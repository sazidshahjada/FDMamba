"""Save training curves and benchmark comparisons as PNG/PDF figures and tables."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.lines import Line2D

METRICS = {
    "accuracy": "Accuracy",
    "auc": "AUC",
    "macro_f1": "Macro-F1",
    "balanced_accuracy": "Balanced accuracy",
}
VARIANTS = ("B0", "B1", "B2", "B3")
COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7")
MARKERS = ("o", "s", "^", "D")


def save_figure(figure, directory, name):
    """Save a readable raster figure and an editable vector counterpart."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        figure.savefig(directory / f"{name}.{extension}", dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_curves(experiment_dir):
    """Plot recorded train/validation history, with the validation-selected epoch marked."""
    directory = Path(experiment_dir)
    history = pd.read_csv(directory / "history.csv")
    if history.empty or not history["epoch"].is_unique:
        raise ValueError("Training history must contain unique recorded epochs")
    metadata = json.loads((directory / "metadata.json").read_text())
    config = yaml.safe_load((directory / "config.yaml").read_text())
    monitor = config["training"]["monitor"]
    scores = history[f"val_{monitor}"].dropna()
    selected = (
        history.loc[scores.idxmin() if monitor == "loss" else scores.idxmax(), "epoch"]
        if len(scores)
        else None
    )
    title = f"{metadata['name']} · {metadata['dataset']}"
    output = directory / "plots"

    def draw(axis, metric, label):
        for split, color in zip(("train", "val"), COLORS):
            axis.plot(history["epoch"], history[f"{split}_{metric}"], label=split, color=color)
        if selected is not None:
            axis.axvline(selected, color="0.45", linestyle=":", label="Selected validation epoch")
        axis.set(xlabel="Epoch", ylabel=label)
        axis.grid(alpha=0.2)
        if metric != "loss":
            axis.set_ylim(0, 1.02)

    figure, axis = plt.subplots(figsize=(8, 4), layout="constrained")
    draw(axis, "loss", "Cross-entropy loss")
    axis.set_title(title)
    axis.legend(fontsize=8)
    save_figure(figure, output, "loss_curves")

    figure, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for axis, (metric, label) in zip(axes.flat, METRICS.items()):
        draw(axis, metric, label)
    axes.flat[0].legend(fontsize=8)
    figure.suptitle(title)
    save_figure(figure, output, "metric_curves")
    print(f"Saved curves: {output}")
    return output


def pareto_mask(parameters, performance):
    """Keep finite points not dominated by a smaller/equal, better/equal model."""
    parameters, performance = np.asarray(parameters), np.asarray(performance)
    finite = np.isfinite(parameters) & np.isfinite(performance)
    result = np.zeros(len(parameters), dtype=bool)
    for index in np.flatnonzero(finite):
        dominated = (
            finite
            & (parameters <= parameters[index])
            & (performance >= performance[index])
            & ((parameters < parameters[index]) | (performance > performance[index]))
        )
        result[index] = not dominated.any()
    return result


def comparison_table(summary):
    """Add parameter counts in millions and paired B0 deltas to benchmark results."""
    table = pd.read_csv(summary)
    required = {"name", "model", "dataset", "parameters", "size", "image_size", *METRICS}
    if not required <= set(table) or table.empty:
        raise ValueError(f"Benchmark CSV needs rows and columns {sorted(required)}")
    if table.duplicated(["dataset", "model"]).any():
        raise ValueError("Compare one checkpoint per model/dataset; separate seeds into reports")
    if len(table[["size", "image_size"]].drop_duplicates()) != 1:
        raise ValueError("Compare checkpoints with the same data and input resolutions")
    parts = table["model"].str.extract(r"^FDMamba-(.+)-(B[0-3])$")
    if parts.isna().any().any():
        raise ValueError("Expected FDMamba model names with variant B0, B1, B2 or B3")
    table["backbone"], table["variant"] = parts[0], parts[1]
    if not np.isfinite(table["parameters"]).all() or (table["parameters"] <= 0).any():
        raise ValueError("Parameter counts must be finite and positive")
    for metric in METRICS:
        scores = table[metric].dropna()
        if not np.isfinite(scores).all() or not scores.between(0, 1).all():
            raise ValueError(f"{metric} must be in [0, 1] or undefined")
    table["parameters_millions"] = table["parameters"] / 1e6
    baseline = table[table["variant"] == "B0"]
    keys = ["dataset", "backbone"]
    paired = baseline[keys + ["parameters", *METRICS]].rename(
        columns={key: f"baseline_{key}" for key in ["parameters", *METRICS]}
    )
    table = table.merge(paired, on=keys, how="left", validate="many_to_one")
    table["added_parameters_vs_B0"] = table["parameters"] - table["baseline_parameters"]
    for metric in METRICS:
        # Percentage-point changes, not relative percentage improvements.
        table[f"delta_{metric}_pp_vs_B0"] = 100 * (table[metric] - table[f"baseline_{metric}"])
    for metric in ("macro_f1", "auc"):
        table[f"pareto_{metric}"] = False
        for _, group in table.groupby("dataset", sort=False):
            table.loc[group.index, f"pareto_{metric}"] = pareto_mask(
                group["parameters"], group[metric]
            )
    return table.sort_values(["dataset", "backbone", "variant"]).reset_index(drop=True)


def plot_comparison(summary, output_dir, *, expected_runs=None):
    """Compare test metrics within each dataset and show parameter/performance tradeoffs."""
    table = comparison_table(summary)
    if expected_runs is not None and (expected_runs < len(table) or expected_runs < 1):
        raise ValueError("expected_runs must be at least the number of benchmark rows")
    count = len(table)
    coverage = f"{count}/{expected_runs or count} runs"
    if expected_runs is not None and count < expected_runs:
        coverage += " · PARTIAL comparison"
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "performance_table.csv", index=False)
    delta_columns = [
        "dataset",
        "backbone",
        "variant",
        "parameters_millions",
        "added_parameters_vs_B0",
        *[f"delta_{metric}_pp_vs_B0" for metric in METRICS],
        "pareto_macro_f1",
        "pareto_auc",
    ]
    table[delta_columns].to_csv(output / "baseline_deltas.csv", index=False)
    visible = ["dataset", "backbone", "variant", "parameters_millions", *METRICS]
    if "images_per_second" in table:
        visible.append("images_per_second")
    html = (
        '<!doctype html><meta charset="utf-8"><title>FDMamba performance</title>'
        "<style>body{font-family:sans-serif}td,th{padding:6px;text-align:right}"
        "tr:nth-child(even){background:#f2f5f8}</style>"
        f"<h1>Test performance — {coverage}</h1>"
        "<p>Metrics are fractions in [0, 1]. Missing values are undefined. "
        "Compare within datasets; a single seed does not establish statistical significance. "
        "Throughput includes preprocessing and data loading.</p>"
        + table[visible].to_html(index=False, float_format=lambda x: f"{x:.4f}", na_rep="—")
    )
    (output / "performance_table.html").write_text(html)
    for dataset, group in table.groupby("dataset", sort=False):
        figure, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
        for axis, (metric, label) in zip(axes.flat, METRICS.items()):
            pivot = group.pivot(index="backbone", columns="variant", values=metric)
            pivot = pivot.reindex(columns=VARIANTS)
            positions = np.arange(len(pivot))
            for index, variant in enumerate(VARIANTS):
                axis.bar(
                    positions + (index - 1.5) * 0.19,
                    pivot[variant],
                    0.19,
                    label=variant,
                    color=COLORS[index],
                )
            axis.set_xticks(positions, pivot.index)
            axis.set(ylabel=label, ylim=(0, 1.05))
            axis.grid(axis="y", alpha=0.2)
        axes.flat[0].legend(ncol=4, fontsize=8)
        figure.suptitle(f"{dataset} · Test performance · {coverage}")
        save_figure(figure, output, f"comparison_{dataset}")

    datasets = table["dataset"].unique()
    families = sorted(table["backbone"].unique())
    for metric in ("macro_f1", "auc"):
        rows = (len(datasets) + 1) // 2
        figure, axes = plt.subplots(rows, 2, figsize=(12, 4.5 * rows), squeeze=False)
        for axis, dataset in zip(axes.flat, datasets):
            group = table[table["dataset"] == dataset]
            for family, color in zip(families, COLORS):
                for variant, marker in zip(VARIANTS, MARKERS):
                    points = group[(group["backbone"] == family) & (group["variant"] == variant)]
                    axis.scatter(
                        points["parameters_millions"],
                        points[metric],
                        color=color,
                        marker=marker,
                        s=70,
                        zorder=3,
                    )
            frontier = group[group[f"pareto_{metric}"]].sort_values("parameters")
            axis.plot(frontier["parameters_millions"], frontier[metric], "k--", alpha=0.45)
            axis.set(
                title=dataset,
                xlabel="Parameters (millions)",
                ylabel=METRICS[metric],
                ylim=(0, 1.02),
            )
            axis.grid(alpha=0.2)
        for axis in list(axes.flat)[len(datasets) :]:
            axis.set_visible(False)
        handles = [
            Line2D([], [], marker="o", linestyle="", color=color, label=family)
            for family, color in zip(families, COLORS)
        ]
        handles += [
            Line2D([], [], marker=marker, linestyle="", color="0.35", label=variant)
            for variant, marker in zip(VARIANTS, MARKERS)
        ]
        handles.append(Line2D([], [], color="k", linestyle="--", label="Pareto frontier"))
        figure.legend(handles=handles, loc="lower center", ncol=4, fontsize=9)
        figure.suptitle(f"Parameters versus test {METRICS[metric]} · {coverage}")
        figure.tight_layout(rect=(0, 0.09, 1, 0.95))
        save_figure(figure, output, f"parameters_vs_{metric}")
    print(f"Saved tables and comparison figures: {output} ({coverage})")
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    curves = commands.add_parser("curves", help="Train/validation curves for one experiment")
    curves.add_argument("--experiment-dir", required=True)
    comparison = commands.add_parser("comparison", help="Benchmark tables and analysis figures")
    comparison.add_argument("--summary", required=True, help="CSV written by benchmark.py")
    comparison.add_argument("--output-dir", default="results/comparison")
    comparison.add_argument("--expected-runs", type=int)
    args = parser.parse_args()
    if args.command == "curves":
        plot_curves(args.experiment_dir)
    else:
        plot_comparison(args.summary, args.output_dir, expected_runs=args.expected_runs)


if __name__ == "__main__":
    main()
