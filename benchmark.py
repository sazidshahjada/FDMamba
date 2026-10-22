"""Benchmark named checkpoints on matching MedMNIST datasets, without retraining."""

import argparse
from pathlib import Path

import torch

from utils.config import read_yaml, run_name
from utils.evaluation import evaluate_checkpoint
from utils.runtime import write_csv, write_json


def benchmark(
    entries,
    *,
    output_dir="results/benchmark",
    root=None,
    batch_size=32,
    num_workers=0,
    device="auto",
    split="test",
    verbose=True,
):
    """Evaluate explicit name/checkpoint/dataset entries and save comparison tables."""
    if not entries:
        raise ValueError("Provide at least one benchmark entry")
    names = []
    for entry in entries:
        if set(entry) != {"name", "checkpoint", "dataset"}:
            raise ValueError("Each benchmark entry needs exactly name, checkpoint and dataset")
        names.append(run_name(entry["name"]))
        if not Path(entry["checkpoint"]).is_file():
            raise FileNotFoundError(entry["checkpoint"])
    if len(set(names)) != len(names):
        raise ValueError("Use unique benchmark names so results cannot overwrite each other")
    output_dir = Path(output_dir)
    rows, reports = [], []
    for entry in entries:
        report = evaluate_checkpoint(
            entry["checkpoint"],
            entry["dataset"],
            name=entry["name"],
            split=split,
            root=root,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            output_dir=output_dir / entry["name"],
            verbose=verbose,
        )
        reports.append(report)
        rows.append(
            {
                key: report[key]
                for key in (
                    "name",
                    "model",
                    "dataset",
                    "checkpoint",
                    "split",
                    "size",
                    "image_size",
                    "num_samples",
                    "parameters",
                    "accuracy",
                    "auc",
                    "macro_f1",
                    "balanced_accuracy",
                    "seconds",
                    "images_per_second",
                    "device",
                )
            }
        )
        # Save after every model so earlier results survive a later failure.
        write_json(output_dir / "summary.json", reports)
        write_csv(output_dir / "summary.csv", rows)
        if verbose:
            print(
                f"{entry['name']}: accuracy={report['accuracy']:.4f}, AUC={report['auc']}, macro-F1={report['macro_f1']:.4f}"
            )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", help="YAML containing a models list of name/checkpoint/dataset entries"
    )
    parser.add_argument("--names", nargs="+", help="Custom labels, in checkpoint order")
    parser.add_argument("--checkpoints", nargs="+")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--data-root")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output-dir", default="results/benchmark")
    args = parser.parse_args()
    if args.config:
        if any((args.names, args.checkpoints, args.datasets)):
            parser.error("Use --config or --names/--checkpoints/--datasets, not both")
        config = read_yaml(args.config)
        if set(config) != {"models"}:
            parser.error("Benchmark YAML must contain a models list")
        entries = config["models"]
    else:
        if not all((args.names, args.checkpoints, args.datasets)):
            parser.error("Provide --config, or all of --names, --checkpoints and --datasets")
        if len({len(args.names), len(args.checkpoints), len(args.datasets)}) != 1:
            parser.error("Names, checkpoints and datasets must have the same length")
        entries = [
            dict(name=name, checkpoint=checkpoint, dataset=dataset)
            for name, checkpoint, dataset in zip(args.names, args.checkpoints, args.datasets)
        ]
    benchmark(
        entries,
        output_dir=args.output_dir,
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        split=args.split,
    )
    print(f"Saved comparison: {Path(args.output_dir) / 'summary.csv'}")


if __name__ == "__main__":
    main()
