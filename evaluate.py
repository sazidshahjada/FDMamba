"""Evaluate a saved native-head model, with no training-data access or fitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fdmambadr import build_model
from utils.loaders import create_eval_loader
from utils.trainer import Trainer


def evaluate_checkpoint(
    checkpoint,
    data_path,
    *,
    split="test",
    output_dir=None,
    batch_size=32,
    num_workers=0,
    device=None,
    verbose=True,
):
    """Reconstruct a trusted train.py/build_model checkpoint and export predictions.

    Uses saved input size, normalization, class order and metric settings. Loss is
    omitted because this path needs no training-fitted loss instance or optimizer.
    """
    checkpoint = Path(checkpoint).expanduser().resolve()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved = state["config"]
    build = saved.get("model_build")
    if build is None:
        raise ValueError(
            "Checkpoint lacks build_model metadata; use Trainer.evaluate with your original constructor"
        )
    data_config = saved.get("run_config", {}).get("data")
    if data_config is None:
        raise ValueError(
            "Checkpoint lacks preprocessing settings; use Trainer.evaluate with an explicit loader"
        )
    model = build_model(**build)
    model.requires_grad_(False)
    train_info = state["dataset_fingerprints"]["train"]
    loader = create_eval_loader(
        data_path,
        image_size=data_config.get("image_size", 224),
        batch_size=batch_size,
        num_workers=num_workers,
        class_names=state["class_names"],
        mean=train_info["normalization_mean"],
        std=train_info["normalization_std"],
    )
    trainer = Trainer(
        model,
        test_loader=loader,
        device=device,
        class_names=state["class_names"],
        metrics_kwargs=saved["metrics_kwargs"],
        output_dir=output_dir or checkpoint.parent,
        verbose=verbose,
        run_config=saved.get("run_config", {}),
    )
    # Release the first checkpoint copy before Trainer loads/evaluates it.
    del state
    return trainer.evaluate(split=split, checkpoint=checkpoint, compute_loss=False)


def main():
    """Evaluate each source separately using a fixed checkpoint and preprocessing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--data", type=Path, required=True, help="Folder containing verified grade folders 0..4"
    )
    parser.add_argument("--split", default="test", help="Output label, e.g. test, idrid or ddr")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--quiet", action="store_true", help="Hide run information and progress; print only JSON"
    )
    args = parser.parse_args()
    report = evaluate_checkpoint(
        args.checkpoint,
        args.data,
        split=args.split,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        verbose=not args.quiet,
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
