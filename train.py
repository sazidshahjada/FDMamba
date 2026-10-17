"""Train one configuration: python train.py --config configs/train.yaml."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from fdmambadr import build_model
from utils.loaders import create_dataloaders
from utils.losses import BalancedCrossEntropy, CrossEntropy, FocalLoss
from utils.runtime import seed_everything, warmup_cosine_scheduler
from utils.trainer import Trainer


def train(config: dict, *, resume: bool = False) -> Trainer:
    """Build the shared pipeline from plain configuration; select on validation QWK."""
    unknown = set(config) - {"model", "data", "training", "loss"}
    if unknown:
        raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
    settings = dict(config.get("training", {}))
    seed = settings.get("seed", 42)
    seed_everything(seed, settings.get("deterministic", False))
    model = build_model(**config.get("model", {}))
    loaders = create_dataloaders(**{**config["data"], "seed": seed})
    loss = dict(config.get("loss", {}))
    loss_name = loss.pop("name", "balanced_ce")
    if loss_name == "balanced_ce":
        criterion = BalancedCrossEntropy.from_dataset(loaders["train"].dataset, **loss)
    elif loss_name == "ce":
        criterion = CrossEntropy(**loss)
    elif loss_name == "focal":
        criterion = FocalLoss(**loss)
    else:
        raise ValueError("loss.name must be balanced_ce, ce or focal")
    epochs = settings.get("epochs", 100)
    warmup = settings.pop("warmup_epochs", min(5, epochs - 1))
    if settings.get("monitor", "qwk") != "qwk" or settings.get("mode", "max") != "max":
        raise ValueError("Primary runs select checkpoints using validation QWK/max")
    trainer = Trainer(
        model,
        loaders["train"],
        loaders["val"],
        loaders.get("test"),
        criterion=criterion,
        scheduler=warmup_cosine_scheduler,
        scheduler_kwargs={"total_epochs": epochs, "warmup_epochs": warmup},
        run_config=config,
        **settings,
    )
    trainer.fit(resume_from=trainer.output_dir / "last.pt" if resume else None)
    return trainer


def main():
    """Read a YAML configuration; paths are relative to the working directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument(
        "--resume", action="store_true", help="Resume last.pt in the configured run directory"
    )
    args = parser.parse_args()
    with args.config.open() as handle:
        config = yaml.safe_load(handle)
    train(config, resume=args.resume)


if __name__ == "__main__":
    main()
