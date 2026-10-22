"""Small helpers for devices, reproducibility, schedules and saved artifacts."""

import csv
import json
import random
from itertools import chain
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed):
    """Seed initialization and augmentation before creating models and loaders."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def select_device(value="auto"):
    """Resolve an automatic, CPU or CUDA device with a useful failure message."""
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if value in (None, "auto")
        else torch.device(value)
    )
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Choose cpu, cuda or auto")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def make_scheduler(optimizer, epochs, warmup_epochs):
    """Warm up from 10% learning rate, then decay with a cosine schedule."""
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs - warmup_epochs)
    if warmup_epochs == 0:
        return cosine
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_epochs
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, cosine], milestones=[warmup_epochs]
    )


def rng_state(loaders):
    """Capture process and loader RNGs for epoch-boundary resume."""
    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        loaders={key: loader.generator.get_state() for key, loader in loaders.items()},
    )


def restore_rng(state, loaders):
    """Restore the matching RNG snapshot after model and optimizer reconstruction."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])
    for key, loader in loaders.items():
        loader.generator.set_state(state["loaders"][key].cpu())


def write_json(path, data):
    """Write strict JSON atomically, without NaN or partially written reports."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path, rows):
    """Write a table from dictionaries with consistent column names."""
    rows = iter(rows)
    first = next(rows, None)
    if first is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(first))
        writer.writeheader()
        writer.writerows(chain([first], rows))


def save_checkpoint(path, state):
    """Keep the previous checkpoint intact until the new file is complete."""
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)
