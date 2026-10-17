"""Reproducibility, scheduling, JSON serialization and RNG snapshots."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.optim import Optimizer


def seed_everything(seed: int = 42, deterministic: bool = False) -> None:
    """Call BEFORE constructing models/loaders to seed their initialization too."""
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False


def warmup_cosine_scheduler(
    optimizer: Optimizer,
    *,
    total_epochs: int,
    warmup_epochs: int = 5,
    min_lr: float = 0.0,
    start_factor: float = 0.1,
):
    """Epoch scheduler factory usable as Trainer(scheduler=..., scheduler_kwargs=...)."""
    if not isinstance(total_epochs, int) or total_epochs <= 0:
        raise ValueError("total_epochs must be positive")
    if not isinstance(warmup_epochs, int) or not 0 <= warmup_epochs < total_epochs:
        raise ValueError("warmup_epochs must be in [0, total_epochs)")
    if not 0 < start_factor <= 1 or not math.isfinite(min_lr) or min_lr < 0:
        raise ValueError("start_factor must be in (0,1] and min_lr must be finite/nonnegative")
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs,
        eta_min=min_lr,
    )
    if not warmup_epochs:
        return cosine
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=start_factor,
        total_iters=warmup_epochs,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )


def _name(value) -> str:
    """Return a stable qualified class name for configuration metadata."""
    cls = value if isinstance(value, type) else type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _jsonable(value):
    """Convert tensors, paths and configurations into JSON-compatible values."""
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (Path, torch.device, torch.dtype)):
        return str(value)
    if callable(value):
        return (
            f"{value.__module__}.{value.__qualname__}"
            if hasattr(value, "__qualname__")
            else _name(value)
        )
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _write_json(path: Path, data) -> None:
    """Write strict JSON atomically so interrupted writes do not truncate reports."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(_jsonable(data), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _loader_rng(loader):
    """Capture the loader and sampler generator states without advancing them."""
    if loader is None:
        return {}
    result = {}
    for name, owner in (("loader", loader), ("sampler", loader.sampler)):
        generator = getattr(owner, "generator", None)
        if isinstance(generator, torch.Generator):
            result[name] = generator.get_state()
    return result


def _restore_loader_rng(loader, state):
    """Restore only generators that were present in the saved loader."""
    if loader is None:
        return
    for name, owner in (("loader", loader), ("sampler", loader.sampler)):
        if name in state:
            generator = getattr(owner, "generator", None)
            if not isinstance(generator, torch.Generator):
                raise ValueError(f"Resuming requires the saved {name} generator")
            generator.set_state(state[name].cpu())


def _rng_state():
    """Capture Python, NumPy, PyTorch and available CUDA RNG states."""
    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    )


def _restore_rng(state):
    """Restore the process RNG snapshot after resume or temporary evaluation."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])
