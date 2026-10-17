"""Optional image grids; no plotting dependencies are imported until requested."""

from __future__ import annotations

import random
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def visualize_images(
    dataloaders: Mapping[str, DataLoader],
    rows: int = 3,
    cols: int = 4,
    split: str = "train",
    *,
    seed: int = 42,
    figsize: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> tuple[Figure, np.ndarray]:
    """Plot labeled samples from a selected split; return ``(figure, axes)``.

    Uses loaders from ``create_dataloaders``. Normalization is undone for display;
    training samples include configured augmentation. Samples are selected with
    a local seed and CPU transform RNG is restored afterwards. The loader's
    iterator/shuffle generator is never advanced. Extra grid cells are hidden.
    Use ``show=False`` for scripts or tests, and optionally save the figure.
    """
    import matplotlib.pyplot as plt

    from .loaders import _ImageFolderDataset

    if split not in ("train", "val", "test") or split not in dataloaders:
        raise ValueError("split must be an available 'train', 'val' or 'test' loader")
    if any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in (rows, cols)):
        raise ValueError("rows and cols must be positive integers")
    dataset = dataloaders[split].dataset
    if not isinstance(dataset, _ImageFolderDataset):
        raise TypeError("Use dataloaders returned by create_dataloaders")
    indices = random.Random(seed).sample(range(len(dataset)), min(rows * cols, len(dataset)))
    figure, axes = plt.subplots(
        rows, cols, squeeze=False, figsize=figsize if figsize is not None else (3 * cols, 3 * rows)
    )
    for axis in axes.flat:
        axis.axis("off")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        for axis, index in zip(axes.flat, indices):
            image, label = dataset[index]
            if dataset.normalization_mean is not None:
                mean = image.new_tensor(dataset.normalization_mean).view(3, 1, 1)
                std = image.new_tensor(dataset.normalization_std).view(3, 1, 1)
                image = image * std + mean
            axis.imshow(image.clamp(0, 1).permute(1, 2, 0).numpy())
            axis.set_title(f"Class {dataset.classes[label]}")
    figure.suptitle(f"{split.capitalize()} images")
    figure.tight_layout()
    if save_path is not None:
        destination = Path(save_path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return figure, axes
