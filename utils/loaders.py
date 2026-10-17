"""ImageFolder loaders and image grids for the supplied train/val/test splits.

Expected layout: ``split_path/<class_name>/*.{jpg,png,...}``. Default class
folders are DR grades 0 through 4. No partitions are rebuilt or reshuffled.
"""
from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import InterpolationMode

if TYPE_CHECKING:
    from matplotlib.figure import Figure


DR_CLASSES = ("0", "1", "2", "3", "4")


def _image_size(value: int | Sequence[int]) -> tuple[int, int]:
    size = (value, value) if isinstance(value, int) else tuple(value)
    if len(size) != 2 or any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in size):
        raise ValueError("image_size must be a positive integer or (height, width)")
    return size


def _classes(names: Sequence[str]) -> tuple[str, ...]:
    names = tuple(names)
    if not names or any(not isinstance(name, str) or not name or Path(name).name != name
                        or name in (".", "..") for name in names):
        raise ValueError("class_names must be an ordered sequence of folder names")
    if len(set(names)) != len(names):
        raise ValueError("class_names must be unique")
    return names


def _normalization(mean, std):
    if (mean is None) != (std is None):
        raise ValueError("Provide both mean and std, or neither")
    if mean is None:
        return None, None
    mean, std = tuple(mean), tuple(std)
    if (len(mean) != 3 or len(std) != 3
            or not all(np.isfinite(v) for v in (*mean, *std))
            or any(v <= 0 for v in std)):
        raise ValueError("mean/std must contain three finite RGB values, with std > 0")
    return mean, std


def _load_rgb(path: str) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def _seed_worker(_worker_id: int) -> None:
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


class _ImageFolderDataset(ImageFolder):
    """ImageFolder with a fixed label mapping, including absent split classes."""

    def __init__(self, root, *, class_names, image_size, augment=False, mean=None, std=None):
        self._class_names = class_names
        self.normalization_mean, self.normalization_std = mean, std
        pipeline = [transforms.Resize(image_size, interpolation=InterpolationMode.BILINEAR,
                                      antialias=True)]
        if augment:
            # Keep the full resized field of view; do not randomly crop lesions.
            pipeline.extend([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10, interpolation=InterpolationMode.BILINEAR),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
            ])
        pipeline.append(transforms.ToTensor())
        if mean is not None:
            pipeline.append(transforms.Normalize(mean, std))
        super().__init__(root=str(root), transform=transforms.Compose(pipeline),
                         loader=_load_rgb, allow_empty=True)
        if not self.samples:
            raise ValueError(f"No supported images found in {root}")
        self.class_counts = dict(zip(self.classes, np.bincount(
            self.targets, minlength=len(self.classes)).tolist()))

    def find_classes(self, directory):
        present, _ = super().find_classes(directory)
        unknown = sorted(set(present) - set(self._class_names))
        if unknown:
            raise ValueError(f"Unexpected class folders in {directory}: {unknown}; "
                             "pass class_names in the verified label order")
        return list(self._class_names), {name: i for i, name in enumerate(self._class_names)}


def _loader(dataset, *, batch_size, num_workers, shuffle, seed, pin_memory, drop_last=False):
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if not isinstance(num_workers, int) or isinstance(num_workers, bool) or num_workers < 0:
        raise ValueError("num_workers must be a nonnegative integer")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if drop_last and len(dataset) < batch_size:
        raise ValueError("drop_last_train would discard the entire training dataset; reduce batch_size or disable it")
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=pin_memory, drop_last=drop_last,
        persistent_workers=num_workers > 0, worker_init_fn=_seed_worker,
        generator=torch.Generator().manual_seed(seed),
    )


def create_dataloaders(
    train_path: str | Path,
    val_path: str | Path,
    test_path: str | Path,
    *,
    image_size: int | Sequence[int] = 224,
    batch_size: int = 32,
    num_workers: int = 0,
    augment: bool = True,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
    class_names: Sequence[str] = DR_CLASSES,
    seed: int = 42,
    pin_memory: bool | None = None,
    drop_last_train: bool = False,
) -> dict[str, DataLoader]:
    """Return ``{'train': ..., 'val': ..., 'test': ...}`` PyTorch loaders.

    Only training is shuffled/augmented. Validation/test always use deterministic
    RGB decoding and resizing and retain their final incomplete batch. Use
    ``augment=False`` when additional augmentation of offline variants is unwanted.
    ``image_size`` is an integer or (height, width).

    Labels follow ``class_names`` order, independently of which class folders are
    present in a split. Defaults map folder '0' to 0, ..., folder '4' to 4.
    Unknown folders and empty splits fail clearly. Each dataset exposes
    ``classes``, ``class_to_idx``, ``class_counts``, ``samples`` and ``targets``.

    Without mean/std, tensors are RGB float32 in [0, 1]. If supplied, use the
    SAME training-fitted statistics for every split; this function never fits
    validation/test statistics. ``compute_train_mean_std`` can fit training only.
    ``seed`` controls shuffle and worker RNGs; seed the training process's global
    PyTorch RNG too, for augmentation with num_workers=0 and model initialization.
    Decode errors propagate rather than silently dropping corrupt images.
    """
    size, classes = _image_size(image_size), _classes(class_names)
    mean, std = _normalization(mean, std)
    roots = {split: Path(path).expanduser().resolve()
             for split, path in zip(("train", "val", "test"), (train_path, val_path, test_path))}
    for split, root in roots.items():
        if not root.is_dir():
            raise FileNotFoundError(f"{split} directory does not exist: {root}")
    paths = list(roots.values())
    for index, path in enumerate(paths):
        for other in paths[index + 1:]:
            if path == other or path in other.parents or other in path.parents:
                raise ValueError("Train, validation and test roots must be separate, non-nested directories")
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    loaders = {}
    for index, (split, root) in enumerate(roots.items()):
        dataset = _ImageFolderDataset(root, class_names=classes, image_size=size,
                                      augment=augment and split == "train", mean=mean, std=std)
        loaders[split] = _loader(
            dataset, batch_size=batch_size, num_workers=num_workers, seed=seed + index,
            shuffle=split == "train", pin_memory=pin_memory,
            drop_last=drop_last_train if split == "train" else False,
        )
    return loaders


@torch.no_grad()
def compute_train_mean_std(
    train_path: str | Path,
    *,
    image_size: int | Sequence[int] = 224,
    batch_size: int = 64,
    num_workers: int = 0,
    class_names: Sequence[str] = DR_CLASSES,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Fit RGB pixel mean/population std on resized TRAIN images, without augmentation.

    This reads the entire training folder once. Save/reuse the result in the run
    configuration; validation/test images are never inputs to this computation.
    """
    root = Path(train_path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Training directory does not exist: {root}")
    dataset = _ImageFolderDataset(root, class_names=_classes(class_names),
                                  image_size=_image_size(image_size))
    loader = _loader(dataset, batch_size=batch_size, num_workers=num_workers,
                     shuffle=False, seed=42, pin_memory=False)
    total = torch.zeros(3, dtype=torch.float64)
    squares = torch.zeros_like(total)
    pixels = 0
    for images, _labels in loader:
        images = images.to(torch.float64)
        total += images.sum(dim=(0, 2, 3))
        squares += images.square().sum(dim=(0, 2, 3))
        pixels += images.shape[0] * images.shape[2] * images.shape[3]
    mean = total / pixels
    std = (squares / pixels - mean.square()).clamp_min(0).sqrt()
    if (std == 0).any():
        raise ValueError("Training images contain a zero-variance RGB channel; cannot normalize by zero")
    return tuple(mean.tolist()), tuple(std.tolist())


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

    if split not in ("train", "val", "test") or split not in dataloaders:
        raise ValueError("split must be an available 'train', 'val' or 'test' loader")
    if any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in (rows, cols)):
        raise ValueError("rows and cols must be positive integers")
    dataset = dataloaders[split].dataset
    if not isinstance(dataset, _ImageFolderDataset):
        raise TypeError("Use dataloaders returned by create_dataloaders")
    indices = random.Random(seed).sample(range(len(dataset)), min(rows * cols, len(dataset)))
    figure, axes = plt.subplots(rows, cols, squeeze=False,
                                figsize=figsize if figsize is not None else (3 * cols, 3 * rows))
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
