"""Official MedMNIST datasets, RGB transforms and reproducible loaders."""

import random
from pathlib import Path

import medmnist
import numpy as np
import torch
from medmnist.dataset import MedMNIST2D
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

DEFAULT_DATASETS = ("pathmnist", "dermamnist", "bloodmnist", "organamnist")


def dataset_info(name):
    """Return metadata for a supported single-label 2D dataset."""
    if name not in medmnist.INFO:
        raise ValueError(f"Unknown MedMNIST dataset: {name}")
    info = medmnist.INFO[name]
    if not issubclass(getattr(medmnist, info["python_class"]), MedMNIST2D):
        raise ValueError("FDMamba supports MedMNIST2D images, not 3D volumes")
    if info["task"] not in ("multi-class", "binary-class", "ordinal-regression"):
        raise ValueError("Use a single-label dataset; ChestMNIST is multilabel")
    return info


def class_names(name):
    """Return human-readable labels in official numeric order."""
    labels = dataset_info(name)["label"]
    return [labels[str(index)] for index in range(len(labels))]


def image_transform(image_size=224, *, augment=False, mean=None, std=None):
    """Resize RGB inputs; optionally augment training data and normalize."""
    if not isinstance(image_size, int) or isinstance(image_size, bool) or image_size < 32:
        raise ValueError("image_size must be an integer >= 32 for these CNNs")
    if (mean is None) != (std is None):
        raise ValueError("Provide both mean and std, or neither")
    if mean is not None and (
        len(mean) != 3 or len(std) != 3 or not np.isfinite([*mean, *std]).all() or min(std) <= 0
    ):
        raise ValueError("mean/std must contain three finite values, with std > 0")
    steps = [transforms.Resize((image_size, image_size), antialias=True)]
    if augment:
        steps += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10, interpolation=InterpolationMode.BILINEAR),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
        ]
    steps.append(transforms.ToTensor())
    if mean is not None:
        steps.append(transforms.Normalize(mean, std))
    return transforms.Compose(steps)


def scalar_label(label):
    """Adapt MedMNIST's one-element target to a scalar class index."""
    return int(label[0])


def load_dataset(
    name,
    split,
    *,
    root="data/medmnist",
    size=28,
    image_size=224,
    download=False,
    augment=False,
    mean=None,
    std=None,
):
    """Load an official split; MedMNIST owns downloading and its permanent cache."""
    info = dataset_info(name)
    if split not in ("train", "val", "test"):
        raise ValueError("split must be train, val or test")
    if not isinstance(size, int) or isinstance(size, bool) or size not in (28, 64, 128, 224):
        raise ValueError("size must be an official MedMNIST resolution: 28, 64, 128 or 224")
    root = Path(root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    dataset = getattr(medmnist, info["python_class"])(
        root=str(root),
        split=split,
        size=size,
        download=download,
        as_rgb=True,
        target_transform=scalar_label,
        transform=image_transform(
            image_size, augment=augment and split == "train", mean=mean, std=std
        ),
    )
    labels = dataset.labels
    if (
        labels.shape != (len(dataset), 1)
        or not len(dataset)
        or not np.issubdtype(labels.dtype, np.integer)
        or labels.min() < 0
        or labels.max() >= len(info["label"])
    ):
        raise ValueError(f"Invalid single-label targets in {name}/{split}")
    return dataset


def seed_worker(_worker_id):
    """Give each data-loader worker reproducible Python and NumPy randomness."""
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def make_loader(name, split, data, *, seed=42, download=False):
    """Create a loader without shuffling or augmenting held-out splits."""
    settings = dict(data)
    batch_size = settings.pop("batch_size", 32)
    num_workers = settings.pop("num_workers", 0)
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative")
    dataset = load_dataset(name, split, download=download, **settings)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "train",
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
    )
