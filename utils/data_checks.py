"""Dataset identities and evaluation-loader validation (no image decoding)."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from torch.utils.data import SequentialSampler, Subset, TensorDataset

from .runtime import _jsonable, _name


def _base_dataset(dataset):
    """Unwrap nested Subsets without loading examples."""
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    return dataset


def _sample(dataset, index):
    """Read a sample path and label from metadata, following Subset indices."""
    if isinstance(dataset, Subset):
        return _sample(dataset.dataset, int(dataset.indices[index]))
    if hasattr(dataset, "samples"):
        path, target = dataset.samples[index]
        return str(path), int(target)
    if hasattr(dataset, "targets"):
        return "", int(dataset.targets[index])
    if isinstance(dataset, TensorDataset) and dataset.tensors[-1].ndim == 1:
        return "", int(dataset.tensors[-1][index])
    return None


def _class_counts(dataset, class_names) -> dict | None:
    """Count split labels from metadata, including Subsets, without decoding images."""
    if hasattr(dataset, "class_counts"):
        return dict(dataset.class_counts)
    counts = Counter(dict.fromkeys(class_names, 0))
    for index in range(len(dataset)):
        sample = _sample(dataset, index)
        if sample is None:
            return None
        label = sample[1]
        name = class_names[label] if 0 <= label < len(class_names) else str(label)
        counts[name] += 1
    return dict(counts)


def _loader_info(loader) -> dict | None:
    """Fingerprint ordered paths/labels and record preprocessing; do not hash pixels."""
    if loader is None:
        return None
    dataset = loader.dataset
    base = _base_dataset(dataset)
    root = Path(base.root).expanduser().resolve() if hasattr(base, "root") else None
    digest = hashlib.sha256()
    known = True
    for index in range(len(dataset)):
        sample = _sample(dataset, index)
        if sample is None:
            known = False
            break
        path, label = sample
        if path and root is not None:
            path = os.path.relpath(path, root)
        digest.update(json.dumps([index, path, label], ensure_ascii=False).encode() + b"\n")
    return _jsonable(
        dict(
            dataset_class=_name(dataset),
            num_samples=len(dataset),
            root=root,
            class_to_idx=getattr(base, "class_to_idx", None),
            classes=getattr(base, "classes", None),
            manifest_sha256=digest.hexdigest() if known else None,
            identity_scope="ordered paths/labels only; image bytes not hashed"
            if known
            else "dataset type/length only",
            transform=repr(getattr(base, "transform", None)),
            normalization_mean=getattr(base, "normalization_mean", None),
            normalization_std=getattr(base, "normalization_std", None),
            batch_size=loader.batch_size,
            drop_last=loader.drop_last,
            num_workers=loader.num_workers,
            sampler_class=_name(loader.sampler),
        )
    )


def _check_eval_loader(loader) -> None:
    """Require a nonempty, sequential loader with deterministic known transforms."""
    if loader is None:
        raise ValueError("An evaluation DataLoader is required")
    if not isinstance(loader.sampler, SequentialSampler) or loader.drop_last:
        raise ValueError("Evaluation requires a sequential loader with drop_last=False")
    if len(loader.dataset) == 0 or len(loader) == 0:
        raise ValueError("Evaluation dataset/loader is empty")

    def check(transform):
        if transform is None:
            return
        if (
            type(transform).__name__.startswith("Random")
            or type(transform).__name__ == "ColorJitter"
        ):
            raise ValueError(
                "Evaluation transforms must be deterministic; remove random augmentation"
            )
        for child in getattr(transform, "transforms", ()):
            check(child)

    check(getattr(_base_dataset(loader.dataset), "transform", None))
