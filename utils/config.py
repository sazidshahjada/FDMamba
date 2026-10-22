"""One plain training YAML, with defaults and early validation."""

import re
from copy import deepcopy
from pathlib import Path

import yaml

DEFAULTS = {
    "model": {"backbone": "mobilenetv3", "variant": "B3", "drop_rate": 0.2},
    "data": {
        "root": "data/medmnist",
        "size": 28,
        "image_size": 224,
        "batch_size": 32,
        "num_workers": 0,
        "augment": False,
        "mean": None,
        "std": None,
    },
    "training": {
        "epochs": 100,
        "lr": 0.001,
        "weight_decay": 0.0001,
        "warmup_epochs": 5,
        "patience": 20,
        "monitor": "macro_f1",
        "class_weighted": True,
        "label_smoothing": 0.0,
        "accumulation_steps": 1,
        "max_grad_norm": 1.0,
        "amp": False,
        "seed": 42,
        "device": "auto",
        "output_dir": None,
        "weights_dir": "saved_weight",
    },
}


def read_yaml(path):
    """Read a YAML mapping; paths inside it are relative to the working directory."""
    with Path(path).open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return config


def run_name(value):
    """Require a portable name for experiment folders and benchmark labels."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(
            "Names must start with a letter/digit and contain only letters, digits, _, . or -"
        )
    return value


def resolve_config(config):
    """Fill known defaults and reject misspelled or inconsistent training settings."""
    unknown = set(config) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown sections: {sorted(unknown)}")
    resolved = deepcopy(DEFAULTS)
    for section, values in config.items():
        allowed = set(DEFAULTS[section])
        if section == "model":
            allowed |= {"fdconv", "mamba"}
        if not isinstance(values, dict) or set(values) - allowed:
            raise ValueError(f"Unknown settings in {section}; see configs/train.yaml")
        resolved[section].update(deepcopy(values))
    from FDMamba import BACKBONES

    from .data import image_transform

    model, data, training = (resolved[key] for key in ("model", "data", "training"))
    if model["backbone"] not in BACKBONES or model["variant"] not in ("B0", "B1", "B2", "B3"):
        raise ValueError("Choose a supported backbone and variant B0/B1/B2/B3")
    if not 0 <= model["drop_rate"] < 1:
        raise ValueError("model.drop_rate must be in [0, 1)")
    if data["size"] not in (28, 64, 128, 224):
        raise ValueError("data.size must be 28, 64, 128 or 224")
    image_transform(data["image_size"], mean=data["mean"], std=data["std"])
    for key, minimum in (
        ("epochs", 1),
        ("accumulation_steps", 1),
        ("warmup_epochs", 0),
        ("seed", 0),
    ):
        value = training[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ValueError(f"training.{key} must be an integer >= {minimum}")
    if training["warmup_epochs"] >= training["epochs"]:
        raise ValueError("warmup_epochs must be less than epochs (use 0 for a one-epoch smoke run)")
    if training["patience"] is not None and training["patience"] < 1:
        raise ValueError("patience must be positive or null")
    if training["lr"] <= 0 or training["weight_decay"] < 0:
        raise ValueError("lr must be positive and weight_decay nonnegative")
    if training["max_grad_norm"] is not None and training["max_grad_norm"] <= 0:
        raise ValueError("max_grad_norm must be positive or null")
    if training["monitor"] not in ("macro_f1", "accuracy", "auc", "balanced_accuracy", "loss"):
        raise ValueError("monitor must be macro_f1, accuracy, auc, balanced_accuracy or loss")
    if not 0 <= training["label_smoothing"] < 1:
        raise ValueError("label_smoothing must be in [0, 1)")
    return resolved


def load_config(path):
    """Load a complete training configuration without inheritance or implicit overrides."""
    return resolve_config(read_yaml(path))
