"""Load saved models, evaluate an official split, or classify a single image."""

from pathlib import Path

import torch
from PIL import Image, ImageOps

from FDMamba import build_model, model_name

from .data import class_names, image_transform, make_loader
from .engine import run_epoch
from .runtime import select_device, write_csv, write_json


def load_model(checkpoint, *, device="auto", model=None):
    """Reconstruct a trusted FDMamba training checkpoint with strict weight loading."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state.get("format_version") != 2:
        raise ValueError("Use a checkpoint created by this repository's current train.py")
    build = state["model_config"]
    expected = {state["name"], build["backbone"], model_name(build["backbone"], build["variant"])}
    if model is not None and model not in expected:
        raise ValueError(
            f"Selected model {model!r} differs from the checkpoint: {sorted(expected)}"
        )
    device = select_device(device)
    if build["variant"] in ("B2", "B3") and device.type != "cuda":
        raise RuntimeError("B2/B3 require CUDA and real Mamba extensions; select a CUDA device")
    network = build_model(**build).to(device)
    network.load_state_dict(state["model_state"], strict=True)
    if network.num_classes != len(state["class_names"]):
        raise ValueError("Checkpoint labels do not match its classifier")
    network.eval().requires_grad_(False)
    metadata = {
        key: state[key] for key in ("name", "dataset", "class_names", "config", "model_config")
    }
    return network, metadata, device


def save_predictions(path, targets, probabilities, *, dataset, split, size):
    """Write probabilities in official split order, with stable archive-row IDs."""
    suffix = "" if size == 28 else f"_{size}"
    rows = (
        dict(
            sample_id=f"{dataset}{suffix}.npz::{split}:{i:06d}",
            true_label=int(target),
            predicted_label=int(scores.argmax()),
            **{f"p{j}": float(score) for j, score in enumerate(scores)},
        )
        for i, (target, scores) in enumerate(zip(targets, probabilities))
    )
    write_csv(path, rows)


def evaluate_checkpoint(
    checkpoint,
    dataset,
    *,
    name=None,
    split="test",
    root=None,
    batch_size=None,
    num_workers=None,
    device="auto",
    output_dir=None,
    verbose=True,
):
    """Evaluate one named checkpoint on its matching MedMNIST label space."""
    network, state, device = load_model(checkpoint, device=device)
    if dataset != state["dataset"] or class_names(dataset) != state["class_names"]:
        raise ValueError("Checkpoint dataset and label order must match the requested dataset")
    if split not in ("val", "test"):
        raise ValueError("Checkpoint evaluation accepts val or test")
    data = dict(state["config"]["data"])
    for key, value in (("root", root), ("batch_size", batch_size), ("num_workers", num_workers)):
        if value is not None:
            data[key] = str(value) if key == "root" else value
    loader = make_loader(dataset, split, data)
    report, targets, probabilities = run_epoch(
        network, loader, device, state["class_names"], verbose=verbose
    )
    report.update(
        name=name or state["name"],
        model=network.display_name,
        dataset=dataset,
        split=split,
        size=data["size"],
        image_size=data["image_size"],
        parameters=sum(p.numel() for p in network.parameters()),
        checkpoint=str(Path(checkpoint).resolve()),
        device=str(device),
    )
    if output_dir is not None:
        output_dir = Path(output_dir)
        write_json(output_dir / "metrics.json", report)
        save_predictions(
            output_dir / "predictions.csv",
            targets,
            probabilities,
            dataset=dataset,
            split=split,
            size=data["size"],
        )
    return report


def predict_image(checkpoint, image, *, model=None, device="auto", top_k=5):
    """Classify one RGB/grayscale image using checkpoint preprocessing and labels."""
    network, state, device = load_model(checkpoint, device=device, model=model)
    if top_k < 1:
        raise ValueError("top_k must be positive")
    data = state["config"]["data"]
    transform = image_transform(data["image_size"], mean=data["mean"], std=data["std"])
    with Image.open(image) as opened:
        tensor = transform(ImageOps.exif_transpose(opened).convert("RGB")).unsqueeze(0)
    with torch.inference_mode():
        probabilities = network(tensor.to(device)).float().softmax(dim=1)[0].cpu()
    if not torch.isfinite(probabilities).all():
        raise FloatingPointError("Nonfinite prediction")
    values, indices = probabilities.topk(min(top_k, len(state["class_names"])))
    return dict(
        name=state["name"],
        model=network.display_name,
        dataset=state["dataset"],
        image=str(image),
        checkpoint=str(checkpoint),
        predictions=[
            dict(label=int(index), class_name=state["class_names"][index], probability=float(value))
            for value, index in zip(values, indices)
        ],
    )
