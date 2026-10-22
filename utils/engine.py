"""A single epoch loop shared by training and checkpoint evaluation."""

from time import perf_counter

import numpy as np
import torch
from tqdm.auto import tqdm

from .metrics import classification_metrics


def run_epoch(
    model,
    loader,
    device,
    classes,
    *,
    criterion=None,
    optimizer=None,
    scaler=None,
    amp=False,
    accumulation_steps=1,
    max_grad_norm=None,
    description="Evaluation",
    verbose=True,
):
    """Run a split; accumulated gradients use the full window's CE denominator."""
    training = optimizer is not None
    model.train(training)
    if training:
        optimizer.zero_grad(set_to_none=True)
    targets, probabilities = [], []
    numerator = denominator = window_weight = 0.0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = perf_counter()
    with tqdm(
        loader, desc=description, leave=False, colour="green", disable=not verbose
    ) as progress:
        for batch_index, (images, labels) in enumerate(progress):
            images, labels = (
                images.to(device, non_blocking=True),
                labels.to(device, non_blocking=True),
            )
            with (
                torch.set_grad_enabled(training),
                torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                    enabled=amp,
                ),
            ):
                logits = model(images)
                if logits.shape != (len(labels), len(classes)) or not torch.isfinite(logits).all():
                    raise ValueError("Model must return finite logits [batch, num_classes]")
                loss = criterion(logits, labels) if criterion is not None else None
            if loss is not None:
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite loss")
                weight = (
                    len(labels)
                    if criterion.weight is None
                    else float(criterion.weight[labels].sum())
                )
                numerator += float(loss.detach())
                denominator += weight
                if training:
                    scaler.scale(loss).backward()
                    window_weight += weight
                    if (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == len(
                        loader
                    ):
                        scaler.unscale_(optimizer)
                        for parameter in model.parameters():
                            if parameter.grad is not None:
                                parameter.grad.div_(window_weight)
                        if max_grad_norm is not None:
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), max_grad_norm, error_if_nonfinite=True
                            )
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                        window_weight = 0.0
                progress.set_postfix(loss=f"{numerator / denominator:.4f}")
            targets.append(labels.detach().cpu().numpy())
            probabilities.append(logits.detach().float().softmax(dim=1).cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = perf_counter() - started
    if not targets:
        raise ValueError("Cannot evaluate an empty loader")
    targets, probabilities = np.concatenate(targets), np.concatenate(probabilities)
    report = classification_metrics(targets, probabilities, classes)
    report["seconds"] = elapsed
    report["images_per_second"] = len(targets) / elapsed
    if criterion is not None:
        report["loss"] = numerator / denominator
    return report, targets, probabilities
