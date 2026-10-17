"""Single-epoch execution shared by Trainer's train, validate and evaluate paths."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping

import torch
from torch import Tensor, nn
from tqdm.auto import tqdm

from .data_checks import _check_eval_loader


class EpochRunner:
    """Internal execution base; Trainer supplies model, loss, optimizer and callbacks.

    Accumulate loss numerators and divide gradients once per optimizer window.
    This preserves weighted-CE semantics even for incomplete minibatches.
    """

    def _batch(self, batch):
        """Parse, validate and transfer a batch to the selected device."""
        if self.batch_parser is not None:
            images, targets = self.batch_parser(batch)
        elif isinstance(batch, Mapping):
            images = batch["images"] if "images" in batch else batch["image"]
            targets = batch["labels"] if "labels" in batch else batch["label"]
        else:
            images, targets = batch[:2]
        if not isinstance(images, Tensor) or not images.is_floating_point():
            raise TypeError("Batches must contain floating-point input tensors")
        if not isinstance(targets, Tensor) or targets.dtype != torch.long or targets.ndim != 1:
            raise TypeError("Batch labels must be one-dimensional torch.long tensors")
        if images.shape[0] != len(targets):
            raise ValueError("Input/label batch sizes differ")
        valid = (
            targets != self.ignore_index
            if self.ignore_index is not None
            else torch.ones_like(targets, dtype=torch.bool)
        )
        if ((targets[valid] < 0) | (targets[valid] >= self.num_classes)).any():
            raise ValueError("Batch labels are outside the configured class range")
        return images.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)

    def _loss_terms(self, logits, targets):
        """Return the summed loss numerator and its correct mean denominator."""
        valid = (
            targets != self.ignore_index
            if self.ignore_index is not None
            else torch.ones_like(targets, dtype=torch.bool)
        )
        if self.loss_normalizer is not None:
            denominator = torch.as_tensor(
                self.loss_normalizer(self.criterion, targets),
                device=self.device,
                dtype=torch.float32,
            )
        elif isinstance(self.criterion, nn.CrossEntropyLoss) and self.criterion.weight is not None:
            denominator = self.criterion.weight[targets[valid]].float().sum()
        else:
            denominator = valid.sum().float()
        if denominator.ndim != 0 or not torch.isfinite(denominator) or denominator < 0:
            raise ValueError("Loss normalizer must be a finite nonnegative scalar")
        if denominator.item() == 0:
            return logits.float().sum() * 0, 0.0
        loss = self.criterion(logits, targets)
        if not isinstance(loss, Tensor) or not torch.isfinite(loss).all():
            raise FloatingPointError("Criterion returned a nonfinite/non-tensor loss")
        reduction = getattr(self.criterion, "reduction", "mean")
        if loss.ndim == 1 and loss.shape == targets.shape:
            numerator = loss[valid].sum()
        elif loss.ndim == 0 and reduction == "sum":
            numerator = loss
        elif loss.ndim == 0 and reduction == "mean":
            numerator = loss * denominator
        else:
            raise ValueError("Criterion must return a mean/sum scalar or one loss per example")
        if not torch.isfinite(numerator):
            raise FloatingPointError("Loss numerator is nonfinite")
        return numerator, float(denominator.item())

    def _optimizer_step(self, denominator):
        """Normalize accumulated gradients, apply optional clipping, and step once."""
        if denominator <= 0:
            self.model.zero_grad(set_to_none=True)
            return
        self.scaler.unscale_(self.optimizer)
        optimized = [
            p
            for group in self.optimizer.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        if not optimized:
            raise RuntimeError("Loss produced no gradients for optimizer parameters")
        for parameter in optimized:
            parameter.grad.div_(denominator)
        norm = nn.utils.clip_grad_norm_(
            optimized,
            self.max_grad_norm or float("inf"),
            error_if_nonfinite=not self.scaler.is_enabled(),
        )
        old_scale = self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        stepped = self.scaler.get_scale() >= old_scale
        if stepped and not math.isfinite(float(norm)):
            raise FloatingPointError("Nonfinite gradient norm")
        if stepped:
            self.global_step += 1
            if self.scheduler is not None and self.scheduler_interval == "step":
                self.scheduler.step()
        self.model.zero_grad(set_to_none=True)

    def _run_epoch(self, loader, *, training, split, epoch, compute_loss=True):
        """Run one split, accumulating metrics over examples and restoring model modes."""
        if loader is None or len(loader) == 0:
            raise ValueError(f"{split} loader is empty or missing")
        if not training:
            _check_eval_loader(loader)
        meter = self._new_metrics()
        total_loss = total_weight = window_weight = 0.0
        sample_indices, offset = [], 0
        started = time.perf_counter()
        model_mode, criterion_mode = self.model.training, self.criterion.training
        self.model.train(training)
        self.criterion.train(training)
        if training:
            self.model.zero_grad(set_to_none=True)
            if callable(getattr(loader.sampler, "set_epoch", None)):
                loader.sampler.set_epoch(epoch)
        description = (
            f"Epoch {epoch}/{self.max_epochs} [{split}]"
            if split in ("train", "val")
            else split.capitalize()
        )
        try:
            with (
                tqdm(
                    loader,
                    desc=description,
                    unit="batch",
                    dynamic_ncols=True,
                    disable=not self.verbose,
                ) as progress,
                torch.enable_grad() if training else torch.inference_mode(),
            ):
                for batch_index, batch in enumerate(progress):
                    images, targets = self._batch(batch)
                    with torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.amp):
                        logits = self.model(images)
                        if not isinstance(logits, Tensor) or logits.shape != (
                            len(targets),
                            self.num_classes,
                        ):
                            raise ValueError(f"Model must return logits [B,{self.num_classes}]")
                        if not torch.isfinite(logits).all():
                            raise FloatingPointError("Model returned nonfinite logits")
                        numerator, weight = (
                            self._loss_terms(logits, targets) if compute_loss else (None, 0.0)
                        )
                    if compute_loss:
                        total_loss += float(numerator.detach())
                        total_weight += weight
                    if training and weight > 0:
                        self.scaler.scale(numerator).backward()
                        window_weight += weight
                    if training and (
                        (batch_index + 1) % self.accumulation_steps == 0
                        or batch_index + 1 == len(loader)
                    ):
                        self._optimizer_step(window_weight)
                        window_weight = 0.0
                    meter.update(logits, targets)
                    valid = (
                        targets != self.ignore_index
                        if self.ignore_index is not None
                        else torch.ones_like(targets, dtype=torch.bool)
                    )
                    sample_indices.extend(
                        (offset + valid.nonzero().flatten().cpu().numpy()).tolist()
                    )
                    offset += len(targets)
                    if compute_loss and total_weight > 0:
                        progress.set_postfix(loss=f"{total_loss / total_weight:.5f}", refresh=False)
                    self._event(
                        "batch_end",
                        split=split,
                        epoch=epoch,
                        batch=batch_index,
                        global_step=self.global_step,
                        loss=float(numerator.detach()) / weight if weight else None,
                    )
            report = meter.compute()
            if compute_loss:
                if total_weight <= 0:
                    raise ValueError(f"{split} has no examples with positive loss weight")
                report["loss"] = total_loss / total_weight
            report["seconds"] = time.perf_counter() - started
            return report, meter, sample_indices
        finally:
            self.model.train(model_mode)
            self.criterion.train(criterion_mode)
