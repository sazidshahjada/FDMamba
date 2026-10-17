"""Multiclass classification losses for raw logits and integer DR grades.

BalancedCrossEntropy uses inverse-frequency TRAIN class weights. FocalLoss is
the softmax multiclass adaptation of Lin et al. (https://arxiv.org/abs/1708.02002).
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _vector(values: Sequence[float] | Tensor, name: str, *, positive=False) -> Tensor:
    try:
        result = torch.as_tensor(values).detach().clone()
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{name} must be a vector ordered by class index") from exc
    if result.is_complex() or result.ndim != 1 or result.numel() == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional vector")
    result = result.to(dtype=torch.float64 if positive or result.dtype == torch.float64 else torch.float32)
    if not torch.isfinite(result).all() or (result <= 0 if positive else result < 0).any():
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must contain finite {qualifier} values")
    return result


def _settings(reduction: str, ignore_index: int) -> None:
    if reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be 'none', 'mean' or 'sum'")
    if not isinstance(ignore_index, int) or isinstance(ignore_index, bool):
        raise TypeError("ignore_index must be an integer")


def _inputs(logits: Tensor, targets: Tensor) -> Tensor:
    if not isinstance(logits, Tensor) or not torch.is_floating_point(logits):
        raise TypeError("logits must be a floating-point tensor")
    if logits.ndim != 2 or logits.shape[1] == 0:
        raise ValueError("logits must have shape [batch, num_classes]")
    if not isinstance(targets, Tensor) or targets.dtype != torch.long:
        raise TypeError("targets must be a torch.long tensor of class indices")
    if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
        raise ValueError("targets must have shape [batch] matching logits")
    if targets.device != logits.device:
        raise ValueError("logits and targets must be on the same device")
    # AMP-safe probability/log-loss arithmetic; preserve double precision for
    # numerical checks. Gradients still flow to half/bfloat16 model outputs.
    return logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits


def _reduce(losses: Tensor, targets: Tensor, reduction: str, ignore_index: int,
            class_weights: Tensor | None = None) -> Tensor:
    if reduction == "none":
        return losses
    total = losses.sum()
    if reduction == "sum":
        return total
    valid = targets != ignore_index
    if class_weights is None:
        denominator = valid.sum().to(losses.dtype)
    else:
        safe_targets = targets.masked_fill(~valid, 0)
        denominator = (class_weights[safe_targets] * valid).sum()
    # An all-ignored/empty batch or one with zero effective CE weight produces
    # differentiable zero rather than division by zero.
    safe_denominator = denominator.clamp_min(torch.finfo(losses.dtype).tiny)
    return torch.where(denominator > 0, total / safe_denominator, total * 0)


class CrossEntropy(nn.CrossEntropyLoss):
    """Cross-entropy for logits [B,C] and torch.long labels [B].

    Supports optional nonnegative class ``weight``, label smoothing, ignored
    labels and all three reductions. Weighted mean divides by the sum of target
    weights, following PyTorch. An all-ignored or zero-effective-weight batch
    returns zero. No softmax should be applied before calling this loss.
    """

    def __init__(self, weight: Sequence[float] | Tensor | None = None,
                 ignore_index: int = -100, reduction: str = "mean",
                 label_smoothing: float = 0.0) -> None:
        _settings(reduction, ignore_index)
        if not isinstance(label_smoothing, Real) or not 0 <= label_smoothing <= 1:
            raise ValueError("label_smoothing must be in [0, 1]")
        weights = None if weight is None else _vector(weight, "weight")
        super().__init__(weight=weights, ignore_index=ignore_index,
                         reduction=reduction, label_smoothing=float(label_smoothing))

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        logits = _inputs(logits, targets)
        weight = self.weight
        if weight is not None:
            if weight.numel() != logits.shape[1]:
                raise ValueError("weight must have one entry per logit class")
            weight = weight.to(device=logits.device, dtype=logits.dtype)
        losses = F.cross_entropy(logits, targets, weight=weight, reduction="none",
                                 ignore_index=self.ignore_index,
                                 label_smoothing=self.label_smoothing)
        return _reduce(losses, targets, self.reduction, self.ignore_index, weight)


class BalancedCrossEntropy(CrossEntropy):
    """Cross-entropy weighted by inverse TRAIN class frequency.

    ``class_counts[c]`` is the number of training examples with label c.
    The fixed weights are ``sum(class_counts) / (C * class_counts[c])``.
    Supply counts in label-index order, not minibatch counts. Every class must
    have a positive training count; zero-count classes fail explicitly.

    ``from_dataset(loaders['train'].dataset)`` obtains counts from ImageFolder's
    existing metadata without reading image pixels. Counts and weights are
    registered buffers, so they are saved in state_dict and follow .to(device).
    """

    def __init__(self, class_counts: Sequence[float] | Tensor,
                 ignore_index: int = -100, reduction: str = "mean",
                 label_smoothing: float = 0.0) -> None:
        counts = _vector(class_counts, "class_counts", positive=True).to(torch.float64)
        weights = (counts.sum() / (counts.numel() * counts)).to(torch.float32)
        super().__init__(weight=weights, ignore_index=ignore_index,
                         reduction=reduction, label_smoothing=label_smoothing)
        self.register_buffer("class_counts", counts)

    @classmethod
    def from_dataset(cls, dataset, **kwargs) -> BalancedCrossEntropy:
        """Build once from the TRAIN ImageFolder dataset's labels/class order."""
        if not hasattr(dataset, "classes") or not hasattr(dataset, "targets"):
            raise TypeError("dataset must expose ImageFolder-style classes and targets")
        labels = torch.as_tensor(dataset.targets)
        num_classes = len(dataset.classes)
        if (labels.ndim != 1 or labels.numel() == 0 or num_classes == 0
                or labels.is_floating_point() or labels.is_complex() or labels.dtype == torch.bool
                or (labels < 0).any() or (labels >= num_classes).any()):
            raise ValueError("Training targets must be integer indices within dataset.classes")
        counts = torch.bincount(labels.to(torch.long), minlength=num_classes)
        return cls(counts, **kwargs)


class FocalLoss(nn.Module):
    """Softmax multiclass focal loss: -alpha_t * (1-p_t)**gamma * log(p_t).

    ``gamma`` is finite and nonnegative. ``alpha`` is None, a nonnegative scalar
    applied uniformly, or a nonnegative vector in label-index order. It is a
    fixed buffer, not a trainable parameter. p_t is the UNWEIGHTED target-class
    probability; alpha is applied after the focusing factor is computed.

    Mean reduction averages over nonignored examples, including alpha weighting
    in the numerator. With gamma=0 and alpha=None this equals CrossEntropy;
    with per-class alpha the mean is a sample average, rather than CE's average
    normalized by the sum of class weights. All-ignored/empty batches return 0.
    """

    def __init__(self, gamma: float = 2.0,
                 alpha: float | Sequence[float] | Tensor | None = None,
                 reduction: str = "mean", ignore_index: int = -100) -> None:
        super().__init__()
        _settings(reduction, ignore_index)
        if not isinstance(gamma, Real) or not math.isfinite(gamma) or gamma < 0:
            raise ValueError("gamma must be finite and nonnegative")
        if alpha is None:
            weights = None
        elif isinstance(alpha, Real) or isinstance(alpha, Tensor) and alpha.ndim == 0:
            scalar = float(alpha.detach()) if isinstance(alpha, Tensor) else float(alpha)
            if not math.isfinite(scalar) or scalar < 0:
                raise ValueError("alpha must be finite and nonnegative")
            weights = torch.tensor(scalar, dtype=torch.float32)
        else:
            weights = _vector(alpha, "alpha")
        self.gamma = float(gamma)
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.register_buffer("alpha", weights)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        logits = _inputs(logits, targets)
        ce = F.cross_entropy(logits, targets, reduction="none", ignore_index=self.ignore_index)
        # -expm1(-CE) computes 1-p_t accurately for confident predictions. The
        # tiny clamp also keeps fractional-gamma gradients finite at p_t == 1.
        one_minus_pt = (-torch.expm1(-ce)).clamp_min(torch.finfo(ce.dtype).tiny)
        losses = one_minus_pt.pow(self.gamma) * ce
        if self.alpha is not None:
            alpha = self.alpha.to(device=logits.device, dtype=logits.dtype)
            if alpha.ndim == 1:
                if alpha.numel() != logits.shape[1]:
                    raise ValueError("alpha must have one entry per logit class")
                safe_targets = targets.masked_fill(targets == self.ignore_index, 0)
                alpha = alpha[safe_targets]
            losses = alpha * losses
        return _reduce(losses, targets, self.reduction, self.ignore_index)
