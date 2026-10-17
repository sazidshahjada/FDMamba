"""Split-level multiclass metrics and exploratory referable-DR measurements.

Predictions are the FIRST argument, matching the loss API. Undefined QWK/AUC or
binary rates are reported as None with reasons; no NaN is written to reports.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score


def _array(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.is_floating_point():
            value = value.double()  # Includes bfloat16, which numpy cannot read directly.
        return value.numpy()
    return np.asarray(value)


def _labels(value, name: str, num_classes: int) -> np.ndarray:
    value = _array(value)
    if value.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array of class indices")
    if value.size and not np.issubdtype(value.dtype, np.integer):
        raise TypeError(f"{name} must contain integer class indices")
    if np.any(value < 0) or np.any(value >= num_classes):
        raise ValueError(f"{name} must contain labels in [0, {num_classes - 1}]")
    return value.astype(np.int64, copy=True)


class ClassificationMetrics:
    """Accumulate batches and compute metrics over all examples in a split.

    ``update(predictions, targets)`` accepts logits/probabilities [B,C] or hard
    predictions [B]. Targets are integer labels [B]; torch tensors and numpy
    arrays/lists are supported. Tensor inputs are detached and stored on CPU.
    Set ``from_logits=False`` only for probability matrices. Use the same input
    form across batches. Call ``reset()`` before the next epoch/split.

    Macro precision/recall/F1 average over ALL configured classes. Per-class
    undefined precision/recall/F1 use zero. Balanced accuracy averages recall
    only over classes represented in the true labels, matching scikit-learn.
    Confusion-matrix rows are true labels; columns are predictions.

    For the exploratory referable endpoint, true grades >= referable_grade are
    positive (default 2). With scores, the prediction is based on SUMMED class
    probability >= referable_threshold, fixed at 0.5 by default. AUC uses that
    continuous sum. With hard grades, the grade boundary supplies the binary
    decision and AUC is unavailable. This does not tune thresholds on test data.
    Set referable_grade=None to disable the binary endpoint.
    """

    def __init__(self, num_classes: int = 5, *, from_logits: bool = True,
                 class_names: Sequence[str] | None = None, ignore_index: int | None = -100,
                 referable_grade: int | None = 2, referable_threshold: float = 0.5) -> None:
        if not isinstance(num_classes, int) or isinstance(num_classes, bool) or num_classes < 2:
            raise ValueError("num_classes must be an integer >= 2")
        if not isinstance(from_logits, bool):
            raise TypeError("from_logits must be a bool")
        names = tuple(str(i) for i in range(num_classes)) if class_names is None else tuple(class_names)
        if (len(names) != num_classes or any(not isinstance(name, str) or not name for name in names)
                or len(set(names)) != num_classes):
            raise ValueError("class_names must contain one unique name per class, in label order")
        if ignore_index is not None and (not isinstance(ignore_index, int) or isinstance(ignore_index, bool)):
            raise TypeError("ignore_index must be an integer or None")
        if referable_grade is not None and (
            not isinstance(referable_grade, int) or isinstance(referable_grade, bool)
            or not 1 <= referable_grade < num_classes
        ):
            raise ValueError("referable_grade must be between 1 and num_classes-1, or None")
        if not 0 <= referable_threshold <= 1:
            raise ValueError("referable_threshold must be in [0, 1]")
        self.num_classes = num_classes
        self.from_logits = from_logits
        self.class_names = names
        self.ignore_index = ignore_index
        self.referable_grade = referable_grade
        self.referable_threshold = float(referable_threshold)
        self.reset()

    def reset(self) -> None:
        """Clear stored examples; configuration is retained."""
        self._targets: list[np.ndarray] = []
        self._predictions: list[np.ndarray] = []
        self._probabilities: list[np.ndarray] = []
        self._has_probabilities: bool | None = None

    def update(self, predictions, targets) -> None:
        """Store one detached batch; metrics are computed later, over the split."""
        targets = _array(targets)
        if targets.ndim != 1 or (targets.size and not np.issubdtype(targets.dtype, np.integer)):
            raise TypeError("targets must be a one-dimensional array of integer labels")
        predictions = _array(predictions)
        if predictions.ndim not in (1, 2) or predictions.shape[0] != len(targets):
            raise ValueError("predictions must be [B] or [B,C], matching targets")
        has_probabilities = predictions.ndim == 2
        if has_probabilities and predictions.shape[1] != self.num_classes:
            raise ValueError("Prediction columns must match num_classes")
        valid = np.ones(len(targets), dtype=bool) if self.ignore_index is None else targets != self.ignore_index
        targets = _labels(targets[valid], "targets", self.num_classes)
        predictions = predictions[valid]
        probabilities = None
        if has_probabilities:
            if not np.issubdtype(predictions.dtype, np.number) or np.iscomplexobj(predictions):
                raise TypeError("Logits/probabilities must contain real numbers")
            predictions = predictions.astype(np.float64, copy=True)
            if not np.isfinite(predictions).all():
                raise ValueError("Logits/probabilities must be finite")
            if self.from_logits:
                # Subtract the maximum before exponentiating. A negative
                # overflow for extreme finite float64 logits correctly gives 0.
                with np.errstate(over='ignore', under='ignore'):
                    scores = np.exp(predictions - predictions.max(axis=1, keepdims=True))
                probabilities = scores / scores.sum(axis=1, keepdims=True)
            else:
                sums = predictions.sum(axis=1, keepdims=True)
                if (np.any(predictions < 0) or np.any(predictions > 1)
                        or not np.allclose(sums, 1, rtol=1e-5, atol=1e-6)):
                    raise ValueError("Probability rows must be nonnegative and sum to one")
                probabilities = predictions / sums
            predictions = probabilities.argmax(axis=1)
        else:
            predictions = _labels(predictions, "predictions", self.num_classes)
        if len(targets) == 0:
            return
        if self._has_probabilities is not None and self._has_probabilities != has_probabilities:
            raise ValueError("Use score matrices for every batch, or hard labels for every batch")
        self._has_probabilities = has_probabilities
        self._targets.append(targets)
        self._predictions.append(predictions.copy())
        if probabilities is not None:
            self._probabilities.append(probabilities)

    def get_predictions(self) -> dict[str, np.ndarray | None]:
        """Return copied CPU arrays in update order, for prediction export."""
        if not self._targets:
            raise ValueError("No nonignored examples have been accumulated")
        return {
            "targets": np.concatenate(self._targets),
            "predictions": np.concatenate(self._predictions),
            "probabilities": np.concatenate(self._probabilities) if self._has_probabilities else None,
        }

    def compute(self) -> dict:
        """Return a JSON-serializable report; this does not clear stored batches."""
        data = self.get_predictions()
        targets, predictions = data["targets"], data["predictions"]
        labels = np.arange(self.num_classes)
        matrix = confusion_matrix(targets, predictions, labels=labels)
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, predictions, labels=labels, zero_division=0,
        )
        undefined = {}
        n = int(len(targets))
        # QWK uses ordinal distances across the complete configured grade axis,
        # including classes absent from this particular validation/test split.
        distances = np.square(labels[:, None] - labels[None, :])
        expected = np.outer(matrix.sum(axis=1).astype(np.float64), matrix.sum(axis=0)) / n
        denominator = float((distances * expected).sum())
        if denominator == 0:
            qwk = None
            undefined["qwk"] = "Expected weighted disagreement is zero (degenerate grade distribution)"
        else:
            qwk = float(1 - (distances * matrix).sum() / denominator)
        report = {
            "num_samples": n,
            "num_classes": self.num_classes,
            "class_names": list(self.class_names),
            "accuracy": float(np.trace(matrix) / n),
            "balanced_accuracy": float(recall[support > 0].mean()),
            "macro_precision": float(precision.mean()),
            "macro_recall": float(recall.mean()),
            "macro_f1": float(f1.mean()),
            "qwk": qwk,
            "confusion_matrix": matrix.tolist(),
            "per_class": [
                {"label": i, "name": self.class_names[i], "precision": float(precision[i]),
                 "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])}
                for i in range(self.num_classes)
            ],
            "undefined_metrics": undefined,
        }
        if self.referable_grade is not None:
            report["referable_dr"] = self._referable(targets, predictions, data["probabilities"], undefined)
        return report

    def _referable(self, targets, predictions, probabilities, undefined) -> dict:
        true = (targets >= self.referable_grade).astype(np.int64)
        if probabilities is None:
            scores = None
            predicted = (predictions >= self.referable_grade).astype(np.int64)
        else:
            scores = probabilities[:, self.referable_grade:].sum(axis=1)
            predicted = (scores >= self.referable_threshold).astype(np.int64)
        matrix = confusion_matrix(true, predicted, labels=[0, 1])
        tn, fp, fn, tp = (int(value) for value in matrix.ravel())
        sensitivity = tp / (tp + fn) if tp + fn else None
        specificity = tn / (tn + fp) if tn + fp else None
        if sensitivity is None:
            undefined["referable_dr.sensitivity"] = "No referable examples in true labels"
        if specificity is None:
            undefined["referable_dr.specificity"] = "No nonreferable examples in true labels"
        if scores is None:
            auc = None
            undefined["referable_dr.roc_auc"] = "Probability scores were not supplied"
        elif not (true == 0).any() or not (true == 1).any():
            auc = None
            undefined["referable_dr.roc_auc"] = "ROC-AUC requires both true binary groups"
        else:
            auc = float(roc_auc_score(true, scores))
        return {
            "grade_boundary": self.referable_grade,
            "threshold": self.referable_threshold if scores is not None else None,
            "decision_rule": "summed_probability" if scores is not None else "predicted_grade",
            "sensitivity": sensitivity, "specificity": specificity, "roc_auc": auc,
            "confusion_matrix": matrix.tolist(),
            "positive_support": tp + fn, "negative_support": tn + fp,
        }


def compute_metrics(predictions, targets, **kwargs) -> dict:
    """Compute a split report from logits/probabilities/hard labels and targets.

    For multiple batches, use ClassificationMetrics.update/compute instead of
    averaging separate reports. Options are the ClassificationMetrics settings.
    """
    metrics = ClassificationMetrics(**kwargs)
    metrics.update(predictions, targets)
    return metrics.compute()
