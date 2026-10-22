"""Whole-split classification metrics using MedMNIST's AUC convention."""

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score


def classification_metrics(targets, probabilities, classes):
    """Measure accuracy, macro-F1, balanced accuracy and binary/macro OvR AUC."""
    targets = np.asarray(targets, dtype=np.int64)
    probabilities = np.asarray(probabilities)
    count = len(classes)
    if (
        targets.ndim != 1
        or not len(targets)
        or probabilities.shape != (len(targets), count)
        or not np.isfinite(probabilities).all()
        or targets.min() < 0
        or targets.max() >= count
    ):
        raise ValueError("Expected nonempty integer labels [N] and finite probabilities [N,C]")
    predicted = probabilities.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        predicted,
        labels=range(count),
        zero_division=0,
    )
    aucs, undefined = [], {}
    for index in [1] if count == 2 else range(count):
        positive = targets == index
        if not positive.any() or positive.all():
            undefined["auc"] = f"Class {index} lacks positive or negative examples"
            break
        aucs.append(roc_auc_score(positive, probabilities[:, index]))
    return {
        "num_samples": len(targets),
        "accuracy": float(np.mean(targets == predicted)),
        "macro_f1": float(f1.mean()),
        "balanced_accuracy": float(recall[support > 0].mean()),
        "auc": None if undefined else float(np.mean(aucs)),
        "class_names": list(classes),
        "confusion_matrix": confusion_matrix(targets, predicted, labels=range(count)).tolist(),
        "per_class": [
            dict(
                name=name,
                precision=float(precision[i]),
                recall=float(recall[i]),
                f1=float(f1[i]),
                support=int(support[i]),
            )
            for i, name in enumerate(classes)
        ],
        "undefined_metrics": undefined,
    }
