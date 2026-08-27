"""Classification metric computation for the AML detector.

All metric handling is centralised here so training, tuning and the update
pipeline report identical numbers.
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Compute binary classification metrics with degenerate-case handling.

    Args:
        y_true: Ground-truth binary labels, shape ``(n,)``.
        y_prob: Predicted probability of the positive (laundering) class,
            shape ``(n,)``.
        threshold: Decision threshold applied to ``y_prob`` for F1/precision/
            recall.

    Returns:
        Mapping with ``f1``, ``precision``, ``recall``, ``roc_auc`` and
        ``pr_auc`` keys. AUC metrics are ``nan`` when either class is absent.
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    metrics: dict[str, float] = {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }

    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    else:  # AUC undefined for a single-class batch; keep keys with NaN.
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")

    return metrics


def format_metrics(metrics: dict[str, float]) -> str:
    """Render a metric mapping as a compact one-line string."""
    return " ".join(f"{key}={value:.4f}" for key, value in metrics.items())
