"""Metrics with the aggregation order used in the paper."""

from __future__ import annotations

import numpy as np


def binary_confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    return {
        "tp": int(np.sum(y_true & y_pred)),
        "tn": int(np.sum(~y_true & ~y_pred)),
        "fp": int(np.sum(~y_true & y_pred)),
        "fn": int(np.sum(y_true & ~y_pred)),
    }


def balanced_accuracy_from_counts(tp: int, tn: int, fp: int, fn: int) -> float:
    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    return float((sensitivity + specificity) / 2.0)
