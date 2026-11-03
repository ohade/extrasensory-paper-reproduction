"""Probability-aware, user-aware metrics for the improvement study."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ResearchMetrics:
    macro_balanced_accuracy: float
    per_label_balanced_accuracy: np.ndarray
    macro_brier: float
    macro_ece: float
    user_macro_class_recall_p20: float
    per_user_macro_class_recall: dict[int, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "macro_balanced_accuracy": self.macro_balanced_accuracy,
            "per_label_balanced_accuracy": self.per_label_balanced_accuracy.tolist(),
            "macro_brier": self.macro_brier,
            "macro_ece": self.macro_ece,
            "user_macro_class_recall_p20": self.user_macro_class_recall_p20,
            "per_user_macro_class_recall": {
                str(user): value for user, value in self.per_user_macro_class_recall.items()
            },
        }


def evaluate_probabilities(
    truth: np.ndarray,
    probabilities: np.ndarray,
    users: np.ndarray,
    thresholds: float | np.ndarray = 0.5,
    *,
    valid_mask: np.ndarray | None = None,
    ece_bins: int = 10,
) -> ResearchMetrics:
    truth = np.asarray(truth, dtype=np.uint8)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    users = np.asarray(users)
    if truth.shape != probabilities.shape or truth.ndim != 2:
        raise ValueError("truth and probabilities must be equally shaped 2-D arrays")
    if len(users) != len(truth):
        raise ValueError("users must have one entry per example")
    if valid_mask is None:
        valid_mask = np.ones_like(truth, dtype=bool)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.shape != truth.shape:
            raise ValueError("valid_mask must match truth")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("probabilities must be finite")
    probabilities = np.clip(probabilities, 0.0, 1.0)
    threshold_values = np.asarray(thresholds)
    if threshold_values.ndim <= 1:
        threshold_array = np.broadcast_to(threshold_values, (truth.shape[1],))
    elif threshold_values.shape == truth.shape:
        threshold_array = threshold_values
    else:
        raise ValueError("thresholds must be scalar, per-label, or match truth")
    predictions = probabilities >= threshold_array

    per_label = np.asarray(
        [
            _balanced_accuracy(
                truth[valid_mask[:, label], label],
                predictions[valid_mask[:, label], label],
            )
            for label in range(truth.shape[1])
        ],
        dtype=np.float64,
    )
    brier = np.asarray(
        [
            np.mean(
                (probabilities[valid_mask[:, label], label] - truth[valid_mask[:, label], label])
                ** 2
            )
            if valid_mask[:, label].any()
            else np.nan
            for label in range(truth.shape[1])
        ]
    )
    ece = np.asarray(
        [
            _equal_mass_ece(
                truth[valid_mask[:, label], label],
                probabilities[valid_mask[:, label], label],
                ece_bins,
            )
            if valid_mask[:, label].any()
            else np.nan
            for label in range(truth.shape[1])
        ]
    )
    per_user = {
        int(user): _macro_class_recall(
            truth[users == user],
            predictions[users == user],
            valid_mask[users == user],
        )
        for user in np.unique(users)
    }
    return ResearchMetrics(
        macro_balanced_accuracy=float(np.nanmean(per_label)),
        per_label_balanced_accuracy=per_label,
        macro_brier=float(np.nanmean(brier)),
        macro_ece=float(np.nanmean(ece)),
        user_macro_class_recall_p20=float(np.percentile(list(per_user.values()), 20)),
        per_user_macro_class_recall=per_user,
    )


def cluster_bootstrap_delta(
    truth: np.ndarray,
    candidate_probabilities: np.ndarray,
    baseline_probabilities: np.ndarray,
    users: np.ndarray,
    *,
    thresholds: float | np.ndarray = 0.5,
    baseline_thresholds: float | np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
    replicates: int = 2_000,
    seed: int = 2017,
) -> tuple[float, float, float]:
    """Return observed and 95% user-cluster bootstrap delta in macro BA."""

    truth = np.asarray(truth, dtype=np.uint8)
    candidate_probabilities = np.asarray(candidate_probabilities, dtype=np.float64)
    baseline_probabilities = np.asarray(baseline_probabilities, dtype=np.float64)
    users = np.asarray(users)
    if truth.shape != candidate_probabilities.shape:
        raise ValueError("candidate probabilities must match truth")
    if truth.shape != baseline_probabilities.shape:
        raise ValueError("baseline probabilities must match truth")
    if len(users) != len(truth):
        raise ValueError("users must have one entry per example")
    unique_users = np.unique(users)
    if len(unique_users) < 2:
        raise ValueError("At least two users are required for cluster bootstrap")
    rng = np.random.default_rng(seed)
    if valid_mask is None:
        valid_mask = np.ones_like(truth, dtype=bool)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.shape != truth.shape:
            raise ValueError("valid_mask must match truth")
    candidate_prediction = candidate_probabilities >= _threshold_array(
        thresholds, truth.shape
    )
    baseline_prediction = baseline_probabilities >= _threshold_array(
        thresholds if baseline_thresholds is None else baseline_thresholds,
        truth.shape,
    )
    candidate_counts = _user_confusion_counts(
        truth, candidate_prediction, users, unique_users, valid_mask
    )
    baseline_counts = _user_confusion_counts(
        truth, baseline_prediction, users, unique_users, valid_mask
    )
    observed = _macro_ba_from_counts(candidate_counts.sum(axis=0)) - _macro_ba_from_counts(
        baseline_counts.sum(axis=0)
    )
    sampled_indices = rng.integers(
        0, len(unique_users), size=(replicates, len(unique_users))
    )
    deltas = np.asarray(
        [
            _macro_ba_from_counts(candidate_counts[indices].sum(axis=0))
            - _macro_ba_from_counts(baseline_counts[indices].sum(axis=0))
            for indices in sampled_indices
        ]
    )
    low, high = np.percentile(deltas, [2.5, 97.5])
    return float(observed), float(low), float(high)


def _threshold_array(
    thresholds: float | np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    values = np.asarray(thresholds)
    if values.ndim <= 1:
        return np.broadcast_to(values, (shape[1],))
    if values.shape == shape:
        return values
    raise ValueError("thresholds must be scalar, per-label, or match truth")


def _user_confusion_counts(
    truth: np.ndarray,
    prediction: np.ndarray,
    users: np.ndarray,
    unique_users: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Return per-user, per-label TP/TN/positive/negative counts."""

    counts = np.zeros((len(unique_users), truth.shape[1], 4), dtype=np.int64)
    for user_index, user in enumerate(unique_users):
        rows = users == user
        valid = valid_mask[rows]
        user_truth = truth[rows]
        user_prediction = prediction[rows]
        positives = (user_truth == 1) & valid
        negatives = (user_truth == 0) & valid
        counts[user_index, :, 0] = np.sum(user_prediction & positives, axis=0)
        counts[user_index, :, 1] = np.sum(~user_prediction & negatives, axis=0)
        counts[user_index, :, 2] = np.sum(positives, axis=0)
        counts[user_index, :, 3] = np.sum(negatives, axis=0)
    return counts


def _macro_ba_from_counts(counts: np.ndarray) -> float:
    valid = (counts[:, 2] > 0) & (counts[:, 3] > 0)
    sensitivity = np.divide(
        counts[:, 0],
        counts[:, 2],
        out=np.full(counts.shape[0], np.nan),
        where=counts[:, 2] > 0,
    )
    specificity = np.divide(
        counts[:, 1],
        counts[:, 3],
        out=np.full(counts.shape[0], np.nan),
        where=counts[:, 3] > 0,
    )
    return float(np.nanmean(((sensitivity + specificity) / 2.0)[valid]))


def _balanced_accuracy(truth: np.ndarray, prediction: np.ndarray) -> float:
    positives = truth == 1
    negatives = ~positives
    if not positives.any() or not negatives.any():
        return float("nan")
    sensitivity = np.mean(prediction[positives])
    specificity = np.mean(~prediction[negatives])
    return float((sensitivity + specificity) / 2.0)


def _macro_class_recall(
    truth: np.ndarray,
    prediction: np.ndarray,
    valid: np.ndarray,
) -> float:
    recalls: list[float] = []
    for label in range(truth.shape[1]):
        positives = (truth[:, label] == 1) & valid[:, label]
        negatives = (truth[:, label] == 0) & valid[:, label]
        if positives.any():
            recalls.append(float(np.mean(prediction[positives, label])))
        if negatives.any():
            recalls.append(float(np.mean(~prediction[negatives, label])))
    return float(np.mean(recalls))


def _equal_mass_ece(truth: np.ndarray, probabilities: np.ndarray, bins: int) -> float:
    order = np.argsort(probabilities, kind="stable")
    error = 0.0
    for indices in np.array_split(order, bins):
        if len(indices):
            error += len(indices) / len(truth) * abs(
                float(np.mean(probabilities[indices]) - np.mean(truth[indices]))
            )
    return error
