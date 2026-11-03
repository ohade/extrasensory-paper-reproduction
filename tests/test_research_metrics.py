import numpy as np

from extrasensory_reproduction.research.metrics import (
    cluster_bootstrap_delta,
    evaluate_probabilities,
)


def test_probability_metrics_and_cluster_bootstrap() -> None:
    truth = np.asarray([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.uint8)
    perfect = np.asarray([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])
    baseline = np.full_like(perfect, 0.5)
    users = np.asarray([0, 0, 1, 1])
    metrics = evaluate_probabilities(truth, perfect, users)
    assert metrics.macro_balanced_accuracy == 1.0
    assert metrics.user_macro_class_recall_p20 == 1.0
    observed, low, high = cluster_bootstrap_delta(
        truth, perfect, baseline, users, replicates=20
    )
    assert observed == 0.5
    assert low <= observed <= high


def test_cluster_bootstrap_supports_distinct_thresholds() -> None:
    truth = np.asarray([[1], [0], [1], [0]], dtype=np.uint8)
    candidate = np.asarray([[0.7], [0.4], [0.7], [0.4]])
    baseline = np.asarray([[0.7], [0.4], [0.7], [0.4]])
    users = np.asarray([0, 0, 1, 1])
    observed, _, _ = cluster_bootstrap_delta(
        truth,
        candidate,
        baseline,
        users,
        thresholds=0.6,
        baseline_thresholds=0.8,
        replicates=20,
    )
    assert observed == 0.5
