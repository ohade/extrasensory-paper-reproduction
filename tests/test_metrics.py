import numpy as np

from extrasensory_reproduction.metrics import (
    balanced_accuracy_from_counts,
    binary_confusion_counts,
)


def test_confusion_counts_and_balanced_accuracy() -> None:
    truth = np.array([1, 1, 0, 0, 0])
    prediction = np.array([1, 0, 1, 0, 0])
    counts = binary_confusion_counts(truth, prediction)
    assert counts == {"tp": 1, "tn": 2, "fp": 1, "fn": 1}
    assert balanced_accuracy_from_counts(**counts) == (0.5 + 2 / 3) / 2
