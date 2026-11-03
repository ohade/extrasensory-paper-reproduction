import numpy as np

from extrasensory_reproduction.research.baselines import (
    combine_probability_sets,
    fit_oof_stacker,
    fit_predict_baseline,
)
from extrasensory_reproduction.research.config import ExperimentConfig


def test_threshold_baseline_and_ensembles() -> None:
    rng = np.random.default_rng(2017)
    features = rng.normal(size=(80, 4)).astype(np.float32)
    users = np.repeat(np.arange(8), 10)
    labels = np.column_stack(
        [
            features[:, 0] + features[:, 1] > 0,
            features[:, 2] - features[:, 3] > 0,
        ]
    ).astype(np.uint8)
    prediction = fit_predict_baseline(
        ExperimentConfig(
            name="test",
            model="threshold_lr",
            max_workers=1,
        ),
        features[:60],
        labels[:60],
        np.zeros_like(labels[:60], dtype=bool),
        users[:60],
        features[60:],
        {"first": np.asarray([0])},
    )
    assert prediction.probabilities.shape == (20, 2)
    assert np.all((prediction.thresholds > 0) & (prediction.thresholds < 1))
    assert prediction.ablations["first"].shape == (20, 2)

    first = np.clip(prediction.probabilities, 0.01, 0.99)
    second = 1.0 - first
    assert np.allclose(combine_probability_sets([first, second], method="average"), 0.5)
    stacked = fit_oof_stacker(
        [np.tile([[0.8, 0.2], [0.2, 0.8]], (30, 1)), labels[:60] * 0.8 + 0.1],
        labels[:60],
        [first, np.clip(first * 0.8 + 0.1, 0, 1)],
    )
    assert stacked.shape == (20, 2)
