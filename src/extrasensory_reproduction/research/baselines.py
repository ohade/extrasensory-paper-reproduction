"""Leakage-safe tabular baselines and ensemble utilities."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.multioutput import ClassifierChain

from ..modeling import _fit_with_paper_grid
from .config import ExperimentConfig

TABULAR_MODELS = {
    "paper_lr",
    "threshold_lr",
    "elastic_net",
    "hist_gradient_boosting",
    "classifier_chain",
}


@dataclass(frozen=True)
class BaselinePrediction:
    probabilities: np.ndarray
    thresholds: np.ndarray
    ablations: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, features: np.ndarray) -> Standardizer:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.nanmean(features, axis=0, dtype=np.float64)
            scale = np.nanstd(features, axis=0, dtype=np.float64)
        mean = np.where(np.isfinite(mean), mean, 0.0)
        scale = np.where(np.isfinite(scale) & (scale > 0.0), scale, 1.0)
        return cls(mean=mean, scale=scale)

    def transform(self, features: np.ndarray) -> np.ndarray:
        transformed = ((features - self.mean) / self.scale).astype(np.float32)
        transformed[~np.isfinite(transformed)] = 0.0
        return transformed


def fit_predict_baseline(
    config: ExperimentConfig,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    train_missing: np.ndarray,
    train_users: np.ndarray,
    test_features: np.ndarray,
    ablation_columns: dict[str, np.ndarray] | None = None,
) -> BaselinePrediction:
    if config.model not in TABULAR_MODELS:
        raise ValueError(f"{config.model} is not a tabular baseline")
    standardizer = Standardizer.fit(train_features)
    standardized_train = standardizer.transform(train_features)
    standardized_test = standardizer.transform(test_features)
    ablation_features = {}
    for name, columns in (ablation_columns or {}).items():
        masked = standardized_test.copy()
        masked[:, columns] = 0.0
        ablation_features[name] = masked

    if config.model == "classifier_chain":
        if config.label_policy != "legacy-negative":
            raise ValueError("classifier_chain currently requires legacy-negative labels")
        estimator = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            solver="liblinear",
            max_iter=1_000,
            random_state=config.seed,
        )
        chain = ClassifierChain(estimator=estimator, order="random", random_state=config.seed)
        chain.fit(standardized_train, train_labels)
        probabilities = np.asarray(chain.predict_proba(standardized_test), dtype=np.float64)
        ablations = {
            name: np.asarray(chain.predict_proba(features), dtype=np.float64)
            for name, features in ablation_features.items()
        }
        return BaselinePrediction(
            probabilities=probabilities,
            thresholds=np.full(train_labels.shape[1], config.threshold),
            ablations=ablations,
        )

    label_results = Parallel(
        n_jobs=config.max_workers,
        prefer="processes",
        batch_size=1,
    )(
        delayed(_fit_predict_label)(
            config,
            label,
            standardized_train,
            train_labels[:, label],
            train_missing[:, label],
            train_users,
            standardized_test,
            ablation_features,
        )
        for label in range(train_labels.shape[1])
    )
    return BaselinePrediction(
        probabilities=np.column_stack([result[0] for result in label_results]),
        thresholds=np.asarray([result[1] for result in label_results]),
        ablations={
            name: np.column_stack([result[2][name] for result in label_results])
            for name in ablation_features
        },
    )


def combine_probability_sets(
    probability_sets: list[np.ndarray],
    *,
    method: str,
) -> np.ndarray:
    if len(probability_sets) < 2:
        raise ValueError("At least two probability sets are required")
    stacked = np.stack(probability_sets)
    if method == "average":
        return stacked.mean(axis=0)
    if method == "hard_vote":
        return (stacked >= 0.5).mean(axis=0)
    raise ValueError(f"Unknown ensemble method: {method}")


def fit_oof_stacker(
    oof_probability_sets: list[np.ndarray],
    truth: np.ndarray,
    test_probability_sets: list[np.ndarray],
    *,
    seed: int = 2017,
) -> np.ndarray:
    if len(oof_probability_sets) != len(test_probability_sets):
        raise ValueError("OOF and test base prediction sets must align")
    oof = np.stack(oof_probability_sets, axis=-1)
    test = np.stack(test_probability_sets, axis=-1)
    output = np.empty(test.shape[:2], dtype=np.float64)
    for label in range(truth.shape[1]):
        model = LogisticRegression(
            class_weight="balanced",
            solver="liblinear",
            random_state=seed + label,
        )
        model.fit(oof[:, label, :], truth[:, label])
        output[:, label] = model.predict_proba(test[:, label, :])[:, 1]
    return output


def _fit_predict_label(
    config: ExperimentConfig,
    label: int,
    train_features: np.ndarray,
    labels: np.ndarray,
    missing: np.ndarray,
    groups: np.ndarray,
    test_features: np.ndarray,
    ablation_features: dict[str, np.ndarray],
) -> tuple[np.ndarray, float, dict[str, np.ndarray]]:
    known = ~missing if config.label_policy == "exclude-missing" else np.ones(len(labels), dtype=bool)
    label_features = train_features[known]
    label_truth = labels[known]
    label_groups = groups[known]
    if np.unique(label_truth).size != 2:
        probability = float(np.mean(label_truth))
        return (
            np.full(len(test_features), probability),
            config.threshold,
            {
                name: np.full(len(features), probability)
                for name, features in ablation_features.items()
            },
        )

    seed = config.seed + label
    if config.model == "paper_lr":
        model, _, _ = _fit_with_paper_grid(
            label_features,
            label_truth,
            seed=seed,
            c_grid=(0.001, 0.01, 0.1, 1.0, 10.0, 100.0),
            max_iter=1_000,
        )
        threshold = config.threshold
    else:
        factory = _model_factory(config.model, seed)
        threshold = _group_oof_threshold(
            factory,
            label_features,
            label_truth,
            label_groups,
        )
        model = factory()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(label_features, label_truth)
    probabilities = np.asarray(model.predict_proba(test_features))[:, 1]
    ablations = {
        name: np.asarray(model.predict_proba(features))[:, 1]
        for name, features in ablation_features.items()
    }
    return probabilities, threshold, ablations


def _model_factory(model_name: str, seed: int) -> Callable[[], BaseEstimator]:
    if model_name == "threshold_lr":
        return lambda: LogisticRegression(
            C=1.0,
            class_weight="balanced",
            solver="liblinear",
            max_iter=1_000,
            random_state=seed,
        )
    if model_name == "elastic_net":
        return lambda: LogisticRegression(
            C=1.0,
            class_weight="balanced",
            penalty="elasticnet",
            l1_ratio=0.5,
            solver="saga",
            max_iter=500,
            n_jobs=1,
            random_state=seed,
        )
    if model_name == "hist_gradient_boosting":
        return lambda: HistGradientBoostingClassifier(
            class_weight="balanced",
            learning_rate=0.05,
            max_iter=100,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=seed,
        )
    raise ValueError(f"Unsupported independent baseline: {model_name}")


def _group_oof_threshold(
    factory: Callable[[], BaseEstimator],
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
) -> float:
    unique_groups = np.unique(groups)
    splits = min(3, len(unique_groups))
    if splits < 2:
        return 0.5
    oof = np.full(len(labels), np.nan, dtype=np.float64)
    for train, validation in GroupKFold(n_splits=splits).split(features, labels, groups):
        if np.unique(labels[train]).size != 2:
            continue
        model = factory()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(features[train], labels[train])
        oof[validation] = np.asarray(model.predict_proba(features[validation]))[:, 1]
    valid = np.isfinite(oof)
    if not valid.any():
        return 0.5
    grid = np.linspace(0.05, 0.95, 37)
    scores = [
        balanced_accuracy_score(labels[valid], oof[valid] >= threshold)
        for threshold in grid
    ]
    return float(grid[int(np.argmax(scores))])
