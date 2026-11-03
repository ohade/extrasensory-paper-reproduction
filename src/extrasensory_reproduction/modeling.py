"""Paper-style logistic-regression training and user-independent evaluation."""

from __future__ import annotations

import json
import warnings
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from .constants import (
    C_GRID,
    DEFAULT_RANDOM_SEED,
    HEADLINE_LABELS,
    PRETTY_LABELS,
    SENSOR_ORDER,
)
from .data import ExtraSensoryDataset
from .metrics import balanced_accuracy_from_counts, binary_confusion_counts


@dataclass(frozen=True)
class FoldLabelResult:
    model: str
    fold: int
    label: str
    pretty_label: str
    selected_c: float
    validation_f1: float
    train_examples: int
    train_positive: int
    test_examples: int
    test_positive: int
    tp: int
    tn: int
    fp: int
    fn: int


def _standardize(
    train_features: np.ndarray,
    test_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(train_features, axis=0, dtype=np.float64)
        std = np.nanstd(train_features, axis=0, dtype=np.float64)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    scale = np.where(np.isfinite(std) & (std > 0.0), std, 1.0)
    standardized_train = ((train_features - mean) / scale).astype(np.float32)
    standardized_test = ((test_features - mean) / scale).astype(np.float32)
    # The official tutorial imputes zero after standardization, representing
    # the training mean for an unavailable individual derived feature.
    standardized_train[~np.isfinite(standardized_train)] = 0.0
    standardized_test[~np.isfinite(standardized_test)] = 0.0
    return standardized_train, standardized_test


def _fit_with_paper_grid(
    features: np.ndarray,
    labels: np.ndarray,
    seed: int,
    c_grid: Sequence[float],
    max_iter: int,
) -> tuple[LogisticRegression, float, float]:
    internal_train, validation = train_test_split(
        np.arange(len(labels)),
        test_size=1.0 / 3.0,
        stratify=labels,
        random_state=seed,
    )
    best_c = float(c_grid[0])
    best_f1 = -1.0

    for cost in c_grid:
        model = LogisticRegression(
            C=float(cost),
            class_weight="balanced",
            fit_intercept=True,
            solver="liblinear",
            max_iter=max_iter,
            random_state=seed,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(features[internal_train], labels[internal_train])
        validation_prediction = model.predict(features[validation])
        score = float(f1_score(labels[validation], validation_prediction, zero_division=0))
        if score > best_f1:
            best_f1 = score
            best_c = float(cost)

    final_model = LogisticRegression(
        C=best_c,
        class_weight="balanced",
        fit_intercept=True,
        solver="liblinear",
        max_iter=max_iter,
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        final_model.fit(features, labels)
    return final_model, best_c, best_f1


def _evaluate_label(
    *,
    model_name: str,
    fold: int,
    label: str,
    label_index: int,
    train_features: np.ndarray,
    test_features: np.ndarray,
    train_rows: np.ndarray,
    test_rows: np.ndarray,
    all_labels: np.ndarray,
    all_missing: np.ndarray,
    label_policy: str,
    seed: int,
    c_grid: Sequence[float],
    max_iter: int,
) -> FoldLabelResult:
    if label_policy == "exclude-missing":
        train_known = ~all_missing[train_rows, label_index]
        test_known = ~all_missing[test_rows, label_index]
    elif label_policy == "legacy-negative":
        train_known = np.ones(len(train_rows), dtype=bool)
        test_known = np.ones(len(test_rows), dtype=bool)
    else:
        raise ValueError(f"Unknown label policy: {label_policy}")

    label_train_features = train_features[train_known]
    label_test_features = test_features[test_known]
    train_labels = all_labels[train_rows[train_known], label_index]
    test_labels = all_labels[test_rows[test_known], label_index]
    if np.unique(train_labels).size != 2:
        raise ValueError(f"{label} fold {fold} training data does not contain both classes")

    trained_model, selected_c, validation_f1 = _fit_with_paper_grid(
        label_train_features,
        train_labels,
        seed=seed,
        c_grid=c_grid,
        max_iter=max_iter,
    )
    prediction = trained_model.predict(label_test_features)
    counts = binary_confusion_counts(test_labels, prediction)
    return FoldLabelResult(
        model=model_name,
        fold=fold,
        label=label,
        pretty_label=PRETTY_LABELS.get(label, label),
        selected_c=selected_c,
        validation_f1=validation_f1,
        train_examples=len(train_labels),
        train_positive=int(train_labels.sum()),
        test_examples=len(test_labels),
        test_positive=int(test_labels.sum()),
        **counts,
    )


def evaluate_models(
    dataset: ExtraSensoryDataset,
    folds: list[dict[str, np.ndarray]],
    *,
    models: Sequence[str] = ("EF",),
    labels: Sequence[str] = HEADLINE_LABELS,
    label_policy: str = "legacy-negative",
    jobs: int = 1,
    seed: int = DEFAULT_RANDOM_SEED,
    c_grid: Sequence[float] = C_GRID,
    max_iter: int = 1000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    unknown_models = set(models) - {"EF", *SENSOR_ORDER}
    if unknown_models:
        raise ValueError(f"Unsupported models: {sorted(unknown_models)}")

    label_lookup = {label: index for index, label in enumerate(dataset.label_names.tolist())}
    unknown_labels = set(labels) - set(label_lookup)
    if unknown_labels:
        raise ValueError(f"Unknown labels: {sorted(unknown_labels)}")

    core_mask = dataset.core_available
    results: list[FoldLabelResult] = []

    for model_index, model_name in enumerate(models):
        model_available = dataset.sensor_available(model_name) & dataset.paper_labeled
        columns = dataset.sensor_columns(model_name)
        fold_inputs: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

        for fold, split in enumerate(folds):
            train_rows = np.flatnonzero(
                np.isin(dataset.user_index, split["train"]) & model_available
            )
            # All methods are evaluated on the same core-complete examples.
            test_rows = np.flatnonzero(np.isin(dataset.user_index, split["test"]) & core_mask)
            train_features, test_features = _standardize(
                dataset.features[train_rows][:, columns],
                dataset.features[test_rows][:, columns],
            )
            fold_inputs.append((fold, train_rows, test_rows, train_features, test_features))

        # Dispatch all fold-label combinations together. Keeping the five
        # standardized fold arrays in shared joblib memmaps prevents a single
        # slow label at the end of each fold from leaving workers idle.
        model_results = Parallel(
            n_jobs=jobs,
            prefer="processes",
            batch_size=1,
        )(
            delayed(_evaluate_label)(
                model_name=model_name,
                fold=fold,
                label=label,
                label_index=label_lookup[label],
                train_features=train_features,
                test_features=test_features,
                train_rows=train_rows,
                test_rows=test_rows,
                all_labels=dataset.labels,
                all_missing=dataset.label_missing,
                label_policy=label_policy,
                seed=seed + 10_000 * model_index + 100 * fold + label_position,
                c_grid=c_grid,
                max_iter=max_iter,
            )
            for (
                fold,
                train_rows,
                test_rows,
                train_features,
                test_features,
            ) in fold_inputs
            for label_position, label in enumerate(labels)
        )
        results.extend(model_results)

    fold_frame = pd.DataFrame([asdict(result) for result in results])
    summary_rows = []
    for (model_name, label), group in fold_frame.groupby(["model", "label"], sort=False):
        totals = group[["tp", "tn", "fp", "fn"]].sum()
        summary_rows.append(
            {
                "model": model_name,
                "label": label,
                "pretty_label": PRETTY_LABELS.get(label, label),
                "test_examples": int(group["test_examples"].sum()),
                "test_positive": int(group["test_positive"].sum()),
                "tp": int(totals["tp"]),
                "tn": int(totals["tn"]),
                "fp": int(totals["fp"]),
                "fn": int(totals["fn"]),
                "balanced_accuracy": balanced_accuracy_from_counts(**totals.to_dict()),
            }
        )
    summary_frame = pd.DataFrame(summary_rows)
    return fold_frame, summary_frame


def write_results(
    fold_frame: pd.DataFrame,
    summary_frame: pd.DataFrame,
    output_dir: Path,
    metadata: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_frame.to_csv(output_dir / "fold_metrics.csv", index=False)
    summary_frame.to_csv(output_dir / "label_metrics.csv", index=False)
    model_averages = (
        summary_frame.groupby("model", sort=False)["balanced_accuracy"]
        .mean()
        .rename("macro_balanced_accuracy")
        .reset_index()
    )
    model_averages.to_csv(output_dir / "model_averages.csv", index=False)
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
