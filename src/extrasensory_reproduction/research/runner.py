"""Execute immutable, stage-gated ExtraSensory research experiments."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..constants import (
    DEFAULT_CACHE_PATH,
    DEFAULT_FOLDS_DIR,
    DEFAULT_ORIGINAL_LABELS_DIR,
    DEFAULT_RAW_DIR,
    HEADLINE_LABELS,
    SENSOR_ORDER,
)
from ..data import load_official_folds, load_or_prepare
from .baselines import TABULAR_MODELS, fit_predict_baseline
from .config import ExperimentConfig
from .ledger import ExperimentStore
from .metrics import evaluate_probabilities
from .protocol import AuditAccessError, Stage, assert_rows_respect_split, build_stage_splits


def run_experiment(
    config_path: Path,
    stage: Stage | str,
    *,
    output_root: Path = Path("results/research"),
    cache_path: Path = DEFAULT_CACHE_PATH,
    raw_dir: Path = DEFAULT_RAW_DIR,
    original_labels_dir: Path = DEFAULT_ORIGINAL_LABELS_DIR,
    folds_dir: Path = DEFAULT_FOLDS_DIR,
) -> dict[str, Any]:
    config = ExperimentConfig.from_toml(config_path)
    return run_experiment_config(
        config,
        stage,
        output_root=output_root,
        cache_path=cache_path,
        raw_dir=raw_dir,
        original_labels_dir=original_labels_dir,
        folds_dir=folds_dir,
    )


def run_experiment_config(
    config: ExperimentConfig,
    stage: Stage | str,
    *,
    output_root: Path = Path("results/research"),
    cache_path: Path = DEFAULT_CACHE_PATH,
    raw_dir: Path = DEFAULT_RAW_DIR,
    original_labels_dir: Path = DEFAULT_ORIGINAL_LABELS_DIR,
    folds_dir: Path = DEFAULT_FOLDS_DIR,
    audit_unlocked: bool = False,
) -> dict[str, Any]:
    selected_stage = Stage(stage)
    if selected_stage in {Stage.AUDIT, Stage.OFFICIAL} and not audit_unlocked:
        raise AuditAccessError(
            f"{selected_stage.value} evaluation is only available through the gated sweep workflow"
        )
    store = ExperimentStore(output_root)
    run_id = f"{config.run_id}-{selected_stage.value}"
    run_dir = store.run_dir(run_id)
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text())

    store.append(
        {
            "run_id": run_id,
            "status": "started",
            "stage": selected_stage.value,
            "config": config.to_dict(),
        }
    )
    started = time.perf_counter()
    try:
        dataset = load_or_prepare(cache_path, raw_dir, original_labels_dir)
        folds = load_official_folds(dataset.user_ids.tolist(), folds_dir)
        label_lookup = {
            label: index for index, label in enumerate(dataset.label_names.tolist())
        }
        label_columns = np.asarray([label_lookup[label] for label in HEADLINE_LABELS])
        fold_outputs: list[dict[str, Any]] = []
        all_truth: list[np.ndarray] = []
        all_valid: list[np.ndarray] = []
        all_probabilities: list[np.ndarray] = []
        all_thresholds: list[np.ndarray] = []
        all_users: list[np.ndarray] = []
        all_rows: list[np.ndarray] = []
        all_fold_ids: list[np.ndarray] = []
        all_ablation_probabilities: dict[str, list[np.ndarray]] = {}

        for split in build_stage_splits(
            folds, selected_stage, audit_unlocked=audit_unlocked
        ):
            train_rows = np.flatnonzero(
                np.isin(dataset.user_index, split.train_users)
                & dataset.paper_labeled
                & dataset.sensor_available("EF")
            )
            test_rows = np.flatnonzero(
                np.isin(dataset.user_index, split.test_users) & dataset.core_available
            )
            assert_rows_respect_split(
                dataset.user_index, train_rows, split.train_users, purpose="train"
            )
            assert_rows_respect_split(
                dataset.user_index, test_rows, split.test_users, purpose="test"
            )

            if config.model in TABULAR_MODELS:
                individual_columns = {
                    sensor: dataset.sensor_columns(sensor) for sensor in SENSOR_ORDER
                }
                ablation_columns = {
                    **individual_columns,
                    "Acc+Gyro": np.concatenate(
                        [individual_columns["Acc"], individual_columns["Gyro"]]
                    ),
                    "WAcc+Loc": np.concatenate(
                        [individual_columns["WAcc"], individual_columns["Loc"]]
                    ),
                    "Aud+PS": np.concatenate(
                        [individual_columns["Aud"], individual_columns["PS"]]
                    ),
                }
                prediction = fit_predict_baseline(
                    config,
                    dataset.features[train_rows],
                    dataset.labels[train_rows][:, label_columns],
                    dataset.label_missing[train_rows][:, label_columns],
                    dataset.user_index[train_rows],
                    dataset.features[test_rows],
                    ablation_columns,
                )
            else:
                from .neural import fit_predict_neural

                prediction = fit_predict_neural(
                    config,
                    dataset,
                    train_rows,
                    test_rows,
                    label_columns,
                )

            truth = dataset.labels[test_rows][:, label_columns]
            label_valid = (
                ~dataset.label_missing[test_rows][:, label_columns]
                if config.label_policy == "exclude-missing"
                else np.ones_like(truth, dtype=bool)
            )
            users = dataset.user_index[test_rows]
            metrics = evaluate_probabilities(
                truth,
                prediction.probabilities,
                users,
                prediction.thresholds,
                valid_mask=label_valid,
            )
            ablation_metrics = {
                name: evaluate_probabilities(
                    truth,
                    probabilities,
                    users,
                    prediction.thresholds,
                    valid_mask=label_valid,
                ).to_dict()
                for name, probabilities in prediction.ablations.items()
            }
            fold_outputs.append(
                {
                    "fold": split.fold,
                    "train_users": len(split.train_users),
                    "test_users": len(split.test_users),
                    "train_examples": len(train_rows),
                    "test_examples": len(test_rows),
                    "thresholds": prediction.thresholds.tolist(),
                    "metrics": metrics.to_dict(),
                    "missing_sensor_metrics": ablation_metrics,
                }
            )
            all_truth.append(truth)
            all_valid.append(label_valid)
            all_probabilities.append(prediction.probabilities)
            all_thresholds.append(
                np.broadcast_to(prediction.thresholds, prediction.probabilities.shape)
            )
            all_users.append(users)
            all_rows.append(test_rows)
            all_fold_ids.append(np.full(len(test_rows), split.fold, dtype=np.int8))
            for name, probabilities in prediction.ablations.items():
                all_ablation_probabilities.setdefault(name, []).append(probabilities)

        combined_truth = np.concatenate(all_truth)
        combined_valid = np.concatenate(all_valid)
        combined_probabilities = np.concatenate(all_probabilities)
        combined_thresholds = np.concatenate(all_thresholds)
        combined_users = np.concatenate(all_users)
        combined_metrics = evaluate_probabilities(
            combined_truth,
            combined_probabilities,
            combined_users,
            combined_thresholds,
            valid_mask=combined_valid,
        )
        combined_ablation_metrics = {
            name: evaluate_probabilities(
                combined_truth,
                np.concatenate(probability_parts),
                combined_users,
                combined_thresholds,
                valid_mask=combined_valid,
            ).to_dict()
            for name, probability_parts in all_ablation_probabilities.items()
        }
        elapsed = time.perf_counter() - started
        summary = {
            "run_id": run_id,
            "stage": selected_stage.value,
            "status": "completed",
            "config": config.to_dict(),
            "labels": list(HEADLINE_LABELS),
            "folds": fold_outputs,
            "metrics": combined_metrics.to_dict(),
            "missing_sensor_metrics": combined_ablation_metrics,
            "elapsed_seconds": elapsed,
        }
        np.savez_compressed(
            run_dir / "predictions.npz",
            rows=np.concatenate(all_rows),
            folds=np.concatenate(all_fold_ids),
            users=combined_users,
            truth=combined_truth,
            valid=combined_valid,
            probabilities=combined_probabilities.astype(np.float32),
            thresholds=combined_thresholds.astype(np.float32),
        )
        store.write_json(summary_path, summary)
        store.append(
            {
                "run_id": run_id,
                "status": "completed",
                "stage": selected_stage.value,
                "macro_balanced_accuracy": combined_metrics.macro_balanced_accuracy,
                "macro_ece": combined_metrics.macro_ece,
                "user_p20": combined_metrics.user_macro_class_recall_p20,
                "elapsed_seconds": elapsed,
            }
        )
        return summary
    except Exception as error:
        store.append(
            {
                "run_id": run_id,
                "status": "failed",
                "stage": selected_stage.value,
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        raise


def summarize_experiments(
    output_root: Path = Path("results/research"),
    *,
    experiment: str | None = None,
) -> list[dict[str, Any]]:
    records = [
        record
        for record in ExperimentStore(output_root).records()
        if record.get("status") == "completed"
    ]
    if experiment is not None:
        records = [record for record in records if str(record.get("run_id", "")).startswith(experiment)]
    return records
