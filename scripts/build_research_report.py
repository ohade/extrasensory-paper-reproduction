"""Build compact, reviewable result tables from immutable research runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from extrasensory_reproduction.research.metrics import cluster_bootstrap_delta


def load_summaries(run_root: Path) -> list[dict[str, Any]]:
    summaries = []
    for path in sorted(run_root.glob("*/summary.json")):
        summary = json.loads(path.read_text())
        summary["_path"] = path
        summaries.append(summary)
    return summaries


def select(
    summaries: list[dict[str, Any]],
    *,
    run_id: str | None = None,
    model: str | None = None,
    stage: str | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    matches = [
        summary
        for summary in summaries
        if (run_id is None or summary["run_id"] == run_id)
        and (model is None or summary["config"]["model"] == model)
        and (stage is None or summary["stage"] == stage)
        and (name is None or summary["config"]["name"] == name)
    ]
    if len(matches) != 1:
        raise ValueError(
            "Expected one summary for "
            f"run_id={run_id}, model={model}, stage={stage}, name={name}; "
            f"found {len(matches)}"
        )
    return matches[0]


def metric_row(
    label: str,
    summary: dict[str, Any],
    *,
    evidence_scope: str,
    protocol_note: str = "",
) -> dict[str, Any]:
    metrics = summary["metrics"]
    return {
        "model": label,
        "stage": summary["stage"],
        "evidence_scope": evidence_scope,
        "status": "executed",
        "macro_balanced_accuracy": metrics["macro_balanced_accuracy"],
        "macro_ece": metrics["macro_ece"],
        "user_macro_class_recall_p20": metrics["user_macro_class_recall_p20"],
        "elapsed_seconds": summary["elapsed_seconds"],
        "protocol_note": protocol_note,
        "run_id": summary["run_id"],
    }


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.10f")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    result_root = root / "results" / "research"
    summaries = load_summaries(result_root / "runs")

    threshold_dev = select(summaries, model="threshold_lr", stage="development")
    tcn_dev = select(
        summaries,
        model="causal_tcn",
        stage="development",
        name="causal-tcn-local-screening-base",
    )
    moe_dev = select(
        summaries,
        model="sensor_moe_tcn",
        stage="development",
        name="sensor-moe-local-screening-base",
    )
    mlp_dev = select(summaries, model="multitask_mlp", stage="development")
    chain_dev = select(summaries, model="classifier_chain", stage="development")
    tcn_promotion = select(summaries, model="causal_tcn", stage="promotion")
    threshold_promotion = select(summaries, model="threshold_lr", stage="promotion")
    tcn_audit = select(summaries, model="causal_tcn", stage="audit")
    threshold_audit = select(summaries, model="threshold_lr", stage="audit")
    tcn_official = select(summaries, model="causal_tcn", stage="official")

    paper_reproduction = float(
        pd.read_csv(root / "results" / "model_averages.csv")
        .set_index("model")
        .loc["EF", "macro_balanced_accuracy"]
    )
    model_rows = [
        {
            "model": "Paper logistic regression reproduction",
            "stage": "official",
            "evidence_scope": "official five-fold paper protocol",
            "status": "executed",
            "macro_balanced_accuracy": paper_reproduction,
            "macro_ece": np.nan,
            "user_macro_class_recall_p20": np.nan,
            "elapsed_seconds": np.nan,
            "protocol_note": "Existing reproduction; calibration outputs not retained.",
            "run_id": "paper-ef-reproduction",
        },
        metric_row(
            "Threshold-tuned logistic regression",
            threshold_dev,
            evidence_scope="development folds 0-2",
        ),
        metric_row(
            "Multi-task MLP",
            mlp_dev,
            evidence_scope="development folds 0-2",
        ),
        metric_row(
            "Causal TCN",
            tcn_dev,
            evidence_scope="development folds 0-2",
        ),
        metric_row(
            "Sensor-gated mixture of experts",
            moe_dev,
            evidence_scope="development folds 0-2",
            protocol_note=(
                "Promising development-only result; not promoted because the TCN "
                "had already consumed the single promotion fold."
            ),
        ),
        metric_row(
            "Classifier chain",
            chain_dev,
            evidence_scope="development folds 0-2",
        ),
        metric_row(
            "Causal TCN",
            tcn_promotion,
            evidence_scope="held-out promotion fold 4",
        ),
        metric_row(
            "Threshold-tuned logistic regression",
            threshold_promotion,
            evidence_scope="held-out promotion fold 4",
        ),
        metric_row(
            "Causal TCN",
            tcn_audit,
            evidence_scope="sealed audit fold 3",
        ),
        metric_row(
            "Threshold-tuned logistic regression",
            threshold_audit,
            evidence_scope="sealed audit fold 3",
        ),
        metric_row(
            "Causal TCN",
            tcn_official,
            evidence_scope="official five-fold evaluation",
        ),
    ]
    write_csv(pd.DataFrame(model_rows), result_root / "model_comparison.csv")

    ablation_specs = [
        ("Causal TCN", tcn_dev),
        (
            "No label refinement",
            select(
                summaries,
                stage="development",
                name="ablation-causal_tcn_no_label_refinement",
            ),
        ),
        (
            "Short history (5 minutes)",
            select(
                summaries,
                stage="development",
                name="ablation-causal_tcn_short_history",
            ),
        ),
        (
            "GroupDRO",
            select(
                summaries,
                stage="development",
                name="ablation-causal_tcn_group_dro",
            ),
        ),
        (
            "Asymmetric loss",
            select(
                summaries,
                stage="development",
                name="causal-tcn-local-trial-00-asymmetric_gamma_negative",
            ),
        ),
    ]
    reference_ba = tcn_dev["metrics"]["macro_balanced_accuracy"]
    ablation_rows = []
    for label, summary in ablation_specs:
        metrics = summary["metrics"]
        ablation_rows.append(
            {
                "variant": label,
                "macro_balanced_accuracy": metrics["macro_balanced_accuracy"],
                "delta_vs_default_tcn": metrics["macro_balanced_accuracy"]
                - reference_ba,
                "macro_ece": metrics["macro_ece"],
                "user_macro_class_recall_p20": metrics[
                    "user_macro_class_recall_p20"
                ],
                "elapsed_seconds": summary["elapsed_seconds"],
                "run_id": summary["run_id"],
            }
        )
    write_csv(pd.DataFrame(ablation_rows), result_root / "ablation_comparison.csv")

    missing_rows = []
    for label, summary in [
        ("Causal TCN", tcn_dev),
        ("Sensor-gated mixture of experts", moe_dev),
        ("Causal TCN", tcn_official),
    ]:
        clean = summary["metrics"]["macro_balanced_accuracy"]
        for sensor, metrics in summary["missing_sensor_metrics"].items():
            ablated = metrics["macro_balanced_accuracy"]
            missing_rows.append(
                {
                    "model": label,
                    "stage": summary["stage"],
                    "missing_sensor_group": sensor,
                    "clean_macro_balanced_accuracy": clean,
                    "ablated_macro_balanced_accuracy": ablated,
                    "degradation": clean - ablated,
                }
            )
    missing_frame = pd.DataFrame(missing_rows)
    write_csv(missing_frame, result_root / "missing_sensor_comparison.csv")

    labels = tcn_official["labels"]
    tcn_per_label = np.asarray(
        tcn_official["metrics"]["per_label_balanced_accuracy"]
    )
    paper_per_label = (
        pd.read_csv(root / "results" / "label_metrics.csv")
        .query("model == 'EF'")
        .set_index("label")
        .loc[labels, "balanced_accuracy"]
        .to_numpy()
    )
    per_label = pd.DataFrame(
        {
            "label": labels,
            "causal_tcn_balanced_accuracy": tcn_per_label,
            "paper_lr_reproduction_balanced_accuracy": paper_per_label,
            "delta": tcn_per_label - paper_per_label,
        }
    )
    write_csv(per_label, result_root / "official_per_label_comparison.csv")

    tcn_predictions = np.load(
        tcn_audit["_path"].parent / "predictions.npz", allow_pickle=False
    )
    baseline_predictions = np.load(
        threshold_audit["_path"].parent / "predictions.npz", allow_pickle=False
    )
    for key in ("rows", "folds", "users", "truth", "valid"):
        if not np.array_equal(tcn_predictions[key], baseline_predictions[key]):
            raise ValueError(f"Audit prediction arrays differ for {key}")
    audit_observed, audit_ci_low, audit_ci_high = cluster_bootstrap_delta(
        tcn_predictions["truth"],
        tcn_predictions["probabilities"],
        baseline_predictions["probabilities"],
        tcn_predictions["users"],
        thresholds=tcn_predictions["thresholds"],
        baseline_thresholds=baseline_predictions["thresholds"],
        valid_mask=tcn_predictions["valid"],
        replicates=2_000,
    )

    def average_degradation(summary: dict[str, Any]) -> float:
        clean = summary["metrics"]["macro_balanced_accuracy"]
        average_ablated = np.mean(
            [
                metrics["macro_balanced_accuracy"]
                for metrics in summary["missing_sensor_metrics"].values()
            ]
        )
        return float(clean - average_ablated)

    tcn_degradation = average_degradation(tcn_dev)
    moe_degradation = average_degradation(moe_dev)
    robustness_reduction = (tcn_degradation - moe_degradation) / tcn_degradation
    findings = {
        "primary_finding": {
            "candidate": "Causal TCN",
            "baseline": "Paper logistic regression reproduction",
            "official_candidate_macro_balanced_accuracy": tcn_official["metrics"][
                "macro_balanced_accuracy"
            ],
            "official_baseline_macro_balanced_accuracy": paper_reproduction,
            "official_delta": tcn_official["metrics"][
                "macro_balanced_accuracy"
            ]
            - paper_reproduction,
            "official_confidence_interval": None,
            "comparison_note": (
                "The tracked paper reproduction retained metrics but not row-level "
                "probabilities, so a paired official confidence interval cannot be "
                "computed without refitting it."
            ),
            "execution_note": (
                "A threshold-tuned five-fold refit was terminated after a rare-label "
                "liblinear job exceeded four hours. No output from that attempt is "
                "used in the report."
            ),
        },
        "paired_audit_finding": {
            "candidate": "Causal TCN",
            "baseline": "Threshold-tuned logistic regression",
            "candidate_macro_balanced_accuracy": tcn_audit["metrics"][
                "macro_balanced_accuracy"
            ],
            "baseline_macro_balanced_accuracy": threshold_audit["metrics"][
                "macro_balanced_accuracy"
            ],
            "delta": audit_observed,
            "user_cluster_bootstrap_95_ci": [audit_ci_low, audit_ci_high],
        },
        "gates": {
            "promotion_delta": tcn_promotion["metrics"]["macro_balanced_accuracy"]
            - threshold_promotion["metrics"]["macro_balanced_accuracy"],
            "audit_delta": tcn_audit["metrics"]["macro_balanced_accuracy"]
            - threshold_audit["metrics"]["macro_balanced_accuracy"],
            "audit_ece_delta": tcn_audit["metrics"]["macro_ece"]
            - threshold_audit["metrics"]["macro_ece"],
            "audit_user_p20_delta": tcn_audit["metrics"][
                "user_macro_class_recall_p20"
            ]
            - threshold_audit["metrics"]["user_macro_class_recall_p20"],
        },
        "moe_development_only": {
            "macro_balanced_accuracy": moe_dev["metrics"][
                "macro_balanced_accuracy"
            ],
            "delta_vs_causal_tcn": moe_dev["metrics"]["macro_balanced_accuracy"]
            - tcn_dev["metrics"]["macro_balanced_accuracy"],
            "average_missing_sensor_degradation": moe_degradation,
            "tcn_average_missing_sensor_degradation": tcn_degradation,
            "relative_degradation_reduction": robustness_reduction,
            "ssl_gate_required_reduction": 0.20,
            "ssl_gate_passed": bool(robustness_reduction >= 0.20),
            "protocol_note": (
                "Development-only: no second promotion attempt was made after the "
                "TCN consumed fold 4."
            ),
        },
        "bootstrap": {
            "unit": "user",
            "replicates": 2_000,
            "seed": 2017,
            "scope": "sealed audit fold 3",
        },
    }
    (result_root / "final_findings.json").write_text(
        json.dumps(findings, indent=2, sort_keys=True) + "\n"
    )

    capabilities = pd.DataFrame(
        [
            ("Paper logistic regression", True, True, "official"),
            (
                "Threshold-tuned logistic regression",
                True,
                True,
                "development, promotion, sealed audit",
            ),
            ("Elastic-net logistic regression", True, False, "not run: compute budget"),
            ("Histogram gradient boosting", True, False, "not run: compute budget"),
            ("Classifier chain", True, True, "development"),
            ("Multi-task MLP", True, True, "development"),
            ("Causal TCN", True, True, "official"),
            ("GroupDRO", True, True, "development ablation"),
            ("Asymmetric loss", True, True, "development ablation"),
            ("Sensor-gated mixture of experts", True, True, "development"),
            ("Reconstruction pretraining", True, True, "smoke test only; gate blocked"),
            ("Soft probability averaging", True, False, "implementation verified by tests"),
            ("Hard majority voting", True, False, "implementation verified by tests"),
            ("Leakage-safe OOF stacking", True, False, "implementation verified by tests"),
        ],
        columns=["method", "implemented", "full_experiment_executed", "evidence"],
    )
    write_csv(capabilities, result_root / "capability_matrix.csv")

    print(json.dumps(findings, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
