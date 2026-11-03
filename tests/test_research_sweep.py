from dataclasses import replace

import pytest

from extrasensory_reproduction.research.config import ExperimentConfig
from extrasensory_reproduction.research.protocol import AuditAccessError, Stage
from extrasensory_reproduction.research.runner import run_experiment_config
from extrasensory_reproduction.research.sweep import (
    SweepSpec,
    _check_prerequisite,
    audit_sweep,
    decide_acceptance,
    promote_sweep,
    run_sweep,
    single_coordinate_schedule,
)


def _summary(score: float, ece: float = 0.1, user_p20: float = 0.7) -> dict:
    return {
        "metrics": {
            "macro_balanced_accuracy": score,
            "macro_ece": ece,
            "user_macro_class_recall_p20": user_p20,
        }
    }


def test_sweep_changes_one_coordinate_and_enforces_guardrails() -> None:
    config = ExperimentConfig(name="base", model="causal_tcn")
    schedule = list(
        single_coordinate_schedule(
            config,
            {"dropout": [0.1, 0.2], "hidden_dim": [64, 128]},
        )
    )
    assert schedule == [("dropout", 0.1), ("hidden_dim", 64)]
    for field, value in schedule:
        candidate = replace(config, **{field: value})
        changed = [
            key
            for key in config.to_dict()
            if config.to_dict()[key] != candidate.to_dict()[key]
        ]
        assert changed == [field]

    accepted = decide_acceptance(
        _summary(0.78),
        _summary(0.77),
        minimum_improvement=0.002,
        maximum_ece_regression=0.01,
        maximum_user_p20_regression=0.005,
    )
    assert accepted.accepted
    rejected = decide_acceptance(
        _summary(0.78, ece=0.12, user_p20=0.69),
        _summary(0.77),
        minimum_improvement=0.002,
        maximum_ece_regression=0.01,
        maximum_user_p20_regression=0.005,
    )
    assert set(rejected.reasons) == {"calibration_regression", "user_p20_regression"}


def test_direct_audit_execution_is_rejected(tmp_path) -> None:
    with pytest.raises(AuditAccessError, match="gated sweep"):
        run_experiment_config(
            ExperimentConfig(name="blocked", model="paper_lr"),
            Stage.AUDIT,
            output_root=tmp_path,
        )


def test_sweep_state_machine_promotes_then_unlocks_audit(tmp_path) -> None:
    base_path = tmp_path / "base.toml"
    base_path.write_text(
        '[experiment]\nname="base"\nmodel="causal_tcn"\ndropout=0.2\n'
    )
    baseline_path = tmp_path / "baseline.toml"
    baseline_path.write_text(
        '[experiment]\nname="baseline"\nmodel="threshold_lr"\n'
    )
    spec = SweepSpec(
        name="test-sweep",
        base_config=base_path,
        baseline_config=baseline_path,
        search_space={"dropout": [0.1, 0.2]},
        max_trials=1,
        screening_epochs=1,
        final_epochs=2,
    )

    def evaluator(
        config: ExperimentConfig,
        stage: Stage,
        audit_unlocked: bool,
    ) -> dict:
        if stage in {Stage.AUDIT, Stage.OFFICIAL}:
            assert audit_unlocked
        score = 0.76 if config.model == "threshold_lr" else 0.77
        if config.model == "causal_tcn" and config.dropout == 0.1:
            score = 0.78
        return {
            "run_id": f"{config.run_id}-{stage.value}",
            "metrics": {
                "macro_balanced_accuracy": score,
                "macro_ece": 0.1,
                "user_macro_class_recall_p20": 0.7,
            },
            "missing_sensor_metrics": {},
        }

    development = run_sweep(
        spec,
        output_root=tmp_path / "results",
        evaluator=evaluator,
    )
    assert development["status"] == "development_complete"
    assert development["trials"][0]["decision"]["accepted"]

    promotion = promote_sweep(
        spec,
        output_root=tmp_path / "results",
        evaluator=evaluator,
    )
    assert promotion["status"] == "promotion_accepted"
    assert promotion["champion_config"]["epochs"] == 2

    audit = audit_sweep(
        spec,
        output_root=tmp_path / "results",
        evaluator=evaluator,
    )
    assert audit["status"] == "audit_accepted"
    assert set(audit["official"]) == {"candidate", "baseline"}


def test_dependent_sweep_unlocks_from_development_without_promotion(tmp_path) -> None:
    results = tmp_path / "results"
    sweeps = results / "sweeps"
    sweeps.mkdir(parents=True)
    (sweeps / "upstream.json").write_text(
        """
        {
          "status": "development_complete",
          "incumbent_summary": {
            "metrics": {"macro_balanced_accuracy": 0.78},
            "missing_sensor_metrics": {
              "drop": {"macro_balanced_accuracy": 0.76}
            }
          },
          "baseline_summary": {
            "metrics": {"macro_balanced_accuracy": 0.77},
            "missing_sensor_metrics": {
              "drop": {"macro_balanced_accuracy": 0.74}
            }
          }
        }
        """
    )
    spec = SweepSpec(
        name="dependent",
        base_config=tmp_path / "unused-base.toml",
        baseline_config=tmp_path / "unused-baseline.toml",
        search_space={},
        requires_sweep="upstream",
        minimum_prerequisite_improvement=0.003,
        minimum_missing_sensor_reduction=0.20,
    )
    _check_prerequisite(spec, results)


def test_audit_can_skip_redundant_official_baseline(tmp_path) -> None:
    base_path = tmp_path / "base.toml"
    base_path.write_text(
        '[experiment]\nname="base"\nmodel="causal_tcn"\ndropout=0.2\n'
    )
    baseline_path = tmp_path / "baseline.toml"
    baseline_path.write_text(
        '[experiment]\nname="baseline"\nmodel="threshold_lr"\n'
    )
    spec = SweepSpec(
        name="bounded",
        base_config=base_path,
        baseline_config=baseline_path,
        search_space={},
        screening_epochs=1,
        final_epochs=2,
        evaluate_official_baseline=False,
    )
    calls: list[tuple[str, Stage]] = []

    def evaluator(
        config: ExperimentConfig,
        stage: Stage,
        audit_unlocked: bool,
    ) -> dict:
        calls.append((config.model, stage))
        score = 0.76 if config.model == "threshold_lr" else 0.78
        return {
            "run_id": f"{config.run_id}-{stage.value}",
            "metrics": {
                "macro_balanced_accuracy": score,
                "macro_ece": 0.1,
                "user_macro_class_recall_p20": 0.7,
            },
            "missing_sensor_metrics": {},
        }

    run_sweep(spec, output_root=tmp_path / "results", evaluator=evaluator)
    promote_sweep(spec, output_root=tmp_path / "results", evaluator=evaluator)
    state = audit_sweep(
        spec,
        output_root=tmp_path / "results",
        evaluator=evaluator,
    )

    assert state["official"]["baseline"] is None
    assert ("threshold_lr", Stage.OFFICIAL) not in calls
