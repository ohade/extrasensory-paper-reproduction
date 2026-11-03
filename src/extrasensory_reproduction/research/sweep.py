"""Bounded config-only autoresearch loop with promotion and audit gates."""

from __future__ import annotations

import json
import time
import tomllib
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .ledger import ExperimentStore
from .protocol import AuditAccessError, Stage
from .runner import run_experiment_config

Evaluator = Callable[[ExperimentConfig, Stage, bool], dict[str, Any]]


@dataclass(frozen=True)
class SweepSpec:
    name: str
    base_config: Path
    baseline_config: Path
    search_space: dict[str, list[Any]]
    max_trials: int = 30
    budget_hours: float = 12.0
    minimum_improvement: float = 0.002
    maximum_ece_regression: float = 0.01
    maximum_user_p20_regression: float = 0.005
    promotion_improvement: float = 0.003
    screening_epochs: int = 3
    final_epochs: int = 12
    requires_sweep: str | None = None
    minimum_prerequisite_improvement: float = 0.0
    minimum_missing_sensor_reduction: float = 0.0
    evaluate_official_baseline: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Sweep name must not be empty")
        if not 1 <= self.max_trials <= 30:
            raise ValueError("max_trials must be between 1 and 30")
        if not 0 < self.budget_hours <= 12:
            raise ValueError("budget_hours must be in (0, 12]")
        if not 1 <= self.screening_epochs <= self.final_epochs:
            raise ValueError("screening_epochs must be positive and <= final_epochs")
        allowed = set(ExperimentConfig.__dataclass_fields__) - {"name", "model"}
        unknown = set(self.search_space) - allowed
        if unknown:
            raise ValueError(f"Unsupported sweep parameters: {sorted(unknown)}")
        if any(not values for values in self.search_space.values()):
            raise ValueError("Every search parameter must have at least one value")

    @classmethod
    def from_toml(cls, path: Path) -> SweepSpec:
        with path.open("rb") as handle:
            payload = tomllib.load(handle)
        sweep = payload.get("sweep")
        search_space = payload.get("space")
        if not isinstance(sweep, dict) or not isinstance(search_space, dict):
            raise TypeError("Sweep spec requires [sweep] and [space] tables")

        def resolve(value: str) -> Path:
            candidate = Path(value)
            if candidate.exists():
                return candidate
            relative = path.parent / candidate
            if relative.exists():
                return relative
            raise FileNotFoundError(f"Sweep config not found: {value}")

        return cls(
            name=str(sweep["name"]),
            base_config=resolve(str(sweep["base_config"])),
            baseline_config=resolve(str(sweep["baseline_config"])),
            search_space={key: list(values) for key, values in search_space.items()},
            max_trials=int(sweep.get("max_trials", 30)),
            budget_hours=float(sweep.get("budget_hours", 12.0)),
            minimum_improvement=float(sweep.get("minimum_improvement", 0.002)),
            maximum_ece_regression=float(sweep.get("maximum_ece_regression", 0.01)),
            maximum_user_p20_regression=float(
                sweep.get("maximum_user_p20_regression", 0.005)
            ),
            promotion_improvement=float(sweep.get("promotion_improvement", 0.003)),
            screening_epochs=int(sweep.get("screening_epochs", 3)),
            final_epochs=int(sweep.get("final_epochs", 12)),
            requires_sweep=sweep.get("requires_sweep"),
            minimum_prerequisite_improvement=float(
                sweep.get("minimum_prerequisite_improvement", 0.0)
            ),
            minimum_missing_sensor_reduction=float(
                sweep.get("minimum_missing_sensor_reduction", 0.0)
            ),
            evaluate_official_baseline=bool(
                sweep.get("evaluate_official_baseline", True)
            ),
        )


@dataclass(frozen=True)
class AcceptanceDecision:
    accepted: bool
    score_delta: float
    ece_delta: float
    user_p20_delta: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "score_delta": self.score_delta,
            "ece_delta": self.ece_delta,
            "user_p20_delta": self.user_p20_delta,
            "reasons": list(self.reasons),
        }


def decide_acceptance(
    candidate: dict[str, Any],
    incumbent: dict[str, Any],
    *,
    minimum_improvement: float,
    maximum_ece_regression: float,
    maximum_user_p20_regression: float,
) -> AcceptanceDecision:
    candidate_metrics = candidate["metrics"]
    incumbent_metrics = incumbent["metrics"]
    score_delta = (
        candidate_metrics["macro_balanced_accuracy"]
        - incumbent_metrics["macro_balanced_accuracy"]
    )
    ece_delta = candidate_metrics["macro_ece"] - incumbent_metrics["macro_ece"]
    user_delta = (
        candidate_metrics["user_macro_class_recall_p20"]
        - incumbent_metrics["user_macro_class_recall_p20"]
    )
    reasons = []
    if score_delta < minimum_improvement:
        reasons.append("insufficient_macro_ba_gain")
    if ece_delta > maximum_ece_regression:
        reasons.append("calibration_regression")
    if user_delta < -maximum_user_p20_regression:
        reasons.append("user_p20_regression")
    return AcceptanceDecision(
        accepted=not reasons,
        score_delta=float(score_delta),
        ece_delta=float(ece_delta),
        user_p20_delta=float(user_delta),
        reasons=tuple(reasons),
    )


def single_coordinate_schedule(
    config: ExperimentConfig,
    search_space: dict[str, list[Any]],
) -> Iterator[tuple[str, Any]]:
    for field in sorted(search_space):
        current = getattr(config, field)
        for value in search_space[field]:
            if value != current:
                yield field, value


def run_sweep(
    spec: SweepSpec,
    *,
    output_root: Path = Path("results/research"),
    max_trials: int | None = None,
    budget_hours: float | None = None,
    evaluator: Evaluator | None = None,
) -> dict[str, Any]:
    store = ExperimentStore(output_root)
    state_path = output_root / "sweeps" / f"{spec.name}.json"
    _check_prerequisite(spec, output_root)
    evaluation = evaluator or _runner_evaluator(output_root)
    trial_limit = min(max_trials or spec.max_trials, spec.max_trials)
    time_limit = min(budget_hours or spec.budget_hours, spec.budget_hours) * 3_600
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("status") == "development_complete":
            return state
        incumbent_config = ExperimentConfig.from_mapping(state["incumbent_config"])
        incumbent_summary = state["incumbent_summary"]
        next_trial = int(state["next_trial"])
        trials = list(state["trials"])
        schedule = [tuple(item) for item in state["schedule"]]
    else:
        base_config = ExperimentConfig.from_toml(spec.base_config)
        incumbent_config = replace(
            base_config,
            name=f"{spec.name}-screening-base",
            epochs=spec.screening_epochs,
        )
        incumbent_summary = evaluation(incumbent_config, Stage.DEVELOPMENT, False)
        baseline_config = ExperimentConfig.from_toml(spec.baseline_config)
        baseline_summary = evaluation(baseline_config, Stage.DEVELOPMENT, False)
        schedule = list(single_coordinate_schedule(incumbent_config, spec.search_space))
        next_trial = 0
        trials = []
        state = {
            "name": spec.name,
            "status": "running",
            "baseline_config": baseline_config.to_dict(),
            "baseline_summary": baseline_summary,
            "schedule": schedule,
        }

    started = time.monotonic()
    for schedule_index in range(next_trial, min(len(schedule), trial_limit)):
        if time.monotonic() - started >= time_limit:
            break
        field, value = schedule[schedule_index]
        candidate = replace(
            incumbent_config,
            name=f"{spec.name}-trial-{schedule_index:02d}-{field}",
            **{field: value},
        )
        summary = evaluation(candidate, Stage.DEVELOPMENT, False)
        decision = decide_acceptance(
            summary,
            incumbent_summary,
            minimum_improvement=spec.minimum_improvement,
            maximum_ece_regression=spec.maximum_ece_regression,
            maximum_user_p20_regression=spec.maximum_user_p20_regression,
        )
        trials.append(
            {
                "trial": schedule_index,
                "changed_field": field,
                "value": value,
                "config": candidate.to_dict(),
                "summary": summary,
                "decision": decision.to_dict(),
            }
        )
        store.append(
            {
                "sweep": spec.name,
                "status": "accepted" if decision.accepted else "rejected",
                "trial": schedule_index,
                "run_id": summary["run_id"],
                "decision": decision.to_dict(),
            }
        )
        if decision.accepted:
            incumbent_config = candidate
            incumbent_summary = summary
        state.update(
            {
                "incumbent_config": incumbent_config.to_dict(),
                "incumbent_summary": incumbent_summary,
                "next_trial": schedule_index + 1,
                "trials": trials,
            }
        )
        store.write_json(state_path, state)

    state["status"] = "development_complete"
    state["incumbent_config"] = incumbent_config.to_dict()
    state["incumbent_summary"] = incumbent_summary
    state["trials"] = trials
    state["next_trial"] = min(len(schedule), trial_limit)
    store.write_json(state_path, state)
    return state


def promote_sweep(
    spec: SweepSpec,
    *,
    output_root: Path = Path("results/research"),
    evaluator: Evaluator | None = None,
) -> dict[str, Any]:
    state_path, state = _load_state(spec, output_root, required="development_complete")
    if "promotion" in state:
        return state
    evaluation = evaluator or _runner_evaluator(output_root)
    candidate = replace(
        ExperimentConfig.from_mapping(state["incumbent_config"]),
        name=f"{spec.name}-champion",
        epochs=spec.final_epochs,
    )
    baseline = ExperimentConfig.from_mapping(state["baseline_config"])
    candidate_summary = evaluation(candidate, Stage.PROMOTION, False)
    baseline_summary = evaluation(baseline, Stage.PROMOTION, False)
    decision = decide_acceptance(
        candidate_summary,
        baseline_summary,
        minimum_improvement=spec.promotion_improvement,
        maximum_ece_regression=spec.maximum_ece_regression,
        maximum_user_p20_regression=spec.maximum_user_p20_regression,
    )
    state["promotion"] = {
        "candidate": candidate_summary,
        "baseline": baseline_summary,
        "decision": decision.to_dict(),
    }
    state["champion_config"] = candidate.to_dict()
    state["status"] = "promotion_accepted" if decision.accepted else "promotion_rejected"
    ExperimentStore(output_root).write_json(state_path, state)
    return state


def audit_sweep(
    spec: SweepSpec,
    *,
    output_root: Path = Path("results/research"),
    evaluator: Evaluator | None = None,
) -> dict[str, Any]:
    state_path, state = _load_state(spec, output_root, required="promotion_accepted")
    if "audit" in state:
        return state
    evaluation = evaluator or _runner_evaluator(output_root)
    candidate = ExperimentConfig.from_mapping(state["champion_config"])
    baseline = ExperimentConfig.from_mapping(state["baseline_config"])
    candidate_summary = evaluation(candidate, Stage.AUDIT, True)
    baseline_summary = evaluation(baseline, Stage.AUDIT, True)
    decision = decide_acceptance(
        candidate_summary,
        baseline_summary,
        minimum_improvement=0.0,
        maximum_ece_regression=spec.maximum_ece_regression,
        maximum_user_p20_regression=spec.maximum_user_p20_regression,
    )
    state["audit"] = {
        "candidate": candidate_summary,
        "baseline": baseline_summary,
        "decision": decision.to_dict(),
    }
    if decision.accepted:
        state["official"] = {
            "candidate": evaluation(candidate, Stage.OFFICIAL, True),
        }
        if spec.evaluate_official_baseline:
            state["official"]["baseline"] = evaluation(
                baseline, Stage.OFFICIAL, True
            )
        else:
            state["official"]["baseline"] = None
            state["official"]["baseline_note"] = (
                "The stage-matched baseline passed the sealed audit. Its official "
                "refit was skipped by the bounded-compute spec; report against the "
                "existing official paper reproduction."
            )
        state["status"] = "audit_accepted"
    else:
        state["status"] = "audit_rejected"
    ExperimentStore(output_root).write_json(state_path, state)
    return state


def _runner_evaluator(output_root: Path) -> Evaluator:
    def evaluate(
        config: ExperimentConfig,
        stage: Stage,
        audit_unlocked: bool,
    ) -> dict[str, Any]:
        return run_experiment_config(
            config,
            stage,
            output_root=output_root,
            audit_unlocked=audit_unlocked,
        )

    return evaluate


def _load_state(
    spec: SweepSpec,
    output_root: Path,
    *,
    required: str,
) -> tuple[Path, dict[str, Any]]:
    state_path = output_root / "sweeps" / f"{spec.name}.json"
    if not state_path.exists():
        raise FileNotFoundError(f"Run development sweep first: {spec.name}")
    state = json.loads(state_path.read_text())
    if state.get("status") != required:
        raise AuditAccessError(
            f"{spec.name} requires state {required}; found {state.get('status')}"
        )
    return state_path, state


def _check_prerequisite(spec: SweepSpec, output_root: Path) -> None:
    if not spec.requires_sweep:
        return
    prerequisite_path = output_root / "sweeps" / f"{spec.requires_sweep}.json"
    if not prerequisite_path.exists():
        raise AuditAccessError(
            f"{spec.name} requires completed sweep {spec.requires_sweep}"
        )
    prerequisite = json.loads(prerequisite_path.read_text())
    if prerequisite.get("status") not in {
        "development_complete",
        "promotion_accepted",
        "audit_accepted",
    }:
        raise AuditAccessError(
            f"{spec.name} requires completed development for {spec.requires_sweep}"
        )
    candidate = prerequisite["incumbent_summary"]
    baseline = prerequisite["baseline_summary"]
    score_gain = (
        candidate["metrics"]["macro_balanced_accuracy"]
        - baseline["metrics"]["macro_balanced_accuracy"]
    )
    if score_gain < spec.minimum_prerequisite_improvement:
        raise AuditAccessError(
            f"{spec.name} requires development gain "
            f"{spec.minimum_prerequisite_improvement:.3f}; observed {score_gain:.3f}"
        )
    if spec.minimum_missing_sensor_reduction:
        reduction = _missing_sensor_reduction(candidate, baseline)
        if reduction < spec.minimum_missing_sensor_reduction:
            raise AuditAccessError(
                f"{spec.name} requires missing-sensor reduction "
                f"{spec.minimum_missing_sensor_reduction:.1%}; observed {reduction:.1%}"
            )


def _missing_sensor_reduction(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
) -> float:
    def degradation(summary: dict[str, Any]) -> float:
        clean = summary["metrics"]["macro_balanced_accuracy"]
        ablations = summary["missing_sensor_metrics"].values()
        average_ablated = sum(
            metrics["macro_balanced_accuracy"] for metrics in ablations
        ) / len(summary["missing_sensor_metrics"])
        return clean - average_ablated

    baseline_degradation = degradation(baseline)
    candidate_degradation = degradation(candidate)
    return (baseline_degradation - candidate_degradation) / max(
        abs(baseline_degradation), 1e-12
    )
