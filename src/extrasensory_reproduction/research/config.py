"""Typed, hashable configuration for bounded research experiments."""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SUPPORTED_MODELS = (
    "paper_lr",
    "threshold_lr",
    "elastic_net",
    "hist_gradient_boosting",
    "multitask_mlp",
    "classifier_chain",
    "causal_tcn",
    "sensor_moe_tcn",
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete experiment definition; no hidden mutable tuning state."""

    name: str
    model: str
    seed: int = 2017
    label_policy: str = "legacy-negative"
    threshold: float = 0.5
    history_steps: int = 30
    max_gap_seconds: int = 90
    hidden_dim: int = 128
    dropout: float = 0.2
    label_rank: int = 8
    batch_size: int = 256
    epochs: int = 12
    patience: int = 3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    group_dro_eta: float = 0.0
    asymmetric_gamma_negative: float = 0.0
    sensor_dropout: float = 0.0
    pretrain_epochs: int = 0
    max_workers: int = 4

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Experiment name must not be empty")
        if self.model not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported research model: {self.model}")
        if self.label_policy not in {"legacy-negative", "exclude-missing"}:
            raise ValueError(f"Unsupported label policy: {self.label_policy}")
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("threshold must be between 0 and 1")
        if self.history_steps < 1:
            raise ValueError("history_steps must be positive")
        if self.max_gap_seconds < 60:
            raise ValueError("max_gap_seconds must be at least 60")
        if self.hidden_dim < 8 or self.label_rank < 0:
            raise ValueError("hidden_dim must be >= 8 and label_rank must be >= 0")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.batch_size < 1 or self.epochs < 1 or self.patience < 1:
            raise ValueError("batch_size, epochs, and patience must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if self.group_dro_eta < 0.0 or self.asymmetric_gamma_negative < 0.0:
            raise ValueError("robustness loss parameters must be non-negative")
        if not 0.0 <= self.sensor_dropout < 1.0:
            raise ValueError("sensor_dropout must be in [0, 1)")
        if self.pretrain_epochs < 0:
            raise ValueError("pretrain_epochs must be non-negative")
        if not 1 <= self.max_workers <= 4:
            raise ValueError("max_workers must be between 1 and 4")

    @classmethod
    def from_toml(cls, path: Path) -> ExperimentConfig:
        with path.open("rb") as handle:
            payload = tomllib.load(handle)
        values = payload.get("experiment", payload)
        if not isinstance(values, dict):
            raise TypeError(f"{path} must contain an [experiment] table")
        return cls.from_mapping(values)

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> ExperimentConfig:
        allowed = set(cls.__dataclass_fields__)
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"Unknown experiment keys: {sorted(unknown)}")
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def run_id(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(canonical.encode()).hexdigest()[:12]
        return f"{self.name}-{digest}"
