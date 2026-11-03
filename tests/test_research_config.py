from pathlib import Path

import pytest

from extrasensory_reproduction.research.config import ExperimentConfig


def test_config_is_stable_and_bounded() -> None:
    path = Path("configs/research/causal_tcn.toml")
    config = ExperimentConfig.from_toml(path)
    assert config.run_id == ExperimentConfig.from_toml(path).run_id
    assert config.model == "causal_tcn"
    assert config.max_workers == 4

    with pytest.raises(ValueError, match="between 1 and 4"):
        ExperimentConfig(name="bad", model="paper_lr", max_workers=5)
