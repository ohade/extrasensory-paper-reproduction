import numpy as np
import pytest

from extrasensory_reproduction.research.protocol import (
    AuditAccessError,
    Stage,
    assert_rows_respect_split,
    build_stage_splits,
)


def _folds() -> list[dict[str, np.ndarray]]:
    users = np.arange(60, dtype=np.int16)
    return [
        {
            "test": users[fold * 12 : (fold + 1) * 12],
            "train": np.setdiff1d(users, users[fold * 12 : (fold + 1) * 12]),
        }
        for fold in range(5)
    ]


def test_development_and_promotion_never_use_audit_users() -> None:
    folds = _folds()
    audit = set(folds[3]["test"].tolist())
    development = build_stage_splits(folds, Stage.DEVELOPMENT)
    promotion = build_stage_splits(folds, Stage.PROMOTION)
    assert [split.fold for split in development] == [0, 1, 2]
    assert promotion[0].fold == 4
    for split in [*development, *promotion]:
        assert len(split.train_users) == 36
        assert len(split.test_users) == 12
        assert not audit.intersection(split.train_users.tolist())
        assert not audit.intersection(split.test_users.tolist())


def test_audit_uses_48_non_audit_train_users() -> None:
    split = build_stage_splits(_folds(), Stage.AUDIT)[0]
    assert split.fold == 3
    assert len(split.train_users) == 48
    assert len(split.test_users) == 12


def test_official_evaluation_requires_audit_unlock() -> None:
    with pytest.raises(AuditAccessError, match="completed audit"):
        build_stage_splits(_folds(), Stage.OFFICIAL)
    splits = build_stage_splits(_folds(), Stage.OFFICIAL, audit_unlocked=True)
    assert len(splits) == 5
    assert all(len(split.train_users) == 48 for split in splits)


def test_row_guard_rejects_disallowed_users() -> None:
    users = np.asarray([0, 0, 1, 2])
    with pytest.raises(AuditAccessError, match="disallowed"):
        assert_rows_respect_split(users, np.asarray([0, 3]), np.asarray([0, 1]), purpose="train")
