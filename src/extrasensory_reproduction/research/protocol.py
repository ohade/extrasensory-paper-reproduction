"""Subject-disjoint development, promotion, and audit split policy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np


class AuditAccessError(RuntimeError):
    """Raised when a pre-audit stage attempts to use an audit subject."""


class Stage(StrEnum):
    DEVELOPMENT = "development"
    PROMOTION = "promotion"
    AUDIT = "audit"
    OFFICIAL = "official"


DEVELOPMENT_FOLDS = (0, 1, 2)
AUDIT_FOLD = 3
PROMOTION_FOLD = 4


@dataclass(frozen=True)
class ResearchSplit:
    fold: int
    train_users: np.ndarray
    test_users: np.ndarray


def audit_users(folds: list[dict[str, np.ndarray]]) -> np.ndarray:
    _validate_official_folds(folds)
    return np.asarray(folds[AUDIT_FOLD]["test"], dtype=np.int16)


def build_stage_splits(
    folds: list[dict[str, np.ndarray]],
    stage: Stage | str,
    *,
    audit_unlocked: bool = False,
) -> list[ResearchSplit]:
    """Build the approved split policy without exposing audit users early."""

    _validate_official_folds(folds)
    selected_stage = Stage(stage)
    all_users = set(
        np.concatenate([folds[AUDIT_FOLD]["train"], folds[AUDIT_FOLD]["test"]]).tolist()
    )
    held_out_audit = set(audit_users(folds).tolist())

    if selected_stage == Stage.DEVELOPMENT:
        selected_folds = DEVELOPMENT_FOLDS
    elif selected_stage == Stage.PROMOTION:
        selected_folds = (PROMOTION_FOLD,)
    elif selected_stage == Stage.OFFICIAL:
        if not audit_unlocked:
            raise AuditAccessError("Official five-fold evaluation requires a completed audit")
        selected_folds = tuple(range(5))
    else:
        selected_folds = (AUDIT_FOLD,)

    splits: list[ResearchSplit] = []
    for fold in selected_folds:
        test_users = set(folds[fold]["test"].tolist())
        if selected_stage in {Stage.AUDIT, Stage.OFFICIAL}:
            train_users = all_users - test_users
        else:
            train_users = all_users - held_out_audit - test_users
            if train_users & held_out_audit or test_users & held_out_audit:
                raise AuditAccessError(
                    f"{selected_stage} fold {fold} attempted to access audit users"
                )
        splits.append(
            ResearchSplit(
                fold=fold,
                train_users=np.asarray(sorted(train_users), dtype=np.int16),
                test_users=np.asarray(sorted(test_users), dtype=np.int16),
            )
        )
    return splits


def assert_rows_respect_split(
    user_index: np.ndarray,
    rows: np.ndarray,
    allowed_users: np.ndarray,
    *,
    purpose: str,
) -> None:
    observed = set(np.unique(user_index[rows]).tolist())
    allowed = set(np.asarray(allowed_users).tolist())
    unexpected = sorted(observed - allowed)
    if unexpected:
        raise AuditAccessError(f"{purpose} contains disallowed users: {unexpected}")


def _validate_official_folds(folds: list[dict[str, np.ndarray]]) -> None:
    if len(folds) != 5:
        raise ValueError("The research protocol requires the five official folds")
    test_users = np.concatenate([np.asarray(fold["test"]) for fold in folds])
    if len(test_users) != len(np.unique(test_users)):
        raise ValueError("Each user must occur in exactly one official test fold")
