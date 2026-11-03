"""Lazy, causal history construction that never crosses user or gap boundaries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


def build_history_indices(
    user_index: np.ndarray,
    timestamps: np.ndarray,
    end_rows: np.ndarray,
    *,
    history_steps: int,
    max_gap_seconds: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return right-aligned row histories and a valid-timestep mask."""

    users = np.asarray(user_index)
    timestamps = np.asarray(timestamps)
    end_rows = np.asarray(end_rows, dtype=np.int64)
    histories = np.full((len(end_rows), history_steps), -1, dtype=np.int32)
    positions = np.full(len(users), -1, dtype=np.int32)
    ordered_by_user: dict[int, np.ndarray] = {}

    for user in np.unique(users[end_rows]):
        rows = np.flatnonzero(users == user)
        ordered = rows[np.argsort(timestamps[rows], kind="stable")]
        ordered_by_user[int(user)] = ordered
        positions[ordered] = np.arange(len(ordered), dtype=np.int32)

    for output_index, end_row in enumerate(end_rows):
        ordered = ordered_by_user[int(users[end_row])]
        position = int(positions[end_row])
        selected = [int(end_row)]
        while position > 0 and len(selected) < history_steps:
            current = int(ordered[position])
            previous = int(ordered[position - 1])
            gap = int(timestamps[current] - timestamps[previous])
            if gap < 0 or gap > max_gap_seconds:
                break
            selected.append(previous)
            position -= 1
        selected.reverse()
        histories[output_index, -len(selected) :] = selected

    return histories, histories >= 0


@dataclass(frozen=True)
class SequenceArrays:
    features: np.ndarray
    sensor_available: np.ndarray
    labels: np.ndarray
    label_valid: np.ndarray
    users: np.ndarray
    histories: np.ndarray
    timestep_valid: np.ndarray


class CausalSequenceDataset(Dataset[tuple[torch.Tensor, ...]]):
    """Materialize only the requested sequence when DataLoader asks for it."""

    def __init__(self, arrays: SequenceArrays) -> None:
        self.arrays = arrays

    def __len__(self) -> int:
        return len(self.arrays.histories)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        history = self.arrays.histories[index]
        valid = self.arrays.timestep_valid[index]
        safe_history = np.where(valid, history, 0)
        features = self.arrays.features[safe_history].copy()
        availability = self.arrays.sensor_available[safe_history].copy()
        features[~valid] = 0.0
        availability[~valid] = False
        return (
            torch.from_numpy(features),
            torch.from_numpy(availability),
            torch.from_numpy(valid),
            torch.from_numpy(self.arrays.labels[index].astype(np.float32, copy=False)),
            torch.from_numpy(self.arrays.label_valid[index]),
            torch.tensor(int(self.arrays.users[index]), dtype=torch.long),
        )
