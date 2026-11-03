import numpy as np

from extrasensory_reproduction.research.sequence import build_history_indices


def test_histories_are_causal_and_stop_at_users_and_gaps() -> None:
    users = np.asarray([0, 0, 0, 0, 1, 1])
    timestamps = np.asarray([100, 160, 220, 500, 110, 170])
    histories, valid = build_history_indices(
        users,
        timestamps,
        np.asarray([2, 3, 5]),
        history_steps=4,
        max_gap_seconds=90,
    )
    assert histories[0].tolist() == [-1, 0, 1, 2]
    assert histories[1].tolist() == [-1, -1, -1, 3]
    assert histories[2].tolist() == [-1, -1, 4, 5]
    assert np.all(histories[valid] <= np.asarray([2, 3, 5]).repeat(valid.sum(axis=1)))
