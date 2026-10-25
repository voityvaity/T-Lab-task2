"""Five-step diagnostics never join unrelated real interactions."""
from __future__ import annotations

import numpy as np

from tlab2.data import Examples
from tlab2.diagnostics import actual_five_step_windows


def fixture() -> Examples:
    ctx = np.asarray([[0, 0, 2, 3], [0, 2, 3, 4], [2, 3, 4, 5],
                      [3, 4, 5, 6], [4, 5, 6, 7]], dtype=np.int32)
    return Examples(
        context=ctx, target=np.asarray([4, 5, 6, 7, 8]),
        completion=np.ones(5, dtype=np.float32),
        user=np.ones(5, dtype=np.int64),
        timestamp=np.arange(100, 105, dtype=np.int64),
    )


def test_strict_consecutive_five_events_are_accepted() -> None:
    e = fixture()
    assert actual_five_step_windows(e) == [[0, 1, 2, 3, 4]]


def test_tied_timestamps_are_rejected() -> None:
    e = fixture()
    e.timestamp[3] = e.timestamp[2]
    assert actual_five_step_windows(e) == []


def test_missing_intervening_event_is_rejected() -> None:
    e = fixture()
    e.context[2] = np.array([2, 3, 6, 7])
    assert actual_five_step_windows(e) == []


def test_oov_observed_previous_item_chains_correctly() -> None:
    e = fixture()
    e.target[0] = 1
    e.context[1] = [0, 2, 3, 1]
    e.context[2] = [2, 3, 1, 5]
    e.context[3] = [3, 1, 5, 6]
    e.context[4] = [1, 5, 6, 7]
    assert actual_five_step_windows(e) == [[0, 1, 2, 3, 4]]


def test_one_window_per_user_limits_overlapping_statistics() -> None:
    e = fixture()
    e.context = np.vstack([e.context, [5, 6, 7, 8]])
    e.target = np.concatenate([e.target, [9]])
    e.completion = np.concatenate([e.completion, [0.8]])
    e.user = np.concatenate([e.user, [1]])
    e.timestamp = np.concatenate([e.timestamp, [105]])
    assert len(actual_five_step_windows(e)) == 1
