"""Leakage-aware five-step evaluation-window construction.

Saved evaluation examples are a capped per-user sample. Two neighboring saved
rows are not necessarily neighboring original events. A valid five-step
comparison requires their full observed contexts to chain exactly.
"""
from __future__ import annotations

from collections import defaultdict
import numpy as np

from tlab2.data import Examples


def actual_five_step_windows(samples: Examples, steps: int = 5) -> list[list[int]]:
    """Return at most one strictly consecutive, untied window per user.

    Every transition must satisfy:
       following_context = previous_context shifted left + previous_real_item.
    This filters dropped/intervening events, tied 5-second timestamps and
    accidentally cherry-picked nonconsecutive targets. It does not establish
    whether a recommendation caused the observed next user event.
    """
    if steps < 2:
        raise ValueError("Need at least two steps for trajectory drift")
    by_user: dict[int, list[int]] = defaultdict(list)
    for i, user in enumerate(samples.user):
        by_user[int(user)].append(i)
    windows: list[list[int]] = []
    for uid in sorted(by_user):
        idx = sorted(by_user[uid], key=lambda i: (int(samples.timestamp[i]), i))
        for start in range(len(idx) - steps + 1):
            segment = idx[start:start + steps]
            okay = all(
                int(samples.timestamp[next_idx]) > int(samples.timestamp[prev_idx])
                and np.array_equal(
                    samples.context[next_idx],
                    np.concatenate((samples.context[prev_idx][1:],
                                    np.asarray([samples.target[prev_idx]], dtype=samples.context.dtype)))
                )
                for prev_idx, next_idx in zip(segment[:-1], segment[1:])
            )
            if okay:
                windows.append(segment)
                break
    return windows
