"""Pre-registered, deterministic validation-only policy selection.

Only an auditor reward head fitted on a disjoint train-user partition can
rank eligible candidates. Never read test metrics during selection.
"""
from __future__ import annotations

import math


def select_policy(
    validation: dict[str, dict[str, float]],
    baseline: str = "gpt2",
    max_relative_ndcg_drop: float = 0.05,
    max_absolute_ndcg_drop: float = 0.005,
    min_auditor_gain: float = 0.0,
) -> dict:
    if baseline not in validation:
        raise ValueError("Baseline missing from validation results")
    if not 0 <= max_relative_ndcg_drop < 1 or max_absolute_ndcg_drop < 0:
        raise ValueError("Invalid NDCG retention constraints")
    if min_auditor_gain < 0:
        raise ValueError("Minimum independent reward gain must be nonnegative")

    base = validation[baseline]
    b_ndcg = float(base["ndcg_at_10"])
    b_reward = float(base["auditor_completion_at_10"])
    if not math.isfinite(b_ndcg) or not math.isfinite(b_reward):
        raise ValueError("Non-finite baseline validation metrics")
    # Enforce BOTH relative and absolute ceilings; near-zero NDCG cannot
    # tolerate an arbitrary absolute 0.01 reduction.
    floor = max(b_ndcg * (1.0 - max_relative_ndcg_drop),
                b_ndcg - max_absolute_ndcg_drop, 0.0)
    eligible = []
    for name, scores in validation.items():
        ndcg = float(scores["ndcg_at_10"])
        reward = float(scores["auditor_completion_at_10"])
        if not (math.isfinite(ndcg) and math.isfinite(reward)):
            continue
        if ndcg >= floor and reward > b_reward + min_auditor_gain and name != baseline:
            eligible.append((name, ndcg, reward))
    # Reward improvement is required to switch away from the baseline.
    # Equal-reward ties favour higher ranking accuracy then lexical identity.
    eligible.sort(key=lambda x: (-x[2], -x[1], x[0]))
    winner = eligible[0][0] if eligible else baseline
    return {
        "selected": winner,
        "baseline_ndcg": b_ndcg,
        "baseline_auditor_reward": b_reward,
        "ndcg_floor": floor,
        "max_relative_ndcg_drop": max_relative_ndcg_drop,
        "max_absolute_ndcg_drop": max_absolute_ndcg_drop,
        "min_auditor_gain": min_auditor_gain,
        "eligible_alternatives": [x[0] for x in eligible],
        "reason": ("auditor_reward_gain_with_both_ndcg_guards"
                   if eligible else "no_candidate_passed_reward_and_ndcg_guards"),
        "independent_online_effect_established": False,
    }
