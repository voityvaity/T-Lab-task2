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
    auditor_is_useful: bool = True,
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
        if (auditor_is_useful and ndcg >= floor
                and reward > b_reward + min_auditor_gain and name != baseline):
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
        "auditor_beats_constant_validation_mae": bool(auditor_is_useful),
        "eligible_alternatives": [x[0] for x in eligible],
        "reason": ("auditor_failed_constant_mae_guard" if not auditor_is_useful
                   else "auditor_reward_gain_with_both_ndcg_guards" if eligible
                   else "no_candidate_passed_reward_and_ndcg_guards"),
        "independent_online_effect_established": False,
    }


def pareto_frontier(validation: dict[str, dict[str, float]]) -> list[dict]:
    """Non-dominated models under (NDCG@10, independent auditor reward).

    Report-only. A test Pareto frontier must never be used for choosing
    training parameters or deciding which experiment to run next.
    """
    rows = []
    for name, score in validation.items():
        x = float(score["ndcg_at_10"])
        y = float(score["auditor_completion_at_10"])
        if math.isfinite(x) and math.isfinite(y):
            rows.append((name, x, y))
    nondominated = []
    for name, x, y in rows:
        if not any(
            xx >= x and yy >= y and (xx > x or yy > y)
            for rival, xx, yy in rows if rival != name
        ):
            nondominated.append({"model": name, "ndcg_at_10": x,
                                 "auditor_completion_at_10": y})
    return sorted(nondominated, key=lambda row: (-row["ndcg_at_10"], row["model"]))
