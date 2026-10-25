"""RL selection criteria must be fixed before test evaluation."""
from __future__ import annotations

import pytest

from tlab2.selection import select_policy, pareto_frontier


def score(ndcg, auditor):
    return {"ndcg_at_10": ndcg, "auditor_completion_at_10": auditor}


def test_relative_ndcg_constraint_prevents_near_zero_collapse():
    validation = {
        "gpt2": score(0.010, 0.40),
        "rl_risky": score(0.002, 0.95),
        "rl_safe": score(0.0098, 0.42),
    }
    r = select_policy(validation, max_relative_ndcg_drop=0.05,
                      max_absolute_ndcg_drop=0.01)
    assert r["selected"] == "rl_safe"
    assert r["ndcg_floor"] == pytest.approx(0.0095)
    assert "rl_risky" not in r["eligible_alternatives"]


def test_absolute_constraint_prevents_large_raw_loss():
    validation = {"gpt2": score(.20, .4), "too_low": score(.194, .9)}
    r = select_policy(validation, max_relative_ndcg_drop=.10,
                      max_absolute_ndcg_drop=.005)
    assert r["selected"] == "gpt2"


def test_without_positive_auditor_gain_keeps_baseline():
    validation = {"gpt2": score(.15, .7), "rl": score(.16, .69)}
    r = select_policy(validation)
    assert r["selected"] == "gpt2"


def test_ties_deterministic_and_never_consults_test_data():
    v = {"gpt2": score(.15, .4), "z": score(.149, .44),
         "a": score(.149, .44)}
    r = select_policy(v)
    assert r["selected"] == "a"


def test_ill_defined_configuration_is_rejected():
    with pytest.raises(ValueError):
        select_policy({"gpt2": score(.1, .1)}, max_relative_ndcg_drop=1.0)
    with pytest.raises(ValueError):
        select_policy({"gpt2": score(.1, .1)}, min_auditor_gain=-.1)
    with pytest.raises(ValueError):
        select_policy({}, baseline="gpt2")


def test_pareto_frontier_keeps_non_dominated_accuracy_reward_tradeoff():
    alternatives = {
        "gpt2": score(.15, .5),
        "rl_balanced": score(.14, .6),
        "dominated": score(.13, .49),
        "best_of_both": score(.16, .61),
    }
    frontier = pareto_frontier(alternatives)
    assert [item["model"] for item in frontier] == ["best_of_both"]
    tied = pareto_frontier({"gpt2": score(.16, .5),
                           "rl": score(.14, .6)})
    assert {item["model"] for item in tied} == {"gpt2", "rl"}
