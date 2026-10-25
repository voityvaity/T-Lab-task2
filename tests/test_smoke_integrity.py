"""Make sure GitHub Actions cannot silently skip the five-step research check."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def checker():
    source = Path(__file__).resolve().parents[1] / "scripts" / "check_smoke.py"
    spec = importlib.util.spec_from_file_location("smoke_validator", source)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module.validate_smoke_results


def payload():
    metrics = {"recall_at_10": .1, "ndcg_at_10": .08,
               "target_catalog_coverage": .3, "queries": 10}
    return {
        "source_kind": "synthetic", "status": "synthetic_smoke_only",
        "test": {key: dict(metrics) for key in (
            "gpt2", "rerank_0.0", "popularity",
            "sequential_rules", "smoothed_transition",
            "rl_completion_h1_kl_0.02",
            "rl_conservative_h5_kl_0.02")},
        "real_five_step_windows": 2,
        "real_context_vs_generated_context_drift": {
            "gpt2": [{"step": k, "paired_trajectories": 2} for k in range(1, 6)]
        },
        "selected_on_validation": "gpt2",
        "selection_diagnostics": {"selected": "gpt2"},
    }


def test_integrity_check_accepts_complete_software_smoke(tmp_path):
    p = tmp_path / "experiment.json"
    p.write_text(json.dumps(payload()))
    result = checker()(p)
    assert result["real_five_step_windows"] == 2


def test_integrity_check_rejects_missing_five_step_drift(tmp_path):
    data = payload()
    data["real_five_step_windows"] = 0
    p = tmp_path / "experiment.json"
    p.write_text(json.dumps(data))
    with pytest.raises(AssertionError):
        checker()(p)


def test_integrity_check_rejects_real_data_mislabeled_as_smoke(tmp_path):
    data = payload()
    data["source_kind"] = "yambda"
    p = tmp_path / "experiment.json"
    p.write_text(json.dumps(data))
    with pytest.raises(AssertionError):
        checker()(p)


def test_integrity_check_rejects_reranking_change_with_zero_weight(tmp_path):
    data = payload()
    data["test"]["rerank_0.0"]["ndcg_at_10"] = .03
    p = tmp_path / "experiment.json"
    p.write_text(json.dumps(data))
    with pytest.raises(AssertionError, match="Zero-reward"):
        checker()(p)


def test_integrity_check_rejects_claimed_rl_win_when_reward_model_is_worse_than_constant(tmp_path):
    data = payload()
    data["selection_diagnostics"] = {
        "selected": "rl_completion_h1_kl_0.02", "min_auditor_gain": .001
    }
    data["selected_on_validation"] = "rl_completion_h1_kl_0.02"
    data["reward_quality_on_observed_events"] = {
        "optimized_valid": {"mae": .19, "constant_mean_mae": .16,
                            "rmse": .45, "constant_mean_rmse": .38},
        "auditor_valid": {"mae": .13, "constant_mean_mae": .15,
                          "rmse": .24, "constant_mean_rmse": .33},
    }
    data["validation"] = {
        "gpt2": {"auditor_completion_at_10": .50},
        "rl_completion_h1_kl_0.02": {"auditor_completion_at_10": .55},
    }
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(data))
    with pytest.raises(AssertionError, match="useless observed-feedback model"):
        checker()(path)


def test_integrity_check_rejects_rl_with_subthreshold_gain(tmp_path):
    data = payload()
    data["selection_diagnostics"] = {
        "selected": "rl_completion_h1_kl_0.02", "min_auditor_gain": .001
    }
    data["selected_on_validation"] = "rl_completion_h1_kl_0.02"
    data["validation"] = {
        "gpt2": {"auditor_completion_at_10": .50},
        "rl_completion_h1_kl_0.02": {"auditor_completion_at_10": .500005},
    }
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(data))
    with pytest.raises(AssertionError, match="sub-threshold"):
        checker()(path)
