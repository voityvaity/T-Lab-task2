"""Prevent invented or synthetic numbers from becoming a competition report."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def load_report_module():
    source = Path(__file__).resolve().parents[1] / "scripts" / "build_report.py"
    spec = importlib.util.spec_from_file_location("tlab2_build_report", source)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def minimal(source: str, status: str) -> dict:
    return {
        "source_kind": source, "status": status,
        "selected_on_validation": "gpt2", "selection_rule": "validation",
        "device": "cpu", "seed": 42, "reward_quality_on_observed_events": {},
        "validation": {}, "test": {}, "test_paired_ndcg_deltas": {},
        "five_step_hypothetical_rollouts": {},
        "real_context_vs_generated_context_drift": {},
        "limitations": ["No ground-truth reward for unseen items"],
    }


def test_report_rejects_missing_and_mislabeled_results(tmp_path: Path) -> None:
    module = load_report_module()
    path = tmp_path / "results.json"
    with pytest.raises(FileNotFoundError):
        module.report(path)
    path.write_text(json.dumps(minimal("synthetic", "measured_offline_model_based_not_causal")))
    with pytest.raises(ValueError, match="mismatched"):
        module.report(path)
    path.write_text(json.dumps(minimal("unverified", "measured_offline_model_based_not_causal")))
    with pytest.raises(ValueError):
        module.report(path)


def test_synthetic_report_never_claims_real_measurements(tmp_path: Path) -> None:
    module = load_report_module()
    path = tmp_path / "results.json"
    path.write_text(json.dumps(minimal("synthetic", "synthetic_smoke_only")))
    text = module.report(path)
    assert "СИНТЕТИЧЕСКИЙ SMOKE" in text
    assert "НЕ КОНКУРСНЫЕ РЕЗУЛЬТАТЫ" in text


def test_yambda_report_labels_measured_offline_only(tmp_path: Path) -> None:
    module = load_report_module()
    path = tmp_path / "results.json"
    path.write_text(json.dumps(minimal("yambda", "measured_offline_model_based_not_causal")))
    text = module.report(path)
    assert "Yambda-50M" in text
    assert "не доказывает" in text


def test_experiment_refuses_synthetic_csv_disguised_as_yambda():
    from argparse import Namespace
    from tlab2.experiment import run
    with pytest.raises(ValueError, match="version-pinned Parquet"):
        run(Namespace(data="data/synthetic_listens.csv", source_kind="yambda"))


def test_one_example_reinforce_batch_fails_instead_of_silent_zero_gradient():
    from argparse import Namespace
    from tlab2.experiment import run
    args = Namespace(
        source_kind="synthetic", data="data/not_needed.csv",
        base_steps=1, reward_steps=1, rl_steps=1, users=2, vocab=16,
        context=8, hidden=32, layers=1, heads=4, train_events=10,
        eval_per_user=5, batch=1, train_fraction=.7, valid_fraction=.15,
        betas=".1", rl_horizons="1", rewards="completion", seed=42,
    )
    with pytest.raises(ValueError, match="batch >= 2"):
        run(args)
