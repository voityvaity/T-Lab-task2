"""Recount returns on the ORIGINAL Yambda cohort without any RL retraining."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest


def recount_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "recompute_session_proxy.py"
    spec = importlib.util.spec_from_file_location("session_recount", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def fake_source(tmp_path):
    rows = [
        (1, 10, 10, 70), (1, 11, 20, 80),
        (2, 10, 10, 60), (2, 12, 20, 50),
        (1, 12, 760000, 70), (1, 11, 800000, 50),
        (2, 12, 760000, 75), (2, 11, 800000, 35),
        (1, 12, 860000, 20), (1, 13, 862400, 95),
        (2, 10, 860200, 55), (2, 11, 862000, 60),
        (1, 13, 999000, 70), (2, 12, 1000000, 70),
    ]
    path = tmp_path / "listens.csv"
    frame = pd.DataFrame(rows, columns=["uid", "item_id", "timestamp", "played_ratio_pct"])
    frame["track_length_seconds"] = 180
    frame.to_csv(path, index=False)
    return path


def test_recount_uses_actual_seconds_and_same_train_only_cohort(tmp_path):
    tool = recount_module()
    path = fake_source(tmp_path)
    options = dict(users=2, seed=42, train_fraction=.7, valid_fraction=.15)
    result = tool.count_sessions(path, **options)
    assert result["timestamp_units"] == "seconds_quantized_to_5_seconds"
    assert result["session_gap_seconds"] == 1800
    assert result["return_window_seconds"] == 86400
    assert result["users"] == 2
    assert result["results"]["low_completion"]["n_session_ends"] == 1
    assert result["results"]["low_completion"]["returned_within_24h"] == 1
    assert result["results"]["high_completion"]["n_session_ends"] == 1
    assert result["results"]["high_completion"]["returned_within_24h"] == 0
    again = tool.count_sessions(path, expected_dataset_hash=result["input_file_sha256"],
                                expected_cohort_hash=result["selected_train_user_cohort_sha256"],
                                **options)
    assert result == again


def test_recount_rejects_swapped_parquet_or_wrong_cohort(tmp_path):
    tool = recount_module()
    path = fake_source(tmp_path)
    common = dict(users=2, seed=42, train_fraction=.7, valid_fraction=.15)
    with pytest.raises(ValueError, match="Input Yambda file differs"):
        tool.count_sessions(path, expected_dataset_hash="0" * 64, **common)
    with pytest.raises(ValueError, match="cohort differs"):
        tool.count_sessions(path, expected_cohort_hash="a" * 64, **common)
