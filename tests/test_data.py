from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tlab2.data import DataConfig, prepare, save_metadata, summarize_session_returns


def fake_events() -> pd.DataFrame:
    rows = []
    for uid in range(10, 18):
        for ts in range(100):
            item = 20 + (uid + ts) % 9
            if ts > 69 and ts % 11 == 0:
                item = 99_000 + uid  # never occurs in training
            rows.append({"uid": uid, "item_id": item, "timestamp": ts,
                         "played_ratio_pct": 150 if ts % 19 == 0 else 20 + ts % 80,
                         "track_length_seconds": 140 + item % 30})
    return pd.DataFrame(rows)


def test_causal_split_and_reproducibility(tmp_path: Path) -> None:
    path = tmp_path / "events.csv"
    fake_events().to_csv(path, index=False)
    cfg = DataConfig(users=6, vocab_size=11, context_length=8,
                     min_history=2, train_events=110, eval_events_per_user=3)
    a = prepare(path, cfg)
    b = prepare(path, cfg)
    assert a.boundaries == b.boundaries
    assert a.boundaries["train_end"] < a.boundaries["valid_end"]
    assert np.array_equal(a.train.context, b.train.context)
    assert np.array_equal(a.train.target, b.train.target)
    assert set(a.train.user) == set(b.train.user)
    assert len(a.train) <= 110
    assert len(a.test) <= 6 * 3
    assert np.all(a.valid.timestamp >= a.boundaries["train_end"])
    assert np.all(a.valid.timestamp < a.boundaries["valid_end"])
    assert np.all(a.test.timestamp >= a.boundaries["valid_end"])
    assert a.coverage["test"] < 1.0
    assert np.all(a.train.target >= 2)
    # Next-item evaluation includes OOV targets as unavoidable misses.
    assert np.all(a.test.target >= 1)
    assert np.any(a.test.target == 1)
    assert a.train.context.shape[1] == 8
    assert np.all((a.train.completion >= 0) & (a.train.completion <= 1))
    assert 1.0 in a.train.completion
    save_metadata(a, tmp_path / "outputs")
    assert (tmp_path / "outputs/dataset_protocol.json").is_file()
    import hashlib
    assert a.input_file_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    import json
    info = json.loads((tmp_path / "outputs/dataset_protocol.json").read_text())
    assert info["input_file_sha256"] == a.input_file_sha256


def test_future_item_change_cannot_modify_train_vocabulary(tmp_path: Path) -> None:
    df = fake_events()
    before, after = tmp_path / "before.csv", tmp_path / "after.csv"
    df.to_csv(before, index=False)
    df.loc[(df.timestamp > 83) & (df.item_id == 24), "item_id"] = 1_234_567
    df.to_csv(after, index=False)
    cfg = DataConfig(users=6, vocab_size=11, context_length=8,
                     train_events=200, eval_events_per_user=2)
    a, b = prepare(before, cfg), prepare(after, cfg)
    assert a.token_to_item == b.token_to_item
    assert np.array_equal(a.train.context, b.train.context)
    assert np.array_equal(a.train.target, b.train.target)
    assert np.array_equal(a.train_duration, b.train_duration)


def test_invalid_configuration_raises(tmp_path: Path) -> None:
    path = tmp_path / "events.csv"
    fake_events().to_csv(path, index=False)
    with pytest.raises(ValueError):
        prepare(path, DataConfig(train_fraction=0.9, valid_fraction=0.15))
    with pytest.raises(ValueError):
        prepare(path, DataConfig(context_length=0))


def test_same_timestamp_does_not_look_ahead(tmp_path: Path) -> None:
    df = fake_events()
    # Two events from one user share a five-second time bucket.
    # Neither can be used as context for the other.
    df = pd.concat([df, pd.DataFrame([{"uid": 10, "item_id": 22,
                                      "timestamp": 35, "played_ratio_pct": 80,
                                      "track_length_seconds": 160}])], ignore_index=True)
    path = tmp_path / "ties.csv"
    df.to_csv(path, index=False)
    all_data = prepare(path, DataConfig(users=8, vocab_size=11, context_length=8,
                                        train_events=2000, eval_events_per_user=20))
    indexes = np.flatnonzero((all_data.train.user == 10) & (all_data.train.timestamp == 35))
    assert len(indexes) == 2
    assert np.array_equal(all_data.train.context[indexes[0]], all_data.train.context[indexes[1]])


def test_session_return_proxy_respects_real_seconds_and_right_censoring() -> None:
    # Yambda timestamps are seconds snapped to a 5-second resolution,
    # NOT counters of five-second units. 30min = 1800, 24h = 86400.
    events = pd.DataFrame([
        {"uid": 1, "timestamp": 100, "played_ratio_pct": 30},
        {"uid": 1, "timestamp": 2500, "played_ratio_pct": 95},
        {"uid": 1, "timestamp": 110_000, "played_ratio_pct": 50},
        {"uid": 2, "timestamp": 250, "played_ratio_pct": 55},
        {"uid": 2, "timestamp": 550, "played_ratio_pct": 70},  # same session
    ])
    summary = summarize_session_returns(events, test_start=0, source_last=140_000)
    assert summary["low_completion"]["n_session_ends"] == 1
    assert summary["low_completion"]["returned_within_24h"] == 1  # 2400 sec later
    assert summary["high_completion"]["n_session_ends"] == 1
    assert summary["high_completion"]["returned_within_24h"] == 0  # > 86400 sec
    # Last user-1 event ts=110000 is right-censored; user-2 last event
    # ts=550 has complete 24h global follow-up and counts as non-return.
    assert summary["medium_completion"]["n_session_ends"] == 1
    assert summary["medium_completion"]["returned_within_24h"] == 0
    only_user_1 = summarize_session_returns(events[events.uid == 1],
                                            test_start=0, source_last=140_000)
    assert only_user_1["medium_completion"]["n_session_ends"] == 0
    assert only_user_1["medium_completion"]["return_rate"] is None


def test_session_return_gap_must_exceed_1800_actual_seconds() -> None:
    # Between two listens there are 900 actual seconds = 15 minutes.
    # Incorrect legacy code would have classified this as a 30min gap.
    events = pd.DataFrame([
        {"uid": 1, "timestamp": 100, "played_ratio_pct": 40},
        {"uid": 1, "timestamp": 1000, "played_ratio_pct": 75},
        {"uid": 1, "timestamp": 4000, "played_ratio_pct": 95},
    ])
    stats = summarize_session_returns(events, test_start=0, source_last=200000)
    assert stats["low_completion"]["n_session_ends"] == 0
    assert stats["medium_completion"]["n_session_ends"] == 1
    assert stats["medium_completion"]["returned_within_24h"] == 1

def test_unused_nominal_vocab_slots_are_never_recommendable(tmp_path: Path) -> None:
    path = tmp_path / "events.csv"
    fake_events().to_csv(path, index=False)
    # Only nine actual tracks occur in TRAIN, despite a generous vocabulary cap.
    result = prepare(path, DataConfig(users=8, vocab_size=1000, train_events=300,
                                       context_length=8, eval_events_per_user=2))
    assert len(result.train_counts) == len(result.token_to_item) + 2
    assert len(result.train_counts) < 1000
    assert np.all(result.train_counts[2:] > 0)
    assert result.train.target.max() < len(result.train_counts)


def test_oov_targets_are_counted_in_evaluation_denominators(tmp_path: Path) -> None:
    path = tmp_path / "events.csv"
    fake_events().to_csv(path, index=False)
    result = prepare(path, DataConfig(users=8, vocab_size=11,
                                       context_length=8, train_events=500,
                                       eval_events_per_user=20))
    # Future-only item IDs have token=1 and MUST remain in held-out queries.
    assert np.any(result.test.target == 1)
    expected = float(np.mean(result.test.target >= 2))
    assert result.coverage["test_evaluation_target_in_vocab"] == pytest.approx(expected)
    assert expected < 1
