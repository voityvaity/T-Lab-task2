from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tlab2.data import DataConfig, prepare, save_metadata


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
    assert np.all(a.train.target >= 2) and np.all(a.test.target >= 2)
    assert a.train.context.shape[1] == 8
    assert np.all((a.train.completion >= 0) & (a.train.completion <= 1))
    assert 1.0 in a.train.completion
    save_metadata(a, tmp_path / "outputs")
    assert (tmp_path / "outputs/dataset_protocol.json").is_file()


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
