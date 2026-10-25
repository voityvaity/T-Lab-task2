"""Causal, globally time-split Yambda listening event preprocessing.

All track mappings and statistics used by the policy come from TRAIN alone.
Test histories may contain events preceding the current test prediction time;
this is a prequential, rather than a frozen-at-test-start evaluation.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import random

import numpy as np
import pandas as pd


REQUIRED = ("uid", "item_id", "timestamp", "played_ratio_pct", "track_length_seconds")


@dataclass(frozen=True)
class DataConfig:
    users: int = 128
    vocab_size: int = 4096  # including PAD and OOV
    context_length: int = 32
    min_history: int = 1
    train_events: int = 30_000
    eval_events_per_user: int = 4
    seed: int = 42
    train_fraction: float = 0.70
    valid_fraction: float = 0.15


@dataclass
class Examples:
    context: np.ndarray
    target: np.ndarray
    completion: np.ndarray
    user: np.ndarray
    timestamp: np.ndarray

    def __len__(self) -> int:
        return len(self.target)


@dataclass
class DataBundle:
    train: Examples
    valid: Examples
    test: Examples
    token_to_item: dict[int, int]
    train_counts: np.ndarray
    train_duration: np.ndarray
    coverage: dict[str, float]
    boundaries: dict[str, int]
    config: DataConfig


def _scan_stats(path: Path) -> tuple[int, int, Counter]:
    """Stream uid,timestamp; global time bounds do not depend on user sampling."""
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, usecols=["uid", "timestamp"])
        return int(frame.timestamp.min()), int(frame.timestamp.max()), Counter()
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Install pyarrow for Parquet input") from exc
    p = pq.ParquetFile(str(path))
    earliest, latest = None, None
    for batch in p.iter_batches(batch_size=262_144, columns=["uid", "timestamp"]):
        df = batch.to_pandas()
        mi, ma = int(df.timestamp.min()), int(df.timestamp.max())
        earliest = mi if earliest is None else min(mi, earliest)
        latest = ma if latest is None else max(ma, latest)
    if earliest is None:
        raise ValueError("Dataset is empty")
    return earliest, latest, Counter()


def _load_events(path: Path, candidate_users: set[int] | None = None) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        if candidate_users is not None:
            frame = frame[frame.uid.isin(candidate_users)]
        return frame.loc[:, list(REQUIRED)].copy()
    import pyarrow.parquet as pq
    chunks = []
    reader = pq.ParquetFile(str(path))
    for batch in reader.iter_batches(batch_size=262_144, columns=list(REQUIRED)):
        frame = batch.to_pandas()
        if candidate_users is not None:
            frame = frame.loc[frame.uid.isin(candidate_users)]
        if not frame.empty:
            chunks.append(frame)
    if not chunks:
        raise ValueError("No events for selected users")
    return pd.concat(chunks, ignore_index=True)


def _select_train_users(path: Path, train_end: int, n: int, seed: int) -> set[int]:
    """Reservoir sampling over eligible TRAIN users, without peeking at their future."""
    counts: Counter = Counter()
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, usecols=["uid", "timestamp"])
        frame = frame[frame.timestamp < train_end]
        counts.update(frame.uid.astype(int).tolist())
    else:
        import pyarrow.parquet as pq
        reader = pq.ParquetFile(str(path))
        for batch in reader.iter_batches(batch_size=262_144, columns=["uid", "timestamp"]):
            df = batch.to_pandas()
            df = df[df.timestamp < train_end]
            counts.update(df.uid.astype(int).tolist())
    eligible = sorted(u for u, c in counts.items() if c >= 2)
    if not eligible:
        raise ValueError("No users have at least two training events")
    rng = random.Random(seed)
    return set(rng.sample(eligible, min(n, len(eligible))))


def _empty(length: int) -> Examples:
    return Examples(np.empty((0, length), dtype=np.int32), np.empty(0, dtype=np.int64),
                    np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int64),
                    np.empty(0, dtype=np.int64))


def _to_examples(rows: list, length: int) -> Examples:
    if not rows:
        return _empty(length)
    ctx, tar, rew, usr, tim = zip(*rows)
    return Examples(np.asarray(ctx, dtype=np.int32), np.asarray(tar, dtype=np.int64),
                    np.asarray(rew, dtype=np.float32), np.asarray(usr, dtype=np.int64),
                    np.asarray(tim, dtype=np.int64))


def prepare(path: str | Path, cfg: DataConfig) -> DataBundle:
    """Read real events, split globally by time and sample only past-aware examples."""
    path = Path(path)
    if cfg.users < 1 or cfg.vocab_size < 4 or cfg.context_length < 1 or cfg.train_events < 1 or cfg.eval_events_per_user < 1:
        raise ValueError("users, vocab_size, context_length and train_events must be positive")
    if cfg.min_history < 1 or not (0 < cfg.train_fraction < 1) or not (0 < cfg.valid_fraction < 1 - cfg.train_fraction):
        raise ValueError("Invalid history or split fractions")
    first, last, _ = _scan_stats(path)
    if first >= last:
        raise ValueError("Dataset needs at least two distinct timestamps")
    span = last - first
    t_train = first + int(span * cfg.train_fraction)
    t_valid = first + int(span * (cfg.train_fraction + cfg.valid_fraction))
    users = _select_train_users(path, t_train, cfg.users, cfg.seed)
    df = _load_events(path, users)
    df = df.sort_values(["uid", "timestamp"], kind="stable").reset_index(drop=True)
    if df[list(REQUIRED)].isna().any().any():
        raise ValueError("Null required event field")
    historic = df[df.timestamp < t_train]
    counts = historic.item_id.value_counts()
    # Deterministic tie-breaking by raw item id, independent of dataframe row order.
    order = sorted(((int(item), int(n)) for item, n in counts.items()), key=lambda p: (-p[1], p[0]))
    selected = order[:cfg.vocab_size - 2]
    vocab = {item: idx + 2 for idx, (item, _) in enumerate(selected)}
    raw_to_token = vocab
    token_to_item = {idx: item for item, idx in vocab.items()}
    item_counts = np.zeros(cfg.vocab_size, dtype=np.int64)
    item_seconds = np.zeros(cfg.vocab_size, dtype=np.float32)
    duration_medians = historic.groupby("item_id", sort=False)["track_length_seconds"].median().to_dict()
    for item, n in selected:
        item_counts[vocab[item]] = n
        item_seconds[vocab[item]] = float(duration_medians.get(item, 0.0))

    rng = random.Random(cfg.seed)
    train_rows: list = []
    train_total = 0
    valid_rows: list = []
    test_rows: list = []
    total = {"valid": 0, "test": 0}
    eligible = {"valid": 0, "test": 0}
    for uid, group in df.groupby("uid", sort=True):
        history: deque[int] = deque(maxlen=cfg.context_length)
        user_valid: deque = deque(maxlen=cfg.eval_events_per_user)
        user_test: deque = deque(maxlen=cfg.eval_events_per_user)
        # All events within the same 5-second timestamp have unknown ordering:
        # evaluate against strictly earlier timestamps, then update the history.
        for ts, tick in group.groupby("timestamp", sort=True):
            pending = []
            for event in tick.itertuples(index=False):
                token = raw_to_token.get(int(event.item_id), 1)
                split = "train" if ts < t_train else ("valid" if ts < t_valid else "test")
                if split != "train":
                    total[split] += 1
                    eligible[split] += int(token >= 2)
                if token >= 2 and len(history) >= cfg.min_history:
                    seq = ([0] * (cfg.context_length - len(history))) + list(history)
                    ratio = min(1.0, max(0.0, float(event.played_ratio_pct) / 100))
                    entry = (seq, token, ratio, int(uid), int(ts))
                    if split == "train":
                        train_total += 1
                        if len(train_rows) < cfg.train_events:
                            train_rows.append(entry)
                        else:
                            j = rng.randrange(train_total)
                            if j < cfg.train_events:
                                train_rows[j] = entry
                    elif split == "valid":
                        user_valid.append(entry)
                    else:
                        user_test.append(entry)
                pending.append(token)
            history.extend(pending)
        valid_rows.extend(user_valid)
        test_rows.extend(user_test)
    if not train_rows or not valid_rows or not test_rows:
        raise ValueError("At least one train/validation/test example required; adjust subcatalog/sample")
    coverage = {name: eligible[name] / max(1, total[name]) for name in ("valid", "test")}
    coverage.update({name + "_eligible": eligible[name] for name in ("valid", "test")})
    coverage.update({name + "_total": total[name] for name in ("valid", "test")})
    return DataBundle(_to_examples(train_rows, cfg.context_length), _to_examples(valid_rows, cfg.context_length),
                      _to_examples(test_rows, cfg.context_length), token_to_item, item_counts, item_seconds,
                      coverage, {"first": first, "last": last, "train_end": t_train, "valid_end": t_valid}, cfg)


def save_metadata(bundle: DataBundle, output: str | Path) -> None:
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    info = {"data_config": asdict(bundle.config), "boundaries": bundle.boundaries,
            "coverage": bundle.coverage, "train_size": len(bundle.train),
            "valid_size": len(bundle.valid), "test_size": len(bundle.test),
            "mapped_items": len(bundle.token_to_item)}
    (root / "dataset_protocol.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
