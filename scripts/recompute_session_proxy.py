"""Recalculate actual Yambda session-return diagnostics without retraining GPT-2.

The cohort is sampled only from TRAIN, exactly as the original completed
experiment. Guard against silently using a different underlying Parquet file
or user cohort. Observation-only descriptive association, NOT causal retention.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tlab2.data import _load_events, _scan_stats, _select_train_users, summarize_session_returns


def dataset_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_sessions(path: Path, *, users: int, seed: int,
                   train_fraction: float, valid_fraction: float,
                   expected_dataset_hash: str = "",
                   expected_cohort_hash: str = "") -> dict:
    if not 0 < train_fraction < 1 or not 0 < valid_fraction < 1 - train_fraction:
        raise ValueError("Invalid train/validation temporal split")
    if users <= 0:
        raise ValueError("users must be positive")
    if not path.is_file():
        raise FileNotFoundError(path)
    src_hash = dataset_hash(path)
    if expected_dataset_hash and src_hash != expected_dataset_hash:
        raise ValueError("Input Yambda file differs from original measured experiment")
    first, last, _ = _scan_stats(path)
    train_end = first + int((last - first) * train_fraction)
    test_start = first + int((last - first) * (train_fraction + valid_fraction))
    selected = _select_train_users(path, train_end, users, seed)
    cohort_hash = hashlib.sha256(",".join(map(str, sorted(selected))).encode()).hexdigest()
    if expected_cohort_hash and cohort_hash != expected_cohort_hash:
        raise ValueError("Train-only user cohort differs from original experiment")
    events = _load_events(path, selected).sort_values(["uid", "timestamp"],
                                                      kind="stable").reset_index(drop=True)
    measured = summarize_session_returns(
        events, test_start, last,
        session_gap_seconds=1800, return_horizon_seconds=86400,
    )
    return {
        "source": "Yambda-50M original Parquet / listens only",
        "not_causal": True,
        "timestamp_units": "seconds_quantized_to_5_seconds",
        "session_gap_seconds": 1800,
        "return_window_seconds": 86400,
        "input_file_sha256": src_hash,
        "selected_train_user_cohort_sha256": cohort_hash,
        "users": len(selected),
        "train_end": train_end,
        "test_start": test_start,
        "last": last,
        "results": measured,
        "interpretation": "Observed return after last event in a session; no exposure propensities.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--users", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=.7)
    parser.add_argument("--valid-fraction", type=float, default=.15)
    parser.add_argument("--expected-dataset-hash", default="")
    parser.add_argument("--expected-cohort-hash", default="")
    args = parser.parse_args()
    results = count_sessions(
        args.data, users=args.users, seed=args.seed,
        train_fraction=args.train_fraction, valid_fraction=args.valid_fraction,
        expected_dataset_hash=args.expected_dataset_hash,
        expected_cohort_hash=args.expected_cohort_hash
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
