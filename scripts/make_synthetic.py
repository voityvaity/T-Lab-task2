"""Synthetic smoke data (not a source of final research metrics)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/synthetic_listens.csv")
    parser.add_argument("--users", type=int, default=24)
    parser.add_argument("--events", type=int, default=110)
    args = parser.parse_args()
    rng = np.random.default_rng(42)
    records = []
    for uid in range(args.users):
        taste = uid % 8
        for t in range(args.events):
            item = 2 + (taste * 7 + int(rng.integers(0, 10))) % 58
            records.append({"uid": uid, "item_id": item, "timestamp": t * 100 + uid,
                            "is_organic": int(rng.integers(0, 2)),
                            "played_ratio_pct": float(np.clip(20 + (item + taste) % 67
                                                            + rng.normal(0, 6), 1, 130)),
                            "track_length_seconds": 160 + item % 140})
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out, index=False)
    print(f"Saved SYNTHETIC smoke fixture to {out}; do not report its metrics as evidence.")


if __name__ == "__main__":
    main()
