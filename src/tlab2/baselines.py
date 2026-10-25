"""Lightweight non-neural ranking controls inspired by sequential-rule research.

These are *our* simple implementations, not a reproduction of Spotify's
published Sequential Rules / PCTM. All statistics originate in TRAIN only.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import torch

from tlab2.data import Examples


class PopularityBaseline:
    def __init__(self, train_counts: np.ndarray):
        counts = np.asarray(train_counts, dtype=np.float64).copy()
        counts[:2] = 0
        if not (counts[2:] > 0).any():
            raise ValueError("Popularity baseline needs at least one known training item")
        self.log_pop = np.log1p(counts).astype(np.float32)
        self.log_pop[:2] = -1e6
        self.vocab_size = len(counts)

    def score(self, context: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(self.log_pop, device=context.device).unsqueeze(0).expand(len(context), -1)


class SequentialRulesBaseline:
    """Weighted last-three-item transition counts + train popularity backoff.

    Score = 0.01 log(1 + popularity) plus 1/distance-weighted conditional
    probabilities for transitions observed among sampled TRAIN examples.
    Only past tokens in each training example contribute; no test labels.
    """

    def __init__(self, train: Examples, train_counts: np.ndarray, history: int = 3):
        if history < 1:
            raise ValueError("history must be positive")
        self.pop = PopularityBaseline(train_counts)
        self.history = history
        edges: dict[int, Counter[int]] = defaultdict(Counter)
        for ctx, target in zip(train.context, train.target):
            src = int(ctx[-1])
            if src >= 2 and target >= 2:
                edges[src][int(target)] += 1
        self.rules = {}
        for src, to_counts in edges.items():
            ids = np.array(sorted(to_counts), dtype=np.int64)
            values = np.array([to_counts[int(j)] for j in ids], dtype=np.float32)
            self.rules[src] = (ids, values / values.sum())

    def score(self, context: torch.Tensor) -> torch.Tensor:
        contexts = context.detach().cpu().numpy()
        result = np.broadcast_to(0.01 * self.pop.log_pop, (len(contexts), self.pop.vocab_size)).copy()
        for row, ctx in enumerate(contexts):
            for age, src in enumerate(ctx[-self.history:][::-1], start=1):
                entry = self.rules.get(int(src))
                if entry is None:
                    continue
                ids, conditional = entry
                result[row, ids] += conditional / age
        result[:, :2] = -1e6
        return torch.as_tensor(result, dtype=torch.float32, device=context.device)
