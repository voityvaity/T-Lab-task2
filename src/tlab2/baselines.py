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


class SmoothedTransitionBaseline:
    """Multi-lag sparse conditional evidence + train-only popularity background.

    Inspired by PCTM's count smoothing but NOT a reproduction of Spotify's
    official implementation. Built from the *same sampled training examples*
    as GPT-2, with deterministic tie-breaking by token ID.
    """

    def __init__(self, train: Examples, train_counts: np.ndarray,
                 max_lag: int = 5, decay: float = 0.7,
                 smoothing: float = 20.0, pop_boost: float = 0.03):
        if max_lag < 1 or not 0 < decay <= 1 or smoothing <= 0 or pop_boost < 0:
            raise ValueError("Invalid smoothed-transition parameters")
        self.pop = PopularityBaseline(train_counts)
        self.max_lag = max_lag
        self.decay = decay
        self.pop_boost = pop_boost
        edges: dict[int, dict[int, float]] = defaultdict(dict)
        for context, target in zip(train.context, train.target):
            dst = int(target)
            if dst < 2:
                continue
            for age, source in enumerate(context[-max_lag:][::-1], start=0):
                src = int(source)
                if src < 2:
                    continue
                w = decay ** age
                edges[src][dst] = edges[src].get(dst, 0.0) + w
        vocabulary = len(train_counts) - 2
        self.rules = {}
        for src, targets in edges.items():
            ids = np.asarray(sorted(targets), dtype=np.int64)
            values = np.asarray([targets[int(item)] for item in ids], dtype=np.float32)
            # Non-parametric smoothed conditional evidence, not a calibrated
            # probability or official PCTM kernel.
            evidence = np.log1p(values * vocabulary / smoothing)
            self.rules[src] = (ids, evidence)

    def score(self, context: torch.Tensor) -> torch.Tensor:
        contexts = context.detach().cpu().numpy()
        background = self.pop_boost * self.pop.log_pop
        scores = np.broadcast_to(background, (len(contexts), len(background))).copy()
        for i, ctx in enumerate(contexts):
            for age, source in enumerate(ctx[-self.max_lag:][::-1], start=0):
                pair = self.rules.get(int(source))
                if pair is None:
                    continue
                indices, evidence = pair
                scores[i, indices] += (self.decay ** age) * evidence
        scores[:, :2] = -1e6
        return torch.as_tensor(scores, dtype=torch.float32, device=context.device)
