"""Train-only empirical-Bayes completion priors.

Each reward model builds its prior from exactly *its own* observed training
subset. Shrinkage discourages memorization of rare tracks; it provides no
counterfactual validity on unobserved recommendations.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from tlab2.data import Examples


@dataclass(frozen=True)
class CompletionPrior:
    item_mean: np.ndarray
    item_count: np.ndarray
    global_mean: float
    global_median: float
    smoothing: float


def estimate_completion_prior(samples: Examples, vocab_size: int,
                              smoothing: float = 40.0) -> CompletionPrior:
    if vocab_size < 3 or smoothing <= 0:
        raise ValueError("Invalid vocabulary or reward prior smoothing")
    ids = samples.target.astype(np.int64)
    labels = samples.completion.astype(np.float64)
    valid = (ids >= 2) & (ids < vocab_size) & np.isfinite(labels)
    if not np.any(valid):
        raise ValueError("Need observed in-catalog training completions for prior")
    global_mean = float(np.mean(labels[valid]))
    global_median = float(np.median(labels[valid]))
    count = np.bincount(ids[valid], minlength=vocab_size).astype(np.float32)
    summed = np.bincount(ids[valid], weights=labels[valid], minlength=vocab_size)
    estimate = (summed + smoothing * global_mean) / (count + smoothing)
    estimate[:2] = global_mean
    return CompletionPrior(estimate.astype(np.float32), count, global_mean,
                           global_median, float(smoothing))
