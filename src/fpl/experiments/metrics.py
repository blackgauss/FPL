"""Small, independent metric functions per output kind.

These are deliberately free of any polars/gym import: quick to test, and only
concerned with arrays of point outcomes.
"""

from __future__ import annotations

import numpy as np


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.abs(actual - predicted).mean())


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def pinball(actual: np.ndarray, predicted: np.ndarray, q: float) -> float:
    error = actual - predicted
    return float(np.mean(np.where(error >= 0, q * error, (q - 1) * error)))


def point_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    cohort: str | None = None,
) -> dict:
    """Point-outcome metrics for one slice (optionally a cohort label)."""
    base = {
        "mae": mae(actual, predicted),
        "rmse": rmse(actual, predicted),
        "n": int(actual.size),
        "bias": float(np.mean(predicted - actual)),
    }
    if cohort is not None:
        base["cohort"] = cohort
    return base