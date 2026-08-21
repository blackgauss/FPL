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


def point_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Point-outcome metrics for one slice (MAE/RMSE/bias are level metrics)."""
    return {
        "mae": mae(actual, predicted),
        "rmse": rmse(actual, predicted),
        "n": int(actual.size),
        "bias": float(np.mean(predicted - actual)),
    }


# --- stage 1: ranking (is the ORDERING useful, scale-free) ---------------

def spearman(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Rank correlation (pred - actual same ordering => 1)."""
    from scipy.stats import spearmanr

    return float(spearmanr(actual, predicted).statistic)


def topk_hit_rate(
    actual: np.ndarray, predicted: np.ndarray,
    source_gw: np.ndarray, *, top: float = 0.10,
) -> float:
    """Fraction of the actual top-`top` that the model placed in its top-`top`
    per gameweek (0..1). 1.0 == perfect selection ordering."""
    total = 0
    hits = 0
    for gw in np.unique(source_gw):
        idx = np.flatnonzero(source_gw == gw)
        k = max(1, int(np.ceil(len(idx) * top)))
        actual_top = set(idx[np.argsort(actual[idx])[-k:]])
        pred_top = set(idx[np.argsort(predicted[idx])[-k:]])
        hits += len(actual_top & pred_top)
        total += k
    return hits / total if total else 0.0


def pairwise_concordance(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Fraction of player pairs whose model ordering matches the actual ordering."""
    n = len(actual)
    if n < 2:
        return 1.0
    a = actual[:, None] - actual[None, :]
    p = predicted[:, None] - predicted[None, :]
    agree = (a * p) > 0
    np.fill_diagonal(agree, 0)
    return float(agree.sum()) / (n * (n - 1))


def ranking_metrics(
    actual: np.ndarray, predicted: np.ndarray, source_gw: np.ndarray,
    *, top: float = 0.10,
) -> dict:
    """The scale-free 'does the model order players correctly' stage.

    Selection happens within a gameweek, so spearman and concordance are
    computed per-GW and averaged, and topk hit-rate counts per-GW recovery.
    """
    rhos: list[float] = []
    concords: list[float] = []
    topk_hits_total = 0
    topk_total = 0
    for gw in np.unique(source_gw):
        idx = np.flatnonzero(source_gw == gw)
        a, p = actual[idx], predicted[idx]
        if a.size < 2:
            continue
        rhos.append(spearman(a, p))
        concords.append(pairwise_concordance(a, p))
        k = max(1, int(np.ceil(a.size * top)))
        topk_hits_total += len(
            set(idx[np.argsort(a)[-k:]]) & set(idx[np.argsort(p)[-k:]]))
        topk_total += k
    return {
        "spearman_rho": sum(rhos) / len(rhos) if rhos else float("nan"),
        "topk_hit_rate": topk_hits_total / topk_total if topk_total else 0.0,
        "pairwise_concordance": sum(concords) / len(concords)
        if concords else float("nan"),
        "n": int(actual.size),
    }


# --- stage 2: calibration (are the MAGNITUDES trustworthy) ----------------

def ece(actual: np.ndarray, predicted: np.ndarray, *, bins: int = 10) -> float:
    """Expected calibration error: weighted |bin actual mean - bin pred mean|."""
    edges = np.quantile(predicted, np.linspace(0, 1, bins + 1))
    edges[-1] += 1e-9
    total = 0.0
    weight_sum = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        mask = (predicted >= lo) & (predicted < hi)
        if not np.any(mask):
            continue
        w = int(mask.sum())
        total += w * abs(float(predicted[mask].mean()) - float(actual[mask].mean()))
        weight_sum += w
    return total / weight_sum if weight_sum else 0.0


def cal_line(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Least-squares actual = slope*pred + intercept; slope 1 / intercept 0 is
    the perfectly calibrated mean line. polyfit returns [intercept, slope]."""
    intercept, slope = np.polynomial.polynomial.polyfit(
        predicted, actual, 1)
    return {"slope": float(slope), "intercept": float(intercept)}


def calibration_metrics(
    actual: np.ndarray, predicted: np.ndarray, *, bins: int = 10,
) -> dict:
    """The 'does the model's magnitude mean what it claims' stage."""
    var_act = float(np.var(actual))
    var_pred = float(np.var(predicted))
    return {
        "mae": mae(actual, predicted),
        "rmse": rmse(actual, predicted),
        "ece": ece(actual, predicted, bins=bins),
        **cal_line(actual, predicted),
        "variance_ratio": var_act / var_pred if var_pred else float("nan"),
        "n": int(actual.size),
    }