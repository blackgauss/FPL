"""Distributional player points: CDFs with contextual variance.

Mean-only forecasts make many teams look alike (identical expected totals),
which is exactly the failure the team H2H surfaced. This module captures the
*shape* of each player's points distribution — not just the mean — using the
model's residual structure.

Two ideas:
1. The model predicts E[points | context]. The *noise* around that is the
   model's residual distribution, which we estimate on held-out data (no
   leakage) and bin by a contextual factor (position) because forwards are
   far more volatile than defenders.
2. For a player with point prediction `pred` and context bin, its CDF is
   `pred + residual_quantile(bin, q)`. We keep the quantile vector (a cheap
   CDF estimate — t-digest would be a drop-in if we ever need to *merge*
   compactly) so a simulator can sample GW outcomes and a squad's total
   distribution is distinct even when means coincide.
"""

from __future__ import annotations

import numpy as np

QS = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]


def fit_residual_cdfs(
    actual: np.ndarray,
    predicted: np.ndarray,
    context: np.ndarray,
    qs: list[float] | None = None,
) -> dict[object, np.ndarray]:
    """Per-context-bin residual quantiles (the additive noise CDF).

    actual/predicted are the held-out pairs; `context` gives each row's bin
    (e.g. position codes). Returns {bin: ndarray of residual quantiles at qs}.
    """
    qs = qs or QS
    bins = np.unique(context)
    cdfs: dict[object, np.ndarray] = {}
    residuals = np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)
    for b in bins:
        err = residuals[context == b]
        if err.size == 0:
            cdfs[b] = np.zeros(len(qs))
            continue
        cdfs[b] = np.quantile(err, qs)
    return cdfs


def player_points_quantiles(
    pred: float,
    residual_cdf: np.ndarray,
    qs: list[float] | None = None,
) -> np.ndarray:
    """Predicted points CDF for one player-GW: pred + residual quantiles.

    `residual_cdf` is the bin's residual-quantile vector (from
    fit_residual_cdfs). Returns points at each quantile in qs.
    """
    qs = qs or QS
    return np.asarray(pred, dtype=float) + np.asarray(residual_cdf, dtype=float)


def moments_from_quantiles(points_at_qs: np.ndarray, qs: list[float]) -> dict[str, float]:
    """Approximate mean/std from a CDF's stored quantiles.

    Linear-interpolates the quantile curve over [0,1] (clamping the tail to
    the first/last stored quantile) and integrates x dq — a standard
    quadrature estimate of E[X] from quantiles. Works on non-uniform qs.
    """
    qs_arr = np.asarray(qs, dtype=float)
    x = np.asarray(points_at_qs, dtype=float)
    # full-range grid: prepend (0, first) and append (1, last)
    full_q = np.concatenate([[0.0], qs_arr, [1.0]])
    full_x = np.concatenate([[x[0]], x, [x[-1]]])
    mean = float(np.trapezoid(full_x, x=full_q))
    var = float(np.trapezoid((full_x - mean) ** 2, x=full_q))
    return {"mean": mean, "std": float(np.sqrt(max(var, 0.0)))}


def sample_from_cdf(points_at_qs: np.ndarray, qs: list[float], rng) -> float:
    """Draw one sample from the CDF via linear interpolation (inverse CDF)."""
    u = rng.uniform()
    return float(np.interp(u, np.asarray(qs), np.asarray(points_at_qs)))