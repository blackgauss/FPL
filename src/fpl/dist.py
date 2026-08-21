"""Distributional player points: per-context t-digests of model residuals.

Mean-only forecasts make many teams look alike (identical expected totals),
which is exactly the failure the team H2H surfaced. This module captures the
*shape* of each player's points distribution — not just the mean — using the
model's residual structure.

Two ideas:
1. The model predicts E[points | context]. The noise around that is the
   residual distribution, estimated on held-out data (no leakage) and binned
   by a contextual factor tuple `(position, price_band)` — cheap players are
   measurably burstier (higher coefficient-of-variation of points in every
   position), so a cheap over-hyped pick gets a wider/tail-heavier posterior
   than a premium star at the same predicted mean.
2. The residual distribution is stored as a **t-digest** (`tdigest.TDigest`):
   a standard, compact CDF estimate that supports positional accuracy at the
   tails, `update()` to fold in new GW residuals over the season without
   refitting, and `merge()` to combine digests across seasons. The quantile
   vector a player's CDF needs is simply read off the digest via
   `.percentile()` — no reinventing of quantile storage.

The team simulator MC-samples squad totals from these digests, so squads
differ by variance and tail risk, not just mean.
"""

from __future__ import annotations

import numpy as np
from tdigest import TDigest

QS = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

# price bands (£m) — cheap players are measurably more volatile (higher CV of
# points) in every position, so binning residuals by price captures a real
# contextual variance difference, not just a modeling artifact.
PRICE_BANDS = [(0, 5.0), (5.0, 7.0), (7.0, 9.0), (9.0, 100.0)]


def price_band(now_cost: float) -> str:
    for lo, hi in PRICE_BANDS:
        if lo <= now_cost < hi:
            return f"£{lo:.0f}-{hi:.0f}m"
    return "£9+m"


def fit_residual_cdfs(
    actual: np.ndarray,
    predicted: np.ndarray,
    context: np.ndarray,
) -> dict[object, TDigest]:
    """Per-context-bin residual distribution as a t-digest.

    actual/predicted are the held-out pairs; `context` gives each row's bin
    (e.g. a position code, or a (position, price_band) tuple). Returns
    {bin: TDigest} of residuals (actual - predicted). Each digest can be
    `update`d later with new GW residuals or `merge`d across seasons.

    Binning choice: position-only is stable at small holdout sizes; price
    bands (see PRICE_BANDS) show real separation but need more held-out GWs
    before per-position price bins have enough samples.
    """
    residuals = np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)
    cdfs: dict[object, TDigest] = {}
    keys = {tuple(k) if isinstance(k, (tuple, list)) else k for k in context.tolist()}
    for b in keys:
        mask = np.asarray(
            [(tuple(k) if isinstance(k, (tuple, list)) else k) == b
             for k in context.tolist()])
        digest = TDigest()
        for e in residuals[mask]:
            digest.update(float(e))
        cdfs[b] = digest
    return cdfs


def update_residual_cdfs(
    cdfs: dict[object, TDigest],
    actual: np.ndarray,
    predicted: np.ndarray,
    context: np.ndarray,
) -> dict[object, TDigest]:
    """Fold new residuals into existing per-bin digests (incremental refresh)."""
    residuals = np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)
    for b in {tuple(k) for k in context.tolist()}:
        digest = cdfs.setdefault(b, TDigest())
        mask = np.asarray([tuple(k) == b for k in context.tolist()])
        for e in residuals[mask]:
            digest.update(float(e))
    return cdfs


def quantiles_of(digest: TDigest, qs: list[float] | None = None) -> np.ndarray:
    """Residual quantiles at `qs` read off a t-digest (percentile stores %)."""
    qs = qs or QS
    return np.asarray([digest.percentile(q * 100) for q in qs], dtype=float)


def moments_from_quantiles(points_at_qs: np.ndarray, qs: list[float]) -> dict[str, float]:
    """Approximate mean/std from a CDF's quantile vector.

    Linear-interpolates the quantile curve over [0,1] (clamping the tail to
    the first/last stored quantile) and integrates x dq — a standard
    quadrature estimate of E[X] from quantiles.
    """
    qs_arr = np.asarray(qs, dtype=float)
    x = np.asarray(points_at_qs, dtype=float)
    full_q = np.concatenate([[0.0], qs_arr, [1.0]])
    full_x = np.concatenate([[x[0]], x, [x[-1]]])
    mean = float(np.trapezoid(full_x, x=full_q))
    var = float(np.trapezoid((full_x - mean) ** 2, x=full_q))
    return {"mean": mean, "std": float(np.sqrt(max(var, 0.0)))}


def sample_from_digest(digest: TDigest, rng) -> float:
    """Inverse-CDF sample from a t-digest (uniform -> percentile)."""
    return float(digest.percentile(rng.uniform() * 100))