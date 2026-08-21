"""Heteroskedastic distributional player points.

Mean-only forecasts make many teams look alike (identical expected totals),
which is exactly the failure the team H2H surfaced. This module captures the
*shape* of each player's points distribution — not just the mean — using the
model's residual structure.

Design (no binning):

1. The point model predicts mu(X) = E[points | features]. The noise around it
   is heteroskedastic: its scale is a continuous function of the features
   (price, form, position, opponent strength...). We model that scale with a
   second regressor  sigma(X) ~= |actual - mu(X)| fit on held-out residuals
   — expensive players and value picks get different widths *continuously*,
   never bucketed into bins where stars would be too few to populate.

2. The standardized residuals z = (actual - mu(X)) / sigma(X) are pooled into
   a single **t-digest** — a standard, compact CDF estimate with `update()`
   (fold in new GW residuals without refitting) and `merge()` across seasons.

3. A player-GW's points CDF is then  pred(X) + sigma(X) * z_q  for each
   quantile q — the mean from the point model, the shape from the shared
   residual digest, scaled by that context's own variance.

This avoids two failure modes binning had: double-counting features already
in the model, and statistically-empty premium-price bins (the very players
team selection cares about).
"""

from __future__ import annotations

import numpy as np
from tdigest import TDigest

QS = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]


def fit_sigma_and_digest(
    actual: np.ndarray,
    predicted: np.ndarray,
    X: np.ndarray,
    feature_names: list[str],
    categorical: list[int],
    learning_params: dict | None = None,
) -> tuple[object, TDigest]:
    """Learned per-row sigma + a global t-digest of standardized residuals.

    - sigma_model: regressor predict |residual| from the model's features.
    - digest: t-digest of z = residual / sigma(X) (the common noise shape).

    Both are fit on held-out material only (no leakage). Returns
    (sigma_model, digest).
    """
    import lightgbm as lgb

    residual = np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)
    ds = lgb.Dataset(X, label=np.abs(residual),
                     feature_name=feature_names,
                     categorical_feature=categorical)
    params = dict(learning_params or {}) or {
        "objective": "regression", "metric": "mae", "num_leaves": 31,
        "learning_rate": 0.05, "min_child_samples": 30, "verbosity": -1,
    }
    sigma_model = lgb.train(params, ds, num_boost_round=100)

    sigma = np.maximum(sigma_model.predict(X), 1e-4)
    z = residual / sigma

    digest = TDigest()
    for zi in z:
        digest.update(float(zi))
    return sigma_model, digest


def update_standardized(
    digest: TDigest,
    actual: np.ndarray,
    predicted: np.ndarray,
    sigma: np.ndarray,
) -> TDigest:
    """Fold new standardized residuals into the digest (incremental refresh)."""
    z = (np.asarray(actual, float) - np.asarray(predicted, float)) / sigma
    for zi in z:
        digest.update(float(zi))
    return digest


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