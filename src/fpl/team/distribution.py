"""Attach distributional point forecasts to the team-search pipeline.

Given a fit GW threshold, train the point model on fit rows (leakage-clean),
predict the held-out GW slice, then fit a heteroskedastic residual model:
    sigma(X) ~ |actual - mu(X)|
plus a global t-digest of standardized residuals z = residual / sigma(X).

For the prediction horizon [gw_start, gw_end], each player-GW's points CDF is
    mu(X) + sigma(X) * z_q   for every quantile q in QS
—— the mean from the point model, the tail shape from the shared digest,
scaled by that context's own (feature-learned) variance. No binning: price
and form move sigma continuously, and stars don't sit in empty price bins.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from fpl.dist import QS, fit_sigma_and_digest, quantiles_of
from fpl.model.train import load_training


def _fit_split(
    processed: str, seasons: list[str], fit_gw_max: int, test_gw_min: int,
    seed: int = 0,
):
    """Train point model; fit sigma-model + global residual digest on its held-out slice."""
    import lightgbm as lgb

    by_season = load_training(processed, seasons)
    first = by_season[seasons[0]]
    last = by_season[seasons[-1]]

    mask_fit = last.gw <= fit_gw_max
    X_tr = np.vstack([first.X, last.X[mask_fit]])
    y_tr = np.concatenate([first.y, last.y[mask_fit]])
    ds = lgb.Dataset(X_tr, label=y_tr, feature_name=first.feature_names,
                     categorical_feature=first.categorical)
    point = lgb.train(
        {"objective": "regression", "metric": "mae", "num_leaves": 63,
         "learning_rate": 0.05, "min_child_samples": 20, "verbosity": -1,
         "seed": seed, "bagging_seed": seed, "feature_fraction_seed": seed,
         "data_random_seed": seed, "deterministic": True,
         "force_col_wise": True, "num_threads": 1},
        ds, num_boost_round=200,
    )

    mask_test = last.gw >= test_gw_min
    X_test, y_test = last.X[mask_test], last.y[mask_test]
    pred_test = point.predict(X_test)
    sigma_model, digest = fit_sigma_and_digest(
        y_test, pred_test, X_test, first.feature_names, first.categorical,
        learning_params={
            "objective": "regression", "metric": "mae", "num_leaves": 31,
            "learning_rate": 0.05, "min_child_samples": 30, "verbosity": -1,
            "seed": seed, "bagging_seed": seed,
            "feature_fraction_seed": seed, "data_random_seed": seed,
            "deterministic": True, "force_col_wise": True,
            "num_threads": 1,
        })
    return point, sigma_model, digest


def distributional_forecast(
    processed: str,
    season: str,
    gw_start: int,
    gw_end: int,
    *,
    fit_gw_max: int = 30,
    test_gw_min: int = 31,
    seed: int = 0,
) -> pl.DataFrame:
    """Per-player-GW distributional forecast for [gw_start, gw_end].

    Returns (player_code, web_name, position, gw, pred, quantiles_struct):
    pred = mu(X); quantiles_struct = mu + sigma * z_q for each q in QS.
    """
    point, sigma_model, digest = _fit_split(
        processed, ["2024-2025", "2025-2026"],
        fit_gw_max=fit_gw_max, test_gw_min=test_gw_min, seed=seed,
    )

    td = load_training(processed, [season])[season]
    players = pl.read_parquet(f"{processed}/players_{season}.parquet")

    # horizon prediction rows are td rows with gw in [gw_start-1, gw_end-1]
    mask = (td.gw >= gw_start - 1) & (td.gw <= gw_end - 1)
    X_h = td.X[mask]
    mu = point.predict(X_h)
    sigma = np.maximum(sigma_model.predict(X_h), 1e-4)
    z_q = quantiles_of(digest)  # standardized residual quantiles (shared shape)
    # 2D: [n_rows, n_q] each row = mu + sigma * z_q
    quant = mu[:, None] + sigma[:, None] * z_q[None, :]

    meta = td.meta.filter(pl.col("gw").is_between(gw_start - 1, gw_end - 1))
    frame = (
        meta.with_columns(
            pl.Series("pred", mu),
            pl.Series("sigma", sigma),
            (pl.col("gw") + 1).alias("gw_target"),
        )
        .join(players.select("player_id", "web_name", "position"),
              on="player_id", how="left")
        .with_columns(
            pl.Series("quantiles_array", [list(r) for r in quant])
        )
    )
    # store quantiles as a struct in canonical QS field names
    qcols = [f"q{int(q*100)}" for q in QS]
    st = frame.with_columns(
        pl.col("quantiles_array").list.to_struct(fields=qcols).alias("quantiles_struct")
    ).drop("quantiles_array")

    return st.select(
        "player_code", "web_name", "position",
        pl.col("gw_target").alias("gw"), "pred", "quantiles_struct",
    ).sort("player_code", "gw")
