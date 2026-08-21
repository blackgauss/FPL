"""Attach distributional point forecasts to the team-search pipeline.

Fits residual CDFs on a held-out GW slice (no leakage), then for the
prediction horizon [gw_start, gw_end] produces a per-player-GW frame carrying
the point prediction AND the quantile structure of its distribution (the
context bin = position). The team simulator consumes this to MC-sample GW
totals so squads differ by variance, not just mean.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from fpl.dist import QS, fit_residual_cdfs
from fpl.model.inference import load_model
from fpl.model.train import load_training

ALL_POSITIONS = ["GKP", "DEF", "MID", "FWD"]


def collect_residuals(
    processed: str, seasons: list[str],
    *,
    fit_gw_max: int, test_gw_min: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Held-out (actual, predicted, position) residual material.

    Trains a fresh ege model on GW <= fit_gw_max across the two seasons, then
    predicts the held-out GW >= test_gw_min slice; returns aligned arrays so
    CDFs are fit without test rows ever entering the fit.
    """
    import lightgbm as lgb

    by_season = load_training(processed, seasons)
    first = by_season[seasons[0]]
    last = by_season[seasons[-1]]

    mask_fit = last.gw <= fit_gw_max
    X_tr = np.vstack([first.X, last.X[mask_fit]])
    y_tr = np.concatenate([first.y, last.y[mask_fit]])
    ds = lgb.Dataset(X_tr, label=y_tr,
                     feature_name=first.feature_names,
                     categorical_feature=first.categorical)
    model = lgb.train(
        {"objective": "regression", "metric": "mae", "num_leaves": 63,
         "learning_rate": 0.05, "min_child_samples": 20, "verbosity": -1},
        ds, num_boost_round=200,
    )

    mask_test = last.gw >= test_gw_min
    act = last.y[mask_test]
    pred = model.predict(last.X[mask_test])

    plr = pl.read_parquet(f"{processed}/players_{seasons[-1]}.parquet")
    meta = last.meta.filter(pl.col("gw") >= test_gw_min)
    pos = (
        meta.join(plr.select("player_code", "position"),
                  on="player_code", how="left")
        .get_column("position").to_numpy()
    )
    return np.asarray(act, float), np.asarray(pred, float), pos.astype(object)


def distributional_forecast(
    processed: str,
    season: str,
    model_path: str,
    *,
    gw_start: int,
    gw_end: int,
    fit_gw_max: int = 30,
    test_gw_min: int = 31,
) -> pl.DataFrame:
    """Per-player-GW distributional forecast for [gw_start, gw_end].

    Returns one row per (player_code, gw): `pred` (point forecast) + a
    `quantiles` list (points at QS for the player's position's residual CDF).
    """
    from fpl.model.inference import expected_points_horizon

    model = load_model(model_path)
    td = load_training(processed, [season])[season]
    players = pl.read_parquet(f"{processed}/players_{season}.parquet")
    horizon = expected_points_horizon(
        td, model, gw_start=gw_start, gw_end=gw_end, players=players,
    )

    act, pred, pos = collect_residuals(
        processed, ["2024-2025", "2025-2026"],
        fit_gw_max=fit_gw_max, test_gw_min=test_gw_min,
    )
    cdfs = fit_residual_cdfs(act, pred, pos)
    for p in ALL_POSITIONS:  # every position must have a bin
        if p not in cdfs:
            cdfs[p] = np.zeros(len(QS))
    # default bin if a horizon player's position is somehow missing
    cdfs.setdefault("MID", np.zeros(len(QS)))

    pos_df = pl.DataFrame({
        "position": list(cdfs),
        "residual_qs": [cdfs[p].tolist() for p in cdfs],
    })
    return (
        horizon.join(pos_df, on="position", how="left")
        .with_columns(
            (pl.col("expected_points") + pl.col("residual_qs").list.eval(
                pl.element().cast(pl.Float64))
             ).alias("quantiles_list")
        )
        .drop("residual_qs")
        .rename({"expected_points": "pred"})
        .with_columns(
            pl.col("quantiles_list").list.to_struct(
                fields=[f"q{int(q*100)}" for q in QS]).alias("quantiles_struct")
        )
        .drop("quantiles_list")
        .select("player_code", "web_name", "position", "gw", "pred",
                "quantiles_struct")
    )