"""Assemble model-ready training data from the feature store.

Reads data/processed/features_{season}.parquet and player/gw_stats frames,
returns a numpy training pair for tree models. No leakage: rows are already
shifted to the next-GW target by the feature store, and model evaluation
splits by game week.

Rows with no Premier League match that GW (opponent/venue null) are kept with
an explicit `had_match` flag — absence of a match is itself signal (0 points).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

FEATURE_COLUMNS = [
    "team_code", "position", "now_cost", "ep_next", "had_match",
    "was_home", "opponent_elo", "home_elo", "pts_avg_3", "pts_avg_5",
]
CATEGORY_COLUMNS = ["team_code", "position"]


@dataclass(frozen=True)
class TrainingData:
    """Frozen training pair: feature matrix (numpy) + target + row metadata."""

    X: np.ndarray
    y: np.ndarray
    gw: np.ndarray            # original GW per row (for time-based split)
    feature_names: list[str]
    categorical: list[int]    # column indices LightGBM should treat as categorical
    meta: pl.DataFrame        # (player_id, gw, season) per row for later joins


def assemble(df: pl.DataFrame, players: pl.DataFrame, gw_stats: pl.DataFrame,
             season: str | None = None) -> TrainingData:
    """Prepare a model-ready matrix from feature-store + player + gw_stats rows.

    `players` MUST be the frame of the same season as `df` (it carries the
    season-local position codes). Joins happen on `player_code` and
    (player_id, gw) for price — the stable code never crosses seasons.
    """
    # join player position on the stable cross-season code
    df = df.join(
        players.select("player_code", "position"),
        on="player_code",
        how="left",
    )
    # join per-GW price (features table has neither now_cost nor ep_next)
    df = df.join(
        gw_stats.select("player_id", "gw", "now_cost", "ep_next"),
        on=["player_id", "gw"],
        how="left",
    ).sort("player_id", "gw")

    had_match = pl.col("opponent_team_code").is_not_null()
    df = df.with_columns(
        had_match.alias("had_match"),
        pl.when(had_match).then(pl.col("was_home")).otherwise(False).alias("was_home"),
        pl.col("opponent_elo").fill_null(0.0),
        pl.col("home_elo").fill_null(0.0),
        pl.col("now_cost").fill_null(0.0),
        pl.col("ep_next").fill_null(0.0),
    )
    if season is not None:
        df = df.with_columns(pl.lit(season).alias("season"))
    else:
        df = df.with_columns(pl.lit("unknown").alias("season"))

    X = df.select(FEATURE_COLUMNS).with_columns(
        # encode string categoricals into int codes; team_code is already int
        *[pl.col(c).cast(pl.Categorical).to_physical() if df.schema[c] == pl.String
          else pl.col(c).cast(pl.Int64)
          for c in CATEGORY_COLUMNS]
    )
    feature_names = FEATURE_COLUMNS
    categorical = [feature_names.index(c) for c in CATEGORY_COLUMNS]

    return TrainingData(
        X=X.to_numpy(),
        y=df.get_column("next_points").to_numpy().astype(np.float64),
        gw=df.get_column("gw").to_numpy().astype(np.int64),
        feature_names=feature_names,
        categorical=categorical,
        meta=df.select("player_id", "player_code", "gw", "season"),
    )