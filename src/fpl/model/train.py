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
from pathlib import Path

import numpy as np
import polars as pl

from fpl.domain import Position

FEATURE_COLUMNS = [
    "team_code", "position", "now_cost", "ep_next", "had_match",
    "was_home", "opponent_elo", "home_elo", "pts_avg_3", "pts_avg_5",
]
CATEGORY_COLUMNS = ["team_code", "position"]


def _stable_categories(column: str) -> list[str] | None:
    """Fixed category vocabulary so integer codes agree ACROSS seasons.

    Per-frame `cast(Categorical)` assigns codes by first-appearance order, so
    vstacking two seasons' matrices (experiments/run) silently mismatches the
    categorical columns. Enum casts use a vocabulary fixed in code instead;
    returning None falls back to frame-local encoding.
    """
    if column == "position":
        return sorted(p.value for p in Position)
    return None


@dataclass(frozen=True)
class TrainingData:
    """Frozen training pair: feature matrix (numpy) + target + row metadata."""

    X: np.ndarray
    y: np.ndarray
    gw: np.ndarray            # original GW per row (for time-based split)
    feature_names: list[str]
    categorical: list[int]    # column indices LightGBM should treat as categorical
    meta: pl.DataFrame        # (player_id, gw, season) per row for later joins


def load_training(processed: str | Path, seasons: list[str],
                  feature_columns: list[str] | None = None,
                  categorical_columns: list[str] | None = None,
                  players: dict[str, pl.DataFrame] | None = None,
                  require_target: bool = True,
                  ) -> dict[str, TrainingData]:
    """Read the feature store for each season and assemble a TrainingData.

    `processed` is data/processed (per config), `seasons` a list of labels.
    Returns {season: TrainingData}. This is the single read path for training,
    experiments, and serving — scripts should not hand-roll the 3-file load.
    `feature_columns`/`categorical_columns` select the model's feature set
    (ablations, experiments); `require_target=False` keeps rows without a
    next-points label (used for scoring a season-start / pre-season window
    where the target doesn't exist yet).
    """
    result: dict[str, TrainingData] = {}
    for season in seasons:
        feat = pl.read_parquet(f"{processed}/features_{season}.parquet")
        gw_stats = pl.read_parquet(f"{processed}/gw_stats_{season}.parquet")
        plr = players[season] if players is not None else \
            pl.read_parquet(f"{processed}/players_{season}.parquet")
        result[season] = assemble(feat, plr, gw_stats, season,
                                  feature_columns=feature_columns,
                                  categorical_columns=categorical_columns,
                                  require_target=require_target)
    return result


def assemble(df: pl.DataFrame, players: pl.DataFrame, gw_stats: pl.DataFrame,
             season: str | None = None,
             feature_columns: list[str] | None = None,
             categorical_columns: list[str] | None = None,
             require_target: bool = True) -> TrainingData:
    """Prepare a model-ready matrix from feature-store + player + gw_stats rows.

    `players` MUST be the frame of the same season as `df` (it carries the
    season-local position codes). Joins happen on `player_code` and
    (player_id, gw) for price — the stable code never crosses seasons.
    `feature_columns` selects a subset of the canonical features (for quick
    ablations); categorical columns that are dropped are simply absent.
    `require_target` drops rows without `next_points` (trainable); set False
    to keep them for inference at season start (target unknown yet).
    """
    if feature_columns is None:
        feature_columns = FEATURE_COLUMNS
    cat_cols = CATEGORY_COLUMNS if categorical_columns is None \
        else categorical_columns
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

    # rows without a target are not trainable; the feature store keeps them
    # (for scoring/inference at season start) but training drops them here
    if require_target:
        df = df.filter(pl.col("next_points").is_not_null())

    cat_selected = [c for c in cat_cols if c in feature_columns]

    def _encode(c: str):
        if df.schema[c] != pl.String:
            return pl.col(c).cast(pl.Int64)  # team_code is already int
        vocab = _stable_categories(c)
        if vocab is not None:
            return pl.col(c).cast(pl.Enum(vocab)).to_physical()
        return pl.col(c).cast(pl.Categorical).to_physical()

    X = df.select(feature_columns).with_columns(
        # encode string categoricals into int codes with a cross-season stable
        # vocabulary where one exists; unknown values raise, not mis-code
        *[_encode(c) for c in cat_selected]
    )
    feature_names = list(feature_columns)
    categorical = [feature_names.index(c) for c in cat_selected]

    return TrainingData(
        X=X.to_numpy(),
        y=df.get_column("next_points").to_numpy().astype(np.float64),
        gw=df.get_column("gw").to_numpy().astype(np.int64),
        feature_names=feature_names,
        categorical=categorical,
        meta=df.select("player_id", "player_code", "gw", "season"),
    )
