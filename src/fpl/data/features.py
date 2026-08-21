"""Feature store: per player-GW features for the modeling pipeline.

One pure function (`build_features`) over the parquet dataset frames. No
abstraction beyond polars — this is the "close to the files" production-job
pattern from docs/style/Data.md.

Produced features (per player_id, gw, season):
    team_code          the club the player actually played for that GW
                       (from team_history — players can transfer mid-season)
    opponent_team_code opponent's team_code; null if the player's team had no
                       Premier League match that GW
    was_home           True if played at home
    opponent_elo       opponent's elo at kickoff (from matches)
    home_elo           player's club's elo that GW
    prev_points        total_points of previous GW (shift 1)
    pts_avg_3          mean points over the previous 3 GWs (null < 3 played)
    pts_avg_5          mean points over the previous 5 GWs (null < 5 played)
    next_points        TARGET — next GW total_points (shift -1); null on the
                       final GW (not part of training)

Semantics: FPL points/rolling features cover Premier League matches only
(matches.tournament == 'prem'). A player-GW with no PL match keeps its row with
null opponent/venue (they scored 0 PL points that GW).
"""

from __future__ import annotations

import polars as pl


def build_features(
    gw_stats: pl.DataFrame,
    team_history: pl.DataFrame,
    matches: pl.DataFrame,
    teams: pl.DataFrame,
) -> pl.DataFrame:
    """Return one feature row per (player_id, gw) for a single season."""
    prem = matches.filter(pl.col("tournament") == "prem")

    base = (
        gw_stats.select("player_id", "gw", "total_points")
        .join(
            team_history.select("player_id", "gw", "team_code"),
            on=["player_id", "gw"],
            how="left",
        )
        .sort("player_id", "gw")
    )

    # figure, for each club-GW, the PL match played (opponent + venue + elos)
    home = prem.select(
        pl.col("gw"), pl.col("home_team").alias("team_code"),
        pl.col("away_team").alias("opponent_team_code"),
        pl.lit(True).alias("was_home"),
        pl.col("home_team_elo").alias("home_elo"),
        pl.col("away_team_elo").alias("opponent_elo"),
    )
    away = prem.select(
        pl.col("gw"), pl.col("away_team").alias("team_code"),
        pl.col("home_team").alias("opponent_team_code"),
        pl.lit(False).alias("was_home"),
        pl.col("away_team_elo").alias("home_elo"),
        pl.col("home_team_elo").alias("opponent_elo"),
    )
    fixture_context = pl.concat([home, away]).unique(
        subset=["gw", "team_code"], keep="first"
    )

    base = base.join(fixture_context, on=["gw", "team_code"], how="left")

    # rolling features over previous GWs only (shift then rolling mean)
    base = base.with_columns(
        pl.col("total_points").shift(1).over("player_id").alias("prev_points"),
        pl.col("total_points").shift(1).rolling_mean(3, min_samples=1)
        .over("player_id").alias("pts_avg_3"),
        pl.col("total_points").shift(1).rolling_mean(5, min_samples=1)
        .over("player_id").alias("pts_avg_5"),
        pl.col("total_points").shift(-1).over("player_id").alias("next_points"),
    )

    return base.select(
        "player_id", "gw", "team_code", "opponent_team_code", "was_home",
        "home_elo", "opponent_elo", "prev_points", "pts_avg_3", "pts_avg_5",
        "total_points", "next_points",
    ).drop_nulls(subset=["prev_points", "pts_avg_3", "pts_avg_5", "next_points"])