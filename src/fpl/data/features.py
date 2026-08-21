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
    pts_avg_3          mean points over previous GWs, window 3 (from the very
                       first available previous GW — no waiting period)
    pts_avg_5          mean points over previous GWs, window 5 (same)
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
    players: pl.DataFrame,
    carryover: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Return one feature row per (player_id, gw) for a single season.

    Rows are keyed by season-local `player_id` (FPL element id) but ALSO carry
    the stable cross-season `player_code`. Downstream training must join player
    metadata on `player_code` — never `player_id`, which is reused for different
    players across seasons.

    `carryover` seeds the season's first GW with end-of-previous-season rolling
    stats (`player_code -> pts_avg_3/5, prev_points`), so a GW1 row isn't
    dropped for lacking current-season history — the "beginning of season"
    case (e.g. 2026-27 GW1 from 2025-26 carryover). Players without a carryover
    (new signings) keep null rolling features at GW1; those rows are dropped.
    """
    prem = matches.filter(pl.col("tournament") == "prem")

    base = (
        gw_stats.select("player_id", "gw", "total_points")
        .join(
            players.select("player_id", "player_code", "team_code"),
            on="player_id",
            how="left",
        )
        # team_history may be absent (legacy layout); fall back to the club the
        # player's season assigned — never leave team_code null (models choke).
        .join(
            team_history.select("player_id", "gw",
                                pl.col("team_code").alias("th_team_code")),
            on=["player_id", "gw"],
            how="left",
        )
        .with_columns(
            pl.coalesce("th_team_code", "team_code").alias("team_code")
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

    # season-start: seed the first GW's rolling features from prior-season
    # carryover instead of dropping the row for lacking current-season history.
    if carryover is not None and base.height:
        first_gw = base.get_column("gw").min()
        carry = carryover.select(
            "player_code",
            pl.col("pts_avg_3").alias("co_avg3"),
            pl.col("pts_avg_5").alias("co_avg5"),
            pl.col("prev_points").alias("co_prev"),
        )
        base = base.join(carry, on="player_code", how="left")
        base = base.with_columns(
            pl.when(pl.col("gw") == first_gw)
            .then(pl.coalesce("prev_points", "co_prev")).otherwise(pl.col("prev_points"))
            .alias("prev_points"),
            pl.when(pl.col("gw") == first_gw)
            .then(pl.coalesce("pts_avg_3", "co_avg3")).otherwise(pl.col("pts_avg_3"))
            .alias("pts_avg_3"),
            pl.when(pl.col("gw") == first_gw)
            .then(pl.coalesce("pts_avg_5", "co_avg5")).otherwise(pl.col("pts_avg_5"))
            .alias("pts_avg_5"),
        ).drop(["co_avg3", "co_avg5", "co_prev"])

    return base.select(
        "player_id", "player_code", "gw", "team_code", "opponent_team_code",
        "was_home", "home_elo", "opponent_elo", "prev_points", "pts_avg_3",
        "pts_avg_5", "total_points", "next_points",
    ).drop_nulls(subset=["prev_points", "pts_avg_3", "pts_avg_5"])
