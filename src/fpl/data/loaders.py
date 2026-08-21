"""Loaders: pure functions turning FPL-Core CSV paths into typed DataFrames.

Each loader fixes the documented raw-format quirks exactly once (float team codes,
stringified numerics, verbose positions), so downstream code only ever sees the
canonical schema. All paths are injected — nothing hardcoded.

Canonical schemas (the contract):

    players:     player_code Int64, player_id Int64, first_name, second_name,
                 web_name, team_code Int64, position GKP|DEF|MID|FWD
    teams:       code Int64, id Int64, name, short_name,
                 strength Float64|null, elo Float64|null
    gw_stats:    player_id Int64, gw Int64, web_name, second_name, status,
                 total_points/minutes/goals_scored/assists/bonus/bps/saves/starts Int64,
                 now_cost/form/ep_next/ep_this/selected_by_percent Float64|null
                 (now_cost in decimal millions)
    match_stats: player_id Int64, match_id str, minutes_played Float64,
                 goals/assists/xg/xa Float64
    matches:     match_id str, gw Int64, kickoff_time str, home_team/away_team Int64
                 (team codes), home_score/away_score Int64|null,
                 home_team_elo/away_team_elo Float64|null, tournament str,
                 finished bool
    team_history: player_id Int64, gw Int64, team_code Int64 (club per GW)
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

POSITION_MAP = {
    "Goalkeeper": "GKP",
    "Defender": "DEF",
    "Midfielder": "MID",
    "Forward": "FWD",
    "Unknown": "UNK",  # legacy (2024-25) players.csv uses this
}

# gw_stats columns the legacy (2024-25) long-table playerstats.csv does NOT have.
# They are typed-null so a legacy season still satisfies the shared gw_stats schema.
LEGACY_GW_STATS_MISSING: dict[str, pl.DataType] = {
    "web_name": pl.String,
    "second_name": pl.String,
    "minutes": pl.Int64,
    "goals_scored": pl.Int64,
    "assists": pl.Int64,
    "saves": pl.Int64,
    "starts": pl.Int64,
}

_GW_STATS_COLUMNS = {
    "id": "player_id",
    "gw": "gw",
    "web_name": "web_name",
    "second_name": "second_name",
    "status": "status",
    "total_points": "total_points",
    "minutes": "minutes",
    "goals_scored": "goals_scored",
    "assists": "assists",
    "bonus": "bonus",
    "bps": "bps",
    "saves": "saves",
    "starts": "starts",
    "now_cost": "now_cost",
    "form": "form",
    "ep_next": "ep_next",
    "ep_this": "ep_this",
    "selected_by_percent": "selected_by_percent",
}


def load_players_csv(path: str | Path) -> pl.DataFrame:
    """Load players.csv with stable codes as ints and short position names."""
    return (
        pl.read_csv(path)
        .with_columns(
            pl.col("player_code").cast(pl.Int64),
            pl.col("player_id").cast(pl.Int64),
            pl.col("team_code").cast(pl.Int64),
            pl.col("position").replace_strict(POSITION_MAP),
        )
        .select("player_code", "player_id", "first_name", "second_name", "web_name",
                "team_code", "position")
    )


def load_teams_csv(path: str | Path) -> pl.DataFrame:
    """Load teams.csv keyed on stable `code`; strength/elo nullable (pre-season gaps)."""
    return (
        pl.read_csv(path, schema_overrides={"strength": pl.Float64, "elo": pl.Float64})
        .with_columns(
            pl.col("code").cast(pl.Int64),
            pl.col("id").cast(pl.Int64),
        )
        .select("code", "id", "name", "short_name", "strength", "elo")
    )


def load_gw_stats_csv(path: str | Path) -> pl.DataFrame:
    """Load player_gameweek_stats.csv (discrete per-GW) with renamed, cast columns."""
    return (
        pl.read_csv(path, schema_overrides={col: pl.Float64 for col in (
            "now_cost", "form", "ep_next", "ep_this", "selected_by_percent")})
        .rename(_GW_STATS_COLUMNS)
        .with_columns(
            pl.col("player_id").cast(pl.Int64),
            pl.col("gw").cast(pl.Int64),
            *[pl.col(c).cast(pl.Int64) for c in (
                "total_points", "minutes", "goals_scored", "assists",
                "bonus", "bps", "saves", "starts")],
        )
        .select(*_GW_STATS_COLUMNS.values())
    )


def load_legacy_gw_stats_csv(path: str | Path) -> pl.DataFrame:
    """Load legacy (2024-25) long-table playerstats.csv into the shared gw_stats schema.

    The legacy table records one row per player per GW (has its own `gw` column)
    and a cumulative `total_points` (confirmed: sums of per-GW `event_points`
    equal the season-end total). `total_points` is therefore recomputed as the
    per-GW `event_points` so the shared, discrete-per-GW contract holds across
    layouts. Several modern columns are absent and emitted as typed-null.
    """
    float_cols = ["now_cost", "form", "ep_next", "ep_this", "selected_by_percent"]
    int_cols = ["bonus", "bps"]

    df = (
        pl.read_csv(path)
        .rename(_GW_STATS_COLUMNS)
        .with_columns(
            pl.col("player_id").cast(pl.Int64),
            pl.col("gw").cast(pl.Int64),
            *[pl.col(c).cast(pl.Float64) for c in float_cols],
            *[pl.col(c).cast(pl.Int64) for c in int_cols],
            pl.col("event_points").cast(pl.Int64).alias("total_points"),
        )
    )
    # emit missing contract columns as typed-null; select in canonical order
    columns = [
        pl.col(c) if c in df.columns else pl.lit(None, dtype=LEGACY_GW_STATS_MISSING[c]).alias(c)
        for c in _GW_STATS_COLUMNS.values()
    ]
    return df.select(columns)


def load_match_stats_csv(path: str | Path) -> pl.DataFrame:
    """Load playermatchstats.csv (full squads incl. 0-minute rows)."""
    return (
        pl.read_csv(path, schema_overrides={
            "minutes_played": pl.Float64, "goals": pl.Float64,
            "assists": pl.Float64, "xg": pl.Float64, "xa": pl.Float64,
        })
        .with_columns(pl.col("player_id").cast(pl.Int64))
        .select("player_id", "match_id", "minutes_played", "goals", "assists", "xg", "xa")
    )


def load_matches_csv(path: str | Path) -> pl.DataFrame:
    """Load matches.csv with team refs (teams.code, stored as floats) cast to ints.

    A missing `tournament` column (legacy layout) defaults to 'prem' — legacy
    per-GW folders are Premier-League-only.
    """
    df = pl.read_csv(path, schema_overrides={
        "gameweek": pl.Float64, "home_team": pl.Float64, "away_team": pl.Float64,
        "home_score": pl.Float64, "away_score": pl.Float64,
    })
    if "tournament" not in df.columns:
        df = df.with_columns(pl.lit("prem").alias("tournament"))
    return (
        df.with_columns(
            pl.col("gameweek").cast(pl.Int64).alias("gw"),
            pl.col("home_team").cast(pl.Int64),
            pl.col("away_team").cast(pl.Int64),
            pl.col("home_score").cast(pl.Int64),
            pl.col("away_score").cast(pl.Int64),
            pl.col("finished").cast(pl.Boolean),
            pl.col("home_team_elo").cast(pl.Float64),
            pl.col("away_team_elo").cast(pl.Float64),
        )
        .select(
            "match_id", "gw", "kickoff_time", "home_team", "away_team",
            "home_score", "away_score", "home_team_elo", "away_team_elo",
            "tournament", "finished")
    )


def load_team_history_csv(path: str | Path) -> pl.DataFrame:
    """Load team_history.csv (player_id, gw, team_code) — player's club per GW.

    Needed because players can change clubs mid-season (27 did in 2025-26);
    opponent/venue features must use the club the player actually played for
    that gameweek, not their final one.
    """
    return pl.read_csv(path).with_columns(
        pl.col("player_id").cast(pl.Int64),
        pl.col("gw").cast(pl.Int64),
        pl.col("team_code").cast(pl.Int64),
    )
