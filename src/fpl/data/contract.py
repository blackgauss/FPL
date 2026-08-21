"""Season catalog: unify per-GW FPL-Core files into single frames per season.

`load_season(root, season)` walks the season folder and produces a `SeasonData`
with unified frames, each carrying `season` (+ `gw` where the raw file lacks it)
and `tournament` on match-level frames. Deduplication guards against postponed
matches reappearing in later GW folders.

Two source layouts are supported (auto-detected):
    modern (>=2025-26):  By Gameweek/GW{n}/{table}.csv, master {table}.csv
    legacy (2024-25):    {table}/{table}.csv, matches/GW{n}/matches.csv,
                         playermatchstats/GW{n}/playermatchstats.csv
Both produce the identical shared schema (legacy missing columns are typed-null).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from fpl.data.loaders import (
    load_gw_stats_csv,
    load_legacy_gw_stats_csv,
    load_match_stats_csv,
    load_matches_csv,
    load_players_csv,
    load_teams_csv,
)

_GW_DIR = re.compile(r"^GW(\d+)$")


@dataclass(frozen=True)
class SeasonData:
    """Canonical per-season frames (see loaders module docstring for schemas)."""

    season: str
    players: pl.DataFrame
    teams: pl.DataFrame
    gw_stats: pl.DataFrame
    match_stats: pl.DataFrame
    matches: pl.DataFrame


def _gw_folders(gw_root: Path) -> list[tuple[int, Path]]:
    """List `GW{n}` subfolders directly under `gw_root`, sorted by GW number."""
    if not gw_root.is_dir():
        return []
    found = [
        (int(m.group(1)), entry)
        for entry in sorted(gw_root.iterdir())
        if (m := _GW_DIR.match(entry.name))
    ]
    return sorted(found)


def _read_optional(path: Path, loader) -> pl.DataFrame | None:
    if path.is_file() and path.stat().st_size > 0:
        return loader(path)
    return None


def _concat(frames: list[pl.DataFrame], subset: list[str]) -> pl.DataFrame:
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="vertical").unique(subset=subset, keep="first")


def detect_layout(season_dir: str | Path) -> str:
    """Return 'modern' (By Gameweek) or 'legacy' (per-table dirs) layout."""
    d = Path(season_dir)
    if (d / "By Gameweek").is_dir():
        return "modern"
    if (d / "players").is_dir() and (d / "matches").is_dir():
        return "legacy"
    raise ValueError(f"unknown FPL-Core layout for {season_dir}")


def _build(season: str, players: pl.DataFrame, teams: pl.DataFrame,
           gw_frames: list, match_stat_frames: list, match_frames: list) -> SeasonData:
    return SeasonData(
        season=season,
        players=players.with_columns(pl.lit(season).alias("season")),
        teams=teams.with_columns(pl.lit(season).alias("season")),
        gw_stats=_concat(gw_frames, ["season", "player_id", "gw"]),
        match_stats=_concat(match_stat_frames, ["season", "player_id", "match_id"]),
        matches=_concat(match_frames, ["season", "match_id"]),
    )


def _load_season_modern(season_dir: Path, season: str) -> SeasonData:
    players = load_players_csv(season_dir / "players.csv")
    teams = load_teams_csv(season_dir / "teams.csv")
    gw_frames: list[pl.DataFrame] = []
    match_stat_frames: list[pl.DataFrame] = []
    match_frames: list[pl.DataFrame] = []
    for gw, folder in _gw_folders(season_dir / "By Gameweek"):
        gw_stats = _read_optional(
            folder / "player_gameweek_stats.csv", load_gw_stats_csv)
        if gw_stats is not None:
            gw_frames.append(gw_stats.with_columns(pl.lit(season).alias("season")))

        match_stats = _read_optional(
            folder / "playermatchstats.csv", load_match_stats_csv)
        if match_stats is not None:
            match_stat_frames.append(
                match_stats.with_columns(
                    pl.lit(gw, dtype=pl.Int64).alias("gw"),
                    pl.lit(season).alias("season"),
                )
            )

        matches = _read_optional(folder / "matches.csv", load_matches_csv)
        if matches is not None:
            match_frames.append(matches.with_columns(pl.lit(season).alias("season")))

    return _build(season, players, teams, gw_frames, match_stat_frames, match_frames)


def _load_season_legacy(season_dir: Path, season: str) -> SeasonData:
    players = load_players_csv(season_dir / "players" / "players.csv")
    teams = load_teams_csv(season_dir / "teams" / "teams.csv")
    gw_frames: list[pl.DataFrame] = []
    match_stat_frames: list[pl.DataFrame] = []
    match_frames: list[pl.DataFrame] = []
    for _, folder in _gw_folders(season_dir / "matches"):
        matches = _read_optional(folder / "matches.csv", load_matches_csv)
        if matches is not None:
            match_frames.append(matches.with_columns(pl.lit(season).alias("season")))

    for gw, folder in _gw_folders(season_dir / "playermatchstats"):
        match_stats = _read_optional(folder / "playermatchstats.csv", load_match_stats_csv)
        if match_stats is not None:
            match_stat_frames.append(
                match_stats.with_columns(
                    pl.lit(gw, dtype=pl.Int64).alias("gw"),
                    pl.lit(season).alias("season"),
                )
            )

    gw_stats = _read_optional(
        season_dir / "playerstats" / "playerstats.csv", load_legacy_gw_stats_csv)
    if gw_stats is not None:
        gw_frames.append(gw_stats.with_columns(pl.lit(season).alias("season")))

    return _build(season, players, teams, gw_frames, match_stat_frames, match_frames)


def load_season(root: str | Path, season: str) -> SeasonData:
    """Build the SeasonData catalog for one season from injected paths."""
    season_dir = Path(root) / season
    if detect_layout(season_dir) == "modern":
        return _load_season_modern(season_dir, season)
    return _load_season_legacy(season_dir, season)
