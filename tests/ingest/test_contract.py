"""Black-box tests: load_season unifies a synthetic season tree correctly."""

from pathlib import Path

import polars as pl
import pytest

from fpl.data.contract import SeasonData, load_season
from tests.fixtures.synthetic import (
    GW_STATS_CSV,
    MATCH_STATS_CSV,
    MATCHES_CSV,
    PLAYERS_CSV,
    TEAMS_CSV,
)


@pytest.fixture()
def season_tree(tmp_path: Path) -> Path:
    """Build a mini FPL-Core season: two GW folders, one postponed match
    reappearing in GW2, a non-prem match mixed into GW1."""
    season_dir = tmp_path / "2025-2026"
    season_dir.mkdir(parents=True)
    (season_dir / "players.csv").write_text(PLAYERS_CSV, encoding="utf-8")
    (season_dir / "teams.csv").write_text(TEAMS_CSV, encoding="utf-8")

    # GW1: normal prem matches + one europa-league match mixed in
    gw1_matches = MATCHES_CSV + (
        "1.0,2025-08-14T19:00:00+00:00,43.0,2050.0,3,0,14.0,1900.0,True,m4,url4,europa-league\n"
    )
    # GW2: repeats m1 (postponed duplicate) plus its own match
    gw2_matches = (
        "gameweek,kickoff_time,home_team,home_team_elo,home_score,away_score,away_team,"
        "away_team_elo,finished,match_id,match_url,tournament\n"
        "1.0,2025-08-16T16:30:00+00:00,43.0,2050.0,2,1,3.0,2000.0,True,m1,url1,prem\n"
        "2.0,2025-08-23T16:30:00+00:00,3.0,2000.0,0,3,43.0,2050.0,True,m2,url2,prem\n"
    )

    for gw, matches in [(1, gw1_matches), (2, gw2_matches)]:
        folder = season_dir / "By Gameweek" / f"GW{gw}"
        folder.mkdir(parents=True)
        (folder / "matches.csv").write_text(matches, encoding="utf-8")
        # GW2 has its own squad list incl. a new match m5 for player 430;
        # rows for m1/m2 are duplicates of GW1's (postponed-match case)
        stats = MATCH_STATS_CSV if gw == 1 else MATCH_STATS_CSV + "430,m5,90,1,0,0.9,0.0\n"
        (folder / "playermatchstats.csv").write_text(stats, encoding="utf-8")
        (folder / "player_gameweek_stats.csv").write_text(GW_STATS_CSV, encoding="utf-8")

    return tmp_path


def test_load_season_returns_catalog(season_tree):
    data = load_season(season_tree, "2025-2026")
    assert isinstance(data, SeasonData)
    assert data.season == "2025-2026"


def test_matches_dedupe_and_tournament_preserved(season_tree):
    data = load_season(season_tree, "2025-2026")
    matches = data.matches
    # GW1: m1, m2, m3, m4; GW2: m1 (dup) + m2 (already there) -> 4 unique
    assert matches.height == 4
    assert sorted(matches.get_column("match_id")) == ["m1", "m2", "m3", "m4"]
    assert "europa-league" in set(matches.get_column("tournament"))
    assert set(matches.get_column("season")) == {"2025-2026"}


def test_gw_stats_unified_across_folders(season_tree):
    data = load_season(season_tree, "2025-2026")
    gw = data.gw_stats
    # GW_STATS_CSV has 3 rows (2 players), duplicated in 2 folders -> deduped
    assert gw.height == 3
    haaland = gw.filter(pl.col("web_name") == "Haaland")
    assert sorted(haaland.get_column("gw").to_list()) == [1, 2]
    assert set(gw.get_column("season")) == {"2025-2026"}


def test_match_stats_get_folder_gw(season_tree):
    data = load_season(season_tree, "2025-2026")
    ms = data.match_stats
    # GW1: 6 rows; GW2 adds m5 (new) — m1/m2 repeats dedupe to GW1's copies
    assert ms.height == 7
    assert set(ms.get_column("gw")) == {1, 2}
    assert set(ms.get_column("season")) == {"2025-2026"}
    # player 430: m1, m2, m4 (from GW1) + m5 (from GW2)
    assert ms.filter(pl.col("player_id") == 430).height == 4


def test_players_and_teams_loaded(season_tree):
    data = load_season(season_tree, "2025-2026")
    assert data.players.height == 4
    assert data.teams.height == 2
    assert set(data.players.get_column("season")) == {"2025-2026"}
