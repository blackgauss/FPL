"""Black-box tests: build_features over a 4-GW synthetic season.

Exercises the feature-job contract: opponent/venue resolution via team_history
(including a mid-season transfer), rolling prev-GW points, and the next-GW
target. An independent hand-computed expectation is asserted for every row.
"""

from pathlib import Path

import polars as pl
import pytest

from fpl.data.contract import load_season
from fpl.data.features import build_features

PLAYERS = (
    "player_code,player_id,first_name,second_name,web_name,team_code,position\n"
    "223094,430,Erling,Haaland,Haaland,43,Forward\n"
    "118748,239,Kevin,De Bruyne,De Bruyne,43,Midfielder\n"
)
TEAMS = (
    "code,id,name,short_name,strength,strength_overall_home,strength_overall_away,"
    "strength_attack_home,strength_attack_away,strength_defence_home,"
    "strength_defence_away,pulse_id,elo,fotmob_name\n"
    "43,1,Man City,MCI,5,5,5,5,5,5,5,10,2064,Man City\n"
    "3,2,Arsenal,ARS,4,4,4,4,4,4,4,2,1991,Arsenal\n"
)
# Haaland (430) plays every GW for MCI (43). De Bruyne (239) transfers:
# MCI for GW1-2, then ARS for GW3-4 (fake, but exercises team_history).
TEAM_HISTORY = (
    "player_id,gw,team_code\n"
    "430,1,43\n430,2,43\n430,3,43\n430,4,43\n"
    "239,1,43\n239,2,43\n239,3,3\n239,4,3\n"
)
# FPL points per GW. Haaland: 13, 2, 4, 1. KDB: 9, 5, 3, 2.
# Full loader contract columns (minutes + friends are all 90 / constant).
GW_STATS = (
    "id,gw,web_name,second_name,status,total_points,minutes,goals_scored,assists,bonus,"
    "bps,saves,starts,now_cost,form,ep_next,ep_this,selected_by_percent\n"
    "430,1,Haaland,Haaland,a,13,90,2,0,2,35,0,1,15.0,5.0,5.5,5.5,73.0\n"
    "430,2,Haaland,Haaland,a,2,90,0,0,0,10,0,1,14.1,3.5,8.0,7.0,70.0\n"
    "430,3,Haaland,Haaland,a,4,90,1,0,0,22,0,1,14.1,3.5,8.0,7.0,70.0\n"
    "430,4,Haaland,Haaland,a,1,90,0,0,0,8,0,1,14.1,3.5,8.0,7.0,70.0\n"
    "239,1,De Bruyne,De Bruyne,a,9,90,1,1,3,30,0,1,11.0,4.0,4.5,4.5,50.0\n"
    "239,2,De Bruyne,De Bruyne,a,5,90,0,1,1,20,0,1,11.0,4.0,4.5,4.5,50.0\n"
    "239,3,De Bruyne,De Bruyne,a,3,90,0,0,0,12,0,1,11.0,4.0,4.5,4.5,50.0\n"
    "239,4,De Bruyne,De Bruyne,a,2,90,0,0,0,10,0,1,11.0,4.0,4.5,4.5,50.0\n"
)
# MCI (43) and ARS (3) alternate home/away. elo stays constant per team for
# simplicity: home_team_elo for 43=2064, for 3=1991 in every match.
MATCHES_TPL = (
    "gameweek,kickoff_time,home_team,home_team_elo,home_score,away_score,away_team,"
    "away_team_elo,finished,match_id,match_url,tournament\n"
)


def _tree(base: Path) -> Path:
    season_dir = base / "2025-2026"
    gw_dir = season_dir / "By Gameweek"
    gw_dir.mkdir(parents=True)
    (season_dir / "players.csv").write_text(PLAYERS, encoding="utf-8")
    (season_dir / "teams.csv").write_text(TEAMS, encoding="utf-8")
    (season_dir / "team_history.csv").write_text(TEAM_HISTORY, encoding="utf-8")
    for gw in range(1, 5):
        folder = gw_dir / f"GW{gw}"
        folder.mkdir(parents=True)
        # MCI (43) home on odd GWs, away on even
        if gw % 2 == 1:
            matches = (
                MATCHES_TPL
                + f"{gw}.0,2025-08-1{gw}T16:30:00+00:00,43.0,2064.0,2,1,3.0,1991.0,True,"
                f"m{gw},u{gw},prem\n"
            )
        else:
            matches = (
                MATCHES_TPL
                + f"{gw}.0,2025-08-2{gw}T16:30:00+00:00,3.0,1991.0,0,3,43.0,2064.0,True,"
                f"m{gw},u{gw},prem\n"
            )
        (folder / "matches.csv").write_text(matches, encoding="utf-8")
        # per-GW slice: header + rows whose gw column == this GW
        lines = [GW_STATS.splitlines()[0]] + [
            ln for ln in GW_STATS.splitlines()[1:]
            if ln.split(",")[1] == str(gw)
        ]
        (folder / "player_gameweek_stats.csv").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
    return season_dir


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    root = _tree(tmp_path_factory.mktemp("features"))
    season = load_season(root.parent, "2025-2026")
    return build_features(season.gw_stats, season.team_history, season.matches,
                          season.players)


def test_columns_are_the_contract(data):
    assert data.columns == [
        "player_id", "player_code", "gw", "team_code", "opponent_team_code",
        "was_home", "home_elo", "opponent_elo", "prev_points", "pts_avg_3",
        "pts_avg_5", "total_points", "next_points",
    ]


def test_team_code_never_null():
    # legacy seasons lack team_history; team_code must backfill from
    # players.team_code, otherwise full-feature models hit NaN (Ridge, etc.)
    root = _tree(Path(__import__("tempfile").mkdtemp()))
    season = load_season(root.parent, "2025-2026")
    empty_history = pl.DataFrame(
        {"player_id": pl.Series([], dtype=pl.Int64),
         "gw": pl.Series([], dtype=pl.Int64),
         "team_code": pl.Series([], dtype=pl.Int64)}
    )
    feats = build_features(season.gw_stats, empty_history, season.matches, season.players)
    assert feats.get_column("team_code").null_count() == 0
    assert set(feats.filter(pl.col("player_id") == 100).get_column("team_code")) <= {3, 43}


def test_haaland_gw2_opponent_and_venue(data):
    row = data.filter((pl.col("player_id") == 430) & (pl.col("gw") == 2))
    assert row.get_column("team_code").item() == 43
    assert row.get_column("opponent_team_code").item() == 3
    assert row.get_column("was_home").item() is False
    assert row.get_column("home_elo").item() == 2064
    assert row.get_column("opponent_elo").item() == 1991


def test_haaland_gw3_venue_toggles(data):
    row = data.filter((pl.col("player_id") == 430) & (pl.col("gw") == 3))
    assert row.get_column("was_home").item() is True
    assert row.get_column("opponent_team_code").item() == 3


def test_rolling_points_hand_computed(data):
    # Haaland: gw1=13, gw2=2, gw3=4
    row = data.filter((pl.col("player_id") == 430) & (pl.col("gw") == 3))
    assert row.get_column("prev_points").item() == 2       # gw2
    assert row.get_column("pts_avg_3").item() == pytest.approx((13 + 2) / 2)
    assert row.get_column("pts_avg_5").item() == pytest.approx((13 + 2) / 2)


def test_target_is_next_gw_points(data):
    row = data.filter((pl.col("player_id") == 430) & (pl.col("gw") == 3))
    assert row.get_column("next_points").item() == 1       # gw4


def test_transfer_reflected_in_opponent(data):
    # KDB plays MCI in GW1-2 (opponent ARS) then transfers to ARS for GW3
    # (opponent MCI). Team code must come from team_history per GW.
    gw2 = data.filter((pl.col("player_id") == 239) & (pl.col("gw") == 2))
    assert gw2.get_column("team_code").item() == 43
    assert gw2.get_column("opponent_team_code").item() == 3
    gw3 = data.filter((pl.col("player_id") == 239) & (pl.col("gw") == 3))
    assert gw3.get_column("team_code").item() == 3
    assert gw3.get_column("opponent_team_code").item() == 43


def test_first_gw_and_last_gw_dropped(data):
    # GW1 has no prev context, GW4 has no target -> both excluded
    gws = set(data.get_column("gw"))
    assert gws == {2, 3}