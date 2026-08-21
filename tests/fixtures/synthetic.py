"""Shared synthetic fixtures: tiny CSVs mimicking FPL-Core layout and quirks."""

from __future__ import annotations

from pathlib import Path

PLAYERS_CSV = """player_code,player_id,first_name,second_name,web_name,team_code,position
223094,430,Erling,Haaland,Haaland,43,Forward
118748,239,Kevin,De Bruyne,De Bruyne,43,Midfielder
111111,100,Duplicate,Wilson,Wilson,3,Defender
111112,101,Other,Anderson,Wilson,3,Goalkeeper
"""

TEAMS_CSV = """code,id,name,short_name,strength,strength_overall_home,strength_overall_away,strength_attack_home,strength_attack_away,strength_defence_home,strength_defence_away,pulse_id,elo,fotmob_name
43,1,Man City,MCI,5,5,5,5,5,5,5,10,2064,Man City
3,2,Arsenal,ARS,,,0,0,0,0,2,,,
"""

GW_STATS_CSV = """id,gw,web_name,second_name,status,total_points,minutes,goals_scored,assists,bonus,bps,saves,starts,now_cost,form,ep_next,ep_this,selected_by_percent
430,1,Haaland,Haaland,a,13,72,2,0,2,35,0,1,15.0,5.0,5.5,5.5,73.0
430,2,Haaland,Haaland,a,2,90,0,0,0,10,0,1,14.1,3.5,8.0,7.0,70.0
239,1,De Bruyne,De Bruyne,a,9,90,1,1,3,30,0,1,11.0,4.0,4.5,4.5,50.0
"""

MATCH_STATS_CSV = """player_id,match_id,minutes_played,goals,assists,xg,xa
430,m1,90,2,0,1.8,0.2
430,m2,90,0,0,0.4,0.1
430,m4,90,1,0,0.9,0.0
239,m1,90,1,1,0.5,0.9
239,m2,0,0,0,0.0,0.0
100,m1,90,0,0,0.0,0.0
"""

MATCHES_CSV = """gameweek,kickoff_time,home_team,home_team_elo,home_score,away_score,away_team,away_team_elo,finished,match_id,match_url,tournament
1.0,2025-08-16T16:30:00+00:00,43.0,2050.0,2,1,3.0,2000.0,True,m1,url1,prem
2.0,2025-08-23T16:30:00+00:00,3.0,2000.0,0,3,43.0,2050.0,True,m2,url2,prem
3.0,2025-08-30T16:30:00+00:00,43.0,2050.0,,,14.0,1900.0,False,m3,url3,europa-league
"""


def write_fixtures(base: Path) -> dict[str, Path]:
    """Write the synthetic CSVs into `base`, returning path per table name."""
    files = {
        "players": base / "players.csv",
        "teams": base / "teams.csv",
        "gw_stats": base / "player_gameweek_stats.csv",
        "match_stats": base / "playermatchstats.csv",
        "matches": base / "matches.csv",
    }
    contents = {
        "players": PLAYERS_CSV,
        "teams": TEAMS_CSV,
        "gw_stats": GW_STATS_CSV,
        "match_stats": MATCH_STATS_CSV,
        "matches": MATCHES_CSV,
    }
    base.mkdir(parents=True, exist_ok=True)
    for name, path in files.items():
        path.write_text(contents[name], encoding="utf-8")
    return files


def build_season_tree(base: Path, season: str = "2025-2026") -> Path:
    """Write a full mini FPL-Core season tree (By Gameweek layout) into `base`.

    Includes the real-data traps:
    - tournament mixing (europa-league match in a GW folder alongside prem)
    - a postponed match (m1) duplicated across GW1 and GW2 folders
    - a 0-minute squad row (player 239 in m2)
    """
    season_dir = base / season
    gw_dir = season_dir / "By Gameweek"
    season_dir.mkdir(parents=True, exist_ok=True)
    (season_dir / "players.csv").write_text(PLAYERS_CSV, encoding="utf-8")
    (season_dir / "teams.csv").write_text(TEAMS_CSV, encoding="utf-8")
    # player 100 transfers from Arsenal (3) to Man City (43) after GW1;
    # player 101 (Anderson) also on 3 and does not transfer.
    (season_dir / "team_history.csv").write_text(
        "player_id,gw,team_code\n"
        "430,1,43\n430,2,43\n"
        "239,1,43\n239,2,43\n"
        "100,1,3\n100,2,43\n"
        "101,1,3\n101,2,3\n",
        encoding="utf-8",
    )

    # matches in both folders; m1 repeats in GW2 (postponed -> dedupe on match_id)
    gw1_matches = (
        "gameweek,kickoff_time,home_team,home_team_elo,home_score,away_score,away_team,"
        "away_team_elo,finished,match_id,match_url,tournament\n"
        "1.0,2025-08-16T16:30:00+00:00,43.0,2050.0,2,1,3.0,2000.0,True,m1,url1,prem\n"
        "3.0,2025-08-30T16:30:00+00:00,43.0,2050.0,,,14.0,1900.0,False,m3,url3,europa-league\n"
        "1.0,2025-08-14T19:00:00+00:00,43.0,2050.0,3,0,3.0,2000.0,True,m4,url4,europa-league\n"
    )
    gw2_matches = (
        "gameweek,kickoff_time,home_team,home_team_elo,home_score,away_score,away_team,"
        "away_team_elo,finished,match_id,match_url,tournament\n"
        "1.0,2025-08-16T16:30:00+00:00,43.0,2050.0,2,1,3.0,2000.0,True,m1,url1,prem\n"
        "2.0,2025-08-23T16:30:00+00:00,3.0,2000.0,0,3,43.0,2050.0,True,m2,url2,prem\n"
    )

    # per-GW player stats (discrete): gw1 rows for both studs, gw2 row for Haaland
    gw1_stats = (
        "id,gw,web_name,second_name,status,total_points,minutes,goals_scored,assists,"
        "bonus,bps,saves,starts,now_cost,form,ep_next,ep_this,selected_by_percent\n"
        "430,1,Haaland,Haaland,a,13,72,2,0,2,35,0,1,15.0,5.0,5.5,5.5,73.0\n"
        "239,1,De Bruyne,De Bruyne,a,9,90,1,1,3,30,0,1,11.0,4.0,4.5,4.5,50.0\n"
    )
    gw2_stats = (
        "id,gw,web_name,second_name,status,total_points,minutes,goals_scored,assists,"
        "bonus,bps,saves,starts,now_cost,form,ep_next,ep_this,selected_by_percent\n"
        "430,2,Haaland,Haaland,a,2,90,0,0,0,10,0,1,14.1,3.5,8.0,7.0,70.0\n"
    )

    # match-level stats: Haaland in m1/m2/m4, KDB in m1 (played) + m2 (0 min bench),
    # Wilson played m1. m1 rows duplicate across GW folders -> dedupe on player+match.
    # GW1 only holds m1 + m4 (europa); m2 is a GW2 match.
    gw1_mstat = (
        "player_id,match_id,minutes_played,goals,assists,xg,xa\n"
        "430,m1,90,2,0,1.8,0.2\n"
        "430,m4,90,1,0,0.9,0.0\n"
        "239,m1,90,1,1,0.5,0.9\n"
        "100,m1,90,0,0,0.0,0.0\n"
    )
    gw2_mstat = (
        "player_id,match_id,minutes_played,goals,assists,xg,xa\n"
        "430,m1,90,2,0,1.8,0.2\n"
        "430,m2,90,0,0,0.4,0.1\n"
        "239,m1,90,1,1,0.5,0.9\n"
        "239,m2,0,0,0,0.0,0.0\n"
    )

    for folder, matches, stats, mstat in [
        (gw_dir / "GW1", gw1_matches, gw1_stats, gw1_mstat),
        (gw_dir / "GW2", gw2_matches, gw2_stats, gw2_mstat),
    ]:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "matches.csv").write_text(matches, encoding="utf-8")
        (folder / "player_gameweek_stats.csv").write_text(stats, encoding="utf-8")
        (folder / "playermatchstats.csv").write_text(mstat, encoding="utf-8")

    return season_dir


LEGACY_PLAYERSTATS_CSV = """id,gw,status,total_points,event_points,bonus,bps,now_cost,form,ep_next,ep_this,selected_by_percent
430,1,a,13,13,2,35,15.0,5.0,5.5,5.5,73.0
430,2,a,15,2,0,10,14.1,3.5,8.0,7.0,70.0
239,1,a,9,9,3,30,11.0,4.0,4.5,4.5,50.0
"""

LEGACY_MATCHES_CSV = """gameweek,kickoff_time,home_team,home_team_elo,home_score,away_score,away_team,away_team_elo,finished,match_id,match_url
1.0,2025-08-16T16:30:00+00:00,43.0,2050.0,2,1,3.0,2000.0,True,m1,url1
2.0,2025-08-23T16:30:00+00:00,3.0,2000.0,0,3,43.0,2050.0,True,m2,url2
"""

LEGACY_MSTAT_CSV = """player_id,match_id,minutes_played,goals,assists,xg,xa
430,m1,90,2,0,1.8,0.2
430,m2,90,0,0,0.4,0.1
239,m1,90,1,1,0.5,0.9
239,m2,0,0,0,0.0,0.0
"""


def build_legacy_season_tree(base: Path, season: str = "2024-2025") -> Path:
    """Write a mini legacy-layout season (per-table dirs, matches/GWn/) into `base`."""
    season_dir = base / season
    (season_dir / "players").mkdir(parents=True, exist_ok=True)
    (season_dir / "teams").mkdir(parents=True, exist_ok=True)
    (season_dir / "playerstats").mkdir(parents=True, exist_ok=True)
    (season_dir / "players" / "players.csv").write_text(PLAYERS_CSV, encoding="utf-8")
    (season_dir / "teams" / "teams.csv").write_text(TEAMS_CSV, encoding="utf-8")
    (season_dir / "playerstats" / "playerstats.csv").write_text(
        LEGACY_PLAYERSTATS_CSV, encoding="utf-8"
    )
    for gw in (1, 2):
        (season_dir / "matches" / f"GW{gw}").mkdir(parents=True, exist_ok=True)
        (season_dir / "playermatchstats" / f"GW{gw}").mkdir(parents=True, exist_ok=True)
    (season_dir / "matches" / "GW1" / "matches.csv").write_text(
        LEGACY_MATCHES_CSV, encoding="utf-8"
    )
    (season_dir / "matches" / "GW2" / "matches.csv").write_text(
        LEGACY_MATCHES_CSV, encoding="utf-8"
    )
    for gw in (1, 2):
        (season_dir / "playermatchstats" / f"GW{gw}" / "playermatchstats.csv").write_text(
            LEGACY_MSTAT_CSV, encoding="utf-8"
        )
    return season_dir
