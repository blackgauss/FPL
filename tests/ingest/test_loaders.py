"""Black-box tests: loaders turn CSV paths into the canonical typed contract."""

from pathlib import Path

import polars as pl
import pytest

from tests.fixtures.synthetic import write_fixtures


@pytest.fixture(scope="module")
def fx(tmp_path_factory) -> dict[str, Path]:
    return write_fixtures(tmp_path_factory.mktemp("loaders"))


def test_players_normalizes_position_and_codes(fx):
    from fpl.data.loaders import load_players_csv

    df = load_players_csv(fx["players"])
    assert df.schema == {
        "player_code": pl.Int64, "player_id": pl.Int64, "first_name": pl.String,
        "second_name": pl.String, "web_name": pl.String, "team_code": pl.Int64,
        "position": pl.String,
    }
    positions = set(df.get_column("position"))
    assert positions <= {"GKP", "DEF", "MID", "FWD"}
    haaland = df.filter(pl.col("web_name") == "Haaland")
    assert haaland.get_column("position").item() == "FWD"
    assert haaland.get_column("player_code").item() == 223094


def test_teams_keyed_on_code_with_nullable_strength(fx):
    from fpl.data.loaders import load_teams_csv

    df = load_teams_csv(fx["teams"])
    assert df.schema["code"] == pl.Int64
    arsenal = df.filter(pl.col("name") == "Arsenal")
    assert arsenal.get_column("strength").item() is None
    assert arsenal.get_column("elo").item() is None
    assert df.filter(pl.col("name") == "Man City").get_column("elo").item() == 2064.0


def test_gw_stats_casts_numerics_and_renames_id(fx):
    from fpl.data.loaders import load_gw_stats_csv

    df = load_gw_stats_csv(fx["gw_stats"])
    assert "player_id" in df.columns and "id" not in df.columns
    assert df.schema["total_points"] == pl.Int64
    assert df.schema["now_cost"] == pl.Float64
    row = df.filter((pl.col("player_id") == 430) & (pl.col("gw") == 1))
    assert row.get_column("total_points").item() == 13
    assert row.get_column("now_cost").item() == 15.0


def test_match_stats_keeps_zero_minute_rows(fx):
    from fpl.data.loaders import load_match_stats_csv

    df = load_match_stats_csv(fx["match_stats"])
    assert df.schema["player_id"] == pl.Int64
    assert df.schema["minutes_played"] == pl.Float64
    bench = df.filter((pl.col("player_id") == 239) & (pl.col("match_id") == "m2"))
    assert bench.get_column("minutes_played").item() == 0.0


def test_matches_casts_float_team_codes_and_gw(fx):
    from fpl.data.loaders import load_matches_csv

    df = load_matches_csv(fx["matches"])
    assert df.schema["home_team"] == pl.Int64
    assert df.schema["away_team"] == pl.Int64
    assert df.schema["gw"] == pl.Int64
    assert df.schema["finished"] == pl.Boolean
    m1 = df.filter(pl.col("match_id") == "m1")
    assert m1.get_column("home_team").item() == 43
    assert m1.get_column("away_team").item() == 3
    # unplayed fixture: null scores preserved, finished=False
    m3 = df.filter(pl.col("match_id") == "m3")
    assert m3.get_column("home_score").item() is None
    assert m3.get_column("finished").item() is False
    assert m3.get_column("tournament").item() == "europa-league"
