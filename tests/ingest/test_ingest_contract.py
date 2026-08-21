"""Black-box contract suite for the parquet dataset.

These tests validate the *output dataset* produced by the public ingest entrypoint
(`fpl.data.ingest.run`). They never depend on how ingest is implemented — any
implementation (polars, duckdb, a Rust port) that produces a conforming parquet
dataset for the synthetic season tree must pass this suite.

Invariants covered: file set, exact schemas, referential integrity, uniqueness,
spot values (pinning the load-time casts + discrete-per-GW semantics), data quality,
idempotent determinism, and the EDA with/without pattern over the produced parquet.
"""


import polars as pl
import pytest

from fpl.data.contract import detect_layout
from fpl.data.ingest import TABLES, run
from tests.fixtures.synthetic import build_legacy_season_tree, build_season_tree

SEASON = "2025-2026"
EXPECTED_COLUMNS = {
    "players": ["player_code", "player_id", "first_name", "second_name", "web_name",
                "team_code", "position", "season"],
    "teams": ["code", "id", "name", "short_name", "strength", "elo", "season"],
    "gw_stats": ["player_id", "gw", "web_name", "second_name", "status",
                 "total_points", "minutes", "goals_scored", "assists", "bonus", "bps",
                 "saves", "starts", "now_cost", "form", "ep_next", "ep_this",
                 "selected_by_percent", "season"],
    "match_stats": ["player_id", "match_id", "minutes_played", "goals", "assists", "xg",
                    "xa", "gw", "season"],
    "matches": ["match_id", "gw", "kickoff_time", "home_team", "away_team", "home_score",
                "away_score", "home_team_elo", "away_team_elo", "tournament", "finished",
                "season"],
    "team_history": ["player_id", "gw", "team_code", "season"],
}


@pytest.fixture()
def dataset(tmp_path_factory) -> dict[str, pl.DataFrame]:
    root = build_season_tree(tmp_path_factory.mktemp("tree"))
    out = tmp_path_factory.mktemp("processed")
    run(root.parent, SEASON, out)
    return {
        table: pl.read_parquet(out / f"{table}_{SEASON}.parquet") for table in TABLES
    }


@pytest.fixture()
def frames(dataset) -> dict[str, pl.DataFrame]:
    return dataset


class TestFileSet:
    def test_exactly_five_tables(self, dataset):
        assert set(dataset) == set(TABLES)


class TestSchema:
    def test_columns_exact(self, frames):
        for table, cols in EXPECTED_COLUMNS.items():
            assert frames[table].columns == cols, f"{table} columns mismatch"

    def test_players_dtypes(self, frames):
        schema = frames["players"].schema
        assert schema["player_code"] == pl.Int64
        assert schema["player_id"] == pl.Int64
        assert schema["team_code"] == pl.Int64
        assert schema["position"] == pl.String

    def test_gw_stats_dtypes(self, frames):
        schema = frames["gw_stats"].schema
        assert schema["total_points"] == pl.Int64
        assert schema["minutes"] == pl.Int64
        assert schema["now_cost"] == pl.Float64  # decimal millions, not tenths
        assert schema["ep_next"] == pl.Float64

    def test_match_stats_dtypes(self, frames):
        schema = frames["match_stats"].schema
        assert schema["minutes_played"] == pl.Float64
        assert schema["xg"] == pl.Float64

    def test_matches_dtypes(self, frames):
        schema = frames["matches"].schema
        assert schema["home_team"] == pl.Int64  # teams.code, cast from float
        assert schema["away_team"] == pl.Int64
        assert schema["gw"] == pl.Int64
        assert schema["finished"] == pl.Boolean
        assert schema["home_score"] == pl.Int64
        assert schema["home_team_elo"] == pl.Float64
        assert schema["away_team_elo"] == pl.Float64

    def test_team_history_dtypes(self, frames):
        schema = frames["team_history"].schema
        assert schema["player_id"] == pl.Int64
        assert schema["gw"] == pl.Int64
        assert schema["team_code"] == pl.Int64


class TestReferentialIntegrity:
    def test_gw_stats_players(self, frames):
        players = set(frames["players"]["player_id"])
        assert set(frames["gw_stats"]["player_id"]) <= players

    def test_match_stats_players(self, frames):
        players = set(frames["players"]["player_id"])
        assert set(frames["match_stats"]["player_id"]) <= players

    def test_match_stats_match_ids(self, frames):
        matches = set(frames["matches"]["match_id"])
        assert set(frames["match_stats"]["match_id"]) <= matches

    def test_players_teams_codes(self, frames):
        teams = set(frames["teams"]["code"])
        assert set(frames["players"]["team_code"]) <= teams


class TestUniqueness:
    def test_players_codes_unique(self, frames):
        assert_frames_unique(frames["players"], ["player_code"])

    def test_teams_codes_unique(self, frames):
        assert_frames_unique(frames["teams"], ["code"])

    def test_gw_stats_unique(self, frames):
        assert_frames_unique(frames["gw_stats"], ["season", "player_id", "gw"])

    def test_match_stats_unique(self, frames):
        assert_frames_unique(frames["match_stats"], ["season", "player_id", "match_id"])

    def test_matches_unique(self, frames):
        assert_frames_unique(frames["matches"], ["season", "match_id"])

    def test_team_history_unique(self, frames):
        assert_frames_unique(frames["team_history"], ["season", "player_id", "gw"])


class TestSpotValues:
    def test_haaland_gw1_discrete_points(self, frames):
        row = frames["gw_stats"].filter(
            (pl.col("player_id") == 430) & (pl.col("gw") == 1)
        )
        assert row.get_column("total_points").item() == 13
        assert row.get_column("minutes").item() == 72
        assert row.get_column("now_cost").item() == 15.0

    def test_haaland_gw2_not_cumulative(self, frames):
        row = frames["gw_stats"].filter(
            (pl.col("player_id") == 430) & (pl.col("gw") == 2)
        )
        assert row.get_column("total_points").item() == 2  # not 15

    def test_position_normalized(self, frames):
        row = frames["players"].filter(pl.col("player_id") == 430)
        assert row.get_column("position").item() == "FWD"

    def test_team_codes_ints(self, frames):
        m1 = frames["matches"].filter(pl.col("match_id") == "m1")
        assert m1.get_column("home_team").item() == 43
        assert m1.get_column("away_team").item() == 3

    def test_postponed_match_deduped(self, frames):
        # m1 appears in both GW1 and GW2 folders -> exactly one row
        assert frames["matches"].filter(pl.col("match_id") == "m1").height == 1

    def test_tournament_mixing_preserved(self, frames):
        tournaments = set(frames["matches"].get_column("tournament"))
        assert tournaments == {"prem", "europa-league"}

    def test_team_history_captures_transfer(self, frames):
        # player 100 moved Arsenal (3) -> Man City (43) after GW1
        moved = frames["team_history"].filter(pl.col("player_id") == 100).sort("gw")
        assert moved.get_column("team_code").to_list() == [3, 43]


class TestDataQuality:
    def test_minutes_non_negative(self, frames):
        assert (frames["match_stats"]["minutes_played"] < 0).sum() == 0
        assert (frames["gw_stats"]["minutes"] < 0).sum() == 0

    def test_zero_minute_squad_rows_preserved(self, frames):
        bench = frames["match_stats"].filter(
            (pl.col("player_id") == 239) & (pl.col("match_id") == "m2")
        )
        assert bench.get_column("minutes_played").item() == 0.0

    def test_now_cost_positive(self, frames):
        assert (frames["gw_stats"]["now_cost"] <= 0).sum() == 0

    def test_no_empty_tables(self, frames):
        for table in TABLES:
            assert frames[table].height > 0


class TestDeterminism:
    def test_idempotent_rerun(self, tmp_path):
        root = build_season_tree(tmp_path / "tree")
        out = tmp_path / "proc"
        run(root.parent, SEASON, out)
        paths = {t: out / f"{t}_{SEASON}.parquet" for t in TABLES}
        first = {t: pl.read_parquet(p) for t, p in paths.items()}
        run(root.parent, SEASON, out)
        second = {t: pl.read_parquet(p) for t, p in paths.items()}
        for table in TABLES:
            a = first[table].sort(first[table].columns)
            b = second[table].sort(second[table].columns)
            assert a.equals(b), f"{table} differs between runs"


class TestEdaPattern:
    """The driving example rerun as plain polars over the parquet dataset."""

    def test_haaland_with_without_kdb(self, frames):
        match_stats = frames["match_stats"]
        gw_stats = frames["gw_stats"]
        matches = frames["matches"]

        # FPL-points stats only exist for Premier League matches: cup matches
        # (m4) have a folder-gw but no gw_stats row and must be excluded first.
        # This mirrors the documented EDA pattern (docs/plans/DataPipeline.md).
        prem_match_ids = set(matches.filter(pl.col("tournament") == "prem").get_column("match_id"))

        haaland_played = match_stats.filter(
            (pl.col("player_id") == 430)
            & (pl.col("minutes_played") > 0)
            & pl.col("match_id").is_in(prem_match_ids)
        )
        kdb_played_matches = set(
            match_stats.filter(
                (pl.col("player_id") == 239)
                & (pl.col("minutes_played") > 0)
                & pl.col("match_id").is_in(prem_match_ids)
            ).get_column("match_id")
        )

        haaland_gw_points = gw_stats.filter(pl.col("player_id") == 430).select(
            "player_id", "gw", "total_points"
        )
        joined = haaland_played.join(
            haaland_gw_points, on=["player_id", "gw"], how="inner"
        ).with_columns(
            pl.col("match_id").is_in(kdb_played_matches).alias("with_other")
        )

        def stat(mask) -> tuple:
            vals = joined.filter(mask).get_column("total_points")
            return vals.mean(), vals.len()

        with_mean, with_n = stat(pl.col("with_other"))
        without_mean, without_n = stat(~pl.col("with_other"))

        assert (with_mean, with_n) == (13.0, 1)
        assert (without_mean, without_n) == (2.0, 1)


def assert_frames_unique(df: pl.DataFrame, subset: list[str]) -> None:
    assert df.height == df.unique(subset=subset).height, f"dup on {subset}"


class TestLegacyLayout:
    """Legacy (2024-25) layout must produce the same shared contract schemas."""

    @pytest.fixture()
    def legacy(self, tmp_path_factory) -> dict[str, pl.DataFrame]:
        root = build_legacy_season_tree(tmp_path_factory.mktemp("legacy"))
        out = tmp_path_factory.mktemp("processed")
        run(root.parent, "2024-2025", out)
        return {
            table: pl.read_parquet(out / f"{table}_2024-2025.parquet")
            for table in TABLES
        }

    def test_layout_detected(self, tmp_path):
        modern = build_season_tree(tmp_path / "m")
        assert detect_layout(modern) == "modern"
        legacy = build_legacy_season_tree(tmp_path / "l")
        assert detect_layout(legacy) == "legacy"

    def test_same_schema_as_modern(self, legacy):
        for table, cols in EXPECTED_COLUMNS.items():
            assert legacy[table].columns == cols, f"{table} columns mismatch"

    def test_legacy_gw_stats_typed_nulls(self, legacy):
        # missing modern columns are typed-null (minutes, goals_scored, ...)
        missing = ["minutes", "goals_scored", "assists", "saves", "starts"]
        for col in missing:
            assert legacy["gw_stats"][col].null_count() == legacy["gw_stats"].height
            assert legacy["gw_stats"].schema[col] == pl.Int64

    def test_legacy_points_preserved(self, legacy):
        row = legacy["gw_stats"].filter(
            (pl.col("player_id") == 430) & (pl.col("gw") == 1)
        )
        assert row.get_column("total_points").item() == 13

    def test_legacy_points_are_discrete_not_cumulative(self, legacy):
        # legacy playerstats total_points is CUMULATIVE (row for 430 GW2 = 15);
        # the loader must emit discrete per-GW event_points (2), matching the
        # modern contract. Guard against the cumulative-total trap.
        row = legacy["gw_stats"].filter(
            (pl.col("player_id") == 430) & (pl.col("gw") == 2)
        )
        assert row.get_column("total_points").item() == 2

    def test_legacy_matches_default_prem(self, legacy):
        assert set(legacy["matches"].get_column("tournament")) == {"prem"}

    def test_legacy_match_stats(self, legacy):
        ms = legacy["match_stats"]
        assert ms.filter(pl.col("player_id") == 430).height == 2
        assert ms.filter(
            (pl.col("player_id") == 239) & (pl.col("match_id") == "m2")
        ).get_column("minutes_played").item() == 0.0

    def test_legacy_team_history_empty_with_schema(self, legacy):
        # legacy layout has no team_history file -> empty frame, correct schema
        th = legacy["team_history"]
        assert th.height == 0
        assert th.columns == ["player_id", "gw", "team_code", "season"]
        assert th.schema["player_id"] == pl.Int64