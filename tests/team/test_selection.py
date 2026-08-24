"""Contract tests: the shared pool -> enumerate -> hydrate tail."""

from pathlib import Path

import polars as pl

from fpl.data.contract import load_season
from fpl.team.selection import SelectionResult, pool_and_squads


def _make_store(tmp_path):
    from tests.fixtures.synthetic import build_season_tree_dense

    root = Path(tmp_path)
    build_season_tree_dense(root, n_players=40, n_gws=4)
    return load_season(root, "2025-2026")


def _scored_from(data) -> pl.DataFrame:
    """A small scored frame with stable codes + a deterministic score."""
    players = data.players.sort("player_code")
    n = players.height
    return players.with_columns(
        (4.0 + (pl.Series(range(n)) % 8).cast(pl.Float64))
        .alias("expected_total")).with_columns(
        pl.lit(1).alias("minutes_in_window"))


def test_pool_and_squads_builds_valid_squads(tmp_path):
    data = _make_store(tmp_path)
    scored = _scored_from(data)
    players = data.players
    gw_stats = data.gw_stats
    # availability frame for filter_pool (price from gw_stats, latest now_cost)
    availability = (gw_stats.join(players.select("player_id", "player_code"),
                                  on="player_id", how="inner")
                    .group_by("player_code").agg(
                        pl.col("now_cost").last().alias("now_cost"))
                    .with_columns(pl.lit(1).alias("minutes_in_window")))

    selection = pool_and_squads(scored, scored, availability, gw=2,
                                n_teams=3, seed=1)
    assert isinstance(selection, SelectionResult)
    assert len(selection.squads) == 3
    assert len(selection.team_ids) == len(selection.squads)
    for squad in selection.squads:
        assert squad.validate() == []
    assert selection.pool_size > 0
    # every picked player is represented in the expected map
    for squad in selection.squads:
        assert selection.expected.get(squad.codes()[0]) is not None