"""Black-box integration tests: stage interfaces of the real pipeline.

Unit suites cover each module in isolation; these tests run the ACTUAL stage
chain back-to-back on a synthetic FPL-Core season and assert the *interface
contracts* between stages — the joins/columns/masks each stage must hand the
next. They are deliberately black-box: only public functions are called, only
outputs are asserted, and each failure names the interface that broke.

Pipeline under test (each arrow is an interface we pin):
    ingest ─► features ─► assemble ─► predict
              ─► score ─► filter ─► enumerate ─► valid 15-man squad
    live   ─► reconcile(clubs/availability) ─► pool built on current world
    live   ─► agree(price units, team) ─► hygiene counts
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from fpl.live.agreement import hygiene_summary
from fpl.live.current import construction_input
from fpl.live.filters import suggest
from fpl.live.live import to_live_frame
from fpl.team.enumerate import greedy_teams
from fpl.team.filtering import availability_from_gw_stats, filter_pool
from fpl.team.scoring import score_players
from tests.fixtures.synthetic import build_season_tree_dense
from tests.live.fixtures_live import make_payload


class ToyModel:
    """Value-spread predictor: distinct scores so pool/enumeration have a
    clean ordering (a constant predictor makes every tie, starving positions
    under the club cap)."""

    def predict(self, X):
        # deterministic per-row value spread, unlike a constant
        return 2.0 + 0.5 * (np.arange(X.shape[0]) % 11)


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    """A synthetic FPL-Core season with a real parquet feature store on disk,
    exactly as the features stage produces it."""
    from fpl.data.contract import load_season
    from fpl.data.features import build_features
    from fpl.model.train import load_training

    root = Path(tmp_path_factory.mktemp("pipeline"))
    build_season_tree_dense(root, n_players=40, n_gws=4)
    data = load_season(root, "2025-2026")
    feats = build_features(data.gw_stats, data.team_history, data.matches,
                           data.players)

    out = root / "processed"
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in [
        ("features", feats), ("gw_stats", data.gw_stats), ("players", data.players),
    ]:
        frame.write_parquet(out / f"{name}_2025-2026.parquet")

    td = load_training(out, ["2025-2026"])["2025-2026"]
    return out, td, data, feats


def test_assemble_predictable_without_target(store):
    """feature store -> assemble(require_target=False) -> predict must work for
    a pre-season / season-start window where the target does not exist yet.
    This was previously broken: it yielded zero rows."""
    from fpl.model.train import load_training

    out, _, _, _ = store
    td = load_training(out, ["2025-2026"], require_target=False)["2025-2026"]
    assert td.X.shape[0] > 0, "require_target=False must keep season-start rows"
    pred = ToyModel().predict(td.X)
    assert pred.shape == (td.X.shape[0],)
    assert np.isfinite(pred).all()


def test_assemble_training_drops_targetless(store):
    """assemble(require_target=True) must return only trainable rows (finite
    targets) but at least one row on a real season."""
    out, td, _, _ = store
    assert td.X.shape[0] > 0
    assert np.isfinite(td.y).all(), "trainable rows must all have targets"


def test_score_players_requires_training_data(store):
    """score -> filter interface: score_players consumes a TrainingData and
    returns the player-level frame filter_pool+availability understand."""
    out, td, data, _ = store
    scored, _ = score_players(td, ToyModel(), gw_start=2, gw_end=4,
                              players=data.players, detail=True)
    assert {"player_code", "web_name", "position", "expected_total"} <= set(scored.columns)

    avail = availability_from_gw_stats(data.gw_stats, data.players,
                                       gw_start=2, gw_end=4)
    pool = filter_pool(scored, avail, reserve_top=4)
    assert pool.height > 0
    assert {"team_code", "now_cost", "expected_total"} <= set(pool.columns)


def test_filter_to_enumerate_produces_valid_squads(store):
    """enumerate must yield only complete 15-player, in-budget, four-position
    squads (a season-start overflow once produced an empty squad)."""
    out, td, data, _ = store
    scored, _ = score_players(td, ToyModel(), gw_start=2, gw_end=4,
                              players=data.players, detail=True)
    avail = availability_from_gw_stats(data.gw_stats, data.players,
                                       gw_start=2, gw_end=4)
    pool = filter_pool(scored, avail, reserve_top=4)
    basket = greedy_teams(pool, n_teams=8, seed=1)

    assert basket.height > 0, "must produce at least one squad"
    squad = basket.group_by("team_id").agg(
        pl.col("player_code").count().alias("n"),
        pl.col("price_tenths").sum().alias("cost"),
        pl.col("position").n_unique().alias("positions"),
    )
    assert (squad["n"] == 15).all(), "every squad must have exactly 15 players"
    assert (squad["cost"] <= 1000).all(), "every squad within £100m budget"
    assert (squad["positions"] == 4).all(), "every squad has all 4 positions"


def test_enumerate_informative_on_infeasible_pool(tmp_path):
    """A genuinely infeasible pool (too few players/clubs for a 15-man squad)
    must raise a clear error, not silently return an empty/partial basket."""
    root = Path(tmp_path)
    build_season_tree_dense(root, n_players=8, n_gws=3, n_clubs=2)
    from fpl.data.contract import load_season
    from fpl.data.features import build_features
    from fpl.model.train import load_training

    data = load_season(root, "2025-2026")
    feats = build_features(data.gw_stats, data.team_history, data.matches,
                           data.players)
    out = root / "proc"
    out.mkdir()
    for name, frame in [("features", feats), ("gw_stats", data.gw_stats),
                        ("players", data.players)]:
        frame.write_parquet(out / f"{name}_2025-2026.parquet")
    td = load_training(out, ["2025-2026"])["2025-2026"]
    scored, _ = score_players(td, ToyModel(), gw_start=2, gw_end=3,
                              players=data.players, detail=True)
    avail = availability_from_gw_stats(data.gw_stats, data.players,
                                       gw_start=2, gw_end=3)
    pool = filter_pool(scored, avail, reserve_top=4)
    with pytest.raises(ValueError, match="could not fill any squad|enlarge pool"):
        greedy_teams(pool, n_teams=5, seed=1)


def test_live_reconcile_feeds_construction(store):
    """live current-world must feed construction: transferred players' club is
    updated and unavailable players excluded BEFORE filter_pool, so the
    team-cap/pool reflect reality not the stale dataset."""
    out, td, data, _ = store
    scored, _ = score_players(td, ToyModel(), gw_start=2, gw_end=4,
                              players=data.players, detail=True)

    # live payload over the scored players; mark 223001 transferred (new club)
    # and 223002 injured
    payload = make_payload(codes=scored["player_code"].to_list(),
                           statuses=["a"] * scored.height, n=scored.height)
    live = to_live_frame(payload).with_columns(
        pl.when(pl.col("player_code") == 223001).then(pl.lit(43))
        .otherwise(pl.col("team_code")).alias("team_code"),
        pl.when(pl.col("player_code") == 223002).then(pl.lit("i"))
        .otherwise(pl.col("status")).alias("status"),
    )
    reconciled = construction_input(scored, live, suggest(live))
    tr = reconciled.filter(pl.col("player_code") == 223001)
    assert tr.height == 1 and tr["team_code"].item() == 43
    assert 223002 not in reconciled["player_code"].to_list()


def test_live_price_units_agree(store):
    """live tenths vs dataset decimal: with price_scale=10 an agreeing price
    must not be flagged (the units interface once flagged every player)."""
    live = to_live_frame(make_payload(
        codes=[430, 239, 100, 101], prices_tenths=[155, 110, 45, 40], n=4))
    dataset = pl.DataFrame({  # decimal millions that AGREE with live
        "player_code": [430, 239, 100, 101],
        "now_cost": [15.5, 11.0, 4.5, 4.0],
        "team_code": live.sort("player_code")["team_code"],
    })
    s = hygiene_summary(live, dataset, dataset_price_col="now_cost",
                        dataset_team_col="team_code", price_scale=10)
    assert s["price_moved"].item() == 0, "units agree -> no false price moves"
    assert s["matched_to_dataset"].item() == 4


def test_full_pipeline_smoke(store):
    """Whole chain once: score -> filter -> enumerate produces a legal 15-man
    squad. Catches a regression that breaks any per-stage interface at once."""
    out, td, data, _ = store
    scored, _ = score_players(td, ToyModel(), gw_start=2, gw_end=4,
                              players=data.players, detail=True)
    avail = availability_from_gw_stats(data.gw_stats, data.players,
                                       gw_start=2, gw_end=4)
    pool = filter_pool(scored, avail, reserve_top=4)
    basket = greedy_teams(pool, n_teams=5, seed=2)
    assert basket.height > 0
    sq = basket.filter(pl.col("team_id") == basket["team_id"].item(0))
    assert sq.height == 15
    assert sq["position"].n_unique() == 4
