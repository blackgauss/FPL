"""End-to-end: the composable pipeline runner.

These assert the search and live-reconcile paths through `fpl.pipeline.run_basket`,
so orchestration stays in one place instead of drifting across scripts.
Synthetic-only; no network.
"""

from pathlib import Path

import polars as pl
import pytest

from fpl.data.contract import load_season
from fpl.data.features import build_features
from fpl.live.live import to_live_frame
from fpl.pipeline import run_basket
from tests.fixtures.synthetic import build_season_tree_dense
from tests.live.fixtures_live import make_payload


class ValueModel:
    def predict(self, X):
        import numpy as np

        return 2.0 + 0.5 * (np.arange(X.shape[0]) % 11)


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    from fpl.model.train import load_training

    root = Path(tmp_path_factory.mktemp("e2e"))
    build_season_tree_dense(root, n_players=40, n_gws=4)
    data = load_season(root, "2025-2026")
    feats = build_features(data.gw_stats, data.team_history, data.matches,
                           data.players)
    out = root / "processed"
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in [("features", feats), ("gw_stats", data.gw_stats),
                        ("players", data.players)]:
        frame.write_parquet(out / f"{name}_2025-2026.parquet")
    load_training(out, ["2025-2026"])
    return out, data


def test_e2e_search_path(store):
    """The search_teams path (no live): assemble -> score -> filter ->
    enumerate -> value, yielding a valid basket + value frame."""
    out, _ = store
    res = run_basket(processed=str(out), season="2025-2026", gw_start=2,
                     gw_end=4, model=ValueModel(), freshness=False,
                     enum_kw={"n_teams": 5, "seed": 1},
                     value_fn="h2h")
    assert res.basket.height > 0
    assert {"win_ratio", "avg_edge"} <= set(res.value.columns)
    # validity contract (mirrors integration suite): 15 players, <=1000t, 4 pos
    sq = res.basket.group_by("team_id").agg(
        pl.col("player_code").count().alias("n"),
        pl.col("price_tenths").sum().alias("cost"),
        pl.col("position").n_unique().alias("pos"),
    )
    assert (sq["n"] == 15).all()
    assert (sq["cost"] <= 1000).all()
    assert (sq["pos"] == 4).all()


def test_e2e_live_apply_path(store):
    """The live_apply path: reconcile live into the input before the pool (the
    exact thing scripts/live_apply.py wires up), and the pool respects it."""
    from fpl.model.train import load_training
    from fpl.team.scoring import score_players

    out, data = store
    td = load_training(str(out), ["2025-2026"])["2025-2026"]
    scored, _ = score_players(td, ValueModel(), gw_start=2, gw_end=4,
                              players=data.players, detail=True)
    codes = scored["player_code"].to_list()
    live = to_live_frame(make_payload(codes=codes, statuses=["a"] * len(codes),
                                      n=len(codes)))

    res = run_basket(processed=str(out), season="2025-2026", gw_start=2,
                     gw_end=4, model=ValueModel(), live=live, freshness=False,
                     enum_kw={"n_teams": 3, "seed": 1}, value_fn="h2h")
    assert res.basket.height > 0
    # reconcile guarantees the POOL's clubs are live clubs (transfer updates);
    # basket has no club column, but every pick comes from the pool
    live_clubs = set(live["team_code"].unique().to_list())
    pool_teams = set(res.pool["team_code"].unique().to_list())
    assert pool_teams <= live_clubs
