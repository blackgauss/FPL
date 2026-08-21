"""Black-box integration: the composable pipeline runner + its gates.

These run the actual `fpl.pipeline.run_basket` end-to-end on a dense synthetic
season — the same path every later stage builds on — and assert the gates and
composition hold:
- freshness gate blocks an empty/missing feature store;
- live reconcile is injected via the pipeline (transf club updated upstream);
- end-to-end produces a valid basket with a value frame.
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


class ToyModel:
    """Value-spread predictor: distinct scores so pool/enumeration have a
    clean ordering (a constant predictor makes every tie, starving positions
    under the club cap)."""

    def predict(self, X):
        import numpy as np

        return 2.0 + 0.5 * (np.arange(X.shape[0]) % 11)


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    from fpl.model.train import load_training

    root = Path(tmp_path_factory.mktemp("pipeline"))
    build_season_tree_dense(root, n_players=40, n_gws=4)
    data = load_season(root, "2025-2026")
    feats = build_features(data.gw_stats, data.team_history, data.matches,
                           data.players)
    out = root / "processed"
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in [("features", feats), ("gw_stats", data.gw_stats),
                        ("players", data.players)]:
        frame.write_parquet(out / f"{name}_2025-2026.parquet")
    # ensure trainable rows exist so load_training works
    load_training(out, ["2025-2026"])
    return out, data


def test_run_basket_freshness_gate(tmp_path):
    """An empty feature store must be blocked before any scoring.

    Uses its own tmp dir — never mutates the shared `store` fixture (a shared
    fixture is read-only for tests that run alongside it).
    """
    pl.DataFrame({"player_code": [], "gw": []}).write_parquet(
        tmp_path / "features_2025-2026.parquet")
    from fpl.live.freshness import FreshenError

    with pytest.raises(FreshenError, match="empty"):
        run_basket(processed=str(tmp_path), season="2025-2026", gw_start=2,
                   gw_end=4, model=ToyModel(), freshness=True)


def test_run_basket_end_to_end(store):
    """Score->filter->enumerate->value via the runner yields a valid basket."""
    out, _ = store
    res = run_basket(processed=str(out), season="2025-2026", gw_start=2,
                     gw_end=4, model=ToyModel(),
                     freshness=False, enum_kw={"n_teams": 5, "seed": 1})
    assert res.basket.height > 0
    assert "win_ratio" in res.value.columns
    sq = res.basket.filter(pl.col("team_id") == res.basket["team_id"].item(0))
    assert sq.height == 15
    assert sq["position"].n_unique() == 4


def test_run_basket_live_reconcile_filters_team(store):
    """Live reconcile is applied upstream: an injured/missing player is
    excluded before enumeration, and a transferred player's club is updated."""
    from fpl.model.train import load_training
    from fpl.team.scoring import score_players

    out, data = store
    # ensure the store has training rows (the other tests may not have loaded)
    td = load_training(str(out), ["2025-2026"])["2025-2026"]
    scored, _ = score_players(td, ToyModel(), gw_start=2, gw_end=4,
                              players=data.players, detail=True)

    codes = scored["player_code"].to_list()
    live = to_live_frame(make_payload(
        codes=codes, statuses=["a"] * len(codes), n=len(codes))).with_columns(
        pl.when(pl.col("player_code") == 223001).then(pl.lit(43))
        .otherwise(pl.col("team_code")).alias("team_code"),
        pl.when(pl.col("player_code") == 223002).then(pl.lit("i"))
        .otherwise(pl.col("status")).alias("status"),
    )

    res = run_basket(processed=str(out), season="2025-2026", gw_start=2,
                     gw_end=4, model=ToyModel(), live=live, freshness=False,
                     enum_kw={"n_teams": 3, "seed": 1})
    basket_codes = set(res.basket["player_code"].to_list())
    assert 223002 not in basket_codes, "injured player must be excluded"
    pool = res.pool
    p1 = pool.filter(pl.col("player_code") == 223001)
    if p1.height:
        assert p1["team_code"].item() == 43


def test_domain_feed_from_training_and_forecast(store):
    """Training -> forecast -> domain objects, with units handled exactly once.

    The seam captains/transfers will stand on: score_players + filter_pool ->
    players_frame (canonical price_tenths) -> the domain builders -> a valid
    Squad, with numpy consumption and per-player forecast lookup all keyed by
    player_code. No algorithm re-derives the tenths conversion.
    """
    import numpy as np

    from fpl.domain import Position, players_from_frame
    from fpl.model.train import load_training
    from fpl.team.enumerate import greedy_teams, squad_for_price
    from fpl.team.filtering import (
        availability_from_gw_stats,
        filter_pool,
        players_frame,
    )
    from fpl.team.harness import basket_squads
    from fpl.team.scoring import score_players

    out, data = store
    td = load_training(str(out), ["2025-2026"])["2025-2026"]
    scored, per_gw = score_players(td, ToyModel(), gw_start=2, gw_end=4,
                                   players=data.players, detail=True)
    avail = availability_from_gw_stats(data.gw_stats, data.players,
                                       gw_start=2, gw_end=4)
    pool = filter_pool(scored, avail, top_k_per_position=25, max_per_team=4)

    # canonical domain frame: the exact _FRAME_PLAYER_COLUMNS contract
    pf = players_frame(pool)
    assert set(pf.columns) == {"player_code", "web_name", "position",
                               "team_code", "price_tenths"}
    # agrees with the existing canonical tenths producer (no re-derived math)
    cross = pf.select("player_code", "price_tenths").join(
        squad_for_price(pool).select("player_code", "price_tenths"),
        on="player_code", how="inner", suffix="_ref")
    assert cross.filter(
        pl.col("price_tenths") != pl.col("price_tenths_ref")).height == 0

    players = players_from_frame(pf)
    assert all(isinstance(p.position, Position) for p in players)

    # forward into the domain and check a valid Squad carries those prices
    basket = greedy_teams(pool, n_teams=1, seed=1)
    _, squad = basket_squads(basket, pf, gw=2)[0]
    assert squad.validate() == []
    assert all(p.cost_tenths == pf.filter(
        pl.col("player_code") == p.code)["price_tenths"].item()
        for p in squad.players)

    # numpy consumption and the captain primitive both key on player_code
    costs = pf.filter(pl.col("player_code").is_in(squad.codes()))[
        "price_tenths"].to_numpy()
    assert isinstance(costs, np.ndarray)
    assert int(costs.sum()) == squad.cost_tenths()
    # captain primitive: every squad player has a per-player-GW forecast
    # (the dense fixture's feature rows start at gw=2, so forecasts begin at 3)
    target_gw = int(per_gw["gw"].min())
    forecast_codes = set(per_gw.filter(pl.col("gw") == target_gw)[
        "player_code"].to_list())
    assert not [c for c in squad.codes() if c not in forecast_codes]