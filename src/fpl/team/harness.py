"""Reusable team-search harness.

The pipeline skeleton is fixed:
    score every player over the horizon
  -> filter to a diverse, budget-relevant pool
  -> enumerate a basket of valid squads
  -> value each team (mean H2H, or distributional MC via h2h_dist)

Nothing about this harness is specific to one algorithm. The strategies are
injected by name from REGISTRY:

    model       a fitted estimator (tree, linear, JAX, ...) with .predict(X)
    enumerate   (pool, **kw) -> long basket frame          [greedy / mcmc later]
    value       (totals) or (dist_forecast, basket) -> value frame
                                                          [h2h / h2h_dist]

Swap a strategy by editing REGISTRY; the pipeline and the search-OOM
bookkeeping stay identical.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from fpl.team.enumerate import greedy_teams, search_space_size
from fpl.team.filtering import availability_from_gw_stats, filter_pool
from fpl.team.scoring import score_players
from fpl.team.simulate import (
    simulate_h2h,
    simulate_h2h_dist,
    simulate_squad_distributions,
    squad_gw_totals,
    weaknesses,
)

REGISTRY = {
    "enumerate": {
        "greedy": greedy_teams,
    },
    "value": {
        "h2h": simulate_h2h,
        "h2h_dist": simulate_h2h_dist,
    },
}


@dataclass(frozen=True)
class SearchResult:
    season: str
    gw_start: int
    gw_end: int
    pool: pl.DataFrame
    basket: pl.DataFrame            # long: team_id, player_code, ...
    value: pl.DataFrame             # team value from the value strategy
    weakness: pl.DataFrame          # window_total, worst_gw, star_dependence
    search_size: tuple[int, float]  # (n possibilities, log10)


def _load(processed: str, season: str):
    from fpl.model.train import load_training

    td = load_training(processed, [season])[season]
    players = pl.read_parquet(f"{processed}/players_{season}.parquet")
    gw_stats = pl.read_parquet(f"{processed}/gw_stats_{season}.parquet")
    return td, players, gw_stats


def run(
    *,
    processed: str,
    season: str,
    gw_start: int,
    gw_end: int,
    model,
    enum: str = "greedy",
    value_fn: str = "h2h",
    pool_kw: dict | None = None,
    enum_kw: dict | None = None,
    value_kw: dict | None = None,
    dist_forecast: pl.DataFrame | None = None,
    scored: pl.DataFrame | None = None,
    per_gw: pl.DataFrame | None = None,
    pool: pl.DataFrame | None = None,
    players: pl.DataFrame | None = None,
) -> SearchResult:
    """Execute score -> filter -> enumerate -> value for one configuration.

    `dist_forecast` (per-player-GW CDFs) is required by value_fn="h2h_dist";
    pass it in — not re-fit inside the harness.

    Composability: `scored`/`per_gw` (and optionally `pool`/`players`) let a
    caller inject a pre-computed scoring — e.g. after live reconcile — instead
    of the harness re-scoring from disk. When omitted the harness assembles
    from `processed` itself. This is the seam `fpl.pipeline` uses to keep the
    leaf modules free of orchestration.
    """
    if scored is None or per_gw is None:
        td, players, gw_stats = _load(processed, season)
        scored, per_gw = score_players(
            td, model, gw_start=gw_start, gw_end=gw_end, players=players,
            detail=True)
        avail = availability_from_gw_stats(gw_stats, players,
                                           gw_start=gw_start, gw_end=gw_end)
        pool = filter_pool(scored, avail, **(pool_kw or {}))
    else:
        # caller injected a pre-scored frame (e.g. after live reconcile);
        # pool must be built from the same scored + its availability
        if pool is None:
            pool_players = players if players is not None else pl.read_parquet(
                f"{processed}/players_{season}.parquet")
            pool_gw = pl.read_parquet(f"{processed}/gw_stats_{season}.parquet")
            avail = availability_from_gw_stats(
                pool_gw, pool_players, gw_start=gw_start, gw_end=gw_end)
            pool = filter_pool(scored, avail, **(pool_kw or {}))

    basket = REGISTRY["enumerate"][enum](pool, **(enum_kw or {}))
    totals = squad_gw_totals(basket, per_gw)
    value = _value(value_fn, basket, totals, dist_forecast, value_kw or {})
    weak = weaknesses(basket, totals)

    return SearchResult(
        season=season, gw_start=gw_start, gw_end=gw_end,
        pool=pool, basket=basket, value=value, weakness=weak,
        search_size=search_space_size(pool),
    )


def _value(value_fn: str, basket, totals, dist_forecast, value_kw):
    fn = REGISTRY["value"][value_fn]
    if value_fn == "h2h":
        return fn(totals, **value_kw)
    if value_fn == "h2h_dist":
        if dist_forecast is None:
            raise ValueError(
                "value_fn='h2h_dist' needs dist_forecast (per-player-GW CDFs); "
                "compute via scripts/fit_dist.py and pass it in")
        n = value_kw.get("n_samples", 150)
        seed = value_kw.get("seed", 0)
        squad_dist = simulate_squad_distributions(
            basket, dist_forecast, n_samples=n, seed=seed)
        return fn(squad_dist, n_samples=n)
    raise ValueError(f"unknown value strategy {value_fn!r}")