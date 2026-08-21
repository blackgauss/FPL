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

from fpl.domain import Squad, squad_from_frame
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
    squads: list[Squad]             # typed teams, ordered by window_total desc
    team_ids: tuple[int, ...]       # aligned with squads
    value: pl.DataFrame             # team value from the value strategy
    weakness: pl.DataFrame          # window_total, worst_gw, star_dependence
    search_size: tuple[int, float]  # (n possibilities, log10)

    def squad(self, team_id: int) -> Squad | None:
        """The typed team behind a search team_id (None if not in the basket)."""
        for tid, s in zip(self.team_ids, self.squads, strict=False):
            if tid == team_id:
                return s
        return None

    def best(self, kind: str = "weakness") -> tuple[int, Squad]:
        """(team_id, Squad) of the top-ranked team.

        kind="weakness": highest expected window total; kind="value": best per
        the value strategy (win_ratio). Consumers never need to touch team_id
        bookkeeping for "give me the best team".
        """
        if kind == "weakness":
            return self.team_ids[0], self.squads[0]
        if kind == "value":
            tid = int(self.value.sort("win_ratio", descending=True)
                      ["team_id"].head(1).item())
            s = self.squad(tid)
            if s is None:
                raise ValueError(f"value ranks team {tid} outside the basket")
            return tid, s
        raise ValueError(f"unknown best kind {kind!r}")


def _load(processed: str, season: str):
    from fpl.model.train import load_training

    td = load_training(processed, [season])[season]
    players = pl.read_parquet(f"{processed}/players_{season}.parquet")
    gw_stats = pl.read_parquet(f"{processed}/gw_stats_{season}.parquet")
    return td, players, gw_stats


def basket_squads(
    basket: pl.DataFrame, names: pl.DataFrame, *, gw: int,
) -> list[tuple[int, Squad]]:
    """Hydrate the basket (long team_id frame) into typed Squad objects.

    `names` (player_code, web_name, team_code) carries the club semantics the
    pool was built on — typically the scored/reconciled frame, so squad clubs
    are current-world and greedy's ≤3/club caps hold. Price is already tenths
    in the basket. Callers (inspect_teams, live check, the weekly optimizers)
    read Squad/Player objects and never see the search internals: team_id
    bookkeeping, column names, or dataset price units.

    Returns [(team_id, Squad)] in basket row order.
    """
    if basket.height == 0:
        return []
    denorm = basket.join(
        names.select("player_code", "web_name", "team_code"),
        on="player_code", how="left",
    )
    need = ["player_code", "web_name", "position", "team_code", "price_tenths"]
    squads: list[tuple[int, Squad]] = []
    for (tid,), g in denorm.group_by("team_id", maintain_order=True):
        squads.append((int(tid), squad_from_frame(g.select(need), gw=gw)))
    return squads


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

    if players is None:
        players = pl.read_parquet(f"{processed}/players_{season}.parquet")
    # hydrate from the frame whose team_code the pool was built on: an injected
    # `scored` already carries live-reconciled clubs (greedy's club caps are
    # live-aware), so re-joining the stale players table would break the ≤3/club
    # invariant on the squads; otherwise the players table is the source.
    names = scored if (scored is not None and "team_code" in scored.columns) \
        else players
    squads = basket_squads(basket, names, gw=gw_start)
    by_id = {tid: sq for tid, sq in squads}
    # order typed squads by window total (weakness order) so consumers just
    # iterate best-first without touching team_id bookkeeping
    order_ids = weak.sort("window_total", descending=True)["team_id"].to_list()
    team_ids = tuple(int(t) for t in order_ids)
    squads = [by_id[t] for t in order_ids]

    return SearchResult(
        season=season, gw_start=gw_start, gw_end=gw_end,
        pool=pool, basket=basket, squads=squads, team_ids=team_ids,
        value=value, weakness=weak,
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