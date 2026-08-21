"""Composable pipeline runner: one path from raw data to a candidate basket.

Every later stage (captain, transfers, horizon sim) should build on this single
runner rather than re-implementing the chain. It is pure orchestration: it
delegates to leaf modules and inserts two **gates** that must hold for any team
selection:

    1. freshness — the feature store is non-empty and live hasn't drifted too
                    far from it (fpl.live.freshness).
    2. leakage   — model training must run validate() before fitting
                    (fpl.model.leakage); callers choosing a split must provide
                    its window (enforced by run_experiment, not duplicated).

Separation of concerns: leaf modules (score/filter/enumerate/harness) stay pure
over their inputs; this module owns the wiring and the gates. The harness
accepts an injected `scored` (the seam) so live reconcile can slot in upstream.
"""

from __future__ import annotations

import polars as pl

from fpl.live.current import construction_input
from fpl.live.filters import suggest
from fpl.team.enumerate import greedy_teams  # noqa: F401  (registry default)
from fpl.team.harness import run as harness_run
from fpl.team.scoring import score_players


def run_basket(
    *,
    processed: str,
    season: str,
    gw_start: int,
    gw_end: int,
    model,
    live: pl.DataFrame | None = None,
    freshness: bool = True,
    **harness_kw,
):
    """score -> reconcile(live) -> filter -> enumerate -> value, with gates.

    When `live` is given it is reconciled into the scored input first (club
    transfers, injured/removed) so construction never sees a stale team — the
    harness is handed a pre-scored frame via its `scored` seam.

    Returns a fpl.team.harness.SearchResult. `harness_kw` (enum/value/...)
    forwards to the harness; `pool_kw`, `enum_kw`, `value_fn` etc. are legal.
    """
    from fpl.live.freshness import check_season_has_rows
    from fpl.model.train import load_training

    if freshness:
        check_season_has_rows(processed, season)

    players = pl.read_parquet(f"{processed}/players_{season}.parquet")
    td = load_training(processed, [season])[season]
    scored, per_gw = score_players(
        td, model, gw_start=gw_start, gw_end=gw_end, players=players,
        detail=True)

    if live is not None:
        scored = construction_input(scored, live, suggest(live))

    return harness_run(
        processed=processed, season=season, gw_start=gw_start, gw_end=gw_end,
        model=model, scored=scored, per_gw=per_gw, players=players,
        **harness_kw,
    )