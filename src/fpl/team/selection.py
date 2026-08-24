"""Shared team-selection tail: pool -> enumerate -> hydrate.

Every "build candidate squads from a scored pool" recipe repeats the same
three steps (filter_pool -> greedy_teams -> basket_squads) plus building the
expected-points map. This is that tail, in one place, so scripts (live_*,
searches) and DVC stages stay thin and identical.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from fpl.domain import Squad
from fpl.team.enumerate import greedy_teams
from fpl.team.filtering import filter_pool
from fpl.team.harness import basket_squads

DEFAULT_POOL_KW = {"top_k_per_position": 25, "max_per_team": 4, "reserve_top": 20}


@dataclass(frozen=True, slots=True)
class SelectionResult:
    squads: tuple[Squad, ...]
    team_ids: tuple[int, ...]
    expected: dict[int, float]      # player_code -> expected_total
    pool_size: int


def pool_and_squads(
    scored: pl.DataFrame,
    names_frame: pl.DataFrame,
    availability: pl.DataFrame,
    *,
    gw: int,
    n_teams: int = 20,
    seed: int = 1,
    pool_kw: dict | None = None,
) -> SelectionResult:
    """Filter a scored pool, enumerate candidate squads, and hydrate them.

    `scored` carries player_code/web_name/position/team_code/expected_total
    (+ minutes_in_window for the never-featuring cut) — the scored pool.
    `names_frame` carries player_code/web_name/team_code for hydration AFTER
    reconcile (pass the same frame you want squads built on — e.g. the
    construction_input() output for live workflows). `availability` supplies
    now_cost. Returns typed Squads, their team ids, and the expected map.
    """
    pool = filter_pool(scored, availability, **(pool_kw or DEFAULT_POOL_KW))
    basket = greedy_teams(pool, n_teams=n_teams, seed=seed)
    hydrated = basket_squads(basket, names_frame, gw=gw)
    expected = dict(zip(scored["player_code"].to_list(),
                        scored["expected_total"].to_list(), strict=False))
    return SelectionResult(
        squads=tuple(s for _, s in hydrated),
        team_ids=tuple(int(t) for t, _ in hydrated),
        expected=expected,
        pool_size=int(pool.height),
    )