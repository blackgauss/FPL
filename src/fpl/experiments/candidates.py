"""Deterministic candidate-squad generation for gym arms.

All gym arms within a run must evaluate the SAME candidate squads; otherwise
differences in enumeration flakiness or seed contaminate the comparison. This
module memoizes squads per (window, model, seed) for the process lifetime and
always uses a fixed seed path.
"""

from __future__ import annotations

from dataclasses import dataclass

from fpl.domain import Squad
from fpl.pipeline import run_basket

_CACHE: dict[tuple, tuple[int, ...]] = {}


@dataclass(frozen=True, slots=True)
class CandidatePack:
    squads: tuple[Squad, ...]
    team_ids: tuple[int, ...]


def candidate_squads(
    *,
    processed: str,
    season: str,
    gw_start: int,
    gw_end: int,
    model,
    n_teams: int,
    seed: int,
) -> CandidatePack:
    """Generate `n_teams` deterministic candidate squads via run_basket.

    Memoized in-process on (season, window, n_teams, seed) so repeated calls
    (e.g. two gym arms) reuse the exact same enumeration.
    """
    key = (season, gw_start, gw_end, n_teams, seed)
    if key not in _CACHE:
        res = run_basket(
            processed=processed, season=season, gw_start=gw_start, gw_end=gw_end,
            model=model, freshness=False, value_fn="h2h",
            enum_kw={"n_teams": n_teams, "seed": seed},
        )
        _CACHE[key] = (tuple(res.squads), tuple(res.team_ids))
    squads, team_ids = _CACHE[key]
    return CandidatePack(squads=squads, team_ids=team_ids)