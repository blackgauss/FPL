"""Ownership-weighted opponent squad sampling for H2H evaluation.

`selected_by_percent` is a global ownership prior, not exact league
ownership, and captain ownership is not exposed by bootstrap. This sampler is
therefore a *probabilistic opponent population*: valid 15-player squads are
weighted by ownership for H2H Monte Carlo, and remain a research estimate
until real league entry picks are available.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from fpl.domain import Player
from fpl.units import DEFAULT_BUDGET_TENTHS, MAX_PER_CLUB, SQUAD_COUNTS

# FPL squad position requirements (single source is fpl.units.SQUAD_COUNTS).
_NEED = list(SQUAD_COUNTS.items())


def sample_opponent(
    rng: np.random.Generator,
    pool: Sequence[Player],
    ownership: dict[int, float],
    *,
    max_attempts: int = 3000,
) -> tuple[Player, ...]:
    """Sample one valid, ownership-weighted opponent squad.

    Selects players by position to satisfy SQUAD_COUNTS, respecting budget and
    the <= MAX_PER_CLUB club cap. Weight for a player is proportional to
    (ownership / reference ownership) ** 0.7, so the conditional inclusion
    probability tracks the API prior while the constraint set is respected.
    """
    reference = float(np.quantile(
        [max(ownership.get(p.code, 0.0), 0.0) for p in pool], 0.95)) or 1.0
    for _ in range(max_attempts):
        chosen: list[Player] = []
        clubs: dict[int, int] = {}
        budget = DEFAULT_BUDGET_TENTHS
        failed = False
        for position, count in _NEED:
            for _ in range(count):
                eligible = [
                    p for p in pool
                    if p.position == position and p not in chosen
                    and clubs.get(p.club, 0) < MAX_PER_CLUB
                    and p.cost_tenths <= budget
                ]
                if not eligible:
                    failed = True
                    break
                rel = np.asarray([
                    max(ownership.get(p.code, 0.0), 0.0) / reference for p in eligible
                ])
                weights = np.maximum(rel, 1e-4) ** 0.7
                weights = weights / weights.sum()
                pick = eligible[int(rng.choice(len(eligible), p=weights))]
                chosen.append(pick)
                clubs[pick.club] = clubs.get(pick.club, 0) + 1
                budget -= pick.cost_tenths
            if failed:
                break
        if not failed and len(chosen) == 15:
            return tuple(chosen)
    raise RuntimeError("could not sample a valid ownership-weighted opponent")


def sample_opponents(
    rng: np.random.Generator,
    pool: Sequence[Player],
    ownership: dict[int, float],
    n: int,
) -> list[tuple[Player, ...]]:
    """Sample `n` valid opponent squads (see sample_opponent)."""
    return [sample_opponent(rng, pool, ownership) for _ in range(n)]