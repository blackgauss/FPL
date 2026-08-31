"""Basic captain and vice-captain policy."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from math import inf

from fpl.domain import Squad


def choose_captains(
    squad: Squad,
    expected: Mapping[int, float],
) -> tuple[int | None, int | None]:
    """Choose the highest-expected starter and a different-club vice.

    This is intentionally a transparent baseline, not a captain model. Missing
    forecasts rank below every forecasted player.
    """
    starters = squad.starters or tuple(p.code for p in squad.players[:11])
    if not starters:
        return None, None
    ranked = sorted(starters, key=lambda c: expected.get(c, -inf), reverse=True)
    captain = ranked[0]
    by_code = squad.by_code()
    vice = next((c for c in ranked[1:]
                 if by_code[c].club != by_code[captain].club), None)
    if vice is None and len(ranked) > 1:
        vice = ranked[1]
    return captain, vice


def set_captains(squad: Squad, expected: Mapping[int, float]) -> Squad:
    """Return a new Squad with the basic captain policy applied."""
    captain, vice = choose_captains(squad, expected)
    updated = replace(squad, captain=captain, vice_captain=vice)
    problems = updated.validate(club_baseline=squad.club_counts())
    if problems:
        raise ValueError("captain policy produced invalid squad: "
                         + "; ".join(problems))
    return updated
