"""Basic multi-Gameweek composition of transfer and captain policies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

from fpl.domain import Player, Squad
from fpl.weekly.captain import set_captains
from fpl.weekly.transfer import apply_transfer, choose_transfer


def plan_weeks(
    squad: Squad,
    forecasts: Mapping[int, Mapping[int, float]],
    candidates: Mapping[int, Sequence[Player]],
    *,
    weeks: int,
    free_transfers: int = 1,
) -> list[Squad]:
    """Return one configured Squad snapshot per week.

    Each week performs at most one greedy transfer, then applies the basic
    captain/vice policy. Free-transfer accounting is FPL's: every week earns
    one free transfer (bank capped at five); spending one consumes a banked
    transfer, so the bank drains across consecutive transfer weeks. This is a
    baseline path, not the final horizon optimizer.
    """
    planned: list[Squad] = []
    problems = squad.validate()
    if problems:
        raise ValueError("cannot plan from invalid squad: " + "; ".join(problems))
    if free_transfers < 0:
        raise ValueError("free_transfers must be non-negative")
    current = squad
    bank = max(0, free_transfers - 1)  # carryover INTO the first planned week
    for gw in range(squad.gw, squad.gw + weeks):
        available = min(5, bank + 1)  # this week's earned FT, capped bank
        expected = forecasts.get(gw, {})
        out, new, _ = choose_transfer(
            current, candidates.get(gw, ()), expected, free_transfers=available)
        if out is not None and new is not None:
            current = apply_transfer(current, out, new, gw=gw)
            bank = max(0, available - 1)
        else:
            bank = available
        current = set_captains(current, expected)
        planned.append(current)
        current = replace(current, gw=gw + 1, transfers_in=())
    return planned


def make_policy(
    forecasts: Mapping[int, Mapping[int, float]],
    candidates: Mapping[int, Sequence[Player]],
    *,
    free_transfers: int = 1,
):
    """Create a gym policy that decides the next gameweek from forecasts.

    Gym calls a policy after settling gameweek ``gw``. This adapter therefore
    applies the transfer and captain policies to ``gw + 1`` and carries the
    free-transfer bank across calls (one FT earned per decision, capped at
    five; spending consumes a banked one). Actual outcomes are never supplied
    to the decision; gym only supplies the settled squad and current gameweek.
    """
    if free_transfers < 0:
        raise ValueError("free_transfers must be non-negative")
    bank = max(0, free_transfers - 1)  # carryover INTO the first decision

    def step(squad: Squad, gw: int) -> Squad:
        nonlocal bank
        next_gw = gw + 1
        current = replace(squad, gw=next_gw, transfers_in=())
        available = min(5, bank + 1)
        expected = forecasts.get(next_gw, {})
        out, new, _ = choose_transfer(
            current, candidates.get(next_gw, ()), expected,
            free_transfers=available,
        )
        if out is not None and new is not None:
            current = apply_transfer(current, out, new, gw=next_gw)
            bank = max(0, available - 1)
        else:
            bank = available
        return set_captains(current, expected)

    return step
