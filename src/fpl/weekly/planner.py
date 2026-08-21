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
    captain/vice policy. A transfer-free week banks one free transfer, capped
    at five; spending one leaves the bank unchanged. This is a baseline path,
    not the final horizon optimizer.
    """
    planned: list[Squad] = []
    current = squad
    bank = free_transfers
    for gw in range(squad.gw, squad.gw + weeks):
        expected = forecasts.get(gw, {})
        out, new, _ = choose_transfer(
            current, candidates.get(gw, ()), expected, free_transfers=bank)
        if out is not None and new is not None:
            current = apply_transfer(current, out, new, gw=gw)
            bank = max(0, bank - 1) + 1
        else:
            bank = min(5, bank + 1)
        current = set_captains(current, expected)
        planned.append(current)
        current = replace(current, gw=gw + 1, transfers_in=())
    return planned
