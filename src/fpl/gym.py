"""Offline replay gym: evaluate a Squad against actual Gameweeks.

Given a Squad at gameweek g, replay g..g+N-1 using REAL outcomes (who played,
how many points) from the stored gw_stats, applying the rigid auto-sub /
captain rule (Squad.gw_settlement). This is the evaluation harness for the
baseline: every week surfaces BOTH the model's forecast (via a `predictor`
hook) and the settled actual points, exposing how the unlearned features of
captains, transfers, and who-plays interact with the baseline.

A `policy` hook (optional) decides the next week's Squad between gameweeks —
this is where later captain / probability-of-playing / transfer models plug
in. No policy == replay in place with no transfers.

Domain-first: a Squad in, chronological WeekResults out. Actuals are read as
dicts keyed by player_code, matching the forecast/result frames.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import polars as pl

from fpl.domain import Squad

Policy = Callable[[Squad, int], Squad]          # (squad, gw) -> next-week squad
Predictor = Callable[[Squad, int], dict[int, float]]  # (squad, gw) -> expected


@dataclass(frozen=True, slots=True)
class WeekResult:
    """One replayed Gameweek: the squad as it entered, and what actually
    happened (settled via the rigid substitution/captain rule)."""

    gw: int
    squad: Squad                    # the squad that played this week
    xi: tuple[int, ...]             # effective scoring XI
    substituted_in: tuple[int, ...] # bench players who came on
    captain_doubled: int | None     # whose points counted twice (or None)
    actual_points: float            # settled points, doubling included
    predicted_points: float | None  # forecast under the same rule, or None
    xi_points: dict[int, float]     # code -> raw points for the XI (pre-double)


def _actuals(gw_stats: pl.DataFrame, players: pl.DataFrame,
             gw: int) -> tuple[dict[int, bool], dict[int, float]]:
    """Who played and scored at `gw`, keyed by player_code.

    "Played" = minutes > 0 (the rules: appeared on pitch / card).
    Points come from the per-GW total_points column.
    """
    rows = (
        gw_stats.filter(pl.col("gw") == gw)
        .join(players.select("player_id", "player_code"), on="player_id")
        .select("player_code", "minutes", "total_points")
        .iter_rows()
    )
    played: dict[int, bool] = {}
    points: dict[int, float] = {}
    for code, minutes, pts in rows:
        played[code] = (minutes or 0) > 0
        points[code] = float(pts or 0.0)
    return played, points


def replay(
    squad: Squad,
    *,
    gw_stats: pl.DataFrame,
    players: pl.DataFrame,
    weeks: int,
    policy: Policy | None = None,
    predictor: Predictor | None = None,
) -> list[WeekResult]:
    """Replay `squad` from squad.gw for `weeks` gameweeks against real data.

    `gw_stats` (player_id, gw, minutes, total_points) and `players`
    (player_id -> player_code) are the actuals store. `policy` mutates the
    squad between weeks (captain/transfers); `predictor` supplies per-player
    expected points so each week reports forecast-vs-actual under the same
    doubling/substitution rule. Weeks with no data are settled as all-players-
    dnps (scores 0) so the harness is total and never crashes.
    """
    results: list[WeekResult] = []
    current = squad
    for gw in range(squad.gw, squad.gw + weeks):
        played, points = _actuals(gw_stats, players, gw)
        settle = current.gw_settlement(played, points)

        predicted = None
        if predictor is not None:
            expected = predictor(current, gw)
            predicted = sum(
                expected.get(code, 0.0) * 2 if code == settle.captain_doubled
                else expected.get(code, 0.0)
                for code in settle.playing
            )

        results.append(WeekResult(
            gw=gw, squad=current, xi=settle.playing,
            substituted_in=settle.substituted_in,
            captain_doubled=settle.captain_doubled,
            actual_points=settle.gw_total, predicted_points=predicted,
            xi_points={code: points.get(code, 0.0) for code in settle.playing},
        ))

        current = policy(current, gw) if policy is not None \
            else replace(current, gw=gw + 1)
    return results