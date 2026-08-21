"""Offline replay gym: evaluate a Squad against actual Gameweeks.

Given a Squad at gameweek g, replay g..g+N-1 using REAL outcomes (who played,
how many points) from the stored gw_stats, applying the rigid auto-sub /
captain rule (Squad.gw_settlement). This is the evaluation harness for the
baseline: every week surfaces BOTH the model's forecast (a `predictor` or a
stored `forecast` frame) and the settled actual points, exposing how the
unlearned features of captains, transfers, and who-plays interact with the
baseline.

What an EVAL is (rigid protocol):

    Eval(squad, gw_stats, players, weeks, policy?, forecast?/predictor?)
        .run() -> EvalResult

A `policy` (squad, gw) -> next Squad is where captain / probability-of-
playing / transfer models plug in; no policy == replay in place. The
EvalResult is also the observability home: it aggregates and NARRATES a run
(total forecast vs actual gap, substitutions, starter dnps, captain usage)
so qualitative model behaviour is visible for rapid iteration.

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

    @property
    def gap(self) -> float | None:
        """predicted - actual for this week (None without a forecast)."""
        if self.predicted_points is None:
            return None
        return self.predicted_points - self.actual_points

    @property
    def dnps(self) -> tuple[int, ...]:
        """Starters who didn't play this week (auto-sub candidates)."""
        return tuple(c for c in self.squad.starters if c not in self.xi)


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


@dataclass(frozen=True, slots=True)
class Eval:
    """A rigidly-specified evaluation: everything needed, named, observable.

    An eval is: take a Squad at gameweek g, replay `weeks` real Gameweeks
    under a (possibly empty) policy, and measure the forecast against how
    the squad ACTUALLY scored through the real rules (subs + captain). Give
    the model's expected points as EITHER `predictor(squad, gw) -> {code:
    float}` OR a stored `forecast` frame (player_code, gw, expected_points)
    — never both. `name` tags the run for observability.
    """

    squad: Squad
    gw_stats: pl.DataFrame
    players: pl.DataFrame
    weeks: int
    policy: Policy | None = None
    predictor: Predictor | None = None
    forecast: pl.DataFrame | None = None
    name: str = "eval"

    def __post_init__(self) -> None:
        if self.predictor is not None and self.forecast is not None:
            raise ValueError("Eval takes a predictor OR a forecast frame, not both")

    def run(self) -> EvalResult:
        predictor = self.predictor
        if predictor is None and self.forecast is not None:
            forecast = self.forecast

            def predictor(squad: Squad, gw: int) -> dict[int, float]:
                rows = forecast.filter(
                    pl.col("player_code").is_in(squad.codes())
                    & (pl.col("gw") == gw))
                return dict(zip(rows["player_code"], rows["expected_points"],
                                strict=False))

        weeks = replay(self.squad, gw_stats=self.gw_stats, players=self.players,
                       weeks=self.weeks, policy=self.policy,
                       predictor=predictor)
        return EvalResult(spec=self, weeks=tuple(weeks))


@dataclass(frozen=True, slots=True)
class EvalResult:
    """Everything an eval produced — the observability surface.

    Week-level traces (WeekResult) plus aggregates that let you read WHY a
    run landed where it did: the forecast-vs-actual gap, how many times bench
    priorities had to act, and captain/formation usage. `summary()` narrates
    it for quick qualitative inspection during iteration.
    """

    spec: Eval
    weeks: tuple[WeekResult, ...]

    @property
    def total_actual(self) -> float:
        return sum(w.actual_points for w in self.weeks)

    @property
    def total_predicted(self) -> float | None:
        pred = [w.predicted_points for w in self.weeks]
        if any(p is None for p in pred):
            return None
        return sum(p for p in pred)  # type: ignore[arg-type]

    @property
    def gap(self) -> float | None:
        """predicted - actual over the whole run (None without a forecast)."""
        if self.total_predicted is None:
            return None
        return self.total_predicted - self.total_actual

    @property
    def substitutions(self) -> int:
        return sum(len(w.substituted_in) for w in self.weeks)

    @property
    def dnps(self) -> int:
        return sum(len(w.dnps) for w in self.weeks)

    @property
    def captain_weeks(self) -> int:
        return sum(1 for w in self.weeks if w.captain_doubled is not None)

    def summary(self) -> str:
        """A one-screen qualitative report for rapid iteration."""
        s = self.spec
        lines = [
            f"[{s.name}] gw {s.squad.gw}..{s.squad.gw + len(self.weeks) - 1} "
            f"({len(self.weeks)} weeks)",
            f"  actual     {self.total_actual:7.1f} pts",
        ]
        if self.total_predicted is not None:
            lines.append(f"  predicted  {self.total_predicted:7.1f} pts "
                         f"(gap {self.gap:+.1f})")
        lines.append(f"  bench acted: {self.substitutions} subs, "
                     f"{self.dnps} starter dnps | captain doubled in "
                     f"{self.captain_weeks} of {len(self.weeks)} weeks")
        worst = max(self.weeks, key=lambda w: w.gap or 0.0)
        detail = f": actual {worst.actual_points:.1f}"
        if worst.predicted_points is not None:
            detail += f" vs pred {worst.predicted_points:.1f}"
        lines.append(f"  worst miss gw{worst.gw}{detail}")
        return "\n".join(lines)