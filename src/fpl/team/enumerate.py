"""Stage 3: enumerate within-budget, valid FPL squads from a filtered pool.

A valid squad: 15 players, budget <= £100m, position counts (2 GKP, 5 DEF,
5 MID, 3 FWD), max 3 per club. To keep the search tractable at pool sizes
~50-90, we use a beam search over positions rather than an exhaustive
N-choose-15: pick per position, in a value order, keeping the top `width`
partial squads at each step, and enforce budget/club constraints as we go.

The journal wants the search space "large enough for complexity, small enough
to be tractable" — beam width is the knob. A small width gives a diverse
shortlist; the top teams by expected total are the "basket".
"""

from __future__ import annotations

import polars as pl

from fpl.units import DEFAULT_BUDGET_TENTHS, MAX_PER_CLUB, SQUAD_COUNTS, to_tenths


def squad_for_price(pool: pl.DataFrame) -> pl.DataFrame:
    """Annotate pool with price in tenths and a player code string."""
    return pool.with_columns(
        pl.col("now_cost").map_elements(to_tenths, return_dtype=pl.Int64)
        .alias("price_tenths"),
        pl.col("player_code").cast(pl.Utf8).alias("code_str"),
    )


def _charge(club_counts: dict[int, int], code: int) -> bool:
    return club_counts.get(code, 0) + 1 <= MAX_PER_CLUB


def search_space_size(pool: pl.DataFrame) -> tuple[int, float]:
    """Theoretical number of squads from `pool`, ignoring budget + club rules.

    Counts C(pick, need) per position (players traded freely between slots):
        teams = C(GKP, 2) * C(DEF, 5) * C(MID, 5) * C(FWD, 3)
    Returns (n_teams, log10(n_teams)) so the order of magnitude of the search
    space is explicit before any optimization. Budget/club constraints shrink
    the true feasible set below this bound.
    """
    from math import comb, log10

    counts = (
        pool.group_by("position").len()
        .with_columns(pl.col("position").cast(pl.String))
    )
    n_per_pos = dict(zip(counts["position"], counts["len"], strict=False))
    total = 1
    for pos, need in SQUAD_COUNTS.items():
        avail = n_per_pos.get(pos, 0)
        if avail < need:
            raise ValueError(
                f"pool has {avail} {pos} but need {need} — enlarge the pool")
        total *= comb(avail, need)
    return total, log10(total)


def greedy_teams(
    pool: pl.DataFrame,
    *,
    budget_tenths: int = DEFAULT_BUDGET_TENTHS,
    n_teams: int = 20,
    max_per_club: int = MAX_PER_CLUB,
    seed: int = 0,
) -> pl.DataFrame:
    """Build `n_teams` distinct, valid squads greedily by position.

    For each position, candidates are ranked by expected value, then a small
    per-team random number of the best are excluded (each team 'misses out on'
    different stars) before greedy fill under budget + club limits — so teams
    are genuinely distinct good squads, not 5 rotations of the same one.
    Deterministic given `seed`. Heuristic baseline, not optimal.

    Returns a long frame: (team_id, position, player_code, expected_total,
    now_cost).
    """
    import random

    pool = squad_for_price(pool)
    teams: list[pl.DataFrame] = []

    by_pos = {
        pos: pool.filter(pl.col("position") == pos)
                 .sort("expected_total", descending=True)
        for pos in SQUAD_COUNTS
    }
    # order positions by scarcity first: a position needing many players from few
    # clubs must be filled before high-value positions eat those clubs. This is
    # what keeps DEF feasible when FWD/MID dominate the same clubs.
    scarcity = [
        (pos, by_pos[pos]["team_code"].n_unique() / need)
        for pos, need in SQUAD_COUNTS.items()
    ]
    fill_order = [p for p, _ in sorted(scarcity, key=lambda t: (t[1], t[0]))]

    # cheapest possible cost per position (club-agnostic lower bound) — used
    # as a safety margin so the greedy never starves a later position
    cheapest_pos: dict[str, int] = {}
    for pos, need in SQUAD_COUNTS.items():
        cheapest_pos[pos] = int(
            by_pos[pos].sort("price_tenths")["price_tenths"].head(need).sum()
        )

    rng = random.Random(seed)

    for team_id in range(n_teams):
        # bounded retries: a fresh attempt may reorder which club star-winners
        # are taken, letting a previously-failed position complete. Each retry
        # uses fresh state + a deterministic per-team rng offset.
        for attempt in range(3):
            picks: list[pl.DataFrame] = []
            budget_left = budget_tenths
            club_counts: dict[int, int] = {}
            remaining = list(fill_order)
            rng = random.Random(seed * 1000 + team_id * 10 + attempt)

            def reserve_for_remaining(rl: list[str], except_pos: str) -> int:
                return sum(cheapest_pos[p] for p in rl if p != except_pos)

            def fill_one(pos: str, row_dict: dict, budget: int, reserve: int,
                         counts: dict[int, int], stack: list,
                         max_club: int) -> tuple[bool, int]:
                if row_dict["price_tenths"] > budget - reserve:
                    return False, budget
                tc = row_dict["team_code"]
                if counts.get(tc, 0) >= max_club:
                    return False, budget
                counts[tc] = counts.get(tc, 0) + 1
                stack.append(pl.DataFrame([row_dict]))
                return True, budget - row_dict["price_tenths"]

            for pos in fill_order:
                remaining.remove(pos)
                need = SQUAD_COUNTS[pos]
                frame = by_pos[pos]
                n_candidates = min(need + 8, frame.height)
                candidate_idx = list(range(n_candidates))
                rng.shuffle(candidate_idx)
                exclude = candidate_idx[: rng.randint(0, 2)]
                keep = [i for i in range(n_candidates) if i not in exclude]
                keep.sort(key=lambda i: -frame[i, "expected_total"])
                taken = 0
                reserve = reserve_for_remaining(remaining, pos)
                for cand in (keep +
                             sorted((i for i in range(frame.height)
                                     if i not in keep),
                                    key=lambda i: -frame[i, "expected_total"])):
                    if taken >= need:
                        break
                    row_dict = dict(zip(pool.columns, frame.row(cand), strict=False))
                    ok, budget_left = fill_one(
                        pos, row_dict, budget_left, reserve,
                        club_counts, picks, max_per_club)
                    if ok:
                        taken += 1
                if taken < need:
                    picks.clear()
                    break
            if picks and len(picks) == 15:
                squad = (pl.concat(picks)
                         .with_columns(pl.lit(team_id).alias("team_id"))
                         .select("team_id", "position", "player_code",
                                 "expected_total", "now_cost", "price_tenths"))
                teams.append(squad)
                break
        # if all attempts failed, skip this team (feasibility genuinely absent)

    if not teams:
        raise ValueError(
            "could not fill any squad under budget/club limits — enlarge pool/budget")
    return pl.concat(teams)


def enumerate_valid_teams(pool: pl.DataFrame, **kw) -> pl.DataFrame:
    """Alias for greedy_teams until an exhaustive/sampled method is added."""
    return greedy_teams(pool, **kw)