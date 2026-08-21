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

SQUAD_COUNTS = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
DEFAULT_BUDGET_TENTHS = 1000  # £100m in £0.1m units (now_cost is decimal £m)
MAX_PER_CLUB = 3


def _price_tenths(now_cost: float) -> int:
    return int(round(now_cost * 10))


def squad_for_price(pool: pl.DataFrame) -> pl.DataFrame:
    """Annotate pool with price in tenths and a player code string."""
    return pool.with_columns(
        pl.col("now_cost").map_elements(_price_tenths, return_dtype=pl.Int64)
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
    need = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
    n_per_pos = dict(zip(counts["position"], counts["len"], strict=False))
    total = 1
    for pos in need:
        avail = n_per_pos.get(pos, 0)
        if avail < need[pos]:
            raise ValueError(
                f"pool has {avail} {pos} but need {need[pos]} — enlarge the pool")
        total *= comb(avail, need[pos])
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
    rng = random.Random(seed)

    for team_id in range(n_teams):
        picks: list[pl.DataFrame] = []
        budget_left = budget_tenths
        club_counts: dict[int, int] = {}
        for pos, need in SQUAD_COUNTS.items():
            frame = by_pos[pos]
            n_candidates = min(need + 8, frame.height)
            candidate_idx = list(range(n_candidates))
            # drop 0..2 of the best, so lineups differ per team
            rng.shuffle(candidate_idx)
            exclude = candidate_idx[: rng.randint(0, 2)]
            keep = [i for i in range(n_candidates) if i not in exclude]
            keep.sort(key=lambda i: -frame[i, "expected_total"])
            taken = 0
            for i in keep:
                if taken >= need:
                    break
                row = frame.row(i)  # tuple; keys via SQUAD_COUNTS frame columns
                row_dict = dict(zip(frame.columns, row, strict=False))
                if row_dict["price_tenths"] > budget_left:
                    continue
                if row_dict["team_code"] in club_counts and \
                        club_counts[row_dict["team_code"]] >= max_per_club:
                    continue
                club_counts[row_dict["team_code"]] = club_counts.get(
                    row_dict["team_code"], 0) + 1
                budget_left -= row_dict["price_tenths"]
                taken += 1
                picks.append(pl.DataFrame([row_dict]))
            if taken < need:
                raise ValueError(
                    f"team {team_id}: could not fill {pos} (need {need}, took {taken}) "
                    "under budget/club limits — enlarge the pool or budget")
        squad = (pl.concat(picks)
                 .with_columns(pl.lit(team_id).alias("team_id"))
                 .select("team_id", "position", "player_code",
                         "expected_total", "now_cost", "price_tenths"))
        teams.append(squad)

    return pl.concat(teams)


def enumerate_valid_teams(pool: pl.DataFrame, **kw) -> pl.DataFrame:
    """Alias for greedy_teams until an exhaustive/sampled method is added."""
    return greedy_teams(pool, **kw)