"""Stage 2: filter the scored player pool to a diverse, tractable subset.

The team search enumerates over this subset, so it must be big enough to be
interesting but bounded enough to be enumerable — and it must keep position,
team, and price diversity so the search isn't trivially one cluster.

Two steps:
1. Drop players "expected to never feature" (no minutes in the window).
2. Rank by expected total; keep top-K per position AND per team (club cap),
   joined with the player's price so cheap value-enablers survive.
"""

from __future__ import annotations

import polars as pl


def availability_from_gw_stats(
    gw_stats: pl.DataFrame,
    players: pl.DataFrame,
    *,
    gw_start: int,
    gw_end: int,
) -> pl.DataFrame:
    """Per-player minutes + price over [gw_start, gw_end].

    'Expected to feature' for a historical window = played at least one minute.
    Price is the player's latest now_cost in the window.
    Returns (player_code, minutes_in_window, now_cost).
    """
    rows = gw_stats.filter(pl.col("gw").is_between(gw_start, gw_end))
    if rows.is_empty():
        raise ValueError(f"no gw_stats rows for GW {gw_start}..{gw_end}")
    agg = rows.join(
        players.select("player_id", "player_code"),
        on="player_id",
        how="inner",
    ).group_by("player_code").agg(
        pl.col("minutes").sum().alias("minutes_in_window"),
        pl.col("now_cost").last().alias("now_cost"),
    )
    return agg


def drop_never_featuring(
    scored: pl.DataFrame,
    availability: pl.DataFrame,
    *,
    min_minutes: int = 0,
) -> pl.DataFrame:
    """Keep players who actually featured in the window."""
    return scored.join(
        availability.select("player_code", "minutes_in_window"),
        on="player_code",
        how="inner",
    ).filter(pl.col("minutes_in_window") > min_minutes)


def filter_pool(
    scored: pl.DataFrame,
    availability: pl.DataFrame,
    *,
    top_k_per_position: int = 30,
    max_per_team: int = 4,
    min_price: float = 0.0,
    max_price: float = 100.0,
) -> pl.DataFrame:
    """Diverse, tractable subset of the scored pool.

    Drops never-featuring players, then keeps the top `top_k_per_position` by
    expected_total per position, capped at `max_per_team` players per club.
    Price is joined from availability; bands are not enforced here — the cap
    per position+team keeps the pool varied, and price survives for budget
    constraints at enumeration time.
    """
    featuring = drop_never_featuring(scored, availability, min_minutes=0)
    pool = featuring.join(
        availability.select("player_code", "now_cost"),
        on="player_code",
        how="inner",
    ).filter(
        (pl.col("now_cost").is_not_null())
        & (pl.col("now_cost") >= min_price)
        & (pl.col("now_cost") <= max_price)
    )

    # position cap (top-K per position), then team cap (a shared club can
    # dominate the pool otherwise)
    ranked = pool.sort("expected_total", descending=True)
    with_pos_rank = ranked.with_columns(
        pl.col("expected_total").rank(descending=True).over("position").alias("__pos_rank")
    )
    position_capped = with_pos_rank.filter(
        pl.col("__pos_rank") <= top_k_per_position
    ).drop("__pos_rank")
    with_team_rank = position_capped.with_columns(
        pl.col("expected_total").rank(descending=True).over("team_code").alias("__team_rank")
    )
    team_capped = with_team_rank.filter(
        pl.col("__team_rank") <= max_per_team
    ).drop("__team_rank")
    return team_capped.sort("position", "expected_total", descending=[False, True])