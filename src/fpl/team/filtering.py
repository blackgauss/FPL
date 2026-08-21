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
    reserve_top: int = 20,
    min_price: float = 0.0,
    max_price: float = 100.0,
) -> pl.DataFrame:
    """Diverse, tractable subset of the scored pool.

    Drops never-featuring players, then keeps:
    - the `reserve_top` players at the top of the overall expected_total
      ranking regardless of club — so genuine stars (Haaland grade) are not
      culled just because a cheaper teammate over-rates above them;
    - among the rest, the top `top_k_per_position` per position, capped at
      `max_per_team` per club — for breadth beyond the stars.

    Without the reserve, `max_per_team`: Man City's four cheap depth picks
    (Guéhi/Donnarumma/Cherki/Semenyo) crowded out Haaland/Salah-like stars
    before enumeration ever saw them.
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

    ranked = pool.sort("expected_total", descending=True)

    reserved = ranked.head(reserve_top)
    rest = ranked.tail(ranked.height - reserve_top) if ranked.height > reserve_top \
        else ranked.clear()

    # position cap on the remainder (stars already saved)
    if rest.height:
        with_pos_rank = rest.with_columns(
            pl.col("expected_total").rank(descending=True).over("position")
            .alias("__pos_rank")
        )
        rest = with_pos_rank.filter(
            pl.col("__pos_rank") <= top_k_per_position
        ).drop("__pos_rank")
        with_team_rank = rest.with_columns(
            pl.col("expected_total").rank(descending=True).over("team_code")
            .alias("__team_rank")
        )
        rest = with_team_rank.filter(
            pl.col("__team_rank") <= max_per_team
        ).drop("__team_rank")

    combined = pl.concat([reserved, rest]) if rest.height else reserved
    return combined.sort("position", "expected_total", descending=[False, True])