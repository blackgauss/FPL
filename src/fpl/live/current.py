"""Reconcile the dataset with live FPL state BEFORE team construction.

The dataset stores each player's club/availability as of its snapshot. Before
we build a team we must make the input reflect the *current* world: a player
who transferred clubs moves in the pool (so `max_per_club` uses their current
club), a player no longer in the live roster (transferred abroad, de-listed,
retired) drops out, and injured/suspended players are excluded — all before
scoring/filtering/enumeration runs.

Two modes use the same rules:
  * frame mode (construction_input) — adjusts a scored *pool* before
    filtering/enumeration, the integer-code layer.
  * typed mode (players_to_replace) — speaks in Squad/Player (fpl.domain):
    which players a candidate team must swap, with reasons. This is the
    action-side of fpl.live.filters.flag_squad (report) and the input the
    weekly transfer step consumes.
"""

from __future__ import annotations

import polars as pl

from fpl.domain import Squad
from fpl.live.filters import suggest


def reconcile_player_clubs(
    frame: pl.DataFrame, live: pl.DataFrame, team_col: str = "team_code"
) -> pl.DataFrame:
    """Overwrite each player's club with their current (live) club; drop
    players missing from the live roster.

    `frame` is any player-keyed frame used downstream (scored pool, players
    table). Players present in live get their `team_code` overwritten with the
    live value (transfer updates — enumeration then respects the current club
    for `max_per_club`); players absent from live are removed (the "missing
    player" case makes them un-pickable).
    """
    live_codes = live.select(
        "player_code", pl.col("team_code").alias("_live_team")
    )
    merged = frame.join(live_codes, on="player_code", how="left")
    merged = merged.filter(pl.col("_live_team").is_not_null())
    return merged.with_columns(pl.col("_live_team").alias(team_col)).drop("_live_team")


def reconcile_availability(
    frame: pl.DataFrame,
    live: pl.DataFrame,
    live_mask: pl.Series,
) -> pl.DataFrame:
    """Drop players `live_mask` excludes (e.g. suggest()) from a player-keyed
    frame. Players absent from live are kept (unknown != excluded)."""
    excluded = live.filter(~live_mask).get_column("player_code")
    if excluded.len() == 0:
        return frame
    return frame.filter(~pl.col("player_code").is_in(excluded.implode()))


def construction_input(
    scored: pl.DataFrame,
    live: pl.DataFrame,
    live_mask: pl.Series,
) -> pl.DataFrame:
    """The scored pool adjusted for the CURRENT world: clubs updated for
    transfers, missing players dropped, injured/suspended/unavailable excluded.

    Call this right before filter_pool so the whole construction pipeline
    (team cap, reserve, enumeration) sees live-consistent clubs.
    """
    out = reconcile_player_clubs(scored, live)
    out = reconcile_availability(out, live, live_mask)
    return out


def players_to_replace(
    squad: Squad,
    live: pl.DataFrame,
    mask: pl.Series | None = None,
) -> dict[int, str]:
    """Which Squad players must be swapped before the next GW, code -> reason.

    The typed action-side of the reconcile rules: a player goes when they are
    absent from the live roster (missing/transferred out of FPL) or excluded
    by the availability mask (default suggest(): injured/suspended/unavailable,
    below the chance-of-playing bar, or not selectable). Players who are
    present and playable are absent from the returned dict.

    `mask` defaults to fpl.live.filters.suggest(live) — the same rule
    construction_input applies to the pool, expressed on a Squad so the weekly
    transfer optimizer knows exactly who to replace and why.
    """
    mask = mask if mask is not None else suggest(live)
    live_status = {
        code: status
        for code, status in live.select("player_code", "status").iter_rows()
    }
    playable = set(live.filter(mask).get_column("player_code").to_list())
    reasons: dict[int, str] = {}
    for p in squad.players:
        if p.code not in live_status:
            reasons[p.code] = "NOT IN LIVE ROSTER (missing/transferred out)"
        elif p.code not in playable:
            status = live_status[p.code]
            if status in ("i", "s", "u", "n"):
                reasons[p.code] = f"UNAVAILABLE[{status}]"
            else:
                reasons[p.code] = "below availability bar"
    return reasons