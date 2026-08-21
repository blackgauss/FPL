"""On-the-fly player filters built from live FPL API state.

The user's design: the live snapshot is kept SEPARATE from the dataset/feature
store; these filters apply it at query time. Each filter is a pure function
taking the live-state frame (from fpl.live.live.load_live_state) plus optional
context, and returning a boolean mask aligned to that frame — so callers can
join/apply it onto any player-level dataset without changing the stored data.

Semantics follow the FPL `status` enum:
    a = available      d = doubtful       i = injured
    s = suspended      u = unavailable    n = not in squad
"""

from __future__ import annotations

import polars as pl

from fpl.domain import Position, Squad

ELEMENT_TYPE_TO_POSITION = {
    1: Position.GKP,
    2: Position.DEF,
    3: Position.MID,
    4: Position.FWD,
}


def position_for_element_type(element_type: int) -> Position:
    """Map the FPL API element_type (1..4) to the domain Position.

    Single canonical projection for live -> domain; callers never build their
    own id map. Raises KeyError for out-of-range element types.
    """
    try:
        return ELEMENT_TYPE_TO_POSITION[int(element_type)]
    except (KeyError, TypeError, ValueError) as exc:
        raise KeyError(f"unknown element_type {element_type!r}") from exc


def status_mask(live: pl.DataFrame, statuses: list[str]) -> pl.Series:
    """True where the player's live status is in `statuses`."""
    return pl.Series(
        [s in statuses for s in live.get_column("status")],
        dtype=pl.Boolean,
    )


def available(live: pl.DataFrame) -> pl.Series:
    """Players fully expected to play: status 'a'. Ignores chance-heuristics
    for week-to-week availability (minute risk); see chance_of_playing next."""
    return status_mask(live, ["a"])


def not_injured_suspended(live: pl.DataFrame) -> pl.Series:
    """Not currently injured, suspended, or unavailable."""
    return status_mask(live, ["a", "d"])


def chance_of_playing(live: pl.DataFrame, *, min_pct: float = 75.0) -> pl.Series:
    """At least `min_pct` chance next round; null (unknown) -> retained.

    FPL leaves chance_of_playing_* null for most players even when available,
    so a null signals 'no flagged doubt'. Doubtful ('d') are kept too — with a
    numeric chance below the bar, the mask is False; with null they pass.
    """
    chance = live.get_column("chance_of_playing_next_round").cast(pl.Float64)
    status = live.get_column("status")
    return pl.Series(
        [
            (s in ("a", "d")) if (c is None) else (float(c) >= min_pct)
            for s, c in zip(status, chance, strict=False)
        ],
        dtype=pl.Boolean,
    )


def no_news(live: pl.DataFrame) -> pl.Series:
    """No injury/availability news text (clearest 'fully available' signal)."""
    return pl.Series(
        [not isinstance(n, str) or n.strip() == "" for n in live.get_column("news")],
        dtype=pl.Boolean,
    )


def in_league(live: pl.DataFrame) -> pl.Series:
    """Still selectable/transactable (not removed, loaned, or de-registered).

    `removed=False` means actively in the league; `can_select=True` means
    priced/selectable. A player is in-league iff not removed and selectable.
    """
    return pl.Series(
        [
            bool(r) is False and bool(c)
            for r, c in zip(live.get_column("removed"),
                            live.get_column("can_select"), strict=False)
        ],
        dtype=pl.Boolean,
    )


def not_transferred(live: pl.DataFrame, team_code: pl.Series) -> pl.Series:
    """Player still at the club the dataset assumes (`team_code` aligned to
    `live` rows — i.e. pass the dataset's per-player current club beside live's
    live.team_code). True where live team_code matches the expected club."""
    return pl.Series(
        [a == b for a, b in zip(team_code, live.get_column("team_code"), strict=False)],
        dtype=pl.Boolean,
    )


def price_unchanged(live: pl.DataFrame, expected_now_cost: pl.Series) -> pl.Series:
    """Price hasn't moved from the value the model/dataset used. expected costs
    are in the same units as live (tenths)."""
    return pl.Series(
        [a == b for a, b in zip(expected_now_cost,
                                live.get_column("now_cost"), strict=False)],
        dtype=pl.Boolean,
    )


def suggest(live: pl.DataFrame, *, min_chance_pct: float = 75.0) -> pl.Series:
    """Composite suggested filter: actually in the league, not injured/
    suspended/unavailable, and meeting the chance-of-playing bar. A one-call
    sensible default for team selection."""
    return (
        in_league(live)
        & not_injured_suspended(live)
        & chance_of_playing(live, min_pct=min_chance_pct)
    ).alias("suggest_playable")


def flag_squad_player(row: dict) -> str:
    """One player's live-health problems as a human string, from a joined row
    carrying live status/team/price vs dataset team/price.

    Detects: injured/suspended/unavailable status, absence from the live
    roster (missing/transferred out of the FPL game), a club transfer, and a
    price move. Returns 'ok' when none apply. Always comparable.
    """
    status = row.get("status")
    problems = []
    if status in ("i", "s", "u", "n"):
        problems.append(f"UNAVAILABLE[{status}]")
    if status == "i":
        problems.append("INJURED")
    live_team = row.get("team_code_live")
    ds_team = row.get("team_code")
    if status is None and live_team is None:
        # dataset player not present in the live roster at all
        problems.append("NOT IN LIVE ROSTER (missing/transferred out)")
    elif ds_team is not None and live_team is not None and ds_team != live_team:
        problems.append(f"TRANSFERRED (ds {ds_team} -> live {live_team})")
    pd = row.get("price_diff_tenths")
    if pd is not None and abs(pd) > 0:
        problems.append(f"price {pd:+.0f}")
    return " | ".join(problems) if problems else "ok"


def flag_squad(squad: Squad, live: pl.DataFrame) -> dict[int, str]:
    """Per-player live problems for a whole Squad, keyed by player_code.

    The row-level `flag_squad_player` forces the caller to hand-join live/
    dataset columns into a dict row. `flag_squad` builds those rows from the
    typed Squad (Player carries club + tenths price) so nothing downstream
    recalls live column names or converts price units. 'ok' when none apply.
    """
    live_idx = {
        code: (status, now_cost, team)
        for code, status, now_cost, team in live.select(
            "player_code", "status", "now_cost", "team_code").iter_rows()
    }
    out: dict[int, str] = {}
    for p in squad.players:
        rec = live_idx.get(p.code)
        if rec is None:
            out[p.code] = "NOT IN LIVE ROSTER (missing/transferred out)"
            continue
        status, live_now_cost, live_team = rec
        out[p.code] = flag_squad_player({
            "status": status,
            "team_code": p.club,
            "team_code_live": live_team,
            "price_diff_tenths": (int(live_now_cost) - p.cost_tenths)
            if live_now_cost is not None else None,
        })
    return out


def apply_filters(live: pl.DataFrame, **filters: pl.Series) -> pl.DataFrame:
    """Join any mask Series (aligned to `live`) back into a live frame, useful
    for reporting which players drop for which reason."""
    out = live
    for name, mask in filters.items():
        out = out.with_columns(mask.alias(name))
    return out


def filter_frame_by_code(
    frame: pl.DataFrame, live: pl.DataFrame, mask: pl.Series,
) -> pl.DataFrame:
    """Drop rows of `frame` (keyed by `player_code`) whose player is excluded
    by `mask` (a boolean Series aligned to `live`, e.g. from `suggest`).

    This is the on-the-fly filter: apply the live-state mask onto any player
    dataset without baking the state into the data. Keys must cover the same
    player_codes; players absent from live are kept (unknown != excluded).
    """
    excluded = live.filter(~mask).get_column("player_code")
    if excluded.len() == 0:
        return frame
    return frame.filter(~pl.col("player_code").is_in(excluded.implode()))