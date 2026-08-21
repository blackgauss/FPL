"""Expression layer: write programs in the shape of their type signature.

The domain model (fpl.domain) is vocabulary; this module is how programs are
WRITTEN with it — as a forward pipe of typed stages:

    program: Data -> [Player] -> [Squad] -> [Score]

Each stage is a pure function; `pipe` threads the value left-to-right so the
code reads exactly like the signature. The `[]` collections are plain Python
lists/tuples (no new container types); the domain names the elements. Heavy
lifting inside a stage is always lowered back onto the raw store (filters /
joins / group-bys keyed by player_code) — never a per-object loop.

Recipe (executed by the test suite):

    >>> import polars as pl
    >>> from functools import partial
    >>> from fpl.domain import players_from_frame, squad_from_frame
    >>> from fpl.express import pipe

    Build the raw store (two valid teams' worth of player rows):

    >>> store = pl.DataFrame({
    ...     "player_code": [100 + i for i in range(15)] + [200 + i for i in range(15)],
    ...     "web_name": [f"P{i}" for i in range(30)],
    ...     "position": (["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3) * 2,
    ...     "team_code": [i % 12 + 1 for i in range(30)],
    ...     "price_tenths": [50 + (i % 3) * 5 for i in range(30)],
    ... })

    The forecast table (model output, keyed by player_code):

    >>> forecast = pl.DataFrame({
    ...     "player_code": store["player_code"],
    ...     "expected_points": [1.0 + (c % 5) for c in store["player_code"]],
    ... })

    Stage definitions — each a small, domain-reading program:

    >>> def to_squads(players):                       # [Player] -> [Squad]
    ...     groups = [players[:15], players[15:]]
    ...     return [squad_from_frame(pl.DataFrame([p._frame_row() for p in g]), gw=1)
    ...             for g in groups]
    >>> def score_squads(squads, forecast):           # [Squad] -> [Score]
    ...     long = pl.DataFrame([{"squad_id": i, "player_code": c}
    ...                          for i, s in enumerate(squads) for c in s.codes()])
    ...     return (long.join(forecast, on="player_code")
    ...             .group_by("squad_id")
    ...             .agg(pl.col("expected_points").sum().alias("gw_total")))

    The whole program, written as its signature
    (Data -> [Player] -> [Squad] -> [Score]):

    >>> scores = pipe(
    ...     store,                                    # Data
    ...     players_from_frame,                       # -> [Player]
    ...     to_squads,                                # -> [Squad]
    ...     partial(score_squads, forecast=forecast), # -> [Score]
    ... )
    >>> scores.height
    2
    >>> scores["gw_total"].sum() > 0
    True
"""

from __future__ import annotations

from collections.abc import Callable


def pipe(value, *steps: Callable):
    """Forward-pipe: value -> step1 -> step2 -> ... Each step is a pure
    function of its predecessor's output; the read matches the type
    signature of the program it expresses."""
    for step in steps:
        value = step(value)
    return value