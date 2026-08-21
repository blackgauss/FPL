"""Domain model: the vocabulary for players and teams.

These are the go-to types whenever code needs to say "a player", "a team", or
"a position". Bulk compute layers (score/filter/enumerate/simulate/model)
work on frames for performance and converge onto these types at the
boundaries — harness hydration (SearchResult.squads), live flags (flag_squad),
scripts, and the weekly optimizers.

Immutability contract (the important part):

  * These are pure value objects — never mutated in place. Change is a NEW
    value (dataclasses.replace, with_form(), or the frame->domain builders).
  * Identity is separated from form as two distinct types, so it is explicit
    what endures and what is refreshed each week:

  PlayerIdentity  immutable FOREVER: code, name, position. Two rows with the
                  same identity are the same player; it can be shared across
                  weeks/squads unchanged.
  PlayerForm      one week's volatile State: club, cost_tenths (transfers and
                  price move every GW). Replaced weekly, never edited.
  Player          a weekly row = identity + form, composed and frozen.
                  `with_form()` is the way form changes (same identity).
  Squad           one gameweek's snapshot (15 Players + XI/captain/vice/gw/
                  transfers made). Everything in it is state; the "team"
                  persists across weeks only as the carried player identities.

Typing note: all three are `@dataclass(frozen=True, slots=True)` — every
attribute is effectively final (no in-place writes), slots close the shape
(no ad-hoc attributes), and all the modern stdlib (StrEnum, dataclasses) is
used rather than inventing machinery.

Public/private surface: the public API is exactly the types (PlayerIdentity,
PlayerForm, Player, Squad, Position, POSITION_ORDER, position_sort_key) plus
the two frame->domain builders (players_from_frame, squad_from_frame).
Everything that bridges to frame column names or dataset layout is private
(underscore-prefixed) so callers depend on the domain, not on the data
layer's spelling.

Typing rule: we don't invent types when a standard type does the job. Player
/ club / gameweek ids are plain `int`s (no aliased NewTypes); prices are
always tenths (£0.1m) by fpl.units convention. `Position` is the one real
enum — closed four-value vocabulary with ordering behaviour (and a
str-subclass, so it stays interoperable with every string column).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import polars as pl

from fpl.units import DEFAULT_BUDGET_TENTHS, MAX_PER_CLUB, SQUAD_COUNTS

# Column names the builders below consume — the data/live layer contract.
_FRAME_PLAYER_COLUMNS = ("player_code", "web_name", "position", "team_code",
                         "price_tenths")


class Position(StrEnum):
    """The four FPL positions — our vocabulary for player/team structure.

    A str-subclass so every existing comparison and polars column (which stores
    the plain strings "GKP"|"DEF"|"MID"|"FWD") keeps working unchanged: no
    conversion cost anywhere.
    """

    GKP = "GKP"
    DEF = "DEF"
    MID = "MID"
    FWD = "FWD"


# One canonical sort order for positions (display + XI structure). Scripts used
# to each declare their own ordering; this is the single home.
POSITION_ORDER: tuple[Position, ...] = (
    Position.GKP, Position.DEF, Position.MID, Position.FWD,
)


def position_sort_key(position: str) -> int:
    """Sort/rank key for one position (canonical GKP < DEF < MID < FWD)."""
    return POSITION_ORDER.index(Position(position))


@dataclass(frozen=True, slots=True)
class PlayerIdentity:
    """A player's permanent identity — never changes across weeks or forms.

    `code` is the stable FPL player_code; `name` is the display name (a rename
    is a NEW identity); `position` is the player's role for this snapshot.
    Two rows sharing an identity are the same player, regardless of form.
    """

    code: int
    name: str
    position: Position


@dataclass(frozen=True, slots=True)
class PlayerForm:
    """A player's volatile weekly state — the part that changes every GW.

    `club` (team_code) changes on transfer, `cost_tenths` changes on price
    moves. A form is replaced (new value), never edited.
    """

    club: int
    cost_tenths: int


@dataclass(frozen=True, slots=True)
class Player:
    """A weekly player row: permanent identity + this week's form, composed.

    Forwarding properties (`code`/`name`/`position` from identity, `club`/
    `cost_tenths` from form) keep everyday call sites terse while the two
    facets stay first-class — e.g. `squad.players[i].identity` is the same
    object every week, only `.form` is replaced.
    """

    identity: PlayerIdentity
    form: PlayerForm

    @property
    def code(self) -> int:
        return self.identity.code

    @property
    def name(self) -> str:
        return self.identity.name

    @property
    def position(self) -> Position:
        return self.identity.position

    @property
    def club(self) -> int:
        return self.form.club

    @property
    def cost_tenths(self) -> int:
        return self.form.cost_tenths

    def with_form(self, *, club: int | None = None,
                  cost_tenths: int | None = None) -> Player:
        """New Player: the SAME identity, replaced weekly form. Never mutates.

        This (and dataclasses.replace) is the only way a player's state moves
        from one week to the next.
        """
        return Player(
            identity=self.identity,
            form=PlayerForm(club=self.club if club is None else club,
                            cost_tenths=self.cost_tenths
                            if cost_tenths is None else cost_tenths),
        )

    def _frame_row(self) -> dict:
        """Boundary adapter back to pool-frame columns. Private on purpose:
        entity code should not depend on the data layer's column spellings."""
        return {"player_code": self.code, "web_name": self.name,
                "position": str(self.position), "team_code": self.club,
                "price_tenths": self.cost_tenths}


@dataclass(frozen=True)
class Squad:
    """A team snapshot for one gameweek: 15 players + its weekly configuration.

    Everything here is state for a specific `gw` — weekly decisions (transfers,
    captain) return a NEW Squad, never mutate this one. The team persists
    across weeks only as the player codes carried forward.

    Invariants (validated by `squad_from_frame` / checked by `validate`):
      - exactly 15 players; position counts == SQUAD_COUNTS;
      - total cost <= budget; <= MAX_PER_CLUB per club;
      - 11 starters (1 GKP, >=3 DEF, >=1 FWD at all times);
      - captain/vice are starters, and a different club.
    """

    players: tuple[Player, ...]
    gw: int = 1
    starters: tuple[int, ...] = ()      # XI (player codes)
    bench: tuple[int, ...] = ()         # substitution priority order
    captain: int | None = None
    vice_captain: int | None = None
    transfers_in: tuple[int, ...] = ()  # codes added this week (log)

    def by_code(self) -> dict[int, Player]:
        return {p.code: p for p in self.players}

    def codes(self) -> tuple[int, ...]:
        """The squad's player codes — the unit the value/sim layers key on."""
        return tuple(p.code for p in self.players)

    def cost_tenths(self) -> int:
        return sum(p.cost_tenths for p in self.players)

    def budget_remaining(self) -> int:
        return DEFAULT_BUDGET_TENTHS - self.cost_tenths()

    def position_counts(self) -> dict[Position, int]:
        counts: dict[Position, int] = dict.fromkeys(POSITION_ORDER, 0)
        for p in self.players:
            counts[p.position] += 1
        return counts

    def validate(self) -> list[str]:
        """Return a list of rule violations (empty == valid Squad)."""
        problems: list[str] = []
        if len(self.players) != 15:
            problems.append(f"must have 15 players, got {len(self.players)}")
        counts = self.position_counts()
        for pos, need in SQUAD_COUNTS.items():
            if counts[pos] != need:
                problems.append(f"{pos} count {counts[pos]} != {need}")
        if self.cost_tenths() > DEFAULT_BUDGET_TENTHS:
            problems.append(f"cost {self.cost_tenths()} > budget")
        clubs: dict[int, int] = {}
        for p in self.players:
            clubs[p.club] = clubs.get(p.club, 0) + 1
        for club, n in clubs.items():
            if n > MAX_PER_CLUB:
                problems.append(f"club {club} has {n} > {MAX_PER_CLUB}")
        starts = self.starters or [p.code for p in self.players[:11]]
        pos_starts = {"GKP": 0, "DEF": 0, "FWD": 0}
        for code in starts:
            p = self.by_code()[code]
            if p.position in pos_starts:
                pos_starts[p.position] += 1
        if pos_starts["GKP"] < 1:
            problems.append("starting XI must include a goalkeeper")
        if pos_starts["DEF"] < 3:
            problems.append("starting XI must include >=3 defenders")
        if pos_starts["FWD"] < 1:
            problems.append("starting XI must include >=1 forward")
        if len(starts) != 11:
            problems.append(f"must have 11 starters, got {len(starts)}")
        if self.captain is not None and self.captain not in starts:
            problems.append("captain must be a starter")
        if self.vice_captain is not None and self.vice_captain not in starts:
            problems.append("vice-captain must be a starter")
        if (self.captain is not None and self.vice_captain is not None
                and self.by_code()[self.captain].club
                == self.by_code()[self.vice_captain].club):
            problems.append("captain and vice-captain should be different clubs")
        return problems


def players_from_frame(frame: pl.DataFrame) -> list[Player]:
    """Build Player objects from a frame with the _FRAME_PLAYER_COLUMNS
    (player_code/web_name/position/team_code/price_tenths)."""
    players: list[Player] = []
    for row in frame.iter_rows(named=True):
        players.append(Player(
            identity=PlayerIdentity(
                code=int(row["player_code"]),
                name=row.get("web_name") or f"p{row['player_code']}",
                position=Position(row["position"])),
            form=PlayerForm(
                club=int(row["team_code"]),
                cost_tenths=int(row.get("price_tenths", 0))),
        ))
    return players


def squad_from_frame(frame: pl.DataFrame, *, gw: int = 1) -> Squad:
    """Build a valid Squad from a 15-row basket frame (greedy output)."""
    players = players_from_frame(frame)
    # legal XI: 1 GKP, 4 DEF, 5 MID, 1 FWD (len 11, all four positions)
    by_pos: dict[str, list[Player]] = {"GKP": [], "DEF": [], "MID": [], "FWD": []}
    for p in players:
        by_pos[p.position].append(p)
    starts = [by_pos["GKP"][0].code]
    starts += [p.code for p in by_pos["DEF"][:4]]   # 4 DEF
    starts += [p.code for p in by_pos["MID"][:5]]   # 5 MID
    starts += [p.code for p in by_pos["FWD"][:1]]   # 1 FWD
    squad = Squad(players=tuple(players), gw=gw,
                  starters=tuple(starts))
    problems = squad.validate()
    if problems:
        raise ValueError("cannot construct valid Squad: " + "; ".join(problems))
    return squad