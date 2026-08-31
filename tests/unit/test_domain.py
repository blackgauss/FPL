"""Black-box tests: the domain model (Player/Squad).

These pin the interface optimizers (captain, transfer) will be built against:
construction from a basket frame, invariants, the validate() reporting, and
the immutability contract (value objects, never mutated). All fixtures are
synthetic — a valid 15-player club with distinct codes so tests run anywhere,
never touching external/.
"""

import polars as pl
import pytest

from fpl.domain import (
    POSITION_ORDER,
    Player,
    PlayerIdentity,
    PlayerState,
    Position,
    Squad,
    players_from_frame,
    players_to_frame,
    position_sort_key,
    squad_from_frame,
)
from fpl.units import DEFAULT_BUDGET_TENTHS, SQUAD_COUNTS

# A valid 15-player club: 2/5/5/3 by position, 11 distinct clubs (2×3, 9×1),
# prices summing to a plausible budget at or below £100m.
_NAMES = {
    1: "G1", 2: "G2",
    10: "D1", 11: "D2", 12: "D3", 13: "D4", 14: "D5",
    20: "M1", 21: "M2", 22: "M3", 23: "M4", 24: "M5",
    30: "F1", 31: "F2", 32: "F3",
}


def valid_frame() -> pl.DataFrame:
    rows: list[dict] = []
    expensive = DEFAULT_BUDGET_TENTHS - 14 * 5  # 1000 - 70 = 930, one anchor price
    pos_pool: list[tuple[str, int]] = []
    for pos, need in SQUAD_COUNTS.items():
        for _ in range(need):
            pos_pool.append((pos, len(pos_pool)))
    for idx, (pos, _) in enumerate(pos_pool):
        code = [1, 2, 10, 11, 12, 13, 14, 20, 21, 22, 23, 24, 30, 31, 32][idx]
        club = {1: 3, 2: 4, 10: 1, 11: 1, 12: 1, 13: 2, 14: 2,
                20: 5, 21: 6, 22: 7, 23: 8, 24: 9, 30: 10, 31: 11, 32: 12}[code]
        rows.append({
            "player_code": code,
            "web_name": _NAMES[code],
            "position": pos,
            "team_code": club,
            "price_tenths": expensive if idx == 0 else 5,
        })
    return pl.DataFrame(rows)


def make_squad(**overrides) -> Squad:
    return squad_from_frame(valid_frame(), **overrides)


class TestPlayer:
    @staticmethod
    def _p(club=1, cost=55, code=10, name="D1", position="DEF"):
        return Player(identity=PlayerIdentity(code, name, Position(position)),
                      state=PlayerState(club, cost))

    def test_fields_and_forwarding(self):
        p = self._p()
        assert p.code == 10 and p.position == "DEF"
        assert p.club == 1 and p.cost_tenths == 55
        # forwarding reads live facets: identity and state are independent
        assert (p.code, p.name, p.position) == (
            p.identity.code, p.identity.name, p.identity.position)
        assert (p.club, p.cost_tenths) == (p.state.club, p.state.cost_tenths)
        assert isinstance(p.code, int) and isinstance(p.club, int)  # plain ints

    def test_hashable(self):
        a = self._p()
        b = self._p()
        assert {a: 1}[b] == 1  # stable identity, usable as map key

    def test_immutable_value_object(self):
        # the immutability contract: no stored field is writable even on the
        # inner facets; change is a NEW value. (Assigning to the read-only
        # forwarding properties also fails via slots; we assert the stored
        # fields so the failure reason stays a clean AttributeError.)
        p = self._p()
        with pytest.raises(AttributeError):
            p.state.club = 2  # type: ignore[misc]
        with pytest.raises(AttributeError):
            p.identity.code = 99  # type: ignore[misc]
        assert p.club == 1 and p.code == 10

    def test_identity_obj_is_shared_across_weeks(self):
        # the point of the split: across weeks/state the identity is the SAME
        # object; only the state is replaced.
        p = self._p(club=1, cost=55)
        q = p.with_state(club=13, cost_tenths=100)  # transferred + price rise
        assert q.identity is p.identity
        assert q.code == p.code and q.position == p.position
        assert q.club == 13 and q.cost_tenths == 100
        assert p.club == 1 and p.cost_tenths == 55  # original untouched

    def test_with_state_is_the_only_transition(self):
        p = self._p(club=1, cost=55)
        same = p.with_state()
        assert same == p  # no-op transition is an equal value
        moved = p.with_state(club=5)
        assert moved == self._p(club=5, cost=55)

    def test_frame_row_roundtrip(self):
        # the private boundary adapter re-produces what the builders consume
        from fpl.domain import players_from_frame

        p = self._p()
        rebuilt = players_from_frame(pl.DataFrame([p._frame_row()]))[0]
        assert rebuilt == p


class TestPositionDomain:
    def test_members_and_str_compat(self):
        assert list(Position) == [Position.GKP, Position.DEF, Position.MID,
                                   Position.FWD]
        assert Position.DEF == "DEF"  # StrEnum: works with plain strings everywhere
        assert hash(Position.DEF) == hash("DEF")

    def test_order_covers_all_positions_for_sorting(self):
        assert position_sort_key("GKP") < position_sort_key("DEF") < \
            position_sort_key("MID") < position_sort_key("FWD")
        assert len(POSITION_ORDER) == 4

    def test_hydration_produces_typed_positions(self):
        assert all(isinstance(p.position, Position) for p in make_squad().players)


class TestSquadFromFrame:
    def test_builds_valid_squad(self):
        squad = make_squad(gw=1)
        assert squad.validate() == []
        assert len(squad.players) == 15
        assert squad.gw == 1

    def test_position_counts_match_constants(self):
        squad = make_squad()
        assert squad.position_counts() == SQUAD_COUNTS

    def test_starters_form_legal_xi(self):
        squad = make_squad()
        assert len(squad.starters) == 11
        starts_pos = [squad.by_code()[c].position for c in squad.starters]
        assert starts_pos.count("GKP") == 1
        assert starts_pos.count("DEF") == 4
        assert starts_pos.count("MID") == 5
        assert starts_pos.count("FWD") == 1

    def test_eleven_unique_codes(self):
        squad = make_squad()
        assert len(set(squad.starters)) == 11

    def test_rejects_missing_position_shape(self):
        # 14 players only -> clear ValueError naming the shape problem
        with pytest.raises(ValueError, match="15 players"):
            squad_from_frame(valid_frame().head(14))

    def test_rejects_non_squad_shape(self):
        # a position permanently short (e.g. 1 GKP) -> names missing positions
        import polars as pl

        frame = valid_frame().with_columns(
            pl.when(pl.col("player_code") == 2).then(pl.lit("DEF"))
            .otherwise(pl.col("position")).alias("position"))
        with pytest.raises(ValueError, match="GKP"):
            squad_from_frame(frame)

    def test_codes_and_budget_conveniences(self):
        squad = make_squad()
        assert len(squad.codes()) == 15
        assert squad.codes() == tuple(p.code for p in squad.players)
        assert squad.budget_remaining() == \
            DEFAULT_BUDGET_TENTHS - squad.cost_tenths()


class TestValidate:
    def test_wrong_player_count_reported(self):
        squad = make_squad()
        shorter = Squad(players=squad.players[:14], starters=squad.starters)
        problems = shorter.validate()
        assert any("15 players" in p for p in problems)

    def test_wrong_position_counts_reported(self):
        squad = make_squad()
        players = list(squad.players)
        extra_fwd = Player(identity=PlayerIdentity(99, "F9", Position.FWD),
                           state=PlayerState(13, 5))
        over = Squad(players=tuple([*players, extra_fwd]),
                     starters=squad.starters)
        problems = over.validate()
        assert any("FWD" in p and "!=" in p for p in problems)

    def test_over_budget_reported(self):
        squad = make_squad()
        players = [p.with_state(cost_tenths=p.cost_tenths * 10)
                   if i == 0 else p
                   for i, p in enumerate(squad.players)]
        over = Squad(players=tuple(players), starters=squad.starters)
        problems = over.validate()
        assert any("cost" in p and "budget" in p for p in problems)

    def test_over_club_cap_reported(self):
        squad = make_squad()
        # move one player into club 1 (already at MAX_PER_CLUB of 3)
        players = tuple(
            p.with_state(club=1)
            if p.club == 12 else p
            for p in squad.players)
        over = Squad(players=players, starters=squad.starters)
        problems = over.validate()
        assert any("club 1 has" in p for p in problems)

    def test_captain_must_be_starter(self):
        squad = make_squad()
        bench_code = [p for p in squad.by_code() if p not in squad.starters][0]
        bad = Squad(players=squad.players, starters=squad.starters,
                    captain=bench_code)
        assert any("captain must be a starter" in p for p in bad.validate())

    def test_vice_must_be_starter(self):
        squad = make_squad()
        bench_code = [p for p in squad.by_code() if p not in squad.starters][0]
        bad = Squad(players=squad.players, starters=squad.starters,
                    vice_captain=bench_code)
        assert any("vice-captain must be a starter" in p for p in bad.validate())

    def test_captain_and_vice_different_club(self):
        squad = make_squad()
        # force captain + vice into the same club by picking two starters
        # of the same club
        byc = squad.by_code()
        same_club = [c for c in squad.starters
                     if byc[c].club == byc[squad.starters[0]].club]
        if len(same_club) < 2:
            # not present: patch club of a second starter
            p0 = byc[squad.starters[0]]
            p1 = byc[squad.starters[1]]
            players = tuple(
                p.with_state(club=p0.club)
                if p.code == p1.code else p for p in squad.players)
            bad = Squad(players=players, starters=squad.starters,
                        captain=byc[squad.starters[0]].code,
                        vice_captain=p1.code)
        else:
            bad = Squad(players=squad.players, starters=squad.starters,
                        captain=same_club[0], vice_captain=same_club[1])
        assert any("different clubs" in p for p in bad.validate())

    def test_duplicate_player_codes_reported(self):
        squad = make_squad()
        dup = Squad(players=squad.players[:14] + (squad.players[0],),
                    starters=squad.starters)
        assert any("duplicate player code" in p for p in dup.validate())

    def test_unknown_starter_reported_not_crashed(self):
        squad = make_squad()
        bad = Squad(players=squad.players, starters=(999999,))
        problems = bad.validate()
        assert any("unknown players" in p for p in problems)

    def test_captain_not_in_squad_reported(self):
        squad = make_squad()
        bad = Squad(players=squad.players, starters=squad.starters,
                    captain=999999, vice_captain=999999)
        problems = bad.validate()
        assert any("captain not in squad" in p for p in problems)
        assert any("vice-captain not in squad" in p for p in problems)

    def test_gameweek_below_one_reported(self):
        squad = make_squad()
        bad = Squad(players=squad.players, starters=squad.starters, gw=0)
        assert any("gameweek must be >= 1" in p for p in bad.validate())


class TestLiveIntegrationShape:
    """Interface documentation-by-example: the columns optimizers depend on."""

    def test_players_to_frame_roundtrip(self):
        # the compatibility seam: domain -> canonical frame -> back, lossless
        squad = make_squad()
        frame = players_to_frame(list(squad.players))
        assert frame.height == 15
        assert set(frame.columns) >= {"player_code", "web_name", "position",
                                      "team_code", "price_tenths"}
        assert players_from_frame(frame) == list(squad.players)

    def test_execute_in_numpy_from_domain(self):
        # speak in the model, execute in numpy (JAX eats the same ndarray)
        import numpy as np

        squad = make_squad()
        frame = players_to_frame(list(squad.players))
        costs = frame["price_tenths"].to_numpy()
        positions = frame["position"].to_numpy()
        assert isinstance(costs, np.ndarray)
        assert int(costs.sum()) == squad.cost_tenths()  # arrays agree with the object
        assert int((positions == "GKP").sum()) == 2

    def test_domain_docstrings_execute(self):
        """The documented recipe is not aspirational — doctests run in-suite.

        This exercises the module-docstring recipe (express a squad GW total,
        captain, form transition — each lowered onto the raw store) and fails
        if the model's documented usage drifts from reality.
        """
        import doctest

        import fpl.domain

        results = doctest.testmod(fpl.domain)
        assert results.failed == 0, results

    def test_boundary_adapter_columns(self):
        squad = make_squad()
        p = squad.players[0]
        # the frame adapter emits exactly the canonical pool columns (the
        # _FRAME_PLAYER_COLUMNS contract), as a private boundary helper
        d = p._frame_row()
        assert set(d) >= {"player_code", "web_name", "position", "team_code",
                          "price_tenths"}

def test_validate_club_baseline_tolerates_observed_drift():
    """FPL caps clubs at selection time; five players sharing a club through
    post-window transfers is state, not a violation — but adding a fifth is."""
    club_of = {1: 1, 2: 2, 10: 3, 11: 3, 12: 4, 13: 5, 14: 6,
               20: 3, 21: 3, 22: 7, 23: 8, 24: 9, 30: 3, 31: 10, 32: 11}
    pos_of = ({c: "GKP" for c in (1, 2)} | {c: "DEF" for c in range(10, 15)}
              | {c: "MID" for c in range(20, 25)} | {c: "FWD" for c in (30, 31, 32)})
    squad = Squad(players=[
        Player(PlayerIdentity(c, f"P{c}", Position(pos_of[c])), PlayerState(club_of[c], 55))
        for c in sorted(club_of)
    ])
    assert any("club 3 has 5 > 3" in p for p in squad.validate())
    assert not [p for p in squad.validate(club_baseline=squad.club_counts())
                if "club" in p]
