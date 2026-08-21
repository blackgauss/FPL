"""Black-box tests for the basic captain/transfer planning baseline."""

import polars as pl

from fpl.domain import squad_from_frame
from fpl.weekly.captain import choose_captains, set_captains
from fpl.weekly.planner import plan_weeks
from fpl.weekly.transfer import apply_transfer, choose_transfer


def make_squad():
    positions = ["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    frame = pl.DataFrame({
        "player_code": list(range(1, 16)),
        "web_name": [f"P{i}" for i in range(1, 16)],
        "position": positions,
        "team_code": [1 + (i % 12) for i in range(15)],
        "price_tenths": [50] * 15,
    })
    base = squad_from_frame(frame, gw=1)
    return base.__class__(
        players=base.players, gw=1, starters=base.starters,
        bench=(2, 7, 14, 15), captain=None, vice_captain=None,
    )


def make_mid(code: int, club: int = 20, cost: int = 50):
    from fpl.domain import Player, PlayerIdentity, PlayerState, Position

    return Player(PlayerIdentity(code, f"P{code}", Position.MID),
                  PlayerState(club, cost))


class TestCaptain:
    def test_highest_starter_and_different_club_vice(self):
        squad = make_squad()
        captain, vice = choose_captains(
            squad, {code: float(code) for code in squad.starters})
        assert captain == max(squad.starters)
        assert vice is not None
        assert squad.by_code()[vice].club != squad.by_code()[captain].club

    def test_set_captains_returns_new_snapshot(self):
        squad = make_squad()
        configured = set_captains(squad, {code: float(code) for code in squad.starters})
        assert configured is not squad
        assert configured.captain == max(squad.starters)


class TestTransfer:
    def test_choose_and_apply_free_transfer(self):
        squad = make_squad()
        out_code = squad.starters[5]
        new = make_mid(99, club=20)
        out, incoming, gain = choose_transfer(
            squad, [new], {code: 2.0 for code in squad.codes()} | {99: 8.0})
        assert out.position.value == "MID" and incoming.code == 99 and gain == 6.0
        updated = apply_transfer(squad, out, incoming, gw=1)
        assert updated.validate() == []
        assert 99 in updated.codes() and out_code not in updated.codes()
        assert updated.transfers_in == (99,)

    def test_infeasible_budget_transfer_skipped(self):
        squad = make_squad()
        out_code = squad.starters[5]
        assert choose_transfer(
            squad, [make_mid(99, club=20, cost=500)],
            {out_code: 2.0, 99: 20.0}) == (None, None, 0.0)


class TestPlanner:
    def test_transfer_then_captain_each_week(self):
        squad = make_squad()
        new = make_mid(99, club=20)
        forecasts = {
            1: {**{code: 2.0 for code in squad.starters}, 99: 8.0},
            2: {**{code: 2.0 for code in squad.starters}, 99: 8.0},
        }
        planned = plan_weeks(squad, forecasts, {1: [new], 2: []}, weeks=2)
        assert [s.gw for s in planned] == [1, 2]
        assert planned[0].transfers_in == (99,)
        assert planned[1].transfers_in == ()
        assert planned[0].captain == 99
        assert planned[0].validate() == []
