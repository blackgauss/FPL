"""Black-box tests: Squad.gw_settlement — auto-subs, benches, captain/vice.

Rigid FPL rules live with the domain (docs/rules.md): bench-priority
auto-substitution, GK-for-GK only, formation-legal outfield swaps, empty
slots score 0, and captain-then-vice doubling. All fixtures synthetic.
"""

import pytest

from fpl.domain import Player, PlayerIdentity, PlayerState, Position, Squad

# 15 players, 11 starters (1 GKP 4 DEF 5 MID 1 FWD) + 4 bench in priority
# order [GKP12, MID13, DEF14, FWD15].
POSITIONS = [Position.GKP, *([Position.DEF] * 4), *([Position.MID] * 5),
             Position.FWD]
CODES = list(range(1, 12))          # starters 1..11
BENCH = (12, 13, 14, 15)
BENCH_POS = {12: Position.GKP, 13: Position.MID, 14: Position.DEF,
             15: Position.FWD}


def _squad(*, bench=BENCH, captain=6, vice=2):
    players = []
    for i in range(15):
        code = i + 1
        pos = POSITIONS[i] if code <= 11 else BENCH_POS[code]
        players.append(Player(
            identity=PlayerIdentity(code, f"P{code}", pos,),
            state=PlayerState(club=1, cost_tenths=50)))
    return Squad(players=tuple(players), starters=tuple(range(1, 12)),
                 bench=bench, captain=captain, vice_captain=vice)


def _pts(overrides=None):
    pts = {code: 2.0 for code in range(1, 16)}
    if overrides:
        pts.update(overrides)
    return pts


class TestSettlement:
    def test_all_starters_play(self):
        squad = _squad()
        out = squad.gw_settlement({c: True for c in range(1, 16)},
                                  _pts({6: 3.0}))
        assert out.playing == tuple(range(1, 12))
        assert out.substituted_in == ()
        assert out.captain_doubled == 6
        # 10 others at 2.0 + captain 3.0*2 + bench irrelevant
        assert out.gw_total == pytest.approx(10 * 2.0 + 6.0)

    def test_bench_priority_fills_dnp_starter(self):
        squad = _squad()
        played = {c: True for c in range(1, 16)}
        played[9] = False                    # MID starter doesn't play
        out = squad.gw_settlement(played, _pts({13: 4.0}))
        assert 9 not in out.playing
        assert 13 in out.playing             # first playing bench (MID) in
        assert out.substituted_in == (13,)
        assert out.captain_doubled == 6

    def test_gk_replaced_only_by_playing_bench_gk(self):
        squad = _squad()
        played = {c: True for c in range(1, 16)}
        played[1] = False                    # starting GK out
        played[12] = True                    # bench GK played
        played[13] = False                   # bench MID didn't
        out = squad.gw_settlement(played, _pts({12: 1.0}))
        assert 1 not in out.playing and 12 in out.playing
        assert 13 not in out.playing         # priority MID skipped (dnp)

    def test_gk_slot_not_filled_by_outfield(self):
        squad = _squad()
        played = {c: True for c in range(1, 16)}
        played[1] = False                    # GK out
        played[12] = False                   # bench GK also out
        # bench MID played but cannot fill a GK slot
        out = squad.gw_settlement(played, _pts())
        assert 1 not in out.playing and 12 not in out.playing
        assert 13 not in out.playing
        assert len(out.playing) == 10        # GK slot empty, scores 0

    def test_no_eligible_sub_slot_scores_zero(self):
        squad = _squad()
        played = {c: True for c in range(1, 16)}
        played[7] = False
        for b in BENCH:
            played[b] = False               # no eligible bench player either
        out = squad.gw_settlement(played, _pts())
        assert 7 not in out.playing
        assert len(out.playing) == 10

    def test_captain_out_uses_vice(self):
        squad = _squad()
        played = {c: True for c in range(1, 16)}
        played[6] = False                    # captain out
        out = squad.gw_settlement(played, _pts({2: 4.0}))
        assert out.captain_doubled == 2
        # 9 starters @2 + sub(13) @2 + vice 4.0*2 = 28.0
        assert out.gw_total == pytest.approx(9 * 2.0 + 2.0 + 8.0)

    def test_captain_and_vice_out_no_double(self):
        squad = _squad()
        played = {c: True for c in range(1, 16)}
        played[6] = played[2] = False
        out = squad.gw_settlement(played, _pts())
        assert out.captain_doubled is None