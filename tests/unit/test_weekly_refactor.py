"""Black-box tests for the refactored weekly building blocks:

1. element_type -> Position projection (single canonical live mapping),
2. reuse of ownership-weighted opponent sampling (valid + deterministic),
3. explicit choose_transfer validity semantics and policy branches.
"""

import numpy as np
import polars as pl
import pytest

from fpl.domain import Player, PlayerIdentity, PlayerState, Position, squad_from_frame
from fpl.live.filters import position_for_element_type
from fpl.live.live import to_live_frame
from fpl.weekly.opponents import sample_opponents
from fpl.weekly.transfer import apply_transfer, choose_transfer


class TestElementTypeProjection:
    def test_maps_full_range(self):
        assert position_for_element_type(1) is Position.GKP
        assert position_for_element_type(2) is Position.DEF
        assert position_for_element_type(3) is Position.MID
        assert position_for_element_type(4) is Position.FWD

    def test_live_frame_roundtrip(self):
        # the canonical live -> domain identity path used by the scripts
        payload = {"elements": [{"id": i, "code": 1000 + i, "web_name": f"P{i}",
                                 "team": 1, "team_code": 10, "element_type": 1 + i % 4,
                                 "now_cost": 50, "status": "a", "news": "",
                                 "news_added": None,
                                 "chance_of_playing_this_round": None,
                                 "chance_of_playing_next_round": None,
                                 "selected_by_percent": "2.0", "minutes": 90,
                                 "ep_next": "3.0", "ep_this": None,
                                 "removed": False, "can_select": True,
                                 "can_transact": True}
                                for i in range(8)]}
        live = to_live_frame(payload)
        positions = [position_for_element_type(e) for e in live["element_type"]]
        assert len(positions) == 8

    def test_unknown_raises(self):
        with pytest.raises(KeyError, match="element_type"):
            position_for_element_type(9)


def _ownership_pool(rng=None, codes=None):
    """A synthetic pool large enough to fill a 15-player squad per position."""
    positions = ["GKP"] * 10 + ["DEF"] * 20 + ["MID"] * 20 + ["FWD"] * 10
    players = [
        Player(PlayerIdentity(i + 1, f"P{i + 1}", Position(pos)), PlayerState(i + 1, 50))
        for i, pos in enumerate(positions)
    ]
    ownership = {p.code: 5.0 + (p.code % 40) for p in players}
    return players, ownership


def _squad_dict(player):
    return {"player_code": player.code, "web_name": player.name,
            "position": player.position.value, "team_code": player.club,
            "price_tenths": player.cost_tenths}


class TestOpponentSampling:
    def test_samples_valid_squads(self):
        players, ownership = _ownership_pool()
        squads = sample_opponents(np.random.default_rng(0), players, ownership, 5)
        assert len(squads) == 5
        for squad in squads:
            frame = pl.DataFrame([_squad_dict(p) for p in squad])
            built = squad_from_frame(frame, gw=1)
            assert built.validate() == []

    def test_deterministic(self):
        players, ownership = _ownership_pool()
        a = sample_opponents(np.random.default_rng(7), players, ownership, 3)
        b = sample_opponents(np.random.default_rng(7), players, ownership, 3)
        assert [tuple(p.code for p in s) for s in a] == \
            [tuple(p.code for p in s) for s in b]


class TestTransferBranches:
    def _squad(self):
        frame = pl.DataFrame({
            "player_code": list(range(1, 16)),
            "web_name": [f"P{i}" for i in range(1, 16)],
            "position": ["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3,
            "team_code": [1 + (i % 12) for i in range(15)],
            "price_tenths": [50] * 15,
        })
        base = squad_from_frame(frame, gw=1)
        return base.__class__(
            players=base.players, gw=1, starters=base.starters,
            bench=(2, 7, 14, 15), captain=5, vice_captain=1,
        )

    def _first_mid_starter(self, squad):
        return next(c for c in squad.starters
                    if squad.by_code()[c].position == Position.MID)

    def _mid(self, code, club=20, cost=50):
        return Player(PlayerIdentity(code, f"N{code}", Position.MID),
                      PlayerState(club, cost))

    def _expected(self, squad, delta_code=None, delta_value=None):
        base = {c: 2.0 for c in squad.codes()}
        if delta_code is not None:
            base[delta_code] = delta_value
        return base

    def test_validate_semantics_consider_only_valid_swaps(self):
        # regression: choose_transfer must only ever return a Proposal that
        # passes validate() (explicit problem-list semantics)
        squad = self._squad()
        expected = self._expected(squad, 900, 10.0)
        new = self._mid(900, cost=500)  # unaffordable -> invalid swap
        assert choose_transfer(squad, [new], expected)[0] is None
        ok = self._mid(901, cost=50)
        out_chosen, in_chosen, _ = choose_transfer(
            squad, [ok], self._expected(squad, 901, 8.0))
        proposal = apply_transfer(squad, out_chosen, in_chosen, gw=1)
        assert proposal.validate() == []

    def test_free_transfer_cost_versus_penalty(self):
        squad = self._squad()
        new = self._mid(910)
        expected = self._expected(squad, 910, 10.0)
        _, _, free_gain = choose_transfer(squad, [new], expected, free_transfers=1)
        _, _, paid_gain = choose_transfer(squad, [new], expected, free_transfers=0)
        assert free_gain == pytest.approx(8.0)
        assert paid_gain == pytest.approx(4.0)

    def test_min_gain_threshold_blocks_small_deltas(self):
        squad = self._squad()
        new = self._mid(920)
        expected = self._expected(squad, 920, 3.0)
        out, _, _ = choose_transfer(squad, [new], expected, min_gain=2.0)
        assert out is None

    def test_cross_position_swap_never_returned(self):
        # choose_transfer must never replace a GKP/DEF out with a forward
        squad = self._squad()
        expected = self._expected(squad, 930, 9.0)
        forward = Player(PlayerIdentity(930, "N930", Position.FWD),
                         PlayerState(99, 50))
        out_chosen, in_chosen, _ = choose_transfer(squad, [forward], expected)
        if out_chosen is not None:
            # a same-position FWD-for-FWD bench swap may be returned, but a
            # cross-position replacement must never happen
            assert in_chosen.position == out_chosen.position
            assert squad.by_code()[out_chosen.code].position == Position.FWD

    def test_missing_expected_defaults_to_zero(self):
        squad = self._squad()
        new = self._mid(940)
        expected = {c: 2.0 for c in squad.codes()}
        # leave the candidate out of expected -> 0.0 forecast -> skipped
        out, in_, gain = choose_transfer(squad, [new], expected)
        assert out is None and in_ is None and gain == 0.0