"""Tests: live snapshot fetch, disk cache, rate-limit safety, and filters.

The FPL API is throttled; a unit test must not hit it. We monkeypatch
`requests.Session.get` and verify:
- first call writes a cache; TTL-valid second call does NOT hit the network;
- a fetch failure falls back to cache (graceful) and raises only with no cache;
- filters behave per the FPL status enum on a synthetic payload.
"""

import json
import time

import polars as pl
import pytest

from fpl.domain import Player, PlayerForm, PlayerIdentity, Position, Squad
from fpl.live.agreement import hygiene_summary, price_diff_tenths, report_agreement, to_tenths
from fpl.live.filters import (
    available,
    chance_of_playing,
    flag_squad_player,
    in_league,
    no_news,
    not_injured_suspended,
    not_transferred,
    price_unchanged,
    suggest,
)
from fpl.live.live import LiveFetchError, fetch_bootstrap, load_live_state, to_live_frame
from tests.live.fixtures_live import make_dataset, make_payload


@pytest.fixture()
def payload():
    return make_payload()


@pytest.fixture()
def live(payload):
    return to_live_frame(payload)


class FakeResponse:
    """Minimal requests.Response stand-in: .raise_for_status(), .json()."""

    def __init__(self, payload=None, error=None):
        self._payload = payload
        self._error = error

    def raise_for_status(self):
        if self._error:
            raise self._error
        return None

    def json(self):
        if self._error:
            raise self._error
        return self._payload


def _patch_get(monkeypatch, fake_get):
    monkeypatch.setattr("fpl.live.live.requests.Session.get", fake_get)


class TestFetchAndCache:
    def test_fetch_then_cache_no_second_request(self, tmp_path, monkeypatch, payload):
        calls = {"n": 0}

        def fake_get(self, url, **kwargs):
            calls["n"] += 1
            return FakeResponse(payload)

        _patch_get(monkeypatch, fake_get)
        cache = tmp_path / "live.json"
        df1, ts1 = load_live_state(cache, max_age_seconds=3600)
        df2, ts2 = load_live_state(cache, max_age_seconds=3600)
        assert calls["n"] == 1, "second call within TTL must use cache"
        assert df1.height == df2.height
        assert ts1 == ts2

    def test_stale_cache_refetches(self, tmp_path, monkeypatch, payload):
        calls = {"n": 0}

        def fake_get(self, url, **kwargs):
            calls["n"] += 1
            return FakeResponse(payload)

        _patch_get(monkeypatch, fake_get)
        cache = tmp_path / "live.json"
        load_live_state(cache, max_age_seconds=3600)
        # force staleness
        rec = json.loads(cache.read_text())
        rec["fetched_epoch"] = time.time() - 10000
        cache.write_text(json.dumps(rec))
        load_live_state(cache, max_age_seconds=60)
        # 1 initial fetch + 1 stale refetch
        assert calls["n"] == 2

    def test_fetch_failure_falls_back_to_cache(self, tmp_path, monkeypatch, payload):
        cache = tmp_path / "live.json"
        cache.write_text(json.dumps(
            {"fetched_at": "x", "fetched_epoch": time.time(), "payload": payload}))

        def boom(self, url, **kwargs):
            raise ConnectionError("rate limited")

        _patch_get(monkeypatch, boom)
        df, ts = load_live_state(cache, max_age_seconds=3600)
        assert df.height == len(payload["elements"])

    def test_fetch_failure_no_cache_raises(self, tmp_path, monkeypatch):
        def boom(self, url, **kwargs):
            raise ConnectionError("rate limited")

        _patch_get(monkeypatch, boom)
        with pytest.raises(LiveFetchError):
            load_live_state(tmp_path / "missing.json", max_age_seconds=3600)

    def test_fetch_bootstrap_returns_json(self, monkeypatch, payload):
        class _Resp:
            def __init__(self, p):
                self._p = p

            def raise_for_status(self):  # noqa
                return None

            def json(self):
                return self._p

        monkeypatch.setattr("fpl.live.live.requests.Session.get",
                            lambda *a, **k: _Resp(payload))
        assert fetch_bootstrap() == payload


class TestLiveFrame:
    def test_keyed_on_stable_code_and_ints(self, live):
        assert live.get_column("player_code").dtype == pl.Int64
        assert live.get_column("now_cost").dtype == pl.Int64
        assert live.height == 8

    def test_now_cost_is_tenths(self, live):
        assert live.get_column("now_cost").min() == 50  # first fixture price


class TestFilters:
    def test_available(self, live):
        assert int(available(live).sum()) == 4  # 'a' count in fixture

    def test_not_injured_suspended(self, live):
        # excludes i, s, u; keeps a, d
        assert int(not_injured_suspended(live).sum()) == 5

    def test_in_league(self, live):
        assert int(in_league(live).sum()) == live.height  # all removable=False

    def test_chance_unknown_kept_if_available(self, live):
        # all chance_of_playing_next_round are None; a/d pass the bar
        m = chance_of_playing(live, min_pct=75)
        assert int(m.sum()) == int(not_injured_suspended(live).sum())

    def test_no_news(self, live):
        assert int(no_news(live).sum()) == live.height  # all news empty

    def test_not_transferred_and_price(self, live):
        # live-sorted columns; expected dataset team_code = live team_code
        live_sorted = live.sort("player_code")
        expected_team = live_sorted.get_column("team_code")
        expected_price = live_sorted.get_column("now_cost")
        assert int(not_transferred(live, expected_team).sum()) == live.height
        assert int(price_unchanged(live, expected_price).sum()) == live.height

    def test_suggest_excludes_injured(self, live):
        # suggest = in-league & not injured/suspended & chance OK:
        # fixture statuses a,a,a,d,i,s,u,a -> keep a,a,a,d,a = 5
        s = suggest(live)
        assert int(s.sum()) == 5
        # exactly the i/s/u rows are excluded
        excluded = live.filter(~s.alias("p")).get_column("status").to_list()
        assert set(excluded) <= {"i", "s", "u"}


class TestCurrentWorld:
    """Reconciliation feeds construction the current clubs/availability."""

    @pytest.fixture()
    def scored(self, live):
        # a scored-like frame: player + dataset-stale team_code + points
        return pl.DataFrame({
            "player_code": [223001, 223002, 223003, 223004, 223005, 223006, 223007, 223008],
            "web_name": [f"P{i}" for i in range(8)],
            "position": ["FWD"] * 2 + ["DEF"] * 6,
            "expected_total": [10.0 + i for i in range(8)],
            "team_code": [1, 1, 2, 2, 3, 3, 4, 4],  # stale clubs
        })

    def test_transfer_updates_club_in_pool(self, scored, live):
        from fpl.live.current import reconcile_player_clubs

        # live gives P0 team 2 (was 1 -> transferred)
        out = reconcile_player_clubs(scored, live)
        p0 = out.filter(pl.col("player_code") == 223001)
        assert p0.get_column("team_code").item() == live.filter(
            pl.col("player_code") == 223001)["team_code"].item()

    def test_missing_from_live_dropped(self, scored, live):
        from fpl.live.current import reconcile_player_clubs

        # drop P7 from live roster -> absent players are removed
        reduced = live.filter(pl.col("player_code") != 223008)
        out = reconcile_player_clubs(scored, reduced)
        assert 223008 not in out.get_column("player_code").to_list()

    def test_construction_input_applies_mask(self, scored, live):
        from fpl.live.current import construction_input

        mask = pl.Series([True] * 8)  # all playable
        out = construction_input(scored, live, mask)
        # still has all 8, clubs now from live
        assert out.height == 8
        assert out.get_column("team_code").eq(live.sort("player_code")["team_code"]).all()

    def test_construction_input_excludes_injured(self, scored, live):
        from fpl.live.current import construction_input
        from fpl.live.filters import suggest

        out = construction_input(scored, live, suggest(live))
        # fixture has i/s/u players (223005..223007) -> excluded
        kept = set(out.get_column("player_code").to_list())
        assert {223005, 223006, 223007}.isdisjoint(kept)


class TestFlagSquadPlayer:
    """Detect missing/injured players in a candidate team (the reported bug)."""

    def test_healthy_ok(self):
        assert flag_squad_player(
            {"status": "a", "team_code": 1, "team_code_live": 1,
             "price_diff_tenths": 0}) == "ok"

    def test_injured_detected(self):
        f = flag_squad_player(
            {"status": "i", "team_code": 1, "team_code_live": 1,
             "price_diff_tenths": 0})
        assert "INJURED" in f and "UNAVAILABLE[i]" in f

    def test_transferred_detected(self):
        f = flag_squad_player(
            {"status": "a", "team_code": 1, "team_code_live": 8,
             "price_diff_tenths": 0})
        assert "TRANSFERRED (ds 1 -> live 8)" in f

    def test_price_move_detected(self):
        f = flag_squad_player(
            {"status": "a", "team_code": 1, "team_code_live": 1,
             "price_diff_tenths": 30})
        assert "price +30" in f

    def test_missing_from_live_flagged(self):
        # player absent from the live roster entirely = missing/transferred out
        f = flag_squad_player(
            {"status": None, "team_code": 1, "team_code_live": None,
             "price_diff_tenths": None})
        assert "NOT IN LIVE ROSTER" in f


class TestFlagSquad:
    """Squad-level live flags from the typed interface (no column joining)."""

    def _squad(self, players):
        return Squad(players=tuple(players),
                     starters=tuple(p.code for p in players))

    @staticmethod
    def _p(code, name, position, club, cost):
        return Player(identity=PlayerIdentity(code, name, Position(position)),
                      form=PlayerForm(club, cost))

    def test_healthy_all_ok(self, live):
        from fpl.live.filters import flag_squad

        squad = self._squad([
            self._p(223001, "P0", "MID", 10, 50),   # team_code 10, £5.0
            self._p(223002, "P1", "MID", 20, 60),   # team_code 20, £6.0
            self._p(223008, "P7", "DEF", 30, 120),  # team_code 30, £12.0
        ])
        out = flag_squad(squad, live)
        assert all(flag == "ok" for flag in out.values())

    def test_injured_detected(self, live):
        from fpl.live.filters import flag_squad

        squad = self._squad([self._p(223005, "P4", "MID", 50, 90)])
        f = flag_squad(squad, live)[223005]
        assert "UNAVAILABLE[i]" in f and "INJURED" in f

    def test_transferred_detected(self, live):
        # squad records old club (99); live says 10
        from fpl.live.filters import flag_squad

        squad = self._squad([self._p(223001, "P0", "MID", 99, 50)])
        f = flag_squad(squad, live)[223001]
        assert "TRANSFERRED (ds 99 -> live 10)" in f

    def test_price_move_detected(self, live):
        from fpl.live.filters import flag_squad

        squad = self._squad([self._p(223001, "P0", "MID", 10, 45)])
        f = flag_squad(squad, live)[223001]
        assert "price +5" in f  # live 50 tenths - squad 45 tenths

    def test_missing_from_live_flagged(self, live):
        from fpl.live.filters import flag_squad

        squad = self._squad([self._p(999999, "ghost", "MID", 10, 50)])
        assert "NOT IN LIVE ROSTER" in flag_squad(squad, live)[999999]


class TestHygieneAgreement:
    # price units: live now_cost is tenths (155 = £15.5m); datasets may store
    # decimal millions (15.5) -> price_scale=10, or already tenths -> scale=1.

    def test_frame_filter_by_code(self, live):
        from fpl.live.filters import filter_frame_by_code, suggest

        frame = live.select("player_code").with_columns(
            pl.lit(1.0).alias("score"))
        out = filter_frame_by_code(frame, live, suggest(live))
        assert out.height == int(suggest(live).sum())
        assert set(live.filter(~suggest(live))["player_code"].to_list()) & \
            set(out["player_code"].to_list()) == set()

    def test_to_tenths_converts_decimal_millions(self):
        s = pl.Series([15.5, 6.0, 4.0], dtype=pl.Float64)
        assert to_tenths(s, 10).to_list() == [155, 60, 40]

    def test_price_diff_tenths_respects_scale(self, live):
        # dataset in decimal millions that AGREES with live -> diff 0
        ds_prices = (live.sort("player_code").get_column("now_cost") / 10)
        assert (price_diff_tenths(live, ds_prices, scale=10).abs() == 0).all()

    def test_no_scale_flags_every_move(self, live):
        """The unit trap: comparing decimal-millions as if tenths shows all
        players 'moved' — this test pins that failure mode explicitly."""
        ds = make_dataset(
            prices_tenths=(live.sort("player_code").get_column("now_cost") / 10).to_list(),
            team_codes=live.sort("player_code").get_column("team_code").to_list())
        rep_wrong = report_agreement(live, ds, dataset_price_col="now_cost",
                                     dataset_team_col="team_code", price_scale=1)
        assert int((rep_wrong["price_diff_tenths"].abs() > 0).sum()) == live.height

    def test_report_matches_if_prices_teams_agree(self, live):
        ds = make_dataset(
            prices_tenths=live.sort("player_code").get_column("now_cost").to_list(),
            team_codes=live.sort("player_code").get_column("team_code").to_list())
        # dataset already tenths -> scale=1
        summary = hygiene_summary(live, ds, dataset_price_col="now_cost",
                                  dataset_team_col="team_code", price_scale=1)
        assert summary["price_moved"].item() == 0
        assert summary["team_transferred"].item() == 0

    def test_decimal_dataset_scale10_agrees(self, live):
        # the real-world case: dataset decimal millions + scale=10 -> no false moves
        ds = make_dataset(
            prices_tenths=(live.sort("player_code").get_column("now_cost") / 10).to_list(),
            team_codes=live.sort("player_code").get_column("team_code").to_list())
        summary = hygiene_summary(live, ds, dataset_price_col="now_cost",
                                  dataset_team_col="team_code", price_scale=10)
        assert summary["price_moved"].item() == 0

    def test_report_flags_price_and_team_moves(self, live):
        # dataset disagrees: prices shifted +1 tenth, teams different
        ds = make_dataset(
            prices_tenths=[51, 61, 71, 81, 91, 101, 111, 121],
            team_codes=[99] * 8)
        rep = report_agreement(live, ds, dataset_price_col="now_cost",
                               dataset_team_col="team_code", price_scale=1)
        assert int((rep["price_diff_tenths"].abs() > 0).sum()) == 8
        assert int((~rep["team_ok"]).sum()) == 8