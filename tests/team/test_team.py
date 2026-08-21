"""Black-box tests: score -> filter -> enumerate stages.

Uses small synthetic frames (no external/ dir, no real parquet) so the maths
of each stage is checked against hand-computed expectations.
"""

import math

import polars as pl
import pytest

from fpl.team.enumerate import greedy_teams, search_space_size, squad_for_price
from fpl.team.filtering import drop_never_featuring, filter_pool
from fpl.team.scoring import score_players


class TestScoring:
    def test_aggregates_over_horizon(self, monkeypatch):
        # stub expected_points_horizon to avoid needing a real model
        def fake_horizon(td, model, gw_start, gw_end, players):
            return pl.DataFrame({
                "player_code": [1, 1, 2], "web_name": ["A", "A", "B"],
                "position": ["FWD", "FWD", "MID"], "team_code": [11, 11, 22],
                "gw": [31, 32, 31], "expected_points": [3.0, 4.0, 7.0],
            })

        import fpl.team.scoring as sc

        monkeypatch.setattr(sc, "expected_points_horizon", fake_horizon)
        players = pl.DataFrame({
            "player_code": [1, 2], "web_name": ["A", "B"],
            "position": ["FWD", "MID"], "team_code": [11, 22],
        })
        agg = score_players(object(), None, gw_start=31, gw_end=32, players=players)
        assert agg.height == 2
        a = agg.filter(pl.col("player_code") == 1)
        assert a.get_column("expected_total").item() == pytest.approx(7.0)
        assert a.get_column("expected_mean_per_gw").item() == pytest.approx(3.5)


class TestFiltering:
    @pytest.fixture()
    def scored(self):
        return pl.DataFrame({
            "player_code": [1, 2, 3, 4, 5],
            "web_name": ["a", "b", "c", "d", "e"],
            "position": ["FWD", "FWD", "FWD", "MID", "MID"],
            "team_code": [11, 11, 33, 22, 44],
            "expected_total": [10.0, 8.0, 6.0, 9.0, 7.0],
        })

    def test_drop_never_featuring(self, scored):
        avail = pl.DataFrame({
            "player_code": [1, 2, 4],
            "minutes_in_window": [90, 0, 180], "now_cost": [8.0, 5.0, 9.0],
        })
        # 0 minutes in the window = expected to never feature -> dropped
        kept = drop_never_featuring(scored, avail, min_minutes=0)
        assert set(kept["player_code"]) == {1, 4}
        kept2 = drop_never_featuring(scored, avail, min_minutes=80)
        assert set(kept2["player_code"]) == {1, 4}

    def test_filter_pool_team_cap(self, scored):
        avail = pl.DataFrame({
            "player_code": [1, 2, 3, 4, 5],
            "minutes_in_window": [90, 90, 90, 90, 90],
            "now_cost": [8.0, 5.0, 6.0, 9.0, 7.0],
        })
        pool = filter_pool(scored, avail, top_k_per_position=10, max_per_team=1)
        # max_per_team=1 -> at most one per club
        grouped = pool.group_by("team_code").len()
        assert (grouped["len"] <= 1).all()

    def test_search_space_size(self):
        # exact C(avail, need) per position; pool sized so the comb is clean
        counts = {"GKP": 6, "DEF": 8, "MID": 10, "FWD": 7}
        rows = []
        code = 0
        for pos, n in counts.items():
            for _ in range(n):
                code += 1
                rows.append((code, f"p{code}", pos, code % 10 + 1, 5.0, float(100 - code)))
        pool = pl.DataFrame(rows, schema=["player_code", "web_name", "position",
                                          "team_code", "now_cost", "expected_total"],
                            orient="row")
        n, lg = search_space_size(pool)
        expect = math.comb(6, 2) * math.comb(8, 5) * math.comb(10, 5) * math.comb(7, 3)
        assert n == expect
        assert lg == pytest.approx(math.log10(expect))


class TestEnumerate:
    @pytest.fixture()
    def pool(self):
        # cheap-ish prices so the £100m budget doesn't starve the greedy fill
        prices = [4.0 + (i % 10) * 0.3 for i in range(1, 61)]  # 4.0..6.7
        return pl.DataFrame({
            "player_code": list(range(1, 61)),
            "web_name": [f"p{i}" for i in range(1, 61)],
            "position": (["GKP"] * 8 + ["DEF"] * 20 + ["MID"] * 22 + ["FWD"] * 10),
            "team_code": (list(range(1, 9)) + list(range(1, 21))
                          + list(range(1, 23)) + list(range(1, 11))),
            "now_cost": prices,
            "expected_total": [float(60 - i) for i in range(60)],
        })

    def test_greedy_teams_valid(self, pool):
        teams = greedy_teams(pool, n_teams=8, seed=1)
        assert teams["team_id"].n_unique() == 8
        summ = teams.group_by("team_id").agg(
            pl.col("player_code").count().alias("n"),
            pl.col("price_tenths").sum().alias("cost"),
            pl.col("position").n_unique().alias("positions"),
        )
        assert (summ["n"] == 15).all()
        assert (summ["cost"] <= 1000).all()
        assert (summ["positions"] == 4).all()

    def test_greedy_diverse(self, pool):
        # The greedy heuristic yields some distinct lineups per team (the
        # 'missed star' jitter); the tiny uniform fixture doesn't force many,
        # real pools give ~16/20 distinct. This asserts the method produces
        # more than a handful, not that every squad is unique.
        t1 = greedy_teams(pool, n_teams=8, seed=1)
        sig1 = t1.group_by("team_id").agg(pl.col("player_code").sort().alias("sig"))
        assert sig1["sig"].n_unique() >= 4

    def test_squad_for_price_annotates(self, pool):
        sp = squad_for_price(pool)
        assert "price_tenths" in sp.columns
        assert sp.get_column("price_tenths").min() == 40  # £4.0 = 40 tenths