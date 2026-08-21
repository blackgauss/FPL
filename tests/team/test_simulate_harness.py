"""Black-box tests: H2H simulator + reusable harness wiring."""

import polars as pl
import pytest

from fpl.team.harness import REGISTRY, SearchResult
from fpl.team.simulate import simulate_h2h, squad_gw_totals, weaknesses


class TestSimulate:
    @pytest.fixture()
    def basket(self):
        # 2 squads, each 2 players
        return pl.DataFrame({
            "team_id": [0, 0, 1, 1],
            "player_code": [1, 2, 3, 4],
            "expected_total": [10.0, 5.0, 8.0, 7.0],
        })

    @pytest.fixture()
    def per_gw(self):
        # GW31: team0 = 1+2 -> 6.0; team1 = 3+4 -> 7.0 (1 wins)
        # GW32: team0 = 7.0; team1 = 6.0 (0 wins)
        return pl.DataFrame({
            "player_code": [1, 2, 3, 4, 1, 3],
            "gw": [31, 31, 31, 31, 32, 32],
            "expected_points": [3.0, 3.0, 4.0, 3.0, 7.0, 6.0],
        })

    def test_squad_gw_totals(self, basket, per_gw):
        t = squad_gw_totals(basket, per_gw).sort("team_id", "gw")
        r0 = t.filter((pl.col("team_id") == 0) & (pl.col("gw") == 31))
        assert r0.get_column("gw_total").item() == 6.0
        assert r0.get_column("n_players").item() == 2

    def test_h2h_record(self, basket, per_gw):
        totals = squad_gw_totals(basket, per_gw)
        v = simulate_h2h(totals)
        v0 = v.filter(pl.col("team_id") == 0)
        # team0 vs team1 over 2 GWs: GW31 lose, GW32 win -> 1W 1L 0D / 2
        assert (v0.get_column("wins").item(), v0.get_column("losses").item(),
                v0.get_column("draws").item()) == (1, 1, 0)
        assert v0.get_column("win_ratio").item() == pytest.approx(0.5)

    def test_weaknesses(self, basket, per_gw):
        totals = squad_gw_totals(basket, per_gw)
        w = weaknesses(basket, totals)
        assert "worst_gw" in w.columns
        assert (w.get_column("star_dependence").is_between(0, 1)).all()


class TestHarness:
    def test_registry_exposes_stages(self):
        assert set(REGISTRY) == {"enumerate", "value"}
        assert "greedy" in REGISTRY["enumerate"]
        assert "h2h" in REGISTRY["value"]

    def test_run_smoke(self, tmp_path):
        """End-to-end on a purely synthetic dataset (no external/)."""
        import polars as pl

        from fpl.team.harness import run as run_search

        class FakeModel:
            def predict(self, X):
                return [1.5] * len(X)

        code = list(range(1, 121))
        positions = ["GKP"] * 20 + ["DEF"] * 40 + ["MID"] * 40 + ["FWD"] * 20
        players = pl.DataFrame({
            "player_id": list(range(1, 121)), "player_code": code,
            "web_name": [f"p{i}" for i in code],
            "team_code": [i % 20 + 1 for i in code],
            "position": positions,
        })
        feat = pl.DataFrame({
            "player_id": [i % 120 + 1 for i in range(240)],
            "player_code": [i % 120 + 1 for i in range(240)],
            "gw": [2 + (i // 120) for i in range(240)],
            "team_code": [i % 20 + 1 for i in range(240)],
            "opponent_team_code": [(i + 5) % 20 + 1 for i in range(240)],
            "was_home": [True] * 240,
            "home_elo": [2000.0] * 240, "opponent_elo": [2000.0] * 240,
            "prev_points": [2.0] * 240, "pts_avg_3": [2.0] * 240,
            "pts_avg_5": [2.0] * 240, "total_points": [2] * 240,
            "next_points": [2] * 240,
        })
        gw = pl.DataFrame({
            "player_id": [i % 120 + 1 for i in range(240)],
            "player_code": [i % 120 + 1 for i in range(240)],
            "gw": [2 + (i // 120) for i in range(240)],
            "minutes": [90] * 240, "now_cost": [5.0] * 240,
            "ep_next": [2.0] * 240,
        })
        season = "synthetic"
        players.write_parquet(tmp_path / f"players_{season}.parquet")
        feat.write_parquet(tmp_path / f"features_{season}.parquet")
        gw.write_parquet(tmp_path / f"gw_stats_{season}.parquet")

        res = run_search(
            processed=str(tmp_path), season=season, gw_start=2, gw_end=3,
            model=FakeModel(), enum="greedy",
            enum_kw={"n_teams": 5, "seed": 0},
            pool_kw={"top_k_per_position": 60, "max_per_team": 10},
        )
        assert isinstance(res, SearchResult)
        assert res.basket["team_id"].n_unique() <= 5
        assert "win_ratio" in res.value.columns
        assert "worst_gw" in res.weakness.columns