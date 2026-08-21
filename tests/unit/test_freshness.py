"""Black-box tests: freshness gate fails fast on stale/drifted inputs."""

import time

import polars as pl
import pytest

from fpl.live.freshness import (
    FreshenError,
    check_drift,
    check_season_has_rows,
    check_snapshot_age,
)


class TestSnapshotAge:
    def test_fresh_ok(self):
        check_snapshot_age(time.time(), max_age_seconds=3600)

    def test_stale_raises(self):
        with pytest.raises(FreshenError, match="old"):
            check_snapshot_age(time.time() - 7200, max_age_seconds=3600)


class TestSeasonRows:
    def test_empty_store_raises(self, tmp_path):
        import polars as pl

        pl.DataFrame({"player_code": [], "gw": []}).write_parquet(
            tmp_path / "features_2025-2026.parquet")
        with pytest.raises(FreshenError, match="empty"):
            check_season_has_rows(str(tmp_path), "2025-2026")

    def test_missing_store_raises(self, tmp_path):
        with pytest.raises(FreshenError, match="no feature store"):
            check_season_has_rows(str(tmp_path), "2025-2026")

    def test_populated_ok(self, tmp_path):
        import polars as pl

        pl.DataFrame({"player_code": [1], "gw": [2]}).write_parquet(
            tmp_path / "features_2025-2026.parquet")
        check_season_has_rows(str(tmp_path), "2025-2026")


class TestDrift:
    def test_no_drift_ok(self):
        live = pl.DataFrame({
            "player_code": [1, 2], "now_cost": [150, 40],
            "team_code": [3, 43],
            "status": ["a", "a"],
        })
        ds = pl.DataFrame({
            "player_code": [1, 2], "now_cost": [15.0, 4.0],  # decimal, agree
            "team_code": [3, 43],
        })
        check_drift(live, ds, price_scale=10)  # tolerances default

    def test_price_drift_raises(self):
        live = pl.DataFrame({
            "player_code": [1, 2], "now_cost": [150, 40],
            "team_code": [3, 43], "status": ["a", "a"],
        })
        ds = pl.DataFrame({
            "player_code": [1, 2], "now_cost": [99.0, 99.0],  # both moved
            "team_code": [3, 43],
        })
        with pytest.raises(FreshenError, match="price drift"):
            check_drift(live, ds, price_scale=10, max_price_moved=0.5)

    def test_team_moves_raise(self):
        live = pl.DataFrame({
            "player_code": [1, 2], "now_cost": [150, 40],
            "team_code": [99, 43], "status": ["a", "a"],
        })
        ds = pl.DataFrame({
            "player_code": [1, 2], "now_cost": [15.0, 4.0],
            "team_code": [3, 43],  # player 1 transferred
        })
        with pytest.raises(FreshenError, match="team transfers"):
            check_drift(live, ds, price_scale=10, max_team_moved=0.25)