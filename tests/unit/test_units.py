"""Black-box tests: shared units / squad-constraint constants.

The price-units trap (live tenths vs dataset decimal) has bitten repeatedly;
this pins the conversion helpers and the single-source squad counts so later
stages (captain, transfers) build on consistent numbers.
"""

import polars as pl

from fpl.units import (
    DEFAULT_BUDGET_TENTHS,
    MAX_PER_CLUB,
    SQUAD_COUNTS,
    now_cost_to_tenths,
    to_millions,
    to_tenths,
)


class TestTenths:
    def test_roundtrip(self):
        assert to_tenths(15.5) == 155
        assert to_millions(155) == 15.5
        assert to_tenths(4.0) == 40

    def test_now_cost_expr_cast(self):
        # in polars a decimal millions column -> tenths ints
        df = pl.DataFrame({"now_cost": [15.5, 4.0]})
        out = df.with_columns(now_cost_to_tenths(pl.col("now_cost")).alias("t"))
        assert out["t"].to_list() == [155, 40]
        assert out["t"].dtype == pl.Int64


class TestSquadConstants:
    def test_budget_in_tenths(self):
        assert DEFAULT_BUDGET_TENTHS == 1000  # £100m

    def test_counts_sum_to_15(self):
        assert sum(SQUAD_COUNTS.values()) == 15
        assert set(SQUAD_COUNTS) == {"GKP", "DEF", "MID", "FWD"}

    def test_max_per_club(self):
        assert MAX_PER_CLUB == 3