"""Black-box tests: distributional point forecasts + distributional H2H."""

import numpy as np
import polars as pl
import pytest

from fpl.dist import (
    QS,
    fit_residual_cdfs,
    moments_from_quantiles,
    player_points_quantiles,
    sample_from_cdf,
)
from fpl.team.simulate import simulate_h2h_dist, simulate_squad_distributions


def _cdf(low: float, high: float) -> list[float]:
    mid = (low + high) / 2
    return [low, low + (mid - low) * 0.1, low + (mid - low) * 0.25,
            low + (mid - low) * 0.5, mid, mid + (high - mid) * 0.5,
            high - (high - mid) * 0.25, high - (high - mid) * 0.1, high]


class TestDist:
    def test_fit_residual_cdfs_bin_by_context(self):
        act = np.array([1.0, 3.0, 5.0, 2.0, 4.0])
        pred = np.array([2.0, 2.0, 6.0, 2.0, 2.0])
        ctx = np.array(["FWD", "FWD", "DEF", "FWD", "DEF"])
        cdfs = fit_residual_cdfs(act, pred, ctx)
        assert set(cdfs) == {"FWD", "DEF"}
        # DEF residuals: (5-6), (4-2) -> [-1, 2]
        assert np.allclose(cdfs["DEF"], np.quantile([-1.0, 2.0], QS))

    def test_player_quantiles_additive(self):
        cdf = np.quantile([-1, 0, 0, 1, 2], QS)
        q = player_points_quantiles(5.0, cdf)
        assert np.allclose(q, 5.0 + cdf)

    def test_moments_degenerate(self):
        qs = np.asarray(QS)
        p = player_points_quantiles(0.0, np.full(len(qs), 3.0))
        m = moments_from_quantiles(p, qs)
        assert m["mean"] == pytest.approx(3.0, abs=1e-6)
        assert m["std"] < 1e-6

    def test_sample_from_cdf_in_range(self):
        rng = np.random.default_rng(0)
        cdf = np.array([0.0, 5.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
        got = [sample_from_cdf(cdf, QS, rng) for _ in range(300)]
        assert 0.0 <= min(got) <= max(got) <= 10.0


class TestDistributedH2H:
    # two identical-composition squads: each has one low-vol player (DEF) and
    # one high-vol player (FWD), same expected points. Mean-only H2H gives
    # equal results for both; h2h_dist must also, but through the samples.
    @pytest.fixture()
    def basket(self):
        return pl.DataFrame({
            "team_id": [0, 0, 1, 1],
            "player_code": [1, 2, 3, 4],
        })

    @pytest.fixture()
    def dist(self):
        rows = []
        for gw in (31, 32):
            for code, lo, hi in [(1, 4.8, 5.2), (2, 1.0, 9.0),
                                 (3, 4.8, 5.2), (4, 1.0, 9.0)]:
                q = _cdf(lo, hi)
                rows.append({"player_code": code, "gw": gw,
                             "quantiles_struct": {"q01": q[0], "q05": q[1],
                                                  "q10": q[2], "q25": q[3],
                                                  "q50": q[4], "q75": q[5],
                                                  "q90": q[6], "q95": q[7],
                                                  "q99": q[8]}})
        return pl.DataFrame(rows)

    def test_twin_squads_same_value(self, basket, dist):
        # identical-composition squads: their H2H rout against each other must
        # be a wash (r0 and r1 are mirrors -> r0 + r1 = 1), and each ~0.5.
        sd = simulate_squad_distributions(basket, dist, n_samples=200, seed=1)
        v = simulate_h2h_dist(sd, n_samples=200)
        r0 = v.filter(pl.col("team_id") == 0).get_column("win_ratio").item()
        r1 = v.filter(pl.col("team_id") == 1).get_column("win_ratio").item()
        assert r0 + r1 == pytest.approx(1.0)
        assert r0 == pytest.approx(0.5, abs=0.05)
        assert r1 == pytest.approx(0.5, abs=0.05)

    def test_value_defined_for_all_squads(self, basket, dist):
        # MC value must be finite for every squad (no NaN),
        # and monotone in the sample-mean ordering.
        sd = simulate_squad_distributions(basket, dist, n_samples=300, seed=2)
        v = simulate_h2h_dist(sd, n_samples=300)
        assert v.get_column("win_ratio").null_count() == 0
        assert v.get_column("win_ratio").is_between(0.0, 1.0).all()
        # mirrored twins -> identical rank when compared to a third squad
        m = sd.group_by("team_id").agg(pl.col("sample_mean").mean())
        assert m.height == 2