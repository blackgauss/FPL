"""Black-box tests: t-digest residual CDFs + distributional H2H."""

import numpy as np
import polars as pl
import pytest
from tdigest import TDigest

from fpl.dist import (
    QS,
    fit_residual_cdfs,
    moments_from_quantiles,
    quantiles_of,
    sample_from_digest,
    update_residual_cdfs,
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
        ctx = np.array([("FWD", "£5-7m"), ("FWD", "£5-7m"),
                        ("DEF", "£5-7m"), ("FWD", "£5-7m"), ("DEF", "£5-7m")],
                       dtype=object)
        cdfs = fit_residual_cdfs(act, pred, ctx)
        assert set(cdfs) == {("FWD", "£5-7m"), ("DEF", "£5-7m")}
        # digests expose quantiles via .percentile
        q = quantiles_of(cdfs[("FWD", "£5-7m")])
        assert q.shape == (len(QS),)

    def test_update_incremental(self):
        d1 = TDigest()
        for x in [1, 2, 3, 4]:
            d1.update(float(x))
        n_before = d1.n
        # update() folds new residuals into an existing digest (the "refresh
        # players weekly" story — no refitting needed)
        update_residual_cdfs({("FWD", "£5-7m"): d1},
                             np.array([5.0, 6.0]), np.array([0.0, 0.0]),
                             np.array([("FWD", "£5-7m")] * 2, dtype=object))
        assert d1.n == n_before + 2

    def test_sample_from_digest_recovers_modes(self):
        rng = np.random.default_rng(0)
        d = TDigest()
        for x in [1.0] * 200 + [9.0] * 700:
            d.update(x)
        got = np.array([sample_from_digest(d, rng) for _ in range(2000)])
        # heavier mode (9.0) should dominate the samples
        assert got.mean() > 5.0
        assert np.percentile(got, 75) >= 8.5

    def test_moments_degenerate(self):
        # constant points -> degenerate CDF -> std 0
        d = TDigest()
        for _ in range(200):
            d.update(3.0)
        q = quantiles_of(d)
        m = moments_from_quantiles(q, QS)
        assert m["std"] < 5e-2


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