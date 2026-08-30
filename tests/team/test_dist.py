"""Black-box tests: heteroskedastic residual CDFs + distributional H2H."""

import numpy as np
import polars as pl
import pytest
from tdigest import TDigest

from fpl.dist import (
    QS,
    fit_sigma_and_digest,
    moments_from_quantiles,
    quantiles_of,
    sample_from_digest,
    update_standardized,
)
from fpl.team.simulate import simulate_h2h_dist, simulate_squad_distributions


def _cdf(low: float, high: float) -> list[float]:
    mid = (low + high) / 2
    return [low, low + (mid - low) * 0.1, low + (mid - low) * 0.25,
            low + (mid - low) * 0.5, mid, mid + (high - mid) * 0.5,
            high - (high - mid) * 0.25, high - (high - mid) * 0.1, high]


class TestDist:
    def test_learning_params_merge_with_defaults(self):
        """A partial override must NOT drop the required defaults (objective
        etc.) — it should merge on top of them."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(120, 2))
        actual = rng.normal(size=120)
        sigma_model, digest = fit_sigma_and_digest(
            actual, np.zeros(120), X, ["x0", "x1"], [],
            learning_params={"num_leaves": 7})
        params = sigma_model.params
        assert params["num_leaves"] == 7           # override applied
        assert params["objective"] == "regression"  # default preserved
        assert digest.n > 0

    def test_sigma_and_digest(self):
        rng = np.random.default_rng(0)
        # two-regime noise: sigma varies with feature x1
        X = rng.normal(size=(400, 3))
        mu = X[:, 0] * 1.0
        true_sigma = 0.5 + 2.0 * (X[:, 1] > 0)
        actual = mu + true_sigma * rng.normal(size=400)
        sigma_model, digest = fit_sigma_and_digest(
            actual, mu, X, ["x0", "x1", "x2"], [])
        # sigma model separates the two regimes
        hi = sigma_model.predict(X[X[:, 1] > 0][:5])
        lo = sigma_model.predict(X[X[:, 1] <= 0][:5])
        assert hi.mean() > lo.mean() * 1.5
        # standardized residuals ~ N(0,1): digest median near 0
        assert abs(quantiles_of(digest, [0.5])[0]) < 0.3

    def test_update_standardized_folds_in(self):
        digest = TDigest()
        for _ in range(10):
            digest.update(0.1)
        n0 = digest.n
        update_standardized(digest, np.array([1.0, 2.0]), np.array([0.0, 0.0]),
                            np.array([1.0, 1.0]))
        assert digest.n == n0 + 2

    def test_moments_degenerate(self):
        d = TDigest()
        for _ in range(200):
            d.update(0.0)
        q = quantiles_of(d)  # all zeros -> points vector = q vector? no: zeros
        m = moments_from_quantiles(q, QS)
        assert m["std"] < 5e-2

    def test_sample_from_digest_recovers_modes(self):
        rng = np.random.default_rng(0)
        d = TDigest()
        for x in [1.0] * 200 + [9.0] * 700:
            d.update(x)
        got = np.array([sample_from_digest(d, rng) for _ in range(2000)])
        assert got.mean() > 5.0
        assert np.percentile(got, 75) >= 8.5


class TestDistributedH2H:
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
        sd = simulate_squad_distributions(basket, dist, n_samples=200, seed=1)
        v = simulate_h2h_dist(sd, n_samples=200)
        r0 = v.filter(pl.col("team_id") == 0).get_column("win_ratio").item()
        r1 = v.filter(pl.col("team_id") == 1).get_column("win_ratio").item()
        assert r0 + r1 == pytest.approx(1.0)
        assert r0 == pytest.approx(0.5, abs=0.05)
        assert r1 == pytest.approx(0.5, abs=0.05)

    def test_value_defined_for_all_squads(self, basket, dist):
        sd = simulate_squad_distributions(basket, dist, n_samples=300, seed=2)
        v = simulate_h2h_dist(sd, n_samples=300)
        assert v.get_column("win_ratio").null_count() == 0
        assert v.get_column("win_ratio").is_between(0.0, 1.0).all()