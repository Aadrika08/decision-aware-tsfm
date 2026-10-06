"""Mathematical invariants of the scoring layer. Numbers here are made up."""
import unittest

import numpy as np
from scipy import stats

from decision_aware_tsfm.analysis.decision.metrics import (
    TAU_GRID, check_taus, crps_from_quantiles, crps_weights, decision_cost,
    pinball, tau_from_costs,
)


class MetricInvariants(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20260924)

    def test_pinball_at_half_is_half_absolute_error(self):
        y = self.rng.normal(size=20_000) * 5
        q = self.rng.normal(size=20_000) * 5
        np.testing.assert_allclose(pinball(y, q, 0.5), 0.5 * np.abs(y - q), rtol=0, atol=1e-12)

    def test_decision_cost_equals_scaled_pinball(self):
        y = self.rng.normal(size=20_000) * 3
        q = self.rng.normal(size=20_000) * 3
        cu = self.rng.uniform(0.1, 10, size=20_000)
        co = self.rng.uniform(0.1, 10, size=20_000)
        tau = cu / (cu + co)
        direct = cu * np.maximum(y - q, 0) + co * np.maximum(q - y, 0)
        np.testing.assert_allclose(direct, (cu + co) * pinball(y, q, tau), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(decision_cost(y, q, 3.0, 1.0),
                                   4.0 * pinball(y, q, tau_from_costs(3.0, 1.0)), atol=1e-12)

    def test_optimal_action_is_the_tau_quantile(self):
        y = self.rng.standard_t(df=4, size=4001)          # heavy-tailed; any sample works
        grid = np.linspace(y.min(), y.max(), 20_001)[::10]
        step = grid[1] - grid[0]
        for tau in TAU_GRID:
            with self.subTest(tau=tau):
                mean_loss = np.array([pinball(y, g, tau).mean() for g in grid])
                best = grid[mean_loss.argmin()]
                lo = np.quantile(y, tau, method="inverted_cdf")
                hi = np.quantile(y, tau, method="higher")
                self.assertTrue(lo - step <= best <= hi + step)
                self.assertLessEqual(pinball(y, lo, tau).mean(), mean_loss.min() + 1e-12)

    def test_tau_outside_open_unit_interval_fails(self):
        for bad in [0.0, 1.0, -0.1, 1.2, np.nan, np.inf]:
            with self.subTest(tau=bad):
                with self.assertRaises(ValueError):
                    check_taus(bad)
                with self.assertRaises(ValueError):
                    pinball(1.0, 0.0, bad)

    def test_nan_quantile_fails(self):
        with self.assertRaises(ValueError):
            pinball([1.0, 2.0], [0.5, np.nan], 0.5)

    def test_pinball_is_never_negative(self):
        y = self.rng.normal(size=1000)
        q = self.rng.normal(size=1000)
        for t in TAU_GRID:
            self.assertTrue((pinball(y, q, t) >= 0).all())

    def test_crps_dense_grid_matches_gaussian_closed_form(self):
        # Forecast N(mu, s), realisation y. Closed form (Gneiting & Raftery 2007):
        # CRPS = s * [ z(2Phi(z)-1) + 2phi(z) - 1/sqrt(pi) ],  z = (y-mu)/s
        mu, s = 0.3, 1.7
        y = self.rng.normal(size=500) * 2
        z = (y - mu) / s
        closed = s * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))
        taus = np.linspace(0.0005, 0.9995, 2000)
        Q = mu + s * stats.norm.ppf(taus)[None, :].repeat(y.size, 0)
        est = crps_from_quantiles(y, Q, taus, "trapezoid")
        np.testing.assert_allclose(est.mean(), closed.mean(), rtol=2e-3)

    def test_crps_weights(self):
        self.assertAlmostEqual(crps_weights(TAU_GRID, "trapezoid").sum(), 0.98, places=12)
        self.assertAlmostEqual(crps_weights(TAU_GRID, "uniform").sum(), 1.0, places=12)
        with self.assertRaises(ValueError):
            crps_weights([0.5, 0.25], "trapezoid")      # not increasing


if __name__ == "__main__":
    unittest.main()
