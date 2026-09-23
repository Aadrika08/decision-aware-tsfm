"""Mathematical invariants of the scoring layer. Numbers here are made up."""
import numpy as np
import pytest
from scipy import stats

from decision_aware_tsfm.analysis.decision.metrics import (
    TAU_GRID, check_taus, crps_from_quantiles, crps_weights, decision_cost,
    pinball, tau_from_costs,
)

RNG = np.random.default_rng(20260924)


def test_pinball_at_half_is_half_absolute_error():
    y = RNG.normal(size=20_000) * 5
    q = RNG.normal(size=20_000) * 5
    np.testing.assert_allclose(pinball(y, q, 0.5), 0.5 * np.abs(y - q), rtol=0, atol=1e-12)


def test_decision_cost_equals_scaled_pinball():
    y = RNG.normal(size=20_000) * 3
    q = RNG.normal(size=20_000) * 3
    cu = RNG.uniform(0.1, 10, size=20_000)
    co = RNG.uniform(0.1, 10, size=20_000)
    tau = cu / (cu + co)
    direct = cu * np.maximum(y - q, 0) + co * np.maximum(q - y, 0)
    np.testing.assert_allclose(direct, (cu + co) * pinball(y, q, tau), rtol=1e-12, atol=1e-12)
    # and the scalar-cost function agrees with itself on one pair
    np.testing.assert_allclose(decision_cost(y, q, 3.0, 1.0),
                               4.0 * pinball(y, q, tau_from_costs(3.0, 1.0)), atol=1e-12)


@pytest.mark.parametrize("tau", TAU_GRID)
def test_optimal_action_is_the_tau_quantile(tau):
    y = RNG.standard_t(df=4, size=4001)          # skewless but heavy-tailed; any sample works
    grid = np.linspace(y.min(), y.max(), 20_001)
    mean_loss = np.array([pinball(y, g, tau).mean() for g in grid[::10]])
    best = grid[::10][mean_loss.argmin()]
    # empirical tau-quantile of the sample (any point in the flat minimising set is optimal)
    lo = np.quantile(y, tau, method="inverted_cdf")
    hi = np.quantile(y, tau, method="higher")
    step = grid[10] - grid[0]
    assert lo - step <= best <= hi + step
    # the quantile itself is at least as good as every grid point
    assert pinball(y, lo, tau).mean() <= mean_loss.min() + 1e-12


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.2, np.nan, np.inf])
def test_tau_outside_open_unit_interval_fails(bad):
    with pytest.raises(ValueError):
        check_taus(bad)
    with pytest.raises(ValueError):
        pinball(1.0, 0.0, bad)


def test_nan_quantile_fails():
    with pytest.raises(ValueError):
        pinball([1.0, 2.0], [0.5, np.nan], 0.5)


def test_regret_components_nonnegative_pinball():
    y = RNG.normal(size=1000); q = RNG.normal(size=1000)
    for t in TAU_GRID:
        assert (pinball(y, q, t) >= 0).all()


def test_crps_dense_grid_matches_gaussian_closed_form():
    # Forecast N(mu, s), realisation y. Closed form (Gneiting & Raftery 2007):
    # CRPS = s * [ z(2Phi(z)-1) + 2phi(z) - 1/sqrt(pi) ],  z = (y-mu)/s
    mu, s = 0.3, 1.7
    y = RNG.normal(size=500) * 2
    z = (y - mu) / s
    closed = s * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))
    taus = np.linspace(0.0005, 0.9995, 2000)
    Q = mu + s * stats.norm.ppf(taus)[None, :].repeat(y.size, 0)
    est = crps_from_quantiles(y, Q, taus, "trapezoid")
    np.testing.assert_allclose(est.mean(), closed.mean(), rtol=2e-3)


def test_crps_weights():
    w = crps_weights(TAU_GRID, "trapezoid")
    assert w.sum() == pytest.approx(0.98, abs=1e-12)
    assert crps_weights(TAU_GRID, "uniform").sum() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        crps_weights([0.5, 0.25], "trapezoid")      # not increasing
