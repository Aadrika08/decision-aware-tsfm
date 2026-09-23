"""Scoring functions. Pure numpy, no pandas, no I/O.

Conventions (fixed here, used everywhere else):
  u = y - q                                   (realised minus forecast)
  pinball_tau(u) = max(tau * u, (tau - 1) * u)
  decision cost C(q, y) = c_u * (y - q)+ + c_o * (q - y)+
                        = (c_u + c_o) * pinball_tau(y - q),  tau = c_u / (c_u + c_o)
"""
from __future__ import annotations

import numpy as np

# Pre-registered operational quantiles. Both sides of 0.5.
TAU_GRID: tuple[float, ...] = (0.01, 0.05, 0.09, 0.25, 0.40, 0.50,
                               0.60, 0.75, 0.91, 0.95, 0.99)


def check_taus(tau) -> np.ndarray:
    """Return tau as a float array; raise if any value is not strictly inside (0, 1)."""
    t = np.asarray(tau, dtype=float)
    if t.size == 0:
        raise ValueError("empty tau")
    bad = ~np.isfinite(t) | (t <= 0.0) | (t >= 1.0)
    if bad.any():
        raise ValueError(f"tau must be strictly between 0 and 1, got {t[bad].tolist()}")
    return t


def pinball(y, q, tau) -> np.ndarray:
    """Elementwise pinball (quantile) loss. Broadcasts like numpy."""
    t = check_taus(tau)
    y = np.asarray(y, dtype=float)
    q = np.asarray(q, dtype=float)
    if np.isnan(q).any():
        raise ValueError("quantile forecast contains NaN")
    u = y - q
    return np.maximum(t * u, (t - 1.0) * u)


def tau_from_costs(c_under: float, c_over: float) -> float:
    """Operational quantile for underage cost c_under and overage cost c_over."""
    if not (c_under > 0 and c_over > 0):
        raise ValueError("both costs must be strictly positive")
    return c_under / (c_under + c_over)


def decision_cost(y, q, c_under: float, c_over: float) -> np.ndarray:
    """Asymmetric linear decision cost, computed directly (not via pinball)."""
    if not (c_under > 0 and c_over > 0):
        raise ValueError("both costs must be strictly positive")
    y = np.asarray(y, dtype=float)
    q = np.asarray(q, dtype=float)
    return c_under * np.maximum(y - q, 0.0) + c_over * np.maximum(q - y, 0.0)


def crps_weights(taus, method: str = "trapezoid") -> np.ndarray:
    """Weights w_k so that CRPS ~= 2 * sum_k w_k * pinball_{tau_k}.

    trapezoid: trapezoid rule over [tau_min, tau_max]. Weights sum to
               tau_max - tau_min (0.98 on our grid). The tails outside the
               grid are not estimated.
    uniform:   w_k = 1/K. GIFT-Eval's convention on an evenly spaced grid.
               On our uneven grid it over-weights the tails, because 6 of the
               11 points sit at or beyond 0.25/0.75.
    """
    t = check_taus(taus)
    if np.any(np.diff(t) <= 0):
        raise ValueError("taus must be strictly increasing")
    if method == "uniform":
        return np.full(t.size, 1.0 / t.size)
    if method != "trapezoid":
        raise ValueError(f"unknown method {method!r}")
    if t.size < 2:
        raise ValueError("trapezoid needs at least two quantiles")
    w = np.empty_like(t)
    w[0] = (t[1] - t[0]) / 2
    w[-1] = (t[-1] - t[-2]) / 2
    w[1:-1] = (t[2:] - t[:-2]) / 2
    return w


def crps_from_quantiles(y, Q, taus, method: str = "trapezoid") -> np.ndarray:
    """Quantile-grid estimate of CRPS. Q has shape (..., K), one column per tau."""
    t = check_taus(taus)
    Q = np.asarray(Q, dtype=float)
    if Q.shape[-1] != t.size:
        raise ValueError(f"Q has {Q.shape[-1]} quantile columns, expected {t.size}")
    w = crps_weights(t, method)
    losses = pinball(np.asarray(y, dtype=float)[..., None], Q, t)
    return 2.0 * (losses * w).sum(axis=-1)
