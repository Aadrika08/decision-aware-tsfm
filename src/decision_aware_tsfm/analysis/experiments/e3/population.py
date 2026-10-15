"""Closed-form (population) answer to the E3 question, with no simulated data.

Added on 2026-10-04 after an independent review. Both idealised forecasters
are handed the true conditional mean and the true scale, so the cost of each
depends only on the innovation law and on where the forecaster puts its
quantile relative to the innovation. In units of sigma:

    G puts its tau-quantile at   z_tau                (Gaussian quantile)
    F puts its tau-quantile at   Q(tau) + b           (true quantile plus location error b)

Let g(c) = E[pinball_tau(e - c)] be the expected cost of putting the quantile
at c. Its derivative is g'(c) = CDF(c) - tau, so g is convex with its minimum
at the true quantile Q(tau), and

    J_F - J_G = g(Q + b) - g(z) = integral from z to Q + b of (CDF(x) - tau) dx.

The structure (phi, sine), the series, Omega and the Omega match do not appear
anywhere in this expression. The lower-cost forecaster is therefore decided
by three things only: the innovation law, tau, and the error size b.

BREAK-EVEN ERROR. For each (member, tau) there is a b* >= 0 such that F has
the lower expected cost exactly when 0 <= b < b*:
    z > Q : b* = z - Q exactly (F and G then sit at the same point);
    z < Q : b* solves g(Q + b*) = g(z) on the far side of the minimum;
    z = Q : b* = 0, F cannot win with any error (every Gaussian cell, and the
            median of a symmetric law).
The same is computed for an error of the opposite sign (breakeven_neg), since
the sign of F's error was an arbitrary choice.

Nothing here uses the engine or any generated series.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats

from decision_aware_tsfm.analysis.decision.metrics import TAU_GRID

from .design import ALTERNATIVES, F_VARIANTS, MEMBERS, innovation_dist

_GL_X, _GL_W = np.polynomial.legendre.leggauss(96)
ERROR_GRID = tuple(round(0.05 * k, 2) for k in range(16))        # 0.00, 0.05, ..., 0.75
QUANTILE_TIE_ATOL = 1e-12


def cost_gap(dist, a: float, b: float, tau: float) -> float:
    """g(b) - g(a) = integral from a to b of (CDF(x) - tau) dx, by Gauss-Legendre
    quadrature on the finite interval (the integrand is smooth)."""
    half, mid = 0.5 * (b - a), 0.5 * (a + b)
    return float(half * np.sum(_GL_W * (dist.cdf(mid + half * _GL_X) - tau)))


def breakeven_error(dist, tau: float, sign: int = 1) -> float:
    """Largest location error (same sign as `sign`, returned as a magnitude,
    sigma units) below which F = true quantile + error has lower expected
    pinball cost than G = Gaussian quantile. 0 means F never wins on that side."""
    z, q = float(stats.norm.ppf(tau)), float(dist.ppf(tau))
    if abs(z - q) <= QUANTILE_TIE_ATOL:
        return 0.0
    if (z - q) * sign > 0:                    # G's quantile lies on the side F errs towards
        return abs(z - q)
    f = lambda b: cost_gap(dist, z, q + sign * b, tau)       # < 0 while F is cheaper than G
    hi = abs(z - q)
    while f(hi) < 0:
        hi *= 2.0
    return float(optimize.brentq(f, 0.0, hi, xtol=1e-13, rtol=1e-13))


def breakeven_table(taus=TAU_GRID) -> pd.DataFrame:
    """One row per (member, tau), every member including the Gaussian ones."""
    rows = []
    for mname, mb in MEMBERS.items():
        dist = innovation_dist(mb)
        for tau in taus:
            z, q = float(stats.norm.ppf(tau)), float(dist.ppf(tau))
            rows.append(dict(member=mname, member_role=mb.role, tau=float(tau),
                             gaussian_quantile=z, true_quantile=q, g_shape_error=z - q,
                             breakeven_pos=breakeven_error(dist, tau, +1),
                             breakeven_neg=breakeven_error(dist, tau, -1)))
    return pd.DataFrame(rows)


def population_delta(member: str, tau: float, location_error: float) -> float:
    """J_F - J_G in units of sigma for a location-error F. Negative: F is cheaper."""
    dist = innovation_dist(MEMBERS[member])
    return cost_gap(dist, float(stats.norm.ppf(tau)), float(dist.ppf(tau)) + location_error, tau)


def f_win_counts(breakeven: pd.DataFrame, errors=ERROR_GRID) -> pd.DataFrame:
    """Number of (alternative member, tau) cells where F has the strictly lower
    expected cost, as a function of the size of F's location error, for both
    signs of the error. Ties go to G, as in the engine."""
    alt = breakeven[breakeven["member"].isin(ALTERNATIVES)]
    declared = {round(v.location_error, 2): v.name for v in F_VARIANTS.values() if v.kind == "location"}
    rows = []
    for err in errors:
        pos, neg = alt["breakeven_pos"] > err, alt["breakeven_neg"] > err
        tails = ~alt["tau"].between(0.25, 0.75)
        rows.append(dict(location_error=float(err), declared_variant=declared.get(round(err, 2), ""),
                         n_cells=len(alt), n_F_wins_positive_error=int(pos.sum()),
                         n_F_wins_positive_error_tails=int((pos & tails).sum()),
                         n_F_wins_positive_error_centre=int((pos & ~tails).sum()),
                         n_F_wins_negative_error=int(neg.sum())))
    return pd.DataFrame(rows)


def near_knife_edge(breakeven: pd.DataFrame, error: float, band: float = 0.05) -> pd.DataFrame:
    """Alternative cells whose break-even lies within `band` of the given error."""
    alt = breakeven[breakeven["member"].isin(ALTERNATIVES)]
    d = alt[(alt["breakeven_pos"] - error).abs() <= band]
    return d.assign(margin=d["breakeven_pos"] - error).sort_values("margin").reset_index(drop=True)
