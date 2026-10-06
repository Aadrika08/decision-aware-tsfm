"""The K2 statistic: relative family difference, its rank correlation with Omega,
a bootstrap interval for that correlation, and sign counts.

The headline statistic is unchanged from the existing record:

    Delta = 100 * (score(A) - score(B)) / score(A)        negative = A better
    rho   = Spearman rank correlation of Omega and Delta across units
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

DEFAULT_RESAMPLES = 10_000
DEFAULT_SEED = 20261004
DEFAULT_LEVEL = 0.95
MIN_UNITS = 3
MAX_SEED = 2**63 - 1
# A table cell is flagged when more than this share of its resamples had no
# defined correlation and were left out of the percentiles.
DEGENERATE_WARN_FRACTION = 0.05


def check_bootstrap_parameters(n_resamples, seed) -> tuple[int, int]:
    """Return (n_resamples, seed) as ints, or raise ValueError saying what is wrong."""
    for name, value in (("number of bootstrap resamples", n_resamples), ("bootstrap seed", seed)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"the {name} must be a whole number, got {value!r}")
    if n_resamples < 1:
        raise ValueError(f"the number of bootstrap resamples must be at least 1, got {n_resamples}")
    if not 0 <= seed <= MAX_SEED:
        raise ValueError(f"the bootstrap seed must be between 0 and {MAX_SEED}, got {seed}")
    return int(n_resamples), int(seed)


def relative_delta(a, b):
    """100 * (a - b) / a. Negative means family A has the lower error."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 100.0 * (a - b) / a


def finite_pairs(tab: pd.DataFrame, family_a: str, family_b: str):
    """Omega and Delta for the rows where both are finite, plus the row mask."""
    d = relative_delta(tab[family_a], tab[family_b])
    omega = np.asarray(tab["omega"], dtype=float)
    ok = np.isfinite(d) & np.isfinite(omega)
    return omega[ok], d[ok], ok


def spearman_xy(x, y) -> tuple[float, float, int]:
    """(rho, two-sided p, n). Fewer than three pairs gives (nan, nan, n)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = int(len(x))
    if n < MIN_UNITS:
        return np.nan, np.nan, n
    rho, p = stats.spearmanr(x, y)
    return float(rho), float(p), n


def spearman(tab: pd.DataFrame, family_a: str, family_b: str) -> tuple[float, float, int]:
    """Spearman correlation of tab['omega'] with Delta(family_a, family_b)."""
    omega, d, _ = finite_pairs(tab, family_a, family_b)
    return spearman_xy(omega, d)


def _resample_weights(codes: np.ndarray, n_clusters: int, n_resamples: int,
                      seed: int) -> np.ndarray:
    """How many times each row appears in each resample (n_resamples x n_rows).

    Clusters are drawn with replacement, n_clusters at a time; a drawn cluster
    brings all of its rows.
    """
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_clusters, size=(n_resamples, n_clusters))
    cluster_counts = np.zeros((n_resamples, n_clusters), dtype=np.int64)
    np.add.at(cluster_counts, (np.arange(n_resamples)[:, None], draws), 1)
    return cluster_counts[:, codes]


def _weighted_average_ranks(v: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Average rank of each original value inside each resample.

    rank = (number of resampled values below it) + (number equal to it + 1) / 2,
    which is the usual average rank with ties. Values with weight 0 get a rank
    too, but they carry no weight in the correlation.
    """
    less = (v[None, :] < v[:, None]).astype(float)       # less[i, j] = v[j] < v[i]
    equal = (v[None, :] == v[:, None]).astype(float)
    return w @ less.T + (w @ equal.T + 1.0) / 2.0


def bootstrap_rhos(x, y, n_resamples: int = DEFAULT_RESAMPLES, seed: int = DEFAULT_SEED,
                   clusters=None) -> np.ndarray:
    """Spearman rho on each bootstrap resample; nan where it is undefined.

    Units (or clusters of rows, if given) are resampled with replacement and
    the ranks are recomputed inside every resample.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if clusters is None:
        codes = np.arange(len(x))
        n_clusters = len(x)
    else:
        codes, uniques = pd.factorize(np.asarray(clusters))
        n_clusters = len(uniques)
    w = _resample_weights(codes, n_clusters, n_resamples, seed).astype(float)
    rx = _weighted_average_ranks(x, w)
    ry = _weighted_average_ranks(y, w)
    n = w.sum(axis=1)
    sx, sy = (w * rx).sum(axis=1), (w * ry).sum(axis=1)
    # Ranks are multiples of 0.5 and weights are integers, so these sums are exact
    # in floating point and a constant variable gives exactly zero.
    cov = n * (w * rx * ry).sum(axis=1) - sx * sy
    vx = n * (w * rx * rx).sum(axis=1) - sx * sx
    vy = n * (w * ry * ry).sum(axis=1) - sy * sy
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = cov / np.sqrt(vx * vy)
    rho[(vx <= 0) | (vy <= 0)] = np.nan
    return rho


def bootstrap_interval(x, y, n_resamples: int = DEFAULT_RESAMPLES, seed: int = DEFAULT_SEED,
                       level: float = DEFAULT_LEVEL, clusters=None) -> dict:
    """Percentile bootstrap interval for Spearman rho.

    Returns ci_low, ci_high, the number of resamples, how many of them were
    degenerate (rho undefined because Omega or Delta was constant in the
    resample; these are left out of the percentiles) and the number of
    resampled units.

    Raises ValueError if x or y contains a value that is not finite (filter
    with finite_pairs first; a NaN would otherwise be ranked as if it were a
    number), if their lengths differ, or if n_resamples or seed is not usable.
    With fewer than three units there is no interval and ci_low, ci_high are NaN.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n_resamples, seed = check_bootstrap_parameters(n_resamples, seed)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError(f"x and y must be one-dimensional and equally long, got {x.shape} and {y.shape}")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError(
            f"bootstrap_interval needs finite values; got {int((~np.isfinite(x)).sum())} non-finite "
            f"in x and {int((~np.isfinite(y)).sum())} in y")
    if clusters is not None and len(np.asarray(clusters)) != len(x):
        raise ValueError("clusters must have one label per row")
    n_units = len(x) if clusters is None else int(pd.Series(np.asarray(clusters)).nunique())
    out = {"ci_low": np.nan, "ci_high": np.nan, "ci_level": level,
           "n_resamples": int(n_resamples), "n_degenerate": 0,
           "n_resampled_units": int(n_units), "seed": int(seed)}
    if len(x) < MIN_UNITS or n_units < MIN_UNITS:
        return out
    rhos = bootstrap_rhos(x, y, n_resamples, seed, clusters)
    good = rhos[np.isfinite(rhos)]
    out["n_degenerate"] = int(len(rhos) - len(good))
    if len(good):
        tail = 100.0 * (1.0 - level) / 2.0
        lo, hi = np.percentile(good, [tail, 100.0 - tail])
        out["ci_low"], out["ci_high"] = float(lo), float(hi)
    return out


def sign_counts(tab: pd.DataFrame, family_a: str, family_b: str) -> dict:
    """On how many units family A has the lower error than family B."""
    a = np.asarray(tab[family_a], dtype=float)
    b = np.asarray(tab[family_b], dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    return {"n": int(ok.sum()),
            "n_a_lower": int((a[ok] < b[ok]).sum()),
            "n_b_lower": int((a[ok] > b[ok]).sum()),
            "n_tie": int((a[ok] == b[ok]).sum())}
