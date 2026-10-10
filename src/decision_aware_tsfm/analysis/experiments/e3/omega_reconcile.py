"""Part A of E3: reconcile the three Omega values quoted for a "moderately noisy" sine.

Three numbers were in circulation:
  0.32          proposal target for a "moderately noisy" signal
  about 0.4864  scripts/validate_omega_synthetics.py (period 24, noise sd 0.5, n = 1000, seed 42)
  0.13 to 0.15  the sanity check in analysis/k3/matched_omega_pair.py (noise sd 1.5, n = 2000, unseeded)

All of them are Omega of  x_t = sin(2 pi t / 24) + sd * e_t,  e_t standard normal,
so they differ only in the noise sd, the series length and the random draw.
This module sweeps noise sd at both lengths with the team's compute_omega.

Nothing here is a research result. It is a calibration of a synthetic signal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from decision_aware_tsfm.omega.compute import compute_omega

PERIOD = 24
LENGTHS = (1000, 2000)
N_SEEDS = 200                      # seeds 0 .. N_SEEDS - 1, numpy default_rng
SD_GRID = tuple(round(0.05 * k, 2) for k in range(2, 61))      # 0.10, 0.15, ..., 3.00
TABLE_SDS = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)        # rows shown in the note
PROPOSAL_TARGET = 0.32


def noisy_sine(n: int, sd: float, rng: np.random.Generator, period: int = PERIOD) -> np.ndarray:
    t = np.arange(n)
    return np.sin(2 * np.pi * t / period) + sd * rng.standard_normal(n)


def omega_sweep(lengths=LENGTHS, sds=SD_GRID, n_seeds: int = N_SEEDS) -> pd.DataFrame:
    """Mean, sd, min and max of Omega (levels) over seeds for every (n, noise sd)."""
    rows = []
    for n in lengths:
        for sd in sds:
            om = np.array([compute_omega(noisy_sine(n, sd, np.random.default_rng(s)))
                           for s in range(n_seeds)])
            rows.append(dict(n=int(n), noise_sd=float(sd), n_seeds=int(n_seeds),
                             omega_mean=om.mean(), omega_sd=om.std(ddof=1),
                             omega_min=om.min(), omega_max=om.max()))
    return pd.DataFrame(rows)


def sd_for_target(sweep: pd.DataFrame, n: int, target: float = PROPOSAL_TARGET) -> float:
    """Noise sd at which the seed-mean Omega crosses `target`, by linear
    interpolation between the two neighbouring grid points. NaN if the mean
    curve never crosses the target on the grid."""
    d = sweep[sweep["n"] == n].sort_values("noise_sd")
    sd, om = d["noise_sd"].to_numpy(), d["omega_mean"].to_numpy()
    for k in range(len(sd) - 1):
        hi, lo = om[k], om[k + 1]                 # Omega falls as noise grows
        if hi >= target >= lo and hi != lo:
            return float(sd[k] + (hi - target) / (hi - lo) * (sd[k + 1] - sd[k]))
    return float("nan")


def reproduce_dayson(n: int = 1000, seed: int = 42) -> float:
    """Same draws, in the same order, as scripts/validate_omega_synthetics.py:
    the legacy global RNG is seeded once, n values are consumed by the
    "pure sine" signal, and the NEXT n values are the sd 0.5 noise."""
    state = np.random.RandomState(seed)           # same stream as np.random.seed(seed)
    t = np.arange(n)
    state.normal(0, 0.05, n)                      # consumed by the pure-sine signal
    return compute_omega(np.sin(2 * np.pi * t / PERIOD) + state.normal(0, 0.5, n))


def reproduce_abdullah(n_seeds: int = N_SEEDS, n: int = 2000, sd: float = 1.5) -> dict:
    """The k3 sanity check is unseeded, so there is no single value to
    reproduce. This reports the distribution over seeds of the same signal."""
    om = np.array([compute_omega(noisy_sine(n, sd, np.random.default_rng(s))) for s in range(n_seeds)])
    return dict(n=n, noise_sd=sd, n_seeds=n_seeds, omega_mean=float(om.mean()),
                omega_sd=float(om.std(ddof=1)), omega_min=float(om.min()), omega_max=float(om.max()),
                share_in_0p13_0p15=float(((om >= 0.13) & (om <= 0.15)).mean()))


def reconcile() -> dict:
    sweep = omega_sweep()
    return dict(sweep=sweep,
                table=sweep[sweep["noise_sd"].isin(TABLE_SDS)].reset_index(drop=True),
                sd_for_target={int(n): sd_for_target(sweep, n) for n in LENGTHS},
                dayson=reproduce_dayson(), abdullah=reproduce_abdullah())
