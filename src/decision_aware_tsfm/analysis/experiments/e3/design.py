"""E3 design constants. Everything in this file was fixed BEFORE the Part B
dataset was generated and before any forecaster was scored. Do not change a
value here after looking at E3 outcomes; a change needs sign-off and a new
dated note.

What had been seen when these values were chosen: only Part A
(omega_reconcile.py), i.e. Omega of a Gaussian noisy sine at n = 1000 and 2000.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

# --------------------------------------------------------------------------
# Lengths. Time convention is the engine's: origin = position of the first
# forecast target, target position = origin + step - 1.
# --------------------------------------------------------------------------
# Context: the history a selector may see. Omega is computed on exactly these
# points. The source paper treats Omega as unreliable below 1000 steps; 2000
# also matches the length used by the existing k3 generator.
N_CONTEXT = 2000
# Test: one-step-ahead targets at positions N_CONTEXT .. N_CONTEXT + N_TEST - 1.
N_TEST = 1000
N_TOTAL = N_CONTEXT + N_TEST
HORIZON = 1
BURN_IN = 500                  # AR(1) warm-up draws discarded before position 0

# --------------------------------------------------------------------------
# Omega tolerance (levels Omega on the context window).
#
# Reasoning, written before generating: Part A measured the seed-to-seed
# standard deviation of Omega for one Gaussian noisy sine at n = 2000 as about
# 0.015 (noise sd 1.0). Two series whose Omega differ by less than about two
# such standard deviations cannot be told apart by Omega beyond its own
# sampling noise, so 0.03 is the smallest tolerance that is meaningful at this
# length. It is also small against the range Omega spans on the proposal's
# reference signals (about 0.06 to 0.87). The price, accepted in advance, is
# that some honest same-process pairs will fail the check; those are reported
# as failed matches and are not regenerated.
#
# CORRECTION, 2026-10-04 (after independent review; the text above is left as
# it was written, and the value below is NOT changed, because changing it
# after seeing results would be post hoc tuning):
#   1. The reasoning above is statistically wrong. The match compares TWO
#      independent draws. If one draw has standard deviation s, the difference
#      of two has standard deviation s * sqrt(2), so 0.03 is about 1.4, not 2,
#      standard deviations of the quantity the check actually looks at.
#   2. The 0.015 came from Part A's single unit-amplitude sine, not from the
#      E3 structures, and one absolute number cannot fit structures whose
#      Omega noise differs by a factor of about five. Measured with
#      omega_only.py (4000 fresh seeds, Gaussian members, context 2000):
#          structure     sd of one draw   sd of difference   0.03 in difference-sd   Gaussian pairs failing
#          ar1_phi0.50       0.0066           0.0093                3.2                    0.15 %
#          ar1_phi0.80       0.0142           0.0199                1.5                   12.7 %
#          ar1_phi0.95       0.0304           0.0430                0.7                   49.3 %
#          sine_sd1.0        0.0141           0.0202                1.5                   14.1 %
#   3. "Sampling noise" above assumed the noise is the same for every member.
#      It is not: on the sine structure student_t3 has 2.5 times the Omega
#      standard deviation of the Gaussian and fails the match 40 % of the time.
#   4. A better rule to have registered: a tolerance per structure, stated in
#      units of the standard deviation of the difference of two Gaussian
#      draws of that structure (for example 2 such units), measured before
#      generating the dataset.
# See docs/results/e3/2026-10-04-e3-matched-omega.md.
# --------------------------------------------------------------------------
OMEGA_TOLERANCE = 0.03
OMEGA_MATCH_COLUMN = "omega_levels"      # first-difference Omega is recorded, not gated on

# Population innovation variance is exactly 1 for every member. This bound is
# for the analytic variance of the standardised distributions.
INNOVATION_VARIANCE_ATOL = 1e-9

DEFAULT_SEEDS = tuple(range(30))


# --------------------------------------------------------------------------
# Structures: the deterministic or autoregressive part shared by a group.
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Structure:
    name: str
    kind: str                  # "ar1" or "sine"
    code: int                  # stable integer used in the RNG seed sequence
    phi: float = 0.0           # ar1: x_t = phi * x_{t-1} + sigma * e_t
    sigma: float = 1.0         # innovation (ar1) or additive-noise (sine) standard deviation
    scored: bool = True        # False: used only as a negative control


def sine_signal(n: int) -> np.ndarray:
    """Same base signal as analysis/k3/matched_omega_pair.py (daily plus weekly cycle)."""
    t = np.arange(n)
    return np.sin(2 * np.pi * t / 24) + 0.5 * np.sin(2 * np.pi * t / (24 * 7))


STRUCTURES: dict[str, Structure] = {s.name: s for s in (
    Structure("ar1_phi0.50", "ar1", 1, phi=0.50),
    Structure("ar1_phi0.80", "ar1", 2, phi=0.80),
    Structure("ar1_phi0.95", "ar1", 3, phi=0.95),
    Structure("sine_sd1.0", "sine", 4, sigma=1.0),
    Structure("sine_sd2.0", "sine", 5, sigma=2.0, scored=False),
)}
SCORED_STRUCTURES = tuple(n for n, s in STRUCTURES.items() if s.scored)


# --------------------------------------------------------------------------
# Members: innovation distributions, each standardised analytically to mean 0
# and variance 1 (population values, not sample values).
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Member:
    name: str
    kind: str                  # gaussian | skewnorm | student_t
    code: int
    param: float = float("nan")   # skew-normal shape a, or Student t degrees of freedom
    role: str = "alternative"     # reference | baseline | alternative


MEMBERS: dict[str, Member] = {m.name: m for m in (
    Member("gaussian", "gaussian", 0, role="reference"),
    # an independent second Gaussian draw: the noise-only baseline for sign differences
    Member("gaussian_b", "gaussian", 1, role="baseline"),
    Member("skewnorm_p2", "skewnorm", 2, 2.0),      # right-skewed, moderate
    Member("skewnorm_p8", "skewnorm", 3, 8.0),      # right-skewed, strong (the k3 generator's value)
    Member("skewnorm_n2", "skewnorm", 4, -2.0),     # left-skewed, moderate
    Member("skewnorm_n8", "skewnorm", 5, -8.0),     # left-skewed, strong
    Member("student_t5", "student_t", 6, 5.0),      # heavy tails, finite kurtosis
    Member("student_t3", "student_t", 7, 3.0),      # heavier tails; variance exists, kurtosis does not
)}
REFERENCE = "gaussian"
BASELINE = "gaussian_b"
ALTERNATIVES = tuple(n for n, m in MEMBERS.items() if m.role == "alternative")


def innovation_dist(member: Member):
    """Frozen scipy distribution with mean exactly 0 and variance exactly 1."""
    if member.kind == "gaussian":
        return stats.norm(0.0, 1.0)
    if member.kind == "skewnorm":
        a = float(member.param)
        d = a / np.sqrt(1.0 + a * a)
        mean = d * np.sqrt(2.0 / np.pi)
        sd = np.sqrt(1.0 - 2.0 * d * d / np.pi)
        return stats.skewnorm(a, loc=-mean / sd, scale=1.0 / sd)
    if member.kind == "student_t":
        df = float(member.param)
        if not df > 2.0:
            raise ValueError(f"Student t needs df > 2 for the variance to exist, got {df}")
        return stats.t(df, loc=0.0, scale=np.sqrt((df - 2.0) / df))
    raise ValueError(f"unknown innovation kind {member.kind!r}")


# --------------------------------------------------------------------------
# Idealised forecasters. The engine only accepts the family labels
# "classical" and "tsfm"; here they are SLOT LABELS for two idealised
# forecasters and say nothing about real classical models or real TSFMs.
#
#   G  (slot "classical"): true conditional mean, true scale, Gaussian shape:
#          q_tau = m_t + sigma * z_tau
#   F  (slot "tsfm"): true conditional mean, a quantile shape that follows the
#          innovation distribution, plus one stated imperfection.
#
# Variants of F, all declared here in advance and all reported:
#   primary      true-shape quantiles shifted by a fixed location error of 0.25 sigma
#   loc_small    same with 0.10 sigma     (sensitivity, reported in full)
#   loc_large    same with 0.50 sigma     (sensitivity, reported in full)
#   empirical    empirical quantiles of the context-window innovations, no
#                knowledge of the true distribution (the alternative design)
#
# NOTE ADDED 2026-10-04 (after independent review; definitions unchanged):
# both forecasters are handed the true conditional mean and scale, so their
# costs depend only on the innovation law, tau and the location error above.
# Which one is cheaper is therefore decided by these definitions, in closed
# form, before any series exists (population.py), and does not depend on the
# structure, on Omega or on the Omega match. The location error is a free
# parameter: F is the population winner in 18 of 66 (alternative member, tau)
# cells at 0.25, in 41 at 0.10 and in 3 at 0.50, and several break-even
# errors lie within 0.05 of 0.25. There is no record of when 0.25 was chosen
# relative to those break-evens (the tree has no version history).
# --------------------------------------------------------------------------
MODEL_G = "ideal_gaussian_G"
FAMILY_SLOT = {"G": "classical", "F": "tsfm"}


@dataclass(frozen=True)
class FVariant:
    name: str
    model: str
    kind: str                  # "location" or "empirical"
    location_error: float = 0.0    # in units of sigma, added to every quantile


F_VARIANTS: dict[str, FVariant] = {v.name: v for v in (
    FVariant("primary", "ideal_flexible_F_loc0.25", "location", 0.25),
    FVariant("loc_small", "ideal_flexible_F_loc0.10", "location", 0.10),
    FVariant("loc_large", "ideal_flexible_F_loc0.50", "location", 0.50),
    FVariant("empirical", "ideal_flexible_F_empirical", "empirical"),
)}
PRIMARY_VARIANT = "primary"

DATASET = "e3_synthetic"       # label carried through the engine; synthetic, not a real dataset
CONFIG = "h1"
