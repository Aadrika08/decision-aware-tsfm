"""Omega-only simulation: does the innovation law move Omega, or the match rate?

    PYTHONPATH=src python -m decision_aware_tsfm.analysis.experiments.e3.omega_only \
        --out artifacts/runs/e3_matched_omega_rev1

Added on 2026-10-04 after an independent review. The first E3 note said that
failed Omega matches were "sampling noise in Omega, not an effect of the
innovation law". That was asserted from 30 seeds per structure, which had no
power to tell. This module measures it directly. Nothing is scored, so many
seeds are cheap.

For every scored structure and every member it draws N_SIM_SEEDS fresh series
with the design's own generator (generator.make_series, full design length),
computes the team's Omega on the 2000-point context window, and compares each
member with the Gaussian reference member of the SAME seed. That reference
has its own independent random stream, exactly as in the E3 dataset, so the
gaussian_b row is the failure rate of two independent Gaussian draws.

The simulation seeds are disjoint from the design seeds (0 to 29). No design
constant is read back or changed here: the tolerance, members, structures and
lengths all come from design.py.

Synthetic series only. No forecaster and no model is involved.
"""
from __future__ import annotations

import argparse
import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from decision_aware_tsfm.omega.compute import compute_omega

from .design import (BASELINE, DEFAULT_SEEDS, MEMBERS, N_CONTEXT, N_TOTAL, OMEGA_TOLERANCE,
                     REFERENCE, SCORED_STRUCTURES, STRUCTURES)
from .generator import context_innovations, make_series
from .scoring import wilson_interval

SIM_SEED_START = 1000            # design seeds are 0 to 29; these do not overlap
N_SIM_SEEDS = 4000
ALPHA = 0.05
POWER_TARGET = 0.80
EXACT_POWER_MAX_N = 100          # exact McNemar power up to this many pairs, normal approximation above
N_DESIGN_SEEDS = len(DEFAULT_SEEDS)


def sim_seeds(n_seeds: int = N_SIM_SEEDS, start: int = SIM_SEED_START) -> np.ndarray:
    return np.arange(start, start + int(n_seeds), dtype="int64")


def simulate(seeds=None, structures=SCORED_STRUCTURES, members=None,
             n_context: int = N_CONTEXT, n_total: int = N_TOTAL) -> pd.DataFrame:
    """One row per (structure, member, seed): Omega on levels of the context
    window and the sample variance of the context innovations (relative to
    sigma squared; its population value is exactly 1 for every member)."""
    seeds = sim_seeds() if seeds is None else np.asarray(list(seeds), dtype="int64")
    members = list(MEMBERS) if members is None else list(members)
    parts = []
    for sname in structures:
        st = STRUCTURES[sname]
        for mname in members:
            mb = MEMBERS[mname]
            om, var = np.empty(seeds.size), np.empty(seeds.size)
            for i, seed in enumerate(seeds):
                x = make_series(st, mb, int(seed), n_total)
                om[i] = compute_omega(x[:n_context], diffs=False)
                var[i] = np.var(context_innovations(x, st, n_context)) / st.sigma ** 2
            parts.append(pd.DataFrame({"structure": sname, "member": mname, "seed": seeds,
                                       "omega_levels": om, "innov_variance": var}))
    return pd.concat(parts, ignore_index=True)


# ------------------------------------------------------------------ the tests

def mcnemar_exact(n10: int, n01: int) -> float:
    """Two-sided exact McNemar p-value: binomial test of the discordant pairs."""
    n = int(n10) + int(n01)
    if n == 0:
        return 1.0
    return float(stats.binomtest(int(n10), n, 0.5).pvalue)


@lru_cache(maxsize=None)
def _rejection_region(d: int, alpha: float) -> np.ndarray:
    """For d discordant pairs: which counts k reject under the exact two-sided
    binomial test (scipy's rule: sum the outcomes that are no more likely)."""
    pmf0 = stats.binom.pmf(np.arange(d + 1), d, 0.5)
    pval = np.array([pmf0[pmf0 <= pmf0[j] * (1 + 1e-7)].sum() for j in range(d + 1)])
    return pval <= alpha


def mcnemar_power(n_pairs: int, p10: float, p01: float, alpha: float = ALPHA) -> float:
    """Power of the two-sided McNemar test with n_pairs pairs when the two
    discordant outcomes have probabilities p10 and p01 per pair.

    Up to EXACT_POWER_MAX_N pairs: exact, by summing the rejection region of
    the exact binomial test over the number of discordant pairs. Above that:
    the usual normal approximation."""
    n_pairs, pd_ = int(n_pairs), float(p10 + p01)
    if pd_ <= 0:
        return 0.0
    if n_pairs <= EXACT_POWER_MAX_N:
        share = p10 / pd_
        power = 0.0
        for d in range(1, n_pairs + 1):
            reject = _rejection_region(d, alpha)
            if reject.any():
                k = np.flatnonzero(reject)
                power += stats.binom.pmf(d, n_pairs, pd_) * stats.binom.pmf(k, d, share).sum()
        return float(power)
    z = stats.norm.ppf(1 - alpha / 2)
    diff = abs(p10 - p01)
    den = np.sqrt(max(pd_ - diff ** 2, 1e-300))
    return float(stats.norm.cdf((diff * np.sqrt(n_pairs) - z * np.sqrt(pd_)) / den))


def detectable_rate_difference(n_pairs: int, base_rate: float, power: float = POWER_TARGET,
                               alpha: float = ALPHA) -> float:
    """Smallest increase in failure rate over `base_rate` that the McNemar test
    detects with the stated power at n_pairs pairs. Working assumption: the two
    failure indicators are independent. In E3 they share a reference series
    and are positively dependent, which makes fewer discordant pairs and more
    power, so this figure is on the cautious side. NaN if even a failure rate
    of 1 is not detectable."""
    def gap(d):
        p1 = base_rate + d
        return mcnemar_power(n_pairs, p1 * (1 - base_rate), base_rate * (1 - p1), alpha) - power
    hi = 1.0 - base_rate - 1e-9
    if gap(hi) < 0:
        return float("nan")
    lo = 1e-6
    for _ in range(60):                       # bisection; exact power is a step function of d
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if gap(mid) >= 0 else (mid, hi)
    return float(hi)


def summarise(draws: pd.DataFrame, tolerance: float = OMEGA_TOLERANCE,
              n_design_seeds: int = N_DESIGN_SEEDS) -> pd.DataFrame:
    """Per (structure, member): Omega mean and sd, match-failure rate against
    the independent Gaussian reference of the same seed, and three tests.

    Failure rate versus the Gaussian baseline (member gaussian_b): exact
    McNemar test, because both members are compared with the SAME reference
    series of each seed. power_run is the power of that test at the number of
    seeds simulated, power_design at the design's seeds per structure, both
    for the difference estimated here. detectable_* is the smallest rate
    difference each sample size detects with 80 percent power.

    Mean of Omega versus the reference: Welch t test (independent streams),
    with the mean shift detectable at 80 percent power.
    Spread of Omega versus the reference: Brown-Forsythe test (Levene on
    deviations from the median).
    """
    rows = []
    zsum = stats.norm.ppf(1 - ALPHA / 2) + stats.norm.ppf(POWER_TARGET)
    for sname, g in draws.groupby("structure", sort=False):
        w = g.pivot(index="seed", columns="member", values="omega_levels")
        v = g.pivot(index="seed", columns="member", values="innov_variance")
        ref = w[REFERENCE].to_numpy()
        n = len(w)
        fail = {m: np.abs(w[m].to_numpy() - ref) > tolerance for m in w.columns if m != REFERENCE}
        if BASELINE in fail:
            base_rate = float(fail[BASELINE].mean())
            det_run = detectable_rate_difference(n, base_rate)
            det_design = detectable_rate_difference(n_design_seeds, base_rate)
        for mname in [m for m in MEMBERS if m in w.columns]:
            om = w[mname].to_numpy()
            row = dict(structure=sname, member=mname, n_seeds=n, omega_mean=om.mean(),
                       omega_se=om.std(ddof=1) / np.sqrt(n), omega_sd=om.std(ddof=1),
                       innov_variance_mean=v[mname].mean(), innov_variance_sd=v[mname].std(ddof=1),
                       corr_omega_innov_variance=float(np.corrcoef(om, v[mname].to_numpy())[0, 1]))
            if mname != REFERENCE:
                f = fail[mname]
                lo, hi = wilson_interval(int(f.sum()), n)
                row.update(n_failed=int(f.sum()), fail_rate=f.mean(), fail_lo=lo, fail_hi=hi,
                           mean_diff_vs_reference=om.mean() - ref.mean(),
                           welch_p=float(stats.ttest_ind(om, ref, equal_var=False).pvalue),
                           mean_diff_detectable=zsum * np.sqrt(om.var(ddof=1) / n + ref.var(ddof=1) / n),
                           sd_ratio_vs_reference=om.std(ddof=1) / ref.std(ddof=1),
                           brown_forsythe_p=float(stats.levene(om, ref, center="median").pvalue))
                if mname != BASELINE and BASELINE in fail:
                    b = fail[BASELINE]
                    n10, n01 = int((f & ~b).sum()), int((~f & b).sum())
                    row.update(fail_rate_baseline=b.mean(), fail_rate_diff=f.mean() - b.mean(),
                               n_member_only_fails=n10, n_baseline_only_fails=n01,
                               mcnemar_p=mcnemar_exact(n10, n01),
                               power_run=mcnemar_power(n, n10 / n, n01 / n),
                               power_design=mcnemar_power(n_design_seeds, n10 / n, n01 / n),
                               detectable_diff_run=det_run, detectable_diff_design=det_design)
            rows.append(row)
    return pd.DataFrame(rows)


def tolerance_in_sd_units(draws: pd.DataFrame, tolerance: float = OMEGA_TOLERANCE) -> pd.DataFrame:
    """Per structure, Gaussian members only: what the absolute tolerance means
    in units of Omega's own sampling noise. The match compares two independent
    draws, so the relevant standard deviation is that of the DIFFERENCE."""
    rows = []
    for sname, g in draws.groupby("structure", sort=False):
        w = g.pivot(index="seed", columns="member", values="omega_levels")
        d = (w[BASELINE] - w[REFERENCE]).to_numpy()
        sd_one, sd_diff = w[REFERENCE].std(ddof=1), d.std(ddof=1)
        rows.append(dict(structure=sname, n_seeds=len(w), omega_mean_gaussian=w[REFERENCE].mean(),
                         omega_sd_gaussian=sd_one, sd_of_difference_of_two_draws=sd_diff,
                         tolerance=tolerance, tolerance_in_single_draw_sd=tolerance / sd_one,
                         tolerance_in_difference_sd=tolerance / sd_diff,
                         gaussian_pair_fail_rate=float((np.abs(d) > tolerance).mean()),
                         normal_theory_fail_rate=float(2 * stats.norm.sf(tolerance / sd_diff)),
                         tolerance_for_2_difference_sd=2 * sd_diff))
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-seeds", type=int, default=N_SIM_SEEDS)
    ap.add_argument("--seed-start", type=int, default=SIM_SEED_START)
    a = ap.parse_args(argv)
    if a.seed_start <= max(DEFAULT_SEEDS):
        raise SystemExit("simulation seeds must not overlap the design seeds")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    t0 = time.time()
    draws = simulate(sim_seeds(a.n_seeds, a.seed_start))
    table, tol = summarise(draws), tolerance_in_sd_units(draws)
    draws.to_csv(out / "omega_only_draws.csv", index=False, lineterminator="\n")
    table.to_csv(out / "omega_only_by_member.csv", index=False, lineterminator="\n")
    tol.to_csv(out / "omega_tolerance_in_sd_units.csv", index=False, lineterminator="\n")
    manifest = dict(command="python -m decision_aware_tsfm.analysis.experiments.e3.omega_only",
                    seeds=[int(a.seed_start), int(a.seed_start + a.n_seeds - 1)], n_seeds=int(a.n_seeds),
                    n_context=N_CONTEXT, n_total=N_TOTAL, omega_tolerance=OMEGA_TOLERANCE,
                    structures=list(SCORED_STRUCTURES), members=list(MEMBERS), alpha=ALPHA,
                    power_target=POWER_TARGET, n_design_seeds=N_DESIGN_SEEDS,
                    runtime_seconds=round(time.time() - t0, 1),
                    label="synthetic series, Omega only; nothing is scored and no model is run")
    (out / "omega_only_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"OMEGA ONLY  {a.n_seeds} seeds per (structure, member), context {N_CONTEXT}, "
          f"tolerance {OMEGA_TOLERANCE}")
    print(table.round(4).to_string(index=False))
    print(tol.round(4).to_string(index=False))
    print(f"wrote omega_only_* tables to {out}  ({manifest['runtime_seconds']} s)")


if __name__ == "__main__":
    main()
