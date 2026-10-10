"""Seeded generator for the E3 matched-Omega dataset, and the match check.

A GROUP is every member generated for one (structure, seed): same
deterministic or autoregressive structure, same length, same population
innovation variance, different innovation distribution. Each member has its
own independent random stream, so a pair of members differs by the innovation
distribution AND by sampling noise; the gaussian / gaussian_b pair differs by
sampling noise only and is the baseline.

The match check is made per PAIR (Gaussian reference, other member) and is
never used to filter what gets generated or persisted: a pair outside the
tolerance stays in every table with status "failed".

Re-implements the idea of analysis/k3/matched_omega_pair.py rather than
importing it; see the E3 note for the reason.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from scipy import stats

from decision_aware_tsfm.omega.compute import compute_omega

from .design import (BASELINE, BURN_IN, CONFIG, DATASET, DEFAULT_SEEDS, MEMBERS, N_CONTEXT,
                     N_TOTAL, OMEGA_MATCH_COLUMN, OMEGA_TOLERANCE, REFERENCE, STRUCTURES,
                     Member, Structure, innovation_dist, sine_signal)

TAIL_PROBS = (0.01, 0.25, 0.50, 0.75, 0.99)
# (first, second) structure of each negative control; Gaussian members, same seed
NEGATIVE_CONTROLS = (("ar1_phi0.50", "ar1_phi0.80"), ("ar1_phi0.80", "ar1_phi0.95"),
                     ("sine_sd1.0", "sine_sd2.0"))


def series_id(structure: str, seed: int, member: str) -> str:
    return f"{structure}/seed{int(seed):03d}/{member}"


def member_rng(seed: int, structure: Structure, member: Member) -> np.random.Generator:
    return np.random.default_rng([int(seed), structure.code, member.code])


def conditional_mean(values: np.ndarray, structure: Structure) -> np.ndarray:
    """True one-step conditional mean at every position. For AR(1), position 0
    has no predecessor in the stored series and is NaN (it is never a target)."""
    x = np.asarray(values, dtype=float)
    if structure.kind == "ar1":
        m = np.full(x.size, np.nan)
        m[1:] = structure.phi * x[:-1]
        return m
    if structure.kind == "sine":
        return sine_signal(x.size)
    raise ValueError(f"unknown structure kind {structure.kind!r}")


def make_series(structure: Structure, member: Member, seed: int, n_total: int = N_TOTAL) -> np.ndarray:
    """One series of length n_total. Innovations are i.i.d. draws from the
    member's distribution (population mean 0, variance 1) times structure.sigma."""
    rng = member_rng(seed, structure, member)
    dist = innovation_dist(member)
    if structure.kind == "ar1":
        e = structure.sigma * dist.rvs(size=BURN_IN + n_total, random_state=rng)
        x = sp_signal.lfilter([1.0], [1.0, -structure.phi], e)
        return x[BURN_IN:]
    if structure.kind == "sine":
        return sine_signal(n_total) + structure.sigma * dist.rvs(size=n_total, random_state=rng)
    raise ValueError(f"unknown structure kind {structure.kind!r}")


def context_innovations(values: np.ndarray, structure: Structure, n_context: int = N_CONTEXT) -> np.ndarray:
    """Innovations recoverable from the context window alone (positions < n_context)."""
    x = np.asarray(values, dtype=float)[:n_context]
    r = x - conditional_mean(x, structure)
    return r[np.isfinite(r)]


def tail_summary(a: np.ndarray) -> dict:
    """Quantile-based tail summary (defined for E3; exists even when kurtosis does not).

    tail_weight = (q99 - q01) / (q75 - q25)      3.449 for a Gaussian; larger = heavier tails
    tail_asym   = ((q99 - q50) - (q50 - q01)) / (q99 - q01)
                  0 for a symmetric law; positive = longer right tail
    """
    q01, q25, q50, q75, q99 = np.quantile(np.asarray(a, dtype=float), TAIL_PROBS)
    return dict(tail_weight=float((q99 - q01) / (q75 - q25)),
                tail_asym=float(((q99 - q50) - (q50 - q01)) / (q99 - q01)))


def describe_series(values: np.ndarray, structure: Structure, member: Member, seed: int,
                    n_context: int = N_CONTEXT) -> dict:
    """Everything persisted per series. All statistics use the context window only."""
    x = np.asarray(values, dtype=float)
    ctx = x[:n_context]
    innov = context_innovations(x, structure, n_context)
    ts, ti = tail_summary(ctx), tail_summary(innov)
    return dict(
        dataset=DATASET, config=CONFIG, series_id=series_id(structure.name, seed, member.name),
        structure=structure.name, structure_kind=structure.kind, phi=structure.phi,
        sigma=structure.sigma, member=member.name, innovation_kind=member.kind,
        innovation_param=member.param, member_role=member.role, seed=int(seed),
        rng_seed_sequence=f"[{int(seed)}, {structure.code}, {member.code}]",
        n_total=int(x.size), n_context=int(n_context), n_test=int(x.size - n_context),
        burn_in=BURN_IN if structure.kind == "ar1" else 0,
        omega_levels=compute_omega(ctx, diffs=False), omega_diffs=compute_omega(ctx, diffs=True),
        skewness=float(stats.skew(ctx)), excess_kurtosis=float(stats.kurtosis(ctx)),
        tail_weight=ts["tail_weight"], tail_asym=ts["tail_asym"],
        innov_variance=float(np.var(innov)), innov_skewness=float(stats.skew(innov)),
        innov_excess_kurtosis=float(stats.kurtosis(innov)),
        innov_tail_weight=ti["tail_weight"], innov_tail_asym=ti["tail_asym"])


def make_dataset(seeds=DEFAULT_SEEDS, structures=None, members=None,
                 n_total: int = N_TOTAL, n_context: int = N_CONTEXT):
    """Returns (values, series_table).

    values: long frame series_id, position, value.
    series_table: one row per series (describe_series).
    Structures with scored=False get the Gaussian reference member only.
    """
    structures = list(STRUCTURES) if structures is None else list(structures)
    members = list(MEMBERS) if members is None else list(members)
    vals, rows = [], []
    for sname in structures:
        st = STRUCTURES[sname]
        for seed in seeds:
            for mname in (members if st.scored else [REFERENCE]):
                mb = MEMBERS[mname]
                x = make_series(st, mb, seed, n_total)
                row = describe_series(x, st, mb, seed, n_context)
                rows.append(row)
                vals.append(pd.DataFrame({"series_id": row["series_id"],
                                          "position": np.arange(x.size, dtype="int64"), "value": x}))
    return pd.concat(vals, ignore_index=True), pd.DataFrame(rows)


# ---------------------------------------------------------------- match check

def check_pair(omega_a: float, omega_b: float, n_a: int, n_b: int,
               tolerance: float = OMEGA_TOLERANCE) -> tuple[str, str]:
    """('matched' | 'failed', reason). Length is checked first because Omega
    depends on length, so Omega values at different lengths are not comparable."""
    if int(n_a) != int(n_b):
        return "failed", f"unequal length {int(n_a)} vs {int(n_b)}"
    if not (np.isfinite(omega_a) and np.isfinite(omega_b)):
        return "failed", "omega not finite"
    d = abs(float(omega_a) - float(omega_b))
    if d > tolerance:
        return "failed", f"omega difference {d:.4f} exceeds tolerance {tolerance:.4f}"
    return "matched", ""


def _pair_row(a: pd.Series, b: pd.Series, kind: str, expected: bool, tolerance: float) -> dict:
    status, reason = check_pair(a[OMEGA_MATCH_COLUMN], b[OMEGA_MATCH_COLUMN],
                                a["n_context"], b["n_context"], tolerance)
    return dict(pair_kind=kind, seed=int(a["seed"]),
                structure_a=a["structure"], member_a=a["member"], series_id_a=a["series_id"],
                structure_b=b["structure"], member_b=b["member"], series_id_b=b["series_id"],
                omega_levels_a=a["omega_levels"], omega_levels_b=b["omega_levels"],
                omega_levels_abs_diff=abs(a["omega_levels"] - b["omega_levels"]),
                omega_diffs_a=a["omega_diffs"], omega_diffs_b=b["omega_diffs"],
                omega_diffs_abs_diff=abs(a["omega_diffs"] - b["omega_diffs"]),
                tolerance=tolerance, expected_match=bool(expected), status=status, reason=reason,
                as_expected=(status == "matched") == bool(expected))


def match_table(series_table: pd.DataFrame, tolerance: float = OMEGA_TOLERANCE) -> pd.DataFrame:
    """One row per pair (Gaussian reference, other member) of every group, plus
    the negative-control pairs. Every pair is kept whatever its status.

    pair_kind: 'alternative' (distribution differs), 'baseline' (two Gaussian
    draws), 'negative_control' (structure differs; a match is NOT expected).
    """
    st = series_table
    rows = []
    for (sname, seed), g in st.groupby(["structure", "seed"], sort=True):
        by = g.set_index("member", drop=False)
        if REFERENCE not in by.index:
            continue
        for mname in by.index:
            if mname == REFERENCE:
                continue
            kind = "baseline" if mname == BASELINE else "alternative"
            rows.append(_pair_row(by.loc[REFERENCE], by.loc[mname], kind, True, tolerance))
    ref = st[st["member"] == REFERENCE].set_index(["structure", "seed"], drop=False)
    for s_a, s_b in NEGATIVE_CONTROLS:
        for seed in sorted(set(st["seed"])):
            if (s_a, seed) in ref.index and (s_b, seed) in ref.index:
                rows.append(_pair_row(ref.loc[(s_a, seed)], ref.loc[(s_b, seed)],
                                      "negative_control", False, tolerance))
    return pd.DataFrame(rows)


def group_table(series_table: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """One row per generated group (structure, seed): Omega spread over all
    members and which members failed the pair check against the reference."""
    m = matches[matches["pair_kind"] != "negative_control"]
    rows = []
    for (sname, seed), g in series_table.groupby(["structure", "seed"], sort=True):
        if len(g) < 2:
            continue
        p = m[(m["structure_a"] == sname) & (m["seed"] == seed)]
        failed = sorted(p.loc[p["status"] == "failed", "member_b"])
        om = g["omega_levels"]
        rows.append(dict(structure=sname, seed=int(seed), n_members=len(g),
                         omega_levels_mean=om.mean(), omega_levels_min=om.min(),
                         omega_levels_max=om.max(), omega_levels_spread=om.max() - om.min(),
                         n_pairs=len(p), n_pairs_failed=len(failed),
                         status="all_matched" if not failed else "failed",
                         failed_members=";".join(failed)))
    return pd.DataFrame(rows)
