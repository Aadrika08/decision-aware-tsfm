"""Score the idealised forecasters with the decision engine and answer the E3
question: within Omega-matched pairs at the same tau, does the sign of Delta
differ between members?

All costs come from the engine (analysis/decision): load_cache validates the
canonical-v1 frame, per_origin_costs and unit_costs compute mean pinball loss,
argmin_family decides the winner (exact ties go to the "classical" slot).
No cost is computed here.

Slot labels: the engine's family names are used as slots only.
    "classical" slot = idealised forecaster G (Gaussian shape)
    "tsfm" slot      = idealised forecaster F (flexible shape, stated imperfection)
    Delta = J_F - J_G, so Delta > 0 means G has the lower cost.

Added 2026-10-04 after independent review. Both forecasters are given the true
conditional mean and scale, so these costs depend on the innovations only and
the lower-cost forecaster in each (member, tau) cell is fixed in closed form
(population.py). The sign counts here measure finite-sample noise around that
closed form; they are not a test that could have come out the other way.
sign_difference_vs_baseline gives one joint interval for the difference from
the Gaussian baseline, and sign_by_population_status separates differences in
cells where the population winner really reverses from near-ties in cells
where it does not.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import integrate, stats

from decision_aware_tsfm.analysis.decision.metrics import TAU_GRID
from decision_aware_tsfm.analysis.decision.pipeline import build_tables
from decision_aware_tsfm.analysis.decision.selectors import argmin_family

from .design import (F_VARIANTS, FAMILY_SLOT, MEMBERS, MODEL_G, N_CONTEXT, OMEGA_MATCH_COLUMN,
                     REFERENCE, SCORED_STRUCTURES, innovation_dist)
from .forecasters import group_frame

SCHEMA = "canonical-v1"
WINNER_OF_SLOT = {FAMILY_SLOT["G"]: "G", FAMILY_SLOT["F"]: "F"}
SERIES_COLS = ["series_id", "structure", "seed", "member", "member_role"]


def engine_unit_costs(frame: pd.DataFrame, omega: pd.DataFrame, f_model: str) -> pd.DataFrame:
    """Run the engine on a canonical-v1 frame holding G and one F model.
    Every origin is a test origin (val_fraction = 0): nothing is selected or
    fitted, so no validation split is needed. Returns one row per (series, tau)."""
    reps = {FAMILY_SLOT["G"]: MODEL_G, FAMILY_SLOT["F"]: f_model}
    fam = {MODEL_G: FAMILY_SLOT["G"], f_model: FAMILY_SLOT["F"]}
    sub = frame[frame["model"].isin([MODEL_G, f_model])]
    tb = build_tables(sub, SCHEMA, reps, fam, omega, val_fraction=0.0, omega_col=OMEGA_MATCH_COLUMN)
    keys = ["dataset", "config", "series_id", "horizon"]
    u = tb["test"].merge(tb["units"], on=keys + ["tau"], validate="1:1")
    u = u.merge(tb["decision_origins"], on=keys, validate="m:1")
    out = pd.DataFrame({
        "series_id": u["series_id"], "tau": u["tau"].astype(float),
        "J_G": u["J_classical"].astype(float), "J_F": u["J_tsfm"].astype(float),
        "scale": u["scale"].astype(float), "omega": u["omega"].astype(float),
        "n_test_origins": u["n_origins"].astype("int64"),
        "decision_origin": u["decision_origin"].astype("int64")})
    out["delta"] = out["J_F"] - out["J_G"]
    out["delta_scaled"] = out["delta"] / out["scale"]          # relative to mean |y| on test targets
    out["delta_rel"] = out["delta"] / out["J_G"]               # relative to G's own cost
    slot = argmin_family(out["J_G"].to_numpy(), out["J_F"].to_numpy())
    out["winner"] = [WINNER_OF_SLOT[s] for s in slot]
    return out


def score_dataset(values: pd.DataFrame, series_table: pd.DataFrame, variants=None) -> pd.DataFrame:
    """Unit costs for every scored series, every tau and every F variant.
    The engine is called once per (group, variant) to bound memory."""
    variants = list(F_VARIANTS) if variants is None else list(variants)
    st = series_table[series_table["structure"].isin(SCORED_STRUCTURES)]
    by_series = {sid: g.sort_values("position")["value"].to_numpy()
                 for sid, g in values[values["series_id"].isin(st["series_id"])].groupby("series_id")}
    models = [MODEL_G] + [F_VARIANTS[v].model for v in variants]
    out = []
    for _, g in st.groupby(["structure", "seed"], sort=True):
        if (g["n_context"] != N_CONTEXT).any():
            raise ValueError("scoring expects the design context length")
        frame = group_frame(by_series, g, models)
        omega = g[["dataset", "config", "series_id", OMEGA_MATCH_COLUMN]]
        for v in variants:
            u = engine_unit_costs(frame, omega, F_VARIANTS[v].model)
            if (u["decision_origin"] != N_CONTEXT).any():
                raise ValueError("decision origin is not the first position after the Omega window")
            u.insert(0, "variant", v)
            out.append(u.merge(g[SERIES_COLS], on="series_id", validate="m:1"))
    return pd.concat(out, ignore_index=True)


# ------------------------------------------------------------ sign differences

def sign_pairs(units: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """One row per (variant, tau, pair). Pairs that failed the Omega check are
    kept with status 'failed'; they are counted, and excluded from the sign count."""
    m = matches[matches["pair_kind"] != "negative_control"]
    m = m[["pair_kind", "seed", "structure_a", "member_b", "series_id_a", "series_id_b",
           "omega_levels_a", "omega_levels_b", "omega_levels_abs_diff", "status"]]
    cols = ["variant", "tau", "series_id", "winner", "delta", "delta_scaled", "delta_rel"]
    a = units[cols].rename(columns={c: f"{c}_a" for c in cols[2:]})
    b = units[cols].rename(columns={c: f"{c}_b" for c in cols[2:]})
    p = m.merge(a, on="series_id_a").merge(b, on=["series_id_b", "variant", "tau"])
    p = p.rename(columns={"structure_a": "structure"})
    p["sign_differs"] = p["winner_a"] != p["winner_b"]
    return p


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    ph = k / n
    den = 1 + z * z / n
    mid = (ph + z * z / (2 * n)) / den
    half = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return float(max(0.0, mid - half)), float(min(1.0, mid + half))


def _cluster_boot(d: pd.DataFrame, n_boot: int, rng: np.random.Generator) -> tuple[float, float]:
    """Percentile interval for the sign-difference rate, resampling whole
    (structure, seed) groups so that pairs sharing one reference series stay together."""
    c = d.groupby(["structure", "seed"])["sign_differs"].agg(["sum", "size"]).to_numpy(dtype=float)
    if len(c) == 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, len(c), size=(n_boot, len(c)))
    rates = c[idx, 0].sum(axis=1) / c[idx, 1].sum(axis=1)
    return float(np.quantile(rates, 0.025)), float(np.quantile(rates, 0.975))


def arm_interval(k: int, n: int, n_groups: int, boot_lo: float, boot_hi: float):
    """(lo, hi, method) for one sign-difference rate k / n.

    Normal case: the group-bootstrap percentile interval. When all n outcomes
    are equal (k = 0 or k = n) the bootstrap cannot move, so a Wilson interval
    is used instead, with the number of independent (structure, seed) groups
    as the sample size rather than the number of pairs. That is the cautious
    choice: pairs of one group share a reference series."""
    if n == 0:
        return float("nan"), float("nan"), "no_matched_pairs"
    if k == 0 or k == n:
        lo, hi = wilson_interval(0 if k == 0 else n_groups, n_groups)
        return lo, hi, "wilson_on_groups"
    return boot_lo, boot_hi, "group_bootstrap"


def count_sign_differences(pairs: pd.DataFrame, by=("variant", "pair_kind", "tau"),
                           n_boot: int = 2000, boot_seed: int = 20261004) -> pd.DataFrame:
    """Per group of `by`: pairs generated, pairs that failed the Omega match,
    matched pairs, and among MATCHED pairs how many have opposite winners.

    wilson_*:   binomial interval on pairs, appropriate when every pair comes
                from a different (structure, seed) group.
    boot_*:     group-resampling interval, appropriate when rows are pooled
                over members that share a reference series. NaN when every
                matched pair has the same outcome (0 of n or n of n), because
                resampling identical outcomes gives a zero-width interval.
    interval_*: the interval to report: the group bootstrap, or in the
                all-equal case a Wilson interval that counts each group once
                (see arm_interval).
    """
    rng = np.random.default_rng(boot_seed)
    rows = []
    for key, d in pairs.groupby(list(by), sort=True):
        key = key if isinstance(key, tuple) else (key,)
        ok = d[d["status"] == "matched"]
        n, k = len(ok), int(ok["sign_differs"].sum())
        lo, hi = wilson_interval(k, n)
        blo, bhi = _cluster_boot(ok, n_boot, rng)
        n_groups = ok.groupby(["structure", "seed"]).ngroups
        ilo, ihi, method = arm_interval(k, n, n_groups, blo, bhi)
        if method != "group_bootstrap":
            blo = bhi = float("nan")              # a bootstrap of all-equal outcomes has no spread
        rows.append({**dict(zip(by, key)), "n_pairs": len(d), "n_failed_match": len(d) - n,
                     "n_matched": n, "n_sign_differs": k,
                     "rate": k / n if n else float("nan"),
                     "wilson_lo": lo, "wilson_hi": hi, "boot_lo": blo, "boot_hi": bhi,
                     "interval_lo": ilo, "interval_hi": ihi, "interval_method": method,
                     "n_groups": n_groups,
                     "mean_abs_delta_scaled_a": ok["delta_scaled_a"].abs().mean() if n else float("nan"),
                     "mean_abs_delta_scaled_b": ok["delta_scaled_b"].abs().mean() if n else float("nan")})
    return pd.DataFrame(rows)


def sign_difference_vs_baseline(pairs: pd.DataFrame, by=("variant", "tau"), n_boot: int = 2000,
                                boot_seed: int = 20261004) -> pd.DataFrame:
    """Alternative sign-difference rate MINUS baseline rate, among matched
    pairs, with one joint interval for the difference.

    Whole (structure, seed) groups are resampled, and the alternative and the
    baseline rate are recomputed from the SAME resample, because the six
    alternative pairs and the baseline pair of a group share one reference
    series. If either rate is 0 of n or n of n the bootstrap understates that
    arm's uncertainty, so the difference interval is then built from the two
    arm intervals (arm_interval) by the square-and-add rule of Newcombe
    (method "newcombe_from_arm_intervals"), which treats the arms as
    independent and is therefore wider than needed.

    CRITERION. above_baseline is True when the lower end of the 95 percent
    interval for (alternative rate - baseline rate) is above zero. It says
    only that alternatives differ from the reference more often than a second
    Gaussian draw does. It is not a statement about the size of any cost gap
    (see sign_by_population_status for that)."""
    rng = np.random.default_rng(boot_seed)
    rows = []
    for key, d in pairs.groupby(list(by), sort=True):
        key = key if isinstance(key, tuple) else (key,)
        ok = d[d["status"] == "matched"]
        alt, base = ok[ok["pair_kind"] == "alternative"], ok[ok["pair_kind"] == "baseline"]
        g = (ok.assign(is_alt=ok["pair_kind"] == "alternative")
               .groupby(["structure", "seed", "is_alt"])["sign_differs"].agg(["sum", "size"])
               .unstack("is_alt").reindex(columns=[("sum", True), ("size", True), ("sum", False),
                                                   ("size", False)]).fillna(0.0).to_numpy(dtype=float))
        ka, na, kb, nb = int(alt["sign_differs"].sum()), len(alt), int(base["sign_differs"].sum()), len(base)
        row = {**dict(zip(by, key)), "alt_n_matched": na, "alt_n_sign_differs": ka,
               "alt_rate": ka / na if na else float("nan"),
               "base_n_matched": nb, "base_n_sign_differs": kb,
               "base_rate": kb / nb if nb else float("nan"), "n_groups": len(g)}
        if na == 0 or nb == 0:
            rows.append({**row, "diff": float("nan"), "diff_lo": float("nan"), "diff_hi": float("nan"),
                         "diff_method": "no_matched_pairs", "above_baseline": False})
            continue
        idx = rng.integers(0, len(g), size=(n_boot, len(g)))
        tot = g[idx].sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            ra, rb = tot[:, 0] / tot[:, 1], tot[:, 2] / tot[:, 3]
        q = lambda a: (float(np.nanquantile(a, 0.025)), float(np.nanquantile(a, 0.975)))
        pa, pb = row["alt_rate"], row["base_rate"]
        la, ua, ma = arm_interval(ka, na, int((g[:, 1] > 0).sum()), *q(ra))
        lb, ub, mb_ = arm_interval(kb, nb, int((g[:, 3] > 0).sum()), *q(rb))
        if ma == "group_bootstrap" and mb_ == "group_bootstrap":
            lo, hi = q(ra - rb)
            method = "joint_group_bootstrap"
        else:
            lo = (pa - pb) - np.sqrt((pa - la) ** 2 + (ub - pb) ** 2)
            hi = (pa - pb) + np.sqrt((ua - pa) ** 2 + (pb - lb) ** 2)
            method = "newcombe_from_arm_intervals"
        rows.append({**row, "alt_lo": la, "alt_hi": ua, "alt_method": ma,
                     "base_lo": lb, "base_hi": ub, "base_method": mb_,
                     "diff": pa - pb, "diff_lo": float(lo), "diff_hi": float(hi),
                     "diff_method": method, "above_baseline": bool(lo > 0)})
    return pd.DataFrame(rows)


def population_cell_status(expected: pd.DataFrame) -> pd.DataFrame:
    """Per (variant, member, tau): is the population winner on this member
    different from the population winner on the Gaussian reference?

    cell_status = "reversal"     the two population winners differ
                  "same_winner"  they are the same, so any realised sign
                                 difference in this cell is a near-tie decided
                                 by sampling noise, not a reversal
    pop_delta_rel_* is the population J_F - J_G as a fraction of G's cost."""
    e = expected[["variant", "member", "tau", "population_winner", "exp_delta_rel"]]
    ref = (e[e["member"] == REFERENCE].drop(columns="member")
           .rename(columns={"population_winner": "population_winner_reference",
                            "exp_delta_rel": "pop_delta_rel_reference"}))
    out = e.rename(columns={"population_winner": "population_winner_member",
                            "exp_delta_rel": "pop_delta_rel_member"}).merge(
        ref, on=["variant", "tau"], validate="m:1")
    out["cell_status"] = np.where(out["population_winner_member"] != out["population_winner_reference"],
                                  "reversal", "same_winner")
    return out


def sign_by_population_status(pairs: pd.DataFrame, expected: pd.DataFrame,
                              by=("variant", "pair_kind", "tau", "cell_status")) -> pd.DataFrame:
    """Sign differences among matched pairs, split by population_cell_status,
    with effect sizes. Location variants only (the empirical variant has no
    closed-form population winner).

    Effect-size columns, all as a fraction of G's own cost:
      pop_delta_rel_member_min / _max   population gap over the cells in the row
      realised_delta_rel_member_mean    mean realised gap on the member series
      realised_delta_rel_reference_mean mean realised gap on the reference series
    """
    cells = population_cell_status(expected).rename(columns={"member": "member_b"})
    ok = pairs[pairs["status"] == "matched"].merge(
        cells, on=["variant", "member_b", "tau"], validate="m:1")       # inner: location variants
    rows = []
    for key, d in ok.groupby(list(by), sort=True):
        key = key if isinstance(key, tuple) else (key,)
        n, k = len(d), int(d["sign_differs"].sum())
        lo, hi = wilson_interval(k, n)
        rows.append({**dict(zip(by, key)), "n_cells": d[["member_b", "tau"]].drop_duplicates().shape[0],
                     "n_matched": n, "n_sign_differs": k, "rate": k / n,
                     "wilson_lo": lo, "wilson_hi": hi,
                     "pop_delta_rel_member_min": d["pop_delta_rel_member"].min(),
                     "pop_delta_rel_member_max": d["pop_delta_rel_member"].max(),
                     "pop_delta_rel_reference": d["pop_delta_rel_reference"].mean(),
                     "realised_delta_rel_member_mean": d["delta_rel_b"].mean(),
                     "realised_delta_rel_reference_mean": d["delta_rel_a"].mean()})
    return pd.DataFrame(rows)


def member_delta_summary(units: pd.DataFrame, by=("variant", "member", "tau")) -> pd.DataFrame:
    """Across series (seeds and, unless `by` says otherwise, structures):
    share where F has the lower cost, and mean and sd of Delta in both scalings."""
    g = units.groupby(list(by), sort=True)
    out = g.agg(n_series=("series_id", "size"),
                share_F_wins=("winner", lambda w: float((w == "F").mean())),
                delta_scaled_mean=("delta_scaled", "mean"), delta_scaled_sd=("delta_scaled", "std"),
                delta_rel_mean=("delta_rel", "mean"), delta_rel_sd=("delta_rel", "std")).reset_index()
    out["delta_rel_se"] = out["delta_rel_sd"] / np.sqrt(out["n_series"])
    return out


# ------------------------------------------------- population (analytic) check

def expected_pinball(dist, c: float, tau: float) -> float:
    """E[pinball_tau(e - c)] for e ~ dist with mean 0, by numerical integration.
    pinball_tau(u) = tau * u + max(-u, 0), so the expectation is
    -tau * c + integral_{-inf}^{c} F(x) dx."""
    val, _ = integrate.quad(dist.cdf, -np.inf, c, limit=200)
    return float(-tau * c + val)


def expected_delta_table(taus=TAU_GRID) -> pd.DataFrame:
    """Population cost of G and of each location variant of F, in units of
    sigma, for every member and tau. Independent of the engine and of any
    simulated data; the empirical variant has no closed form and is omitted."""
    z = stats.norm.ppf(np.asarray(taus, dtype=float))
    rows = []
    for mname, mb in MEMBERS.items():
        dist = innovation_dist(mb)
        q = dist.ppf(np.asarray(taus, dtype=float))
        for k, tau in enumerate(taus):
            eg = expected_pinball(dist, z[k], tau)
            for v in F_VARIANTS.values():
                if v.kind != "location":
                    continue
                ef = expected_pinball(dist, q[k] + v.location_error, tau)
                rows.append(dict(variant=v.name, member=mname, tau=float(tau),
                                 gaussian_quantile=z[k], true_quantile=q[k],
                                 exp_J_G=eg, exp_J_F=ef, exp_delta=ef - eg,
                                 exp_delta_rel=(ef - eg) / eg,
                                 population_winner="F" if ef < eg else "G"))
    return pd.DataFrame(rows)
