"""Verification tables showing that the E3 costs do not depend on the series.

Added on 2026-10-04 after an independent review. Three checks:

1. innovation_only_costs / compare_with_engine. Recompute every unit cost from
   the innovation sequence e_t = x_t - m_t alone, in plain numpy, with no
   engine call and no forecast frame, and compare with the engine's costs.
   Agreement to rounding error shows that the conditional mean, and with it
   the structure and Omega, cancel out of both forecasters' costs.
2. no_series_simulation. Draw i.i.d. innovations with no time series at all
   (so there is no Omega and nothing to match) and count how often the
   lower-cost forecaster differs between a member and an independent Gaussian
   draw. If this reproduces the rates of the matched-pair simulation, the
   match contributed nothing to them.
3. sign_by_match_status. The same sign-difference rates for the pairs that
   PASSED and the pairs that FAILED the Omega match.

The pinball loss below is written out here on purpose: it is the independent
route. E3's reported costs still come from the engine (scoring.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from decision_aware_tsfm.analysis.decision.metrics import TAU_GRID

from .design import (BASELINE, F_VARIANTS, MEMBERS, N_CONTEXT, N_TEST, PRIMARY_VARIANT,
                     REFERENCE, SCORED_STRUCTURES, STRUCTURES, innovation_dist)
from .generator import conditional_mean

TAUS = np.asarray(TAU_GRID, dtype=float)
TIE_RTOL = 1e-12                 # the engine's tie rule (selectors.argmin_family), restated
NO_SERIES_REPLICATES = 4000
NO_SERIES_SEED = 20261004


def _pinball_mean(e: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Mean pinball loss of quantile offsets (one per tau) against innovations e.
    e may be (n,) or (replicates, n); the mean is over the last axis of e."""
    u = e[..., None] - offsets
    return np.maximum(TAUS * u, (TAUS - 1.0) * u).mean(axis=-2)


def _f_wins(j_g: np.ndarray, j_f: np.ndarray) -> np.ndarray:
    return j_f < j_g - TIE_RTOL * np.maximum(1.0, np.maximum(np.abs(j_g), np.abs(j_f)))


def innovation_only_costs(values: pd.DataFrame, series_table: pd.DataFrame, variants=None,
                          n_context: int = N_CONTEXT) -> pd.DataFrame:
    """Costs of G and of every F variant for every scored series and tau,
    computed from the innovations only. One row per (variant, series, tau)."""
    variants = list(F_VARIANTS) if variants is None else list(variants)
    z = stats.norm.ppf(TAUS)
    st_rows = series_table[series_table["structure"].isin(SCORED_STRUCTURES)]
    by = {sid: g.sort_values("position")["value"].to_numpy()
          for sid, g in values[values["series_id"].isin(st_rows["series_id"])].groupby("series_id")}
    out = []
    for r in st_rows.itertuples():
        st, mb = STRUCTURES[r.structure], MEMBERS[r.member]
        x = by[r.series_id]
        e = x - conditional_mean(x, st)                 # innovations in x units, NaN at AR(1) position 0
        e_test, e_ctx = e[n_context:], e[:n_context]
        e_ctx = e_ctx[np.isfinite(e_ctx)]
        q = innovation_dist(mb).ppf(TAUS)
        j_g = _pinball_mean(e_test, st.sigma * z)
        for vname in variants:
            v = F_VARIANTS[vname]
            off = st.sigma * (q + v.location_error) if v.kind == "location" else np.quantile(e_ctx, TAUS)
            j_f = _pinball_mean(e_test, off)
            out.append(pd.DataFrame({"variant": vname, "series_id": r.series_id, "tau": TAUS,
                                     "J_G_innov": j_g, "J_F_innov": j_f,
                                     "winner_innov": np.where(_f_wins(j_g, j_f), "F", "G")}))
    return pd.concat(out, ignore_index=True)


def compare_with_engine(units: pd.DataFrame, innov: pd.DataFrame) -> pd.DataFrame:
    """Per variant: how far the innovation-only costs are from the engine's."""
    m = units.merge(innov, on=["variant", "series_id", "tau"], validate="1:1")
    if len(m) != len(units) or len(m) != len(innov):
        raise ValueError("engine units and innovation-only costs do not cover the same rows")
    rows = []
    for vname, d in m.groupby("variant", sort=False):
        rows.append(dict(variant=vname, n_units=len(d),
                         max_abs_diff_J_G=float((d["J_G"] - d["J_G_innov"]).abs().max()),
                         max_abs_diff_J_F=float((d["J_F"] - d["J_F_innov"]).abs().max()),
                         n_winner_mismatches=int((d["winner"] != d["winner_innov"]).sum())))
    tot = dict(variant="all", n_units=len(m),
               max_abs_diff_J_G=max(r["max_abs_diff_J_G"] for r in rows),
               max_abs_diff_J_F=max(r["max_abs_diff_J_F"] for r in rows),
               n_winner_mismatches=sum(r["n_winner_mismatches"] for r in rows))
    return pd.DataFrame(rows + [tot])


def no_series_simulation(variant: str = PRIMARY_VARIANT, n_replicates: int = NO_SERIES_REPLICATES,
                         n_test: int = N_TEST, seed: int = NO_SERIES_SEED) -> pd.DataFrame:
    """Sign-difference rate per (member, tau) from i.i.d. innovations only.

    Each replicate draws n_test innovations from the member's law and n_test
    from an independent Gaussian (the reference), scores G and F on each, and
    records whether the lower-cost forecaster differs. There is no series, no
    conditional mean, no Omega and no match. Location variants only."""
    v = F_VARIANTS[variant]
    if v.kind != "location":
        raise ValueError("the no-series simulation is defined for location variants only")
    z = stats.norm.ppf(TAUS)

    def f_wins(mname, stream):
        dist = innovation_dist(MEMBERS[mname])
        rng = np.random.default_rng([int(seed), MEMBERS[mname].code, stream])
        e = dist.rvs(size=(int(n_replicates), int(n_test)), random_state=rng)
        return _f_wins(_pinball_mean(e, z), _pinball_mean(e, dist.ppf(TAUS) + v.location_error))

    ref = f_wins(REFERENCE, 0)
    rows = []
    for mname in MEMBERS:
        if mname == REFERENCE:
            continue
        differs = f_wins(mname, 1) != ref
        for k, tau in enumerate(TAUS):
            rows.append(dict(variant=variant, member=mname,
                             pair_kind="baseline" if mname == BASELINE else "alternative",
                             tau=float(tau), n_replicates=int(n_replicates), n_test=int(n_test),
                             n_sign_differs=int(differs[:, k].sum()), rate=float(differs[:, k].mean())))
    return pd.DataFrame(rows)


def no_series_vs_matched(no_series: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """Per tau: sign-difference rate from the no-series simulation next to the
    rate among Omega-matched pairs of the series simulation. The alternative
    rate of the no-series simulation gives the six alternatives equal weight."""
    variant = no_series["variant"].iloc[0]
    p = pairs[(pairs["variant"] == variant) & (pairs["status"] == "matched")]
    rows = []
    for tau in TAUS:
        row = dict(variant=variant, tau=float(tau))
        for kind in ("alternative", "baseline"):
            ns = no_series[(no_series["pair_kind"] == kind) & np.isclose(no_series["tau"], tau)]
            mp = p[(p["pair_kind"] == kind) & np.isclose(p["tau"], tau)]
            row[f"{kind}_rate_no_series"] = float(ns["rate"].mean())
            row[f"{kind}_rate_matched_pairs"] = float(mp["sign_differs"].mean())
            row[f"{kind}_n_matched_pairs"] = len(mp)
        rows.append(row)
    return pd.DataFrame(rows)


def sign_by_match_status(pairs: pd.DataFrame, by=("variant", "pair_kind", "tau")) -> pd.DataFrame:
    """Sign-difference rate separately for pairs that passed and pairs that
    failed the Omega match, with a Fisher exact test of the two rates. The
    test treats pairs as independent although pairs of one group share a
    reference series, so its p-value is too small if anything."""
    rows = []
    for key, d in pairs.groupby(list(by), sort=True):
        key = key if isinstance(key, tuple) else (key,)
        ok, bad = d[d["status"] == "matched"], d[d["status"] == "failed"]
        k1, n1, k2, n2 = int(ok["sign_differs"].sum()), len(ok), int(bad["sign_differs"].sum()), len(bad)
        p = stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1] if n1 and n2 else float("nan")
        rows.append({**dict(zip(by, key)), "n_matched": n1, "n_differs_matched": k1,
                     "rate_matched": k1 / n1 if n1 else float("nan"),
                     "n_failed_match": n2, "n_differs_failed_match": k2,
                     "rate_failed_match": k2 / n2 if n2 else float("nan"), "fisher_p": float(p)})
    return pd.DataFrame(rows)


