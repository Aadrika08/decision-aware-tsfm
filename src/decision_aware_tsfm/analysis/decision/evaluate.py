"""Attach realised cost, oracle, regret and best-fixed family to any selector's
decisions, producing the common schema. Plus a group summary."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import COLUMNS, SCHEMA_VERSION, UNIT_KEYS, validate_selection_frame
from .selectors import SelectorInput, exact_argmin_family


def best_fixed(test: pd.DataFrame, group=("horizon", "tau")) -> pd.DataFrame:
    """Per group, the single family with lowest total scaled test cost (hindsight).
    Compared exactly; only an exact tie goes to classical."""
    t = test.copy()
    t["sc"] = t.J_classical / t.scale
    t["st"] = t.J_tsfm / t.scale
    g = t.groupby(list(group))[["sc", "st"]].sum().reset_index()
    g["best_fixed_family"] = exact_argmin_family(g.sc, g.st)
    return g[list(group) + ["best_fixed_family"]]


def evaluate(decisions: pd.DataFrame, selector_name: str, test: pd.DataFrame,
             units: pd.DataFrame, representatives: dict[str, str],
             n_val: pd.DataFrame) -> pd.DataFrame:
    d = units[UNIT_KEYS + ["omega"]].merge(decisions, on=UNIT_KEYS, how="left", validate="1:1")
    d = d.merge(test[UNIT_KEYS + ["J_classical", "J_tsfm", "scale", "n_origins"]],
                on=UNIT_KEYS, how="left", validate="1:1")
    d = d.merge(n_val, on=["dataset", "config", "series_id", "horizon"], how="left")
    d = d.merge(best_fixed(test), on=["horizon", "tau"], how="left")
    if d[["J_classical", "J_tsfm"]].isna().any().any():
        raise ValueError("some units have no test costs")

    sel = d.status == "selected"
    fam = d.chosen_family.where(sel)
    realized = np.where(fam == "classical", d.J_classical, np.where(fam == "tsfm", d.J_tsfm, np.nan))
    oracle_cost = np.minimum(d.J_classical, d.J_tsfm)
    regret = np.where(sel, realized - oracle_cost, np.nan)
    out = pd.DataFrame({
        "selector": selector_name,
        "dataset": d.dataset, "config": d.config, "series_id": d.series_id,
        "horizon": d.horizon.astype("int64"), "tau": d.tau.astype(float),
        "status": d.status,
        "chosen_family": fam.astype(object),
        "chosen_model": fam.map(representatives).astype(object),
        "realized_cost": realized.astype(float),
        "oracle_family": exact_argmin_family(d.J_classical, d.J_tsfm),
        "oracle_cost": oracle_cost.astype(float),
        "regret": regret.astype(float),
        "scale": d.scale.astype(float),
        "scaled_regret": (regret / d.scale).astype(float),
        "best_fixed_family": d.best_fixed_family,
        "best_fixed_cost": np.where(d.best_fixed_family == "classical", d.J_classical, d.J_tsfm).astype(float),
        "omega": d.omega.astype(float),
        "n_val_origins": d.n_validation_origins.fillna(0).astype("int64"),
        "n_test_origins": d.n_origins.astype("int64"),
        "note": d.note.fillna("").astype(object),
        "schema_version": SCHEMA_VERSION,
    })
    return out[list(COLUMNS)]


def run_selectors(selectors, units, val, test, representatives, n_val) -> pd.DataFrame:
    frames = []
    for s in selectors:
        inp = SelectorInput(units=units, val=val, test=test if s.needs_hindsight else None)
        frames.append(evaluate(s.select(inp), s.name, test, units, representatives, n_val))
    return validate_selection_frame(pd.concat(frames, ignore_index=True))


def summarize(res: pd.DataFrame, by=("selector", "horizon", "tau")) -> pd.DataFrame:
    """Per group: mean scaled regret, and fraction of oracle value recovered
        (sum BF - sum SEL) / (sum BF - sum ORACLE)   on scaled costs, selected units only.
    The fraction is undefined when best-fixed already equals the oracle
    (a dominance region or an exact tie), or when no unit in the group was
    selected, and is reported as NaN with the reason."""
    r = res.copy()
    s = r.status == "selected"
    r["sel_s"] = np.where(s, r.realized_cost / r.scale, np.nan)
    r["orc_s"] = np.where(s, r.oracle_cost / r.scale, np.nan)
    r["bf_s"] = np.where(s, r.best_fixed_cost / r.scale, np.nan)
    g = r.groupby(list(by))
    out = g.agg(n_units=("status", "size"),
                n_selected=("status", lambda x: int((x == "selected").sum())),
                mean_scaled_regret=("scaled_regret", "mean"),
                sel=("sel_s", "sum"), orc=("orc_s", "sum"), bf=("bf_s", "sum")).reset_index()
    denom = out.bf - out.orc
    ok = denom > 1e-12 * np.maximum(1.0, out.bf.abs())
    out["frac_oracle_value"] = np.where(ok, (out.bf - out.sel) / denom.where(ok, 1.0), np.nan)
    out["frac_note"] = np.where(ok, "", np.where(out.n_selected > 0, "undefined: best-fixed equals oracle",
                                                 "undefined: no selected units"))
    return out.drop(columns=["sel", "orc", "bf"])
