"""Attach realised cost, oracle, regret and best-fixed family to any selector's
decisions, producing the common schema. Plus a group summary.

Deferrals. A selector may answer "defer" for a unit (today only the K3
selector in the D_Omega slot does). A deferred row in the common schema carries
no family and no cost. summarize() covers decided units only, so it refuses a
frame with deferred rows; charge_deferrals() and summarize_charged() price each
deferral at the engine's pinball-validation comparator for the same unit (pilot
decision 8, #29) and report the deferral rate and the cost including deferrals
next to the decided-only numbers."""
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
    selected, and is reported as NaN with the reason.

    A frame with deferred rows is refused: averaging over selected units only
    would drop them without a trace. Use summarize_charged() for such a frame."""
    n_defer = int((res["status"] == "defer").sum())
    if n_defer:
        raise ValueError(f"{n_defer} deferred row(s): summarize() covers decided units only and would "
                         "drop them; use summarize_charged(), which prices each deferral")
    return _summarize_decided(res, by)


def _summarize_decided(res: pd.DataFrame, by) -> pd.DataFrame:
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


# --------------------------------------------------------------------------- deferrals

DEFERRAL_PRICING = "validation_comparator"     # the name #54's evaluate_selector uses for this pricing
DEFERRAL_PRICE_SELECTOR = "pinball_validation"
CHARGE_COLUMNS = ("deferred", "deferral_family", "deferral_cost", "charged_cost", "charged_regret",
                  "charged_scaled_regret", "deferral_pricing", "charge_note")


def charge_deferrals(res: pd.DataFrame) -> pd.DataFrame:
    """Every row of a selection frame, plus the cost the unit is charged.

    selected           charged its own realized_cost.
    defer              charged deferral_cost: the realized_cost of the engine's
                       pinball_validation comparator on the SAME unit (dataset,
                       config, series_id, horizon, tau), taken from that
                       comparator's rows in the same frame. Pilot decision 8
                       (#29). This is the per-unit deferral_cost #54's
                       evaluate_selector takes. It is never the oracle: there is
                       no option for oracle pricing here, and a deferral that
                       cannot be priced this way stays unpriced.
    defer, unpriced    pinball_validation made no choice for that unit (no
                       validation costs): charged_cost is NaN and charge_note says so.
    insufficient_data  not charged (NaN), with the reason in charge_note.

    Columns added (CHARGE_COLUMNS): deferred (bool), deferral_family (the
    family pinball_validation chose, deferred rows only), deferral_cost,
    charged_cost, charged_regret (= charged_cost - oracle_cost),
    charged_scaled_regret, deferral_pricing ("validation_comparator" on every
    row, so a row read on its own still says how deferrals were priced) and
    charge_note. The schema columns are returned unchanged."""
    validate_selection_frame(res)
    r = res.reset_index(drop=True)
    deferred = (r["status"] == "defer").to_numpy()
    selected = (r["status"] == "selected").to_numpy()
    pv = r.loc[r["selector"] == DEFERRAL_PRICE_SELECTOR,
               UNIT_KEYS + ["status", "chosen_family", "realized_cost"]]
    pv = pv.rename(columns={"status": "_pv_status", "chosen_family": "_pv_family",
                            "realized_cost": "_pv_cost"})
    m = r[UNIT_KEYS].merge(pv, on=UNIT_KEYS, how="left", validate="m:1")
    if deferred.any() and m.loc[deferred, "_pv_status"].isna().any():
        n = int(m.loc[deferred, "_pv_status"].isna().sum())
        raise ValueError(f"{n} deferred unit(s) have no {DEFERRAL_PRICE_SELECTOR} row in this frame; "
                         "a deferral is priced only at that comparator, never at the oracle")
    priced = deferred & (m["_pv_status"] == "selected").to_numpy()
    deferral_family = np.where(priced, m["_pv_family"].to_numpy(dtype=object), None)
    deferral_cost = np.where(priced, m["_pv_cost"].to_numpy(dtype=float), np.nan)
    realized = r["realized_cost"].to_numpy(dtype=float)
    charged = np.where(selected, realized, np.where(priced, deferral_cost, np.nan))
    charged_regret = charged - r["oracle_cost"].to_numpy(dtype=float)
    note = np.select([selected, priced, deferred],
                     ["", f"deferral charged at {DEFERRAL_PRICE_SELECTOR}",
                      f"deferral unpriced: {DEFERRAL_PRICE_SELECTOR} made no choice for this unit"],
                     default="not charged: status " + r["status"].astype(str))
    out = r.copy()
    out["deferred"] = deferred
    out["deferral_family"] = pd.Series(deferral_family, dtype=object)
    out["deferral_cost"] = deferral_cost.astype(float)
    out["charged_cost"] = charged.astype(float)
    out["charged_regret"] = charged_regret.astype(float)
    out["charged_scaled_regret"] = (charged_regret / r["scale"].to_numpy(dtype=float)).astype(float)
    out["deferral_pricing"] = DEFERRAL_PRICING
    out["charge_note"] = pd.Series(note, dtype=object)
    return out


def summarize_charged(res: pd.DataFrame, by=("selector", "horizon", "tau")) -> pd.DataFrame:
    """summarize() for a frame that may contain deferrals.

    The decided-only columns are exactly summarize()'s (n_units, n_selected,
    mean_scaled_regret, frac_oracle_value, frac_note: selected units only). Added,
    over EVERY unit in the group, with each deferral charged as in charge_deferrals():
      n_deferred, deferral_rate         deferred units and their share of n_units
      n_not_charged                     units with no charged cost (insufficient_data,
                                        or a deferral pinball_validation could not price)
      mean_scaled_cost_charged          mean of charged_cost / scale: the realised cost
                                        including deferrals
      mean_scaled_regret_charged        mean of charged_scaled_regret
      frac_oracle_value_charged         (sum BF - sum CHARGED) / (sum BF - sum ORACLE), scaled
      frac_note_charged                 why that fraction is undefined, if it is
      deferral_pricing                  "validation_comparator"
    The three charged numbers are NaN when any unit in the group is not charged,
    so a partly priced group never looks complete. With the default grouping
    (selector, horizon, tau) every number is per tau (pilot decision 9)."""
    c = charge_deferrals(res)
    base = _summarize_decided(res, by)
    c["_sc"] = c["charged_cost"] / c["scale"]
    c["_oc"] = c["oracle_cost"] / c["scale"]
    c["_bc"] = c["best_fixed_cost"] / c["scale"]
    c["_uncharged"] = c["charged_cost"].isna()
    g = c.groupby(list(by))
    ex = g.agg(n_deferred=("deferred", "sum"), n_not_charged=("_uncharged", "sum"),
               mean_scaled_cost_charged=("_sc", "mean"),
               mean_scaled_regret_charged=("charged_scaled_regret", "mean"),
               sc=("_sc", "sum"), oc=("_oc", "sum"), bc=("_bc", "sum")).reset_index()
    out = base.merge(ex, on=list(by), how="left", validate="1:1")
    out["n_deferred"] = out["n_deferred"].astype("int64")
    out["n_not_charged"] = out["n_not_charged"].astype("int64")
    out["deferral_rate"] = out["n_deferred"] / out["n_units"]
    complete = out["n_not_charged"] == 0
    for col in ("mean_scaled_cost_charged", "mean_scaled_regret_charged"):
        out[col] = np.where(complete, out[col], np.nan)
    denom = out.bc - out.oc
    ok = complete & (denom > 1e-12 * np.maximum(1.0, out.bc.abs()))
    out["frac_oracle_value_charged"] = np.where(ok, (out.bc - out.sc) / denom.where(ok, 1.0), np.nan)
    out["frac_note_charged"] = np.where(
        ok, "", np.where(~complete, "undefined: " + out["n_not_charged"].astype(str) + " unit(s) not charged",
                         "undefined: best-fixed equals oracle"))
    out["deferral_pricing"] = DEFERRAL_PRICING
    cols = list(base.columns) + ["n_deferred", "deferral_rate", "n_not_charged", "mean_scaled_cost_charged",
                                 "mean_scaled_regret_charged", "frac_oracle_value_charged",
                                 "frac_note_charged", "deferral_pricing"]
    return out[cols]
