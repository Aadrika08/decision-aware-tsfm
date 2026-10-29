"""Pilot real-cache tables, generated from run_pilot outputs (no hand-typed numbers).

usage: python -I make_tables.py RUNS_DIR OMEGA_JSON > tables.md
RUNS_DIR holds py311-{placeholder,k3_selector}[-last4096] run_pilot output folders.
"""
import json, sys
from pathlib import Path
import pandas as pd

runs = Path(sys.argv[1]); omega_path = Path(sys.argv[2])
SEL = ["always_classical", "always_tsfm", "static_omega", "pinball_validation", "crps_validation", "hindsight_oracle"]

def head(run):
    t = pd.read_csv(runs / run / "tables" / "headline_by_selector.csv")
    return t[(t.headline) & (t.static_threshold == 0.45)].set_index("selector")

def md(df, floatfmt=4):
    cols = list(df.columns); idx = df.index.name or ""
    out = ["| " + " | ".join([idx] + [str(c) for c in cols]) + " |", "|" + "---|" * (len(cols) + 1)]
    for i, r in df.iterrows():
        cells = []
        for v in r:
            cells.append(f"{v:.{floatfmt}f}" if isinstance(v, float) else str(v))
        out.append("| " + " | ".join([str(i)] + cells) + " |")
    return "\n".join(out)

pc, pl = head("py311-placeholder"), head("py311-placeholder-last4096")
kc, kl = head("py311-k3_selector"), head("py311-k3_selector-last4096")
rows = {}
for s in SEL:
    rows[s] = [pc.loc[s, "mean_scaled_regret"], pl.loc[s, "mean_scaled_regret"]]
rows["D_Omega, placeholder"] = [pc.loc["d_omega", "mean_scaled_regret"], pl.loc["d_omega", "mean_scaled_regret"]]
rows["D_Omega, K3, decided units only"] = [kc.loc["d_omega", "mean_scaled_regret"], kl.loc["d_omega", "mean_scaled_regret"]]
pricedcol = [c for c in kc.columns if c.startswith("mean_scaled_regret_deferrals")][0]
rows["D_Omega, K3, deferrals priced (all 1,815)"] = [kc.loc["d_omega", pricedcol], kl.loc["d_omega", pricedcol]]
t1 = pd.DataFrame(rows, index=["Omega `current`", "Omega `last_4096`"]).T; t1.index.name = "selector"
n_dec = (int(kc.loc["d_omega", "n_selected"]), int(kl.loc["d_omega", "n_selected"]))
print("## Table 1. Mean scaled regret, all horizons and taus, static threshold 0.45 (1,815 units)\n")
print(md(t1, 4))
print(f"\nK3 decided units: {n_dec[0]} of 1,815 (`current`), {n_dec[1]} of 1,815 (`last_4096`); the rest deferred.\n")

# thresholds for static omega
def allthr(run, sel):
    t = pd.read_csv(runs / run / "tables" / "headline_by_selector.csv")
    return t[t.selector == sel].set_index("static_threshold").mean_scaled_regret
t1b = pd.DataFrame({"current": allthr("py311-placeholder", "static_omega"), "last_4096": allthr("py311-placeholder-last4096", "static_omega")}); t1b.index.name = "static threshold"
print("## Table 1b. Static Omega rule by threshold (mean scaled regret)\n"); print(md(t1b, 4)); print()

# by horizon
def byh(run):
    t = pd.read_csv(runs / run / "tables" / "headline_by_selector_horizon.csv"); t = t[t.headline]
    return t.pivot(index="selector", columns="horizon", values="mean_scaled_regret")
hp, hk = byh("py311-placeholder"), byh("py311-k3_selector")
t2 = hp.loc[SEL].copy(); t2.loc["D_Omega, placeholder"] = hp.loc["d_omega"]; 
kt = pd.read_csv(runs / "py311-k3_selector/tables/headline_by_selector_horizon.csv"); kt = kt[(kt.headline) & (kt.selector == "d_omega")].set_index("horizon")
t2.loc["D_Omega, K3, deferrals priced"] = kt[pricedcol]
t2.columns = [f"h={c}" for c in t2.columns]; t2.index.name = "selector"
print("## Table 2. Mean scaled regret by horizon (Omega `current`, threshold 0.45)\n"); print(md(t2, 4)); print()

# family winners by horizon, tau
e1 = pd.read_csv(runs / "py311-placeholder/e1/summary_by_horizon_tau.csv")
iv = pd.read_csv(runs / "py311-placeholder/e1/mean_delta_intervals.csv")
t3 = e1.merge(iv[["horizon", "tau", "mean_delta_lower", "mean_delta_upper"]], on=["horizon", "tau"])
t3["lower_cost_family"] = t3.share_tsfm_lower.map(lambda s: "TSFM" if s > 0.5 else ("classical" if s < 0.5 else "tie"))
t3["interval excludes 0"] = ((t3.mean_delta_lower > 0) | (t3.mean_delta_upper < 0)).map({True: "yes", False: "no"})
out = pd.DataFrame({"h": t3.horizon, "tau": t3.tau, "units": t3.n_units, "TSFM lower (share)": t3.share_tsfm_lower.round(3),
                    "mean Delta": t3.mean_delta.round(4), "95% interval": [f"[{a:.4f}, {b:.4f}]" for a, b in zip(t3.mean_delta_lower, t3.mean_delta_upper)],
                    "lower-cost family (by share)": t3.lower_cost_family, "excludes 0": t3["interval excludes 0"]})
out.index = [""] * len(out)
print("## Table 3. Which family has the lower cost, by horizon and tau (Delta = TSFM cost minus classical cost, scaled; Delta < 0 favours TSFM)\n")
print("| h | tau | units | TSFM lower (share) | mean Delta | 95% interval | lower-cost family (by share) | excludes 0 |\n|---|---|---|---|---|---|---|---|")
for _, r in out.iterrows():
    print(f"| {r['h']} | {r['tau']} | {r['units']} | {r['TSFM lower (share)']} | {r['mean Delta']} | {r['95% interval']} | {r['lower-cost family (by share)']} | {r['excludes 0']} |")
print(f"\nCells where the interval excludes 0: {int((t3['interval excludes 0']=='yes').sum())} of {len(t3)}. Intervals are nominal 95% percentile block-bootstrap intervals, approximate and known to under-cover (STATUS, verified findings); do not quote one as exact.\n")

# by dataset
d = pd.read_csv(runs / "py311-placeholder/e1/summary_by_dataset_horizon_tau.csv")
g = d.groupby("dataset")[list(d.columns)].apply(lambda x: pd.Series({"units": int(x.n_units.sum()), "TSFM lower (share)": x.n_tsfm_lower.sum() / x.n_units.sum(),
    "mean Delta": (x.mean_delta * x.n_units).sum() / x.n_units.sum(),
    "regret always_classical": (x.regret_always_classical * x.n_units).sum() / x.n_units.sum(),
    "regret always_tsfm": (x.regret_always_tsfm * x.n_units).sum() / x.n_units.sum(),
    "regret pinball_validation": (x.regret_pinball_validation * x.n_units).sum() / x.n_units.sum()}))
g["units"] = g["units"].astype(int); g.index.name = "dataset"
print("## Table 4. By dataset (all horizons and taus pooled, Omega `current`)\n")
print(md(g.astype(object).apply(lambda c: c.map(lambda v: str(v) if isinstance(v, int) else f"{float(v):.4f}")), 4)); print()

# K3 deferrals
def dfr(run):
    t = pd.read_csv(runs / run / "engine/threshold_0.45/selection_results.csv"); t = t[t.selector == "d_omega"]
    d = t[t.status != "selected"]
    return pd.DataFrame({"units": t.groupby("dataset").size(), "series": t.groupby("dataset").series_id.nunique(),
                         "deferred units": d.groupby("dataset").size(), "deferred series": d.groupby("dataset").series_id.nunique()}).fillna(0).astype(int)
a, b = dfr("py311-k3_selector"), dfr("py311-k3_selector-last4096")
t5 = pd.DataFrame({"units": a.units, "series": a.series, "deferred units (`current`)": a["deferred units"], "deferred series (`current`)": a["deferred series"],
                   "deferred units (`last_4096`)": b["deferred units"], "deferred series (`last_4096`)": b["deferred series"]}).astype(int)
t5.index.name = "dataset"
print("## Table 5. K3 selector deferrals by dataset (threshold 0.45; every deferral is reason omega_out_of_range)\n"); print(md(t5)); print()

# omega band
om = json.loads(omega_path.read_text(encoding="utf-8"))
recs = om["rows"]
o = pd.DataFrame(recs)
vcol, vcols = "omega", "variant"
res = {}
for v, x in o.groupby(vcols):
    w = x[vcol].astype(float)
    res[v] = {"origin records": len(x), "in 0.40 to 0.50": int(((w >= 0.40) & (w <= 0.50)).sum()), "below 0.40": int((w < 0.40).sum()), "above 0.50": int((w > 0.50).sum())}
print("## Table 6. Omega values by variant (series-origin records)\n"); print(md(pd.DataFrame(res).T.rename_axis("variant"), 0).replace(".0 ", " ").replace(".0|", "|"))
