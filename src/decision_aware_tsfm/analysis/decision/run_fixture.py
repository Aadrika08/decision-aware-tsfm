"""python -m decision_aware_tsfm.analysis.decision.run_fixture --out DIR

--d-omega k3_selector puts the K3 Omega-plus-tau selector (#26) in the D_Omega
slot instead of the placeholder. It needs scikit-learn. Fixture numbers only.
"""
from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path

import pandas as pd

from .evaluate import summarize
from .fixture import (FAMILY_OF_MODEL, FIT_ONLY_DATASETS, FIXTURE_SCHEMA,
                      FIXTURE_STATIC_THRESHOLD, REPRESENTATIVES, make_fixture)
from .pipeline import build_tables, run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/runs/decision_fixture")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--d-omega", choices=("placeholder", "k3_selector"), default="placeholder")
    ap.add_argument("--k3-margin", type=float, default=None, help="k3_selector only; default: his 0.0")
    ap.add_argument("--k3-min-distinct", type=int, default=None, help="k3_selector only; default: his 3")
    ap.add_argument("--k3-cost", choices=("raw", "scaled"), default=None, help="k3_selector only; default raw")
    a = ap.parse_args()
    k3_options = {k: v for k, v in (("margin", a.k3_margin), ("min_distinct", a.k3_min_distinct),
                                    ("cost", a.k3_cost)) if v is not None}
    if k3_options and a.d_omega != "k3_selector":
        ap.error("--k3-* options need --d-omega k3_selector")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)

    fc, omega, planted = make_fixture(a.seed)
    print(f"fixture: {len(fc):,} forecast rows, {fc.series_id.nunique()} series, "
          f"{fc.model.nunique()} models, horizons {sorted(int(h) for h in fc.horizon.unique())}")
    fc.to_parquet(out / "fixture_cache.parquet", index=False)
    omega.to_csv(out / "fixture_omega.csv", index=False, lineterminator="\n")
    planted.to_csv(out / "planted_answers.csv", index=False, lineterminator="\n")

    tables = build_tables(out / "fixture_cache.parquet", FIXTURE_SCHEMA, REPRESENTATIVES,
                          FAMILY_OF_MODEL, omega, fit_only_datasets=FIT_ONLY_DATASETS)
    if a.d_omega == "k3_selector":
        from . import k3_adapter               # imported only on request: it needs scikit-learn
        res, d = run(tables, REPRESENTATIVES, FIXTURE_STATIC_THRESHOLD,
                     d_omega_fit=functools.partial(k3_adapter.K3SelectorAdapter.fit, **k3_options))
        folds = d.describe()
        (out / "d_omega_folds.json").write_text(json.dumps(folds, indent=1, sort_keys=True) + "\n",
                                                encoding="utf-8")
        print("D_Omega K3 Omega-plus-tau selector, leave-one-dataset-out folds "
              "(frozen specs in d_omega_folds.json):")
        for ds, fold in folds["folds"].items():
            if "specs" not in fold:
                print(f"  {ds}: {fold}")
                continue
            rules = [r["status"] for s in fold["specs"].values() for r in s["rules"].values()]
            print(f"  {ds}: margin={fold['margin']} min_distinct={fold['min_distinct']} cost={fold['cost']} "
                  f"rules fitted {rules.count('ok')}/{len(rules)} fitted_datasets={fold['fitted_datasets']}")
    else:
        res, d = run(tables, REPRESENTATIVES, FIXTURE_STATIC_THRESHOLD)
        print("D_Omega placeholder, leave-one-dataset-out folds:")
        for ds, fold in d.describe()["folds"].items():
            print(f"  {ds}: {fold}")
    print(f"evaluation units per selector: {len(tables['units'])}; result rows: {len(res)}; "
          f"schema validated")
    res.to_parquet(out / "selection_results.parquet", index=False)

    s = summarize(res, by=("selector", "dataset"))
    s.to_csv(out / "summary_by_dataset.csv", index=False, lineterminator="\n")
    print("\nmean scaled regret by selector and dataset (both horizons, all 11 tau):")
    print(s.pivot(index="selector", columns="dataset", values="mean_scaled_regret").round(5))
    if a.d_omega == "k3_selector":
        dm = res[res.selector == "d_omega"]
        none = dm[dm.status != "selected"]
        print(f"\nD_Omega units without a choice: {len(none)} of {len(dm)}. They carry no cost, are left out "
              "of the means above and count as not choosing TSFM below.")
        if len(none):
            print(none.assign(reason=k3_adapter.reason_of(none.note).fillna(none.note))
                  .groupby(["dataset", "status", "reason"]).size().rename("units").to_string())

    flip = res[(res.dataset == "fx_tau_flip") & (res.horizon == 24)]
    tab = (flip.groupby(["selector", "tau"]).chosen_family
           .agg(lambda x: f"{(x == 'tsfm').sum()}/{len(x)}").unstack("tau"))
    print("\nfx_tau_flip, horizon 24: series choosing TSFM, out of 6")
    print(tab)
    print("planted pinball winner: ",
          " ".join(planted[planted.dataset == "fx_tau_flip"].planted_pinball_winner.str[:4]))


if __name__ == "__main__":
    main()
