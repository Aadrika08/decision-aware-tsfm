"""python -m decision_aware_tsfm.analysis.decision.run_fixture --out DIR"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .evaluate import summarize
from .fixture import (FAMILY_OF_MODEL, FIT_ONLY_DATASETS, FIXTURE_SCHEMA,
                      FIXTURE_STATIC_THRESHOLD, REPRESENTATIVES, make_fixture,
                      make_fixture_scales)
from .pipeline import build_tables, run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/runs/decision_fixture")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)

    fc, omega, planted = make_fixture(a.seed)
    print(f"fixture: {len(fc):,} forecast rows, {fc.series_id.nunique()} series, "
          f"{fc.model.nunique()} models, horizons {sorted(int(h) for h in fc.horizon.unique())}")
    fc.to_parquet(out / "fixture_cache.parquet", index=False)
    omega.to_csv(out / "fixture_omega.csv", index=False, lineterminator="\n")
    planted.to_csv(out / "planted_answers.csv", index=False, lineterminator="\n")
    scales = make_fixture_scales(a.seed)
    scales.to_csv(out / "fixture_scales.csv", index=False, lineterminator="\n")

    tables = build_tables(out / "fixture_cache.parquet", FIXTURE_SCHEMA, REPRESENTATIVES,
                          FAMILY_OF_MODEL, omega, fit_only_datasets=FIT_ONLY_DATASETS, scales=scales)
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

    flip = res[(res.dataset == "fx_tau_flip") & (res.horizon == 24)]
    tab = (flip.groupby(["selector", "tau"]).chosen_family
           .agg(lambda x: f"{(x == 'tsfm').sum()}/{len(x)}").unstack("tau"))
    print("\nfx_tau_flip, horizon 24: series choosing TSFM, out of 6")
    print(tab)
    print("planted pinball winner: ",
          " ".join(planted[planted.dataset == "fx_tau_flip"].planted_pinball_winner.str[:4]))


if __name__ == "__main__":
    main()
