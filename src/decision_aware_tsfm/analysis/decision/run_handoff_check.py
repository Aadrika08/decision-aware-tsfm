"""Consumer check of a producer handoff bundle with the decision engine.

    PYTHONPATH=src python -m decision_aware_tsfm.analysis.decision.run_handoff_check BUNDLE \
        --variant current --classical AutoETS --tsfm Moirai-small \
        --static-threshold 0.45 --fit-only-datasets fx_dfit --out artifacts/runs/decision-handoff

Reads the bundle (hashes, provenance, Omega join), runs all seven comparators,
and recomputes every test-period cost directly from the raw quantile columns as
an independent check that the engine reads the right column for each tau.
Writes selection_results.parquet and report.json to --out (a new directory).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .cache_io import qcol
from .handoff import read_handoff
from .metrics import TAU_GRID
from .pipeline import FIT_SCHEMES, build_tables, run


def direct_test_costs(forecasts: pd.DataFrame, per_origin: pd.DataFrame,
                      representatives: dict) -> pd.DataFrame:
    """Mean pinball per (series, horizon, tau, family) over test origins, computed
    straight from producer rows without the engine's cost code."""
    test = per_origin.loc[per_origin.role == "test",
                          ["dataset", "config", "series_id", "horizon", "origin"]].drop_duplicates()
    f = forecasts[forecasts.model.isin(representatives.values())].merge(
        test, on=["dataset", "config", "series_id", "horizon", "origin"])
    out = []
    for tau in TAU_GRID:
        u = f["y_true"].to_numpy() - f[qcol(tau)].to_numpy()
        loss = np.where(u >= 0, tau * u, (tau - 1) * u)
        g = f[["dataset", "config", "series_id", "horizon", "family"]].assign(loss=loss)
        m = g.groupby(["dataset", "config", "series_id", "horizon", "family"])["loss"].mean()
        out.append(m.unstack("family").add_prefix("direct_J_").reset_index().assign(tau=tau))
    return pd.concat(out, ignore_index=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundle", type=Path)
    ap.add_argument("--variant", required=True, help="Omega variant, e.g. current or last_4096")
    ap.add_argument("--classical", required=True, help="frozen classical representative model")
    ap.add_argument("--tsfm", required=True, help="frozen TSFM representative model")
    ap.add_argument("--static-threshold", required=True, type=float,
                    help="static Omega threshold (no default: must be the recorded value)")
    ap.add_argument("--fit-only-datasets", nargs="*", default=[])
    ap.add_argument("--val-fraction", type=float, default=1 / 3)
    ap.add_argument("--fit-scheme", choices=FIT_SCHEMES, default="leave_one_dataset_out")
    ap.add_argument("--min-fit-datasets", type=int, default=3,
                    help="other datasets a leave-one-dataset-out fold needs (decision record 0007: 3)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"--out {a.out} already exists; choose a new directory")

    forecasts, omega, report = read_handoff(a.bundle, a.variant)
    reps = {"classical": a.classical, "tsfm": a.tsfm}
    family_of_model = forecasts.groupby("model")["family"].first().to_dict()
    tables = build_tables(forecasts, "producer-v1", reps, family_of_model, omega,
                          fit_only_datasets=tuple(a.fit_only_datasets), val_fraction=a.val_fraction,
                          omega_col="omega", omega_variant=a.variant,
                          # interface check: a bundle may have too few origins per series for a
                          # validation third; keep those series so every column is still checked
                          exclude_without_validation=False)
    res, d_omega = run(tables, reps, a.static_threshold, fit_scheme=a.fit_scheme,
                       min_fit_datasets=a.min_fit_datasets)

    direct = direct_test_costs(forecasts, tables["per_origin"], reps)
    cmp = tables["test"].merge(direct, on=["dataset", "config", "series_id", "horizon", "tau"],
                               how="outer", indicator=True)
    unmatched = int((cmp["_merge"] != "both").sum())
    rel = max(float((np.abs(cmp[f"J_{f}"] - cmp[f"direct_J_{f}"])
                     / np.maximum(1e-12, np.abs(cmp[f"direct_J_{f}"]))).max()) for f in reps)
    # what a reader that used q_0.50 for every tau would have produced
    fm = forecasts[forecasts.model.isin(reps.values())]
    med_diff = float(np.abs(fm[qcol(0.01)] - fm[qcol(0.50)]).max())

    report.update(
        schema="producer-v1", representatives=reps, static_threshold=a.static_threshold,
        fit_only_datasets=list(a.fit_only_datasets), val_fraction=a.val_fraction,
        fit_scheme=a.fit_scheme, d_omega=d_omega.describe(),
        series_without_validation_kept="interface check only; an analysis run excludes them",
        evaluation_units=int(len(tables["units"])), result_rows=int(len(res)),
        status_counts={f"{s}/{st}": int(n) for (s, st), n in res.groupby(["selector", "status"]).size().items()},
        undefined_omega_decisions={
            f"{u}/{r}": int(n) for (u, r), n in
            tables["undefined_omega"].groupby(["used_for", "reason"], dropna=False).size().items()},
        column_check=dict(units_compared=int((cmp["_merge"] == "both").sum()), unmatched=unmatched,
                          max_relative_difference=rel,
                          q001_vs_q050_max_gap_in_bundle=med_diff))
    a.out.mkdir(parents=True)
    res.to_parquet(a.out / "selection_results.parquet", index=False)
    (a.out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if report["fixture"]:
        print("SYNTHETIC FIXTURE: checks the interface only, not research results.")
    print(json.dumps(report, indent=2, sort_keys=True))
    if unmatched or rel > 1e-10:
        raise SystemExit("column check FAILED: engine costs differ from direct recomputation")


if __name__ == "__main__":
    main()
