"""Regenerate the E3 dataset, score it with the decision engine, write the tables.

    # full run (generates the series; about 8 minutes)
    PYTHONPATH=src python -m decision_aware_tsfm.analysis.experiments.e3.run \
        --out artifacts/runs/e3_matched_omega

    # rebuild every table in a NEW directory from series saved by an earlier run
    PYTHONPATH=src python -m decision_aware_tsfm.analysis.experiments.e3.run \
        --out artifacts/runs/e3_matched_omega_rev1 --series-from artifacts/runs/e3_matched_omega

Scoring always reads the series back from a saved series_values.parquet. With
--reuse-units the engine is not called again: the saved unit_costs.csv of the
series directory is read back instead (seconds instead of minutes).

What the tables are (see the E3 note for how to read them):
  population (closed form, the result)   breakeven_error.csv, f_win_count_by_error.csv,
                                         breakeven_near_primary_error.csv, expected_delta.csv
  simulation (a check on noise)          unit_costs.csv, sign_*.csv, member_delta_summary*.csv
  verification                           verify_innovation_only.csv, verify_no_series.csv,
                                         verify_no_series_vs_matched.csv, verify_sign_by_match_status.csv
  dataset and match                      series_*.csv, match_*.csv

Synthetic series and idealised forecasters only. No model is run or imported.
The Omega-only simulation is a separate command (omega_only.py).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from decision_aware_tsfm.analysis.decision.metrics import TAU_GRID

from . import design
from .generator import group_table, make_dataset, match_table
from .omega_reconcile import reconcile
from .population import breakeven_table, f_win_counts, near_knife_edge
from .scoring import (count_sign_differences, expected_delta_table, member_delta_summary,
                      score_dataset, sign_by_population_status, sign_difference_vs_baseline,
                      sign_pairs, wilson_interval)
from .verify import (NO_SERIES_REPLICATES, compare_with_engine, innovation_only_costs,
                     no_series_simulation, no_series_vs_matched, sign_by_match_status)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_table(path) -> pd.DataFrame:
    """Read a saved CSV table back exactly. The default pandas float parser
    can be off in the last digit, which is enough to change an Omega value
    and, at the tolerance boundary, a match status."""
    return pd.read_csv(path, float_precision="round_trip")


def match_summary(matches: pd.DataFrame) -> pd.DataFrame:
    g = matches.groupby(["pair_kind", "structure_a", "structure_b"], sort=True)
    return g.agg(n_pairs=("status", "size"),
                 n_matched=("status", lambda s: int((s == "matched").sum())),
                 n_failed=("status", lambda s: int((s == "failed").sum())),
                 n_as_expected=("as_expected", "sum"),
                 abs_diff_mean=("omega_levels_abs_diff", "mean"),
                 abs_diff_max=("omega_levels_abs_diff", "max"),
                 diffs_abs_diff_mean=("omega_diffs_abs_diff", "mean")).reset_index()


def match_by_member(matches: pd.DataFrame) -> pd.DataFrame:
    """Match outcome per (structure, member) and per member over all scored
    structures (structure = "all"), with a Wilson interval for the failure rate."""
    m = matches[matches["pair_kind"] != "negative_control"]
    parts = [m.assign(structure=m["structure_a"]), m.assign(structure="all")]
    rows = []
    for (sname, mname), d in pd.concat(parts).groupby(["structure", "member_b"], sort=False):
        n, k = len(d), int((d["status"] == "failed").sum())
        lo, hi = wilson_interval(k, n)
        rows.append(dict(structure=sname, member=mname, pair_kind=d["pair_kind"].iloc[0], n_pairs=n,
                         n_failed=k, fail_rate=k / n, fail_lo=lo, fail_hi=hi,
                         abs_diff_mean=d["omega_levels_abs_diff"].mean()))
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="artifacts/runs/e3_matched_omega")
    ap.add_argument("--n-seeds", type=int, default=len(design.DEFAULT_SEEDS))
    ap.add_argument("--skip-part-a", action="store_true")
    ap.add_argument("--reuse-series", action="store_true",
                    help="do not regenerate; score the series_values.parquet already in --out")
    ap.add_argument("--series-from", default=None,
                    help="directory of an earlier run whose saved series are read (never written to)")
    ap.add_argument("--reuse-units", action="store_true",
                    help="read unit_costs.csv from the series directory instead of calling the engine")
    ap.add_argument("--no-series-replicates", type=int, default=NO_SERIES_REPLICATES,
                    help="replicates per member in the no-series verification simulation")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    src = Path(a.series_from) if a.series_from else out
    if a.series_from and src.resolve() == out.resolve():
        raise SystemExit("--series-from must be a different directory from --out")
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_rows", 200)
    t0 = time.time()
    seeds = tuple(range(a.n_seeds))

    if not a.skip_part_a:
        r = reconcile()
        r["sweep"].to_csv(out / "part_a_omega_sweep.csv", index=False, lineterminator="\n")
        summ = dict(sd_for_omega_0p32=r["sd_for_target"], dayson_n1000_sd0p5_seed42=r["dayson"],
                    abdullah_n2000_sd1p5=r["abdullah"])
        (out / "part_a_summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
        print("PART A  Omega of sin(2 pi t / 24) + sd * noise")
        print(r["table"].round(4).to_string(index=False))
        print(json.dumps(summ, indent=2))

    vpath = src / "series_values.parquet"
    if not (a.reuse_series or a.series_from):
        values, series = make_dataset(seeds)
        values.to_parquet(vpath, index=False)
        series.to_csv(out / "series_table.csv", index=False, lineterminator="\n")
    # everything below uses the saved inputs, read back exactly
    values = pd.read_parquet(vpath)
    series = read_table(src / "series_table.csv")
    if a.series_from:
        series.to_csv(out / "series_table.csv", index=False, lineterminator="\n")
    seeds = tuple(sorted(int(s) for s in series.seed.unique()))
    print(f"\nPART B  {len(series)} series, context {design.N_CONTEXT} + test {design.N_TEST}, "
          f"seeds {seeds[0]}..{seeds[-1]}, Omega tolerance {design.OMEGA_TOLERANCE}"
          + (f"  (series read from {src})" if a.series_from else ""))

    matches = match_table(series)
    groups = group_table(series, matches)
    msum = match_summary(matches)
    mmem = match_by_member(matches)
    matches.to_csv(out / "match_pairs.csv", index=False, lineterminator="\n")
    groups.to_csv(out / "match_groups.csv", index=False, lineterminator="\n")
    msum.to_csv(out / "match_summary.csv", index=False, lineterminator="\n")
    mmem.to_csv(out / "match_by_member.csv", index=False, lineterminator="\n")
    by_member = (series.groupby(["structure", "member"])
                 [["omega_levels", "omega_diffs", "skewness", "excess_kurtosis", "tail_weight",
                   "tail_asym", "innov_variance", "innov_skewness", "innov_excess_kurtosis",
                   "innov_tail_weight", "innov_tail_asym"]].agg(["mean", "std"]))
    by_member.columns = [f"{c}_{s}" for c, s in by_member.columns]
    by_member.reset_index().to_csv(out / "series_summary_by_member.csv", index=False, lineterminator="\n")
    print(msum.round(4).to_string(index=False))
    print("match failures by member (n_failed of n_pairs):")
    print(mmem.pivot(index="member", columns="structure", values="n_failed").to_string())
    print("groups (structure, seed) with every member within tolerance of the reference:")
    print(groups.groupby("structure").status.value_counts().unstack(fill_value=0))

    # ---- population (closed form): the result. Uses no generated series.
    expected = expected_delta_table()
    breakeven = breakeven_table()
    wins = f_win_counts(breakeven)
    primary_error = design.F_VARIANTS[design.PRIMARY_VARIANT].location_error
    knife = near_knife_edge(breakeven, primary_error)
    expected.to_csv(out / "expected_delta.csv", index=False, lineterminator="\n")
    breakeven.to_csv(out / "breakeven_error.csv", index=False, lineterminator="\n")
    wins.to_csv(out / "f_win_count_by_error.csv", index=False, lineterminator="\n")
    knife.to_csv(out / "breakeven_near_primary_error.csv", index=False, lineterminator="\n")
    alt = breakeven[breakeven.member.isin(design.ALTERNATIVES)]
    print("\nPOPULATION  break-even location error (sigma units); F is cheaper than G only below it")
    print(alt.pivot(index="tau", columns="member", values="breakeven_pos")
          [list(design.ALTERNATIVES)].round(3).to_string())
    print("cells (of 66) where F is the population winner, by size of F's location error:")
    print(wins.to_string(index=False))
    print(f"break-evens within 0.05 of the primary error {primary_error}: {len(knife)} cells")

    # ---- simulation: a check on finite-sample noise
    if a.reuse_units:
        units = read_table(src / "unit_costs.csv")
        if set(units["series_id"]) != set(series.loc[series.structure.isin(design.SCORED_STRUCTURES),
                                                     "series_id"]):
            raise SystemExit("saved unit_costs.csv does not cover the saved series")
    else:
        units = score_dataset(values, series)
    units.to_csv(out / "unit_costs.csv", index=False, lineterminator="\n")
    pairs = sign_pairs(units, matches)
    pairs.to_csv(out / "sign_pairs.csv", index=False, lineterminator="\n")
    by_tau = count_sign_differences(pairs, by=("variant", "pair_kind", "tau"))
    by_member_tau = count_sign_differences(pairs, by=("variant", "pair_kind", "member_b", "tau"))
    by_struct_tau = count_sign_differences(pairs, by=("variant", "pair_kind", "structure", "tau"))
    vs_base = sign_difference_vs_baseline(pairs)
    by_status = sign_by_population_status(pairs, expected)
    by_status_member = sign_by_population_status(
        pairs, expected, by=("variant", "pair_kind", "member_b", "tau", "cell_status"))
    by_status_all = sign_by_population_status(pairs, expected, by=("variant", "pair_kind", "cell_status"))
    by_tau.to_csv(out / "sign_summary_by_tau.csv", index=False, lineterminator="\n")
    by_member_tau.to_csv(out / "sign_summary_by_member_tau.csv", index=False, lineterminator="\n")
    by_struct_tau.to_csv(out / "sign_summary_by_structure_tau.csv", index=False, lineterminator="\n")
    vs_base.to_csv(out / "sign_difference_vs_baseline.csv", index=False, lineterminator="\n")
    by_status.to_csv(out / "sign_by_population_status_tau.csv", index=False, lineterminator="\n")
    by_status_member.to_csv(out / "sign_by_population_status_member_tau.csv", index=False, lineterminator="\n")
    by_status_all.to_csv(out / "sign_by_population_status_total.csv", index=False, lineterminator="\n")
    member_delta_summary(units).to_csv(out / "member_delta_summary.csv", index=False, lineterminator="\n")
    member_delta_summary(units, by=("variant", "structure", "member", "tau")).to_csv(
        out / "member_delta_summary_by_structure.csv", index=False, lineterminator="\n")

    print("\nSIMULATION  sign differences within Omega-matched pairs (engine costs), by F variant")
    for v in design.F_VARIANTS:
        d = vs_base[vs_base.variant == v]
        print(f"\nvariant {v}: {design.F_VARIANTS[v].model}   (alternative rate minus baseline rate)")
        print(d.drop(columns=["variant"]).round(4).to_string(index=False))
    print("\nsplit by population-winner status (location variants):")
    print(by_status_all.round(4).to_string(index=False))

    # ---- verification: the costs do not depend on the series, Omega or the match
    innov = innovation_only_costs(values, series)
    vinn = compare_with_engine(units, innov)
    nos = no_series_simulation(n_replicates=a.no_series_replicates)
    nosm = no_series_vs_matched(nos, pairs)
    bystat = sign_by_match_status(pairs)
    vinn.to_csv(out / "verify_innovation_only.csv", index=False, lineterminator="\n")
    nos.to_csv(out / "verify_no_series.csv", index=False, lineterminator="\n")
    nosm.to_csv(out / "verify_no_series_vs_matched.csv", index=False, lineterminator="\n")
    bystat.to_csv(out / "verify_sign_by_match_status.csv", index=False, lineterminator="\n")
    print("\nVERIFICATION  engine costs against costs recomputed from the innovations alone")
    print(vinn.to_string(index=False))
    print("sign-difference rate with no series at all, next to the matched-pair rate (primary variant):")
    print(nosm.drop(columns=["variant"]).round(4).to_string(index=False))
    print("pairs that passed and pairs that failed the Omega match (primary variant):")
    print(bystat[bystat.variant == design.PRIMARY_VARIANT].drop(columns=["variant"])
          .round(4).to_string(index=False))

    manifest = dict(
        command="python -m decision_aware_tsfm.analysis.experiments.e3.run",
        out=str(out), series_from=str(src), units_from_engine=not a.reuse_units,
        seeds=list(map(int, seeds)), n_context=design.N_CONTEXT, n_test=design.N_TEST,
        horizon=design.HORIZON, burn_in=design.BURN_IN, omega_tolerance=design.OMEGA_TOLERANCE,
        omega_match_column=design.OMEGA_MATCH_COLUMN, tau_grid=list(TAU_GRID),
        structures={k: dict(kind=s.kind, phi=s.phi, sigma=s.sigma, scored=s.scored)
                    for k, s in design.STRUCTURES.items()},
        members={k: dict(kind=m.kind, param=None if np.isnan(m.param) else m.param, role=m.role)
                 for k, m in design.MEMBERS.items()},
        f_variants={k: dict(model=v.model, kind=v.kind, location_error=v.location_error)
                    for k, v in design.F_VARIANTS.items()},
        series_values_sha256=_sha256(vpath), n_series=int(len(series)),
        no_series_simulation=dict(replicates=int(nos.n_replicates.iloc[0]), n_test=int(nos.n_test.iloc[0])),
        versions=dict(python=platform.python_version(), numpy=np.__version__,
                      pandas=pd.__version__, scipy=scipy.__version__),
        runtime_seconds=round(time.time() - t0, 1),
        label="synthetic series, idealised forecasters; mechanism check, not a result about real models")
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nwrote tables to {out}  ({manifest['runtime_seconds']} s)")


if __name__ == "__main__":
    main()
