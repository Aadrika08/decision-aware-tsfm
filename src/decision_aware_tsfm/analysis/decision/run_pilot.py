"""One command from a forecast cache handoff bundle to every pilot result (#37).

    PYTHONPATH=src python -m decision_aware_tsfm.analysis.decision.run_pilot BUNDLE --out DIR

BUNDLE is a producer handoff bundle (handoff.json, partitions, omega.json). The
workload the cache was planned from is read from BUNDLE/workload.json, or from
--workload FILE (Dayson's workload.json). DIR must not exist; the run writes into
DIR.partial and renames it to DIR only when every step has passed.

Steps, in order. Every default is a decision posted on #29 (DECISIONS_REFERENCE):
  1. Load. handoff.read_handoff, exactly as run_handoff_check: every partition hash,
     the producer-v1 columns and types, job provenance, the Omega join for the chosen
     variant (default `current`; `last_4096` with --variant). Held-out names (M5, PJM)
     in dataset, config or series_id stop the run.
  2. Roles. The role table is built from the workload with the #60 code
     (role_table.build_role_table, record 0007). It must agree with the roles the
     engine gives the origins in the cache, for every (series, horizon) the cache
     holds; a bundle-supplied role table (BUNDLE/role_table.csv, BUNDLE/role_table/
     or --role-table) must agree with the workload. Any disagreement stops the run.
     A (series, horizon) of the workload with no rows in the cache is not an error;
     it is counted in the coverage table.
  3. Representatives. representatives.best_of_family, loss "scaled", validation
     origins only, frozen to DIR/representatives/. --fixed-classical NAME adds the
     fixed-named sensitivity run (that classical model against the same TSFM).
  4. Engine. pipeline.build_tables and pipeline.run: the seven comparators, leave-one-
     dataset-out with at least 3 fitting datasets, one run per static threshold
     (default 0.45, then 0.4 and 0.5; the first is the headline). D_Omega is the
     placeholder on main; d_omega_builder in run_pilot() is the seam where the K3
     selector adapter plugs in. The test costs are recomputed straight from the raw
     quantile columns (run_handoff_check.direct_test_costs) and must match.
  5. E1 and E2 on the headline run: e1.run_e1 and e1.mean_delta_intervals; e2.fit,
     freeze, load_frozen and evaluate once per leave-one-dataset-out fold (its choices
     must equal the engine's d_omega choices), and e2's regret figures with
     moving-block bootstrap intervals (bootstrap.fixed_choice_costs, the default
     fixed-oracle statistic, datasets and series as clusters).
  6. Paper tables in DIR/tables and, with --figures and matplotlib installed (it is
     not pinned), two figures. Without matplotlib the figures are skipped with a
     message; the tables they would draw are named in summary.md.
  7. summary.md, manifest.json (input and code SHA-256, every output file's SHA-256)
     and environment.json (interpreter and package versions; the only file that
     differs between Python 3.10 and 3.11).

Two runs on the same bundle give byte-identical files, apart from environment.json
when the interpreter differs. Nothing here runs a forecasting model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow
import scipy

from ..experiments import bootstrap, e1, e2
from . import role_table as rt
from .cache_io import CacheError
from .evaluate import run_selectors
from .guards import HeldOutLeakError, assert_no_held_out
from .handoff import HandoffError, file_sha256, read_handoff
from .metrics import TAU_GRID
from .pipeline import build_tables, fit_pool, run, seven_selectors
from .representatives import (EvaluationRowsError, best_of_family, fixed_named, freeze,
                              score_table)
from .run_handoff_check import direct_test_costs
from .run_role_table import _git_commit, _tree_sha256
from .schema import UNIT_KEYS

DECISIONS_REFERENCE = ("https://github.com/Aadrika08/decision-aware-tsfm/issues/29"
                       "#issuecomment-6185355042")
# hashes named in the #29 decisions; a run on other inputs says so in summary.md
APPROVED_WORKLOAD_SHA256 = "1474953eb7898c3f65986b6c54974af02b52b0be63eb39b58bdc246d3d979516"
APPROVED_ROLE_TABLE_SHA256 = "f5d5024359a9f7e69364a49229f65e316337424c241cd87688a95dc78e7ed7da"
PRODUCER_SCHEMA = "producer-v1"
DEFAULT_VARIANT = "current"                        # #29 item 5
DEFAULT_THRESHOLDS = (0.45, 0.4, 0.5)              # #29 item 10, headline first
MIN_FIT_DATASETS = 3                               # #29 item 6, record 0007 section 2
VAL_FRACTION = 1 / 3                               # #29 item 4, record 0007 section 1
REPRESENTATIVE_LOSS = "scaled"                     # #29 item 7
FIT_SCHEME = "leave_one_dataset_out"
BOOT_CLUSTER = "dataset+series"                    # record 0007, consequences
SELECTOR_ORDER = ("always_classical", "always_tsfm", "static_omega", "pinball_validation",
                  "crps_validation", "d_omega", "hindsight_oracle")
ORACLE = "hindsight_oracle"
DEFERRAL_REFERENCE = e2.VALIDATION_COMPARATOR      # #29 item 8: a deferral costs what this rule pays
ENVIRONMENT_FILE = "environment.json"
MANIFEST_FILE = "manifest.json"
SUMMARY_FILE = "summary.md"
FIGURES_MISSING = "skipped: matplotlib is not installed (it is not a pinned dependency)"
FIGURES_OFF = "not requested (pass --figures)"
CODE_DIRS = ("analysis/decision", "analysis/experiments")


class PilotRunError(RuntimeError):
    """The pilot run cannot go on: an input is missing or a check failed."""


# --------------------------------------------------------------------------- small helpers

def _csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", encoding="utf-8")


def _json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def _label(threshold: float) -> str:
    return f"threshold_{threshold:g}"


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(name))


def _order_selectors(frame: pd.DataFrame, keys) -> pd.DataFrame:
    rank = {s: i for i, s in enumerate(SELECTOR_ORDER)}
    return (frame.assign(_o=frame["selector"].map(lambda s: rank.get(s, len(rank))))
            .sort_values(["_o", "selector"] + [k for k in keys if k != "selector"])
            .drop(columns="_o").reset_index(drop=True))


def code_sha256(package_root: Path | None = None) -> tuple[str, list[str]]:
    """SHA-256 over the source files of the decision and experiments packages:
    sorted lines of (path relative to the package, file SHA-256)."""
    root = package_root or Path(__file__).resolve().parents[2]
    files = sorted(p for d in CODE_DIRS for p in (root / d).glob("*.py"))
    lines = [f"{p.relative_to(root).as_posix()} {file_sha256(p)}\n" for p in files]
    return (hashlib.sha256("".join(lines).encode("utf-8")).hexdigest(),
            [p.relative_to(root).as_posix() for p in files])


# --------------------------------------------------------------------------- step 1: load

def load_bundle(bundle: Path, variant: str):
    """read_handoff (every hash and format check), then the held-out guard on names."""
    forecasts, omega, report = read_handoff(bundle, variant)
    for col in ("dataset", "config", "series_id"):
        assert_no_held_out(forecasts[col].unique(), f"pilot run: bundle {col}")
        assert_no_held_out(omega[col].unique(), f"pilot run: omega.json {col}")
    family_of_model = forecasts.groupby("model")["family"].first().to_dict()
    return forecasts, omega, report, family_of_model


# --------------------------------------------------------------------------- step 2: roles

def _read_table_input(path: Path):
    """A role table CSV, or a directory written by run_role_table export."""
    folder = path if path.is_dir() else None
    table = rt.read_csv(folder / rt.FILES["table"] if folder else path)
    side = {name: (rt.read_csv(folder / rt.FILES[name])
                   if folder and (folder / rt.FILES[name]).is_file() else None)
            for name in ("excluded", "decision_origins")}
    return table, side


def _restrict(frame: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    return frame.merge(pairs, on=rt.SERIES_KEYS, how="inner")


def check_roles(forecasts: pd.DataFrame, workload_origins: pd.DataFrame,
                bundle_table=None) -> tuple[rt.RoleTable, dict]:
    """Build the role table from the workload and check it against the cache (and a
    bundle-supplied table). Raises rt.RoleTableMismatch on any disagreement."""
    want = rt.build_role_table(workload_origins, VAL_FRACTION)
    report = dict(bundle_role_table=None)
    if bundle_table is not None:
        table, side = bundle_table
        r = rt.check_role_table(table, workload_origins, VAL_FRACTION, excluded=side["excluded"],
                                decision_origins=side["decision_origins"], strict=True)
        report["bundle_role_table"] = dict(ok=r["ok"], rows=r["table_rows"], matched=r["matched_rows"])

    cache = rt.distinct_origins(forecasts, "handoff bundle")
    cache_pairs = cache[rt.SERIES_KEYS].drop_duplicates()
    work_pairs = want.table[rt.SERIES_KEYS].drop_duplicates()
    unplanned = cache_pairs.merge(work_pairs, how="left", indicator=True)
    unplanned = unplanned[unplanned["_merge"] == "left_only"]
    if len(unplanned):
        raise rt.RoleTableMismatch(
            f"{len(unplanned)} (series, horizon) pairs in the cache are not in the workload, e.g. "
            f"{unplanned[rt.SERIES_KEYS].head(3).to_dict('records')}")
    # roles are assigned per (series, horizon) from all its origins, so a pair the cache holds
    # only in part (a first slice, an interrupted run) would get roles that differ from the table
    n_cache = cache.groupby(rt.SERIES_KEYS).size().rename("in_cache")
    n_work = _restrict(want.table, cache_pairs).groupby(rt.SERIES_KEYS).size().rename("in_workload")
    part = pd.concat([n_cache, n_work], axis=1).reset_index()
    part = part[part["in_cache"] != part["in_workload"]]
    if len(part):
        raise rt.RoleTableMismatch(
            f"the cache holds only part of the workload's origins for {len(part)} (series, horizon) "
            f"pairs, e.g. {part.head(3).to_dict('records')}. Roles come from all of a pair's origins, "
            "so run on a cache that holds every origin of each pair it holds")
    r = rt.check_role_table(_restrict(want.table, cache_pairs), cache, VAL_FRACTION,
                            excluded=_restrict(want.excluded, cache_pairs),
                            decision_origins=_restrict(want.decision_origins, cache_pairs),
                            strict=True)
    report.update(cache_vs_workload=dict(ok=r["ok"], rows=r["table_rows"], matched=r["matched_rows"]),
                  workload_series_horizons=int(len(work_pairs)),
                  cache_series_horizons=int(len(cache_pairs)),
                  workload_series_horizons_absent_from_cache=int(len(work_pairs) - len(cache_pairs)))
    if len(_restrict(want.excluded, cache_pairs)) == len(cache_pairs):
        raise PilotRunError(f"none of the {len(cache_pairs)} (series, horizon) pairs in the cache has a "
                            "validation origin (record 0007 excludes them all); nothing can be scored")
    return want, report


def check_engine_roles(tables: dict, want: rt.RoleTable) -> dict:
    """The roles and decision origins the engine actually used (its representatives'
    origins, after its exclusions) must equal the role table on the pairs it kept."""
    used = tables["per_origin"][rt.SERIES_KEYS + ["origin", "role"]].drop_duplicates()
    kept = used[rt.SERIES_KEYS].drop_duplicates()
    table = _restrict(want.table, kept).assign(role=lambda d: d["role"].map(rt.ENGINE_ROLE))
    m = used.merge(table, on=rt.SERIES_KEYS + ["origin"], how="outer", suffixes=("_engine", "_table"),
                   indicator=True)
    bad = m[(m["_merge"] != "both") | (m["role_engine"] != m["role_table"])]
    dec = tables["decision_origins"].merge(_restrict(want.decision_origins, kept), on=rt.SERIES_KEYS,
                                           how="outer", suffixes=("_engine", "_table"), indicator=True)
    bad_dec = dec[(dec["_merge"] != "both")
                  | (dec["decision_origin_engine"] != dec["decision_origin_table"])]
    if len(bad) or len(bad_dec):
        raise rt.RoleTableMismatch(
            f"the engine used roles that differ from the role table: {len(bad)} origin rows, "
            f"{len(bad_dec)} decision origins, e.g. "
            f"{(bad if len(bad) else bad_dec).drop(columns='_merge').head(2).to_dict('records')}")
    return dict(ok=True, origin_rows=int(len(used)), series_horizons=int(len(kept)))


# --------------------------------------------------------------------------- step 3: representatives

def choose_representatives(forecasts, family_of_model, source: dict, fixed_classical=None):
    """(best_of_family payload, fixed_named payload or None). Validation origins only."""
    best = best_of_family(forecasts, PRODUCER_SCHEMA, family_of_model, val_fraction=VAL_FRACTION,
                          loss=REPRESENTATIVE_LOSS, source=source)
    fixed = None
    if fixed_classical is not None:
        named = {"classical": fixed_classical, "tsfm": best["representatives"]["tsfm"]}
        fixed = fixed_named(forecasts, PRODUCER_SCHEMA, family_of_model, named,
                            val_fraction=VAL_FRACTION, loss=REPRESENTATIVE_LOSS, source=source)
    return best, fixed


# --------------------------------------------------------------------------- step 4: engine

def engine_tables(forecasts, omega, reps, family_of_model, variant):
    return build_tables(forecasts, PRODUCER_SCHEMA, reps, family_of_model, omega,
                        val_fraction=VAL_FRACTION, omega_col="omega", omega_variant=variant)


def run_engine(tables, reps, threshold, min_fit_datasets, d_omega_builder=None):
    """The seven comparators at one static threshold. d_omega_builder(tables) -> selector
    replaces the placeholder D_Omega (the K3 adapter's seam); None runs main's pipeline.run."""
    if d_omega_builder is None:
        return run(tables, reps, threshold, fit_scheme=FIT_SCHEME, min_fit_datasets=min_fit_datasets)
    d_omega = d_omega_builder(tables)
    res = run_selectors(seven_selectors(threshold, d_omega), tables["units"], tables["val"],
                        tables["test"], reps, tables["n_val"])
    return res, d_omega


def column_check(forecasts, tables, reps) -> dict:
    """The engine's test costs against a direct recomputation from the raw quantile
    columns (run_handoff_check's check). Raises PilotRunError on any difference."""
    direct = direct_test_costs(forecasts, tables["per_origin"], reps)
    cmp = tables["test"].merge(direct, on=["dataset", "config", "series_id", "horizon", "tau"],
                               how="outer", indicator=True)
    unmatched = int((cmp["_merge"] != "both").sum())
    rel = max(float((np.abs(cmp[f"J_{f}"] - cmp[f"direct_J_{f}"])
                     / np.maximum(1e-12, np.abs(cmp[f"direct_J_{f}"]))).max()) for f in reps)
    if unmatched or not rel <= 1e-10:
        raise PilotRunError(f"column check failed: {unmatched} unmatched units, largest relative "
                            f"difference {rel}")
    return dict(units_compared=int(len(cmp)), unmatched=unmatched, max_relative_difference=rel)


# --------------------------------------------------------------------------- step 5: E1 and E2

def _interval_frame(ci, horizon: int, **extra) -> pd.DataFrame:
    f = ci.to_frame().rename(columns={"label": "tau"})
    f.insert(0, "horizon", int(horizon))
    for i, (k, v) in enumerate(extra.items()):
        f.insert(i, k, v)
    return f


def delta_intervals(tables, n_boot, boot_seed) -> tuple[pd.DataFrame, pd.DataFrame]:
    """e1.mean_delta_intervals one horizon at a time, so a horizon that cannot be
    resampled (for example a dataset left with one series) is skipped with the reason
    and the others are still computed. Returns (intervals, skipped)."""
    parts, skipped = [], []
    for h, po in tables["per_origin"].groupby("horizon", sort=True):
        try:
            parts.append(e1.mean_delta_intervals({"per_origin": po}, n_boot=n_boot, seed=boot_seed,
                                                 cluster=BOOT_CLUSTER))
        except bootstrap.DegenerateResampleError as exc:
            skipped.append(dict(horizon=int(h), reason=str(exc)))
    return (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(),
            pd.DataFrame(skipped, columns=["horizon", "reason"]))


def _fold_tables(tables, dataset) -> dict:
    keep = lambda f: f[f["dataset"] == dataset].reset_index(drop=True)
    return dict(units=keep(tables["units"]), val=keep(tables["val"]), test=keep(tables["test"]),
                n_val=keep(tables["n_val"]), settings=tables["settings"])


def run_e2_folds(tables, selection, threshold, min_fit_datasets, frozen_dir: Path,
                 selector_kind=e2.DOmegaLinearPlaceholder.kind):
    """E2's fit, freeze, load and evaluate once per leave-one-dataset-out fold. The fit
    units are the engine's fit pool without the fold's dataset (LeaveOneDatasetOut's
    pool), so the frozen choices must equal the engine's d_omega choices."""
    pool = fit_pool(tables).dropna(subset=["omega", "J_classical", "J_tsfm", "scale"])
    engine = selection[selection["selector"] == "d_omega"]
    rows, selections = [], []
    for d in sorted(tables["units"]["dataset"].unique()):
        fit_units = pool[pool["dataset"] != d].reset_index(drop=True)
        n_fit = int(fit_units["dataset"].nunique())
        ev = _fold_tables(tables, d)
        row = dict(fold=d, n_fit_datasets=n_fit, n_fit_units=int(len(fit_units)),
                   n_units=int(len(ev["units"])), fitted=False, reason="", frozen_file="",
                   frozen_sha256="", selector_params="", n_same_as_engine=0, n_different=0)
        if n_fit < min_fit_datasets:
            row["reason"] = f"{n_fit} other dataset(s) with usable units, need {min_fit_datasets}"
        elif ev["units"]["omega"].notna().sum() == 0:
            row["reason"] = "no unit of this dataset has a defined Omega"
        else:
            spec = e2.fit(dict(fit_units=fit_units, settings=tables["settings"]),
                          selector_kind=selector_kind, static_threshold=threshold)
            name = f"frozen_{_safe(d)}.json"
            e2.freeze(spec, frozen_dir / name)
            res = e2.evaluate(e2.load_frozen(frozen_dir / name), ev)
            mine = res.selection[res.selection["selector"] == res.selector_name]
            cmp = mine.merge(engine, on=UNIT_KEYS, how="left", suffixes=("", "_engine"), validate="1:1")
            same = (cmp["chosen_family"].fillna("") == cmp["chosen_family_engine"].fillna("")) \
                & (cmp["status"] == cmp["status_engine"])
            row.update(fitted=True, frozen_file=f"e2/frozen/{name}", frozen_sha256=res.frozen_sha256,
                       selector_params=json.dumps(spec.payload["selector_params"], sort_keys=True),
                       n_same_as_engine=int(same.sum()), n_different=int((~same).sum()))
            selections.append(res.selection.assign(fold=d, frozen_sha256=res.frozen_sha256))
        rows.append(row)
    check = pd.DataFrame(rows)
    if check["n_different"].sum():
        raise PilotRunError(f"E2's frozen d_omega chose differently from the engine on "
                            f"{int(check['n_different'].sum())} units: {check.to_dict('records')}")
    fold_sel = pd.concat(selections, ignore_index=True) if selections else pd.DataFrame()
    return check, fold_sel


def regret_intervals(tables, selection, n_boot, boot_seed) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Nominal 95% moving-block bootstrap intervals for each selector's mean scaled regret
    per (horizon, tau), on the units it decided, with its family and the oracle family
    held fixed (bootstrap.stat_mean_scaled_regret). Returns (intervals, skipped)."""
    parts, skipped = [], []
    for name in [s for s in SELECTOR_ORDER if s != ORACLE]:
        dec = selection[(selection["selector"] == name) & (selection["status"] == "selected")]
        for h, po in tables["per_origin"].groupby("horizon", sort=True):
            units = dec.loc[dec["horizon"] == h, UNIT_KEYS]
            if units.empty:
                skipped.append(dict(selector=name, horizon=int(h), reason="no decided units"))
                continue
            cost = bootstrap.fixed_choice_costs(po.merge(units, on=UNIT_KEYS), dec, name)
            try:
                ci = bootstrap.bootstrap_ci(
                    cost, lambda x: bootstrap.stat_mean_scaled_regret(x, by="tau"), n_boot=n_boot,
                    seed=boot_seed, cluster=BOOT_CLUSTER)
            except bootstrap.DegenerateResampleError as exc:
                skipped.append(dict(selector=name, horizon=int(h), reason=str(exc)))
                continue
            parts.append(_interval_frame(ci, h, selector=name, n_units_decided=int(len(units)))
                         .rename(columns={"estimate": "mean_scaled_regret"}))
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return out, pd.DataFrame(skipped, columns=["selector", "horizon", "reason"])


# --------------------------------------------------------------------------- step 6: tables

def deferral_priced(selection: pd.DataFrame, by) -> pd.DataFrame:
    """Per group: n_deferred (units the selector did not decide) and the mean scaled regret
    over all units when each such unit is charged what pinball_validation paid on it
    (#29 item 8). A unit neither decided is left out and counted in n_unpriced."""
    by = list(by)
    ref = selection.loc[selection["selector"] == DEFERRAL_REFERENCE,
                        UNIT_KEYS + ["status", "scaled_regret"]].rename(
        columns={"status": "ref_status", "scaled_regret": "ref_scaled_regret"})
    m = selection.merge(ref, on=UNIT_KEYS, how="left", validate="m:1")
    sel = m["status"] == "selected"
    priced = np.where(sel, m["scaled_regret"],
                      np.where(m["ref_status"] == "selected", m["ref_scaled_regret"], np.nan))
    g = m.assign(_deferred=~sel, _priced=priced, _unpriced=np.isnan(priced)).groupby(by)
    return g.agg(n_deferred=("_deferred", "sum"), n_unpriced=("_unpriced", "sum"),
                 mean_scaled_regret_deferrals_priced=("_priced", "mean")).reset_index().astype(
        {"n_deferred": "int64", "n_unpriced": "int64"})


def headline_table(selection: pd.DataFrame, by) -> pd.DataFrame:
    """e2.regret_summary (mean scaled regret, regret over oracle cost, fraction of oracle
    value recovered) plus the deferral columns, selectors in the engine's order."""
    by = list(by)
    t = e2.regret_summary(selection, by=by).merge(deferral_priced(selection, by), on=by,
                                                  validate="1:1")
    return _order_selectors(t, by)


def role_coverage(want: rt.RoleTable, cache_pairs: pd.DataFrame, tables: dict) -> pd.DataFrame:
    """Per dataset, config and horizon: the #60 role summary of the workload, then how many
    series the cache holds, how many the engine scored and how many it excluded and why."""
    s = rt.summarize(want)
    keys = rt.SUMMARY_KEYS
    count = lambda f, name: (f.groupby(keys)["series_id"].nunique().rename(name).reset_index()
                             if len(f) else pd.DataFrame(columns=keys + [name]))
    s = s.merge(count(cache_pairs, "series_in_cache"), on=keys, how="left")
    s = s.merge(count(tables["units"][rt.SERIES_KEYS].drop_duplicates(), "series_scored"),
                on=keys, how="left")
    ex = tables["excluded_series_horizons"]
    ex = ex[ex["used_for"] == "evaluation"] if len(ex) else ex
    reasons = {"zero scale": "excluded_zero_scale", "no validation": "excluded_no_validation"}
    for word, col in reasons.items():
        part = ex[ex["reason"].str.startswith(word)] if len(ex) else ex
        s = s.merge(count(part, col), on=keys, how="left")
    cols = ["series_in_cache", "series_scored"] + list(reasons.values())
    for c in cols:
        s[c] = pd.to_numeric(s[c]).fillna(0).astype("int64")
    return s


# --------------------------------------------------------------------------- figures

def write_figures(out: Path, headline_threshold: float) -> str:
    """Two PNG figures from the tables, if matplotlib can be imported. Returns a status."""
    try:
        import matplotlib
    except ImportError:
        return FIGURES_MISSING
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = pd.read_csv(out / "tables" / "headline_by_horizon_tau.csv")
    h = h[np.isclose(h["static_threshold"], headline_threshold)]
    dmap = pd.read_csv(out / "e1" / "decision_map.csv")
    horizons = sorted(int(x) for x in h["horizon"].unique())
    (out / "figures").mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(horizons), figsize=(5 * len(horizons), 4), squeeze=False)
    for ax, hz in zip(axes[0], horizons):
        for sel in SELECTOR_ORDER:
            d = h[(h["horizon"] == hz) & (h["selector"] == sel)].sort_values("tau")
            if len(d):
                ax.plot(d["tau"], d["mean_scaled_regret"], marker="o", ms=3, lw=1, label=sel)
        ax.set_title(f"horizon {hz}, static threshold {headline_threshold:g}", fontsize=10)
        ax.set_xlabel("tau")
        ax.set_ylabel("mean scaled regret")
        ax.grid(alpha=.25)
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "figures" / "regret_by_tau.png", dpi=150, metadata={"Software": None})
    plt.close(fig)

    fig, axes = plt.subplots(1, len(horizons), figsize=(5 * len(horizons), 4), squeeze=False)
    for ax, hz in zip(axes[0], horizons):
        d = dmap[(dmap["horizon"] == hz) & dmap["omega"].notna()]
        pts = ax.scatter(d["omega"], d["delta"], c=d["tau"], s=10, cmap="viridis")
        ax.axhline(0, color="#555", lw=.8)
        ax.set_title(f"horizon {hz}: Delta below 0 means TSFM cheaper", fontsize=10)
        ax.set_xlabel("Omega")
        ax.set_ylabel("Delta (scaled)")
        ax.grid(alpha=.25)
        fig.colorbar(pts, ax=ax, label="tau")
    fig.tight_layout()
    fig.savefig(out / "figures" / "decision_map.png", dpi=150, metadata={"Software": None})
    plt.close(fig)
    return "written: figures/regret_by_tau.png, figures/decision_map.png"


# --------------------------------------------------------------------------- the run

def _source_files(bundle: Path, workload: Path | None, role_table: Path | None):
    workload = workload or bundle / "workload.json"
    if not workload.is_file():
        raise PilotRunError(f"no workload: {workload} does not exist. Pass --workload with the "
                            "workload.json the cache was planned from")
    if role_table is None:
        for cand in (bundle / "role_table", bundle / rt.FILES["table"]):
            if cand.exists():
                role_table = cand
                break
    return workload, role_table


def run_pilot(bundle, out, variant: str = DEFAULT_VARIANT, thresholds=DEFAULT_THRESHOLDS,
              workload=None, role_table=None, fixed_classical=None,
              min_fit_datasets: int = MIN_FIT_DATASETS, n_boot: int = 200, boot_seed: int = 0,
              figures: bool = False, d_omega_builder=None,
              e2_selector_kind: str = e2.DOmegaLinearPlaceholder.kind) -> dict:
    """Run steps 1 to 7 (module docstring) and return the manifest."""
    bundle, out = Path(bundle), Path(out)
    thresholds = [float(t) for t in thresholds]
    if not thresholds or len(set(thresholds)) != len(thresholds):
        raise PilotRunError(f"static thresholds must be distinct and at least one: {thresholds}")
    partial = out.with_name(out.name + ".partial")
    for p in (out, partial):
        if p.exists():
            raise PilotRunError(f"{p} already exists; choose a new --out")
    workload, role_table = _source_files(bundle, None if workload is None else Path(workload),
                                         None if role_table is None else Path(role_table))

    # 1 and 2: every input check happens before anything is written
    forecasts, omega, report, fom = load_bundle(bundle, variant)
    origins = rt.read_workload_origins(workload)
    assert_no_held_out(origins["series_id"].unique(), "pilot run: workload series_id")
    bundle_table = _read_table_input(role_table) if role_table is not None else None
    want, role_report = check_roles(forecasts, origins, bundle_table)

    try:
        partial.mkdir(parents=True)
        manifest = _run_steps(partial, bundle, forecasts, omega, report, fom, want, role_report,
                              workload, role_table, variant, thresholds, fixed_classical,
                              min_fit_datasets, n_boot, boot_seed, figures, d_omega_builder,
                              e2_selector_kind)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    partial.rename(out)
    return manifest


def _run_steps(out, bundle, forecasts, omega, report, fom, want, role_report, workload, role_table,
               variant, thresholds, fixed_classical, min_fit_datasets, n_boot, boot_seed, figures,
               d_omega_builder, e2_selector_kind):
    head = thresholds[0]
    files_note = {}
    workload_sha = file_sha256(workload)
    code_sha, code_files = code_sha256()
    commit = _git_commit()

    # 2: role table from the workload, with the #60 writer
    sidecar = rt.write_role_table(want, out / "role_table", dict(
        input=dict(kind="workload", name=workload.name, sha256=workload_sha),
        fit_only_datasets=[], code_commit=commit))
    role_sha = sidecar["files"][rt.FILES["table"]]

    # 3: representatives, frozen
    source = dict(kind="handoff bundle", fixture=report["fixture"], omega_variant=variant,
                  handoff_sha256=report["handoff_sha256"],
                  partitions_hash_verified=report["partitions_hash_verified"])
    best, fixed = choose_representatives(forecasts, fom, source, fixed_classical)
    freeze(best, out / "representatives" / "representatives.json")
    scores = [score_table(best).assign(rule="best_of_family")]
    if fixed is not None:
        freeze(fixed, out / "representatives" / "representatives_fixed_named.json")
        scores.append(score_table(fixed).assign(rule="fixed_named"))
    reps = dict(best["representatives"])

    # 4: engine, once per threshold on one set of tables
    tables = engine_tables(forecasts, omega, reps, fom, variant)
    engine_roles = check_engine_roles(tables, want)
    col = column_check(forecasts, tables, reps)
    runs = []
    for t in thresholds:
        res, d_omega = run_engine(tables, reps, t, min_fit_datasets, d_omega_builder)
        d = out / "engine" / _label(t)
        d.mkdir(parents=True)
        res.to_parquet(d / "selection_results.parquet", index=False)
        _csv(res, d / "selection_results.csv")
        _json(d_omega.describe() if hasattr(d_omega, "describe") else {"selector": d_omega.name},
              d / "d_omega.json")
        runs.append((t, res, d_omega))
    _csv(tables["excluded_series_horizons"], out / "engine" / "excluded_series_horizons.csv")
    _csv(tables["undefined_omega"], out / "engine" / "undefined_omega.csv")
    selection = runs[0][1]

    # 5: E1 and E2 on the headline run
    r1 = e1.run_e1(tables, selection)
    _csv(r1["map"], out / "e1" / "decision_map.csv")
    _csv(r1["summary"], out / "e1" / "summary_by_horizon_tau.csv")
    _csv(r1["summary_by_dataset"], out / "e1" / "summary_by_dataset_horizon_tau.csv")
    _csv(r1["symmetry"], out / "e1" / "symmetry.csv")
    if n_boot > 0:
        delta_ci, delta_skipped = delta_intervals(tables, n_boot, boot_seed)
        if len(delta_ci):
            _csv(delta_ci, out / "e1" / "mean_delta_intervals.csv")
        _csv(delta_skipped, out / "e1" / "mean_delta_intervals_skipped.csv")
        delta_ci_status = (f"{len(delta_ci)} rows, {len(delta_skipped)} horizon(s) skipped "
                           "(e1/mean_delta_intervals_skipped.csv)")
    else:
        delta_ci_status = "not run (--n-boot 0)"
    fold_check, fold_sel = run_e2_folds(tables, selection, head, min_fit_datasets,
                                        out / "e2" / "frozen", e2_selector_kind)
    _csv(fold_check, out / "e2" / "fold_check.csv")
    if len(fold_sel):
        _csv(fold_sel, out / "e2" / "fold_selection.csv")
    _csv(headline_table(selection, ("selector", "horizon", "tau")), out / "e2" / "regret_summary.csv")
    decided = selection[selection["status"] == "selected"].groupby("selector").size()
    gaps = [e2.gap_to_validation(selection, s) for s in SELECTOR_ORDER
            if s != e2.VALIDATION_COMPARATOR and decided.get(s, 0)]
    _csv(pd.concat(gaps, ignore_index=True), out / "e2" / "gap_to_validation.csv")
    worst = [e2.worst_decile_regret(selection, s) for s in SELECTOR_ORDER if decided.get(s, 0)]
    _csv(pd.concat(worst, ignore_index=True), out / "e2" / "worst_decile.csv")
    if n_boot > 0:
        reg_ci, reg_skipped = regret_intervals(tables, selection, n_boot, boot_seed)
        if len(reg_ci):
            _csv(reg_ci, out / "e2" / "regret_intervals.csv")
        _csv(reg_skipped, out / "e2" / "regret_intervals_skipped.csv")
    else:
        reg_ci, reg_skipped = pd.DataFrame(), pd.DataFrame()

    # 6: paper tables
    tdir = out / "tables"
    for by, name in ((("selector", "horizon", "tau"), "headline_by_horizon_tau.csv"),
                     (("selector", "horizon"), "headline_by_selector_horizon.csv"),
                     (("selector",), "headline_by_selector.csv")):
        parts = [headline_table(res, by).assign(static_threshold=t, headline=(t == head))
                 for t, res, _ in runs]
        tab = pd.concat(parts, ignore_index=True)
        _csv(tab[["static_threshold", "headline"] + [c for c in tab.columns
                                                     if c not in ("static_threshold", "headline")]],
             tdir / name)
    _csv(pd.concat(scores, ignore_index=True), tdir / "representative_scores.csv")
    cache_pairs = rt.distinct_origins(forecasts, "handoff bundle")[rt.SERIES_KEYS].drop_duplicates()
    coverage = role_coverage(want, cache_pairs, tables)
    _csv(coverage, tdir / "role_coverage.csv")
    sens = None
    if fixed is not None:
        f_reps = dict(fixed["representatives"])
        f_tables = engine_tables(forecasts, omega, f_reps, fom, variant)
        check_engine_roles(f_tables, want)
        f_res, _ = run_engine(f_tables, f_reps, head, min_fit_datasets, d_omega_builder)
        d = out / "sensitivity" / f"fixed_classical_{_safe(fixed_classical)}"
        d.mkdir(parents=True)
        f_res.to_parquet(d / "selection_results.parquet", index=False)
        _csv(f_res, d / "selection_results.csv")
        sens = pd.concat([headline_table(res_, ("selector", "horizon", "tau")).assign(
            rule=rule, classical=r_["classical"], tsfm=r_["tsfm"])
            for rule, res_, r_ in (("best_of_family", selection, reps),
                                   ("fixed_named", f_res, f_reps))], ignore_index=True)
        _csv(sens, tdir / "representative_sensitivity.csv")
    figure_status = write_figures(out, head) if figures else FIGURES_OFF

    # 7: summary, environment, manifest
    settings = dict(omega_variant=variant, static_thresholds=thresholds, headline_threshold=head,
                    val_fraction=VAL_FRACTION, fit_scheme=FIT_SCHEME,
                    min_fit_datasets=min_fit_datasets,
                    representatives=dict(rule="best_of_family", loss=REPRESENTATIVE_LOSS,
                                         tau=0.5, chosen=reps,
                                         fixed_classical_sensitivity=fixed_classical),
                    d_omega=("placeholder DOmegaLinear, leave-one-dataset-out (main)"
                             if d_omega_builder is None else getattr(d_omega_builder, "__name__",
                                                                     "custom builder")),
                    e2_selector_kind=e2_selector_kind, deferral_pricing=DEFERRAL_REFERENCE,
                    n_boot=n_boot, boot_seed=boot_seed, boot_cluster=BOOT_CLUSTER,
                    boot_statistics=["e1.mean_delta_intervals",
                                     "bootstrap.stat_mean_scaled_regret (fixed oracle)"],
                    tau_grid=list(TAU_GRID))
    inputs = dict(bundle_name=bundle.name, handoff_sha256=report["handoff_sha256"],
                  omega_sha256=report["omega_sha256"], fixture=report["fixture"],
                  evidence_kind=report["evidence_kind"],
                  forecast_rows_sha256=best["input"]["sha256"],
                  workload=dict(name=workload.name, sha256=workload_sha,
                                matches_29=workload_sha == APPROVED_WORKLOAD_SHA256),
                  role_table_from_workload=dict(sha256=role_sha,
                                                matches_29=role_sha == APPROVED_ROLE_TABLE_SHA256),
                  bundle_role_table=(None if role_table is None else
                                     dict(name=role_table.name, sha256=_tree_sha256(role_table))))
    _json(dict(python=platform.python_version(), implementation=platform.python_implementation(),
               numpy=np.__version__, pandas=pd.__version__, pyarrow=pyarrow.__version__,
               scipy=scipy.__version__),
          out / ENVIRONMENT_FILE)
    facts = dict(report=report, role=role_report, engine_roles=engine_roles, column_check=col,
                 tables=tables, runs=runs, best=best, fixed=fixed, coverage=coverage,
                 fold_check=fold_check, delta_ci_status=delta_ci_status, reg_ci=reg_ci,
                 reg_skipped=reg_skipped, figure_status=figure_status, settings=settings,
                 inputs=inputs, sens=sens, n_boot=n_boot)
    (out / SUMMARY_FILE).write_text(_summary(facts), encoding="utf-8", newline="\n")
    files = {p.relative_to(out).as_posix(): file_sha256(p) for p in sorted(out.rglob("*"))
             if p.is_file() and p.name not in (MANIFEST_FILE, ENVIRONMENT_FILE)}
    manifest = dict(command="python -m decision_aware_tsfm.analysis.decision.run_pilot",
                    decisions_reference=DECISIONS_REFERENCE, inputs=inputs, settings=settings,
                    code=dict(git_commit=commit, code_sha256=code_sha, code_files=code_files),
                    checks=dict(roles=role_report, engine_roles=engine_roles, column_check=col),
                    environment_file=ENVIRONMENT_FILE, files=files)
    _json(manifest, out / MANIFEST_FILE)
    return manifest


# --------------------------------------------------------------------------- summary.md

def _fmt(x) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{x:.6f}" if isinstance(x, float) else str(x)


def _summary(f: dict) -> str:
    rep, s, i, t = f["report"], f["settings"], f["inputs"], f["tables"]
    runs, best = f["runs"], f["best"]
    head_res = runs[0][1]
    units = t["units"]
    ex = t["excluded_series_horizons"]
    L = ["# Pilot analysis run", ""]
    if i["fixture"]:
        L += ["**SYNTHETIC FIXTURE: this run checks the interface only. It is not a research result.**", ""]
    L += [f"Decisions applied: {DECISIONS_REFERENCE}", "",
          "## Inputs", "",
          f"- Bundle `{i['bundle_name']}`: handoff.json sha256 `{i['handoff_sha256']}`, "
          f"{rep['partitions_hash_verified']} of {rep['partitions_listed']} partitions hash-checked, "
          f"{rep['rows']} forecast rows, {rep['jobs']} jobs, {rep['series']} series, "
          f"models {', '.join(rep['models'])}, seeds {rep['seeds']}.",
          f"- Omega variant `{s['omega_variant']}`: {rep['omega_rows_for_variant']} rows, "
          f"{rep['omega_undefined_rows_for_variant']} undefined.",
          f"- Workload `{i['workload']['name']}` sha256 `{i['workload']['sha256']}`; the one named "
          f"on #29: {'yes' if i['workload']['matches_29'] else 'NO'}.",
          f"- Role table built from the workload: role_table.csv sha256 "
          f"`{i['role_table_from_workload']['sha256']}`; the one named on #29: "
          f"{'yes' if i['role_table_from_workload']['matches_29'] else 'NO'}.",
          "- Role table carried by the bundle: " + (
              "none" if i["bundle_role_table"] is None else
              f"`{i['bundle_role_table']['name']}`, agrees with the workload on every row."), "",
          "## What ran", ""]
    r = f["role"]
    L += ["1. Load and validate: passed. Held-out names: none.",
          f"2. Roles: the workload has {r['workload_series_horizons']} (series, horizon) pairs, the "
          f"cache holds {r['cache_series_horizons']} ({r['workload_series_horizons_absent_from_cache']}"
          f" not in the cache yet). The cache's origins get the same roles as the role table on all "
          f"{r['cache_vs_workload']['rows']} rows; the engine used those roles on "
          f"{f['engine_roles']['origin_rows']} origin rows.",
          f"3. Representatives (best of family, scaled pinball at tau 0.50, validation origins only): "
          + "; ".join(f"{fam} `{m}` ({best['reasons'][fam]})" for fam, m in best["representatives"].items())
          + "." + ("" if f["fixed"] is None else
                   f" Sensitivity: fixed classical `{f['fixed']['representatives']['classical']}`."),
          f"4. Engine: {len(units)} evaluation units ({units[rt.SERIES_KEYS].drop_duplicates().shape[0]}"
          f" series-horizons, {units['dataset'].nunique()} datasets) per selector, static thresholds "
          f"{', '.join(f'{x:g}' for x in s['static_thresholds'])} (headline {s['headline_threshold']:g}). "
          f"Column check: {f['column_check']['units_compared']} units recomputed from the raw quantile "
          f"columns, largest relative difference {f['column_check']['max_relative_difference']:.3g}.",
          f"5. E1: decision map and summaries; mean Delta intervals {f['delta_ci_status']}. E2: "
          f"{int(f['fold_check']['fitted'].sum())} of {len(f['fold_check'])} folds fitted, frozen and "
          f"evaluated; their d_omega choices equal the engine's on "
          f"{int(f['fold_check']['n_same_as_engine'].sum())} units. Regret intervals: "
          + (f"{len(f['reg_ci'])} rows, {len(f['reg_skipped'])} (selector, horizon) cells skipped "
             "(e2/regret_intervals_skipped.csv)."
             if f["n_boot"] > 0 else "not run (--n-boot 0)."),
          f"6. Paper tables written. Figures: {f['figure_status']}.", "",
          "## Counts the paper reports (record 0007)", "",
          "- Validation fraction one third, gap rule as in record 0007; role counts per dataset and "
          "horizon in `tables/role_coverage.csv`.",
          f"- Excluded (series, horizon) pairs: {len(ex)}" + (
              "" if not len(ex) else " (" + "; ".join(
                  f"{k}: {n}" for k, n in ex.groupby("reason").size().items()) + ")") + ".",
          f"- Units with undefined Omega: {t['n_units_undefined_omega']}.",
          "- Folds: " + "; ".join(
              f"{row.fold} " + ("fitted" if row.fitted else f"not fitted ({row.reason})")
              for row in f["fold_check"].itertuples()) + ".",
          "- Deferred units at the headline threshold (charged the pinball-validation cost in the "
          "deferral column): " + (", ".join(
              f"{k} {int(v)}" for k, v in (head_res[head_res.status != "selected"]
                                           .groupby("selector").size().items())) or "none") + ".", ""]
    pooled = headline_table(head_res, ("selector",))
    L += [f"## Headline, all horizons and taus pooled, static threshold {s['headline_threshold']:g}", "",
          "| selector | units | decided | mean scaled regret | fraction of oracle value recovered |",
          "|---|---|---|---|---|"]
    for row in pooled.itertuples():
        L.append(f"| {row.selector} | {row.n_units} | {row.n_selected} | {_fmt(row.mean_scaled_regret)} | "
                 f"{_fmt(row.frac_oracle_value_recovered)} |")
    L += ["", "Mean scaled regret and the fraction are over decided units; the per-(horizon, tau) "
          "table also gives regret over oracle cost and the deferral-priced mean.", "",
          "## Where each result is", "",
          "| file | what it holds |", "|---|---|"]
    where = [
        ("tables/headline_by_horizon_tau.csv", "the seven rules per horizon and tau, every threshold "
         "(headline first): regret, regret over oracle cost, fraction of oracle value, deferrals. "
         "Source of the regret-by-tau figure"),
        ("tables/headline_by_selector_horizon.csv", "the same pooled over taus"),
        ("tables/headline_by_selector.csv", "the same pooled over horizons and taus"),
        ("tables/representative_scores.csv", "validation score of every candidate model"),
        ("tables/role_coverage.csv", "roles per dataset and horizon, series in cache, scored, excluded"),
        ("role_table/", "the role table built from the workload (#60 writer)"),
        ("representatives/", "the frozen representative choice"),
        ("engine/threshold_*/", "selection results of the seven rules and the D_Omega folds"),
        ("engine/excluded_series_horizons.csv", "excluded pairs with the reason"),
        ("engine/undefined_omega.csv", "decisions with undefined Omega and the reason"),
        ("e1/decision_map.csv", "per unit Omega, tau and Delta; source of the decision-map figure"),
        ("e1/", "E1 summaries, symmetry table, mean Delta intervals"),
        ("e2/", "frozen fold specifications, fold check, regret summary, gap to validation, worst "
                "decile, regret intervals"),
        ("manifest.json", "input and code SHA-256 and the SHA-256 of every file"),
        ("environment.json", "interpreter and package versions")]
    if f["sens"] is not None:
        where.insert(4, ("tables/representative_sensitivity.csv",
                         "best of family against the fixed classical model, per horizon and tau"))
    L += [f"| `{a}` | {b} |" for a, b in where]
    L += ["", "Intervals are nominal 95% percentile block-bootstrap intervals (approximate, pointwise; "
          "see analysis/experiments/bootstrap.py before quoting one).", ""]
    return "\n".join(L)


# --------------------------------------------------------------------------- command line

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundle", type=Path, help="producer handoff bundle directory")
    ap.add_argument("--out", type=Path, required=True, help="new output directory")
    ap.add_argument("--variant", default=DEFAULT_VARIANT, help="Omega variant: current or last_4096")
    ap.add_argument("--static-thresholds", type=float, nargs="+", default=list(DEFAULT_THRESHOLDS),
                    help="static Omega thresholds; the first is the headline")
    ap.add_argument("--workload", type=Path, help="workload.json (default BUNDLE/workload.json)")
    ap.add_argument("--role-table", type=Path,
                    help="role table CSV or export directory to check (default: one in the bundle)")
    ap.add_argument("--fixed-classical", help="also run the fixed-named sensitivity with this model")
    ap.add_argument("--min-fit-datasets", type=int, default=MIN_FIT_DATASETS)
    ap.add_argument("--n-boot", type=int, default=200, help="bootstrap resamples; 0 skips intervals")
    ap.add_argument("--boot-seed", type=int, default=0)
    ap.add_argument("--figures", action="store_true", help="draw PNG figures (needs matplotlib)")
    a = ap.parse_args(argv)
    try:
        m = run_pilot(a.bundle, a.out, variant=a.variant, thresholds=a.static_thresholds,
                      workload=a.workload, role_table=a.role_table, fixed_classical=a.fixed_classical,
                      min_fit_datasets=a.min_fit_datasets, n_boot=a.n_boot, boot_seed=a.boot_seed,
                      figures=a.figures)
    except (PilotRunError, HandoffError, CacheError, rt.RoleTableError, HeldOutLeakError,
            EvaluationRowsError) as exc:
        raise SystemExit(f"pilot run stopped, nothing written: {type(exc).__name__}: {exc}")
    if m["inputs"]["fixture"]:
        print("SYNTHETIC FIXTURE: checks the interface only, not research results.")
    print((a.out / SUMMARY_FILE).read_text(encoding="utf-8"))
    print(f"wrote {len(m['files']) + 2} files to {a.out}")


if __name__ == "__main__":
    main()
