"""E1: the (Omega, tau) decision map over the full 11-value tau grid.

    python -m decision_aware_tsfm.analysis.experiments.e1 --out artifacts/runs/e1_fixture

The command above runs on the SYNTHETIC FIXTURE. Its output is an interface
check, not a result about real model families.

Orientation of Delta, used everywhere in this module and in
bootstrap.stat_mean_delta. docs/project-overview.md defines

    Delta(x, tau) = J_TSFM(x, tau) - J_classical(x, tau)
    "Delta < 0 -> the TSFM family is decision-optimal. Otherwise the classical family is."

This module reports that quantity divided by the unit's scale:

    delta = J_tsfm / scale  -  J_classical / scale

where J_f is the unit's mean pinball loss at tau for family f's representative
over the unit's test origins and scale is the engine's scale (the series'
seasonal-naive MAE, season 24, on its history before the first origin, #29
item 7a; always positive, so the sign is the overview's). delta < 0 means
TSFM is the lower-cost family; delta > 0 means classical is. The E3 code uses
the same orientation. A unit whose two SCALED costs are equal within 1e-12
(relative, with an absolute floor of 1e-12) is a tie: it counts for neither
family in the shares and is skipped by the sign-reversal count.

What this module does not do: it does not fit the boundary Omega*(tau). The
proposal writes the boundary in |tau - 0.5|, which assumes the map is
symmetric about 0.5. Whether to fit it at all is gated on K3 (owner:
Abdullah). This module delivers the map and the symmetry table that the fit
and the K3 decision consume. See BOUNDARY_INPUT_COLUMNS.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..decision.metrics import TAU_GRID
from ..decision.schema import UNIT_KEYS
from ..decision.selectors import TIE_RTOL
from . import bootstrap

TAU_ATOL = 1e-9
# The columns of decision_map.parquet that a boundary fit needs. Stable contract.
BOUNDARY_INPUT_COLUMNS = UNIT_KEYS + ["omega", "delta", "lower_cost_family", "is_tie",
                                      "scaled_cost_classical", "scaled_cost_tsfm", "scale",
                                      "n_test_origins"]


SERIES_HORIZON = ("dataset", "config", "series_id", "horizon")


class MissingTauError(ValueError):
    """The results do not cover the pre-registered tau grid."""


class UnequalPopulationError(ValueError):
    """Two things that are compared were computed on different sets of units."""


def _match(values, grid) -> tuple[list[float], list[float]]:
    """(grid values with no match in values, values with no match in grid)."""
    v = np.unique(np.asarray(values, dtype=float))
    g = np.asarray(grid, dtype=float)
    missing = [float(x) for x in g if not np.isclose(v, x, rtol=0, atol=TAU_ATOL).any()]
    extra = [float(x) for x in v if not np.isclose(g, x, rtol=0, atol=TAU_ATOL).any()]
    return missing, extra


def assert_full_grid(frame: pd.DataFrame, what: str, grid=TAU_GRID, by=("horizon",)) -> None:
    """Raise MissingTauError unless every group of `frame` has exactly the grid's taus.

    A tau outside the grid also raises: the grid is pre-registered, so an extra
    value means the wrong table was passed. `by` sets how fine the check is.
    by=("horizon",) only checks each horizon as a whole, which lets a tau be
    absent for one dataset or one series; run_e1 checks every series at every
    horizon (SERIES_HORIZON).
    """
    if frame.empty:
        raise MissingTauError(f"{what}: no rows, so every tau in the grid is missing")
    g = np.sort(np.asarray(grid, dtype=float))
    tau = frame["tau"].to_numpy(dtype=float)
    k = np.abs(tau[:, None] - g[None, :]).argmin(axis=1)
    on_grid = np.abs(tau - g[k]) <= TAU_ATOL
    if not on_grid.all():
        extra = sorted({float(x) for x in tau[~on_grid]})
        raise MissingTauError(f"{what}: unexpected tau {extra[:5]}; expected {list(grid)}")
    by = list(by)
    if not by:
        present = np.unique(k)
        if present.size < g.size:
            missing = [float(x) for i, x in enumerate(g) if i not in set(present)]
            raise MissingTauError(f"{what}: missing tau {missing}; expected {list(grid)}")
        return
    seen = frame[by].assign(_k=k).drop_duplicates()
    n = seen.groupby(by, sort=True)["_k"].nunique()
    short = n[n < g.size]
    if len(short):
        key = short.index[0]
        key = key if isinstance(key, tuple) else (key,)
        have = set(seen.loc[(seen[by] == pd.Series(key, index=by)).all(axis=1), "_k"])
        missing = [float(x) for i, x in enumerate(g) if i not in have]
        raise MissingTauError(f"{what} {dict(zip(by, key))}: missing tau {missing} "
                              f"({len(short)} group(s) incomplete); expected {list(grid)}")


def decision_map(tables: dict) -> pd.DataFrame:
    """One row per unit: omega, tau, both scaled costs, Delta and the lower-cost family.

    Costs are the unit's TEST-origin costs (tables['test']), the same costs the
    selectors are scored on. Omega is the value the engine attached to the unit
    (tables['units']). delta = scaled_cost_tsfm - scaled_cost_classical, so
    delta < 0 means TSFM has the lower cost (module docstring).

    Ties are judged on SCALED cost. Judging them on raw cost with an absolute
    floor, as the engine's argmin_family does, would call two costs of 1e-13
    and 5e-13 a tie on a series whose scale is 1e-12, although one family is
    five times cheaper there.

    Undefined Omega. A unit whose Omega is undefined (NaN: flat or empty
    history, see decision.pipeline.attach_omega) stays in the map with omega
    NaN. Its costs, Delta and lower-cost family are still reported; it is only
    left out of anything ordered by Omega (the sign-reversal count), and
    summarize_map reports how many such units each group has.
    """
    t = tables["units"][UNIT_KEYS + ["omega"]].merge(
        tables["test"][UNIT_KEYS + ["J_classical", "J_tsfm", "scale", "n_origins"]],
        on=UNIT_KEYS, how="left", validate="1:1")
    if t[["J_classical", "J_tsfm", "scale"]].isna().any().any():
        raise ValueError("some units have no test cost or scale")
    if (t["scale"] <= 0).any():
        raise ValueError("scale must be positive")
    if not np.isfinite(t[["J_classical", "J_tsfm", "scale"]].to_numpy(dtype=float)).all():
        raise ValueError("costs and scale must be finite")
    om = t["omega"].to_numpy(dtype=float)
    if np.isinf(om).any():
        raise ValueError("omega must be finite, or NaN when undefined")
    scale = t["scale"].to_numpy(float)
    a, b = t["J_classical"].to_numpy(float) / scale, t["J_tsfm"].to_numpy(float) / scale
    tie = np.abs(a - b) <= TIE_RTOL * np.maximum(1.0, np.maximum(np.abs(a), np.abs(b)))
    out = t[UNIT_KEYS + ["omega"]].copy()
    out["scaled_cost_classical"] = a
    out["scaled_cost_tsfm"] = b
    out["delta"] = out["scaled_cost_tsfm"] - out["scaled_cost_classical"]
    out["is_tie"] = tie
    out["lower_cost_family"] = np.where(tie, "tie", np.where(out["delta"] < 0, "tsfm", "classical"))
    out["scale"] = t["scale"].to_numpy(float)
    out["n_test_origins"] = t["n_origins"].astype("int64").to_numpy()
    return out[BOUNDARY_INPUT_COLUMNS]


def count_sign_reversals(omega, delta, is_tie=None) -> int:
    """Number of times the sign of Delta changes as Omega increases.

    Steps, in order:
      1. Units flagged in is_tie are dropped. If is_tie is not given, a unit
         with delta exactly 0 is a tie.
      2. Units that share exactly the same Omega are replaced by one point with
         their mean Delta. Their order along Omega is not defined, so counting
         them one by one would make the result depend on row order.
      3. A point whose mean Delta is exactly 0 is dropped.
      4. The remaining points are sorted by Omega and the number of adjacent
         pairs with opposite signs is returned.

    A single clean boundary gives 1. No boundary (one family lower everywhere)
    gives 0. Because ties are dropped rather than treated as a sign, the
    sequence + tie - counts one reversal and + tie + counts none.
    """
    om = np.asarray(omega, dtype=float)
    de = np.asarray(delta, dtype=float)
    if om.shape != de.shape:
        raise ValueError("omega and delta must have the same length")
    if not (np.isfinite(om).all() and np.isfinite(de).all()):
        raise ValueError("omega and delta must be finite")
    tie = (de == 0.0) if is_tie is None else np.asarray(is_tie, dtype=bool)
    om, de = om[~tie], de[~tie]
    if om.size == 0:
        return 0
    pts = pd.Series(de).groupby(om).mean()          # index sorted by omega
    s = np.sign(pts.to_numpy())
    s = s[s != 0]
    return int((s[1:] != s[:-1]).sum())


def selector_regret(selection: pd.DataFrame, by=("horizon", "tau")) -> pd.DataFrame:
    """Mean scaled regret of each selector per group, one column per selector
    (regret_<selector>), plus n_selected_<selector>. Undecided units are not
    in the mean; n_selected shows how many units the mean is over."""
    by = list(by)
    s = selection[selection["status"] == "selected"]
    m = s.groupby(by + ["selector"])["scaled_regret"].mean().unstack("selector")
    n = s.groupby(by + ["selector"]).size().unstack("selector")
    names = list(dict.fromkeys(selection["selector"]))     # order of appearance
    m = m.reindex(columns=names)
    n = n.reindex(columns=names).fillna(0).astype("int64")
    m.columns = [f"regret_{c}" for c in names]
    n.columns = [f"n_selected_{c}" for c in names]
    return m.join(n).reset_index()


def summarize_map(dmap: pd.DataFrame, selection: pd.DataFrame | None = None,
                  by=("horizon", "tau")) -> pd.DataFrame:
    """Per group (default horizon and tau):

      n_units              units in the group
      n_tsfm_lower, n_classical_lower, n_ties
      share_tsfm_lower     n_tsfm_lower / n_units (ties are in the denominator
                           and count for neither family)
      mean_delta           mean Delta over all units, ties included (their Delta is zero within tolerance)
      n_sign_reversals     count_sign_reversals along Omega within the group,
                           over the units whose Omega is defined
      n_undefined_omega    units whose Omega is undefined (NaN); they are in every
                           count above except n_sign_reversals
      regret_<selector>    mean scaled regret of each selector (if selection given)
    """
    by = list(by)
    rows = []
    for key, g in dmap.groupby(by, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        n_t = int((g["lower_cost_family"] == "tsfm").sum())
        n_c = int((g["lower_cost_family"] == "classical").sum())
        has_omega = g["omega"].notna().to_numpy()
        d = g[has_omega]
        rows.append(dict(zip(by, key), n_units=len(g), n_tsfm_lower=n_t, n_classical_lower=n_c,
                         n_ties=int(g["is_tie"].sum()), share_tsfm_lower=n_t / len(g),
                         mean_delta=float(g["delta"].mean()),
                         n_sign_reversals=count_sign_reversals(d["omega"], d["delta"], d["is_tie"]),
                         n_undefined_omega=int((~has_omega).sum())))
    out = pd.DataFrame(rows)
    if selection is not None:
        out = out.merge(selector_regret(selection, by), on=by, how="left", validate="1:1")
    return out


def tau_pairs(grid=TAU_GRID) -> list[tuple[float, float]]:
    """(lower tau, upper tau) pairs with lower + upper = 1. tau = 0.5 pairs with
    itself and is left out. Raises MissingTauError if a tau has no mirror."""
    g = sorted(float(x) for x in grid)
    pairs = []
    for lo in g:
        if lo >= 0.5 - TAU_ATOL:
            continue
        hi = [x for x in g if abs(x - (1.0 - lo)) <= TAU_ATOL]
        if not hi:
            raise MissingTauError(f"tau {lo} has no mirror value {1.0 - lo:g} in {g}")
        pairs.append((lo, hi[0]))
    unpaired = [x for x in g if x > 0.5 + TAU_ATOL and x not in {p[1] for p in pairs}]
    if unpaired:
        raise MissingTauError(f"tau {unpaired} have no mirror value below 0.5 in {g}")
    return pairs


def symmetry_table(summary: pd.DataFrame, grid=TAU_GRID, by=("horizon",)) -> pd.DataFrame:
    """Compare each tau below 0.5 with its mirror 1 - tau, within each group.

    Differences are UPPER MINUS LOWER (the tau above 0.5 minus the tau below):
      diff_share_tsfm_lower = share(1 - tau) - share(tau)
      diff_mean_delta       = mean_delta(1 - tau) - mean_delta(tau)
    A map that is symmetric about 0.5 gives zero for both. This table only
    shows the differences; it does not test them.

    n_units_lower and n_units_upper are the numbers of units behind each side.
    If they differ the two sides were computed on different populations and
    the difference would mix a tau effect with a population effect, so the
    function raises UnequalPopulationError instead of returning the row.
    """
    by = list(by)
    rows = []
    groups = [((), summary)] if not by else summary.groupby(by, sort=True)
    for key, g in groups:
        key = key if isinstance(key, tuple) else (key,)
        for lo, hi in tau_pairs(grid):
            a = g[np.isclose(g["tau"], lo, rtol=0, atol=TAU_ATOL)]
            b = g[np.isclose(g["tau"], hi, rtol=0, atol=TAU_ATOL)]
            if len(a) != 1 or len(b) != 1:
                raise MissingTauError(f"symmetry pair ({lo}, {hi}) needs exactly one summary row "
                                      f"per side in group {dict(zip(by, key))}; "
                                      f"found {len(a)} and {len(b)}")
            a, b = a.iloc[0], b.iloc[0]
            rows.append(dict(zip(by, key), tau_lower=lo, tau_upper=hi,
                             distance_from_half=round(0.5 - lo, 12),
                             n_units_lower=int(a["n_units"]), n_units_upper=int(b["n_units"]),
                             share_tsfm_lower_at_tau_lower=float(a["share_tsfm_lower"]),
                             share_tsfm_lower_at_tau_upper=float(b["share_tsfm_lower"]),
                             diff_share_tsfm_lower=float(b["share_tsfm_lower"] - a["share_tsfm_lower"]),
                             mean_delta_at_tau_lower=float(a["mean_delta"]),
                             mean_delta_at_tau_upper=float(b["mean_delta"]),
                             diff_mean_delta=float(b["mean_delta"] - a["mean_delta"]),
                             n_sign_reversals_lower=int(a["n_sign_reversals"]),
                             n_sign_reversals_upper=int(b["n_sign_reversals"])))
            row = rows[-1]              # the check reads the reported counts, so they cannot drift apart
            if row["n_units_lower"] != row["n_units_upper"]:
                raise UnequalPopulationError(
                    f"symmetry pair ({lo}, {hi}) in group {dict(zip(by, key))}: "
                    f"{row['n_units_lower']} units at tau {lo} but {row['n_units_upper']} at tau {hi}")
    return pd.DataFrame(rows)


def mean_delta_intervals(tables: dict, n_boot: int = 200, seed: int = 0, level: float = 0.95,
                         cluster: str = "dataset+series", block_length="auto",
                         time_blocks: str = "per_series") -> pd.DataFrame:
    """Nominal-level percentile bootstrap interval for mean Delta at every
    (horizon, tau), TEST origins only (the origins the map itself uses).

    One set of resamples per horizon is shared by all taus. Datasets and series
    are resampled as clusters and origins in moving blocks (bootstrap.py). The
    intervals are approximate and pointwise: read bootstrap.py's module
    docstring before quoting one. Every row carries the number of resamples,
    the seed, the cluster level, the block lengths used and the time-block
    mode, so an interval cannot be separated from how it was made.
    """
    cost = bootstrap.family_cost_table(tables["per_origin"], role="test")
    out = []
    for h, g in cost.groupby("horizon", sort=True):
        ci = bootstrap.bootstrap_ci(g, lambda d: bootstrap.stat_mean_delta(d, by="tau"),
                                    n_boot=n_boot, level=level, seed=seed,
                                    block_length=block_length, cluster=cluster,
                                    time_blocks=time_blocks)
        f = ci.to_frame().rename(columns={"label": "tau", "estimate": "mean_delta",
                                          "lower": "mean_delta_lower", "upper": "mean_delta_upper"})
        f.insert(0, "horizon", int(h))
        out.append(f)
    return pd.concat(out, ignore_index=True)


def _assert_same_units(dmap: pd.DataFrame, selection: pd.DataFrame) -> None:
    """Every selector must have exactly the units of the map."""
    want = dmap[UNIT_KEYS].drop_duplicates()
    for name, g in selection.groupby("selector", sort=False):
        got = g[UNIT_KEYS]
        both = got.merge(want, on=UNIT_KEYS, how="inner")
        if len(got) != len(want) or len(both) != len(want):
            raise UnequalPopulationError(
                f"selector {name!r} has {len(got)} units, the decision map has {len(want)}, "
                f"{len(both)} in common")


def run_e1(tables: dict, selection: pd.DataFrame, grid=TAU_GRID) -> dict:
    """Decision map, per-(horizon, tau) summary and symmetry table.

    Fails with MissingTauError if any tau of the grid is absent, or any tau
    outside the grid is present, for any series at any horizon in the units,
    in the unit test costs, or in any selector's results. Fails with
    UnequalPopulationError if a selector's units are not the map's units. So
    every (horizon, tau) cell, and both sides of every symmetry pair, are
    computed on the same series.
    """
    assert_full_grid(tables["units"], "units", grid, by=SERIES_HORIZON)
    assert_full_grid(tables["test"], "unit test costs", grid, by=SERIES_HORIZON)
    assert_full_grid(selection, "selection results", grid, by=("selector",) + SERIES_HORIZON)
    dmap = decision_map(tables)
    assert_full_grid(dmap, "decision map", grid, by=SERIES_HORIZON)
    _assert_same_units(dmap, selection)
    summary = summarize_map(dmap, selection)
    return dict(map=dmap, summary=summary, symmetry=symmetry_table(summary, grid),
                summary_by_dataset=summarize_map(dmap, selection, by=("dataset", "horizon", "tau")))


def main(argv=None):
    from ..decision.fixture import (FAMILY_OF_MODEL, FIT_ONLY_DATASETS, FIXTURE_SCHEMA,
                                    FIXTURE_STATIC_THRESHOLD, REPRESENTATIVES, make_fixture,
                                    make_fixture_scales)
    from ..decision.pipeline import build_tables, run

    ap = argparse.ArgumentParser(description="E1 on the synthetic fixture (interface check only)")
    ap.add_argument("--out", default="artifacts/runs/e1_fixture")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-boot", type=int, default=200, help="0 skips the bootstrap intervals")
    ap.add_argument("--boot-seed", type=int, default=0)
    ap.add_argument("--time-blocks", default="per_series", choices=bootstrap.TIME_BLOCK_MODES)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)

    fc, omega, _ = make_fixture(a.seed)
    tables = build_tables(fc, FIXTURE_SCHEMA, REPRESENTATIVES, FAMILY_OF_MODEL, omega,
                          fit_only_datasets=FIT_ONLY_DATASETS, scales=make_fixture_scales(a.seed))
    selection, _ = run(tables, REPRESENTATIVES, FIXTURE_STATIC_THRESHOLD)
    r = run_e1(tables, selection)

    print("SYNTHETIC FIXTURE. Interface check, not a research result.")
    print(f"units: {len(r['map'])}; horizons {sorted(int(h) for h in r['map'].horizon.unique())}; "
          f"taus {sorted(r['map'].tau.unique())}")
    r["map"].to_parquet(out / "decision_map.parquet", index=False)
    r["map"].to_csv(out / "decision_map.csv", index=False, lineterminator="\n")
    r["summary"].to_csv(out / "summary_by_horizon_tau.csv", index=False, lineterminator="\n")
    r["summary_by_dataset"].to_csv(out / "summary_by_dataset_horizon_tau.csv", index=False, lineterminator="\n")
    r["symmetry"].to_csv(out / "symmetry.csv", index=False, lineterminator="\n")
    head = ["horizon", "tau", "n_units", "share_tsfm_lower", "mean_delta", "n_ties", "n_sign_reversals"]
    print("\nper (horizon, tau):")
    print(r["summary"][head].round(5).to_string(index=False))
    print("\nDelta = J_tsfm/scale - J_classical/scale; Delta < 0 means TSFM has the lower cost.")
    print("\nsymmetry, upper minus lower:")
    print(r["symmetry"][["horizon", "tau_lower", "tau_upper", "n_units_lower", "n_units_upper",
                         "diff_share_tsfm_lower", "diff_mean_delta"]].round(5).to_string(index=False))
    m = r["map"]
    print("\nchecks used in the result note:")
    print(f"  ties: {int(m.is_tie.sum())} units, by dataset {m[m.is_tie].groupby('dataset').size().to_dict()}")
    print(f"  units by lower-cost family and dataset: "
          f"{m.groupby(['dataset', 'lower_cost_family']).size().to_dict()}")
    print(f"  sign reversals per (horizon, tau): min {int(r['summary'].n_sign_reversals.min())}, "
          f"max {int(r['summary'].n_sign_reversals.max())}")
    print(f"  mean Delta range over the {len(r['summary'])} cells: "
          f"{r['summary'].mean_delta.min():.5f} to {r['summary'].mean_delta.max():.5f}")
    files = ["decision_map.parquet", "decision_map.csv", "summary_by_horizon_tau.csv",
             "summary_by_dataset_horizon_tau.csv", "symmetry.csv"]
    if a.n_boot > 0:
        ci = mean_delta_intervals(tables, n_boot=a.n_boot, seed=a.boot_seed,
                                  time_blocks=a.time_blocks)
        ci.to_csv(out / "mean_delta_intervals.csv", index=False, lineterminator="\n")
        files.append("mean_delta_intervals.csv")
        print(f"\nmean Delta with a NOMINAL 95% percentile block-bootstrap interval (approximate, "
              f"pointwise; not verified coverage). {a.n_boot} resamples, seed {a.boot_seed}, "
              f"datasets and series as clusters, time blocks {a.time_blocks}:")
        print(ci[["horizon", "tau", "mean_delta", "mean_delta_lower", "mean_delta_upper",
                  "n_boot", "block_lengths_used"]].round(5).to_string(index=False))
    manifest = dict(experiment="E1", data="synthetic fixture (decision.fixture.make_fixture)",
                    evidence="interface check, not a research result", fixture_seed=a.seed,
                    tau_grid=list(TAU_GRID), n_boot=a.n_boot, boot_seed=a.boot_seed,
                    time_blocks=a.time_blocks,
                    delta_definition="delta = J_tsfm/scale - J_classical/scale (docs/project-overview.md: "
                                     "Delta = J_TSFM - J_classical); delta < 0 means TSFM has the lower cost",
                    interval_kind=bootstrap.INTERVAL_KIND,
                    boundary_fit="not done here; gated on K3", files=files,
                    boundary_input_columns=BOUNDARY_INPUT_COLUMNS)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {len(files) + 1} files to {out}")


if __name__ == "__main__":
    main()
