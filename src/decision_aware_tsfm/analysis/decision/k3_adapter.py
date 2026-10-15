"""The K3 Omega-plus-tau selector (#26, analysis/k3/selector.py) in the D_Omega slot.

Entry point: build_k3_d_omega(tables, ...) returns the D_Omega selector for
the engine's tables, ready for pipeline.seven_selectors. Score its results with
evaluate.summarize_charged(), which prices deferrals; evaluate.summarize()
refuses a frame with deferred rows.

A thin adapter. It turns the engine's fit pool into the arrays fit_selector
takes, keeps the frozen spec it returns, and asks apply_selector for every
unit. No rule is fitted or applied in this file and nothing in analysis/k3 is
changed.

What the adapter adds, all of it recorded in describe():
  horizon  His rule is per tau and his unit is the series; the engine's unit
           also has a horizon. One spec is fitted per horizon, so every spec
           sees each series once per tau. A unit whose horizon has no spec is
           deferred with reason unfitted_horizon (the one reason that is the
           adapter's and not his).
  cost     "scaled" (default, pilot decision 9) hands him J / scale, the
           engine's scaled cost from costs.py, the objective the placeholder
           also minimises, so one dataset with large values cannot dominate
           the fit. "raw" hands him J_classical and J_tsfm as they are
           (sensitivity only).
  Omega    undefined is NaN in the engine and None in his interface.
  defer    his "defer" becomes status "defer" with his reason in the note and
           no family. The engine charges each deferral the cost of its
           pinball_validation comparator on the same unit
           (evaluate.charge_deferrals, pilot decision 8). His evaluate_selector
           is not used here; its default prices a deferral at the oracle.

Two floors, both kept (pilot decision 6):
  min_fit_datasets  the ENGINE's floor (LeaveOneDatasetOut): a fold whose pool
                    has fewer distinct datasets with usable units is not fitted
                    and its units are insufficient_data. Default 3 (#61).
  k3_min_distinct   HIS floor, passed to fit_selector as min_distinct. The
                    adapter always passes dataset labels, so #54 counts the
                    distinct datasets with a defined Omega at each (horizon,
                    tau); it also needs at least 2 distinct Omega values to
                    place a threshold. Below either, that rule is
                    insufficient_data and its units are deferred. Default 3.

His module is imported on first use, never when this module is imported, so
the decision package loads without analysis/k3.
"""
from __future__ import annotations

import copy
import functools
import importlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .guards import FitEvalOverlapError, assert_no_held_out
from .pipeline import FIT_SCHEMES, fit_d_omega
from .schema import FAMILIES, UNIT_KEYS
from .selectors import _decisions

K3_SELECTOR_MODULE = "decision_aware_tsfm.analysis.k3.selector"
DEFAULT_MARGIN = 0.0              # his apply_selector default: no weak-evidence band
DEFAULT_K3_MIN_DISTINCT = 3       # his MIN_DISTINCT_OMEGA_THRESHOLD (record 0008 section 3)
DEFAULT_MIN_FIT_DATASETS = 3      # the engine's leave-one-dataset-out floor (#61, record 0007 section 2)
K3_MIN_DISTINCT_COUNTS = ("distinct datasets with a defined Omega at the (horizon, tau); "
                          "a threshold also needs at least 2 distinct Omega values")
COSTS = ("raw", "scaled")
DEFAULT_COST = "scaled"           # pilot decision 9: the engine's J / scale
UNFITTED_HORIZON = "unfitted_horizon"


class K3SelectorUnavailable(ImportError):
    """analysis/k3/selector.py, or a module it imports, cannot be imported."""


def k3_module():
    try:
        return importlib.import_module(K3_SELECTOR_MODULE)
    except ImportError as e:
        raise K3SelectorUnavailable(
            f"cannot import {K3_SELECTOR_MODULE} ({e}). The K3 selector needs "
            "analysis/k3/selector.py (PR #54) and analysis/k3/boundary.py (PR #32)") from e


def _omega_or_none(omega) -> list:
    return [None if np.isnan(v) else float(v) for v in np.asarray(omega, dtype=float)]


def _check_tau_keys(tau) -> None:
    """His rules are keyed by tau printed to two decimals. A tau the key does not
    reproduce would silently share another tau's rule, so it is refused."""
    bad = [float(t) for t in np.unique(np.asarray(tau, dtype=float)) if float(f"{t:.2f}") != t]
    if bad:
        raise ValueError(f"tau must be exact at two decimals for the K3 selector's rule keys, got {bad[:5]}")


def reason_of(note) -> pd.Series:
    """The K3 reason in each note (rule, omega_out_of_range, ...); missing where there is none."""
    return pd.Series(note, dtype=object).str.extract(r"\breason=(\w+)", expand=False)


@dataclass(frozen=True)
class K3SelectorAdapter:
    """specs: horizon -> the frozen spec fit_selector returned for that horizon."""
    specs: dict = field(repr=False)
    margin: float
    k3_min_distinct: int
    cost: str
    fitted_series: frozenset = field(repr=False)
    fitted_datasets: tuple = ()
    name: str = "d_omega"
    needs_hindsight: bool = False

    @classmethod
    def fit(cls, fit_units: pd.DataFrame, margin: float = DEFAULT_MARGIN,
            k3_min_distinct: int = DEFAULT_K3_MIN_DISTINCT, cost: str = DEFAULT_COST) -> "K3SelectorAdapter":
        """fit_units: UNIT_KEYS + omega + J_classical + J_tsfm + scale.

        margin and k3_min_distinct are his parameters (k3_min_distinct is his
        min_distinct), passed through unchanged. Units with an undefined Omega
        are handed over as None (he counts them and fits on the rest); units
        with a missing cost are left out."""
        assert_no_held_out(fit_units["dataset"], "D_Omega fit")
        assert_no_held_out(fit_units["config"], "D_Omega fit")
        if cost not in COSTS:
            raise ValueError(f"cost must be one of {COSTS}, got {cost!r}")
        if not margin >= 0:
            raise ValueError("margin must be non-negative")
        if k3_min_distinct != int(k3_min_distinct) or k3_min_distinct < 1:
            raise ValueError("k3_min_distinct must be a whole number of datasets, at least 1")
        k3 = k3_module()
        u = fit_units.dropna(subset=["J_classical", "J_tsfm"] + (["scale"] if cost == "scaled" else []))
        if u.empty:
            raise ValueError("D_Omega fit received no usable units")
        if cost == "scaled" and not (u["scale"].to_numpy(dtype=float) > 0).all():
            raise ValueError("scaled cost needs a positive scale on every unit")
        if u.duplicated(UNIT_KEYS).any():
            raise ValueError("D_Omega fit pool has duplicate units; his unit is one series per tau")
        _check_tau_keys(u["tau"])
        specs = {}
        for h, g in u.groupby("horizon", sort=True):
            jc, jt = g["J_classical"].to_numpy(dtype=float), g["J_tsfm"].to_numpy(dtype=float)
            if cost == "scaled":
                jc, jt = jc / g["scale"].to_numpy(dtype=float), jt / g["scale"].to_numpy(dtype=float)
            specs[int(h)] = k3.fit_selector(_omega_or_none(g["omega"]), g["tau"].to_numpy(dtype=float), jc, jt,
                                            g["dataset"].to_numpy(dtype=object), min_distinct=int(k3_min_distinct))
        return cls(specs=specs, margin=float(margin), k3_min_distinct=int(k3_min_distinct), cost=cost,
                   fitted_series=frozenset(u["series_id"]),
                   fitted_datasets=tuple(sorted(u["dataset"].unique())))

    def select(self, inp):
        overlap = self.fitted_series & set(inp.units["series_id"])
        if overlap:
            raise FitEvalOverlapError(f"{len(overlap)} series used both to fit and to evaluate D_Omega")
        k3 = k3_module()
        _check_tau_keys(inp.units["tau"])
        fam, reasons = [], []
        for h, tau, om in zip(inp.units["horizon"], inp.units["tau"], _omega_or_none(inp.units["omega"])):
            spec = self.specs.get(int(h))
            choice, reason = ((k3.DEFER, UNFITTED_HORIZON) if spec is None
                              else k3.apply_selector(spec, om, float(tau), margin=self.margin))
            if choice != k3.DEFER and choice not in FAMILIES:
                raise ValueError(f"K3 selector returned an unknown choice {choice!r}")
            fam.append(None if choice == k3.DEFER else choice)
            reasons.append(reason)
        d = _decisions(inp.units, fam)
        d["status"] = np.where(pd.isna(d["chosen_family"]), "defer", "selected")
        d["note"] = [f"reason={r}" for r in reasons]
        return d

    def describe(self) -> dict:
        return dict(selector="k3_omega_tau", margin=self.margin, k3_min_distinct=self.k3_min_distinct,
                    k3_min_distinct_counts=K3_MIN_DISTINCT_COUNTS, cost=self.cost,
                    fitted_datasets=list(self.fitted_datasets),
                    specs={str(h): copy.deepcopy(s) for h, s in sorted(self.specs.items())})


def build_k3_d_omega(tables, *, fit_scheme: str = "leave_one_dataset_out",
                     min_fit_datasets: int = DEFAULT_MIN_FIT_DATASETS,
                     k3_min_distinct: int = DEFAULT_K3_MIN_DISTINCT,
                     margin: float = DEFAULT_MARGIN, cost: str = DEFAULT_COST):
    """The K3 selector for the D_Omega slot, fitted on the engine's tables.

    tables: pipeline.build_tables(...). Returns what pipeline.run() would put in
    the slot with d_omega_fit=K3SelectorAdapter.fit and these settings: under
    leave-one-dataset-out (default) a LeaveOneDatasetOut with one
    K3SelectorAdapter per evaluation dataset (or NotFitted below
    min_fit_datasets); under fit_only_datasets one K3SelectorAdapter.

    Use it as
        d = build_k3_d_omega(tables, min_fit_datasets=3)
        res = run_selectors(seven_selectors(threshold, d), tables["units"], tables["val"],
                            tables["test"], representatives, tables["n_val"])
        summary = evaluate.summarize_charged(res)          # per selector, horizon and tau
        charged = evaluate.charge_deferrals(res)           # per unit, deferrals priced

    Fails before fitting anything if a held-out name is among the evaluation
    units, or if the K3 module cannot be imported (K3SelectorUnavailable)."""
    assert_no_held_out(tables["units"]["dataset"], "D_Omega evaluation units")
    assert_no_held_out(tables["units"]["config"], "D_Omega evaluation units")
    if fit_scheme not in FIT_SCHEMES:
        raise ValueError(f"fit_scheme must be one of {FIT_SCHEMES}, got {fit_scheme!r}")
    if min_fit_datasets != int(min_fit_datasets) or min_fit_datasets < 1:
        raise ValueError("min_fit_datasets must be a whole number of datasets, at least 1")
    k3_module()
    fit = functools.partial(K3SelectorAdapter.fit, margin=margin, k3_min_distinct=k3_min_distinct, cost=cost)
    return fit_d_omega(tables, fit_scheme, int(min_fit_datasets), fit)
