"""The K3 Omega-plus-tau selector (#26, analysis/k3/selector.py) in the D_Omega slot.

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
  cost     "raw" hands him J_classical and J_tsfm as they are; "scaled" hands
           him J / scale, the objective the placeholder minimises.
  Omega    undefined is NaN in the engine and None in his interface.
  defer    his "defer" becomes status "defer" with his reason in the note and
           no family. A deferred unit is NOT priced: how to price a deferral is
           an open team question. His evaluate_selector prices it as the
           oracle; that function is not used here.

His module needs scikit-learn (through analysis/k3/boundary.py), which the
pinned decision environment does not have. It is therefore imported on first
use, never when this module is imported.
"""
from __future__ import annotations

import copy
import importlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .guards import FitEvalOverlapError, assert_no_held_out
from .schema import FAMILIES, UNIT_KEYS
from .selectors import _decisions

K3_SELECTOR_MODULE = "decision_aware_tsfm.analysis.k3.selector"
DEFAULT_MARGIN = 0.0            # his apply_selector default: no weak-evidence band
DEFAULT_MIN_DISTINCT = 3        # his MIN_DISTINCT_OMEGA_THRESHOLD, counted in distinct datasets
COSTS = ("raw", "scaled")
DEFAULT_COST = "raw"
UNFITTED_HORIZON = "unfitted_horizon"


class K3SelectorUnavailable(ImportError):
    """analysis/k3/selector.py, or scikit-learn which it needs, cannot be imported."""


def k3_module():
    try:
        return importlib.import_module(K3_SELECTOR_MODULE)
    except ImportError as e:
        raise K3SelectorUnavailable(
            f"cannot import {K3_SELECTOR_MODULE} ({e}). The K3 selector needs "
            "analysis/k3/selector.py (#26) and scikit-learn (requirements/analysis.in, #22)") from e


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
    min_distinct: int
    cost: str
    fitted_series: frozenset = field(repr=False)
    fitted_datasets: tuple = ()
    name: str = "d_omega"
    needs_hindsight: bool = False

    @classmethod
    def fit(cls, fit_units: pd.DataFrame, margin: float = DEFAULT_MARGIN,
            min_distinct: int = DEFAULT_MIN_DISTINCT, cost: str = DEFAULT_COST) -> "K3SelectorAdapter":
        """fit_units: UNIT_KEYS + omega + J_classical + J_tsfm + scale.

        margin and min_distinct are his parameters, passed through unchanged.
        Units with an undefined Omega are handed over as None (he counts them
        and fits on the rest); units with a missing cost are left out."""
        assert_no_held_out(fit_units["dataset"], "D_Omega fit")
        assert_no_held_out(fit_units["config"], "D_Omega fit")
        if cost not in COSTS:
            raise ValueError(f"cost must be one of {COSTS}, got {cost!r}")
        if not margin >= 0:
            raise ValueError("margin must be non-negative")
        if min_distinct != int(min_distinct) or min_distinct < 1:
            raise ValueError("min_distinct must be a whole number of datasets, at least 1")
        k3 = k3_module()
        u = fit_units.dropna(subset=["J_classical", "J_tsfm"] + (["scale"] if cost == "scaled" else []))
        if u.empty:
            raise ValueError("D_Omega fit received no usable units")
        if u.duplicated(UNIT_KEYS).any():
            raise ValueError("D_Omega fit pool has duplicate units; his unit is one series per tau")
        _check_tau_keys(u["tau"])
        specs = {}
        for h, g in u.groupby("horizon", sort=True):
            jc, jt = g["J_classical"].to_numpy(dtype=float), g["J_tsfm"].to_numpy(dtype=float)
            if cost == "scaled":
                jc, jt = jc / g["scale"].to_numpy(dtype=float), jt / g["scale"].to_numpy(dtype=float)
            specs[int(h)] = k3.fit_selector(_omega_or_none(g["omega"]), g["tau"].to_numpy(dtype=float), jc, jt,
                                            g["dataset"].to_numpy(dtype=object), min_distinct=int(min_distinct))
        return cls(specs=specs, margin=float(margin), min_distinct=int(min_distinct), cost=cost,
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
        return dict(selector="k3_omega_tau", margin=self.margin, min_distinct=self.min_distinct,
                    cost=self.cost, fitted_datasets=list(self.fitted_datasets),
                    specs={str(h): copy.deepcopy(s) for h, s in sorted(self.specs.items())})
