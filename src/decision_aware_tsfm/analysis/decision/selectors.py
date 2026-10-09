"""The seven comparators. Each one only CHOOSES a family per unit.

Costs, regret and the output format are attached afterwards by
evaluate.evaluate(), identically for every selector, so no selector can
compute its own score differently from the others.

What each selector may see:
  units : UNIT_KEYS + omega                         (cheap, pre-forecast)
  val   : UNIT_KEYS + J_*/crps_* on validation origins
  test  : UNIT_KEYS + J_* on test origins            (hindsight oracle ONLY)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .guards import FitEvalOverlapError, assert_no_held_out
from .schema import FAMILIES, UNIT_KEYS

TIE_RTOL = 1e-12


def argmin_family(cost_classical, cost_tsfm) -> np.ndarray:
    """Lower-cost family per unit, for choices made from validation costs. Ties
    (within 1e-12 relative) go to classical, the cheaper family to run. NaN in
    either cost gives None. Hindsight columns use exact_argmin_family instead."""
    a = np.asarray(cost_classical, dtype=float)
    b = np.asarray(cost_tsfm, dtype=float)
    tol = TIE_RTOL * np.maximum(1.0, np.maximum(np.abs(a), np.abs(b)))
    out = np.where(b < a - tol, FAMILIES[1], FAMILIES[0]).astype(object)
    out[~(np.isfinite(a) & np.isfinite(b))] = None
    return out


def exact_argmin_family(cost_classical, cost_tsfm) -> np.ndarray:
    """The family whose cost is the exact minimum, with no tolerance. Only an
    exact tie goes to classical. NaN in either cost gives None.

    Used wherever a family is named in hindsight (the oracle, oracle_family,
    best_fixed_family), so the family named always carries the minimum cost that
    regret is measured against, and the oracle's regret is exactly zero."""
    a = np.asarray(cost_classical, dtype=float)
    b = np.asarray(cost_tsfm, dtype=float)
    out = np.where(b < a, FAMILIES[1], FAMILIES[0]).astype(object)
    out[~(np.isfinite(a) & np.isfinite(b))] = None
    return out


def _decisions(units: pd.DataFrame, family, note: str = "") -> pd.DataFrame:
    d = units[UNIT_KEYS].copy()
    fam = np.asarray(family, dtype=object)
    d["chosen_family"] = fam
    d["status"] = np.where(pd.isna(fam), "insufficient_data", "selected")
    d["note"] = note
    return d


@dataclass(frozen=True)
class SelectorInput:
    units: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame | None = None


class AlwaysClassical:
    name, needs_hindsight = "always_classical", False

    def select(self, inp):
        return _decisions(inp.units, np.full(len(inp.units), "classical", dtype=object))


class AlwaysTSFM:
    name, needs_hindsight = "always_tsfm", False

    def select(self, inp):
        return _decisions(inp.units, np.full(len(inp.units), "tsfm", dtype=object))


@dataclass
class StaticOmegaRule:
    """TSFM when omega > threshold, else classical. tau is ignored.

    threshold has no default on purpose: it must be the value read from the
    source paper and recorded, not a number typed from memory.
    """
    threshold: float
    name: str = "static_omega"
    needs_hindsight: bool = False

    def select(self, inp):
        om = inp.units["omega"].to_numpy(dtype=float)
        fam = np.where(om > self.threshold, "tsfm", "classical").astype(object)
        fam[~np.isfinite(om)] = None
        return _decisions(inp.units, fam, f"threshold={self.threshold}")


def _val_choice(inp, prefix):
    v = inp.units[UNIT_KEYS].merge(inp.val, on=UNIT_KEYS, how="left")
    return argmin_family(v[f"{prefix}_classical"], v[f"{prefix}_tsfm"])


class PinballValidation:
    """Lower mean pinball at the unit's own tau on validation origins."""
    name, needs_hindsight = "pinball_validation", False

    def select(self, inp):
        return _decisions(inp.units, _val_choice(inp, "J"))


class CRPSValidation:
    """Lower mean CRPS (quantile-grid estimate) on validation origins; same choice at every tau."""
    name, needs_hindsight = "crps_validation", False

    def select(self, inp):
        return _decisions(inp.units, _val_choice(inp, "crps"))


class HindsightOracle:
    name, needs_hindsight = "hindsight_oracle", True

    def select(self, inp):
        if inp.test is None:
            raise ValueError("oracle needs test costs")
        t = inp.units[UNIT_KEYS].merge(inp.test, on=UNIT_KEYS, how="left")
        return _decisions(inp.units, exact_argmin_family(t["J_classical"], t["J_tsfm"]))


@dataclass(frozen=True)
class NotFitted:
    """Stands in for a selector that could not be fitted (e.g. no fit-only data).
    Every unit is reported as insufficient_data, with the reason in the note."""
    name: str
    reason: str
    needs_hindsight: bool = False

    def select(self, inp):
        fam = np.full(len(inp.units), None, dtype=object)
        return _decisions(inp.units, fam, f"not fitted: {self.reason}")

    def describe(self) -> dict:
        return dict(not_fitted=self.reason)


DEFAULT_BETA_GRID = (-0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5)


@dataclass(frozen=True)
class DOmegaLinear:
    """PLACEHOLDER for D_Omega: the proposal's minimal boundary
        TSFM iff omega > omega0 + beta * |tau - 0.5|
    fitted by minimising total scaled regret on the units it is given (one
    leave-one-dataset-out fold, or the fit-only datasets under the older scheme).

    Abdullah owns the frozen Omega-plus-tau selector (K3/E4). This class exists
    so the comparator slot runs end to end; his selector should replace it by
    exposing the same select(inp) method.
    """
    omega0: float
    beta: float
    fitted_series: frozenset = field(repr=False)
    fitted_datasets: tuple = ()
    name: str = "d_omega"
    needs_hindsight: bool = False

    @classmethod
    def fit(cls, fit_units: pd.DataFrame, beta_grid=DEFAULT_BETA_GRID) -> "DOmegaLinear":
        """fit_units: UNIT_KEYS + omega + J_classical + J_tsfm + scale."""
        assert_no_held_out(fit_units["dataset"], "D_Omega fit")
        assert_no_held_out(fit_units["config"], "D_Omega fit")
        u = fit_units.dropna(subset=["omega", "J_classical", "J_tsfm"])
        if u.empty:
            raise ValueError("D_Omega fit received no usable units")
        om = np.sort(u["omega"].unique())
        cand0 = np.concatenate([[om[0] - 0.01], (om[1:] + om[:-1]) / 2, [om[-1] + 0.01]])
        best_j = np.minimum(u.J_classical, u.J_tsfm).to_numpy()
        dist = np.abs(u["tau"].to_numpy() - 0.5)
        rows = []
        for b in beta_grid:
            for w0 in cand0:
                tsfm = u["omega"].to_numpy() > w0 + b * dist
                j = np.where(tsfm, u.J_tsfm, u.J_classical)
                rows.append((float(((j - best_j) / u.scale.to_numpy()).sum()), abs(b), w0, b))
        # lowest objective; ties -> smallest |beta|, then smallest omega0
        obj = np.array([r[0] for r in rows])
        tol = 1e-12 * max(1.0, float(np.abs(obj).max()))
        near = [r for r in rows if r[0] <= obj.min() + tol]
        _, _, w0, b = sorted(near, key=lambda r: (r[1], r[2]))[0]
        return cls(omega0=float(w0), beta=float(b),
                   fitted_series=frozenset(u["series_id"]),
                   fitted_datasets=tuple(sorted(u["dataset"].unique())))

    def select(self, inp):
        overlap = self.fitted_series & set(inp.units["series_id"])
        if overlap:
            raise FitEvalOverlapError(f"{len(overlap)} series used both to fit and to evaluate D_Omega")
        om = inp.units["omega"].to_numpy(dtype=float)
        bound = self.omega0 + self.beta * np.abs(inp.units["tau"].to_numpy() - 0.5)
        fam = np.where(om > bound, "tsfm", "classical").astype(object)
        fam[~np.isfinite(om)] = None
        return _decisions(inp.units, fam, f"omega0={self.omega0:.4f} beta={self.beta:+.3f}")

    def describe(self) -> dict:
        return dict(omega0=self.omega0, beta=self.beta, fitted_datasets=list(self.fitted_datasets))


FIT_POOL_COLUMNS = UNIT_KEYS + ["omega", "J_classical", "J_tsfm", "scale"]


@dataclass(frozen=True)
class LeaveOneDatasetOut:
    """Fit a selector once per evaluation dataset, never on that dataset.

    For evaluation dataset d the selector is fitted on every unit in the pool
    whose dataset is not d, and applied only to d's units. The pool holds the
    fit-only datasets (all origins) and the other evaluation datasets' test
    units, so d itself is never seen while its rule is fitted. A fold whose
    pool covers fewer than min_fit_datasets distinct datasets is not fitted,
    and its units are reported as insufficient_data with the reason.

    fit: callable(fit_units) -> selector with select(inp), e.g. DOmegaLinear.fit.
    """
    folds: dict
    min_fit_datasets: int
    name: str = "d_omega"
    needs_hindsight: bool = False

    @classmethod
    def build(cls, pool: pd.DataFrame, eval_datasets, fit=None, min_fit_datasets: int = 2,
              name: str = "d_omega") -> "LeaveOneDatasetOut":
        fit = DOmegaLinear.fit if fit is None else fit
        if min_fit_datasets < 1:
            raise ValueError("min_fit_datasets must be at least 1")
        missing = [c for c in FIT_POOL_COLUMNS if c not in pool.columns]
        if missing:
            raise ValueError(f"fit pool lacks columns {missing}")
        usable = pool.dropna(subset=["omega", "J_classical", "J_tsfm", "scale"])
        folds = {}
        for d in sorted(set(eval_datasets)):
            fold = usable[usable["dataset"] != d]
            n = int(fold["dataset"].nunique())
            if n < min_fit_datasets:
                folds[d] = NotFitted(name, f"leave-one-dataset-out: {n} other dataset(s) with usable "
                                           f"units, need {min_fit_datasets}")
            else:
                folds[d] = fit(fold)
        return cls(folds=folds, min_fit_datasets=int(min_fit_datasets), name=name)

    @property
    def fitted_series(self) -> frozenset:
        return frozenset().union(*[getattr(f, "fitted_series", frozenset()) for f in self.folds.values()])

    def select(self, inp):
        parts = []
        for d, units in inp.units.groupby("dataset", sort=True):
            fold = self.folds.get(d, NotFitted(self.name, f"leave-one-dataset-out: no fold for {d!r}"))
            if d in getattr(fold, "fitted_datasets", ()):
                raise FitEvalOverlapError(f"dataset {d!r} was used to fit its own rule")
            out = fold.select(SelectorInput(units=units, val=inp.val, test=None))
            out["note"] = f"fold={d} " + out["note"].astype(str)
            parts.append(out)
        if not parts:
            return _decisions(inp.units, np.full(0, None, dtype=object))
        res = pd.concat(parts, ignore_index=True)
        # same row order as the units handed in
        order = inp.units[UNIT_KEYS].reset_index(drop=True).reset_index().rename(columns={"index": "_row"})
        return (res.merge(order, on=UNIT_KEYS, how="left", validate="1:1")
                .sort_values("_row").drop(columns="_row").reset_index(drop=True))

    def describe(self) -> dict:
        return dict(scheme="leave_one_dataset_out", min_fit_datasets=self.min_fit_datasets,
                    folds={d: f.describe() for d, f in sorted(self.folds.items())})
