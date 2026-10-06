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
    """Lower-cost family per unit. Exact ties (within 1e-12 relative) go to
    classical, the cheaper family to run. NaN in either cost gives None."""
    a = np.asarray(cost_classical, dtype=float)
    b = np.asarray(cost_tsfm, dtype=float)
    tol = TIE_RTOL * np.maximum(1.0, np.maximum(np.abs(a), np.abs(b)))
    out = np.where(b < a - tol, FAMILIES[1], FAMILIES[0]).astype(object)
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
        return _decisions(inp.units, argmin_family(t["J_classical"], t["J_tsfm"]))


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


DEFAULT_BETA_GRID = (-0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5)


@dataclass(frozen=True)
class DOmegaLinear:
    """PLACEHOLDER for D_Omega: the proposal's minimal boundary
        TSFM iff omega > omega0 + beta * |tau - 0.5|
    fitted by minimising total scaled regret on fit-only datasets.

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
