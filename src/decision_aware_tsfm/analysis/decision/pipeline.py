"""Wire the pieces together: cache -> costs -> seven selectors -> common schema."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .cache_io import SERIES_ID_KEYS, load_cache, restrict_to_representatives
from .costs import SERIES_KEYS, assign_roles, n_origins_by_role, per_origin_costs, unit_costs
from .evaluate import run_selectors
from .guards import assert_no_held_out
from .metrics import TAU_GRID
from .schema import UNIT_KEYS
from .selectors import (AlwaysClassical, AlwaysTSFM, CRPSValidation, DOmegaLinear,
                        HindsightOracle, NotFitted, PinballValidation, StaticOmegaRule)


class OmegaJoinError(ValueError):
    pass


def attach_omega(decisions: pd.DataFrame, omega: pd.DataFrame, omega_col: str,
                 omega_variant: str | None = None, allow_zero_omega: bool = False) -> pd.DataFrame:
    """Add one Omega value per (series, horizon) decision.

    decisions: SERIES_KEYS + decision_origin, where decision_origin is the
    origin at which the family choice is made.

    omega may be
      per series : dataset, config, series_id, <omega_col>
      per origin : dataset, config, series_id, origin, <omega_col>
    optionally with a 'variant' column (e.g. full history vs last 4,096 steps),
    in which case omega_variant must be named explicitly.

    With per-origin Omega the value used is the one at decision_origin, i.e.
    computed on the history available when the choice is made. Averaging over
    later origins would let history after the decision into the decision.

    Undefined Omega. Omega cannot be computed for a flat or empty history. Such
    a value must arrive as null (NaN) in a row that EXISTS for the decision; it
    is kept as NaN, and Omega-based selectors then report insufficient_data for
    that unit while the other comparators still run. A decision with NO row in
    the omega table is still an error: a missing join is a pipeline fault, not
    an undefined statistic.

    Omega exactly 0.0 is refused unless allow_zero_omega is set: the team's
    compute_omega returns 0.0 for degenerate input, which cannot be told apart
    from a real, very low Omega and would make a threshold rule choose
    confidently. Finite values outside [0, 1] and infinite values are refused.
    """
    om = omega.copy()
    if "variant" in om.columns:
        if omega_variant is None:
            raise OmegaJoinError(f"omega table has variants {sorted(om.variant.unique())}; "
                                 "pass omega_variant explicitly")
        om = om[om["variant"] == omega_variant]
        if om.empty:
            raise OmegaJoinError(f"omega variant {omega_variant!r} not present")
    elif omega_variant is not None:
        raise OmegaJoinError("omega_variant given but the omega table has no 'variant' column")
    if omega_col not in om.columns:
        raise OmegaJoinError(f"omega table has no column {omega_col!r}")

    key = list(SERIES_ID_KEYS)
    if "origin" in om.columns:
        om = om.rename(columns={"origin": "decision_origin"})
        key.append("decision_origin")
    om = om[key + [omega_col]].rename(columns={omega_col: "omega"})
    if om.duplicated(key).any():
        raise OmegaJoinError(f"omega table has duplicate keys on {key}")

    out = decisions.merge(om, on=key, how="left", validate="m:1", indicator="_omega_row")
    miss = out["_omega_row"] != "both"
    if miss.any():
        ex = out.loc[miss, key].drop_duplicates().head(3).to_dict("records")
        raise OmegaJoinError(f"{int(miss.sum())} decisions have no omega row, e.g. {ex}")
    out = out.drop(columns="_omega_row")
    v = pd.to_numeric(out["omega"], errors="coerce").astype(float)
    if (out["omega"].notna() & v.isna()).any():
        raise OmegaJoinError("omega values must be numeric or null")
    out["omega"] = v
    defined = v.notna()
    if np.isinf(v[defined]).any() or ((v[defined] < 0) | (v[defined] > 1)).any():
        raise OmegaJoinError("omega values must lie in [0, 1]; use null for an undefined value")
    zero = defined & (v == 0.0)
    if zero.any() and not allow_zero_omega:
        ex = out.loc[zero, key].drop_duplicates().head(3).to_dict("records")
        raise OmegaJoinError(
            f"{int(zero.sum())} decisions have omega exactly 0.0, e.g. {ex}. compute_omega returns 0.0 "
            "for a flat or empty history; store an undefined Omega as null, or pass "
            "allow_zero_omega=True if 0.0 is a real value")
    return out


def _decision_origins(po: pd.DataFrame, role: str | None) -> pd.DataFrame:
    d = po if role is None else po[po["role"] == role]
    return (d.groupby(SERIES_KEYS)["origin"].min().rename("decision_origin").reset_index())


ZERO_SCALE_REASON = "zero scale: every realised target in the scored period is 0, so scaled cost is undefined"


def _drop_zero_scale(po: pd.DataFrame, role: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove (series, horizon) groups whose mean |y| over the scored origins is 0.

    Scaled cost divides by that mean, so it is undefined there. The groups are
    returned separately with a reason; they are never dropped silently."""
    d = po if role is None else po[po["role"] == role]
    tot = d.groupby(SERIES_KEYS)["abs_y_sum"].sum()
    bad = tot[tot <= 0].index
    if len(bad) == 0:
        return po, pd.DataFrame(columns=SERIES_KEYS + ["reason"])
    idx = pd.MultiIndex.from_frame(po[SERIES_KEYS])
    excluded = bad.to_frame(index=False).assign(reason=ZERO_SCALE_REASON)
    return po[~idx.isin(bad)].copy(), excluded


def build_tables(forecasts, schema_version, representatives, family_of_model, omega,
                 fit_only_datasets=(), val_fraction=1 / 3, taus=TAU_GRID, crps_method="trapezoid",
                 omega_col="omega_levels", omega_variant=None, allow_zero_omega=False):
    fc = load_cache(forecasts, schema_version, taus)
    assert_no_held_out(fc["dataset"], "development pipeline")
    assert_no_held_out(fc["config"], "development pipeline")
    fc = restrict_to_representatives(fc, representatives, family_of_model)
    po = per_origin_costs(fc, taus, crps_method)

    fit_mask = po.dataset.isin(fit_only_datasets)
    fit_po, fit_excluded = _drop_zero_scale(po[fit_mask], None)
    # fit-only datasets: every origin is used for fitting, so the choice is made before the first one
    fit_om = attach_omega(_decision_origins(fit_po, None), omega, omega_col, omega_variant, allow_zero_omega)
    fit_units = unit_costs(fit_po, role=None).merge(fit_om.drop(columns="decision_origin"),
                                                    on=SERIES_KEYS, how="left")

    ev, ev_excluded = _drop_zero_scale(assign_roles(po[~fit_mask], val_fraction), "test")
    val = unit_costs(ev, "validation")
    test = unit_costs(ev, "test")
    # evaluation datasets: the choice is made at the first test origin
    ev_om = attach_omega(_decision_origins(ev, "test"), omega, omega_col, omega_variant, allow_zero_omega)
    units = test[UNIT_KEYS].merge(ev_om, on=SERIES_KEYS, how="left", validate="m:1")
    # settings recorded so a later step (e.g. a frozen E2 evaluation) can check that
    # these tables were built the way a frozen specification says they must be
    settings = dict(schema_version=schema_version, representatives=dict(representatives),
                    val_fraction=float(val_fraction), omega_col=omega_col, omega_variant=omega_variant,
                    crps_method=crps_method, taus=tuple(float(t) for t in taus),
                    fit_only_datasets=tuple(sorted(fit_only_datasets)),
                    models_scored=tuple(sorted(po["model"].unique().tolist())),
                    allow_zero_omega=bool(allow_zero_omega))
    excluded = pd.concat([ev_excluded.assign(used_for="evaluation"),
                          fit_excluded.assign(used_for="fit")], ignore_index=True)
    return dict(per_origin=ev, fit_units=fit_units,
                units=units.drop(columns="decision_origin"), val=val, test=test,
                decision_origins=ev_om[SERIES_KEYS + ["decision_origin"]],
                n_val=n_origins_by_role(ev, "validation"),
                n_targets_dropped_missing_y=fc.attrs.get("n_targets_dropped_missing_y", 0),
                omega_variant=omega_variant, settings=settings,
                excluded_series_horizons=excluded,
                n_units_undefined_omega=int(units["omega"].isna().sum()))


def seven_selectors(static_threshold, d_omega):
    return [AlwaysClassical(), AlwaysTSFM(), StaticOmegaRule(static_threshold),
            PinballValidation(), CRPSValidation(), d_omega, HindsightOracle()]


def run(tables, representatives, static_threshold):
    usable = tables["fit_units"].dropna(subset=["omega", "J_classical", "J_tsfm"])
    if usable.empty:
        d_omega = NotFitted("d_omega", "no fit-only datasets supplied")
    else:
        d_omega = DOmegaLinear.fit(tables["fit_units"])
    res = run_selectors(seven_selectors(static_threshold, d_omega), tables["units"],
                        tables["val"], tables["test"], representatives, tables["n_val"])
    return res, d_omega
