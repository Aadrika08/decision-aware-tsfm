"""Wire the pieces together: cache -> costs -> seven selectors -> common schema."""
from __future__ import annotations

import pandas as pd

from .cache_io import load_cache, restrict_to_representatives
from .costs import assign_roles, n_origins_by_role, per_origin_costs, unit_costs
from .evaluate import run_selectors
from .guards import assert_no_held_out
from .metrics import TAU_GRID
from .schema import UNIT_KEYS
from .selectors import (AlwaysClassical, AlwaysTSFM, CRPSValidation, DOmegaLinear,
                        HindsightOracle, PinballValidation, StaticOmegaRule)


def build_tables(forecasts, schema_version, representatives, family_of_model, omega,
                 fit_only_datasets=(), val_fraction=1 / 3, taus=TAU_GRID, crps_method="trapezoid",
                 omega_col="omega_levels"):
    fc = load_cache(forecasts, schema_version, taus)
    assert_no_held_out(fc["dataset"], "development pipeline")
    assert_no_held_out(fc["config"], "development pipeline")
    fc = restrict_to_representatives(fc, representatives, family_of_model)
    po = per_origin_costs(fc, taus, crps_method)

    om = omega[["series_id", omega_col]].rename(columns={omega_col: "omega"})
    if om.series_id.duplicated().any():
        raise ValueError("omega table has duplicate series_id")
    missing = set(po.series_id) - set(om.series_id)
    if missing:
        raise ValueError(f"{len(missing)} cached series have no omega, e.g. {sorted(missing)[:3]}")

    fit_mask = po.dataset.isin(fit_only_datasets)
    fit_units = unit_costs(po[fit_mask], role=None).merge(om, on="series_id")
    ev = assign_roles(po[~fit_mask], val_fraction)
    val = unit_costs(ev, "validation")
    test = unit_costs(ev, "test")
    units = test[UNIT_KEYS].merge(om, on="series_id", how="left")
    return dict(per_origin=ev, fit_units=fit_units, units=units, val=val, test=test,
                n_val=n_origins_by_role(ev, "validation"),
                n_targets_dropped_missing_y=fc.attrs.get("n_targets_dropped_missing_y", 0))


def seven_selectors(static_threshold, d_omega):
    return [AlwaysClassical(), AlwaysTSFM(), StaticOmegaRule(static_threshold),
            PinballValidation(), CRPSValidation(), d_omega, HindsightOracle()]


def run(tables, representatives, static_threshold):
    d_omega = DOmegaLinear.fit(tables["fit_units"])
    res = run_selectors(seven_selectors(static_threshold, d_omega), tables["units"],
                        tables["val"], tables["test"], representatives, tables["n_val"])
    return res, d_omega
