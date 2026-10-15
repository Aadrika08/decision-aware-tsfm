"""The one output format shared by all seven comparators.

A row is one decision unit: (selector, dataset, config, series_id, horizon, tau).
The selector chooses a family for that unit using only what it is allowed to
see; realised cost is the mean pinball loss of that family's representative
over the unit's test origins and forecast steps.

Why per unit and not per origin: regret is only guaranteed non-negative at
the level where the choice is made. The oracle picks the family with the
lower MEAN test cost for the unit, so at a single origin the selector's
family can beat the oracle's family. Per-origin costs are kept separately
(costs.per_origin_costs) for the block bootstrap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import check_taus

SCHEMA_VERSION = "selection-v1"
FAMILIES = ("classical", "tsfm")          # order = tie-break order (classical wins exact ties)
STATUSES = ("selected", "defer", "insufficient_data")
UNIT_KEYS = ["dataset", "config", "series_id", "horizon", "tau"]

# column -> kind: s=string, i=integer, f=float
COLUMNS: dict[str, str] = {
    "selector": "s",
    "dataset": "s", "config": "s", "series_id": "s", "horizon": "i", "tau": "f",
    "status": "s",
    "chosen_family": "s",        # missing unless status == selected
    "chosen_model": "s",
    "realized_cost": "f",        # mean pinball_tau of chosen family on test origins
    "oracle_family": "s",
    "oracle_cost": "f",
    "regret": "f",               # realized_cost - oracle_cost, >= 0
    "scale": "f",                # series scale: seasonal-naive MAE (season 24) of its history before its first origin (#29 item 7a)
    "scaled_regret": "f",        # regret / scale
    "best_fixed_family": "s",    # one family for the whole (horizon, tau) group, chosen in hindsight
    "best_fixed_cost": "f",
    "omega": "f",
    "n_val_origins": "i",
    "n_test_origins": "i",
    "note": "s",
    "schema_version": "s",
}


class SchemaError(ValueError):
    pass


def validate_selection_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Raise SchemaError on any violation; return df unchanged if valid."""
    if list(df.columns) != list(COLUMNS):
        missing = [c for c in COLUMNS if c not in df.columns]
        extra = [c for c in df.columns if c not in COLUMNS]
        raise SchemaError(f"columns differ from schema. missing={missing} extra={extra}")
    for col, kind in COLUMNS.items():
        s = df[col]
        if kind == "i" and not pd.api.types.is_integer_dtype(s):
            raise SchemaError(f"{col} must be integer, got {s.dtype}")
        if kind == "f" and not pd.api.types.is_float_dtype(s):
            raise SchemaError(f"{col} must be float, got {s.dtype}")
    if (df["schema_version"] != SCHEMA_VERSION).any():
        raise SchemaError("schema_version mismatch")
    check_taus(df["tau"].to_numpy())
    if not df["status"].isin(STATUSES).all():
        raise SchemaError(f"unknown status {sorted(set(df.status) - set(STATUSES))}")
    key = ["selector"] + UNIT_KEYS
    if df.duplicated(key).any():
        raise SchemaError(f"duplicate decision units: {df[df.duplicated(key)][key].head(3).to_dict('records')}")

    sel = df["status"] == "selected"
    if not df.loc[sel, "chosen_family"].isin(FAMILIES).all():
        raise SchemaError("selected rows must name a valid family")
    if df.loc[~sel, "chosen_family"].notna().any() or df.loc[~sel, "realized_cost"].notna().any():
        raise SchemaError("non-selected rows must not carry a family or a cost")
    for c in ("realized_cost", "oracle_cost", "regret", "scale"):
        if not np.isfinite(df.loc[sel, c].to_numpy()).all():
            raise SchemaError(f"{c} must be finite on selected rows")
    if (df.loc[sel, "regret"] < 0).any():
        raise SchemaError("negative regret: oracle is not the minimum of the same costs")
    if (df.loc[sel, "scale"] <= 0).any():
        raise SchemaError("scale must be positive")
    orc = sel & (df["selector"] == "hindsight_oracle")
    if (df.loc[orc, "regret"] != 0).any():
        raise SchemaError("oracle has non-zero regret against itself")
    if (df.loc[orc, "chosen_family"] != df.loc[orc, "oracle_family"]).any():
        raise SchemaError("oracle chose a family other than oracle_family")
    return df
