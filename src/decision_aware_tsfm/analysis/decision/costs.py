"""From canonical forecasts to per-origin and per-unit family costs."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .cache_io import qcol
from .metrics import TAU_GRID, crps_weights, pinball

SERIES_KEYS = ["dataset", "config", "series_id", "horizon"]
ROLES = ("validation", "test", "gap")


def per_origin_costs(fc: pd.DataFrame, taus=TAU_GRID, crps_method: str = "trapezoid") -> pd.DataFrame:
    """One row per (series, horizon, origin, family, tau) with summed losses.

    Sums (not means) are stored so later aggregation is exact when origins have
    different numbers of valid steps.
    """
    t = np.asarray(taus, dtype=float)
    Q = fc[[qcol(x) for x in t]].to_numpy(dtype=float)
    y = fc["y"].to_numpy(dtype=float)
    L = pinball(y[:, None], Q, t)                       # (rows, K)
    crps = 2.0 * (L * crps_weights(t, crps_method)).sum(axis=1)

    g = fc[SERIES_KEYS + ["origin", "family", "model"]].copy()
    for k, x in enumerate(t):
        g[f"pb_{k}"] = L[:, k]
    g["crps"] = crps
    g["abs_y"] = np.abs(y)
    g["n"] = 1
    agg = g.groupby(SERIES_KEYS + ["origin", "family", "model"], sort=True).sum().reset_index()

    long = agg.melt(id_vars=SERIES_KEYS + ["origin", "family", "model", "crps", "abs_y", "n"],
                    value_vars=[f"pb_{k}" for k in range(t.size)],
                    var_name="k", value_name="pinball_sum")
    long["tau"] = t[long["k"].str[3:].astype(int).to_numpy()]
    long = long.rename(columns={"crps": "crps_sum", "abs_y": "abs_y_sum"}).drop(columns="k")
    return long[SERIES_KEYS + ["origin", "family", "model", "tau",
                               "pinball_sum", "crps_sum", "abs_y_sum", "n"]]


def assign_roles(po: pd.DataFrame, val_fraction: float) -> pd.DataFrame:
    """Split each series' origins in time: earliest -> validation, rest -> test.

    A validation origin is demoted to 'gap' if its targets (origin+1 .. origin+H)
    reach past the first test origin, so validation never sees a value that
    falls inside the test period.
    """
    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must be in [0, 1)")
    origins = po[SERIES_KEYS + ["origin"]].drop_duplicates().sort_values(SERIES_KEYS + ["origin"])
    rank = origins.groupby(SERIES_KEYS).cumcount()
    count = origins.groupby(SERIES_KEYS)["origin"].transform("size")
    n_val = np.floor(val_fraction * count + 1e-9).astype(int)   # 0.29 * 100 = 28.999... in floats
    origins["role"] = np.where(rank < n_val, "validation", "test")
    first_test = (origins[origins.role == "test"].groupby(SERIES_KEYS)["origin"].min()
                  .rename("first_test").reset_index())
    origins = origins.merge(first_test, on=SERIES_KEYS, how="left")
    leak = (origins.role == "validation") & (origins.origin + origins.horizon > origins.first_test)
    origins.loc[leak, "role"] = "gap"
    return po.merge(origins[SERIES_KEYS + ["origin", "role"]], on=SERIES_KEYS + ["origin"])


def unit_costs(po: pd.DataFrame, role: str | None) -> pd.DataFrame:
    """Aggregate to one row per unit (series, horizon, tau), both families side by side.

    role=None aggregates every origin (used only for datasets set aside to fit a selector).
    """
    d = po if role is None else po[po["role"] == role]
    keys = SERIES_KEYS + ["tau"]
    a = d.groupby(keys + ["family"]).agg(
        pinball_sum=("pinball_sum", "sum"), crps_sum=("crps_sum", "sum"),
        abs_y_sum=("abs_y_sum", "sum"), n=("n", "sum"),
        n_origins=("origin", "nunique")).reset_index()
    a["J"] = a.pinball_sum / a.n
    a["C"] = a.crps_sum / a.n
    a["scale"] = a.abs_y_sum / a.n
    w = a.pivot_table(index=keys, columns="family", values=["J", "C"], aggfunc="first")
    w.columns = [f"{'J' if m == 'J' else 'crps'}_{f}" for m, f in w.columns]
    meta = a[a.family == "classical"].set_index(keys)[["scale", "n_origins", "n"]]
    return w.join(meta).reset_index()


def n_origins_by_role(po: pd.DataFrame, role: str) -> pd.DataFrame:
    d = po[po.role == role]
    return (d.groupby(SERIES_KEYS)["origin"].nunique().rename(f"n_{role}_origins").reset_index())

