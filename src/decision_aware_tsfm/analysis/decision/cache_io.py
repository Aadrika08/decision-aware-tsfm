"""Forecast-cache loading and validation.

load_cache() is the ONLY function that knows the producer's cache layout.
Everything downstream sees the canonical frame below. When Aadrika's schema
is final, add a branch to _to_canonical() and nothing else changes.

Canonical frame, one row per (dataset, config, series_id, model, horizon, origin, step):
    dataset, config, series_id, model : str
    horizon  : int  forecast length H of the job
    origin   : int  position of the last observed step, in the series' own step units
    step     : int  1..H; the target sits at position origin + step
    y        : float realised value (NaN allowed; dropped identically for all models)
    q_0.01 ... q_0.99 : float quantile forecasts on TAU_GRID
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import TAU_GRID, check_taus
from .schema import FAMILIES

KEY_COLS = ["dataset", "config", "series_id", "model", "horizon", "origin", "step"]
TARGET_KEYS = ["dataset", "config", "series_id", "horizon", "origin", "step"]
SUPPORTED_SCHEMAS = ("fixture-v0",)


class CacheError(ValueError):
    pass


class DuplicateCacheKeyError(CacheError):
    pass


def qcol(tau: float) -> str:
    return f"q_{tau:.2f}"


def load_cache(source, schema_version: str, taus=TAU_GRID) -> pd.DataFrame:
    """Read a cache (DataFrame, parquet file, or parquet directory) and return
    the validated canonical frame."""
    if isinstance(source, pd.DataFrame):
        raw = source.copy()
    else:
        p = Path(source)
        if not p.exists():
            raise FileNotFoundError(p)
        raw = pd.read_parquet(p)
    return validate_forecasts(_to_canonical(raw, schema_version), taus)


def _to_canonical(raw: pd.DataFrame, schema_version: str) -> pd.DataFrame:
    if schema_version == "fixture-v0":
        return raw
    raise NotImplementedError(
        f"cache schema {schema_version!r} is not mapped yet. Supported: {SUPPORTED_SCHEMAS}. "
        "Pending from the cache owner: how origin/step/horizon become a timestamp, "
        "and how dataset/config map to manifest subset names.")


def validate_forecasts(df: pd.DataFrame, taus=TAU_GRID) -> pd.DataFrame:
    t = check_taus(taus)
    qcols = [qcol(x) for x in t]
    need = KEY_COLS + ["y"] + qcols
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise CacheError(f"missing columns {missing}")
    for c in ("horizon", "origin", "step"):
        if not pd.api.types.is_integer_dtype(df[c]):
            raise CacheError(f"{c} must be integer, got {df[c].dtype}")
    if ((df["step"] < 1) | (df["step"] > df["horizon"])).any():
        raise CacheError("step must lie in 1..horizon")

    dup = df.duplicated(KEY_COLS, keep=False)
    if dup.any():
        raise DuplicateCacheKeyError(
            f"{int(dup.sum())} rows share a cache key, e.g. {df.loc[dup, KEY_COLS].head(2).to_dict('records')}")

    Q = df[qcols].to_numpy(dtype=float)
    if not np.isfinite(Q).all():
        bad = ~np.isfinite(Q).all(axis=1)
        raise CacheError(f"{int(bad.sum())} rows have missing or infinite quantiles")
    # tolerance absorbs float noise (~1e-15); any real crossing is far larger
    tol = 1e-9 * np.maximum(1.0, np.abs(Q[:, :-1]))
    crossing = (np.diff(Q, axis=1) < -tol).any(axis=1)
    if crossing.any():
        raise CacheError(f"{int(crossing.sum())} rows have crossing quantiles "
                         f"(q must be non-decreasing in tau), e.g. "
                         f"{df.loc[crossing, KEY_COLS].head(2).to_dict('records')}")

    # every model must be scored against the same realised value
    yk = df["y"].fillna(np.inf)             # treat NaN as a value for the consistency check
    n_distinct = yk.groupby([df[k] for k in TARGET_KEYS]).nunique()
    if (n_distinct > 1).any():
        raise CacheError(f"{int((n_distinct > 1).sum())} targets have different y across models")
    return df


def restrict_to_representatives(df: pd.DataFrame, representatives: dict[str, str],
                                family_of_model: dict[str, str]) -> pd.DataFrame:
    """Keep one frozen representative model per family and label its family.

    Fails if a representative is absent, is labelled with the wrong family, or
    if the two families do not cover exactly the same targets.
    """
    if set(representatives) != set(FAMILIES):
        raise CacheError(f"need one representative for each of {FAMILIES}")
    for fam, model in representatives.items():
        if family_of_model.get(model) != fam:
            raise CacheError(f"representative {model!r} is not labelled {fam!r}")
    out = df[df["model"].isin(representatives.values())].copy()
    for fam, model in representatives.items():
        if not (out["model"] == model).any():
            raise CacheError(f"representative {model!r} has no rows in the cache")
    out["family"] = out["model"].map(family_of_model)

    cover = out.groupby(TARGET_KEYS)["family"].nunique()
    if (cover != len(FAMILIES)).any():
        raise CacheError(f"{int((cover != len(FAMILIES)).sum())} targets are not covered "
                         "by both families; families must be compared on identical windows")

    n_nan = int(out["y"].isna().sum() // len(FAMILIES))
    out = out[out["y"].notna()].copy()
    out.attrs["n_targets_dropped_missing_y"] = n_nan
    return out
