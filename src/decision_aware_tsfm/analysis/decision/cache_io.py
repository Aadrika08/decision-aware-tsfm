"""Forecast-cache loading and validation.

load_cache() is the ONLY entry point that knows a producer's cache layout.
Everything downstream sees the canonical frame below.

Time convention (shared with the producer, agreed in #45):
    origin is the zero-based position of the FIRST forecast target;
    step runs 1..horizon; the target sits at position origin + step - 1;
    the model's context ends immediately before origin.

Canonical frame, one row per (dataset, config, series_id, model, seed, horizon, origin, step):
    dataset, config, series_id, model : str   (config may be "" under dataset-config-v1)
    seed     : int
    horizon  : int  forecast length H of the job
    origin   : int  position of the first target
    step     : int  1..H
    y        : float realised value (NaN allowed in canonical input only)
    q_0.01 ... q_0.99 : float quantile forecasts on TAU_GRID
    optional : family, job_id, timestamp (carried through when present)

Supported schemas
    canonical-v1  the frame above (used by the synthetic fixture)
    producer-v1   the production cache described by the cache owner (#45 / #47):
                  job_id, dataset, config, series_id, model, family (str);
                  origin, horizon, seed, step (int64); y_true (float64);
                  timestamp (ISO 8601 str); q_0.01 ... q_0.99 (float64).
                  (job_id, step) unique, no nulls or non-finite values,
                  quantiles non-decreasing, family in {classical, tsfm}.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import TAU_GRID, check_taus
from .schema import FAMILIES

KEY_COLS = ["dataset", "config", "series_id", "model", "seed", "horizon", "origin", "step"]
TARGET_KEYS = ["dataset", "config", "series_id", "horizon", "origin", "step"]
SERIES_ID_KEYS = ["dataset", "config", "series_id"]
SUPPORTED_SCHEMAS = ("canonical-v1", "producer-v1")

PRODUCER_STR = ["job_id", "dataset", "config", "series_id", "model", "family"]
PRODUCER_INT = ["origin", "horizon", "seed", "step"]
JOB_FIELDS = ["dataset", "config", "series_id", "model", "seed", "horizon", "origin"]


class CacheError(ValueError):
    pass


class DuplicateCacheKeyError(CacheError):
    pass


def qcol(tau: float) -> str:
    return f"q_{tau:.2f}"


def target_position(df: pd.DataFrame) -> pd.Series:
    """Position of each row's target in its series (shared convention)."""
    return df["origin"] + df["step"] - 1


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
    return validate_forecasts(_to_canonical(raw, schema_version, taus), taus)


def _to_canonical(raw: pd.DataFrame, schema_version: str, taus=TAU_GRID) -> pd.DataFrame:
    if schema_version == "canonical-v1":
        return raw
    if schema_version == "producer-v1":
        return _from_producer_v1(raw, taus)
    if schema_version == "fixture-v0":
        raise NotImplementedError(
            "fixture-v0 used origin = last observed step. The engine now uses the shared "
            "convention (origin = first target). Regenerate the fixture as canonical-v1.")
    raise NotImplementedError(f"cache schema {schema_version!r} is not mapped. "
                              f"Supported: {SUPPORTED_SCHEMAS}")


def _from_producer_v1(raw: pd.DataFrame, taus=TAU_GRID) -> pd.DataFrame:
    qcols = [qcol(x) for x in check_taus(taus)]
    need = PRODUCER_STR + PRODUCER_INT + ["y_true", "timestamp"] + qcols
    missing = [c for c in need if c not in raw.columns]
    if missing:
        raise CacheError(f"producer cache is missing columns {missing}")
    df = raw[need].copy()

    nulls = df.columns[df.isna().any()].tolist()
    if nulls:
        raise CacheError(f"producer cache has null values in {nulls}")
    for c in PRODUCER_INT:
        if not pd.api.types.is_integer_dtype(df[c]):
            raise CacheError(f"{c} must be int64, got {df[c].dtype}")
    for c in ["y_true"] + qcols:
        if not pd.api.types.is_float_dtype(df[c]):
            raise CacheError(f"{c} must be float64, got {df[c].dtype}")
    if not np.isfinite(df["y_true"].to_numpy()).all():
        raise CacheError("producer cache has non-finite y_true")
    if (df["origin"] < 0).any():
        raise CacheError("origin must be zero-based (>= 0)")
    bad_fam = sorted(set(df["family"]) - set(FAMILIES))
    if bad_fam:
        raise CacheError(f"family must be one of {FAMILIES}, got {bad_fam}")

    dup = df.duplicated(["job_id", "step"], keep=False)
    if dup.any():
        raise DuplicateCacheKeyError(f"{int(dup.sum())} rows share (job_id, step), e.g. "
                                     f"{df.loc[dup, ['job_id', 'step']].head(2).to_dict('records')}")
    per_job = df.groupby("job_id")[JOB_FIELDS].nunique()
    if (per_job > 1).any(axis=None):
        bad = per_job[(per_job > 1).any(axis=1)].index[:3].tolist()
        raise CacheError(f"jobs whose rows disagree on dataset/config/series/model/seed/horizon/origin: {bad}")
    fam_per_model = df.groupby("model")["family"].nunique()
    if (fam_per_model > 1).any():
        raise CacheError(f"models labelled with more than one family: {fam_per_model[fam_per_model > 1].index.tolist()}")

    try:
        ts = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)
    except (ValueError, TypeError) as e:
        raise CacheError(f"timestamp is not ISO 8601: {e}") from None
    _check_timestamps(df.assign(_ts=ts))

    return df.rename(columns={"y_true": "y"})


def _check_timestamps(df: pd.DataFrame) -> None:
    """Within a series, each target position has exactly one timestamp, and
    timestamps strictly increase with position."""
    d = df[SERIES_ID_KEYS + ["_ts"]].assign(_pos=target_position(df))
    per_pos = d.groupby(SERIES_ID_KEYS + ["_pos"])["_ts"].nunique()
    if (per_pos > 1).any():
        raise CacheError(f"{int((per_pos > 1).sum())} target positions map to more than one timestamp")
    u = d.drop_duplicates(SERIES_ID_KEYS + ["_pos"]).sort_values(SERIES_ID_KEYS + ["_pos"])
    step_ok = u.groupby(SERIES_ID_KEYS)["_ts"].diff().dropna() > pd.Timedelta(0)
    if not step_ok.all():
        raise CacheError("timestamps do not increase with target position within a series")


def validate_forecasts(df: pd.DataFrame, taus=TAU_GRID) -> pd.DataFrame:
    t = check_taus(taus)
    qcols = [qcol(x) for x in t]
    if "seed" not in df.columns:
        df = df.assign(seed=np.int64(0))
    need = KEY_COLS + ["y"] + qcols
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise CacheError(f"missing columns {missing}")
    for c in ("seed", "horizon", "origin", "step"):
        if not pd.api.types.is_integer_dtype(df[c]):
            raise CacheError(f"{c} must be integer, got {df[c].dtype}")
    if ((df["step"] < 1) | (df["step"] > df["horizon"])).any():
        raise CacheError("step must lie in 1..horizon")
    if (df["origin"] < 0).any():
        raise CacheError("origin must be >= 0")

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

    # one realised value per target position, across models, seeds and horizons
    yk = df["y"].fillna(np.inf)             # treat NaN as a value for the consistency check
    n_distinct = yk.groupby([df[k] for k in SERIES_ID_KEYS] + [target_position(df)]).nunique()
    if (n_distinct > 1).any():
        raise CacheError(f"{int((n_distinct > 1).sum())} target positions have different y "
                         "across models, seeds or horizons")
    return df


def restrict_to_representatives(df: pd.DataFrame, representatives: dict[str, str],
                                family_of_model: dict[str, str]) -> pd.DataFrame:
    """Keep one frozen representative model per family and label its family.

    Fails if a representative is absent or mislabelled (including against a
    family column supplied by the producer), if the two families do not cover
    exactly the same targets, or if they were run with different seed sets.
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
    expected = out["model"].map(family_of_model)
    if "family" in out.columns and (out["family"] != expected).any():
        bad = out.loc[out["family"] != expected, ["model", "family"]].drop_duplicates()
        raise CacheError(f"cache family labels disagree with the frozen mapping: {bad.to_dict('records')}")
    out["family"] = expected

    cover = out.groupby(TARGET_KEYS)["family"].nunique()
    if (cover != len(FAMILIES)).any():
        raise CacheError(f"{int((cover != len(FAMILIES)).sum())} targets are not covered "
                         "by both families; families must be compared on identical windows")

    seeds = (out.groupby(TARGET_KEYS[:-1] + ["family"])["seed"]
             .agg(lambda s: tuple(sorted(set(s)))).unstack("family"))
    diff = seeds[FAMILIES[0]] != seeds[FAMILIES[1]]
    if diff.any():
        raise CacheError(f"{int(diff.sum())} windows were run with different seed sets for the two "
                         f"families, e.g. {seeds[diff].head(2).reset_index().to_dict('records')}")

    nan_targets = out.loc[out["y"].isna(), TARGET_KEYS].drop_duplicates()
    out = out[out["y"].notna()].copy()
    out.attrs["n_targets_dropped_missing_y"] = int(len(nan_targets))
    return out
