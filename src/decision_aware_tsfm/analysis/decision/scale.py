"""The scale that makes costs comparable across series (#29, item 7a).

Decision: https://github.com/Aadrika08/decision-aware-tsfm/issues/29#issuecomment-6185355042
(item 7a), the MASE-style denominator Abdullah suggested on #62.

    scale = mean |y_t - y_{t-24}| over every t in [history_start + 24, history_end)
            where both y_t and y_{t-24} are finite

that is the in-sample seasonal-naive MAE with season 24 (hourly data), computed
on the series' history BEFORE its earliest forecast origin in the workload.
history_end is that origin (the first forecast target, #45 convention), so no
target of any window can enter the scale. history_start is the workload's
usable_start (0 in the pilot). There is ONE scale per series: every role,
horizon and tau of the series is divided by the same number.

It replaces mean |y| over the unit's targets, which came from after the origin
and was 0 at horizon 1 for every solar series and four jena_weather series of
the pilot, so those (series, horizon) pairs were dropped.

A scale of 0, or a history with no finite pair (y_t, y_{t-24}), leaves the
series without a usable scale: its value is NaN with a status, and the engine
excludes the series with that reason. Nothing is ever divided by 0 and there
is no fallback to another scale.

Where the history comes from. The cache bundle does not carry it. The workload
(Dayson's workload.json, which run_pilot already reads) names each series'
source file (source_path, relative to the workload's folder or to an explicit
--sources folder) and its SHA-256. read_workload_sources checks every hash
before it reads a file and refuses a missing file, a hash mismatch, or rows of
one series that disagree about their source.

The engine takes the result as a table (SCALE_COLUMNS); attach_scale is the
one place that joins it to cost rows, for the engine and for the
representatives alike.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

from .cache_io import SERIES_ID_KEYS, CacheError
from .guards import assert_no_held_out
from .handoff import file_sha256

SEASON = 24
SCALE_KIND = "seasonal_naive_mae_season_24_history_before_first_origin"
SCALE_REFERENCE = ("https://github.com/Aadrika08/decision-aware-tsfm/issues/29"
                   "#issuecomment-6185355042 item 7a")
SCALE_COLUMNS = SERIES_ID_KEYS + ["scale", "status", "history_start", "history_end", "n_pairs"]
REQUIRED_COLUMNS = SERIES_ID_KEYS + ["scale", "history_end"]
STATUS_OK = "ok"
STATUS_ZERO = "zero_seasonal_naive_mae"            # every finite pair has y_t == y_{t-24}
STATUS_NO_PAIRS = "no_finite_seasonal_pair"        # history shorter than 25 points, or no finite pair
STATUS_NOT_POSITIVE = "not_positive_or_not_finite"  # a supplied table whose scale cannot be divided by
MISSING_SCALES = ("no scale table given. Scaled cost divides by each series' seasonal-naive MAE "
                  "(season 24) on its history before the first origin (#29 item 7a); build the "
                  "table with scale.read_workload_sources(workload).scales or scale.scale_table")
WORKLOAD_SOURCE_FIELDS = ("source_path", "source_sha256")


class ScaleError(CacheError):
    """The series scale cannot be built or does not fit the cost rows."""


# --------------------------------------------------------------------------- the statistic

def seasonal_naive_mae(history, season: int = SEASON) -> tuple[float, int]:
    """(mean |y_t - y_{t-season}| over the pairs where both are finite, number of such pairs).
    Returns (nan, 0) when there is no such pair."""
    if not isinstance(season, (int, np.integer)) or season < 1:
        raise ValueError(f"season must be a positive integer, got {season!r}")
    y = np.asarray(history, dtype=np.float64)
    if y.ndim != 1:
        raise ScaleError(f"a history must be one-dimensional, got shape {y.shape}")
    if y.size <= season:
        return float("nan"), 0
    now, before = y[season:], y[:-season]
    ok = np.isfinite(now) & np.isfinite(before)
    n = int(ok.sum())
    if n == 0:
        return float("nan"), 0
    return float(np.mean(np.abs(now[ok] - before[ok]))), n


def scale_row(key, values, history_end: int, history_start: int = 0, season: int = SEASON) -> dict:
    """One row of the scale table for the series key = (dataset, config, series_id), from its
    values at positions history_start .. history_end - 1."""
    values = np.asarray(values, dtype=np.float64)
    history_end, history_start = int(history_end), int(history_start)
    if not 0 <= history_start <= history_end <= values.size:
        raise ScaleError(f"{tuple(key)}: history [{history_start}, {history_end}) does not lie in a "
                         f"series of length {values.size}")
    s, n = seasonal_naive_mae(values[history_start:history_end], season)
    if n == 0:
        status = STATUS_NO_PAIRS
    elif not (np.isfinite(s) and s > 0):
        status = STATUS_ZERO
    else:
        status = STATUS_OK
    dataset, config, series_id = key
    return dict(dataset=str(dataset), config=str(config), series_id=str(series_id),
                scale=(s if status == STATUS_OK else float("nan")), status=status,
                history_start=history_start, history_end=history_end, n_pairs=n)


def scale_table(rows) -> pd.DataFrame:
    """The scale table (SCALE_COLUMNS) from scale_row dicts, sorted by series."""
    t = pd.DataFrame(list(rows), columns=SCALE_COLUMNS)
    t = t.astype({"scale": "float64", "history_start": "int64", "history_end": "int64",
                  "n_pairs": "int64"})
    return t.sort_values(SERIES_ID_KEYS).reset_index(drop=True)


def usable(scale) -> np.ndarray:
    """True where a scale can be divided by: finite and > 0."""
    s = np.asarray(scale, dtype=np.float64)
    return np.isfinite(s) & (s > 0)


# --------------------------------------------------------------------------- joining to cost rows

def check_scale_table(scales: pd.DataFrame) -> pd.DataFrame:
    """Validate a scale table and return SERIES_ID_KEYS + scale, status, history_end.
    A table built elsewhere may leave out status; a row then gets STATUS_OK if its scale is
    usable and STATUS_NOT_POSITIVE if not."""
    if scales is None:
        raise ScaleError(MISSING_SCALES)
    if not isinstance(scales, pd.DataFrame):
        raise ScaleError(f"the scale table must be a DataFrame, got {type(scales).__name__}")
    missing = [c for c in REQUIRED_COLUMNS if c not in scales.columns]
    if missing:
        raise ScaleError(f"the scale table lacks columns {missing}")
    s = scales[REQUIRED_COLUMNS + (["status"] if "status" in scales.columns else [])].copy()
    if s.duplicated(SERIES_ID_KEYS).any():
        raise ScaleError(f"the scale table has more than one row for "
                         f"{s.loc[s.duplicated(SERIES_ID_KEYS), SERIES_ID_KEYS].head(3).to_dict('records')}")
    v = pd.to_numeric(s["scale"], errors="coerce").astype("float64")
    if (s["scale"].notna() & v.isna()).any():
        raise ScaleError("scale values must be numeric or null")
    s["scale"] = v
    he = pd.to_numeric(s["history_end"], errors="coerce")
    if he.isna().any() or (he != np.floor(he)).any() or (he < 0).any():
        raise ScaleError("history_end must be a non-negative integer position for every series")
    s["history_end"] = he.astype("int64")
    ok = usable(s["scale"])
    if "status" not in s.columns:
        s["status"] = np.where(ok, STATUS_OK, STATUS_NOT_POSITIVE)
    elif (ok & (s["status"] != STATUS_OK)).any() or (~ok & (s["status"] == STATUS_OK)).any():
        raise ScaleError(f"scale and status disagree: status {STATUS_OK!r} must go with a finite "
                         "scale above 0, and only with one")
    s["scale"] = np.where(ok, s["scale"], np.nan)
    return s[SERIES_ID_KEYS + ["scale", "status", "history_end"]]


def attach_scale(frame: pd.DataFrame, scales: pd.DataFrame, context: str) -> pd.DataFrame:
    """Add each row's series scale as `scale` and its status as `scale_status`.

    frame holds SERIES_ID_KEYS and origin. Raises ScaleError when a series of frame has no
    row in the table (a missing join is a fault, never a reason to drop the series) or when
    a series' history_end lies after its earliest origin in frame (the scale would then
    contain target values). Rows of the table for series not in frame are ignored."""
    s = check_scale_table(scales)
    first = frame.groupby(SERIES_ID_KEYS, sort=True)["origin"].min().rename("first_origin").reset_index()
    m = first.merge(s, on=SERIES_ID_KEYS, how="left", indicator="_scale_row")
    absent = m["_scale_row"] != "both"
    if absent.any():
        raise ScaleError(f"{context}: {int(absent.sum())} series have no row in the scale table, e.g. "
                         f"{m.loc[absent, SERIES_ID_KEYS].head(3).to_dict('records')}. {MISSING_SCALES}")
    late = m["history_end"] > m["first_origin"]
    if late.any():
        raise ScaleError(f"{context}: the scale history of {int(late.sum())} series ends after the "
                         f"series' first origin, so it would contain target values, e.g. "
                         f"{m.loc[late, SERIES_ID_KEYS + ['history_end', 'first_origin']].head(3).to_dict('records')}")
    add = s[SERIES_ID_KEYS + ["scale", "status"]].rename(columns={"status": "scale_status"})
    return frame.merge(add, on=SERIES_ID_KEYS, how="left", validate="m:1")


def table_sha256(scales: pd.DataFrame) -> str:
    """SHA-256 of a scale table's rows (sorted by series; floats as repr, NaN as null)."""
    t = scales.sort_values(SERIES_ID_KEYS)
    recs = [{k: (None if isinstance(v, float) and not np.isfinite(v) else
                 (repr(float(v)) if isinstance(v, (float, np.floating)) else
                  (int(v) if isinstance(v, (int, np.integer)) else str(v))))
             for k, v in r.items()} for r in t.to_dict("records")]
    return hashlib.sha256(json.dumps(recs, sort_keys=True, separators=(",", ":"))
                          .encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- history from the workload

@dataclass(frozen=True)
class SeriesSources:
    """What read_workload_sources returns."""
    scales: pd.DataFrame                              # SCALE_COLUMNS + source_sha256
    values: dict = field(repr=False)                  # (dataset, config, series_id) -> read-only array
    report: dict = field(default_factory=dict)


def _relative(path_text: str, key) -> PurePosixPath:
    rel = PurePosixPath(str(path_text).replace("\\", "/"))
    if not str(path_text) or rel.is_absolute() or ".." in rel.parts:
        raise ScaleError(f"{key}: source_path {path_text!r} must be a relative path inside the "
                         "sources folder")
    return rel


def read_workload_sources(workload, sources=None, season: int = SEASON) -> SeriesSources:
    """Read every series' source named by the workload, check its SHA-256, and compute the
    scale on the history before the series' earliest origin in the workload.

    workload  path of workload.json (rows with dataset, config, series_id, origin,
              source_path, source_sha256; usable_start and source_length are used when present)
    sources   folder that source_path is relative to; default: the workload's folder

    Held-out names (M5, PJM) in a series name or source path stop the read before any file
    is opened."""
    workload = Path(workload)
    if not workload.is_file():
        raise ScaleError(f"no workload at {workload}; the scale needs the workload's source files")
    doc = json.loads(workload.read_text(encoding="utf-8"))
    rows = doc.get("rows") if isinstance(doc, dict) else None
    if not isinstance(rows, list) or not rows or not all(isinstance(r, dict) for r in rows):
        raise ScaleError(f"{workload.name}: rows must be a nonempty list of mappings")
    need = SERIES_ID_KEYS + ["origin"] + list(WORKLOAD_SOURCE_FIELDS)
    lacking = sorted({k for r in rows for k in need if r.get(k) is None})
    if lacking:
        raise ScaleError(f"{workload.name}: rows lack {lacking}. The scale (#29 item 7a) is computed "
                         "on each series' history, which the workload names by source_path and "
                         "source_sha256")
    for k in SERIES_ID_KEYS + ["source_path"]:
        assert_no_held_out({str(r[k]) for r in rows}, f"scale history: workload {k}")
    base = Path(sources) if sources is not None else workload.parent
    if not base.is_dir():
        raise ScaleError(f"sources folder {base} does not exist")

    by_series: dict = {}
    for r in rows:
        by_series.setdefault((str(r["dataset"]), str(r["config"]), str(r["series_id"])), []).append(r)
    out_rows, values = [], {}
    for key in sorted(by_series):
        rs = by_series[key]
        ident = {(r["source_path"], r["source_sha256"], r.get("usable_start", 0), r.get("source_length"))
                 for r in rs}
        if len(ident) != 1:
            raise ScaleError(f"{key}: workload rows disagree on source_path, source_sha256, "
                             f"usable_start or source_length: {sorted(map(str, ident))[:3]}")
        path_text, sha, start, length = next(iter(ident))
        path = base / _relative(path_text, key)
        if not path.is_file():
            raise ScaleError(f"{key}: history source {path_text!r} is missing from {base.name or base}. "
                             "The scale needs every series' history before its first origin; pass "
                             "the folder that holds the workload's source files (--sources)")
        if file_sha256(path) != sha:
            raise ScaleError(f"{key}: hash mismatch for source {path_text!r}: the file is not the "
                             "one the workload planned")
        arr = np.load(path, allow_pickle=False)
        if (arr.ndim != 1 or not np.issubdtype(arr.dtype, np.number)
                or np.issubdtype(arr.dtype, np.complexfloating) or arr.dtype == np.bool_):
            raise ScaleError(f"{key}: source {path_text!r} must be a one-dimensional real array")
        if length is not None and arr.size != int(length):
            raise ScaleError(f"{key}: source {path_text!r} has {arr.size} values, workload says {length}")
        arr = np.asarray(arr, dtype=np.float64)
        arr.flags.writeable = False
        first = min(int(r["origin"]) for r in rs)
        out_rows.append(dict(scale_row(key, arr, first, int(start or 0), season), source_sha256=sha))
        values[key] = arr
    scales = pd.DataFrame(out_rows, columns=SCALE_COLUMNS + ["source_sha256"])
    scales = scales.astype({"scale": "float64", "history_start": "int64", "history_end": "int64",
                            "n_pairs": "int64"}).sort_values(SERIES_ID_KEYS).reset_index(drop=True)
    report = dict(kind=SCALE_KIND, season=int(season), reference=SCALE_REFERENCE,
                  workload_name=workload.name,
                  sources_folder=("workload folder" if sources is None else Path(sources).name),
                  series=int(len(scales)), sources_hash_verified=int(len(values)),
                  series_without_usable_scale=int((~usable(scales["scale"])).sum()))
    return SeriesSources(scales=scales, values=values, report=report)


def check_targets_match_sources(frame: pd.DataFrame, sources: SeriesSources, y_col: str = "y") -> dict:
    """Every realised value in a forecast frame must equal its source value at position
    origin + step - 1, so the history the scale came from is the series the forecasts were
    scored on. Series of frame that the sources do not hold raise."""
    keys = list(SERIES_ID_KEYS)
    have = set(sources.values)
    need = {tuple(k) for k in frame[keys].drop_duplicates().itertuples(index=False, name=None)}
    absent = sorted(need - have)
    if absent:
        raise ScaleError(f"{len(absent)} series of the cache have no source in the workload, e.g. "
                         f"{absent[:3]}")
    checked = 0
    for key, g in frame.groupby(keys, sort=True):
        arr = sources.values[tuple(key)]
        pos = (g["origin"] + g["step"] - 1).to_numpy(dtype=np.int64)
        if (pos < 0).any() or (pos >= arr.size).any():
            raise ScaleError(f"{key}: target positions fall outside the source of length {arr.size}")
        y = g[y_col].to_numpy(dtype=np.float64)
        bad = ~((y == arr[pos]) | (np.isnan(y) & np.isnan(arr[pos])))
        if bad.any():
            raise ScaleError(f"{key}: {int(bad.sum())} realised values differ from the source file, "
                             "so the source is not the series the forecasts were scored on")
        checked += int(len(g))
    return dict(series=int(len(need)), target_rows_checked=checked)


def check_bundle_against_sources(forecasts: pd.DataFrame, omega: pd.DataFrame,
                                 sources: SeriesSources) -> dict:
    """A handoff bundle against the workload's sources. handoff.read_handoff has already
    checked that each job's source_sha256 equals its Omega rows'; here the Omega rows'
    source_sha256 must equal the workload's for every series of the cache, and every
    realised value (y_true) must equal the source at its position."""
    keys = list(SERIES_ID_KEYS)
    cache_series = forecasts[keys].drop_duplicates()
    bundle_sha = omega[keys + ["source_sha256"]].drop_duplicates().merge(cache_series, on=keys)
    if bundle_sha.duplicated(keys).any():
        raise ScaleError("the bundle names more than one source hash for one series")
    m = bundle_sha.merge(sources.scales[keys + ["source_sha256"]], on=keys, how="left",
                         suffixes=("_bundle", "_workload"), indicator="_row")
    absent = m["_row"] != "both"
    if absent.any():
        raise ScaleError(f"{int(absent.sum())} series of the bundle are not in the workload, e.g. "
                         f"{m.loc[absent, keys].head(3).to_dict('records')}")
    differ = m["source_sha256_bundle"] != m["source_sha256_workload"]
    if differ.any():
        raise ScaleError(f"{int(differ.sum())} series were forecast from a source other than the one "
                         f"the workload names, e.g. {m.loc[differ, keys].head(3).to_dict('records')}")
    out = check_targets_match_sources(forecasts, sources, y_col="y_true")
    out["source_hashes_match_bundle"] = int(len(m))
    return out
