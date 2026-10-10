"""Loading for K2: published results, the Omega manifest, and the Omega tables.

Manifest reading is isolated in `read_manifest`, which returns the same
canonical series table for every manifest layout. Everything downstream
(`aggregate_omega`, the statistics, the command) depends only on that table.

Two layouts are read. 'dev-v1' is the single development file and the default.
'manifest-v1' is the data owner's two-file layout (a series file and an Omega
file joined on series_id), in which an undefined Omega is null with a status.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..decision.guards import HELD_OUT_TOKENS, HeldOutLeakError, assert_no_held_out
from .heldout import check_listed, scan_names

# Stage0Prep subset directory names -> GIFT-Eval (base dataset, frequency).
# GIFT-Eval configs are base/frequency/term; one raw directory backs all three terms.
SUBSET_MAP: dict[str, tuple[str, str]] = {
    "bitbrains_fast_storage_5T": ("bitbrains_fast_storage", "5T"),
    "bitbrains_fast_storage_H": ("bitbrains_fast_storage", "H"),
    "bitbrains_rnd_5T": ("bitbrains_rnd", "5T"),
    "bitbrains_rnd_H": ("bitbrains_rnd", "H"),
    "bizitobs_application": ("bizitobs_application", "10S"),
    "bizitobs_l2c_5T": ("bizitobs_l2c", "5T"),
    "bizitobs_l2c_H": ("bizitobs_l2c", "H"),
    "bizitobs_service": ("bizitobs_service", "10S"),
    "electricity_15T": ("electricity", "15T"),
    "electricity_H": ("electricity", "H"),
    "electricity_D": ("electricity", "D"),
    "ett1_15T": ("ett1", "15T"), "ett1_H": ("ett1", "H"), "ett1_D": ("ett1", "D"),
    "ett2_15T": ("ett2", "15T"), "ett2_H": ("ett2", "H"), "ett2_D": ("ett2", "D"),
    "hierarchical_sales_D": ("hierarchical_sales", "D"),
    "jena_weather_10T": ("jena_weather", "10T"),
    "jena_weather_H": ("jena_weather", "H"),
    "jena_weather_D": ("jena_weather", "D"),
    "kdd_cup_2018_with_missing_H": ("kdd_cup_2018", "H"),
    "LOOP_SEATTLE_5T": ("loop_seattle", "5T"),
    "LOOP_SEATTLE_H": ("loop_seattle", "H"),
    "M_DENSE_H": ("m_dense", "H"),
    "m4_daily": ("m4_daily", "D"),
    "m4_hourly": ("m4_hourly", "H"),
    "m4_monthly": ("m4_monthly", "M"),
    "m4_weekly": ("m4_weekly", "W"),
    "saugeenday_D": ("saugeen", "D"),
    "saugeenday_W": ("saugeen", "W"),
    "solar_10T": ("solar", "10T"),
    "solar_H": ("solar", "H"),
    "SZ_TAXI_15T": ("sz_taxi", "15T"),
    "us_births_D": ("us_births", "D"),
    "us_births_W": ("us_births", "W"),
}

# Subset directories the existing record treats as duplicates of a mapped one
# and therefore leaves out: {duplicate: the mapped subset it duplicates}.
# The reader checks the claim (same lengths and Omega values) on every run.
KNOWN_DUPLICATE_SUBSETS: dict[str, str] = {"jena_weather": "jena_weather_10T"}

MANIFEST_VERSIONS: tuple[str, ...] = ("dev-v1", "frozen-v2", "manifest-v1")
DEFAULT_MANIFEST_VERSION = "dev-v1"
TWO_FILE_VERSION = "manifest-v1"
DEFAULT_OMEGA_COLUMN = "omega_levels"
OMEGA_PREFIX = "omega"
# manifest-v1: the key of the Omega file, and how each Omega column says whether
# its value is defined. A value is present exactly when its status is "ok".
OMEGA_VALUES_KEY = "omega_values"
OMEGA_STATUS_SUFFIX = "_status"
OMEGA_STATUS_OK = "ok"
# A base dataset that keeps fewer than this share of its series with a defined
# Omega is listed in the report. It is reported, not removed.
LOW_DEFINED_SHARE = 0.5
# Column names aggregate_omega computes itself. A manifest column with one of
# these names would be overwritten without notice, so such a manifest is refused.
COMPUTED_OMEGA_COLUMNS: tuple[str, ...] = ("omega_sd",)
# Keys a manifest file may use to describe itself. None is required; what is
# found is recorded in the run manifest.
MANIFEST_VERSION_KEYS: tuple[str, ...] = ("manifest_version", "version")
MANIFEST_FROZEN_KEY = "frozen"
HELD_OUT_LIST_KEY = "held_out_strict"
MAX_NAMES_IN_MESSAGE = 200


class OmegaColumnError(ValueError):
    """The requested Omega column cannot be used (absent, or not a float column)."""


class OmegaMissingError(OmegaColumnError):
    """The requested Omega column has no value for some series."""


class ManifestLayoutError(ValueError):
    """The manifest does not have the layout the chosen version expects."""


class ManifestJoinError(ManifestLayoutError):
    """The series file and the Omega file of a two-file manifest do not join one to one."""


class ResultsLayoutError(ValueError):
    """The results file does not have the expected columns."""


@dataclass
class ManifestSeries:
    """Canonical series table, the same for every manifest layout.

    series columns: series_id, source_group, base, freq, length, and every
    Omega column the manifest carries. One row per series that could be
    matched to a GIFT-Eval (base dataset, frequency). For 'manifest-v1' the
    table also carries the text column <Omega column>_status; an undefined
    Omega is NaN there, never 0.0.
    """
    series: pd.DataFrame
    report: dict = field(default_factory=dict)


@dataclass
class OmegaTables:
    by_base: pd.DataFrame        # base, <omega columns>, omega_sd, n_series, median_length
    by_base_freq: pd.DataFrame   # base, freq, omega, n_series
    # n_series counts the series that have a value in the selected Omega column.
    omega_col: str
    report: dict = field(default_factory=dict)


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------
# Published results
# --------------------------------------------------------------------------
def load_results(path: str, held_out_listed=()) -> tuple[pd.DataFrame, dict]:
    """Read the concatenated GIFT-Eval results and split the configuration name.

    Adds base, freq and term from 'dataset' = base/frequency/term. Rows whose
    dataset name is held out (see heldout.py; held_out_listed is the manifest's
    held_out_strict list) are removed here, before anything is computed. Every
    removed dataset name is listed in the report with its row count and the
    reason, and the command prints them, so a false positive of the substring
    rule is seen. Names that only match the punctuation-insensitive rule are
    reported and kept.
    """
    return prepare_results(pd.read_csv(path), held_out_listed)


def prepare_results(raw: pd.DataFrame, held_out_listed=()) -> tuple[pd.DataFrame, dict]:
    for col in ("dataset", "model"):
        if col not in raw.columns:
            raise ResultsLayoutError(f"results have no {col!r} column; columns: {list(raw.columns)}")
    listed = check_listed(held_out_listed)
    names = raw.dataset.astype(str)
    strict, fuzzy = scan_names(names.unique(), listed)
    held = names.isin(set(strict))
    rows_per_name = names.value_counts()
    report = {
        "n_rows_read": int(len(raw)),
        "held_out_tokens": list(HELD_OUT_TOKENS),
        "held_out_listed": list(listed),
        "held_out_rows_excluded": int(held.sum()),
        "held_out_datasets_excluded": sorted(strict),
        "held_out_exclusions": [{"dataset": name, "n_rows": int(rows_per_name[name]),
                                 "reasons": strict[name]} for name in sorted(strict)],
        "held_out_fuzzy_reports": [{"dataset": name, "n_rows": int(rows_per_name[name]),
                                    "reasons": fuzzy[name], "action": "kept, reported only"}
                                   for name in sorted(fuzzy)],
    }
    r = raw[~held].copy()
    parts = r.dataset.astype(str).str.split("/")
    bad = parts.map(len) != 3
    if bad.any():
        raise ResultsLayoutError(
            "expected dataset names of the form base/frequency/term, got "
            f"{sorted(r.dataset[bad].unique())[:5]}")
    r["base"] = parts.str[0]
    r["freq"] = parts.str[1]
    r["term"] = parts.str[2]
    dup = int(r.duplicated(["dataset", "model"]).sum())
    if dup:
        raise ResultsLayoutError(f"{dup} duplicate (dataset, model) rows in results")
    assert_no_held_out(r.dataset.unique(), "K2 results")
    report.update(n_rows_used=int(len(r)), n_models=int(r.model.nunique()),
                  n_configs=int(r.dataset.nunique()), n_base_datasets=int(r.base.nunique()))
    return r, report


# --------------------------------------------------------------------------
# Omega manifest
# --------------------------------------------------------------------------
def read_manifest(path: str, version: str = DEFAULT_MANIFEST_VERSION,
                  omega_values_path: str | None = None,
                  exclude_held_out: bool = False) -> ManifestSeries:
    """Read an Omega manifest of the given layout version into the canonical table.

    omega_values_path and exclude_held_out belong to the two-file layout
    'manifest-v1' only; with any other layout they are refused, not ignored.
    """
    if version != TWO_FILE_VERSION and (omega_values_path is not None or exclude_held_out):
        raise ManifestLayoutError(
            f"--omega-values and --exclude-held-out-matches belong to the two-file layout "
            f"{TWO_FILE_VERSION!r}; the layout selected is {version!r}, which is one file. "
            f"Pass --manifest-version {TWO_FILE_VERSION}, or leave the two options out.")
    if version == TWO_FILE_VERSION:
        if omega_values_path is None:
            raise ManifestLayoutError(
                f"layout {TWO_FILE_VERSION!r} is two files: --manifest is the series file "
                "(series_manifest.json) and --omega-values is the Omega file (omega_values.json). "
                "--omega-values was not given.")
        with open(path, encoding="utf-8") as f:
            series_raw = json.load(f)
        with open(omega_values_path, encoding="utf-8") as f:
            omega_raw = json.load(f)
        return _from_manifest_v1(series_raw, omega_raw, exclude_held_out=exclude_held_out)
    if version == "dev-v1":
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        return _from_dev_v1(raw)
    if version == "frozen-v2":
        raise NotImplementedError(
            "Manifest layout 'frozen-v2' is not implemented, because the frozen manifest was "
            "not available when this reader was written and its layout must not be guessed. "
            "Needed from the Omega owner: (1) the frozen manifest file; (2) the name of the "
            "key or table holding the active development series; (3) the field names for "
            "dataset, config, series id and length; (4) the names of the Omega columns, "
            "including the one computed on the last 4,096 steps; (5) how each dataset and "
            "config value maps to a GIFT-Eval 'base/frequency' name, which replaces SUBSET_MAP. "
            "Then add a reader that returns the canonical ManifestSeries table "
            "(series_id, source_group, base, freq, length, Omega columns) and nothing "
            "downstream of read_manifest needs to change.")
    raise ValueError(f"unknown manifest version {version!r}; known: {list(MANIFEST_VERSIONS)}")


def _from_dev_v1(raw: dict, subset_map: dict[str, tuple[str, str]] | None = None,
                 known_duplicates: dict[str, str] | None = None) -> ManifestSeries:
    """Current layout: {'active_dev_series': [{series_id, subset, length, omega_*...}, ...]}."""
    subset_map = SUBSET_MAP if subset_map is None else subset_map
    known_duplicates = KNOWN_DUPLICATE_SUBSETS if known_duplicates is None else known_duplicates
    if not isinstance(raw, dict) or "active_dev_series" not in raw:
        keys = sorted(raw) if isinstance(raw, dict) else type(raw).__name__
        raise ManifestLayoutError(f"dev-v1 manifest needs the key 'active_dev_series'; found {keys}")
    a = pd.DataFrame(raw["active_dev_series"])
    for col in ("series_id", "subset", "length"):
        if col not in a.columns:
            raise ManifestLayoutError(f"dev-v1 manifest rows need {col!r}; columns: {list(a.columns)}")
    clash = [c for c in COMPUTED_OMEGA_COLUMNS if c in a.columns]
    if clash:
        raise ManifestLayoutError(
            f"manifest column(s) {clash} have the name of a column K2 computes itself "
            "(the spread of the selected Omega column across series) and would be overwritten. "
            "Rename the manifest column before running K2.")
    # Held-out data in the development list is an upstream error: refuse the file.
    listed = check_listed(raw.get(HELD_OUT_LIST_KEY))
    fields = identity_fields(a)
    strict_hits, fuzzy_hits = held_out_hits(a, fields, listed)
    if strict_hits:
        raise HeldOutLeakError(_refusal_message(strict_hits, fields))
    assert_no_held_out(a.subset.unique(), "K2 manifest (active_dev_series)")

    counts = a.groupby("subset").size()
    duplicates_dropped, unmapped = {}, {}
    for subset in sorted(set(a.subset) - set(subset_map)):
        twin = known_duplicates.get(subset)
        if twin is not None and _same_series_values(a, subset, twin):
            duplicates_dropped[subset] = {"n_series": int(counts[subset]), "duplicate_of": twin}
        else:
            unmapped[subset] = int(counts[subset])

    s = a[a.subset.isin(subset_map)].copy()
    s["base"] = s.subset.map(lambda x: subset_map[x][0])
    s["freq"] = s.subset.map(lambda x: subset_map[x][1])
    s = s.rename(columns={"subset": "source_group"})
    omega_cols = omega_columns(s)
    keep = ["series_id", "source_group", "base", "freq", "length"] + omega_cols
    report = {
        "manifest_version": "dev-v1",
        "manifest_file": manifest_file_record(raw, "dev-v1"),
        "n_active_series_read": int(len(a)),
        "n_series_used": int(len(s)),
        "held_out_rows_found": 0,
        "held_out_listed": list(listed),
        "held_out_fields_checked": fields,
        "held_out_fuzzy_reports": [dict(hit, action="kept, reported only",
                                        in_a_mapped_subset=bool(hit["subsets"]
                                                                and set(hit["subsets"]) & set(subset_map)))
                                   for hit in fuzzy_hits],
        "omega_columns_available": omega_cols,
        "omega_columns_rejected": rejected_omega_columns(a),
        "unmapped_subsets_not_used": unmapped,
        "known_duplicate_subsets_not_used": duplicates_dropped,
        "mapped_subsets_absent_from_manifest": sorted(set(subset_map) - set(a.subset)),
        "subsets_used": {k: int(v) for k, v in s.groupby("source_group").size().items()},
    }
    return ManifestSeries(series=s[keep].reset_index(drop=True), report=report)


def _from_manifest_v1(series_raw: dict, omega_raw: dict, exclude_held_out: bool = False,
                      subset_map: dict[str, tuple[str, str]] | None = None) -> ManifestSeries:
    """Two-file layout of the data owner's manifest-v1.

    series file: {'active_dev_series': [{series_id, subset, length, ...}, ...],
                  'held_out_strict': [...]}
    Omega file:  {'omega_values': [{series_id, <column>, <column>_status, ...}, ...]}

    The two are joined on series_id and the join must be one to one. An
    undefined Omega is null with a status that is not "ok"; it stays undefined
    (NaN) in the table and is counted by status, it is never read as 0.0. A
    value of exactly 0.0 with status "ok" is a value (the file rounds to four
    places) and is counted. A value and a status that disagree are refused.

    Subsets are matched to GIFT-Eval names through SUBSET_MAP, as in dev-v1.

    Held-out names stop the read, as in dev-v1. With exclude_held_out=True the
    matching rows are left out instead, before any of their values is used, and
    the report lists them.
    """
    subset_map = SUBSET_MAP if subset_map is None else subset_map
    for raw, key, what in ((series_raw, "active_dev_series", "series"),
                           (omega_raw, OMEGA_VALUES_KEY, "Omega")):
        if not isinstance(raw, dict) or key not in raw:
            keys = sorted(raw) if isinstance(raw, dict) else type(raw).__name__
            raise ManifestLayoutError(
                f"{TWO_FILE_VERSION} {what} file needs the key {key!r}; found {keys}")
    a = pd.DataFrame(series_raw["active_dev_series"])
    o = pd.DataFrame(omega_raw[OMEGA_VALUES_KEY])
    for col in ("series_id", "subset", "length"):
        if col not in a.columns:
            raise ManifestLayoutError(
                f"{TWO_FILE_VERSION} series rows need {col!r}; columns: {list(a.columns)}")
    if "series_id" not in o.columns:
        raise ManifestLayoutError(
            f"{TWO_FILE_VERSION} Omega rows need 'series_id'; columns: {list(o.columns)}")
    in_both = sorted((set(a.columns) & set(o.columns)) - {"series_id"})
    if in_both:
        raise ManifestLayoutError(
            f"column(s) {in_both} are in the series file and in the Omega file; K2 cannot tell "
            "which one to use")
    clash = [c for c in COMPUTED_OMEGA_COLUMNS if c in o.columns or c in a.columns]
    if clash:
        raise ManifestLayoutError(
            f"manifest column(s) {clash} have the name of a column K2 computes itself "
            "(the spread of the selected Omega column across series) and would be overwritten. "
            "Rename the manifest column before running K2.")
    _check_one_to_one(a.series_id, o.series_id)
    merged = a.merge(o, on="series_id", how="inner", validate="one_to_one")
    if len(merged) != len(a):
        raise ManifestJoinError(
            f"joining the series file and the Omega file on series_id gave {len(merged)} rows "
            f"for {len(a)} series; the ids do not have the same type in the two files")

    # Held-out names: checked on every text field of the joined rows, before any value is used.
    listed = check_listed(series_raw.get(HELD_OUT_LIST_KEY))
    fields = identity_fields(merged)
    strict_hits, fuzzy_hits = held_out_hits(merged, fields, listed)
    flagged = pd.Series(False, index=merged.index)
    for hit in strict_hits:
        flagged |= merged[hit["field"]].astype(str) == hit["value"]
    if strict_hits and not exclude_held_out:
        raise HeldOutLeakError(
            _refusal_message(strict_hits, fields)
            + "\n  To run without these rows pass --exclude-held-out-matches; they are then left "
              "out before any value is used, and printed and recorded with their count.")
    excluded_by_subset = {str(k): int(v) for k, v in
                          merged.subset[flagged].astype(str).value_counts().sort_index().items()}
    merged = merged[~flagged]
    for field_name in fields:
        assert_no_held_out(merged[field_name].dropna().unique(),
                           f"K2 manifest (active_dev_series, {field_name})")

    value_cols = [c for c in o.columns if str(c).startswith(OMEGA_PREFIX)
                  and not str(c).endswith(OMEGA_STATUS_SUFFIX)]
    undefined, exact_zero = {}, {}
    for col in value_cols:
        status_col = f"{col}{OMEGA_STATUS_SUFFIX}"
        if status_col not in o.columns:
            raise ManifestLayoutError(
                f"{TWO_FILE_VERSION} Omega column {col!r} has no {status_col!r} column, so an "
                "undefined value could not be told from a value")
        if not pd.api.types.is_float_dtype(merged[col]):
            continue                      # not usable as Omega; listed in omega_columns_rejected
        status = merged[status_col]
        values = merged[col].to_numpy(dtype=float)
        is_text = status.map(lambda v: isinstance(v, str)).to_numpy(dtype=bool)
        is_ok = (status == OMEGA_STATUS_OK).to_numpy(dtype=bool)
        problems = {
            "have a status that is not text": ~is_text,
            f"are null with status {OMEGA_STATUS_OK!r}": np.isnan(values) & is_ok,
            f"have a value with a status other than {OMEGA_STATUS_OK!r}":
                ~np.isnan(values) & is_text & ~is_ok,
            "have a value that is not finite": np.isinf(values),
        }
        for problem, mask in problems.items():
            if mask.any():
                raise ManifestLayoutError(
                    f"{TWO_FILE_VERSION} Omega column {col!r}: {int(mask.sum())} row(s) {problem}, "
                    f"for example series {merged.series_id[mask].astype(str).head(3).tolist()}. "
                    f"A value is expected exactly when {status_col!r} is {OMEGA_STATUS_OK!r}; "
                    "the file is refused.")
        undefined[str(col)] = {str(k): int(v) for k, v in
                               status[np.isnan(values)].value_counts().sort_index().items()}
        exact_zero[str(col)] = int((values == 0.0).sum())

    unmapped = {str(k): int(v) for k, v in
                merged.subset[~merged.subset.isin(subset_map)].value_counts().sort_index().items()}
    s = merged[merged.subset.isin(subset_map)].copy()
    s["base"] = s.subset.map(lambda x: subset_map[x][0])
    s["freq"] = s.subset.map(lambda x: subset_map[x][1])
    s = s.rename(columns={"subset": "source_group"})
    omega_cols = omega_columns(s)
    status_cols = [f"{c}{OMEGA_STATUS_SUFFIX}" for c in omega_cols]
    keep = ["series_id", "source_group", "base", "freq", "length"] + omega_cols + status_cols
    report = {
        "manifest_version": TWO_FILE_VERSION,
        "manifest_file": manifest_file_record(series_raw, TWO_FILE_VERSION),
        "n_active_series_read": int(len(a)),
        "n_omega_rows_read": int(len(o)),
        "join": "series file and Omega file joined on series_id, checked one to one",
        "n_series_used": int(len(s)),
        "held_out_rows_found": int(flagged.sum()),
        "held_out_exclusion_requested": bool(exclude_held_out),
        "held_out_rows_excluded": int(flagged.sum()),
        "held_out_rows_excluded_by_subset": excluded_by_subset,
        "held_out_exclusions": [dict(hit, action="left out, reported") for hit in strict_hits],
        "held_out_listed": list(listed),
        "held_out_fields_checked": fields,
        "held_out_fuzzy_reports": [dict(hit, action="kept, reported only",
                                        in_a_mapped_subset=bool(hit["subsets"]
                                                                and set(hit["subsets"]) & set(subset_map)))
                                   for hit in fuzzy_hits],
        "omega_columns_available": omega_cols,
        "omega_columns_rejected": {name: what for name, what in rejected_omega_columns(merged).items()
                                   if not name.endswith(OMEGA_STATUS_SUFFIX)},
        "omega_undefined_by_column": undefined,
        "omega_exactly_zero_by_column": exact_zero,
        "unmapped_subsets_not_used": unmapped,
        "known_duplicate_subsets_not_used": {},
        "mapped_subsets_absent_from_manifest": sorted(set(subset_map) - set(merged.subset)),
        "subsets_used": {k: int(v) for k, v in s.groupby("source_group").size().items()},
    }
    return ManifestSeries(series=s[keep].reset_index(drop=True), report=report)


def _check_one_to_one(series_ids: pd.Series, omega_ids: pd.Series) -> None:
    """Refuse the two files unless every series id is exactly once in each."""
    series_ids, omega_ids = series_ids.astype(str), omega_ids.astype(str)
    problems = []
    for ids, where in ((series_ids, "series file"), (omega_ids, "Omega file")):
        repeated = sorted(ids[ids.duplicated()].unique())
        if repeated:
            problems.append(f"{len(repeated)} series id(s) occur more than once in the {where}, "
                            f"for example {repeated[:3]}")
    no_omega = sorted(set(series_ids) - set(omega_ids))
    no_series = sorted(set(omega_ids) - set(series_ids))
    if no_omega:
        problems.append(f"{len(no_omega)} series id(s) have no row in the Omega file, "
                        f"for example {no_omega[:3]}")
    if no_series:
        problems.append(f"{len(no_series)} Omega row id(s) are not in the series file, "
                        f"for example {no_series[:3]}")
    if problems:
        raise ManifestJoinError(
            f"the series file ({len(series_ids)} rows) and the Omega file ({len(omega_ids)} rows) "
            "do not join one to one on series_id: " + "; ".join(problems)
            + ". Nothing was read from either file; the two files must come from the same run.")


def _same_series_values(a: pd.DataFrame, subset: str, twin: str) -> bool:
    """True if two subsets carry the same lengths and Omega values, series for series."""
    cols = ["length"] + omega_columns(a)
    x = a.loc[a.subset == subset, cols].sort_values(cols).to_numpy(dtype=float)
    y = a.loc[a.subset == twin, cols].sort_values(cols).to_numpy(dtype=float)
    return x.shape == y.shape and x.size > 0 and bool(np.array_equal(x, y, equal_nan=True))


def omega_columns(series: pd.DataFrame) -> list[str]:
    """Float columns whose name starts with 'omega', in file order.

    Only float columns qualify. A boolean, integer or text column is not an
    Omega value even if its name starts with 'omega'; see rejected_omega_columns.
    """
    return [c for c in series.columns
            if str(c).startswith(OMEGA_PREFIX) and pd.api.types.is_float_dtype(series[c])]


def rejected_omega_columns(series: pd.DataFrame) -> dict[str, str]:
    """Columns named like an Omega column that are not float: {name: what they hold}."""
    out = {}
    for c in series.columns:
        if str(c).startswith(OMEGA_PREFIX) and not pd.api.types.is_float_dtype(series[c]):
            kinds = sorted({type(v).__name__ for v in series[c].dropna().head(1000)})
            out[str(c)] = f"{series[c].dtype} ({', '.join(kinds) or 'no values'})"
    return out


def identity_fields(rows: pd.DataFrame) -> list[str]:
    """The manifest columns that hold names: every column whose values are text.

    This covers subset and series_id and any dataset or config field a later
    layout adds, without having to know its name.
    """
    fields = []
    for c in rows.columns:
        values = rows[c].dropna()
        if len(values) and values.map(lambda v: isinstance(v, str)).all():
            fields.append(str(c))
    return fields


def held_out_hits(rows: pd.DataFrame, fields: list[str], listed=()) -> tuple[list[dict], list[dict]]:
    """Held-out matches in the name fields of manifest rows: (strict, fuzzy).

    One entry per distinct (field, value), with the number of rows carrying it,
    the subsets they sit under and the first few series ids.
    """
    strict_hits, fuzzy_hits = [], []
    for field_name in fields:
        strict, fuzzy = scan_names(rows[field_name].dropna().unique(), listed)
        for found, hits in ((strict, strict_hits), (fuzzy, fuzzy_hits)):
            for value, reasons in found.items():
                match = rows[rows[field_name].astype(str) == value]
                hits.append({
                    "field": field_name, "value": value, "reasons": reasons,
                    "n_rows": int(len(match)),
                    "subsets": sorted(match.subset.astype(str).unique())[:5]
                    if "subset" in rows.columns else [],
                    "series_ids": match.series_id.astype(str).head(3).tolist()
                    if "series_id" in rows.columns else []})
    return strict_hits, fuzzy_hits


def _refusal_message(hits: list[dict], fields: list[str]) -> str:
    n_rows = sum(h["n_rows"] for h in hits)
    lines = [f"K2 manifest (active_dev_series): held-out data present. {len(hits)} name(s) on "
             f"{n_rows} row(s) match the held-out rule; the manifest is refused and nothing is "
             f"written. Fields checked: {fields}."]
    for h in hits[:MAX_NAMES_IN_MESSAGE]:
        lines.append(f"  {h['field']} = {h['value']!r}: {h['n_rows']} row(s), "
                     f"subset(s) {h['subsets']}, series {h['series_ids']}; "
                     f"matched {', '.join(h['reasons'])}")
    if len(hits) > MAX_NAMES_IN_MESSAGE:
        lines.append(f"  ... and {len(hits) - MAX_NAMES_IN_MESSAGE} more name(s)")
    lines.append("  If a name is not held-out data this is a false positive of the rule; the row "
                 "must be renamed or removed upstream, K2 does not skip it silently.")
    return "\n".join(lines)


def manifest_file_record(raw: dict, layout: str) -> dict:
    """What the manifest file says about itself: version and whether it is frozen.

    The dev-v1 file carries neither, so `frozen` is False unless the file has
    an explicit boolean 'frozen' key set to true.
    """
    declared_version = next((raw[k] for k in MANIFEST_VERSION_KEYS if k in raw), None)
    declared_frozen = raw.get(MANIFEST_FROZEN_KEY)
    if isinstance(declared_frozen, bool):
        basis = f"the file's {MANIFEST_FROZEN_KEY!r} key"
    else:
        declared_frozen = None
        basis = (f"the file has no boolean {MANIFEST_FROZEN_KEY!r} key; layout {layout} is the "
                 "development manifest, so it is recorded as not frozen")
    return {"layout_version": layout,
            "declared_version": declared_version if isinstance(declared_version, (str, int, float))
            else None,
            "frozen": bool(declared_frozen), "frozen_basis": basis,
            "top_level_keys": sorted(str(k) for k in raw)}


def aggregate_omega(manifest: ManifestSeries, omega_col: str = DEFAULT_OMEGA_COLUMN,
                    allow_missing: bool = False) -> OmegaTables:
    """Average Omega over series: one value per base dataset, and per (base, frequency).

    Every series counts once, so a base dataset's value is weighted towards
    its larger frequency subsets. omega_sd is the across-series standard
    deviation of the selected Omega column.

    The selected column must be a float column with a finite value for every
    series. If any series has none, the run stops and names the base datasets
    affected, because a dataset with no value would leave the analysis and
    change n without notice. allow_missing=True continues instead: series
    without a value are left out of the averages, base datasets and frequency
    subsets with no value at all are removed, and the report says which.
    """
    s = manifest.series
    available = omega_columns(s)
    if omega_col not in available:
        rejected = manifest.report.get("omega_columns_rejected", {})
        if omega_col in rejected:
            raise OmegaColumnError(
                f"Omega column {omega_col!r} is in the manifest but is not a float column: it is "
                f"{rejected[omega_col]}. Only float columns can be used as Omega. "
                f"Available Omega columns: {available}")
        raise OmegaColumnError(
            f"Omega column {omega_col!r} is not in the manifest. Available Omega columns: {available}")
    clash = [c for c in COMPUTED_OMEGA_COLUMNS if c in s.columns]
    if clash:
        raise ManifestLayoutError(
            f"manifest column(s) {clash} have the name of a column K2 computes itself")
    assert_no_held_out(s.base.unique(), "K2 Omega aggregation")

    has_value = pd.Series(np.isfinite(s[omega_col].to_numpy(dtype=float)), index=s.index)
    missing_by_base = {
        str(base): {"n_missing": int((~ok).sum()), "n_series": int(len(ok))}
        for base, ok in has_value.groupby(s.base) if (~ok).any()}
    without_value = sorted(b for b, c in missing_by_base.items() if c["n_missing"] == c["n_series"])
    status_report = _status_report(s, omega_col, has_value, missing_by_base)
    if missing_by_base and not allow_missing:
        lines = [f"Omega column {omega_col!r} has no value for {int((~has_value).sum())} of "
                 f"{len(s)} series, in {len(missing_by_base)} base dataset(s):"]
        for base, c in missing_by_base.items():
            note = ("no series has a value: the dataset would leave the analysis"
                    if c["n_missing"] == c["n_series"] else "its Omega would use the remaining series")
            lines.append(f"  {base}: {c['n_missing']} of {c['n_series']} series ({note})")
        if status_report:
            lines.append(f"  by status: {status_report['undefined_selected_omega_by_status']}")
        lines.append("  Nothing was written. Fix the column upstream, or pass --allow-missing-omega "
                     "to run on the series that have a value; the changed n is then printed.")
        raise OmegaMissingError("\n".join(lines))

    n_base_before = int(s.base.nunique())
    n_subsets_before = int(s.groupby(["base", "freq"]).ngroups)
    if missing_by_base:
        s = s.copy()
        s[omega_col] = s[omega_col].where(has_value)
        # Drop every frequency subset in which no series has a value. A base dataset
        # with no value at all loses all of its subsets this way and leaves the tables.
        subset_has_value = s.groupby(["base", "freq"])[omega_col].transform("count") > 0
        s = s[subset_has_value]
    grouped = s.groupby("base")
    by_base = grouped[available].mean()
    by_base["omega_sd"] = grouped[omega_col].std()
    by_base["n_series"] = grouped[omega_col].count()
    by_base["median_length"] = grouped["length"].median()
    by_base = by_base.reset_index()
    by_base_freq = (s.groupby(["base", "freq"])
                     .agg(omega=(omega_col, "mean"), n_series=(omega_col, "count"))
                     .reset_index())
    report = {
        "omega_col": omega_col,
        "allow_missing_omega": bool(allow_missing),
        "n_base_datasets": int(len(by_base)),
        "n_base_datasets_before_missing_omega": n_base_before,
        "n_base_frequency_subsets": int(len(by_base_freq)),
        "n_base_frequency_subsets_before_missing_omega": n_subsets_before,
        "n_series_missing_selected_omega": int((~has_value).sum()),
        "missing_selected_omega_by_base": missing_by_base,
        "base_datasets_removed_no_selected_omega": without_value,
    }
    report.update(status_report)
    return OmegaTables(by_base=by_base, by_base_freq=by_base_freq, omega_col=omega_col,
                       report=report)


def _status_report(s: pd.DataFrame, omega_col: str, has_value: pd.Series,
                   missing_by_base: dict) -> dict:
    """Why values of the selected column are undefined, and how many are exactly 0.0.

    Only for a layout that carries <Omega column>_status (manifest-v1); for the
    single-file layout the result is empty and the report keeps its old keys.
    """
    status_col = f"{omega_col}{OMEGA_STATUS_SUFFIX}"
    if status_col not in s.columns:
        return {}
    undefined = s.loc[~has_value, ["base", status_col]].astype(str)
    zero = s.loc[(s[omega_col] == 0.0).to_numpy(), "base"].astype(str)
    return {
        "undefined_selected_omega_by_status": {
            k: int(v) for k, v in undefined[status_col].value_counts().sort_index().items()},
        "undefined_selected_omega_by_base": {
            base: {k: int(v) for k, v in rows[status_col].value_counts().sort_index().items()}
            for base, rows in undefined.groupby("base")},
        "base_datasets_under_half_defined": {
            base: {"n_defined": c["n_series"] - c["n_missing"], "n_series": c["n_series"]}
            for base, c in missing_by_base.items()
            if 0 < c["n_series"] - c["n_missing"] < LOW_DEFINED_SHARE * c["n_series"]},
        "n_series_selected_omega_exactly_zero": int(len(zero)),
        "selected_omega_exactly_zero_by_base": {
            k: int(v) for k, v in zero.value_counts().sort_index().items()},
    }
