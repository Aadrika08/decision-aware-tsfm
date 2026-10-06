"""Loading for K2: published results, the Omega manifest, and the Omega tables.

Manifest reading is isolated in `read_manifest`, which returns the same
canonical series table for every manifest layout. Everything downstream
(`aggregate_omega`, the statistics, the command) depends only on that table.
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

MANIFEST_VERSIONS: tuple[str, ...] = ("dev-v1", "frozen-v2")
DEFAULT_MANIFEST_VERSION = "dev-v1"
DEFAULT_OMEGA_COLUMN = "omega_levels"
OMEGA_PREFIX = "omega"
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


class ResultsLayoutError(ValueError):
    """The results file does not have the expected columns."""


@dataclass
class ManifestSeries:
    """Canonical series table, the same for every manifest layout.

    series columns: series_id, source_group, base, freq, length, and every
    Omega column the manifest carries. One row per series that could be
    matched to a GIFT-Eval (base dataset, frequency).
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
def read_manifest(path: str, version: str = DEFAULT_MANIFEST_VERSION) -> ManifestSeries:
    """Read an Omega manifest of the given layout version into the canonical table."""
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
    if missing_by_base and not allow_missing:
        lines = [f"Omega column {omega_col!r} has no value for {int((~has_value).sum())} of "
                 f"{len(s)} series, in {len(missing_by_base)} base dataset(s):"]
        for base, c in missing_by_base.items():
            note = ("no series has a value: the dataset would leave the analysis"
                    if c["n_missing"] == c["n_series"] else "its Omega would use the remaining series")
            lines.append(f"  {base}: {c['n_missing']} of {c['n_series']} series ({note})")
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
    return OmegaTables(by_base=by_base, by_base_freq=by_base_freq, omega_col=omega_col,
                       report=report)
