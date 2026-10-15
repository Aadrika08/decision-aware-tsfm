"""Role table (validation / gap / evaluation) written from the engine's own rule.

Decision record 0007, section 1, sets the roles per (dataset, config, series_id,
horizon): the earliest floor(n / 3) origins are validation candidates, the rest
are evaluation, a candidate whose last target reaches the first evaluation
origin E becomes gap, and a (series, horizon) left with no validation origin is
excluded with a reason. The engine applies that rule to the origins it finds in
a cache and never reads a table. This module runs the same two engine functions
(costs.assign_roles, pipeline._drop_without_validation) on a list of origins, so
a table written here agrees with the engine by construction, and
check_role_table() compares any table (this one, or one built by hand) with
what the engine would assign.

Table, as in the record's last paragraph: one row per (dataset, config,
series_id, origin, horizon) with a role. Models and seeds share it. The engine
calls the evaluation role "test"; the table says "evaluation" (ENGINE_ROLE).
Rows of an excluded (series, horizon) stay in the table with the role the rule
gave them; the pair is named in the separate excluded list.

Not covered: the engine can also exclude a (series, horizon) whose realised
targets are all zero. That needs the values, so it cannot be known from a
workload. Origins are taken from every row of a cache, while the engine uses
the two representative models; the two agree when every model covers every
origin, which the pilot readiness check requires.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from .cache_io import SERIES_ID_KEYS, load_cache
from .costs import SERIES_KEYS, assign_roles
from .guards import assert_no_held_out
from .handoff import file_sha256, read_handoff
from .pipeline import NO_VALIDATION_REASON, _decision_origins, _drop_without_validation

TABLE_KEYS = ["dataset", "config", "series_id", "origin", "horizon"]
TABLE_COLUMNS = TABLE_KEYS + ["role"]
ENGINE_ROLE = {"validation": "validation", "gap": "gap", "evaluation": "test"}   # table word -> engine word
TABLE_ROLE = {engine: table for table, engine in ENGINE_ROLE.items()}
DECISION_COLUMNS = SERIES_KEYS + ["decision_origin"]
EXCLUDED_COLUMNS = SERIES_KEYS + ["reason"]
SUMMARY_KEYS = ["dataset", "config", "horizon"]
RULE_RECORD = "docs/decisions/0007-development-split-and-fitting-roles.md, section 1 (PR #57)"
RULE_FUNCTIONS = ("decision_aware_tsfm.analysis.decision.costs.assign_roles",
                  "decision_aware_tsfm.analysis.decision.pipeline._drop_without_validation")
FILES = dict(table="role_table.csv", decision_origins="decision_origins.csv",
             excluded="excluded_series_horizons.csv", summary="summary.csv")
SIDECAR = "role_table.json"
DIFFERENCES = ("missing_rows", "extra_rows", "role_disagreements", "duplicate_rows", "unknown_roles",
               "decision_origin_disagreements", "exclusion_disagreements")
_ROW_ORDER = SERIES_KEYS + ["origin"]


class RoleTableError(ValueError):
    pass


class RoleTableMismatch(RoleTableError):
    """A role table and the engine's rule disagree."""


class RoleTable(NamedTuple):
    table: pd.DataFrame             # TABLE_COLUMNS, every origin (excluded pairs included)
    decision_origins: pd.DataFrame  # SERIES_KEYS + decision_origin (E), kept pairs only
    excluded: pd.DataFrame          # SERIES_KEYS + reason
    val_fraction: float


def _keyed(frame: pd.DataFrame, columns, context: str) -> pd.DataFrame:
    """The named columns with string names and integer positions, checked for held-out data."""
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise RoleTableError(f"{context}: missing columns {missing}")
    d = frame[list(columns)].copy()
    if d.isna().any(axis=None):
        raise RoleTableError(f"{context}: null values in {d.columns[d.isna().any()].tolist()}")
    for c in set(columns) & {"origin", "horizon", "decision_origin"}:
        if len(d) and not pd.api.types.is_integer_dtype(d[c]):      # an empty list read from CSV has no dtype
            raise RoleTableError(f"{context}: {c} must be integer, got {d[c].dtype}")
        d[c] = d[c].astype("int64")
    for c in SERIES_ID_KEYS:
        d[c] = d[c].astype(str)
    assert_no_held_out(d["dataset"], context)
    assert_no_held_out(d["config"], context)
    return d


def distinct_origins(frame: pd.DataFrame, context: str = "origins") -> pd.DataFrame:
    """Distinct (dataset, config, series_id, origin, horizon), sorted. Seeds and models collapse."""
    d = _keyed(frame, TABLE_KEYS, context)
    if d.empty:
        raise RoleTableError(f"{context}: no origins")
    if (d["origin"] < 0).any() or (d["horizon"] < 1).any():
        raise RoleTableError(f"{context}: origin must be >= 0 and horizon >= 1")
    return d.drop_duplicates().sort_values(_ROW_ORDER).reset_index(drop=True)


def build_role_table(origins: pd.DataFrame, val_fraction: float = 1 / 3) -> RoleTable:
    """Roles for every origin, the decision origin E of every kept (series, horizon),
    and the excluded (series, horizon) pairs with the engine's reason.

    As in build_tables, nothing is excluded for a missing validation origin when
    val_fraction is 0, because the caller has then chosen to have no validation."""
    po = assign_roles(distinct_origins(origins, "role table"), val_fraction)
    kept, excluded = (_drop_without_validation(po) if val_fraction > 0
                      else (po, pd.DataFrame(columns=EXCLUDED_COLUMNS)))
    table = po.assign(role=po["role"].map(TABLE_ROLE))[TABLE_COLUMNS]
    decisions = _decision_origins(kept, "test")[DECISION_COLUMNS].astype({"decision_origin": "int64"})
    excluded = excluded[EXCLUDED_COLUMNS].astype({"horizon": "int64"})
    return RoleTable(table.sort_values(_ROW_ORDER).reset_index(drop=True),
                     decisions.sort_values(SERIES_KEYS).reset_index(drop=True),
                     excluded.sort_values(SERIES_KEYS).reset_index(drop=True), float(val_fraction))


def _records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records"))


def _differences(engine: pd.DataFrame, table: pd.DataFrame, keys, value: str) -> pd.DataFrame:
    """Rows present on one side only, or on both with a different value."""
    m = engine.merge(table, on=keys, how="outer", suffixes=("_engine", "_table"), indicator="side")
    bad = (m["side"] != "both") | (m[f"{value}_engine"] != m[f"{value}_table"])
    return m[bad].sort_values(keys)


def check_role_table(table: pd.DataFrame, origins: pd.DataFrame, val_fraction: float = 1 / 3,
                     excluded: pd.DataFrame | None = None, decision_origins: pd.DataFrame | None = None,
                     strict: bool = False) -> dict:
    """Compare a role table with the roles the engine assigns to these origins.

    origins: distinct_origins() of a workload or of a cache. excluded and
    decision_origins are the table's companion lists; when one is not supplied
    it is read off the table itself (a pair with no validation row counts as
    excluded, and E is the earliest evaluation origin of a pair that is kept).

    Returns a report with the number of matched rows and one list per kind of
    difference (DIFFERENCES); report["ok"] is True only if all are empty. With
    strict=True any difference raises RoleTableMismatch. Held-out names raise
    HeldOutLeakError in either mode.
    """
    want = build_role_table(origins, val_fraction)
    got = _keyed(table, TABLE_COLUMNS, "role table")
    duplicates = got[got.duplicated(TABLE_KEYS, keep=False)]
    unknown = got[~got["role"].isin(ENGINE_ROLE)]
    known = got[got["role"].isin(ENGINE_ROLE)].drop_duplicates(TABLE_KEYS)
    m = want.table.merge(got.drop_duplicates(TABLE_KEYS), on=TABLE_KEYS, how="outer",
                         suffixes=("_engine", "_table"), indicator="side")
    both = m[m["side"] == "both"]
    wrong = both[(both["role_engine"] != both["role_table"]) & both["role_table"].isin(ENGINE_ROLE)]

    if excluded is None:
        has_val = (known["role"] == "validation").groupby([known[k] for k in SERIES_KEYS]).any()
        none = has_val[~has_val].index.to_frame(index=False) if val_fraction > 0 else known.iloc[:0][SERIES_KEYS]
        excluded = none.assign(reason=NO_VALIDATION_REASON)
    got_excluded = _keyed(excluded, EXCLUDED_COLUMNS, "excluded list")
    if decision_origins is None:
        kept = known.merge(got_excluded[SERIES_KEYS], on=SERIES_KEYS, how="left", indicator="side")
        decision_origins = (kept[(kept["side"] == "left_only") & (kept["role"] == "evaluation")]
                            .groupby(SERIES_KEYS)["origin"].min().rename("decision_origin").reset_index())
    got_decisions = _keyed(decision_origins, DECISION_COLUMNS, "decision origins")

    report = dict(
        val_fraction=float(val_fraction), table_rows=int(len(got)), engine_rows=int(len(want.table)),
        matched_rows=int((both["role_engine"] == both["role_table"]).sum()),
        missing_rows=_records(m.loc[m["side"] == "left_only", TABLE_KEYS + ["role_engine"]]),
        extra_rows=_records(m.loc[m["side"] == "right_only", TABLE_KEYS + ["role_table"]]),
        role_disagreements=_records(wrong[TABLE_KEYS + ["role_engine", "role_table"]]),
        duplicate_rows=_records(duplicates), unknown_roles=_records(unknown),
        decision_origin_disagreements=_records(
            _differences(want.decision_origins, got_decisions, SERIES_KEYS, "decision_origin")),
        exclusion_disagreements=_records(_differences(want.excluded, got_excluded, SERIES_KEYS, "reason")))
    report["ok"] = not any(report[k] for k in DIFFERENCES)
    if strict and not report["ok"]:
        raise RoleTableMismatch(describe_differences(report))
    return report


def describe_differences(report: dict) -> str:
    parts = [f"{len(report[k])} {k}, e.g. {report[k][0]}" for k in DIFFERENCES if report[k]]
    return ("role table disagrees with the engine (val_fraction "
            f"{report['val_fraction']:.6g}): " + "; ".join(parts))


# ---- readers ---------------------------------------------------------------------------

def workload_origins(document: dict) -> pd.DataFrame:
    """Origins of the producer's workload document (cache.planner.plan input): a mapping
    with split "development" and rows that each carry dataset, config, series_id, origin
    and horizon, repeated per seed. Only those five fields are read here."""
    if not isinstance(document, dict) or document.get("split") != "development":
        raise RoleTableError("workload must be a mapping with split 'development'")
    rows = document.get("rows")
    if not isinstance(rows, list) or not rows or not all(isinstance(r, dict) for r in rows):
        raise RoleTableError("workload requires a nonempty list of row mappings")
    return distinct_origins(pd.DataFrame([{k: r.get(k) for k in TABLE_KEYS} for r in rows]), "workload")


def read_workload_origins(path) -> pd.DataFrame:
    return workload_origins(json.loads(Path(path).read_text(encoding="utf-8")))


def cache_origins(source, schema_version: str) -> pd.DataFrame:
    """Origins of a forecast cache, read and validated by the engine's own loader."""
    return distinct_origins(load_cache(source, schema_version), "cache")


def handoff_origins(bundle, variant: str) -> tuple[pd.DataFrame, dict]:
    """Origins of a producer handoff bundle, read by the engine's bundle reader."""
    forecasts, _, report = read_handoff(bundle, variant)
    return distinct_origins(load_cache(forecasts, "producer-v1"), "handoff bundle"), report


# ---- files -----------------------------------------------------------------------------

def read_csv(path) -> pd.DataFrame:
    """Names stay strings: an empty config is kept as "", and an ID such as 007 keeps its zeros."""
    return pd.read_csv(path, dtype={c: str for c in SERIES_ID_KEYS + ["role", "reason"]},
                       keep_default_na=False, encoding="utf-8")


def summarize(result: RoleTable) -> pd.DataFrame:
    """Rows of the table per dataset, config and horizon by role, and the excluded pairs."""
    t = result.table
    s = (t.groupby(SUMMARY_KEYS + ["role"]).size().unstack("role", fill_value=0)
         .reindex(columns=list(ENGINE_ROLE), fill_value=0))
    s.insert(0, "series", t.groupby(SUMMARY_KEYS)["series_id"].nunique())
    s["excluded_series_horizons"] = (result.excluded.groupby(SUMMARY_KEYS).size()
                                     .reindex(s.index, fill_value=0))
    return s.astype("int64").reset_index().rename_axis(columns=None)


def write_role_table(result: RoleTable, out_dir, provenance: dict) -> dict:
    """Write the table, E per kept pair, the excluded list, the summary and a sidecar.

    The four CSV files depend only on the set of origins, so they are identical for
    a reordered input. The sidecar adds the settings, the rule's source, the caller's
    provenance (input hash, code commit) and the SHA-256 of each CSV."""
    out = Path(out_dir)
    out.mkdir(parents=True)
    frames = dict(table=result.table, decision_origins=result.decision_origins,
                  excluded=result.excluded, summary=summarize(result))
    for name, frame in frames.items():
        frame.to_csv(out / FILES[name], index=False, lineterminator="\n", encoding="utf-8")
    roles = result.table["role"].value_counts()
    sidecar = dict(
        keys=TABLE_KEYS, roles=ENGINE_ROLE,
        roles_note="table word -> the engine's word; 'evaluation' is the role the engine calls 'test'",
        val_fraction=result.val_fraction, rule_record=RULE_RECORD, rule_functions=list(RULE_FUNCTIONS),
        excluded_reason=NO_VALIDATION_REASON,
        counts=dict(rows=int(len(result.table)),
                    series=int(len(result.table[SERIES_ID_KEYS].drop_duplicates())),
                    series_horizons=int(len(result.table[SERIES_KEYS].drop_duplicates())),
                    **{r: int(roles.get(r, 0)) for r in ENGINE_ROLE},
                    excluded_series_horizons=int(len(result.excluded))),
        files={FILES[name]: file_sha256(out / FILES[name]) for name in frames},
        **provenance)
    (out / SIDECAR).write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return sidecar
