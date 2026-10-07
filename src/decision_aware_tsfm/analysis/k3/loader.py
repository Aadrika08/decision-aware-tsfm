"""K3 input loader: verified cache + Omega sidecar -> validated, joined K3 inputs.

Reads the forecast cache (v1 consumer payload, #16/#31) and the separately
versioned Omega sidecar (proposed exchange format, docs/runbooks/development-pilot.md),
joins them on stable identity, and runs the K3 input checks required for
acceptance on #29.

Scope: this module validates and joins. It does NOT compute pinball loss or
decision cost -- that is the evaluation engine's job (Veer, #16 consumer contract).

Rules taken from docs/runbooks/cache-consumer-contract.md:
  * read partitions with pyarrow.parquet.ParquetFile(path).read(); never infer
    types from directory names; never glob -- only files in the published
    SHA-256 list are read;
  * join on (dataset, config, series_id, origin), never on row position or a
    global counter;
  * never attach Omega whose history extends past the forecast origin;
  * match actuals and windows across models before any comparison;
  * one explicit Omega variant per analysis -- variants are never mixed;
  * production names follow dataset-config-v1: `dataset` and `config` are
    separate fields and neither carries the legacy slash form, and a source
    repository such as Salesforce/GiftEval is header provenance, never a
    row's dataset name.

PROVISIONAL: the Omega sidecar format is a proposal until Dayson's Stage 0
export is agreed, and the per-partition hash list is assumed to be a mapping
{relative_path: sha256}. Both may change; keep them isolated here.

PROVISIONAL: the naming rules below are my reading of the producer contract
("Names/repository" in #45). dataset-config-v1 is required for real inputs;
NAMING_LEGACY exists only so the historical smoke release, whose rows carry
dataset="Salesforce/GiftEval" and config="ett1/H", can still be read for
comparison. Confirm before the real pilot.
"""
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from decision_aware_tsfm.cache.schema import TAUS, validate_payload

OMEGA_KEY = ("dataset", "config", "series_id", "origin")
# Bands straddle the published rule's transition region (below 0.40 no clear
# TSFM advantage, above 0.50 TSFM wins). Coverage is reported per band because
# a count of distinct Omega values says nothing about where they sit: eight
# values all above 0.50 are no evidence about a boundary at 0.45 (#32).
OMEGA_BANDS = (("below_0.40", -float("inf"), 0.40),
               ("0.40_to_0.50", 0.40, 0.50),
               ("above_0.50", 0.50, float("inf")))
COMPARISON_KEY = ("dataset", "config", "series_id", "origin", "horizon", "seed")
CAP = {"last_4096": 4096}
NAMING_V1 = "dataset-config-v1"
NAMING_LEGACY = "legacy-slash-form"
NAMINGS = (NAMING_V1, NAMING_LEGACY)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class K3InputError(ValueError):
    """Input is malformed or violates a join/leakage/coverage rule."""


def _fail(message, examples=None):
    if examples:
        shown = list(examples)[:5]
        more = len(examples) - len(shown) if hasattr(examples, "__len__") else 0
        message += f": {shown}" + (f" (+{more} more)" if more > 0 else "")
    raise K3InputError(message)


# ------------------------------------------------------------------ cache
def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_cache(base_dir, partition_hashes):
    """Read exactly the published partitions, verified against their SHA-256s.

    Returns (table, job_metadata) where job_metadata maps job_id -> the parsed
    `job` schema-metadata dict (empty for fixture partitions without one).
    """
    base = Path(base_dir)
    if not partition_hashes:
        _fail("Partition hash list is empty")
    bad = [p for p, h in partition_hashes.items() if not _SHA256.match(str(h).lower())]
    if bad:
        _fail("Malformed SHA-256 entries", bad)

    tables, job_meta, fixture_flags = [], {}, set()
    mismatched, missing = [], []
    for relative, expected in sorted(partition_hashes.items()):
        path = base / relative
        if not path.is_file():
            missing.append(relative)
            continue
        if _sha256(path) != expected.lower():
            mismatched.append(relative)
            continue
        table = pq.ParquetFile(path).read()
        metadata = table.schema.metadata or {}
        fixture_flags.add(b"fixture" in metadata)
        if b"job" in metadata:
            job = json.loads(metadata[b"job"])
            for job_id in set(table["job_id"].to_pylist()):
                job_meta[job_id] = job
        tables.append(table.replace_schema_metadata(None))

    if missing:
        _fail("Listed partitions not found", missing)
    if mismatched:
        _fail("Partition SHA-256 mismatch", mismatched)
    if len(fixture_flags) > 1:
        _fail("Cache mixes fixture and real partitions")

    table = validate_payload(pa.concat_tables(tables))
    return table, job_meta, fixture_flags == {True}


# ------------------------------------------------------------ omega sidecar
def load_omega(path, allow_fixture=False):
    """Parse and validate an Omega sidecar document."""
    doc = json.loads(Path(path).read_text())
    if doc.get("schema_version") != 1:
        _fail("Unsupported Omega schema_version", [doc.get("schema_version")])
    if not isinstance(doc.get("fixture"), bool):
        _fail("Omega document must declare fixture: true/false")
    if doc["fixture"] and not allow_fixture:
        _fail("Fixture Omega refused; pass allow_fixture=True only for tests")
    repository = doc.get("source_repository")
    if repository is not None and (not isinstance(repository, str) or not repository.strip()):
        _fail("source_repository must be a non-empty string when present")
    variants = doc.get("variants")
    if not isinstance(variants, dict) or not variants:
        _fail("Omega document must declare at least one variant with a description")
    rows = doc.get("rows")
    if not isinstance(rows, list) or not rows:
        _fail("Omega document has no rows")

    required = {"dataset": str, "config": str, "series_id": str, "source_revision": str,
                "source_sha256": str, "origin": int, "variant": str,
                "history_start": int, "history_end": int, "omega": (int, float)}
    seen, problems = {}, []
    for i, row in enumerate(rows):
        missing = [f for f, kind in required.items()
                   if row.get(f) is None or isinstance(row.get(f), bool)
                   or not isinstance(row.get(f), kind)]
        if missing:
            problems.append(f"row {i}: missing or wrong type {missing}")
            continue
        if row["variant"] not in variants:
            problems.append(f"row {i}: undeclared variant {row['variant']!r}")
        if not math.isfinite(row["omega"]):
            problems.append(f"row {i}: nonfinite omega")
        if not _SHA256.match(row["source_sha256"]):
            problems.append(f"row {i}: malformed source_sha256")
        if not 0 <= row["history_start"] < row["history_end"] <= row["origin"]:
            problems.append(f"row {i}: history bounds must satisfy 0 <= start < end <= origin")
        key = tuple(row[k] for k in OMEGA_KEY) + (row["variant"],)
        if key in seen:
            problems.append(f"row {i}: duplicate of row {seen[key]}")
        seen[key] = i
    if problems:
        _fail("Invalid Omega rows", problems)

    by_source = {}
    for row in rows:
        by_source.setdefault(tuple(row[k] for k in OMEGA_KEY), {})[row["variant"]] = row
    for source, found in by_source.items():
        absent = set(variants) - set(found)
        if absent:
            problems.append(f"{source}: missing {sorted(absent)}")
        if len({(r["source_revision"], r["source_sha256"]) for r in found.values()}) > 1:
            problems.append(f"{source}: variants disagree on source revision/hash")
        current = found.get("current")
        for name, cap in CAP.items():
            capped = found.get(name)
            if current and capped:
                start = max(current["history_start"], current["history_end"] - cap)
                if (capped["history_start"], capped["history_end"]) != (start, current["history_end"]):
                    problems.append(f"{source}: {name} bounds inconsistent with current")
    if problems:
        _fail("Inconsistent Omega variants", problems)
    return {"fixture": doc["fixture"], "variants": variants, "by_source": by_source,
            "source_repository": repository}


# -------------------------------------------------------------------- join
def join_omega(table, job_meta, omega, variant, require_provenance=True):
    """Attach one Omega variant to every cache (dataset, config, series_id, origin).

    With require_provenance (the default for real inputs), every job must carry
    source revision/hash metadata and it must match the Omega row. A real
    partition without that metadata is refused rather than silently skipped.
    """
    if variant not in omega["variants"]:
        _fail("Requested variant not declared", [variant])

    keys = set(zip(*(table[k].to_pylist() for k in OMEGA_KEY)))
    missing = sorted(k for k in keys if k not in omega["by_source"])
    if missing:
        _fail("Cache series/origins with no Omega row", missing)

    if require_provenance:
        unprovenanced = sorted(
            job_id for job_id in set(table["job_id"].to_pylist())
            if "source_revision" not in job_meta.get(job_id, {})
            or "source_sha256" not in job_meta.get(job_id, {}))
        if unprovenanced:
            _fail("Jobs without source revision/hash metadata; provenance is "
                  "mandatory for real inputs", unprovenanced)

    mismatched, wrong_repository, checked = [], [], 0
    for job_id, job in job_meta.items():
        if "source_revision" not in job or "source_sha256" not in job:
            continue
        rows = table.filter(pc.equal(table["job_id"], job_id))
        key = tuple(rows[k][0].as_py() for k in OMEGA_KEY)
        ref = omega["by_source"][key][variant]
        checked += 1
        if (ref["source_revision"], ref["source_sha256"]) != (job["source_revision"], job["source_sha256"]):
            mismatched.append(key)
        job_repository = job.get("source_repository")
        if (omega["source_repository"] and job_repository
                and job_repository != omega["source_repository"]):
            wrong_repository.append((key, job_repository, omega["source_repository"]))
    if mismatched:
        _fail("Omega computed on a different source revision than the forecasts", mismatched)
    if wrong_repository:
        _fail("Omega and forecasts name different source repositories", wrong_repository)

    omega_by_key = {k: omega["by_source"][k][variant]["omega"] for k in keys}
    unused = sorted(set(omega["by_source"]) - keys)
    return omega_by_key, {"provenance_checked_jobs": checked,
                          "unused_omega_sources": len(unused),
                          "source_repository": omega["source_repository"]}


def validate_naming(pairs, naming, where):
    """Check (dataset, config) pairs against the agreed production naming.

    Under dataset-config-v1 the two fields are separate: `dataset` is a source
    name such as ett1 or m4, `config` is its variant such as H or daily and may
    be empty for a source with no variant. Neither carries the legacy slash
    form, and neither is a source repository: Salesforce/GiftEval is header
    provenance, not a dataset name.
    """
    if naming not in NAMINGS:
        _fail(f"Unknown naming scheme; expected one of {NAMINGS}", [naming])
    if naming == NAMING_LEGACY:
        return
    problems = []
    for dataset, config in sorted(pairs):
        if not isinstance(dataset, str) or not dataset.strip():
            problems.append(f"{where}: empty dataset")
            continue
        if "/" in dataset:
            problems.append(
                f"{where}: dataset {dataset!r} is in the legacy slash form; "
                f"{NAMING_V1} wants separate dataset and config fields, with any "
                "source repository recorded as header provenance instead")
        if not isinstance(config, str):
            problems.append(f"{where}: config for {dataset!r} is not a string")
        elif "/" in config:
            problems.append(f"{where}: config {config!r} is in the legacy slash form")
    if problems:
        _fail(f"Names do not follow {NAMING_V1}", problems)


def omega_band_coverage(omega_by_key):
    """Per band: how many datasets, series-origins and distinct Omega values."""
    coverage = {}
    for label, low, high in OMEGA_BANDS:
        inside = {k: v for k, v in omega_by_key.items() if low <= v < high}
        coverage[label] = {
            "range": [None if low == -float("inf") else low,
                      None if high == float("inf") else high],
            "datasets": sorted({k[0] for k in inside}),
            "series_origins": len(inside),
            "distinct_omega": len({round(v, 12) for v in inside.values()}),
        }
    return coverage


# ----------------------------------------------------------- K3 input checks
def k3_input_checks(table):
    """Every comparison must contain both families with identical actuals/windows."""
    families, actuals, problems = {}, {}, []
    for row in table.to_pylist():
        key = tuple(row[k] for k in COMPARISON_KEY)
        families.setdefault(key, set()).add(row["family"])
        target = (row["y_true"], row["timestamp"])
        step_key = key + (row["step"],)
        if actuals.setdefault(step_key, target) != target:
            problems.append(step_key)

    incomplete = sorted(k for k, f in families.items() if f != {"classical", "tsfm"})
    if incomplete:
        _fail("Comparisons missing a family", incomplete)
    if problems:
        _fail("Actuals or timestamps differ across models for the same window", sorted(set(problems)))

    models = {}
    for model, family in set(zip(table["model"].to_pylist(), table["family"].to_pylist())):
        models.setdefault(family, []).append(model)
    return {
        "comparisons": len(families),
        "series": len({k[:3] for k in families}),
        "series_origins": len({k[:4] for k in families}),
        "horizons": sorted({k[4] for k in families}),
        "seeds": sorted({k[5] for k in families}),
        "models_by_family": {f: sorted(m) for f, m in sorted(models.items())},
        "taus": list(TAUS),
    }


# ------------------------------------------------------------------- entry
def load_k3_inputs(base_dir, partition_hashes, omega_path, variant,
                   allow_fixture=False, naming=NAMING_V1):
    """Full K3 input path. Returns (table, omega_by_key, report).

    report["status"] is "ok" for real inputs and "fixture_valid" when any
    input is a fixture -- the latter is never real-pilot acceptance (#29).
    """
    table, job_meta, cache_fixture = load_cache(base_dir, partition_hashes)
    if cache_fixture and not allow_fixture:
        _fail("Fixture cache refused; pass allow_fixture=True only for tests")
    omega = load_omega(omega_path, allow_fixture=allow_fixture)
    is_fixture = cache_fixture or omega["fixture"]

    validate_naming(set(zip(table["dataset"].to_pylist(), table["config"].to_pylist())),
                    naming, "cache")
    validate_naming({(k[0], k[1]) for k in omega["by_source"]}, naming, "omega sidecar")
    omega_by_key, join_report = join_omega(table, job_meta, omega, variant,
                                           require_provenance=not is_fixture)
    checks = k3_input_checks(table)

    values = np.array(list(omega_by_key.values()), dtype=float)
    distinct = len(np.unique(values))
    report = {
        "status": "fixture_valid" if is_fixture else "ok",
        "provenance_required": not is_fixture,
        "naming": naming,
        "variant": variant,
        "partitions": len(partition_hashes),
        "rows": len(table),
        **checks,
        **join_report,
        "distinct_omega": distinct,
        "distinct_datasets": len({k[0] for k in omega_by_key}),
        "omega_range": [float(values.min()), float(values.max())],
        "omega_band_coverage": omega_band_coverage(omega_by_key),
    }
    return table, omega_by_key, report
