"""Model-free admission checks for the bounded four-model development pilot."""
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import json
import math
from pathlib import Path

from .cli import atomic_output, read_json
from .executor import _guard_source, load_source
from .pilot import pilot_models
from .planner import MODEL_FIELDS, NAMING_FIELDS, _naming, canonical, digest, plan


SERIES = ("dataset", "config", "series_id", "source_revision", "source_sha256")
PAIRS = {(h, seed) for h in (1, 24, 168) for seed in (42, 43)}


def _key(row):
    return tuple(row[field] for field in SERIES) + (row["origin"],)


def _omega_check(omega, groups, fixture, naming):
    if not isinstance(omega, dict) or omega.get("schema_version") != 1:
        raise ValueError("Omega handoff requires schema_version 1")
    if _naming(omega) != naming:
        raise ValueError("Omega and workload naming metadata must agree")
    if not isinstance(omega.get("fixture"), bool) or omega["fixture"] != fixture:
        raise ValueError("Omega and workload fixture labels must agree")
    variants = omega.get("variants")
    if (not isinstance(variants, dict) or not {"current", "last_4096"} <= variants.keys()
            or any(not isinstance(k, str) or not k or not isinstance(v, str) or not v.strip()
                   for k, v in variants.items())):
        raise ValueError("Omega variants need current/last_4096 calculation descriptions")
    if not isinstance(omega.get("rows"), list):
        raise ValueError("Omega handoff requires rows")
    indexed = {}
    for row in omega["rows"]:
        if not isinstance(row, dict):
            raise ValueError("Omega row must be a mapping")
        if set(NAMING_FIELDS) & row.keys():
            raise ValueError("Omega naming metadata belongs in the sidecar header")
        key = _key(row)
        variant = row["variant"]
        if key not in groups or variant not in variants:
            raise ValueError("Omega row has no matching pilot source/origin or declared variant")
        joined = key + (variant,)
        if joined in indexed:
            raise ValueError("Duplicate Omega source/origin/variant")
        start, end = row["history_start"], row["history_end"]
        if (any(isinstance(x, bool) or not isinstance(x, int) for x in (start, end))
                or not 0 <= start < end <= key[-1]):
            raise ValueError("Omega history must be nonempty and end before the forecast origin")
        value = row["omega"]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError("Omega must be finite and within [0, 1]")
        indexed[joined] = row
    expected = {key + (variant,) for key in groups for variant in variants}
    if indexed.keys() != expected:
        raise ValueError(f"Missing Omega source/origin/variant rows: {len(expected - indexed.keys())}")
    for key in groups:
        current, capped = indexed[key + ("current",)], indexed[key + ("last_4096",)]
        if (capped["history_end"] != current["history_end"]
                or capped["history_start"] != max(current["history_start"], current["history_end"] - 4096)):
            raise ValueError("last_4096 must cap current history at the same cutoff")
    summary = {}
    for variant in variants:
        rows = [r for k, r in indexed.items() if k[-1] == variant]
        values = [r["omega"] for r in rows]
        summary[variant] = {"description": variants[variant], "rows": len(rows),
                            "minimum": min(values), "maximum": max(values),
                            "history_lengths": sorted({r["history_end"] - r["history_start"] for r in rows})}
    if summary["current"]["minimum"] == summary["current"]["maximum"]:
        raise ValueError("Pilot has no documented variation in current Omega")
    return summary


def inspect_pilot(workload, models, omega, *, base_dir):
    """Return evidence and plans; failure never yields an executable output plan."""
    report = {"status": "blocked", "evidence_kind": "pilot_readiness", "reasons": [],
              "checked_at": datetime.now(timezone.utc).isoformat(),
              "checker_sha256": digest(Path(__file__).read_bytes()),
              "source_windows_verified": 0}
    try:
        planned = plan(workload, models, base_dir=base_dir)
        naming = {field: planned["jobs"][0][field] for field in NAMING_FIELDS
                  if field in planned["jobs"][0]}
        report.update(fixture=planned["fixture"], workload_sha256=planned["workload_sha256"],
                      model_document_sha256=digest(canonical(models).encode()),
                      omega_document_sha256=digest(canonical(omega).encode()),
                      provenance=planned["provenance"])
        actual_models = [{field: row[field] for field in MODEL_FIELDS} for row in models["models"]]
        if sorted(map(canonical, actual_models)) != sorted(map(canonical, pilot_models())):
            raise ValueError("Pilot requires exactly the four qualified CPU/RTX 4050 model configurations")
        groups, mappings = defaultdict(list), {}
        for job in planned["jobs"]:
            # Guard EVERY source, including resolved aliases, before opening any source.
            _guard_source(job)
            short = tuple(job[k] for k in SERIES[:3])
            identity = (job.get("source_repository"),) + tuple(
                job[k] for k in ("source_revision", "source_sha256", "item_id", "channel"))
            if short in mappings and mappings[short] != identity:
                raise ValueError("One stable series ID maps to inconsistent source identities")
            mappings[short] = identity
            groups[_key(job)].append(job)
        if len(mappings) < 2 or len({(k[0], k[1]) for k in mappings}) < 2:
            raise ValueError("Pilot requires multiple series and dataset configurations")
        for jobs in groups.values():
            coverage = {(j["model"], j["horizon"], j["seed"]) for j in jobs}
            expected = {(m["model"], h, s) for m in pilot_models() for h, s in PAIRS}
            if len(jobs) != len(expected) or coverage != expected:
                raise ValueError("Every series/origin requires all four models, horizons 1/24/168 and seeds 42/43")
            # The shared comparison context and split must be identical across jobs.
            fields = ("context_start", "context_length", "training_end", "validation_end",
                      "usable_start", "usable_end", "source_length", "start_timestamp", "frequency")
            if len({tuple(j[k] for k in fields) for j in jobs}) != 1:
                raise ValueError("Comparison jobs disagree on context or split bounds")
        report["omega"] = _omega_check(omega, groups, planned["fixture"], naming)
        verified = set()
        for job in planned["jobs"]:
            window = (job["source_path"], job["source_sha256"], job["source_length"],
                      job["context_start"], job["origin"], job["horizon"])
            if window not in verified:
                load_source(job)
                verified.add(window)
                report["source_windows_verified"] = len(verified)
        first_key = min(groups, key=lambda k: (k[2], k[0], k[1], k[-1], k[3], k[4]))
        first = deepcopy(workload)
        first["rows"] = sorted((row for row in first["rows"] if _key(row) == first_key), key=canonical)
        # Derive in memory so relative source paths keep their original base.
        # Exported plans carry absolute paths and retain the full plan's job IDs.
        first_plan = plan(first, models, base_dir=base_dir)
        if not {j["job_id"] for j in first_plan["jobs"]} <= {j["job_id"] for j in planned["jobs"]}:
            raise ValueError("Derived first slice changed forecast identities")
        counts = Counter((j["dataset"], j["config"], j["model"], j["horizon"]) for j in planned["jobs"])
        report.update(status="fixture_valid" if planned["fixture"] else "ready", planned_jobs=planned["planned_jobs"],
                      planned_rows=planned["planned_rows"], series=len(mappings), origins=len(groups),
                      coverage=[dict(dataset=d, config=c, model=m, horizon=h, jobs=n, rows=h*n)
                                for (d, c, m, h), n in sorted(counts.items())],
                      first_slice=dict(zip(SERIES + ("origin",), first_key)),
                      first_jobs=first_plan["planned_jobs"], first_rows=first_plan["planned_rows"],
                      approval_note="References are recorded provenance; readiness does not confer scientific approval.")
        return report, planned, first_plan
    except (ValueError, TypeError, KeyError, OSError) as error:
        report["reasons"].append(f"{type(error).__name__}: {error}")
        return report, None, None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", required=True, type=Path)
    parser.add_argument("--models", required=True, type=Path)
    parser.add_argument("--omega", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    args = parser.parse_args(argv)
    output = args.output_directory.resolve()
    try:
        if output.exists():
            raise ValueError("Readiness output directory must be new")
        inputs = (args.workload, args.models, args.omega)
        if any(path.resolve().is_relative_to(output) for path in inputs):
            raise ValueError("Output directory must not contain handoff inputs")
        try:
            workload, models, omega = (read_json(path) for path in inputs)
            # Do not create a directory over a source file, even for invalid input.
            if isinstance(workload, dict) and isinstance(workload.get("rows"), list):
                for row in workload["rows"]:
                    if isinstance(row, dict) and isinstance(row.get("source_path"), str):
                        source = (args.workload.resolve().parent / row["source_path"]).resolve()
                        if source.is_relative_to(output):
                            raise ValueError("Output directory must not contain staged sources")
            report, planned, first = inspect_pilot(workload, models, omega,
                                                   base_dir=args.workload.resolve().parent)
        except (ValueError, TypeError, KeyError, OSError) as error:
            report, planned, first = {"status": "blocked", "evidence_kind": "pilot_readiness",
                                     "reasons": [f"{type(error).__name__}: {error}"]}, None, None
        output.mkdir(parents=True, exist_ok=False)
        report["inputs"] = {name: str(path.resolve()) for name, path in zip(("workload", "models", "omega"), inputs)}
        if planned is not None:
            for name, document in (("plan.json", planned), ("first-plan.json", first)):
                atomic_output(output / name, json.dumps(document, indent=2, allow_nan=False) + "\n", refuse_existing=True)
        atomic_output(output / "report.json", json.dumps(report, indent=2, allow_nan=False) + "\n", refuse_existing=True)
        print(json.dumps({"status": report["status"], "report": str(output / "report.json"), "reasons": report["reasons"]}))
        return 0 if report["status"] in ("ready", "fixture_valid") else 2
    except (ValueError, OSError) as error:
        print(f"pilot-readiness: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
