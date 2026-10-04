"""Pure validation and content-addressed planning for configured forecasts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import subprocess
import sys
import warnings

from .adapters import validate_adapter
from .schema import TAUS


ROW_FIELDS = (
    "source_revision", "source_sha256", "dataset", "config", "series_id", "item_id",
    "channel", "frequency", "start_timestamp", "source_length", "usable_start",
    "usable_end", "context_start", "context_length", "origin", "horizon", "seed",
    "training_end", "validation_end", "origin_timestamp", "context_start_timestamp",
)
MODEL_FIELDS = (
    "model", "family", "adapter", "adapter_version", "checkpoint", "device",
    "batch_size", "quantile_method", "settings",
)
IDENTITY_FIELDS = ("schema_version", "fixture", "quantiles") + ROW_FIELDS + MODEL_FIELDS + (
    "implementation_sha256", "environment_sha256",
)
REQUIRED_ROW_FIELDS = (set(ROW_FIELDS) - {"origin_timestamp", "context_start_timestamp"}) | {"source_path"}


def canonical(value) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Value is not canonical JSON") from error


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def job_identity(job) -> dict:
    missing = [field for field in IDENTITY_FIELDS if field not in job]
    if missing:
        raise ValueError(f"Job identity is missing fields: {', '.join(missing)}")
    return {field: job[field] for field in IDENTITY_FIELDS}


def _timestamps(start, frequency, positions):
    import pandas as pd
    from pandas.tseries.frequencies import to_offset

    try:
        timestamp = pd.Timestamp(start)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*'H' is deprecated.*", category=FutureWarning)
            offset = to_offset(frequency)
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid start_timestamp or pandas frequency offset") from error
    if timestamp.tz is not None:
        raise ValueError("start_timestamp must be timezone-naive")
    if pd.isna(timestamp) or getattr(offset, "n", 1) <= 0:
        raise ValueError("Timestamp and frequency must advance from a real anchor")
    if not offset.is_on_offset(timestamp):
        raise ValueError("start_timestamp is not aligned to frequency")
    return [(timestamp + position * offset).isoformat() for position in positions]


def target_timestamps(job) -> list[str]:
    timestamps = _timestamps(job["start_timestamp"], job["frequency"],
                            range(job["origin"], job["origin"] + job["horizon"]))
    if job.get("origin_timestamp") != timestamps[0]:
        raise ValueError("origin_timestamp is inconsistent with source position")
    return timestamps


def _plain_int(value, name, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _approval(document, name, required):
    value = document.get("approval_reference")
    if required and (not isinstance(value, str) or not value.strip()):
        raise ValueError(f"Real {name} requires approval_reference")
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{name} approval_reference must be a string")
    return value


def _held_out(value):
    return bool({"m5", "pjm"} & set(re.split(r"[^a-z0-9]+", str(value).lower())))


def _validate_row(row, fixture, base_dir):
    if not isinstance(row, dict) or REQUIRED_ROW_FIELDS - row.keys():
        raise ValueError("Workload row is missing required fields")
    for name in ("source_revision", "dataset", "config", "series_id", "item_id", "frequency",
                 "start_timestamp"):
        if not isinstance(row[name], str) or not row[name]:
            raise ValueError(f"{name} must be a nonempty string")
    if not isinstance(row["source_sha256"], str) or not re.fullmatch(r"[0-9a-fA-F]{64}", row["source_sha256"]):
        raise ValueError("source_sha256 must contain 64 hexadecimal characters")
    if not isinstance(row["source_path"], str) or not row["source_path"]:
        raise ValueError("source_path must be a relative string")
    for name in ("source_revision", "source_path", "dataset", "config", "series_id", "item_id"):
        if _held_out(row[name]):
            raise ValueError("M5 and PJM are held out")
    relative = Path(row["source_path"])
    root = Path(base_dir).resolve()
    resolved = (root / relative).resolve()
    if relative.is_absolute() or not resolved.is_relative_to(root):
        raise ValueError("source_path must remain under base_dir")
    if not fixture and "_idx_" in row["series_id"].lower():
        raise ValueError("Real workloads require stable upstream series IDs")
    for name, minimum in (("channel", 0), ("source_length", 1), ("usable_start", 0),
                          ("usable_end", 1), ("context_start", 0), ("context_length", 1000),
                          ("origin", 0), ("horizon", 1), ("seed", 0),
                          ("training_end", 0), ("validation_end", 0)):
        _plain_int(row[name], name, minimum=minimum)
    valid = (row["usable_start"] <= row["context_start"]
             and row["usable_end"] <= row["source_length"]
             and row["origin"] == row["context_start"] + row["context_length"]
             and row["training_end"] <= row["origin"]
             and row["origin"] + row["horizon"] <= row["validation_end"] <= row["usable_end"])
    if not valid:
        raise ValueError("Invalid source, context, training, or validation bounds")
    context_time, origin_time = _timestamps(
        row["start_timestamp"], row["frequency"], (row["context_start"], row["origin"]))
    clean = {field: row[field] for field in REQUIRED_ROW_FIELDS}
    clean.update(context_start_timestamp=context_time, origin_timestamp=origin_time)
    return clean, str(resolved)


def _implementation():
    directory = Path(__file__).resolve().parent
    content = bytearray()
    for name in ("planner.py", "executor.py", "adapters.py", "pilot.py", "schema.py"):
        path = directory / name
        if path.exists():
            normalized = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
            content.extend(name.encode() + b"\0" + normalized + b"\0")
    return digest(bytes(content))


def _environment():
    versions = {"python": sys.version.split()[0]}
    for package in ("numpy", "scipy", "pandas", "pyarrow", "torch", "uni2ts",
                    "statsforecast", "statsmodels", "coreforecast", "utilsforecast",
                    "threadpoolctl"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _git_commit():
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                            capture_output=True, text=True)
    value = result.stdout.strip()
    return value if result.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", value) else None


def plan(workload: dict, models: dict, *, base_dir: Path) -> dict:
    if not isinstance(workload, dict) or workload.get("schema_version") != 1:
        raise ValueError("Workload schema_version must be 1")
    if workload.get("split") != "development" or not isinstance(workload.get("fixture"), bool):
        raise ValueError("Workload must use the development split and explicit fixture boolean")
    if workload.get("quantiles") != list(TAUS) or not isinstance(workload.get("rows"), list) or not workload["rows"]:
        raise ValueError("Workload requires the exact quantile grid and nonempty rows")
    if not isinstance(models, dict) or models.get("schema_version") != 1:
        raise ValueError("Models schema_version must be 1")
    if not isinstance(models.get("models"), list) or not models["models"]:
        raise ValueError("Models requires a nonempty models list")

    fixture = workload["fixture"]
    workload_approval = _approval(workload, "workload", not fixture)
    model_approval = _approval(models, "models", not fixture)
    labels = [model.get("model") for model in models["models"] if isinstance(model, dict)]
    if len(labels) != len(models["models"]) or len(labels) != len(set(labels)):
        raise ValueError("Model labels must be present and unique")
    for model in models["models"]:
        if set(MODEL_FIELDS) - model.keys():
            raise ValueError("Model is missing required fields")
        for field in ("model", "family", "adapter", "adapter_version", "quantile_method"):
            if not isinstance(model[field], str) or not model[field]:
                raise ValueError(f"Model {field} must be a nonempty string")
        if model["device"] not in ("cpu", "cuda"):
            raise ValueError("Model device must be cpu or cuda")
        _plain_int(model["batch_size"], "batch_size", minimum=1)
        protected = (set(ROW_FIELDS) | {"source_path", "schema_version", "fixture", "quantiles",
                                       "implementation_sha256", "environment_sha256"}) - set(MODEL_FIELDS)
        if protected & model.keys():
            raise ValueError("Model contains protected workload or identity fields")

    cleaned = [_validate_row(row, fixture, base_dir) for row in workload["rows"]]
    logical = set()
    for row, _ in cleaned:
        key = tuple(row[field] for field in ("dataset", "config", "source_revision", "source_sha256",
                                             "item_id", "channel", "context_start", "origin",
                                             "horizon", "seed"))
        if key in logical:
            raise ValueError("Duplicate logical workload row")
        logical.add(key)

    hashed_workload = dict(workload)
    hashed_workload["rows"] = sorted(workload["rows"], key=canonical)
    workload_hash = digest(canonical(hashed_workload).encode())
    implementation_hash = _implementation()
    environment = _environment()
    environment_hash = digest(canonical(environment).encode())
    jobs = []
    for row, source_path in cleaned:
        for model in models["models"]:
            configured = {field: model[field] for field in MODEL_FIELDS}
            job = dict(schema_version=2, fixture=fixture, quantiles=list(TAUS), **row, **configured,
                       implementation_sha256=implementation_hash,
                       environment_sha256=environment_hash)
            validate_adapter(job)
            job["job_id"] = digest(canonical(job_identity(job)).encode())
            job.update(source_path=source_path, workload_sha256=workload_hash)
            if workload_approval is not None:
                job["approval_reference"] = workload_approval
            if model_approval is not None:
                job["model_approval_reference"] = model_approval
            jobs.append(job)
    jobs.sort(key=lambda job: job["job_id"])
    counts = {label: sum(job["model"] == label for job in jobs) for label in sorted(labels)}
    provenance = {"environment": environment, "git_commit": _git_commit(),
                  "implementation_sha256": implementation_hash,
                  "environment_sha256": environment_hash}
    return {"schema_version": 2, "fixture": fixture, "workload_sha256": workload_hash,
            "jobs": jobs, "planned_jobs": len(jobs),
            "planned_rows": sum(job["horizon"] for job in jobs), "counts_by_model": counts,
            "provenance": provenance}
