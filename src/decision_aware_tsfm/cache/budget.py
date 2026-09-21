"""Measured, compatibility-bounded forecast budget projections."""

from collections import defaultdict
import hashlib
import json
import math
from statistics import mean


SIGNATURE_FIELDS = (
    "model", "family", "adapter", "adapter_version", "checkpoint", "device",
    "batch_size", "quantile_method", "settings", "horizon", "context_length",
    "frequency", "implementation_sha256", "environment_sha256", "fixture", "config",
    "quantiles",
)
DEVICES = ("cpu", "cuda")


def measurement_key(job):
    """Return the stable timing-compatibility key for a planned job."""
    if not isinstance(job, dict):
        raise ValueError("Job must be a mapping")
    missing = [field for field in SIGNATURE_FIELDS if field not in job]
    if missing:
        raise ValueError(f"Job lacks timing signature fields: {', '.join(missing)}")
    try:
        canonical = json.dumps(
            {field: job[field] for field in SIGNATURE_FIELDS},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("Job timing signature is not canonical JSON") from error
    return hashlib.sha256(canonical.encode()).hexdigest()


def _nonnegative(value, label, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    if value < 0 or (positive and value == 0):
        raise ValueError(f"{label} must be {'positive' if positive else 'nonnegative'}")
    return value


def _verified_measurements(measurements):
    usable = []
    seen = {}
    for record in measurements:
        if not isinstance(record, dict):
            raise ValueError("Measurement must be a mapping")
        if record.get("status") != "verified":
            continue
        required = {
            "timing_key", "hardware", "device", "inference_seconds", "output_bytes",
            "peak_memory_bytes", "fixture", "job_id", "run_id", "attempt",
        }
        if "run_id" not in record:
            raise ValueError("Verified measurement is missing run_id")
        if required - record.keys():
            raise ValueError("Verified measurement is missing required fields")
        if record["device"] not in DEVICES or not isinstance(record["hardware"], str) or not record["hardware"]:
            raise ValueError("Measurement has an unsupported device or hardware label")
        if (not isinstance(record["timing_key"], str) or not isinstance(record["job_id"], str)
                or not isinstance(record["run_id"], str) or not record["run_id"]):
            raise ValueError("Measurement timing_key, job_id, and run_id must be nonempty strings")
        if not isinstance(record["fixture"], bool):
            raise ValueError("Measurement fixture must be boolean")
        if isinstance(record["attempt"], bool) or not isinstance(record["attempt"], int) or record["attempt"] < 1:
            raise ValueError("Measurement attempt must be a positive integer")
        _nonnegative(record["inference_seconds"], "inference_seconds", positive=True)
        _nonnegative(record["output_bytes"], "output_bytes")
        _nonnegative(record["peak_memory_bytes"], "peak_memory_bytes")
        evidence = (record["run_id"], record["job_id"], record["attempt"])
        encoded = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if evidence in seen:
            if seen[evidence] != encoded:
                raise ValueError("Conflicting duplicate measurement evidence")
            continue
        seen[evidence] = encoded
        usable.append(record)
    return usable


def estimate_budget(jobs, measurements, *, hardware, loading_seconds, io_seconds,
                    restart_fraction=0.0, failure_fraction=0.0):
    """Project wall time and storage only from exactly compatible measurements."""
    restart_fraction = _nonnegative(restart_fraction, "restart_fraction")
    failure_fraction = _nonnegative(failure_fraction, "failure_fraction")
    jobs = list(jobs)
    if not jobs:
        raise ValueError("At least one planned job is required")
    if not all(isinstance(value, dict) for value in (hardware, loading_seconds, io_seconds)):
        raise ValueError("Hardware and overhead evidence must be mappings")

    scenarios = set()
    keyed_jobs = []
    job_ids = []
    for planned in jobs:
        if planned.get("device") not in DEVICES:
            raise ValueError("Only cpu and cuda projections are supported")
        if not isinstance(planned.get("fixture"), bool):
            raise ValueError("Each job must explicitly identify fixture status")
        scenarios.add(planned["fixture"])
        if not isinstance(planned.get("job_id"), str) or not planned["job_id"]:
            raise ValueError("Each planned job requires a job_id")
        job_ids.append(planned["job_id"])
        keyed_jobs.append((planned, measurement_key(planned)))
    if len(job_ids) != len(set(job_ids)):
        raise ValueError("Duplicate planned job_id")
    if len(scenarios) != 1:
        raise ValueError("Fixture and real jobs require separate projections")

    verified = _verified_measurements(measurements)
    compatible = defaultdict(list)
    for record in verified:
        compatible[(record["device"], record["timing_key"], record["hardware"], record["fixture"])].append(record)

    resources = {}
    reasons = []
    storage = 0.0
    matched_jobs = 0
    margin = 1 + restart_fraction + failure_fraction
    fixture = scenarios.pop()
    for device in DEVICES:
        planned = [(job, key) for job, key in keyed_jobs if job["device"] == device]
        if not planned:
            resources[device] = {"planned_jobs": 0, "measured_sample_count": 0,
                                 "matched_signature_count": 0, "inference_seconds": 0.0,
                                 "loading_seconds": 0.0, "io_seconds": 0.0,
                                 "resource_hours": 0.0}
            continue
        label = hardware.get(device)
        if not isinstance(label, str) or not label:
            reasons.append(f"{device}: missing hardware identity")
        loading = loading_seconds.get(device)
        io = io_seconds.get(device)
        if loading is None:
            reasons.append(f"{device}: missing loading overhead evidence")
        else:
            _nonnegative(loading, f"{device} loading_seconds")
        if io is None:
            reasons.append(f"{device}: missing I/O overhead evidence")
        else:
            _nonnegative(io, f"{device} io_seconds")

        inference = 0.0
        sample_ids = set()
        matched_signatures = set()
        resource_complete = bool(label) and loading is not None and io is not None
        for planned_job, key in planned:
            records = compatible.get((device, key, label, fixture), [])
            if not records:
                resource_complete = False
                continue
            matched_jobs += 1
            matched_signatures.add(key)
            inference += mean(record["inference_seconds"] for record in records)
            storage += mean(record["output_bytes"] for record in records)
            sample_ids.update((record["run_id"], record["job_id"], record["attempt"])
                              for record in records)
        if any(not compatible.get((device, key, label, fixture)) for _, key in planned):
            reasons.append(f"{device}: no compatible timing evidence")
        resources[device] = {
            "planned_jobs": len(planned),
            "measured_sample_count": len(sample_ids),
            "matched_signature_count": len(matched_signatures),
            "inference_seconds": inference,
            "loading_seconds": loading,
            "io_seconds": io,
            "resource_hours": ((inference + loading + io) * margin / 3600
                               if resource_complete else None),
        }

    if fixture:
        reasons.append("fixture projection is not a real compute budget")
    return {
        "status": "complete" if not reasons else "incomplete",
        "scope": "resource_projection",
        "scenario": "fixture" if fixture else "real",
        "resources": resources,
        "storage_bytes": (int(storage) if storage.is_integer() else storage)
                         if matched_jobs == len(jobs) else None,
        "measured_storage_projection_bytes": int(storage) if storage.is_integer() else storage,
        "wall_clock_hours": None,
        "coverage": {"matched_jobs": matched_jobs, "planned_jobs": len(jobs)},
        "safety_margins": {
            "restart_fraction": restart_fraction,
            "failure_fraction": failure_fraction,
        },
        "assumptions": ["Loading and I/O inputs are explicit totals for the planned schedule",
                        "Restart and failure fractions multiply resource occupancy"],
        "limitations": ["Resource sums are not elapsed wall time; use measured run schedule"],
        "reasons": reasons,
    }
