"""Crash-safe execution and audit for configured forecast-cache plans."""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from collections import OrderedDict
from contextlib import contextmanager
from datetime import datetime, timezone
import errno
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import platform
import re
import tempfile
import time
from urllib.parse import quote
import uuid

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .adapters import load_adapter, validate_adapter
from .budget import measurement_key
from . import planner
from .planner import job_identity, target_timestamps
from .schema import QCOLS, SCHEMA, validate_payload


_ADAPTERS = {}
_SOURCES = OrderedDict()
_WINDOWS = OrderedDict()
_CACHE_LIMIT = 4
_TRANSIENT_ERRNOS = {errno.EAGAIN, errno.ETIMEDOUT, errno.ECONNRESET, errno.EINTR}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata_job(job):
    return {**job_identity(job), "job_id": job["job_id"]}


def _validate_job_id(job):
    expected = _sha(_canonical(job_identity(job)).encode())
    if job.get("job_id") != expected:
        raise ValueError("job_id does not match the canonical job identity")


def job_path(cache, job):
    return (Path(cache) / f"model={quote(str(job['model']), safe='')}"
            / f"config={quote(str(job['config']), safe='')}" / f"{job['job_id']}.parquet")


def _payload_hash(table):
    return _sha(_canonical(table.replace_schema_metadata(None).to_pydict()).encode())


def _table(job, actual, quantiles):
    columns = {name: [job[name]] * job["horizon"] for name in
               ("job_id", "dataset", "config", "series_id", "model", "family",
                "origin", "horizon", "seed")}
    columns.update(step=list(range(1, job["horizon"] + 1)), y_true=actual,
                   timestamp=target_timestamps(job))
    columns.update({name: quantiles[:, index] for index, name in enumerate(QCOLS)})
    return validate_payload(pa.table(columns, schema=SCHEMA))


def _with_metadata(table, job, measurement):
    encoded_measurement = _canonical(measurement).encode()
    metadata = {
        b"job": _canonical(_metadata_job(job)).encode(),
        b"measurement": encoded_measurement,
        b"measurement_sha256": _sha(encoded_measurement).encode(),
        b"payload_sha256": _payload_hash(table).encode(),
    }
    return table.replace_schema_metadata(metadata)


def _atomic_parquet(path, table):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            pq.write_table(table, stream, compression="zstd")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(_canonical(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def generation_lock(cache):
    """Hold one cross-platform coordinator lock and release it on every exit."""
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    stream = (cache / ".writer.lock").open("a+b")
    try:
        # Windows byte-range locks also forbid reading the locked byte.
        try:
            if os.fstat(stream.fileno()).st_size == 0:
                stream.write(b"0")
                stream.flush()
        except PermissionError as error:
            raise BlockingIOError("Another cache generator holds the coordinator lock") from error
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise BlockingIOError("Another cache generator holds the coordinator lock") from error
        else:
            import fcntl
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise BlockingIOError("Another cache generator holds the coordinator lock") from error
        yield
    finally:
        try:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        stream.close()


def _guard_source(job):
    path = Path(job.get("source_path", ""))
    raw = tuple(job.get(field, "") for field in
                ("source_revision", "source_repository", "dataset", "config", "series_id", "item_id")) + (str(path),)
    if any(planner._held_out(value) for value in raw):
        raise ValueError("M5 and PJM are held out and cannot be accessed")
    if planner._held_out(path.resolve(strict=False)):
        raise ValueError("M5 and PJM are held out and cannot be accessed")


def _plain_int(job, name, minimum=0):
    value = job.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _validate_serialized_job(job):
    _validate_job_id(job)
    _guard_source(job)
    for name, minimum in (("channel", 0), ("source_length", 1), ("usable_start", 0),
                          ("usable_end", 1), ("context_start", 0), ("context_length", 1000),
                          ("origin", 0), ("horizon", 1), ("seed", 0),
                          ("training_end", 0), ("validation_end", 0)):
        _plain_int(job, name, minimum)
    if (job["usable_start"] > job["context_start"]
            or job["usable_end"] > job["source_length"]
            or job["origin"] != job["context_start"] + job["context_length"]
            or job["training_end"] > job["origin"]
            or job["origin"] + job["horizon"] > job["validation_end"]
            or job["validation_end"] > job["usable_end"]):
        raise ValueError("Invalid source, context, training, or validation bounds")
    if not isinstance(job.get("source_sha256"), str) or not re.fullmatch(r"[0-9a-fA-F]{64}", job["source_sha256"]):
        raise ValueError("source_sha256 must contain 64 hexadecimal characters")
    path = Path(job.get("source_path", ""))
    if not path.is_absolute():
        raise ValueError("source_path must be absolute")
    expected_context_timestamp = planner._timestamps(
        job["start_timestamp"], job["frequency"], (job["context_start"],))[0]
    if job.get("context_start_timestamp") != expected_context_timestamp:
        raise ValueError("context_start_timestamp is inconsistent with source position")
    target_timestamps(job)


def _lru_put(cache, key, value):
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > _CACHE_LIMIT:
        cache.popitem(last=False)


def load_source(job):
    """Verify a planned NPY source and return only its context and target window."""
    _validate_serialized_job(job)
    path = Path(job["source_path"])
    stat = path.stat()
    source_key = (str(path.resolve()), job["source_sha256"], stat.st_size, stat.st_mtime_ns,
                  job["source_length"])
    source = _SOURCES.get(source_key)
    if source is None:
        if _file_sha(path) != job["source_sha256"]:
            raise ValueError("Source checksum differs from the planned source")
        source = np.load(path, allow_pickle=False, mmap_mode="r")
        if source.ndim != 1 or len(source) != job["source_length"]:
            raise ValueError("Source must be the planned one-dimensional length")
        if (not np.issubdtype(source.dtype, np.number)
                or np.issubdtype(source.dtype, np.bool_)
                or np.issubdtype(source.dtype, np.complexfloating)):
            raise ValueError("Source dtype must be real numeric")
        _lru_put(_SOURCES, source_key, source)
    else:
        _SOURCES.move_to_end(source_key)
    start, origin, horizon = job["context_start"], job["origin"], job["horizon"]
    window_key = source_key + (start, origin, horizon)
    cached = _WINDOWS.get(window_key)
    if cached is not None:
        _WINDOWS.move_to_end(window_key)
        return cached
    context = np.asarray(source[start:origin], dtype=np.float64).copy()
    actual = np.asarray(source[origin:origin + horizon], dtype=np.float64).copy()
    if not np.isfinite(context).all() or not np.isfinite(actual).all():
        raise ValueError("Context and target window must be finite")
    context.flags.writeable = actual.flags.writeable = False
    _lru_put(_WINDOWS, window_key, (context, actual))
    return context, actual


def _number(value, name, *, positive=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not np.isfinite(value) or value < 0 or (positive and value == 0)):
        raise ValueError(f"Measurement {name} must be a {'positive' if positive else 'nonnegative'} number")


def _measurement(table, job):
    metadata = table.schema.metadata or {}
    encoded = metadata.get(b"measurement")
    if encoded is None or metadata.get(b"measurement_sha256") != _sha(encoded).encode():
        raise ValueError("Cache measurement metadata is missing or corrupt")
    try:
        value = json.loads(encoded)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("Cache measurement metadata is invalid JSON") from error
    required = {"hardware", "device", "inference_seconds", "peak_memory_bytes", "fixture",
                "job_id", "run_id", "attempt", "model_load_seconds", "source_load_seconds"}
    if not isinstance(value, dict) or required - value.keys():
        raise ValueError("Cache measurement metadata is incomplete")
    if (not isinstance(value["hardware"], str) or not value["hardware"]
            or value["device"] != job["device"] or value["fixture"] is not job["fixture"]
            or value["job_id"] != job["job_id"]
            or not isinstance(value["run_id"], str) or not value["run_id"]):
        raise ValueError("Cache measurement provenance mismatch")
    if isinstance(value["attempt"], bool) or not isinstance(value["attempt"], int) or value["attempt"] < 1:
        raise ValueError("Cache measurement attempt must be a positive integer")
    _number(value["inference_seconds"], "inference_seconds", positive=True)
    for name in ("peak_memory_bytes", "model_load_seconds", "source_load_seconds"):
        _number(value[name], name)
    return value


def _validate_table(table, job):
    metadata = table.schema.metadata or {}
    if metadata.get(b"job") != _canonical(_metadata_job(job)).encode():
        raise ValueError("Cache job identity metadata mismatch")
    if metadata.get(b"payload_sha256") != _payload_hash(table).encode():
        raise ValueError("Cache payload checksum mismatch")
    validate_payload(table)
    _measurement(table, job)
    rows = table.to_pylist()
    identity = ("job_id", "dataset", "config", "series_id", "model", "family",
                "origin", "horizon", "seed")
    if any(any(row[key] != job[key] for key in identity) for row in rows):
        raise ValueError("Cache row identity mismatch")
    if [row["timestamp"] for row in rows] != target_timestamps(job):
        raise ValueError("Cache timestamps are misaligned")
    return table


def _read_parquet(path):
    # Close even when ParquetFile construction fails and its traceback survives.
    with Path(path).open("rb") as stream:
        return pq.ParquetFile(stream).read()


def read_job(cache, job):
    """Read and fully validate one cache partition without source or model access."""
    _validate_job_id(job)
    table = _read_parquet(job_path(cache, job))
    return _validate_table(table, job)


def _adapter_key(job):
    fields = ("model", "family", "adapter", "adapter_version", "checkpoint", "device",
              "batch_size", "quantile_method", "settings", "frequency", "context_length",
              "horizon", "fixture")
    return _canonical({key: job[key] for key in fields})


def _execute_job(job):
    source_started = time.perf_counter()
    try:
        context, actual = load_source(job)
    except BaseException as error:
        error.execution_phase = "source_preparation"
        raise
    source_seconds = time.perf_counter() - source_started
    key = _adapter_key(job)
    load_seconds = 0.0
    if key not in _ADAPTERS:
        started = time.perf_counter()
        try:
            _ADAPTERS[key] = load_adapter(job)
        except BaseException as error:
            error.execution_phase = "model_loading"
            raise
        load_seconds = time.perf_counter() - started
    started = time.perf_counter()
    try:
        quantiles, adapter_measurement = _ADAPTERS[key].predict(context, job)
    except BaseException as error:
        error.execution_phase = "inference"
        raise
    inference_seconds = max(time.perf_counter() - started, 1e-12)
    quantiles = np.asarray(quantiles, dtype=np.float64)
    if quantiles.shape != (job["horizon"], len(QCOLS)) or not np.isfinite(quantiles).all():
        raise ValueError("Adapter returned invalid quantile shape or values")
    if (np.diff(quantiles, axis=1) < 0).any():
        raise ValueError("Adapter returned crossing quantiles")
    return actual, quantiles, adapter_measurement, inference_seconds, load_seconds, source_seconds


def _transient(error):
    return isinstance(error, (TimeoutError, ConnectionError)) or (
        isinstance(error, OSError) and error.errno in _TRANSIENT_ERRNOS)


class _Ledger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, status, **values):
        event = {"status": status, "at": time.time(), **values}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(_canonical(event) + "\n")
            stream.flush()
            os.fsync(stream.fileno())


class _PostPublishError(Exception):
    pass


def _hardware(job, adapter_measurement=None):
    adapter_measurement = adapter_measurement or {}
    return (adapter_measurement.get("device", "cuda") if job["device"] == "cuda"
            else platform.platform())


def _verified_measurement(job, embedded, attempt, output_bytes, io_seconds):
    return {
        "status": "verified", "timing_key": measurement_key(job),
        "hardware": embedded["hardware"], "device": job["device"],
        "inference_seconds": embedded["inference_seconds"], "output_bytes": output_bytes,
        "peak_memory_bytes": embedded["peak_memory_bytes"], "fixture": job["fixture"],
        "job_id": job["job_id"], "run_id": embedded["run_id"], "attempt": attempt,
        "model_load_seconds": embedded["model_load_seconds"],
        "source_load_seconds": embedded["source_load_seconds"], "io_seconds": io_seconds,
    }


def _write_result(cache, job, result, attempt, run_id, ledger):
    actual, quantiles, adapter_measurement, inference, model_load, source_load = result
    embedded = {**adapter_measurement, "hardware": _hardware(job, adapter_measurement),
                "device": job["device"], "fixture": job["fixture"], "job_id": job["job_id"],
                "run_id": run_id,
                "peak_memory_bytes": adapter_measurement.get("peak_memory_bytes", 0),
                "inference_seconds": inference,
                "model_load_seconds": model_load, "source_load_seconds": source_load,
                "attempt": attempt}
    table = _with_metadata(_table(job, actual, quantiles), job, embedded)
    started = time.perf_counter()
    published = False
    try:
        _atomic_parquet(job_path(cache, job), table)
        published = True
        verified = read_job(cache, job)
        output_bytes = job_path(cache, job).stat().st_size
        io_seconds = time.perf_counter() - started
        measurement = _verified_measurement(job, embedded, attempt, output_bytes, io_seconds)
        ledger.add("complete", job_id=job["job_id"], attempt=attempt)
        ledger.add("verified", measurement=measurement, rows=len(verified))
        return measurement
    except BaseException as error:
        if not published or isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        try:
            verified = read_job(cache, job)
            recovered = _measurement(verified, job)
            return _verified_measurement(job, recovered, attempt, job_path(cache, job).stat().st_size,
                                         time.perf_counter() - started)
        except Exception as recovery_error:
            raise _PostPublishError(
                f"Partition was published but post-write verification failed: {recovery_error}") from error


def _quarantine(cache, path):
    destination = Path(cache) / "quarantine" / f"{uuid.uuid4().hex}-{Path(path).name}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(path, destination)
    return destination


def _failure(job, attempt, run_id, phase, started_at, started, error):
    phase = getattr(error, "execution_phase", phase)
    ended = datetime.now(timezone.utc).isoformat()
    elapsed = max(time.perf_counter() - started, 1e-12)
    return {
        "status": "failed", "timing_key": measurement_key(job),
        "hardware": _hardware(job), "device": job["device"], "fixture": job["fixture"],
        "job_id": job["job_id"], "run_id": run_id, "attempt": attempt, "phase": phase,
        "started_at": started_at, "ended_at": ended, "elapsed_seconds": elapsed,
        "inference_seconds": elapsed if phase == "execution" else 1e-12,
        "output_bytes": 0, "peak_memory_bytes": 0,
        "model_load_seconds": 0.0, "source_load_seconds": 0.0,
        "io_seconds": elapsed if phase == "publication" else 0.0,
        "error": f"{type(error).__name__}: {error}", "transient": _transient(error),
    }


def _recover_published(cache, job, attempt, run_id, publication_started):
    if not job_path(cache, job).exists():
        return None
    verified = read_job(cache, job)
    embedded = _measurement(verified, job)
    if embedded["run_id"] != run_id:
        return None
    return _verified_measurement(job, embedded, attempt, job_path(cache, job).stat().st_size,
                                 max(time.perf_counter() - publication_started, 1e-12))


def _attempt_serial(cache, job, run_id, ledger):
    failures = []
    for attempt in range(1, 3):
        started_at = datetime.now(timezone.utc).isoformat()
        started = time.perf_counter()
        ledger.add("running", job_id=job["job_id"], attempt=attempt)
        phase = "execution"
        publication_started = None
        try:
            result = _execute_job(job)
            phase = "publication"
            publication_started = time.perf_counter()
            return _write_result(cache, job, result, attempt, run_id, ledger), None, failures
        except BaseException as error:
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            if publication_started is not None:
                try:
                    recovered = _recover_published(
                        cache, job, attempt, run_id, publication_started)
                    if recovered is not None:
                        return recovered, None, failures
                except Exception:
                    pass
            failure = _failure(job, attempt, run_id, phase, started_at, started, error)
            failures.append(failure)
            ledger.add(**failure)
            if not _transient(error) or attempt == 2:
                return None, failure["error"], failures
    raise AssertionError("unreachable")


def _planned_jobs(plan, *, generate=False):
    if not isinstance(plan, dict) or not isinstance(plan.get("jobs"), list) or not plan["jobs"]:
        raise ValueError("Plan requires a nonempty jobs list")
    if (plan.get("schema_version") != 2 or isinstance(plan.get("planned_jobs"), bool)
            or plan.get("planned_jobs") != len(plan["jobs"])):
        raise ValueError("Expected a complete schema_version=2 plan")
    if (isinstance(plan.get("planned_rows"), bool)
            or plan.get("planned_rows") != sum(job["horizon"] for job in plan["jobs"])):
        raise ValueError("Plan row count mismatch")
    jobs = list(plan["jobs"])
    identifiers = [job.get("job_id") for job in jobs]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Plan contains duplicate job IDs")
    if any(job.get("fixture") != plan.get("fixture") for job in jobs):
        raise ValueError("Plan mixes fixture and real jobs")
    if len({tuple(job.get(field) for field in planner.NAMING_FIELDS) for job in jobs}) != 1:
        raise ValueError("Plan mixes naming conventions")
    current_implementation = planner._implementation() if generate else None
    current_environment = (_sha(planner.canonical(planner._environment()).encode())
                           if generate else None)
    for job in jobs:
        _validate_serialized_job(job)
        validate_adapter(job)
        if generate and job["batch_size"] != 1:
            raise ValueError("batch_size > 1 is not qualified for cache generation")
        if generate and (job["implementation_sha256"] != current_implementation
                         or job["environment_sha256"] != current_environment):
            raise ValueError("Plan implementation or environment fingerprint differs from this executor")
    planner.validate_series_mappings(jobs)
    return jobs


def audit(plan, cache):
    """Audit all planned and discovered partitions without loading sources or models."""
    jobs = _planned_jobs(plan)
    by_metadata = {_canonical(_metadata_job(job)).encode(): job for job in jobs}
    occurrences = {job["job_id"]: [] for job in jobs}
    corrupt, unexpected = [], []
    root = Path(cache)
    for path in root.rglob("*.parquet") if root.exists() else ():
        if "quarantine" in path.relative_to(root).parts:
            continue
        try:
            table = _read_parquet(path)
            metadata = table.schema.metadata or {}
            job = by_metadata.get(metadata.get(b"job"))
            if job is None:
                unexpected.append(str(path))
                continue
            _validate_table(table, job)
            occurrences[job["job_id"]].append((path, table))
            if path != job_path(cache, job):
                unexpected.append(str(path))
        except Exception as error:
            corrupt.append({"path": str(path), "error": f"{type(error).__name__}: {error}"})
    missing = [str(job_path(cache, job)) for job in jobs
               if job_path(cache, job) not in {path for path, _ in occurrences[job["job_id"]]}]
    duplicate = {job_id: [str(path) for path, _ in found]
                 for job_id, found in occurrences.items() if len(found) > 1}
    actuals = {}
    mismatched_actuals = []
    for job in jobs:
        found = occurrences[job["job_id"]]
        if len(found) != 1:
            continue
        table = found[0][1]
        key = (job.get("source_repository"), job["dataset"], job["config"], job["series_id"],
               job["item_id"], job["channel"], job["source_sha256"],
               job["origin"], job["horizon"])
        actual = np.asarray(table["y_true"].to_pylist())
        if key in actuals and not np.array_equal(actual, actuals[key]):
            mismatched_actuals.append(job["job_id"])
        actuals[key] = actual
    status = "complete" if not (missing or corrupt or unexpected or duplicate or mismatched_actuals) else "incomplete"
    verified = sum(any(path == job_path(cache, job) for path, _ in occurrences[job["job_id"]])
                   for job in jobs)
    return {"status": status, "planned_jobs": len(jobs),
            "verified_jobs": verified,
            "missing": missing, "corrupt": corrupt, "unexpected": unexpected,
            "duplicate": duplicate, "mismatched_actuals": mismatched_actuals}


def generate(plan, cache, *, workers=1, repair=False, limit_jobs=None, real=False):
    """Generate missing jobs, preserving evidence from failures and corrupt partitions."""
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if limit_jobs is not None and (isinstance(limit_jobs, bool) or not isinstance(limit_jobs, int) or limit_jobs < 1):
        raise ValueError("limit_jobs must be a positive integer")
    jobs = _planned_jobs(plan, generate=True)
    if not plan["fixture"]:
        if not real:
            raise ValueError("Real generation requires real=True")
        if any(not job.get("approval_reference") or not job.get("model_approval_reference") for job in jobs):
            raise ValueError("Real generation requires workload and model approval references")
    cache = Path(cache)
    run_id = uuid.uuid4().hex
    run_dir = cache / "runs" / run_id
    ledger = _Ledger(run_dir / "events.jsonl")
    started = time.perf_counter()
    generated, skipped, failed, measurements = [], [], {}, []
    run = {"status": "incomplete", "run_id": run_id,
           "plan_provenance": {key: value for key, value in plan.items() if key != "jobs"}}
    try:
        with generation_lock(cache):
            _atomic_json(run_dir / "plan.json", plan)
            ledger.add("planned", planned_jobs=len(jobs), workload_sha256=plan["workload_sha256"])
            pending = []
            for job in jobs:
                path = job_path(cache, job)
                if path.exists():
                    try:
                        read_job(cache, job)
                        skipped.append(job["job_id"])
                        ledger.add("skipped", job_id=job["job_id"], reason="verified cache hit")
                        continue
                    except Exception:
                        if not repair:
                            raise ValueError(f"Corrupt cache partition requires repair=True: {path}")
                        preserved = _quarantine(cache, path)
                        ledger.add("failed", job_id=job["job_id"], error="corrupt cache quarantined",
                                   quarantine=str(preserved))
                pending.append(job)
            if limit_jobs is not None:
                pending = pending[:limit_jobs]
            pending.sort(key=lambda job: (job["device"] != "cpu", _adapter_key(job), job["job_id"]))

            # CUDA stays in the coordinator; CPU jobs may use a bounded spawn pool.
            cpu = [job for job in pending if job["device"] == "cpu"]
            gpu = [job for job in pending if job["device"] != "cpu"]
            if workers == 1:
                for job in cpu + gpu:
                    measurement, error, attempt_failures = _attempt_serial(cache, job, run_id, ledger)
                    measurements.extend(attempt_failures)
                    if error:
                        failed[job["job_id"]] = attempt_failures[-1]
                    else:
                        generated.append(job["job_id"])
                        measurements.append(measurement)
            else:
                # Keep at most one submitted job per worker; retries and writes remain coordinated.
                attempts = {job["job_id"]: 1 for job in cpu}
                queue = iter(cpu)
                with ProcessPoolExecutor(max_workers=workers,
                                         mp_context=multiprocessing.get_context("spawn")) as pool:
                    active = {}
                    for _ in range(min(workers, len(cpu))):
                        job = next(queue)
                        ledger.add("running", job_id=job["job_id"], attempt=1)
                        active[pool.submit(_execute_job, job)] = (
                            job, datetime.now(timezone.utc).isoformat(), time.perf_counter())
                    while active:
                        done, _ = wait(active, return_when=FIRST_COMPLETED)
                        for future in done:
                            job, started_at, attempt_started = active.pop(future)
                            attempt = attempts[job["job_id"]]
                            resubmitted = False
                            phase = "execution"
                            try:
                                result = future.result()
                                phase = "publication"
                                publication_started = time.perf_counter()
                                measurement = _write_result(cache, job, result, attempt, run_id, ledger)
                                generated.append(job["job_id"])
                                measurements.append(measurement)
                            except BaseException as error:
                                if isinstance(error, (KeyboardInterrupt, SystemExit)):
                                    raise
                                recovered = None
                                if phase == "publication":
                                    try:
                                        recovered = _recover_published(
                                            cache, job, attempt, run_id, publication_started)
                                    except Exception:
                                        pass
                                if recovered is not None:
                                    generated.append(job["job_id"])
                                    measurements.append(recovered)
                                    resubmitted = False
                                else:
                                    failure = _failure(job, attempt, run_id, phase, started_at,
                                                       attempt_started, error)
                                    measurements.append(failure)
                                    ledger.add(**failure)
                                    resubmitted = _transient(error) and attempt < 2
                                    if resubmitted:
                                        attempts[job["job_id"]] += 1
                                        ledger.add("running", job_id=job["job_id"], attempt=attempt + 1)
                                        active[pool.submit(_execute_job, job)] = (
                                            job, datetime.now(timezone.utc).isoformat(), time.perf_counter())
                                    else:
                                        failed[job["job_id"]] = failure
                            if resubmitted:
                                continue
                            try:
                                following = next(queue)
                            except StopIteration:
                                continue
                            ledger.add("running", job_id=following["job_id"], attempt=1)
                            active[pool.submit(_execute_job, following)] = (
                                following, datetime.now(timezone.utc).isoformat(), time.perf_counter())
                for job in gpu:
                    measurement, error, attempt_failures = _attempt_serial(cache, job, run_id, ledger)
                    measurements.extend(attempt_failures)
                    if error:
                        failed[job["job_id"]] = attempt_failures[-1]
                    else:
                        generated.append(job["job_id"])
                        measurements.append(measurement)
            report = audit(plan, cache)
            run.update(status="complete" if report["status"] == "complete" and not failed else "incomplete",
                       generated=len(generated), skipped=len(skipped), failed=len(failed),
                       generated_job_ids=generated, skipped_job_ids=skipped, failures=failed,
                       measurements=measurements, wall_seconds=time.perf_counter() - started,
                       model_load_seconds={device: sum(m.get("model_load_seconds", 0) for m in measurements
                                                        if m["device"] == device)
                                           for device in ("cpu", "cuda")},
                       source_load_seconds=sum(m.get("source_load_seconds", 0) for m in measurements),
                       io_seconds={device: sum(m.get("io_seconds", 0) for m in measurements
                                               if m["device"] == device)
                                   for device in ("cpu", "cuda")}, audit=report)
            return run
    except BaseException as error:
        run.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                   error=f"{type(error).__name__}: {error}", generated=len(generated),
                   skipped=len(skipped), failed=len(failed), generated_job_ids=generated,
                   skipped_job_ids=skipped, failures=failed, measurements=measurements,
                   wall_seconds=time.perf_counter() - started)
        raise
    finally:
        _atomic_json(run_dir / "run.json", run)
