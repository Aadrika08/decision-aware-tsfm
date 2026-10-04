"""Issue #3: one development channel, two forecast models, resumable Parquet.

Run from this checkout with the pinned Python 3.11 smoke environment.
Model libraries are imported only when an uncached Moirai job is executed.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import sys
import tempfile
import time
import urllib.request
import warnings

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
DATA_REV = "30841734ac5cfddbd0c3bad6d09d2b6b32becbb0"
DATA_SHA = "e4f9484726edb0c28581df4d03b255239bab9b2c0ea14acfdb1fea69d79dd325"
MODEL_REV = "0c24ab99db2c1a70ea2a0fc03bf113329772ac64"
MODEL_REPO = "Salesforce/moirai-1.1-R-small"
TAUS = [0.01, 0.05, 0.09, 0.25, 0.40, 0.50, 0.60, 0.75, 0.91, 0.95, 0.99]
QCOLS = [f"q_{t:.2f}" for t in TAUS]
HORIZON = 24
CONTEXT = 2048
ORIGINS = [16412, 16436]
START = datetime(2016, 7, 1)
MODELS = ["seasonal_naive", "moirai_1.1_R_small"]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def planned_jobs(device="cuda"):
    jobs = []
    for model in MODELS:
        for index, origin in enumerate(ORIGINS):
            job = dict(
                schema_version=1, dataset="Salesforce/GiftEval", config="ett1/H",
                source_revision=DATA_REV, source_sha256=DATA_SHA,
                item_id="item_0", channel=0, series_id="item_0_dim0", frequency="H",
                origin=origin, origin_timestamp=(START + timedelta(hours=origin)).isoformat(),
                context_start=origin - CONTEXT, context_length=CONTEXT,
                training_end=16412, validation_end=16460,
                horizon=HORIZON, seed=42 + index, model=model,
                family="classical" if model == MODELS[0] else "tsfm",
                compute_backend="cpu" if model == MODELS[0] else device,
                checkpoint=None if model == MODELS[0] else MODEL_REV,
                quantiles=TAUS, quantile_method="numpy.linear",
                settings=({"season_length": 24, "residual_lag": 24}
                          if model == MODELS[0] else
                          {"repository": MODEL_REPO, "patch_size": 32, "samples": 4096,
                           "batch_size": 1, "target_dim": 1, "covariates": 0}),
                implementation_sha256=digest(Path(__file__).read_bytes()),
                environment_sha256=digest((ROOT / "requirements" / "smoke.txt").read_bytes()),
            )
            job["job_id"] = digest(canonical(job).encode())
            jobs.append(job)
    return jobs


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".partial")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_series(data_dir):
    manifest = json.loads((ROOT / "artifacts" / "reference" / "manifests" / "dev_manifest.json").read_text())
    if not any(r["subset"] == "ett1_H" for r in manifest["active_dev_series"]):
        raise ValueError("ett1/H is not in the development manifest")
    path = Path(data_dir) / f"ett1-H-{DATA_REV}.arrow"
    if not path.exists():
        url = (f"https://huggingface.co/datasets/Salesforce/GiftEval/resolve/{DATA_REV}"
               "/ett1/H/data-00000-of-00001.arrow")
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        if digest(data) != DATA_SHA:
            raise ValueError("Downloaded dataset checksum differs from the pinned source")
        atomic_bytes(path, data)
    data = path.read_bytes()
    if digest(data) != DATA_SHA:
        raise ValueError("Local dataset checksum differs from the pinned source")
    table = ipc.open_stream(pa.BufferReader(data)).read_all()
    if len(table) != 1:
        raise ValueError("Expected exactly one source item")
    row = table.to_pylist()[0]
    targets = np.asarray(row["target"], dtype=np.float64)
    if (row["item_id"] != "item_0" or row["freq"] != "H" or row["start"] != START
            or targets.shape != (7, 17420)):
        raise ValueError("Source identity, timestamps, frequency or channel shape changed")
    series = targets[0]  # Keep the time axis intact; never flatten channels.
    if not np.isfinite(series[:16460]).all():
        raise ValueError("Smoke context/validation contains nonfinite observations")
    return series


def slice_window(series, job):
    series = np.asarray(series, dtype=np.float64)
    origin = job["origin"]
    if series.ndim != 1 or not (CONTEXT <= origin < origin + HORIZON <= len(series)):
        raise ValueError("Invalid one-dimensional time series or window bounds")
    if origin < job["training_end"] or origin + HORIZON > job["validation_end"]:
        raise ValueError("Forecast window is outside the approved validation block")
    context = series[origin - CONTEXT:origin].copy()
    actual = series[origin:origin + HORIZON].copy()
    if not np.isfinite(context).all() or not np.isfinite(actual).all():
        raise ValueError("Nonfinite context or actuals")
    return context, actual


def seasonal_quantiles(context):
    context = np.asarray(context, dtype=np.float64)
    if context.ndim != 1 or len(context) < 1000 or not np.isfinite(context).all():
        raise ValueError("SeasonalNaive needs at least 1000 finite training timestamps")
    residuals = context[24:] - context[:-24]
    return context[-24:, None] + np.quantile(residuals, TAUS, method="linear")[None, :]


def job_path(cache, job):
    return Path(cache) / f"model={job['model']}" / "dataset=ett1_H" / f"{job['job_id']}.parquet"


def validate_table(table, job):
    metadata = table.schema.metadata or {}
    if metadata.get(b"job") != canonical(job).encode():
        raise ValueError("Cache job configuration mismatch")
    if len(table) != HORIZON or any(table[c].null_count for c in table.column_names):
        raise ValueError("Cache must contain a complete, non-null forecast job")
    if table["step"].to_pylist() != list(range(1, HORIZON + 1)):
        raise ValueError("Missing, duplicate or unordered forecast steps")
    for col in ["job_id", "dataset", "config", "series_id", "model", "family", "origin", "horizon", "seed"]:
        if table[col].to_pylist() != [job[col]] * HORIZON:
            raise ValueError(f"Inconsistent cache column: {col}")
    expected_times = [(START + timedelta(hours=job["origin"] + i)).isoformat() for i in range(HORIZON)]
    if table["timestamp"].to_pylist() != expected_times:
        raise ValueError("Target timestamps are misaligned")
    actual = np.asarray(table["y_true"].to_pylist(), dtype=float)
    quantiles = np.column_stack([table[c].to_numpy() for c in QCOLS])
    if not np.isfinite(actual).all() or not np.isfinite(quantiles).all():
        raise ValueError("Nonfinite cache values")
    if (np.diff(quantiles, axis=1) < 0).any():
        raise ValueError("Crossing quantiles")
    return actual, quantiles


def write_job(cache, job, actual, quantiles, measurement):
    actual = np.asarray(actual, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)
    if actual.shape != (HORIZON,) or quantiles.shape != (HORIZON, len(TAUS)):
        raise ValueError("Unexpected actual/quantile shape")
    columns = {c: [job[c]] * HORIZON for c in
               ["job_id", "dataset", "config", "series_id", "model", "family", "origin", "horizon", "seed"]}
    columns.update(step=list(range(1, HORIZON + 1)), y_true=actual,
                   timestamp=[(START + timedelta(hours=job["origin"] + i)).isoformat() for i in range(HORIZON)])
    columns.update({col: quantiles[:, i] for i, col in enumerate(QCOLS)})
    table = pa.table(columns).replace_schema_metadata(
        {b"job": canonical(job).encode(), b"measurement": canonical(measurement).encode()})
    validate_table(table, job)
    path = job_path(cache, job)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, compression="zstd")
    atomic_bytes(path, sink.getvalue().to_pybytes())


def read_job(cache, job):
    table = pq.ParquetFile(job_path(cache, job)).read()
    actual, quantiles = validate_table(table, job)
    measurement = json.loads(table.schema.metadata[b"measurement"])
    return actual, quantiles, measurement


def load_moirai(device):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; explicitly choose --device cpu")
    # CUDA categorical sampling has no strict deterministic kernel in Torch 2.4.
    # Preserve seeds and record warnings; verify repeatability on the measured environment.
    torch.use_deterministic_algorithms(True, warn_only=device == "cuda")
    module = MoiraiModule.from_pretrained(MODEL_REPO, revision=MODEL_REV)
    model = MoiraiForecast(module=module, prediction_length=HORIZON, context_length=CONTEXT,
                          patch_size=32, num_samples=4096, target_dim=1,
                          feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0).eval()
    return model.create_predictor(batch_size=1, device=device)


def moirai_quantiles(predictor, context, job, device):
    import torch
    import pandas as pd
    from gluonts.dataset.common import ListDataset
    random.seed(job["seed"])
    np.random.seed(job["seed"])
    torch.manual_seed(job["seed"])
    if device == "cuda":
        torch.cuda.manual_seed_all(job["seed"])
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    dataset = ListDataset([dict(item_id=job["series_id"], target=context,
                               start=pd.Period(START + timedelta(hours=job["context_start"]), freq="h"))], freq="h")
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        forecast = next(iter(predictor.predict(dataset)))
    if forecast.start_date != pd.Period(job["origin_timestamp"], freq="h"):
        raise ValueError("Moirai forecast starts at the wrong timestamp")
    samples = np.asarray(forecast.samples)
    if samples.shape != (4096, HORIZON) or not np.isfinite(samples).all():
        raise ValueError(f"Invalid Moirai samples: {samples.shape}")
    quantiles = np.quantile(samples, TAUS, axis=0, method="linear").T
    if device == "cuda":
        torch.cuda.synchronize()
    return quantiles, dict(inference_sampling_seconds=time.perf_counter() - started,
                          device=torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
                          warnings=sorted({str(w.message) for w in caught}),
                          peak_gpu_memory_bytes=torch.cuda.max_memory_allocated() if device == "cuda" else 0)


def verify_environment():
    if sys.version_info[:2] != (3, 11):
        raise ValueError("Generation requires the pinned Python 3.11 environment")
    for line in (ROOT / "requirements" / "smoke.txt").read_text().splitlines():
        if line and not line.startswith(("#", " ")):
            name, version = line.split("==")
            if importlib.metadata.version(name) != version:
                raise ValueError(f"Environment differs from requirements/smoke.txt: {name}")


@contextmanager
def generation_lock(cache):
    Path(cache).mkdir(parents=True, exist_ok=True)
    # ponytail: one writer for the smoke cache; partitioned workers only if the full grid needs them.
    with (Path(cache) / ".writer.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def generate(args):
    started = time.perf_counter()
    run = dict(started_at=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
               platform=platform.platform(), jobs=[], model_load_seconds=0.0, data_load_seconds=0.0)
    run["packages"] = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    series = predictor = None
    with generation_lock(args.cache):
        for job in planned_jobs(args.device):
            if job_path(args.cache, job).exists():
                read_job(args.cache, job)
                run["jobs"].append(dict(job_id=job["job_id"], status="skipped", rows=HORIZON))
                continue
            if args.limit_jobs is not None and sum(j["status"] == "generated" for j in run["jobs"]) >= args.limit_jobs:
                break
            if series is None:
                verify_environment()
                t = time.perf_counter()
                series = load_series(args.data)
                run["data_load_seconds"] = time.perf_counter() - t
            context, actual = slice_window(series, job)
            if job["model"] == MODELS[0]:
                t = time.perf_counter()
                quantiles = seasonal_quantiles(context)
                measurement = dict(inference_sampling_seconds=time.perf_counter() - t,
                                   device="cpu", peak_gpu_memory_bytes=0)
            else:
                if predictor is None:
                    t = time.perf_counter()
                    predictor = load_moirai(args.device)
                    run["model_load_seconds"] = time.perf_counter() - t
                quantiles, measurement = moirai_quantiles(predictor, context, job, args.device)
            t = time.perf_counter()
            write_job(args.cache, job, actual, quantiles, measurement)
            write_seconds = time.perf_counter() - t
            cached_actual, cached_quantiles, _ = read_job(args.cache, job)
            np.testing.assert_array_equal(cached_actual, actual)
            np.testing.assert_array_equal(cached_quantiles, quantiles)
            entry = dict(job_id=job["job_id"], status="generated", rows=HORIZON,
                         cache_write_seconds=write_seconds, **measurement)
            run["jobs"].append(entry)
            print(canonical(entry), flush=True)
        run["wall_seconds"] = time.perf_counter() - started
        atomic_bytes(Path(args.cache) / "runs" / f"{time.time_ns()}.json", canonical(run).encode())
    return run


def analyze(cache, device="cuda"):
    by_model = {m: [] for m in MODELS}
    actual_by_origin = {}
    timings = {m: [] for m in MODELS}
    for job in planned_jobs(device):
        actual, quantiles, measurement = read_job(cache, job)
        if job["origin"] in actual_by_origin:
            np.testing.assert_array_equal(actual, actual_by_origin[job["origin"]])
        actual_by_origin[job["origin"]] = actual
        error = actual[:, None] - quantiles
        loss = np.maximum(np.asarray(TAUS) * error, (np.asarray(TAUS) - 1) * error)
        np.testing.assert_allclose(loss[:, 5].mean(), np.abs(error[:, 5]).mean() / 2)
        by_model[job["model"]].append(loss)
        timings[job["model"]].append(measurement)
    costs = {m: np.concatenate(v).mean(axis=0) for m, v in by_model.items()}
    oracle = np.minimum(costs[MODELS[0]], costs[MODELS[1]])
    return dict(rows=4 * HORIZON, experimental_unit="one channel, two consecutive validation windows",
                oracle="better fixed representative over both windows, separately at each tau",
                costs=[dict(tau=t, model=m, mean_pinball=float(costs[m][i]),
                            representative_regret=float(costs[m][i] - oracle[i]))
                       for i, t in enumerate(TAUS) for m in MODELS],
                measurements=timings, models_imported="torch" in sys.modules)


def dry_run(cache, project_jobs, device="cuda"):
    jobs = planned_jobs(device)
    projection = {}
    for model in MODELS:
        measurements = [read_job(cache, j)[2] for j in jobs
                        if j["model"] == model and job_path(cache, j).exists()]
        projection[model] = None if not measurements else dict(
            jobs_per_model=project_jobs, horizon=HORIZON, context=CONTEXT,
            estimated_inference_sampling_hours=project_jobs * np.mean(
                [m["inference_sampling_seconds"] for m in measurements]) / 3600,
            measured_jobs=len(measurements), devices=sorted({m["device"] for m in measurements}))
    return dict(planned_jobs=len(jobs), planned_rows=len(jobs) * HORIZON,
                missing_jobs=sum(not job_path(cache, j).exists() for j in jobs),
                projection=projection,
                limitation="Same two models, hardware, context and h=24 only; excludes model loading, I/O and unmeasured roster/horizons. Not a full-grid budget confirmation.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["generate", "analyze", "dry-run"])
    parser.add_argument("--cache", type=Path, default=ROOT / "artifacts" / "runs" / "cache" / "forecast-smoke")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "forecast-smoke")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--limit-jobs", type=int, help="Stop after this many new complete jobs; exercise resume")
    parser.add_argument("--project-jobs", type=int, default=1000, help="Projection scenario: jobs per measured model")
    args = parser.parse_args()
    if args.project_jobs < 1 or (args.limit_jobs is not None and args.limit_jobs < 1):
        parser.error("Job counts must be positive")
    result = (generate(args) if args.command == "generate" else
              analyze(args.cache, args.device) if args.command == "analyze" else dry_run(args.cache, args.project_jobs, args.device))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
