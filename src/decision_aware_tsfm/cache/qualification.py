"""Synthetic, reproducible qualification of the four issue-20 pilot adapters."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
from typing import Callable

import numpy as np

from .schema import TAUS

MODEL_NAMES = ("autoarima_native", "autoets_simulated", "seasonal_naive_native", "moirai_pilot")
HORIZONS = (1, 24, 168)
SEEDS = (42, 43)
START = "2000-01-01T00:00:00"
ORIGIN = (datetime.fromisoformat(START) + timedelta(hours=2048)).isoformat()
GPU_REPEAT_TOLERANCE = 1e-6


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_json(path: Path, value) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _save_array(path: Path, value: np.ndarray) -> str:
    with path.open("xb") as stream:
        np.save(stream, value, allow_pickle=False)
    return _hash(path.read_bytes())


def _sources() -> tuple[dict, dict[str, np.ndarray]]:
    step = np.arange(2048, dtype=np.float64)
    rng = np.random.default_rng(20260920)
    values = {
        "positive_seasonal_trend_noise": (
            20.0 + 0.01 * step + 3.0 * np.sin(2 * np.pi * step / 24)
            + 1.5 * np.cos(2 * np.pi * step / 168) + rng.normal(0.0, 0.2, step.size)
        ),
        "constant_nonpositive": np.full(2048, -3.0, dtype=np.float64),
    }
    definitions = {
        "positive_seasonal_trend_noise": {
            "formula": "20 + 0.01*t + 3*sin(2*pi*t/24) + 1.5*cos(2*pi*t/168) + Normal(0,0.2)",
            "noise_generator": "numpy.default_rng(PCG64)", "noise_seed": 20260920,
        },
        "constant_nonpositive": {"formula": "-3.0 for every timestamp", "noise_seed": None},
    }
    return definitions, values


def _environment() -> dict:
    packages = dict(sorted((dist.metadata["Name"], dist.version)
                           for dist in importlib.metadata.distributions()))
    hardware = {
        "platform": platform.platform(), "machine": platform.machine(),
        "processor": platform.processor(), "cpu_count": os.cpu_count(),
    }
    torch = sys.modules.get("torch")
    if torch is not None:
        hardware["cuda_available"] = torch.cuda.is_available()
        hardware["cuda_devices"] = ([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
                                    if torch.cuda.is_available() else [])
    else:
        hardware.update(cuda_available=None, cuda_devices=[], cuda_probe="torch not loaded")
    runtime = {name: os.environ.get(name) for name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "CUBLAS_WORKSPACE_CONFIG", "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE")}
    return {"python": sys.version, "packages": packages, "hardware": hardware,
            "runtime_environment": runtime}


def _provenance() -> dict:
    root = Path(__file__).resolve().parents[3]
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip() or None
    files = [Path(__file__), Path(__file__).with_name("pilot.py"),
             Path(__file__).with_name("adapters.py"), Path(__file__).with_name("schema.py"),
             root / "requirements/qualification.txt"]
    code = {str(path.relative_to(root)): _hash(path.read_bytes()) for path in files if path.exists()}
    environment = _environment()
    return {"git_commit": commit, "code_sha256": code,
            "combined_code_sha256": _hash(_canonical(code).encode()),
            "environment": environment, "environment_sha256": _hash(_canonical(environment).encode())}


def _job(model: dict, horizon: int, seed: int) -> dict:
    return {
        "fixture": True, "frequency": "H", "context_length": 2048,
        "horizon": horizon, "seed": seed, "start_timestamp": START,
        "context_start_timestamp": START, "origin_timestamp": ORIGIN,
        **model,
    }


def _validate(context: np.ndarray, before: np.ndarray, forecast, horizon: int) -> np.ndarray:
    output = np.asarray(forecast, dtype=np.float64)
    if output.shape != (horizon, len(TAUS)):
        raise ValueError(f"forecast shape {output.shape}, expected {(horizon, len(TAUS))}")
    if not np.isfinite(output).all():
        raise ValueError("forecast contains nonfinite values")
    if (np.diff(output, axis=1) < 0).any():
        raise ValueError("forecast quantiles are not ordered")
    if not np.array_equal(context, before, equal_nan=True):
        raise ValueError("adapter mutated its input context")
    return output


def _validate_measurement(measurement) -> dict:
    if not isinstance(measurement, dict):
        raise ValueError("adapter measurement must be a mapping")
    seconds = measurement.get("fitting_seconds")
    peak = measurement.get("peak_memory_bytes")
    if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
            or not np.isfinite(seconds) or seconds < 0):
        raise ValueError("adapter measurement requires finite fitting_seconds >= 0")
    if not isinstance(measurement.get("fitted_model"), dict):
        raise ValueError("adapter measurement requires fitted_model mapping")
    for name in ("hardware_platform", "device"):
        if not isinstance(measurement.get(name), str) or not measurement[name]:
            raise ValueError(f"adapter measurement requires nonempty {name}")
    if isinstance(peak, bool) or not isinstance(peak, int) or peak < 0:
        raise ValueError("adapter measurement requires integer peak_memory_bytes >= 0")
    if not isinstance(measurement.get("warnings"), list):
        raise ValueError("adapter measurement requires warnings list")
    return measurement


def _persist(report: dict, path: Path, status: str) -> None:
    report["status"] = status
    report["counts"] = {"attempted": len(report["attempts"]),
                        "passed": sum(a["status"] == "passed" for a in report["attempts"]),
                        "failed": sum(a["status"] == "failed" for a in report["attempts"])}
    _atomic_json(path, report)


def qualify(output_dir: Path, *, device="cuda", models=None, horizons=HORIZONS,
            adapter_loader: Callable | None = None) -> dict:
    """Run selected qualification cases and always leave an atomic final report."""
    output_dir = Path(output_dir)
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be 'cpu' or 'cuda'")
    if device == "cuda":
        cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if cublas not in (None, ":4096:8"):
            raise ValueError("CUDA qualification requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    from .pilot import pilot_models
    canonical_models = {model["model"]: model for model in pilot_models(device=device)}
    if models is None:
        models = list(canonical_models.values())
    test_harness = adapter_loader is not None
    if adapter_loader is None:
        from .adapters import load_adapter
        adapter_loader = load_adapter
    configured = {model["model"]: model for model in models}
    unknown = sorted(set(configured) - set(MODEL_NAMES))
    if not configured or unknown or len(configured) != len(models):
        raise ValueError(f"models must be a unique, nonempty subset of {MODEL_NAMES}; unknown={unknown}")
    mismatched = [name for name, model in configured.items() if model != canonical_models[name]]
    if mismatched:
        raise ValueError(f"models do not match canonical pilot configurations: {sorted(mismatched)}")
    selected_horizons = tuple(dict.fromkeys(horizons))
    if not selected_horizons or set(selected_horizons) - set(HORIZONS):
        raise ValueError(f"horizons must be selected from {HORIZONS}")

    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "sources").mkdir()
    (output_dir / "forecasts").mkdir()
    report_path = output_dir / "report.json"
    started = time.time()
    definitions, sources = _sources()
    source_records = {}
    for name, context in sources.items():
        path = output_dir / "sources" / f"{name}.npy"
        source_records[name] = {**definitions[name], "path": str(path.relative_to(output_dir)),
                                "sha256": _save_array(path, context), "shape": list(context.shape),
                                "dtype": str(context.dtype)}

    report = {
        "schema_version": 1,
        "evidence_kind": "test_harness" if test_harness else "synthetic_adapter_qualification",
        "timing_scope": "synthetic qualification only; not a production compute budget",
        "status": "running", "started_unix": started, "device_requested": device,
        "required_coverage": {"models": list(MODEL_NAMES), "horizons": list(HORIZONS),
                              "seeds": list(SEEDS), "contexts": list(sources)},
        "selected_coverage": {"models": list(configured), "horizons": list(selected_horizons)},
        "model_configs": list(configured.values()),
        "fixture_config": {"context_length": 2048, "frequency": "H", "start_timestamp": START,
                           "context_start_timestamp": START, "origin_timestamp": ORIGIN},
        "quantiles": list(TAUS), "sources": source_records, "provenance": _provenance(),
        "attempts": [], "failures": [],
    }
    expected_attempts = len(configured) * len(selected_horizons) * len(sources) * 3
    fatal = None
    _persist(report, report_path, "running")
    try:
        for model_name, model in configured.items():
            for horizon in selected_horizons:
                adapter = None
                load_seconds = None
                load_error = None
                try:
                    load_started = time.perf_counter()
                    adapter = adapter_loader(_job(model, horizon, 42))
                    load_seconds = time.perf_counter() - load_started
                except BaseException as error:
                    load_seconds = time.perf_counter() - load_started
                    load_error = f"{type(error).__name__}: {error}"
                    if not isinstance(error, Exception):
                        report["failures"].append({"id": f"{model_name}__h{horizon}__load",
                                                   "error": load_error,
                                                   "adapter_load_seconds": load_seconds})
                        raise
                report["execution_environment"] = _environment()
                _persist(report, report_path, "running")
                for source_name, context in sources.items():
                    for seed in SEEDS:
                        repeats = 2 if seed == 42 else 1
                        baseline = None
                        for repeat in range(repeats):
                            identity = f"{model_name}__h{horizon}__{source_name}__s{seed}__r{repeat}"
                            record = {"id": identity, "model": model_name, "horizon": horizon,
                                      "context": source_name, "seed": seed, "repeat": repeat,
                                      "adapter_load_seconds": load_seconds if repeat == 0 and seed == 42 and source_name == next(iter(sources)) else 0.0}
                            predict_started = time.perf_counter()
                            try:
                                if load_error:
                                    raise RuntimeError(f"adapter load failed: {load_error}")
                                working = context.copy()
                                before = working.copy()
                                forecast, measurement = adapter.predict(working, _job(model, horizon, seed))
                                predict_seconds = time.perf_counter() - predict_started
                                output = _validate(working, before, forecast, horizon)
                                measurement = _validate_measurement(measurement)
                                path = output_dir / "forecasts" / f"{identity}.npy"
                                digest = _save_array(path, output)
                                record.update(status="passed", predict_wall_seconds=predict_seconds,
                                              forecast_path=str(path.relative_to(output_dir)),
                                              forecast_sha256=digest, adapter_measurement=measurement)
                                if repeat == 0:
                                    baseline = output
                                else:
                                    delta = float(np.max(np.abs(output - baseline)))
                                    tolerance = GPU_REPEAT_TOLERANCE if model.get("device") == "cuda" else 0.0
                                    record["repeatability"] = {"max_abs_delta": delta,
                                        "tolerance": tolerance, "exact": bool(np.array_equal(output, baseline)),
                                        "gpu_determinism_warning": ("Seeded GPU operations may remain nondeterministic; observed delta is recorded."
                                                                    if model.get("device") == "cuda" else None)}
                                    if delta > tolerance:
                                        raise ValueError(f"repeatability delta {delta} exceeds tolerance {tolerance}")
                            except BaseException as error:
                                record.update(status="failed", error=f"{type(error).__name__}: {error}",
                                              elapsed_seconds=time.perf_counter() - predict_started)
                                report["failures"].append({"id": identity, "error": record["error"]})
                                if not isinstance(error, Exception):
                                    report["attempts"].append(record)
                                    _persist(report, report_path, "failed")
                                    raise
                            report["attempts"].append(record)
                            _persist(report, report_path, "running")
    except BaseException as error:
        fatal = error
        if not report["failures"]:
            report["failures"].append({"id": "runner", "error": f"{type(error).__name__}: {error}"})
    finally:
        full = set(configured) == set(MODEL_NAMES) and set(selected_horizons) == set(HORIZONS)
        report["finished_unix"] = time.time()
        report["wall_seconds"] = report["finished_unix"] - started
        report["execution_hardware"] = _environment()["hardware"]
        complete = len(report["attempts"]) == expected_attempts
        status = ("failed" if report["failures"] or not complete else
                  "test_only" if full and test_harness else "passed" if full else "partial")
        _persist(report, report_path, status)
    if fatal is not None:
        raise fatal
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES)
    parser.add_argument("--horizons", nargs="+", type=int, choices=HORIZONS, default=list(HORIZONS))
    args = parser.parse_args(argv)
    try:
        from .pilot import pilot_models
        models = pilot_models(device=args.device)
        if args.models:
            models = [model for model in models if model["model"] in args.models]
        report = qualify(args.output_dir, device=args.device, models=models, horizons=args.horizons)
        counts = report["counts"]
        print(f"status={report['status']} attempted={counts['attempted']} passed={counts['passed']} "
              f"failed={counts['failed']} report={args.output_dir / 'report.json'}")
        return 0 if report["status"] == "passed" else 2
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"pilot-qualification: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
