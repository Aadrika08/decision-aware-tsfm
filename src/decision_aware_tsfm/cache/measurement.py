"""Measure cache generation on Linux without changing the planned experiment."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import time

from .cli import atomic_output, read_json
from .executor import _planned_jobs


def _stop(process, sig=signal.SIGTERM):
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def _gpu_summary(path):
    """Keep sampled device-wide memory separate from process and Torch memory."""
    devices = {}
    with path.open(newline="") as stream:
        for row in csv.reader(stream, skipinitialspace=True):
            if len(row) != 5:
                raise ValueError("Malformed nvidia-smi sample")
            timestamp, uuid, memory, utilization, total = row
            values = [float(value) for value in (memory, utilization, total)]
            if not all(value >= 0 and value < float("inf") for value in values):
                raise ValueError("Invalid nvidia-smi measurement")
            memory, utilization, total = values
            if utilization > 100 or memory > total:
                raise ValueError("Invalid nvidia-smi measurement bounds")
            device = devices.setdefault(uuid, {"sample_count": 0, "max_used_mib": 0,
                "max_utilization_percent": 0, "total_mib": total, "first_sample": timestamp})
            device.update(sample_count=device["sample_count"] + 1,
                max_used_mib=max(device["max_used_mib"], memory),
                max_utilization_percent=max(device["max_utilization_percent"], utilization),
                last_sample=timestamp)
    if not devices:
        raise ValueError("No nvidia-smi samples captured")
    return devices


def measure(plan_file, cache, output_directory, *, real=False, workers=1):
    """Persist logs and measurements; never treat fixture measurements as real evidence."""
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if sys.platform != "linux":
        raise ValueError("Pilot resource measurement requires Linux")
    plan_file, cache, output = (Path(path).resolve() for path in
                              (plan_file, cache, output_directory))
    planned = read_json(plan_file)
    jobs = _planned_jobs(planned, generate=True)
    gpu_required = any(job["device"] == "cuda" for job in jobs)
    if gpu_required and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (None, ":4096:8"):
        raise ValueError("CUDA measurement requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
    if not isinstance(planned.get("fixture"), bool):
        raise ValueError("Plan must explicitly identify fixture status")
    if not planned["fixture"]:
        if not real:
            raise ValueError("Real measurement requires --real")
        if any(not isinstance(job.get(field), str) or not job[field].strip()
               for job in jobs for field in ("approval_reference", "model_approval_reference")):
            raise ValueError("Real measurement requires workload and model approval references")
    protected = [plan_file, *(Path(job["source_path"]).resolve() for job in jobs)]
    if (output.is_relative_to(cache) or cache.is_relative_to(output)
            or any(path.is_relative_to(output) or path.is_relative_to(cache) for path in protected)):
        raise ValueError("Measurement output, cache and input files must be separate")
    output.mkdir(parents=True, exist_ok=False)
    encoded = json.dumps(planned, indent=2, allow_nan=False) + "\n"
    atomic_output(output / "plan.json", encoded, refuse_existing=True)
    root = Path(__file__).resolve().parents[3]
    environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", LC_ALL="C", PYTHONDONTWRITEBYTECODE="1",
                       PYTHONPATH=str(root / "src"))
    if gpu_required:
        environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    command = ["/usr/bin/time", "-v", "-o", str(output / "time.txt"), sys.executable,
        str(root / "scripts/forecast_cache.py"), "generate", "--plan-file", str(output / "plan.json"),
        "--cache", str(cache), "--workers", str(workers), "--output", str(output / "generation.json")]
    if real:
        command.append("--real")
    report = {"status": "incomplete", "evidence_kind": "fixture" if planned["fixture"] else "real_pilot",
        "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "generation_exit_code": None, "generation_status": "not_started", "command": command,
        "cpu_workers": workers,
        "cwd": str(root), "plan_file": str(plan_file),
        "plan_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
        "environment": {key: environment[key] for key in (
            "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "LC_ALL", "PYTHONPATH",
            "PYTHONDONTWRITEBYTECODE", "CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG",
            "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE") if key in environment},
        "hardware": {"platform": platform.platform(), "logical_cpus": os.cpu_count()},
        "started_at": datetime.now(timezone.utc).isoformat(), "errors": [],
        "gpu_sampling_interval_seconds": 1, "maximum_process_rss_kib": None,
        "gpu_device_samples": {},
        "limitations": ["Device memory/utilization includes other processes and is sampled, not an exact peak",
            "Maximum process RSS is not summed process-tree memory or PyTorch allocation",
            "Per-job PyTorch allocation and timings remain in generation.json",
            "Fixture measurements do not establish a real compute budget"]}
    generation = sampler = None
    started = time.perf_counter()
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        cpu_text = Path("/proc/cpuinfo").read_text()
        cpu_models = sorted(set(re.findall(r"^model name\s*:\s*(.+)$", cpu_text, re.MULTILINE)))
        if not cpu_models:
            raise ValueError("Cannot identify CPU model from /proc/cpuinfo")
        report["hardware"]["cpu_models"] = cpu_models
        if not os.access("/usr/bin/time", os.X_OK):
            raise ValueError("Required measurement tool /usr/bin/time is missing")
        nvidia = shutil.which("nvidia-smi")
        with (output / "gpu.csv").open("w") as gpu_log, (output / "gpu.stderr.log").open("w") as gpu_errors:
            if nvidia:
                identity_command = [nvidia, "--query-gpu=uuid,name,driver_version,memory.total",
                                    "--format=csv,noheader,nounits"]
                try:
                    identity = subprocess.run(identity_command, text=True, capture_output=True,
                                              timeout=10, env=environment)
                    report["hardware"]["gpu_probe"] = {"command": identity_command,
                        "exit_code": identity.returncode, "stdout": identity.stdout, "stderr": identity.stderr}
                    if identity.returncode or not identity.stdout.strip():
                        raise ValueError("nvidia-smi identity probe failed")
                    devices = list(csv.reader(identity.stdout.splitlines(), skipinitialspace=True))
                    if gpu_required:
                        if len(devices) != 1 or len(devices[0]) != 4:
                            raise ValueError("GPU measurement requires exactly one physical CUDA device")
                        gpu_uuid = devices[0][0]
                        if environment.get("CUDA_VISIBLE_DEVICES", "0") not in ("0", gpu_uuid):
                            raise ValueError("CUDA_VISIBLE_DEVICES does not select the measured GPU")
                        environment["CUDA_VISIBLE_DEVICES"] = gpu_uuid
                        report["environment"]["CUDA_VISIBLE_DEVICES"] = gpu_uuid
                        report["hardware"]["measured_gpu_uuid"] = gpu_uuid
                    sample_command = [nvidia,
                        "--query-gpu=timestamp,uuid,memory.used,utilization.gpu,memory.total",
                        "--format=csv,noheader,nounits", "--loop=1"]
                    report["gpu_sampling_command"] = sample_command
                    sampler = subprocess.Popen(sample_command, stdout=gpu_log, stderr=gpu_errors,
                                               env=environment, start_new_session=True)
                except (OSError, ValueError, subprocess.TimeoutExpired) as error:
                    report["errors"].append(f"GPU measurement: {error}")
                    if gpu_required and isinstance(error, ValueError):
                        raise
            else:
                report["errors"].append("GPU measurement: nvidia-smi is unavailable")
            with (output / "stdout.log").open("w") as stdout, (output / "stderr.log").open("w") as stderr:
                generation = subprocess.Popen(command, stdout=stdout, stderr=stderr, cwd=root,
                                              env=environment, start_new_session=True)
                report["generation_status"] = "running"
                report["generation_exit_code"] = generation.wait()
            if sampler is not None and sampler.poll() is not None:
                report["errors"].append(f"GPU sampler exited prematurely: {sampler.returncode}")
            _stop(sampler)
        report["generation_status"] = "failed"
        if (output / "generation.json").exists():
            report["generation_status"] = read_json(output / "generation.json").get("status", "unknown")
        raw_time = (output / "time.txt").read_text()
        rss = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", raw_time)
        if rss and int(rss.group(1)) > 0:
            report["maximum_process_rss_kib"] = int(rss.group(1))
        else:
            report["errors"].append("Maximum process RSS missing or invalid in time.txt")
        if sampler is not None:
            try:
                report["gpu_device_samples"] = _gpu_summary(output / "gpu.csv")
            except ValueError as error:
                report["errors"].append(f"GPU measurement: {error}")
        complete = (report["maximum_process_rss_kib"] is not None
                    and (not gpu_required or (report["gpu_device_samples"] and not report["errors"])))
        if report["generation_exit_code"] == 0 and report["generation_status"] == "complete":
            report["measurement_status"] = "complete" if complete else "incomplete"
            report["status"] = "complete" if complete else "incomplete"
        else:
            report["status"] = "failed"
    except BaseException as error:
        report["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        if report["generation_status"] == "running":
            report["generation_status"] = report["status"]
        report["errors"].append(f"{type(error).__name__}: {error}")
        raise
    finally:
        _stop(generation, signal.SIGINT)
        _stop(sampler)
        if generation is not None:
            report["generation_exit_code"] = generation.returncode
        report["wall_seconds"] = time.perf_counter() - started
        report.setdefault("measurement_status", "incomplete")
        try:
            atomic_output(output / "report.json", json.dumps(report, indent=2, allow_nan=False) + "\n")
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-file", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--workers", type=int, default=1, help="Bounded CPU worker count")
    args = parser.parse_args(argv)
    try:
        result = measure(args.plan_file, args.cache, args.output_directory,
                         real=args.real, workers=args.workers)
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"measure-pilot: {error}", file=sys.stderr)
        return 2
    print(json.dumps({key: result[key] for key in ("status", "evidence_kind", "generation_exit_code")}))
    return 0 if result["status"] == "complete" else 2
