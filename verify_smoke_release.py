"""Independently verify the issue #3 release, without model inference.

Proposed sub-issue under #3: verify the published cache handoff and quantify
the remaining full-grid budget inputs. This reads tag blobs, not working-tree
source files, so Windows line endings cannot change the recorded identities.
"""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
from pathlib import Path
import platform
import subprocess
import sys
import zipfile

import numpy as np
import pyarrow
import pyarrow.parquet as pq

TAG = "v0.1.0-smoke"
COMMIT = "457dd1d9bd31db5cb539d5778d1ad49a71bc244f"
ZIP_SHA = "9d2e50d43b7898d88ecbfc70cd3642f0b142eb1155bd1574bf1974c9229a6876"
ROOT = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def blob(path):
    return subprocess.check_output(["git", "show", f"{COMMIT}:{path}"], cwd=ROOT)


def verify(archive):
    raw = Path(archive).read_bytes()
    require(sha(raw) == ZIP_SHA, "Release archive SHA-256 mismatch")
    report = json.loads(blob("docs/forecast-smoke-results.json"))
    implementation = sha(blob("forecast_smoke.py"))
    environment = sha(blob("requirements-smoke.txt"))
    losses = {}
    actuals = {}
    measurements = {}
    verified_jobs = []
    taus = np.array([.01, .05, .09, .25, .40, .50, .60, .75, .91, .95, .99])
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        require(z.testzip() is None, "ZIP CRC failure")
        names = z.namelist()
        require(len(names) == len(set(names)), "Duplicate archive members")
        expected_paths = set()
        for job in report["jobs"]:
            identity = dict(job)
            job_id = identity.pop("job_id")
            canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)
            require(sha(canonical.encode()) == job_id, "Invalid content-addressed job ID")
            require(job["implementation_sha256"] == implementation, "Implementation hash mismatch")
            require(job["environment_sha256"] == environment, "Environment hash mismatch")
            require(job["config"] == "ett1/H" and job["channel"] == 0, "Unexpected dataset/channel")
            require(job["horizon"] == 24 and job["context_length"] == 2048, "Unexpected horizon/context")
            require(job["quantiles"] == taus.tolist(), "Quantile grid mismatch")
            path = f"cache/forecast-smoke/model={job['model']}/dataset=ett1_H/{job_id}.parquet"
            expected_paths.add(path)
            table = pq.ParquetFile(io.BytesIO(z.read(path))).read()
            metadata = table.schema.metadata
            require(json.loads(metadata[b"job"]) == job, "Parquet job metadata mismatch")
            require(len(table) == 24, "Incomplete job")
            require(all(table[c].null_count == 0 for c in table.column_names), "Null cache values")
            for col in ["job_id", "dataset", "config", "series_id", "model", "family", "origin", "horizon", "seed"]:
                require(table[col].to_pylist() == [job[col]] * 24, f"Identity mismatch: {col}")
            require(table["step"].to_pylist() == list(range(1, 25)), "Missing/duplicate/unordered steps")
            origin = datetime.fromisoformat(job["origin_timestamp"])
            times = [(origin + timedelta(hours=i)).isoformat() for i in range(24)]
            require(table["timestamp"].to_pylist() == times, "Timestamp mismatch")
            y = table["y_true"].to_numpy()
            q = np.column_stack([table[f"q_{t:.2f}"].to_numpy() for t in taus])
            require(np.isfinite(y).all() and np.isfinite(q).all(), "Nonfinite predictions/actuals")
            require((np.diff(q, axis=1) >= 0).all(), "Crossing quantiles")
            if job["origin"] in actuals:
                np.testing.assert_array_equal(y, actuals[job["origin"]])
            actuals[job["origin"]] = y
            # Independent branch-form pinball calculation.
            residual = y[:, None] - q
            loss = np.where(residual >= 0, taus * residual, (1 - taus) * -residual)
            np.testing.assert_allclose(loss[:, 5], .5 * np.abs(residual[:, 5]), rtol=0, atol=1e-12)
            losses.setdefault(job["model"], []).append(loss)
            measurement = json.loads(metadata[b"measurement"])
            require(measurement["inference_sampling_seconds"] > 0, "Invalid inference timing")
            measurements.setdefault(job["model"], []).append(measurement)
            verified_jobs.append(job_id)
        require({n for n in names if n.endswith(".parquet")} == expected_paths, "Unexpected/missing cache jobs")
        ledgers = [json.loads(z.read(n)) for n in names if n.endswith(".json")]
        generated = {}
        unrelated_ledger_jobs = []
        completed_reruns = 0
        for ledger in ledgers:
            entries = ledger["jobs"]
            for entry in entries:
                if entry["job_id"] not in verified_jobs:
                    unrelated_ledger_jobs.append(entry["job_id"])
                    continue
                if entry["status"] == "generated":
                    generated.setdefault(entry["job_id"], []).append(entry)
            if ({e["job_id"] for e in entries} == set(verified_jobs)
                    and len(entries) == 4 and all(e["status"] == "skipped" for e in entries)):
                require(ledger["model_load_seconds"] == 0 and ledger["data_load_seconds"] == 0,
                        "Completed rerun performed loading")
                completed_reruns += 1
        require(set(generated) == set(verified_jobs), "Missing generation provenance")
        require(completed_reruns > 0, "Missing completed-rerun ledger")
        for job in report["jobs"]:
            path = f"cache/forecast-smoke/model={job['model']}/dataset=ett1_H/{job['job_id']}.parquet"
            measurement = json.loads(pq.ParquetFile(io.BytesIO(z.read(path))).schema_arrow.metadata[b"measurement"])
            require(any(all(entry.get(k) == v for k, v in measurement.items())
                        for entry in generated[job["job_id"]]), "Measurement/ledger mismatch")
    costs = {model: np.concatenate(parts).mean(axis=0) for model, parts in losses.items()}
    oracle = np.min(list(costs.values()), axis=0)
    recomputed = []
    for row in report["costs"]:
        index = taus.tolist().index(row["tau"])
        cost = float(costs[row["model"]][index])
        regret = float(cost - oracle[index])
        np.testing.assert_allclose([cost, regret], [row["mean_pinball"], row["representative_regret"]], rtol=0, atol=1e-12)
        recomputed.append(dict(tau=row["tau"], model=row["model"], mean_pinball=cost, representative_regret=regret))
    scenarios = {}
    for model, entries in measurements.items():
        seconds = [m["inference_sampling_seconds"] for m in entries]
        hours = float(np.mean(seconds) * 1000 / 3600)
        recorded = report["projection"]["projection"][model]["estimated_inference_sampling_hours"]
        np.testing.assert_allclose(hours, recorded, rtol=0, atol=1e-12)
        scenarios[model] = dict(measured_jobs=len(seconds), seconds_per_job=seconds,
                               mean_seconds_per_job=float(np.mean(seconds)),
                               inference_hours_for_1000_identical_jobs=hours,
                               devices=sorted({m["device"] for m in entries}))
    return dict(verified_at=datetime.now(timezone.utc).isoformat(), tag=TAG, commit=COMMIT,
                archive_sha256=sha(raw), python=platform.python_version(), platform=platform.platform(),
                numpy=np.__version__, pyarrow=pyarrow.__version__, verified_jobs=verified_jobs,
                rows=96, quantiles=11, run_ledgers=len(ledgers), completed_rerun_ledgers=completed_reruns,
                unrelated_ledger_jobs=unrelated_ledger_jobs,
                costs=recomputed, measured_scenarios=scenarios, torch_imported="torch" in sys.modules,
                full_grid_budget_status="blocked_missing_frozen_workload_and_other_model_horizon_timings",
                limitations=["Historical ledgers were checked; inference and resume were not rerun.",
                             "The separate uninterrupted cache is not in this release; exact cross-run equality cannot be independently rechecked from this ZIP.",
                             "Scenarios apply only to the recorded two models, h=24, context=2048 and hardware; exclude loading and I/O."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.archive)
    rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
