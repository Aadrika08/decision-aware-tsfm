import hashlib
import errno
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

import numpy as np
import pyarrow.parquet as pq

from decision_aware_tsfm.cache import executor
from decision_aware_tsfm.cache.executor import audit, generate, job_path, load_source, read_job
from decision_aware_tsfm.cache.planner import plan
from decision_aware_tsfm.cache.schema import TAUS


class ExecutorTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.cache = self.root / "cache"
        self.source = self.root / "source.npy"
        np.save(self.source, np.arange(1100.0))
        executor._SOURCES.clear()
        executor._WINDOWS.clear()
        executor._ADAPTERS.clear()

    def tearDown(self):
        executor._SOURCES.clear()
        executor._WINDOWS.clear()
        executor._ADAPTERS.clear()
        shutil.rmtree(self.root)

    def make_plan(self, rows=1, models=1, prediction=7.5):
        digest = hashlib.sha256(self.source.read_bytes()).hexdigest()
        workload_rows = []
        for index in range(rows):
            origin = 1000 + index
            workload_rows.append(dict(
                source_revision="fixture-v1", source_sha256=digest,
                source_path=self.source.name, dataset="synthetic_fixture", config="hourly",
                series_id="series-0", item_id="item-0", channel=0, frequency="H",
                start_timestamp="2020-01-01T00:00:00", source_length=1100,
                usable_start=0, usable_end=1100, context_start=index,
                context_length=1000, origin=origin, horizon=2, seed=42 + index,
                training_end=origin, validation_end=origin + 2))
        model_rows = []
        for index in range(models):
            family = "classical" if index == 0 else "tsfm"
            model_rows.append(dict(model=f"fixture_{family}_{index}", family=family,
                                   adapter="fixture", adapter_version="1", checkpoint=None,
                                   device="cpu", batch_size=1, quantile_method="constant",
                                   settings={"prediction": prediction + index}))
        workload = dict(schema_version=1, split="development", fixture=True,
                        quantiles=list(TAUS), rows=workload_rows)
        return plan(workload, {"schema_version": 1, "models": model_rows}, base_dir=self.root)

    def test_planted_roundtrip_and_complete_resume_do_not_load(self):
        planned = self.make_plan(models=2)
        first = generate(planned, self.cache)
        self.assertEqual(first["status"], "complete")
        self.assertEqual(first["generated"], 2)
        for job in planned["jobs"]:
            table = read_job(self.cache, job)
            self.assertEqual(table.num_rows, 2)
            self.assertEqual(table["q_0.50"].to_pylist(), [job["settings"]["prediction"]] * 2)
        with mock.patch.object(executor, "load_source", side_effect=AssertionError("source touched")), \
             mock.patch.object(executor, "load_adapter", side_effect=AssertionError("model touched")):
            resumed = generate(planned, self.cache)
            self.assertEqual(audit(planned, self.cache)["status"], "complete")
        self.assertEqual(resumed["generated"], 0)
        self.assertEqual(resumed["skipped"], 2)

    def test_limit_then_resume_equals_uninterrupted(self):
        planned = self.make_plan(rows=2, models=2)
        limited = generate(planned, self.cache, limit_jobs=1)
        self.assertEqual(limited["status"], "incomplete")
        generate(planned, self.cache)
        other = self.root / "other"
        generate(planned, other)
        for job in planned["jobs"]:
            self.assertEqual(read_job(self.cache, job).to_pydict(), read_job(other, job).to_pydict())

    def test_source_hash_shape_window_and_future_are_bounded(self):
        planned = self.make_plan()
        job = planned["jobs"][0]
        context, actual = load_source(job)
        np.testing.assert_array_equal(context, np.arange(1000.0))
        np.testing.assert_array_equal(actual, [1000.0, 1001.0])
        values = np.arange(1100.0)
        values[1099] = np.nan  # Outside the requested target window is not inspected.
        executor._SOURCES.clear()
        executor._WINDOWS.clear()
        np.save(self.source, values)
        job = self.make_plan()["jobs"][0]
        load_source(job)
        damaged = dict(job, source_sha256="0" * 64)
        damaged["job_id"] = hashlib.sha256(executor._canonical(
            executor.job_identity(damaged)).encode()).hexdigest()
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_source(damaged)
        executor._SOURCES.clear()
        executor._WINDOWS.clear()
        np.save(self.source, np.arange(1100.0).reshape(100, 11))
        shaped = self.make_plan()["jobs"][0]
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            load_source(shaped)

    def test_corrupt_duplicate_and_repair_preserves_bytes(self):
        planned = self.make_plan()
        job = planned["jobs"][0]
        generate(planned, self.cache)
        path = job_path(self.cache, job)
        original = path.read_bytes()
        duplicate = self.cache / "renamed.parquet"
        duplicate.write_bytes(original)
        report = audit(planned, self.cache)
        self.assertIn(job["job_id"], report["duplicate"])
        self.assertIn(str(duplicate), report["unexpected"])
        duplicate.unlink()
        path.write_bytes(b"corrupt partition")
        with self.assertRaisesRegex(ValueError, "repair=True"):
            generate(planned, self.cache)
        repaired = generate(planned, self.cache, repair=True)
        self.assertEqual(repaired["status"], "complete")
        quarantined = list((self.cache / "quarantine").iterdir())
        self.assertEqual(len(quarantined), 1)
        self.assertEqual(quarantined[0].read_bytes(), b"corrupt partition")

    def test_missing_and_atomic_write_failure(self):
        planned = self.make_plan()
        self.assertEqual(len(audit(planned, self.cache)["missing"]), 1)
        job = planned["jobs"][0]
        context, actual = load_source(job)
        adapter = executor.load_adapter(job)
        quantiles, measurement = adapter.predict(context, job)
        table = executor._with_metadata(executor._table(job, actual, quantiles), job, measurement)
        with mock.patch.object(pq, "write_table", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                executor._atomic_parquet(job_path(self.cache, job), table)
        self.assertFalse(job_path(self.cache, job).exists())
        self.assertEqual(list(self.cache.rglob("*.partial")), [])

    def test_transient_has_two_total_attempts_and_terminal_does_not_retry(self):
        planned = self.make_plan()
        with mock.patch.object(executor, "_execute_job",
                               side_effect=[TimeoutError("one"), ConnectionError("two")]) as call:
            result = generate(planned, self.cache)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(call.call_count, 2)
        failures = [record for record in result["measurements"] if record["status"] == "failed"]
        self.assertEqual([record["attempt"] for record in failures], [1, 2])
        for record in failures:
            self.assertGreater(record["elapsed_seconds"], 0)
            self.assertEqual(record["phase"], "execution")
            self.assertTrue(record["started_at"] and record["ended_at"])
        shutil.rmtree(self.cache)
        with mock.patch.object(executor, "_execute_job", side_effect=ValueError("terminal")) as call:
            result = generate(planned, self.cache)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(call.call_count, 1)

    def test_one_transient_attempt_can_recover(self):
        planned = self.make_plan()
        real = executor._execute_job
        with mock.patch.object(executor, "_execute_job",
                               side_effect=[TimeoutError("one"), real(planned["jobs"][0])]) as call:
            result = generate(planned, self.cache)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(call.call_count, 2)

    def test_source_preparation_failure_is_identified(self):
        planned = self.make_plan()
        with mock.patch.object(executor, "load_source", side_effect=TimeoutError("source unavailable")):
            result = generate(planned, self.cache)
        failures = [record for record in result["measurements"] if record["status"] == "failed"]
        self.assertEqual(len(failures), 2)
        self.assertEqual({record["phase"] for record in failures}, {"source_preparation"})

    def test_postpublication_failure_never_repeats_inference(self):
        planned = self.make_plan()
        original_add = executor._Ledger.add
        raised = False

        def flaky(ledger, status, **values):
            nonlocal raised
            if status == "complete" and not raised:
                raised = True
                raise TimeoutError("ledger unavailable after publish")
            return original_add(ledger, status, **values)

        with mock.patch.object(executor, "_execute_job", wraps=executor._execute_job) as execute, \
             mock.patch.object(executor._Ledger, "add", new=flaky):
            result = generate(planned, self.cache)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(execute.call_count, 1)
        read_job(self.cache, planned["jobs"][0])

    def test_source_window_cache_is_bounded_and_rechecks_changes(self):
        planned = self.make_plan(models=2)
        with mock.patch.object(executor, "_file_sha", wraps=executor._file_sha) as hashed:
            generate(planned, self.cache)
        self.assertEqual(hashed.call_count, 1)
        self.assertLessEqual(len(executor._SOURCES), 4)
        self.assertLessEqual(len(executor._WINDOWS), 4)
        job = planned["jobs"][0]
        values = np.arange(1100.0)
        values[1000] = -1
        executor._SOURCES.clear()
        executor._WINDOWS.clear()
        np.save(self.source, values)
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_source(job)

    def test_rejects_forged_bounds_heldout_empty_duplicates_fingerprints_and_batching(self):
        planned = self.make_plan()
        job = planned["jobs"][0]

        def reidentify(**updates):
            changed = dict(job, **updates)
            changed["job_id"] = hashlib.sha256(executor._canonical(
                executor.job_identity(changed)).encode()).hexdigest()
            return changed

        for changed, message in ((reidentify(dataset="M5_daily"), "held out"),
                                 (reidentify(origin=1001), "bounds")):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                load_source(changed)
        heldout = self.root / "pjm_daily"
        heldout.mkdir()
        target = heldout / "source.npy"
        shutil.copyfile(self.source, target)
        link = self.root / "safe-source.npy"
        try:
            link.symlink_to(target)
        except OSError:
            pass  # Windows may deny unprivileged symlink creation.
        else:
            with self.assertRaisesRegex(ValueError, "held out"):
                load_source(dict(job, source_path=str(link)))
        empty = dict(planned, jobs=[], planned_jobs=0, planned_rows=0)
        with self.assertRaisesRegex(ValueError, "nonempty"):
            audit(empty, self.cache)
        duplicate = dict(planned, jobs=[job, job], planned_jobs=2, planned_rows=4)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            audit(duplicate, self.cache)
        stale = self.make_plan()
        stale["jobs"][0] = reidentify(implementation_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            generate(stale, self.cache)
        batched = self.make_plan()
        batched["jobs"][0] = reidentify(batch_size=2)
        with self.assertRaisesRegex(ValueError, "not qualified"):
            generate(batched, self.cache)

    def test_run_archives_plan_and_measurement_corruption_is_not_verified(self):
        planned = self.make_plan()
        result = generate(planned, self.cache)
        run_dir = self.cache / "runs" / result["run_id"]
        self.assertEqual(json.loads((run_dir / "plan.json").read_text()), planned)
        self.assertEqual(result["plan_provenance"]["workload_sha256"], planned["workload_sha256"])
        job = planned["jobs"][0]
        path = job_path(self.cache, job)
        table = pq.ParquetFile(path).read()
        metadata = dict(table.schema.metadata)
        metadata[b"measurement"] = b"{}"
        executor._atomic_parquet(path, table.replace_schema_metadata(metadata))
        report = audit(planned, self.cache)
        self.assertEqual(report["verified_jobs"], 0)
        self.assertEqual(len(report["corrupt"]), 1)

    def test_rejects_non_real_numeric_sources_and_bad_context_timestamp(self):
        for dtype in (np.bool_, np.complex128, "U2"):
            executor._SOURCES.clear()
            executor._WINDOWS.clear()
            np.save(self.source, np.arange(1100).astype(dtype))
            job = self.make_plan()["jobs"][0]
            with self.subTest(dtype=str(dtype)), self.assertRaisesRegex(ValueError, "real numeric"):
                load_source(job)
        executor._SOURCES.clear()
        executor._WINDOWS.clear()
        np.save(self.source, np.arange(1100.0))
        job = self.make_plan()["jobs"][0]
        changed = dict(job, context_start_timestamp="1999-01-01T00:00:00")
        changed["job_id"] = hashlib.sha256(executor._canonical(
            executor.job_identity(changed)).encode()).hexdigest()
        with self.assertRaisesRegex(ValueError, "context_start_timestamp"):
            load_source(changed)

    def test_parallel_recovers_published_partition_without_resubmit(self):
        planned = self.make_plan(rows=2)
        original = executor._atomic_parquet
        raised = False

        def after_replace(path, table):
            nonlocal raised
            original(path, table)
            if not raised:
                raised = True
                raise OSError(errno.EINTR, "directory fsync interrupted")

        with mock.patch.object(executor, "_atomic_parquet", side_effect=after_replace):
            result = generate(planned, self.cache, workers=2)
        self.assertEqual(result["status"], "complete")
        events = [json.loads(line) for line in
                  (self.cache / "runs" / result["run_id"] / "events.jsonl").read_text().splitlines()]
        attempts = [(event.get("job_id"), event.get("attempt")) for event in events
                    if event["status"] == "running"]
        self.assertEqual(len(attempts), 2)

    def test_keyboard_interrupt_after_publication_is_not_swallowed(self):
        planned = self.make_plan()
        original = executor._write_result

        def interrupt(*args, **kwargs):
            original(*args, **kwargs)
            raise KeyboardInterrupt

        with mock.patch.object(executor, "_write_result", side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                generate(planned, self.cache)

    def test_workers_two_matches_one(self):
        planned = self.make_plan(rows=2, models=2)
        serial, parallel = self.root / "serial", self.root / "parallel"
        generate(planned, serial, workers=1)
        result = generate(planned, parallel, workers=2)
        self.assertEqual(result["status"], "complete")
        for job in planned["jobs"]:
            self.assertEqual(read_job(serial, job).to_pydict(), read_job(parallel, job).to_pydict())

    def test_coordinator_lock_blocks_another_process(self):
        code = """
import sys, time
from decision_aware_tsfm.cache.executor import generation_lock
with generation_lock(sys.argv[1]):
    print('locked', flush=True)
    time.sleep(2)
"""
        env = dict(os.environ, PYTHONPATH="src")
        process = subprocess.Popen([sys.executable, "-c", code, str(self.cache)], cwd=Path.cwd(),
                                   env=env, stdout=subprocess.PIPE, text=True)
        self.assertEqual(process.stdout.readline().strip(), "locked")
        try:
            with self.assertRaises(BlockingIOError):
                with executor.generation_lock(self.cache):
                    pass
        finally:
            process.terminate()
            process.wait(timeout=5)
            process.stdout.close()

    def test_audit_detects_actual_mismatch_across_models(self):
        planned = self.make_plan(models=2)
        generate(planned, self.cache)
        job = planned["jobs"][1]
        path = job_path(self.cache, job)
        table = pq.ParquetFile(path).read()
        columns = table.to_pydict()
        columns["y_true"][0] += 1
        changed = executor.pa.table(columns, schema=executor.SCHEMA)
        measurement = json.loads(table.schema.metadata[b"measurement"])
        executor._atomic_parquet(path, executor._with_metadata(changed, job, measurement))
        self.assertEqual(audit(planned, self.cache)["mismatched_actuals"], [job["job_id"]])


if __name__ == "__main__":
    unittest.main()
