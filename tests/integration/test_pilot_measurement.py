from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest.mock import patch

from decision_aware_tsfm.cache import measurement
from decision_aware_tsfm.cache.fixtures import create_fixture
from decision_aware_tsfm.cache.planner import plan


@unittest.skipUnless(sys.platform == "linux", "Linux measurement wrapper")
class MeasurementTest(unittest.TestCase):
    def setUp(self):
        hardware = patch.object(measurement.platform, "platform", return_value="test-platform")
        hardware.start()
        self.addCleanup(hardware.stop)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        create_fixture(self.root / "inputs")
        workload = json.loads((self.root / "inputs/workload.json").read_text())
        models = json.loads((self.root / "inputs/models.json").read_text())
        self.planned = plan(workload, models, base_dir=self.root / "inputs")
        self.plan_file = self.root / "plan.json"
        self.plan_file.write_text(json.dumps(self.planned))
        self.cache = self.root / "cache"
        self.output = self.root / "measurement"
        self.expected_workers = 1

    def fake_generation(self, command, **kwargs):
        self.assertIsInstance(command, list)
        self.assertNotIn("shell", kwargs)
        self.assertEqual(command[command.index("--workers") + 1], str(self.expected_workers))
        self.assertEqual(kwargs["env"]["OPENBLAS_NUM_THREADS"], "1")
        self.assertEqual(kwargs["env"]["OMP_NUM_THREADS"], "1")
        self.assertEqual(kwargs["env"]["MKL_NUM_THREADS"], "1")
        Path(command[command.index("-o") + 1]).write_text(
            "Maximum resident set size (kbytes): 12345\n")
        Path(command[command.index("--output") + 1]).write_text('{"status":"complete"}')
        process = unittest.mock.Mock()
        process.wait.return_value = process.returncode = 0
        process.poll.return_value = 0
        return process

    def test_fixture_report_preserves_provenance_and_refuses_output_reuse(self):
        before = self.plan_file.read_bytes()
        with patch.object(measurement.shutil, "which", return_value=None), \
                patch.object(measurement.subprocess, "Popen", side_effect=self.fake_generation):
            result = measurement.measure(self.plan_file, self.cache, self.output)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["evidence_kind"], "fixture")
            self.assertEqual(result["cpu_workers"], 1)
            self.assertEqual(result["maximum_process_rss_kib"], 12345)
            self.assertIn("unavailable", result["errors"][0])
            self.assertTrue(result["hardware"]["cpu_models"])
            self.assertEqual(json.loads((self.output / "plan.json").read_text()), self.planned)
            self.assertEqual(self.plan_file.read_bytes(), before)
            with self.assertRaises(FileExistsError):
                measurement.measure(self.plan_file, self.cache, self.output)

    def test_parallel_workers_are_forwarded_and_persisted(self):
        self.expected_workers = 4
        with patch.object(measurement.shutil, "which", return_value=None), \
                patch.object(measurement.subprocess, "Popen", side_effect=self.fake_generation):
            result = measurement.measure(self.plan_file, self.cache, self.output, workers=4)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["cpu_workers"], 4)
        self.assertEqual(json.loads((self.output / "report.json").read_text())["cpu_workers"], 4)

    def test_invalid_workers_never_create_output_or_spawn(self):
        for workers in (0, -1, True, False, 1.0, "2", None):
            with self.subTest(workers=workers), \
                    patch.object(measurement.subprocess, "Popen") as spawned:
                with self.assertRaisesRegex(ValueError, "workers must be a positive integer"):
                    measurement.measure(self.plan_file, self.cache, self.output, workers=workers)
                self.assertFalse(self.output.exists())
                spawned.assert_not_called()

    def test_cli_preserves_default_and_forwards_opt_in_workers(self):
        for workers, flags in ((1, []), (4, ["--workers", "4"])):
            self.expected_workers = workers
            output = self.root / f"cli-{workers}"
            with self.subTest(workers=workers), redirect_stdout(io.StringIO()), \
                    patch.object(measurement.shutil, "which", return_value=None), \
                    patch.object(measurement.subprocess, "Popen", side_effect=self.fake_generation):
                result = measurement.main(["--plan-file", str(self.plan_file), "--cache", str(self.cache),
                                           "--output-directory", str(output), *flags])
            self.assertEqual(result, 0)
            self.assertEqual(json.loads((output / "report.json").read_text())["cpu_workers"], workers)

    def test_cli_rejects_invalid_workers_before_output_or_spawn(self):
        for workers in ("0", "-1", "1.5"):
            with self.subTest(workers=workers), redirect_stderr(io.StringIO()), \
                    patch.object(measurement.subprocess, "Popen") as spawned:
                arguments = ["--plan-file", str(self.plan_file), "--cache", str(self.cache),
                             "--output-directory", str(self.output), "--workers", workers]
                if workers == "1.5":
                    with self.assertRaises(SystemExit) as error:
                        measurement.main(arguments)
                    self.assertEqual(error.exception.code, 2)
                else:
                    self.assertEqual(measurement.main(arguments), 2)
                self.assertFalse(self.output.exists())
                spawned.assert_not_called()

    def test_bad_plan_and_overlapping_paths_never_spawn(self):
        with patch.object(measurement.subprocess, "Popen") as spawned:
            with self.assertRaisesRegex(ValueError, "separate"):
                measurement.measure(self.plan_file, self.root / "inputs", self.output)
            self.planned["jobs"][0]["source_path"] = "/tmp/pjm/untouched.npy"
            self.plan_file.write_text(json.dumps(self.planned))
            with self.assertRaisesRegex(ValueError, "held out"):
                measurement.measure(self.plan_file, self.cache, self.output)
            self.assertFalse(self.output.exists())
            spawned.assert_not_called()

    def test_real_requires_opt_in_and_approval_before_spawning(self):
        self.planned["fixture"] = False
        self.plan_file.write_text(json.dumps(self.planned))
        with patch.object(measurement, "_planned_jobs", return_value=self.planned["jobs"]), \
                patch.object(measurement.subprocess, "Popen") as spawned:
            with self.assertRaisesRegex(ValueError, "--real"):
                measurement.measure(self.plan_file, self.cache, self.output)
            with self.assertRaisesRegex(ValueError, "approval"):
                measurement.measure(self.plan_file, self.cache, self.output, real=True)
            self.assertFalse(self.output.exists())
            spawned.assert_not_called()

    def test_failed_generation_cannot_be_successful_measurement(self):
        def failed(command, **kwargs):
            process = self.fake_generation(command, **kwargs)
            process.wait.return_value = process.returncode = 2
            process.poll.return_value = 2
            return process
        with patch.object(measurement.shutil, "which", return_value=None), \
                patch.object(measurement.subprocess, "Popen", side_effect=failed):
            result = measurement.measure(self.plan_file, self.cache, self.output)
        self.assertEqual(result["generation_exit_code"], 2)
        self.assertEqual(result["status"], "failed")

    def test_interrupt_terminates_children_and_persists_report(self):
        child = unittest.mock.Mock(pid=123)
        child.poll.return_value = None
        child.wait.side_effect = [KeyboardInterrupt, 130]
        child.returncode = 130
        previous = signal.getsignal(signal.SIGTERM)
        with patch.object(measurement.shutil, "which", return_value=None), \
                patch.object(measurement.subprocess, "Popen", return_value=child), \
                patch.object(measurement.os, "killpg") as kill:
            with self.assertRaises(KeyboardInterrupt):
                measurement.measure(self.plan_file, self.cache, self.output)
        kill.assert_called_once_with(123, signal.SIGINT)
        result = json.loads((self.output / "report.json").read_text())
        self.assertEqual(result["status"], "interrupted")
        self.assertEqual(result["generation_status"], "interrupted")
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous)

    def test_gpu_samples_are_distinct_and_invalid_values_rejected(self):
        samples = self.root / "samples.csv"
        samples.write_text("2026/09/21 10:00:00, GPU-test, 100, 20, 6000\n"
                           "2026/09/21 10:00:01, GPU-test, 120, 10, 6000\n")
        summary = measurement._gpu_summary(samples)["GPU-test"]
        self.assertEqual(summary["sample_count"], 2)
        self.assertEqual(summary["max_used_mib"], 120)
        self.assertEqual(summary["max_utilization_percent"], 20)
        samples.write_text("2026/09/21, GPU-test, N/A, 20, 6000\n")
        with self.assertRaises(ValueError):
            measurement._gpu_summary(samples)

    def test_missing_time_is_explicit_and_does_not_start_generation(self):
        with patch.object(measurement.os, "access", return_value=False), \
                patch.object(measurement.subprocess, "Popen") as spawned:
            with self.assertRaisesRegex(ValueError, "/usr/bin/time"):
                measurement.measure(self.plan_file, self.cache, self.output)
            spawned.assert_not_called()
        result = json.loads((self.output / "report.json").read_text())
        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["generation_exit_code"])

    def test_gpu_sampling_binds_device_and_missing_tool_is_incomplete(self):
        self.planned["jobs"][0]["device"] = "cuda"
        self.plan_file.write_text(json.dumps(self.planned))
        sampler = unittest.mock.Mock(pid=456, returncode=None)
        sampler.poll.return_value = None
        def stopped(*args, **kwargs):
            sampler.returncode = sampler.poll.return_value = -15
            return -15
        sampler.wait.side_effect = stopped
        def spawn(command, **kwargs):
            if command[0] == "/fake/nvidia-smi":
                kwargs["stdout"].write("2026/09/21 10:00:00, GPU-test, 100, 20, 6000\n")
                self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "GPU-test")
                return sampler
            return self.fake_generation(command, **kwargs)
        identity = unittest.mock.Mock(returncode=0, stdout="GPU-test, Test GPU, 1.2, 6000\n", stderr="")
        with patch.object(measurement, "_planned_jobs", return_value=self.planned["jobs"]), \
                patch.dict(measurement.os.environ, {"CUDA_VISIBLE_DEVICES": "0"}), \
                patch.object(measurement.shutil, "which", return_value="/fake/nvidia-smi"), \
                patch.object(measurement.subprocess, "run", return_value=identity), \
                patch.object(measurement.subprocess, "Popen", side_effect=spawn), \
                patch.object(measurement.os, "killpg") as kill:
            result = measurement.measure(self.plan_file, self.cache, self.output)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["hardware"]["measured_gpu_uuid"], "GPU-test")
        self.assertEqual(result["gpu_device_samples"]["GPU-test"]["max_used_mib"], 100)
        kill.assert_called_once_with(456, signal.SIGTERM)
        with patch.object(measurement, "_planned_jobs", return_value=self.planned["jobs"]), \
                patch.object(measurement.shutil, "which", return_value=None), \
                patch.object(measurement.subprocess, "Popen", side_effect=self.fake_generation):
            result = measurement.measure(self.plan_file, self.cache, self.root / "missing-gpu")
        self.assertEqual(result["generation_exit_code"], 0)
        self.assertEqual(result["generation_status"], "complete")
        self.assertEqual(result["measurement_status"], "incomplete")
        self.assertEqual(result["status"], "incomplete")

    def test_ambiguous_gpu_device_cannot_start_generation(self):
        self.planned["jobs"][0]["device"] = "cuda"
        self.plan_file.write_text(json.dumps(self.planned))
        identity = unittest.mock.Mock(returncode=0, stdout="GPU-a, A, 1, 6000\nGPU-b, B, 1, 6000\n", stderr="")
        with patch.object(measurement, "_planned_jobs", return_value=self.planned["jobs"]), \
                patch.object(measurement.shutil, "which", return_value="/fake/nvidia-smi"), \
                patch.object(measurement.subprocess, "run", return_value=identity), \
                patch.object(measurement.subprocess, "Popen") as spawned:
            with self.assertRaisesRegex(ValueError, "exactly one"):
                measurement.measure(self.plan_file, self.cache, self.output)
        spawned.assert_not_called()

    def test_unusable_optional_gpu_probe_does_not_block_cpu_generation(self):
        for index, (exit_code, stdout, stderr) in enumerate((
                (1, "", "Failed to initialize NVML: Insufficient Permissions"),
                (0, "", ""))):
            with self.subTest(exit_code=exit_code, stderr=stderr):
                identity = unittest.mock.Mock(returncode=exit_code, stdout=stdout, stderr=stderr)
                with patch.object(measurement.shutil, "which", return_value="/fake/nvidia-smi"), \
                        patch.object(measurement.subprocess, "run", return_value=identity), \
                        patch.object(measurement.subprocess, "Popen", side_effect=self.fake_generation) as spawned:
                    result = measurement.measure(self.plan_file, self.cache, self.root / f"probe-{index}")
                spawned.assert_called_once()
                self.assertEqual(result["status"], "complete")
                self.assertEqual(result["measurement_status"], "complete")
                self.assertEqual(result["generation_exit_code"], 0)
                self.assertIn("GPU measurement: nvidia-smi identity probe failed", result["errors"])
                self.assertEqual(result["hardware"]["gpu_probe"]["stderr"], stderr)
                self.assertEqual(result["gpu_device_samples"], {})
        self.planned["jobs"][0]["device"] = "cuda"
        self.plan_file.write_text(json.dumps(self.planned))
        with patch.object(measurement, "_planned_jobs", return_value=self.planned["jobs"]), \
                patch.object(measurement.shutil, "which", return_value="/fake/nvidia-smi"), \
                patch.object(measurement.subprocess, "run", return_value=identity), \
                patch.object(measurement.subprocess, "Popen") as spawned:
            with self.assertRaisesRegex(ValueError, "identity probe failed"):
                measurement.measure(self.plan_file, self.cache, self.output)
        spawned.assert_not_called()
        result = json.loads((self.output / "report.json").read_text())
        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["generation_exit_code"])

    def test_cuda_rejects_invalid_cublas_before_generation_or_output(self):
        self.planned["jobs"][0]["device"] = "cuda"
        self.plan_file.write_text(json.dumps(self.planned))
        with patch.object(measurement, "_planned_jobs", return_value=self.planned["jobs"]), \
                patch.dict(measurement.os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":16:8"}), \
                patch.object(measurement.subprocess, "Popen") as spawned:
            with self.assertRaisesRegex(ValueError, "CUBLAS_WORKSPACE_CONFIG=:4096:8"):
                measurement.measure(self.plan_file, self.cache, self.output)
        spawned.assert_not_called()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
