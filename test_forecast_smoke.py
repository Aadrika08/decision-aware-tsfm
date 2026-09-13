"""Offline checks for #3. No model downloads or other owners' code required."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa

import forecast_smoke as smoke


class SmokeTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.cache = Path(self.directory.name)
        self.job = smoke.planned_jobs()[0]
        self.actual = np.arange(24, dtype=float)
        self.quantiles = np.repeat(self.actual[:, None], 11, axis=1)
        self.measurement = dict(inference_sampling_seconds=1., device="test", peak_gpu_memory_bytes=0)

    def write(self, job=None, actual=None, quantiles=None):
        smoke.write_job(self.cache, self.job if job is None else job,
                        self.actual if actual is None else actual,
                        self.quantiles if quantiles is None else quantiles, self.measurement)

    def test_window_and_no_future_leakage(self):
        series = np.arange(17420, dtype=float)
        context, actual = smoke.slice_window(series, self.job)
        np.testing.assert_array_equal(context, series[14364:16412])
        np.testing.assert_array_equal(actual, series[16412:16436])
        before = smoke.seasonal_quantiles(context)
        series[16412:] = -999
        after, _ = smoke.slice_window(series, self.job)
        np.testing.assert_array_equal(before, smoke.seasonal_quantiles(after))
        np.testing.assert_array_equal(before[:, 5], context[-24:] + 24)
        with self.assertRaises(ValueError):
            smoke.slice_window(np.zeros((7, 17420)), self.job)
        series[15000] = np.nan
        with self.assertRaises(ValueError):
            smoke.slice_window(series, self.job)

    def test_roundtrip_and_invalid_cache(self):
        self.write()
        a, q, _ = smoke.read_job(self.cache, self.job)
        np.testing.assert_array_equal(a, self.actual)
        np.testing.assert_array_equal(q, self.quantiles)
        with self.assertRaises(FileExistsError):
            self.write()
        table = smoke.pq.ParquetFile(smoke.job_path(self.cache, self.job)).read()
        with self.assertRaises(ValueError):
            smoke.validate_table(table.slice(0, 23), self.job)
        with self.assertRaises(ValueError):
            smoke.validate_table(table.set_column(table.schema.get_field_index('step'), 'step', pa.array([1] * 24)), self.job)
        with self.assertRaises(ValueError):
            smoke.validate_table(table, dict(self.job, seed=123))
        for bad in [np.full((24, 11), np.nan), np.tile(np.arange(11)[::-1], (24, 1)), np.zeros((23, 11))]:
            with self.assertRaises(ValueError):
                self.write(quantiles=bad)

    def test_atomic_interruption_and_resume(self):
        with patch.object(smoke.os, 'replace', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.write()
        self.assertFalse(smoke.job_path(self.cache, self.job).exists())
        self.write()
        for job in smoke.planned_jobs()[1:]:
            self.write(job=job)
        args = argparse.Namespace(cache=self.cache, data=self.cache / 'missing', device='cuda', limit_jobs=None)
        with patch.object(smoke, 'load_moirai', side_effect=AssertionError('must not load model')), \
             patch.object(smoke, 'load_series', side_effect=AssertionError('must not load source')):
            run = smoke.generate(args)
        self.assertEqual([j['status'] for j in run['jobs']], ['skipped'] * 4)
        self.assertEqual(run['model_load_seconds'], 0)

    def test_cache_only_metrics_and_projection(self):
        for job in smoke.planned_jobs():
            # A deliberately worse TSFM fixture: this is an arithmetic test, not a model result.
            q = self.quantiles + (2 if job['family'] == 'tsfm' else 0)
            self.write(job=job, quantiles=q)
        report = smoke.analyze(self.cache)
        median = [r for r in report['costs'] if r['tau'] == .5]
        self.assertEqual([r['mean_pinball'] for r in median], [0, 1])
        self.assertEqual([r['representative_regret'] for r in median], [0, 1])
        self.assertTrue(all(r['representative_regret'] >= 0 for r in report['costs']))
        self.assertEqual(report['rows'], 96)
        raw = subprocess.check_output([sys.executable, str(smoke.ROOT / 'forecast_smoke.py'),
                                       'analyze', '--cache', str(self.cache)], text=True)
        self.assertFalse(json.loads(raw)['models_imported'])
        projection = smoke.dry_run(self.cache, 3600)['projection']
        self.assertEqual(projection[smoke.MODELS[0]]['estimated_inference_sampling_hours'], 1)

    def test_stale_configuration_and_missing_jobs(self):
        self.write()
        with self.assertRaises(FileNotFoundError):
            smoke.analyze(self.cache)
        changed = dict(self.job, settings={'season_length': 12})
        changed.pop('job_id')
        changed['job_id'] = smoke.digest(smoke.canonical(changed).encode())
        self.assertNotEqual(smoke.job_path(self.cache, changed), smoke.job_path(self.cache, self.job))
        self.assertFalse(smoke.job_path(self.cache, changed).exists())


if __name__ == '__main__':
    unittest.main()
