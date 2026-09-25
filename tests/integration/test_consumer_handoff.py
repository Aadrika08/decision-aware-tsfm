"""Producer handoff checks; the optional source run uses the real pinned PR #44 generator."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

import pandas as pd
import pyarrow.parquet as pq

from decision_aware_tsfm.cache.handoff_fixture import _bounded, _jobs, export_handoff
from decision_aware_tsfm.cache.schema import QCOLS, SCHEMA, validate_payload


def small_fixture():
    rows = []
    for dataset in ("fx_tsfm_wins", "fx_tau_flip"):
        for origin in (1000, 1024):
            for model in ("AutoETS", "SeasonalNaive", "Moirai-small"):
                for step in range(1, 25):
                    row = dict(dataset=dataset, config="H", series_id=f"{dataset}/s0",
                               model=model, horizon=24, origin=origin, step=step,
                               y=float(step))
                    row.update({q: float(step + i / 10) for i, q in enumerate(QCOLS)})
                    rows.append(row)
    omega = pd.DataFrame([dict(dataset=ds, series_id=f"{ds}/s0", omega_levels=0.5)
                          for ds in ("fx_tsfm_wins", "fx_tau_flip")])
    return pd.DataFrame(rows), omega


class ConsumerHandoffTest(unittest.TestCase):
    def test_schema_origins_seeds_and_full_quantiles(self):
        frame, omega = small_fixture()
        partitions, omega_rows = _jobs(frame, omega, "a" * 64)
        self.assertEqual(len(partitions), 24)
        self.assertEqual(len(omega_rows), 8)
        by_seed = {}
        for _, table in partitions:
            validate_payload(table)
            self.assertEqual(table.schema.remove_metadata(), SCHEMA)
            row = table.slice(0, 1).to_pylist()[0]
            self.assertEqual(row["origin"], 1001 if row["timestamp"] < "2020-02-12" else 1025)
            self.assertEqual(row["timestamp"],
                             (pd.Timestamp("2020-01-01") + pd.Timedelta(hours=row["origin"])).isoformat())
            self.assertEqual([row[q] for q in QCOLS], [1 + i / 10 for i in range(11)])
            job = json.loads(table.schema.metadata[b"job"])
            self.assertEqual(job["source_sha256"], next(r["source_sha256"] for r in omega_rows
                             if (r["dataset"], r["origin"]) == (row["dataset"], row["origin"])))
            key = (row["dataset"], row["origin"], row["model"])
            by_seed.setdefault(key, {})[row["seed"]] = table.to_pydict()
        for versions in by_seed.values():
            self.assertEqual(set(versions), {42, 43})
            for field in ("y_true", "timestamp", *QCOLS):
                self.assertEqual(versions[42][field], versions[43][field])
            self.assertNotEqual(versions[42]["job_id"], versions[43]["job_id"])
        for row in omega_rows:
            self.assertEqual(row["history_end"], row["origin"])
            self.assertLessEqual(row["history_start"], row["origin"])

    def test_median_only_reader_would_lose_fixture_signal(self):
        frame, omega = small_fixture()
        table = _jobs(frame, omega, "a" * 64)[0][0][1]
        row = table.slice(0, 1).to_pylist()[0]
        self.assertNotEqual(row["q_0.01"], row["q_0.50"])
        self.assertNotEqual(row["q_0.91"], row["q_0.50"])
        # A reader that copies median into every quantile fails this invariant.
        median_only = {q: row["q_0.50"] for q in QCOLS}
        self.assertNotEqual([median_only[q] for q in QCOLS], [row[q] for q in QCOLS])

    def test_invalid_frame_and_existing_output_are_rejected(self):
        frame, omega = small_fixture()
        one_origin = frame.loc[~((frame.dataset == "fx_tau_flip") & (frame.origin == 1024))]
        with self.assertRaisesRegex(ValueError, "two origins per series"):
            _bounded(one_origin)
        for name in ("origin", "step"):
            malformed = frame.copy()
            malformed[name] = malformed[name].astype(float)
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "non-null integers"):
                _bounded(malformed)
        frame.loc[0, "q_0.01"] = float("nan")
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            _jobs(frame, omega, "a" * 64)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "keep").write_text("unchanged")
            with self.assertRaises(FileExistsError):
                export_handoff(path, path)
            self.assertEqual((path / "keep").read_text(), "unchanged")

    @unittest.skipUnless(os.environ.get("DECISION_FIXTURE_SOURCE"), "set DECISION_FIXTURE_SOURCE to pinned PR #44 checkout")
    def test_pinned_generator_bundle_is_deterministic(self):
        source = Path(os.environ["DECISION_FIXTURE_SOURCE"])
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "a", Path(directory) / "b"
            first = export_handoff(source, a)
            second = export_handoff(source, b)
            self.assertEqual(first, second)
            self.assertGreater(len(first["partition_hashes"]), 20)
            self.assertGreaterEqual(len({p.split("/")[0] for p in first["partition_hashes"]}), 3)
            for relative, expected in first["partition_hashes"].items():
                path = a / relative
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)
                validate_payload(pq.ParquetFile(path).read())
            omega = json.loads((a / "omega.json").read_text())
            self.assertEqual({r["variant"] for r in omega["rows"]}, {"current", "last_4096"})
            self.assertEqual(omega["evidence_kind"], "synthetic_omega_labels")
            answers = json.loads((a / "planted_answers.json").read_text())
            self.assertEqual(answers["evidence_kind"], "analytic_expected_costs_not_finite_sample_scores")


if __name__ == "__main__":
    unittest.main()
