import io
from pathlib import Path
import unittest
import zipfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from decision_aware_tsfm.cache.schema import SCHEMA, TAUS, decision_fixture, validate_payload


class ConsumerSchemaTest(unittest.TestCase):
    def test_roundtrip_and_known_answers(self):
        table = decision_fixture()
        sink = pa.BufferOutputStream()
        pq.write_table(table, sink)
        restored = pq.ParquetFile(pa.BufferReader(sink.getvalue())).read()
        validate_payload(restored)
        for tau in TAUS:
            y = restored["y_true"].to_numpy()
            q = restored[f"q_{tau:.2f}"].to_numpy()
            loss = np.maximum(tau * (y - q), (tau - 1) * (y - q))
            np.testing.assert_allclose(loss, [2*tau, 2*tau, 1-tau, 1-tau])

    def test_bad_payloads(self):
        table = decision_fixture()
        for bad in (table.slice(0, 3), pa.concat_tables([table, table]),
                    table.drop(["q_0.99"]), table.slice(0, 0)):
            with self.assertRaises(ValueError):
                validate_payload(bad)
        for value in (None, float("nan"), -100.):
            bad = table.set_column(table.schema.get_field_index("q_0.99"),
                                   "q_0.99", pa.array([value]*4, type=pa.float64()))
            with self.assertRaises(ValueError):
                validate_payload(bad)
        bad = table.set_column(table.schema.get_field_index("seed"), "seed",
                               pa.array([42]*4, type=pa.int32()))
        with self.assertRaises(ValueError):
            validate_payload(bad)

    def test_reordered_rows(self):
        validate_payload(decision_fixture().take(pa.array([3, 0, 2, 1])))

    def test_release_compatibility(self):
        archive = Path(__file__).resolve().parents[2] / "cache/forecast-smoke-sample.zip"
        if not archive.exists():
            self.skipTest("Download the v0.1.0-smoke release to check real-file compatibility")
        count = 0
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                if name.endswith(".parquet"):
                    table = pq.ParquetFile(io.BytesIO(z.read(name))).read()
                    self.assertEqual(table.schema.remove_metadata(), SCHEMA)
                    validate_payload(table)
                    count += len(table)
        self.assertEqual(count, 96)
