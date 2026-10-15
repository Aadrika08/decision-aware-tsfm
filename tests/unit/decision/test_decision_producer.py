"""Producer-v1 ingress (#45 / #47): format checks, seeds, time convention,
per-origin Omega join, and the handoff-bundle reader."""
import contextlib
import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from decision_aware_tsfm.analysis.decision.cache_io import (
    CacheError, DuplicateCacheKeyError, load_cache, qcol, restrict_to_representatives)
from decision_aware_tsfm.analysis.decision.costs import per_origin_costs, unit_costs
from decision_aware_tsfm.analysis.decision.evaluate import run_selectors
from decision_aware_tsfm.analysis.decision.fixture import (
    FAMILY_OF_MODEL, FIT_ONLY_DATASETS, FIXTURE_STATIC_THRESHOLD, FLAT_FAMILY_OF_MODEL,
    FLAT_REPRESENTATIVES, PRODUCER_SCHEMA, REPRESENTATIVES, flat_fixture_producer,
    stated_scales, to_producer_format)
from decision_aware_tsfm.analysis.decision.handoff import (
    PRODUCER_ARROW_SCHEMA, HandoffError, read_handoff)
from decision_aware_tsfm.analysis.decision.metrics import TAU_GRID
from decision_aware_tsfm.analysis.decision.pipeline import OmegaJoinError, build_tables, run
from decision_aware_tsfm.analysis.decision.run_handoff_check import main as handoff_check
from decision_aware_tsfm.analysis.decision.scale import attach_scale
from decision_aware_tsfm.analysis.decision.selectors import (
    AlwaysClassical, AlwaysTSFM, HindsightOracle)

from . import _shared, _sources


def quiet_check(argv):
    with contextlib.redirect_stdout(io.StringIO()):
        handoff_check(argv)

QCOLS = [qcol(t) for t in TAU_GRID]
FLAT_OMEGA = pd.DataFrame([dict(dataset="fx_flat", config="H", series_id="fx_flat/s0", omega=0.5)])


def producer_small():
    return to_producer_format(_shared.small())


class ProducerIngress(unittest.TestCase):
    def test_round_trip_gives_identical_results(self):
        """The whole fixture through producer-v1 gives exactly the canonical results."""
        fc, omega, _ = _shared.fixture_data()
        t = build_tables(to_producer_format(fc), PRODUCER_SCHEMA, REPRESENTATIVES, FAMILY_OF_MODEL,
                         omega, fit_only_datasets=FIT_ONLY_DATASETS, scales=_shared.scales())
        res, _ = run(t, REPRESENTATIVES, FIXTURE_STATIC_THRESHOLD)
        ref = _shared.results()[0]
        key = ["selector", "dataset", "config", "series_id", "horizon", "tau"]
        a = res.sort_values(key).reset_index(drop=True)
        b = ref.sort_values(key).reset_index(drop=True)
        num = a.select_dtypes("number").columns
        self.assertTrue((a.drop(columns=num).fillna("NA") == b.drop(columns=num).fillna("NA")).all().all())
        self.assertEqual(float((a[num].fillna(0) - b[num].fillna(0)).abs().max().max()), 0.0)

    def test_flat_fixture_costs_are_2tau_and_1_minus_tau(self):
        """y = 10, classical = 8, TSFM = 11 at every quantile (the producer's flat fixture)."""
        t = build_tables(flat_fixture_producer(), PRODUCER_SCHEMA, FLAT_REPRESENTATIVES,
                         FLAT_FAMILY_OF_MODEL, FLAT_OMEGA, omega_col="omega", val_fraction=0,
                         scales=stated_scales(flat_fixture_producer()))
        test = t["test"].sort_values("tau")
        np.testing.assert_allclose(test.J_classical, 2 * np.array(TAU_GRID), atol=1e-12)
        np.testing.assert_allclose(test.J_tsfm, 1 - np.array(TAU_GRID), atol=1e-12)
        res = run_selectors([AlwaysClassical(), AlwaysTSFM(), HindsightOracle()], t["units"], t["val"],
                            t["test"], FLAT_REPRESENTATIVES, t["n_val"])
        orc = res[res.selector == "hindsight_oracle"].set_index("tau")
        for tau in TAU_GRID:
            self.assertEqual(orc.loc[tau, "chosen_family"], "classical" if tau < 1 / 3 else "tsfm")
        cls = res[res.selector == "always_classical"].set_index("tau")
        self.assertAlmostEqual(cls.loc[0.5, "regret"], 0.5, places=12)     # costs 1 and 0.5
        self.assertAlmostEqual(cls.loc[0.91, "regret"], 1.73, places=12)   # costs 1.82 and 0.09

    def test_producer_decision_fixture_if_available(self):
        try:
            from decision_aware_tsfm.cache.schema import decision_fixture
        except ImportError:
            self.skipTest("cache.schema.decision_fixture not in this checkout")
        fc = load_cache(decision_fixture().to_pandas(), PRODUCER_SCHEMA)
        fam = fc.groupby("model")["family"].first().to_dict()
        reps = {v: k for k, v in fam.items()}
        po = per_origin_costs(restrict_to_representatives(fc, reps, fam))
        u = unit_costs(attach_scale(po, stated_scales(fc), "test"), None).sort_values("tau")
        np.testing.assert_allclose(u.J_classical, 2 * np.array(TAU_GRID), atol=1e-12)
        np.testing.assert_allclose(u.J_tsfm, 1 - np.array(TAU_GRID), atol=1e-12)

    def test_reader_uses_each_quantile_column(self):
        """A reader that used q_0.50 for every tau would give different costs."""
        p = producer_small()
        med = p.copy()
        for c in QCOLS:
            med[c] = p[qcol(0.5)]
        def J(frame):
            fc = restrict_to_representatives(load_cache(frame, PRODUCER_SCHEMA), REPRESENTATIVES, FAMILY_OF_MODEL)
            po = attach_scale(per_origin_costs(fc), _shared.scales(), "test")
            return unit_costs(po, None).set_index(["series_id", "tau"])["J_tsfm"]
        a, b = J(p), J(med)
        at = lambda s, tau: s.xs(tau, level="tau")
        np.testing.assert_allclose(at(a, 0.5), at(b, 0.5), atol=1e-12)
        self.assertTrue((np.abs(at(a, 0.01) - at(b, 0.01)) > 1e-3).all())

    def _fails(self, frame, exc=CacheError, pattern=None):
        with self.assertRaisesRegex(exc, pattern or ".*"):
            load_cache(frame, PRODUCER_SCHEMA)

    def test_bad_producer_input_fails_loudly(self):
        base = producer_small()
        cases = {
            "missing column": (base.drop(columns="seed"), "missing columns"),
            "null value": (base.assign(config=base.config.where(base.index != 0, None)), "null"),
            "non-finite y": (base.assign(y_true=base.y_true.where(base.index != 0, np.inf)), "non-finite"),
            "unknown family": (base.assign(family=base.family.replace("tsfm", "foundation")), "family"),
            "negative origin": (base.assign(origin=base.origin - 10_000), "origin"),
            "float origin": (base.assign(origin=base.origin.astype(float)), "int64"),
            "non-ISO timestamp": (base.assign(timestamp="yesterday"), "ISO 8601"),
            "crossing quantiles": (base.assign(**{qcol(0.99): base[qcol(0.01)] - 1}), "crossing"),
        }
        for name, (frame, pattern) in cases.items():
            with self.subTest(name):
                self._fails(frame, pattern=pattern)

    def test_duplicate_job_step_fails(self):
        base = producer_small()
        self._fails(pd.concat([base, base.iloc[:1]]), DuplicateCacheKeyError, "job_id, step")

    def test_job_with_two_origins_fails(self):
        base = producer_small()
        j = base.job_id.iloc[0]
        bad = base.copy()
        bad.loc[(bad.job_id == j) & (bad.step == 2), "origin"] += 1
        self._fails(bad, pattern="jobs whose rows disagree")

    def test_model_with_two_families_fails(self):
        bad = producer_small()
        bad.loc[bad.index[0], "family"] = "tsfm" if bad.family.iloc[0] == "classical" else "classical"
        self._fails(bad, pattern="more than one family")

    def test_timestamp_conflicts_fail(self):
        base = producer_small()
        two = base.copy()
        i = two.index[(two.model == "Moirai-small")][0]
        two.loc[i, "timestamp"] = "1999-01-01T00:00:00"
        self._fails(two, pattern="more than one timestamp")
        backwards = base.copy()
        backwards["timestamp"] = (pd.Timestamp("2030-01-01")
                                  - pd.to_timedelta(backwards.origin + backwards.step - 1, unit="h")
                                  ).dt.strftime("%Y-%m-%dT%H:%M:%S")
        self._fails(backwards, pattern="do not increase")

    def test_same_position_different_y_across_horizons_fails(self):
        fc = _shared.fixture_data()[0]
        s = fc[(fc.series_id == "fx_tie/s0") & (fc.model == "AutoETS")].copy()
        pos = s.origin + s.step - 1
        h1 = s.index[(s.horizon == 1) & (pos == int(pos[s.horizon == 24].iloc[0]))]
        s.loc[h1, "y"] += 1.0
        with self.assertRaisesRegex(CacheError, "different y"):
            load_cache(s, "canonical-v1")

    def test_family_column_must_match_frozen_mapping(self):
        p = producer_small()
        p.loc[p.model == "Moirai-small", "family"] = "classical"
        fc = load_cache(p, PRODUCER_SCHEMA)
        with self.assertRaisesRegex(CacheError, "family labels disagree"):
            restrict_to_representatives(fc, REPRESENTATIVES, FAMILY_OF_MODEL)


class Seeds(unittest.TestCase):
    def _two_seeds(self, shift=0.5):
        p0 = producer_small()
        p1 = p0.copy()
        p1["seed"] = np.int64(1)
        p1["job_id"] = p1["job_id"] + "|s1"
        tsfm = p1.model == "Moirai-small"
        p1.loc[tsfm, QCOLS] = p1.loc[tsfm, QCOLS] + shift
        return p0, p1

    def _J(self, frame):
        fc = restrict_to_representatives(load_cache(frame, PRODUCER_SCHEMA), REPRESENTATIVES, FAMILY_OF_MODEL)
        po = attach_scale(per_origin_costs(fc), _shared.scales(), "test")
        return unit_costs(po, None).set_index(["series_id", "tau"])

    def test_costs_are_averaged_over_seeds(self):
        p0, p1 = self._two_seeds()
        both = self._J(pd.concat([p0, p1], ignore_index=True))
        mean = (self._J(p0)["J_tsfm"] + self._J(p1.assign(seed=np.int64(0)))["J_tsfm"]) / 2
        np.testing.assert_allclose(both["J_tsfm"], mean.loc[both.index], rtol=1e-12)

    def test_identical_seed_copies_change_nothing(self):
        p0, p1 = self._two_seeds(shift=0.0)
        np.testing.assert_allclose(self._J(pd.concat([p0, p1], ignore_index=True))["J_tsfm"],
                                   self._J(p0)["J_tsfm"], rtol=1e-12)

    def test_each_window_counts_once_whatever_its_seed_count(self):
        """A window run with two seeds must not weigh twice as much as a window run with one."""
        p0, p1 = self._two_seeds()
        extra = p1[p1.origin == p1.groupby("series_id").origin.transform("min")]   # 2nd seed on first window only
        mixed = self._J(pd.concat([p0, extra], ignore_index=True))["J_tsfm"]
        fc = restrict_to_representatives(load_cache(pd.concat([p0, extra], ignore_index=True), PRODUCER_SCHEMA),
                                         REPRESENTATIVES, FAMILY_OF_MODEL)
        po = per_origin_costs(fc)
        po = po[po.family == "tsfm"]
        per_window = po.pinball_sum / po.n                                  # already seed-averaged
        want = per_window.groupby([po.series_id, po.tau]).mean()
        np.testing.assert_allclose(mixed, want.loc[mixed.index], rtol=1e-12)
        self.assertEqual(sorted(po.n_seeds.unique()), [1, 2])

    def test_families_with_different_seed_sets_fail(self):
        p0, p1 = self._two_seeds()
        only_tsfm = p1[p1.model == "Moirai-small"]
        fc = load_cache(pd.concat([p0, only_tsfm], ignore_index=True), PRODUCER_SCHEMA)
        with self.assertRaisesRegex(CacheError, "seed sets"):
            restrict_to_representatives(fc, REPRESENTATIVES, FAMILY_OF_MODEL)


def per_origin_omega(fc):
    """Distinct Omega per (series, origin) and per variant, so a wrong join is visible."""
    so = fc[["dataset", "config", "series_id", "origin"]].drop_duplicates()
    rows = []
    for v, off in (("current", 0.0), ("last_4096", 0.25)):
        rows.append(so.assign(variant=v, omega=so.origin / 100_000 + off))
    return pd.concat(rows, ignore_index=True)


class PerOriginOmega(unittest.TestCase):
    def setUp(self):
        self.fc, _, _ = _shared.fixture_data()
        self.omega = per_origin_omega(self.fc)

    def test_uses_omega_at_first_test_origin(self):
        t = build_tables(self.fc, "canonical-v1", REPRESENTATIVES, FAMILY_OF_MODEL, self.omega,
                         fit_only_datasets=FIT_ONLY_DATASETS, omega_col="omega", omega_variant="last_4096",
                         scales=_shared.scales())
        first_test = (t["per_origin"][t["per_origin"].role == "test"]
                      .groupby(["series_id", "horizon"]).origin.min())
        u = t["units"].drop_duplicates(["series_id", "horizon"]).set_index(["series_id", "horizon"])
        np.testing.assert_allclose(u.omega, first_test.loc[u.index] / 100_000 + 0.25, atol=1e-15)
        fit = t["fit_units"].drop_duplicates(["series_id", "horizon"])
        first_any = self.fc[self.fc.dataset == "fx_dfit"].groupby(["series_id", "horizon"]).origin.min()
        np.testing.assert_allclose(fit.set_index(["series_id", "horizon"]).omega,
                                   first_any.loc[list(zip(fit.series_id, fit.horizon))] / 100_000 + 0.25)

    def test_variant_must_be_named(self):
        with self.assertRaisesRegex(OmegaJoinError, "omega_variant"):
            build_tables(self.fc, "canonical-v1", REPRESENTATIVES, FAMILY_OF_MODEL, self.omega,
                         fit_only_datasets=FIT_ONLY_DATASETS, omega_col="omega", scales=_shared.scales())

    def test_variant_without_variant_column_fails(self):
        with self.assertRaisesRegex(OmegaJoinError, "no 'variant' column"):
            build_tables(self.fc, "canonical-v1", REPRESENTATIVES, FAMILY_OF_MODEL,
                         self.omega.drop(columns="variant").drop_duplicates(["series_id", "origin"]),
                         fit_only_datasets=FIT_ONLY_DATASETS, omega_col="omega", omega_variant="current",
                         scales=_shared.scales())

    def test_missing_origin_row_fails(self):
        t = build_tables(self.fc, "canonical-v1", REPRESENTATIVES, FAMILY_OF_MODEL, self.omega,
                         fit_only_datasets=FIT_ONLY_DATASETS, omega_col="omega", omega_variant="current",
                         scales=_shared.scales())
        d = t["decision_origins"].iloc[0]
        gone = self.omega[~((self.omega.series_id == d.series_id) & (self.omega.origin == d.decision_origin))]
        with self.assertRaisesRegex(OmegaJoinError, "no omega"):
            build_tables(self.fc, "canonical-v1", REPRESENTATIVES, FAMILY_OF_MODEL, gone,
                         fit_only_datasets=FIT_ONLY_DATASETS, omega_col="omega", omega_variant="current",
                         scales=_shared.scales())

    def test_duplicate_omega_key_fails(self):
        dup = pd.concat([self.omega, self.omega.iloc[:1]], ignore_index=True)
        with self.assertRaisesRegex(OmegaJoinError, "duplicate"):
            build_tables(self.fc, "canonical-v1", REPRESENTATIVES, FAMILY_OF_MODEL, dup,
                         fit_only_datasets=FIT_ONLY_DATASETS, omega_col="omega", omega_variant="current",
                         scales=_shared.scales())


# ---- handoff bundles --------------------------------------------------------------

NAMING, REPO = "dataset-config-v1", "synthetic:test"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_bundle(root, frame, omega_value=0.5, omega_hash=None, history_end_shift=0, job_meta=True,
                 source_hashes=None):
    """Write a bundle in the producer layout: one parquet per job with job metadata,
    handoff.json with hashes, omega.json with per-origin rows for two variants.
    source_hashes: {(dataset, config, series_id): sha256} of real source files (see
    _sources.write_workload); without it each series gets a made-up hash."""
    root = Path(root)
    hashes, omega_rows = {}, {}
    for job_id, g in frame.groupby("job_id", sort=True):
        g = g.sort_values("step")
        r = g.iloc[0]
        src_hash = (source_hashes[(r.dataset, r.config, r.series_id)] if source_hashes is not None
                    else hashlib.sha256(f"{r.dataset}|{r.series_id}".encode()).hexdigest())
        job = dict(job_id=job_id, naming_scheme=NAMING, source_repository=REPO, source_revision="rev1",
                   source_sha256=src_hash, **{k: (int(r[k]) if k in ("origin", "horizon", "seed") else r[k])
                                              for k in ("dataset", "config", "series_id", "model",
                                                        "origin", "horizon", "seed")})
        table = pa.Table.from_pandas(g, schema=PRODUCER_ARROW_SCHEMA, preserve_index=False)
        if job_meta:
            table = table.replace_schema_metadata({b"job": json.dumps(job).encode()})
        rel = f"model={r.model}/{hashlib.sha256(job_id.encode()).hexdigest()[:16]}.parquet"
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, root / rel)
        hashes[rel] = _sha(root / rel)
        for v in ("current", "last_4096"):
            omega_rows[(r.dataset, r.config, r.series_id, int(r.origin), v)] = dict(
                dataset=r.dataset, config=r.config, series_id=r.series_id, origin=int(r.origin),
                variant=v, source_revision="rev1", source_sha256=omega_hash or src_hash,
                history_start=0, history_end=int(r.origin) + history_end_shift, omega=omega_value)
    (root / "omega.json").write_text(json.dumps(dict(naming_scheme=NAMING, source_repository=REPO,
                                                     rows=list(omega_rows.values()))), encoding="utf-8")
    (root / "handoff.json").write_text(json.dumps(dict(
        fixture=True, evidence_kind="test", naming_scheme=NAMING, source_repository=REPO,
        partition_hashes=hashes, omega_sha256=_sha(root / "omega.json"))), encoding="utf-8")
    return hashes


class HandoffBundle(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "bundle"
        self.frame = flat_fixture_producer()

    def tearDown(self):
        self.tmp.cleanup()

    def test_reads_valid_bundle(self):
        write_bundle(self.root, self.frame)
        fc, omega, rep = read_handoff(self.root, "current")
        self.assertEqual(rep["partitions_hash_verified"], rep["partitions_listed"])
        self.assertEqual(rep["rows"], len(self.frame))
        self.assertEqual(rep["seeds"], [0, 1])
        self.assertEqual(rep["unlisted_parquet_files"], 0)
        self.assertEqual(set(omega.variant), {"current", "last_4096"})

    def test_changed_partition_fails_hash_check(self):
        hashes = write_bundle(self.root, self.frame)
        rel = sorted(hashes)[0]
        t = pq.ParquetFile(self.root / rel).read()
        pq.write_table(t.set_column(t.schema.get_field_index("y_true"), "y_true",
                                    pa.array(np.full(len(t), 99.0))), self.root / rel)
        with self.assertRaisesRegex(HandoffError, "hash mismatch"):
            read_handoff(self.root, "current")

    def test_unlisted_partition_is_ignored_and_counted(self):
        write_bundle(self.root, self.frame)
        extra = next(self.root.rglob("*.parquet"))
        (extra.parent / "stray.parquet").write_bytes(extra.read_bytes())
        _, _, rep = read_handoff(self.root, "current")
        self.assertEqual(rep["unlisted_parquet_files"], 1)
        self.assertEqual(rep["rows"], len(self.frame))

    def test_missing_job_provenance_fails(self):
        write_bundle(self.root, self.frame, job_meta=False)
        with self.assertRaisesRegex(HandoffError, "provenance"):
            read_handoff(self.root, "current")

    def test_omega_history_after_origin_fails(self):
        write_bundle(self.root, self.frame, history_end_shift=1)
        with self.assertRaisesRegex(HandoffError, "after its origin"):
            read_handoff(self.root, "current")

    def test_omega_from_other_source_fails(self):
        write_bundle(self.root, self.frame, omega_hash="0" * 64)
        with self.assertRaisesRegex(HandoffError, "different source"):
            read_handoff(self.root, "current")

    def test_unknown_variant_fails(self):
        write_bundle(self.root, self.frame)
        with self.assertRaisesRegex(HandoffError, "variant"):
            read_handoff(self.root, "full_history")

    def test_wrong_column_order_fails(self):
        cols = list(self.frame.columns)
        cols[0], cols[1] = cols[1], cols[0]
        bad_schema = pa.schema([PRODUCER_ARROW_SCHEMA.field(c) for c in cols])
        write_bundle(self.root, self.frame)
        rel = next(p for p in self.root.rglob("*.parquet"))
        t = pq.ParquetFile(rel).read()
        pq.write_table(pa.Table.from_pandas(t.to_pandas()[cols], schema=bad_schema,
                                            preserve_index=False).replace_schema_metadata(t.schema.metadata), rel)
        man = json.loads((self.root / "handoff.json").read_text(encoding="utf-8"))
        man["partition_hashes"][rel.relative_to(self.root).as_posix()] = _sha(rel)
        (self.root / "handoff.json").write_text(json.dumps(man), encoding="utf-8")
        with self.assertRaisesRegex(HandoffError, "columns, order or Arrow types"):
            read_handoff(self.root, "current")

    def _bundle_with_sources(self):
        """The flat bundle plus a workload and source files (history before origin 48)."""
        _, hashes = _sources.write_workload(self.root, self.frame, y_col="y_true")
        write_bundle(self.root, self.frame, source_hashes=hashes)

    def test_check_command_runs_and_reports(self):
        self._bundle_with_sources()
        out = Path(self.tmp.name) / "out"
        quiet_check([str(self.root), "--variant", "current", "--classical", "flat-classical",
                       "--tsfm", "flat-tsfm", "--static-threshold", "0.45", "--val-fraction", "0",
                       "--out", str(out)])
        rep = json.loads((out / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(rep["column_check"]["unmatched"], 0)
        self.assertLess(rep["column_check"]["max_relative_difference"], 1e-10)
        # one dataset in the bundle: its leave-one-dataset-out fold has nothing to fit on
        self.assertEqual(rep["fit_scheme"], "leave_one_dataset_out")
        self.assertEqual(set(rep["d_omega"]["folds"]), {"fx_flat"})
        self.assertIn("0 other dataset", rep["d_omega"]["folds"]["fx_flat"]["not_fitted"])
        res = pd.read_parquet(out / "selection_results.parquet")
        self.assertTrue((res[res.selector == "d_omega"].status == "insufficient_data").all())
        self.assertTrue((res[res.selector == "always_tsfm"].status == "selected").all())

    def test_check_command_old_fit_scheme(self):
        self._bundle_with_sources()
        out = Path(self.tmp.name) / "out_old"
        quiet_check([str(self.root), "--variant", "current", "--classical", "flat-classical",
                     "--tsfm", "flat-tsfm", "--static-threshold", "0.45", "--val-fraction", "0",
                     "--fit-scheme", "fit_only_datasets", "--out", str(out)])
        rep = json.loads((out / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(rep["d_omega"], {"not_fitted": "no fit-only datasets supplied"})


@unittest.skipUnless(os.environ.get("DECISION_HANDOFF_BUNDLE"),
                     "set DECISION_HANDOFF_BUNDLE to a producer handoff directory")
class RealHandoffBundle(unittest.TestCase):
    """Runs against the producer's exported bundle when one is supplied."""
    def test_bundle_loads_for_both_variants_and_costs_match_raw_columns(self):
        root = Path(os.environ["DECISION_HANDOFF_BUNDLE"])
        for variant in ("current", "last_4096"):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as d:
                quiet_check([str(root), "--variant", variant, "--classical", "AutoETS",
                               "--tsfm", "Moirai-small", "--static-threshold", "0.45",
                               "--fit-only-datasets", "fx_dfit", "--out", str(Path(d) / "out")])
                rep = json.loads((Path(d) / "out" / "report.json").read_text(encoding="utf-8"))
                self.assertEqual(rep["partitions_hash_verified"], rep["partitions_listed"])
                self.assertEqual(rep["column_check"]["unmatched"], 0)


if __name__ == "__main__":
    unittest.main()
