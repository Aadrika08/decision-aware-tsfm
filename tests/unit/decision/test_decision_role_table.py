"""Role table written from the engine (decision record 0007, section 1) and the check
that a table and the engine agree."""
import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from decision_aware_tsfm.analysis.decision import role_table as rt
from decision_aware_tsfm.analysis.decision.cache_io import qcol
from decision_aware_tsfm.analysis.decision.costs import ROLES, SERIES_KEYS
from decision_aware_tsfm.analysis.decision.fixture import (
    FIT_ONLY_DATASETS, FIXTURE_SCHEMA, FLAT_FAMILY_OF_MODEL, FLAT_REPRESENTATIVES, PRODUCER_SCHEMA,
    stated_scales, to_producer_format)
from decision_aware_tsfm.analysis.decision.guards import HeldOutLeakError
from decision_aware_tsfm.analysis.decision.metrics import TAU_GRID
from decision_aware_tsfm.analysis.decision.pipeline import build_tables
from decision_aware_tsfm.analysis.decision.run_role_table import main as role_table_main

from . import _shared
from .test_decision_producer import write_bundle

# the exclusion reason exactly as decision record 0007 section 1 writes it
RECORD_REASON = ("no validation origins: the earliest-third validation window is empty after the gap "
                 "rule, so validation comparators cannot be scored")
NINE = list(range(1000, 1000 + 24 * 9, 24))            # 1000, 1024, ..., 1192
THIRTY = list(range(5000, 5000 + 24 * 30, 24))

# Shape of the real pilot workload as reported in the producer's preflight of 2026-10-12:
# (dataset, config): (series, origins per series), origins 48 steps apart, horizons 1, 24, 168.
PILOT_SHAPE = {("electricity", "H"): (10, 17), ("ett1", "H"): (7, 17), ("ett2", "H"): (7, 17),
               ("jena_weather", "H"): (21, 16), ("solar", "H"): (10, 16)}
# its proposed_role_coverage.counts, all 15 rows: dataset, horizon, validation, gap, test
PILOT_COUNTS = [
    ("electricity", 1, 50, 0, 120), ("electricity", 24, 50, 0, 120), ("electricity", 168, 20, 30, 120),
    ("ett1", 1, 35, 0, 84), ("ett1", 24, 35, 0, 84), ("ett1", 168, 14, 21, 84),
    ("ett2", 1, 35, 0, 84), ("ett2", 24, 35, 0, 84), ("ett2", 168, 14, 21, 84),
    ("jena_weather", 1, 105, 0, 231), ("jena_weather", 24, 105, 0, 231), ("jena_weather", 168, 42, 63, 231),
    ("solar", 1, 50, 0, 110), ("solar", 24, 50, 0, 110), ("solar", 168, 20, 30, 110)]


def origins(series, horizons, dataset="dev", config="H"):
    """series: {series_id: origins}. One row per series, horizon and origin."""
    return pd.DataFrame([dict(dataset=dataset, config=config, series_id=s, origin=o, horizon=h)
                         for s, os_ in series.items() for h in horizons for o in os_])


def workload(frame, seeds=(42, 43)):
    """A workload document in the producer's layout (cache.planner.plan input, as built in
    tests/integration/test_pilot_readiness.py): one row per series, origin, horizon and seed."""
    rows = [dict(dataset=r.dataset, config=r.config, series_id=r.series_id, source_revision="invented-v1",
                 source_sha256="0" * 64, source_path="invented.npy", item_id=r.series_id, channel=0,
                 frequency="H", start_timestamp="2020-01-01T00:00:00", context_start=int(r.origin) - 2048,
                 context_length=2048, origin=int(r.origin), horizon=int(r.horizon), seed=seed)
            for r in frame.itertuples() for seed in seeds]
    return dict(schema_version=1, split="development", fixture=True, quantiles=list(TAU_GRID), rows=rows)


def pilot_shape():
    parts = [origins({f"{ds}_{cfg}__s{j:02d}": [30000 + 1000 * i + 7 * j + 48 * k for k in range(n)]
                      for j in range(n_series)}, (1, 24, 168), ds, cfg)
             for i, ((ds, cfg), (n_series, n)) in enumerate(PILOT_SHAPE.items())]
    return pd.concat(parts, ignore_index=True)


def by_horizon(result):
    """{horizon: (validation, gap, evaluation)} over the whole table."""
    c = result.table.groupby(["horizon", "role"]).size().unstack("role", fill_value=0)
    c = c.reindex(columns=["validation", "gap", "evaluation"], fill_value=0)
    return {int(h): tuple(int(x) for x in row) for h, row in c.iterrows()}


def found(report):
    return {k: len(report[k]) for k in rt.DIFFERENCES if report[k]}


def flat_cache(series_origins, horizon=24):
    """Producer-v1 cache with chosen origins per series: y = 10, classical 8, TSFM 11."""
    rows = []
    for sid, os_ in series_origins.items():
        for model, value in (("flat-classical", 8.0), ("flat-tsfm", 11.0)):
            for o in os_:
                for step in range(1, horizon + 1):
                    r = dict(dataset="fx_flat", config="H", series_id=sid, model=model, seed=0,
                             horizon=horizon, origin=o, step=step, y=10.0)
                    r.update({qcol(x): value for x in TAU_GRID})
                    rows.append(r)
    return to_producer_format(pd.DataFrame(rows), FLAT_FAMILY_OF_MODEL)


def run_cli(argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        role_table_main([str(x) for x in argv])
    return out.getvalue()


class RecordWorkedExamples(unittest.TestCase):
    def test_one_series_origins_100_to_399_horizon_24(self):
        r = rt.build_role_table(origins({"s": range(100, 400)}, (24,)))
        self.assertEqual(by_horizon(r), {24: (77, 23, 200)})
        role = r.table.set_index("origin")["role"]
        self.assertEqual((role[100], role[176], role[177], role[199], role[200], role[399]),
                         ("validation", "validation", "gap", "gap", "evaluation", "evaluation"))
        self.assertEqual(r.decision_origins["decision_origin"].tolist(), [200])
        self.assertEqual(len(r.excluded), 0)

    def test_same_origins_at_other_horizons(self):
        r = rt.build_role_table(origins({"s": range(100, 400)}, (1, 100, 101, 168)))
        self.assertEqual(by_horizon(r), {1: (100, 0, 200), 100: (1, 99, 200), 101: (0, 100, 200),
                                         168: (0, 100, 200)})
        self.assertEqual(r.excluded["horizon"].tolist(), [101, 168])
        self.assertEqual(r.decision_origins["horizon"].tolist(), [1, 100])

    def test_nine_origins_spaced_24(self):
        r = rt.build_role_table(origins({"s": NINE}, (24, 48, 168)))
        self.assertEqual(by_horizon(r), {24: (3, 0, 6), 48: (2, 1, 6), 168: (0, 3, 6)})
        t = r.table
        self.assertEqual(t[(t.horizon == 48) & (t.role == "gap")]["origin"].tolist(), [1048])
        self.assertEqual(t[(t.horizon == 168) & (t.role == "gap")]["origin"].tolist(), [1000, 1024, 1048])
        # E is the earliest evaluation origin, for kept pairs only
        self.assertEqual(r.decision_origins[["horizon", "decision_origin"]].values.tolist(),
                         [[24, 1072], [48, 1072]])
        self.assertEqual(r.excluded.to_dict("records"),
                         [dict(dataset="dev", config="H", series_id="s", horizon=168, reason=RECORD_REASON)])

    def test_table_layout_and_role_words(self):
        r = rt.build_role_table(origins({"s": NINE}, (24, 48, 168)))
        self.assertEqual(list(r.table.columns), ["dataset", "config", "series_id", "origin", "horizon", "role"])
        self.assertEqual(len(r.table), 27)
        self.assertFalse(r.table.duplicated(rt.TABLE_KEYS).any())
        self.assertEqual(rt.ENGINE_ROLE, {"validation": "validation", "gap": "gap", "evaluation": "test"})
        self.assertEqual(sorted(rt.ENGINE_ROLE.values()), sorted(ROLES))
        self.assertEqual(set(r.table["role"]), {"validation", "gap", "evaluation"})

    def test_validation_fraction_is_used(self):
        self.assertEqual(by_horizon(rt.build_role_table(origins({"s": NINE}, (24,)), 0.5)), {24: (4, 0, 5)})
        none = rt.build_role_table(origins({"s": NINE}, (24,)), 0.0)
        self.assertEqual(by_horizon(none), {24: (0, 0, 9)})
        self.assertEqual(len(none.excluded), 0)        # as in build_tables: no validation was asked for

    def test_seeds_and_input_order_do_not_matter(self):
        o = origins({"a": NINE, "b": THIRTY}, (24, 168))
        ref = rt.build_role_table(o)
        again = rt.build_role_table(pd.concat([o, o]).sample(frac=1.0, random_state=5))
        for a, b in zip(ref[:3], again[:3]):
            pd.testing.assert_frame_equal(a, b)


class PilotShape(unittest.TestCase):
    """A synthetic workload with the pilot's shape. Not the real workload.json."""

    def test_counts_equal_the_producers_diagnostic(self):
        doc = workload(pilot_shape())
        self.assertEqual(len(doc["rows"]), 5424)
        o = rt.workload_origins(doc)
        self.assertEqual(len(o), 2712)                                             # source windows
        self.assertEqual(len(o[rt.SERIES_ID_KEYS].drop_duplicates()), 55)
        self.assertEqual(len(o[rt.SERIES_ID_KEYS + ["origin"]].drop_duplicates()), 904)
        r = rt.build_role_table(o)
        s = rt.summarize(r).groupby(["dataset", "horizon"])[["validation", "gap", "evaluation"]].sum()
        got = [(ds, int(h), *(int(x) for x in row)) for (ds, h), row in s.iterrows()]
        self.assertEqual(sorted(got), sorted(PILOT_COUNTS))
        self.assertEqual(len(r.excluded), 0)
        self.assertEqual(len(r.decision_origins), 55 * 3)
        self.assertEqual(int(rt.summarize(r)["excluded_series_horizons"].sum()), 0)


class CheckDetectsEachDisagreement(unittest.TestCase):
    def setUp(self):
        self.origins = origins({"short": NINE, "long": THIRTY}, (24, 48, 168))
        self.want = rt.build_role_table(self.origins)
        self.table = self.want.table

    def row(self, series, horizon, origin):
        t = self.table
        return t.index[(t.series_id == series) & (t.horizon == horizon) & (t.origin == origin)][0]

    def test_the_engines_own_table_passes(self):
        rep = rt.check_role_table(self.table, self.origins, strict=True)
        self.assertTrue(rep["ok"])
        self.assertEqual((rep["matched_rows"], rep["table_rows"], rep["engine_rows"]), (117, 117, 117))
        self.assertEqual(found(rep), {})
        rt.check_role_table(self.table, self.origins, excluded=self.want.excluded,
                            decision_origins=self.want.decision_origins, strict=True)

    def test_a_different_role(self):
        bad = self.table.copy()
        bad.loc[self.row("long", 24, 5000), "role"] = "gap"
        rep = rt.check_role_table(bad, self.origins)
        self.assertEqual(found(rep), {"role_disagreements": 1})
        self.assertEqual(rep["matched_rows"], 116)
        self.assertEqual(rep["role_disagreements"][0],
                         dict(dataset="dev", config="H", series_id="long", origin=5000, horizon=24,
                              role_engine="validation", role_table="gap"))
        self.assertFalse(rep["ok"])

    def test_a_missing_origin(self):
        rep = rt.check_role_table(self.table.drop(self.row("long", 24, 5000 + 24 * 29)), self.origins)
        self.assertEqual(found(rep), {"missing_rows": 1})
        self.assertEqual(rep["missing_rows"][0]["origin"], 5000 + 24 * 29)

    def test_an_extra_origin(self):
        extra = dict(dataset="dev", config="H", series_id="long", origin=9000, horizon=24, role="evaluation")
        rep = rt.check_role_table(pd.concat([self.table, pd.DataFrame([extra])], ignore_index=True), self.origins)
        self.assertEqual(found(rep), {"extra_rows": 1})
        self.assertEqual(rep["matched_rows"], 117)

    def test_a_horizon_missing_for_a_series(self):
        t = self.table
        rep = rt.check_role_table(t[~((t.series_id == "long") & (t.horizon == 168))], self.origins)
        # all 30 rows are missing, and with them the pair's decision origin
        self.assertEqual(found(rep), {"missing_rows": 30, "decision_origin_disagreements": 1})
        self.assertEqual({r["horizon"] for r in rep["missing_rows"]}, {168})

    def test_roles_of_two_horizons_swapped(self):
        """Origin 1048 is validation at horizon 24 and gap at horizon 48. A check that
        dropped horizon from the key could not tell these two tables apart."""
        bad = self.table.copy()
        short = bad.series_id == "short"
        bad.loc[short & (bad.horizon == 24), "horizon"] = -1
        bad.loc[short & (bad.horizon == 48), "horizon"] = 24
        bad.loc[bad.horizon == -1, "horizon"] = 48
        rep = rt.check_role_table(bad, self.origins)
        self.assertEqual(found(rep), {"role_disagreements": 2})
        self.assertEqual({(r["origin"], r["horizon"]) for r in rep["role_disagreements"]}, {(1048, 24), (1048, 48)})

    def test_duplicate_rows(self):
        dup = pd.concat([self.table, self.table.iloc[[5]]], ignore_index=True)
        rep = rt.check_role_table(dup, self.origins)
        self.assertEqual(found(rep), {"duplicate_rows": 2})

    def test_an_unknown_role_word(self):
        for word in ("test", "Validation", "train", ""):
            bad = self.table.copy()
            bad.loc[self.row("long", 24, 5000 + 24 * 29), "role"] = word      # the engine's word is not the table's
            with self.subTest(word=word):
                rep = rt.check_role_table(bad, self.origins)
                self.assertEqual(found(rep), {"unknown_roles": 1})
                self.assertEqual(rep["matched_rows"], 116)

    def test_a_table_built_with_a_different_validation_fraction(self):
        half = rt.build_role_table(self.origins, 0.5)
        rep = rt.check_role_table(half.table, self.origins)
        self.assertFalse(rep["ok"])
        # by hand: short 1 + 2 + 1 and long 5 + 6 + 10 rows change role at horizons 24, 48, 168
        self.assertEqual(len(rep["role_disagreements"]), 25)
        self.assertTrue(rt.check_role_table(half.table, self.origins, 0.5)["ok"])
        self.assertFalse(rt.check_role_table(self.table, self.origins, 0.5)["ok"])

    def test_a_wrong_decision_origin(self):
        d = self.want.decision_origins.copy()
        d.loc[(d.series_id == "short") & (d.horizon == 24), "decision_origin"] = 1048   # last validation origin
        rep = rt.check_role_table(self.table, self.origins, excluded=self.want.excluded, decision_origins=d)
        self.assertEqual(found(rep), {"decision_origin_disagreements": 1})
        self.assertEqual((rep["decision_origin_disagreements"][0]["decision_origin_engine"],
                          rep["decision_origin_disagreements"][0]["decision_origin_table"]), (1072, 1048))
        # read off the table alone: the row at E relabelled moves the table's E
        bad = self.table.copy()
        bad.loc[self.row("short", 24, 1072), "role"] = "gap"
        self.assertEqual(found(rt.check_role_table(bad, self.origins)),
                         {"role_disagreements": 1, "decision_origin_disagreements": 1})

    def test_exclusions_that_differ(self):
        ex, d = self.want.excluded, self.want.decision_origins
        self.assertEqual(ex[["series_id", "horizon"]].values.tolist(), [["short", 168]])
        cases = dict(
            not_listed=ex.iloc[:0],
            other_reason=ex.assign(reason="too short"),
            one_too_many=pd.concat([ex, ex.assign(series_id="long")], ignore_index=True))
        for name, listed in cases.items():
            with self.subTest(name):
                rep = rt.check_role_table(self.table, self.origins, excluded=listed, decision_origins=d)
                self.assertEqual(found(rep), {"exclusion_disagreements": 1})
        # read off the table alone: a pair whose only validation rows are relabelled looks excluded
        bad = self.table.copy()
        bad.loc[(bad.series_id == "short") & (bad.horizon == 48) & (bad.role == "validation"), "role"] = "gap"
        self.assertEqual(found(rt.check_role_table(bad, self.origins)),
                         {"role_disagreements": 2, "decision_origin_disagreements": 1,
                          "exclusion_disagreements": 1})

    def test_strict_mode_raises_and_names_the_difference(self):
        bad = self.table.copy()
        bad.loc[self.row("long", 24, 5000), "role"] = "evaluation"
        with self.assertRaisesRegex(rt.RoleTableMismatch, "1 role_disagreements.*'origin': 5000"):
            rt.check_role_table(bad, self.origins, strict=True)
        with self.assertRaisesRegex(rt.RoleTableMismatch, "1 missing_rows"):
            rt.check_role_table(self.table.iloc[1:], self.origins, strict=True)

    def test_held_out_names_are_refused(self):
        for column, name in (("dataset", "M5"), ("config", "pjm_hourly"), ("dataset", "gift/PJM/H")):
            with self.subTest(column=column, name=name):
                held = self.origins.assign(**{column: name})
                with self.assertRaises(HeldOutLeakError):
                    rt.build_role_table(held)
                with self.assertRaises(HeldOutLeakError):
                    rt.workload_origins(workload(held))
                with self.assertRaises(HeldOutLeakError):
                    rt.check_role_table(self.table.assign(**{column: name}), self.origins)
                with self.assertRaises(HeldOutLeakError):
                    rt.check_role_table(self.table, held)

    def test_malformed_tables_are_refused(self):
        with self.assertRaisesRegex(rt.RoleTableError, "missing columns"):
            rt.check_role_table(self.table.drop(columns="horizon"), self.origins)
        with self.assertRaisesRegex(rt.RoleTableError, "must be integer"):
            rt.check_role_table(self.table.astype({"origin": float}), self.origins)
        with self.assertRaisesRegex(rt.RoleTableError, "null values"):
            rt.check_role_table(self.table.assign(role=None), self.origins)


class WorkloadReader(unittest.TestCase):
    def test_reads_only_the_five_fields_and_collapses_seeds(self):
        o = origins({"a": NINE}, (1, 24))
        doc = workload(o)
        self.assertEqual(len(doc["rows"]), 36)
        pd.testing.assert_frame_equal(rt.workload_origins(doc), rt.distinct_origins(o))

    def test_bad_documents_are_refused(self):
        good = workload(origins({"a": NINE}, (24,)))
        cases = dict(not_development=dict(good, split="held_out"), no_rows=dict(good, rows=[]),
                     rows_not_mappings=dict(good, rows=[1, 2]), not_a_mapping=good["rows"])
        for field, value in (("origin", 1000.5), ("origin", True), ("horizon", "24"), ("origin", -1),
                             ("horizon", 0), ("series_id", None)):
            cases[f"{field}={value!r}"] = dict(good, rows=[dict(good["rows"][0], **{field: value})] + good["rows"][1:])
        cases["missing_horizon"] = dict(good, rows=[{k: v for k, v in good["rows"][0].items() if k != "horizon"}])
        for name, doc in cases.items():
            with self.subTest(name), self.assertRaises(rt.RoleTableError):
                rt.workload_origins(doc)


class ExportRoundTrip(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        # an empty config and an ID with leading zeros must survive the CSV round trip
        self.frame = pd.concat([origins({"007": NINE, "long": THIRTY}, (24, 48, 168), config=""),
                                origins({"s": NINE}, (1, 24), dataset="other")], ignore_index=True)
        self.workload = self.root / "workload.json"
        self.workload.write_text(json.dumps(workload(self.frame)), encoding="utf-8")

    def export(self, name, workload_path=None, *extra):
        out = self.root / name
        run_cli(["export", "--workload", workload_path or self.workload, "--out", out, *extra])
        return out

    def test_export_read_back_and_check(self):
        out = self.export("a")
        self.assertEqual(sorted(p.name for p in out.iterdir()),
                         ["decision_origins.csv", "excluded_series_horizons.csv", "role_table.csv",
                          "role_table.json", "summary.csv"])
        table = rt.read_csv(out / "role_table.csv")
        want = rt.build_role_table(self.frame)
        pd.testing.assert_frame_equal(table, want.table)
        pd.testing.assert_frame_equal(rt.read_csv(out / "decision_origins.csv"), want.decision_origins)
        pd.testing.assert_frame_equal(rt.read_csv(out / "excluded_series_horizons.csv"), want.excluded)
        self.assertEqual(set(table.loc[table.dataset == "dev", "config"]), {""})
        self.assertIn("007", set(table.series_id))
        rep = rt.check_role_table(table, rt.read_workload_origins(self.workload), strict=True)
        self.assertEqual(rep["matched_rows"], len(self.frame))
        self.assertIn("role table check passed", run_cli(["check", out, "--workload", self.workload]))
        self.assertIn("role table check passed",
                      run_cli(["check", out / "role_table.csv", "--workload", self.workload]))

    def test_sidecar_records_the_settings_and_hashes(self):
        out = self.export("a")
        side = json.loads((out / "role_table.json").read_text(encoding="utf-8"))
        self.assertEqual(side["val_fraction"], 1 / 3)
        self.assertEqual(side["roles"], {"validation": "validation", "gap": "gap", "evaluation": "test"})
        self.assertIn("0007", side["rule_record"])
        self.assertEqual(side["rule_functions"][0], "decision_aware_tsfm.analysis.decision.costs.assign_roles")
        self.assertEqual(side["excluded_reason"], RECORD_REASON)
        self.assertEqual(side["input"], dict(kind="workload", name="workload.json",
                                             sha256=hashlib.sha256(self.workload.read_bytes()).hexdigest()))
        # by hand: 007 gives 5/4/18, long 23/7/60, other/s 6/0/12 (validation/gap/evaluation)
        self.assertEqual(side["counts"], dict(rows=135, series=3, series_horizons=8, validation=34,
                                              gap=11, evaluation=90, excluded_series_horizons=1))
        self.assertIn("code_commit", side)
        for name, digest in side["files"].items():
            self.assertEqual(hashlib.sha256((out / name).read_bytes()).hexdigest(), digest, name)
        summary = pd.read_csv(out / "summary.csv", keep_default_na=False)
        self.assertEqual(list(summary.columns), ["dataset", "config", "horizon", "series", "validation",
                                                 "gap", "evaluation", "excluded_series_horizons"])
        self.assertEqual(summary[["validation", "gap", "evaluation"]].sum().tolist(), [34, 11, 90])

    def test_second_run_and_shuffled_input_give_identical_files(self):
        a, b = self.export("a"), self.export("b")
        doc = json.loads(self.workload.read_text(encoding="utf-8"))
        doc["rows"] = pd.Series(doc["rows"]).sample(frac=1.0, random_state=11).tolist()
        shuffled = self.root / "shuffled" / "workload.json"
        shuffled.parent.mkdir()
        shuffled.write_text(json.dumps(doc), encoding="utf-8")
        c = self.export("c", shuffled)
        for name in list(rt.FILES.values()) + [rt.SIDECAR]:
            self.assertEqual((a / name).read_bytes(), (b / name).read_bytes(), name)
        for name in rt.FILES.values():
            self.assertEqual((a / name).read_bytes(), (c / name).read_bytes(), name)
        sa, sc = (json.loads((d / rt.SIDECAR).read_text(encoding="utf-8")) for d in (a, c))
        self.assertNotEqual(sa["input"]["sha256"], sc["input"]["sha256"])
        sa["input"]["sha256"] = sc["input"]["sha256"] = None
        self.assertEqual(sa, sc)

    def test_check_command_fails_on_a_changed_row(self):
        out = self.export("a")
        path = out / "role_table.csv"
        lines = path.read_text(encoding="utf-8").split("\n")
        row = lines.index("dev,,long,5000,24,validation")
        lines[row] = "dev,,long,5000,24,evaluation"
        path.write_text("\n".join(lines), encoding="utf-8")
        with self.assertRaises(SystemExit) as failed:
            run_cli(["check", out, "--workload", self.workload])
        self.assertIn("role table check FAILED", str(failed.exception))
        self.assertIn("1 role_disagreements", str(failed.exception))
        self.assertNotEqual(failed.exception.code, 0)

    def test_check_command_reads_the_companion_lists(self):
        """The table is right; only the excluded list or the decision origins are wrong."""
        for name, old, new, kind in (
                # the reason holds a comma, so the CSV writer quotes it
                ("excluded_series_horizons.csv", f'dev,,007,168,"{RECORD_REASON}"\n', "", "1 exclusion_disagreements"),
                ("decision_origins.csv", "dev,,007,24,1072\n", "dev,,007,24,1048\n",
                 "1 decision_origin_disagreements")):
            out = self.export(name.split(".")[0])
            text = (out / name).read_text(encoding="utf-8")
            self.assertEqual(text.count(old), 1, name)
            (out / name).write_text(text.replace(old, new), encoding="utf-8")
            with self.subTest(name):
                with self.assertRaises(SystemExit) as failed:
                    run_cli(["check", out, "--workload", self.workload])
                self.assertIn(kind, str(failed.exception))
                # the table file alone still agrees with the engine
                run_cli(["check", out / "role_table.csv", "--workload", self.workload])

    def test_check_command_passes_when_nothing_is_excluded(self):
        """The pilot's case: the excluded list is a header with no rows."""
        path = self.root / "kept.json"
        path.write_text(json.dumps(workload(origins({"a": NINE, "b": THIRTY}, (1, 24)))), encoding="utf-8")
        out = self.export("kept", path)
        self.assertEqual((out / "excluded_series_horizons.csv").read_text(encoding="utf-8"),
                         "dataset,config,series_id,horizon,reason\n")
        self.assertIn("role table check passed", run_cli(["check", out, "--workload", path]))
        (out / "excluded_series_horizons.csv").write_text(
            "dataset,config,series_id,horizon,reason\ndev,H,a,24,invented\n", encoding="utf-8")
        with self.assertRaisesRegex(SystemExit, "1 exclusion_disagreements"):
            run_cli(["check", out, "--workload", path])

    def test_check_command_fails_on_another_validation_fraction(self):
        out = self.export("half", None, "--val-fraction", "0.5")
        run_cli(["check", out, "--workload", self.workload, "--val-fraction", "0.5"])
        with self.assertRaises(SystemExit) as failed:
            run_cli(["check", out, "--workload", self.workload])
        self.assertIn("role_disagreements", str(failed.exception))
        self.assertIn("sidecar records val_fraction 0.5", str(failed.exception))

    def test_export_refuses_an_existing_directory_and_held_out_names(self):
        out = self.export("a")
        with self.assertRaisesRegex(SystemExit, "already exists"):
            self.export("a")
        self.assertTrue((out / "role_table.csv").is_file())
        held = self.root / "held.json"
        held.write_text(json.dumps(workload(self.frame.assign(dataset="m5"))), encoding="utf-8")
        with self.assertRaises(HeldOutLeakError):
            self.export("held", held)
        self.assertFalse((self.root / "held").exists())


class TableEqualsTheEngine(unittest.TestCase):
    """The table built from a workload is what the engine assigns when it later reads a
    cache with the same origins."""

    def assert_same_as_engine(self, result, tables):
        engine = tables["per_origin"][rt.TABLE_KEYS + ["role"]].drop_duplicates()
        engine = engine.assign(role=engine["role"].map(rt.TABLE_ROLE))
        m = result.table.merge(engine, on=rt.TABLE_KEYS, how="outer", suffixes=("", "_engine"), indicator=True)
        kept = m[m["_merge"] == "both"]
        self.assertEqual(len(kept), len(engine))
        self.assertTrue((kept["role"] == kept["role_engine"]).all())
        self.assertFalse((m["_merge"] == "right_only").any())
        # rows the engine no longer holds belong to the excluded pairs, and to nothing else
        dropped = m.loc[m["_merge"] == "left_only", SERIES_KEYS].drop_duplicates()
        self.assertEqual(sorted(map(tuple, dropped.values.tolist())),
                         sorted(map(tuple, result.excluded[SERIES_KEYS].values.tolist())))
        ex = tables["excluded_series_horizons"]
        self.assertEqual(sorted(map(tuple, ex[rt.EXCLUDED_COLUMNS].values.tolist())),
                         sorted(map(tuple, result.excluded.values.tolist())))
        self.assertTrue((ex["used_for"] == "evaluation").all())
        pd.testing.assert_frame_equal(
            tables["decision_origins"].sort_values(SERIES_KEYS).reset_index(drop=True),
            result.decision_origins, check_dtype=False)
        return len(kept)

    def test_full_fixture(self):
        fc = _shared.fixture_data()[0]
        doc = workload(fc.loc[~fc.dataset.isin(FIT_ONLY_DATASETS), rt.TABLE_KEYS].drop_duplicates())
        result = rt.build_role_table(rt.workload_origins(doc))
        self.assertEqual(self.assert_same_as_engine(result, _shared.tables()), 22 * (480 + 30))
        self.assertEqual(by_horizon(result), {1: (22 * 160, 0, 22 * 320), 24: (22 * 10, 0, 22 * 20)})
        cache = rt.cache_origins(fc, FIXTURE_SCHEMA)
        pd.testing.assert_frame_equal(cache[~cache.dataset.isin(FIT_ONLY_DATASETS)].reset_index(drop=True),
                                      rt.workload_origins(doc))

    def test_cache_with_gap_and_excluded_series_through_the_producer_path(self):
        series = {"fx_flat/gap": list(range(48, 48 + 12 * 9, 12)),        # 2 validation, 1 gap, 6 evaluation
                  "fx_flat/all_gap": list(range(600, 600 + 6 * 9, 6)),    # every candidate is gap
                  "fx_flat/two": [900, 924]}                              # floor(2 / 3) = 0 candidates
        frame = flat_cache(series)
        omega = pd.DataFrame([dict(dataset="fx_flat", config="H", series_id=s, omega=0.5) for s in series])
        tables = build_tables(frame, PRODUCER_SCHEMA, FLAT_REPRESENTATIVES, FLAT_FAMILY_OF_MODEL, omega,
                              omega_col="omega", scales=stated_scales(frame))
        result = rt.build_role_table(rt.workload_origins(workload(origins(series, (24,), "fx_flat"))))
        self.assertEqual(self.assert_same_as_engine(result, tables), 9)
        self.assertEqual(by_horizon(result), {24: (2, 1 + 3, 6 + 6 + 2)})
        self.assertEqual(result.excluded["series_id"].tolist(), ["fx_flat/all_gap", "fx_flat/two"])
        self.assertEqual(set(result.excluded["reason"]), {RECORD_REASON})
        self.assertEqual(result.decision_origins["decision_origin"].tolist(), [84])
        # the same cache read directly, and as a handoff bundle
        pd.testing.assert_frame_equal(rt.build_role_table(rt.cache_origins(frame, PRODUCER_SCHEMA)).table,
                                      result.table)
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "bundle"
            write_bundle(bundle, frame)
            from_bundle, report = rt.handoff_origins(bundle, "current")
            pd.testing.assert_frame_equal(rt.build_role_table(from_bundle).table, result.table)
            self.assertEqual(report["series_origins"], 20)
            out = Path(tmp) / "out"
            run_cli(["export", "--handoff", bundle, "--variant", "current", "--out", out])
            side = json.loads((out / rt.SIDECAR).read_text(encoding="utf-8"))
            self.assertEqual(side["input"]["sha256"],
                             hashlib.sha256((bundle / "handoff.json").read_bytes()).hexdigest())
            self.assertEqual(side["counts"]["excluded_series_horizons"], 2)
            run_cli(["check", out, "--handoff", bundle, "--variant", "current"])


if __name__ == "__main__":
    unittest.main()
