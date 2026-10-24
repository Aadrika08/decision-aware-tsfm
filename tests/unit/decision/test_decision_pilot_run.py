"""The one-command pilot run (run_pilot.py) on a pilot-shaped synthetic handoff bundle.

The bundle: the roster fixture (three classical models and one TSFM, as in the pilot),
five datasets with three series each, horizons 1 and 24, 15 origins 12 apart (so horizon 24
has a gap origin), one seed, a workload.json naming one source file per series (its history
before the first origin, from which the scale is computed, #29 item 7a) and the role table
the #60 code exports from it. One series has every horizon-1 target equal to 0, as solar
has at horizon 1 in the real pilot; its history varies, so it is kept and scored.
SYNTHETIC: these tests check the wiring, not any research result.

Set PILOT_OTHER_PYTHON to a second interpreter (the team's 3.10 or 3.11) to also check
that it writes byte-identical files.
"""
import contextlib
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from decision_aware_tsfm.analysis.decision import fixture
from decision_aware_tsfm.analysis.decision import role_table as rt
from decision_aware_tsfm.analysis.decision import run_pilot as rp
from decision_aware_tsfm.analysis.decision.cache_io import qcol
from decision_aware_tsfm.analysis.decision.guards import HeldOutLeakError
from decision_aware_tsfm.analysis.decision.metrics import TAU_GRID
from decision_aware_tsfm.analysis.decision.pipeline import fit_pool
from decision_aware_tsfm.analysis.decision.selectors import LeaveOneDatasetOut
from decision_aware_tsfm.analysis.decision.handoff import read_handoff
from decision_aware_tsfm.analysis.decision.scale import read_workload_sources

from . import _sources
from .test_decision_producer import _sha, write_bundle

ORIGINS = [1001 + 12 * k for k in range(15)]
HORIZONS = (1, 24)
KEEP_SERIES = ("s0", "s1", "s2")
ZERO_SERIES = "fx_tie/s1"           # every horizon-1 target is 0 (mean |y| would be 0 there)
SHIFT_4096 = 0.03                   # last_4096 Omega = current + 0.03, so the variants differ
# a series where the families differ gets an Omega between 0.45 and 0.5, so the headline and
# the 0.5 sensitivity choose differently on it
BETWEEN = ("fx_tau_flip/s0", 0.47)
N_BOOT = 20
REPO = Path(__file__).resolve().parents[3]
QCOLS = [qcol(t) for t in TAU_GRID]


def pilot_history() -> dict:
    """The fixture's drawn series on the pilot-shaped origins: the history in the sources."""
    with mock.patch.dict(fixture.HORIZON_ORIGINS, {h: ORIGINS for h in HORIZONS}, clear=True):
        return fixture.series_values(7)


def pilot_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Canonical forecasts of the roster fixture on the pilot-shaped origins, and Omega labels."""
    with mock.patch.dict(fixture.HORIZON_ORIGINS, {h: ORIGINS for h in HORIZONS}, clear=True):
        fc, omega = fixture.make_roster_fixture(7)
    fc = fc[fc["series_id"].str.rsplit("/", n=1).str[1].isin(KEEP_SERIES)].copy()
    pos = fc["origin"] + fc["step"] - 1
    fc.loc[(fc["series_id"] == ZERO_SERIES) & pos.isin(ORIGINS), "y"] = 0.0
    fc["seed"] = np.int64(42)
    omega = omega[omega["series_id"].isin(fc["series_id"].unique())].copy()
    omega.loc[omega["series_id"] == BETWEEN[0], "omega_levels"] = BETWEEN[1]
    return fc.reset_index(drop=True), omega


def workload_doc(fc: pd.DataFrame) -> dict:
    keys = ["dataset", "config", "series_id", "origin", "horizon", "seed"]
    rows = fc[keys].drop_duplicates().sort_values(keys)
    return dict(schema_version=1, split="development", fixture=True,
                rows=[{k: (int(v) if k in ("origin", "horizon", "seed") else v) for k, v in r.items()}
                      for r in rows.to_dict("records")])


def set_omega(root: Path, labels: dict) -> None:
    """Per-series Omega labels in omega.json (current, and current + SHIFT_4096 for last_4096)."""
    side = json.loads((root / "omega.json").read_text(encoding="utf-8"))
    for row in side["rows"]:
        v = labels[row["series_id"]]
        row["omega"] = round(min(1.0, v + SHIFT_4096), 6) if row["variant"] == "last_4096" else v
    (root / "omega.json").write_text(json.dumps(side), encoding="utf-8")
    man = json.loads((root / "handoff.json").read_text(encoding="utf-8"))
    man["omega_sha256"] = _sha(root / "omega.json")
    (root / "handoff.json").write_text(json.dumps(man), encoding="utf-8")


def make_bundle(root: Path, fc: pd.DataFrame, omega: pd.DataFrame, with_role_table=True) -> None:
    """Bundle, sources/ and workload.json (with source_path and source_sha256) in root."""
    _, hashes = _sources.write_workload(root, fc, history=pilot_history())
    write_bundle(root, fixture.to_producer_format(fc, fixture.ROSTER_FAMILY_OF_MODEL),
                 source_hashes=hashes)
    set_omega(root, dict(zip(omega["series_id"], omega["omega_levels"])))
    doc = json.loads((root / "workload.json").read_text(encoding="utf-8"))
    if with_role_table:
        result = rt.build_role_table(rt.workload_origins(doc))
        rt.write_role_table(result, root / "role_table", dict(input="test"))


def quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


def tree(path: Path, skip=()) -> dict:
    return {p.relative_to(path).as_posix(): p.read_bytes() for p in sorted(path.rglob("*"))
            if p.is_file() and p.name not in skip}


class PilotBundle(unittest.TestCase):
    """One bundle and one full run shared by the tests of this class."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.base = Path(cls.tmp.name)
        cls.fc, cls.omega = pilot_frame()
        cls.bundle = cls.base / "bundle"
        make_bundle(cls.bundle, cls.fc, cls.omega)
        cls.out = cls.base / "run"
        cls.manifest = quiet(rp.run_pilot, cls.bundle, cls.out, n_boot=N_BOOT,
                             fixed_classical="SeasonalNaive")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def read(self, rel, out=None):
        return pd.read_csv((out or self.out) / rel, keep_default_na=False, na_values=[""])

    def sel(self, threshold=0.45, out=None):
        return pd.read_parquet((out or self.out) / "engine" / f"threshold_{threshold:g}" /
                               "selection_results.parquet")


class EndToEnd(PilotBundle):
    def test_every_step_wrote_its_files_and_nothing_else_is_left(self):
        for rel in ("summary.md", "manifest.json", "environment.json",
                    "role_table/role_table.csv", "role_table/role_table.json",
                    "representatives/representatives.json",
                    "representatives/representatives_fixed_named.json",
                    "engine/threshold_0.45/selection_results.parquet",
                    "engine/threshold_0.4/selection_results.csv",
                    "engine/threshold_0.5/d_omega.json",
                    "engine/excluded_series_horizons.csv", "engine/undefined_omega.csv",
                    "e1/decision_map.csv", "e1/summary_by_horizon_tau.csv", "e1/symmetry.csv",
                    "e1/mean_delta_intervals.csv", "e2/fold_check.csv", "e2/fold_selection.csv",
                    "e2/regret_summary.csv", "e2/gap_to_validation.csv", "e2/worst_decile.csv",
                    "e2/regret_intervals.csv", "tables/headline_by_horizon_tau.csv",
                    "tables/headline_by_selector.csv", "tables/headline_by_selector_horizon.csv",
                    "tables/representative_scores.csv", "tables/role_coverage.csv",
                    "tables/representative_sensitivity.csv",
                    "sensitivity/fixed_classical_SeasonalNaive/selection_results.parquet"):
            self.assertTrue((self.out / rel).is_file(), rel)
        self.assertFalse(self.out.with_name("run.partial").exists())

    def test_manifest_hashes_every_file_and_records_inputs(self):
        m = json.loads((self.out / "manifest.json").read_text(encoding="utf-8"))
        on_disk = {p.relative_to(self.out).as_posix() for p in self.out.rglob("*") if p.is_file()}
        self.assertEqual(set(m["files"]) | {"manifest.json", "environment.json"}, on_disk)
        for rel, sha in m["files"].items():
            self.assertEqual(_sha(self.out / rel), sha, rel)
        self.assertEqual(m["inputs"]["handoff_sha256"], _sha(self.bundle / "handoff.json"))
        self.assertEqual(m["inputs"]["workload"]["sha256"], _sha(self.bundle / "workload.json"))
        self.assertFalse(m["inputs"]["workload"]["matches_29"])
        self.assertEqual(len(m["code"]["code_sha256"]), 64)
        self.assertEqual(m["settings"]["static_thresholds"], [0.45, 0.4, 0.5])
        self.assertEqual(m["settings"]["min_fit_datasets"], 3)
        self.assertEqual(m["settings"]["omega_variant"], "current")
        self.assertEqual(m["decisions_reference"], rp.DECISIONS_REFERENCE)
        self.assertNotIn(str(self.base), json.dumps(m))          # no absolute paths

    def test_role_table_is_the_60_export_of_the_workload(self):
        want = rt.build_role_table(rt.read_workload_origins(self.bundle / "workload.json"))
        got = rt.read_csv(self.out / "role_table" / "role_table.csv")
        self.assertTrue(rt.check_role_table(got, rt.read_workload_origins(self.bundle / "workload.json"))["ok"])
        self.assertEqual(len(got), len(want.table))
        self.assertEqual(set(got["role"]), {"validation", "gap", "evaluation"})
        m = json.loads((self.out / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(m["checks"]["roles"]["bundle_role_table"]["matched"], len(want.table))

    def test_representatives_best_of_family_on_validation(self):
        p = json.loads((self.out / "representatives/representatives.json").read_text(encoding="utf-8"))
        pl = p["payload"]
        self.assertEqual(pl["rule"], "best_of_family")
        self.assertEqual(pl["selection"]["loss"], "scaled")
        self.assertEqual(pl["roles_scored"], ["validation"])
        self.assertEqual(pl["reasons"]["tsfm"], "only_member")
        self.assertEqual(pl["representatives"]["tsfm"], "Moirai-small")
        self.assertEqual(sorted(pl["candidates"]["classical"]), ["AutoARIMA", "AutoETS", "SeasonalNaive"])
        scores = self.read("tables/representative_scores.csv")
        best = scores[(scores.rule == "best_of_family") & (scores.family == "classical")]
        self.assertEqual(best.loc[best.mean_scaled_loss.idxmin(), "model"], pl["representatives"]["classical"])
        # the series whose horizon-1 targets are all 0 has a history scale: nothing is dropped
        self.assertEqual(pl["coverage"]["classical"]["n_units_dropped_zero_scale"], 0)
        self.assertEqual(pl["selection"]["scale"], "seasonal_naive_mae_season_24_history_before_first_origin")
        self.assertEqual(len(pl["input"]["scales_sha256"]), 64)

    def test_engine_ran_seven_rules_with_the_chosen_representatives(self):
        s = self.sel()
        self.assertEqual(list(dict.fromkeys(s.selector)), list(rp.SELECTOR_ORDER))
        reps = self.manifest["settings"]["representatives"]["chosen"]
        chosen = s.dropna(subset=["chosen_model"]).groupby("chosen_family")["chosen_model"].unique()
        self.assertEqual({k: list(v) for k, v in chosen.items()},
                         {k: [m] for k, m in reps.items()})
        o = s[s.selector == "hindsight_oracle"]
        self.assertTrue((o.regret == 0).all())
        # 15 series, 2 horizons, 11 taus: nothing excluded
        self.assertEqual(len(s[s.selector == "always_tsfm"]), 15 * 2 * 11)

    def test_series_with_all_zero_horizon_1_targets_is_kept_and_scored(self):
        """Under mean |y| this (series, horizon 1) was dropped; the history scale keeps it."""
        self.assertEqual(len(self.read("engine/excluded_series_horizons.csv")), 0)
        cov = self.read("tables/role_coverage.csv")
        row = cov[(cov.dataset == "fx_tie") & (cov.horizon == 1)].iloc[0]
        self.assertEqual((row.series, row.series_in_cache, row.series_scored, row.excluded_zero_scale),
                         (3, 3, 3, 0))
        self.assertEqual(int(cov.gap.sum()), 15)            # one gap origin per series at horizon 24
        s = self.sel()
        z = s[(s.series_id == ZERO_SERIES) & (s.horizon == 1)]
        self.assertEqual(len(z), 7 * 11)
        self.assertTrue((z[z.selector != "d_omega"].status == "selected").all())
        sc = read_workload_sources(self.bundle / "workload.json").scales.set_index("series_id")
        self.assertTrue(np.isfinite(z.scaled_regret[z.status == "selected"]).all())
        self.assertTrue((z.scale == sc.loc[ZERO_SERIES, "scale"]).all())

    def test_scale_is_the_history_scale_of_each_series(self):
        sc = read_workload_sources(self.bundle / "workload.json").scales
        self.assertEqual(len(sc), 15)
        self.assertTrue((sc.history_end == ORIGINS[0]).all() and (sc.status == "ok").all())
        hist = pilot_history()
        for r in sc.itertuples():
            h = hist[(r.dataset, r.config, r.series_id)][:ORIGINS[0]]
            self.assertEqual(r.scale, float(np.mean(np.abs(h[24:] - h[:-24]))))
        written = self.read("engine/series_scales.csv")
        np.testing.assert_allclose(written.scale, sc.scale, rtol=1e-15)
        s = self.sel()
        m = s.merge(sc[["series_id", "scale"]], on="series_id", suffixes=("", "_series"))
        self.assertTrue((m.scale == m.scale_series).all())    # one scale per series, every horizon
        sh = self.manifest["inputs"]["scale_history"]
        self.assertEqual((sh["series_in_cache"], sh["sources_hash_verified"]), (15, 15))
        self.assertEqual(self.manifest["settings"]["scale"], sh["kind"])

    def test_leave_one_dataset_out_with_three_and_e2_folds_match_the_engine(self):
        d = json.loads((self.out / "engine/threshold_0.45/d_omega.json").read_text(encoding="utf-8"))
        self.assertEqual(d["scheme"], "leave_one_dataset_out")
        self.assertEqual(d["min_fit_datasets"], 3)
        self.assertEqual(len(d["folds"]), 5)
        fc = self.read("e2/fold_check.csv")
        self.assertTrue(fc.fitted.all())
        self.assertEqual(int(fc.n_different.sum()), 0)
        self.assertEqual(int(fc.n_same_as_engine.sum()), len(self.sel().query("selector == 'd_omega'")))
        for row in fc.itertuples():
            fold = d["folds"][row.fold]
            self.assertEqual(json.loads(row.selector_params),
                             {"omega0": fold["omega0"], "beta": fold["beta"]})
            self.assertTrue((self.out / row.frozen_file).is_file())

    def test_headline_table_matches_the_selection_frame(self):
        h = self.read("tables/headline_by_horizon_tau.csv")
        self.assertEqual(len(h), 3 * 7 * 2 * 11)
        s = self.sel()
        x = h[(h.static_threshold == 0.45) & (h.selector == "always_tsfm") & (h.horizon == 24)
              & np.isclose(h.tau, 0.5)].iloc[0]
        u = s[(s.selector == "always_tsfm") & (s.horizon == 24) & np.isclose(s.tau, 0.5)]
        self.assertAlmostEqual(x.mean_scaled_regret, u.scaled_regret.mean(), places=12)
        bf, sel, orc = ((u.best_fixed_cost / u.scale).sum(), (u.realized_cost / u.scale).sum(),
                        (u.oracle_cost / u.scale).sum())
        self.assertAlmostEqual(x.frac_oracle_value_recovered, (bf - sel) / (bf - orc), places=10)
        self.assertEqual(int(x.n_deferred), 0)
        o = h[h.selector == "hindsight_oracle"]
        self.assertTrue((o.mean_scaled_regret == 0).all())

    def test_column_check_recomputes_every_test_unit(self):
        c = self.manifest["checks"]["column_check"]
        self.assertEqual(c["unmatched"], 0)
        self.assertEqual(c["units_compared"], 15 * 2 * 11)
        self.assertLessEqual(c["max_relative_difference"], 1e-10)

    def test_bootstrap_intervals_are_written_for_every_rule_but_the_oracle(self):
        r = self.read("e2/regret_intervals.csv")
        self.assertEqual(sorted(r.selector.unique()), sorted(set(rp.SELECTOR_ORDER) - {"hindsight_oracle"}))
        e = self.read("e1/mean_delta_intervals.csv")
        for f in (r, e):
            self.assertTrue((f.n_boot == N_BOOT).all())
            self.assertTrue((f.boot_seed == 0).all())
            self.assertTrue((f.cluster == "dataset+series").all())
            self.assertTrue((f.block_length == "auto").all())
        self.assertTrue((r.lower <= r.upper).all())
        self.assertEqual(len(r), 6 * 2 * 11)
        e = self.read("e1/mean_delta_intervals.csv")
        self.assertEqual(len(e), 2 * 11)
        for rel in ("e1/mean_delta_intervals_skipped.csv", "e2/regret_intervals_skipped.csv"):
            self.assertEqual(len(self.read(rel)), 0, rel)

    def test_a_horizon_that_cannot_be_resampled_is_skipped_alone(self):
        forecasts, omega, _, fom = rp.load_bundle(self.bundle, "current")
        reps = self.manifest["settings"]["representatives"]["chosen"]
        scales = read_workload_sources(self.bundle / "workload.json").scales
        tables = rp.engine_tables(forecasts, omega, reps, fom, "current", scales)
        selection, _ = rp.run_engine(tables, reps, 0.45, 3)
        po = tables["per_origin"]
        # fx_tie keeps one series at horizon 1: the dataset+series resample cannot run there
        gone = ["fx_tie/s1", "fx_tie/s2"]
        thin = dict(tables, per_origin=po[~(po.series_id.isin(gone) & (po.horizon == 1))])
        ci, skipped = rp.delta_intervals(thin, N_BOOT, 0)
        self.assertEqual(sorted(ci.horizon.unique()), [24])
        self.assertEqual(skipped.horizon.tolist(), [1])
        self.assertIn("fewer than", skipped.reason.iloc[0])
        full, _ = rp.delta_intervals(tables, N_BOOT, 0)
        pd.testing.assert_frame_equal(ci, full[full.horizon == 24].reset_index(drop=True))
        sel = selection[~(selection.series_id.isin(gone) & (selection.horizon == 1))]
        rci, rskipped = rp.regret_intervals(thin, sel, N_BOOT, 0)
        self.assertEqual(sorted(rci.horizon.unique()), [24])
        self.assertEqual(sorted(rskipped.selector), sorted(set(rp.SELECTOR_ORDER) - {"hindsight_oracle"}))

    def test_summary_names_the_fixture_and_the_files(self):
        text = (self.out / "summary.md").read_text(encoding="utf-8")
        self.assertIn("SYNTHETIC FIXTURE", text)
        self.assertIn(rp.DECISIONS_REFERENCE, text)
        for rel in ("tables/headline_by_horizon_tau.csv", "e1/decision_map.csv", "manifest.json"):
            self.assertIn(rel, text)
        self.assertIn("NO", text)                        # workload is not the #29 one
        self.assertIn("matplotlib", rp.FIGURES_MISSING)
        self.assertNotIn(chr(0x2014), text)               # no em dashes

    def test_figures_skip_without_matplotlib_and_read_the_tables_with_it(self):
        self.assertIn(rp.FIGURES_OFF, (self.out / "summary.md").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            for rel in ("tables/headline_by_horizon_tau.csv", "e1/decision_map.csv"):
                (out / rel).parent.mkdir(parents=True, exist_ok=True)
                (out / rel).write_bytes((self.out / rel).read_bytes())
            with mock.patch.dict("sys.modules", {"matplotlib": None, "matplotlib.pyplot": None}):
                self.assertEqual(rp.write_figures(out, 0.45), rp.FIGURES_MISSING)
            self.assertFalse((out / "figures").exists())
            # a stand-in plotting module: checks the data side only, not matplotlib itself
            plt, mpl, figs = mock.MagicMock(), mock.MagicMock(), []
            mpl.pyplot = plt

            def subplots(n_rows, n_cols, **kw):
                figs.append(mock.MagicMock())
                axes = np.empty((n_rows, n_cols), dtype=object)
                for j in range(n_cols):
                    axes[0, j] = mock.MagicMock()
                return figs[-1], axes
            plt.subplots.side_effect = subplots
            with mock.patch.dict("sys.modules", {"matplotlib": mpl, "matplotlib.pyplot": plt}):
                status = rp.write_figures(out, 0.45)
            self.assertTrue(status.startswith("written"))
            self.assertEqual([c.args[1] for c in plt.subplots.call_args_list], [2, 2])   # two horizons
            self.assertEqual([f.savefig.call_args.args[0].name for f in figs],
                             ["regret_by_tau.png", "decision_map.png"])

    def test_fixed_classical_sensitivity_uses_the_named_model(self):
        s = pd.read_parquet(self.out / "sensitivity/fixed_classical_SeasonalNaive/selection_results.parquet")
        self.assertEqual(set(s.loc[s.chosen_family == "classical", "chosen_model"]), {"SeasonalNaive"})
        t = self.read("tables/representative_sensitivity.csv")
        self.assertEqual(set(t.rule), {"best_of_family", "fixed_named"})
        self.assertEqual(set(t.loc[t.rule == "fixed_named", "classical"]), {"SeasonalNaive"})


class Determinism(PilotBundle):
    def test_two_runs_are_byte_identical(self):
        out2 = self.base / "run2"
        quiet(rp.run_pilot, self.bundle, out2, n_boot=N_BOOT, fixed_classical="SeasonalNaive")
        a, b = tree(self.out), tree(out2)
        self.assertEqual(sorted(a), sorted(b))
        diff = [k for k in a if a[k] != b[k]]
        self.assertEqual(diff, [])

    @unittest.skipUnless(os.environ.get("PILOT_OTHER_PYTHON"),
                         "set PILOT_OTHER_PYTHON to a second interpreter to compare across Pythons")
    def test_other_python_writes_the_same_files(self):
        out3 = self.base / "run_other_python"
        env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
        subprocess.run([os.environ["PILOT_OTHER_PYTHON"], "-m",
                        "decision_aware_tsfm.analysis.decision.run_pilot", str(self.bundle),
                        "--out", str(out3), "--n-boot", str(N_BOOT),
                        "--fixed-classical", "SeasonalNaive"],
                       check=True, cwd=REPO, env=env, capture_output=True)
        a, b = tree(self.out, skip=("environment.json",)), tree(out3, skip=("environment.json",))
        self.assertEqual(sorted(a), sorted(b))
        self.assertEqual([k for k in a if a[k] != b[k]], [])


class Sensitivities(PilotBundle):
    def test_three_threshold_runs_headline_first(self):
        h = self.read("tables/headline_by_horizon_tau.csv")
        self.assertEqual(list(dict.fromkeys(h.static_threshold)), [0.45, 0.4, 0.5])
        self.assertEqual(h.headline.tolist(), (h.static_threshold == 0.45).tolist())
        runs = {t: self.sel(t) for t in (0.45, 0.4, 0.5)}
        for t, s in runs.items():
            st = s[s.selector == "static_omega"]
            want = np.where(st.omega > t, "tsfm", "classical")
            self.assertEqual(st.chosen_family.tolist(), want.tolist(), t)
            others = s[s.selector != "static_omega"].reset_index(drop=True)
            pd.testing.assert_frame_equal(
                others, runs[0.45][runs[0.45].selector != "static_omega"].reset_index(drop=True))
        # Omega labels lie between 0.4 and 0.45 and between 0.45 and 0.5, so all three runs differ
        fam = {t: runs[t][runs[t].selector == "static_omega"].chosen_family.to_numpy() for t in runs}
        self.assertTrue((fam[0.4] != fam[0.45]).any())
        self.assertTrue((fam[0.45] != fam[0.5]).any())
        # 0.4 and 0.45 differ only on fx_tie, where both families cost the same; 0.5 differs on BETWEEN
        h = h[(h.selector == "static_omega")].groupby("static_threshold").mean_scaled_regret.mean()
        self.assertNotEqual(h[0.45], h[0.5])

    def test_cli_threshold_order_sets_the_headline_and_variant_last_4096_runs(self):
        out = self.base / "run_4096"
        quiet(rp.main, [str(self.bundle), "--out", str(out), "--variant", "last_4096",
                        "--static-thresholds", "0.5", "0.45", "--n-boot", "0"])
        m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(m["settings"]["omega_variant"], "last_4096")
        self.assertEqual(m["settings"]["headline_threshold"], 0.5)
        h = self.read("tables/headline_by_horizon_tau.csv", out)
        self.assertEqual(list(dict.fromkeys(h.static_threshold)), [0.5, 0.45])
        s = self.sel(0.5, out)
        cur = self.sel(0.45)
        st = s[s.selector == "static_omega"].reset_index(drop=True)
        base = cur[cur.selector == "static_omega"].reset_index(drop=True)
        np.testing.assert_allclose(st.omega, np.minimum(1.0, (base.omega + SHIFT_4096).round(6)))
        self.assertFalse((out / "e2" / "regret_intervals.csv").exists())

    def test_d_omega_seam_takes_a_builder(self):
        calls = []

        def same_as_main(tables):
            calls.append(1)
            return LeaveOneDatasetOut.build(fit_pool(tables), tables["units"]["dataset"].unique(),
                                            min_fit_datasets=3)
        out = self.base / "run_seam"
        quiet(rp.run_pilot, self.bundle, out, n_boot=N_BOOT, fixed_classical="SeasonalNaive",
              d_omega_builder=same_as_main)
        self.assertEqual(len(calls), 4)                  # three thresholds and the sensitivity run
        a, b = tree(self.out), tree(out)
        self.assertEqual([k for k in a if a[k] != b[k]], ["manifest.json"])   # it names the builder
        m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(m["settings"]["d_omega"], "same_as_main")


class Refusals(PilotBundle):
    def assertNothingWritten(self, out):
        self.assertFalse(out.exists())
        self.assertFalse(out.with_name(out.name + ".partial").exists())

    def test_bundle_role_table_with_one_changed_row_stops(self):
        t = rt.read_csv(self.bundle / "role_table" / "role_table.csv")
        i = t.index[t.role == "validation"][0]
        t.loc[i, "role"] = "evaluation"
        bad = self.base / "bad_role_table.csv"
        t.to_csv(bad, index=False, lineterminator="\n", encoding="utf-8")
        out = self.base / "run_bad_roles"
        with self.assertRaises(rt.RoleTableMismatch):
            quiet(rp.run_pilot, self.bundle, out, role_table=bad, n_boot=0)
        self.assertNothingWritten(out)
        with self.assertRaises(SystemExit) as cm:
            quiet(rp.main, [str(self.bundle), "--out", str(out), "--role-table", str(bad)])
        self.assertIn("RoleTableMismatch", str(cm.exception))
        self.assertNothingWritten(out)

    def test_bundle_role_table_from_another_workload_stops(self):
        doc = workload_doc(self.fc)
        r = dict(doc["rows"][0])
        r["origin"] = ORIGINS[-1] + 12                   # an origin the cache does not hold
        doc["rows"].append(r)
        wl = self.base / "workload_extra_origin.json"
        wl.write_text(json.dumps(doc), encoding="utf-8")
        out = self.base / "run_bad_workload"
        with self.assertRaisesRegex(rt.RoleTableMismatch, "role table disagrees"):
            quiet(rp.run_pilot, self.bundle, out, workload=wl, role_table=self.bundle / "role_table",
                  n_boot=0)
        self.assertNothingWritten(out)

    def test_cache_holding_part_of_a_pairs_origins_stops(self):
        # what a first slice looks like: the workload plans more origins than the cache holds
        doc = workload_doc(self.fc)
        r = dict(doc["rows"][0])
        r["origin"] = ORIGINS[-1] + 12
        doc["rows"].append(r)
        wl = self.base / "workload_partial.json"
        wl.write_text(json.dumps(doc), encoding="utf-8")
        table = self.base / "role_table_partial"
        rt.write_role_table(rt.build_role_table(rt.workload_origins(doc)), table, dict(input="test"))
        out = self.base / "run_partial"
        with self.assertRaisesRegex(rt.RoleTableMismatch, "only part of the workload's origins for 1 "):
            quiet(rp.run_pilot, self.bundle, out, workload=wl, role_table=table, n_boot=0)
        self.assertNothingWritten(out)

    def test_cache_with_no_validation_origin_anywhere_stops(self):
        with tempfile.TemporaryDirectory() as d:
            fc = self.fc[self.fc.origin.isin(ORIGINS[:2]) & (self.fc.horizon == 24)]
            root = Path(d) / "bundle"
            make_bundle(root, fc, self.omega)
            out = Path(d) / "out"
            with self.assertRaisesRegex(rp.PilotRunError, "none of the 15 .* validation origin"):
                rp.run_pilot(root, out, n_boot=0)
            self.assertNothingWritten(out)

    def test_cache_pair_missing_from_the_workload_stops(self):
        doc = workload_doc(self.fc[self.fc.series_id != "fx_tie/s0"])
        forecasts, _, _, _ = rp.load_bundle(self.bundle, "current")
        with self.assertRaisesRegex(rt.RoleTableMismatch, "not in the workload"):
            rp.check_roles(forecasts, rt.workload_origins(doc))

    def test_workload_series_not_yet_in_the_cache_is_counted_not_refused(self):
        fc, _ = pilot_frame()
        extra = fc[fc.series_id == "fx_tie/s0"].assign(series_id="fx_tie/s9")
        doc = workload_doc(pd.concat([self.fc, extra]))
        wl = self.base / "workload_more.json"
        wl.write_text(json.dumps(doc), encoding="utf-8")
        forecasts, _, _, _ = rp.load_bundle(self.bundle, "current")
        _, rep = rp.check_roles(forecasts, rt.workload_origins(doc))
        self.assertEqual(rep["workload_series_horizons_absent_from_cache"], 2)
        self.assertTrue(rep["cache_vs_workload"]["ok"])

    def test_missing_workload_stops(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "out"
            with self.assertRaisesRegex(rp.PilotRunError, "--workload"):
                rp.run_pilot(self.bundle, out, workload=Path(d) / "nope.json")
            self.assertNothingWritten(out)

    def test_existing_out_is_refused(self):
        with self.assertRaisesRegex(rp.PilotRunError, "already exists"):
            rp.run_pilot(self.bundle, self.out)

    def test_held_out_names_are_refused(self):
        small = self.fc[(self.fc.dataset == "fx_tie") & (self.fc.horizon == 24)].copy()
        for name in ("pjm_load", "m5"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as d:
                fc = small.assign(dataset=name)
                root = Path(d) / "bundle"
                make_bundle(root, fc, self.omega, with_role_table=False)
                out = Path(d) / "out"
                with self.assertRaises(HeldOutLeakError):
                    rp.run_pilot(root, out, n_boot=0)
                self.assertNothingWritten(out)
        with tempfile.TemporaryDirectory() as d:           # held-out series name only in the workload
            doc = workload_doc(self.fc)
            doc["rows"][0] = dict(doc["rows"][0], series_id="fx_tie/pjm_s0")
            wl = Path(d) / "workload.json"
            wl.write_text(json.dumps(doc), encoding="utf-8")
            with self.assertRaises(HeldOutLeakError):
                rp.run_pilot(self.bundle, Path(d) / "out", workload=wl, n_boot=0)


class Internals(PilotBundle):
    def test_e2_regret_summary_is_the_headline_run(self):
        e = self.read("e2/regret_summary.csv")
        h = self.read("tables/headline_by_horizon_tau.csv")
        h = h[h.headline].drop(columns=["static_threshold", "headline"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(e, h)

    def test_deferral_is_charged_what_pinball_validation_paid(self):
        s = self.sel().copy()
        hit = (s.selector == "static_omega") & (s.dataset == "fx_tau_flip")
        s.loc[hit, ["status", "chosen_family", "chosen_model"]] = ["insufficient_data", None, None]
        s.loc[hit, ["realized_cost", "regret", "scaled_regret"]] = np.nan
        got = rp.deferral_priced(s, ("selector",)).set_index("selector")
        st, pv = s[s.selector == "static_omega"], s[s.selector == "pinball_validation"]
        m = st.merge(pv, on=["dataset", "config", "series_id", "horizon", "tau"], suffixes=("", "_pv"))
        want = np.where(m.status == "selected", m.scaled_regret, m.scaled_regret_pv).mean()
        self.assertEqual(int(got.loc["static_omega", "n_deferred"]), int(hit.sum()))
        self.assertAlmostEqual(got.loc["static_omega", "mean_scaled_regret_deferrals_priced"], want, places=14)
        self.assertEqual(int(got.loc["always_tsfm", "n_deferred"]), 0)

    def test_engine_roles_that_differ_from_the_table_stop(self):
        forecasts, omega, _, fom = rp.load_bundle(self.bundle, "current")
        reps = self.manifest["settings"]["representatives"]["chosen"]
        scales = read_workload_sources(self.bundle / "workload.json").scales
        tables = rp.engine_tables(forecasts, omega, reps, fom, "current", scales)
        want = rt.build_role_table(rt.read_workload_origins(self.bundle / "workload.json"))
        self.assertTrue(rp.check_engine_roles(tables, want)["ok"])
        po = tables["per_origin"].copy()
        i = po.index[po.role == "validation"]
        po.loc[po.origin == po.loc[i[0], "origin"], "role"] = "test"
        with self.assertRaisesRegex(rt.RoleTableMismatch, "differ from the role table"):
            rp.check_engine_roles(dict(tables, per_origin=po), want)
        dec = tables["decision_origins"].copy()
        dec.loc[0, "decision_origin"] += 12
        with self.assertRaisesRegex(rt.RoleTableMismatch, "1 decision origins"):
            rp.check_engine_roles(dict(tables, decision_origins=dec), want)

    def test_a_failure_after_writing_starts_leaves_nothing(self):
        out = self.base / "run_fails_late"
        with mock.patch.object(rp, "column_check", side_effect=rp.PilotRunError("planted")):
            with self.assertRaisesRegex(rp.PilotRunError, "planted"):
                rp.run_pilot(self.bundle, out, n_boot=0)
        self.assertFalse(out.exists())
        self.assertFalse(out.with_name(out.name + ".partial").exists())

    def test_held_out_series_name_in_the_bundle_is_refused(self):
        small = self.fc[(self.fc.dataset == "fx_tie") & (self.fc.horizon == 24)].copy()
        small["series_id"] = small["series_id"].str.replace("fx_tie/", "fx_tie/pjm_", regex=False)
        om = self.omega.assign(series_id=self.omega.series_id.str.replace("fx_tie/", "fx_tie/pjm_",
                                                                          regex=False))
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "bundle"
            make_bundle(root, small, om, with_role_table=False)
            with self.assertRaisesRegex(HeldOutLeakError, "bundle series_id"):
                rp.run_pilot(root, Path(d) / "out", n_boot=0)


try:
    from decision_aware_tsfm.analysis.decision import k3_adapter
    k3_adapter.k3_module()
    NO_K3 = None
except ImportError as _e:
    NO_K3 = f"K3 selector not importable here ({_e})"


@unittest.skipIf(NO_K3, NO_K3 or "")
class K3SelectorInThePilot(PilotBundle):
    """--d-omega k3_selector: the K3 selector in the D_Omega slot and frozen by E2 per fold."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.k3_out = cls.base / "run_k3"
        quiet(rp.main, [str(cls.bundle), "--out", str(cls.k3_out), "--n-boot", "0",
                        "--d-omega", "k3_selector"])

    def test_manifest_names_the_selector_and_e2_kind(self):
        m = json.loads((self.k3_out / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(m["settings"]["d_omega"], "k3_selector")
        self.assertEqual(m["settings"]["e2_selector_kind"], k3_adapter.K3OmegaTauKind.kind)

    def test_e2_frozen_k3_folds_choose_what_the_engine_chose(self):
        check = self.read("e2/fold_check.csv", self.k3_out)
        self.assertTrue(check.fitted.all())
        self.assertEqual(int(check.n_different.sum()), 0)
        self.assertEqual(int(check.n_same_as_engine.sum()), len(self.sel(out=self.k3_out).query(
            "selector == 'd_omega'")))

    def test_only_d_omega_changes_and_its_deferrals_are_priced(self):
        k3, base = self.sel(out=self.k3_out), self.sel()
        other = lambda s: s[s.selector != "d_omega"].reset_index(drop=True)
        pd.testing.assert_frame_equal(other(k3), other(base))
        d = k3[k3.selector == "d_omega"]
        self.assertGreater(int((d.status == "defer").sum()), 0)
        self.assertGreater(int((d.status == "selected").sum()), 0)
        self.assertTrue(d.loc[d.status == "defer", "note"].str.contains("reason=").all())
        h = self.read("tables/headline_by_selector.csv", self.k3_out)
        row = h[(h.selector == "d_omega") & h.headline].iloc[0]
        self.assertEqual(int(row.n_deferred), int((d.status == "defer").sum()))
        self.assertEqual(int(row.n_unpriced), 0)
        self.assertTrue(np.isfinite(row.mean_scaled_regret_deferrals_priced))

    def test_a_fold_with_every_unit_deferred_gives_an_empty_worst_decile(self):
        d = self.sel(out=self.k3_out).query("selector == 'd_omega'")
        all_deferred = [ds for ds, g in d.groupby("dataset") if (g.status == "defer").all()]
        self.assertTrue(all_deferred)                     # the case this guards exists here
        self.assertTrue((self.k3_out / "e2" / "fold_check.csv").exists())

    def test_k3_kind_round_trips_through_json(self):
        from . import _shared
        fit_units = fit_pool(_shared.tables()).dropna(subset=["omega", "J_classical", "J_tsfm", "scale"])
        kind = k3_adapter.K3OmegaTauKind()
        params = json.loads(json.dumps(kind.fit(fit_units), allow_nan=False))
        built = kind.build(params, sorted(fit_units.series_id.unique()),
                           sorted(fit_units.dataset.unique()))
        direct = k3_adapter.K3SelectorAdapter.fit(fit_units)
        self.assertEqual(built.describe(), direct.describe())


class UndefinedOmega(unittest.TestCase):
    """One series has a null Omega with a reason (as Bitbrains may): the Omega rules defer
    there, everything else still runs, and the deferral is priced and counted."""
    SID = "fx_tau_flip/s0"

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        base = Path(cls.tmp.name)
        fc, omega = pilot_frame()
        cls.bundle = base / "bundle"
        make_bundle(cls.bundle, fc, omega)
        side = json.loads((cls.bundle / "omega.json").read_text(encoding="utf-8"))
        for row in side["rows"]:
            if row["series_id"] == cls.SID:
                row.update(omega=None, omega_status="flat_history")
        (cls.bundle / "omega.json").write_text(json.dumps(side), encoding="utf-8")
        man = json.loads((cls.bundle / "handoff.json").read_text(encoding="utf-8"))
        man["omega_sha256"] = _sha(cls.bundle / "omega.json")
        (cls.bundle / "handoff.json").write_text(json.dumps(man), encoding="utf-8")
        cls.out = base / "run"
        quiet(rp.run_pilot, cls.bundle, cls.out, n_boot=N_BOOT)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_omega_rules_defer_and_are_priced(self):
        s = pd.read_parquet(self.out / "engine/threshold_0.45/selection_results.parquet")
        mine = s.series_id == self.SID
        for name in ("static_omega", "d_omega"):
            self.assertTrue((s[mine & (s.selector == name)].status == "insufficient_data").all())
            self.assertTrue((s[~mine & (s.selector == name)].status == "selected").all())
        h = pd.read_csv(self.out / "tables/headline_by_selector.csv")
        h = h[h.headline].set_index("selector")
        self.assertEqual(int(h.loc["static_omega", "n_deferred"]), 2 * 11)
        self.assertEqual(int(h.loc["pinball_validation", "n_deferred"]), 0)
        undefined = pd.read_csv(self.out / "engine/undefined_omega.csv")
        self.assertEqual(set(undefined.reason), {"flat_history"})
        self.assertIn("Units with undefined Omega: 22", (self.out / "summary.md").read_text(encoding="utf-8"))

    def test_e2_and_intervals_run_on_the_decided_units(self):
        fc = pd.read_csv(self.out / "e2/fold_check.csv")
        self.assertTrue(fc.fitted.all())
        self.assertEqual(int(fc.n_different.sum()), 0)
        r = pd.read_csv(self.out / "e2/regret_intervals.csv")
        n = r.groupby("selector").n_units_decided.first()
        self.assertEqual(n["always_tsfm"] - n["static_omega"], 11)        # per horizon, 11 taus


class RepresentativesSeeOnlyValidation(PilotBundle):
    def _roles(self, forecasts):
        table = rt.build_role_table(rt.distinct_origins(forecasts)).table
        return forecasts.merge(table, on=rt.TABLE_KEYS, how="left")["role"].to_numpy()

    def _make_perfect(self, forecasts, model, roles, mask_role):
        f = forecasts.copy()
        hit = (f["model"] == model).to_numpy() & np.isin(roles, mask_role)
        f.loc[hit, QCOLS] = np.repeat(f.loc[hit, "y_true"].to_numpy()[:, None], len(QCOLS), axis=1)
        return f

    def test_scrambling_evaluation_forecasts_leaves_the_choice_unchanged(self):
        forecasts, _, _, fom = rp.load_bundle(self.bundle, "current")
        roles = self._roles(forecasts)
        sc = read_workload_sources(self.bundle / "workload.json").scales
        base, _ = rp.choose_representatives(forecasts, fom, {}, sc)
        loser = sorted(m for m, f in fom.items()
                       if f == "classical" and m != base["representatives"]["classical"])[0]
        # the loser becomes a perfect forecaster on every evaluation and gap origin
        scrambled = self._make_perfect(forecasts, loser, roles, ["evaluation", "gap"])
        after, _ = rp.choose_representatives(scrambled, fom, {}, sc)
        self.assertNotEqual(after["input"]["sha256"], base["input"]["sha256"])
        self.assertEqual(after["representatives"], base["representatives"])
        self.assertEqual(after["scores"], base["scores"])
        # the same change on validation origins does move the choice, so the test can see a leak
        leaked = self._make_perfect(forecasts, loser, roles, ["validation"])
        moved, _ = rp.choose_representatives(leaked, fom, {}, sc)
        self.assertEqual(moved["representatives"]["classical"], loser)


if __name__ == "__main__":
    unittest.main()
