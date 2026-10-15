"""The runner: exact read-back of saved tables, per-member match table, and a small end-to-end rebuild."""
import io
import json
import pathlib
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

import numpy as np
import pandas as pd

from decision_aware_tsfm.analysis.experiments.e3 import run as run_module
from decision_aware_tsfm.analysis.experiments.e3.design import F_VARIANTS
from decision_aware_tsfm.analysis.experiments.e3.run import main, match_by_member, read_table
from decision_aware_tsfm.analysis.experiments.e3.scoring import wilson_interval

from . import _shared

FLOAT_COLS = ["omega_levels", "omega_diffs", "skewness", "excess_kurtosis", "innov_variance"]


class ReadBack(unittest.TestCase):
    def test_saved_series_table_is_read_back_bit_for_bit(self):
        s = _shared.dataset()[1]
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "series_table.csv"
            s.to_csv(path, index=False)
            back = read_table(path)
            for col in FLOAT_COLS:
                with self.subTest(column=col):
                    np.testing.assert_array_equal(back[col].to_numpy(), s[col].to_numpy())

    def test_awkward_floats_survive_the_round_trip(self):
        # values on which a fast decimal parser is known to be off by one unit in the last place
        x = np.array([0.1213283718475503, 0.0965282301678676, 0.024800141679682697, 1 / 3, 2 / 3,
                      0.1 + 0.2, 1e-17, 5e-324, 1.7976931348623157e308, 0.30000000000000004])
        x = np.concatenate([x, np.random.default_rng(1).random(5000)])
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "t.csv"
            pd.DataFrame({"x": x}).to_csv(path, index=False)
            np.testing.assert_array_equal(read_table(path)["x"].to_numpy(), x)


class MatchByMember(unittest.TestCase):
    def test_hand_built_table(self):
        m = pd.DataFrame({
            "pair_kind": ["baseline", "baseline", "alternative", "alternative", "alternative",
                          "negative_control"],
            "structure_a": ["s1", "s2", "s1", "s1", "s2", "s1"],
            "member_b": ["gaussian_b", "gaussian_b", "t3", "t3", "t3", "gaussian"],
            "status": ["matched", "failed", "failed", "failed", "matched", "failed"],
            "omega_levels_abs_diff": [0.01, 0.05, 0.04, 0.06, 0.02, 0.3]})
        t = match_by_member(m).set_index(["structure", "member"])
        self.assertEqual(len(t), 6)                                  # 2 members x (2 structures + all)
        self.assertEqual((t.loc[("s1", "t3"), "n_pairs"], t.loc[("s1", "t3"), "n_failed"]), (2, 2))
        self.assertEqual(t.loc[("all", "t3"), "fail_rate"], 2 / 3)
        self.assertEqual(t.loc[("all", "gaussian_b"), "pair_kind"], "baseline")
        self.assertEqual((t.loc[("all", "gaussian_b"), "fail_lo"], t.loc[("all", "gaussian_b"), "fail_hi"]),
                         wilson_interval(1, 2))
        self.assertAlmostEqual(t.loc[("all", "t3"), "abs_diff_mean"], 0.04)
        self.assertNotIn("gaussian", t.index.get_level_values("member"))

    def test_real_two_seed_table_counts_every_design_pair_once(self):
        t = match_by_member(_shared.matches())
        per = t[t.structure != "all"]
        self.assertEqual(int(per.n_pairs.sum()), 4 * 2 * 7)
        self.assertTrue((t[t.structure == "all"].n_pairs == 8).all())


class EndToEnd(unittest.TestCase):
    def test_rebuild_from_saved_series_writes_every_table_and_never_touches_the_source(self):
        values, _ = _shared.dataset()
        rows = _shared.scored_rows()
        with tempfile.TemporaryDirectory() as tmp:
            src, out = pathlib.Path(tmp) / "src", pathlib.Path(tmp) / "out"
            src.mkdir()
            values[values.series_id.isin(rows.series_id)].to_parquet(src / "series_values.parquet", index=False)
            rows.to_csv(src / "series_table.csv", index=False)
            _shared.units_all().to_csv(src / "unit_costs.csv", index=False)
            before = {f.name: f.read_bytes() for f in src.iterdir()}
            # the population table takes several seconds to integrate; reuse the cached one
            with redirect_stdout(io.StringIO()) as log, \
                    mock.patch.object(run_module, "expected_delta_table", _shared.expected):
                main(["--out", str(out), "--series-from", str(src), "--reuse-units", "--skip-part-a",
                      "--no-series-replicates", "50"])
            self.assertEqual({f.name: f.read_bytes() for f in src.iterdir()}, before)
            want = {"series_table.csv", "match_pairs.csv", "match_groups.csv", "match_summary.csv",
                    "match_by_member.csv", "series_summary_by_member.csv", "expected_delta.csv",
                    "breakeven_error.csv", "f_win_count_by_error.csv", "breakeven_near_primary_error.csv",
                    "unit_costs.csv", "sign_pairs.csv", "sign_summary_by_tau.csv",
                    "sign_summary_by_member_tau.csv", "sign_summary_by_structure_tau.csv",
                    "sign_difference_vs_baseline.csv", "sign_by_population_status_tau.csv",
                    "sign_by_population_status_member_tau.csv", "sign_by_population_status_total.csv",
                    "member_delta_summary.csv", "member_delta_summary_by_structure.csv",
                    "verify_innovation_only.csv", "verify_no_series.csv",
                    "verify_no_series_vs_matched.csv", "verify_sign_by_match_status.csv",
                    "run_manifest.json"}
            self.assertEqual({f.name for f in out.iterdir()}, want)
            man = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(man["units_from_engine"])
            self.assertEqual(man["series_from"], str(src))
            self.assertIn("not a result about real models", man["label"])
            v = pd.read_csv(out / "verify_innovation_only.csv").set_index("variant")
            self.assertEqual(v.loc["all", "n_units"], 3 * 11 * len(F_VARIANTS))
            self.assertEqual(v.loc["all", "n_winner_mismatches"], 0)
            self.assertLess(v.loc["all", "max_abs_diff_J_F"], 1e-12)
            pd.testing.assert_frame_equal(read_table(out / "unit_costs.csv"),
                                          read_table(src / "unit_costs.csv"))
            self.assertIn("POPULATION", log.getvalue())
            wins = pd.read_csv(out / "f_win_count_by_error.csv").set_index("location_error")
            self.assertEqual(wins.loc[0.25, "n_F_wins_positive_error"], 18)

    def test_series_source_and_output_must_differ(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                main(["--out", tmp, "--series-from", tmp, "--reuse-units", "--skip-part-a"])


if __name__ == "__main__":
    unittest.main()
