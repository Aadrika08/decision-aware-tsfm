"""Family aggregation on a tiny hand-built table. Numbers here are made up."""
import unittest

import pandas as pd

from decision_aware_tsfm.analysis.decision.guards import HeldOutLeakError
from decision_aware_tsfm.analysis.k2.families import (
    FamilyMappingError, GROUPS, family_table, model_to_family, roster_report)

GROUPS_TINY = {"fam_a": ["a1", "a2", "a3"], "fam_b": ["b1"]}


def tiny():
    """One base dataset with two configurations; three models in fam_a, one in fam_b."""
    rows = [
        ("d/H/short", "a1", 1.0), ("d/H/long", "a1", 9.0),      # model mean 5
        ("d/H/short", "a2", 2.0), ("d/H/long", "a2", 4.0),      # model mean 3
        ("d/H/short", "a3", 6.0), ("d/H/long", "a3", 8.0),      # model mean 7
        ("d/H/short", "b1", 10.0), ("d/H/long", "b1", 20.0),
        ("d/H/short", "stranger", 1e6), ("d/H/long", "stranger", 1e6),
    ]
    df = pd.DataFrame(rows, columns=["dataset", "model", "score"])
    df["base"] = "d"
    return df


class FamilyAggregation(unittest.TestCase):
    def cell(self, within, pooling, unit="base", row="d"):
        t = family_table(tiny(), "score", within, unit=unit, pooling=pooling, groups=GROUPS_TINY)
        return t.loc[row, "fam_a"]

    def test_mean_is_the_same_under_both_poolings(self):
        self.assertAlmostEqual(self.cell("mean", "rows"), 5.0)      # (1+9+2+4+6+8)/6
        self.assertAlmostEqual(self.cell("mean", "models"), 5.0)    # (5+3+7)/3

    def test_median_over_rows_and_over_models(self):
        self.assertAlmostEqual(self.cell("median", "rows"), 5.0)    # median of 1,2,4,6,8,9
        self.assertAlmostEqual(self.cell("median", "models"), 5.0)  # median of 5,3,7
        # They are different quantities in general: change one row and only one moves.
        df = tiny()
        df.loc[(df.model == "a1") & (df.dataset == "d/H/long"), "score"] = 90.0   # a1 mean 45.5
        rows = family_table(df, "score", "median", pooling="rows", groups=GROUPS_TINY)
        models = family_table(df, "score", "median", pooling="models", groups=GROUPS_TINY)
        self.assertAlmostEqual(rows.loc["d", "fam_a"], 5.0)         # 1,2,4,6,8,90
        self.assertAlmostEqual(models.loc["d", "fam_a"], 7.0)       # 45.5,3,7

    def test_best_over_rows_is_a_single_configuration_and_over_models_is_a_model(self):
        self.assertAlmostEqual(self.cell("min", "rows"), 1.0)       # a1 on its easy configuration
        self.assertAlmostEqual(self.cell("min", "models"), 3.0)     # a2, the best model overall

    def test_at_configuration_level_the_poolings_agree(self):
        for within, expected in (("mean", 3.0), ("median", 2.0), ("min", 1.0)):
            with self.subTest(within=within):
                for pooling in ("rows", "models"):
                    self.assertAlmostEqual(
                        self.cell(within, pooling, unit="config", row="d/H/short"), expected)

    def test_model_outside_the_mapping_is_ignored(self):
        with_stranger = family_table(tiny(), "score", groups=GROUPS_TINY)
        without = family_table(tiny()[lambda d: d.model != "stranger"], "score", groups=GROUPS_TINY)
        pd.testing.assert_frame_equal(with_stranger, without)
        self.assertEqual(list(with_stranger.columns), ["fam_a", "fam_b"])
        report = roster_report(tiny(), GROUPS_TINY)
        self.assertEqual(report["models_ignored"], ["stranger"])
        self.assertEqual(report["n_models_ignored"], 1)

    def test_mapped_model_missing_from_results_is_reported(self):
        report = roster_report(tiny()[lambda d: d.model != "a3"], GROUPS_TINY)
        self.assertEqual(report["families"]["fam_a"]["absent_from_results"], ["a3"])
        self.assertEqual(report["families"]["fam_a"]["n_present"], 2)

    def test_model_in_two_families_is_an_error(self):
        with self.assertRaises(FamilyMappingError):
            model_to_family({"x": ["m"], "y": ["m"]})

    def test_real_mapping_is_a_function(self):
        mapping = model_to_family()
        self.assertEqual(len(mapping), sum(len(v) for v in GROUPS.values()))

    def test_bad_arguments_are_refused(self):
        for kwargs in ({"within_family": "max"}, {"pooling": "series"}, {"unit": "series"}):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                family_table(tiny(), "score", groups=GROUPS_TINY, **kwargs)

    def test_held_out_unit_is_refused(self):
        df = tiny()
        df["base"] = "m5_made_up_name"          # name only; the numbers are made up
        with self.assertRaises(HeldOutLeakError):
            family_table(df, "score", groups=GROUPS_TINY)


if __name__ == "__main__":
    unittest.main()
