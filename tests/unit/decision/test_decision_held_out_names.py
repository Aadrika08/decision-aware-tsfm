"""The held-out name rule: M5 and PJM names are refused, and a name that only happens to
contain the letters "m5" inside a word (a channel suffix such as "_dim5") is not."""
import unittest

from decision_aware_tsfm.analysis.decision.guards import (
    HELD_OUT_TOKENS, HeldOutLeakError, assert_no_held_out, held_out_tokens_in, is_held_out)

HELD_OUT = [
    "m5", "M5", "M5_Retail", "m5_daily", "m5/daily", "gift_m5", "gift-m5-daily", "Salesforce/M5",
    "data\\M5\\sales", " m5 ", "m5.parquet", "kaggle.m5", "2m5", "m50", "m5m5",
    "m5_store_1", "store_m5", "store_1_m5", "x_M5_y", "1m5", "_m5", "M5a", "(m5)", "a.M5",
    "pjm", "PJM", "PJM_Electricity", "pjm_hourly", "gift_pjm", "energy/PJM/H", "pjmw", "xpjm", "dimpjm",
    "PJM_hourly", "mypjmdata", "MyPjMData", "1pjm", "pjm5",
]
# the first 19 contain the letters "m5" straight after another letter; the rest have no token
NOT_HELD_OUT = [
    "ett1_H__item_0_dim5", "ett1_15T__item_0_dim5", "ett2_H__item_0_dim5", "ett2_15T__item_0_dim5",
    "jena_weather_H__item_0_dim5", "jena_weather_10T__item_0_dim5",
    "bizitobs_l2c_H__L2C_dim5", "bizitobs_l2c_5T__L2C_dim5",
    "x_dim50", "x_dim59", "DIM5", "item5", "team5", "stream5_H",
    "ram5gb_cluster", "am5", "AM5", "x_dim5_dim5", "jm5x_pj",
    "ett1", "H", "m4_daily", "m4", "solar_10T__item_5", "item_50", "5m", "m_5", "pj_m", "",
    "x_dim15", "p_jm", "pj",
]


class HeldOutNames(unittest.TestCase):
    def test_tokens_are_unchanged(self):
        self.assertEqual(HELD_OUT_TOKENS, ("m5", "pjm"))

    def test_held_out_names_are_refused(self):
        for name in HELD_OUT:
            with self.subTest(name=name):
                self.assertTrue(is_held_out(name))

    def test_m5_inside_a_word_is_not_held_out(self):
        for name in NOT_HELD_OUT:
            with self.subTest(name=name):
                self.assertFalse(is_held_out(name))

    def test_a_held_out_name_is_still_refused_when_a_channel_suffix_follows(self):
        for name in ("M5_Retail__item_0_dim5", "m5_daily__item_3_dim0", "pjm_H__item_0_dim5",
                     "x_dim5/m5", "dim5_m5", "dim5_dim5_m5", "dim5m5_m5", "x_dim5_pjm", "am5/M5"):
            with self.subTest(name=name):
                self.assertTrue(is_held_out(name))

    def test_known_gap_run_together_name_is_not_caught(self):
        """The accepted limit of the rule (see the 2026-10-12 note): a held-out name written
        with a letter straight before "m5" passes. If the rule is tightened, change this test."""
        for name in ("retailm5", "RetailM5", "walmartm5_daily"):
            with self.subTest(name=name):
                self.assertFalse(is_held_out(name))

    def test_tokens_found_are_named_in_token_order(self):
        cases = {
            "M5_Retail": ["m5"], "PJM_Electricity": ["pjm"],
            "pjm_and_m5": ["m5", "pjm"], "m5/pjm/m5": ["m5", "pjm"],
            "pjm_H__item_0_dim5": ["pjm"],          # the channel suffix is not an "m5" match
            "dim5_m5": ["m5"],
            "ett1_H__item_0_dim5": [], "ram5gb_cluster": [], "": [],
        }
        for name, expected in cases.items():
            with self.subTest(name=name):
                self.assertEqual(held_out_tokens_in(name), expected)
                self.assertEqual(is_held_out(name), bool(expected))

    def test_non_string_and_empty_names(self):
        self.assertTrue(is_held_out(b"m5".decode()))
        for name in (5, 5.0, None, float("nan"), "", " ", True):
            with self.subTest(name=name):
                self.assertFalse(is_held_out(name))
                self.assertIs(type(is_held_out(name)), bool)
        self.assertIs(is_held_out("M5"), True)

    def test_assert_names_every_hit_and_only_hits(self):
        assert_no_held_out(NOT_HELD_OUT, "clean")                # no error
        with self.assertRaises(HeldOutLeakError) as cm:
            assert_no_held_out(["ett1_H__item_0_dim5", "M5_Retail", "ett1"], "mixed")
        msg = str(cm.exception)
        self.assertIn("M5_Retail", msg)
        self.assertNotIn("dim5", msg)


if __name__ == "__main__":
    unittest.main()
