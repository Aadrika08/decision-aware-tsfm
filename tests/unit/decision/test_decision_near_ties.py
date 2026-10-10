"""Near ties: the oracle names the family with the exact minimum, validation keeps its tolerance."""
import unittest

import numpy as np
import pandas as pd

from decision_aware_tsfm.analysis.decision.evaluate import best_fixed, run_selectors
from decision_aware_tsfm.analysis.decision.fixture import (
    FIXTURE_STATIC_THRESHOLD as THRESHOLD, REPRESENTATIVES)
from decision_aware_tsfm.analysis.decision.pipeline import seven_selectors
from decision_aware_tsfm.analysis.decision.schema import SchemaError, validate_selection_frame
from decision_aware_tsfm.analysis.decision.selectors import (
    TIE_RTOL, HindsightOracle, NotFitted, PinballValidation, argmin_family, exact_argmin_family)

GAP = 5e-13                      # inside the 1e-12 tolerance, so argmin_family calls it a tie


def _frames(j_classical, j_tsfm, scale=1.0, val=None):
    """Hand-built engine tables: one unit per cost pair, all in one (horizon, tau) group."""
    jc, jt = np.asarray(j_classical, dtype=float), np.asarray(j_tsfm, dtype=float)
    keys = pd.DataFrame({"dataset": "d", "config": "c", "series_id": [f"s{i}" for i in range(jc.size)],
                         "horizon": 1, "tau": 0.5})
    vc, vt = (jc, jt) if val is None else val
    return dict(units=keys.assign(omega=0.5),
                val=keys.assign(J_classical=vc, J_tsfm=vt, crps_classical=vc, crps_tsfm=vt),
                test=keys.assign(J_classical=jc, J_tsfm=jt, scale=scale, n_origins=4),
                n_val=keys[["dataset", "config", "series_id", "horizon"]].assign(n_validation_origins=2))


def _run(selectors, t):
    return run_selectors(selectors, t["units"], t["val"], t["test"], REPRESENTATIVES, t["n_val"])


class OracleOnNearTies(unittest.TestCase):
    def test_review_example_tsfm_lower_inside_the_tolerance(self):
        # the case from the #44 review: this raised "oracle has non-zero regret against itself"
        res = _run([HindsightOracle()], _frames([1.0], [1.0 - GAP]))
        r = res.iloc[0]
        self.assertEqual((r.chosen_family, r.oracle_family), ("tsfm", "tsfm"))
        self.assertEqual(r.oracle_cost, 1.0 - GAP)
        self.assertEqual(r.regret, 0.0)

    def test_classical_lower_inside_the_tolerance(self):
        r = _run([HindsightOracle()], _frames([1.0 - GAP], [1.0])).iloc[0]
        self.assertEqual((r.chosen_family, r.oracle_family), ("classical", "classical"))
        self.assertEqual(r.oracle_cost, 1.0 - GAP)
        self.assertEqual(r.regret, 0.0)

    def test_exact_tie_goes_to_classical(self):
        r = _run([HindsightOracle()], _frames([1.0], [1.0])).iloc[0]
        self.assertEqual((r.chosen_family, r.oracle_family), ("classical", "classical"))
        self.assertEqual(r.regret, 0.0)

    def test_oracle_family_always_has_the_exact_minimum_cost(self):
        """Near ties at several magnitudes, both directions, down to one float step apart."""
        jc, jt = [], []
        for base in (1e-13, 1e-6, 1.0, 37.5, 1e6):
            for lower in (base * (1 - GAP), np.nextafter(base, 0.0), base * 0.1):
                jc += [base, lower]
                jt += [lower, base]
        t = _frames(jc, jt)
        res = _run([HindsightOracle()], t)
        cost = np.where(res.oracle_family == "classical", t["test"].J_classical, t["test"].J_tsfm)
        self.assertTrue((cost == np.minimum(t["test"].J_classical, t["test"].J_tsfm)).all())
        self.assertTrue((res.regret == 0.0).all())
        self.assertEqual(res.oracle_family.tolist(), ["tsfm", "classical"] * (len(jc) // 2))

    def test_all_seven_stay_valid_on_near_ties(self):
        t = _frames([1.0, 1.0 - GAP, 1.0, 2.0], [1.0 - GAP, 1.0, 1.0, 1.0])
        res = _run(seven_selectors(THRESHOLD, NotFitted("d_omega", "not needed here")), t)
        self.assertTrue((res.regret.dropna() >= 0).all())
        orc = res[res.selector == "hindsight_oracle"]
        self.assertEqual(orc.chosen_family.tolist(), ["tsfm", "classical", "classical", "tsfm"])
        # every selector reports the same oracle for the same unit
        self.assertTrue((res.groupby("series_id").oracle_family.nunique() == 1).all())


class ToleranceKeptForValidation(unittest.TestCase):
    def test_argmin_family_still_calls_a_near_tie_for_classical(self):
        self.assertEqual(argmin_family([1.0, 1.0, 1.0], [1.0 - GAP, 1.0, 1.0 - 3 * TIE_RTOL]).tolist(),
                         ["classical", "classical", "tsfm"])
        self.assertEqual(exact_argmin_family([1.0, 1.0, 1.0], [1.0 - GAP, 1.0, 1.0 - 3 * TIE_RTOL]).tolist(),
                         ["tsfm", "classical", "tsfm"])

    def test_validation_selector_keeps_the_tolerance(self):
        # validation says "tie" (classical); the test costs clearly favour tsfm
        t = _frames([2.0], [1.0], val=([1.0], [1.0 - GAP]))
        r = _run([PinballValidation()], t).iloc[0]
        self.assertEqual(r.chosen_family, "classical")
        self.assertEqual(r.oracle_family, "tsfm")
        self.assertEqual(r.regret, 1.0)

    def test_exact_argmin_gives_none_for_missing_or_infinite_costs(self):
        got = exact_argmin_family([np.nan, 1.0, np.inf, 1.0], [1.0, np.nan, 1.0, 2.0])
        self.assertEqual(got.tolist(), [None, None, None, "classical"])


class BestFixedIsExact(unittest.TestCase):
    def test_group_total_lower_inside_the_tolerance(self):
        # one (horizon, tau) group; tsfm's total scaled cost is lower by less than the tolerance
        t = _frames([0.5, 0.5], [0.5, 0.5 - GAP])
        self.assertEqual(best_fixed(t["test"]).best_fixed_family.tolist(), ["tsfm"])
        t = _frames([0.5, 0.5 - GAP], [0.5, 0.5])
        self.assertEqual(best_fixed(t["test"]).best_fixed_family.tolist(), ["classical"])
        t = _frames([0.5, 0.5], [0.5, 0.5])
        self.assertEqual(best_fixed(t["test"]).best_fixed_family.tolist(), ["classical"])      # exact tie

    def test_best_fixed_cost_follows_the_family(self):
        res = _run([HindsightOracle()], _frames([0.5, 0.5], [0.5, 0.5 - GAP]))
        self.assertEqual(res.best_fixed_family.tolist(), ["tsfm", "tsfm"])
        self.assertEqual(res.best_fixed_cost.tolist(), [0.5, 0.5 - GAP])


class SchemaChecksTheOracleRow(unittest.TestCase):
    def test_oracle_choice_must_equal_oracle_family(self):
        res = _run([HindsightOracle()], _frames([1.0], [1.0]))            # exact tie: regret stays 0
        validate_selection_frame(res)
        bad = res.copy()
        bad["oracle_family"] = "tsfm"
        with self.assertRaisesRegex(SchemaError, "oracle_family"):
            validate_selection_frame(bad)

    def test_check_applies_only_to_the_oracle_selector(self):
        t = _frames([2.0], [1.0], val=([1.0], [2.0]))                     # validation picks classical
        validate_selection_frame(_run([PinballValidation(), HindsightOracle()], t))


if __name__ == "__main__":
    unittest.main()
