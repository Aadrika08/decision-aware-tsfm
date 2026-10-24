"""Near ties: the bootstrap's own oracle names the family the engine's exact oracle names."""
import unittest

import numpy as np
import pandas as pd

from decision_aware_tsfm.analysis.decision.costs import unit_costs
from decision_aware_tsfm.analysis.decision.evaluate import run_selectors
from decision_aware_tsfm.analysis.decision.fixture import REPRESENTATIVES
from decision_aware_tsfm.analysis.decision.schema import UNIT_KEYS
from decision_aware_tsfm.analysis.decision.selectors import HindsightOracle, argmin_family
from decision_aware_tsfm.analysis.experiments import bootstrap as bs

GAP = 5e-13                      # inside the 1e-12 tolerance, so argmin_family calls it a tie
N_ORIGINS = 8


def _per_origin(pairs):
    """Hand-built engine per-origin table, test origins only. pairs: one (classical, tsfm)
    cost pair per series, paid at EVERY origin (one target per origin, series scale 1), so the
    two families of a unit are as far apart on any resample as on the original table."""
    rows = []
    for i, pair in enumerate(pairs):
        for family, cost in zip(("classical", "tsfm"), pair):
            rows.append(pd.DataFrame(dict(dataset="hand", config="c", series_id=f"s{i:02d}", horizon=1,
                                          origin=np.arange(N_ORIGINS), family=family, tau=0.5,
                                          pinball_sum=float(cost), crps_sum=float(cost),
                                          scale=1.0, n=1, role="test")))
    return pd.concat(rows, ignore_index=True)


def _engine_oracle(po):
    """The engine's hindsight-oracle rows for the units of po."""
    test = unit_costs(po, "test")
    series = ["dataset", "config", "series_id", "horizon"]
    n_val = test[series].drop_duplicates().assign(n_validation_origins=0)
    # the oracle reads test costs only, so the validation table is a stand-in
    return run_selectors([HindsightOracle()], test[UNIT_KEYS].assign(omega=0.5), test, test,
                         REPRESENTATIVES, n_val)


class SampleOracleOnNearTies(unittest.TestCase):
    def _check(self, pairs, want):
        po = _per_origin(pairs)
        res = _engine_oracle(po)
        fixed = bs.fixed_choice_costs(po, res, "hindsight_oracle")      # both labels read from the engine
        own = bs.attach_sample_oracle(bs.family_cost_table(po))          # oracle named by the bootstrap
        keys = UNIT_KEYS + ["origin"]
        self.assertTrue(fixed[keys].equals(own[keys]))                   # same rows, same order
        # the same family as the engine for every unit, and the family that was expected
        self.assertEqual(own.oracle_family.tolist(), fixed.oracle_family.tolist())
        self.assertEqual(own.groupby("series_id").oracle_family.first().tolist(), want)
        self.assertTrue(np.array_equal(own.pinball_oracle.to_numpy(), fixed.pinball_oracle.to_numpy()))
        # the selector is the ENGINE's oracle, the oracle is the bootstrap's own
        own["pinball_selected"] = fixed["pinball_selected"].to_numpy()
        u = bs._unit_means(bs.OriginPanel(own).identity(), extra=bs.FIXED_COLUMNS)
        self.assertTrue((u.pinball_oracle == np.minimum(u.J_classical, u.J_tsfm)).all())
        ci = bs.bootstrap_ci(own, lambda d: bs.stat_mean_scaled_regret(d, by="boot_series"),
                             n_boot=40, seed=0)
        self.assertEqual(ci.replicates.shape, (40, len(pairs)))
        self.assertTrue((ci.estimate == 0.0).all())
        self.assertTrue((ci.replicates == 0.0).all())                    # zero for every unit on every resample

    def test_tsfm_lower_inside_the_tolerance(self):
        # the engine review's example (near-tie note of 2026-10-09) as per-origin costs
        self.assertEqual(argmin_family([1.0], [1.0 - GAP]).tolist(), ["classical"])
        self._check([(1.0, 1.0 - GAP)], ["tsfm"])

    def test_classical_lower_inside_the_tolerance(self):
        self._check([(1.0 - GAP, 1.0)], ["classical"])

    def test_exact_tie_goes_to_classical(self):
        self._check([(1.0, 1.0)], ["classical"])

    def test_several_magnitudes_both_directions(self):
        """Near ties and clear gaps from 1e-13 to 1e6. At 1e-13 the tolerance's absolute
        floor of 1e-12 calls a family that is ten times cheaper a tie."""
        pairs = []
        for base in (1e-13, 1e-6, 1.0, 37.5, 1e6):
            for lower in (base * (1 - GAP), base * 0.1):
                pairs += [(base, lower), (lower, base)]
        self.assertEqual(argmin_family([1e-13], [1e-14]).tolist(), ["classical"])
        self._check(pairs, ["tsfm", "classical"] * (len(pairs) // 2))


if __name__ == "__main__":
    unittest.main()
