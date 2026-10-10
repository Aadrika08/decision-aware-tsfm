"""Undefined Omega and undefined scale: handled explicitly, never silently."""
import unittest

import numpy as np
import pandas as pd

from decision_aware_tsfm.analysis.decision.fixture import (
    FAMILY_OF_MODEL, FIT_ONLY_DATASETS, FIXTURE_SCHEMA, FIXTURE_STATIC_THRESHOLD, REPRESENTATIVES)
from decision_aware_tsfm.analysis.decision.metrics import TAU_GRID
from decision_aware_tsfm.analysis.decision.pipeline import (
    ZERO_SCALE_REASON, OmegaJoinError, build_tables, run)
from decision_aware_tsfm.analysis.decision.schema import validate_selection_frame

from . import _shared

SID = "fx_tsfm_wins/s0"
OMEGA_BASED = ("static_omega", "d_omega")
OTHERS = ("always_classical", "always_tsfm", "pinball_validation", "crps_validation", "hindsight_oracle")


def _tables(omega, fc=None, **kw):
    fc = _shared.fixture_data()[0] if fc is None else fc
    return build_tables(fc, FIXTURE_SCHEMA, REPRESENTATIVES, FAMILY_OF_MODEL, omega,
                        fit_only_datasets=FIT_ONLY_DATASETS, **kw)


def _omega(**changes):
    om = _shared.fixture_data()[1].copy()
    for sid, value in changes.items():
        om.loc[om.series_id == sid.replace("__", "/"), "omega_levels"] = value
    return om


class UndefinedOmega(unittest.TestCase):
    def test_null_omega_makes_omega_selectors_defer_and_others_still_run(self):
        t = _tables(_omega(fx_tsfm_wins__s0=np.nan))
        self.assertEqual(t["n_units_undefined_omega"], 2 * len(TAU_GRID))      # two horizons x 11 taus
        res, _ = run(t, REPRESENTATIVES, FIXTURE_STATIC_THRESHOLD)
        validate_selection_frame(res)
        mine = res[res.series_id == SID]
        for name in OMEGA_BASED:
            self.assertTrue((mine[mine.selector == name].status == "insufficient_data").all(), name)
        for name in OTHERS:
            self.assertTrue((mine[mine.selector == name].status == "selected").all(), name)
        rest = res[res.series_id != SID]
        self.assertTrue((rest.status == "selected").all())                      # nobody else is affected

    def test_null_omega_in_fit_data_is_left_out_of_the_fit(self):
        t = _tables(_omega(fx_dfit__s0=np.nan))
        _, d = run(t, REPRESENTATIVES, FIXTURE_STATIC_THRESHOLD)
        self.assertNotIn("fx_dfit/s0", d.fitted_series)
        self.assertIn("fx_dfit/s1", d.fitted_series)

    def test_missing_omega_row_is_still_an_error(self):
        om = _shared.fixture_data()[1]
        with self.assertRaisesRegex(OmegaJoinError, "no omega row"):
            _tables(om[om.series_id != SID])

    def test_omega_exactly_zero_is_refused_unless_allowed(self):
        with self.assertRaisesRegex(OmegaJoinError, "exactly 0.0"):
            _tables(_omega(fx_tsfm_wins__s0=0.0))
        t = _tables(_omega(fx_tsfm_wins__s0=0.0), allow_zero_omega=True)
        self.assertTrue(t["settings"]["allow_zero_omega"])
        res, _ = run(t, REPRESENTATIVES, FIXTURE_STATIC_THRESHOLD)
        st = res[(res.series_id == SID) & (res.selector == "static_omega")]
        self.assertTrue((st.chosen_family == "classical").all())                # what 0.0 would have meant

    def test_out_of_range_or_infinite_omega_is_refused(self):
        for bad in (1.5, -0.1, np.inf):
            with self.subTest(omega=bad):
                with self.assertRaisesRegex(OmegaJoinError, r"\[0, 1\]"):
                    _tables(_omega(fx_tsfm_wins__s0=bad))

    def test_non_numeric_omega_is_refused(self):
        om = _shared.fixture_data()[1].copy()
        om["omega_levels"] = om["omega_levels"].astype(object)
        om.loc[om.series_id == SID, "omega_levels"] = "flat"
        with self.assertRaisesRegex(OmegaJoinError, "numeric or null"):
            _tables(om)

    def test_default_settings_record_that_zero_is_not_allowed(self):
        self.assertFalse(_shared.tables()["settings"]["allow_zero_omega"])
        self.assertEqual(_shared.tables()["n_units_undefined_omega"], 0)


class ZeroScale(unittest.TestCase):
    def _flat_zero(self, sid):
        fc = _shared.fixture_data()[0].copy()
        fc.loc[fc.series_id == sid, "y"] = 0.0
        return fc

    def test_all_zero_series_is_excluded_with_a_reason(self):
        t = _tables(_shared.fixture_data()[1], fc=self._flat_zero("fx_tie/s0"))
        ex = t["excluded_series_horizons"]
        self.assertEqual(sorted(ex.horizon.tolist()), [1, 24])
        self.assertTrue((ex.series_id == "fx_tie/s0").all())
        self.assertTrue((ex.reason == ZERO_SCALE_REASON).all())
        self.assertTrue((ex.used_for == "evaluation").all())
        for name in ("units", "test", "val", "per_origin"):
            self.assertNotIn("fx_tie/s0", set(t[name].series_id), name)
        res, _ = run(t, REPRESENTATIVES, FIXTURE_STATIC_THRESHOLD)
        validate_selection_frame(res)
        self.assertTrue(np.isfinite(res.scaled_regret).all())
        self.assertEqual(res.series_id.nunique(), _shared.results()[0].series_id.nunique() - 1)

    def test_all_zero_fit_series_is_excluded_from_the_fit(self):
        t = _tables(_shared.fixture_data()[1], fc=self._flat_zero("fx_dfit/s0"))
        ex = t["excluded_series_horizons"]
        self.assertTrue((ex.used_for == "fit").all())
        self.assertNotIn("fx_dfit/s0", set(t["fit_units"].series_id))
        _, d = run(t, REPRESENTATIVES, FIXTURE_STATIC_THRESHOLD)
        self.assertNotIn("fx_dfit/s0", d.fitted_series)

    def test_nothing_is_excluded_on_the_ordinary_fixture(self):
        self.assertEqual(len(_shared.tables()["excluded_series_horizons"]), 0)


if __name__ == "__main__":
    unittest.main()
