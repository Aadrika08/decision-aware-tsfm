"""Idealised forecasters and the exported canonical-v1 frame."""
import unittest

import numpy as np
from scipy import stats

from decision_aware_tsfm.analysis.decision.cache_io import (
    TARGET_KEYS, load_cache, qcol, restrict_to_representatives)
from decision_aware_tsfm.analysis.decision.metrics import TAU_GRID
from decision_aware_tsfm.analysis.experiments.e3.design import (
    F_VARIANTS, FAMILY_SLOT, MEMBERS, MODEL_G, N_CONTEXT, N_TOTAL, STRUCTURES, innovation_dist)
from decision_aware_tsfm.analysis.experiments.e3.forecasters import (
    QCOLS, TAUS, group_frame, quantile_offsets, series_frame, true_innovation_quantiles)
from decision_aware_tsfm.analysis.experiments.e3.generator import (
    conditional_mean, make_series, series_id)

from . import _shared

ALL_MODELS = [MODEL_G] + [v.model for v in F_VARIANTS.values()]
F_PRIMARY = F_VARIANTS["primary"].model


def _frame(sname, mname, seed=0, models=ALL_MODELS):
    st, mb = STRUCTURES[sname], MEMBERS[mname]
    x = make_series(st, mb, seed)
    return x, series_frame(x, series_id(sname, seed, mname), st, mb, models)


class CanonicalFrame(unittest.TestCase):
    def test_engine_loader_accepts_the_frame(self):
        rows = _shared.scored_rows()
        by = {sid: _shared.values_of(sid) for sid in rows.series_id}
        fc = load_cache(group_frame(by, rows, ALL_MODELS), "canonical-v1")
        self.assertEqual(len(fc), len(rows) * len(ALL_MODELS) * (N_TOTAL - N_CONTEXT))
        self.assertEqual([qcol(t) for t in TAU_GRID], QCOLS)

    def test_both_forecasters_cover_identical_targets(self):
        rows = _shared.scored_rows()
        by = {sid: _shared.values_of(sid) for sid in rows.series_id}
        fc = load_cache(group_frame(by, rows, [MODEL_G, F_PRIMARY]), "canonical-v1")
        g = fc[fc.model == MODEL_G][TARGET_KEYS].reset_index(drop=True)
        f = fc[fc.model == F_PRIMARY][TARGET_KEYS].reset_index(drop=True)
        self.assertTrue(g.equals(f))
        # and the engine's own identical-window check passes
        reps = {FAMILY_SLOT["G"]: MODEL_G, FAMILY_SLOT["F"]: F_PRIMARY}
        out = restrict_to_representatives(fc, reps, {m: s for s, m in reps.items()})
        self.assertEqual(len(out), len(fc))

    def test_time_convention_and_targets(self):
        for sname in ("ar1_phi0.80", "sine_sd1.0"):
            with self.subTest(structure=sname):
                x, fr = _frame(sname, "skewnorm_n2", models=[MODEL_G])
                self.assertTrue((fr.step == 1).all() and (fr.horizon == 1).all())
                np.testing.assert_array_equal(fr.origin.to_numpy(), np.arange(N_CONTEXT, N_TOTAL))
                pos = (fr.origin + fr.step - 1).to_numpy()
                np.testing.assert_array_equal(fr.y.to_numpy(), x[pos])

    def test_quantiles_are_non_decreasing_for_every_model(self):
        for mname in MEMBERS:
            with self.subTest(member=mname):
                _, fr = _frame("ar1_phi0.95", mname)
                self.assertTrue((np.diff(fr[QCOLS].to_numpy(), axis=1) >= 0).all())


class ForecasterDefinitions(unittest.TestCase):
    def test_gaussian_member_G_equals_true_conditional_quantiles(self):
        for sname in ("ar1_phi0.50", "ar1_phi0.95", "sine_sd1.0"):
            with self.subTest(structure=sname):
                st, mb = STRUCTURES[sname], MEMBERS["gaussian"]
                x, fr = _frame(sname, "gaussian", models=[MODEL_G])
                pos = np.arange(N_CONTEXT, N_TOTAL)
                # true conditional law written out here, independently of forecasters.py
                mean = st.phi * x[pos - 1] if st.kind == "ar1" else (
                    np.sin(2 * np.pi * pos / 24) + 0.5 * np.sin(2 * np.pi * pos / 168))
                true_q = mean[:, None] + st.sigma * innovation_dist(mb).ppf(TAUS)[None, :]
                np.testing.assert_allclose(fr[QCOLS].to_numpy(), true_q, rtol=0, atol=1e-12)

    def test_G_keeps_the_gaussian_shape_for_every_member(self):
        # G must not adapt to the member: its offsets are sigma * z_tau whatever the true law is
        z = stats.norm.ppf(TAUS)
        for sname in ("ar1_phi0.80", "sine_sd2.0"):
            for mname in MEMBERS:
                with self.subTest(structure=sname, member=mname):
                    st = STRUCTURES[sname]
                    x, fr = _frame(sname, mname, models=[MODEL_G])
                    mean = conditional_mean(x, st)[N_CONTEXT:]
                    np.testing.assert_allclose(fr[QCOLS].to_numpy() - mean[:, None],
                                               np.tile(st.sigma * z, (mean.size, 1)), atol=1e-12)

    def test_location_variants_are_true_quantiles_plus_the_stated_error(self):
        for sname in ("ar1_phi0.80", "sine_sd2.0"):
            for vname in ("primary", "loc_small", "loc_large"):
                with self.subTest(structure=sname, variant=vname):
                    st, mb, v = STRUCTURES[sname], MEMBERS["skewnorm_p2"], F_VARIANTS[vname]
                    x = make_series(st, mb, 0)
                    off = quantile_offsets(v.model, x, st, mb)
                    np.testing.assert_allclose(off - st.sigma * v.location_error,
                                               st.sigma * true_innovation_quantiles(mb), atol=1e-12)
        self.assertEqual([F_VARIANTS[v].location_error for v in ("loc_small", "primary", "loc_large")],
                         [0.10, 0.25, 0.50])

    def test_true_quantiles_have_the_stated_coverage(self):
        # long series: share of targets at or below the true conditional quantile is tau
        n_test = 20000
        for sname, mname in (("ar1_phi0.80", "skewnorm_p8"), ("ar1_phi0.80", "student_t5"),
                             ("sine_sd1.0", "skewnorm_n8")):
            with self.subTest(structure=sname, member=mname):
                st, mb = STRUCTURES[sname], MEMBERS[mname]
                x = make_series(st, mb, 11, n_total=N_CONTEXT + n_test)
                pos = np.arange(N_CONTEXT, x.size)
                q = conditional_mean(x, st)[pos][:, None] + st.sigma * true_innovation_quantiles(mb)
                cover = (x[pos][:, None] <= q).mean(axis=0)
                tol = 4 * np.sqrt(TAUS * (1 - TAUS) / n_test)
                self.assertTrue((np.abs(cover - TAUS) < tol).all(), cover)

    def test_empirical_variant_uses_context_only(self):
        st, mb = STRUCTURES["ar1_phi0.80"], MEMBERS["student_t5"]
        model = F_VARIANTS["empirical"].model
        x = make_series(st, mb, 4)
        base = quantile_offsets(model, x, st, mb)
        future = x.copy()
        future[N_CONTEXT:] = 1e6                        # change every test-period value
        np.testing.assert_array_equal(quantile_offsets(model, future, st, mb), base)
        past = x.copy()
        past[100:1900] *= 3.0                           # change the context
        self.assertFalse(np.array_equal(quantile_offsets(model, past, st, mb), base))
        self.assertLess(float(np.abs(base - true_innovation_quantiles(mb)).max()), 0.5)


if __name__ == "__main__":
    unittest.main()
