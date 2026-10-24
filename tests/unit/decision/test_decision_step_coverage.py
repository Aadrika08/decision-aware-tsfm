"""Producer-v1 ingress: every job must hold exactly steps 1..horizon, checked per job
before any seed is aggregated (review of #44, 2026-10-10)."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from decision_aware_tsfm.analysis.decision import pipeline
from decision_aware_tsfm.analysis.decision.cache_io import (
    TARGET_KEYS, CacheError, DuplicateCacheKeyError, load_cache)
from decision_aware_tsfm.analysis.decision.fixture import (
    FLAT_FAMILY_OF_MODEL, FLAT_REPRESENTATIVES, PRODUCER_SCHEMA, flat_fixture_producer, stated_scales)
from decision_aware_tsfm.analysis.decision.handoff import read_handoff
from decision_aware_tsfm.analysis.decision.pipeline import build_tables, run

from .test_decision_producer import FLAT_OMEGA, write_bundle

INCOMPLETE = r"exactly steps 1\.\.horizon"
H = 24


def flat_with_spike():
    """The flat fixture with the shared realised value at step 24 set to 1000."""
    f = flat_fixture_producer()
    f.loc[f.step == H, "y_true"] = 1000.0
    return f


def without(frame, step, model="flat-classical", seed=0):
    """Drop one step from every job of one (model, seed): two jobs, one per origin."""
    gone = (frame.model == model) & (frame.seed == seed) & (frame.step == step)
    assert int(gone.sum()) == 2
    return frame[~gone].reset_index(drop=True)


def restep(frame, old, new):
    """Relabel one step of the first job, keeping its row count."""
    out = frame.copy()
    out.loc[(out.job_id == out.job_id.iloc[0]) & (out.step == old), "step"] = new
    return out


def _oracle_at_half(frame):
    t = build_tables(frame, PRODUCER_SCHEMA, FLAT_REPRESENTATIVES, FLAT_FAMILY_OF_MODEL, FLAT_OMEGA,
                     omega_col="omega", val_fraction=0, scales=stated_scales(frame))
    res, _ = run(t, FLAT_REPRESENTATIVES, 0.45)
    return res[(res.selector == "hindsight_oracle") & (res.tau == 0.5)].iloc[0]


class StepCoverage(unittest.TestCase):
    def test_complete_cache_still_passes(self):
        fc = load_cache(flat_with_spike(), PRODUCER_SCHEMA)
        self.assertEqual(len(fc), 2 * 2 * 2 * H)                 # models x origins x seeds x steps
        o = _oracle_at_half(flat_with_spike())
        self.assertEqual(o.chosen_family, "tsfm")
        self.assertAlmostEqual(o.realized_cost, 253 / 12, places=12)   # (23 * 0.5 + 494.5) / 24
        self.assertEqual(o.scale, 1.0)        # stated scale (stated_scales); mean |y| would be 51.25

    def test_review_case_missing_step_hidden_by_the_other_seed_is_refused(self):
        # the case from the #44 review: this was accepted, and the tau=0.50 oracle moved from
        # tsfm (cost 21.0833, scale 51.25) to classical (cost 11.5319, scale 31.0638)
        bad = without(flat_with_spike(), step=H)
        # the other seed still covers every target for both families, so target coverage sees nothing
        self.assertTrue((bad.groupby(TARGET_KEYS)["family"].nunique() == 2).all())
        self.assertEqual(bad.groupby(TARGET_KEYS).ngroups, 2 * H)
        with self.assertRaisesRegex(CacheError, INCOMPLETE):
            load_cache(bad, PRODUCER_SCHEMA)
        with self.assertRaisesRegex(CacheError, INCOMPLETE):
            _oracle_at_half(bad)

    def test_error_names_the_incomplete_jobs(self):
        bad = without(flat_with_spike(), step=H)
        with self.assertRaises(CacheError) as cm:
            load_cache(bad, PRODUCER_SCHEMA)
        msg = str(cm.exception)
        self.assertIn("2 jobs", msg)
        self.assertIn("fx_flat|H|fx_flat/s0|flat-classical|h24|o48|s0", msg)
        self.assertNotIn("flat-tsfm", msg)

    def test_incomplete_bundle_is_refused_after_its_hashes_verify(self):
        bad = without(flat_with_spike(), step=H)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "bundle"
            hashes = write_bundle(root, bad)
            fc, omega, rep = read_handoff(root, "current")
            self.assertEqual((rep["partitions_hash_verified"], rep["rows"]), (len(hashes), len(bad)))
            with self.assertRaisesRegex(CacheError, INCOMPLETE):
                build_tables(fc, PRODUCER_SCHEMA, FLAT_REPRESENTATIVES, FLAT_FAMILY_OF_MODEL, omega,
                             omega_col="omega", omega_variant="current", val_fraction=0,
                             scales=stated_scales(fc))

    def test_any_missing_step_is_refused(self):
        for step in (1, 12, H):
            for model, seed in (("flat-classical", 0), ("flat-tsfm", 1)):
                with self.subTest(step=step, model=model, seed=seed):
                    with self.assertRaisesRegex(CacheError, INCOMPLETE):
                        load_cache(without(flat_fixture_producer(), step, model, seed), PRODUCER_SCHEMA)

    def test_horizon_one_passes_and_horizon_two_lacking_a_step_is_refused(self):
        one = flat_fixture_producer(horizon=1)
        load_cache(one, PRODUCER_SCHEMA)                         # complete: passes
        short = flat_fixture_producer(horizon=2)
        with self.assertRaisesRegex(CacheError, INCOMPLETE):
            load_cache(without(short, 2), PRODUCER_SCHEMA)

    def test_extra_step_is_refused(self):
        base = flat_fixture_producer(origins=(48,))
        extra = base[(base.job_id == base.job_id.iloc[0]) & (base.step == H)].assign(
            step=H + 1, timestamp="2020-01-04T00:00:00")          # position 48 + 25 - 1 = 72
        with self.assertRaisesRegex(CacheError, INCOMPLETE):
            load_cache(pd.concat([base, extra], ignore_index=True), PRODUCER_SCHEMA)

    def test_out_of_range_step_is_refused(self):
        base = flat_fixture_producer(origins=(48,))
        cases = {"step 0 in place of step 1": (1, 0), "step H+1 in place of step H": (H, H + 1),
                 "negative step": (1, -3)}
        for name, (old, new) in cases.items():
            with self.subTest(name):
                bad = restep(base, old, new)
                self.assertEqual(len(bad), len(base))            # every job still has H rows
                with self.assertRaisesRegex(CacheError, INCOMPLETE):
                    load_cache(bad, PRODUCER_SCHEMA)

    def test_duplicate_step_is_refused(self):
        base = flat_fixture_producer(origins=(48,))
        with self.assertRaisesRegex(DuplicateCacheKeyError, "job_id, step"):
            load_cache(restep(base, 12, 13), PRODUCER_SCHEMA)    # H rows, steps 1..H, 13 twice, no 12
        with self.assertRaisesRegex(DuplicateCacheKeyError, "job_id, step"):
            load_cache(pd.concat([base, base.iloc[:1]], ignore_index=True), PRODUCER_SCHEMA)

    def test_check_runs_before_representatives_are_chosen_or_seeds_aggregated(self):
        bad = without(flat_with_spike(), step=H)
        later = AssertionError("reached a later stage with an incomplete job")
        with mock.patch.object(pipeline, "restrict_to_representatives", side_effect=later), \
                mock.patch.object(pipeline, "per_origin_costs", side_effect=later):
            with self.assertRaisesRegex(CacheError, INCOMPLETE):
                build_tables(bad, PRODUCER_SCHEMA, FLAT_REPRESENTATIVES, FLAT_FAMILY_OF_MODEL, FLAT_OMEGA,
                             omega_col="omega", val_fraction=0, scales=stated_scales(bad))

    def test_incomplete_job_of_a_model_that_is_not_a_representative_is_refused(self):
        other = flat_fixture_producer()
        other = other[other.model == "flat-classical"].assign(model="flat-other")
        other["job_id"] = other["job_id"].str.replace("flat-classical", "flat-other", regex=False)
        frame = pd.concat([flat_fixture_producer(), without(other, H, model="flat-other")], ignore_index=True)
        with self.assertRaisesRegex(CacheError, INCOMPLETE):
            load_cache(frame, PRODUCER_SCHEMA)


if __name__ == "__main__":
    unittest.main()
