"""Moving-block and cluster bootstrap: block structure, refusal of stages that
cannot resample, clustering, shared time blocks, interval level, coverage,
and the regret statistics.

Everything here is synthetic: hand-built tables with known answers and the
engine's fixture.
"""
import unittest

import numpy as np
import pandas as pd

from decision_aware_tsfm.analysis.decision.evaluate import summarize
from decision_aware_tsfm.analysis.experiments import bootstrap as bs
from decision_aware_tsfm.analysis.experiments import coverage_sim, e1

from . import _shared

Z95, Z80 = 1.959964, 1.281552


def _panel(datasets, n_origins, horizon=1, step=1, first=100, taus=(0.5,), value=None, seed=0):
    """datasets: {name: number of series}. One row per (series, tau, origin)."""
    rng = np.random.default_rng(seed)
    rows = []
    for ds, n_series in datasets.items():
        for j in range(n_series):
            v = rng.standard_normal(n_origins) if value is None else np.full(n_origins, value)
            for tau in taus:
                rows.append(pd.DataFrame(dict(dataset=ds, config="c", series_id=f"{ds}/s{j}",
                                              horizon=horizon, tau=tau,
                                              origin=first + step * np.arange(n_origins), cost=v)))
    return pd.concat(rows, ignore_index=True)


def _one_series(cost, horizon=1, step=1):
    return pd.DataFrame(dict(dataset="d", config="c", series_id="s", horizon=horizon, tau=0.5,
                             origin=np.arange(len(cost)) * step, cost=cost))


def _mean_cost(d):
    return float(d["cost"].mean())


def _blocks(idx, L):
    return [idx[i:i + L] for i in range(0, len(idx), L)]


def _costs(rows):
    """rows: (series, origin, pinball_classical, pinball_tsfm, scale). One target per origin;
    the scale is the series' own and must be the same at every origin of a series."""
    d = pd.DataFrame(rows, columns=["series_id", "origin", "pinball_classical", "pinball_tsfm", "scale"])
    d.insert(0, "config", "c")
    d.insert(0, "dataset", "hand")
    d["horizon"], d["tau"], d["n"] = 1, 0.5, 1
    return d


class BlockLength(unittest.TestCase):
    def test_overlap_count(self):
        for h, step, want in ((24, 1, 24), (24, 24, 1), (24, 12, 2), (24, 7, 4), (1, 1, 1), (48, 24, 2)):
            with self.subTest(horizon=h, step=step):
                self.assertEqual(bs.n_overlapping_origins(h, step), want)

    def test_default_rule(self):
        cases = [((24, 1, 480), 24),        # overlap floor dominates
                 ((24, 24, 30), 4),         # no overlap: ceil(30 ** (1/3)) = 4
                 ((1, 1, 320), 7),          # ceil(320 ** (1/3)) = 7
                 ((24, 1, 48), 24),         # exactly two blocks: the shortest series accepted
                 ((1, 1, 27), 3), ((1, 1, 28), 4)]
        for args, want in cases:
            with self.subTest(args=args):
                self.assertEqual(bs.default_block_length(*args), want)

    def test_default_rule_refuses_a_series_too_short_for_its_block(self):
        """The old rule capped the block at the number of origins, which made
        every resample the original series."""
        for args in ((24, 1, 10),      # windows overlap across the whole series
                     (24, 1, 24), (24, 1, 47),      # fewer than two blocks
                     (1, 1, 1), (24, 24, 2), (1, 1, 4)):       # fewer than 5 block starts
            with self.subTest(args=args):
                with self.assertRaises(bs.DegenerateResampleError):
                    bs.default_block_length(*args)

    def test_check_block_floors(self):
        self.assertEqual((bs.MIN_BLOCK_STARTS, bs.MIN_BLOCKS, bs.MIN_CLUSTERS), (5, 2, 2))
        for n, L, ok in ((5, 1, True), (4, 1, False),         # 5 starts needed
                         (10, 5, True), (9, 5, False),        # two blocks needed
                         (6, 2, True), (5, 2, False), (100, 100, False), (100, 101, False)):
            with self.subTest(n=n, L=L):
                if ok:
                    bs.check_block(n, L)
                else:
                    with self.assertRaises(bs.DegenerateResampleError):
                        bs.check_block(n, L)

    def test_origin_step(self):
        self.assertEqual(bs.origin_step([1001, 1025, 1049]), 24)
        self.assertEqual(bs.origin_step([5, 6, 7, 9]), 1)
        self.assertEqual(bs.origin_step([10, 14, 20]), 4)      # irregular: the smallest gap
        self.assertEqual(bs.origin_step([7]), 1)

    def test_panel_reports_block_length_per_series(self):
        p = bs.OriginPanel(_panel({"a": 2}, 30, horizon=24, step=24))
        self.assertTrue((p.block_lengths("auto").block_length == 4).all())
        self.assertEqual(p.describe_block_lengths("auto"), "h24: L=4, n=30")
        p = bs.OriginPanel(_panel({"a": 2}, 480, horizon=24, step=1))
        self.assertTrue((p.block_lengths("auto").block_length == 24).all())
        self.assertEqual(p.describe_block_lengths(None), "origins not resampled")


class StagesThatCannotResample(unittest.TestCase):
    """Fix 1. Each case used to return an interval of width exactly zero."""

    def test_reviewer_block_cases_raise_instead_of_zero_width(self):
        rng = np.random.default_rng(0)
        for n, H, step in ((1, 1, 1), (2, 1, 1), (2, 24, 24), (10, 24, 1), (24, 24, 1), (3, 24, 24)):
            with self.subTest(n_origins=n, horizon=H, step=step):
                f = _one_series(rng.standard_normal(n), H, step)
                with self.assertRaises(bs.DegenerateResampleError):
                    bs.bootstrap_ci(f, _mean_cost, n_boot=20, seed=0)
                with self.assertRaises(bs.DegenerateResampleError):
                    bs.OriginPanel(f).resample(np.random.default_rng(0))

    def test_single_cluster_raises_at_every_cluster_level(self):
        one = _one_series(np.random.default_rng(0).standard_normal(50))
        for cluster in ("series", "dataset", "dataset+series"):
            with self.subTest(cluster=cluster, table="one series, one dataset"):
                with self.assertRaises(bs.DegenerateResampleError):
                    bs.bootstrap_ci(one, _mean_cost, n_boot=20, seed=0, block_length=None, cluster=cluster)
        five = _panel({"only": 5}, 50)
        for cluster in ("dataset", "dataset+series"):
            with self.subTest(cluster=cluster, table="five series, one dataset"):
                with self.assertRaises(bs.DegenerateResampleError) as cm:
                    bs.bootstrap_ci(five, _mean_cost, n_boot=20, seed=0, block_length=None, cluster=cluster)
                self.assertIn("dataset", str(cm.exception))
        # the same table can be resampled over series, and then the width is not zero
        ci = bs.bootstrap_ci(five, _mean_cost, n_boot=50, seed=0, block_length=None, cluster="series")
        self.assertGreater(ci.width[0], 0.01)

    def test_one_dataset_with_a_single_series_blocks_the_series_stage(self):
        f = _panel({"a": 4, "lonely": 1}, 30)
        for cluster in ("series", "dataset+series"):
            with self.subTest(cluster=cluster):
                with self.assertRaises(bs.DegenerateResampleError) as cm:
                    bs.bootstrap_ci(f, _mean_cost, n_boot=20, seed=0, cluster=cluster)
                self.assertIn("lonely", str(cm.exception))
        bs.bootstrap_ci(f, _mean_cost, n_boot=20, seed=0, cluster="dataset")

    def test_explicit_block_length_is_not_silently_clamped(self):
        f = _one_series(np.random.default_rng(1).standard_normal(30))
        for L in (30, 31, 500, 16):                 # 16: fewer than two blocks
            with self.subTest(block_length=L):
                with self.assertRaises(bs.DegenerateResampleError):
                    bs.bootstrap_ci(f, _mean_cost, n_boot=20, seed=0, block_length=L)
        self.assertGreater(bs.bootstrap_ci(f, _mean_cost, n_boot=50, seed=0, block_length=15).width[0], 0)

    def test_one_short_series_among_long_ones_is_named(self):
        f = pd.concat([_panel({"a": 3}, 40), _panel({"b": 1}, 3)], ignore_index=True)
        with self.assertRaises(bs.DegenerateResampleError) as cm:
            bs.bootstrap_ci(f, _mean_cost, n_boot=20, seed=0)
        self.assertIn("b/s0", str(cm.exception))

    def test_error_is_raised_before_any_resample_and_statistic_call(self):
        calls = []
        f = _one_series(np.arange(10.0), 24, 1)
        with self.assertRaises(bs.DegenerateResampleError):
            bs.bootstrap_ci(f, lambda d: calls.append(1) or 0.0, n_boot=20, seed=0)
        self.assertEqual(calls, [])


class MovingBlocks(unittest.TestCase):
    def test_indices_are_contiguous_blocks(self):
        for n, L in ((10, 3), (10, 5), (7, 1), (100, 24), (48, 24), (9, 2), (46, 23)):
            for seed in range(20):
                with self.subTest(n=n, L=L, seed=seed):
                    idx = bs.moving_block_indices(n, L, np.random.default_rng(seed))
                    self.assertEqual(len(idx), n)
                    self.assertTrue(((idx >= 0) & (idx < n)).all())
                    blocks = _blocks(idx, L)
                    self.assertEqual(len(blocks), -(-n // L))
                    for k, b in enumerate(blocks):
                        self.assertTrue((np.diff(b) == 1).all())
                        if k < len(blocks) - 1:
                            self.assertEqual(len(b), L)
                        self.assertLessEqual(b[0], n - L)

    def test_every_admissible_start_is_drawn_about_equally_often(self):
        """Starts are uniform on 0 .. n-L, both ends included. A resampler that
        never draws the last start can never begin a block at the last origins."""
        n, L = 12, 4                                           # starts 0..8, three blocks per draw
        rng = np.random.default_rng(5)
        starts = np.concatenate([[b[0] for b in _blocks(bs.moving_block_indices(n, L, rng), L)]
                                 for _ in range(1500)])
        self.assertEqual(sorted(set(starts.tolist())), list(range(n - L + 1)))
        freq = np.bincount(starts, minlength=n - L + 1) / starts.size
        self.assertTrue((np.abs(freq - 1 / 9) < 0.02).all(), freq)
        for n, L in ((5, 1), (10, 5), (48, 24)):
            with self.subTest(n=n, L=L):
                seen = set()
                for seed in range(400):
                    seen |= {b[0] for b in _blocks(bs.moving_block_indices(n, L, np.random.default_rng(seed)), L)}
                self.assertEqual(seen, set(range(n - L + 1)))

    def test_blocks_are_actually_resampled(self):
        idx = [tuple(bs.moving_block_indices(50, 5, np.random.default_rng(s))) for s in range(30)]
        self.assertGreater(len(set(idx)), 25)
        starts = {b[0] for i in idx for b in _blocks(np.array(i), 5)}
        self.assertGreater(len(starts), 30)           # starts are not confined to multiples of L
        self.assertTrue(any(s % 5 for s in starts))

    def test_block_as_long_as_the_series_is_refused(self):
        """Replaces the old test that expected the original order back."""
        with self.assertRaises(bs.DegenerateResampleError):
            bs.moving_block_indices(9, 9, np.random.default_rng(1))

    def test_bad_block_length(self):
        for L in (0, -1):
            with self.subTest(L=L):
                with self.assertRaises(ValueError):
                    bs.moving_block_indices(10, L, np.random.default_rng(0))
        with self.assertRaises(bs.DegenerateResampleError):
            bs.moving_block_indices(10, 11, np.random.default_rng(0))

    def test_resampled_table_has_contiguous_origin_blocks_and_keeps_taus_together(self):
        taus = (0.05, 0.5, 0.95)
        f = _panel({"a": 2, "b": 1}, 40, horizon=24, step=24, first=1001, taus=taus)
        f["cost"] = f["origin"] * 1000 + f["tau"]              # value identifies (origin, tau)
        panel = bs.OriginPanel(f)
        for seed in range(5):
            r = panel.resample(np.random.default_rng(seed), block_length=4, cluster="none")
            self.assertEqual(len(r), len(f))
            self.assertEqual(list(r.columns), list(f.columns) + ["boot_dataset", "boot_series"])
            self.assertEqual(r.dtypes[:-2].tolist(), panel.frame.dtypes.tolist())
            for sid, g in r.groupby("boot_series"):
                with self.subTest(seed=seed, series=sid):
                    self.assertEqual(g.series_id.nunique(), 1)
                    o = g.origin.to_numpy().reshape(-1, len(taus))
                    self.assertTrue((o == o[:, :1]).all())               # one origin, all taus, adjacent
                    self.assertTrue(np.allclose(g.tau.to_numpy().reshape(-1, len(taus)), taus))
                    self.assertTrue(np.allclose(g.cost, g.origin * 1000 + g.tau))
                    for b in _blocks(o[:, 0], 4):
                        self.assertTrue((np.diff(b) == 24).all())        # consecutive origins
        self.assertFalse(r.origin.reset_index(drop=True).equals(panel.frame.origin))


class Clusters(unittest.TestCase):
    def setUp(self):
        self.f = _panel({"a": 4, "b": 3, "c": 5}, 12, taus=(0.25, 0.75))
        self.panel = bs.OriginPanel(self.f)
        self.origins = set(self.f.origin)
        self.series_of = self.f.groupby("dataset").series_id.agg(lambda x: frozenset(x)).to_dict()

    def test_series_level_keeps_all_origins_of_a_series_together(self):
        repeats = 0
        for seed in range(20):
            r = self.panel.resample(np.random.default_rng(seed), block_length=None, cluster="series")
            self.assertEqual(len(r), len(self.f))
            for sid, g in r.groupby("boot_series"):
                self.assertEqual(g.series_id.nunique(), 1)
                self.assertEqual(len(g), 12 * 2)
                self.assertEqual(set(g.origin), self.origins)
                self.assertTrue((g.groupby("tau").origin.apply(list).map(sorted)
                                 .map(lambda o: o == sorted(self.origins))).all())
            # drawn within dataset: the number of series per dataset is unchanged
            n = r.groupby("dataset").boot_series.nunique().to_dict()
            self.assertEqual(n, {"a": 4, "b": 3, "c": 5})
            repeats += int(r.groupby("boot_series").series_id.first().duplicated().any())
        self.assertGreater(repeats, 15)             # drawing is with replacement

    def test_dataset_level_keeps_all_series_of_a_dataset_together(self):
        seen_counts = set()
        for seed in range(30):
            r = self.panel.resample(np.random.default_rng(seed), block_length=None, cluster="dataset")
            self.assertEqual(r.boot_dataset.nunique(), 3)
            for d, g in r.groupby("boot_dataset"):
                self.assertEqual(g.dataset.nunique(), 1)
                ds = g.dataset.iloc[0]
                self.assertEqual(frozenset(g.series_id), self.series_of[ds])
                self.assertEqual(len(g), len(self.series_of[ds]) * 12 * 2)
                self.assertEqual(g.boot_series.nunique(), len(self.series_of[ds]))
            seen_counts.add(r.dataset.nunique())
        self.assertGreater(len(seen_counts), 1)     # some draws repeat a dataset, some do not

    def test_two_stage_resamples_series_inside_drawn_datasets(self):
        """Both stages must happen: datasets repeat across the table AND series
        repeat inside a drawn dataset. A resampler that skips the series stage
        returns every dataset copy with each of its series exactly once."""
        series_repeat = dataset_repeat = 0
        for seed in range(30):
            r = self.panel.resample(np.random.default_rng(seed), block_length=None, cluster="dataset+series")
            per_copy = r.groupby("boot_dataset")
            for d, g in per_copy:
                ds = g.dataset.iloc[0]
                self.assertEqual(g.dataset.nunique(), 1)
                self.assertLessEqual(set(g.series_id), self.series_of[ds])
                self.assertEqual(g.boot_series.nunique(), len(self.series_of[ds]))
            distinct = per_copy.series_id.nunique()
            copies = per_copy.boot_series.nunique()
            series_repeat += int((distinct < copies).any())
            dataset_repeat += int(r.dataset.nunique() < 3)
        self.assertGreater(series_repeat, 25)
        self.assertGreater(dataset_repeat, 15)

    def test_identity_marks_every_series_once(self):
        i = self.panel.identity()
        self.assertEqual(i.boot_series.nunique(), 12)
        self.assertEqual(i.groupby("boot_series").series_id.nunique().max(), 1)
        self.assertEqual(i.boot_dataset.nunique(), 3)

    def test_unknown_cluster_level_and_reserved_columns(self):
        with self.assertRaises(ValueError):
            self.panel.resample(np.random.default_rng(0), cluster="domain")
        with self.assertRaises(ValueError):
            bs.OriginPanel(self.f.assign(boot_series=0))
        with self.assertRaises(ValueError):
            bs.OriginPanel(self.f.drop(columns="origin"))

    def test_missing_key_values_raise(self):
        f = self.f.copy()
        f["series_id"] = f["series_id"].astype(object)
        f.loc[:3, "series_id"] = None
        with self.assertRaises(ValueError):
            bs.OriginPanel(f)


class SharedTimeBlocks(unittest.TestCase):
    """Fix 5: one set of block positions per dataset, for series with the same origins."""

    def _origins_by_copy(self, r):
        return {k: tuple(g.origin) for k, g in r.groupby("boot_series")}

    def test_all_series_of_a_dataset_get_the_same_positions(self):
        f = _panel({"a": 5, "b": 4}, 40, horizon=24, step=24)
        panel = bs.OriginPanel(f)
        differ_between_datasets = 0
        for seed in range(10):
            r = panel.resample(np.random.default_rng(seed), 4, "none", "shared_within_dataset")
            seqs = self._origins_by_copy(r)
            by_ds = r.groupby("dataset").boot_series.unique()
            a, b = {seqs[k] for k in by_ds["a"]}, {seqs[k] for k in by_ds["b"]}
            self.assertEqual((len(a), len(b)), (1, 1))
            differ_between_datasets += int(a != b)
            own = self._origins_by_copy(panel.resample(np.random.default_rng(seed), 4, "none", "per_series"))
            self.assertEqual(len(set(own.values())), 9)        # default: every series its own draw
        self.assertEqual(differ_between_datasets, 10)          # datasets are drawn independently

    def test_shared_positions_also_under_series_clustering_and_per_horizon(self):
        f = pd.concat([_panel({"a": 4}, 40, horizon=24, step=24),
                       _panel({"a": 4}, 60, horizon=1, step=1)], ignore_index=True)
        r = bs.OriginPanel(f).resample(np.random.default_rng(3), "auto", "series", "shared_within_dataset")
        for h, g in r.groupby("horizon"):
            with self.subTest(horizon=h):
                self.assertEqual(g.boot_series.nunique(), 4)
                self.assertEqual(len({tuple(x.origin) for _, x in g.groupby("boot_series")}), 1)

    def test_misaligned_origins_fail_clearly(self):
        shifted = pd.concat([_panel({"a": 3}, 40), _panel({"a": 1}, 40, first=101).assign(series_id="a/late")],
                            ignore_index=True)
        shorter = pd.concat([_panel({"a": 3}, 40), _panel({"a": 1}, 39).assign(series_id="a/short")],
                            ignore_index=True)
        for name, f in (("shifted by one origin", shifted), ("one origin fewer", shorter)):
            with self.subTest(case=name):
                with self.assertRaises(bs.MisalignedOriginsError) as cm:
                    bs.bootstrap_ci(f, _mean_cost, n_boot=10, seed=0, time_blocks="shared_within_dataset")
                self.assertIn("identical origins", str(cm.exception))
                bs.bootstrap_ci(f, _mean_cost, n_boot=10, seed=0)          # the default still runs
        ok = _panel({"a": 3, "b": 2}, 40)                   # alignment is needed within a dataset only
        ok.loc[ok.dataset == "b", "origin"] += 7
        bs.bootstrap_ci(ok, _mean_cost, n_boot=10, seed=0, time_blocks="shared_within_dataset")

    def test_shared_needs_a_block_stage_and_a_known_mode(self):
        f = _panel({"a": 3}, 40)
        with self.assertRaises(ValueError):
            bs.bootstrap_ci(f, _mean_cost, n_boot=5, block_length=None, cluster="series",
                            time_blocks="shared_within_dataset")
        with self.assertRaises(ValueError):
            bs.bootstrap_ci(f, _mean_cost, n_boot=5, time_blocks="by_calendar")

    def test_reviewer_common_shock_construction(self):
        """20 series, 60 origins, every origin carries a shock shared by all
        series. True sd of the mean is sqrt(1/60 + 1/1200), so the true width
        of a 95% interval is 0.5186. Independent per-series blocks average the
        shock away; shared blocks keep it."""
        true_width = 2 * Z95 * np.sqrt(1 / 60 + 1 / 1200)
        self.assertAlmostEqual(true_width, 0.5186, places=4)
        for seed in (5000, 5001):
            f, stat, truth, sd = coverage_sim.make_shock(np.random.default_rng(seed))
            self.assertAlmostEqual(2 * Z95 * sd, true_width, places=12)
            own = bs.bootstrap_ci(f, stat, n_boot=150, seed=seed)
            own_series = bs.bootstrap_ci(f, stat, n_boot=150, seed=seed, cluster="series")
            shared = bs.bootstrap_ci(f, stat, n_boot=150, seed=seed, time_blocks="shared_within_dataset")
            with self.subTest(seed=seed):
                self.assertLess(own.width[0] / true_width, 0.45)
                self.assertLess(own_series.width[0] / true_width, 0.55)
                self.assertGreater(shared.width[0] / true_width, 0.60)
                self.assertLess(shared.width[0] / true_width, 1.35)
                self.assertEqual(shared.time_blocks, "shared_within_dataset")


class Interval(unittest.TestCase):
    def test_same_seed_same_interval(self):
        f = _panel({"a": 3, "b": 3}, 60, seed=4)
        for cluster in bs.CLUSTER_LEVELS:
            for mode in bs.TIME_BLOCK_MODES:
                with self.subTest(cluster=cluster, time_blocks=mode):
                    a = bs.bootstrap_ci(f, _mean_cost, n_boot=60, seed=11, cluster=cluster, time_blocks=mode)
                    b = bs.bootstrap_ci(f, _mean_cost, n_boot=60, seed=11, cluster=cluster, time_blocks=mode)
                    c = bs.bootstrap_ci(f, _mean_cost, n_boot=60, seed=12, cluster=cluster, time_blocks=mode)
                    self.assertEqual((a.lower[0], a.upper[0]), (b.lower[0], b.upper[0]))
                    self.assertTrue(np.array_equal(a.replicates, b.replicates))
                    self.assertNotEqual((a.lower[0], a.upper[0]), (c.lower[0], c.upper[0]))
                    self.assertGreater(a.width[0], 0.01)
                    self.assertAlmostEqual(a.estimate[0], f.cost.mean(), places=14)

    def test_row_order_of_input_does_not_matter(self):
        f = _panel({"a": 3}, 40, seed=5)
        a = bs.bootstrap_ci(f, _mean_cost, n_boot=50, seed=1)
        b = bs.bootstrap_ci(f.sample(frac=1.0, random_state=0), _mean_cost, n_boot=50, seed=1)
        self.assertTrue(np.allclose(a.replicates, b.replicates, rtol=0, atol=1e-14))

    def test_constant_statistic_gives_zero_width(self):
        f = _panel({"a": 3, "b": 2}, 30, value=3.25)
        for cluster in bs.CLUSTER_LEVELS:
            with self.subTest(cluster=cluster, statistic="mean of a constant column"):
                ci = bs.bootstrap_ci(f, _mean_cost, n_boot=60, seed=2, cluster=cluster)
                self.assertEqual(ci.width[0], 0.0)
                self.assertEqual((ci.lower[0], ci.upper[0], ci.estimate[0]), (3.25, 3.25, 3.25))
        g = _panel({"a": 3, "b": 2}, 30, seed=1)
        ci = bs.bootstrap_ci(g, lambda d: 7.0, n_boot=60, seed=2, cluster="dataset+series")
        self.assertEqual((ci.lower[0], ci.upper[0]), (7.0, 7.0))

    def test_interval_level_against_a_known_answer(self):
        """400 independent N(0,1) costs, single origins resampled. The bootstrap
        distribution of the mean is close to normal with sd s/sqrt(n), so a 95%
        interval is 2*1.96*s/sqrt(n) wide and an 80% interval 2*1.2816*s/sqrt(n).
        A 90% interval mislabelled as 95% would be 0.839 of the first."""
        n, B = 400, 1600
        x = np.random.default_rng(42).standard_normal(n)
        se = x.std() / np.sqrt(n)
        f = _one_series(x)
        for level, z in ((0.95, Z95), (0.80, Z80)):
            ci = bs.bootstrap_ci(f, _mean_cost, n_boot=B, level=level, seed=3, block_length=1)
            with self.subTest(level=level):
                self.assertAlmostEqual(ci.width[0] / (2 * z * se), 1.0, delta=0.06)
                self.assertAlmostEqual((ci.lower[0] + ci.upper[0]) / 2, x.mean(), delta=0.15 * se)
                tail = (1 - level) / 2 * B
                self.assertAlmostEqual(int((ci.replicates[:, 0] < ci.lower[0]).sum()), tail, delta=1.01)
                self.assertAlmostEqual(int((ci.replicates[:, 0] > ci.upper[0]).sum()), tail, delta=1.01)
                self.assertEqual(ci.level, level)

    def test_coverage_is_near_nominal_only_without_dependence(self):
        """An honest range, not a pass mark. With 60 independent costs the
        nominal 95% interval covers somewhat under 95% (52 of 60 when this was written,
        at these seeds and this small number of resamples). With overlapping windows (horizon
        24, an origin every step, 120 origins) it covers far less (38 of 60),
        and the test fails if that ever looks like 95%: the note and the
        docstrings say these intervals are approximate, and this pins why. The
        full simulation is coverage_sim.py."""
        reps, B = 60, 80
        hit = dict(independent=0, overlapping=0)
        for r in range(reps):
            x = np.random.default_rng(900 + r).standard_normal(60)
            ci = bs.bootstrap_ci(_one_series(x), _mean_cost, n_boot=B, seed=r)
            hit["independent"] += ci.lower[0] <= 0.0 <= ci.upper[0]
            f, stat, truth, _ = coverage_sim.make_overlap(np.random.default_rng(900 + r), n=120)
            ci = bs.bootstrap_ci(f, stat, n_boot=B, seed=r)
            hit["overlapping"] += ci.lower[0] <= truth <= ci.upper[0]
        independent, overlapping = hit["independent"] / reps, hit["overlapping"] / reps
        self.assertGreaterEqual(independent, 0.82, independent)
        self.assertLessEqual(independent, 0.99, independent)
        self.assertGreaterEqual(overlapping, 0.55, overlapping)
        self.assertLessEqual(overlapping, 0.85, overlapping)
        self.assertGreater(independent - overlapping, 0.08)

    def test_overlapping_windows_block_interval_against_the_true_width(self):
        """Hand-built case of the problem the block bootstrap is for.

        One series, horizon 24, an origin at every step. The cost of a window is
        the sum of 24 independent per-target errors, so windows at neighbouring
        origins share 23 of their 24 targets (lag-1 correlation about 23/24).
        The variance of the mean of 480 such costs is known exactly, so the true
        width of a 95% interval is known: 4.258. The block interval is far wider
        than the single-origin one and still narrower than the truth.
        """
        H, n = 24, 480
        f, stat, truth, sd = coverage_sim.make_overlap(np.random.default_rng(20261004), n=n, horizon=H)
        cost = f.cost.to_numpy()
        self.assertEqual(len(cost), n)
        self.assertGreater(np.corrcoef(cost[:-1], cost[1:])[0, 1], 0.9)
        k = np.arange(1, H)
        by_hand = np.sqrt((n * H + 2 * np.sum((n - k) * (H - k))) / n ** 2)
        self.assertAlmostEqual(sd, by_hand, places=12)
        true_width = 2 * Z95 * sd
        self.assertAlmostEqual(true_width, 4.258, places=3)
        self.assertEqual(int(bs.OriginPanel(f).block_lengths("auto").block_length.iloc[0]), H)
        for seed in range(5):
            block = bs.bootstrap_ci(f, stat, n_boot=400, seed=seed, block_length="auto")
            naive = bs.bootstrap_ci(f, stat, n_boot=400, seed=seed, block_length=1)
            with self.subTest(seed=seed):
                self.assertGreater(block.width[0], 2.0 * naive.width[0])
                self.assertLess(naive.width[0] / true_width, 0.30)
                self.assertGreater(block.width[0] / true_width, 0.55)
                self.assertLess(block.width[0] / true_width, 1.00)

    def test_independent_costs_block_and_naive_intervals_are_similar(self):
        """Control: with no autocorrelation the two intervals are about the same width."""
        f = _panel({"hand": 1}, 480, seed=9)
        block = bs.bootstrap_ci(f, _mean_cost, n_boot=400, seed=0, block_length=8)
        naive = bs.bootstrap_ci(f, _mean_cost, n_boot=400, seed=0, block_length=1)
        self.assertLess(abs(np.log(block.width[0] / naive.width[0])), np.log(1.5))

    def test_vector_statistic_and_frame(self):
        f = _panel({"a": 3, "b": 3}, 40, taus=(0.05, 0.5, 0.95), seed=6)
        f["cost"] = f["cost"] + 10 * f["tau"]                  # the three taus are told apart
        ci = bs.bootstrap_ci(f, lambda d: d.groupby("tau").cost.mean(), n_boot=50, seed=0,
                             cluster="dataset+series")
        self.assertEqual(ci.labels, (0.05, 0.5, 0.95))
        self.assertEqual(ci.replicates.shape, (50, 3))
        t = ci.to_frame()
        self.assertEqual(list(t.label), [0.05, 0.5, 0.95])
        want = f.groupby("tau").cost.mean()
        for j, tau in enumerate((0.05, 0.5, 0.95)):
            with self.subTest(tau=tau):
                self.assertAlmostEqual(t.estimate[j], want[tau], places=12)
                lo, hi = np.quantile(ci.replicates[:, j], [0.025, 0.975])     # per column, 95%
                self.assertAlmostEqual(t.lower[j], lo, places=12)
                self.assertAlmostEqual(t.upper[j], hi, places=12)
                self.assertLess(abs(t.estimate[j] - 10 * tau), 1.0)
                self.assertLess(t.lower[j], t.estimate[j])
                self.assertGreater(t.upper[j], t.estimate[j])
        self.assertEqual(set(t.n_boot), {50})
        self.assertEqual(set(t.nominal_level), {0.95})
        self.assertEqual(set(t.block_lengths_used), {"h1: L=4, n=40"})
        self.assertTrue(t.interval_kind.str.contains("approximate").all())
        self.assertNotIn("level", t.columns)                   # only ever called nominal

    def test_bad_arguments(self):
        f = _panel({"a": 2}, 20)
        with self.assertRaises(ValueError):
            bs.bootstrap_ci(f, _mean_cost, block_length=None, cluster="none")
        with self.assertRaises(ValueError):
            bs.bootstrap_ci(f, _mean_cost, level=1.0)
        with self.assertRaises(ValueError):
            bs.bootstrap_ci(f, _mean_cost, n_boot=0)
        with self.assertRaises(ValueError):
            bs.bootstrap_ci(f, lambda d: float("nan"), n_boot=5)
        with self.assertRaises(ValueError):
            bs.bootstrap_ci(f, _mean_cost, n_boot=5, block_length="longest")


class RegretStatistics(unittest.TestCase):
    """Fix 2 and the built-in statistics on hand-built tables with known answers."""

    def _equal_families(self):
        """The reviewer's construction: 50 series, 40 origins, two families with
        the same expected cost |1 + 0.3 N|, scale 1."""
        rng = np.random.default_rng(1)
        rows = []
        for s in range(50):
            a, b = 1 + 0.3 * rng.standard_normal(40), 1 + 0.3 * rng.standard_normal(40)
            rows.append(pd.DataFrame(dict(dataset="d", config="c", series_id=f"s{s:02d}", horizon=1,
                                          tau=0.5, origin=np.arange(40), pinball_classical=np.abs(a),
                                          pinball_tsfm=np.abs(b), scale=1.0, n=1)))
        return bs.attach_sample_oracle(pd.concat(rows, ignore_index=True))

    def test_default_interval_contains_the_estimate_on_the_reviewer_construction(self):
        c = self._equal_families()
        c["pinball_selected"] = c["pinball_oracle"]            # selector = sample-best family
        for cluster in ("none", "series"):
            with self.subTest(cluster=cluster, statistic="default, oracle family fixed"):
                ci = bs.bootstrap_ci(c, bs.stat_mean_scaled_regret, n_boot=80, seed=0, cluster=cluster)
                self.assertEqual(ci.estimate[0], 0.0)
                self.assertLessEqual(ci.lower[0], ci.estimate[0])
                self.assertGreaterEqual(ci.upper[0], ci.estimate[0])
            with self.subTest(cluster=cluster, statistic="recomputed oracle, kept as the biased option"):
                ci = bs.bootstrap_ci(c, bs.stat_mean_scaled_regret_recomputed_oracle, n_boot=80, seed=0,
                                     cluster=cluster)
                self.assertEqual(ci.estimate[0], 0.0)
                self.assertGreater(ci.lower[0], 0.002)         # the whole interval is above the estimate

    def test_default_interval_is_centred_and_replicates_can_be_negative(self):
        c = self._equal_families()
        c["pinball_selected"] = c["pinball_classical"]         # always classical
        fixed = bs.bootstrap_ci(c, bs.stat_mean_scaled_regret, n_boot=100, seed=0)
        self.assertGreater(fixed.estimate[0], 0.01)            # reported regret, biased upward (truth 0)
        self.assertLess(fixed.lower[0], fixed.estimate[0])
        self.assertGreater(fixed.upper[0], fixed.estimate[0])
        self.assertLess(abs(fixed.replicates.mean() - fixed.estimate[0]), 0.1 * fixed.width[0])
        # single units go negative on resamples: the fixed oracle family is not always the cheaper one
        one = bs.OriginPanel(c).resample(np.random.default_rng(0))
        u = bs._unit_means(one, extra=bs.FIXED_COLUMNS)
        self.assertGreater(int((u.pinball_selected < u.pinball_oracle).sum()), 0)
        recomputed = bs.bootstrap_ci(c, bs.stat_mean_scaled_regret_recomputed_oracle, n_boot=100, seed=0)
        self.assertGreater(recomputed.replicates.mean() - recomputed.estimate[0], 0.2 * recomputed.width[0])

    def test_fixed_oracle_unit_values_go_negative_on_a_resample(self):
        # two origins: TSFM cheaper on the whole (oracle), classical cheaper at origin 0
        rows = [("s0", o, c, t, 1.0) for o, (c, t) in enumerate([(1.0, 3.0)] + [(5.0, 1.0)] * 9)]
        c = bs.attach_sample_oracle(_costs(rows))
        self.assertEqual(set(c.oracle_family), {"tsfm"})
        c["pinball_selected"] = c["pinball_classical"]
        only_first = c[c.origin == 0].assign(boot_series=0)
        self.assertEqual(bs.stat_mean_scaled_regret(only_first), -2.0)            # 1 - 3, oracle held at TSFM
        self.assertEqual(bs.stat_mean_scaled_regret_recomputed_oracle(only_first), 0.0)

    def test_known_answers_scalar_and_grouped(self):
        """Three units with scaled regrets 0, 0 and 0.9 and scaled deltas -0.1,
        -0.2 and +0.9: the mean differs from the median in both."""
        rows = []
        for sid, (c, t, y) in {"a": (1.2, 1.0, 2.0), "b": (2.0, 1.0, 5.0), "c": (1.0, 5.5, 5.0)}.items():
            rows += [(sid, o, c, t, y) for o in range(6)]
        cost = bs.attach_sample_oracle(_costs(rows))
        self.assertEqual(cost.groupby("series_id").oracle_family.first().to_dict(),
                         {"a": "tsfm", "b": "tsfm", "c": "classical"})
        cost["pinball_selected"] = cost["pinball_tsfm"]        # always TSFM: regret only on c
        i = bs.OriginPanel(cost).identity()
        self.assertAlmostEqual(bs.stat_mean_scaled_regret(i), (0 + 0 + 4.5 / 5.0) / 3, places=15)
        self.assertAlmostEqual(bs.stat_mean_scaled_regret_recomputed_oracle(i), 0.3, places=15)
        self.assertAlmostEqual(bs.stat_mean_delta(i), (-0.1 - 0.2 + 0.9) / 3, places=15)
        by = bs.stat_mean_scaled_regret(i, by="tau")
        self.assertEqual(list(by.index), [0.5])
        self.assertAlmostEqual(by.iloc[0], 0.3, places=15)
        self.assertAlmostEqual(bs.stat_mean_delta(i, by="tau").iloc[0], 0.2, places=15)

    def test_delta_orientation_is_tsfm_minus_classical(self):
        cost = _costs([("a", o, 2.0, 1.0, 10.0) for o in range(6)])    # TSFM cheaper
        self.assertAlmostEqual(bs.stat_mean_delta(bs.OriginPanel(cost).identity()), -0.1, places=15)

    def test_missing_costs_raise_instead_of_being_skipped(self):
        cost = _costs([("a", o, 2.0, 1.0, 10.0) for o in range(30)])
        self.assertAlmostEqual(bs.stat_mean_delta(bs.OriginPanel(cost).identity()), -0.1)
        broken = cost.copy()
        broken.loc[:14, "pinball_classical"] = np.nan          # the reviewer's 15 of 30
        with self.assertRaises(ValueError):
            bs.stat_mean_delta(bs.OriginPanel(broken).identity())
        with self.assertRaises(ValueError):
            bs.bootstrap_ci(broken, bs.stat_mean_delta, n_boot=10, seed=0)
        with self.assertRaises(ValueError):
            bs.attach_sample_oracle(broken)

    def test_regret_statistic_needs_both_fixed_columns(self):
        cost = _costs([("a", o, 2.0, 1.0, 10.0) for o in range(6)])
        cost["pinball_selected"] = cost["pinball_classical"]
        with self.assertRaises(ValueError) as cm:
            bs.stat_mean_scaled_regret(bs.OriginPanel(cost).identity())
        self.assertIn("pinball_oracle", str(cm.exception))


class OnEngineTables(unittest.TestCase):
    """The cost tables and statistics agree with the engine on the fixture (horizon 24)."""

    def setUp(self):
        t = _shared.tables()
        self.res, _ = _shared.results()
        self.po = t["per_origin"][t["per_origin"].horizon == 24]
        self.tables = t

    def test_fixed_choice_statistic_equals_engine_mean_scaled_regret(self):
        eng = summarize(self.res).set_index(["selector", "horizon", "tau"]).mean_scaled_regret
        for sel in ("always_classical", "static_omega", "pinball_validation", "d_omega",
                    "hindsight_oracle"):
            with self.subTest(selector=sel):
                cost = bs.fixed_choice_costs(self.po, self.res, sel)
                i = bs.OriginPanel(cost).identity()
                got = bs.stat_mean_scaled_regret(i, by="tau")
                want = eng.xs((sel, 24), level=("selector", "horizon")).reindex(got.index)
                self.assertTrue(np.allclose(got, want, rtol=1e-10, atol=1e-14))
                units = self.res[(self.res.selector == sel) & (self.res.horizon == 24)]
                self.assertAlmostEqual(bs.stat_mean_scaled_regret(i), units.scaled_regret.mean(), places=13)

    def test_delta_statistic_equals_e1_mean_delta(self):
        cost = bs.family_cost_table(self.po)
        got = bs.stat_mean_delta(bs.OriginPanel(cost).identity(), by="tau")
        s = e1.summarize_map(e1.decision_map(self.tables))
        want = s[s.horizon == 24].set_index("tau").mean_delta.reindex(got.index)
        self.assertTrue(np.allclose(got, want, rtol=1e-10, atol=1e-14))
        self.assertGreater(np.abs(want).min(), 1e-5)           # so a flipped sign cannot pass

    def test_only_test_origins_enter(self):
        cost = bs.family_cost_table(self.po)
        test_origins = set(self.po[self.po.role == "test"].origin)
        self.assertEqual(set(cost.origin), test_origins)
        self.assertEqual(len(test_origins), 20)
        self.assertEqual(set(bs.fixed_choice_costs(self.po, self.res, "d_omega").origin), test_origins)

    def test_choice_and_oracle_family_stay_fixed_under_resampling(self):
        po = self.po[self.po.tau.isin([0.25, 0.5, 0.75])]      # 0.25 and 0.75 are near-ties on fx_tau_flip
        cost = bs.fixed_choice_costs(po, self.res, "pinball_validation")
        orc = self.res[(self.res.selector == "hindsight_oracle") & (self.res.horizon == 24)]
        want_orc = orc.set_index(["series_id", "tau"]).oracle_family
        want = cost.groupby(["series_id", "tau"]).chosen_family.first()
        panel = bs.OriginPanel(cost)
        negative = 0
        for seed in range(4):
            r = panel.resample(np.random.default_rng(seed), "auto", "dataset+series")
            per = r.groupby(["series_id", "tau"])[["chosen_family", "oracle_family"]].agg(["nunique", "first"])
            self.assertTrue((per[("chosen_family", "nunique")] == 1).all())
            self.assertTrue((per[("oracle_family", "nunique")] == 1).all())
            self.assertTrue((per[("chosen_family", "first")] == want.reindex(per.index)).all())
            self.assertTrue((per[("oracle_family", "first")] == want_orc.reindex(per.index)).all())
            for col, fam in (("pinball_selected", r.chosen_family), ("pinball_oracle", r.oracle_family)):
                picked = np.where(fam == "classical", r.pinball_classical, r.pinball_tsfm)
                self.assertTrue(np.array_equal(picked, r[col].to_numpy()))
            u = bs._unit_means(r, extra=bs.FIXED_COLUMNS)
            negative += int((u.pinball_selected < u.pinball_oracle).sum())
        self.assertGreater(negative, 0)             # the fixed oracle is not re-chosen per resample

    def test_fixture_intervals_contain_their_estimate_at_every_tau(self):
        for sel in ("static_omega", "pinball_validation"):
            cost = bs.fixed_choice_costs(self.po, self.res, sel)
            ci = bs.bootstrap_ci(cost, lambda d: bs.stat_mean_scaled_regret(d, by="tau"), n_boot=60,
                                 seed=0, cluster="dataset+series")
            with self.subTest(selector=sel):
                self.assertEqual(len(ci.estimate), 11)
                self.assertTrue((ci.lower <= ci.estimate).all())
                self.assertTrue((ci.upper >= ci.estimate).all())
                self.assertGreater(ci.width.max(), 1e-4)
        orc = bs.fixed_choice_costs(self.po[self.po.tau == 0.5], self.res, "hindsight_oracle")
        ci = bs.bootstrap_ci(orc, bs.stat_mean_scaled_regret, n_boot=30, seed=0)
        # the oracle against itself is zero by definition, on every resample
        self.assertEqual((ci.estimate[0], ci.lower[0], ci.upper[0]), (0.0, 0.0, 0.0))

    def test_selection_frame_from_other_costs_is_refused(self):
        res = self.res.copy()
        flip = (res.selector == "d_omega") & (res.dataset == "fx_tsfm_wins") & (res.tau == 0.5)
        res.loc[flip, "oracle_family"] = "classical"           # not the cheaper family there
        with self.assertRaises(ValueError) as cm:
            bs.fixed_choice_costs(self.po, res, "d_omega")
        self.assertIn("do not belong together", str(cm.exception))

    def test_undecided_selector_or_unknown_selector_raises(self):
        with self.assertRaises(ValueError):
            bs.fixed_choice_costs(self.po, self.res, "no_such_selector")
        res = self.res.copy()
        res.loc[res.selector == "d_omega", "status"] = "insufficient_data"
        with self.assertRaises(ValueError):
            bs.fixed_choice_costs(self.po, res, "d_omega")

    def test_e1_interval_table_uses_test_origins_and_says_how_it_was_made(self):
        t = dict(self.tables, per_origin=self.po)
        ci = e1.mean_delta_intervals(t, n_boot=40, seed=0)
        self.assertEqual(len(ci), 11)
        s = e1.summarize_map(e1.decision_map(self.tables))
        want = s[s.horizon == 24].set_index("tau").mean_delta.reindex(ci.tau)
        # the estimate is the map's own mean Delta, which is computed on TEST origins only
        self.assertTrue(np.allclose(ci.mean_delta, want, rtol=1e-10, atol=1e-14))
        self.assertTrue((ci.mean_delta_lower < ci.mean_delta).all())
        self.assertTrue((ci.mean_delta_upper > ci.mean_delta).all())
        self.assertEqual(set(ci.cluster), {"dataset+series"})
        self.assertEqual(set(ci.n_boot), {40})
        self.assertEqual(set(ci.horizon), {24})
        self.assertEqual(set(ci.block_lengths_used), {"h24: L=3, n=20"})
        self.assertEqual(set(ci.time_blocks), {"per_series"})
        shared = e1.mean_delta_intervals(t, n_boot=40, seed=0, time_blocks="shared_within_dataset")
        self.assertEqual(set(shared.time_blocks), {"shared_within_dataset"})
        self.assertFalse(np.allclose(shared.mean_delta_lower, ci.mean_delta_lower))


if __name__ == "__main__":
    unittest.main()
