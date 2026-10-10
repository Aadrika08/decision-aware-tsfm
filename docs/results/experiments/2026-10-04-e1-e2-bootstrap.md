# E1 runner, E2 fit/freeze/evaluate, block bootstrap: fixes after independent review

Status: synthetic evidence only. Everything below was run on the engine's synthetic fixture, on
hand-built tables with known answers, or on simulated processes with a closed-form truth. No real
forecast cache was read. No M5 or PJM data exists in the tree or was touched. The numbers are
interface and mechanism checks. None of them is a result about real model families.

This note replaces the earlier note of the same name. An independent reviewer found three defects
that gave wrong output without an error, and reported that 15 of the 16 bugs they planted passed
the test suite. This change fixes the defects, adds tests that catch all sixteen, and corrects what
the earlier note claimed about the bootstrap intervals and the frozen-file hash.

## Commands

All from the tree root with the team's pinned interpreter (Python 3.11.15, numpy 1.26.4,
pandas 2.2.3, scipy 1.11.4). `python` below means that interpreter.

    PYTHONPATH=src python -m unittest discover -s tests/unit/experiments -t .
    PYTHONPATH=src python -m unittest discover -s tests/unit/decision -t .
    PYTHONPATH=src python -m decision_aware_tsfm.analysis.experiments.e1 --out artifacts/runs/e1_fixture
    PYTHONPATH=src python -m decision_aware_tsfm.analysis.experiments.e2 --out artifacts/runs/e2_fixture
    PYTHONPATH=src python -m decision_aware_tsfm.analysis.experiments.coverage_sim --out artifacts/runs/bootstrap_coverage --jobs 2
    python tests/unit/experiments/planted_bugs.py

Every number in this note is printed by one of these six commands or is in a file they write
under `artifacts/runs/` (git-ignored). Defaults: fixture seed 7; E1 and E2 use 200 bootstrap
resamples with bootstrap seed 0; the coverage simulation uses 300 replications of 300 resamples
with seed 20261004 and takes about 85 CPU-minutes. `planted_bugs.py` copies the source to a
temporary folder and never writes to the working tree.

## What was built

Code in `src/decision_aware_tsfm/analysis/experiments/` (`e1.py`, `e2.py`, `bootstrap.py`, new
`coverage_sim.py`), tests and the planted-bug tool in `tests/unit/experiments/`. Nothing outside
those paths and this note was changed. The engine change this work relies on
(`pipeline.build_tables` returning `tables["settings"]`) was made by the coordinator.

### bootstrap.py

What is resampled, from the outside in:

1. Datasets (`cluster="dataset"` or `"dataset+series"`): whole datasets, with replacement.
2. Series (`cluster="series"` or `"dataset+series"`): whole series, with replacement, within each
   dataset, so the number of series per dataset is fixed.
3. Origins (`block_length` not `None`): within each (series, horizon), moving blocks of
   consecutive origins. All rows of an origin (every tau) move together. With
   `time_blocks="per_series"` (default) each series draws its own block positions. With
   `time_blocks="shared_within_dataset"` (new) one set of positions is drawn per (dataset,
   horizon) and used by every series of that dataset.

What is held fixed: the set of units and the tau grid, the number of series per dataset and of
origins per series, Omega, the validation/test split, the fitted selector, each selector's chosen
family per unit, and (new, in the default regret statistic) the hindsight-oracle family per unit.

What an interval is an interval for: the value the statistic would take on average over
evaluation samples drawn the way the resampling draws them, with everything in the previous
paragraph unchanged. For `stat_mean_delta` this is the mean scaled cost difference between the two
families' representatives. For `stat_mean_scaled_regret` it is the mean scaled cost gap between
the family the selector chose and the family that was cheaper on this test sample.

What an interval is known not to cover:

- The variability of fitting or validating a selector on another sample.
- The choice of the oracle family from the same test costs. Reported regret is biased upward as
  an estimate of regret against the truly better family, and the interval is centred on the
  reported value. See "regret equal families" in the coverage table: truth 0, the interval
  contained 0 in 0 of 300 replications.
- Shocks shared by series at the same origin, when `time_blocks="per_series"`.
- Shocks shared across datasets at the same calendar time, in either mode.
- Dataset-level uncertainty when there is one dataset. M5 and PJM are one dataset each, so
  `cluster="dataset"` cannot be used for the transfer test (it now raises). The transfer intervals
  can only describe series-level and time-block variation within each held-out dataset.
- Serial dependence longer than a block. See the coverage table.
- In the other direction: stacking the series stage and the block stage counts the noise within
  a series twice, so that combination is too wide when series are alike.
- Multiple comparisons. The 22 (horizon, tau) intervals are pointwise.

Changes in this module:

- A stage that cannot resample raises `DegenerateResampleError` before any resample is drawn. It
  used to return the original table every time and report an interval of width zero. Refused:
  fewer than 5 distinct block start positions in any series, a series shorter than 2 blocks,
  fewer than 2 datasets when datasets are resampled, any dataset with fewer than 2 series when
  series are resampled. Block lengths are no longer capped at the number of origins.
- The default regret statistic holds both the selector's family and the oracle family fixed, so
  the bootstrap distribution is centred on the reported regret. The earlier statistic (oracle
  recomputed on each resample) is kept as `stat_mean_scaled_regret_recomputed_oracle`; its
  docstring says it is biased upward and is not a confidence interval for the reported regret.
- `time_blocks="shared_within_dataset"` as above. It needs identical origins for all series of a
  dataset at a horizon and raises `MisalignedOriginsError` otherwise.
- Missing costs raise in every built-in statistic. They used to be skipped by pandas.
- Every interval table carries `nominal_level`, `n_boot`, `boot_seed`, `cluster`,
  `block_lengths_used` (per horizon), `time_blocks` and `interval_kind`. The column is called
  `nominal_level`, not `level`, on purpose.
- The docstring of the default block length no longer claims that the cube-root term protects
  against autocorrelation. The rule itself is unchanged: `max(ceil(horizon / origin step),
  ceil(n_origins^(1/3)))`.

### e1.py

- Delta now follows `docs/project-overview.md`, which says: "Delta(x, tau) = J_TSFM(x, tau) -
  J_classical(x, tau)" and "`Delta < 0` -> the TSFM family is decision-optimal. Otherwise the
  classical family is." (the file uses a typographic minus and arrow). The code reports that
  quantity divided by the unit's scale: `delta = scaled_cost_tsfm - scaled_cost_classical`. The
  same orientation is used in `bootstrap.stat_mean_delta`, in every CSV, and in the manifest
  (`delta_definition`, replacing `delta_sign`). Column names did not need to change because none
  of them encoded the old orientation; `share_tsfm_lower` is unaffected.
- Ties are judged on scaled cost.
- The tau grid is checked for every series at every horizon, in the units, the test costs and
  every selector's results. An extra tau raises as well. A selector whose units are not the map's
  units raises.
- The symmetry table reports the number of units on each side and raises
  `UnequalPopulationError` when they differ.
- The boundary Omega*(tau) is still not fitted here. That is gated on K3 (Abdullah).

### e2.py

- `fit(fit_tables, selector_kind, static_threshold, selector_name)` reads representatives,
  val_fraction, omega_col, omega_variant, crps_method and the tau grid from
  `fit_tables["settings"]`. They are no longer typed in by the caller.
- `evaluate(frozen, tables, allow_held_out=False, ledger_path=None)` refuses tables with no
  `settings`, refuses a mismatch in any of those six settings, and checks that the models scored
  are exactly the frozen representatives.
- The specification records a SHA-256 of the selector's source code (the adapter class and the
  module that defines the selector it builds). `evaluate` recomputes it and refuses when it
  differs. Code that the selector calls in other modules is not covered.
- `freeze` validates the specification and accepts only a `FitSpec` made by `fit`.
- Held-out evaluation needs `allow_held_out=True` (the bool) and a `ledger_path`. One JSON line
  (UTC time, frozen hash, selector code hash, datasets, held-out names, number of units) is
  appended for every evaluation that touched held-out names, before any selector runs.
- Every derived result table has a `frozen_sha256` column; `EvalResult.write` puts the hash in the
  manifest and in the file name of the selection frame.
- The frozen selector and the baselines receive copies of the units and validation costs. The
  test table is restricted to the evaluated units, so the engine's best-fixed family is chosen on
  the evaluated population.
- Fit/evaluation overlap is checked on names after Unicode normalisation, trimming and case
  folding. The tau grid is checked for every series at every horizon in `fit` and `evaluate`.
- `regret_summary` replaces `normalized_regret` and reports two separately named figures (see
  design choice 2).
- What the hash is: an unkeyed SHA-256 of the file's own content. It catches accidental edits. It
  is not a security control. A person who edits the file and recomputes the hash, or writes a
  file by hand that never went through `fit`, gets a file the module accepts; a test documents
  this. The fit step can therefore be skipped by someone who sets out to skip it. What makes a
  held-out evaluation credible is the ledger and the frozen file kept under version control.
- Only the engine's placeholder `DOmegaLinear` is registered as a selector kind. The real D_Omega
  (Abdullah) has not been run through this code.

### coverage_sim.py and planted_bugs.py

`coverage_sim.py` simulates processes with a known answer and reports how often the nominal 95%
interval contains it. `planted_bugs.py` plants one bug at a time in a copy of the tree, runs the
suite, restores the file in a `finally` block, compares it byte for byte with the original, and
at the end checks that a clean copy passes and that the working tree was not modified.

## Verified in this change

Tests: `Ran 159 tests in 30.571s`, `OK`. By file: test_bootstrap 53, test_e1 34, test_e2 68,
test_experiments_no_inference 4. The decision-engine suite still passes (`Ran 77 tests in
23.364s`, `OK (skipped=1)`).

### E1 on the fixture (synthetic)

- 484 units, horizons 1 and 24, 11 taus, 22 units per (horizon, tau) on both sides of every
  symmetry pair.
- Ties: 88 units, all in fx_tie. Lower-cost family by dataset: fx_tsfm_wins 124 TSFM and 8
  classical; fx_classical_wins 128 classical and 4 TSFM; fx_tau_flip 54 TSFM and 78 classical.
- Mean Delta over the 22 cells ranges from -0.00047 to 0.00089. It is negative (TSFM cheaper) at
  taus 0.40, 0.50 and 0.60 at both horizons and positive elsewhere.
- Sign reversals per cell: 1 to 5.
- Symmetry table (upper minus lower): share differences from -0.136 to 0.182, mean-Delta
  differences from -0.00025 to 0.00019. The fixture was not designed to be symmetric or
  asymmetric; these only show the table is filled.
- Mean Delta intervals (nominal 95%, 200 resamples, seed 0, datasets and series as clusters,
  per-series time blocks, block length 7 over 320 test origins at horizon 1 and 3 over 20 test
  origins at horizon 24): every one of the 22 intervals contains zero. For example horizon 24,
  tau 0.50: -0.00044, interval -0.02938 to 0.02269. With four datasets the dataset stage is very
  coarse; these intervals show the plumbing, nothing else.

### E2 on the fixture (synthetic)

- Placeholder fitted on fx_dfit (12 series, 264 units): omega0 = 0.45, beta = 0.0. The fixture's
  static threshold is also 0.45, so on the fixture d_omega and static_omega are the same rule:
  they choose differently on 0 of 484 units. The fixture cannot tell the two selectors apart, and
  no number here says anything about a tau-dependent boundary.
- 484 evaluation units, 7 selectors. Minimum regret 0, maximum oracle regret 0.
- 154 groups (7 selectors, 2 horizons, 11 taus); both regret figures defined in all 154.
- The two figures differ widely. For d_omega at horizon 24, tau 0.09: mean scaled regret 0.00061,
  regret over oracle cost 0.0375, fraction of oracle value recovered -23.43. At taus 0.25 to 0.60
  the regret is 0 and the fraction is 1.
- Gap to pinball validation over the 22 cells: -0.000107 to 0.000849 in scaled cost.
- Worst decile: 22 series per cell, 3 in the worst decile, values 0 to 0.00370.
- Regret intervals for d_omega (nominal 95%, 200 resamples, seed 0, datasets and series as
  clusters, per-series time blocks, block lengths as above): the default statistic's interval
  contains its estimate in 22 of 22 cells. Example, horizon 24, tau 0.95: 0.000849, interval
  0.000000 to 0.002560. At taus 0.25 to 0.60 the selector's family costs exactly what the
  oracle family costs on every unit (the same family, or an exact tie), so the statistic is zero
  on every resample and the interval is 0 to 0. That is a
  definition, not certainty about the true regret.
- Selector timing is written to `selector_cost.csv` and labelled as a single run that is not
  reproducible. No timing is quoted here.
- A second run of the command into the same folder reuses the frozen file (it is named by its
  hash) and prints the same output.

### Coverage simulation (synthetic, closed-form truth)

300 replications, 300 resamples each, nominal level 95%, seed 20261004. "Width ratio" is the mean
interval width divided by the true width `2 * 1.96 * sd` of a correct 95% interval. The standard
error of a coverage figure is at most 0.029. The first 21 rows come from one run of the command. The
last row was appended to the design list afterwards and run alone with `--only`; appending does not
change the seeds of earlier rows, and one earlier row rerun alone reproduced its figures exactly.

| design | block lengths | truth | mean estimate | mean width | true width | width ratio | coverage | contains own estimate |
|---|---|---|---|---|---|---|---|---|
| overlap H=24 stride 1 n=480, auto block | h24: L=24, n=480 | 0.0000 | 0.0324 | 3.2240 | 4.258 | 0.76 | 0.803 | 1.000 |
| overlap H=24 stride 1 n=480, single origins (naive) | h24: L=1, n=480 | 0.0000 | -0.0106 | 0.8420 | 4.258 | 0.20 | 0.310 | 1.000 |
| overlap H=24 stride 1 n=480, block 48 | h24: L=48, n=480 | 0.0000 | 0.0783 | 3.5745 | 4.258 | 0.84 | 0.870 | 1.000 |
| overlap H=24 stride 1 n=120, auto block | h24: L=24, n=120 | 0.0000 | -0.1473 | 5.1691 | 8.297 | 0.62 | 0.707 | 1.000 |
| ar1 phi=0.0 n=320 H=1, auto block | h1: L=7, n=320 | 0.0000 | -0.0080 | 0.2144 | 0.219 | 0.98 | 0.930 | 1.000 |
| ar1 phi=0.5 n=320 H=1, auto block | h1: L=7, n=320 | 0.0000 | 0.0066 | 0.3790 | 0.437 | 0.87 | 0.913 | 1.000 |
| ar1 phi=0.9 n=320 H=1, auto block | h1: L=7, n=320 | 0.0000 | 0.0407 | 1.0947 | 2.159 | 0.51 | 0.637 | 1.000 |
| ar1_h24 phi=0.0 n=20 H=24 step 24, auto block | h24: L=3, n=20 | 0.0000 | -0.0206 | 0.7906 | 0.877 | 0.90 | 0.880 | 1.000 |
| ar1_h24 phi=0.5 n=20 H=24 step 24, auto block | h24: L=3, n=20 | 0.0000 | 0.0477 | 1.0656 | 1.694 | 0.63 | 0.763 | 1.000 |
| shock 20x60, per-series blocks, cluster none | h24: L=4, n=60 | 0.0000 | -0.0007 | 0.1530 | 0.519 | 0.30 | 0.463 | 1.000 |
| shock 20x60, per-series blocks, cluster series | h24: L=4, n=60 | 0.0000 | -0.0014 | 0.1893 | 0.519 | 0.37 | 0.560 | 1.000 |
| shock 20x60, shared blocks, cluster none | h24: L=4, n=60 | 0.0000 | 0.0046 | 0.4876 | 0.519 | 0.94 | 0.940 | 1.000 |
| shock 20x60, shared blocks, cluster series | h24: L=4, n=60 | 0.0000 | 0.0089 | 0.5023 | 0.519 | 0.97 | 0.917 | 1.000 |
| shock_none 20x60, per-series blocks, cluster none | h24: L=4, n=60 | 0.0000 | 0.0010 | 0.1072 | 0.113 | 0.95 | 0.927 | 1.000 |
| shock_none 20x60, shared blocks, cluster none | h24: L=4, n=60 | 0.0000 | -0.0023 | 0.1045 | 0.113 | 0.92 | 0.917 | 1.000 |
| regret clear gap 0.3, always classical, fixed oracle | h1: L=4, n=40 | 0.2981 | 0.2983 | 0.0444 | n/a | n/a | 0.920 | 1.000 |
| regret clear gap 0.3, always classical, fixed oracle, cluster series | h1: L=4, n=40 | 0.2981 | 0.2968 | 0.0641 | n/a | n/a | 0.993 | 1.000 |
| regret equal families, always classical, fixed oracle | h1: L=4, n=40 | 0.0000 | 0.0268 | 0.0245 | n/a | n/a | 0.000 | 1.000 |
| regret equal families, always classical, recomputed oracle | h1: L=4, n=40 | 0.0000 | 0.0267 | 0.0217 | n/a | n/a | 0.000 | 0.533 |
| regret equal families, sample-best selector, fixed oracle | h1: L=4, n=40 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | 1.000 | 1.000 |
| regret equal families, sample-best selector, recomputed oracle | h1: L=4, n=40 | 0.0000 | 0.0000 | 0.0130 | n/a | n/a | 0.000 | 0.000 |
| regret clear gap 0.3, always classical, fixed oracle, cluster series, origins not resampled | origins not resampled | 0.2981 | 0.2990 | 0.0456 | n/a | n/a | 0.937 | 1.000 |

Reading the table:

- Independent costs (ar1 phi=0.0, shock_none): coverage 0.93 and 0.93, width ratio 0.98 and 0.95.
  Close to nominal, slightly under.
- Overlapping windows, horizon 24, an origin every step: the block interval is 0.76 of the true
  width and covers 0.80 with 480 origins (20 blocks), and 0.62 and 0.71 with 120 origins (5
  blocks). Resampling single origins gives 0.20 of the true width and covers 0.31, so blocks help
  a great deal and are still not enough. Doubling the block to 48 gives 0.84 and 0.87. This is
  the hand-built overlap case of the unit tests: true width 4.258, mean block-bootstrap width
  3.224, mean single-origin width 0.842.
- Autocorrelated costs without overlap: phi 0.5 covers 0.91, phi 0.9 covers 0.64 at width ratio
  0.51. The cube-root term of the default block length (7 for 320 origins) does not protect
  against strong autocorrelation.
- The fixture's horizon-24 shape (20 origins, block 3): 0.88 with independent costs, 0.76 with
  phi 0.5.
- Common shock at each origin, 20 series: per-series blocks cover 0.46 (0.56 with series
  clustering) at 0.30 and 0.37 of the true width. Shared blocks cover 0.94 (0.92 with series
  clustering) at 0.94 and 0.97. Without a common shock shared blocks cover 0.92 against 0.93 for
  per-series blocks, so sharing costs little when it is not needed.
- Regret, clear gap: the default interval covers the true gap 0.92 with blocks only and 0.94
  with series resampled and origins left alone. With both stages stacked it is 1.44 times as wide
  as with blocks only (0.0641 against 0.0444) and covers 0.99: the noise within a series is
  counted twice.
- Regret, two families with equal expected cost (true regret 0): the always-classical selector
  is reported with mean regret 0.027. The default interval contains its own estimate in every
  replication and contains the truth in none: it is an interval around a biased number. The
  recomputed-oracle interval contains its own estimate in 0.53 of replications. For the
  sample-best selector (reported regret exactly 0) the default interval is 0 to 0 and the
  recomputed-oracle interval excludes 0 in every replication.

### Planted bugs

The reviewer's sixteen bugs (M1 to M16) were replanted against the fixed code, one at a time, by
`planted_bugs.py`, together with nine more (N1 to N9) aimed at the new code. The reviewer
reported that before this change only M12 was caught; that run was not repeated here.

| bug | what was planted | outcome | tests that failed |
|---|---|---|---|
| M1 | bootstrap_ci uses quantiles (a, 1-a): a 90% interval labelled 95% | caught | Interval.test_coverage_is_near_nominal_only_without_dependence, Interval.test_interval_level_against_a_known_answer, Interval.test_vector_statistic_and_frame |
| M2 | moving_block_indices never draws the last admissible block start | caught | MovingBlocks.test_every_admissible_start_is_drawn_about_equally_often |
| M3 | cluster='dataset+series' skips the series stage | caught | Clusters.test_two_stage_resamples_series_inside_drawn_datasets |
| M4 | e1.mean_delta_intervals uses all origins, not test only | caught | DecisionMapOnFixture.test_cli_writes_outputs, OnEngineTables.test_e1_interval_table_uses_test_origins_and_says_how_it_was_made |
| M5 | e2.evaluate hands test costs to the frozen selector | caught | DropInSelector.test_frozen_selector_never_receives_test_costs |
| M6 | _check_payload drops the held-out check on fit configs and fit series | caught | TamperAndFrozenOnly.test_file_with_a_correct_hash_naming_held_out_fit_data_is_refused |
| M7 | e2.evaluate: dataset-name overlap check removed | caught | Separation.test_dataset_name_check_works_without_the_series_check |
| M8 | p90_series_regret is actually the median | caught | MetricsOnHandBuiltTables.test_p90_is_the_ninetieth_percentile |
| M9 | symmetry table reads n_units_upper from the lower row | caught | Symmetry.test_unequal_populations_are_refused |
| M10 | static threshold hard-coded to 0.45, frozen value ignored | caught | EvaluateAndMetrics.test_static_rule_uses_the_frozen_threshold |
| M11 | e1.selector_regret counts undecided units in n_selected | caught | SelectorRegret.test_undecided_units_are_not_counted_as_selected |
| M12 | e1.decision_map: tie tolerance loosened from 1e-12 to 1e-4 | caught | DecisionMapOnFixture.test_delta_agrees_with_engine_oracle, DecisionMapOnFixture.test_exact_tie_dataset_is_reported_as_ties |
| M13 | e2.evaluate: selector-name check removed | caught | DropInSelector.test_selector_name_must_match_the_frozen_name_at_evaluation |
| M14 | scalar stat_mean_scaled_regret returns the median | caught | OnEngineTables.test_fixed_choice_statistic_equals_engine_mean_scaled_regret, RegretStatistics.test_default_interval_is_centred_and_replicates_can_be_negative, RegretStatistics.test_known_answers_scalar_and_grouped |
| M15 | gap_to_validation counts undecided units as gap 0 | caught | MetricsOnHandBuiltTables.test_gap_to_validation_leaves_undecided_units_out |
| M16 | regret summary sums over all units, not only selected ones | caught | MetricsOnHandBuiltTables.test_regret_summary_sums_over_selected_units_only |
| N1 | shared time blocks are drawn per series after all | caught | SharedTimeBlocks.test_all_series_of_a_dataset_get_the_same_positions, SharedTimeBlocks.test_reviewer_common_shock_construction, SharedTimeBlocks.test_shared_positions_also_under_series_clustering_and_per_horizon |
| N2 | default regret statistic recomputes the oracle per resample | caught | OnEngineTables.test_fixture_intervals_contain_their_estimate_at_every_tau, RegretStatistics.test_default_interval_contains_the_estimate_on_the_reviewer_construction, RegretStatistics.test_default_interval_is_centred_and_replicates_can_be_negative (+1 more) |
| N3 | check_block accepts a single block start (zero-width intervals again) | caught | BlockLength.test_check_block_floors, BlockLength.test_default_rule_refuses_a_series_too_short_for_its_block, MovingBlocks.test_block_as_long_as_the_series_is_refused (+3 more) |
| N4 | evaluate does not compare crps_method with the frozen value | caught | FrozenSettings.test_every_frozen_setting_is_enforced |
| N5 | selector code hash is computed but not compared | caught | DropInSelector.test_selector_code_changed_after_freezing_is_refused |
| N6 | held-out evaluation is not written to the ledger | caught | HeldOutGate.test_every_held_out_evaluation_appends_one_ledger_line, HeldOutGate.test_ledger_line_is_written_before_any_selector_runs |
| N7 | Delta orientation flipped back to classical minus TSFM in the map | caught | DecisionMapOnFixture.test_cli_writes_outputs, DecisionMapOnFixture.test_delta_agrees_with_engine_oracle, DecisionMapOnFixture.test_tau_flip_is_tsfm_near_median_and_classical_in_both_tails (+6 more) |
| N8 | evaluate does not restrict the test table to the evaluated units | caught | Separation.test_best_fixed_is_chosen_on_the_evaluated_units_only |
| N9 | allow_held_out accepts any truthy value | caught | HeldOutGate.test_flag_must_be_the_bool_true |

25 of 25 caught, 0 survived. A clean copy passed after all restores and the working tree was
not modified by the tool.

A planted-bug run shows that these particular defects are detected. It does not show that the
suite detects defects in general.

Assertions that could not fail were removed or replaced: `lower <= upper` in the two interval
tests, the formula-against-itself line in `test_overlap_count`, the test with no assertion on
renamed fit data, `seconds > 0` on the timing table, and the symmetry test that repeated the
code's own subtraction.

## Design choices needing sign-off

1. **Delta orientation.** Now the documented one: Delta = J_TSFM - J_classical, negative means
   TSFM is cheaper. Alternative: the earlier classical-minus-TSFM ("TSFM advantage is positive").
   Reason: the overview defines it and E3 already uses it; two orientations in one project would
   produce sign errors in the paper.

2. **Which metric is "normalized regret".** Two are reported, each under its own name, and
   neither is called normalized regret:
   - `regret_over_oracle_cost` = (SEL - ORACLE) / ORACLE. The overview's glossary says of the
     hindsight family oracle: "The ceiling, and the denominator for normalised regret." The plan's
     comparator table says: "Ceiling, and the denominator for normalized regret".
   - `frac_oracle_value_recovered` = (BEST_FIXED - SEL) / (BEST_FIXED - ORACLE), undefined when
     best-fixed equals the oracle. The proposal lists, after "Primary: normalized decision regret
     J_m-hat - min_m J_m": "fraction of hindsight-oracle selection value recovered" as a separate
     metric.
   The earlier code called (SEL - ORACLE) / (BEST_FIXED - ORACLE) normalized regret, which is one
   minus the second metric. A mentor has to say which quantity the paper means. The proposal's own
   formula for the primary metric has no denominator at all. (Plan and proposal were read from the
   text extracts of `docs/implementation-plan.pdf` and `docs/proposal.pdf` supplied by the
   coordinator; the PDFs themselves were not opened.)
   Both are ratios of sums of scaled cost over the units the selector decided. Alternative: the
   mean of per-unit ratios. Not chosen because a unit with oracle cost near zero would dominate.

3. **What the bootstrap intervals may be called.** Proposed wording: "nominal 95% percentile
   block-bootstrap interval (approximate)". Alternative: "95% confidence interval". Not
   supportable: in the simulation with the default settings the nominal 95% interval
   covered the truth between 0.64 and 0.93 of the time depending on the dependence in the costs,
   and 0.46 when series shared a shock. A mentor
   should decide whether that wording is acceptable in the paper or whether a better-calibrated
   method is needed first (see Open).

4. **Regret under resampling.** Default: selector family and oracle family both fixed.
   Alternative: recompute the oracle per resample (kept, named, not default). Reason: with the
   oracle recomputed every unit contributes a non-negative amount on every resample, so the
   interval can exclude the reported value. Cost of the choice: the interval is for a gap between
   two fixed labels and ignores that the oracle label was chosen from the same data.

5. **Time blocks.** Default per series; option shared within a dataset. Alternative: make shared
   the default wherever origins align. In the simulation shared blocks were close to nominal with
   a common shock and not worse without one, so shared looks like the better default for real
   data, where series of a dataset share a calendar. It is left as an option here because the
   fix list asked for an option and because it fails on unaligned origins. A mentor should choose
   the default for the paper.

6. **Refusal floors.** 5 block starts, 2 blocks, 2 clusters. Alternative: 2 block starts, the
   bare minimum that is not degenerate. Reason for 5: with 2 starts and 2 blocks there are 4
   distinct resamples, and a percentile interval over 4 values is not an interval. The floors
   stop degenerate output only; the n=20 and n=120 rows of the coverage table pass them and still
   under-cover.

7. **Block length default.** Unchanged rule, corrected docstring. Alternatives: a longer fixed
   multiple of the overlap (the block-48 row), or a data-driven rule (Politis and White; not
   implemented, and the paper was not fetched in this session). Not changed here because tuning
   the rule to the simulation would be fitting the method to its own test.

8. **Ties on scaled cost.** A unit is a tie when the two scaled costs differ by at most 1e-12
   relative (absolute floor 1e-12). Alternative: the engine's rule on raw cost, whose absolute
   floor calls costs of 1e-13 and 5e-13 a tie. Consequence: on a series with very small scale E1
   can report a lower-cost family where the engine's oracle reports a tie. On the fixture the two
   agree on all 484 units.

9. **Tau grid checks are per series and horizon.** Stricter than the fix list asked (per horizon
   in E2, per dataset and horizon in E1). Reason: the engine computes every tau from the same
   quantile forecasts, so a unit without a tau means a broken table.

10. **Settings enforced.** representatives, val_fraction, omega_col, omega_variant, crps_method,
    tau grid, and models scored. The cache schema version is recorded but not enforced, because a
    held-out cache may arrive in another layout. Alternative: enforce it too.

11. **Selector code hash scope.** Adapter class plus the selector's defining module. Alternative:
    hash the whole package or record a commit id. Not chosen: the tree is not a git checkout here,
    and a whole-package hash would refuse an evaluation after an unrelated docstring edit.

12. **Ledger.** JSON lines, appended before selectors run so that a crashed run is still recorded,
    written only when held-out names are present. Alternative: also log non-held-out evaluations.

13. **Best-fixed family.** The engine's: one family per (horizon, tau) over all evaluated units,
    not re-chosen within a dataset. When results are grouped by dataset the "0 is the best fixed
    family" reading of the fraction does not hold; the docstring says so.

14. **Stacked stages in E1's default.** `mean_delta_intervals` resamples datasets, series and
    origin blocks together. Alternative: clusters only (`block_length=None`), which already
    carries all dependence within a series. In the simulation stacking series and blocks made the
    interval 1.44 times as wide and it covered 0.99; series alone covered 0.94. Stacking is kept
    as the default because it errs on the wide side, but it is not calibrated either. A mentor
    should choose which stages the paper's intervals use.

15. **Earlier choices kept:** units with identical Omega are averaged before counting sign
    reversals; symmetry differences are upper minus lower; series are resampled within dataset;
    worst-decile regret is the mean of the worst ceil(10%) of series, with the 90th percentile
    next to it; fit/evaluation overlap is by name.

## Open

- **Intervals under-cover.** No bias correction, no studentised or calibrated interval, no
  circular or stationary bootstrap (moving blocks draw the first and last origins less often), no
  data-driven block length, no multiple-comparison handling. Owner: whoever takes the statistics
  for the paper, with a mentor's decision on item 3.
- **Cross-dataset shocks and single-dataset transfer.** Not addressed. For M5 and PJM only
  within-dataset variation can be resampled.
- **Held-out tables cannot be built.** `pipeline.build_tables` calls the held-out guard
  unconditionally, so the M5/PJM evaluation tables, with the `settings` record `evaluate` now
  requires, cannot come from it. Engine owner. The held-out path was exercised only with a fixture
  dataset renamed `m5_invented_for_test`.
- **Engine changes wanted, not made:** (a) the point above; (b) `argmin_family` and
  `evaluate.best_fixed` judge ties on raw cost with an absolute floor (design choice 8); (c)
  `evaluate.evaluate` chooses the best-fixed family on the whole test table rather than on the
  units passed (E2 works around it by restricting the table first); (d) `guards.is_held_out` is a
  plain substring rule: it does not normalise Unicode (E2 normalises before calling it) and it
  matches unrelated names that contain "m5" or "pjm".
- **Renamed fit data is not detected.** The overlap check compares names. A test documents it.
- **The hash is unkeyed.** See What was built. A signed or externally recorded hash would be
  needed for a real control.
- **Real D_Omega selector:** Abdullah. Only the placeholder has been run.
- **Real forecast cache:** nothing here has run on real forecasts (cache owner).
- **Static Omega threshold:** still the fixture placeholder 0.45. The published value has to be
  read from the source paper and recorded. It was not fetched in this session.
- **Selector compute cost** in the proposal's sense (model evaluations avoided, GPU-seconds) is
  not measured. It needs the producer's runtime logs.
- **E1 boundary fit:** blocked on K3 (Abdullah).
- **Dataset-level intervals with few datasets** are coarse (four datasets on the fixture). Not
  assessed beyond refusing fewer than two.
