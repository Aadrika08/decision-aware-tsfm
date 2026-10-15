# Scale for comparing costs across series: seasonal-naive MAE on the history before the first origin

Status: code change, tested on synthetic fixtures and checked against Dayson's real source files.
No real forecast has been read. Not a result.
Scope: #37 (parent #36). Decision: #29 item 7a,
https://github.com/Aadrika08/decision-aware-tsfm/issues/29#issuecomment-6185355042, which follows
Abdullah's point 2 on #62, https://github.com/Aadrika08/decision-aware-tsfm/pull/62#issuecomment-6184302719
("MASE-style in-sample seasonal-naive MAE ... has to be computed from data before the forecast origin").

## What changed

Every "scaled" quantity now divides by one number per series:

    scale = mean |y_t - y_{t-24}| over t in [history_start + 24, history_end), both values finite

where `history_end` is the series' earliest origin in the workload (the first forecast target, so
no target of any window is inside it) and `history_start` is the workload's `usable_start` (0 in the
pilot). The same scale is used for every role, horizon and tau of a series. It replaces mean |y|
over the unit's targets, which came from after the origin, changed between validation and test, and
was 0 at horizon 1 for all 10 solar series and 4 jena_weather series of the pilot.

If the scale is 0, or the history has no finite pair, the series is excluded for every horizon with
the reason `zero scale: ... (zero_seasonal_naive_mae)` or `(no_finite_seasonal_pair)`. Nothing is
divided by 0 and there is no fallback to another scale. The old scale is not kept as an option:
keeping it would need a second path through the engine, the representatives and the bootstrap. The
old numbers can be reproduced from main at `8468258`.

Where the scale is used (all through `analysis/decision/scale.py`, new):

- engine: `pipeline.build_tables(..., scales=TABLE)`; `scales` is required. Every series in the cache
  needs a row, and its `history_end` may not be after the series' first origin in the cache.
  `costs.unit_costs` reads the scale from the per-origin rows instead of computing it.
- representatives: `best_of_family` and `fixed_named` take the same `scales=`; the payload records
  `selection.scale` and `input.scales_sha256`. `representative_robustness` passes it on.
- downstream: E1, E2 and K3 use the engine's `scale` column, so they follow. The bootstrap
  (`bootstrap.family_cost_table`, `_unit_means`) now carries the series scale and keeps it fixed on
  every resample instead of recomputing mean |y| on the resampled origins. E3's `delta_scaled` uses
  the same statistic on each series' context (positions before `n_context`); E3's winners use raw
  costs and are unchanged.
- leftover fixed: `representative_robustness` now defaults to `min_fit_datasets=3` (decision 6, #61).

## Design questions settled

1. **Where the history comes from.** The cache bundle does not carry it: `handoff.json`, the
   partitions and `omega.json` hold forecasts, provenance hashes and Omega only. The workload does:
   every row of Dayson's `workload.json` names `source_path` (relative to the workload's folder) and
   `source_sha256`, and the producer's jobs carry the same `source_sha256`. `run_pilot` already
   requires the workload, so the only new input is optional: `--sources DIR` when the source files are
   not next to the workload. `scale.read_workload_sources` refuses a held-out name (M5, PJM) before
   opening any file, a missing file, a hash mismatch, a path outside the folder, and rows of one series
   that disagree about their source. `run_pilot` and `run_handoff_check` then also check that the
   bundle's source hash equals the workload's for every series, and that every `y_true` in the cache
   equals the source value at its position. Any failure stops the run before anything is written.
   `run_handoff_check` and `run_representatives` take `--workload` and `--sources` the same way.
2. **Which history.** Per series, positions `usable_start` to the earliest workload origin minus 1,
   so one scale per series for every role and horizon. In the pilot this history also covers the
   positions between GIFT-Eval's training end and the first origin (528 to 2,880 hours); they come
   before every target and are part of the models' context for the first origin.
3. **Old scale.** Removed (see above).
4. **Synthetic fixtures.** `fixture.series_values` returns each fixture series' full draw and
   `fixture.make_fixture_scales` its scale table (history before origin 1001; about 1.13 per series,
   where mean |y| was about 10). `make_fixture` itself is unchanged: its cache is byte-identical.
   `fixture.stated_scales` gives hand-built caches with no history (the flat fixture) a stated scale.
   `cache/handoff_fixture.py` is Aadrika's file and loads the fixture from the pinned #44 commit, so it
   is not changed; its bundle needs a workload with sources before the engine can read it. The
   engine's own fake bundles (`tests/unit/decision/_sources.py`) now write `sources/*.npy` and a
   workload naming them.

## Dayson's real sources

`read_workload_sources` on `Dayson_pilot_handoff_2026-10-13/workload.json` (run with `python -I`;
55 series, 55 source hashes verified, every scale usable, M5 and PJM not opened):

| dataset | series | min scale | max scale | history end |
|---|---|---|---|---|
| electricity | 10 | 6.0008 | 205.0364 | 34104 |
| ett1 | 7 | 0.8909 | 10.5154 | 16460 |
| ett2 | 7 | 2.0177 | 15.3034 | 16460 |
| jena_weather | 21 | 0.0244 | 135.3374 | 7872 |
| solar | 10 | 9.6923 | 31.2052 | 7848 |

The 14 series whose horizon-1 targets are all 0 now all have a positive scale: solar items 6, 12, 14,
68, 77, 94, 95, 97, 124, 125 (9.69 to 31.21) and jena_weather dims 14, 15, 17, 18 (0.0244, 45.8,
109.5, 135.3), so horizon 1 is scored on all 5 datasets. The 10 electricity histories hold 70,920 missing
values between them; only finite pairs count. One thing to watch: jena_weather dim14 has a scale
of 0.0244, far below the others, so its scaled costs will be large and can weigh heavily in means
over series.

## Effect on the synthetic outputs

`run_fixture` (seed 7), old = main `8468258`, new = this branch, same on Python 3.11 and 3.10:

| file | change |
|---|---|
| `fixture_cache.parquet`, `fixture_omega.csv`, `planted_answers.csv` | identical |
| `fixture_scales.csv` | new: the scale table used |
| `selection_results.parquet` | 3,388 rows, same keys and columns. Changed columns only: `scale` (all rows; old 9.92 to 10.08, new 1.08 to 1.18), `scaled_regret` (687 rows, each about 8.4 to 9.2 times the old value), `best_fixed_family` (308 rows) and `best_fixed_cost` (252 rows); the best fixed family is chosen on summed scaled cost, so it depends on the scale. `status`, `chosen_family`, `chosen_model`, `realized_cost`, `oracle_family`, `oracle_cost`, `regret` and Omega are identical |
| `summary_by_dataset.csv` | `mean_scaled_regret` and `frac_oracle_value` change, for the same reasons |

The fake pilot bundle: the series with every horizon-1 target 0 (`fx_tie/s1`) is now kept and scored
(it was excluded for zero scale), so the engine scores 15 x 2 x 11 units instead of 29 x 11, and
`engine/series_scales.csv` is a new output file.

## Tests

New `tests/unit/decision/test_decision_scale.py` (36 tests, 1 skipped without a real workload): hand-computed statistic and finite
pairs; changing every value from the first origin on (and scrambling the cache's targets) leaves the
scale unchanged; one scale per series in every table; history reaching into the targets refused;
all-zero horizon-1 targets with a varying history kept and scored; a flat or too-short history
excluded with its reason (evaluation and fit-only datasets, and counted by the representatives); no
scale table, a missing series and malformed tables refused; missing source, changed source, missing
source fields, disagreeing rows, paths outside the folder, held-out names before any file is opened,
`usable_start`, `--sources` through the API and the command; bundle hash and values against the
sources; representatives and engine use the same scale; the bootstrap keeps it fixed; E3 context
scale; robustness floor 3; an optional check on a real workload (`SCALE_REAL_WORKLOAD`).

Existing tests changed, and why:
- every call of `build_tables`, `best_of_family`, `fixed_named`, `representative_robustness` and
  `unit_costs` now passes a scale table (the fixture's, or a stated one for hand-built caches). No
  expected number changed for that reason.
- `test_decision_representatives`: hand caches get a stated scale equal to their constant |y| (10),
  so the hand-computed scores stay; the command test writes a workload and sources, with origins
  moved to 48 so there is a history.
- `test_decision_pilot_run`: `fx_tie/s1` is now kept (counts 29 x 11 to 30 x 11, nothing excluded,
  role coverage 2 scored and 1 excluded to 3 scored and 0 excluded, `n_units_dropped_zero_scale` 1 to
  0); the resample-skip test removes two series at horizon 1 instead of one; new checks on the scale
  file and manifest.
- `test_decision_undefined.ZeroScale`: zero targets alone no longer exclude a series; the exclusion
  tests now use a series that is 0 in its history too, and the reason names the status.
- `test_decision_step_coverage`: the oracle row's `scale` is the stated 1.0 (mean |y| was 51.25).
- `test_decision_producer`, `test_decision_null_omega_handoff`: the check command's bundles now come
  with a workload and sources (`write_bundle(..., source_hashes=...)`).
- `test_bootstrap`, `test_bootstrap_near_ties`: hand-built cost tables name the column `scale`
  instead of `abs_y_sum` (same constant values).

Counts (`-m unittest discover -s tests/unit/<suite> -t .`, team pins), both Pythons identical:

| suite | main `8468258` | this branch, 3.11 | this branch, 3.10 |
|---|---|---|---|
| decision | 279 (2 skipped) | 317 (3 skipped) | 317 (3 skipped) |
| experiments | 170 | 170 | 170 |
| e3 | 107 | 107 | 107 |
| k2 | 184 | 184 | 184 |
| `tests.unit.test_package_layout` | | 3 | 3 |
| integration | | 102 (17 skipped) | 102 (17 skipped) |

All OK. The third decision skip is the optional real-workload check.

Planted bugs: 22 bugs planted one at a time in throwaway copies (lag 1 instead of
24; a pair kept with a non-finite earlier value; history running into the targets; history ending at
the last origin; no hash check; history after the first origin accepted; a series with no scale row
silently kept; zero scale not dropped; silent scale 1 when no table is given; representatives on
mean |y|; engine unit scale back to mean |y|; bootstrap summing the scale; cache values not checked
against the sources; bundle hash not compared; `--sources` dropped by the command; `usable_start`
ignored; scale and status disagreement accepted; E3 scale on a shorter context; engine scale
different from the representatives'; robustness floor back to 2; `series_scales.csv` not written;
exclusion reason without its status). First round: 18 of 22 caught. The four misses (cache values,
`--sources` on the command line, `usable_start`, E3 context) got tests; second round: 22 of 22
caught, and an unmodified copy passes every test.

Determinism: two `run_pilot` runs on the fake bundle are byte-identical, and the Python 3.10 run
writes the same files as the 3.11 run apart from `environment.json` (`PILOT_OTHER_PYTHON`).
`run_fixture` gives byte-identical files on 3.11 and 3.10.

## Open

- The note files of 2026-09-24, 2026-10-12 (representatives) and 2026-10-15 (pilot run) describe the
  old scale. They are records of that time and are not edited; this note supersedes them on the scale.
- Aadrika's `handoff_fixture` bundle has no workload or sources, so it cannot go through the engine
  until it carries them (or a workload is written for it).
- The K3 adapter branch (`feat/37-k3-selector-adapter`) passes the engine's J/scale to K3, so it picks
  this up when merged with main; not run here.
