# Pilot analysis run: one command from the cache handoff to every pilot result

Status: new command, tested on synthetic bundles only. No real forecast has been read. Not a result.
Scope: `analysis/decision/run_pilot.py` (new) and `tests/unit/decision/test_decision_pilot_run.py`
(new). No existing file was changed. Part of #37 (parent #36).

## Why

Every piece the pilot needs is on main (`d25f732`): the bundle reader (#45), the role table (#60),
leave-one-dataset-out with a floor of 3 (#61), the representatives (#62), the engine, E1, E2 and
the block bootstrap (#52). Nothing joined them. When Aadrika's real cache lands, the analysis should
be one line, run with the settings the team has already agreed, and it should refuse to run when the
inputs do not match what was planned.

## The command

Team pinned environment (Python 3.11.15 or 3.10.20, NumPy 1.26.4, pandas 2.2.3, PyArrow 16.1.0,
SciPy 1.11.4), from the repository root:

```bash
PYTHONPATH=src python -m decision_aware_tsfm.analysis.decision.run_pilot BUNDLE --out DIR
```

`BUNDLE` is the producer handoff bundle (a folder with `handoff.json`, the parquet partitions and
`omega.json`). The workload is read from `BUNDLE/workload.json`; if the bundle does not carry it, add
`--workload HANDOFF/workload.json` (Dayson's file). `DIR` must not exist.

Options: `--variant last_4096` (Omega sensitivity), `--static-thresholds 0.45 0.4 0.5` (the first is
the headline), `--fixed-classical NAME` (fixed-named representative sensitivity), `--role-table PATH`
(a role table to check, if not in the bundle), `--min-fit-datasets` (default 3), `--n-boot` (default
200, 0 skips the intervals), `--boot-seed` (default 0), `--figures`.

The run writes into `DIR.partial` and renames it to `DIR` only when every step has passed. If any
check fails, the partial folder is removed and nothing is left behind.

## What each step does

1. **Load and check the bundle.** `handoff.read_handoff`, the same reader `run_handoff_check` uses:
   every partition hash, the exact producer-v1 columns and types, job provenance (where each forecast
   came from), and the Omega join for the chosen variant. Then the held-out guard (M5, PJM) on the
   dataset, config and series names of the bundle, `omega.json` and the workload.
2. **Roles.** The role table is built from the workload with the #60 code (`build_role_table`,
   record 0007: earliest third validation, then gap, then evaluation) and written to `role_table/`
   with the #60 writer. Three checks, each stops the run on any difference:
   - a role table carried by the bundle (`BUNDLE/role_table.csv`, `BUNDLE/role_table/`, or
     `--role-table`) must equal the one built from the workload;
   - every (series, horizon) in the cache must be in the workload and must hold all of the
     workload's origins for it, and the roles the engine's rule gives the cache's origins must equal
     the table;
   - after the engine has run, the roles and decision origins it actually used must equal the table.

   A (series, horizon) of the workload that is not in the cache at all is not an error; it is
   counted (`workload_series_horizons_absent_from_cache`) so a cache that covers some series
   completely can still be analysed.
3. **Representatives.** `best_of_family` with loss `scaled`, validation origins only, frozen to
   `representatives/representatives.json` (#62). With one TSFM it is `only_member`.
   `--fixed-classical NAME` also freezes the fixed-named choice (that classical model against the
   same TSFM) and runs the engine with it as a sensitivity.
4. **Engine.** `build_tables` once, then `pipeline.run` once per static threshold: the seven
   comparators, leave-one-dataset-out with at least 3 fitting datasets. The test costs are
   recomputed directly from the raw quantile columns (`run_handoff_check.direct_test_costs`) and
   must agree to a relative difference of 1e-10. D_Omega is the placeholder on main. The seam for
   the K3 selector adapter is `run_pilot(..., d_omega_builder=...)`: a function that takes the
   tables and returns the selector. With no builder the code calls `pipeline.run` exactly as on main.
5. **E1 and E2**, on the headline run. E1: `e1.run_e1` (decision map, summaries, symmetry table)
   and `e1.mean_delta_intervals`, one horizon at a time. E2: for every leave-one-dataset-out fold,
   `e2.fit` on the engine's fit pool without that dataset, `freeze`, `load_frozen`, `evaluate` on
   that dataset; the frozen rule's choices must equal the engine's `d_omega` choices unit by unit,
   or the run stops. Then E2's regret figures on the pooled headline run (`regret_summary`,
   `gap_to_validation`, `worst_decile_regret`) and moving-block bootstrap intervals for the mean
   scaled regret of every rule except the oracle (`bootstrap.fixed_choice_costs` and the default
   fixed-oracle statistic, datasets and series as clusters, automatic block length). A horizon that
   cannot be resampled is skipped with the bootstrap's own reason, written to a `_skipped.csv`
   file; the other horizons still run.
6. **Paper tables** in `tables/`:
   - `headline_by_horizon_tau.csv`: the seven rules per horizon and tau, for every threshold
     (headline rows first, flagged `headline`): units, decided units, mean scaled regret, regret
     over oracle cost, fraction of oracle value recovered (with the reason when it is undefined),
     number of deferrals and the mean scaled regret with each deferral charged what
     `pinball_validation` paid on that unit;
   - the same pooled over taus (`headline_by_selector_horizon.csv`) and over everything
     (`headline_by_selector.csv`);
   - `representative_scores.csv` (validation score of every candidate),
     `representative_sensitivity.csv` (with `--fixed-classical`);
   - `role_coverage.csv`: per dataset and horizon, the #60 role counts, series in the cache, series
     scored, and series excluded for zero scale or for no validation origin.

   Figures: the repository pins no plotting library (`analysis/k2/figure.py` says the same) and
   neither environment has matplotlib. `--figures` draws two PNG files only if matplotlib can be
   imported; otherwise it is skipped with a message. The figure data are already tables:
   `tables/headline_by_horizon_tau.csv` (regret by tau) and `e1/decision_map.csv` (Omega against
   Delta). No dependency was added.
7. **`summary.md`** (what ran, counts, the pooled headline table, where every file is),
   **`manifest.json`** (SHA-256 of `handoff.json`, `omega.json`, the workload, the role table, a
   fingerprint of every forecast row, the git commit, a SHA-256 over the decision and experiments
   source files, all settings, and the SHA-256 of every output file) and **`environment.json`**
   (interpreter and package versions). No file holds an absolute path or a time stamp.

## Decisions the defaults encode

All from the #29 decisions comment
(https://github.com/Aadrika08/decision-aware-tsfm/issues/29#issuecomment-6185355042):

- Item 4: the role split of record 0007, one third validation. The summary says whether the
  workload's SHA-256 and the role table's SHA-256 equal the two named on #29 (`1474953e...` and
  `f5d50243...`). A different file is reported as `NO`, not refused.
- Item 5: Omega `current` by default, `last_4096` with `--variant`.
- Item 6: at least 3 fitting datasets per leave-one-dataset-out fold.
- Item 7: best of family, once, on validation origins, pinball at tau 0.50 scaled by mean |y|;
  a fixed named model as a sensitivity.
- Item 8: a deferral is charged what the pinball-validation rule paid on that unit (the
  `mean_scaled_regret_deferrals_priced` column). Oracle pricing is not reported here.
- Item 10: static threshold 0.45, with 0.4 and 0.5 as sensitivity runs.
- Record 0007, consequences: bootstrap clusters are datasets and whole series.

## What was measured (2026-10-15)

Tests, `PYTHONPATH=src python -m unittest ...` from the worktree root:

| suite | Python 3.11 | Python 3.10 |
|---|---|---|
| decision | 279 OK (2 skipped) | 279 OK (2 skipped) |
| experiments | 170 OK | 170 OK |
| e3 | 107 OK | 107 OK |
| k2 | 184 OK | 184 OK |
| package layout | 3 OK | 3 OK |
| integration | 102 OK (17 skipped) | 102 OK (17 skipped) |

Main has 243 decision tests (1 skipped); the 36 new ones are in `test_decision_pilot_run.py`. One
of them compares the files written by the other Python and skips unless `PILOT_OTHER_PYTHON` names
that interpreter. With it set, the file ran 36 OK and 0 skipped on 3.11 (comparing against 3.10)
and on 3.10 (comparing against 3.11).

The test bundle: the roster fixture (AutoARIMA, AutoETS, SeasonalNaive, Moirai-small), five
datasets with three series each, horizons 1 and 24, 15 origins 12 apart (so horizon 24 has a gap
origin), a `workload.json` and the role table exported from it. One series has every horizon-1
target equal to 0, so the engine excludes it there for zero scale. One series gets an Omega label
of 0.47, so the 0.45 and 0.5 runs choose differently on a series where the families differ. A
second bundle gives one series a null Omega with reason `flat_history`.

What the tests check: the full run writes every file and the manifest hashes match; the role
table equals the #60 export; representatives are best of family on validation, `only_member` for
the TSFM, the zero-scale unit counted and left out; the seven rules use the frozen
representatives; the zero-scale pair is excluded with its reason; every fold has 3 or more fitting
datasets and E2's frozen choices equal the engine's; one headline cell recomputed by hand; the
column check covers every unit; interval settings (resamples, seed, clusters, block length); a
horizon that cannot be resampled is skipped alone; the threshold runs (headline first, only
`static_omega` changes, the 0.45 and 0.5 runs differ); `last_4096` and a different threshold order
from the command line; the D_Omega seam; refusals (a bundle role table with one changed row, a role
table from another workload, a cache holding part of a pair's origins, a cache pair missing from
the workload, no validation origin anywhere, no workload, existing output, held-out names in the
bundle's dataset or series names and in the workload); nothing is left behind after a late
failure; null Omega deferrals are priced and counted; representatives do not move when the losing
classical model is made a perfect forecaster on every evaluation and gap origin, and do move when
the same is done on validation origins.

Determinism: two runs on the same bundle give byte-identical files (all 47 files of the
end-to-end run, parquet included). Python 3.11 and 3.10 give byte-identical files except
`environment.json`, which records the interpreter.

Default fixture output: `run_fixture` on this branch and on a clean export of `origin/main`, 5 of
5 files byte-identical, on 3.11 and on 3.10, and 3.11 equal to 3.10.

Planted bugs: 25 bugs planted one at a time in throwaway copies of the tree (outside the
worktree); the pilot test file catches 25 of 25 on 3.11 and 25 of 25 on 3.10, and the clean copy
passes. The first round caught 22 of 25; the three misses (E1 and E2 reading the last threshold
run instead of the headline, E1 intervals clustered by series only, the bootstrap seed ignored) led
to the Omega label of 0.47 in the test bundle and to checks on the interval settings. The bugs:
headline threshold not first; floor 2; raw loss for representatives; bundle role table check not
strict; partial-pair check off; unplanned-pair check off; series names not guarded in the bundle;
series names not guarded in the workload; fit-only scheme instead of leave-one-dataset-out; Omega
variant ignored; E2 fit seeing its own dataset; oracle given intervals; deferral priced at the
oracle; every threshold flagged headline; exclusion reasons swapped in the coverage table; engine
role check ignoring role values; every run using the headline threshold; `summary.md` left out of
the manifest; partial output kept on failure; E1 and E2 on the last threshold run; fixed
sensitivity ignoring the named model; E2 compared with `static_omega`; E1 intervals clustered by
series only; validation fraction one quarter; bootstrap seed ignored.

Real producer format: `consumer-handoff-45.zip` (ZIP sha256 `5f1f0df0...`, Aadrika's exported
bundle) with a workload written from its own origins, read with `python -I`. Every hash, format and
role check passes, then the run stops with "none of the 34 (series, horizon) pairs in the cache has
a validation origin": that bundle has 2 origins per series, and one third of 2 rounds down to 0.

Scale: a synthetic bundle of pilot size (34 series, 17 origins 48 apart, horizons 1, 24 and 168,
four models, seeds 42 and 43: 13,872 jobs, 892,432 rows) ran end to end in 26 seconds with
about 2.1 GB peak memory on this laptop, 200 resamples, no interval skipped.

## What the real cache may hit that the fakes do not

- **Zero-scale exclusions at horizon 1.** From Dayson's sources and workload: at horizon 1 every
  evaluation target is 0 for all 10 solar series and for 4 of 21 jena_weather series (the origins
  are 48 hours apart, so they fall at the same hour of day). The engine excludes those pairs with
  its zero-scale reason, so horizon 1 is scored on 4 datasets and 41 series. The table and summary
  report them.
- **The first 24-job slice cannot be analysed.** It holds one origin of one series. The run stops
  with "the cache holds only part of the workload's origins". It needs every origin of each series
  it holds, and at least 4 datasets for every fold to be fitted.
- **Undefined Omega reasons.** `handoff.py` accepts only `empty_history`, `below_floor`,
  `flat_history` and `nonfinite`. Aadrika's readiness check also knows `near_flat`; a bundle with
  it stops at load. The real pilot Omega has 0 nulls, so this only matters once Bitbrains is added.
- **Workload location.** Dayson's handoff has `workload.json`; Aadrika's bundle format does not
  list it. If her bundle does not carry it, pass `--workload`.
- **Model and checkpoint names** are taken from the cache as they are. The run does not check them
  against `models.json`.
- **Figures** are drawn only where matplotlib is installed. The drawing code was checked against a
  stand-in plotting module, not against matplotlib itself (no environment here has it, and the
  sandbox blocked installing it in a throwaway one).
