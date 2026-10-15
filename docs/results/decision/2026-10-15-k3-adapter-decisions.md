# Decision engine: pilot decisions 6, 8 and 9 wired into the K3 adapter

Status: tested on hand-built tables and the synthetic fixture only. No real cache has been read. Not
a result. Follows `2026-10-09-k3-selector-adapter.md`, whose line "a deferral carries no cost at
all" is superseded here.

Source of the decisions: the pilot decision record posted on #29 on 2026-10-15, items 6, 8 and 9.
Part of #37. The adapter still depends on PR #32 (`analysis/k3/boundary.py`) and PR #54
(`analysis/k3/selector.py`), which are approved and not yet on main.

Scope: `analysis/decision/evaluate.py`, `analysis/decision/k3_adapter.py`,
`analysis/decision/pipeline.py`, `analysis/decision/run_fixture.py`,
`tests/unit/decision/test_decision_k3_adapter.py`. Nothing under `analysis/k3/` is changed. The
placeholder stays the default and the default run is byte-identical to main.

## What changed

Main merged into the branch first (`9a2adf0`). One conflict, in `pipeline.run()`: main's
`min_fit_datasets=3` (#61) kept, with the adapter's `d_omega_fit` argument.

Decision 8, deferral pricing.
- New `evaluate.charge_deferrals(res)`. Every row of a selection frame, plus `deferred` (the defer
  flag), `deferral_family`, `deferral_cost`, `charged_cost`, `charged_regret`,
  `charged_scaled_regret`, `deferral_pricing` and `charge_note`. A deferred unit is charged the
  `realized_cost` of the engine's `pinball_validation` comparator on the same (dataset, config,
  series_id, horizon, tau), read from that comparator's rows in the same frame. That per-unit price
  is what #54's `evaluate_selector` takes as `deferral_cost`.
- There is no option for oracle pricing. A frame with no `pinball_validation` rows for a deferred
  unit is refused ("never at the oracle"). A deferral where `pinball_validation` itself made no
  choice stays unpriced: `charged_cost` is NaN and `charge_note` says why. Every row carries
  `deferral_pricing = "validation_comparator"`, the name #54 uses, so a row read on its own still
  says how it was priced. A frame where a deferred row already carries a cost fails the schema check.
- New `evaluate.summarize_charged(res, by)`. The decided-only columns are exactly `summarize()`'s.
  Added, over every unit in the group: `n_deferred`, `deferral_rate`, `n_not_charged`,
  `mean_scaled_cost_charged` (the realised cost including deferrals), `mean_scaled_regret_charged`,
  `frac_oracle_value_charged`, `frac_note_charged`, `deferral_pricing`. If any unit in a group is
  not charged, the three charged numbers are NaN and the note gives the count, so a partly priced
  group never looks complete.
- `summarize()` now refuses a frame with deferred rows and points to `summarize_charged()`. Before,
  it averaged selected units only and dropped deferrals without a trace. Main never produces a
  deferred row, so nothing on main changes.
- The schema itself is unchanged: a deferred row in `selection_results.parquet` still has status
  `defer`, no family and no cost. The charged cost lives in the added columns.

Decision 9, scaled cost and one tau at a time.
- `K3SelectorAdapter.fit` defaults to `cost="scaled"`: it hands #54 `J / scale`, the engine's
  scaled cost from `costs.py`. `raw` stays available as a sensitivity option. A scaled fit refuses
  a non-positive scale.
- `summarize_charged` groups by (selector, horizon, tau) by default, the engine's own grain.

Decision 6, two floors named apart.
- `min_fit_datasets` is the engine's floor (leave-one-dataset-out): distinct datasets with usable
  units in a fold's pool. Default 3 everywhere the adapter can reach it.
- `k3_min_distinct` is #54's floor (`min_distinct` in his code; renamed in the adapter from
  `min_distinct`). Default 3. `describe()` records both and what the K3 one counts.

Entry point for a pilot command: `k3_adapter.build_k3_d_omega(tables, *, fit_scheme, min_fit_datasets=3,
k3_min_distinct=3, margin=0.0, cost="scaled")`. It takes the engine's tables and returns the D_Omega
selector, the same object `run(..., d_omega_fit=K3SelectorAdapter.fit)` builds. It refuses held-out
names among the evaluation units, and checks that the K3 module can be imported before fitting
anything. `pipeline.run()` now calls a new `pipeline.fit_d_omega()` for the slot; its output is
unchanged.

`run_fixture --d-omega k3_selector` uses the entry point, stops with a plain error (exit 2, nothing
written) when the K3 module is missing, and also writes `selection_charged.parquet` and
`summary_by_horizon_tau.csv`. Its `summary_by_dataset.csv` gains the charged columns.

## What #54 and #32 actually do here (read from their code)

- #32 now imports scikit-learn inside the functions that need it. The selector calls none of them.
  So with #32 and #54 present, the K3 option runs without scikit-learn; a test runs it with
  scikit-learn blocked. The clear error is for a missing K3 module, not for missing scikit-learn.
- The adapter always passes dataset labels, so #54's `min_distinct` counts distinct datasets with a
  defined Omega at each (horizon, tau), not distinct Omega values. #54 separately needs at least 2
  distinct Omega values to place a threshold. Decision 6, Abdullah's note on #61 and record 0008
  section 3 describe the K3 floor as a count of distinct Omega values. A test pins what the code
  does: 3 datasets that share 2 Omega values pass his floor.

## Commands

From the worktree root, `PYTHONPATH=src`:
- `python -m unittest discover -s tests/unit/<suite> -t .` for decision, experiments, e3, k2, k3
- `python -m unittest tests.unit.test_package_layout`, `python -m unittest discover -s tests/integration -t .`
- `python -m decision_aware_tsfm.analysis.decision.run_fixture --out DIR [--d-omega k3_selector]`

Environments: pinned team environments (Python 3.11.15 and 3.10.20, NumPy 1.26.4, pandas 2.2.3,
PyArrow 16.1.0, SciPy 1.11.4, no scikit-learn), and throwaway ones with the same pins plus
scikit-learn 1.5.2.

## Verified

Test counts, every suite, Python 3.11 and 3.10 gave the same counts:

| Suite | Branch alone, pinned envs | Branch + #32 + #35 + #54, scikit-learn envs | Same merge, pinned envs |
|---|---|---|---|
| decision | 322 OK, 56 skipped | 322 OK, 1 skipped | 322 OK, 1 skipped |
| experiments | 170 OK | 170 OK | 170 OK |
| e3 | 107 OK | 107 OK | 107 OK |
| k2 | 184 OK | 184 OK | 184 OK |
| k3 | (not on the branch) | 102 OK | 102 run, 5 errors |
| test_package_layout | 3 OK | 3 OK | 3 OK |
| integration | 102 OK, 17 skipped | 102 OK, 17 skipped | 102 OK, 17 skipped |

The 56 skipped on the branch alone are 55 adapter tests that need the K3 module plus the
handoff-bundle test. The 5 k3 errors in pinned envs are #32's own boundary tests that call
scikit-learn; they pass where it is installed. The adapter test file has 79 tests (24 run without
the K3 module).

- Default run unchanged: `run_fixture --out DIR` on main (`d25f732`) and on this branch, Python 3.11
  and 3.10: all 5 files byte-identical (`selection_results.parquet` sha256 `63c5dfb8911d`,
  `summary_by_dataset.csv` `cc17666045f9`) and the printed output identical.
- `run_fixture --d-omega k3_selector` on the merge: same 8 files byte-identical on 3.11 and 3.10.
  With the default margin no unit is deferred on the fixture; with `--k3-margin 0.2`, 258 of 484
  D_Omega units are deferred and every one is charged.
- Without the K3 module: `--d-omega k3_selector` exits 2 with the import message and writes nothing.
- The engine's charged numbers equal #54's `evaluate_selector`, one tau at a time, on scaled costs
  with `deferral_cost` from `pinball_validation` (deferrals, deferral rate, realised cost, fraction
  of oracle value). Its oracle-priced default gives a lower cost, flagged `oracle_upper_bound`.
- A dataset with costs and scale 1000 times larger leaves the scaled fit unchanged, on a hand-built
  pool and through the entry point on the fixture; the raw fit flips orientation.
- 25 bugs planted one at a time in copies of the merged tree; the adapter test file caught 21 of
  the first 21. Four more aimed at gaps: 1 caught, 3 missed (a non-positive scale accepted, the
  entry point's early import check removed, the schema check in `charge_deferrals` removed). A test
  was added for each and all 25 are now caught.

## Not settled

- E1, E2 and the bootstrap read the selection frame and use selected rows only (the bootstrap
  refuses a selector with non-selected rows). They do not price deferrals yet. A pilot run with the
  K3 selector should report the D_Omega numbers from `summarize_charged`, not from E1 or E2.
- Units of a leave-one-dataset-out fold that is not fitted (engine floor) are `insufficient_data`,
  not deferrals. They are not charged, and they make that group's charged numbers NaN.
- The K3 floor's wording in decision 6 and record 0008 (distinct Omega values) differs from what the
  code counts (distinct datasets). No result changes on the fixture.
