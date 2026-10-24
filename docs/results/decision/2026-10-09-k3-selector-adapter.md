# Decision engine: the K3 Omega-plus-tau selector in the D_Omega slot (adapter)

Status: plumbing tested on hand-built tables and the synthetic fixture only. No real cache has been
read. Not a result. Three things are not settled and this change settles none of them:

- It depends on unmerged work: PR #54 (`analysis/k3/selector.py`), PR #32 (`analysis/k3/boundary.py`,
  which the selector imports) and PR #33 (the scikit-learn pin). It was built on the engine branch
  (`cab61d4`) with #54 and #32 merged in locally, uncommitted.
- It needs scikit-learn, which the pinned decision environment and the decision CI job do not have.
  There, 44 of the 53 new tests are skipped.
- How a deferred unit is priced is undecided. Here a deferral carries no cost at all.

Scope: one new module `analysis/decision/k3_adapter.py`, one new test file
`tests/unit/decision/test_decision_k3_adapter.py`, and small edits to `pipeline.py` (one new
argument of `run`), `run_fixture.py` (an opt-in flag) and `evaluate.py` (one note text in
`summarize`). Nothing under `analysis/k3/` is changed. The placeholder stays the default.
Part of #37; the selector itself is #26.

## Commands

Team pinned environment (Python 3.11 and 3.10, NumPy 1.26.4, pandas 2.2.3, PyArrow 16.1.0, SciPy
1.11.4, no scikit-learn):

- `PYTHONPATH=src python -m unittest discover -s tests/unit/decision -t .`
- `PYTHONPATH=src python -m decision_aware_tsfm.analysis.decision.run_fixture --out DIR`

Throwaway environment, the same four pins plus scikit-learn 1.5.2 (the pin in PR #33), Python 3.11.15
and 3.10.20, the repository not installed (`PYTHONPATH=src`):

- `PYTHONPATH=src python -m unittest discover -s tests/unit/decision -t .`
- `PYTHONPATH=src python -m decision_aware_tsfm.analysis.decision.run_fixture --out DIR --d-omega k3_selector`
  (optional: `--k3-margin M`, `--k3-min-distinct N`, `--k3-cost raw|scaled`)

## What changed

- New `k3_adapter.K3SelectorAdapter`. `fit(fit_units, margin=0.0, min_distinct=3, cost="raw")` applies
  the held-out guards to dataset and config names, turns the fit pool into the arrays `fit_selector`
  takes (undefined Omega: NaN becomes None) and keeps the frozen spec it returns. `select(inp)` calls
  `apply_selector` once per unit. `describe()` returns the settings and a copy of every frozen spec,
  JSON-serialisable. It also exposes `fitted_series` and `fitted_datasets`, which the fold wrapper
  reads. No rule is fitted or applied in the adapter itself.
- A choice of classical or tsfm becomes a `selected` row. His `defer` becomes `status="defer"` with no
  family and his reason in the note (`reason=omega_out_of_range`, and so on; `reason=rule` on selected
  rows). The fold wrapper puts `fold=<dataset>` in front as before. `k3_adapter.reason_of(note)` reads
  the reason back.
- His module is imported on first use, not when `k3_adapter` is imported, so importing the decision
  package never loads `analysis/k3` or scikit-learn. Without them, `fit` raises
  `K3SelectorUnavailable` with the reason. The held-out guards run before that import.
- `pipeline.run()` gained `d_omega_fit=None`. None is the placeholder (`DOmegaLinear.fit`), as before.
  Passing `K3SelectorAdapter.fit` (or a `functools.partial` of it with settings) puts the K3 selector
  in the slot, under either fit scheme. `pipeline.py` itself does not import the adapter.
- `run_fixture` gained `--d-omega {placeholder,k3_selector}` (default placeholder) and three
  `--k3-*` settings. With `k3_selector` it also writes `d_omega_folds.json` (every fold's frozen specs)
  and prints how many D_Omega units have no choice, by dataset, status and reason.
- `summarize()`: a group with no selected unit now says `undefined: no selected units` in
  `frac_note`. It used to say `undefined: best-fixed equals oracle`, which is not why the fraction is
  undefined there. No number changes. This is the one edit that is not strictly the adapter; it can be
  dropped on its own.
- Two guards the engine did not need before: the adapter refuses a tau that is not exact at two
  decimals (his rules are keyed by tau printed to two decimals, so 0.251 would silently use the rule
  for 0.25) and refuses a fit pool with duplicate units (his unit is one series per tau).

## Verified in this change

- Decision suite, pinned environment, Python 3.11 and 3.10: 170 tests, OK, 45 skipped (44 of the 53
  new tests, with the reason printed, plus the handoff-bundle test). Before this change: 117 tests,
  1 skipped. The 9 new tests that run without scikit-learn cover the default path, the lazy import,
  the held-out guards, the clear failure when the K3 module is missing, `run()` handing the fit
  callable each fold's pool, and the summary note.
- Decision suite, throwaway environment with scikit-learn, Python 3.11 and 3.10: 170 tests, OK,
  1 skipped (the handoff-bundle test).
- The adapter reproduces his selector. On the fixture, for every fold and horizon his two functions
  were called directly on arrays built by hand from the pool: the frozen spec is identical and all
  484 of 484 units have the same choice and reason. On hand-built pools with a planted boundary the
  answers are known without his code: both orientations, a unit on the threshold, the lowest and
  highest fitted Omega, out-of-range Omega, undefined Omega, an unfitted tau, an unfitted horizon and
  the weak-evidence margin.
- Default path unchanged: `run_fixture --out DIR` before and after, Python 3.11 and 3.10, all 5 files
  byte-identical and the printed output identical (`selection_results.parquet` sha256 `63c5dfb8911d`,
  `summary_by_dataset.csv` `cc17666045f9`). Also identical when run in the environment that has
  scikit-learn.
- Deferred rows in the common schema: `validate_selection_frame` accepts them; they carry no family,
  model, cost or regret; `oracle_cost` and `oracle_family` are still filled; the other six comparators
  are row-for-row identical to a placeholder run. `summarize` counts them in `n_units` and not in
  `n_selected`, and its means are over selected units only. `run_fixture` completes and reports when
  none, some or all D_Omega units are deferred.
- Fold wrapper: with the adapter inside, the frame handed to each fold never contains that fold's
  dataset; a fold whose `fitted_datasets` contains its own dataset is refused; a held-out name in the
  pool is refused.
- Team tests, all four environments: `tests.unit.test_package_layout` 3 tests OK;
  `python -m unittest discover -s tests/integration -t .` 66 tests OK, 5 skipped.
- His own tests: `tests/unit/test_k3_selector.py` and `test_k3_boundary.py` are pytest-style
  functions that import pytest. With `python -m unittest` they fail to import (no pytest); with pytest
  present, unittest collects 0 tests. Run with pytest 8.3.4 (his pin) in a separate throwaway
  environment: 41 passed (25 selector, 16 boundary), Python 3.11 and 3.10. His PR text says 77; the
  other 36 were not found in these two files.
- 36 deliberately planted bugs, one at a time, in copies of the tree (family flipped, undefined Omega
  as 0.0 or as NaN, a deferral reported as a choice or with the wrong status, either held-out guard
  removed, `fitted_datasets` or `fitted_series` not reported, wrong tau key, margin or `min_distinct`
  or cost setting ignored, horizons pooled, another horizon's rule borrowed, eager import, `run()`
  ignoring the option under either scheme, a deferral priced as the oracle, reason dropped or wrong,
  defaults changed, guards removed, live spec handed out, missing K3 module swallowed, fixture command
  dropping its settings or its spec file): all 36 are caught with scikit-learn present. In the pinned
  environment, where 44 tests skip, 8 of the 36 are caught.

Fixture numbers (synthetic data with planted answers; they show the plumbing, not the method). Mean
scaled regret is over selected units only.

| setting | folds fitted | units without a choice | K3 mean scaled regret | placeholder, same units | placeholder, all units |
|---|---|---|---|---|---|
| defaults (raw cost, margin 0, min_distinct 3) | 4 of 4, 22 of 22 rules each, 4 datasets per rule | 0 of 484 | 0.000353 | 0.000390 | 0.000390 |
| `cost="scaled"` | same | 0 of 484 | 0.000353 | 0.000390 | 0.000390 |
| `margin=0.05` | same | 77 of 484 (15.9%), all weak_evidence | 0.000383 | 0.000431 | 0.000390 |
| `margin=0.1` | same | 92 of 484 (19.0%), all weak_evidence | 0.000397 | 0.000411 | 0.000390 |
| `margin=0.2` | same | 258 of 484 (53.3%), all weak_evidence | 0.000178 | 0.000178 | 0.000390 |
| `min_distinct=5` | 4 of 4 built, 0 of 22 rules each | 484 of 484, all insufficient_data | none | none | 0.000390 |
| `fit_scheme="fit_only_datasets"` (1 dataset) | 1 fit, 0 of 22 rules | 484 of 484, all insufficient_data | none | none | 0.000390 |
| no fit-only dataset (3 other datasets per fold) | 4 of 4, 22 of 22 rules each | 242 of 484 (50.0%), all omega_out_of_range | 0.000726 | 0.001099 | 0.000571 |
| 3 datasets in total (2 others per fold) | 3 of 3 built, 0 of 22 rules each | 352 of 352, all insufficient_data | none | none | 0.000784 |
| 2 datasets in total (1 other per fold) | 0 of 2 (the wrapper needs 2) | 220 of 220, status insufficient_data from the wrapper | none | none | none |

With the defaults, per dataset (K3 against the placeholder): fx_classical_wins 0.000002 and 0.000002,
fx_tau_flip 0.001212 and 0.001419, fx_tie 0 and 0, fx_tsfm_wins 0.000079 and 0.000010. Raw and scaled
cost agree on the fixture because every fixture series has the same level; a hand-built test shows
they can differ.

## Design choices needing sign-off

1. Horizons. His rule is per tau and his unit is the series; the engine's unit also has a horizon.
   Chosen: one frozen spec per horizon, so each spec sees every series once per tau, which is his
   function called on exactly the population he defines. It also matches the engine, where best-fixed
   is per (horizon, tau). Alternatives: pool the horizons into one rule per tau (a series then enters
   once per horizon, which is the repeated-observation problem ADR 0006 section 2 argues against, and
   raw costs of different horizons are summed together); or average the horizons to one cost per series
   first. Cost of the choice: a horizon the fold never saw has no rule, and the unit is deferred with
   `unfitted_horizon`, a reason that is the adapter's and not in his list; and if datasets do not share
   horizons, fewer datasets stand behind each rule.
2. Cost he is fitted on. Chosen default: raw `J_classical` and `J_tsfm`, exactly what the pool holds,
   nothing transformed, `scale` not used. Alternative, implemented and tested: `cost="scaled"`
   (J / scale), the placeholder's objective. Against the default: on real data raw sums are dominated
   by the datasets with large values, while the engine's regret, best-fixed family and oracle-value
   fraction are all on scaled cost, and ADR 0006 section 1 says the best-fixed definition must be the
   same on both sides. My reading is that scaled is the better choice for the real run; raw is the
   default only because it changes nothing about what his code is given. Needs the selector owner.
3. Fold pool. Unchanged from the development-roles note, item 1: the fit-only datasets (all origins)
   plus the other evaluation datasets' test units. One consequence for his spec: the wrapper drops
   units with an undefined Omega before the adapter sees them, so `training_undefined_omega` reads 0
   and `training_series` counts defined rows only.
4. `margin`. Chosen: 0.0, his default (no weak-evidence band). It is in absolute Omega units. It must
   be fixed before the real run and not tuned on evaluation datasets. Because means are over selected
   units only, a wider margin lowers the reported regret by construction (0.000353 to 0.000178 on the
   fixture at 0.2, with 53% of units deferred), so margins cannot be compared until deferrals are priced.
5. `min_distinct=3` datasets, his default, and the wrapper's `min_fit_datasets=2`. On the fixture
   every fold has 4 other datasets, so all 4 folds fit all 22 rules. The two thresholds disagree at
   exactly 2 other datasets: the fold is built, every rule says insufficient_data and every unit comes
   out `defer`; with 1 other dataset the wrapper answers `insufficient_data` itself. Same cause, two
   statuses. Alternative: pass `min_fit_datasets=3` whenever his selector is in the slot. Not done,
   because it would change a setting silently.
6. Status of a deferral. Chosen: every `defer` from his selector is `status="defer"`, whatever the
   reason, and the reason is in the note. Alternative: report his `insufficient_data` and
   `omega_undefined` reasons with the engine's `insufficient_data` status, which is what the
   placeholder and the static rule report for an undefined Omega. The note lets either be rebuilt.
7. How the selector is chosen. Chosen: `run(..., d_omega_fit=callable)`, default the placeholder.
   Alternative: a name such as `d_omega="k3_selector"`. Reason: `pipeline.py` stays free of any import
   from `analysis/k3`, and the settings are bound where the caller can see them. The default was not
   flipped: his code and its dependency are not merged, and deferral pricing is not agreed.

## Open

- Deferral pricing. His `evaluate_selector` prices a deferral as the oracle; it is not used here. The
  other proposal on the table is to price it as `pinball_validation`. Until one is agreed, the D_Omega
  row of any summary covers fewer units than the other six and is not comparable with them.
- CI. The decision CI job installs no scikit-learn, so it would skip 44 of the 53 new tests. Either
  the job gains the pin from PR #33 or the selector stops needing it: `selector.py` takes four names
  from `boundary.py` (`MIN_DISTINCT_OMEGA_THRESHOLD`, `ORIENTATIONS`, `_validate`, `distinct_count`)
  and none of them uses scikit-learn; only `boundary.py`'s own top-level imports do.
- For the selector owner, found while reading and reproduced: (a) `fit_selector` raises
  `TypeError: cannot unpack non-iterable NoneType object` when at least `min_distinct` datasets share
  one single Omega value at a tau (three datasets, Omega 0.5 each: no midpoint to try). The adapter
  lets it surface. (b) Rules are keyed by `f"{tau:.2f}"`: taus 0.251 and 0.25 become one rule. The
  adapter refuses such taus; the 11 pre-registered taus are all exact at two decimals. (c) After PR
  #32, `analysis/k3/__init__.py` imports `boundary`, so `analysis.k3.matched_omega_pair` can no longer
  be imported without scikit-learn. (d) His two test files need pytest; the team runs unittest.
- E2 freezes selectors through its own registry (`register_selector_kind` in `experiments/e2.py`, on
  the branch `feat/42-e1-e2-bootstrap`). Freezing the K3 selector there needs a kind that rebuilds an
  adapter from `describe()`; not built here. E2's selector-code fingerprint covers `selectors.py`,
  which this change does not touch.
- `run_handoff_check` was not given the option. The handoff bundle is not available in this
  workspace and nothing was run on it.
- The adapter's own series-overlap check compares `series_id` alone, as the placeholder's does.
