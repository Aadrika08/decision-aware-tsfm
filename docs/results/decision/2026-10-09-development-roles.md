# Decision engine: agreed development roles (leave-one-dataset-out, empty validation third)

Status: behaviour change tested on the synthetic fixture and the rebuilt handoff bundle only. No real
cache has been read. Not a result.
Scope: `analysis/decision/pipeline.py`, `selectors.py`, `run_handoff_check.py`, `run_fixture.py`
and their tests; two test helpers in `tests/unit/experiments/`. Part of #38.
Agreed on 2026-10-07 (selector owner): earliest third of origins for validation, the rest for
evaluation, leave-one-dataset-out for fitting, and a series only qualifies if its validation third
is not empty; otherwise it is excluded with a reason.

## Commands

Team pinned environment (Python 3.11, NumPy 1.26.4, pandas 2.2.3, PyArrow 16.1.0, SciPy 1.11.4):

- `PYTHONPATH=src python -m unittest discover -s tests/unit/decision -t .`
- `PYTHONPATH=src python -m unittest discover -s tests/unit/experiments -t .`

## What changed

- `run()` now fits D_Omega leave-one-dataset-out by default (`fit_scheme="leave_one_dataset_out"`).
  For each evaluation dataset d, the rule is fitted on every other dataset and applied only to d.
  The fitting pool is the fit-only datasets (all origins) plus the other evaluation datasets' test
  units with their decision-origin Omega (`pipeline.fit_pool`). The note column of each D_Omega row
  names its fold (`fold=<dataset>`).
- A fold whose pool has fewer than `min_fit_datasets` (default 2) other datasets is not fitted; its
  units report `insufficient_data` with the reason. The other six comparators are unaffected.
- `LeaveOneDatasetOut` refuses a fold that lists its own dataset among its fit datasets, independent
  of the selector inside, so the real D_Omega (#26) can be dropped in without its own overlap check.
- The earlier scheme is still available as `fit_scheme="fit_only_datasets"`. E2 freezes one rule
  fitted that way, so E2 itself is unchanged.
- `build_tables()` now excludes a (series, horizon) with no validation origin and lists it in
  `excluded_series_horizons` with the reason and `used_for="evaluation"`. The check is per horizon
  because the gap rule depends on horizon. Default: on whenever `val_fraction > 0`; with
  `val_fraction = 0` the caller has chosen no validation, so nothing is excluded for this reason (E3
  scores this way). `exclude_without_validation=False` restores the earlier behaviour (validation
  comparators report `insufficient_data`). The setting is recorded in `tables["settings"]`.
- `run_handoff_check` keeps such series (interface check: the bundle has two origins per series) and
  says so in its report; it gained `--fit-scheme` and reports every fold.

## Verified in this change

- Decision suite: 105 tests (104 pass, 1 skipped without `DECISION_HANDOFF_BUNDLE`; with it set, all
  105 pass). Experiments suite: 165 tests pass. E3 suite: 107 pass. Same on Python 3.10.
- Every D_Omega choice on the fixture equals an independent refit per dataset (fit on the pool without
  d, apply the threshold rule to d's units): 484 of 484 units.
- On the fixture the four folds give omega0 0.41, 0.41, 0.45 and 0.41 with beta 0 (the fit-only scheme
  gives 0.45). No fixture series lacks a validation origin, so nothing is excluded there.
- E1 fixture run: all 7 output files byte-identical to the 2026-10-06 run. E2 fixture run: every table
  identical apart from the selector-code fingerprint (selectors.py changed, so the frozen file name and
  hash change: 80713c79bbf7 before, 3efbf0bcf105 now) and the measured wall times.
- Handoff bundle check, both Omega variants: 408 partitions verified, 242 units compared, 0 unmatched,
  largest relative difference 3.3e-16; all four folds fitted.
- 12 deliberately planted bugs in the new code (fold sees its own dataset, pool without evaluation
  datasets, pool built from validation costs, off-by-one on the minimum, row order lost, exclusion
  skipped or inverted, wrong default, guard removed, handoff check excluding, fold note dropped) are
  each caught by the tests.

## Design choices needing sign-off

1. The pool for fold d uses the other datasets' test units. Alternative: only their validation
   origins. Reason: more data per fold, and d itself is never seen, which mirrors E2 (fit on whole
   development collections, evaluate on a new one). Cost: if two datasets cover the same calendar
   period, a shared shock can reach the fit for the other. The validation-only pool removes that at
   the price of a noisier fit.
2. A fold needs at least 2 other datasets. Alternative: 1. Reason: a boundary fitted on one dataset
   says little about transfer.
3. Exclusion is per (series, horizon). Alternative: per series (drop all horizons if any horizon has
   no validation origin). Reason: the gap rule makes availability depend on horizon, and the engine's
   unit already includes horizon.

## Open

- The real D_Omega (#26) replaces the placeholder inside the same fold wrapper once its code is in.
- Old E2 frozen files carry the previous selector-code fingerprint and will be refused by the current
  code by design; refreeze after this change.
