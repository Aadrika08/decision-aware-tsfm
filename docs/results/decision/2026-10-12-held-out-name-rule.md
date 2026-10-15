# Held-out name rule: "m5" inside a word is no longer a match

Status: change to the held-out name check, tested on names only. No held-out data was opened. No
forecast or cache was read. Not a result. This rule protects M5 and PJM, so it is put to the team
for review in its own pull request.
Scope: `analysis/decision/guards.py` (the rule), `analysis/k2/heldout.py` (the reasons K2 prints
and the rule text K2 records), one new test file, and the K2 and E2 tests that pinned the old
rule. Branch `fix/37-held-out-name-rule`, from main at `39d49e7`. Part of #37.

## The defect

`is_held_out` refused any name that contains the letters "m5" or "pjm" anywhere. The manifest on
`feat/1-omega-validation` (PR #48, `23af713`) names a series
`<dataset>_<config>__<item>_dim<channel>`, so channel 5 of a multivariate series ends in `_dim5`,
which contains "m5". On main, `is_held_out("ett1_H__item_0_dim5")` is True.

Measured on the two manifest-v1 files at `23af713` (names only, no value read): the old rule
flags 8 of the 8,761 active series ids, one in each of `bizitobs_l2c_5T`, `bizitobs_l2c_H`,
`ett1_15T`, `ett1_H`, `ett2_15T`, `ett2_H`, `jena_weather_10T` and `jena_weather_H`, and 5 of the
139,127 excluded ids. All 13 end in `_dim5`. It flags 0 values in `subset`, `config`, `dataset`
and `item_id`. None of the 13 is M5 or PJM.

There is one `guards.py` on main, used by the engine, by E1/E2, by E3 (through the engine) and by
K2. Who was affected:

- The engine and E3 check dataset and config names only, so they were not.
- E2 checks series ids as well, in `fit`, `freeze`, `load_frozen` and `evaluate`. With real
  series ids, `fit` would refuse the 8 ids, and `evaluate` would refuse them unless it was given
  `allow_held_out=True` and a ledger, which is meant for the final transfer test only.
- The K2 manifest-v1 reader refused the whole manifest unless `--exclude-held-out-matches` was
  given, which left the 8 series out
  (see [the provisional rerun](../k2/2026-10-10-provisional-rerun-manifest-v1.md)).

## What changed

- "m5" now counts only when no letter (a to z, after lower-casing) comes straight before it.
  "pjm" still counts anywhere. Case is ignored and a backslash is read as a slash, as before.
  The tokens are unchanged.
- `guards.py` has one new function, `held_out_tokens_in(name)`, which returns the tokens the rule
  finds in a name. `is_held_out` is true when that list is not empty. K2's `strict_reasons` now
  takes its reasons from it, so a name such as `pjm_H__item_0_dim5` is reported as matching
  'pjm' and not also 'm5'.
- The rule text that K2 writes to `run_manifest.json` (`held_out.rule`) now describes this rule.
  It used to say "substring match on the tokens".
- Nothing else changed: no dataset, split, horizon, tau grid, metric or gate, and no other
  behaviour of the engine, E1, E2, E3 or K2.

Old rule against new rule (`held out` means the name is refused):

| name | old rule | new rule | changed |
|---|---|---|---|
| `m5` | held out | held out |  |
| `M5` | held out | held out |  |
| `m5_store_1` | held out | held out |  |
| `store_m5` | held out | held out |  |
| `store_1_m5` | held out | held out |  |
| `x_M5_y` | held out | held out |  |
| `M5_Retail` | held out | held out |  |
| `Salesforce/M5` | held out | held out |  |
| `data\M5\sales` | held out | held out |  |
| `1m5` | held out | held out |  |
| `_m5` | held out | held out |  |
| `m50` | held out | held out |  |
| `M5a` | held out | held out |  |
| `dim5_m5` | held out | held out |  |
| `M5_Retail__item_0_dim5` | held out | held out |  |
| `pjm` | held out | held out |  |
| `PJM_hourly` | held out | held out |  |
| `PJM_Electricity` | held out | held out |  |
| `mypjmdata` | held out | held out |  |
| `pjm_H__item_0_dim5` | held out | held out |  |
| `ett1_H__item_dim5` | held out | passes | yes |
| `ett1_H__item_0_dim5` | held out | passes | yes |
| `jena_weather_H__item_dim5` | held out | passes | yes |
| `jena_weather_10T__item_0_dim5` | held out | passes | yes |
| `bizitobs_l2c_H__L2C_dim5` | held out | passes | yes |
| `ett1_H__item_dim15` | passes | passes |  |
| `ett1_H__item_dim50` | held out | passes | yes |
| `ett1_H__item_dim4` | passes | passes |  |
| `ram5gb_cluster` | held out | passes | yes |
| `item5` | held out | passes | yes |
| `am5` | held out | passes | yes |
| `retailm5` | held out | passes | yes |
| `m4_daily` | passes | passes |  |
| `solar_10T__item_5` | passes | passes |  |
| `M-5_Retail` | passes | passes |  |
| `ém5` | held out | held out |  |
| (empty name) | passes | passes |  |

37 names, 10 change. On the manifest-v1 files the new rule flags 0 names in every text field of
the active and the excluded rows. The two names in the manifest's own `held_out_strict` list,
`M5_Retail` and `PJM_Electricity`, are refused by both rules.

## What this rule does not catch (needs sign-off)

A held-out name written with a letter straight before "m5", such as `retailm5`, is not caught by
this rule. It was caught by the old one. The team's names for the held-out sets (`M5_Retail`,
`PJM_Electricity`, and `m5` or `pjm` as a folder or dataset name) are all still caught. A test
pins this limit by name so that it cannot change unnoticed.

Two things still stand behind it in K2: the manifest's own `held_out_strict` list, compared on
letters and digits only, which this change does not touch; and the punctuation-insensitive layer,
which has no letter condition, so K2 still prints `retailm5` and every `_dim5` id as a possible
held-out name to check by hand. E2 and the engine have neither.

Only a to z count as a letter. A name with another letter before "m5", such as `ém5`, is still
refused. The alternative that keeps the old rule and exempts only the `_dim<n>` ending was not
chosen, because the same false positive exists for any name such as `item5`.

Two other name checks exist in the repository and were not changed: Stage0Prep's
`STRICT_HELD_OUT` (the same two tokens, matched against source folder paths) and the cache
planner's own check in `cache/planner.py`.

## Every place the rule is applied, and what changes there

| Call site | Names checked | Change |
|---|---|---|
| `decision/pipeline.py` (`build_tables`) | forecast `dataset`, `config` | None for the 29 real configs and the 1 dataset name in manifest-v1. A dataset named like `item5` is no longer refused. |
| `decision/selectors.py` (D_Omega fit) | fit `dataset`, `config` | Same. |
| `experiments/e2.py` `fit` | fit `dataset`, `config`, `series_id` | `_dim5` series ids are accepted. There is still no switch to fit on a held-out name. |
| `experiments/e2.py` `freeze`, `load_frozen` | frozen fit datasets, configs, series | A specification listing `_dim5` fit series is frozen and loads. |
| `experiments/e2.py` `evaluate` | evaluation `dataset`, `config`, `series_id` | `_dim5` series ids are evaluated without the flag and are not written to the held-out ledger. A held-out name still needs `allow_held_out=True` and a ledger. |
| `k2/heldout.py` `strict_reasons` | every text field of a manifest row; results `dataset` | `_dim5` and `ram5gb`-style names are no longer strict matches. Reasons name only the tokens the rule found. |
| `k2/heldout.py` `fuzzy_reasons` | the same names | Code unchanged. Names the strict layer now lets through are reported here and kept. |
| `k2/loading.py` `prepare_results` | results `dataset` | A results dataset with "m5" inside a word is kept and printed, not dropped. None of the 97 dataset names in the saved results contains "m5" or "pjm", so the default run's tables are unchanged. |
| `k2/loading.py` manifest readers | every text field | manifest-v1 is read without `--exclude-held-out-matches`. |
| `k2/loading.py` `aggregate_omega`, `k2/families.py`, `k2/sensitivity.py` | base dataset names | None for the real names. |

## `--exclude-held-out-matches`

The option is kept. It still leaves out manifest-v1 rows whose names match the rule, and a
manifest with a real match is still refused without it. On manifest-v1 at `23af713` it now has
nothing to leave out, so it changes nothing.

PROVISIONAL, same status as the 2026-10-10 rerun: manifest-v1 is from the unmerged PR #48 and is
not frozen. Reanalysis of published results; no forecasting model was run. Measured on 2026-10-12
with the command of the 2026-10-10 note (10,000 resamples, seed 20261004, `--allow-missing-omega`),
Python 3.11. Run outputs are in a scratch folder and are not committed. Input hashes as in that
note: `series_manifest.json` `f20898d3...`, `omega_values.json` `fd4ca660...`.

| Run | Rule | Flag | Outcome | Series kept | With a value, full history | With a value, last 4,096 |
|---|---|---|---|---|---|---|
| main `39d49e7` | old | no | refused, 8 ids named | | | |
| main `39d49e7` | old | yes | runs, 8 rows left out | 8,753 | 8,687 | 8,562 |
| this branch | new | no | runs, 0 rows left out | 8,761 | 8,695 | 8,570 |
| this branch | new | yes | runs, 0 rows left out | 8,761 | 8,695 | 8,570 |

- This branch, with the flag against without it: all 14 output tables are byte-identical, for
  both Omega columns.
- This branch without the flag against main with the flag (8 more series): all 60 dataset-level
  correlations and all 12 headline correlations are equal, with their intervals, for both Omega
  columns. 10 of the 14 tables are byte-identical. The configuration-level table differs in 36 of
  its 72 rows, by at most 0.061 in the correlation on full history and 0.032 on the last 4,096
  steps; the per-dataset Omega means in the joined tables differ.
- K2 prints 41 names as possible held-out names to check by hand, 8 more than before: the 8
  `_dim5` ids. The other 33 are names such as `solar_10T__item_5`.

## Commands

Team pinned environment (Python 3.11.15 and 3.10.20, NumPy 1.26.4, pandas 2.2.3, PyArrow 16.1.0,
SciPy 1.11.4), from the repository root:

- `PYTHONPATH=src python -m unittest discover -s tests/unit/<suite> -t .` for `decision`,
  `experiments`, `e3`, `k2`
- `PYTHONPATH=src python -m unittest tests.unit.test_package_layout`
- `PYTHONPATH=src python -m unittest discover -s tests/integration -t .`
- `PYTHONPATH=src python -m decision_aware_tsfm.analysis.decision.run_fixture --out <directory>`
- `PYTHONPATH=src python -m decision_aware_tsfm.analysis.experiments.e1 --out <directory>`
- `PYTHONPATH=src python -m decision_aware_tsfm.analysis.experiments.e2 --out <directory>`
- `PYTHONPATH=src python -m decision_aware_tsfm.analysis.experiments.e3.run --out <directory>`
- `PYTHONPATH=src python -m decision_aware_tsfm.analysis.k2.cli --outdir <directory>`
- the manifest-v1 command in the 2026-10-10 note, with and without `--exclude-held-out-matches`

## Tests

- New file `tests/unit/decision/test_decision_held_out_names.py`, 8 tests: 38 held-out names are
  refused (24 with "m5", 14 with "pjm"); 32 other names are not, 19 of them with "m5" straight
  after a letter (the 8 real ids among them); a held-out name is still refused when a `_dim5`
  suffix or another `dim5` comes with it; the `retailm5` limit; the tokens found, in token
  order; the tokens are unchanged; non-string and empty names; the error names the held-out
  name and not the `_dim5` id.
- K2, 6 tests rewritten because they pinned the old rule, and 2 added. Three in
  `test_k2_heldout.py` used `ram5gb_cluster` as a name the strict layer must refuse; they now
  use `store_1_m5`, and assert that `ram5gb_cluster` is kept and reported. Three in
  `test_k2_manifest_v1.py` used a `_dim5` id to exercise the refusal and the opt-in; they now
  use made-up ids with `m5` as a token. The 2 new tests assert that a `_dim5` id is read and
  used without the opt-in, is printed as a possible name, and gives the same tables with it.
- E2, 1 test added: series ids ending in `_dim5` fit, freeze, load and evaluate without the
  flag, with the regrets of the plain run, and the suffix does not hide a held-out name in
  front of it.

## Verified in this change

- Every suite, same counts on Python 3.11 and 3.10: decision 160 pass (152 on main, plus 8), 1
  skipped without `DECISION_HANDOFF_BUNDLE`; experiments 170 (169 plus 1); e3 107; k2 184 (182
  plus 2); layout 3; integration 102 (17 skipped).
- The new tests fail on the old rule. With `guards.py` exactly as on main the new file cannot be
  imported (it uses `held_out_tokens_in`); the 6 tests this file started from, which need no
  new function, fail in 2 of 6 there. With the old substring rule put back and the new function
  kept: 4 of the 8 new tests fail, and so do 5 K2 tests and the new E2 test.
- Synthetic fixture run: all 5 output files byte-identical to main, on both Pythons
  (`fixture_cache.parquet` `bcecefbdb163`, `fixture_omega.csv` `6e8679fa77c0`,
  `planted_answers.csv` `bd92190ccef0`, `selection_results.parquet` `63c5dfb8911d`,
  `summary_by_dataset.csv` `cc17666045f9`).
- Default runs on main and on this branch, Python 3.11. E1: 7 of 7 files byte-identical. E2: 7 of
  9; the other two are `selector_cost.csv` (wall times, marked not reproducible) and
  `manifest.json` (the output folder path only). E3: 28 of 29; the other is `run_manifest.json`
  (output folder path and runtime only). K2 on the default single-file manifest: all 14 tables
  byte-identical; `run_manifest.json` differs in the hash of `heldout.py`, the rule text and
  the folder paths, and in nothing else.
- Planted bugs, 21, each put alone into a throwaway copy of `guards.py` with the new test file
  run on both Pythons: 21 of 21 caught, and the copy with no bug passes. Among them: the old
  substring rule; "pjm" given the letter condition; a digit before "m5" treated as a letter; an
  underscore treated as a letter; case not folded; only the first "m5" in a name checked; a
  non-string name raising; an empty name held out; "m5" only as a whole word; only the `dim`
  ending exempt; either token not matched; nothing ever held out.

## Open

- No record under `docs/decisions/` exists for this rule. The rule does not change which
  datasets are held out; the team may still want one.
- A run-together name such as `retailm5` passes the rule (see above).
- The 2026-10-10 provisional K2 note says K2 stops on manifest-v1 by default. That holds for the
  code it was run on, not for this branch. That note is not edited.
- Not run on the real pilot cache, and not run on GitHub Actions.
