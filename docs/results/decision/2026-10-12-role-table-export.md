# Decision engine: role table export and the table-versus-engine check

Status: tooling, tested on synthetic inputs only. No real workload, no real forecast and no
held-out data was read. Not a result. Decision record 0007 is still proposed (PR #57), so a table
written with this tool is not an approved role table until that record is signed.
Scope: two new modules, `analysis/decision/role_table.py` and `analysis/decision/run_role_table.py`,
and one new test file, `tests/unit/decision/test_decision_role_table.py`. No existing file was
changed. Part of #45.

## Why

The real pilot is waiting for "the approved per-origin/horizon role table". Record 0007 ends with:
"The engine derives roles from the origins in the cache and does not read a window table, so a
check that the two agree still has to be written." A table built by hand can drift from the engine.
This change writes the table from the engine's own functions, so the two agree by construction,
and adds the check for any table, whoever built it.

## What the producer expects

No role table format exists in the producer's code on main (`39d49e7`).

- `cache/readiness.py` takes `--workload`, `--models`, `--omega` and `--output-directory`
  (lines 162 to 165). It reads no role or window table. The words "role table", "window table"
  and `selector_roles` appear nowhere under `src/`, `scripts/` or `tests/`.
- `selector_roles_present: false` is a field of the preflight report on PR #58
  (`docs/results/cache/2026-10-12-pilot-execution-preflight/report.json`). No code on main writes
  or reads it.
- Dayson's `scripts/export_pilot_handoff.py` (PR #48, `23af713`) writes `workload.json`,
  `omega.json` and `summary.md`, and its summary says "Selector role assignments are not included
  yet (pending Veer/Abdullah/Pratham)."

The sentences that constrain the format:

- Aadrika on #45: "Put the approved assignments in a separate window table keyed by `(dataset,
  config, series_id, origin, horizon)`; models and seeds share that assignment. Dataset folds are
  a separate dataset-level assignment."
- Dayson on #45: "put role assignments in a separate window table keyed by (dataset, config,
  series_id, origin, horizon) once Veer, Abdullah and Pratham agree them".
- Aadrika's plan, `docs/superpowers/plans/2026-10-10-aadrika-integration-and-pilot-plan.md` line
  89: "Preserve existing forecast `training_end`/`validation_end` semantics; these bounds are not
  interchangeable with selector validation/evaluation roles. Do not invent a new table schema
  independently of its consumers."
- Aadrika on PR #58: "the role table must match the approved protocol, not merely the forecast
  bounds."

So the key is fixed by the team, and the file name, file type and role words are not. The output
follows the last paragraph of record 0007: one row per (dataset, config, series_id, origin,
horizon) with a role of validation, gap or evaluation, plus a separate list of excluded
(series, horizon) pairs. The workload's `training_end` and `validation_end` are not read.

The input is the producer's `workload.json` as `cache/planner.py::plan` accepts it (`split`
equal to `development`, `rows` with one mapping per series, origin, horizon and seed; reference:
`handoff()` in `tests/integration/test_pilot_readiness.py`). Only `dataset`, `config`,
`series_id`, `origin` and `horizon` are read. The document that `handoff()` builds was accepted
by the producer's `plan` (96 jobs from 24 workload rows) and read by the new reader in the same
run: 12 distinct windows, 12 table rows.

## What was built

`role_table.py`:

- `build_role_table(origins, val_fraction=1/3)` calls `costs.assign_roles` for the roles,
  `pipeline._drop_without_validation` for the exclusion and `pipeline._decision_origins` for E.
  The rule is not written a second time. The reason string is `pipeline.NO_VALIDATION_REASON`,
  imported, and a test pins it to the wording in record 0007. It returns the table, E for every
  kept (series, horizon), and the excluded pairs.
- The engine calls the evaluation role `test`. The table says `evaluation`. The mapping is
  `ENGINE_ROLE` in the module and `roles` in the sidecar.
- `check_role_table(table, origins, val_fraction=1/3, excluded=None, decision_origins=None,
  strict=False)` returns the number of matched rows and one list per kind of difference:
  `missing_rows`, `extra_rows`, `role_disagreements`, `duplicate_rows`, `unknown_roles`,
  `decision_origin_disagreements`, `exclusion_disagreements`. `strict=True` raises
  `RoleTableMismatch` on any of them. Held-out names raise `HeldOutLeakError` in either mode.
- Readers: `read_workload_origins` (the producer's `workload.json`), `cache_origins` (through
  `cache_io.load_cache`) and `handoff_origins` (through `handoff.read_handoff`). No new parquet
  reader.
- Held-out names in `dataset` or `config` are refused with `guards.assert_no_held_out`, in the
  origins and in a table being checked.

Files written by the export, all utf-8 with `\n` line ends:

| File | Content |
|---|---|
| `role_table.csv` | dataset, config, series_id, origin, horizon, role |
| `decision_origins.csv` | dataset, config, series_id, horizon, decision_origin (E), kept pairs only |
| `excluded_series_horizons.csv` | dataset, config, series_id, horizon, reason |
| `summary.csv` | per dataset, config and horizon: series, validation, gap, evaluation, excluded pairs |
| `role_table.json` | validation fraction, role words, record 0007 and the engine functions, the input's name and SHA-256, the code commit, the SHA-256 of each CSV |

## Commands

```bash
PYTHONPATH=src python -m decision_aware_tsfm.analysis.decision.run_role_table export \
    --workload HANDOFF/workload.json --out artifacts/runs/role-table
PYTHONPATH=src python -m decision_aware_tsfm.analysis.decision.run_role_table check \
    artifacts/runs/role-table --workload HANDOFF/workload.json
```

The origins come from one of `--workload FILE`, `--cache PATH --schema NAME` or
`--handoff DIR --variant V`. Both commands take `--val-fraction` (default 1/3) and
`--fit-only-datasets` (those datasets get no roles). `export` refuses an existing `--out`.
`check` takes the export directory or a single role table CSV, and exits with status 1 and one
line naming the difference when the table and the engine disagree.

## Pilot shape (a synthetic reconstruction, NOT the real workload)

The real `workload.json` was not available. The test builds a workload with the shape reported in
the producer's preflight of 2026-10-12: electricity/H 10 series, ett1/H 7, ett2/H 7,
jena_weather/H 21, solar/H 10; 17 origins per series for the first three and 16 for the other
two; origins 48 steps apart; horizons 1, 24 and 168; seeds 42 and 43. Series names and origin
positions are invented. It has 5,424 workload rows, 55 series, 904 series-origins and 2,712
windows.

Output of the export on it (identical on Python 3.11 and 3.10):

```
     dataset config  horizon  series  validation  gap  evaluation  excluded_series_horizons
 electricity      H        1      10          50    0         120                         0
 electricity      H       24      10          50    0         120                         0
 electricity      H      168      10          20   30         120                         0
        ett1      H        1       7          35    0          84                         0
        ett1      H       24       7          35    0          84                         0
        ett1      H      168       7          14   21          84                         0
        ett2      H        1       7          35    0          84                         0
        ett2      H       24       7          35    0          84                         0
        ett2      H      168       7          14   21          84                         0
jena_weather      H        1      21         105    0         231                         0
jena_weather      H       24      21         105    0         231                         0
jena_weather      H      168      21          42   63         231                         0
       solar      H        1      10          50    0         110                         0
       solar      H       24      10          50    0         110                         0
       solar      H      168      10          20   30         110                         0
```

Totals: 2,712 rows, 660 validation, 165 gap, 1,887 evaluation, 0 excluded pairs, 165 decision
origins. All 15 rows equal `proposed_role_coverage.counts` in the preflight report, where the
`test` column is this table's `evaluation`. The 15 expected rows are literals in the test.

The check on the untouched export: 2,712 matched rows, exit status 0. After changing the first
row (electricity_H__s00, origin 30000, horizon 1) from validation to evaluation: 2,711 matched
rows, exit status 1, and

```
role table check FAILED: role table disagrees with the engine (val_fraction 0.333333): 1 role_disagreements, e.g. {'dataset': 'electricity', 'config': 'H', 'series_id': 'electricity_H__s00', 'origin': 30000, 'horizon': 1, 'role_engine': 'validation', 'role_table': 'evaluation'}
```

## Released consumer fixture

`consumer-handoff-45.zip` (SHA-256
`5f1f0df0de17282ffdbae4c0cd4fc8e356511dfbf774bf6e6da54ed54a16b94c`, verified after download), read
with `--handoff`, `--fit-only-datasets fx_dfit`, for both `--variant current` and
`--variant last_4096`: 22 series at horizon 24 with 2 origins each give 44 rows, 0 validation,
0 gap, 44 evaluation, and all 22 (series, horizon) pairs excluded, because floor(2 / 3) = 0. The
role table is byte-identical for the two variants. The check passes on it (44 matched rows).

## Tests

`tests/unit/decision/test_decision_role_table.py`, 33 tests:

- both worked examples of record 0007 section 1 (77 / 23 / 200 at H = 24, the other horizons of
  the same series, and the 9-origin example at H = 24, 48 and 168 with its exclusion);
- the pilot shape above;
- one table per kind of disagreement, each wrong in one way: a different role, a missing origin,
  an extra origin, a horizon missing for a series, roles of two horizons swapped, duplicate rows,
  an unknown role word (including the engine's own `test`), another validation fraction, a wrong
  E, exclusions that differ, held-out names, malformed columns;
- round trip through the two commands, with an empty config and an ID with leading zeros; a
  second run and a shuffled workload give byte-identical CSV files (the sidecar then differs only
  in the input's SHA-256);
- the table built from a workload equals the roles inside the engine's own tables when
  `build_tables` reads a cache with the same origins: the full fixture (11,220 kept rows), and a
  small producer-format cache with a gap origin and two excluded series, also read as a handoff
  bundle.

Planted bugs, one at a time in a throwaway copy of the two modules: 33 planted, 31 caught. The
two that were not caught each remove one of two sorts that do the same job, so the output does
not change; removing both together is one of the 31 caught. The list covers the role mapping
swapped, the gap rule dropped, E taken as the last validation origin, excluded pairs not listed,
the check ignoring horizon, extra rows, missing rows, duplicates, unknown role words, E or
exclusions, the validation fraction ignored when building or checking, and the held-out guard
removed.

One defect was found by these tests and fixed before this note: an excluded list with no rows,
which is what the pilot shape produces, came back from CSV without a number type and the check
refused it.

Suites, run from the worktree with `PYTHONPATH=src`:

| Suite | Python 3.11 | Python 3.10 | main before |
|---|---|---|---|
| decision | 185 OK (1 skipped) | 185 OK (1 skipped) | 152 OK (1 skipped) |
| experiments | 169 OK | 169 OK | 169 OK |
| e3 | 107 OK | 107 OK | 107 OK |
| k2 | 182 OK | 182 OK | 182 OK |
| `tests.unit.test_package_layout` | 3 OK | 3 OK | 3 OK |
| integration | 102 OK (17 skipped) | 102 OK (17 skipped) | 102 OK (17 skipped) |

The default fixture output (`run_fixture --out DIR`) is byte-identical to main on both Pythons:
`fixture_cache.parquet bcecefbdb163`, `fixture_omega.csv 6e8679fa77c0`,
`planted_answers.csv bd92190ccef0`, `selection_results.parquet 63c5dfb8911d`,
`summary_by_dataset.csv cc17666045f9`.

## Choices the team should confirm

1. File type and names. CSV files plus a JSON sidecar. Nothing in the producer reads them yet.
2. Rows of an excluded (series, horizon) stay in the table with the role the rule gave them
   (gap or evaluation), and the pair is named in the separate excluded list, as record 0007 words
   it. A reader that uses the table alone would treat those evaluation rows as scored.
3. The table says `evaluation` and refuses `test` as an unknown word.
4. `decision_origins.csv` is an addition to the record's format. It lists E, the origin where an
   Omega row is required for every kept pair.
5. The check compares content. It does not verify the hashes recorded in the sidecar.

## Limits

- The engine can also exclude a (series, horizon) whose realised targets are all zero. That needs
  the values, so a table built from a workload cannot contain it.
- From a cache, origins are taken from every row. The engine uses the two representative models.
  The two agree when every model covers every origin, which the pilot readiness check requires.
- `code_commit` in the sidecar is the checkout's HEAD. Run the export from a committed checkout,
  or the commit will not describe the code that ran.
- `min_fit_datasets`, the fold assignment and the Omega rows are outside this change.

## Still needed for the real table

The real `workload.json` from Dayson's `HANDOFF.zip` (SHA-256
`70e814bf323178df84b771f920f6a6114c6796cb5b143ba28ccce4833eb56ba0` in the preflight report), then
one run of the export command above. Record 0007 has to be signed before the result is the
approved table. The producer has no step that reads a role table, so how readiness uses it is for
Aadrika to decide.
