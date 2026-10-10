# Decision engine: null Omega in a handoff bundle, and step coverage per producer job (re-review of #44)

Status: input-validation change tested on synthetic bundles and the synthetic fixture only. No real
cache has been read. Not a result.
Scope: `analysis/decision/handoff.py`, `cache_io.py`, `pipeline.py`, `run_handoff_check.py` and two
new test files, `tests/unit/decision/test_decision_null_omega_handoff.py` and
`tests/unit/decision/test_decision_step_coverage.py`. Part of #37. Raised by the cache owner in the
re-review of #44 at `cab61d4` on 2026-10-10, as two inline comments.

## The defects

1. A null Omega was read as a missing join. `read_handoff` decided whether a forecast
   (series, origin) had an Omega row by testing the joined value for null. A row that is present
   with `"omega": null` and a reason was therefore reported as absent, and the agreed undefined
   case never reached the engine. The reason (`omega_status`) was also dropped from the frame that
   `read_handoff` returns. Reproduced before the change on a disposable copy of the original
   bundle (current Omega at `fx_classical_wins/s0`, origin 1001, set to null with
   `omega_status=flat_history`, sidecar hash updated in `handoff.json`): `HandoffError: 1 forecast
   (series, origin) pairs have no Omega row for variant 'current'`. Removing that row instead gave
   the same message, word for word.
2. A producer job with a missing step was accepted. `_from_producer_v1` checked that (job_id, step)
   is unique and that a job's rows agree on its identity, but not that a job holds steps
   1..horizon. Target coverage is checked later across seeds, so a second seed hid the gap.
   Reproduced before the change with the reviewer's case (`flat_fixture_producer()`, shared
   `y_true` at step 24 set to 1000, step 24 removed for classical seed 0 at both origins, exported
   with `write_bundle()`): all 8 partition hashes verified, 190 rows read, and `build_tables`/`run`
   accepted it. The tau=0.50 hindsight oracle moved from tsfm, cost 21.0833, scale 51.2500 (complete
   bundle; classical cost 21.6250) to classical, cost 11.5319, scale 31.0638.

## Commands

Team pinned environment (Python 3.11 and 3.10, NumPy 1.26.4, pandas 2.2.3, PyArrow 16.1.0, SciPy 1.11.4):

- `PYTHONPATH=src python -m unittest discover -s tests/unit/decision -t .`
- `DECISION_HANDOFF_BUNDLE=<unpacked consumer-fixture> PYTHONPATH=src python -m unittest discover -s tests/unit/decision -t .`
- `PYTHONPATH=src python -m decision_aware_tsfm.analysis.decision.run_handoff_check <unpacked consumer-fixture> --variant current --classical AutoETS --tsfm Moirai-small --static-threshold 0.45 --fit-only-datasets fx_dfit --out <new directory>`
  (and the same with `--variant last_4096`)
- `PYTHONPATH=src python -m decision_aware_tsfm.analysis.decision.run_fixture --out <directory>`

The original bundle is the release asset `consumer-handoff-45.zip`
(SHA-256 `5f1f0df0de17282ffdbae4c0cd4fc8e356511dfbf774bf6e6da54ed54a16b94c`, verified after download).

## What changed

- `read_handoff` takes join membership from the merge itself. A forecast (series, origin) with no
  sidecar row for the chosen variant is still an error, with the same message as before. A row
  that is present with a null Omega is no longer counted as absent.
- A null Omega must carry a recognised `omega_status`: `empty_history`, `below_floor`,
  `flat_history` or `nonfinite`. A null with no status, or with any other value (`ok`, an empty
  string, an unknown word, a number, a list), is refused: `omega.json rows have a null Omega
  without a recognised omega_status`. A `NaN` literal in the sidecar is treated as null.
- A defined Omega may carry `omega_status` `ok` or no status. A defined Omega with any other
  status, including one of the four undefined reasons, is refused: `omega.json rows have a defined
  Omega but omega_status [...]`.
- Both status checks run on every sidecar row, for all variants and origins, as the existing
  history check does. The source revision and hash check still applies to null rows.
- The Omega frame returned by `read_handoff` keeps `omega_status` (the column is always present;
  it is empty for a sidecar that has none, such as the original bundle). The handoff report gains
  `omega_undefined_rows_for_variant` and `omega_undefined_reasons_for_variant` (reason to count).
- `attach_omega` carries a reason column through when the Omega table has one. The column name is
  `<omega_col>_status`: `omega_status` for a handoff sidecar, `omega_levels_status` for the frozen
  manifest. `build_tables` returns a new table, `tables["undefined_omega"]`: one row per decision
  whose Omega is undefined, with dataset, config, series_id, horizon, decision_origin, reason and
  used_for (evaluation or fit). The reason is missing when the Omega table gave none. `units` and
  `fit_units` are unchanged: the status column does not reach the selectors.
- `run_handoff_check` writes `undefined_omega_decisions` to `report.json` (counts keyed
  `used_for/reason`).
- `_from_producer_v1` (called by `load_cache`) requires every job to hold exactly steps 1..horizon:
  per `job_id`, the row count equals the horizon, the lowest step is 1 and the highest is the
  horizon. With the existing uniqueness of (job_id, step) this is exact. The check runs on every
  job in the cache, before representatives are chosen and before seeds are aggregated. The error
  names up to three jobs: `2 jobs do not hold exactly steps 1..horizon, e.g. [...]`.
- Not changed: `selectors.py`, `evaluate.py`, `schema.py` (the selection-v1 output format), the
  fixture, `run_fixture.py`, and how the engine treats an undefined Omega once it arrives (kept as
  NaN, the two Omega-based comparators report `insufficient_data`, 0.0 is refused).

## Verified in this change

- Finding 1, reviewer's case after the change: `read_handoff` accepts the bundle (408 partitions,
  9,792 rows). The report shows 1 undefined row for the variant, reason `flat_history`. The engine
  sees Omega as NaN for that series: 11 units undefined (1 horizon x 11 taus), and
  `tables["undefined_omega"]` lists the decision (`fx_classical_wins/s0`, horizon 24, decision
  origin 1001, `flat_history`, evaluation). `static_omega` and `d_omega` report `insufficient_data`
  on those 11 units and `selected` on the other 231. `always_classical`, `always_tsfm` and
  `hindsight_oracle` report `selected` on all 242.
- On that bundle `pinball_validation` and `crps_validation` report `insufficient_data` on all 242
  units, in the original bundle before and after the change and in the copy with the null after
  it: the bundle has two origins per series, so there is no validation origin. The new tests
  therefore use a bundle with three origins per series (one validation, two test), where all five
  non-Omega comparators report `selected` for the series with the null and the two Omega-based
  ones report `insufficient_data`.
- Finding 1, other cases after the change, on copies of the original bundle: the row removed is
  still refused (`1 forecast (series, origin) pairs have no Omega row for variant 'current'`);
  null with no status and null with status `invented` are both refused.
- Finding 2, reviewer's case after the change: the complete bundle still gives tsfm, cost 21.0833,
  scale 51.2500. The incomplete bundle is read by `read_handoff` (8 of 8 hashes verified, 190 rows)
  and refused by `build_tables`: `CacheError: 2 jobs do not hold exactly steps 1..horizon`, naming
  the two classical seed 0 jobs with 23 rows each.
- Decision suite: 152 tests on Python 3.11 and 3.10. Without `DECISION_HANDOFF_BUNDLE`: 151 pass,
  1 skipped. With it pointing at the original bundle: 152 pass, no skips. 35 of the 152 are new
  (24 in `test_decision_null_omega_handoff.py`, 11 in `test_decision_step_coverage.py`). On the
  unchanged code 30 of the 35 fail (21 and 9); the other 5 pin behaviour that must not change (an
  absent row is refused, 0.0 is refused, a complete cache passes, a duplicate step is refused, the
  selector tables keep their columns).
- Team tests on this branch, Python 3.11 and 3.10: `tests.unit.test_package_layout` 3 pass;
  `tests/integration` 66 tests, 61 pass, 5 skipped.
- Fixture run (`run_fixture`): all 5 output files byte-identical before and after, on both Pythons.
- Handoff check on the original bundle, both variants, both Pythons: `selection_results.parquet`
  byte-identical before and after. `report.json` differs only by the three added keys
  (`omega_undefined_rows_for_variant: 0`, `omega_undefined_reasons_for_variant: {}`,
  `undefined_omega_decisions: {}`). Still 408 of 408 partitions verified, 9,792 rows, 242 units
  compared, 0 unmatched, maximum relative cost difference 3.308343066616648e-16.
- Downstream, on throwaway copies with this change applied as a patch (Python 3.11): the E1/E2
  branch (`feat/42-e1-e2-bootstrap` at `cc18132`) passes its 169 experiments tests before and
  after, and its decision suite goes from 117 to 152 tests (1 skipped). The K3 adapter branch
  (`feat/37-k3-selector-adapter` at `89d4dd1`) goes from 170 decision tests (45 skipped) to 205
  (45 skipped), all passing; the patch applies cleanly to its own `pipeline.py`.
- 40 deliberately planted bugs, one at a time in throwaway copies, are each caught by the decision
  suite. 28 concern finding 1 (membership read from the value again, the absent-row error
  swallowed or inverted, the reason check removed, inverted, weakened or limited to one variant, a
  reason dropped from the list, `ok` accepted for a null, the contradiction check removed or
  widened, the reason dropped from the returned frame or the tables or the report, null turned
  into 0.0 in the reader or in the engine, the hash check skipped for null rows, a missing reason
  filled in, every decision listed as undefined, fit and evaluation mixed up, the status column
  leaking into the selector tables). 12 concern finding 2 (the check removed, inverted, done only
  when every job is incomplete, reduced to the row count or to the first or last step, counted
  across seeds, made relative to the rows present, off by one, moved after aggregation or after
  the choice of representatives). 37 of the 40 are caught only by the two new test files.

## Design choices needing sign-off

1. Recognised reasons are exactly the producer's four, and a defined value is marked `ok` or not
   marked. Source: `omega.compute_omega_checked` and `scripts/export_pilot_handoff.py` on
   `feat/1-omega-validation` (`23af713`), and `cache/readiness.py` on `fix/55-null-omega-readiness`
   (PR #56, `dea0214`). Both sides list the same four reasons and the same field name. Alternative:
   accept any non-empty reason. Reason for the strict list: an unknown word is a contract change
   that should be seen, not absorbed.
2. A defined Omega with an undefined reason, or with any status other than `ok`, is refused.
   Alternatives: trust the value, or trust the status and treat the value as undefined. Reason:
   the row contradicts itself and the reader cannot know which half is wrong. This is stricter than
   `cache/readiness.py` in PR #56, which does not look at the status of a defined row.
3. The reason is kept in three places and not in the selection-v1 rows: the returned Omega frame,
   the handoff report (rows of the chosen variant), and `tables["undefined_omega"]` (one row per
   deferred decision, also summarised in the check command's `report.json`). Alternative: write it
   into the `note` column of the two Omega-based comparators. Reason: that needs a change in
   `selectors.py`, which changes the selector-code fingerprint that E2 freezes, and the per-decision
   table says the same thing without touching the shared output format.
4. The engine does not require or validate a reason. `attach_omega` carries one through when the
   table has a `<omega_col>_status` column. Alternative: require a reason for every null in
   `attach_omega`. Reason: the engine without a bundle accepts a bare null by the 2026-10-06
   agreement and existing callers and tests rely on that; the bundle reader is where the producer
   contract is enforced.
5. Complete coverage means: per `job_id`, exactly the steps 1..horizon, nothing missing and nothing
   extra. It is required of every job in the cache, including models that are not a frozen
   representative. Alternative: check representatives only. Reason: the producer's own payload
   check is per job, and a cache with a short job is a malformed cache whichever model it is.
6. The step check is in the loader, not in `read_handoff`. `read_handoff` still returns the rows of
   a bundle that has a short job; `load_cache` (and so `build_tables` and the check command)
   refuses them. Alternative: refuse in `read_handoff` as well. Reason: the loader is the one place
   every producer-v1 frame passes through, bundle or not.

## Open

- The canonical-v1 path has the same gap and was not changed. The reviewer's case rebuilt as a
  canonical frame (no `job_id`) is still accepted and gives classical, cost 11.5319, scale 31.0638.
  canonical-v1 is used by the synthetic fixture; real caches arrive as producer-v1. A check per
  (dataset, config, series, model, seed, horizon, origin) would close it and needs a decision,
  because canonical input may carry missing realised values.
- The reason vocabulary comes from two branches that are not merged (`feat/1-omega-validation` and
  PR #56). If either changes before merging, `UNDEFINED_OMEGA_REASONS` in `handoff.py` must follow.
- The frozen manifest (`omega_values.json` on `feat/1-omega-validation`) stores one status per
  Omega column (`omega_levels_status`, `omega_levels_4096_status`, ...), per series. Values seen in
  the 8,753 rows inspected (entries naming M5 or PJM were skipped): `ok` 8,687 and `flat_history`
  66 for `omega_levels_status`; `ok` 8,562, `flat_history` 166 and `empty_history` 25 for
  `omega_levels_4096_status`. The engine picks such a column up by name, but nothing in this change
  reads the manifest, and there is no test against it.
- E1 and E2 (PR #52) do not report the reason yet. `tables["undefined_omega"]` is there for them.
- The change was not run on GitHub Actions; nothing was pushed.
