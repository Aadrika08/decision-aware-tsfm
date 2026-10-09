# Decision engine: the hindsight oracle on near ties (review of #44, item 1)

Status: behaviour change tested on hand-built tables and the synthetic fixture only. No real cache
has been read. Not a result.
Scope: `analysis/decision/selectors.py`, `evaluate.py`, `schema.py` and one new test file,
`tests/unit/decision/test_decision_near_ties.py`. Part of #37.

## The defect

`argmin_family` treats two costs within 1e-12 (relative, with an absolute floor of 1e-12) as a tie
and gives the tie to classical. The hindsight oracle and the `oracle_family` column used it, while
`oracle_cost` is the exact minimum of the two costs. On a near tie where TSFM is lower by less than
the tolerance, the oracle chose classical, paid the higher cost, and reported a regret against
itself. Reproduced before the change on Python 3.11 and 3.10 with the reviewer's example
(`J_classical=1.0`, `J_tsfm=1.0-5e-13`, `scale=1.0`): `run_selectors([HindsightOracle()], ...)` raised
`SchemaError: oracle has non-zero regret against itself`.

## Commands

Team pinned environment (Python 3.11, NumPy 1.26.4, pandas 2.2.3, PyArrow 16.1.0, SciPy 1.11.4):

- `PYTHONPATH=src python -m unittest discover -s tests/unit/decision -t .`

## What changed

- New `selectors.exact_argmin_family`: the family whose cost is the exact minimum, no tolerance.
  Only an exact tie goes to classical. A missing or infinite cost still gives no family.
- It is now used wherever a family is named in hindsight: `HindsightOracle`, the `oracle_family`
  column, and `best_fixed_family` (group totals of scaled test cost are compared exactly).
- `argmin_family` keeps its tolerance and is now used only for choices made from validation costs
  (`pinball_validation`, `crps_validation`). `TIE_RTOL` is unchanged.
- `validate_selection_frame` gained one check: on the oracle's rows, `chosen_family` must equal
  `oracle_family`. The existing check that the oracle's regret is exactly zero is unchanged.

## Verified in this change

- The reviewer's example now gives chosen family tsfm, `oracle_family` tsfm, regret 0.0. The other
  direction (`J_classical=1.0-5e-13`, `J_tsfm=1.0`) and an exact tie both give classical, regret 0.0.
- Decision suite: 117 tests (116 pass, 1 skipped without `DECISION_HANDOFF_BUNDLE`), on Python 3.11
  and 3.10. 12 of the 117 are new. With only the new helper added and nothing wired to it, 6 of
  the 12 fail; the other 6 pin behaviour that must not change (the validation tolerance, the exact
  tie, the classical-lower direction).
- All suites together: a throwaway tree made of main (`554b41c`) plus this change plus the E1/E2,
  E3, K2 and roster branches passes decision 117, experiments 165, e3 107, k2 149, the 3 team unit
  tests and the 96 team integration tests (17 skipped), on Python 3.11 and 3.10.
- Fixture run (`run_fixture`): all 5 output files byte-identical before and after.
- E1 fixture run: all 7 output files byte-identical before and after.
- E2 fixture run: every table identical before and after (the 3,388-row selection file included),
  apart from the selector-code fingerprint and the measured wall times. The frozen file name and
  hash change because `selectors.py` changed: 3efbf0bcf105 before, 5692688509f5 now.
- 12 deliberately planted bugs (tolerance put back into the exact rule, exact tie given to tsfm,
  comparison reversed, oracle or `oracle_family` or `best_fixed_family` left on the tolerance rule,
  missing-cost guard dropped, validation tolerance removed, `oracle_cost` taken as the maximum, new
  schema check removed, applied to every selector, or pointed at the wrong column) are each caught
  by the decision suite. Three of them (tolerance back in the exact rule, oracle left on the
  tolerance rule, `oracle_family` left on the tolerance rule) are caught only by the new tests.

## Design choices needing sign-off

1. Hindsight columns are exact; validation choices keep the tolerance. Alternative: one rule
   everywhere. Reason: the oracle is defined by the exact minimum that regret is measured against,
   so it must name that family. A validation selector choosing between two costs that differ by
   rounding noise should still prefer the cheaper family to run.
2. `best_fixed_family` is exact as well. Reason: it is also a hindsight label, and its cost feeds
   the oracle-value fraction. On the fixture this changes nothing.

## Open

- `analysis/experiments/bootstrap.py` (PR #52) builds its own `oracle_family` for per-origin cost
  tables with `argmin_family`. It should move to `exact_argmin_family` when that branch is updated,
  so both agree on near ties. E1's tie label keeps `TIE_RTOL` by design.
- `analysis/experiments/e3/scoring.py` (PR #53) uses `argmin_family` to name the winner between two
  synthetic processes. That is a comparison label, not the engine's oracle; left for the E3 review.
- E2 frozen files made before this change carry the previous selector-code fingerprint and will be
  refused by the current code by design; refreeze after this change.
- The handoff bundle check was not rerun: the bundle is not available in this workspace.
- Review item 2 (the decision tests are not in CI) is a separate change.
