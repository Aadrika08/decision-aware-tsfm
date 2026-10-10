# Bootstrap: the sample oracle on near ties (follow-up to the engine's near-tie note)

Status: behaviour change tested on hand-built tables and the synthetic fixture only. No real cache
has been read. Not a result.
Scope: `analysis/experiments/bootstrap.py` (`attach_sample_oracle`), one new test file,
`tests/unit/experiments/test_bootstrap_near_ties.py`, and three entries in
`tests/unit/experiments/planted_bugs.py`. Part of #42. Closes the first item under "Open" in
`docs/results/decision/2026-10-09-near-tie-oracle.md`.

## The defect

`bootstrap.attach_sample_oracle` named a unit's oracle family with the engine's `argmin_family`,
which treats two costs within 1e-12 (relative, with an absolute floor of 1e-12) as a tie and gives
the tie to classical. Since the engine change of the same day, the engine's hindsight oracle and
its `oracle_family` column use `exact_argmin_family`. On a near tie where TSFM is lower by less
than the tolerance the two disagreed. Reproduced before the change on Python 3.11 and 3.10 with a
hand-built unit (classical 1.0 and TSFM 1.0 - 5e-13 at each of 8 test origins, scale 1.0): the
engine named tsfm, the bootstrap named classical, the bootstrap's mean oracle cost was 1.0 while
the lower mean cost was 0.9999999999995, and the engine's oracle scored against the bootstrap's
oracle had mean scaled regret -5.000444502911705e-13 on the original table and on 40 of 40
resamples, where it must be zero.

## Commands

Team pinned environment (Python 3.11.15 and 3.10.20, NumPy 1.26.4, pandas 2.2.3, PyArrow 16.1.0,
SciPy 1.11.4), from the tree root with `PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1`:

- `python -m unittest discover -s tests/unit/decision -t .`
- `python -m unittest discover -s tests/unit/experiments -t .`
- `python -m unittest tests.unit.test_package_layout`
- `python -m unittest discover -s tests/integration -t .`
- `python tests/unit/experiments/planted_bugs.py --jobs 4`
- `python -m decision_aware_tsfm.analysis.experiments.e1 --out <dir>`
- `python -m decision_aware_tsfm.analysis.experiments.e2 --out <dir>`
- `python -m decision_aware_tsfm.analysis.experiments.coverage_sim --out <dir> --only regret --reps 20 --n-boot 50`

## What changed

- `attach_sample_oracle` now calls `exact_argmin_family`: the family with the exact lower mean
  cost on the rows given, no tolerance, only an exact tie goes to classical. Its docstring says so.
  `bootstrap.py` no longer imports `argmin_family`.
- Nothing else in `analysis/experiments/` used `argmin_family`. The other tolerances were read and
  left as they are, because none of them names a family in hindsight:
  - `e1.decision_map` labels a unit a tie with `TIE_RTOL` on scaled cost. Kept by design: it is a
    label for the map ("neither family"), not the oracle.
  - `bootstrap.fixed_choice_costs` reads `oracle_family` from the engine's selection frame, so it
    is exact already. Its `ORACLE_CHECK_RTOL` (1e-9) only checks that the selection frame and the
    per-origin table belong together.
  - `e2.regret_summary` uses 1e-12 to decide whether a ratio is defined (oracle cost zero,
    best-fixed equal to the oracle). Same zero rule as the engine's `evaluate.summarize`.
  - `stat_mean_scaled_regret_recomputed_oracle` already takes the exact minimum.
  - `coverage_sim.py` and `e2.py` contain no tie rule of their own for naming a family.

## Verified in this change

- New tests, 4: TSFM lower inside the tolerance, classical lower inside the tolerance, an exact
  tie, and near ties plus clear gaps at five magnitudes from 1e-13 to 1e6 in both directions. Each
  builds a per-origin table by hand, runs the engine's hindsight oracle on it, and checks that
  `attach_sample_oracle` names the same family for every unit, that the unit's mean oracle cost
  equals the lower of its two mean costs exactly, and that the engine's oracle scored against the
  bootstrap's oracle is exactly zero for every unit on the original table and on each of 40
  resamples.
- Against the unchanged code 2 of the 4 fail (TSFM lower inside the tolerance, and the magnitudes
  test); the other 2 pin behaviour that must not change (the classical-lower direction, the exact
  tie). After the change all 4 pass. Same on Python 3.11 and 3.10.
- After the change the hand-built unit of "The defect" gives tsfm from both, mean oracle cost
  0.9999999999995, regret 0.0 on the original table and 0 of 40 resamples non-zero.
- Suites after the change, same counts on Python 3.11 and 3.10: decision 117 (116 pass, 1 skipped
  without `DECISION_HANDOFF_BUNDLE`), experiments 169 (165 before, plus the 4 new), team unit
  tests 3 (`tests.unit.test_package_layout`), team integration tests 66 (61 pass, 5 skipped
  because the release files they check are not downloaded).
- Planted bugs: 28 of 28 caught, 0 survived, on Python 3.11 and 3.10; a clean copy passed after
  all restores and the working tree was not modified by the tool. 25 are the earlier M1 to M16 and
  N1 to N9. The 3 new ones:

  | bug | what was planted | outcome | tests that failed |
  |---|---|---|---|
  | T1 | `attach_sample_oracle` is back on the tolerance rule (`argmin_family`) | caught | 2, both new |
  | T2 | exact rule reversed, the dearer family is named | caught | 7: 4 earlier tests and 3 new |
  | T3 | an exact tie goes to tsfm | caught | 1, new |

- E1 fixture run: all 7 output files byte-identical before and after, on both Pythons.
- E2 fixture run, 9 files: 7 byte-identical before and after on both Pythons, the 200-resample
  `regret_intervals.csv` and the 3,388-row selection file included. `selector_cost.csv` differs
  only in the two wall-time columns (`seconds`, `seconds_per_unit`). `manifest.json` differs only
  in `frozen_path`, which holds the output folder. The text printed by the E1 and E2 commands is
  identical before and after apart from the output folder name.
- E2 hashes did not move: selector code hash 60caf131b2af and frozen-file hash 5692688509f5
  before and after. The selector code hash covers the source of the adapter class in `e2.py` and
  of the whole `decision/selectors.py`; `bootstrap.py` is not in it. No refreeze is needed for
  this change.
- Coverage simulation, the 7 regret designs (the only callers of `attach_sample_oracle` outside
  the tests), at 20 replications of 50 resamples: `coverage.csv` byte-identical before and after
  on both Pythons.

## Open

- The full coverage simulation (300 replications of 300 resamples) was not rerun. The table in
  `2026-10-04-e1-e2-bootstrap.md` was produced with the tolerance rule; the reduced run above
  found no difference.
- The engine compares unit means computed by `costs.unit_costs`; `attach_sample_oracle` computes
  its own means from the per-origin rows. Both now use the exact rule, so they agree whenever the
  two means are the same floats. Two families whose means differ only by summation rounding (far
  below the 5e-13 tested here) could still be named differently. E2 is not exposed to this: it
  reads the engine's `oracle_family` through `fixed_choice_costs`.
- E1's map label can differ from the engine's `oracle_family` on a near tie: the map says "tie",
  the oracle names the exactly cheaper family. By design. Design choice 8 of
  `2026-10-04-e1-e2-bootstrap.md` still describes the engine's oracle as reporting a tie; that
  note was not edited.
- `e1.decision_map`'s docstring mentions `argmin_family` as the engine's rule on raw cost. That is
  still true of `argmin_family`, which the engine now uses for validation choices only. Not edited.
- `analysis/experiments/e3/scoring.py` (PR #53) is not on this branch and was not looked at.
- The handoff bundle check was not rerun: the bundle is not available in this workspace.
