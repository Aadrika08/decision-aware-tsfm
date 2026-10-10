# Decision engine and E1: undefined Omega and zero scale

Status: behaviour change tested on the synthetic fixture only. No real cache has been read. Not a result.
Scope: `analysis/decision/pipeline.py`, `analysis/experiments/e1.py` and their tests. Part of #38.
Raised by the data owner on 2026-10-05: `compute_omega` returns 0.0 for a flat or empty history.

## Commands

Team pinned environment (Python 3.11, NumPy 1.26.4, pandas 2.2.3, PyArrow 16.1.0, SciPy 1.11.4):

- `PYTHONPATH=src python -m unittest discover -s tests/unit/decision -t .`
- `PYTHONPATH=src python -m unittest discover -s tests/unit/experiments -t .`

## What changed

- Undefined Omega. A decision whose Omega row exists but holds null is kept with Omega NaN. The two
  Omega-based comparators (static Omega rule, D_Omega) report `insufficient_data` for that unit; the
  other five comparators run as usual. `tables["n_units_undefined_omega"]` gives the count. Fit-only
  series with undefined Omega are left out of the D_Omega fit.
- A decision with no row at all in the Omega table is still an error (a missing join is a pipeline
  fault, not an undefined statistic).
- Omega exactly 0.0 is refused with a message naming the affected series, unless
  `allow_zero_omega=True` is passed. Finite values outside [0, 1], infinite values and non-numeric
  values are refused. The setting is recorded in `tables["settings"]`.
- Zero scale. A (series, horizon) whose realised targets are all 0 over the scored origins has mean
  |y| = 0, so scaled cost is undefined. It is removed from the evaluation (or fit) tables and listed
  in `tables["excluded_series_horizons"]` with the reason and whether it was evaluation or fit data.
  Before this change such a series stopped the whole run at the output-schema check.
- E1. `decision_map` keeps a unit with undefined Omega (costs, Delta and lower-cost family are still
  reported). `summarize_map` leaves it out of the sign-reversal count only and reports
  `n_undefined_omega` per group. Infinite Omega is refused.
- E2 needed no change: with an undefined Omega the frozen Omega-based selectors defer and the
  others are scored.

## Verified in this change

- Decision suite: 87 tests (86 pass, 1 skipped: the real-bundle test, which needs
  `DECISION_HANDOFF_BUNDLE`). Experiments suite: 164 tests pass.
- On the fixture with one evaluation series set to undefined Omega: 22 units (2 horizons x 11 taus)
  are undefined; the static rule and D_Omega report `insufficient_data` on exactly those units and
  every other comparator and every other series is unaffected. E1 reports 1 undefined unit in each
  of the 22 (horizon, tau) cells, 22 units per cell, and 21 selected units for the static rule.
- Delta and the lower-cost family of the undefined-Omega series are identical to the ordinary
  fixture run, since costs do not depend on Omega.

## Design choices needing sign-off

1. Undefined Omega is kept and deferred on, not excluded. Alternative: exclude those series from the
   workload. Reason: Omega is computed per origin, so one flat window should not remove a whole
   series, and removing the cases where Omega cannot be computed would make an Omega-based rule look
   better than it is. The count of deferred units is reported.
2. Omega exactly 0.0 is an error by default. Alternative: accept it as a value. Reason: the team's
   `compute_omega` returns 0.0 for degenerate input, and a threshold rule would turn that into a
   confident "classical" choice. A real series essentially never has Omega exactly 0.
3. Zero-scale series are excluded with a recorded reason. Alternative: keep them and report unscaled
   cost only. Reason: every summary in the engine uses scaled cost, and an undefined scale would
   propagate silently; exclusion with a reason keeps the population explicit.

## Open

- Data owner: store undefined Omega as null with a reason code in the frozen manifest and Omega
  sidecar, per origin and per variant, not as 0.0.
- `omega/compute.py` (not edited here; another owner) returns 0.0 from five separate checks (fewer
  than 2 finite points, fewer than 2 after differencing, no frequency bins, zero total power, a
  single frequency bin). Returning NaN there would remove the ambiguity at the source.
- How many pilot units are affected is unknown until the frozen manifest is available. The data
  owner reports 2 electricity series in the pilot and more in Bitbrains (not in the pilot).
