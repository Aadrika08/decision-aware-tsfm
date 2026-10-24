# Decision engine: the K3 selector in run_pilot (`--d-omega k3_selector`)

Status: tested on the synthetic pilot bundle of `test_decision_pilot_run.py` only. No real cache
has been read. Not a result. Part of #37. Follows `2026-10-15-k3-adapter-decisions.md`.

## What changed

- `run_pilot --d-omega k3_selector` puts the K3 Omega-plus-tau selector (`k3_adapter.build_k3_d_omega`,
  leave-one-dataset-out, the run's `--min-fit-datasets`, his margin 0.0, his floor of 3, scaled cost)
  in the D_Omega slot of every threshold run and of the fixed-classical sensitivity run. The default
  (`placeholder`) is unchanged. If the K3 module cannot be imported the command stops with the reason.
- E2 freezes the same selector per fold: new `k3_adapter.K3OmegaTauKind` (kind `k3_omega_tau`),
  registered with `e2.register_selector_kind`. `fit` returns his settings plus one frozen spec per
  horizon as plain JSON; `build` turns them back into the `K3SelectorAdapter` that `fit` made. The
  existing fold check (frozen E2 choices must equal the engine's d_omega choices) therefore applies to
  K3 as it did to the placeholder.
- `e2.evaluate`: a fold where the selector defers every unit (K3 below its floor) now gives an empty
  worst-decile table instead of raising "has no selected units". Every other E2 output is unchanged.
- Deferrals are priced as before in `run_pilot` (`deferral_priced`: the cost `pinball_validation` paid
  on the same unit, #29 item 8). The manifest names `d_omega = k3_selector` and
  `e2_selector_kind = k3_omega_tau`.

## Checks

On the test bundle (5 datasets, 330 d_omega units at the headline threshold), Python 3.11 and 3.10
give the same numbers: E2's frozen K3 choices equal the engine's on all 5 folds (330 same, 0
different); 264 units selected, 66 deferred (one fold where every unit defers), 0 unpriced; the other
six rules are identical to the placeholder run.

New tests: `K3SelectorInThePilot` (5 tests; skipped where the K3 module is not importable).
Planted bugs, each caught, with a clean control that passes: E2 left on the placeholder kind (99 units
differ, run stops), `build` ignoring the frozen margin (261 differ, run stops), the empty-fold guard
removed (the run errors).
