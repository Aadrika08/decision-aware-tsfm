# 0007 - Development split and fitting roles

- **Date:** 2026-10-10, revised 2026-10-12 after Abdullah's sign-off.
- **Decided by:** not yet decided. Proposed by Veer. Abdullah (selector owner) signed off
  sections 1 to 4 on 2026-10-11
  ([comment on #57](https://github.com/Aadrika08/decision-aware-tsfm/pull/57#issuecomment-6111850251)),
  and the points he raised there are settled in sections 2 and 5. Sign-off by Pratham
  (mentor) is still to be recorded.
- **Issue:** [#45](https://github.com/Aadrika08/decision-aware-tsfm/issues/45). Code:
  PR [#44](https://github.com/Aadrika08/decision-aware-tsfm/pull/44), merged into main on
  2026-10-10 (the roles work is part of #38).
- **Status:** proposed. It becomes a decision when Pratham signs. Until then nothing below
  is frozen and Dayson should not freeze a window table from it.

## Context

Issue #45 states: "All-development alone is not the unseen-development evaluation split
required by the sprint." Three people are waiting for the split in writing:

- Dayson will "put role assignments in a separate window table keyed by (dataset, config,
  series_id, origin, horizon)" once the roles are agreed.
- Aadrika asked on 2026-10-10 for the "agreed split/roles (including horizon-dependent
  overlap purge and insufficient-origin exclusions)" to be recorded. The producer validates
  supplied bounds; it does not choose them.
- Abdullah's selector (#26) is to be fitted and scored under this protocol. The engine
  runs a placeholder in its slot until his code is in.

I proposed the outline on #45 on 2026-09-26: a time split inside each series (validation,
evaluation, gap), a dataset-level split for fitting selectors (leave-one-dataset-out), and
Omega taken at the first evaluation origin. The decision engine implements it in PR #44,
which is on main. This record describes what the code on main (`39d49e7`) does
(`analysis/decision/costs.py`, `pipeline.py`, `selectors.py`, `guards.py`) and the tests
that pin it (`tests/unit/decision/test_decision_roles.py`, `test_decision_selectors.py`,
`test_decision_undefined.py`). Everything so far has run on synthetic fixtures only. No
real forecast has been split with this rule.

Naming: the engine's code and output call the evaluation role `test`. This record says
evaluation. They are the same role.

## Decision

### 1. Roles inside a series: validation, gap, evaluation

Roles are assigned per (dataset, config, series_id, horizon). Models and seeds share them.
Origin follows the agreed convention: zero-based position of the first target, so a window
with horizon H at origin o covers target positions o to o + H - 1.

1. Take the distinct origins that have forecasts for that (series, horizon). Sort them
   ascending. Call their number n.
2. `n_val = floor(n / 3)`. The code computes `floor((1/3) * n + 1e-9)`; the 1e-9 only
   guards float rounding. n = 1..12 gives 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4.
3. The earliest `n_val` origins are validation candidates. The other `n - n_val` origins
   are evaluation. Let E be the earliest evaluation origin. E always exists when n >= 1.
4. Gap rule: a validation candidate at origin o becomes gap if `o + H > E`, that is, if
   its last target position `o + H - 1` is at or after E. It stays validation if
   `o + H <= E`. A gap origin is used for nothing: not validation, not evaluation, not
   fitting.
5. Exclusion: if no validation origin is left (n is 1 or 2, or every candidate became
   gap), that (series, horizon) is removed from the analysis. Its evaluation origins are
   not scored by any comparator and do not enter any fitting pool. It is listed in
   `excluded_series_horizons` with `used_for = "evaluation"` and exactly this reason:
   `no validation origins: the earliest-third validation window is empty after the gap
   rule, so validation comparators cannot be scored`. Other horizons of the same series
   are not affected.

There are no ties: origins are distinct integers and each gets one role per horizon. The
same origin can hold different roles at different horizons, which is why horizon is in the
table key. Datasets named as fit-only (section 2) get no roles; all their origins are used
for fitting.

Worked example (the numbers in `OriginRoles`, rerun on the code): one series, H = 24,
origins 100 to 399 at every step, n = 300. `n_val = 100`, candidates 100..199, E = 200.

| Role | Origins | Count | Why |
|---|---|---|---|
| validation | 100..176 | 77 | 176 + 24 = 200, not greater than E |
| gap | 177..199 | 23 | 177 + 24 = 201 > E |
| evaluation | 200..399 | 200 | the later two thirds |

Same origins, other horizons: H = 1 gives 100 validation and no gap. H = 100 gives 1
validation (origin 100) and 99 gap. H = 101 or H = 168 gives 0 validation and 100 gap, so
the (series, horizon) is excluded.

Second example, 9 origins spaced 24 apart (1000, 1024, ..., 1192), so `n_val = 3` and
E = 1072:

| H | validation | gap | evaluation | excluded |
|---|---|---|---|---|
| 24 | 1000, 1024, 1048 | none | 1072..1192 (6) | no |
| 48 | 1000, 1024 | 1048 | 1072..1192 (6) | no |
| 168 | none | 1000, 1024, 1048 | 1072..1192 (6) | yes |

For evenly spaced origins with spacing s, the number of validation origins kept is
`min(n_val, max(0, n_val - ceil(H / s) + 1))`. This matched the code on all 1,080 cases
tried (n = 1..60, s in 1, 6, 12, 24, 48, 168, H in 1, 24, 168). With s = 24 and H = 168 a
series needs n >= 21 origins to keep one validation origin (n = 21 gives 1 validation,
6 gap, 14 evaluation; n = 20 gives none). In general a series keeps a validation origin
at horizon H when `n >= 3 * ceil(H / s)`.

### 2. Dataset level: leave-one-dataset-out for fitted selectors

This applies to the fitted selector only (the D_Omega slot). The fixed-family rules, the
static Omega rule and the hindsight oracle are not fitted. The two validation comparators
choose from each series' own validation origins.

- One fold per evaluation dataset d. The fold key is the `dataset` field alone, so all
  configs of a dataset are left out together.
- The rule for d is fitted on every pool unit whose dataset is not d, and applied only to
  d's units. There are as many fitted rules as evaluation datasets.
- The pool has one row per (series, horizon, tau). For an evaluation dataset the row holds
  the costs over its evaluation origins and the Omega of section 3. For a fit-only dataset
  it holds the costs over all its origins and the Omega at its earliest origin. Validation
  and gap origins are not in the pool. Excluded (series, horizon) pairs are not in the
  pool. Rows with undefined Omega, cost or scale are dropped before fitting.
- A fold needs at least 3 other datasets with usable rows (agreed with Abdullah on
  2026-10-11). This is the same floor as record 0008 section 3, where his selector needs
  3 datasets with distinct Omega before it fits a threshold, so the two records carry one
  number. With fewer the fold is not fitted: every unit of d reports `insufficient_data`
  for the fitted selector, with the note
  `leave-one-dataset-out: N other dataset(s) with usable units, need 3`. The other six
  comparators still run. The engine takes this number as `min_fit_datasets`, and its
  default on main is still 2, so a run under this record passes 3 until the default is
  changed.
- A fold whose fitted datasets include d is refused with an error, whatever selector is
  inside.

### 3. Decision-time Omega

Each (series, horizon) gets one Omega: the value at its decision origin E, the earliest
evaluation origin, for the named variant. The join key is (dataset, config, series_id,
origin = E, variant). The history behind that value must not end after E. Omega is never
averaged over origins.

- The row must exist. A missing row is an error.
- A row that exists with a null value means Omega is undefined. The unit is kept. The two
  Omega-based comparators (static rule, D_Omega) report `insufficient_data` for it; the
  other five run as usual. The number of such units is reported. They are left out of fits.
- Omega exactly 0.0 is refused unless explicitly allowed. Values outside [0, 1] and
  infinite values are refused.

A null is accepted when it carries an `omega_status` of `empty_history`, `below_floor`,
`flat_history` or `nonfinite`, and the reason is kept and reported. A null without one of
these reasons is refused. (Before commit `461546c` the bundle reader, `handoff.py`,
rejected a present null row as if it were missing. Aadrika raised it in review of #44 and
confirmed the fix on main on 2026-10-11.)

### 4. What stays untouched: M5 and PJM

M5 and PJM get no role, no fold and no pool row. The engine refuses any forecast whose
dataset or config name contains `m5` or `pjm` (case-insensitive) before any cost is
computed, and the D_Omega fit refuses the same. The window table and the fold assignment
must contain no M5 or PJM rows. How a frozen rule is later scored on them is not set here.

### 5. Not decided here

Two design choices in the code are still open (from
`docs/results/decision/2026-10-09-development-roles.md`):

1. Pool for fold d uses the other datasets' evaluation units. Alternative: only their
   validation origins. Current choice gives more data per fold; its cost is that a shock
   shared by two datasets over the same calendar period can reach the fit for the other.
2. Exclusion is per (series, horizon). Alternative: per series, dropping every horizon of
   a series if any horizon has no validation origin.

Also not decided here: the static Omega rule and its threshold (the code has no default);
the model roster and the family representatives; the origin schedule and the number of
origins per series; which Omega variant is the headline; the single frozen rule that E2
carries to M5 and PJM. On the static rule, Abdullah proposes (on #57, 2026-10-11) that it
defer between 0.40 and 0.50 and that both edges be reported as a sensitivity, because 2 of
the 20 datasets sit in that band. That is for the group to decide.

How a deferred unit is priced is no longer open here. Record 0008 section 5 (PR #32)
proposes it: a deferred unit is priced from the engine's pinball-validation comparator,
supplied to the selector by the caller. Pricing at the oracle stays only as a labelled
upper bound (`deferral_pricing: "oracle_upper_bound"`).

Abdullah's selector (`origin/exp/26-omega-tau-selector`, `analysis/k3/selector.py`)
differed from the engine in four places. Settled with him on 2026-10-11:

- Unit: the engine's, (series, horizon, tau). His selector has no horizon and is not given
  one. The caller supplies the (horizon, tau) grouping, so the selector is fitted once per
  horizon, and section 3 fixes which Omega a series carries.
- Minimum data: 3, as in section 2.
- A fold that cannot be fitted: both behaviours are kept, because they answer different
  questions. In the comparator table the unit reports `insufficient_data` with no cost. In
  the E4 frontier every deferred unit needs a cost, so it is priced like any other
  deferral, from the validation comparator. His selector tags why it deferred
  (`insufficient_data`, `weak_evidence` or `omega_undefined`), so the two uses can be told
  apart.
- Fold membership: no fit-only datasets are planned for the real run. Every development
  dataset is rotated, so "rotate everything" and "never evaluate a fit-only dataset" are
  the same thing. The fit-only scheme stays a non-default option.

## Alternatives considered

- **Treat all development data as one set, with no split.** Rejected in #45: it gives no
  unseen-development evaluation.
- **One role per series.** Rejected: one series holds validation, gap and evaluation
  origins, and the overlap depends on horizon (Aadrika, #45, 2026-10-04).
- **A single fixed split of datasets into fit and evaluation.** Rejected in favour of
  rotation: Abdullah's K3 record counts 2 of 20 datasets in the 0.40 to 0.50 Omega band,
  so a fixed split can leave that band out of one side. The older fit-only scheme stays in
  the code as a non-default option (`fit_scheme="fit_only_datasets"`).
- **Omega averaged over a series' origins.** Rejected: it lets history after the decision
  into the decision.
- **Keep a series with no validation origin and let the validation comparators report
  insufficient_data.** This was the earlier behaviour. Replaced on 2026-10-07 by the
  exclusion with a recorded reason, so no unit is scored against an empty validation set.
  The earlier behaviour stays available as `exclude_without_validation=False`.

## Consequences

Numbers that depend on this record:

- Every development-data number for the seven comparators (realised cost, regret, scaled
  regret, fraction of oracle value) is computed on evaluation origins only.
- The two validation comparators choose from validation origins only, so their regret
  depends on `n_val` and on the gap rule.
- Every development number for the fitted selector comes from a rule that never saw the
  dataset it is scored on. This protocol produces one rule per dataset, not one boundary.
- The population differs by horizon, because longer horizons lose more (series, horizon)
  pairs to the gap rule. Tables across horizons do not compare the same series unless
  that is checked.
- One series gives a row at every horizon and tau, and those rows share one history, so
  they are not separate trials. Uncertainty is therefore clustered by series across
  horizons, not by (series, horizon): a resample draws whole series, and a drawn series
  brings all its horizons and taus with it. `analysis/experiments/bootstrap.py` does this
  when `cluster` is `series` or `dataset+series` (E2 uses `dataset+series`). Raised by
  Abdullah on #57.

To report in the paper: the one-third fraction, the gap rule, the number of excluded
(series, horizon) pairs per dataset and horizon with the reason, the number of units with
undefined Omega, the fold list, and any fold that was not fitted.

If decided differently: a fraction other than one third moves origins between validation
and evaluation and can change every number above. No gap rule would let validation costs
include targets from the evaluation period. Per-series exclusion would keep the same
series at every horizon and fewer of them. A validation-only pool would remove the
shared-period risk and give noisier fits. A lower minimum than 3 other datasets would fit
more folds on less evidence. Averaged Omega could change any Omega-based choice.

For Dayson: the window table can be built from section 1 alone, with one row per
(dataset, config, series_id, origin, horizon) and a role of validation, gap or evaluation,
plus a separate list of excluded (series, horizon) pairs. The fold assignment is one row
per evaluation dataset. An Omega row is required at (dataset, config, series_id, E,
variant) for every (series, horizon) that is kept. The engine derives roles from the
origins in the cache and does not read a window table, so a check that the two agree
still has to be written.
