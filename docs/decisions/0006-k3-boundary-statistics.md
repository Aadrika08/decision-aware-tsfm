# 0006 — K3 boundary statistics: oracle baseline, unit of analysis, Omega requirements

- Date: 2026-09-25
- Owner: Abdullah
- Issue: [#23](https://github.com/Aadrika08/decision-aware-tsfm/issues/23), parent [#21](https://github.com/Aadrika08/decision-aware-tsfm/issues/21)
- Status: proposed. Pending sign-off from the project lead. Veer's positions on
  the oracle definition and Omega coverage are recorded below and were given as
  reasoned positions rather than measured results; all three items are
  pre-registered before any real multi-series K3 result is seen.

## Scope

Fix the statistical choices K3 depends on: what the oracle is measured against,
what counts as one observation, and how much Omega variation is required before
a boundary is fitted at all. These choices change the headline K3 number, so
they are fixed before real data rather than after.

This decision does not freeze datasets, splits, the model roster, within-family
representatives, tau values or the Omega variant used for the headline result.
M5 and PJM remain untouched.

## 1. Oracle baseline: best fixed family

The fraction of oracle value captured by a threshold is measured against the
**best** fixed family, not the worst:

```
fraction = (best_fixed_cost - threshold_cost) / (best_fixed_cost - oracle_cost)
```

The oracle is zero and the best fixed family is the scale. The feasibility-sprint
skeleton used the worst fixed family, which flatters any threshold: a rule that
merely avoids the worse family scores well without using Omega at all. Candidate
thresholds include both extremes, so "always one family" is always available and
the fraction cannot be negative under this baseline.

When the best fixed family already matches the oracle the denominator is zero.
This is reported as `no_oracle_gap` with the fraction left undefined; it is never
divided through or filled with a placeholder. Veer's evaluation engine marks the
same case undefined, and both use oracle-as-zero with best-fixed as the scale.
The definition must stay identical in both: it is the regret denominator in the
K3 tables and in the decision engine, and silent drift would only surface as
disagreeing tables.

The level at which the oracle is taken follows decision 2: the minimum is taken
per series, after aggregating origins, not per row and not per dataset. A coarser
level raises the oracle's cost and therefore shrinks reported regret.

`baseline="worst_fixed"` remains available so the two definitions can be compared
in a sensitivity check. It is not the headline definition.

## 2. Unit of analysis: series, not series-origin

Omega is computed per series **and** origin, from the history preceding each
forecast origin, so one series can carry several Omega values. K3 treats the
**series** as the unit of analysis and aggregates origins within a series.
Treating each series-origin as an independent observation requires an explicit
argument and is not the default.

Evidence: in K2, the same data and the same comparison gave rho = -0.088
(p = 0.41) at config level and rho = -0.417 (p = 0.068) at dataset level. The
unit of analysis moved the answer substantially. Overlapping origins are a
stronger version of that problem, because they share actual observations and not
only a descriptor value.

Uncertainty uses the moving block bootstrap at series and domain level already
required by the implementation plan, so nearby windows are never counted as
independent trials. Cross-validated comparisons in `fit_monotone_boundary`
accept group labels and use grouped folds so one series never appears in both
the training and test side of a fold.

## 3. Minimum Omega requirements and coverage reporting

Before any fit:

- at least **3** distinct Omega values for a monotone fit or a threshold search;
- at least **8** distinct Omega values for an unconstrained flexible fit.

Below these, the analysis returns an explicit `insufficient_data` result naming
the count and the requirement. It never returns a fitted curve, a threshold, or
a number that could be read as a boundary estimate.

These floors guard against fitting a curve to too few points. They do not
establish that the available Omega values are well spread. Of the 20 datasets
in the K2 replication, 6 sit below 0.40, 2 between 0.40 and 0.50, and 12 above
0.50. The published transition band is exactly the region with 2 datasets, so a
boundary estimated there rests on those two. Coverage per Omega band is
therefore reported alongside every K3 result, and any boundary landing in the
sparse band is reported as resting on that coverage rather than presented as a
supported estimate.

## 4. Fitting choices fixed in advance

- Both threshold orientations are searched (classical below the threshold and
  TSFM above, and the reverse). The earlier skeleton tested only one.
- The monotone direction is either pre-registered before seeing the data, or
  both increasing and decreasing fits are reported. The better-fitting direction
  is never selected after the fact and presented as the only one considered.
- Monotone and flexible fits are compared by out-of-fold error, not in-sample
  error, which always favours the more flexible model.
- Malformed input raises rather than returning a result. Insufficient input
  returns `insufficient_data`. The two are distinct conditions.

## Alternatives considered

- **Worst fixed family as baseline.** Rejected as the headline: it makes a
  threshold look useful for avoiding the worse family rather than for using
  Omega. Retained as a sensitivity option.
- **Series-origin as the unit.** Rejected as the default: more observations, but
  correlated ones, and the K2 evidence shows the unit changes the answer.
- **No minimum Omega requirement.** Rejected: a fitted curve always returns a
  boundary, including where none exists, which is the failure mode K3 is meant
  to detect.

## Consequences

A no-boundary or insufficient-data outcome stays a valid, reportable result.
None of the choices above may be revisited after seeing real K3 results in order
to obtain a boundary; changing any of them requires a new decision record.

## Sources

- [PR #32](https://github.com/Aadrika08/decision-aware-tsfm/pull/32) — implementation and tests
- Veer's K2 unit-of-analysis figures and Omega band coverage, recorded on [#21](https://github.com/Aadrika08/decision-aware-tsfm/issues/21)
- `docs/runbooks/cache-consumer-contract.md` — oracle and regret conventions in the evaluation engine
