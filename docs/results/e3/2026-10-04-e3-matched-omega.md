# E3: matched-Omega study (synthetic series, idealised forecasters)

Status: SYNTHETIC MECHANISM DEMONSTRATION, NOT A TEST. This is the corrected version of the note,
written on 2026-10-04 after an independent statistical review. It replaces the first version of
the same date; the section "Corrections to the first version" lists what changed and why. Three
things have to be understood before reading any number below.

1. The outcome is fixed by the definitions. Both "forecasters" are closed-form idealisations that
   are handed the true conditional mean and the true scale. Which of the two has the lower cost
   for a given innovation law and tau follows from how they are defined and from one free
   parameter, the size of the location error given to F. It can be worked out exactly before any
   series exists. E3 as built could not have failed to show a flip.
2. The outcome does not depend on Omega or on the Omega match. The structure of the series, its
   Omega, and whether a pair passed the match all cancel out of both forecasters' costs. Pairs
   that failed the match behave like pairs that passed, and the same rates appear with no time
   series at all. "Matched Omega" describes how the dataset was built; it is not an ingredient of
   the result.
3. It says nothing about real classical models or real foundation models. No forecasting model
   was run or imported. The engine's family names `classical` and `tsfm` appear only because the
   engine requires two slot labels. G is not a classical model and F is not a TSFM.

What it is good for. It demonstrates, with exact algebra and a simulation that measures only the
noise around that algebra, one mechanism by which a rule that sees only Omega and tau can be
wrong: two processes can share their second-order structure, and therefore the population
spectrum that Omega summarises, while their innovation shape differs, and a forecaster with a
Gaussian quantile shape and a forecaster with the right shape but a location error then change
order at some quantiles. How often they change order is set by the error size, not found by the experiment:
F is the population winner in 64 of the 66 (alternative law, tau) cells with no error, in 41
with error 0.10 sigma, in 18 with 0.25 sigma (the primary variant) and in 3 with 0.50 sigma, and
13 of the 66 break-even errors lie within 0.05 of 0.25.

What a version that could fail would need. Forecasters that have to estimate the conditional
mean and the distribution from data, which means the team's real forecast pipeline run on these
series (Open, item 1). Until that exists, E3 must not be used as the gate for a later experiment
("if E3 fires", "pairs do not flip" in `docs/project-overview.md`).

Dated 2026-10-04. Every number below comes from a command run in the session that produced this
version.

## Commands

    PY=<team python 3.11.15; numpy 1.26.4, pandas 2.2.3, scipy 1.11.4, pyarrow 16.1.0>

    # tests (107 tests; 40 s wall on a busy two-core machine, 30 s of CPU)
    PYTHONPATH=src $PY -m unittest discover -s tests/unit/e3 -t .

    # Omega-only simulation: 4000 fresh seeds per (structure, member), nothing scored (about 2.5 minutes)
    PYTHONPATH=src $PY -m decision_aware_tsfm.analysis.experiments.e3.omega_only \
        --out artifacts/runs/e3_matched_omega_rev1

    # every other table, rebuilt in a new directory from the series saved by the original run
    # (engine scoring included: about 8 minutes on a quiet machine. With --reuse-units --skip-part-a
    # the saved unit costs are read back instead of calling the engine: about 35 s)
    PYTHONPATH=src $PY -m decision_aware_tsfm.analysis.experiments.e3.run \
        --out artifacts/runs/e3_matched_omega_rev1 --series-from artifacts/runs/e3_matched_omega

    # the original full run (generates the series; not repeated for this version)
    PYTHONPATH=src $PY -m decision_aware_tsfm.analysis.experiments.e3.run --out artifacts/runs/e3_matched_omega

Run record: design seeds 0 to 29; `series_values.parquet` sha256
0511b1ce2c22f5cb952264af49fede21f4d9dc59394e3c0995b80bddeb894206 (read from
`artifacts/runs/e3_matched_omega/`, which was not modified); Omega-only seeds 1000 to 4999;
settings and package versions in `run_manifest.json` and `omega_only_manifest.json` in
`artifacts/runs/e3_matched_omega_rev1/`. The working tree is not a git repository, so there is no
commit hash, and no version history of `design.py`.

## What was built

Code, all under `src/decision_aware_tsfm/analysis/experiments/e3/`:

| File | Purpose | Changed in this version |
|---|---|---|
| `design.py` | Design constants: lengths, Omega tolerance, structures, innovation laws, forecaster variants. | Comments only: two dated corrections added beneath the original reasoning. No value changed. |
| `omega_reconcile.py` | Part A sweep of Omega against noise sd. | No |
| `generator.py` | Seeded generator, per-series statistics, Omega match check, negative controls. | No |
| `forecasters.py` | The two idealised forecasters in the engine's canonical-v1 frame. | No |
| `scoring.py` | Engine costs and winners, sign counts, population cost table. | Added: joint interval for the difference from baseline, split by population-winner status, Wilson fallback for all-equal counts. |
| `population.py` | NEW. Closed-form break-even error per (law, tau) and the F-win count as a function of error size. | New |
| `verify.py` | NEW. Costs recomputed from the innovations alone; simulation with no series; matched against failed pairs. | New |
| `omega_only.py` | NEW. Omega-only simulation of the match failure rate by member, with tests and power. | New |
| `run.py` | The single command. | Writes the new tables; can read saved series from another directory. |

Omega is always the team's `compute_omega` (imported, not copied). Reported costs are always the
decision engine's. `verify.py` contains its own short pinball loss on purpose: it is the
independent route used to check the engine's costs. The only numbers reported from it are the
verification tables of section 2.

Dataset (unchanged): AR(1) with phi = 0.50, 0.80, 0.95 and a sine-plus-noise structure (daily
plus half-weekly sine from `analysis/k3/matched_omega_pair.py`, noise sd 1.0); a fifth structure
(sine, noise sd 2.0) is a negative control only. Members: `gaussian` (reference), `gaussian_b`
(a second independent Gaussian draw, the noise-only baseline), skew-normal with shape +2, +8,
-2, -8, Student t with 5 and 3 degrees of freedom, each standardised analytically to mean 0 and
variance 1. Each member has its own random stream. 3000 points per series: 2000 context (Omega
and all recorded statistics use only these) and 1000 one-step test targets. 30 seeds, 990 series.
A pair is (Gaussian reference, other member) of one (structure, seed); nothing is filtered, and
failed matches stay in every table.

Idealised forecasters (unchanged), horizon 1, both given the true conditional mean `m_t` and the
true scale `sigma`:

- G: `q_tau = m_t + sigma * z_tau`. Gaussian shape. On a Gaussian law this is the exact oracle.
- F: `q_tau = m_t + sigma * (Q_law(tau) + b)`. The true quantile of the law plus a location error
  `b`: 0.25 (primary), 0.10 and 0.50. A fourth variant uses the empirical quantiles of the 2000
  context innovations.

Delta = J_F - J_G (mean pinball loss), so Delta > 0 means G is cheaper. Ties go to G (engine rule).

Tables in `artifacts/runs/e3_matched_omega_rev1/`: population `breakeven_error.csv`,
`f_win_count_by_error.csv`, `breakeven_near_primary_error.csv`, `expected_delta.csv`; verification
`verify_innovation_only.csv`, `verify_no_series.csv`, `verify_no_series_vs_matched.csv`,
`verify_sign_by_match_status.csv`; simulation `unit_costs.csv`, `sign_pairs.csv`,
`sign_difference_vs_baseline.csv`, `sign_by_population_status_tau.csv`,
`sign_by_population_status_member_tau.csv`, `sign_by_population_status_total.csv`,
`sign_summary_by_tau.csv`, `sign_summary_by_member_tau.csv`, `sign_summary_by_structure_tau.csv`,
`member_delta_summary.csv`, `member_delta_summary_by_structure.csv`; dataset and match
`series_table.csv`, `series_summary_by_member.csv`, `match_pairs.csv`, `match_groups.csv`,
`match_summary.csv`, `match_by_member.csv`; Omega only `omega_only_draws.csv`,
`omega_only_by_member.csv`, `omega_tolerance_in_sd_units.csv`; Part A `part_a_omega_sweep.csv`,
`part_a_summary.json`.

## Verified in this change

### 1. The result: a closed-form table, no data involved

In units of sigma, G puts its quantile at `z_tau` and F at `Q(tau) + b`. Let
`g(c) = E[pinball_tau(e - c)]`. Then `g'(c) = CDF(c) - tau`, so g is convex with its minimum at
the true quantile, and

    J_F - J_G = g(Q + b) - g(z) = integral from z to Q + b of (CDF(x) - tau) dx.

The series, phi, the sine, Omega and the match do not appear. For each law and tau there is a
break-even error `b*`: F is cheaper than G exactly when `0 <= b < b*`. If G's quantile lies on
the side F errs towards, `b* = z - Q` exactly; otherwise `b*` is where g returns to `g(z)` on the
far side of its minimum. On a Gaussian law `z = Q`, so `b* = 0` and F cannot win with any error.

Break-even location error `b*` (sigma units). F wins a cell only if its error is below the entry.
Entries above the primary error 0.25 are the 18 cells where F is the population winner (marked *).

| tau | skewnorm_p2 | skewnorm_p8 | skewnorm_n2 | skewnorm_n8 | student_t5 | student_t3 |
|---|---|---|---|---|---|---|
| 0.01 | 0.238 | 0.298* | 0.332* | 0.594* | 0.280* | 0.295* |
| 0.05 | 0.135 | 0.269* | 0.134 | 0.268* | 0.080 | 0.242 |
| 0.09 | 0.080 | 0.182 | 0.061 | 0.139 | 0.125 | 0.274* |
| 0.25 | 0.029 | 0.100 | 0.053 | 0.086 | 0.108 | 0.209 |
| 0.40 | 0.072 | 0.184 | 0.082 | 0.167 | 0.046 | 0.092 |
| 0.50 | 0.083 | 0.192 | 0.083 | 0.187 | 0.000 | 0.000 |
| 0.60 | 0.083 | 0.172 | 0.072 | 0.181 | 0.046 | 0.094 |
| 0.75 | 0.053 | 0.087 | 0.029 | 0.100 | 0.112 | 0.233 |
| 0.91 | 0.062 | 0.146 | 0.084 | 0.205 | 0.134 | 0.335* |
| 0.95 | 0.143 | 0.300* | 0.149 | 0.389* | 0.084 | 0.286* |
| 0.99 | 0.415* | 0.861* | 0.315* | 0.848* | 0.329* | 0.340* |

Cells (of 66 = 6 alternative laws x 11 taus) where F is the population winner, by error size:

| error b (sigma) | 0.00 | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 | 0.30 | 0.35 | 0.40 | 0.45 | 0.50 | 0.60 | 0.75 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F wins | 64 | 60 | 41 | 30 | 23 | 18 | 10 | 5 | 4 | 3 | 3 | 2 | 2 |
| of which tau outside 0.25 to 0.75 | 36 | 36 | 30 | 22 | 21 | 18 | 10 | 5 | 4 | 3 | 3 | 2 | 2 |

- At error 0 the count is 64, not 66, because the two Student t laws have median 0 like the
  Gaussian, an exact tie that goes to G.
- The counts are identical for an error of the opposite sign (`breakeven_neg`), because the
  member set is symmetric (each skew-normal has its mirror image, the t laws are symmetric).
- 0.25 sits near a knife edge. 13 break-evens lie within 0.05 of it: five just below (0.205,
  0.209, 0.233, 0.238, 0.242) and eight just above (0.268, 0.269, 0.274, 0.280, 0.286, 0.295,
  0.298, 0.300). Moving the error from 0.20 to 0.30 moves the count from 23 to 10.
- Size of the population gap at error 0.25, as a fraction of G's cost: on a Gaussian law G is
  cheaper by 2.6 to 3.8 percent at every tau. In the 18 cells where F wins, F is cheaper by 0.3
  to 25.5 percent, and in 8 of the 18 by less than 1 percent.
- There is no evidence either way about when 0.25 was chosen relative to these break-evens: the
  tree has no version history. Pre-declaration protects nothing here in any case, because the
  whole table is computable before any data exists.
- Cross-checks: the 264 rows of `expected_delta.csv` (integration from minus infinity) agree with
  the finite-interval identity above to 2.4e-13; the counts at 0.10, 0.25 and 0.50 equal the
  population winners in that table (41, 18, 3).

### 2. Verification that structure, Omega and the match play no part

- Innovation-only recomputation. All 42,240 unit costs (960 series x 11 taus x 4 F variants)
  were recomputed from `e_t = x_t - m_t` alone in plain numpy. Largest difference from the engine:
  1.1e-15 for J_G and 1.7e-15 for J_F. Winner mismatches: 0. (`verify_innovation_only.csv`.)
- No series at all. 4000 replicates per member of 1000 i.i.d. innovations, compared with an
  independent Gaussian draw, primary variant. Sign-difference rate in percent, next to the rate
  among Omega-matched pairs of the series simulation:

| tau | 0.01 | 0.05 | 0.09 | 0.25 | 0.40 | 0.50 | 0.60 | 0.75 | 0.91 | 0.95 | 0.99 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| alternatives, no series | 76.9 | 41.0 | 14.0 | 4.9 | 2.9 | 3.6 | 2.8 | 0.3 | 22.8 | 47.9 | 69.9 |
| alternatives, matched pairs | 81.1 | 39.7 | 12.6 | 4.5 | 3.2 | 3.5 | 2.5 | 0.0 | 22.4 | 48.8 | 71.2 |
| baseline, no series | 21.4 | 3.6 | 1.5 | 0.1 | 0.1 | 0.1 | 0.2 | 0.4 | 2.5 | 5.3 | 22.9 |
| baseline, matched pairs | 24.2 | 3.2 | 2.1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 2.1 | 3.2 | 17.9 |

- Failed against matched. The 126 alternative pairs that FAILED the Omega match differ at the
  same rates as the 594 that passed (primary variant, percent, failed then matched): tau 0.01
  81.0 and 81.1; 0.05 37.3 and 39.7; 0.95 47.6 and 48.8; 0.99 68.3 and 71.2. Over the 22 (pair
  kind, tau) cells of the primary variant the smallest Fisher p is 0.098; over all 88 cells of
  the four variants 4 have p below 0.05, which is what chance alone gives.
- Through the engine. A unit test puts one innovation sequence into an AR(1) series with
  phi = 0.95 and into the sine series. The two Omegas differ by more than 0.1; the engine's costs
  and winners for G and for the primary F are identical to 1e-12.

Consequence: "the four structures agree" (first version, reading 5) is an identity, not a
robustness result. All four scored structures also have sigma = 1.

### 3. The simulation, read as a check on noise

Sign of Delta within Omega-matched pairs, primary variant, engine costs. A pair "differs" when
the lower-cost forecaster is G for one member and F for the other.

Criterion (defined here; the first version used "above baseline" without defining it): a tau is
"above baseline" when the lower end of the 95 percent interval for (alternative rate minus
baseline rate) is above zero. The interval is one joint bootstrap (2000 resamples of whole
(structure, seed) groups, both rates recomputed from the same resample, since the six alternative
pairs and the baseline pair of a group share one reference series). Where either rate is 0 of n
the bootstrap cannot move; the interval is then built from the two arm intervals by Newcombe's
square-and-add rule, using a Wilson interval on the number of groups for the empty arm (marked N).
The criterion says only that alternatives differ from the reference more often than a second
Gaussian draw does. It says nothing about the size of a cost gap.

| tau | Alternative differ / matched | % | Baseline differ / matched | % | Difference, points | 95 % interval | Above baseline |
|---|---|---|---|---|---|---|---|
| 0.01 | 482/594 | 81.1 | 23/95 | 24.2 | 56.9 | 46.1 to 67.5 | yes |
| 0.05 | 236/594 | 39.7 | 3/95 | 3.2 | 36.6 | 31.8 to 40.7 | yes |
| 0.09 | 75/594 | 12.6 | 2/95 | 2.1 | 10.5 | 6.4 to 13.9 | yes |
| 0.25 | 27/594 | 4.5 | 0/95 | 0.0 | 4.5 | 0.4 to 6.3 (N) | yes |
| 0.40 | 19/594 | 3.2 | 0/95 | 0.0 | 3.2 | -0.9 to 4.6 (N) | no |
| 0.50 | 21/594 | 3.5 | 0/95 | 0.0 | 3.5 | -0.6 to 5.0 (N) | no |
| 0.60 | 15/594 | 2.5 | 0/95 | 0.0 | 2.5 | -1.5 to 3.8 (N) | no |
| 0.75 | 0/594 | 0.0 | 0/95 | 0.0 | 0.0 | -3.9 to 3.1 (N) | no |
| 0.91 | 133/594 | 22.4 | 2/95 | 2.1 | 20.3 | 17.1 to 23.1 | yes |
| 0.95 | 290/594 | 48.8 | 3/95 | 3.2 | 45.7 | 41.4 to 49.7 | yes |
| 0.99 | 423/594 | 71.2 | 17/95 | 17.9 | 53.3 | 43.3 to 62.8 | yes |

Arm intervals (percent): the alternative rate 0/594 at tau 0.75 is 0.0 to 3.1 (Wilson on 119
groups), not "0.0 to 0.0"; the baseline rate 0/95 is 0.0 to 3.9.

The sign count mixes two different things. Split by population-winner status ("reversal" = the
population winner on the member differs from the population winner on the Gaussian reference;
"same winner" = it does not, so a realised difference there is a near-tie decided by noise):

| tau | Reversal cells | differ / matched | % | Same-winner cells | differ / matched | % |
|---|---|---|---|---|---|---|
| 0.01 | 5 | 437/498 | 87.8 | 1 | 45/96 | 46.9 |
| 0.05 | 2 | 172/201 | 85.6 | 4 | 64/393 | 16.3 |
| 0.09 | 1 | 55/99 | 55.6 | 5 | 20/495 | 4.0 |
| 0.25 | 0 | | | 6 | 27/594 | 4.5 |
| 0.40 | 0 | | | 6 | 19/594 | 3.2 |
| 0.50 | 0 | | | 6 | 21/594 | 3.5 |
| 0.60 | 0 | | | 6 | 15/594 | 2.5 |
| 0.75 | 0 | | | 6 | 0/594 | 0.0 |
| 0.91 | 1 | 98/99 | 99.0 | 5 | 35/495 | 7.1 |
| 0.95 | 3 | 260/300 | 86.7 | 3 | 30/294 | 10.2 |
| 0.99 | 6 | 423/594 | 71.2 | 0 | | |
| All | 18 | 1445/1791 | 80.7 | 48 | 276/4743 | 5.8 |

- Of the 1721 sign differences among matched alternative pairs, 1445 are in the 18 reversal
  cells and 276 are in the 48 same-winner cells. The baseline rate over all taus is 50 of 1045
  (4.8 percent).
- Tau 0.25 meets the "above baseline" criterion, yet all 27 of its differences are in
  same-winner cells (22 of them student_t3, where G's population margin is 1.8 percent). Passing
  the criterion is therefore not evidence of a reversal.
- A high count does not mean a large effect. skewnorm_n8 at tau 0.05 differs in 101 of 101
  matched pairs on a population gap of 0.3 percent of G's cost (its break-even, 0.268, is 0.018
  above the error). student_t3 at tau 0.05 differs in 37 of 99 pairs in a same-winner cell where
  G's margin is 0.2 percent.
- Not every population reversal is resolved at this sample size. For student_t3 and student_t5
  at tau 0.99 the population gap is -0.6 and -0.7 percent, the realised mean gap over 120 series
  has the opposite sign (+0.4 and +0.6 percent), and the pairs differ 56 of 99 and 49 of 98
  times. The realised majority winner over 120 series equals the population winner in 87 of 88
  (member, tau) cells; the exception is student_t5 at tau 0.99.
- Realised mean Delta relative to G's cost against its population value: largest gap 0.014 over
  the 88 cells. Gaussian members: +0.024 to +0.044 realised, +0.026 to +0.038 in population.
- In the engine's own scaling (Delta divided by mean absolute y on the test targets, which ranges
  from 0.70 to 4.45 over series) the effects are small numbers: the largest single-series value
  is 0.034 and (member, tau) means lie between -0.005 and +0.017.
- Baseline noise at the extreme taus (24.2 and 17.9 percent) comes from having about 10 test
  targets beyond those quantiles. 30 seeds give no power to refine any of this; the closed form
  in section 1 is the result and this section only shows that 1000 test targets reproduce it.

By member (differ / matched, primary variant; * marks reversal cells):

| tau | gaussian_b | skewnorm_p2 | skewnorm_p8 | skewnorm_n2 | skewnorm_n8 | student_t5 | student_t3 |
|---|---|---|---|---|---|---|---|
| 0.01 | 23/95 | 45/96 | 86/100* | 92/100* | 94/101* | 86/98* | 79/99* |
| 0.05 | 3/95 | 11/96 | 71/100* | 2/100 | 101/101* | 14/98 | 37/99 |
| 0.09 | 2/95 | 0/96 | 4/100 | 2/100 | 0/101 | 14/98 | 55/99* |
| 0.25 | 0/95 | 0/96 | 0/100 | 2/100 | 2/101 | 1/98 | 22/99 |
| 0.40 | 0/95 | 0/96 | 0/100 | 6/100 | 13/101 | 0/98 | 0/99 |
| 0.50 | 0/95 | 0/96 | 0/100 | 4/100 | 17/101 | 0/98 | 0/99 |
| 0.60 | 0/95 | 0/96 | 0/100 | 1/100 | 14/101 | 0/98 | 0/99 |
| 0.75 | 0/95 | 0/96 | 0/100 | 0/100 | 0/101 | 0/98 | 0/99 |
| 0.91 | 2/95 | 10/96 | 22/100 | 1/100 | 1/101 | 1/98 | 98/99* |
| 0.95 | 3/95 | 25/96 | 65/100* | 2/100 | 99/101* | 3/98 | 96/99* |
| 0.99 | 17/95 | 61/96* | 83/100* | 87/100* | 87/101* | 49/98* | 56/99* |

Other variants (all declared with the primary one, all reported):

| F variant | Population cells where F wins (of 66) | Alternative rate over tau, % | Baseline rate over tau, % | Taus above baseline |
|---|---|---|---|---|
| location error 0.10 | 41 | 42.1 to 73.9 | 16.8 to 46.3 | all except 0.99 |
| location error 0.25 (primary) | 18 | 0.0 to 81.1 | 0.0 to 24.2 | 0.01, 0.05, 0.09, 0.25, 0.91, 0.95, 0.99 |
| location error 0.50 | 3 | 0.0 to 47.5 | 0.0 to 3.2 | 0.01 and 0.99 |
| empirical quantiles from context | no closed form | 53.7 to 64.6 | 35.8 to 57.9 | all except 0.50 and 0.95 |

- With error 0.10, G's population margin on a Gaussian law is 0.5 percent of its cost, so the
  baseline itself differs in 17 to 46 percent of pairs. With 0.50, F wins in 3 cells.
- Empirical variant, Gaussian members: F has the lower realised cost in 29.2 to 45.8 percent of
  series per (member, tau) cell (120 series each), and in 31.2 to 41.2 percent per tau when the
  two Gaussian members are pooled (240 series each); mean Delta relative to G's cost is 0.0004 to
  0.0045 per tau. It is close to a tie, so its sign count is hardly distinguishable from its
  baseline. On non-Gaussian members F has the lower cost in 33.3 to 100 percent of series per
  (member, tau) cell, with mean Delta down to -0.32 of G's cost. This variant is also handed the
  true conditional mean, including the true phi.

Exact ties: 4 of the 10,560 primary units (49 of 42,240 over all variants) have Delta within
1e-12 of zero and go to G.

### 4. Omega-only simulation: failed matches are not only sampling noise

The first version said the failed matches were "sampling noise in Omega, not an effect of the
innovation law". That is false for the sine structure. `omega_only.py` draws 4000 fresh seeds
(1000 to 4999) per (structure, member) with the design's own generator, computes Omega on the
2000-point context and compares each member with the independent Gaussian reference of the same
seed. Nothing is scored.

| Structure | Member | Omega mean | Omega sd | Fails 0.03 match, % | Difference from baseline, points | McNemar p |
|---|---|---|---|---|---|---|
| sine_sd1.0 | gaussian (reference) | 0.2528 | 0.0141 | | | |
| sine_sd1.0 | gaussian_b (baseline) | 0.2527 | 0.0143 | 14.12 | | |
| sine_sd1.0 | skewnorm_p2 | 0.2528 | 0.0146 | 14.07 | -0.05 | 0.97 |
| sine_sd1.0 | skewnorm_p8 | 0.2531 | 0.0150 | 14.60 | +0.48 | 0.53 |
| sine_sd1.0 | skewnorm_n2 | 0.2528 | 0.0143 | 13.43 | -0.70 | 0.33 |
| sine_sd1.0 | skewnorm_n8 | 0.2531 | 0.0148 | 13.55 | -0.57 | 0.43 |
| sine_sd1.0 | student_t5 | 0.2534 | 0.0177 | 19.40 | +5.28 | 1.0e-11 |
| sine_sd1.0 | student_t3 | 0.2609 | 0.0349 | 40.45 | +26.33 | 6.7e-155 |
| ar1_phi0.50 | gaussian (reference) | 0.1023 | 0.0066 | | | |
| ar1_phi0.50 | gaussian_b (baseline) | 0.1024 | 0.0065 | 0.15 | | |
| ar1_phi0.50 | four skew-normal, student_t5 | 0.1023 to 0.1024 | 0.0063 to 0.0065 | 0.10 to 0.22 | -0.05 to +0.07 | 0.61 to 1 |
| ar1_phi0.50 | student_t3 | 0.1012 | 0.0071 | 0.53 | +0.38 | 0.0059 |
| ar1_phi0.80 | gaussian (reference) | 0.2075 | 0.0142 | | | |
| ar1_phi0.80 | gaussian_b (baseline) | 0.2073 | 0.0140 | 12.68 | | |
| ar1_phi0.80 | four skew-normal, student_t5 | 0.2069 to 0.2072 | 0.0139 to 0.0141 | 13.10 to 13.95 | +0.43 to +1.28 | 0.072 to 0.56 |
| ar1_phi0.80 | student_t3 | 0.2061 | 0.0141 | 13.70 | +1.03 | 0.15 |
| ar1_phi0.95 | gaussian (reference) | 0.3875 | 0.0304 | | | |
| ar1_phi0.95 | gaussian_b (baseline) | 0.3884 | 0.0307 | 49.30 | | |
| ar1_phi0.95 | four skew-normal, student_t5 | 0.3884 to 0.3891 | 0.0299 to 0.0307 | 47.90 to 48.77 | -1.40 to -0.52 | 0.18 to 0.63 |
| ar1_phi0.95 | student_t3 | 0.3884 | 0.0308 | 48.35 | -0.95 | 0.39 |

Every (structure, member) row is in `omega_only_by_member.csv`; the grouped AR(1) rows above give
the range over the five members named.

The test and its power. Failure rate of a member against the baseline's is tested with the exact
McNemar test, because both are compared with the same reference series of each seed. 24
(structure, member) comparisons were made; with a Bonferroni threshold of 0.05 / 24 = 0.002 only
student_t5 and student_t3 on the sine pass. At 4000 seeds the test detects, with 80 percent
power, a difference in failure rate of 2.3 points on the sine (2.2 at phi 0.80, 3.1 at phi 0.95,
0.4 at phi 0.50), computed assuming the two failure indicators are independent, which is
cautious because they share a reference. At the design's 30 seeds per structure the same test
needs a difference of 36 points (25 at phi 0.50) for 80 percent power, and its power for the
differences found here is 52 percent for student_t3 on the sine and 4 percent for student_t5.
The first version did not test per member at all: it compared 126 of 720 pooled alternative
failures with 25 of 120 baseline failures (Fisher p 0.37 on those counts), which dilutes one
affected cell among 24. In the 30 design seeds student_t3 failed 7 of 30 on the sine against 2 of
30 for the baseline (Fisher p 0.15).

What the simulation shows:

- Sine structure, student_t3: Omega mean is shifted by +0.0080 (Welch p 8e-41; a shift of 0.0017
  is detectable at 80 percent power), its sd is 2.47 times the Gaussian's (Brown-Forsythe p
  5e-228), and 40.5 percent of pairs fail the match against 14.1 percent for two Gaussian draws.
  student_t5: sd ratio 1.25 (p 1e-35), 19.4 percent fail. skewnorm_p8 and skewnorm_n8: sd ratio
  1.065 and 1.045 (p 0.0003 and 0.005), no detectable change in the failure rate.
- Mechanism. For sine plus noise, Omega falls when the realised noise variance of the window is
  high: the correlation between Omega and the sample innovation variance is -0.35 for the
  Gaussian and -0.44 for student_t3. The population variance is 1 for every member, but
  student_t3 has no fourth moment, so its sample variance over 2000 points has sd 0.48 against
  0.03 for the Gaussian. Equal population variance is not approached at n = 2000 for t with 3
  degrees of freedom. In the 30 design seeds its sample innovation variance ranged from 0.73 to
  2.03 (other members: 0.88 to 1.26).
- AR(1) structures: no material effect of the law on the match rate. Omega of an AR(1) does not
  depend on the innovation scale, and the correlation above is near zero there for the Gaussian.
  A small mean shift for student_t3 is nonetheless detectable at phi 0.50 and 0.80 (-0.0012 and
  -0.0014, Welch p 2e-14 and 1e-5), about one twenty-fifth of the tolerance; at phi 0.50 it
  moves the failure rate from 6 to 21 of 4000 pairs (McNemar p 0.006, not below the Bonferroni
  threshold).
- Two Gaussian draws of the same process fail the 0.03 tolerance 0.15, 12.7, 49.3 and 14.1
  percent of the time (phi 0.50, 0.80, 0.95, sine). Most failures are therefore the tolerance
  being tight relative to Omega's noise, and on the sine the heavy-tailed laws add to that.

Match outcome in the 30 design seeds, failures out of 30 pairs by member (first version reported
only the pooled 126 of 720 and 25 of 120):

| Structure | gaussian_b | skewnorm_p2 | skewnorm_p8 | skewnorm_n2 | skewnorm_n8 | student_t5 | student_t3 |
|---|---|---|---|---|---|---|---|
| ar1_phi0.50 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| ar1_phi0.80 | 3 | 6 | 3 | 3 | 4 | 5 | 2 |
| ar1_phi0.95 | 20 | 15 | 11 | 13 | 12 | 13 | 12 |
| sine_sd1.0 | 2 | 3 | 6 | 4 | 3 | 4 | 7 |
| All (of 120) | 25 | 24 | 20 | 20 | 19 | 22 | 21 |

Negative controls (unchanged): 90 pairs with different structure, all 90 reported as mismatches;
smallest levels difference 0.067. First-difference Omega would not have separated them (largest
gap 0.018), which is why the match is gated on levels.

### 5. The tolerance rationale was statistically wrong (value kept)

`design.py` says 0.03 is "two seed standard deviations" (2 x 0.015). The match compares two
independent draws, whose difference has sd `sqrt(2)` times that of one draw, and 0.015 came from
Part A's single sine rather than from the E3 structures. Measured (Gaussian members, 4000 seeds):

| Structure | sd of one draw | sd of the difference of two draws | 0.03 in difference-sd units | Gaussian pairs failing, % | Tolerance equal to 2 difference-sd |
|---|---|---|---|---|---|
| ar1_phi0.50 | 0.0066 | 0.0093 | 3.2 | 0.15 | 0.019 |
| ar1_phi0.80 | 0.0142 | 0.0199 | 1.5 | 12.7 | 0.040 |
| ar1_phi0.95 | 0.0304 | 0.0430 | 0.7 | 49.3 | 0.086 |
| sine_sd1.0 | 0.0141 | 0.0202 | 1.5 | 14.1 | 0.040 |

The value 0.03 is NOT changed: changing it after seeing results would be post hoc tuning. The
original comment in `design.py` is left as written, with a dated correction beneath it. A better
rule to have registered: a tolerance per structure, in units of the sd of the difference of two
Gaussian draws of that structure (for example 2 units), measured before the dataset is generated.
Because of section 2, no choice of tolerance would have changed the cost results.

### 6. Part A: three Omega values quoted for a "moderately noisy" sine (unchanged, one figure corrected)

Signal `sin(2 pi t / 24) + sd * Gaussian noise`, Omega on levels, 200 seeds per cell; mean (sd):

| noise sd | n = 1000 | n = 2000 |
|---|---|---|
| 0.25 | 0.7184 (0.0073) | 0.7344 (0.0052) |
| 0.50 | 0.4983 (0.0165) | 0.5120 (0.0121) |
| 0.75 | 0.3351 (0.0202) | 0.3443 (0.0153) |
| 1.00 | 0.2354 (0.0198) | 0.2405 (0.0153) |
| 1.25 | 0.1766 (0.0178) | 0.1786 (0.0140) |
| 1.50 | 0.1411 (0.0156) | 0.1410 (0.0123) |
| 2.00 | 0.1048 (0.0118) | 0.1019 (0.0094) |
| 3.00 | 0.0805 (0.0076) | 0.0751 (0.0059) |

- `reproduce_dayson()` returns 0.486436, and a unit test confirms it equals the value computed
  from the signal built by `scripts/validate_omega_synthetics.py` (noise sd 0.5, n = 1000, seed
  42). The sd at that setting is 0.0165; the first version printed 0.017.
- The k3 sanity check is unseeded with noise sd 1.5 at n = 2000: mean 0.1410, sd 0.0123, range
  0.1085 to 0.1702, 54 percent of seeds inside 0.13 to 0.15.
- Omega of about 0.32 needs noise sd of about 0.78 (n = 1000) or 0.80 (n = 2000).
- Not verified: which signal the proposal's 0.32 target was taken from. The source paper was not
  fetched in this session.
- Not repeated in this session, and therefore removed from this version: the five individual
  values printed by the k3 `sanity_check()` and the Omega of the existing k3 pair.

### 7. Tests and planted bugs

`Ran 107 tests in 39.866s`, `OK` (the first version had 53 tests). The repository layout test
(`tests.unit.test_package_layout`) also passes: `Ran 3 tests`, `OK`.

Planted bugs. Each bug was planted alone, the whole E3 suite was run, and the file was restored
in a `finally` block and compared with a backup. The planting was done in a private copy of
`src/`, `scripts/` and `tests/unit/e3/` so that the shared tree never contained a planted bug;
afterwards the copy and the shared tree were confirmed identical (`diff -r`) and the shared tree
identical to the backup taken before planting. Interpreter: the team's (numpy 1.26.4, pandas
2.2.3, scipy 1.11.4). Result: 26 planted, 26 caught, 0 survived.

| | Planted bug | Caught | By (first test listed; number of failing checks) |
|---|---|---|---|
| A | Bootstrap resamples single pairs instead of groups | yes | `test_bootstrap_resamples_whole_groups_not_single_pairs` (1) |
| B | Baseline pair labelled as alternative in `match_table` | yes | `test_pair_kind_separates_the_baseline_from_the_alternatives` (1) |
| C | Random stream ignores the structure | yes | `test_seed_structure_and_member_each_change_the_stream` (1) |
| D | Bootstrap interval includes failed matches | yes | `test_failed_matches_change_neither_the_rate_nor_any_interval` (1) |
| E | Population F cost uses the Gaussian quantile | yes | `test_every_row_of_the_population_table_satisfies_the_identity` and 4 other tests (65) |
| F | `delta_rel` divided by J_F instead of J_G | yes | `test_delta_rel_is_relative_to_the_cost_of_G` (1) |
| G | Persisted skewness and kurtosis computed on the whole series | yes | `test_no_persisted_statistic_uses_the_test_period` (2) |
| H | Part A signal amplitude doubled | yes | `test_signal_is_a_unit_amplitude_period_24_sine_plus_scaled_noise` and 2 other tests (3) |
| I | AR(1) burn-in discarded (series starts from a zero state) | yes | `test_ar1_series_includes_the_burn_in_history`, `test_ar1_starts_in_the_stationary_distribution` (3) |
| J | Sign difference taken from the sign of Delta, wrong on exact ties | yes | `test_sign_differs_follows_the_engine_winner_on_exact_ties` (1) |
| K | `reproduce_abdullah` run at n = 1000 | yes | `test_abdullah_setting_is_n_2000_noise_sd_1p5` (1) |
| L | Empirical F quantiles inflated by 1.3 | yes | `test_costs_and_winners_equal_the_engine_for_every_variant` and 3 other tests (4) |
| M | Match tolerance applied to the difference divided by 1.5 | yes | `test_check_pair_boundary_and_length_rule` and 1 other test (2) |
| N | Round-trip CSV parsing removed | yes | `test_saved_series_table_is_read_back_bit_for_bit`, `test_awkward_floats_survive_the_round_trip` (6) |
| O | skewnorm_n8 generated with shape -2 | yes | `test_mirror_members_have_mirrored_break_evens` and 4 other tests (22) |
| P | Omega-only: failure judged against twice the tolerance | yes | `test_failure_uses_the_registered_tolerance_and_the_same_seed_reference` and 2 other tests (4) |
| Q | Omega-only: Omega on the full series instead of the context | yes | `test_each_draw_is_the_design_generator_and_the_team_omega_on_the_context` (8) |
| R | Joint bootstrap resamples the baseline arm separately | yes | `test_the_two_rates_come_from_the_same_resample` (1) |
| S | Break-even: quantile-gap branch taken on the wrong side | yes | `test_break_even_is_where_the_two_expected_costs_cross` and 2 other tests (8) |
| T | Innovation-only F ignores the location error | yes | `test_hand_computed_cost_from_innovations` and 3 other tests (4) |
| U | F-win count treats a tie as a win for F | yes | `test_hand_built_count` and 1 other test (2) |
| V | Reversal status inverted | yes | `test_cell_status_compares_each_member_with_the_reference` and 2 other tests (3) |
| W | All-equal fallback uses pairs instead of groups | yes | `test_all_equal_outcomes_get_a_wilson_interval_not_a_zero_width_one` and 1 other test (2) |
| X | No-series simulation: member shares the reference stream | yes | `test_rates_follow_the_population_winners_without_any_series` (1) |
| Y | McNemar discordant counts swapped with concordant | yes | `test_failure_rates_and_discordant_counts_on_a_hand_built_example` (1) |
| Z | Newcombe lower bound ignores the baseline arm | yes | `test_one_empty_arm_widens_the_difference_interval` and 1 other test (2) |

A to O are the reviewer's fifteen (the ten named in the fix list as survivors are A, B, D, E, F,
G, H, I, K, N). P to Z are planted in the code added in this version. Two things went wrong on
the way and are recorded here rather than hidden. First, bug J survived the first replant pass:
it differs from the correct rule only on exact ties, and no test had a tie. A test with a tie
was added and J is now caught. Second, the first replant pass was made with the wrong
interpreter (the system Python with numpy 2.4.4, reached through a resolved symlink) and inside
the shared tree; it was stopped, the tree was confirmed identical to its backup, and the pass
reported above was rerun from scratch with the team interpreter in a private copy.

Tests repaired or removed (each could not fail, or was fragile):

| Old test | Problem | Now |
|---|---|---|
| `test_seed_structure_and_member_each_change_the_stream` | Third assertion compared sine sd 1.0 with sd 2.0, which differ whatever the stream. | Compares the recovered innovations of two AR(1) structures and the first draw of the two streams. |
| `test_sweep_is_seeded` | Passed the same seed twice. | Replaced by a test of the signal itself (unit amplitude, period 24, noise equal to sd times the generator's stream) and by a sweep cell checked against a hand computation over seeds 0 to n - 1. |
| `test_tolerance_is_the_registered_value` | Echoed a constant. | Removed. The behaviour at 0.329 and 0.331 is pinned by `test_check_pair_boundary_and_length_rule`. |
| `test_G_is_not_the_true_law_for_non_gaussian_members` | Tested scipy, not E3. | Removed. |
| `test_existing_k3_generator_is_still_in_place` | Existence only; duplicated the repository layout test. | Removed. |
| `test_matched_design_pairs_pass_in_low_persistence_groups` | Asserted that all sine pairs match, which is false 14 to 40 percent of the time in population and held only on seeds 0 and 1. | `test_status_is_the_tolerance_rule_applied_to_the_persisted_omegas`: the status must equal the stated rule; no claim that pairs match except at phi 0.50. |
| `test_sample_innovation_variances_agree_within_a_group` | Excluded student_t3, the one member where it matters, and used a fixed 0.2. | Per-series bound of 5 sampling standard deviations for members with finite kurtosis, and a separate test that the student_t3 sample variance does not settle at n = 2000. |
| `test_realised_costs_track_the_population_values` | 15 percent tolerance on three taus let the wrong population formula through. | Population table checked against an independent integration (1e-7), against the finite-interval identity on all 264 rows (1e-8), and, for two members, against 400,000 i.i.d. draws within 5 standard errors. |

### 8. Reproducibility

The rebuild from the saved series, with engine scoring, reproduced the 12 tables of the original
run that are unchanged in definition byte for byte (`cmp`): `series_table`,
`series_summary_by_member`, `match_pairs`, `match_groups`, `match_summary`, `unit_costs`,
`sign_pairs`, `member_delta_summary`, `member_delta_summary_by_structure`, `expected_delta`,
`part_a_omega_sweep` (all `.csv`) and `part_a_summary.json`. The three `sign_summary_*` tables
differ from the original only in the rows where every matched pair has the same outcome (the
bootstrap columns are now empty there and `interval_*` columns were added); their bootstrap
values in all other rows are identical. The engine rebuild was run twice in this session (475 s
and, on a busy machine, 801 s) and a `--reuse-units` rebuild once; all three give byte-identical
CSV tables. A second Omega-only run of the first 200 seeds reproduced the corresponding 6400
rows of `omega_only_draws.csv` exactly. The original directory `artifacts/runs/e3_matched_omega/`
was only read.

### Corrections to the first version

1. False claim removed: failed matches are not only sampling noise (section 4).
2. The headline (sign of Delta differs within Omega-matched pairs) is reframed: it follows from
   the definitions and cannot fail (status paragraph, sections 1 and 2).
3. The location error is shown as the free parameter it is (section 1).
4. Sign counts are split by population-winner status with effect sizes; the criterion is
   defined; one joint interval replaces two separate ones; "0.0 to 0.0" is replaced (section 3).
5. Tolerance rationale corrected, value kept (section 5).
6. The sentence that G is Gaussian "because the team's classical adapters emit Gaussian
   quantiles" is withdrawn. It invited reading G as a stand-in for the classical family. G is
   Gaussian because that makes it the exact oracle on the reference member and wrong in shape on
   the others, which is the contrast the mechanism needs.
7. The closing sentence that Omega "cannot distinguish a Gaussian series from a skewed or
   heavy-tailed one with the same second-order structure" needs a caveat: it holds for the
   population spectrum, but in finite samples the distribution of Omega does depend on the law
   (sine structure, student_t3: mean +0.008, sd 2.5 times larger).
8. Minor: Part A sd printed as 0.017 is 0.0165; the empirical-variant F share is now stated per
   (member, tau) and pooled; the range of the engine's scale was misstated (it is 0.70 to 4.45
   over series; the first version gave 0.90 to 2.55).

Limitations not stated in the first version: structure and Omega cancel out of the costs; every
scored structure has sigma = 1; F and the empirical F are both handed the true conditional mean,
including the true phi; the reference member is the one law on which G is exactly right; 30 seeds
give no power for the claim about failed matches; horizon 1 only.

## Design choices needing sign-off

1. Present the closed-form table as the result and the simulation as a noise check. Alternative:
   keep the sign-difference table as the headline, as in the first version. Reason: the costs
   depend only on the innovation law, tau and the error size, so the simulation can only
   reproduce the closed form plus noise.
2. Keep the Omega tolerance at 0.03 although its rationale was wrong. Alternative: move to a
   per-structure tolerance in sd units now. Reason: changing it after seeing results is post hoc
   tuning, and it cannot change any cost result (section 2). The better rule is recorded for a
   future pre-registration.
3. Keep the forecaster definitions, members, seeds and lengths. Alternative: redefine F (for
   example a scale error, or an estimated mean) so that the result depends on the series. Reason:
   redefinition after the fact would be tuning; a design that can fail needs real forecasts and
   belongs to the cache owner's pipeline (Open, item 1).
4. Criterion "above baseline" = lower end of the joint 95 percent interval for the rate
   difference above zero. Alternative: a threshold on the effect size. Reason: it is the usual
   reading of the first version's undefined phrase; the note states that it is not evidence of a
   reversal and reports the split by population-winner status next to it.
5. Reversal status is defined by the population winner only, with no minimum gap. Alternative:
   require a gap of at least some percent of G's cost. Reason: any threshold chosen now would be
   chosen after seeing the table; effect sizes are reported instead (8 of the 18 reversal cells
   have a gap below 1 percent).
6. All-equal counts (0 of n): Wilson interval with the number of (structure, seed) groups as the
   sample size. Alternative: Wilson on the number of pairs (0 of 594 gives 0.0 to 0.6 percent
   instead of 0.0 to 3.1). Reason: pairs of a group share a reference series, so groups are the
   independent units. When one arm is all-equal the difference interval treats the arms as
   independent (Newcombe), which is wider than a joint interval would be.
7. Omega-only simulation: 4000 seeds, seeds 1000 to 4999, the design's generator at full design
   length, McNemar test against the baseline member, Bonferroni over 24 comparisons. Alternative:
   an unpaired two-proportion test, or fewer seeds. Reason: the members share the reference of
   each seed, and 4000 seeds detect a 2.3 point difference on the sine.
8. The engine's family labels are used as slots (G in `classical`, F in `tsfm`) with
   `val_fraction = 0`. They are slot labels only. Alternative: neutral labels in the engine,
   which is outside E3's paths.
9. Carried over unchanged from the first version and still needing sign-off: 2000 context points
   and 1000 test targets; independent random streams per member rather than common random
   numbers; population-exact standardisation of the innovations rather than sample
   standardisation; student_t3 included although its kurtosis is infinite; the quantile-based
   tail summary; horizon 1 only.

## Open

1. (Aadrika, cache owner) A version of E3 that could fail needs forecasters that estimate the
   conditional mean and the distribution from data: the team's real forecast pipeline run on
   these series. Inputs: `artifacts/runs/e3_matched_omega/series_values.parquet` and
   `series_table.csv`; decision origin at position 2000. Until then nothing here says anything
   about real classical models or real TSFMs, and the "flip" is not an empirical finding.
2. (Project lead) `docs/project-overview.md` makes a later experiment conditional on "if E3
   fires" (E5 paragraph) and has a risk-table rule for "matched-Omega pairs do not flip". E3 as
   built always fires, so it must not be used as that gate. The wording there needs a decision;
   that file is outside E3's paths and was not edited.
3. (Team) If the matched-Omega design is used with real forecasts: register a per-structure
   tolerance in sd units (section 5), and decide what to do about heavy-tailed members on the
   sine, where 40 percent of student_t3 pairs fail the present tolerance.
4. (Team) Levels Omega has large sampling noise for persistent series at n = 2000 (phi = 0.95:
   sd 0.030 for one draw). Any Omega threshold rule inherits this. First-difference Omega is more
   stable but did not separate the negative controls.
5. (Owners of k3 and of `pyproject.toml`) `analysis/k3/matched_omega_pair.py` is untouched; the
   entry point `decision-aware-matched-omega` still points at it. Decide whether to retire it.
6. (Dayson) `scripts/validate_omega_synthetics.py` uses noise sd 0.5 for the "moderately noisy"
   signal and gets 0.486 against a 0.32 target; a noise sd near 0.8 gives about 0.32. The origin
   of the 0.32 target is unverified.
7. (Abdullah) The k3 sanity check is unseeded and uses noise sd 1.5, which gives about 0.14.
8. (Engine owner) With horizon 1 the engine scoring of the 120 groups takes most of the 8
   minutes of a full rebuild (475 s in total on a quiet machine). No change was made to the
   engine.
9. Not examined: horizons above 1, CRPS, the seven selectors applied to these series, a scale
   error as F's imperfection, other series lengths, a location error whose sign or size varies
   with tau. The date on which the 0.25 error was chosen cannot be established.
