# K2a and K2b: Replication of the spectral-predictability relationship

Feasibility sprint, decision-aware model-family selection.
Owner: Veer. Inputs: Dayson's corrected Omega manifest, published GIFT-Eval results.
No forecasting model was run at any point in this gate.

## Summary

K2a passes with qualifications. The published relationship reproduces in direction
across all three model-family comparisons on both of the source paper's own metrics.
The strongest match is on MSE against deep-learning baselines, where the measured
correlation equals the published value to three decimals.

K2b shows the relationship weakening under the decision metric. The sign is preserved
against deep-learning and pretrained baselines but the correlations lose significance,
and against statistical baselines the sign reverses.

The most consequential finding is not the replication itself. The relationship exists
only when model families are summarised by their mean. Summarised by median, or by the
best member, it disappears entirely.

## Method

The source paper (arXiv:2511.08884) measures relative family advantage as

    Delta = 100 * (score(A) - score(B)) / score(A)

where negative Delta means family A is better, and correlates Delta against Omega
using Spearman rank correlation. It uses the GIFT-Eval taxonomy of four model
families: statistical, deep-learning, pretrained, and zero-shot. Zero-shot models are
those whose published weights carry no leakage against the benchmark. Pretrained
models had to be re-pretrained on a leakage-free corpus before evaluation. The paper's
result is specific to the zero-shot family; it reports no consistent Omega dependence
for pretrained models.

This replication uses the paper's exact family membership lists, scores each family by
its mean across models on a dataset, and aggregates Omega to one value per base
dataset by averaging across that dataset's series and channels. Published per-model
results were taken from the GIFT-Eval leaderboard repository: 129 models across 97
dataset configurations, 12,416 rows.

## K2a results

Spearman correlation between Omega and Delta. Negative indicates the zero-shot
advantage grows as Omega rises, which is the published direction.

| Metric | vs deep-learning | vs statistical | vs pretrained |
|---|---|---|---|
| sMAPE | -0.417 (p=0.068) | -0.123 (p=0.605) | -0.514 (p=0.020) |
| MSE | -0.556 (p=0.011) | -0.398 (p=0.082) | -0.520 (p=0.019) |

Published values: -0.556 against deep-learning, -0.234 against statistical, with the
pretrained comparison falling between the two.

All six correlations carry the published sign. Three reach p < 0.05 and two more fall
below p = 0.10. The MSE comparison against deep-learning matches the published figure
exactly.

**K2a verdict: pass.** The direction reproduces consistently. Magnitudes are weaker
than published on sMAPE and equal to published on MSE.

## K2b results

Mean pinball loss at tau = 0.5 equals half of mean absolute error, so the decision
metric at the median is available directly from the published MAE column. No forecasts
and no quantile cache were required. CRPS is also published and is included as the
generic probabilistic comparator.

| Metric | vs deep-learning | vs statistical | vs pretrained |
|---|---|---|---|
| sMAPE (K2a reference) | -0.417 | -0.123 | -0.514 |
| MAE, equivalently pinball at tau=0.5 | -0.248 (p=0.292) | +0.251 (p=0.286) | -0.405 (p=0.077) |
| CRPS | -0.280 (p=0.232) | +0.376 (p=0.102) | -0.415 (p=0.069) |

The relationship survives the move to the decision metric against deep-learning and
pretrained baselines, retaining its sign while losing roughly 40% of its magnitude and
falling out of significance. Against statistical baselines the sign reverses.

**K2b verdict: the relationship weakens materially under the decision metric.** This is
consistent with the project's premise that the scoring metric carries information about
family choice, though at n = 20 the change is not itself statistically distinguishable
from noise.

## The aggregation-sensitivity finding

Repeating the sMAPE analysis with different within-family summaries:

| Family summary | zero-shot vs deep-learning |
|---|---|
| Mean across models (the paper's choice) | -0.417 (p=0.068) |
| Best model in family | -0.063 (p=0.791) |
| Median model in family | +0.045 (p=0.851) |

A mean is sensitive to its worst members. The disappearance of the effect under median
and best-of-family is consistent with the relationship being driven by a small number
of poorly performing models in the comparison family rather than by a property of the
families themselves.

This matters for deployment, which is the setting the published rule addresses. A
practitioner selects one model, not the average of nine. Under that reading the rule has
no measured effect at any Omega.

## Known divergences from the published method

1. **Twenty datasets rather than twenty-eight.** Configurations below the 1000-timestep
   floor were excluded per the source paper's own recommendation, and a further seven
   subsets fell below the floor after test-window trimming.

2. **Test-window trimming.** The source paper computed Omega over full datasets because
   train-test splits were not public at the time of writing. This replication computes
   Omega on training regions only, replicating GIFT-Eval's own windows logic. This is the
   more defensible procedure but is a divergence from the published method.

3. **No 4096-step truncation.** The source paper truncates each data stream at 4096
   steps. This is not yet implemented. Omega is length-dependent: the same random walk
   scores 0.695 at 1,000 points and 0.840 at 65,536, while an AR(1) process moves in the
   opposite direction. Across the twenty datasets used here, Omega and median series
   length correlate at -0.058 (p=0.808), so the confound is not observably active, but
   the divergence should be closed before publication.

4. **Model roster drift.** The published result used the leaderboard as it stood at
   submission. The current leaderboard carries 129 models including entries that
   postdate the paper. Family membership follows the paper's appendix lists, so
   post-paper models are excluded.

5. **Omega aggregation.** One Omega per dataset conceals substantial within-dataset
   spread. In bitbrains_fast_storage the per-series values span nearly the full range.
   Aggregating by median rather than mean does not change the conclusion
   (-0.427 versus -0.417).

## Implications for the project design

The project defines its two candidate families as TSFM against classical, where
classical means AutoARIMA, AutoETS and SeasonalNaive. That is the statistical group,
and it is the comparison with the weakest measured relationship on both metrics
(-0.123 on sMAPE, -0.398 on MSE) and the one whose sign reverses under the decision
metric.

The planned model roster also mixes the two large-model families. TimesFM 2.5 and
Moirai-small are zero-shot; Chronos-Bolt and Tiny Time Mixers r2 are pretrained. The
source paper reports the Omega relationship only for zero-shot models. Mixing them
within one family is likely to attenuate any boundary the project attempts to locate.

Both points should be settled before the roster is frozen.

## Files

| File | Contents |
|---|---|
| `deliverable_1_joined_table.csv` | Dataset, Omega on levels and differences, within-dataset spread, series count, median length, model family, sMAPE, MSE, MAE, pinball at tau=0.5, CRPS |
| `deliverable_2_replication_figure.png` | Omega against Delta for both comparisons on both metrics, with published thresholds at 0.4 and 0.5 marked |
| `deliverable_4_summary_stats.csv` | All twelve correlations with sample sizes and p-values |
| `src/decision_aware_tsfm/analysis/k2/replication.py` | Full analysis, self-contained. `decision-aware-run-k2 --fetch-results` downloads the published results; `--manifest` points at the Omega manifest; `--figure` writes the plot. Reproduces every number above. |
| `artifacts/reference/k2/gift_results.csv` | Concatenated published results, 129 models, 97 configurations |
