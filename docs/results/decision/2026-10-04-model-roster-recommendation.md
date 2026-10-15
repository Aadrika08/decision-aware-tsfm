# Model-roster recommendation (Veer, task 2)

**Status: a proposal awaiting team and mentor approval. It is not a decision.** Nothing in the
roster, the cache or the code was changed. Whatever is approved must be recorded in
`docs/decisions/` by that area's owner (section 8). This note does not create that record.

Evidence type: reading of the source paper (arXiv:2511.08884v1), the GIFT-Eval benchmark paper
(arXiv:2410.10393v2), the GIFT-Eval leaderboard files (`results/<model>/config.json`) and
evaluation notebook fetched on 2026-10-04, four model cards, and this repository. No model was
run. "K2" numbers come from Veer's reanalysis of published leaderboard results on 20 datasets
(`artifacts/runs/k2-2026-10-04/omega_levels/`), not from the paper.

## 1. The short version

1. **One question comes first (section 2).** The paper reports its effect only for models whose
   fixed weights never saw the benchmark data [zero-shot]. We have not established that our only
   pilot foundation model, the public `Salesforce/moirai-1.1-R-small`, is such a model. The
   benchmark's own paper says the original Moirai had seen some of its test data [leakage].
2. Proposed, subject to that answer: the main "TSFM family" contains zero-shot models only.
   Chronos-Bolt and Tiny Time Mixers (TTM) r2 are reported, if at all, as a separate group.
3. The proposed pilot is the four models whose adapters are qualified under decision record
   [ADR] 0005: Moirai-1.1-R-small, SeasonalNaive, AutoETS, AutoARIMA. Qualified is not approved;
   the roster needs its own record.
4. Every further candidate, including TimesFM 2.5, must first show it produces forecasts at our
   11 probability levels [quantiles; the levels are called tau]. Until a second zero-shot model
   passes, results are "Moirai-1.1-R-small against classical", not "TSFMs against classical".

## 2. Open item 1: is our Moirai "zero-shot" in the paper's sense?

**What is known, and from where.**

- Source paper, Section 5: zero-shot models are "TSFMs with no data leakage in their published
  weights and thus not 're-pretrained'". Models retrained on a leak-free dataset are called
  "pretrained". Appendix A.1 lists `Moirai_small` as zero-shot.
- GIFT-Eval paper, Section 4, paragraph "Models": "pre-training datasets of TimesFM, Chronos, and
  Moirai exhibit partial data leakage issues for GIFT-Eval. Since Moirai provides pretraining
  code, here we pretrain a series of Moirai models using GIFT-Eval's pretraining split".
  Appendix F.2: "we pre-trained a new series of Moirai models on our GIFT-Eval pre-training data
  and report the results from these models in the main paper", and "We call the original model
  Moirai-Leakage". Its Table 23 names the datasets that Moirai's original training collection
  (LOTSA) shares with GIFT-Eval: hierarchical_sales, loop_seattle, m_dense, restaurant and
  sz_taxi. Four of these five are among the 20 K2 datasets (all but restaurant).
- Leaderboard, `results/Moirai_small/config.json`: type "zero-shot", `testdata_leakage` "No".
  Its `model_link` points to `Salesforce/moirai-1.1-R-large`, the large saved set of weights
  [checkpoint], not a small one. The
  notebook named in its `code_link` (`notebooks/moirai.ipynb` in the gift-eval repository) loads
  `Salesforce/moirai-1.0-R-small`.
- Model card of `Salesforce/moirai-1.1-R-small`: "This is new updated version of Moirai-1.0-R".
  It does not list training data.
- We run `Salesforce/moirai-1.1-R-small` at the revision pinned in ADR 0005.

So three sources suggest three different checkpoints behind the entry (a retrained model, 1.1-R
and 1.0-R), and none says what the public 1.1-R-small was trained on. Our only pilot TSFM may
belong with the leakage group.

**What must be checked.** (a) Which checkpoint produced the `Moirai_small` results that the
paper used. (b) What data the public 1.1-R-small was trained on, and whether it includes the
Table 23 datasets or other GIFT-Eval test data. (c) Whether the retrained Moirai weights are
published.

**Who could answer.** The GIFT-Eval maintainers at Salesforce AI Research for (a) and (c), for
example through an issue on github.com/SalesforceAIResearch/gift-eval. The Moirai maintainers
(same organisation) for (b). The source paper's authors for which leaderboard snapshot they
analysed. The mentor should decide who writes.

**How the recommendation changes.** I have not assumed any of these answers.

| Answer | Consequence |
|---|---|
| The entry used public 1.1-R weights, and those were not trained on GIFT-Eval test data | The label `tsfm / zero-shot` holds and section 6 stands as written. |
| The entry used the retrained model, and the public 1.1-R-small was trained on overlapping data | Our pilot model is not in the paper's zero-shot group, and the project has no qualified zero-shot TSFM. The pilot still tests the pipeline but not the paper's claim. A zero-shot model (TimesFM 2.5, or the retrained Moirai weights if published) must qualify before the main comparison. |
| The entry used public weights, but those weights saw some GIFT-Eval test data | The model matches the paper's group by name, but the comparison is unfair on the overlapping datasets. Keep it, label it `leakage possible`, and report with and without those datasets. |
| No answer before the pilot | Label it `zero-shot per leaderboard, checkpoint unverified`, report with and without the Table 23 datasets, and do not describe the pilot as a test of the zero-shot claim. |

## 3. What the source paper claims, checked against the paper

| Project belief | Outcome | What the paper says, and where |
|---|---|---|
| The effect is for zero-shot only: -0.556 against deep learning, -0.234 against statistical | CONFIRMED | Paragraph "Model-Family-Specific Patterns" (after the 5.1 heading): "Spearman correlations ranging from -0.234 for statistical to -0.556 for deep learning"; "this advantage is specific to zero-shot models". No p-values; the value against pretrained is not stated. |
| Four groups taken from GIFT-Eval | CONFIRMED as wording; see section 2 | Section 5: "categorized as statistical, deep-learning, pretrained, or zero-shot following GIFT-Eval's taxonomy". Members: Appendix A.1. |
| One Omega threshold picks the family; series cut at 4,096 steps and unreliable below 1,000 | CONFIRMED; the rule has three zones | Section 5.2: zero-shot "optimal" for "Ω > 0.5", "no clear advantage" for "Ω < 0.4", "less clear-cut" between. Appendix A.2: "truncated the incoming data at 4096 steps". Section 5.3: "we recommend T > 1000 timesteps". |
| A family's score is the mean over its models | CONFIRMED for Figure 4 only | Section 5 body text: "Each plotted point represents the averaged sMAPE for one of the 4 given model classes". Not stated for the correlations given with Figure 6. No correlation values are given for Figure 7. |

The paper's analysis uses sMAPE (a percentage error of the single-number forecast), 51 models
and 28 datasets. Its weakest reported value, -0.234, is for zero-shot against statistical, the
comparison this project is about.

**The paper and the leaderboard describe "pretrained" differently.** The paper says pretrained
models were retrained on a leak-free dataset. On the leaderboard today, 10 of the paper's 11
pretrained models have `testdata_leakage: "Yes"` (Moirai2 is the exception) and link to ordinary
public checkpoints such as `amazon/chronos-bolt-base`; all 18 of the paper's zero-shot models
still present have `"No"`. On that evidence "pretrained" means "public weights that may have
seen benchmark data".

`GROUPS`, now in `analysis/k2/families.py`, matches Appendix A.1 (19, 11, 9 and 5
models). `granite-flowstate-r1` is absent from the saved results, so the K2 zero-shot family
has 18 members.

## 4. The comparability risk

A fair comparison needs both sides to have had the same kind of access to the data. Our
candidates are of three kinds.

- **Fitted per series.** AutoARIMA, AutoETS and SeasonalNaive estimate their settings from the
  history of the series being forecast, afresh at each forecast start point [origin].
- **Frozen, never trained on the benchmark** [zero-shot]. One fixed set of weights, no fitting.
- **Frozen, but flagged for leakage** [the leaderboard's "pretrained" group]. The model may
  already have seen the answers.

Three consequences. Numbers are rank correlations between Omega and the relative error gap
[Spearman rho], with the range a resampling check gives [95% bootstrap interval].

1. **Mixing the two frozen kinds in one family is a risk.** The paper reports the Omega
   relationship for zero-shot models only, and our development datasets come from GIFT-Eval,
   where a leakage-flagged model can look better than on new data. Whether the two groups
   respond to Omega differently is not settled by K2. Zero-shot against pretrained (sMAPE,
   n = 20) is -0.514, interval [-0.800, -0.054], when each family is summarised by its mean, and
   -0.337 at the finer configuration level (n = 71). With the median model or the best model it
   is +0.129 and +0.009.
2. **Classical and TSFM models are not on the same footing; this can only be labelled.**
   `docs/project-overview.md` says "every forecaster is frozen and zero-shot", but the classical
   family is fitted per series.
3. **How a family is summarised changes the answer.** K2, sMAPE, n = 20, from
   `correlations_dataset_level.csv`:

   | Zero-shot against | Family mean | Median model | Best model |
   |---|---|---|---|
   | deep learning | -0.417 [-0.793, +0.091] | -0.105 | +0.009 |
   | statistical | -0.123 [-0.603, +0.397] | -0.179 | -0.074 |

   Here each model is first averaged over a dataset's configurations. The older
   `within_family_sensitivity.csv` took median and best over (model, configuration) rows, giving
   +0.045 and -0.063 against deep learning. Against statistical, our own comparison, the
   interval includes zero and the value on MAE is +0.251. The same check on MSE is in
   `docs/results/k2/2026-10-04-k2-hardening.md`. One student reanalysis on 20 datasets is a
   warning, not a finding about the paper.

## 5. Each candidate

Paper group is from Appendix A.1, leaderboard from `config.json`, status from
`docs/runbooks/cache-consumer-contract.md` and the 2026-09-21 qualification note in
`docs/results/cache/`.

| Model | Paper group | Leaderboard type, leakage | Status in this repo | Proposed label |
|---|---|---|---|---|
| Moirai-1.1-R-small | zero-shot (as `Moirai_small`) | zero-shot, No; checkpoint not established (section 2) | Passed, 18/18 synthetic attempts, 11 quantiles from 4,096 samples | Until section 2 is answered: `tsfm / zero-shot per leaderboard, checkpoint unverified` |
| TimesFM 2.5 | zero-shot | zero-shot, No; its card lists GiftEvalPretrain among its training data | "No adapter-contract pass in our evidence". Card: output is "mean, then 10th to 90th quantiles" | `tsfm / zero-shot`, only after it passes |
| Chronos-Bolt-base, -small | pretrained | pretrained, Yes | "Not cleared for the full quantile grid" | `tsfm-pretrained (leakage flagged)`, secondary only |
| TTM r2 | pretrained (as `TTM-R2-Pretrained`) | pretrained, Yes | "No adapter-contract pass in our evidence". Card: r2 releases "support point forecasting use-cases"; quantiles are not mentioned | `tsfm-pretrained (leakage flagged)`, secondary only |
| AutoARIMA, AutoETS, SeasonalNaive | statistical | statistical, No | Each passed 18/18 (Gaussian quantiles; AutoETS from 4,096 simulated paths) | `classical / fitted per series` |

Chronos-Bolt is not cleared because ADR 0002 records that its "current quantile API clamps
requests outside its trained 0.1–0.9 range to its endpoint predictions", and the consumer
contract says "Do not fabricate tails or silently substitute repeated endpoints." Six of our
eleven tau values lie outside that range. The TimesFM 2.5 card suggests the same question applies to it. That TTM
r2 gives single-number forecasts only is my inference from its card. Neither was tested.

## 6. Recommendation (proposal)

- **Pilot:** the four models with adapters qualified under ADR 0005. That ADR is accepted for
  synthetic-input qualification and says it does not freeze "the final model roster".
- **Main TSFM family: zero-shot models only**, once section 2 says which models those are.
- **Expansion candidate:** TimesFM 2.5, if it passes the same 11-quantile qualification with
  tails the model itself produces. A larger Moirai-1.1-R checkpoint could reuse the qualified
  adapter, but it carries the same section 2 question and is not a second design.
- **Outside the main comparison:** Chronos-Bolt-base and -small, and TTM r2. If run later, a
  separately labelled group at tau values they natively cover.
- **Labels in every results table:** keep the cache's `family` column and add the access label
  from section 5 and the quantile source (`Gaussian`, `simulated paths`, `sampled`, `native`).
- **What is lost:** the TSFM side may have one or two models, so the paper's claim is tested
  narrowly. Nothing can be said about Chronos, although the paper's Section 5.2 names "Moirai or
  Chronos" as zero-shot examples (its Appendix A.1 lists every Chronos entry as pretrained).
- **The single-TSFM pilot.** The within-family rule picks the member with the lowest mean
  pinball loss at tau = 0.5 on development data. With one TSFM there is no choice, while
  classical picks the best of three, a small advantage. So do not freeze a TSFM representative
  from the pilot, and report per-model results beside any family result.

## 7. Alternatives considered

- **All five TSFMs as one family.** Mixes zero-shot with leakage-flagged models; four of the
  five have not passed the 11-quantile check.
- **Two TSFM groups side by side in the main analysis.** Chronos-Bolt is recorded as not cleared
  for the full grid and TTM r2 has no pass in our evidence, so the groups would likely be scored
  on different tau sets.
- **Build tails for Chronos-Bolt or TTM by extrapolation.** The contract forbids fabricated
  tails.
- **Compare at tau = 0.5 only.** Removes the asymmetric-cost question the project tests.

## 8. Decisions needed before the expensive run

Each needs an issue and a numbered file in `docs/decisions/`, whose README lists "the model
roster" and "the within-family representative rule" among items that must be recorded.

1. How Moirai-1.1-R-small is labelled, and what is done if section 2 stays unanswered.
2. Whether the main TSFM family is zero-shot only, which second zero-shot model is tried first,
   and whether Chronos-Bolt and TTM r2 are dropped or kept as a secondary group.
3. The within-family rule when a family has one member, and which summaries are reported.
4. The static Omega rule: the paper gives 0.5 and 0.4 with an undecided middle zone, so the team
   must choose how to treat it and replace the placeholder `FIXTURE_STATIC_THRESHOLD = 0.45`
   in `analysis/decision/fixture.py`, which is not a number the paper states.
5. The wording in `docs/project-overview.md` ("frozen and zero-shot"), once 1 and 2 are settled.

## 9. Other open items and things I could not verify

1. Section 2 is the first open item.
2. Whether any candidate TSFM was trained on M5 or PJM, our held-out datasets. Not checked; to
   be read from each model's published training-data list before the held-out run.
3. Which description of "pretrained" is right model by model (section 3). The leaderboard has
   140 model folders today against 51 models in the paper.
4. Whether TimesFM 2.5 can produce quantiles outside 0.1 to 0.9, and whether TTM r2 can produce
   any. Model cards only.
5. `k2a-k2b-findings.md` says the MSE result against deep learning (-0.556) "matches the
   published figure exactly". The paper's -0.556 is from its sMAPE analysis, where K2 gives
   -0.417. The K2 hardening note (path in section 4) also records that one model,
   `Toto_Open_Base_1.0`, dominates the zero-shot family's mean MSE on many datasets. This is
   not an exact reproduction.
6. Moirai-1.1-R-small is licensed cc-by-nc-4.0 (non-commercial) on its model card.
