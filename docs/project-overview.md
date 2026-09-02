# Project overview

The 10-minute version. Read this instead of the two PDFs when you are getting oriented;
read the PDFs when you need the detail behind a specific number.

Sources: [`implementation-plan.pdf`](implementation-plan.pdf) (execution),
[`proposal.pdf`](proposal.pdf) (research argument).

---

## The chain

```
research question
      ↓
core hypothesis
      ↓
Omega  (the cheap descriptor)
      ↓
decision-aware selector
      ↓
feasibility gates   ← we are here
      ↓
full experiment, if the gates pass
```

---

## Research question

Given a new time series and a **known asymmetric downstream cost**, can cheap pre-forecast
properties predict whether a **classical** model family or a **TSFM** family will minimise
downstream decision cost — without running expensive validation across both families?

And when the cheap descriptor is insufficient, can the selector detect that and defer to
full validation?

## Why the question is not already answered

A 2025 paper introduced **spectral predictability (Omega)**, a fast statistic derived from
the power spectrum, and showed across GIFT-Eval datasets that zero-shot TSFMs gain relative
advantage in high-Omega regimes. That motivates Omega as a first-pass family-selection rule.

But that rule was validated on **forecast error** — sMAPE and MSE. It does not tell us
whether the same family boundary holds when under-prediction and over-prediction have
unequal operational consequences. The authors themselves name probabilistic scoring as
future work.

Separately, a 2026 paper finds descriptor-based model selection correct in only a small
fraction of cases. That is the strongest published challenge to our premise, and it is why
K2a exists as a gate rather than a formality.

## The arithmetic that makes this precise

For the asymmetric linear decision with underage cost `cu` and overage cost `co`:

```
C(q, y) = cu·(y − q)+ + co·(q − y)+          tau = cu / (cu + co)
```

the optimal action is the **tau-quantile**, and

```
C(q, y) = (cu + co) · rho_tau(y − q)
```

where `rho_tau` is pinball loss. So **minimising downstream decision cost is exactly
equivalent, up to scale, to minimising pinball loss at the operational quantile.**

That gives an expensive but exact target: run every candidate, validate under pinball-tau,
pick the best family. The whole project is about approximating that choice cheaply.

Define, for family `m`:

```
J_m(x, tau) = E[ L_tau( a_m(x), Y ) ]
Delta(x, tau) = J_TSFM(x, tau) − J_classical(x, tau)
```

`Delta < 0` → the TSFM family is decision-optimal. Otherwise the classical family is.
**The paper is about predicting the sign and practical magnitude of Delta cheaply.**

### Relation to CRPS

```
CRPS(F, y) = 2 · ∫ rho_tau( y − F⁻¹(tau) ) dtau
```

CRPS is the uniform average of decision cost over *all* operational quantiles; pinball-tau
is the single slice a given deployment actually faces. So the CRPS comparator is a precise
question, not an arbitrary one: if the operational quantile carries no information, the two
agree and the cost model can be ignored. If it does, they diverge — and the divergence
should grow as tau moves away from 0.5.

## Core hypothesis

Cheap spectral structure contains useful, transferable information about model-family
choice, but it is **not universally sufficient** once downstream costs emphasise asymmetric
quantiles. A cost-aware selector using tau can recover much of the value of exhaustive
decision-aligned validation; where Omega is insufficient, the missing information should be
identifiable.

## The intended contribution

Not "a slightly better threshold". A **characterisation of when cheap pre-forecast
information is sufficient** for decision-aware model-family selection, when it is not, and
what minimum additional information would repair the failure.

---

## The pipeline

```
new time series x   →   cheap descriptor   →   frozen selector   →   confident: classical / TSFM
known cost tau          Omega(x)               P(T* | Omega, tau)     ambiguous: defer
                                                                              ↓
                                                                    full pinball-tau validation
```

The expensive reference procedure is fixed: evaluate both families under historical
pinball-tau validation and choose the lower-cost one. The proposed selector sees only cheap
descriptors and tau at deployment time. **Nothing is trained at any stage** — every
forecaster is frozen and zero-shot.

---

## Feasibility gates (E0)

Four gates, run before any expensive compute, at essentially zero GPU cost. They answer
whether the project is viable.

| Gate | Question | What it does |
|------|----------|--------------|
| **K1** | What does Omega actually contain? | Implement the statistic; confirm from the published definition that it is determined by second-order and spectral information — the assumption the matched-Omega construction depends on. Validate against synthetic signals of known structure. Decide levels vs first differences. |
| **K2a** | Does the published relationship replicate? | Compute Omega on GIFT-Eval training splits, join to the **published** per-model sMAPE and MSE results, check the pattern is there. No forecasting model is run and no metric is changed. |
| **K2b** | Does it survive the move to the decision metric? | Only after K2a passes: check whether the relationship holds under pinball loss at tau = 0.5. |
| **K3** | Is a scalar boundary meaningful? | Before fitting anything: count sign reversals in Delta, compare a monotone fit against an unconstrained one, measure how much oracle value the best single threshold captures. |

**Why K2a and K2b are separate.** Pinball at 0.5 is proportional to mean absolute error and
is *not* a metric the original result was established on. Collapsing the two would leave a
failure ambiguous between a false premise, a broken pipeline, and a metric change. A K2b
failure is informative in its own right.

**The go/no-go gate is after K2a.** If K2a fails we change direction in week one rather than
week eight. The pivot is already written: we ask instead whether decision-aligned selection
is more reliable than error-based selection — arguably the stronger paper.

**If K3 shows a scalar threshold captures little oracle value**, we stop fitting
`Omega*(tau)` and the paper reframes around insufficiency. A fitted regression always
returns *some* curve; K3 checks whether the boundary is real before we build a paper on it.

---

## The full experiment, if the gates pass

Six stages. Predictions are generated **once**, cached to disk, and every later experiment
is arithmetic on those files — that is why one stage is heavy and the rest are nearly free.

| Stage | What |
|-------|------|
| 0 | Pre-registration and data preparation. Freeze the grid, horizons, tau values, splits. Apply the 1000-timestep floor. |
| 1 | Premise gates K1–K3 (above). |
| 2 | **Generate all predictions.** The only expensive stage. Full quantile grids, not point forecasts. Partitioned Parquet, idempotent writes, resumable. |
| 3 | Decision layer and baselines. Freeze the within-family rule; two decision tasks; sweep all eleven tau values. Reads the cache; no model runs. |
| 4 | The four experiments E1–E4. |
| 5 | Statistics and writing. Block bootstrap at series and domain level; forecast-comparison tests on headline claims only. |

### The four experiments

| | Question | Output |
|---|---|---|
| **E1** | Where does the boundary sit as costs get lopsided? | Sweep tau on both sides of 0.5. The candidate boundary is written in `\|tau − 0.5\|` and therefore *assumes* symmetry; sweeping both sides tests that instead of asserting it. Primary output: the (Omega, tau) decision map with fitted boundary and bootstrap uncertainty. |
| **E2** | Does it transfer? | Fit on development collections, freeze every coefficient, evaluate on M5 and PJM — never touched during fitting. The honesty test. |
| **E3** | Is Omega sufficient in principle? | Construct synthetic series identical under Omega but different underneath (same spectrum, different innovation skew or tail weight). If they need different families, **no** Omega-only selector could separate them. A statement about the descriptor, not about our fitted rule. |
| **E4** | What is it worth in deployment? | Selector answers classical / TSFM / defer. Plot validations avoided against excess decision regret: "skip X% of the work for Y% worse decisions." |

**E5** (one minimal repair, adding a single pre-registered descriptor if E3 fires) is
explicitly **deferred and out of scope** for this submission cycle.

### The seven comparators

| Comparator | What it establishes |
|---|---|
| Always classical | Cheap fixed-family floor |
| Always TSFM | Expensive fixed-family floor |
| Published static Omega rule | The prior method being tested and corrected |
| Pinball-tau validation | Exact decision-aligned selector; the approximation target |
| CRPS validation | Generic probabilistic selector: does decision alignment add anything? |
| **D_Omega** | Our proposed cheap decision-aware selector |
| Hindsight family oracle | Ceiling, and the denominator for normalised regret |

---

## Terminology

### Gates and experiments

| Term | Meaning |
|------|---------|
| `K1` | Gate: what does Omega actually contain? |
| `K2a` | Gate: replicate the published relationship on sMAPE / MSE |
| `K2b` | Gate: bridge to pinball loss at tau = 0.5 |
| `K3` | Gate: is a scalar Omega boundary meaningful? |
| `E0` | Collective name for the K gates |
| `E1` | Decision-boundary sweep over tau |
| `E2` | Frozen unseen-domain transfer (M5, PJM) |
| `E3` | Matched-Omega sufficiency test |
| `E4` | Selective validation / deferral frontier |
| `E5` | One minimal repair — **deferred, out of scope** |

### Quantities

| Term | Meaning |
|------|---------|
| **Omega** | Spectral predictability. A fast statistic derived from the power spectrum. Reference values: pure sine ≈ 0.87, daily + weekly cycle ≈ 0.79, moderate noise ≈ 0.32, white noise ≈ 0.06. |
| **The random-walk trap** | A random walk scores Omega ≈ 0.74 **on levels** despite being the standard example of an unpredictable series, because the statistic rewards low-frequency energy and a drifting series is smooth. Whether Omega is computed on levels or first differences must be decided and justified on day one, and **both must be reported**. |
| **1000-timestep floor** | The source paper recommends Omega only for series longer than 1000 timesteps; FFT estimates become unstable below that. Configurations below the floor are excluded, and the exclusion list is recorded and reported. |
| **tau** | The operational quantile, `tau = cu / (cu + co)`. Pre-registered grid: `{0.01, 0.05, 0.09, 0.25, 0.40, 0.50, 0.60, 0.75, 0.91, 0.95, 0.99}`. |
| **cu, co** | Underage and overage cost. Their ratio defines tau. |
| **pinball loss** `rho_tau` | The loss whose minimiser is the tau-quantile. Proportional to downstream decision cost. At tau = 0.5 it is proportional to MAE. |
| **Delta(x, tau)** | `J_TSFM − J_classical`. Its **sign** is what the selector predicts. |
| **decision regret** `R_sel` | `J_m̂ − min_m J_m`. The **primary metric**. Family-selection accuracy is a secondary diagnostic only. |
| **D_Omega** | The proposed cheap decision-aware selector. |
| **CRPS** | Continuous ranked probability score. The uniform average of pinball loss over all tau — hence the "generic probabilistic" comparator. |
| **hindsight family oracle** | Picks the better family with full knowledge. The ceiling, and the denominator for normalised regret. |
| **dominance region** | Where one family wins nearly always regardless of Omega, so no selector is needed. |
| **selector-reliable region** | Where expected selector regret is below a pre-registered tolerance, so full validation can be skipped. |

### Data and models

| Term | Meaning |
|------|---------|
| **TSFM family** | Zero-shot probabilistic time-series foundation models. Provisional roster: Chronos-Bolt-base / -small, TimesFM 2.5, Moirai-1.1-R-small, Tiny Time Mixers r2. Chosen for a consistent quantile interface, small size, and comparability with the replicated result. |
| **classical family** | Probabilistic or distribution-wrapped classical forecasters: AutoARIMA, AutoETS, SeasonalNaive. **These must produce distributions too** — a probabilistic foundation model compared against a point-forecast ARIMA is not a fair experiment. |
| **within-family rule** | Within each family, take the candidate with the lowest mean pinball-tau on **development** datasets at tau = 0.5, and freeze it before any held-out evaluation. Applied identically to both families. |
| **development datasets** | A pre-registered diverse subset of 8–12 GIFT-Eval and Monash collections. Candidates: M4, Electricity, Solar, ETT, Jena Weather, Hospital, US Births, Hierarchical Sales, Loop Seattle, SZ-Taxi, Bitbrains. |
| **held-out datasets** | **M5** (retail demand → newsvendor ordering under asymmetric overage/underage costs) and **PJM** (electricity → reserve procurement under asymmetric shortage/excess penalties). Never touched during fitting, tuning, or debugging. No within-benchmark split counts as cross-domain generalisation. |
| **baseline model** vs **baseline selector** | Two different things. A *baseline model* is a cheap classical forecaster — one of the two candidate families. A *baseline selector* is one of the seven comparators above. "Always classical" is a baseline selector that always picks a baseline model. |
| **cache contract** | Every row keyed by a content hash of (dataset, config, window start, model, horizon, seed). Writes are idempotent, so a rerun resumes rather than duplicating. Partitioned Parquet, synced to persistent storage immediately. |

---

## Compute posture

We train nothing — this is all inference. The entire forecaster suite is about 499M
parameters, roughly 1 GB in half precision, so a free T4 is ample and nothing is
memory-bound. No hosted language model is called at any stage, so the API budget is zero.

Estimated 22–30 GPU-hours total, almost all in Stage 2. Free tiers first, always. Every
line of code is validated on a free T4 with a tiny subset before any larger machine is
touched.

---

## Known risks

| Risk | Response |
|---|---|
| K2a replication fails | Tested in week one at zero GPU cost. Project reframes around whether decision-aligned selection is more reliable than error-based selection. |
| Omega misreads integrated series | Random walk ≈ 0.74 on levels. Levels vs first differences decided and documented on day one; both reported. |
| E3 overruns | The most likely schedule risk. Dedicated owner, started early. |
| Classical models are slow | AutoARIMA over tens of thousands of fits is a **CPU** bottleneck. Parallelise across cores and cache fits. This must not be solved by requesting a GPU. |
| Synthetic-to-real gap | Matched-Omega pairs prove a controlled failure exists but may not capture every higher-order structure in real data. |
| Negative outcomes | If the boundary is non-monotone or matched-Omega pairs do not flip, the paper narrows to the strongest surviving result rather than adding experiments to rescue the original claim. |
