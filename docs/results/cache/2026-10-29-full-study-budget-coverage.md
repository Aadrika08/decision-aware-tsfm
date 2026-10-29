# Full-study budget input coverage — 2026-10-29

Owner: Aadrika. Issue: #3. Pilot resource evidence: #30; parent pipeline: #16.

**All approved pilot forecast jobs now have reconciled measurements. A numeric
full-study estimate remains incomplete because the final executable workload,
model roster and target capacity have not been approved.** The published catalog
sets `full_study_estimate` to null rather than extrapolating invented counts.

## Measured inputs now available

The [completed pilot](2026-10-29-full-pilot.md) supplies 21,696 unique jobs, each
matched to exactly one verified native measurement. Its 55 series, 904 origins,
four models, horizons 1/24/168, seeds 42/43, context 2,048, batch one and eleven
quantiles are covered by the [existing pilot approval](https://github.com/Aadrika08/decision-aware-tsfm/issues/29#issuecomment-6185355042).
Those settings and the pilot's representative/scale/split decisions do not need
reapproval. No new forecasts were generated to produce this budget catalog.

The measurements form **eight native run/device/hardware cohorts and 27
cohort-specific timing groups**, including all 25 recovered original-run
measurements. Every group records its model/checkpoint/method, horizon, context,
batch and device settings, as well as dataset/config and series coverage.
Counts, hashes and native attempt identities reconcile with the complete release.

All 21,696 jobs contain source loading, model loading, I/O, output bytes and native
peak-memory fields. Recorded zeros remain zeros, and absent fields would remain
missing. The 16,272 CPU peak-memory zeros are uninstrumented placeholders;
CUDA peaks are native PyTorch allocations, not process RSS or device-wide memory.
CPU cohort hardware labels are platform strings; the original measurement wrappers
retain the physical machine identities. Do not pool different cohorts silently.

| Successful-job component totals | CPU-labelled jobs | CUDA-labelled jobs |
|---|---:|---:|
| Measured base hours | 171.247048 | 0.303645 |
| Proposed additive 10% restart + 10% failure scenario | 205.496458 | 0.364374 |

These totals are not elapsed, active-GPU or billed machine hours. Successful jobs
exclude incomplete historical interrupted/OOM costs, whose totals remain partly
unknown. The margin scenario is proposed, not a measured observation or approval
for the full study. Qualification and diagnostic observations remain separately
recorded in the original evidence archive.

## Reproducible catalog

[Download the budget-input archive](https://github.com/Aadrika08/decision-aware-tsfm/releases/download/pilot-full-2026-10-29/full-study-budget-inputs-20261029.tar.gz)
from the existing full-pilot release.

- Archive SHA-256: `a313af367c766010ad203304a6f852b57e1b3447ed4b395bdd1f521efa002124`.
- `timing-coverage.json` SHA-256: `81d2544fa9c32db7dec18f6e1f4c6b58a35d072584152e274d646ae73cd289f9`.
- Builder SHA-256: `cdabd38ad5f385216ab08af744bda4287811f1838abccda63e3905b73403f1bd`.

The six-file archive contains the builder, catalog, reconciliation summary,
reviewer's rebuild output, reproduction README and per-file checksums. The builder
requires checkout `a7be4db1e16512f5cc2501973c6203cb15fbc353` and exact original
archive, plan, resource-summary and budget-module hashes. It reuses the existing
budget API, reads only accounting JSON from the evidence tar, and never opens
forecast-source arrays or loads models. Repeated builds were byte-identical;
the central review reproduced the catalog with the same hash.

The JSON includes unexecuted, checked CLI templates for model-free development
planning and per-cohort measurement coverage. A matching timing key is only
mechanical compatibility; it does not establish that pilot timings represent a
new domain, series length or model configuration.

## What is still needed to finish #3

1. **Dayson and project lead:** an approved final workload containing exact
   dataset/config/series, origin/window, horizon, seed, context and split/role
   bounds. The 8,761-series manifest is an inventory, not a forecast job schedule.
2. **Team/project lead:** the final model roster and quantile methods for additions.
   The four-model pilot approval does not freeze a larger study. A second TSFM
   remains an expansion choice; it is not a missing condition for the completed
   four-model pilot.
3. **Compute owner/project lead:** target CPU/GPU identity, worker/batch settings,
   available parallel capacity, and agreed restart/failure allowances.
4. **Aadrika, after those inputs:** report exact job/row counts and uncovered
   timing groups; obtain approved small timings only for uncovered or
   non-representative configurations; then calculate device/cohort totals and
   schedule bounds. State separately whether analysis/selector computation is
   included in the requested budget.

The current development planner refuses M5/PJM even for a model-free plan.
Transfer-stage budgeting therefore needs approved metadata-only counts and an
appropriate approved path; do not bypass that guard or inspect held-out arrays.
No full-grid forecast run is authorized by this record.

The historical approximately 36 configurations × eight models × three horizons
and 22–30 GPU-hour proposal is not a measured full-study estimate. Eleven quantiles
are emitted together and do not multiply forecast-job counts. Older smoke notes
that only h=24 was measured remain historical: the completed pilot now measures
all four approved models at all three horizons. #30 is complete; #3 stays open for
the final workload-based estimate rather than repeating the original smoke work.
