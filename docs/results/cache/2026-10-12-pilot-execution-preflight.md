# Aadrika pilot execution preflight — October 12

Owner: Aadrika. Part of [#29](https://github.com/Aadrika08/decision-aware-tsfm/issues/29),
parent #16; resource evidence remains [#30](https://github.com/Aadrika08/decision-aware-tsfm/issues/30).
This records verified preparation, not pilot completion or permission to run the full grid.
The [machine-readable report](2026-10-12-pilot-execution-preflight/report.json)
contains the input fingerprints and proposed split coverage.

## Verified input packet

Dayson's original `HANDOFF.zip` SHA-256 is
`70e814bf323178df84b771f920f6a6114c6796cb5b143ba28ccce4833eb56ba0`.
All 55 staged NPY hashes match the workload. ZIP entry separators were normalized
from Windows backslashes to POSIX slashes on extraction; every entry's bytes
and the original ZIP were preserved. No data or metadata values were edited.

Source: `Salesforce/GiftEval`, revision
`30841734ac5cfddbd0c3bad6d09d2b6b32becbb0`, naming `dataset-config-v1`.

| Configuration | Series | Series-origins | training_end | validation_end |
| --- | ---: | ---: | ---: | ---: |
| electricity/H | 10 | 170 | 31224 | 35064 |
| ett1/H | 7 | 119 | 15260 | 17420 |
| ett2/H | 7 | 119 | 15260 | 17420 |
| jena_weather/H | 21 | 336 | 7344 | 8784 |
| solar/H | 10 | 160 | 7320 | 8760 |

The unmodified real document is **blocked**: `Real workload requires approval_reference`.
Both workload and model references are absent; the selector role/window table is absent.
These bounds are supplied proposals, not an approved selector split.

A disposable in-memory fixture-labelled copy passed model-free technical readiness:
2,712 source windows verified, 55 series, 904 origins, 5,424 workload rows,
21,696 forecast jobs and 1,395,776 forecast rows. Both per-origin Omega variants
join completely; the capped histories are 4,096 steps. No fixture-labelled
executable plan was saved or used for forecasting.

The planned first slice is electricity/H, `electricity_H__MT_003`, origin 34104:
24 jobs and 1,544 rows. It covers all four models, horizons 1/24/168 and seeds 42/43.
The model document specifies `autoarima_native`, `autoets_simulated`,
`seasonal_naive_native` and `moirai_pilot`, exactly matching `pilot_models()`;
context is 2,048, classical season length 24, and the original eleven taus remain.
One origin alone checks the interface and measurement path; it cannot establish
the full validation/evaluation comparison or the K3 boundary.

## Approved inputs still needed

1. Pratham's written approval identifying this exact ZIP/workload and the four
   model configurations in its `models.json`. Record genuine workload and model
   references; one comment can cover both if it explicitly approves both.
2. Final sign-off on [#57](https://github.com/Aadrika08/decision-aware-tsfm/pull/57)
   and Dayson's corresponding origin/horizon role table. Abdullah signed sections
   1–4 on October 11; Pratham's sign-off was not present at this check.
3. Confirmation of the supplied bounds and exported selection. The summary says
   selection uses defined training Omega; this packet has zero undefined rows.
   It does not demonstrate retention of undefined candidates. Approval must
   acknowledge the selection or Dayson must provide an amended packet.

Using the merged engine's `assign_roles` against these origin schedules, the
proposed one-third split and horizon-dependent gap rule leave nonempty validation
for every series at all three horizons: **zero excluded series/horizon pairs**.
This is a diagnostic of #57's proposal, not a frozen role table or a split decision.
No origins or Omega values were changed to achieve it.

The inventory's #48 freeze/checksum corrections remain Dayson's work. The already
staged executable packet can be approved separately by its exact hash. The final
main-study roster, second TSFM, static threshold, deferral pricing and E3 claim
are separate milestones; they do not add models or settings to this first slice.

## Runtime and integration evidence

Use the preserved `cache/qualification-20/.venv/bin/python`, Python 3.11.15.
All 99 qualification pins match, required numerical/classical/Moirai imports pass,
and an independent installed-requirement check finds zero active violations.
`uv pip check` failed on a read-only cache mount; it was not counted as a pass.
The smoke environment lacks StatsForecast and 13 other qualification additions;
it was preserved and is not the inference runtime.

The sole CUDA device is RTX 4050 (6,141 MiB total, 5,796 MiB free at probe),
driver 610.57.04. GNU time and nvidia-smi are installed. The pinned Moirai revision
`0c24ab99db2c1a70ea2a0fc03bf113329772ac64` is cached; its weights hash is
`cc46831272ea07e99d78fe05c09e730c7c76a1b272cddbdaf5173dae371cb238`.
No packages were installed, checkpoints loaded or forecasts generated.

The merged decision engine at main `7ecc09a` includes fixes `461546c`:
**152 decision tests passed without skips**, and both original fixture variants
verified 408 partitions / 9,792 rows with zero unmatched comparisons.
The reasoned-null join and exact per-job step-coverage regressions passed.
Veer's own fixture acceptance is recorded on #45; this does not replace either
consumer's acceptance of the future real cache.

PR #56's review follow-up `a22812d` accepts an omitted status or `ok` for defined
Omega, matching the consumer. All **105 PR tests** pass without skips; current-main
integration passes **99 cache/layout tests** and **15 final readiness tests**.
An initial integration invocation used an incorrect test-module name and lacked
the smoke ZIP: 97 tests, one import error and five skips. Correcting only the
invocation/evidence paths yielded the complete 99-test pass; both logs are retained.

## Execution after approval

Preserve the original archive. Create a separate approved input copy only after
the actual approval URLs and role table arrive. Do not replace fixture flags or
invent approval links to bypass admission. Use a fresh output directory for each
check/attempt and retain every failed or interrupted attempt.

From the execution checkout, with its verified qualification environment:

```bash
set -e
export PILOT_PY=/home/utkarsh/Work/shraddha/decision-aware-tsfm/cache/qualification-20/.venv/bin/python
export PYTHONPATH=src HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
"$PILOT_PY" scripts/check_pilot_readiness.py --workload artifacts/runs/pilot-handoff-2026-10-12/HANDOFF-approved/workload.json --models artifacts/runs/pilot-handoff-2026-10-12/HANDOFF-approved/models.json --omega artifacts/runs/pilot-handoff-2026-10-12/HANDOFF-approved/omega.json --output-directory artifacts/runs/pilot-ready
"$PILOT_PY" scripts/measure_pilot.py --plan-file artifacts/runs/pilot-ready/first-plan.json --cache artifacts/runs/pilot-cache --output-directory artifacts/runs/pilot-first-measurement --real
"$PILOT_PY" scripts/forecast_cache.py audit --plan-file artifacts/runs/pilot-ready/first-plan.json --cache artifacts/runs/pilot-cache --output artifacts/runs/pilot-first-audit.json
```

Require readiness `ready`; the role table must match the approved protocol, not
merely the forecast bounds. The existing wrapper enforces one CPU worker, one
BLAS thread, batch one and unambiguous GPU selection. Deliver the audited first
slice and its plan/checksums before extending to the approved pilot plan.

Then follow the [development-pilot runbook](../../runbooks/development-pilot.md):
resume the pilot, verify a completed rerun generates zero jobs, exercise missing
and corrupt partitions on a disposable copy, package provenance/sidecars/audits,
and obtain explicit Veer and Abdullah acceptance. #30 needs actual CPU/GPU/storage
measurements and the measured base plus the separately labelled proposed margins.
Synthetic qualification timings cannot establish either the budget or finish date.

The October 13 first slice and October 14 pilot are targets conditional on timely
approvals and successful measurement; a firm completion time follows the real
first-slice evidence. The proposed October 15 full run is not authorized here:
all eight project gates, including real K3 and budget acceptance, still apply.

No real pilot forecasts, M5/PJM access, study-default selection or full-grid run
occurred. #29, #30, #45 and parent #16 remain open for their actual conditions.
