# Aadrika's next contribution — 2026-10-01

Scope: preparation/handoff [#28](https://github.com/Aadrika08/decision-aware-tsfm/issues/28)
and [#45](https://github.com/Aadrika08/decision-aware-tsfm/issues/45), followed by
the already authorized real pilot [#29](https://github.com/Aadrika08/decision-aware-tsfm/issues/29)
and resource evidence [#30](https://github.com/Aadrika08/decision-aware-tsfm/issues/30).

This checklist is the 08:35 IST preparation snapshot. Publication and subsequent
owner updates are tracked on #45. GitHub readers can use the
[portable reproduction instructions](2026-09-30-stage0-handoff-review/README.md)
to obtain the fixture; the ZIP paths below refer to locally prepared packets.

**Next action: resolve the published handoff questions and obtain the missing inputs/reviews.**
The runner, qualification and fixture work already exist. No further implementation
is justified by today's unchanged upstream revisions. The next new experimental
deliverable is a measured, audited 24-job real comparison slice, once its gates pass.

## Completed locally

- [x] Refresh GitHub as `Aadrika08`. At 08:35 IST on October 1, PRs #48, #47,
  #44 and #35 have the same heads reviewed on September 30. The published branch
  list also contains no newer workload or consumer revision.
- [x] Retain the [September 30 report](2026-09-30-stage0-handoff-review.md) and
  [evidence](2026-09-30-stage0-handoff-review.json): 24 targeted tests passed;
  408 fixture partitions / 9,792 rows verified. These are September 30 results,
  not new real-data measurements. Unchanged tests/models were not rerun today.
- [x] Check producer PRs: #46, #31 and #47 are open, each has three successful
  CI checks and no submitted GitHub reviews at the current snapshot.
- [x] Check measurement tool availability: GNU Time 1.10 and one RTX 4050 Laptop
  GPU (6,141 MiB, driver 610.57.04) are visible. This does not qualify the full
  runtime or establish inference throughput or a budget.
- [x] Prepare and verify a local team handoff ZIP
  (`artifacts/runs/aadrika-next-20261001/aadrika-team-handoff-2026-10-01.zip`)
  containing the report, evidence, cache contract/schema, and unchanged fixture ZIP.
  Payload hashes and every fixture partition are checked. The archive remains local.

## Work in order

- [ ] **Aadrika — collect owner responses through the handoff PR and #45.**
  The report and portable reproduction material are prepared for that handover.
  The report's separate PR #48 review and the Slack text below remain drafts;
  sharing the handoff does not submit that review or send a Slack message.
- [ ] **Aadrika + reviewers — land the existing producer stack after review:**
  integration #46, readiness #31, then contract #47. Retarget/recheck the remaining
  PRs as their bases land. Passing CI is not a submitted review or team acceptance.
- [ ] **Dayson + analysis owners — supply the frozen executable handoff.** Keep
  the published stable IDs; reconcile dataset/config and actual source provenance;
  preserve subset hashes separately from staged NPY hashes; provide exact approved
  origins, source start/frequency, bounds, horizons/seeds, and matching Omega
  variants/cutoffs. Record validation/evaluation/gap assignments per window and
  dataset folds separately, with the agreed selector decision-time cutoff.
  Record descriptor availability for any affected selected series. Track the
  25 empty capped histories recorded as Omega 0.0 and the US Births discrepancy
  upstream; neither requires delaying an unrelated approved pilot configuration.
  Aadrika must not select substitutes or fill
  missing approvals. The series inventory itself need not become a forecast table.
- [ ] **Veer + Abdullah — accept the same fixture format at updated commits.**
  Veer owns producer ingress, first-target origin, seed retention and per-origin
  Omega joins (#44). Abdullah owns mandatory real-input provenance and matching
  repository/naming checks (#35). Save each owner's command, revision and result;
  Aadrika's local probes are not their acceptance. Reprobe only when revisions change.
- [ ] **Aadrika — run existing readiness on the supplied frozen inputs (#28).**
  Require a real `ready` report plus genuine approval records; `fixture_valid`
  does not authorize inference. Preserve full `plan.json` and `first-plan.json`.
  The first slice must contain 24 jobs / 1,544 rows with unchanged job IDs from
  the full pilot. Use the [existing runbook](../../runbooks/development-pilot.md#validate-and-plan-without-inference).
- [ ] **Aadrika — measure the first real comparison slice (#29/#30).** Use the
  existing measurement wrapper: four qualified models, hourly context 2,048,
  horizons 1/24/168, seeds 42/43, all eleven quantiles, batch one, one CPU worker
  and one BLAS thread. Audit the slice and deliver it before extending the run.
  Preserve failures and raw CPU/GPU/runtime/storage evidence.
- [ ] **Aadrika — complete only the frozen pilot and close its evidence loop.**
  Verify complete reruns generate zero jobs; exercise interruption and corrupt/missing
  partition checks on disposable copies; obtain both owners' real cache-only
  acceptances against identical hashes. Publish measured base costs and the separately
  labelled proposed 10% restart + 10% failure scenario. Keep full-study budget gaps
  explicit. #29/#30 and parents #16/#3 remain open until their own conditions pass.

The available evidence does not support starting a real run today. No source
arrays, model weights, M5/PJM, experimental settings or production code were changed.
The LFS/storage decision and within-TSFM representative expansion are separate work.

## Ready-to-send team text — not sent

I've checked the latest GitHub state and packaged my handoff report, cache schema
and the 408-partition fixture together. My preparation is complete; the next run
on my side is the measured 24-job real pilot slice once the frozen inputs and
consumer integration are ready. The September 30 checks passed all 24 tests.

Dayson: your stable IDs and full active-series/Omega join check out. One new issue
from my review: 25 capped histories have zero finite observations but Omega is
recorded as 0.0. Please agree how those unavailable values are represented with
the analysis owners. For the runner I still need the approved window workload,
staged-source mapping/hashes and matching per-origin Omega; this can stay separate
from your series inventory. The report answers the naming/checksum/ID questions.

Veer and Abdullah: your current PR heads are unchanged from my fixture probes.
Once the importer/provenance changes you already own are pushed, I'll rerun against
those revisions; please record acceptance against the same fixture hashes.
Pratham/reviewers: #46, #31 and #47 have passing CI and await submitted reviews.
After readiness passes I'll deliver the first audited real slice, followed by
the pilot's measured resource estimate. No real multi-series result is claimed yet.

## Evidence for this update

Local refresh and packaging records are under
`artifacts/runs/aadrika-next-20261001/`: `github-snapshot.json`,
`producer-review-status.json`, `runtime-preflight.json`, and `handoff-package.json`.
The new package has its own SHA-256; the embedded fixture ZIP retains its September
30 hash. This checklist records local preparation, not publication or merge.
