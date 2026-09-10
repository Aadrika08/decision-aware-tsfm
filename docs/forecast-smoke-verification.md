# Issue #3: release verification and remaining budget inputs

Owner: Aadrika. Parent: https://github.com/Aadrika08/decision-aware-tsfm/issues/3

Proposed follow-up sub-issue: **Verify the published cache handoff and resolve
full-grid budget inputs**. This is required by #3's cache handoff and measured
compute completion condition. The work here verifies the existing experiment;
it does not freeze a new dataset, split, horizon, model roster or gate.

## Published handoff

The sample is available in the [v0.1.0-smoke release](https://github.com/Aadrika08/decision-aware-tsfm/releases/tag/v0.1.0-smoke).
The issue comment and original runbook saying it is only local are outdated.
The release points to commit `457dd1d9bd31db5cb539d5778d1ad49a71bc244f`.
Its asset `forecast-smoke-sample.zip` has SHA-256
`9d2e50d43b7898d88ecbfc70cd3642f0b142eb1155bd1574bf1974c9229a6876`.

Independent verification reads the archive without extracting or executing its
contents. It checks the four jobs against the original Git blobs, including
content hashes, timestamps, completeness, identical actuals across models,
finite ordered quantiles and generation provenance. It independently recomputes
all 22 model/tau pinball costs, representative regrets and the recorded timing
scenario. All agree with the original report within absolute tolerance 1e-12.
No Torch import, source download or model inference is required.

The archive includes one historical ledger for job
`34c4642270e75e1627dd3d6ade5bc4190528c167efdeed92fd714a34cbaf77ed`, which is not
one of the delivered four jobs. The verifier reports it separately and does
not use it for their timing or generation provenance. The other three ledgers
cover generation, resume, and a completed rerun that skipped all four jobs
with zero model/data load time.

The original uninterrupted cache is not included. Therefore this verification
does **not** independently reproduce the historical claim of exact equality
between interrupted/resumed and uninterrupted predictions. That requires the
second cache or a new run in the pinned Linux GPU environment.

## Reproduce on Windows or Linux

Download the release asset to `cache/forecast-smoke-sample.zip` in this branch.
Use Python 3.11 with `numpy==1.26.4` and `pyarrow==16.1.0`:

```text
python verify_smoke_release.py cache/forecast-smoke-sample.zip --output docs/forecast-smoke-release-verification.json
python -m unittest test_smoke_release -v
```

The verifier reads immutable Git blobs from the release commit. This avoids
two Windows limitations in the original entry point: its unconditional
`fcntl` import and working-tree byte hashes affected by `core.autocrlf`.
The original forecasting script and lock file remain unchanged so existing
cache identities are preserved. This verifier is specifically for this release.
It is not a replacement forecasting runner.

Machine-readable evidence: [forecast-smoke-release-verification.json](forecast-smoke-release-verification.json).

Verification on 2026-09-10: all four release regression tests passed, covering
the actual release, a changed archive, a mismatched reported metric, and missing
generation provenance. `git diff --check` passed. The original five Linux
forecasting tests and GPU inference were not rerun on this Windows host.

## Full-grid budget: still missing inputs

The release explicitly leaves the full-grid estimate outstanding. The tag
contains no frozen full-grid workload. Its historical measurements establish:

| Representative | Measured inference seconds/job | 1,000 identical jobs |
|---|---:|---:|
| SeasonalNaive | 0.0003674415 mean | 0.0001020671 CPU-hours |
| Moirai-small | 0.4143433695 mean | 0.1150953804 GPU-hours on RTX 4050 |

These are two observations per model at context 2,048 and horizon 24, with
batch size one. The Moirai observations each pay first-inference overhead.
The numbers exclude loading and I/O and are not T4 measurements.

To finish the unchecked task in #3, the following inputs are required:

1. An agreed workload listing retained development series and forecast origins
   per configuration/horizon, with the frozen model roster and seed repetitions.
   The corrected K1 manifest alone does not specify forecast windows or jobs.
2. Approved tiny timing measurements for the remaining model/horizon combinations
   on the intended hardware, plus loading and I/O measurements. The issue currently
   limits inference to one dataset and two models; expanding probes needs an
   agreed issue update before execution.
3. The agreed loading schedule/parallelism and safety margin for the projection.

Then compute separately by resource:
`hours = (sum(jobs * measured_seconds_per_job) + loading_seconds + io_seconds) / 3600`,
with the agreed margin. Eleven quantiles do not multiply the job count.
Do not treat a two-model scenario as confirmation of the 22–30 GPU-hour budget.
Issue #3 must remain open until this completion condition is met.
