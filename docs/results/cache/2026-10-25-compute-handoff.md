# Aadrika's compute handoff, 25 October

Owner: Aadrika. Scope: [#29](https://github.com/Aadrika08/decision-aware-tsfm/issues/29)
and [#30](https://github.com/Aadrika08/decision-aware-tsfm/issues/30).
The operator is obtaining compute and explicitly excluded forecast execution from
this work. No forecasting model was run, no host was rented, and the original
pilot and monitor remain stopped.

[PR #72](https://github.com/Aadrika08/decision-aware-tsfm/pull/72) is merged as
`49af263582a97290193b3c28eac0cf58d52e0ff7`, following Veer's approval of the exact
head, four successful CI checks and no unresolved review threads. The merge
closes #71; the run and measured-budget issues stay open.

## Transfer available now

The [partial-pilot release](https://github.com/Aadrika08/decision-aware-tsfm/releases/tag/pilot-resume-2026-10-23)
contains the existing archive, its checksum and fresh verification records. It is
available independently of the laptop; it is not a completed consumer cache.

- Archive: `pilot-resume-2026-10-23.tar.gz`, 48,498,484 bytes.
- SHA-256: `823c44d67eb155b9cf96e1c2dea270635b4ad29c288d83216b9f56394727437a`.
- Execution code: `1937c7ba11e045449ef5469685b405ab51da58f3`, pinned in the archive's
  `transfer-manifest.json`. This pin includes the measured worker option.
- Original and approved inputs, 55 source arrays, both Omega variants, roles,
  plans, 575 saved jobs, and historical attempt evidence are included.
- Python 3.11.15, the locked environment and pinned model checkpoint must be
  installed and qualified on the selected host.

All 913 original files were rehashed against both the archive and the unchanged
local originals. The archive contains only regular files and directories, with
no duplicate or unsafe paths. A fresh GitHub download matched the same archive
checksum. See [transport verification](2026-10-25-compute-handoff/transport-verification.json)
and [download verification](2026-10-25-compute-handoff/release-download-verification.json).

The fresh [saved-cache audit](2026-10-25-compute-handoff/saved-cache-audit-summary.json)
verifies **575 / 21,696 jobs**: **21,121 missing**, zero corrupt/unexpected/
duplicate partitions and zero mismatched actuals. Its exit code 2 means the
pilot is incomplete. No missing job was generated during this check.

## Existing inputs and decisions

Abdullah has now [verified the complete Omega sidecar](https://github.com/Aadrika08/decision-aware-tsfm/issues/29#issuecomment-6396103789):
1,808 rows, 904 for each variant. That input is delivered; it does not need to be
requested again. His coverage caveats and suggestion to make `last_4096` primary
are analysis considerations. This handoff preserves the [approved pilot decision](https://github.com/Aadrika08/decision-aware-tsfm/issues/29#issuecomment-6185355042):
`current` primary, `last_4096` sensitivity. No research setting was changed.

## Accounting and validation prepared

The cache-only accounting check matched the 575 saved partitions to their own
native verified events: 24 first-slice jobs, 25 recovered original jobs and 526
user-stopped service jobs. It retains four `(run_id, device, hardware)` cohorts
so new-host measurements will not be pooled with laptop timings.

The [partial budget summary](2026-10-25-compute-handoff/partial-budget-summary.json)
records 12.514740 CPU and 0.001450 CUDA resource hours in successful-job
components, plus the separate proposed 10% restart + 10% failure scenario.
These are the 575 saved jobs only, not elapsed time or the complete pilot budget.
Known wrapper durations and unknown OOM totals remain separate. All eight
per-cohort base/proposed budget commands passed their subset checks.

Validation:

- [Pre-merge suites](2026-10-25-compute-handoff/premerge-tests.json): 900 collected,
  897 passed, three optional decision integrations skipped (original bundle,
  real-workload scale, second interpreter). Producer integration tests had no skips.
- After the merge, 57 measurement/executor/planner/budget/layout checks passed.
- The closeout's five Bash blocks and eight embedded Python snippets pass syntax
  checks. Its accounting collector was exercised on the existing cache without
  importing model packages; a mocked gate rejects partial coverage before reading jobs.
- Future full-cache rerun, real repair, new-host qualification and consumer
  acceptance were not executed. See the [safe-check record](2026-10-25-compute-handoff/closeout-safe-checks.json)
  and [partial accounting record](2026-10-25-compute-handoff/partial-accounting-check.json).

## Remaining work that needs the compute run

1. Record the selected host, resource caps and worker count; qualify the exact
   environment/checkpoint and pass the representative admission benchmark in the
   [parallel resume runbook](../../runbooks/pilot-parallel-resume.md).
2. Resume all remaining jobs, preserving the original 575 and every attempt.
3. Perform the full audit, completed rerun with zero generation, and disposable
   missing/corrupt recovery. The recovery check intentionally runs two approved
   forecasts later; it is not performed by this preparation.
4. Publish the complete bundle and measured resource accounting, keeping old
   laptop and new-host attempts separate. Earlier unknown OOM totals remain
   unknown; a successful-job total is not complete workflow accounting.
5. Obtain Veer's and Abdullah's explicit acceptance of the same complete bundle
   through their own consumers. Existing fixture/first-slice approvals do not
   substitute for this.

The two-day execution scenario remains conditional on the actual host's
throughput and memory checks. The final study roster/workload and its budget
are separate from this bounded pilot. #29, #30, #16 and the full-grid budget
item in #3 remain open until their evidence requirements are satisfied.

The [portable closeout runbook](../../runbooks/pilot-closeout.md) supplies the
post-run steps. Do not use the archived laptop monitor: it assumes laptop paths,
service names, code revision and one hardware label for the entire run.
