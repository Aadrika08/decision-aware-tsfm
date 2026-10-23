# Aadrika real pilot completion implementation plan

Goal: complete issue #29 with the approved real cache and both consumers' acceptance;
complete #30 with measured resource evidence and the bounded budget.

Authority: Veer's October 15 decision on #29,
https://github.com/Aadrika08/decision-aware-tsfm/issues/29#issuecomment-6185355042,
explicitly accepted by the operator. No Pratham sign-off is claimed.

Frozen generation source: main d2e1b1b40334ee89207be3eab4078b0d291d346a.
Use the existing local RTX4050 and preserved qualification environment, one CPU
worker, one BLAS thread and batch one. No model/metric/data/split changes, paid
infrastructure, M5/PJM access or full-study grid.

1. Preserve the original ZIP and verify 55 source hashes. Add the approval URL
   only to a separate workload/models copy. Record old/new hashes and exact code/runtime.
2. Require real readiness ready: 21,696 jobs / 1,395,776 rows; first slice 24 jobs /
   1,544 rows. Export/check 2,712 role rows: 660 validation, 165 gap, 1,887 evaluation,
   zero exclusions, CSV hash f5d5024359a9f7e69364a49229f65e316337424c241cd87688a95dc78e7ed7da.
3. Measure, audit and deliver the first slice before extending the cache. It is
   interface/resource evidence; one origin cannot support the full K3 analysis.
4. Resume the full approved pilot with measurement; exercise one controlled
   interruption/resume and preserve every attempt. Final audit must verify all
   21,696 jobs with no unexplained omissions, corruption, duplicates or mismatched actuals.
5. Verify a completed rerun generates zero jobs. Exercise missing/corrupt recovery
   on a disposable copy and preserve bad bytes. Bundle cache, plan, both Omega
   variants, roles, source arrays, approved inputs, hashes and all resource reports.
6. Publish the real handoff and request explicit Veer/Abdullah acceptance on #29.
   Publish #30's measured base and separately labelled proposed 10% restart plus
   10% failure scenario, with actual CPU/GPU time, memory, bytes and coverage gaps.

Only close #29 after both actual consumers accept; only close #30 after measured
coverage and budget are complete. Keep parent #16 and full-study gates separate.
Unexpected real failures get recorded, not repaired by altering scientific settings.
