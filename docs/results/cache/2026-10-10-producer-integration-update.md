# Producer integration update — October 10

Owner: Aadrika. Compatibility issue [#55](https://github.com/Aadrika08/decision-aware-tsfm/issues/55), parent #3; related handoff #45, real pilot #29 and measurement #30.

The producer readiness fix retains reasoned null Omega without dropping forecast
jobs. Existing source, cutoff, finite-forecast and defined-current variation gates
remain intact. No forecasting models ran and M5/PJM were not accessed.

## Implemented and checked

- Commit `93c245f` adds nullable-descriptor admission, per-variant defined/undefined
  counts and reasons, and defined-value ranges. Blocked variation reports retain
  those counts; the gate is unchanged. Forecast schema and job identities are unchanged.
- Baseline: nine readiness tests passed. New nullable regressions produced six
  failed assertions against the old checker, then all 14 readiness tests passed.
- Full repository discovery passed all **104 tests, zero skips**, using Python
  3.11.15 with NumPy 1.26.4, pandas 2.2.3, PyArrow 16.1.0 and SciPy 1.11.4,
  the preserved smoke archive and a clean pinned consumer-fixture source.
- A first discovery run without those evidence inputs passed with six skips;
  the supplied-input run above covers those checks. Neither run certifies a
  real-model runtime or measured throughput.
- Initial PR CI passed Linux but failed the new acceptance test on Windows:
  it regenerated NPY files while readiness retained memory maps. The test now
  keeps its staged arrays immutable and changes only Omega metadata between
  cases, matching the real handoff contract. No executor behavior was changed;
  the corrected pushed-head CI must pass before this PR is considered green.

Reproduce from this PR checkout in the pinned environment, after downloading
`v0.1.0-smoke/forecast-smoke-sample.zip` into `cache/` and creating a clean
detached generator checkout at the following commit:

```bash
git worktree add --detach artifacts/runs/decision-fixture-source dd9505d2389a191327e4fb5da937a8c0b289a6a1
PYTHONPATH=src DECISION_FIXTURE_SOURCE=artifacts/runs/decision-fixture-source python -m unittest discover -s tests -t . -v
```

The [execution checklist](../../superpowers/plans/2026-10-10-aadrika-integration-and-pilot-plan.md)
keeps the remaining owner dependencies and gates explicit.

## Original consumer artifact delivered

### October 12 review follow-up

Veer approved #56 and requested a nonblocking consistency check for defined
Omega. Readiness now accepts an omitted status or `ok` for a defined value and
refuses contradictory or malformed statuses before reading sources, matching
the merged consumer. The added test produced seven failed subcases before the
two-line guard. No Omega value, forecast identity or scientific setting changed.
The full PR suite passes 105 tests without skips in the preserved qualification
environment; final pushed-head CI is verified separately before landing.

The two #44 ingress findings were fixed in `461546c` and merged into main.
Veer's fixture acceptance is recorded on #45. The real pilot remains dependent
on its actual approval references and the proposed #57 split becoming approved.

The [consumer-handoff-45 release](https://github.com/Aadrika08/decision-aware-tsfm/releases/tag/consumer-handoff-45)
contains the original September 30 ZIP, renamed for the requested handoff. It was
recovered without regeneration; GitHub's asset digest matches its local hash.

| Artifact | SHA-256 |
| --- | --- |
| `consumer-handoff-45.zip` | `5f1f0df0de17282ffdbae4c0cd4fc8e356511dfbf774bf6e6da54ed54a16b94c` |
| Embedded `handoff.json` | `a6e1ec824e6e8fcf4d897e699bee93a20169b0ff69a5212eb22d6ac535591e5c` |
| Embedded `omega.json` | `af504a50fdf0362d3b3d7e787cc9c5505e023233e9ec3238ec20d8ed3162b822` |

All 408 partition hashes and both sidecar hashes were verified before upload.
The fixture contains 9,792 planted forecast rows, 34 synthetic series, two origins
and seeds 42/43. Those seeds duplicate coverage; they are not independent model
runs. This artifact does not approve a pilot or establish research results.

## Fresh #44 review

Reviewed `cab61d49e9e1a875f4b6efca9acc31e9794051c8`; both prior review requests
are resolved: exact near-tie oracle consistency and decision tests in CI. All
117 decision tests passed without skips when supplied the original bundle.
Both Omega variants verified 408 partitions/9,792 rows, 242 compared units,
zero unmatched units and maximum relative direct-column cost difference
`3.308343066616648e-16`. The fixture's validation comparators report insufficient
data; this remains an interface check, not a selector-performance experiment.

The [submitted re-review](https://github.com/Aadrika08/decision-aware-tsfm/pull/44#pullrequestreview-5478474252)
requests two additional fixes, reproduced on disposable synthetic inputs:

1. `read_handoff` detects a missing join with `omega.isna()`, rejecting a present
   null-with-reason row before engine deferral; its projection also drops
   `omega_status`. Join membership and descriptor validity need separate checks.
2. Producer job completeness is not checked before seed aggregation. Removing
   step 24 from classical seed 0 at two origins, while the other seed retains it,
   passes hash/schema/provenance checks and changes the tau=.50 oracle from TSFM
   cost 21.0833/scale 51.25 to classical cost 11.5319/scale 31.0638. The producer's
   existing payload validator rejects those incomplete partitions. The importer
   must enforce full `1..horizon` coverage per job before computing costs.

These malformed-input probes do not show that the producer emitted bad data.
Veer retains ownership of the consumer fixes. Aadrika's probes are not Veer's or
Abdullah's explicit handoff acceptance.

## Remaining admission inputs

The repository is public. The 20 previously refused workflows were restarted and
passed; generic cache CI does not prove collection of separate K3 pytest suites.

The `manifest-v1` tag and #48 exporter are available, but the tracked branch does
not contain an approved executable workload/model document or origin-role table.
No pilot/budget results or roster approval were recorded on #29/#30/#51 at this
refresh. The exporter currently filters undefined training Omega; Dayson must
reconcile that with the keep-and-report agreement without unapproved selection changes.

- Dayson: supply immutable approved workload/staged sources, source mapping and
  checksums, matching current/capped Omega and approved per-origin/horizon roles.
- Veer/Abdullah/Pratham: record the agreed role decision; Pratham supplies the
  genuine bounded-workload/model approval reference. Aadrika cannot invent it.
- Veer/Abdullah: record acceptance against the downloadable original fixture,
  then the audited real cache when it exists.

After admission passes, Aadrika's next compute deliverable remains the measured
24-job/1,544-row first slice, followed by the approved pilot and measured budget.
A second TSFM belongs to the expansion/representative milestone and does not
block delivery of the already defined four-model engineering pilot. Selector
objective, operational deferral pricing, static-band policy and E3 interpretation
remain separately owned scientific decisions.
