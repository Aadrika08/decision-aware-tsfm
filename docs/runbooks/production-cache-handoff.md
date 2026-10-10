# Production cache handoff — issue #45

Owner: Aadrika. This is the proposed exchange contract for Dayson, Veer and
Abdullah, tracked in [#45](https://github.com/Aadrika08/decision-aware-tsfm/issues/45).
The naming decision is [ADR 0006](../decisions/0006-production-cache-naming.md).
Review and real-pilot acceptance are pending; the included bundle is synthetic.

## Exact names, payload and time

The [v1 Parquet columns and dtypes](cache-consumer-contract.md#columns-and-types)
are unchanged. New production workload and Omega headers contain:

```json
{"naming_scheme": "dataset-config-v1", "source_repository": "Salesforce/GiftEval"}
```

Rows use the explicit dataset/config mapping in ADR 0006, for example `ett1`/`H`
and `m4`/`daily`. A configuration that is known not to exist is the empty string;
an unknown one must be resolved upstream. Both naming fields are preserved and
hashed in job metadata. Do not reinterpret old `Salesforce/GiftEval`/`ett1/H`
payloads as already migrated; preserve the historical release.

Indices and window ends are zero-based/end-exclusive. For one-based `step`:

```text
context          = source[context_start:origin]
targets          = source[origin:origin + horizon]
target_position  = origin + step - 1
target_timestamp = source_start + target_position * frequency_offset
```

Use the recorded calendar offset, including months; do not assume fixed seconds.
The current runner requires timezone-naive timestamps. With an hourly source
starting `2020-01-01T00:00:00`, `origin=24`, `horizon=2` yields targets January 2
at 00:00 and 01:00. The last context observation is position 23. Horizon counts
observations, not hours.

Keep seeds in forecast and comparison identities. Omega joins require dataset,
config, stable series ID, origin, source revision/hash and one named variant,
under the same repository/naming convention. History cannot end after origin.
`last_4096` caps the current history at the same cutoff; shorter histories stay
shorter. Never attach one full-series value to every earlier origin.

These bounds establish forecast chronology, not the complete selector split.
`split="development"` excludes external held-out domains but does not distinguish
selector fitting, validation and unseen-development evaluation. Veer and Abdullah
must agree the exact assignments and treatment of overlapping origins; Dayson
encodes them. Aadrika must not silently adopt PR #44's default one-third split.

## Reproduce the shared synthetic bundle

Use the existing Python 3.11 environment with NumPy 1.26.4, pandas 2.2.3,
PyArrow 16.1.0 and SciPy 1.11.4. No forecasting packages or model inference are
needed. The exporter reuses Veer's PR #44 fixture at an exact clean commit:

```bash
git fetch origin
git worktree add --detach artifacts/runs/decision-fixture-source dd9505d2389a191327e4fb5da937a8c0b289a6a1
PYTHONPATH=src python scripts/export_consumer_fixture.py --decision-source artifacts/runs/decision-fixture-source --output artifacts/runs/consumer-handoff
DECISION_FIXTURE_SOURCE=artifacts/runs/decision-fixture-source PYTHONPATH=src python -m unittest tests.integration.test_consumer_handoff -v
```

Choose new output paths when repeating. The exporter refuses existing output and
dirty/different generator checkouts. It retains the first two h=24 origins of
each of the generator's 34 series and all three fixture models. It shifts the
generator's last-observation origin by +1, preserving the targets and forecasts.
Seeds 42/43 are explicitly labelled copies for identity coverage, not independent
model runs. Result: 408 partitions, 9,792 rows and eleven distinct quantiles.
This three-model fixture does not change the four-model real-pilot roster.

Bundle contents:

- `handoff.json`: exact partition paths/hashes, sidecar/answer hashes, pinned
  generator provenance and origin/seed meanings.
- Parquet partitions: v1 payload with explicit synthetic job metadata. Their
  source hash identifies the generator/seed/series specification, not a staged
  real `.npy` file; metadata records that distinction. They are not executable
  configured-forecast plans or measured forecast jobs.
- `omega.json`: per-origin current/last-4096 **planted labels**, not calculated
  Omega. Identical variant labels do not establish empirical length invariance.
- `planted_answers.json`: the existing generator's analytic expected costs,
  not exact scores of the bounded finite sample. Its fitting-only dataset has no
  planted comparator table; no new representative or threshold is chosen here.

Read only listed partitions using `pyarrow.parquet.ParquetFile(path).read()`.
Check file hashes and metadata; directory-derived Arrow types are not authoritative.
Use the saved columns to calculate finite-sample costs. Replacing every quantile
with `q_0.50` must fail the column-preservation checks.

## Consumer changes and acceptance

| Owner | Required change / evidence |
|---|---|
| Veer, #44 / #38 | Add producer payload ingress (`y_true` to internal `y`), align origin convention, retain seeds through uniqueness and cost aggregation, and join source/origin-specific Omega. Agree later seed/origin reduction explicitly. |
| Abdullah, #35 / #24 | Require real job provenance rather than skipping absent source metadata; check the new naming/repository metadata across cache and sidecar. His loader currently reads forecast partitions directly. |
| Veer + Abdullah | Record selector development roles, oracle/regret definitions and overlapping-origin treatment before real analysis. D_Omega remains Abdullah's #26; Aadrika does not choose scientific definitions. |
| Dayson | Supply the post-migration frozen manifest, stable item/channel mapping, sources/checksums, exact windows, approved splits, exclusions/dedup summary and both described Omega variants; repeat reproducibility after renaming. |
| Aadrika | Publish the producer bundle and validation evidence; receive both owners' commands, revisions and acceptance against the same hashes. |

PR #35 can be checked without merging its analysis code into this producer PR:

```bash
git worktree add --detach artifacts/runs/k3-consumer 24967444a6f36d49a1d539ec23e01597204df596
PYTHONPATH=artifacts/runs/k3-consumer/src python - <<'PY'
import json
from pathlib import Path
from decision_aware_tsfm.analysis.k3.loader import load_k3_inputs
root = Path('artifacts/runs/consumer-handoff')
manifest = json.loads((root / 'handoff.json').read_text())
for variant in ('current', 'last_4096'):
    _, _, report = load_k3_inputs(root, manifest['partition_hashes'],
                                 root / 'omega.json', variant, allow_fixture=True)
    print(json.dumps(report, sort_keys=True))
PY
```

A local `fixture_valid` probe is not Abdullah's sign-off and is not real-pilot
acceptance. PR #44 currently accepts only `fixture-v0`; do not discard a seed,
collapse origins or relabel the producer bundle to make its current reader pass.

## Integration and next real run

The integration-to-main PR is [#46](https://github.com/Aadrika08/decision-aware-tsfm/pull/46).
It includes the existing layout, cache and adapter work; #31/#35/#44 remain
separate. This contract implementation is stacked on #31 for its readiness
checks. After the base PRs land, retarget remaining PRs to main; do not merge
unreviewed analysis changes into the producer branch.

Once shared review and the frozen input requirements are met, follow the existing
[development pilot runbook](development-pilot.md): readiness, first comparison
slice, measured generation, audit/resume, both consumers, then the budget.
No new real pilot, full-grid estimate or consumer-owner acceptance is established
by this contract work. Issues #29/#30 and parents #3/#16 remain open.
