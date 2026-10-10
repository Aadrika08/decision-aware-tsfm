# ADR 0006: explicit production dataset/config naming

Status: proposed for shared review in [#45](https://github.com/Aadrika08/decision-aware-tsfm/issues/45).
Aadrika approved implementing the producer changes on 2026-09-25. Dayson, Veer
and Abdullah have not yet signed off this exchange contract or a real manifest.

## Reason

The historical smoke stores `dataset=Salesforce/GiftEval`, `config=ett1/H`.
Dayson's proposed manifest uses `dataset=ett1`, `config=H`. Silently changing those
meanings would break joins and discard repository provenance. The production
sprint required a shared contract but did not freeze this exact spelling.

## Proposed contract

New workload and Omega documents carry the same top-level
`naming_scheme: "dataset-config-v1"` and `source_repository: "Salesforce/GiftEval"`.
The repository value identifies the actual source; synthetic fixtures use an
explicit synthetic repository label. Dataset/config are separate literal
components. The planner consumes agreed values; it does not guess them from paths.

| Source layout | `dataset` | `config` |
|---|---|---|
| `ett1/H` | `ett1` | `H` |
| `electricity/15T` | `electricity` | `15T` |
| `m4_daily` | `m4` | `daily` |
| `m4_weekly` | `m4` | `weekly` |
| `m4_monthly` | `m4` | `monthly` |
| `m4_hourly` | `m4` | `hourly` |
| `m4_quarterly` | `m4` | `quarterly` |
| `m4_yearly` | `m4` | `yearly` |
| A source with no configuration | Agreed dataset name | `""` |

This table specifies names, not dataset eligibility: excluded M4 variants and
Hospital do not become eligible through a naming change. Frequency is a separate
field. Preserve spelling/case from the approved mapping; do not split arbitrary
underscores or infer a missing frequency. Empty config means a known absence of a
variant; missing/null config is an error. Slash-delimited composite values and
leading/trailing whitespace are rejected under the new convention.

The two header fields are copied into each forecast job and included in its
identity hash. Physical Parquet columns stay v1 and configured job metadata stays
version 2; the naming scheme is versioned independently. Missing both header
fields means the historical convention; partial/unknown headers and mixed plans
are rejected. Historical artifacts and identity serialization are preserved.
New code still changes the existing implementation fingerprint for new plans.

Within a workload, `(dataset, config, series_id)` must map to exactly one source
revision/hash, item and channel. Item IDs may repeat in different configurations;
different channels need distinct stable series identities. Dayson owns their
construction. Forecast rows retain `(job_id, step)` as their unique key, and seeds
remain distinct. Matching source and naming metadata is required across joins.

## Unchanged research decisions

Origin remains the zero-based first forecast target; target position is
`origin + step - 1`. Omega remains a separate source/origin/variant sidecar.
No dataset, split, model roster, horizon, tau, Omega calculation, threshold,
representative rule or oracle definition changes here. Development fitting and
unseen-development evaluation assignments still need owner agreement. M5/PJM
remain unavailable for development.

## Acceptance

The implementation and synthetic exchange checks are described in the
[production handoff](../runbooks/production-cache-handoff.md). Shared approval,
Dayson's post-migration reproducibility check, and real consumer acceptance remain
separate prerequisites for freezing a real workload.
