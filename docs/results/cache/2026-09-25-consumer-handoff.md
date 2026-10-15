# Producer contract and consumer fixture — 2026-09-25

Owner: Aadrika. Scope: [#45](https://github.com/Aadrika08/decision-aware-tsfm/issues/45),
under #3 and related to #16/#28/#29. This records implementation and synthetic
compatibility evidence, not a real pilot or consumer-owner acceptance.

## Implemented

- Explicit `dataset-config-v1` naming with repository provenance in job identity;
  known empty configs are accepted and historical serialization is preserved.
- Stable-ID mapping checks apply to both planning and saved-plan execution/audit.
- Omega/workload naming headers must agree before source access.
- Exact first-target origin/timestamp semantics and the shared review requirements
  are documented in the [handoff runbook](../../runbooks/production-cache-handoff.md).
- The existing integration branch has been proposed for main in
  [PR #46](https://github.com/Aadrika08/decision-aware-tsfm/pull/46). Its Linux,
  Windows and structure checks passed. It has not been merged.

## Verification

Python 3.11.15; NumPy 1.26.4, pandas 2.2.3, PyArrow 16.1.0, SciPy 1.11.4.
The existing CI cache/measurement tests, new handoff tests and package-layout tests
ran 94 checks: 89 passed and five release checks initially skipped because the
archive had not yet been copied into the new worktree. All eight schema/release
checks then passed against the downloaded, unchanged `v0.1.0-smoke` archive,
covering those five skips. After adding exporter-file provenance, all four handoff
tests passed again, including two byte-identical bundle generations in this
environment. Python compilation and `git diff --check` also passed.

The fixture reuses PR #44 at `dd9505d2389a191327e4fb5da937a8c0b289a6a1`:
34 series, 68 series/origins, three fixture models, h=24 and seeds 42/43, giving
408 partitions / 9,792 rows / eleven quantiles. The two seeds duplicate one
synthetic draw for identity coverage. Omega values are planted labels; analytic
expected costs are not exact finite-sample results. This does not qualify a new
model roster or an Omega threshold.

| Artifact | SHA-256 |
|---|---|
| `handoff.json` | `a6e1ec824e6e8fcf4d897e699bee93a20169b0ff69a5212eb22d6ac535591e5c` |
| `consumer-handoff-45.zip` (2,443,124 bytes) | `4b8be161e5a74293f84c416976504d2d0f09ae4a0d6f29b5966a9803c0a25c13` |
| Generator `fixture.py` | `e4f00736faf0eaaf14356f9cb67ab4714a692a62d73cdec0a521fbb54dec0719` |
| Producer `handoff_fixture.py` | `799aca3d9cb391adba38cf9c672e3567dedb272f99cbdd8a968011a8d518d75e` |

Generated files live under `artifacts/runs/consumer-handoff-45/`; the sibling ZIP
contains that directory. Raw forecasts are not committed. Reproduction commands
and input semantics are in the runbook. Archive metadata can differ when repacked;
validate the listed partition and sidecar hashes in the handoff manifest.

## Consumer probes

At PR #35 commit `24967444a6f36d49a1d539ec23e01597204df596`, both `current` and
`last_4096` load as `fixture_valid`, with all 408 jobs' source provenance checked,
136 comparisons, both seeds, 29 distinct planted Omega labels and no unused
Omega sources. This local probe is not Abdullah's sign-off. His real-input path
still needs missing provenance to be fatal and naming/repository agreement checked.

At the pinned PR #44 commit, loading the producer shape returns
`NotImplementedError`: only `fixture-v0` is supported. A diagnostic that merely
renames `y_true` to `y` then raises `DuplicateCacheKeyError` for all 9,792 rows
because seed is absent from the key. No seed was discarded to make the probe pass.
The source also uses last-observation origins and joins Omega on series ID alone.
These are Veer's consumer changes, not reasons to alter the producer's windows.

## Remaining inputs and team update

The fetched Stage 0 branch remains at
`b05b844ae9cff0dbcd27e8b043f775143f047723`; its published `dev_manifest.json` has
8,782 active rows with the older `subset` field. Dayson's newly described stable-ID,
window and 4,096-history work is not present at that head. The post-renaming freeze,
source checksums, approved development roles and both consumer acceptances are
still outstanding. No new real forecasts or full-grid budget were produced.

Suggested update from Aadrika:

> I've implemented the producer-side naming change and documented the time
> convention: origin is the first target, so target position is origin + step - 1.
> Dataset/config are separate under an explicit naming version, with repository
> provenance retained; empty config means there is no variant. The old smoke
> cache stays readable. Please review the shared contract in #45 before freeze.
>
> I reused Veer's distinct-quantile fixture and exported the same hashed producer
> bundle for both consumers. Abdullah's current loader reads both Omega variants
> in my local probe. Veer's importer still needs the producer mapping, seed keys
> and origin/Omega alignment. These are fixture checks, not real results or your
> acceptance on my behalf.
>
> The integration-to-main PR is #46. For the real pilot, I still need Dayson's
> frozen post-renaming handoff and the agreed fitting/validation/evaluation roles
> within development. I'll use the existing four-model pilot once those inputs
> and the shared contract are approved; within-TSFM selection remains deferred.
