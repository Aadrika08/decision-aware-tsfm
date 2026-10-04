# Stage 0 handoff review — 2026-09-30

Owner: Aadrika. Scope: [#28](https://github.com/Aadrika08/decision-aware-tsfm/issues/28)
and [#45](https://github.com/Aadrika08/decision-aware-tsfm/issues/45), under the
existing producer work. This is a metadata/fixture review, not a real pilot.

**Outcome:** Dayson's new manifest is published and its snapshot identities and
Omega coverage check out. It is a series inventory, not yet an executable forecast
workload. The production naming, staged-source mapping and per-origin exchange
still need reconciliation. An additional finding is that **25 active series have
zero finite observations in the capped history but record capped Omega as 0.0**.
No upstream data, scientific definition or production code was changed here.

[Machine-readable evidence](2026-09-30-stage0-handoff-review.json) contains complete
counts, the 25 affected IDs, input hashes, consumer outcomes and the test command.

For a fresh checkout, use the [published reproduction material](2026-09-30-stage0-handoff-review/README.md).
It includes the exact audit/probe scripts and regenerates the fixture without
depending on the reviewer's local files. Publication is authorized on October 1;
the activity and unsent-draft statements below describe the September 30 review.

## Pinned inputs and verification boundary

| Component | PR | Reviewed commit |
|---|---|---|
| Stage 0 inventory/export | #48 | `5da6edd43cd880c89bde0c076466ea672251697f` |
| Producer contract/readiness | #47, stacked on #31 | `df74d5150cd0ec710380c51048baeeeeb54ec3bf` |
| Decision consumer | #44 | `dd9505d2389a191327e4fb5da937a8c0b289a6a1` |
| K3 consumer | #35 | `24967444a6f36d49a1d539ec23e01597204df596` |
| Statistical decision text | #32 | `dd685915ed456e5856e2402fa56d1cf5f031fb25` |

GitHub was read as `Aadrika08`; no comments, reviews, issues, releases, PRs or
merges were published. PRs #31/#35/#44/#46/#47/#48 remain open in the saved snapshot.
Veer's agreements on #45, #32 and #21 are recorded; the #32 sign-off includes
conditions on dataset-level coverage/counting and best-fixed grouping. The
supplied Slack conversation records Abdullah's agreement on the split approach
and planned provenance fix. Agreement is distinct from a published implementation
or a frozen assignment file.

| Downloaded file | Bytes | SHA-256 |
|---|---:|---|
| `series_manifest.json` | 87,571,416 | `f20898d3954c2c60c600d764bcb97dc4b75da245716d537bec474d0c3a0f7094` |
| `omega_values.json` | 2,750,426 | `5bb86ae341f87de7029e80c10f2d8aca299514fcf68b61162a2dec1b43190563` |

Both JSON files were read completely. No raw time-series arrays, development
sources, held-out sources or model weights were opened. Source/subset hashes were
not recomputed from raw data, Omega was not recomputed, and Stage 0 was not rerun.
Dayson's reported repeated-run reproducibility and real-data spot checks remain
producer-reported evidence; this review verifies the published metadata itself.

## Checks that passed

- **8,761 active series**, **139,127 excluded series**, and **8,761 Omega rows**.
- Zero duplicate IDs, active/excluded ID overlap, conflicting native-item/channel
  mappings, positional fallback IDs or legacy `_idx_` IDs in either inventory group.
- Every active ID has exactly one Omega row; no extra Omega IDs exist.
- All four Omega numeric fields are finite and in [0, 1]. Numeric validity alone
  does not establish descriptor availability; see the empty-history finding below.
- All recorded capped starts equal `max(0, split_boundary_index - 4096)` and all
  finite-count fields lie between zero and the corresponding raw span.
- No M5/PJM dataset/config/subset entries appear in either inventory group. This
  checks metadata membership; it does not independently prove raw-data provenance.
- Hospital has zero active rows and the duplicate flat Jena directory has zero
  active rows. There is still **one active `us_births/D` row**; Dayson should resolve
  the discrepancy with the earlier dataset review rather than having the cache
  owner silently remove it.

The ID spelling needs no cosmetic change for the runner. These checks establish
uniqueness/mapping within the published snapshot, not independently rerun stability.

## Findings and exact remaining handoff

### 1. Empty capped histories must not be interpreted as measured low Omega

There are **25** active rows with `history_length_4096 == 0` and both capped Omega
fields equal to `0.0`: **18 Bitbrains fast-storage/5T channels and 7 KDD hourly
series**. Their raw cutoff positions are internally consistent; the problem is
descriptor availability, not an incorrect start index.

Example: `bitbrains_fast_storage_5T__fastStorage_694_dim0` records raw history
`[3104, 7200)`, zero finite observations in that span, and capped Omega `0.0`,
while its full-history level Omega is `0.1387`.

Dayson and the analysis owners should explicitly classify these capped values as
unavailable/insufficient input and agree downstream handling before those values
enter K2/K3. This review does not change values, exclude rows, impute observations
or choose a new minimum-history rule. This finding does not itself exclude an
unrelated approved pilot configuration.

More generally, 3,618 capped windows contain fewer finite observations than their
raw span, and 1,764 active series have a raw history shorter than 4,096 steps.
The cap does not produce an equal finite sample size for every series.

### 2. Answer to the three assumptions raised in PR #48

| Question | Producer contract and required action |
|---|---|
| Constant dataset field | #48 uses historical `dataset=Salesforce/GiftEval`, `config=ett1/H`. The new contract uses header `naming_scheme=dataset-config-v1`, the actual `source_repository`, and row `dataset=ett1`, `config=H`. M4 suffixes and no-variant empty configs follow ADR 0006's explicit mapping. Confirm the actual repository/revision per source; the GIFT-Eval constant is not independently verified for every downloaded subset here. Preserve the original inventory and released smoke naming. |
| Subset checksum | Keep `source_checksum=sha256:…` as upstream subset provenance. The runner's `source_sha256` is 64 hexadecimal characters hashing the exact staged one-dimensional `.npy` bytes. Removing the prefix does **not** turn a directory hash into a staged-file hash. Both links need an explicit mapping; the original data revision is separate from the Stage 0 code revision. |
| Series ID format | Existing native-item/channel IDs pass snapshot checks; the runner treats them as opaque. Preserve them and their native mapping. Record any future fallback or collision explicitly. There are also 5,265 active univariate rows with `channel=null`; the executable contract requires a nonnegative integer channel. The staging handoff must document the univariate mapping, without changing the IDs or flattening multivariate series. |

### 3. An inventory must be accompanied by a frozen forecast workload

The inventory already supplies stable IDs, original item/channel references,
raw length, trim boundary, finite history length and candidate horizons. It does
not supply the executable workload document or source staging bundle.

The remaining exchange needs:

- Actual data revisions, staged file paths/hashes, raw source start timestamps
  and frequencies. `length` is a finite-observation count and must not silently
  replace raw positional length or delete internal timestamps.
- Explicit approved series/origin schedule, context and usable bounds, individual
  horizons/seeds, and genuine workload/model approval references. The existing
  bounded pilot uses context 2,048, h=1/24/168 and seeds 42/43; this review selects
  no series or origins.
- A separate window assignment keyed by dataset/config/series/origin/horizon for
  validation/evaluation/gap, common across models and seeds. A series can contain
  more than one role; overlap depends on horizon. Keep dataset-level fold
  assignments separate. Do not overload the runner's existing forecast bounds
  with selector-fitting/evaluation semantics.
- Recorded analysis-owner decisions for the time split, dataset folds and
  selector decision-time cutoff. Veer proposed using Omega at the first evaluation
  origin; that proposal is not silently implemented by this review.

Dayson retains ownership of the final workload/export and freeze. Veer and
Abdullah define the analysis assignments. Aadrika validates their supplied
workload with the existing checker; no new converter is included here.

### 4. Omega exchange still needs source/origin/variant identities

The published sidecar has one row per series at the trim boundary. It contains
the four scalar fields and the capped start/finite-count metadata; it is not yet
the readiness sidecar with explicit repository/revision/staged-source linkage,
origin, named variant, raw `history_start`/`history_end`, method descriptions and
fixture/schema headers. The levels-versus-differences mapping must remain explicit.

Keep these published values for their documented cutoff. Do not copy them to an
earlier origin, infer raw end from the finite count, or choose a new Omega default.
Dayson supplies the appropriate cutoff calculations for the approved workload.

## Existing readiness and consumer results

Passing the unmodified inventory to `inspect_pilot` yields an expected blocked
report: `Workload schema_version must be 1`, with no plans and zero source-loader
or adapter-loader calls. This is a **shape/admission probe**, not proof that every
semantic gap above was diagnosed by the checker and not evidence that the series
inventory itself is corrupt.

| Probe | Observed result | Interpretation |
|---|---|---|
| Existing planner/readiness/handoff tests | **24 passed, 0 failed, 0 skipped** | Includes the pinned generator's two-export determinism test and negative admission cases. |
| Regenerated fixture | **408 partitions / 9,792 rows**, 34 series, 68 series-origins | Every partition hash verified, including after ZIP extraction/readback. |
| K3, `current` and `last_4096` | Both `fixture_valid`; 408 provenance-checked jobs, 136 comparisons, no unused Omega | Local compatibility evidence, not Abdullah's owner acceptance. |
| Decision consumer, native producer schema | `NotImplementedError` | Published importer still supports only `fixture-v0`. |
| Decision consumer, diagnostic truth-column rename only | `DuplicateCacheKeyError` for 9,792 rows | Seed-aware uniqueness remains necessary; no seeds were dropped. |
| K3 negative control: mismatched repository header | Accepted as `fixture_valid` | Repository/naming metadata validation remains incomplete. |
| K3 negative control: three invented partitions without job metadata, presented to its non-fixture path | Accepted as `ok`, **0 provenance-checked jobs** | Confirms the acknowledged mandatory-provenance gap. This is intentionally malformed invented data, never real acceptance. |

The last two rows are findings about the unchanged consumer. They are not passing
validation tests or reasons to modify producer payloads. Both consumer branches
were left unchanged. A new commit from either owner requires a new probe.

Fixture seeds are duplicate synthetic coverage; Omega labels are planted and
the three fixture models do not redefine the four-model real-pilot roster.

## Deliverables and reproduction

- Original local fixture ZIP:
  `artifacts/runs/stage0-handoff-review-20260930/consumer-handoff-2026-09-30.zip`
- [Regenerate the shared fixture and reproduce this review](2026-09-30-stage0-handoff-review/README.md)
- [Evidence JSON](2026-09-30-stage0-handoff-review.json)
- Local input snapshots, scripts, logs and probes:
  `artifacts/runs/stage0-handoff-review-20260930/`

The ZIP is local and has not been published. Its SHA-256 is
`5f1f0df0de17282ffdbae4c0cd4fc8e356511dfbf774bf6e6da54ed54a16b94c`.
The regenerated `handoff.json` hash is recorded in the adjacent evidence JSON.
Repacking ZIP metadata may change the archive hash; verify the handoff's partition
hashes as the payload authority.

The following preserves the original local replay procedure. GitHub readers should
use the fresh-checkout instructions linked above; the local paths below are not
published artifacts.

Run from this review worktree using Python 3.11.15, NumPy 1.26.4, pandas 2.2.3,
PyArrow 16.1.0 and SciPy 1.11.4. The `sources/decision` and `sources/k3` clean
worktrees are pinned to the commits above. The installed environment used here is
`/home/utkarsh/Work/shraddha/decision-aware-tsfm-aadrika3/.venv/bin/python`.

The following replay creates a fresh directory on every invocation, copies the
immutable metadata inputs, and points the audit, export and both probes at that
same directory. It preserves the original report, fixture and negative controls.
Use the pinned Python environment above for `python`.

```bash
python - <<'PY'
import hashlib, os, shutil, subprocess, sys, tempfile
from pathlib import Path

repo = Path.cwd()
saved = repo / "artifacts/runs/stage0-handoff-review-20260930"
repeat = Path(tempfile.mkdtemp(prefix="stage0-handoff-repeat-", dir=saved.parent))
def fingerprints():
    return {str(p.relative_to(saved)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in saved.iterdir() if p.is_file()}
before = fingerprints()
shutil.copytree(saved / "inputs", repeat / "inputs")
for name in ("audit_manifest.py", "probe_consumers.py"):
    shutil.copy2(saved / name, repeat / name)
def run(label, pythonpath, *args):
    env = dict(os.environ, PYTHONPATH=str(pythonpath),
               DECISION_FIXTURE_SOURCE=str(saved / "sources/decision"))
    with (repeat / (label + ".log")).open("w") as log:
        subprocess.run([sys.executable, *map(str, args)], env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
run("audit", repo / "src", repeat / "audit_manifest.py")
run("tests", repo / "src", "-m", "unittest",
    "tests.integration.test_cache_planner", "tests.integration.test_pilot_readiness",
    "tests.integration.test_consumer_handoff", "-v")
run("export", repo / "src", "scripts/export_consumer_fixture.py",
    "--decision-source", saved / "sources/decision", "--output", repeat / "consumer-fixture")
for mode in ("decision", "k3"):
    run(mode, saved / "sources" / mode / "src", repeat / "probe_consumers.py", mode)
assert fingerprints() == before, "Saved review evidence changed"
print("Independent replay saved at", repeat)
PY
```

Each run retains its own logs and negative-control files. The original collection
script requires the saved `Aadrika08` account and downloads only published
metadata/code; this replay uses the already pinned inputs without network access.
No new dependency, data download pipeline or model run is required.

The exact replay above was verified in a fresh directory: all 24 tests passed
again, the manifest audit and both consumer results matched, and the saved
evidence fingerprints remained unchanged.

## Owner checklist and follow-up scope

- **Dayson:** confirm production naming/source mapping; classify the 25 empty
  capped histories; resolve the US Births count discrepancy; provide the approved
  window workload, source staging and matching per-origin Omega exchange.
- **Veer:** complete the four already accepted #44/#38 integration items and record
  acceptance of the regenerated producer fixture.
- **Abdullah:** complete the acknowledged real-provenance requirement and matching
  repository/naming checks; confirm fixture consumption at the updated commit.
- **Analysis owners/project lead:** record the remaining split/fold/decision-time
  specifications and conditional statistical sign-offs before real evaluation.
- **Aadrika:** deliver this review and bundle, rerun readiness on the corrected
  frozen handoff, then perform the existing #29 first slice and #30 measurements.

The 87,571,416-byte manifest's storage policy is still a separate repository
decision. No LFS configuration, history rewrite or large-file commit was added.
The real pilot, measured budget and consumer-owner acceptances remain pending;
the validation/review contribution itself is complete independently of those gates.

## Draft Slack update — not sent

Pratham, I've completed the producer-side review of Dayson's new handoff and
regenerated the shared fixture. The 8,761 active IDs are unique, there are no
positional fallbacks, every active series joins to Omega, and the metadata guard
finds no M5/PJM entries. All 24 targeted tests passed.

One new finding for Dayson and the analysis owners: 25 active series have zero
finite observations in the last-4,096 window but capped Omega is recorded as 0.0.
We need to distinguish unavailable descriptors from measured low Omega before
using those rows; I haven't changed values or excluded anything.

The remaining producer handoff is now documented precisely: reconcile the older
dataset/config naming with #47, preserve the subset hash alongside the actual
staged-file hash, and supply the approved per-origin workload/Omega mapping. The
existing series-ID spelling can stay. Window roles belong in a separate
series/origin/horizon table; the series manifest does not need to become a forecast
table.

Abdullah's current loader reads both fixture variants in my local check; the
missing-provenance and repository-header checks still need his planned changes.
Veer's published importer still needs the integration he's already taking on.
The regenerated 408-partition ZIP and commands are ready locally for handoff.

My review/report is ready. Once Dayson's corrected frozen workload and the
consumer updates are available, my next execution milestone is the audited
24-job real slice and its resource measurements. No real multi-series result or
budget is being claimed yet.

## Draft PR #48 review — not submitted

Reviewed the complete published manifest and Omega sidecar at `5da6edd` against
producer `df74d51`. Snapshot identity/coverage checks pass: 8,761 unique active
IDs, no fallback IDs, and a one-to-one active/Omega join.

Before the forecast handoff is frozen:

1. Reconcile the old repository-as-dataset naming with `dataset-config-v1` in
   #45/#47. Keep the actual repository as separate provenance.
2. Preserve the subset checksum, and provide the exact staged-file checksum plus
   data revision/source mapping required by the existing runner. The ID spelling
   itself needs no cosmetic change; document the null-channel/univariate mapping.
3. Explicitly classify 25 capped histories with zero finite observations, currently
   reported as Omega 0.0 (18 Bitbrains channels and 7 KDD series). No exclusion or
   Omega-definition change has been made by this review.
4. Keep the series inventory, but provide the separately frozen window/role and
   source/origin/variant handoff needed by #28. Whole-series cutoff values cannot
   automatically serve earlier forecast origins.

The inventory's rejection by the executable-workload checker is expected format
separation; it is not a claim that the inventory is corrupt. Source-data
recomputation and your repeated-run checks were not independently rerun here.
