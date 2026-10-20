# 0009 — Manifest freeze (manifest-v2), Omega variants, and undefined Omega

- Date: 2026-10-12
- Owner: Dayson Du
- Pull request: #48
- Status: proposed, waiting for review

## Decision

### 1. The frozen manifest

The development manifest is frozen as tag `manifest-v2` (commit `061f31e`).
It is three files. Hashes are SHA-256 of the committed bytes, which are the
same on every operating system.

| File | Where | SHA-256 |
|---|---|---|
| `series_manifest.json` | `artifacts/reference/manifests/` | `db22d3facdb141b16c0abd9fbcbb2c68d609d6f85dc5b448c005290dff5fb0be` |
| `omega_values.json` | `artifacts/reference/manifests/` | `2c80f4247ec190de812108d2493457fad72e91d3c086378c23a490b04811b224` |
| `excluded_series.json` | release asset on `manifest-v2` | `e240e4c3c128eb407e622a81cca4d9e4746ecd92c67fc2e913e742e47346015b` |

The excluded list (139,127 rows, most of the old 84 MiB) is a release asset
so the committed manifest stays small. `series_manifest.json` names it in
`excluded_series_file`.

The files are produced by `scripts/stage0prep_fixed_v2.py` on GIFT-Eval data
revision `30841734ac5cfddbd0c3bad6d09d2b6b32becbb0`. They are written as
utf-8 with LF line endings, dataset folders are processed in sorted order,
and `.gitattributes` marks both committed files `-text` so git never
rewrites their line endings. `scripts/hash_check.py` prints the hash of the
committed bytes and of the working copy.

`manifest-v1` stays as it is, because the provisional K2 rerun on #50 points
at it. It should not be used for anything new.

Contents, unchanged from manifest-v1 except where stated below: 8,761 active
series and 139,127 excluded, the exact GIFT-Eval test-window trim (0004), the
1,000-timestep floor, stable series IDs `{subset}__{item_id}[_dim{channel}]`,
M5 and PJM never opened, and the flat `jena_weather` duplicate skipped. New
in manifest-v2: `frequency` and `freq_code` on every row, Omega rounded to 6
decimals (was 4), the `near_flat` status, and `mode_share` /
`mode_share_4096` per series.

### 2. Omega variants and the default

Every active series has two Omega histories:

- **Full history** (`omega_levels`, `omega_diffs`): the whole training part of
  the series, after the test-window trim, NaNs removed.
- **Last 4,096 steps** (`omega_levels_4096`, `omega_diffs_4096`): the last
  4,096 steps of the same raw trimmed array, NaNs removed afterwards, with at
  least 1,000 finite points required. `history_start_index_4096` records
  where the window starts, so it can never include future observations.

**The default is full-history levels.** The 4,096-step variant is reported
as a sensitivity check. Reason: full history reproduces the K2 numbers
exactly, and the provisional K2 rerun (#50) shows the 4,096 variant moves the
headline correlations by at most 0.134 with no change of sign.

### 3. Undefined Omega

Omega is `null` with a status, never `0.0`, when it is not defined. The
series stays in the manifest and consumers count it as undefined.

| Status | Meaning |
|---|---|
| `ok` | defined; the value is in (0, 1] |
| `empty_history` | fewer than 2 finite points |
| `below_floor` | fewer finite points than the floor (4,096 variants: 1,000) |
| `flat_history` | every point Omega is computed on has the same value |
| `near_flat` | at least 99% of the points Omega is computed on share one value |
| `nonfinite` | the computation returned NaN or infinity (none occur) |

"Points Omega is computed on" means the levels, or the differences for the
`_diffs` columns.

Why `near_flat`: for a series that sits on one value with a rare spike,
Omega measures where the spike falls, not how predictable the series is.
It can come out anywhere from 0 to almost 1: with the spike 5 points from
the end of a flat series it is 0.9999. Under manifest-v1, 22 such values
were exactly 0.0 with status `ok`, and the rule also removes 84
full-history levels values that were not zero (the largest 0.201). All
of them are now `near_flat`.

The 99% cut is a judgement, not a natural gap in the data. Full-history
mode share across active series:

| Mode share | Series |
|---|---|
| 0.90 to 0.95 | 76 |
| 0.95 to 0.98 | 79 |
| 0.98 to 0.99 | 37 |
| 0.99 to 0.999 | 56 |
| 0.999 and above (includes the 66 flat series) | 104 |

`mode_share` (full history) and `mode_share_4096` are stored per series and
are taken on the levels. A different cut can be re-applied from the file
for the two levels columns without
regenerating the manifest. It cannot for the `_diffs` columns, where the
rule is applied to the differences: 22 series have a levels share of 0.99
or more while `omega_diffs` is `ok`, and 10 are `near_flat` on differences
with a levels share below 0.99.

## Consequences

Undefined counts in manifest-v2:

| Column | empty_history | flat_history | near_flat |
|---|---|---|---|
| `omega_levels` | 0 | 66 | 94 |
| `omega_diffs` | 0 | 66 | 82 |
| `omega_levels_4096` | 25 | 166 | 93 |
| `omega_diffs_4096` | 25 | 166 | 80 |

All 94 full-history `near_flat` series are Bitbrains (58
`bitbrains_fast_storage_5T`, 36 `bitbrains_rnd_5T`). Every value that is
defined in both manifests is equal up to the old 4-decimal rounding (largest
difference 0.00005). No value is exactly 0.0.

The final K2 rerun should use manifest-v2. `compute_omega` itself is
unchanged; the status logic lives in `compute_omega_checked`, which is
covered by `tests/unit/omega/test_compute_omega_checked.py`.
