# 0003 — Omega channel splitting and test-window trim fix

- Date: 2026-09-10
- Owner: Dayson Du
- Issue: https://github.com/Aadrika08/decision-aware-tsfm/issues/1
- Status: accepted, merged into `main` via PR #11

## Decision

`stage0prep.py` made multivariate targets into one glued series per
dataset instead of splitting per channel. ETT (7 sensors), jena_weather (21),
and bitbrains (2) were affected. `ett1_H` length 121,940 == 17,420 × 7
confirmed this. Every multivariate subset's length divided exactly by its
channel count, while univariate subsets did not.

`stage0prep_fixed.py` adds `split_channels()`. A 2D target splits into one
1D series per channel before Omega runs. A 1D target passes through
unchanged. `series_id` gets a `_ch{n}` suffix when a channel index applies,
and a `channel` field is recorded on every manifest entry, both active and
excluded.

Omega was also being computed on the full series including the held-out
test window. `stage0prep_fixed.py` trims the last `max(HORIZONS)` = 168
timesteps before Omega runs, guarded so series shorter than 168 are left
untouched. `load_train_only()` also prefers an explicit train split
when the dataset exposes one. This does not detect the case where
`load_from_disk` already returns a train-only Dataset with no split dict.
the 168-step trim is a conservative backstop for that case, does not
guarantee. GIFT-Eval's true test-window convention, single trailing chunk
versus multiple rolling windows, is still unconfirmed.

Channel splitting is verified against the manifest: `ett1_H` produces 7
series (was 1), `jena_weather` 21 (was 1), `bitbrains_fast_storage_H`
splits into channels 0 and 1. All 2,500 resulting bitbrains_H series fall
below the 1,000-timestep floor at hourly resolution (max 553 timesteps)
and are excluded. This is correct, since the subset previously cleared the floor
only by being glued to twice its real length. `bitbrains_fast_storage_5T`
is unaffected. 2,462 series remain active.

The trim is verified by independently recomputing NaN-stripped, trimmed
lengths from raw data and matching against manifest entries by
`series_id`, across every series in a subset, not a sample. Exact match on
three subsets: `m4_daily`, 4,227 series, mixed short/long, both branches
of the trim guard exercised; `covid_deaths`, 266 series, all under 168,
trim correctly no-ops; `electricity_H`, 370 series, all over 168, trim
correctly applies.

## Consequences

`dev_manifest.json` is regenerated: 8,652 active series, down from 9,055.
The drop is from `bitbrains_fast_storage_H` dropping out, which is described
above.

This changes which datasets and Omega values feed K2a. ETT, jena_weather,
and bitbrains subsets carry different Omega values and series counts than
the prior manifest. K2a should be rerun against this manifest, not the
pre-fix one.

The trim is not yet checked on the remaining subsets, bitbrains variants,
ETT, jena_weather. The logic is subset-agnostic and three independent
passes already cover both trim branches, so this is lower priority. Not
blocking.

`K1_Omega_Validation_1.pdf`'s manifest-ingestion numbers, 9,055 active
series, mean Ω_levels 0.5871, reflect the pre-fix manifest and need
regenerating before further citation. The synthetic-signal validation
table in the same PDF is unaffected, since it does not depend on `stage0prep.py`.
