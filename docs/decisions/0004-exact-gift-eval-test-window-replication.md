# 0004 — Exact GIFT-Eval test-window replication

- Date: 2026-09-10
- Owner: Dayson Du
- Issue: https://github.com/Aadrika08/decision-aware-tsfm/issues/1
- Status: implemented, verified on m4_daily, covid_deaths, and ett1_H

## Decision

0003 trimmed a flat `max(HORIZONS)` = 168 timesteps off every series,
regardless of length or frequency. This is as a conservative placeholder pending
confirmation of GIFT-Eval's actual test-window convention. Checking
GIFT-Eval's own paper (Aksu et al., Appendix C: "we use last 10% of each
timeseries ... for testing") suggested a percentage-based rule instead of
a flat count, but the actual implementation, found in GIFT-Eval's own
`src/gift_eval/data.py`, is more specific than either the flat 168 or a
simple per-series 10%.

The real test region size is `prediction_length * windows`, computed once
per dataset configuration:

TEST_SPLIT = 0.1
MAX_WINDOW = 20
windows = ceil(TEST_SPLIT * min_series_length / prediction_length)
windows = clamp(windows, 1, MAX_WINDOW)

`min_series_length` is the minimum length across all series in that
dataset/frequency group, not each series' own length. Every series in a
group is trimmed by the same amount. `prediction_length` depends
on frequency (`PRED_LENGTH_MAP` / `M4_PRED_LENGTH_MAP`) and term
(short/medium/long, multipliers 1/10/15). M4 datasets are hard-coded to
`windows = 1` regardless of the formula, matching GIFT-Eval's published
results table, which has no medium/long entries for any M4 subset.

`stage0prep_fixed.py` does not track "term" per subset the way GIFT-Eval's
`Dataset` class does one raw directory here (e.g. `ett1/H`) backs all
three term configs in the real benchmark. For non-M4 datasets, the test
region is computed under all three terms and the trim uses the largest,
guaranteeing Omega never overlaps the true test window under any term the
published K2a results might use. This overtrims subsets that only ever
have a short-term entry in GIFT-Eval's results table (e.g. `covid_deaths`:
correct trim 30, our trim 450), but never undertrims.

Frequency is read from the `freq` field GIFT-Eval's own HF datasets store,
not parsed from the subset's folder name, since folder naming isn't a
reliable frequency signal across all datasets. If `freq` can't be
determined or doesn't map to a known prediction-length code, the subset
falls back to the flat 10%-of-min-length approximation from 0003, with a
warning printed.

## Validation

Frequency normalization and trim-formula logic checked in isolation
(no dependency on real data) against numbers hand-computed from GIFT-Eval's
own Table 7:

- `ett1_H` (min length 17,420, freq H): computed trim 2,160 (long-term
  wins: 720 x 3 windows).
- `m4_daily` (min length 107): computed trim 14 (M4 special case, matches
  M4's published daily horizon exactly).
- `m4_hourly` (min length 748): computed trim 48 (M4 special case).
- `electricity_15T` (min length 140,256): computed trim 14,400 (long-term,
  windows capped at 20).
- `covid_deaths` (min length 212): computed trim 450, flagged as the known
  overtrim tradeoff for a short-only dataset.

Verified end-to-end against real data on three subsets, full-subset
series-ID-matched comparison (not a sample), same method as 0003:

- `m4_daily`: 4,227/4,227 series, exact match.
- `covid_deaths`: 266/266 series, exact match.
- `ett1_H`: 7/7 series, exact match, after a correction below.

`ett1_H` channel lengths dropped from 17,252 (0003's flat-168 trim) to
15,260 (this trim) -- a difference of 1,992, exactly matching
`2,160 - 168 = 1,992` predicted from the formula. Raw untrimmed length
(17,420, Table 7) minus the trim (2,160) equals 15,260, confirming the
full chain end to end.

Subset-level active-series counts shifted in the predicted direction
across all 12 subsets spot-checked by count alone: M4 subsets gained
series (0003's flat 168 trim was far larger than M4's true trim, wrongly
excluding some series), non-M4 subsets stayed flat or lost a small number
(e.g. `electricity_D` 160 -> 158, consistent with series sitting close to
the 1,000 floor).

**Correction found during verification:** the first version of the
updated `check_testwindow_trim.py` did not replicate `split_channels()` on
the raw-data side. On `ett1_H` this fed the checker a single glued
121,940-length series (the old channel-gluing bug, reproduced inside the
checker only) instead of 7 real 17,420-length series, producing a wrong
`min_length` and a false mismatch (computed trim 12,240 instead of 2,160).
`m4_daily` and `covid_deaths` are univariate and were unaffected. Fixed by
adding the same `split_channels()` logic to the checker; re-verified
`ett1_H` afterward with an exact match.

## Consequences

`dev_manifest.json` regenerated again under the new trim. Manifest active
series count changed from 8,652 (0003's manifest) to 8,841. Non-M4
Omega values will generally differ from 0003's manifest since more of
each series' tail is now excluded; M4 Omega values are computed over
correspondingly more of the series than under the flat-168 trim.

## Follow-ups

- Trim verified end-to-end on 3 subsets (`m4_daily`, `covid_deaths`,
  `ett1_H`), covering the M4 special case, a short univariate dataset, and
  a multivariate dataset with the max-over-terms/capping logic. Not yet
  run on the bitbrains subsets.
- The max-over-terms approximation for non-M4 datasets is a known,
  documented overtrim for subsets without a real medium/long entry in
  GIFT-Eval's results table. A more precise fix would require
  transcribing which terms each subset actually has from Table 7,
  judged not worth the effort given the overtrim direction is safe
  (no leakage) and K2a's dominant sources of uncertainty are elsewhere
  (dataset/model roster matching, per-series-to-per-dataset aggregation).
- `K1_Omega_Validation_1.pdf`'s manifest-ingestion numbers still reflect
  the original pre-fix manifest (9,055 active, mean Omega_levels 0.5871)
  and need regenerating against this manifest before further citation.

