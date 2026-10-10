# K2 provisional rerun on manifest-v1 (two files, undefined Omega as null with a status)

Status: PROVISIONAL, run on the frozen manifest-v1 from the unmerged PR #48 at `23af713`; not
final until #48 merges and the team records which history variant is frozen. Reanalysis of
published results; no forecasting model was run. Every number below was produced on 2026-10-10 by
the commands in this note. Run outputs are in a scratch folder, not in the repository; no saved
result file in `docs/results/k2/` was changed or added. Part of #39.

In short:

1. On the full-history Omega column the rerun gives the same numbers as the saved run. All 60
   dataset-level correlations and their bootstrap intervals are identical to the last digit
   (largest difference 0.0), because the order of the 20 datasets by Omega did not change. The
   like-for-like replication of the published -0.556 is still the sMAPE value -0.417, 95%
   bootstrap interval [-0.793, +0.091], n = 20, and the interval still includes zero.
2. On the 4,096-step Omega column the same value is -0.426 [-0.786, +0.089], n = 20. Across the
   12 headline cells the largest change from full history to 4,096 steps is 0.134 (MAE versus
   statistical, +0.251 to +0.117). No interval changes between including and excluding zero.
3. K2 stops on this manifest by default. 8 active series ids match the team's held-out rule
   because the new id format ends multivariate ids in `_dim<channel>`, and `_dim5` contains `m5`.
   No M5 or PJM entry is in the manifest. Both runs here were made with the opt-in that leaves
   those 8 rows out. That has no effect on the dataset-level numbers on full history, but it does
   move the configuration-level numbers (by up to 0.061), and its effect on the 4,096 column
   was not measured. See "Held-out rule" and "Open".
4. Two of the data owner's statements did not check out as written (the hash of
   `series_manifest.json`, and the hash of `omega_values.json` unless line endings are converted).
   The counts 25, 66 and 166 and "every other value is unchanged" are confirmed.

## Commands

Run from the repository root with the pinned interpreter (Python 3.11.15, numpy 1.26.4, pandas
2.2.3, pyarrow 16.1.0, scipy 1.11.4). `<scratch>` is a working folder outside the repository;
the scripts and logs named below were kept there and are not committed.

    # the two manifest files, taken from the shared git store (not copied into the repository)
    git show 23af713:series_manifest.json > <scratch>/manifest-v1/series_manifest.json
    git show 23af713:omega_values.json    > <scratch>/manifest-v1/omega_values.json

    # full history
    PYTHONPATH=src python -m decision_aware_tsfm.analysis.k2.replication \
        --manifest-version manifest-v1 \
        --manifest <scratch>/manifest-v1/series_manifest.json \
        --omega-values <scratch>/manifest-v1/omega_values.json \
        --omega-col omega_levels --allow-missing-omega --exclude-held-out-matches \
        --outdir <scratch>/runs/v1_full_history

    # last 4,096 steps
    (the same command with --omega-col omega_levels_4096 --outdir <scratch>/runs/v1_last_4096)

    # the saved computation, unchanged default (old single-file manifest)
    PYTHONPATH=src python -m decision_aware_tsfm.analysis.k2.replication \
        --outdir <scratch>/runs/default_before_py311

    # tests
    PYTHONPATH=src python -m unittest discover -s tests/unit/k2 -t .

Settings are the defaults of the saved run in all three: 10,000 bootstrap resamples, seed
20261004, results file `artifacts/reference/k2/gift_results.csv`. The tables in this note were
read from the run folders by `<scratch>/scripts/report_tables.py` (log:
`<scratch>/logs/report_tables.log`).

## Inputs (SHA-256, as recorded in each run manifest)

    gift_results.csv                 5bdf59d70d5e72c3c6593b194d68cecdb35d9f435b2109b42b8668054886c464
    dev_manifest.json (old)          a944b4a9b02ae2bc72e4ff6cd4fbe869a778d2ad3a2bc80498574919f7d35b4b
    series_manifest.json at 23af713  f20898d3954c2c60c600d764bcb97dc4b75da245716d537bec474d0c3a0f7094
    omega_values.json at 23af713     fd4ca6604b80dd3aaa9010c3ecdda27ed12cedcd6773482d7022651e46c44a96

The two manifest-v1 hashes are of the files as git stores them (87,571,416 and 4,263,022 bytes).
They are not the hashes in the data owner's message; see the next section. The manifest is
recorded as not frozen (the series file has no `frozen` key and declares no version).

## Check of the input against the data owner's statements

| Statement | Result |
|---|---|
| PR #48, branch `feat/1-omega-validation`, frozen at `23af713` | Confirmed. The pull request is open and its head is `23af713a23956cb3910cc6cb4d50f022b49de74d`, which is also the tip of the branch. |
| `series_manifest.json` has SHA-256 `6f876e27...` | Not confirmed. The file at `23af713` hashes to `f20898d3...` as stored, and to `6698e371...` with Windows line endings. The file is the same git object as at `5da6edd` (2026-09-26); neither later commit changed it. Its rows do not have the `frequency` and `freq_code` fields that the producing script at `23af713` writes, so the committed file was not produced by the committed script. The quoted hash is probably of a newer local file that was not pushed. |
| `omega_values.json` has SHA-256 `96e6df7b...` | Confirmed only after converting line endings: the file as stored hashes to `fd4ca660...`; with every line feed turned into carriage return plus line feed it hashes to `96e6df7b99484a5e7e35a3cba0f27ec2283aca7d09fb64f7ce0d0a2d3d9ccf62`. A hash taken on a Windows checkout will not match a checkout elsewhere. |
| Undefined Omega is null with a status field | Confirmed. Every Omega column has a `<column>_status` column; a value is present exactly when the status is `ok` (0 exceptions in 8,761 rows, four columns). |
| 66 `flat_history` on full history | Confirmed: `omega_levels` and `omega_diffs` each have 66 nulls, all `flat_history`. |
| 25 `empty_history` and 166 `flat_history` on the 4,096 columns | Confirmed: `omega_levels_4096` and `omega_diffs_4096` each have 191 nulls, 166 `flat_history` and 25 `empty_history`. The 25 are the rows with `history_length_4096` equal to 0. |
| Every other value is unchanged | Confirmed twice. Against `omega_values.json` at the previous commit `5da6edd`: in each of the four Omega columns every non-null value is equal and every null was 0.0 before; the two history columns are equal in all 8,761 rows. Against the old `dev_manifest.json`, matching rows by subset and position: 8,761 of 8,761 lengths equal; `omega_levels` and `omega_diffs` equal in 8,695 rows and null (0.0 before) in 66; largest absolute difference 0.0. |
| `omega_levels_4096` is included, with `omega_diffs_4096`, `history_start_index_4096`, `history_length_4096` | Confirmed, in every row. |
| The two files replace the old single-file `dev_manifest.json` | The producing script writes the two files. The old file is still in the tree at `23af713`, unchanged. |
| 8,761 unique active series ids, one-to-one join (cache owner's audit) | Confirmed: 8,761 rows and 8,761 distinct ids in each file, the same ids, in the same order. |
| No M5 or PJM rows (`guard_check.py`) | Confirmed for what that script checks (subset names). See "Held-out rule" for the series ids. |

Other facts about the files:

- `series_manifest.json` has the keys `active_dev_series` (8,761 rows), `excluded_series` (139,127
  rows, all with the reason "Below 1000 timestep floor"; K2 does not read them) and
  `held_out_strict` (2 names). `omega_values.json` has one key, `omega_values` (8,761 rows).
- 29 subsets and 29 configurations, one to one; `dataset` is `Salesforce/GiftEval` in every row.
  The subset names are the 29 the old manifest used.
- Against the old manifest: the 21 rows of the duplicate subset `jena_weather` are gone (the old
  reader already left them out), no series was added, and every series id changed (0 ids in
  common), so the comparison above had to be by subset and position.
- Values of exactly 0.0 remain, all with status `ok`: 10 in `omega_levels`, 2 in `omega_diffs`,
  10 in `omega_levels_4096`, 0 in `omega_diffs_4096`. The old manifest had 76 and 68 zeros in its
  two columns; 66 of each became null. Whether the remaining zeros are real values that round to
  0.0000 or histories the producing check does not recognise as degenerate cannot be told from
  the manifest.

Commands and outputs for this section: `<scratch>/scripts/inspect_v1.py`,
`compare_old_new.py`, `compare_prev_omega.py`, `heldout_hits_v1.py`, and the logs of the same
names in `<scratch>/logs/`.

## Held-out rule: 8 series ids match it, and none is M5 or PJM

The manifest lists no M5 or PJM entry: 0 rows match the rule in `subset`, `config`, `dataset` or
`item_id`. But 8 active rows (and 5 excluded rows) match it in `series_id`. All 8 are channel 5
of a multivariate series, all 8 ids end in `_dim5`, and none still matches when the `_dim<n>`
ending is removed. They are one series in each of `bizitobs_l2c_5T`, `bizitobs_l2c_H`, `ett1_15T`,
`ett1_H`, `ett2_15T`, `ett2_H`, `jena_weather_10T` and `jena_weather_H`. The values of these rows
were not looked at.

K2 checks every text field of a manifest row and refuses the manifest on a match. That rule was
added in the hardening on purpose and is unchanged: run without an option, K2 stops on
manifest-v1 and names the 8 ids (`<scratch>/logs/v1_refused_default.log`). To get any number,
this change adds an explicit opt-in, `--exclude-held-out-matches`, which leaves the matching rows
out before any of their values is used and prints and records them. Both runs below use it. So:

- 8,761 series were read and 8,753 used. `ett1`, `ett2` and `bizitobs_l2c` are computed on 12 of
  their 14 series and `jena_weather` on 40 of its 42.
- This is a false positive of the substring rule, produced by the id format. The clean fix is
  upstream (an id format that cannot contain a held-out token), or a team decision on the rule.
  Until then the numbers here are on 8 series fewer than the manifest intends.

In addition 33 names matched the punctuation-insensitive rule and were printed and kept, as that
rule prescribes: 22 series ids and 11 item ids of the form `item_5` and `item_50` to `item_59`,
all in `solar_10T` and `solar_H`.

## What changed in the code

Default behaviour is unchanged: the single-file layout `dev-v1` and
`artifacts/reference/manifests/dev_manifest.json` stay the default, and no statistic, metric,
grouping or default was touched.

| File | Change |
|---|---|
| `loading.py` | New layout `manifest-v1` in `read_manifest`, read by `_from_manifest_v1`: joins the series file and the Omega file on `series_id` after checking the join is one to one (`ManifestJoinError` otherwise); keeps null as undefined (NaN), never 0.0; refuses a row whose value and status disagree; applies the existing held-out checks to every text field of the joined rows; returns the same canonical table as the old reader, plus the status columns. `aggregate_omega` adds, only when a status column exists, the count of undefined values by status (overall and per base dataset), the base datasets kept on fewer than half of their series, and the count of values that are exactly 0.0. Subsets are matched through the existing `SUBSET_MAP`. |
| `pipeline.py` | `run_k2` takes `omega_values_path` and `exclude_held_out`; the Omega file is hashed into `inputs.omega_values` of the run manifest; the banner prints the status of undefined values, left-out held-out matches and exact zeros. |
| `cli.py` | Options `--omega-values` and `--exclude-held-out-matches`; `manifest-v1` becomes a choice of `--manifest-version`. Both options are refused with any other layout. |
| `tests/unit/k2/test_k2_manifest_v1.py` | 33 new tests on hand-built two-file fixtures. |

How undefined and zero values are handled:

- null with a status: the value stays undefined. By default the run stops and lists the base
  datasets affected with the count by status (the existing behaviour for a missing value). With
  `--allow-missing-omega` the series is left out of its dataset's Omega mean, spread and series
  count, and the count by status is printed and written to `omega_report` in the run manifest.
- A base dataset with no defined series is removed and printed (existing behaviour). A base
  dataset kept on fewer than half of its series is listed. Neither happened in these runs.
- 0.0 with status `ok` is used as a value and counted and printed. The decision engine refuses
  an Omega of exactly 0.0 because in the old manifest 0.0 could mean "undefined"; in this layout
  the status says which it is. See "Design choices".
- In `joined_table.csv` the Omega columns other than the selected one are plain means over their
  own defined values; their undefined counts are in `manifest_report.omega_undefined_by_column`.

## Results

Delta = 100 (A - B) / A with A the zero-shot family, family score = mean across the family's
models, one row per base dataset, Omega averaged over the dataset's series that have a value.
Negative rho means the zero-shot advantage grows with Omega. Interval: 95% percentile bootstrap,
10,000 resamples of the 20 datasets, seed 20261004; no resample was degenerate in any run. "Old"
is the saved computation on `dev_manifest.json`, rerun here (`<scratch>/runs/default_before_py311`);
"v1 full" is `<scratch>/runs/v1_full_history`; "v1 4,096" is `<scratch>/runs/v1_last_4096`.

Series behind each run: old 8,761 (0 undefined); v1 full 8,687 (8,753 used, 66 undefined, all
`flat_history`: 56 in `bitbrains_fast_storage`, 10 in `bitbrains_rnd`); v1 4,096 8,562 (8,753
used, 191 undefined: `bitbrains_fast_storage` 132 `flat_history` and 18 `empty_history`,
`bitbrains_rnd` 28 `flat_history`, `bizitobs_service` 4 `flat_history`, `electricity` 2
`flat_history`, `kdd_cup_2018` 7 `empty_history`).

### 1. Headline: family mean, dataset level

| Metric | Zero-shot versus | Old manifest | manifest-v1, full history | manifest-v1, last 4,096 steps |
|---|---|---|---|---|
| sMAPE | deep | -0.417 [-0.793, +0.091], n = 20 | -0.417 [-0.793, +0.091], n = 20 | -0.426 [-0.786, +0.089], n = 20 |
| sMAPE | statistical | -0.123 [-0.603, +0.397], n = 20 | -0.123 [-0.603, +0.397], n = 20 | -0.174 [-0.629, +0.338], n = 20 |
| sMAPE | pretrained | -0.514 [-0.800, -0.054], n = 20 | -0.514 [-0.800, -0.054], n = 20 | -0.537 [-0.824, -0.090], n = 20 |
| MSE (MSE[mean]; see caveat) | deep | -0.556 [-0.828, -0.123], n = 20 | -0.556 [-0.828, -0.123], n = 20 | -0.522 [-0.810, -0.077], n = 20 |
| MSE (MSE[mean]; see caveat) | statistical | -0.398 [-0.807, +0.101], n = 20 | -0.398 [-0.807, +0.101], n = 20 | -0.388 [-0.804, +0.123], n = 20 |
| MSE (MSE[mean]; see caveat) | pretrained | -0.520 [-0.813, -0.097], n = 20 | -0.520 [-0.813, -0.097], n = 20 | -0.498 [-0.791, -0.063], n = 20 |
| MAE (2 x pinball at tau 0.5) | deep | -0.248 [-0.681, +0.256], n = 20 | -0.248 [-0.681, +0.256], n = 20 | -0.253 [-0.677, +0.273], n = 20 |
| MAE (2 x pinball at tau 0.5) | statistical | +0.251 [-0.315, +0.684], n = 20 | +0.251 [-0.315, +0.684], n = 20 | +0.117 [-0.427, +0.587], n = 20 |
| MAE (2 x pinball at tau 0.5) | pretrained | -0.405 [-0.789, +0.103], n = 20 | -0.405 [-0.789, +0.103], n = 20 | -0.403 [-0.799, +0.108], n = 20 |
| CRPS | deep | -0.280 [-0.718, +0.265], n = 20 | -0.280 [-0.718, +0.265], n = 20 | -0.299 [-0.746, +0.248], n = 20 |
| CRPS | statistical | +0.376 [-0.132, +0.750], n = 20 | +0.376 [-0.132, +0.750], n = 20 | +0.325 [-0.207, +0.720], n = 20 |
| CRPS | pretrained | -0.415 [-0.812, +0.097], n = 20 | -0.415 [-0.812, +0.097], n = 20 | -0.429 [-0.829, +0.084], n = 20 |

Two-sided Spearman p-values, same order (old and v1 full are equal; v1 4,096 in brackets):
0.0677 (0.0614), 0.6045 (0.4620), 0.0203 (0.0147), 0.0108 (0.0183), 0.0818 (0.0910), 0.0187
(0.0255), 0.2915 (0.2826), 0.2855 (0.6224), 0.0769 (0.0781), 0.2323 (0.1999), 0.1023 (0.1623),
0.0688 (0.0594).

What the numbers show, and no more:

- Old against v1 full: identical in all 60 cells of `correlations_dataset_level.csv` (12
  comparisons by 5 within-family summaries), rho and interval ends; the largest difference is
  exactly 0.0. Six datasets have a different Omega (next section) but the order of the 20
  datasets is the same, and a rank correlation depends only on the order.
- v1 full against v1 4,096: every headline cell keeps its sign. The largest change is 0.134 (MAE
  versus statistical); the next are 0.051 (sMAPE versus statistical) and 0.051 (CRPS versus
  statistical). The three intervals that excluded zero still do (sMAPE versus pretrained, MSE
  versus deep, MSE versus pretrained) and the nine that included zero still do.
- The count of datasets on which the zero-shot family has the lower error does not depend on
  Omega and is the same in all three runs: sMAPE 20, 15, 20; MSE 8, 5, 6; MAE 19, 18, 19; CRPS
  20, 19, 20 (deep, statistical, pretrained; of 20).
- n is 20 in every cell of every run.

### 2. Like for like with the paper (sMAPE)

| Zero-shot versus | Published (sMAPE, 28 datasets) | Old | v1 full | v1 4,096 | Published value inside the interval |
|---|---|---|---|---|---|
| deep | -0.556 | -0.417 [-0.793, +0.091] | -0.417 [-0.793, +0.091] | -0.426 [-0.786, +0.089] | yes, in all three |
| statistical | -0.234 | -0.123 [-0.603, +0.397] | -0.123 [-0.603, +0.397] | -0.174 [-0.629, +0.338] | yes, in all three |
| pretrained | not given | -0.514 [-0.800, -0.054] | -0.514 [-0.800, -0.054] | -0.537 [-0.824, -0.090] | not applicable |

On either Omega column the replication is weaker than published in both comparisons that have a
published value, both published values lie inside the interval, and for deep and statistical so
does zero. With 20 datasets this can neither confirm nor contradict the published magnitudes;
the 4,096 column does not change that.

### 3. Within-family summaries (rho, 95% interval; n = 20 in every cell)

Old manifest, and manifest-v1 full history (identical):

| Metric | Versus | Mean | Median, rows (recorded) | Best, rows (recorded) | Median, models | Best, models |
|---|---|---|---|---|---|---|
| sMAPE | deep | -0.417 [-0.79, +0.09] | +0.045 [-0.50, +0.58] | -0.063 [-0.49, +0.35] | -0.105 [-0.59, +0.41] | +0.009 [-0.47, +0.47] |
| sMAPE | statistical | -0.123 [-0.60, +0.40] | +0.056 [-0.42, +0.57] | -0.101 [-0.56, +0.39] | -0.179 [-0.62, +0.31] | -0.074 [-0.55, +0.46] |
| sMAPE | pretrained | -0.514 [-0.80, -0.05] | +0.164 [-0.46, +0.65] | -0.248 [-0.65, +0.22] | +0.129 [-0.41, +0.59] | +0.009 [-0.49, +0.48] |
| MSE | deep | -0.556 [-0.83, -0.12] | -0.254 [-0.67, +0.31] | -0.349 [-0.68, +0.12] | -0.113 [-0.55, +0.43] | -0.078 [-0.53, +0.37] |
| MSE | statistical | -0.398 [-0.81, +0.10] | +0.047 [-0.45, +0.50] | -0.051 [-0.48, +0.38] | +0.208 [-0.25, +0.56] | +0.134 [-0.36, +0.54] |
| MSE | pretrained | -0.520 [-0.81, -0.10] | -0.104 [-0.55, +0.40] | +0.000 [-0.48, +0.49] | +0.075 [-0.46, +0.52] | -0.203 [-0.56, +0.21] |
| MAE | deep | -0.248 [-0.68, +0.26] | -0.119 [-0.55, +0.39] | -0.047 [-0.50, +0.43] | -0.220 [-0.65, +0.32] | +0.158 [-0.29, +0.54] |
| MAE | statistical | +0.251 [-0.32, +0.68] | +0.253 [-0.30, +0.68] | -0.033 [-0.57, +0.46] | +0.314 [-0.19, +0.67] | +0.209 [-0.27, +0.62] |
| MAE | pretrained | -0.405 [-0.79, +0.10] | -0.104 [-0.61, +0.42] | -0.060 [-0.54, +0.38] | +0.105 [-0.45, +0.56] | -0.117 [-0.52, +0.31] |
| CRPS | deep | -0.280 [-0.72, +0.26] | -0.307 [-0.65, +0.15] | -0.188 [-0.56, +0.25] | -0.317 [-0.69, +0.18] | -0.044 [-0.47, +0.39] |
| CRPS | statistical | +0.376 [-0.13, +0.75] | +0.388 [-0.12, +0.75] | +0.532 [+0.17, +0.75] | +0.183 [-0.37, +0.63] | +0.313 [-0.22, +0.67] |
| CRPS | pretrained | -0.415 [-0.81, +0.10] | -0.101 [-0.68, +0.44] | -0.102 [-0.53, +0.34] | +0.086 [-0.46, +0.58] | -0.096 [-0.56, +0.37] |

manifest-v1, last 4,096 steps:

| Metric | Versus | Mean | Median, rows (recorded) | Best, rows (recorded) | Median, models | Best, models |
|---|---|---|---|---|---|---|
| sMAPE | deep | -0.426 [-0.79, +0.09] | +0.033 [-0.50, +0.58] | -0.143 [-0.53, +0.28] | -0.095 [-0.57, +0.42] | +0.042 [-0.42, +0.49] |
| sMAPE | statistical | -0.174 [-0.63, +0.34] | +0.017 [-0.45, +0.52] | -0.114 [-0.57, +0.36] | -0.206 [-0.65, +0.28] | -0.092 [-0.55, +0.42] |
| sMAPE | pretrained | -0.537 [-0.82, -0.09] | +0.137 [-0.46, +0.63] | -0.268 [-0.69, +0.21] | +0.132 [-0.40, +0.60] | +0.023 [-0.47, +0.48] |
| MSE | deep | -0.522 [-0.81, -0.08] | -0.250 [-0.68, +0.31] | -0.480 [-0.73, -0.05] | -0.134 [-0.56, +0.40] | -0.077 [-0.51, +0.38] |
| MSE | statistical | -0.388 [-0.80, +0.12] | +0.024 [-0.49, +0.50] | -0.066 [-0.50, +0.38] | +0.050 [-0.38, +0.45] | +0.077 [-0.44, +0.51] |
| MSE | pretrained | -0.498 [-0.79, -0.06] | -0.168 [-0.61, +0.34] | +0.011 [-0.47, +0.52] | -0.029 [-0.51, +0.42] | -0.268 [-0.63, +0.15] |
| MAE | deep | -0.253 [-0.68, +0.27] | -0.144 [-0.58, +0.36] | -0.126 [-0.54, +0.34] | -0.253 [-0.66, +0.28] | +0.137 [-0.29, +0.52] |
| MAE | statistical | +0.117 [-0.43, +0.59] | +0.226 [-0.35, +0.65] | -0.036 [-0.60, +0.48] | +0.167 [-0.32, +0.58] | +0.144 [-0.36, +0.58] |
| MAE | pretrained | -0.403 [-0.80, +0.11] | -0.144 [-0.65, +0.38] | -0.017 [-0.47, +0.44] | -0.012 [-0.51, +0.45] | -0.167 [-0.59, +0.27] |
| CRPS | deep | -0.299 [-0.75, +0.25] | -0.320 [-0.67, +0.11] | -0.302 [-0.61, +0.12] | -0.268 [-0.66, +0.23] | -0.060 [-0.48, +0.39] |
| CRPS | statistical | +0.325 [-0.21, +0.72] | +0.380 [-0.14, +0.71] | +0.463 [+0.05, +0.74] | +0.168 [-0.40, +0.60] | +0.301 [-0.26, +0.68] |
| CRPS | pretrained | -0.429 [-0.83, +0.08] | -0.065 [-0.66, +0.48] | -0.099 [-0.54, +0.37] | +0.078 [-0.49, +0.59] | -0.074 [-0.56, +0.40] |

The pattern of the hardening note holds on the 4,096 column for sMAPE versus deep: -0.426 under
the family mean; +0.033 and -0.143 under the recorded median and best; -0.095 and +0.042 under
the model-level median and best. Over the 60 cells the largest change between the two columns is
0.158 (MSE versus statistical, median over models, +0.208 to +0.050). Two cells change sign,
both under the median over models and both close to zero on either column: MSE versus pretrained
(+0.075 to -0.029) and MAE versus pretrained (+0.105 to -0.012). One cell that did not exclude
zero on full history does on 4,096: MSE versus deep under the recorded "best, rows" summary,
-0.349 [-0.68, +0.12] to -0.480 [-0.73, -0.05]; that summary is the single lowest
model-configuration row, on the MSE[mean] column. No other cell changes between excluding and
including zero (4 of 60 exclude zero on full history, 5 on 4,096). The MSE rows of these tables
carry the caveat below.

### 4. MSE sensitivity (the MSE rows must not be cited without this table)

The model with the largest median share of the zero-shot family's summed MSE[mean] is
`Toto_Open_Base_1.0` in all three runs, chosen by the same rule. rho and 95% interval; the last
column is the number of datasets (of 20) on which the zero-shot family has the lower error, the
same in all three runs.

| MSE variant | Versus | Old | v1 full | v1 4,096 | Zero-shot lower |
|---|---|---|---|---|---|
| As recorded: MSE[mean] | deep | -0.556 [-0.828, -0.123] | -0.556 [-0.828, -0.123] | -0.522 [-0.810, -0.077] | 8 |
| As recorded: MSE[mean] | statistical | -0.398 [-0.807, +0.101] | -0.398 [-0.807, +0.101] | -0.388 [-0.804, +0.123] | 5 |
| As recorded: MSE[mean] | pretrained | -0.520 [-0.813, -0.097] | -0.520 [-0.813, -0.097] | -0.498 [-0.791, -0.063] | 6 |
| MSE[mean], dominant model removed | deep | -0.454 [-0.773, +0.024] | -0.454 [-0.773, +0.024] | -0.438 [-0.771, +0.048] | 15 |
| MSE[mean], dominant model removed | statistical | -0.164 [-0.607, +0.338] | -0.164 [-0.607, +0.338] | -0.232 [-0.649, +0.271] | 14 |
| MSE[mean], dominant model removed | pretrained | -0.654 [-0.842, -0.305] | -0.654 [-0.842, -0.305] | -0.589 [-0.824, -0.189] | 16 |
| MSE[0.5] column | deep | -0.347 [-0.708, +0.144] | -0.347 [-0.708, +0.144] | -0.325 [-0.703, +0.180] | 17 |
| MSE[0.5] column | statistical | +0.250 [-0.249, +0.635] | +0.250 [-0.249, +0.635] | +0.123 [-0.357, +0.547] | 18 |
| MSE[0.5] column | pretrained | -0.526 [-0.768, -0.107] | -0.526 [-0.768, -0.107] | -0.516 [-0.771, -0.090] | 19 |
| MSE[0.5] column, dominant model removed | deep | -0.334 [-0.702, +0.173] | -0.334 [-0.702, +0.173] | -0.313 [-0.696, +0.207] | 17 |
| MSE[0.5] column, dominant model removed | statistical | +0.242 [-0.256, +0.629] | +0.242 [-0.256, +0.629] | +0.116 [-0.361, +0.537] | 18 |
| MSE[0.5] column, dominant model removed | pretrained | -0.481 [-0.775, -0.028] | -0.481 [-0.775, -0.028] | -0.471 [-0.779, -0.014] | 19 |
| MAE[0.5], for comparison | deep | -0.248 [-0.681, +0.256] | -0.248 [-0.681, +0.256] | -0.253 [-0.677, +0.273] | 19 |
| MAE[0.5], for comparison | statistical | +0.251 [-0.315, +0.684] | +0.251 [-0.315, +0.684] | +0.117 [-0.427, +0.587] | 18 |
| MAE[0.5], for comparison | pretrained | -0.405 [-0.789, +0.103] | -0.405 [-0.789, +0.103] | -0.403 [-0.799, +0.108] | 19 |

On the 4,096 column the statements of the hardening note about MSE stay as they were: versus
deep the interval excludes zero only as recorded (-0.522) and includes it with the dominant
model removed (-0.438) and on MSE[0.5] (-0.325); versus statistical the sign on MSE[0.5] is the
MAE sign (+0.123 and +0.117); versus pretrained the interval excludes zero in every variant.

### 5. Leave one model out

Removing one model of the comparison family at a time. Each cell: rho with all models; range of
rho over the removals; largest change and the model removed; number of sign changes; number of
removals after which the interval changes between excluding and including zero.

| Metric | Versus | Old and v1 full (identical) | v1 4,096 |
|---|---|---|---|
| sMAPE | deep | -0.417; -0.456 to +0.195; 0.612 (`Crossformer`); 1; 0 | -0.426; -0.463 to +0.224; 0.650 (`Crossformer`); 1; 0 |
| sMAPE | statistical | -0.123; -0.141 to -0.026; 0.098 (`Auto_Arima`); 0; 0 | -0.174; -0.183 to -0.053; 0.122 (`Auto_Arima`); 0; 0 |
| sMAPE | pretrained | -0.514; -0.605 to -0.048; 0.466 (`Lag-Llama`); 0; 2 | -0.537; -0.623 to -0.059; 0.478 (`Lag-Llama`); 0; 2 |
| MSE | deep | -0.556; -0.556 to -0.371; 0.185 (`Crossformer`); 0; 1 | -0.522; -0.522 to -0.298; 0.224 (`Crossformer`); 0; 1 |
| MSE | statistical | -0.398; -0.426 to -0.340; 0.059 (`Auto_Arima`); 0; 0 | -0.388; -0.415 to -0.320; 0.068 (`Auto_Arima`); 0; 0 |
| MSE | pretrained | -0.520; -0.520 to -0.459; 0.062 (`Lag-Llama`); 0; 0 | -0.498; -0.498 to -0.438; 0.060 (`Lag-Llama`); 0; 1 |
| MAE | deep | -0.248; -0.268 to +0.117; 0.365 (`Crossformer`); 1; 0 | -0.253; -0.290 to +0.108; 0.361 (`Crossformer`); 1; 0 |
| MAE | statistical | +0.251; +0.223 to +0.408; 0.156 (`Auto_Arima`); 0; 0 | +0.117; +0.083 to +0.283; 0.165 (`Auto_Arima`); 0; 0 |
| MAE | pretrained | -0.405; -0.511 to +0.012; 0.417 (`Lag-Llama`); 1; 1 | -0.403; -0.507 to -0.071; 0.332 (`Lag-Llama`); 0; 1 |
| CRPS | deep | -0.280; -0.299 to -0.030; 0.250 (`Crossformer`); 0; 0 | -0.299; -0.320 to +0.003; 0.302 (`Crossformer`); 1; 0 |
| CRPS | statistical | +0.376; +0.221 to +0.450; 0.155 (`Auto_ETS`); 0; 0 | +0.325; +0.194 to +0.423; 0.131 (`Auto_ETS`); 0; 0 |
| CRPS | pretrained | -0.415; -0.496 to -0.078; 0.337 (`Lag-Llama`); 0; 0 | -0.429; -0.507 to -0.083; 0.346 (`Lag-Llama`); 0; 0 |
| MSE[0.5] | deep | -0.347; -0.364 to +0.098; 0.445 (`Crossformer`); 1; 0 | -0.325; -0.358 to +0.095; 0.420 (`Crossformer`); 1; 0 |
| MSE[0.5] | statistical | +0.250; +0.165 to +0.314; 0.084 (`Auto_ETS`); 0; 0 | +0.123; +0.053 to +0.188; 0.071 (`Auto_ETS`); 0; 0 |
| MSE[0.5] | pretrained | -0.526; -0.588 to -0.023; 0.504 (`Lag-Llama`); 0; 1 | -0.516; -0.576 to -0.048; 0.468 (`Lag-Llama`); 0; 1 |

Removing one zero-shot model at a time (18 removals per row), same cell layout:

| Metric | Versus | Old and v1 full (identical) | v1 4,096 |
|---|---|---|---|
| sMAPE | deep | -0.417; -0.468 to -0.374; 0.051 (`Moirai_base`); 0; 0 | -0.426; -0.477 to -0.392; 0.051 (`Moirai_base`); 0; 0 |
| sMAPE | statistical | -0.123; -0.135 to -0.116; 0.012 (`Moirai_base`); 0; 0 | -0.174; -0.177 to -0.167; 0.008 (`VisionTS`); 0; 0 |
| sMAPE | pretrained | -0.514; -0.576 to -0.495; 0.062 (`YingLong_6m`); 0; 0 | -0.537; -0.603 to -0.517; 0.066 (`YingLong_6m`); 0; 0 |
| MSE | deep | -0.556; -0.556 to -0.454; 0.102 (`Toto_Open_Base_1.0`); 0; 1 | -0.522; -0.529 to -0.438; 0.084 (`Toto_Open_Base_1.0`); 0; 1 |
| MSE | statistical | -0.398; -0.426 to -0.164; 0.235 (`Toto_Open_Base_1.0`); 0; 0 | -0.388; -0.429 to -0.232; 0.156 (`Toto_Open_Base_1.0`); 0; 0 |
| MSE | pretrained | -0.520; -0.654 to -0.514; 0.134 (`Toto_Open_Base_1.0`); 0; 0 | -0.498; -0.589 to -0.498; 0.092 (`Toto_Open_Base_1.0`); 0; 0 |
| MAE | deep | -0.248; -0.298 to -0.227; 0.050 (`Moirai_base`); 0; 0 | -0.253; -0.310 to -0.232; 0.057 (`Moirai_base`); 0; 0 |
| MAE | statistical | +0.251; +0.212 to +0.281; 0.039 (`Moirai_small`); 0; 0 | +0.117; +0.075 to +0.147; 0.042 (`Moirai_base`); 0; 0 |
| MAE | pretrained | -0.405; -0.505 to -0.347; 0.101 (`Moirai_base`); 0; 1 | -0.403; -0.505 to -0.362; 0.102 (`Moirai_base`); 0; 1 |
| CRPS | deep | -0.280; -0.317 to -0.253; 0.038 (`Moirai_base`); 0; 0 | -0.299; -0.338 to -0.275; 0.039 (`Moirai_base`); 0; 0 |
| CRPS | statistical | +0.376; +0.305 to +0.380; 0.071 (`Moirai_base`); 0; 0 | +0.325; +0.250 to +0.337; 0.075 (`Moirai_base`); 0; 0 |
| CRPS | pretrained | -0.415; -0.456 to -0.382; 0.041 (`Moirai_base`); 0; 0 | -0.429; -0.463 to -0.389; 0.039 (`TabPFN-TS`); 0; 0 |
| MSE[0.5] | deep | -0.347; -0.415 to -0.334; 0.068 (`Moirai_base`); 0; 0 | -0.325; -0.406 to -0.313; 0.081 (`Moirai_base`); 0; 0 |
| MSE[0.5] | statistical | +0.250; +0.214 to +0.274; 0.036 (`Moirai_small`); 0; 0 | +0.123; +0.090 to +0.147; 0.033 (`Moirai_small`); 0; 0 |
| MSE[0.5] | pretrained | -0.526; -0.603 to -0.463; 0.077 (`Moirai_large`); 0; 0 | -0.516; -0.592 to -0.457; 0.077 (`Moirai_large`); 0; 2 |

The two dependences on a single comparison model that the hardening note reports are still
there on the 4,096 column: sMAPE versus deep goes from -0.426 to +0.224 [-0.251, +0.641] without
`Crossformer` (old and v1 full: -0.417 to +0.195 [-0.272, +0.597]); sMAPE versus pretrained goes
from -0.537 to -0.059 [-0.602, +0.448] without `Lag-Llama` (old and v1 full: -0.514 to -0.048
[-0.580, +0.464]). No model was removed from any headline number.

### 6. Configuration level (sensitivity only), family mean

One row per GIFT-Eval configuration; the interval resamples the 20 base datasets; the p-value
treats configurations as independent, which they are not. 71 of 97 configurations join under the
dataset-and-frequency rule and 89 under the dataset-broadcast rule, in all three runs.

Dataset and frequency (n = 71):

| Metric | Versus | Old | v1 full | v1 4,096 |
|---|---|---|---|---|
| sMAPE | deep | -0.165 [-0.615, +0.227], p = 0.168 | -0.115 [-0.622, +0.296], p = 0.339 | -0.082 [-0.569, +0.304], p = 0.494 |
| sMAPE | statistical | -0.089 [-0.510, +0.402], p = 0.459 | -0.107 [-0.520, +0.394], p = 0.373 | -0.138 [-0.506, +0.310], p = 0.252 |
| sMAPE | pretrained | -0.337 [-0.638, -0.024], p = 0.004 | -0.349 [-0.634, -0.022], p = 0.003 | -0.436 [-0.680, -0.123], p < 0.001 |
| MSE | deep | -0.191 [-0.494, +0.110], p = 0.110 | -0.186 [-0.486, +0.112], p = 0.120 | -0.211 [-0.485, +0.060], p = 0.077 |
| MSE | statistical | -0.163 [-0.439, +0.139], p = 0.173 | -0.163 [-0.430, +0.137], p = 0.174 | -0.183 [-0.428, +0.091], p = 0.126 |
| MSE | pretrained | -0.242 [-0.553, +0.087], p = 0.042 | -0.244 [-0.541, +0.080], p = 0.040 | -0.274 [-0.545, +0.019], p = 0.021 |
| MAE | deep | -0.097 [-0.520, +0.271], p = 0.421 | -0.066 [-0.513, +0.304], p = 0.583 | -0.088 [-0.534, +0.304], p = 0.463 |
| MAE | statistical | -0.151 [-0.404, +0.191], p = 0.208 | -0.182 [-0.416, +0.173], p = 0.129 | -0.233 [-0.450, +0.112], p = 0.051 |
| MAE | pretrained | -0.375 [-0.660, -0.058], p = 0.001 | -0.400 [-0.664, -0.073], p = 0.001 | -0.488 [-0.715, -0.187], p < 0.001 |
| CRPS | deep | -0.054 [-0.413, +0.251], p = 0.653 | -0.029 [-0.395, +0.269], p = 0.813 | -0.055 [-0.404, +0.236], p = 0.648 |
| CRPS | statistical | +0.133 [-0.191, +0.482], p = 0.267 | +0.127 [-0.198, +0.472], p = 0.290 | +0.079 [-0.253, +0.416], p = 0.514 |
| CRPS | pretrained | -0.306 [-0.595, -0.026], p = 0.010 | -0.329 [-0.599, -0.031], p = 0.005 | -0.389 [-0.633, -0.092], p = 0.001 |

Dataset broadcast (n = 89):

| Metric | Versus | Old and v1 full (identical) | v1 4,096 |
|---|---|---|---|
| sMAPE | deep | -0.088 [-0.519, +0.266], p = 0.411 | -0.054 [-0.515, +0.316], p = 0.614 |
| sMAPE | statistical | -0.058 [-0.516, +0.432], p = 0.588 | -0.133 [-0.539, +0.337], p = 0.214 |
| sMAPE | pretrained | -0.328 [-0.591, -0.017], p = 0.002 | -0.380 [-0.635, -0.056], p < 0.001 |
| MSE | deep | -0.199 [-0.470, +0.061], p = 0.062 | -0.225 [-0.480, +0.024], p = 0.034 |
| MSE | statistical | -0.141 [-0.409, +0.127], p = 0.188 | -0.197 [-0.434, +0.054], p = 0.064 |
| MSE | pretrained | -0.274 [-0.541, -0.022], p = 0.009 | -0.304 [-0.548, -0.053], p = 0.004 |
| MAE | deep | -0.069 [-0.474, +0.299], p = 0.518 | -0.124 [-0.547, +0.265], p = 0.248 |
| MAE | statistical | -0.113 [-0.377, +0.211], p = 0.293 | -0.253 [-0.461, +0.036], p = 0.017 |
| MAE | pretrained | -0.398 [-0.650, -0.067], p < 0.001 | -0.466 [-0.693, -0.149], p < 0.001 |
| CRPS | deep | -0.051 [-0.369, +0.243], p = 0.633 | -0.086 [-0.427, +0.219], p = 0.422 |
| CRPS | statistical | +0.200 [-0.050, +0.442], p = 0.061 | +0.095 [-0.137, +0.339], p = 0.374 |
| CRPS | pretrained | -0.363 [-0.625, -0.057], p < 0.001 | -0.421 [-0.659, -0.112], p < 0.001 |

Under the dataset-and-frequency rule the v1 full column differs from the old one, although the
Omega values are the same. The side check in section 8 shows that this difference comes almost
entirely from the 8 rows the held-out rule leaves out, not from the change to null.

## 7. The 4,096 sensitivity itself

Omega per base dataset (mean over the series with a value), from `dataset_deltas.csv` of each
run, sorted by full-history Omega. Rank 1 is the lowest Omega.

| Base dataset | Series, old | Omega, old | Series, v1 full | Omega, v1 full | v1 full minus old | Series, v1 4,096 | Omega, v1 4,096 | 4,096 minus full | Rank, full | Rank, 4,096 | Rank change |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hierarchical_sales | 118 | 0.1144 | 118 | 0.1144 | +0.0000 | 118 | 0.1144 | +0.0000 | 1 | 1 | 0 |
| bitbrains_rnd | 936 | 0.2587 | 926 | 0.2615 | +0.0028 | 908 | 0.2605 | -0.0010 | 2 | 2 | 0 |
| bitbrains_fast_storage | 2432 | 0.2734 | 2376 | 0.2798 | +0.0064 | 2282 | 0.2719 | -0.0080 | 3 | 3 | 0 |
| saugeen | 2 | 0.2903 | 2 | 0.2903 | +0.0000 | 2 | 0.2861 | -0.0042 | 4 | 4 | 0 |
| sz_taxi | 156 | 0.3070 | 156 | 0.3070 | +0.0000 | 156 | 0.3070 | +0.0000 | 5 | 5 | 0 |
| kdd_cup_2018 | 269 | 0.3168 | 269 | 0.3168 | +0.0000 | 262 | 0.3164 | -0.0004 | 6 | 6 | 0 |
| loop_seattle | 646 | 0.4368 | 646 | 0.4368 | +0.0000 | 646 | 0.4216 | -0.0152 | 7 | 7 | 0 |
| jena_weather | 42 | 0.4888 | 40 | 0.4813 | -0.0075 | 40 | 0.5110 | +0.0297 | 8 | 10 | +2 |
| ett1 | 14 | 0.5475 | 12 | 0.5439 | -0.0037 | 12 | 0.4882 | -0.0556 | 9 | 9 | 0 |
| m_dense | 30 | 0.5821 | 30 | 0.5821 | +0.0000 | 30 | 0.5389 | -0.0432 | 10 | 11 | +1 |
| electricity | 897 | 0.6126 | 897 | 0.6126 | +0.0000 | 895 | 0.5932 | -0.0195 | 11 | 13 | +2 |
| ett2 | 14 | 0.6211 | 12 | 0.6140 | -0.0071 | 12 | 0.4447 | -0.1693 | 12 | 8 | -4 |
| us_births | 1 | 0.6259 | 1 | 0.6259 | +0.0000 | 1 | 0.5918 | -0.0341 | 13 | 12 | -1 |
| bizitobs_service | 42 | 0.6381 | 42 | 0.6381 | +0.0000 | 38 | 0.6157 | -0.0224 | 14 | 14 | 0 |
| bizitobs_l2c | 14 | 0.6711 | 12 | 0.6813 | +0.0101 | 12 | 0.6750 | -0.0063 | 15 | 16 | +1 |
| m4_monthly | 46 | 0.6845 | 46 | 0.6845 | +0.0000 | 46 | 0.6845 | +0.0000 | 16 | 17 | +1 |
| bizitobs_application | 2 | 0.7125 | 2 | 0.7125 | +0.0000 | 2 | 0.7032 | -0.0093 | 17 | 18 | +1 |
| solar | 274 | 0.7254 | 274 | 0.7254 | +0.0000 | 274 | 0.6630 | -0.0625 | 18 | 15 | -3 |
| m4_weekly | 168 | 0.7339 | 168 | 0.7339 | +0.0000 | 168 | 0.7339 | +0.0000 | 19 | 19 | 0 |
| m4_daily | 2658 | 0.7508 | 2658 | 0.7508 | +0.0000 | 2658 | 0.7509 | +0.0001 | 20 | 20 | 0 |

- Difference, 4,096 minus full history: from -0.1693 (`ett2`) to +0.0297 (`jena_weather`); median
  -0.0071, mean -0.0211, mean absolute difference 0.0240. Omega is lower on the 4,096 column in
  14 datasets, higher in 2 (`jena_weather`, `m4_daily`) and equal in 4 (`hierarchical_sales`,
  `sz_taxi`, `m4_monthly`, `m4_weekly`).
- Rank: 9 of 20 datasets change rank; the largest moves are `ett2` (12 to 8) and `solar` (18 to
  15). The 7 datasets with the lowest Omega keep their ranks, as do the top two.
- The 0.4 mark: no dataset crosses it. 14 datasets are at or above 0.4 on both columns (and on
  the old manifest).
- The 0.5 mark: 3 datasets cross it. `ett1` (0.5439 to 0.4882) and `ett2` (0.6140 to 0.4447) go
  below; `jena_weather` (0.4813 to 0.5110) goes above. 12 datasets are at or above 0.5 on full
  history and 11 on the 4,096 column. All three datasets that cross are among the four that are
  one series short per subset because of the held-out rule, so these three crossings in
  particular must be rechecked once the ids are fixed.
- Rank correlation between the two Omega columns across the 20 datasets: Spearman 0.9714
  (Pearson 0.9774).
- Between the old manifest and v1 full history: 6 datasets have a different Omega and 14 are
  equal; the order is the same (Spearman 1.0000) and no dataset crosses 0.4 or 0.5.

## 8. Which datasets entered or left, and why values moved

No dataset entered or left the analysis. The same 20 base datasets and the same 29
dataset-frequency subsets are analysed in all three runs; n = 20 everywhere. What changed is the
number of series behind some datasets:

| Cause | Datasets | Series |
|---|---|---|
| Undefined on full history (null, `flat_history`), left out of the mean; these were 0.0 in the old manifest and were averaged in | `bitbrains_fast_storage` (56 of 2,432), `bitbrains_rnd` (10 of 936) | 66 |
| Left out by the held-out rule (`_dim5` ids) | `ett1`, `ett2`, `bizitobs_l2c` (2 of 14 each), `jena_weather` (2 of 42) | 8 |
| Undefined on the 4,096 column (the 4,096 run only) | `bitbrains_fast_storage` (150), `bitbrains_rnd` (28), `kdd_cup_2018` (7), `bizitobs_service` (4), `electricity` (2) | 191 in total, in place of the 66 |

Side check 1, the two causes apart. A scratch copy of the old manifest was made in which the 66
series are null instead of 0.0 and nothing else is changed (no row renamed or removed), and the
default layout was run on it (`<scratch>/runs/decomp_old_flat_as_null`, input
`<scratch>/decomp/old_flat_as_null.json`).

- Null instead of 0.0, alone: Omega of `bitbrains_fast_storage` +0.0064 and `bitbrains_rnd`
  +0.0028; no other dataset moves. All 60 dataset-level cells are identical to the old run. At
  configuration level under the dataset-and-frequency rule the largest change in rho over the 36
  cells is 0.0054.
- The 8 left-out rows, on top of that: Omega of `bizitobs_l2c` +0.0101, `ett1` -0.0037, `ett2`
  -0.0071, `jena_weather` -0.0075. All 60 dataset-level cells are still identical. At
  configuration level under the dataset-and-frequency rule the largest change in rho is 0.0611
  (for example sMAPE versus deep: -0.165 old, -0.163 with null alone, -0.115 with the 8 rows
  also left out). Under the dataset-broadcast rule nothing changes in either step.

So the held-out exclusion, not the null change, is what moves the configuration-level numbers
on full history. The same exclusion is in the 4,096 run and its effect there was not measured.

Side check 2, the zeros. A scratch copy of `omega_values.json` was made in which every 0.0 with
status `ok` is null with a made-up status, and both columns were rerun
(`<scratch>/runs/sens_zero_undefined_full`, `sens_zero_undefined_4096`). All 60 dataset-level
cells are identical to the corresponding run with the zeros used as values, on both columns. The
choice between using and leaving out those 10 values changes no dataset-level number here.

## Verified in this change

- Input checks: the table above.
- Tests, Python 3.11.15 and 3.10.20, same pinned packages.

  | Suite | Command | Python 3.11 | Python 3.10 |
  |---|---|---|---|
  | K2 | `python -m unittest discover -s tests/unit/k2 -t .` | 182 tests, OK | 182 tests, OK |
  | Layout | `python -m unittest tests.unit.test_package_layout` | 3 tests, OK | 3 tests, OK |
  | Integration | `python -m unittest discover -s tests/integration -t .` | 96 tests, OK (17 skipped) | 96 tests, OK (17 skipped) |

  The K2 suite had 149 tests before this change; 33 are new.

- Tests were written first. Before the code existed the new file could not be imported (1 error);
  with only empty placeholders for the two new names, 33 tests ran with 11 failures and 25
  errors (`<scratch>/logs/red_before_stub_py311.log`). After the change all 33 pass.
- Default path unchanged. The documented command with no option was run before and after the change, on both Pythons.
  All 14 tables are byte-identical before and after (and between the two Pythons). The run
  manifest differs only in `created_utc`, `command` (the output folder) and `code_sha256` (the
  three edited files); it has the same keys as before. Of the saved copies in
  `docs/results/k2/`, `summary_stats.csv` and `within_family_sensitivity.csv` are byte-identical
  to the run; `joined_table.csv` has the same 80 rows, 12 columns and values in a different
  column order, as recorded earlier (`<scratch>/logs/compare_default.log`).

- Planted bugs. 32 bugs were planted one at a time, each in a throwaway copy of `src` and `tests`
  (never in the working tree; its files were hashed before and after and are unchanged), and the
  K2 suite was run on each copy. A control copy with no bug passes (182 tests). 32 of 32 were
  caught (`<scratch>/scripts/planted_bugs_k2_v1.py`, `<scratch>/logs/planted_bugs_round2.log`).
  A first round of the same 32 was thrown away: its control copy failed because of how the copy
  was built, so its verdicts meant nothing; the copy was fixed and everything rerun.

  | Planted bug | Outcome | Failing tests | A test that fails |
  |---|---|---|---|
  | V1: null read as 0.0, in the table handed downstream | caught | 10 | `test_allow_missing_prints_count_and_status_and_records_them` |
  | V1b: null read as 0.0 when the Omega file is read | caught | 11 | `test_allow_missing_prints_count_and_status_and_records_them` |
  | V2: rows with an undefined Omega silently dropped, no count | caught | 10 | `test_allow_missing_prints_count_and_status_and_records_them` |
  | V2b: status of undefined values not reported at all | caught | 12 | `test_allow_missing_prints_count_and_status_and_records_them` |
  | V2c: manifest-v1 nulls no longer stop the run without the flag | caught | 1 | `test_undefined_omega_stops_the_command_and_writes_nothing` |
  | V3: the 4,096 column carries the full-history values | caught | 13 | `test_allow_missing_prints_count_and_status_and_records_them` |
  | V3b: the run ignores `--omega-col` | caught | 7 | `test_allow_missing_prints_count_and_status_and_records_them` |
  | V3c: status of the wrong column reported | caught | 3 | `test_allow_missing_prints_count_and_status_and_records_them` |
  | V4: join by row position, not by `series_id` | caught | 1 | `test_join_is_by_series_id_not_by_row_order` |
  | V4b: one-to-one check removed | caught | 2 | `test_join_mismatch_and_missing_second_file_are_clean_errors` |
  | V4c: left join that keeps series with no Omega row, checks removed | caught | 2 | `test_join_mismatch_and_missing_second_file_are_clean_errors` |
  | V5: held-out guard skipped in the two-file reader | caught | 4 | `test_held_out_name_in_any_text_field_is_refused` |
  | V5b: held-out guard looks at the subset field only | caught | 4 | `test_held_out_name_in_any_text_field_is_refused` |
  | V5c: held-out matches left out without being asked | caught | 3 | `test_held_out_name_in_any_text_field_is_refused` |
  | V5d: opt-in exclusion counts the rows but keeps them | caught | 2 | `test_held_out_name_stops_the_command_and_the_opt_in_leaves_it_out` |
  | V5e: excluded rows reported as 0 | caught | 2 | `test_held_out_name_stops_the_command_and_the_opt_in_leaves_it_out` |
  | V5f: the manifest's own `held_out_strict` list ignored | caught | 3 | `test_both_columns_run_and_both_input_hashes_are_recorded` |
  | V6: default layout switched to the two-file format | caught | 52 | `test_defaults_of_the_command_and_of_the_run` |
  | V6b: new options silently accepted with the single-file layout | caught | 2 | `test_reading_from_files_needs_both_and_only_for_this_layout` |
  | V6c: status keys added to the report of the single-file layout | caught | 1 | `test_the_reference_run_reads_the_single_file_and_nothing_else` |
  | V7: hash of the Omega file not recorded | caught | 1 | `test_both_columns_run_and_both_input_hashes_are_recorded` |
  | V7b: Omega file record holds the hash of the series file | caught | 1 | `test_both_columns_run_and_both_input_hashes_are_recorded` |
  | V8: undefined count by column counts every status | caught | 3 | `test_allow_missing_prints_count_and_status_and_records_them` |
  | V8b: undefined count taken from the defined rows | caught | 9 | `test_allow_missing_prints_count_and_status_and_records_them` |
  | V8c: undefined values counted as exact zeros | caught | 1 | `test_null_stays_undefined_in_the_table_and_the_reader_counts_it_by_status` |
  | V8d: under-half list uses "at most half" | caught | 1 | `test_dataset_kept_on_fewer_than_half_of_its_series_is_listed` |
  | V9: 0.0 with status `ok` treated as undefined | caught | 28 | `test_the_reference_run_reads_the_single_file_and_nothing_else` |
  | V9b: exact zeros not counted | caught | 2 | `test_zero_omega_is_used_and_printed` |
  | V10: a value and a status that disagree are accepted | caught | 1 | `test_value_and_status_must_agree` |
  | V11: status not printed in the missing-Omega banner | caught | 2 | `test_allow_missing_prints_count_and_status_and_records_them` |
  | V12: zero-Omega warning never printed | caught | 1 | `test_zero_omega_is_used_and_printed` |
  | V13: held-out exclusions not printed by the command | caught | 1 | `test_held_out_name_stops_the_command_and_the_opt_in_leaves_it_out` |

  Ten of these are caught by exactly one test (V2c, V4, V6c, V7, V7b, V8c, V8d, V10, V12, V13).

- The two manifest-v1 runs were repeated on Python 3.10: all 14 tables of each are byte-identical
  to the Python 3.11 run.
- Not exercised: the figure (matplotlib is not installed) and the download.

## Design choices needing sign-off

| Choice | Alternative | Reason |
|---|---|---|
| The two-file layout is a new, opt-in value of `--manifest-version` (`manifest-v1`) with a second path option `--omega-values`. The single file stays the default. The placeholder layout `frozen-v2` is left as it was. | Make the two-file layout the default, or fill in `frozen-v2`. | The saved results must stay reproducible by the documented command, and the file is not yet recorded as frozen. The name is the data owner's. Whether `frozen-v2` is still needed is for the team. |
| Rows are grouped by the `subset` field through the existing subset map. | Map the new `config` field to GIFT-Eval names. | The 29 subset names are unchanged, so the grouping is exactly the saved one. On this file subset and config are one to one. A file without `subset` is refused. |
| An undefined Omega stops the run unless `--allow-missing-omega` is given; with it the series is left out and the count by status is printed and recorded. | Accept null without a flag because the layout defines it. | It is the existing rule for a missing value. On this manifest the flag is always needed, which keeps the choice visible in the command. |
| A base dataset stays in the analysis while at least one of its series has a defined Omega; one kept on fewer than half of its series is listed in the banner and the run manifest. | Remove a dataset below a minimum number or share of defined series. | Unchanged rule; three analysed datasets have only 1 or 2 series in total, so an absolute minimum would remove datasets the saved run uses. The lowest share of defined series in these runs is 38 of 42 (`bizitobs_service`, 4,096 column). |
| 0.0 with status `ok` is used as a value, counted and printed. | Refuse it, as the decision engine does, or treat it as undefined. | The engine refuses 0.0 because the old files used 0.0 for "undefined"; here the status carries that. Leaving the values out would be a silent change of the data. The side check shows no dataset-level number depends on it. Whether those histories should count as defined is a question for the data owner. |
| A value and a status that disagree (null with `ok`, a number with another status, a status that is not text, an infinite value) stop the run. | Trust the value, or trust the status. | Either guess could turn an undefined value into a number. |
| The join must be one to one on `series_id`; a missing, extra or repeated id stops the run before anything is read. | Inner join and report the count. | A partial join means the two files are not from the same run. |
| A held-out match in any text field stops the run, as before. `--exclude-held-out-matches` leaves the matching rows out instead, prints each and records them. | Stop only (no number could be produced on this manifest), or skip the check for `series_id`. | The rule and its default are unchanged. The opt-in keeps flagged rows out of the analysis, which is the safe direction for held-out data; the cost is that 8 legitimate series are missing here. Weakening the check for one field was not done. |

## Open

1. For the data owner: (a) the `series_manifest.json` in the pull request does not have the hash
   in the message and was not produced by the script in the same commit; (b) the hash of
   `omega_values.json` matches only with Windows line endings, so hashes should be taken on the
   stored form or the files marked as binary; (c) the series-id format produces `_dim5`, which the
   team's held-out substring rule matches (8 active and 5 excluded ids); (d) 10, 2 and 10 values
   of exactly 0.0 with status `ok` remain in `omega_levels`, `omega_diffs` and
   `omega_levels_4096`.
2. For the team: how to resolve (c). Until then K2 refuses the manifest by default, and any
   other consumer that applies the same substring rule to series ids will do the same.
3. Rerun both columns when the ids are fixed and #48 is merged, and record which history variant
   is frozen. The numbers that most need the recheck are the configuration-level ones and the
   three 0.5 crossings on the 4,096 column.
4. The effect of the 8 left-out rows on the 4,096 run was not measured.
5. The saved result files in `docs/results/k2/` are those of the old manifest and were not
   regenerated; the earlier findings note still needs the corrections listed in the hardening
   note.

## Standing caveats (from the hardening note; they apply to every table above)

- The published -0.556 is an sMAPE figure. The like-for-like row is sMAPE; the MSE row equal to
  -0.556 on the old manifest is a different metric and the agreement is a coincidence. On the
  4,096 column that MSE value is -0.522.
- The MSE rows use the MSE[mean] column and depend on one model's column (`Toto_Open_Base_1.0`).
- sMAPE versus deep depends on one deep model (`Crossformer`) and sMAPE versus pretrained on one
  pretrained model (`Lag-Llama`).
- The family mean is this replication's choice; the paper does not state one. Under the median
  or best model the sMAPE correlation versus deep is near zero on both Omega columns.
- n = 20 base datasets against 28 in the paper. The intervals are approximate at this n, are
  not corrected for the number of comparisons, share their resamples, and treat Omega and the
  published scores as fixed.
- Omega here is the team's own, computed on training regions; the 4,096 column closes the
  "no 4,096-step truncation" divergence of the earlier findings note only for this provisional
  manifest.
- Family scores for a base dataset average all of its configurations, including frequencies for
  which the manifest has no Omega.
