# K2 final rerun on the frozen manifest-v2

Status: final K2 rerun on the data owner's frozen manifest, tag `manifest-v2` (commit `061f31e`,
PR #48). Reanalysis of published results; no forecasting model was run. Every number below was
produced on 2026-10-15 by the commands in this note, on main at `d209ae7`, and read from the run
folders named below. Run outputs are in a scratch folder, not in the repository; no saved result
file in `docs/results/k2/` was changed or added. Part of #39. This replaces the provisional
manifest-v1 note of 2026-10-10 as the K2 record on the frozen manifest.

PR #48 was still open when this was run, and its decision record
`0008-manifest-v2-freeze-omega-variants-and-undefined-omega.md` says "proposed, waiting for
review". That record names full-history levels (`omega_levels`) as the default column and the
4,096-step column as a sensitivity check. This note follows it: the full-history run is the
headline, the 4,096 run is the sensitivity.

In short:

1. Full history: all 60 dataset-level correlations and their intervals are identical to the run
   on the old single-file manifest (largest difference 0.0). The like-for-like replication of the
   published -0.556 is the sMAPE value -0.417, 95% bootstrap interval [-0.793, +0.091], n = 20.
   The interval includes zero and includes the published value.
2. Last 4,096 steps: the same value is -0.426 [-0.786, +0.089], n = 20. Across the 12 headline
   cells the largest change from full history is 0.134 (MAE versus statistical, +0.251 to
   +0.117). No headline cell changes sign and no headline interval changes between including and
   excluding zero.
3. manifest-v1 and manifest-v2 give byte-identical correlation tables on both columns. The 160
   (full) and 284 (4,096) undefined Omega values, including the new `near_flat` status, move the
   Omega of the two Bitbrains datasets but not the order of the 20 datasets on full history.
4. With the held-out name rule now on main (#59), no manifest row is left out: 8,761 of 8,761
   series are read and used. `--exclude-held-out-matches` changes nothing (outputs byte-identical
   with and without it). The 8 `_dim5` rows that the 2026-10-10 note had to leave out are back in.
5. Every output table is byte-identical between Python 3.11 and Python 3.10.

## Inputs (SHA-256)

    series_manifest.json (manifest-v2, 6,530,131 bytes)   db22d3facdb141b16c0abd9fbcbb2c68d609d6f85dc5b448c005290dff5fb0be
    omega_values.json    (manifest-v2, 5,005,058 bytes)   2c80f4247ec190de812108d2493457fad72e91d3c086378c23a490b04811b224
    excluded_series.json (release asset, not read by K2)  e240e4c3c128eb407e622a81cca4d9e4746ecd92c67fc2e913e742e47346015b
    gift_results.csv     (artifacts/reference/k2)          5bdf59d70d5e72c3c6593b194d68cecdb35d9f435b2109b42b8668054886c464

- The two manifest files were taken with `git show` from the tag
  (`artifacts/reference/manifests/`). They are the same git objects at the tag and at the PR head
  `961dd57`, and their hashes equal the ones the data owner stated. The release asset was
  downloaded with `gh release download manifest-v2` and its hash equals the digest GitHub reports
  for it. K2 records the first two and the results file in `inputs` of each `run_manifest.json`.
- K2 records the manifest as "frozen: no", because the series file has no boolean `frozen` key.
  The freeze is the tag and record 0008, not a field in the file.
- For comparison, manifest-v1 (`23af713`) was rerun on the same code: `series_manifest.json`
  `f20898d3...`, `omega_values.json` `fd4ca660...` (the hashes recorded in the 2026-10-10 note).

## Commands

Run from the root of a worktree of main at `d209ae7`. `<scratch>` is a working folder outside the
repository. Python 3.11.15 and 3.10.20, both with numpy 1.26.4, pandas 2.2.3, pyarrow 16.1.0,
scipy 1.11.4 (`environment.pin_mismatches` is empty in every run manifest).

    git show manifest-v2:artifacts/reference/manifests/series_manifest.json > <scratch>/tag/series_manifest.json
    git show manifest-v2:artifacts/reference/manifests/omega_values.json    > <scratch>/tag/omega_values.json

    # full history (headline)
    PYTHONPATH=src python -m decision_aware_tsfm.analysis.k2.replication \
        --manifest-version manifest-v1 \
        --manifest <scratch>/tag/series_manifest.json \
        --omega-values <scratch>/tag/omega_values.json \
        --omega-col omega_levels --allow-missing-omega --exclude-held-out-matches \
        --outdir <scratch>/runs/py311_v2_full_history

    # last 4,096 steps (sensitivity)
    (the same with --omega-col omega_levels_4096 --outdir <scratch>/runs/py311_v2_last_4096)

`manifest-v1` is the name of the two-file layout in the reader, not of the manifest version;
manifest-v2 uses the same layout. Each command was run with both Pythons (`py310_*` folders).
Also run, on Python 3.11 unless named:

- the same two commands without `--exclude-held-out-matches` (`*_noflag`);
- the same two commands on the manifest-v1 files (`py311_v1_full_history`, `py311_v1_last_4096`);
- the default command on the old single-file manifest, on both Pythons
  (`PYTHONPATH=src python -m decision_aware_tsfm.analysis.k2.replication --outdir <scratch>/runs/py311_default`).

Settings are the defaults in every run: 10,000 bootstrap resamples, seed 20261004, 95% percentile
intervals over the 20 base datasets. No resample was degenerate in any headline cell. The tables
below were read from the run folders by `<scratch>/scripts/report_tables.py` and
`<scratch>/scripts/extra.py` (not committed).

## What the run read

| | Full history | Last 4,096 steps |
|---|---|---|
| Series read and used | 8,761 of 8,761 | 8,761 of 8,761 |
| Held-out rows left out | 0 | 0 |
| Series with an Omega value | 8,601 | 8,477 |
| Undefined, by status | 66 `flat_history`, 94 `near_flat` | 25 `empty_history`, 166 `flat_history`, 93 `near_flat` |
| Datasets with undefined values | `bitbrains_fast_storage` 114 of 2,432 (56 flat, 58 near_flat); `bitbrains_rnd` 46 of 936 (10 flat, 36 near_flat) | `bitbrains_fast_storage` 209, `bitbrains_rnd` 62, `kdd_cup_2018` 7, `bizitobs_service` 4, `electricity` 2 |
| Omega exactly 0.0 | 0 | 0 |
| Base datasets, dataset-frequency subsets | 20, 29 | 20, 29 |

- The undefined counts on all four Omega columns equal the table in record 0008 (from
  `manifest_report.omega_undefined_by_column`). No dataset was removed and none was kept on fewer
  than half of its series.
- Held-out rule: 0 rows matched M5 or PJM, in the manifest or in the results file. The
  punctuation-insensitive check printed 41 names and kept them, as that check prescribes: 30
  series ids (the 8 `_dim5` ids and 22 `solar` ids such as `solar_H__item_5`) and 11 item ids
  (`item_5`, `item_50` to `item_59`). None is an M5 or PJM series. M5 and PJM were not opened.
- `granite-flowstate-r1` is mapped to the zero-shot family but is not in the results file, as in
  every earlier run; 18 zero-shot models are used.

## Results

Delta = 100 (A - B) / A with A the zero-shot family, family score = mean across the family's
models, one row per base dataset, Omega averaged over the dataset's series that have a value.
Negative rho means the zero-shot advantage grows with Omega.

### 1. Headline: family mean, dataset level (`correlations_dataset_level.csv`)

| Metric | Zero-shot versus | Old single-file manifest | manifest-v2, full history | manifest-v2, last 4,096 steps |
|---|---|---|---|---|
| sMAPE | deep | -0.417 [-0.793, +0.091] | -0.417 [-0.793, +0.091] | -0.426 [-0.786, +0.089] |
| sMAPE | statistical | -0.123 [-0.603, +0.397] | -0.123 [-0.603, +0.397] | -0.174 [-0.629, +0.338] |
| sMAPE | pretrained | -0.514 [-0.800, -0.054] | -0.514 [-0.800, -0.054] | -0.537 [-0.824, -0.090] |
| MSE (MSE[mean]; see section 3) | deep | -0.556 [-0.828, -0.123] | -0.556 [-0.828, -0.123] | -0.522 [-0.810, -0.077] |
| MSE (MSE[mean]; see section 3) | statistical | -0.398 [-0.807, +0.101] | -0.398 [-0.807, +0.101] | -0.388 [-0.804, +0.123] |
| MSE (MSE[mean]; see section 3) | pretrained | -0.520 [-0.813, -0.097] | -0.520 [-0.813, -0.097] | -0.498 [-0.791, -0.063] |
| MAE (2 x pinball at tau 0.5) | deep | -0.248 [-0.681, +0.256] | -0.248 [-0.681, +0.256] | -0.253 [-0.677, +0.273] |
| MAE (2 x pinball at tau 0.5) | statistical | +0.251 [-0.315, +0.684] | +0.251 [-0.315, +0.684] | +0.117 [-0.427, +0.587] |
| MAE (2 x pinball at tau 0.5) | pretrained | -0.405 [-0.789, +0.103] | -0.405 [-0.789, +0.103] | -0.403 [-0.799, +0.108] |
| CRPS | deep | -0.280 [-0.718, +0.265] | -0.280 [-0.718, +0.265] | -0.299 [-0.746, +0.248] |
| CRPS | statistical | +0.376 [-0.132, +0.750] | +0.376 [-0.132, +0.750] | +0.325 [-0.207, +0.720] |
| CRPS | pretrained | -0.415 [-0.812, +0.097] | -0.415 [-0.812, +0.097] | -0.429 [-0.829, +0.084] |

n = 20 in every cell. Two-sided Spearman p-values, same order, full history (4,096 in brackets):
0.0677 (0.0614), 0.6045 (0.4620), 0.0203 (0.0147), 0.0108 (0.0183), 0.0818 (0.0910), 0.0187
(0.0255), 0.2915 (0.2826), 0.2855 (0.6224), 0.0769 (0.0781), 0.2323 (0.1999), 0.1023 (0.1623),
0.0688 (0.0594).

- Old against manifest-v2 full: identical in all 60 cells (12 comparisons by 5 within-family
  summaries), rho and both interval ends; the largest difference is 0.0.
- Full against 4,096, the 12 headline cells: every cell keeps its sign. The largest change is
  0.134 (MAE versus statistical), then 0.051 and 0.051. The same three intervals exclude zero on
  both columns (sMAPE versus pretrained, MSE versus deep, MSE versus pretrained).
- Full against 4,096, all 60 cells: the largest change is 0.158 (MSE versus statistical, median
  over models, +0.208 to +0.050). Two cells change sign, both near zero: MSE versus pretrained,
  median over models (+0.075 to -0.029), and MAE versus pretrained, median over models (+0.105 to
  -0.012). A third goes from exactly 0.0 to +0.011 (MSE versus pretrained, lowest single
  model-configuration row). One interval changes
  from including to excluding zero: MSE versus deep, lowest single model-configuration row (on
  the MSE[mean] column, so section 3 applies), -0.349 [-0.683, +0.124] to
  -0.480 [-0.728, -0.049]. 4 of 60 cells exclude zero on full history, 5 on 4,096.
- The number of datasets on which the zero-shot family has the lower error does not depend on
  Omega and is the same in all runs (`sign_counts.csv`, deep, statistical, pretrained, of 20):
  sMAPE 20, 15, 20; MSE 8, 5, 6; MAE 19, 18, 19; CRPS 20, 19, 20.

### 2. Like for like with the paper, sMAPE (`paper_like_for_like.csv`)

| Zero-shot versus | Published (sMAPE, 28 datasets) | manifest-v2 full | manifest-v2 4,096 | Published value inside the interval |
|---|---|---|---|---|
| deep | -0.556 | -0.417 [-0.793, +0.091] | -0.426 [-0.786, +0.089] | yes, both |
| statistical | -0.234 | -0.123 [-0.603, +0.397] | -0.174 [-0.629, +0.338] | yes, both |
| pretrained | not given | -0.514 [-0.800, -0.054] | -0.537 [-0.824, -0.090] | not applicable |

On either column the replication is weaker than published in both comparisons that have a
published value, both published values lie inside the interval, and for deep and statistical so
does zero. With 20 datasets this can neither confirm nor contradict the published magnitudes.
The same file records the MSE[mean] value of the deep row as -0.556 on full history: the
match with the published sMAPE figure is a different metric agreeing by coincidence, not a
replication (see the 2026-10-04 hardening note, section 3).

### 3. MSE sensitivity (`mse_sensitivity.csv`; the MSE rows must not be cited without it)

The model with the largest median share of the zero-shot family's summed MSE[mean] is
`Toto_Open_Base_1.0` on both columns. Last column: datasets (of 20) on which the zero-shot family
has the lower error.

| MSE variant | Versus | Full history | Last 4,096 | Zero-shot lower |
|---|---|---|---|---|
| As recorded, MSE[mean] | deep | -0.556 [-0.828, -0.123] | -0.522 [-0.810, -0.077] | 8 |
| As recorded, MSE[mean] | statistical | -0.398 [-0.807, +0.101] | -0.388 [-0.804, +0.123] | 5 |
| As recorded, MSE[mean] | pretrained | -0.520 [-0.813, -0.097] | -0.498 [-0.791, -0.063] | 6 |
| MSE[mean], dominant model removed | deep | -0.454 [-0.773, +0.024] | -0.438 [-0.771, +0.048] | 15 |
| MSE[mean], dominant model removed | statistical | -0.164 [-0.607, +0.338] | -0.232 [-0.649, +0.271] | 14 |
| MSE[mean], dominant model removed | pretrained | -0.654 [-0.842, -0.305] | -0.589 [-0.824, -0.189] | 16 |
| MSE[0.5] column | deep | -0.347 [-0.708, +0.144] | -0.325 [-0.703, +0.180] | 17 |
| MSE[0.5] column | statistical | +0.250 [-0.249, +0.635] | +0.123 [-0.357, +0.547] | 18 |
| MSE[0.5] column | pretrained | -0.526 [-0.768, -0.107] | -0.516 [-0.771, -0.090] | 19 |
| MAE[0.5], for comparison | deep | -0.248 [-0.681, +0.256] | -0.253 [-0.677, +0.273] | 19 |
| MAE[0.5], for comparison | statistical | +0.251 [-0.315, +0.684] | +0.117 [-0.427, +0.587] | 18 |
| MAE[0.5], for comparison | pretrained | -0.405 [-0.789, +0.103] | -0.403 [-0.799, +0.108] | 19 |

What this shows on both columns: the zero-shot family has the lower MSE on only 8, 5 and 6 of 20
datasets as recorded, and on 15, 14 and 16 once `Toto_Open_Base_1.0` is removed. On the MSE[0.5]
column the sign versus statistical is the MAE sign (+0.250 and +0.251 on full history). So the
apparent opposite ranking of families by MSE and by MAE comes from one model's MSE[mean] column.
Versus deep the interval excludes zero only as recorded; with the dominant model removed and on
MSE[0.5] it includes zero.

### 4. Dependence on one comparison model (`leave_one_model_out.csv`)

| Headline | Model removed | Full history | Last 4,096 |
|---|---|---|---|
| sMAPE versus deep | `Crossformer` | -0.417 to +0.195 [-0.272, +0.597] | -0.426 to +0.224 [-0.251, +0.641] |
| sMAPE versus pretrained | `Lag-Llama` | -0.514 to -0.048 [-0.580, +0.464] | -0.537 to -0.059 [-0.602, +0.448] |

The like-for-like replication of the published figure changes sign when one deep model is
removed, and the sMAPE result versus pretrained, the only family-mean sMAPE interval that
excludes zero, falls to near zero without `Lag-Llama`. No model was removed from any headline
number.

### 5. Configuration level (sensitivity only, `correlations_config_level.csv`)

71 of 97 configurations join under the dataset-and-frequency rule and 89 under the
dataset-broadcast rule, in every run. The p-value of these rows treats configurations as
independent, which they are not; the interval resamples the 20 base datasets.

- Old against manifest-v2 full: the largest change in rho over the 72 rows is 0.0054. No sign
  changes and no interval changes between including and excluding zero (7 of 72 exclude zero in
  both).
- Full against 4,096: the largest change is 0.202 (CRPS versus statistical, dataset and
  frequency, within-family summary `min`, +0.293 to +0.091). 8 rows change sign; no interval changes between
  including and excluding zero (7 of 72 in both).

### 6. The 4,096 sensitivity, Omega per base dataset (`dataset_deltas.csv`)

Sorted by full-history Omega; rank 1 is the lowest.

| Base dataset | Old | v2 full | v2 4,096 | 4,096 minus full | Rank full | Rank 4,096 |
|---|---|---|---|---|---|---|
| hierarchical_sales | 0.1144 | 0.1144 | 0.1144 | 0.0000 | 1 | 1 |
| bitbrains_rnd | 0.2587 | 0.2698 | 0.2677 | -0.0020 | 2 | 2 |
| bitbrains_fast_storage | 0.2734 | 0.2857 | 0.2779 | -0.0078 | 3 | 3 |
| saugeen | 0.2904 | 0.2903 | 0.2861 | -0.0042 | 4 | 4 |
| sz_taxi | 0.3070 | 0.3070 | 0.3070 | 0.0000 | 5 | 5 |
| kdd_cup_2018 | 0.3168 | 0.3168 | 0.3164 | -0.0004 | 6 | 6 |
| loop_seattle | 0.4368 | 0.4368 | 0.4216 | -0.0152 | 7 | 7 |
| jena_weather | 0.4888 | 0.4888 | 0.5148 | +0.0261 | 8 | 10 |
| ett1 | 0.5475 | 0.5475 | 0.4696 | -0.0779 | 9 | 9 |
| m_dense | 0.5821 | 0.5821 | 0.5389 | -0.0433 | 10 | 11 |
| electricity | 0.6126 | 0.6126 | 0.5932 | -0.0195 | 11 | 13 |
| ett2 | 0.6211 | 0.6211 | 0.4429 | -0.1782 | 12 | 8 |
| us_births | 0.6259 | 0.6259 | 0.5918 | -0.0340 | 13 | 12 |
| bizitobs_service | 0.6381 | 0.6381 | 0.6157 | -0.0224 | 14 | 14 |
| bizitobs_l2c | 0.6711 | 0.6711 | 0.6680 | -0.0031 | 15 | 16 |
| m4_monthly | 0.6845 | 0.6845 | 0.6845 | 0.0000 | 16 | 17 |
| bizitobs_application | 0.7125 | 0.7125 | 0.7032 | -0.0093 | 17 | 18 |
| solar | 0.7254 | 0.7254 | 0.6630 | -0.0625 | 18 | 15 |
| m4_weekly | 0.7339 | 0.7339 | 0.7339 | 0.0000 | 19 | 19 |
| m4_daily | 0.7508 | 0.7508 | 0.7509 | +0.0001 | 20 | 20 |

- Old against v2 full: only the two Bitbrains datasets move by more than 0.0001
  (`bitbrains_fast_storage` +0.0123, `bitbrains_rnd` +0.0110), because 114 and 46 of their series
  are now undefined and left out of the mean; the largest change in any other dataset is 0.00004
  (rounding). The rank order of the 20 datasets is the same (Spearman 1.0), which is why every
  dataset-level correlation is identical.
- 4,096 minus full: from -0.1782 (`ett2`) to +0.0261 (`jena_weather`); median -0.0060, mean
  -0.0227, mean absolute 0.0253. Lower in 14 datasets, higher in 2 (`jena_weather`, `m4_daily`),
  equal in 4.
- Rank: 9 of 20 datasets change rank; the largest moves are `ett2` (12 to 8) and `solar` (18 to
  15). Spearman between the two columns across the 20 datasets: 0.9714 (Pearson 0.9739).
- The 0.4 mark: no dataset crosses it; 14 datasets are at or above it on both columns. The 0.5
  mark: 12 datasets at or above on full history, 11 on 4,096; `ett1` and `ett2` go below it and
  `jena_weather` goes above it. 0.4 and 0.5 are reference marks only, not the paper's zones.
- The 2026-10-10 note flagged these three crossings for a recheck because each dataset was then
  missing its `_dim5` series. They are rechecked here with all series in: all three still cross.

## Byte-identity checks

| Comparison | Tables identical (of 14) |
|---|---|
| Python 3.11 against 3.10, manifest-v2 full | 14 |
| Python 3.11 against 3.10, manifest-v2 4,096 | 14 |
| Python 3.11 against 3.10, old single-file manifest | 14 |
| With against without `--exclude-held-out-matches`, full and 4,096 | 14 and 14 (logs equal too) |
| manifest-v1 against manifest-v2, full | 11; differ: `dataset_deltas.csv`, `joined_table.csv`, `joined_config_table.csv` (Omega values); both correlation tables identical |
| manifest-v1 against manifest-v2, 4,096 | 11; same three differ; both correlation tables identical |
| Old single-file manifest against manifest-v2 full | 10; differ: the same three and `correlations_config_level.csv` (largest change 0.0054, section 5) |

Saved copies in `docs/results/k2/`: `summary_stats.csv` and `within_family_sensitivity.csv` are
byte-identical to today's manifest-v2 full run and to the default run. `joined_table.csv`
differs from both, as recorded in the earlier notes (column order, and here also the Omega
values).

## Tests on main at `d209ae7` (this note adds no code)

| Suite | Python 3.11 | Python 3.10 |
|---|---|---|
| `tests/unit/decision` | 160 OK (1 skipped) | 160 OK (1 skipped) |
| `tests/unit/experiments` | 170 OK | 170 OK |
| `tests/unit/e3` | 107 OK | 107 OK |
| `tests/unit/k2` | 184 OK | 184 OK |
| `tests/unit/test_package_layout.py` | 3 OK | 3 OK |
| `tests/integration` | 102 OK (17 skipped) | 102 OK (17 skipped) |

Each suite run as `PYTHONPATH=src python -m unittest discover -s <suite> -t .` from the
worktree root (layout: `python -m unittest tests.unit.test_package_layout`). No planted-bug round
was run: this change adds a note only.

## What this allows us to claim, and what not

- The published zero-shot versus deep figure, -0.556, is an sMAPE figure. Our like-for-like value
  is -0.417 [-0.793, +0.091], n = 20, on the frozen manifest. It is weaker than published, its
  interval includes zero, and it becomes +0.195 without `Crossformer`. Versus pretrained, -0.514
  becomes -0.048 without `Lag-Llama`.
- The MSE[mean] value of -0.556 for the same comparison is not a replication of the published
  number. The MSE rows depend on `Toto_Open_Base_1.0` (section 3) and must be cited with that
  table.
- Choosing the 4,096-step column instead of full history does not change any headline sign or
  any headline interval's relation to zero; the largest headline change is 0.134. It does reorder
  datasets in the middle of the Omega range (`ett2`, `solar`, `jena_weather`, `ett1`). For
  example any threshold between 0.4429 and 0.6211 puts `ett2` on a different side depending on
  the column.
- The intervals are percentile bootstrap intervals over 20 datasets. Their coverage was not
  measured for K2.
- Not covered here: the figure (matplotlib is not installed) and the download of the results
  file (`gift_results.csv` is the stored copy, hash above).

## Open

- Record 0008 and PR #48 are still open. If the team changes the default column or the manifest,
  this rerun has to be repeated; the commands above do that.
