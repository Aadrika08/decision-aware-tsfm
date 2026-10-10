# K2 hardening and rerun on the current development manifest

Status: reanalysis of published results. No forecasting model was run. Every number below was
produced on 2026-10-04 by the commands in this note, from the saved GIFT-Eval results file and the
current (not yet frozen) Omega manifest, and is read from the output directory
`artifacts/runs/k2-2026-10-04/review_fixes/`. This is the second version of this note. An independent reviewer reran the first
version and found no wrong number; the findings were a missing sensitivity analysis, two
statements about the source paper that the paper does not support, guards that could be
bypassed, and tests that checked the shape of tables and not their values. This version fixes
those. The recorded headline computation is unchanged and its tables are byte-identical to the
record. The rerun on the frozen manifest and the 4,096-step variant are still not done: the
frozen manifest was not available. Figures were not regenerated (matplotlib is not installed in
the pinned environment).

What changed in what may be claimed, in short:

1. The published -0.556 (zero-shot versus deep learning) is an sMAPE figure. The like-for-like
   replication is the sMAPE row: rho = -0.417, 95% bootstrap interval
   [-0.793, +0.091], n = 20. The MSE row also reads -0.556, but that
   is a different metric, which the paper does not use for this comparison; the agreement to
   three decimals is a coincidence (section 3).
2. The MSE rows depend on one model's MSE[mean] column (`Toto_Open_Base_1.0`) and must not be cited without
   the sensitivity table in section 1.
3. A leave-one-model-out table was added for every metric (section 2). No other headline depends
   on a single zero-shot model. Two headlines do depend on a single model of the comparison
   family: sMAPE versus deep goes from -0.417 to +0.195 when `Crossformer` is
   removed from the deep family, and sMAPE versus pretrained goes from -0.514 to
   -0.048 when `Lag-Llama` is removed from the pretrained family.

## Commands

Run from the repository root with the pinned interpreter (Python 3.11.15).

    # the one command: regenerates every K2 output into a new directory
    PYTHONPATH=src python -m decision_aware_tsfm.analysis.k2.replication \
        --outdir artifacts/runs/k2-2026-10-04/review_fixes --figure

    # tests
    PYTHONPATH=src python -m unittest discover -s tests/unit/k2 -t .

`scripts/run_k2.py` and the `decision-aware-run-k2` console script call the same entry point.
The command refuses to write into a directory that already exists, or into a directory inside
an existing run. Defaults: results `artifacts/reference/k2/gift_results.csv`, manifest
`artifacts/reference/manifests/dev_manifest.json`, Omega column `omega_levels`, manifest layout
`dev-v1`, 10,000 bootstrap resamples, seed 20261004. The two runs took 26 and 25 seconds of wall time in this session (about 19 seconds of user CPU time each), with other jobs running on the same machine.

Outputs of this run are in `artifacts/runs/k2-2026-10-04/review_fixes/` (git-ignored). A second run with the same arguments
(`artifacts/runs/k2-2026-10-04/review_fixes_rerun/`) produced byte-identical CSV files. The eight tables that the first
hardening run also wrote are byte-identical to that run (`artifacts/runs/k2-2026-10-04/omega_levels/`),
so the fixes changed no number that was already reported.

## What was built

`replication.py` is the entry point and keeps the names older scripts import. The code is a small
library; the modules marked new were added in this version.

| Module | Job |
|---|---|
| `loading.py` | Reads the results file and the manifest. Manifest reading sits behind one function, `read_manifest(path, version)`, which returns the same series table for any layout. Builds the Omega tables per base dataset and per (base dataset, frequency). Applies the held-out guard and the checks on the Omega column. |
| `heldout.py` (new) | Held-out name checks on top of `analysis/decision/guards.py`: a strict layer that is acted on and a punctuation-insensitive layer that is only reported. |
| `families.py` | Model-to-family mapping, metric columns, the per-family score table (mean, median, best; two ways of pooling), and the labels for the within-family summaries. |
| `statistic.py` | Delta, Spearman correlation, bootstrap interval, sign counts, and the checks on the bootstrap parameters. |
| `joins.py` (new) | Joining family scores to Omega. Moved out of `pipeline.py` unchanged. |
| `sensitivity.py` (new) | MSE sensitivity, leave one model out, model score diagnostics, the like-for-like table against the paper, and the relabelled within-family table. |
| `pipeline.py` | Computes every table and writes the output directory and the run manifest. |
| `environment.py` | Records running versions and compares them with the team pins. |
| `figure.py` | Optional figure, only imported with `--figure`. |
| `fetch.py` | Download of the published results (network; not used by the analysis, not run here). |
| `cli.py` | Argument parsing. |

Files written by the command. The first three are in the recorded format and are not changed by
anything in this version.

| File | Contents |
|---|---|
| `summary_stats.csv` | The twelve headline correlations, same format and rounding as the recorded file. |
| `within_family_sensitivity.csv` | The recorded three-row table (sMAPE, zero-shot versus deep), same format, including the recorded label text. |
| `joined_table.csv` | Per base dataset and family: mean scores, Omega summary, series count, median length. |
| `correlations_dataset_level.csv` | 60 rows: 4 metrics x 3 comparisons x 5 within-family summaries, full precision, with n, bootstrap interval and the count of degenerate resamples. |
| `sign_counts.csv` | For the same 60 cells: on how many datasets the zero-shot family has the lower error. |
| `dataset_deltas.csv` | The 20 points (Omega, Delta) behind each headline correlation. |
| `correlations_config_level.csv` | Sensitivity only: the correlation at configuration level under two matching rules. |
| `joined_config_table.csv` | Per configuration and family: mean scores and the Omega value under each matching rule. |
| `mse_sensitivity.csv` (new) | The MSE correlations as recorded, with the dominant zero-shot model removed, and on the MSE[0.5] column, with intervals and sign counts (15 rows). |
| `leave_one_model_out.csv` (new) | Every family-mean correlation with one model removed, for the four metrics and MSE[0.5] (395 rows). |
| `leave_one_model_out_summary.csv` (new) | The range of those per metric, comparison and family of the removed model (30 rows). |
| `model_score_diagnostics.csv` (new) | Per metric and model: its share of the family's summed score, how often it has the family's highest score, and two unit checks on the MSE column (215 rows). |
| `paper_like_for_like.csv` (new) | The correlations the paper publishes beside the replication on the paper's metric (sMAPE). |
| `within_family_sensitivity_relabelled.csv` (new) | The recorded sensitivity table with accurate labels, plus the two model-level rows. |
| `run_manifest.json` | SHA-256 of both inputs and of every output and source file, interpreter and package versions, team pins and any mismatch, the command, parameters, the model-to-family mapping, held-out rule, fields checked, every exclusion and every reported name, manifest layout version and frozen flag, missing-Omega report, the dominant model and the rule that chose it, degenerate-resample warnings, figure status. |

Environment: the run manifest records Python 3.11.15, numpy 1.26.4, pandas 2.2.3, pyarrow 16.1.0,
scipy 1.11.4, which equal the team pins (no mismatch recorded). If the running versions differ the
command prints a warning naming each difference and continues.

Inputs of this run (SHA-256):

    gift_results.csv   5bdf59d70d5e72c3c6593b194d68cecdb35d9f435b2109b42b8668054886c464
    dev_manifest.json  a944b4a9b02ae2bc72e4ff6cd4fbe869a778d2ad3a2bc80498574919f7d35b4b

The manifest is recorded as not frozen: layout `dev-v1`, no version declared in the file, no
`frozen` key in the file. All numbers in this note are on the current, not frozen, manifest.

## Verified in this change

### 1. Headline numbers with uncertainty, and the MSE sensitivity table

Delta = 100 (A - B) / A with A the zero-shot family, family score = mean across the family's
models, one row per base dataset, Omega = `omega_levels` averaged over the dataset's series.
Negative rho means the zero-shot advantage grows with Omega. Interval: 95% percentile bootstrap,
10,000 resamples of the 20 datasets, seed 20261004. No resample was degenerate in any cell of any
table (0 of 10,000 everywhere). p is the usual two-sided Spearman p-value, unchanged from the
record. The last column counts datasets where the zero-shot family mean has the lower error.

| Metric | Zero-shot versus | n | rho | 95% bootstrap interval | p | Zero-shot lower |
|---|---|---|---|---|---|---|
| sMAPE | deep | 20 | -0.417 | [-0.793, +0.091] | 0.0677 | 20 of 20 |
| sMAPE | statistical | 20 | -0.123 | [-0.603, +0.397] | 0.6045 | 15 of 20 |
| sMAPE | pretrained | 20 | -0.514 | [-0.800, -0.054] | 0.0203 | 20 of 20 |
| MSE (see the table below) | deep | 20 | -0.556 | [-0.828, -0.123] | 0.0108 | 8 of 20 |
| MSE (see the table below) | statistical | 20 | -0.398 | [-0.807, +0.101] | 0.0818 | 5 of 20 |
| MSE (see the table below) | pretrained | 20 | -0.520 | [-0.813, -0.097] | 0.0187 | 6 of 20 |
| MAE (2 x pinball at tau 0.5) | deep | 20 | -0.248 | [-0.681, +0.256] | 0.2915 | 19 of 20 |
| MAE (2 x pinball at tau 0.5) | statistical | 20 | +0.251 | [-0.315, +0.684] | 0.2855 | 18 of 20 |
| MAE (2 x pinball at tau 0.5) | pretrained | 20 | -0.405 | [-0.789, +0.103] | 0.0769 | 19 of 20 |
| CRPS | deep | 20 | -0.280 | [-0.718, +0.265] | 0.2323 | 20 of 20 |
| CRPS | statistical | 20 | +0.376 | [-0.132, +0.750] | 0.1023 | 19 of 20 |
| CRPS | pretrained | 20 | -0.415 | [-0.812, +0.097] | 0.0688 | 20 of 20 |

What the interval tells the reader: if the 20 datasets are treated as a sample of the kind of
dataset we care about, and we redraw 20 of them with replacement many times, the middle 95% of
the recomputed correlations falls in this range. It shows how much a correlation across 20
datasets moves when a few datasets are swapped in or out. Here the intervals are wide (widths
from 0.70 to 1.00, median 0.90) and 9 of the 12 include zero. Only sMAPE versus
pretrained, MSE versus deep and MSE versus pretrained exclude zero. Of these, MSE versus deep
does not survive the check below, and sMAPE versus pretrained does not survive the removal of
one pretrained model (section 2). MSE versus pretrained keeps an interval that excludes zero in
every MSE variant below.

What it does not tell the reader:

- It is not a new hypothesis test, and it is not corrected for looking at 12 comparisons.
- The 20 datasets were not drawn at random from a population, so it is not a guarantee about
  other benchmarks.
- It treats Omega and the published scores as fixed. It does not include uncertainty in Omega
  itself, in the choice of models in each family, or in the choice of within-family summary.
  Section 2 shows that the choice of models matters.
- With n = 20 a percentile interval for a rank correlation is approximate and tends to be
  somewhat too narrow. Read it as a lower bound on the uncertainty.
- All intervals use the same resampled sets of datasets, so they are not independent of
  each other.

#### The MSE rows depend on one model's MSE[mean] column

The recorded MSE rows use the published column `eval_metrics/MSE[mean]` and average it over the
18 zero-shot models present in the results. One of them, `Toto_Open_Base_1.0`, has MSE[mean] values far out
of line with the other 17 and with its own other columns (from `model_score_diagnostics.csv`,
on the 20 analysed datasets):

| Check | `Toto_Open_Base_1.0` | The other 17 zero-shot models |
|---|---|---|
| Median share of the family's summed MSE[mean] (an even share is 0.056) | 0.539 | at most 0.023 (`Moirai_large`) |
| Datasets on which it has the highest MSE[mean] in the family | 17 of 20 | `Moirai_base` 1, `Moirai_small` 2 |
| Median ratio of its MSE[mean] to the family's median model | 104.2 | 0.80 to 1.46 |
| Median of sqrt(MSE[mean]) / MAE[0.5] over its rows | 14.85 | 1.94 to 2.18 |
| Median of MSE[mean] / MSE[0.5] over its rows | 15.84 | 0.99 to 1.02 |

On its MAE, sMAPE, CRPS and MSE[0.5] columns the same model is ordinary: its median share of the
family sum is 0.054, 0.055, 0.052 and 0.053. MSE[mean] is the squared error of the mean
forecast and MSE[0.5] the squared error of the median forecast, by the GluonTS naming that
GIFT-Eval uses; that reading of the column names was not re-verified here. Whether this model's
published MSE[mean] values are correct was not checked. What is established is that one column
of one model carries more than half of the zero-shot family's MSE[mean] on a typical dataset.

The table below is `mse_sensitivity.csv`. "Removed" means `Toto_Open_Base_1.0` taken out of the
zero-shot family; this is the model with the largest median share, chosen by that rule and not
by name. Same statistic, same 20 datasets, same bootstrap as the headline.

| MSE variant | Zero-shot models | Versus deep | Versus statistical | Versus pretrained | Zero-shot lower (deep / statistical / pretrained, of 20) |
|---|---|---|---|---|---|
| As recorded: MSE[mean] | 18 | -0.556 [-0.828, -0.123] | -0.398 [-0.807, +0.101] | -0.520 [-0.813, -0.097] | 8 / 5 / 6 |
| MSE[mean], `Toto_Open_Base_1.0` removed | 17 | -0.454 [-0.773, +0.024] | -0.164 [-0.607, +0.338] | -0.654 [-0.842, -0.305] | 15 / 14 / 16 |
| MSE[0.5] column | 18 | -0.347 [-0.708, +0.144] | +0.250 [-0.249, +0.635] | -0.526 [-0.768, -0.107] | 17 / 18 / 19 |
| MSE[0.5] column, `Toto_Open_Base_1.0` removed | 17 | -0.334 [-0.702, +0.173] | +0.242 [-0.256, +0.629] | -0.481 [-0.775, -0.028] | 17 / 18 / 19 |
| MAE[0.5], for comparison | 18 | -0.248 [-0.681, +0.256] | +0.251 [-0.315, +0.684] | -0.405 [-0.789, +0.103] | 19 / 18 / 19 |

In plain words:

- As recorded, the zero-shot family looks worse than every other family on MSE on most datasets
  (8, 5 and 6 of 20 datasets with the lower error). With the one model removed it is better on
  15, 14 and 16 of 20, and on the MSE[0.5] column with all 18 models it is better on 17, 18 and 19
  of 20, which is what MAE shows (19, 18, 19). So "MSE and MAE give opposite family rankings" is
  produced by one model's column and is not a finding about the families.
- Versus deep: -0.556 as recorded, -0.454 with the model removed, -0.347 on MSE[0.5]. The recorded
  interval excludes zero; the other two include it. The value -0.556 does not survive
  either variant.
- Versus statistical: -0.398 as recorded, -0.164 with the model removed, +0.250 on MSE[0.5]. The sign
  changes. The MSE[0.5] value is the same as the MAE value (+0.251). All three intervals
  include zero.
- Versus pretrained: -0.520, -0.654 and -0.526. This is the one MSE comparison that stays
  negative with an interval that excludes zero in every variant. Section 2 shows that it still
  depends on one pretrained model under MSE[0.5].
- On the MSE[0.5] column, removing `Toto_Open_Base_1.0` changes little (-0.347 to -0.334, +0.250 to
  +0.242, -0.526 to -0.481), which confirms that the issue is that model's MSE[mean] column and
  not the model.

What the student can claim about MSE:

- "Computed as in the existing record (MSE[mean], family mean), the correlations are -0.556,
  -0.398 and -0.520, and they depend on one model's MSE[mean] column." Always with the table
  above.
- "On the squared error of the median forecast (MSE[0.5]) the picture matches MAE: the zero-shot
  family has the lower error on 17, 18 and 19 of 20 datasets, and the correlations with Omega are
  -0.347, +0.250 and -0.526."

What the student cannot claim about MSE:

- That the MSE -0.556 replicates the paper's published -0.556. The published figure is an
  sMAPE figure (section 3).
- That the relationship is stronger, or more significant, on MSE than on sMAPE or MAE.
- That the zero-shot family is worse than the other families on MSE, or that MSE and MAE rank
  the families in opposite ways.
- That the sign against statistical baselines reverses between MSE and MAE. On MSE[0.5] it has
  the MAE sign.
- Anything about whether `Toto_Open_Base_1.0`'s published MSE[mean] values are right or wrong.

### 2. Leave one model out, every metric

`leave_one_model_out.csv` recomputes each family-mean correlation with one model removed, for
every mapped model present in the results: the 18 zero-shot models (each changes all three comparisons)
and the 9 deep, 5 statistical and 11 pretrained models (each changes the comparison against its own
family). `leave_one_model_out_summary.csv` gives the range. The reviewer asked for the zero-shot
family; the comparison families were added because the same question applies to them, and that
is where the dependence turned out to be.

Removing one zero-shot model at a time (18 removals per row):

| Metric | Versus | rho, all models | Range of rho over the removals | Largest change (model removed) | Sign changes | Interval switches | Zero-shot lower, range (all models) |
|---|---|---|---|---|---|---|---|
| sMAPE | deep | -0.417 | -0.468 to -0.374 | 0.051 (`Moirai_base`) | 0 | 0 | 20 to 20 (20) |
| sMAPE | statistical | -0.123 | -0.135 to -0.116 | 0.012 (`Moirai_base`) | 0 | 0 | 15 to 15 (15) |
| sMAPE | pretrained | -0.514 | -0.576 to -0.495 | 0.062 (`YingLong_6m`) | 0 | 0 | 20 to 20 (20) |
| MSE | deep | -0.556 | -0.556 to -0.454 | 0.102 (`Toto_Open_Base_1.0`) | 0 | 1 | 8 to 15 (8) |
| MSE | statistical | -0.398 | -0.426 to -0.164 | 0.235 (`Toto_Open_Base_1.0`) | 0 | 0 | 5 to 14 (5) |
| MSE | pretrained | -0.520 | -0.654 to -0.514 | 0.134 (`Toto_Open_Base_1.0`) | 0 | 0 | 6 to 16 (6) |
| MAE | deep | -0.248 | -0.298 to -0.227 | 0.050 (`Moirai_base`) | 0 | 0 | 18 to 19 (19) |
| MAE | statistical | +0.251 | +0.212 to +0.281 | 0.039 (`Moirai_small`) | 0 | 0 | 18 to 18 (18) |
| MAE | pretrained | -0.405 | -0.505 to -0.347 | 0.101 (`Moirai_base`) | 0 | 1 | 19 to 19 (19) |
| CRPS | deep | -0.280 | -0.317 to -0.253 | 0.038 (`Moirai_base`) | 0 | 0 | 20 to 20 (20) |
| CRPS | statistical | +0.376 | +0.305 to +0.380 | 0.071 (`Moirai_base`) | 0 | 0 | 19 to 20 (19) |
| CRPS | pretrained | -0.415 | -0.456 to -0.382 | 0.041 (`Moirai_base`) | 0 | 0 | 20 to 20 (20) |
| MSE[0.5] | deep | -0.347 | -0.415 to -0.334 | 0.068 (`Moirai_base`) | 0 | 0 | 16 to 18 (17) |
| MSE[0.5] | statistical | +0.250 | +0.214 to +0.274 | 0.036 (`Moirai_small`) | 0 | 0 | 18 to 19 (18) |
| MSE[0.5] | pretrained | -0.526 | -0.603 to -0.463 | 0.077 (`Moirai_large`) | 0 | 0 | 19 to 19 (19) |

Removing one model of the comparison family at a time (9 deep, 5 statistical, 11 pretrained
removals):

| Metric | Versus | rho, all models | Range of rho over the removals | Largest change (model removed) | Sign changes | Interval switches | Zero-shot lower, range (all models) |
|---|---|---|---|---|---|---|---|
| sMAPE | deep | -0.417 | -0.456 to +0.195 | 0.612 (`Crossformer`) | 1 | 0 | 19 to 20 (20) |
| sMAPE | statistical | -0.123 | -0.141 to -0.026 | 0.098 (`Auto_Arima`) | 0 | 0 | 14 to 17 (15) |
| sMAPE | pretrained | -0.514 | -0.605 to -0.048 | 0.466 (`Lag-Llama`) | 0 | 2 | 17 to 20 (20) |
| MSE | deep | -0.556 | -0.556 to -0.371 | 0.185 (`Crossformer`) | 0 | 1 | 5 to 9 (8) |
| MSE | statistical | -0.398 | -0.426 to -0.340 | 0.059 (`Auto_Arima`) | 0 | 0 | 4 to 5 (5) |
| MSE | pretrained | -0.520 | -0.520 to -0.459 | 0.062 (`Lag-Llama`) | 0 | 0 | 3 to 6 (6) |
| MAE | deep | -0.248 | -0.268 to +0.117 | 0.365 (`Crossformer`) | 1 | 0 | 17 to 19 (19) |
| MAE | statistical | +0.251 | +0.223 to +0.408 | 0.156 (`Auto_Arima`) | 0 | 0 | 17 to 18 (18) |
| MAE | pretrained | -0.405 | -0.511 to +0.012 | 0.417 (`Lag-Llama`) | 1 | 1 | 16 to 19 (19) |
| CRPS | deep | -0.280 | -0.299 to -0.030 | 0.250 (`Crossformer`) | 0 | 0 | 20 to 20 (20) |
| CRPS | statistical | +0.376 | +0.221 to +0.450 | 0.155 (`Auto_ETS`) | 0 | 0 | 19 to 19 (19) |
| CRPS | pretrained | -0.415 | -0.496 to -0.078 | 0.337 (`Lag-Llama`) | 0 | 0 | 19 to 20 (20) |
| MSE[0.5] | deep | -0.347 | -0.364 to +0.098 | 0.445 (`Crossformer`) | 1 | 0 | 15 to 18 (17) |
| MSE[0.5] | statistical | +0.250 | +0.165 to +0.314 | 0.084 (`Auto_ETS`) | 0 | 0 | 17 to 18 (18) |
| MSE[0.5] | pretrained | -0.526 | -0.588 to -0.023 | 0.504 (`Lag-Llama`) | 0 | 1 | 17 to 19 (19) |

"Sign changes" counts removals after which rho has the other sign. "Interval switches" counts
removals after which the bootstrap interval changes between excluding zero and including it.

What the table shows, metric by metric:

- Zero-shot family, sMAPE, MAE, CRPS and MSE[0.5]: no headline depends on a single zero-shot
  model. Over all 12 rows the largest change in rho from removing one zero-shot model is
  0.101 (MAE versus pretrained, `Moirai_base`), no sign changes, and the sign counts move by
  at most 1 dataset. One interval switches: MAE versus pretrained is -0.405
  [-0.789, +0.103] with all models and -0.505 [-0.807, -0.042] without `Moirai_base`; the interval was
  already close to zero on one side.
- Zero-shot family, MSE[mean]: the dependence on `Toto_Open_Base_1.0` described in section 1. It is the model
  with the largest effect in all three MSE rows (changes of 0.102, 0.235 and 0.134), and
  the sign counts move from 8, 5 and 6 to 15, 14 and 16.
- Deep family: every "versus deep" correlation moves towards zero or changes sign when
  `Crossformer` is removed. Without it the correlation
  is +0.195 [-0.272, +0.597] on sMAPE (from -0.417), -0.371 on MSE (from -0.556),
  +0.117 on MAE (from -0.248), -0.030 on CRPS (from -0.280) and +0.098 on MSE[0.5]
  (from -0.347). The sign changes on sMAPE, MAE and MSE[0.5]. The sign counts barely move
  (19 of 20 on sMAPE without it, 20 with it): the zero-shot family is still better than the
  deep family almost everywhere; what goes away is the trend of that advantage with Omega.
- Pretrained family: every "versus pretrained" correlation except the MSE[mean] one is close to
  zero when `Lag-Llama` is removed. Without it the correlation is -0.048 [-0.580, +0.464] on sMAPE (from -0.514), +0.012 on MAE
  (from -0.405), -0.078 on CRPS (from -0.415) and -0.023 on MSE[0.5] (from -0.526). On
  MSE[mean] the change is 0.062; there the zero-shot mean is dominated by `Toto_Open_Base_1.0`.
  For sMAPE the interval also comes to include zero when `TTM-R2-Pretrained` (-0.376 [-0.734, +0.121]) is removed.
- Statistical family: no dependence on a single model. The largest change is 0.156
  (MAE, `Auto_Arima`), with no sign change and no interval switch in any row.

These two models are not outliers of the kind `Toto_Open_Base_1.0` is on MSE[mean]. On sMAPE `Crossformer`
carries a median 0.160 of the deep family's sum (an even share is 0.111) and has the
highest sMAPE in its family on 11 of 20 datasets; `Lag-Llama` carries 0.104 (even share
0.091) and has the highest on 14. Each is its family's weakest member on more than half of the
datasets, and a family mean follows its weakest member. This is the same thing the existing record found from
the other side, when the correlation went away under the median and the best member (section 4).

What follows for the claim. The like-for-like replication value (sMAPE, zero-shot versus deep,
-0.417) is the correlation of Omega with how much worse the mean of nine deep models is, and that
mean moves with one of the nine. It is correct to report -0.417 as what the stated computation
gives. It is not correct to describe it as a property of the deep family as a whole without
saying that it is +0.195 without `Crossformer`. The same applies to sMAPE versus pretrained
(-0.514, and -0.048 without `Lag-Llama`). Nothing here says either model's published scores are
wrong, and no model was removed from any headline number.

### 3. What the source paper says, and the like-for-like comparison

The text of the paper (arXiv:2511.08884v1) was read in this session from a local text copy,
`paper_pdf.txt`; line numbers below refer to that file.

- The family comparison is defined on sMAPE only. Section 5 analyses "51 models from the GIFT-Eval
  Time Series Forecasting Leaderboard, using their reported sMAPE performance across 28 datasets"
  (lines 263 to 265). The comparison statistic is defined at lines 339 to 344: "we compared
  relative accuracies between model types for each dataset. For each model pair A -> B evaluated
  on dataset i: Delta sMAPE A->B (i) = 100 x (sMAPE(A, i) - sMAPE(B, i)) / sMAPE(A, i)."
  (The formula is transcribed here in plain text.)
- The published correlations are for that quantity: "Spearman correlations ranging from -0.234
  for statistical to -0.556 for deep learning" (lines 401 to 402). No number is given for the
  pretrained comparison, and no p-values or sample size are given with these two.
- MSE is used only in the controlled experiments: "we verify the robustness of our observations
  using the popular MSE metric on controlled experiments" (lines 224 to 226). There is no MSE
  version of the family comparison in the paper.
- The paper does not say how the scores of a family's models are combined into one score per
  dataset for these correlations. It speaks of a "model pair A -> B" and of "model types", and
  does not say whether A and B are single models or family summaries, nor which summary. The
  only averaging it describes is for a different figure (line 282, binned averages by Omega).
  So the label "mean across family (paper's choice)" in the existing record is not supported
  by the paper. The family mean is this replication's choice.

Consequences for the record:

| Zero-shot versus | Published (sMAPE, 28 datasets) | This replication on sMAPE (n = 20) | 95% bootstrap interval | Published value inside the interval | MSE[mean] row, not like for like |
|---|---|---|---|---|---|
| deep | -0.556 | -0.417 | [-0.793, +0.091] | yes | -0.556 |
| statistical | -0.234 | -0.123 | [-0.603, +0.397] | yes | -0.398 |
| pretrained | not given | -0.514 | [-0.800, -0.054] | not applicable | -0.520 |

- The like-for-like replication of the published -0.556 is the sMAPE value -0.417, not the MSE
  value -0.556. The existing findings note says the MSE comparison "matches the published
  figure exactly"; the two numbers are equal to three decimals, but they measure different
  things, and the MSE one depends on one model's column (section 1). The statement should be
  withdrawn.
- On the paper's metric the replication is weaker than published in both comparisons that have a
  published value (-0.417 against -0.556, -0.123 against -0.234). Both published values lie inside
  the replication's bootstrap interval, and so does zero. With 20 datasets this replication can
  neither confirm nor contradict the published magnitudes.
- The comparison is like for like in metric and statistic only. It differs from the paper in the
  datasets (20 base datasets, those with an Omega value in the current manifest, against 28 in the
  paper), in the Omega values (the team's own; the existing findings note says they are computed
  on training regions, where the paper used full datasets), and possibly in how a family score is
  formed, which the paper does not state.
- The family lists in `families.py` were compared name by name with the paper's Appendix A.1
  (lines 658 to 665) in this session: the four lists have 19, 11, 9 and 5 names in both, and
  agree except for capitalisation in two names (`Chronos_bolt_base` and `Chronos_bolt_small` in
  the paper, `chronos_bolt_base` and `chronos_bolt_small` in the result files).

Labels. The recorded file `within_family_sensitivity.csv` keeps its recorded label text,
including "mean across family (paper's choice)", because a regression test compares it byte for
byte with the record. The corrected labels are written only to the new file
`within_family_sensitivity_relabelled.csv` and to the run manifest:

| Recorded label (legacy file) | Corrected label (new file) | rho | p |
|---|---|---|---|
| mean across family (paper's choice) | mean across family (this replication's choice; the paper does not state one) | -0.417 | 0.0677 |
| median model in family | median over the family's model-configuration rows | +0.045 | 0.8502 |
| best model in family | lowest single model-configuration row in the family | -0.063 | 0.7914 |
| (not in the recorded file) | median model in family (each model averaged over configurations first) | -0.105 | 0.6587 |
| (not in the recorded file) | best model in family (each model averaged over configurations first) | +0.009 | 0.9699 |

### 4. Within-family sensitivity, all metrics and comparisons

The existing record reports this only for sMAPE versus deep. The recorded "median model in
family" and "best model in family" rows are computed over every (model, configuration) row of a
family inside a base dataset. The "best model" row is therefore the single lowest score on any
one configuration, and the "median model" row is the median over model-configuration rows.
Neither is a model. An analysed base dataset has between 1 and 8 configurations. The table below reports
both poolings: "rows" is the recorded behaviour, kept unchanged; "models" first averages each
model over the dataset's configurations and then takes the median or the best over models, which
is what the recorded labels say. For the mean the two are identical (checked by a test).

rho with 95% bootstrap interval, n = 20 in every cell.

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

Reading: for sMAPE and MSE the negative correlation appears under the family mean. Under the
model-level median and best model it is close to zero for all three comparisons (largest
magnitude 0.208, MSE versus statistical). That statement holds for the model-level pooling only:
under the recorded row pooling, MSE versus deep is -0.254 under the median and -0.349 under the
best, which is not close to zero (both intervals include zero). The recorded statement for
sMAPE versus deep holds under both poolings: -0.417 under the mean; +0.045 and -0.063 under
the recorded median and best; -0.105 and +0.009 under the model-level median and best. For CRPS versus
deep the median rows are not near zero (-0.307 and -0.317), so "disappears under the median"
should not be stated for every metric. Every interval in the median and best columns includes
zero except CRPS versus statistical under the recorded best (+0.532). The MSE rows of this table
use the MSE[mean] column and carry the caveat of section 1.

Sign counts, n = 20, no ties in any cell: number of datasets where the zero-shot family has the
lower error.

| Metric | Versus | Mean | Median, rows | Best, rows | Median, models | Best, models |
|---|---|---|---|---|---|---|
| sMAPE | deep | 20 | 19 | 16 | 19 | 19 |
| sMAPE | statistical | 15 | 14 | 17 | 15 | 15 |
| sMAPE | pretrained | 20 | 17 | 10 | 16 | 17 |
| MSE | deep | 8 | 16 | 12 | 17 | 10 |
| MSE | statistical | 5 | 20 | 20 | 18 | 18 |
| MSE | pretrained | 6 | 10 | 11 | 12 | 10 |
| MAE | deep | 19 | 16 | 15 | 17 | 15 |
| MAE | statistical | 18 | 18 | 20 | 17 | 18 |
| MAE | pretrained | 19 | 13 | 10 | 12 | 12 |
| CRPS | deep | 20 | 18 | 12 | 19 | 18 |
| CRPS | statistical | 19 | 19 | 19 | 20 | 19 |
| CRPS | pretrained | 20 | 18 | 12 | 16 | 16 |

### 5. Unit of analysis: configuration level (sensitivity only)

Unchanged from the first version of this note; the table is byte-identical. The headline uses
one row per base dataset. The same correlation was computed with one row per GIFT-Eval
configuration (base/frequency/term). The results file has 97 configurations over 28 base
datasets. Two ways of giving a configuration an Omega value were run and both are reported.

- Dataset and frequency (the briefed rule): a configuration gets the mean Omega of the manifest
  subset with the same base dataset and frequency, through the existing subset-name map. All terms
  (short, medium, long) of that dataset and frequency share the value. 71 of 97 configurations
  join, from 29 dataset-frequency subsets in 20 base datasets. The 26 that do not join are listed
  in the run manifest: 8 belong to base datasets with no Omega at all, 18 belong to the 20 analysed
  datasets but to a frequency the manifest has no Omega for (for example electricity/W, solar/D).
- Dataset broadcast: every configuration of a base dataset gets that dataset's single Omega (the
  value the headline uses), including frequencies with no Omega of their own. 89 of 97 join, with
  20 distinct Omega values.

Family mean, zero-shot versus each family. The interval resamples the 20 base datasets and takes
all configurations of a drawn dataset together. The p-value treats configurations as independent,
which they are not, and is shown only so the quoted figure can be compared.

| Metric | Versus | Dataset level (n = 20) | Dataset and frequency (n = 71) | Dataset broadcast (n = 89) |
|---|---|---|---|---|
| sMAPE | deep | -0.417 | -0.165 [-0.615, +0.227], p = 0.168 | -0.088 [-0.519, +0.266], p = 0.411 |
| sMAPE | statistical | -0.123 | -0.089 [-0.510, +0.402], p = 0.459 | -0.058 [-0.516, +0.432], p = 0.588 |
| sMAPE | pretrained | -0.514 | -0.337 [-0.638, -0.024], p = 0.004 | -0.328 [-0.591, -0.017], p = 0.002 |
| MSE | deep | -0.556 | -0.191 [-0.494, +0.110], p = 0.110 | -0.199 [-0.470, +0.061], p = 0.062 |
| MSE | statistical | -0.398 | -0.163 [-0.439, +0.139], p = 0.173 | -0.141 [-0.409, +0.127], p = 0.188 |
| MSE | pretrained | -0.520 | -0.242 [-0.553, +0.087], p = 0.042 | -0.274 [-0.541, -0.022], p = 0.009 |
| MAE | deep | -0.248 | -0.097 [-0.520, +0.271], p = 0.421 | -0.069 [-0.474, +0.299], p = 0.518 |
| MAE | statistical | +0.251 | -0.151 [-0.404, +0.191], p = 0.208 | -0.113 [-0.377, +0.211], p = 0.293 |
| MAE | pretrained | -0.405 | -0.375 [-0.660, -0.058], p = 0.001 | -0.398 [-0.650, -0.067], p < 0.001 |
| CRPS | deep | -0.280 | -0.054 [-0.413, +0.251], p = 0.653 | -0.051 [-0.369, +0.243], p = 0.633 |
| CRPS | statistical | +0.376 | +0.133 [-0.191, +0.482], p = 0.267 | +0.200 [-0.050, +0.442], p = 0.061 |
| CRPS | pretrained | -0.415 | -0.306 [-0.595, -0.026], p = 0.010 | -0.363 [-0.625, -0.057], p < 0.001 |

The quoted figure (rho = -0.088, p = 0.41 for sMAPE zero-shot versus deep): it reproduces, but
only under the dataset-broadcast rule (n = 89, rho = -0.088, p = 0.411). Under the briefed
dataset-and-frequency rule it does not: the result is rho = -0.165, p = 0.168, n = 71. Anyone
quoting -0.088 must say that each dataset's Omega was copied to all of its configurations,
including 18 configurations whose frequency has no Omega in the manifest. Under either rule the
sMAPE and MSE correlations against deep and statistical are smaller in magnitude than at dataset
level and their intervals include zero; the comparisons against pretrained keep intervals that
exclude zero under at least one rule. The MAE versus statistical sign is positive at dataset level
(+0.251) and negative at configuration level (-0.151, -0.113); all three intervals include zero.
The same table for the median and best summaries is in `correlations_config_level.csv`. The MSE
rows use the MSE[mean] column and carry the caveat of section 1; the leave-one-model-out check
was not run at configuration level.

### 6. Guards (what was fixed, and what the run manifest records)

Held-out guard. The rule itself stays in `analysis/decision/guards.py` (a name containing `m5`
or `pjm`, ignoring case), which was not edited. What changed is where it is applied and what
is shown:

- Manifest: every text field of every active row is checked, not only `subset`. In the current
  manifest these are `series_id`, `subset`, `status`; a `dataset` or `config` field in a later layout is
  covered without a code change. The reviewer's case, a series with id `m5_retail_item_0001`
  under subset `electricity_H`, now stops the run and the message names the field, the value,
  the subset and the row count.
- The manifest's own `held_out_strict` list (`M5_Retail`, `PJM_Electricity`) is read and applied, to the manifest
  rows and to the results file. An entry matches on letters and digits only, so `M-5_Retail` and
  `M5 Retail` match `M5_Retail`.
- Nothing is dropped or refused silently. Every results dataset dropped by the rule is printed
  with its row count and the reason, with a line saying it may be a false positive. The
  reviewer's made-up `ram5gb_cluster/H/short` is now printed as dropped.
- Second line of defence: the tokens are also looked for after removing everything that is not
  a letter or a digit, which catches `M-5`, `P.J.M`, `pj_m_hourly`. This rule has false positives
  (`item_5` becomes `item5`), so a match is printed and recorded and is not acted on.
- In this run: 0 results rows excluded, 0 manifest rows refused, and no name matched the
  second rule in either file.

Omega column switch (`--omega-col`), the path the 4,096-step rerun will use:

- Only float columns are accepted. A boolean, integer or text column whose name starts with
  `omega` stops the run with a message saying what the column holds.
- If the chosen column has no finite value for any series, the run stops and lists each affected
  base dataset with its counts. The reviewer's case (a column empty for one whole dataset, which
  silently gave n = 19) now stops. `--allow-missing-omega` continues instead, removes datasets
  and frequency subsets that have no value, and prints the changed n in a banner; the run
  manifest records what was removed. On the reviewer's case rebuilt in this session the banner
  read "n CHANGES: 19 base datasets have an Omega value, not 20; 27 dataset-frequency subsets, not 29.".
- A manifest column named `omega_sd` stops the run, because the command computes a column of
  that name (the spread of the selected Omega column) and would overwrite it.
- In this run the selected column has a value for all 8,761 series used (0 missing).

Other checks added: the output directory may not lie inside an existing run directory;
`--bootstrap-resamples` below 1 and a negative `--seed` are reported as one-line errors;
`bootstrap_interval` raises on NaN or infinite input instead of returning an interval; the
command prints a warning, and records it in the run manifest, for any table row where more than
5% of the resamples were degenerate (none in this run); the run manifest records the manifest
layout version and a frozen flag.

Roster and manifest (from the run manifest):

- Roster: 129 models in the results; 43 are in the family mapping and present, 86 are ignored. One
  mapped model, `granite-flowstate-r1`, is absent from the results file, so the zero-shot family is
  18 models, not 19. Deep 9, statistical 5, pretrained 11.
- Manifest: 8,782 active series read, 8,761 used, over 29 mapped subsets and 20 base datasets. The
  subset `jena_weather` (21 series) is left out as a duplicate of `jena_weather_10T`; the reader
  checked on this run that lengths and Omega values are identical. No subset was unmapped. 7
  mapped subset names are not in this manifest: `bitbrains_fast_storage_H`, `bitbrains_rnd_H`, `ett1_D`, `ett2_D`, `jena_weather_D`, `m4_hourly`, `us_births_W`.

### 7. Reproduction of the recorded numbers

| Recorded item | Result |
|---|---|
| `summary_stats.csv` (12 correlations, n, p) | Byte-identical to the file written by this run (SHA-256 `a2c693085e82...` for both). |
| `within_family_sensitivity.csv` (3 rows) | Byte-identical to the file written by this run (SHA-256 `87158a3086c6...` for both), including the recorded label text. |
| `joined_table.csv` (80 rows) | All values identical. The recorded file has its columns in a different order from the one the code writes. The command keeps the code's order. |
| Median-model p-value, recorded as 0.850 or 0.851 | The computed value is 0.850206: 0.8502 in the CSV, 0.850 at three decimals. The "0.851" in `k2a-k2b-findings.md` is not what the computation gives. |
| Counts in the findings note: 129 models, 97 configurations, 12,416 rows, 20 datasets | Reproduced. |

Not checked in this run: the figure, and the other prose numbers in `k2a-k2b-findings.md` (for
example the Omega versus length correlation and the median-Omega variant).

### 8. Tests and planted bugs

    PYTHONPATH=src python -m unittest discover -s tests/unit/k2 -t .
    Ran 149 tests in 34.933s
    OK

Wall time 37 seconds with other jobs running on the same machine; the limit for a test directory is about 60 seconds. The first version had 71 tests.

The first version of this note said that no reference number is typed into a test. That was not
strictly true and the sentence is withdrawn. The regression tests read the recorded CSV files
and compare, and they also type the counts 97, 71, 89, 29, 20 and 28 (configurations, joined
configurations, subsets, datasets), each next to an independent recount in the same test, and
the formatted p-value "0.850". The tests added in this version for the new tables on the
reference inputs type no reference number: each value is compared with another table of the
same run or with a recomputation in plain pandas and scipy inside the test.

Value-level tests were added for every table on tiny hand-built inputs whose answers are worked
by hand in the test files (for example a five-dataset table in which the five within-family
summaries give rho = -0.1, +0.5, +1.0, +0.3 and +0.8, and a table in which one model carries a
known 0.75 median share of the family's MSE). These inputs are made up; the tests are mechanism
checks and say nothing about real model families.

Planted bugs. 57 bugs were planted one at a time and the suite run each time: the reviewer's
eight survivors, the reviewer's other twelve (to check nothing regressed), and 37 new ones
covering each thing added in this version. The planting was done in two isolated copies of the
tree, never in the shared working tree. Each file was restored from a backup in a `finally`
block and the copy compared with the backup after every bug; at the end both copies and the
working tree were compared with the backup (all identical) and the suite passed in both copies.
The first ten bugs were run against the whole suite; the others with the suite stopped at its
first failure, and a survivor confirmed by a full run. Result: 55 caught, 2 survived.

The two survivors and what was done about them:

- N26 removed one line of the `--allow-missing-omega` path (dropping base datasets with no Omega
  value). The suite still passed because the next line, which drops every frequency subset with
  no value, already removes those datasets: the planted change had no effect on behaviour. The
  redundant line was deleted from the code. A bug on the remaining line (N26b: datasets and
  subsets with no value are kept) was then planted and caught.
- N30 (`n_series` counts series that have no Omega value) was a real gap: no test had a subset in
  which only some series lack a value. A test with that case was added, the bug replanted, and it
  was caught.

After these two changes the planting harness was run again on fresh copies (N26b and N30, both
caught), with the same restore and comparison checks, and the outputs of the command were
regenerated; they are the ones this note reports.

The reviewer's eight survivors:

| Planted bug | Outcome | A test that fails |
|---|---|---|
| B2: configuration-level intervals ignore --seed | caught | `test_configuration_level_values_follow_column_matching_and_seed` |
| B4: dataset_deltas.csv delta divided by the other family's score | caught | `test_delta_is_relative_to_the_zero_shot_score` |
| B8: dataset-and-frequency Omega always uses omega_levels | caught | `test_configuration_level_values_follow_column_matching_and_seed` |
| B10: omega_sd always from omega_levels | caught | `test_omega_tables_follow_the_selected_column` |
| B11: "Best, models" column actually computes the median | caught | `test_every_summary_column_has_its_own_known_rho` |
| B17: version-pin check compares major.minor only | caught | `test_a_difference_in_the_last_version_part_is_a_mismatch` |
| B19: sign counts for "models" pooling use row pooling | caught | `test_sign_counts_known_answers_for_both_poolings` |
| B20: omega_dataset_frequency column in joined_config_table.csv set to a constant | caught | `test_joined_config_table_holds_both_omega_values_per_configuration` |

New bugs, in fix-list order (N1 to N12 item 1, N13 to N15 item 2, N16 to N23 item 3, N24 to N30
item 4, N31 to N37 item 6):

| Planted bug | Outcome | A test that fails |
|---|---|---|
| N1: MSE table: the 'dominant model removed' rows do not remove it | caught | `test_command_prints_the_mse_table_beside_the_headline_and_writes_it` |
| N2: MSE table: the MSE[0.5] variant reads the MSE[mean] column | caught | `test_other_variants_equal_an_independent_recomputation` |
| N3: leave one model out: the model is not removed | caught | `test_leave_one_model_out_equals_an_independent_recomputation` |
| N4: leave one model out: rho_change has the wrong sign | caught | `test_leave_one_model_out_rows` |
| N5: leave one model out: removing a model of another family writes rows for all comparisons | caught | `test_leave_one_model_out_rows` |
| N6: summary: sign changes counted as sign agreements | caught | `test_summary_is_derived_from_the_rows` |
| N7: summary: largest change taken as the largest positive change | caught | `test_summary_is_derived_from_the_rows` |
| N8: summary: interval status changes counted as non-changes | caught | `test_summary_is_derived_from_the_rows` |
| N9: dominant model picked as the smallest share | caught | `test_command_prints_the_mse_table_beside_the_headline_and_writes_it` |
| N10: diagnostics: mean share reported as the median share | caught | `test_command_prints_the_mse_table_beside_the_headline_and_writes_it` |
| N11: command picks the dominant model on sMAPE, not MSE | caught | `test_command_prints_the_mse_table_beside_the_headline_and_writes_it` |
| N12: leave-one-out fast path averages model means, not rows | caught | `test_leave_one_model_out_equals_an_independent_recomputation` |
| N13: paper table: 'like for like' value taken from MSE[mean] | caught | `test_paper_table_carries_the_smape_rows` |
| N14: relabelled table still uses the recorded labels | caught | `test_command_prints_the_mse_table_beside_the_headline_and_writes_it` |
| N15: legacy label text changed (breaks byte identity with the record) | caught | `test_command_prints_the_mse_table_beside_the_headline_and_writes_it` |
| N16: manifest guard checks the subset field only | caught | `test_dropped_results_rows_are_printed_and_fuzzy_names_are_printed_and_kept` |
| N17: manifest's held_out_strict list ignored | caught | `test_manifest_list_reaches_the_results_and_the_run_manifest` |
| N18: held_out_strict list not applied to the results | caught | `test_manifest_list_reaches_the_results_and_the_run_manifest` |
| N19: fuzzy matches not reported at all | caught | `test_dropped_results_rows_are_printed_and_fuzzy_names_are_printed_and_kept` |
| N20: fuzzy matches silently dropped from the results | caught | `test_dropped_results_rows_are_printed_and_fuzzy_names_are_printed_and_kept` |
| N21: dropped results rows not printed | caught | `test_dropped_results_rows_are_printed_and_fuzzy_names_are_printed_and_kept` |
| N22: list entries compared with punctuation left in | caught | `test_manifest_list_reaches_the_results_and_the_run_manifest` |
| N23: fuzzy manifest matches not printed by the command | caught | `test_dropped_results_rows_are_printed_and_fuzzy_names_are_printed_and_kept` |
| N24: boolean and integer Omega columns accepted | caught | `test_omega_column_must_be_a_float_column` |
| N25: missing Omega values do not stop the run | caught | `test_omega_column_missing_for_a_dataset_stops_the_run` |
| N26: with the flag, a dataset with no Omega value is kept | survived: the planted change had no effect (see below) | none |
| N26b: with the flag, datasets and subsets with no Omega value are kept | caught | `test_allow_missing_omega_prints_the_changed_n_and_records_it` |
| N27: infinite Omega counted as a value | caught | `test_column_missing_for_some_series_stops_and_gives_the_counts` |
| N28: omega_sd collision guard removed (both places) | caught | `test_colliding_manifest_column_is_a_clean_error` |
| N29: changed n not printed under --allow-missing-omega | caught | `test_allow_missing_omega_prints_the_changed_n_and_records_it` |
| N30: n_series counts series without an Omega value | survived at first; caught after a test was added | `test_series_counts_under_the_flag_count_only_series_with_a_value` |
| N31: output directory inside an existing run accepted | caught | `test_check_outdir_accepts_a_new_directory_under_plain_directories` |
| N32: bootstrap_interval accepts NaN | caught | `test_non_finite_input_is_refused` |
| N33: zero resamples accepted | caught | `test_bad_resample_count_and_seed_are_clean_errors` |
| N34: negative seed not checked | caught | `test_bad_resample_count_and_seed_are_clean_errors` |
| N35: degenerate-resample warning never fires | caught | `test_cells_above_the_threshold_are_listed` |
| N36: manifest always recorded as frozen | caught | `test_clean_run_prints_no_warning` |
| N37: missing metric column gives a raw traceback | caught | `test_results_without_a_needed_metric_column_is_a_clean_error` |

The reviewer's other twelve (B3, B1, B5, B6, B7, B9, B12, B13, B14, B15, B16, B18): 12 of 12 caught again.

Not exercised: the figure code (matplotlib absent; only the "matplotlib not installed" path is
tested) and the download (`--fetch-results`, needs the network and `requests`, neither pinned).

## Design choices needing sign-off

New in this version:

| Choice | Alternative | Reason |
|---|---|---|
| The recorded MSE headline is kept as computed, and the sensitivity table is printed directly under it by the command and placed beside it in this note. | Replace MSE[mean] by MSE[0.5] in the headline, or drop `Toto_Open_Base_1.0` from the zero-shot family. | The recorded numbers must stay reproducible, and removing a model or switching a column after seeing the result is a decision for the team, not for the code. The table makes the dependence visible wherever the MSE row is shown. |
| The model removed in the MSE table is chosen by a rule: the zero-shot model with the largest median share of the family's summed MSE[mean] across the analysed datasets. | Name `Toto_Open_Base_1.0` in the code. | A named model would be a choice made after looking at the result. The rule is recorded in the run manifest and picks whatever model dominates on a later results file or manifest. |
| Leave one model out covers the comparison families as well as the zero-shot family. | Zero-shot family only, as the review asked. | The question "does a headline depend on one model" applies equally to the nine deep, five statistical and eleven pretrained models, and that is where a dependence was found. |
| The leave-one-model-out and MSE tables use the family mean only. | Repeat them for the median and best summaries. | The dependence on one model is a property of the mean, which is the headline summary. The median and best are already in section 4. |
| A strict held-out match is acted on (manifest refused, results rows dropped and printed). A punctuation-insensitive match on the tokens is printed and recorded, and not acted on. | Stop the run on a punctuation-insensitive match as well. | The loose rule has ordinary false positives (`item_5`), so stopping on it would block clean manifests. The review asked for it to be reported. The cost is that a held-out series whose name only matches loosely enters the run with a printed warning; a person must read the warning. The manifest's own `held_out_strict` entries are matched on letters and digits and are acted on, because a whole entry such as `m5retail` is specific. |
| Every text field of a manifest row is treated as a name field. | A fixed list (`subset`, `series_id`, `dataset`, `config`). | The frozen layout's field names are not known. Checking every text field cannot miss a renamed field; the cost is a possible false positive in a free-text field, which stops the run with a message naming the field. |
| A missing Omega value stops the run unless `--allow-missing-omega` is given. With the flag, datasets with no value are removed and the changed n is printed in a banner. | Always continue and report n, or never allow it. | A dataset silently leaving the analysis changes every correlation. Stopping by default makes the 4,096-step rerun fail loudly if the column is incomplete; the flag exists because a partially filled column may be a legitimate interim state. |
| Only float columns can be selected as Omega; integer columns are refused along with boolean and text. | Accept any numeric column. | A boolean column is numeric to pandas and gave NaN correlations with exit 0. An all-integer column is not a plausible Omega. |
| A manifest column named `omega_sd` stops the run. | Rename the computed column. | The recorded `joined_table.csv` has a column of that name and the regression test compares it, so the computed name cannot change. |
| `n_series` counts the series that have a value in the selected Omega column. | Count all series, as before. | Identical when no value is missing (this run), and correct under `--allow-missing-omega`. |
| At least 1 bootstrap resample and a seed of 0 or more are required; anything else is a one-line error. | A higher minimum number of resamples. | The review asked for clean errors. A sensible minimum for a 95% interval is a judgement the team can add. |
| A row is flagged when more than 5% of its resamples are degenerate. | Another threshold, or fail. | Degenerate resamples are left out of the percentiles, so a large share means the interval rests on fewer resamples than requested. The threshold is a constant in `statistic.py`. |
| The manifest is recorded as frozen only if the file has a boolean `frozen` key set to true. | Infer it from the layout version or the file name. | The current file says nothing about being frozen, and guessing would put a wrong flag in the record. |
| The corrected within-family labels go into a new file; the legacy file keeps the recorded text. | Correct the legacy file. | The legacy file is compared byte for byte with the record. |

Carried over from the first version, unchanged:

| Choice | Alternative | Reason |
|---|---|---|
| Percentile bootstrap over the 20 base datasets, 10,000 resamples, fixed seed. | Fisher z interval, or a bias-corrected bootstrap. | The dataset is the independent unit and the percentile interval is the simplest one to explain. It is approximate at n = 20, as stated above. |
| Configuration-level intervals resample base datasets, taking all of a dataset's configurations together. | Resample the 71 or 89 configurations as if independent. | Configurations of one dataset share data and Omega. Resampling them separately would give a narrower interval that overstates the evidence. |
| Configuration-level default matching is dataset and frequency; the dataset-broadcast rule is reported beside it. | Report only one. | The briefed rule uses the Omega that belongs to the configuration's data. The broadcast rule is the only one that reproduces the quoted -0.088, so both are shown and labelled. |
| Kept the recorded "median model" and "best model" computation (over model-configuration rows) and added the model-level version beside it. | Replace the recorded computation. | The task was to keep recorded numbers identical. The team should choose which one the paper reports; the model-level version is the one that matches the recorded labels. |
| Held-out rows in the results file are removed and counted; a held-out row in the manifest's development list stops the run. | Stop in both cases, or drop in both. | The results file is third-party and may legitimately list held-out datasets, so removing them before any statistic and printing them is enough. A held-out series in our own development list is an upstream error that someone must fix. |
| The command refuses any existing output directory, even an empty one, and writes to a temporary sibling directory that is renamed at the end. | Allow overwrite with a flag. | A run directory then always corresponds to exactly one run manifest, and a failed run leaves nothing behind. |
| `omega_sd` in `joined_table.csv` is the spread of the selected Omega column. | Always the spread of `omega_levels`, as before. | Identical for the default column, and correct when another column is selected. Now covered by a value-level test. |
| The duplicate subset `jena_weather` is left out only after the reader confirms it equals `jena_weather_10T`; otherwise it is reported as unmapped. | Hard-coded exclusion, as before. | The claim is then checked on every manifest, including the frozen one. |
| A version mismatch against the team pins warns and continues. | Fail. | As briefed; the mismatch is recorded in the run manifest. The comparison is on the full version string. |
| `--fetch-results` refuses to replace an existing results file. | Overwrite, as before. | The saved file is a frozen input and the leaderboard changes over time. |

## Open

Blocked on the Omega owner (Dayson):

1. The frozen development manifest file, with: the name of the key holding the active series; the
   field names for dataset, config, series id and length; the names of the Omega columns including
   the one computed on the last 4,096 steps; how dataset and config values map to GIFT-Eval
   base/frequency names (this replaces the current subset-name map); and whether the file carries
   a version and a `frozen` flag (the run manifest records both if present). With these, one
   reader function is added in `loading.py`; nothing after `read_manifest` changes. The commands
   that will then be run are:

       PYTHONPATH=src python -m decision_aware_tsfm.analysis.k2.replication \
           --manifest <frozen manifest path> --manifest-version frozen-v2 \
           --omega-col <full-history Omega column> \
           --outdir artifacts/runs/k2-<date>/frozen_full_history

       PYTHONPATH=src python -m decision_aware_tsfm.analysis.k2.replication \
           --manifest <frozen manifest path> --manifest-version frozen-v2 \
           --omega-col <4,096-step Omega column> \
           --outdir artifacts/runs/k2-<date>/frozen_last_4096

   Until then both fail on purpose: `--manifest-version frozen-v2` stops with the list above, and
   an unknown `--omega-col` stops with the available columns (`omega_levels`, `omega_diffs` in the
   current manifest). If the 4,096-step column is not a float column, is missing for any series,
   or is named `omega_sd`, the run stops and says why (section 6). The held-out checks on the
   frozen layout depend on the new reader passing its rows through the same functions; that is
   not written yet.

For the team to decide or apply:

2. The existing findings note `k2a-k2b-findings.md` (not edited here; one new dated record per
   run) needs these corrections, all from sections 1 to 3:
   (a) "The MSE comparison against deep-learning matches the published figure exactly" and
   "both of the source paper's own metrics": the paper's family comparison is on sMAPE only, and
   the MSE row depends on one model's column.
   (b) "Mean across models (the paper's choice)": the paper does not state a choice.
   (c) Its "Implications" paragraph quotes -0.398 on MSE for the statistical comparison. That value
   depends on one model's MSE[mean] column; on MSE[0.5] it is +0.250, the same sign as MAE
   (+0.251). Its K2b statement that the sign reverses against statistical baselines compares sMAPE
   (-0.123) with MAE and is not affected by this; both of those intervals include zero.
   (e) Its explanation of the aggregation finding ("driven by a small number of poorly performing
   models in the comparison family") is now shown directly for one model per family (section 2).
   (d) The median-model p is 0.850, not 0.851; the "Files" table names files that are not the
   names in `docs/results/k2/`; the recorded `joined_table.csv` has a different column order from
   the one the code writes.
3. Corrections to the team planning document (not edited here):
   (a) It contrasts rho = -0.556 with rho = +0.251 as one comparison under two metrics. They are
   different comparisons: -0.556 is MSE[mean], zero-shot versus deep; +0.251 is MAE, zero-shot
   versus statistical. The like-for-like pairs are: versus deep, MSE -0.556 and MAE -0.248 (sMAPE -0.417,
   CRPS -0.280); versus statistical, MSE -0.398 and MAE +0.251 (sMAPE -0.123, CRPS +0.376); versus
   pretrained, MSE -0.520 and MAE -0.405 (sMAPE -0.514, CRPS -0.415). Every MSE value here is the
   MSE[mean] one and carries the caveat of section 1.
   (b) It says the effect "weakened further when aggregated within families". The record shows
   something different: for sMAPE, zero-shot versus deep, rho is -0.417 (interval [-0.793, +0.091])
   when each family is summarised by its mean, and +0.045 and -0.063 when it is summarised by the
   recorded median and best (-0.105 and +0.009 with the model-level median and best). The
   relationship is seen under the family mean and not under the median or best model, and under
   the mean it depends on one deep model (section 2). The weakening at a finer unit is a
   separate result: at configuration level the same correlation is -0.165 (n = 71).
4. Whether `Crossformer` and `Lag-Llama` stay in their families for the headline, and how the
   paper describes the dependence on them (section 2). No model was removed here.
5. Whether the MSE result is reported on MSE[mean] with the sensitivity table, on MSE[0.5], or
   not at all, and whether someone checks `Toto_Open_Base_1.0`'s published MSE[mean] values against its
   result file upstream (section 1).
6. Which within-family pooling the paper reports for "median model" and "best model"
   (section 4), and whether `granite-flowstate-r1` should stay in the mapping when it is absent from the
   saved results.
7. A mismatch in the existing dataset-level analysis, left unchanged: family scores for a base
   dataset average all of its configurations, including 18 configurations at frequencies for which
   the manifest has no Omega, while the dataset's Omega averages only the series of the subsets
   that are present, each series counting once.
8. `analysis/decision/guards.py` (not edited; outside this area): the token rule is a plain
   substring match, so it has false positives (`ram5gb`) and misses spelling variants (`M-5`).
   K2 now prints both kinds. Whether the shared rule should change is for the owner of that file.

Unverified in this run:

9. Figures were not regenerated; the existing `replication_figure.png` is unchanged and the figure
   code path was not executed.
10. The published values quoted from the source paper were read from a local text copy of the
    arXiv v1 PDF, not from a fresh download, and the paper's Figure 6 was not inspected.
11. The saved results file was not compared with the live leaderboard, and `Toto_Open_Base_1.0`'s MSE[mean]
    values were not compared with any other source.
12. The leave-one-model-out check was run at dataset level only, not at configuration level and
    not for the median and best summaries.
