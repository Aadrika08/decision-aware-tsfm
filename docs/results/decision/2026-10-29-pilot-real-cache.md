# Decision engine on the full pilot cache (#37, #29)

Status: first run of the engine on the real cache. Pilot only: 5 dataset groups, 55 series, one TSFM.
Not a general result and not a claim about selector effectiveness. Part of #37. Written by Veer's side;
Abdullah's K3 reader acceptance on #29 is separate.

## Inputs

- Release `pilot-full-2026-10-29`. Forecast archive sha256 `4ec7b9ad45c921581562cc7bbb118c0988b318179fd4e5cfac97ae904cfc03a5`,
  `handoff.json` sha256 `82d2234855ba4b67df0aa0b2994946c1a46c70498375600378a6e6eb5b7aa0a9`, evidence archive
  `f983217fdce4538d9bc059576ed413d7521dd31478833ea4e84eb2fc0ac4bf99`. 21,696 jobs, 1,395,776 rows, 55 series, 904 series-origins.
- Code: main `289b38c1190f5a1974a399e38e0bd1b7cfe84242`, `run_pilot`. Acceptance comment: https://github.com/Aadrika08/decision-aware-tsfm/issues/29#issuecomment-6501484345
- Representatives (best of family, validation only, scaled pinball at tau 0.50): classical `autoarima_native`
  (of 3 candidates), TSFM `moirai_pilot` (only member). 1,815 evaluation units (165 series-horizons x 11 taus), 0 excluded, 0 undefined Omega.
- Four runs on Python 3.11 and the same four on 3.10 (placeholder or K3 selector in the D_Omega slot, Omega `current` or `last_4096`).
  All exit 0. For each pair the 3.11 and 3.10 outputs are identical file by file except `environment.json`.
  Manifest sha256 (first 16 hex) on 3.11: placeholder `09cbb5d1b17d1cee`, K3 `4b3f32be3f064305`, placeholder last_4096 `46f02dacac75118e`, K3 last_4096 `469b6328ae803d7d`.

## How the tables were made

`make_tables.py` in this folder reads the run outputs and prints every table below; no number is typed by hand.
`python -I make_tables.py RUNS_DIR full-bundle/omega.json`. The run outputs (about 13 MB each) are not committed.

## Checks done before writing this

- For all 1,815 units the realized pinball cost of AutoARIMA and Moirai was recomputed from the raw forecast partitions and the role
  table with separate code: max relative difference to the engine 3.3e-13. The hindsight oracle matches exactly. The seasonal-naive
  scale of all 55 series was recomputed from the source files: max relative difference 3.7e-15. The mean scaled regret of
  always_classical, always_tsfm, static_omega, pinball_validation and crps_validation equals the engine's to 1e-14.
- No source code changed, so no suites were rerun for this note.

## Results

### Table 1. Mean scaled regret, all horizons and taus, static threshold 0.45 (1,815 units)

| selector | Omega `current` | Omega `last_4096` |
|---|---|---|
| always_classical | 0.0430 | 0.0430 |
| always_tsfm | 0.0178 | 0.0178 |
| static_omega | 0.0244 | 0.0243 |
| pinball_validation | 0.0070 | 0.0070 |
| crps_validation | 0.0117 | 0.0117 |
| hindsight_oracle | 0.0000 | 0.0000 |
| D_Omega, placeholder | 0.0253 | 0.0249 |
| D_Omega, K3, decided units only | 0.0231 | 0.0155 |
| D_Omega, K3, deferrals priced (all 1,815) | 0.0209 | 0.0138 |

K3 decided units: 1584 of 1,815 (`current`), 1485 of 1,815 (`last_4096`); the rest deferred.

### Table 1b. Static Omega rule by threshold (mean scaled regret)

| static threshold | current | last_4096 |
|---|---|---|
| 0.45 | 0.0244 | 0.0243 |
| 0.4 | 0.0247 | 0.0246 |
| 0.5 | 0.0250 | 0.0255 |

### Table 2. Mean scaled regret by horizon (Omega `current`, threshold 0.45)

| selector | h=1 | h=24 | h=168 |
|---|---|---|---|
| always_classical | 0.0275 | 0.0346 | 0.0670 |
| always_tsfm | 0.0178 | 0.0222 | 0.0133 |
| static_omega | 0.0222 | 0.0287 | 0.0223 |
| pinball_validation | 0.0049 | 0.0068 | 0.0093 |
| crps_validation | 0.0110 | 0.0124 | 0.0117 |
| hindsight_oracle | 0.0000 | 0.0000 | 0.0000 |
| D_Omega, placeholder | 0.0232 | 0.0291 | 0.0235 |
| D_Omega, K3, deferrals priced | 0.0251 | 0.0225 | 0.0150 |

### Table 3. Which family has the lower cost, by horizon and tau (Delta = TSFM cost minus classical cost, scaled; Delta < 0 favours TSFM)

| h | tau | units | TSFM lower (share) | mean Delta | 95% interval | lower-cost family (by share) | excludes 0 |
|---|---|---|---|---|---|---|---|
| 1 | 0.01 | 55 | 0.473 | -0.0033 | [-0.0103, 0.0046] | classical | no |
| 1 | 0.05 | 55 | 0.527 | -0.0185 | [-0.0427, 0.0067] | TSFM | no |
| 1 | 0.09 | 55 | 0.564 | -0.0295 | [-0.0657, 0.0071] | TSFM | no |
| 1 | 0.25 | 55 | 0.582 | -0.039 | [-0.0925, 0.0181] | TSFM | no |
| 1 | 0.4 | 55 | 0.491 | -0.0143 | [-0.0519, 0.0347] | classical | no |
| 1 | 0.5 | 55 | 0.309 | 0.0175 | [-0.0008, 0.0464] | classical | no |
| 1 | 0.6 | 55 | 0.455 | -0.0063 | [-0.0428, 0.0512] | classical | no |
| 1 | 0.75 | 55 | 0.491 | -0.028 | [-0.0818, 0.0449] | classical | no |
| 1 | 0.91 | 55 | 0.6 | -0.0156 | [-0.0593, 0.0307] | TSFM | no |
| 1 | 0.95 | 55 | 0.527 | 0.0013 | [-0.0259, 0.0262] | TSFM | no |
| 1 | 0.99 | 55 | 0.091 | 0.0293 | [0.0054, 0.0554] | classical | yes |
| 24 | 0.01 | 55 | 0.782 | -0.0212 | [-0.0375, 0.0002] | TSFM | no |
| 24 | 0.05 | 55 | 0.782 | -0.0456 | [-0.0966, 0.0142] | TSFM | no |
| 24 | 0.09 | 55 | 0.764 | -0.0575 | [-0.1238, 0.0186] | TSFM | no |
| 24 | 0.25 | 55 | 0.709 | -0.0532 | [-0.1326, 0.0429] | TSFM | no |
| 24 | 0.4 | 55 | 0.655 | -0.0041 | [-0.0491, 0.0627] | TSFM | no |
| 24 | 0.5 | 55 | 0.309 | 0.0367 | [0.0013, 0.0790] | classical | yes |
| 24 | 0.6 | 55 | 0.473 | 0.0029 | [-0.0490, 0.0496] | classical | no |
| 24 | 0.75 | 55 | 0.455 | -0.0363 | [-0.1142, 0.0194] | classical | no |
| 24 | 0.91 | 55 | 0.527 | -0.0123 | [-0.0417, 0.0100] | TSFM | no |
| 24 | 0.95 | 55 | 0.4 | 0.0123 | [-0.0118, 0.0258] | classical | no |
| 24 | 0.99 | 55 | 0.2 | 0.0417 | [0.0018, 0.0916] | classical | yes |
| 168 | 0.01 | 55 | 0.782 | -0.0354 | [-0.0605, -0.0058] | TSFM | yes |
| 168 | 0.05 | 55 | 0.891 | -0.0822 | [-0.1428, -0.0104] | TSFM | yes |
| 168 | 0.09 | 55 | 0.855 | -0.1073 | [-0.1826, -0.0109] | TSFM | yes |
| 168 | 0.25 | 55 | 0.745 | -0.1183 | [-0.1918, 0.0055] | TSFM | no |
| 168 | 0.4 | 55 | 0.745 | -0.0595 | [-0.1030, 0.0287] | TSFM | no |
| 168 | 0.5 | 55 | 0.455 | -0.0079 | [-0.0494, 0.0570] | classical | no |
| 168 | 0.6 | 55 | 0.709 | -0.0497 | [-0.0868, 0.0233] | TSFM | no |
| 168 | 0.75 | 55 | 0.782 | -0.0979 | [-0.1707, -0.0040] | TSFM | yes |
| 168 | 0.91 | 55 | 0.782 | -0.0527 | [-0.0966, -0.0127] | TSFM | yes |
| 168 | 0.95 | 55 | 0.764 | -0.0152 | [-0.0490, 0.0118] | TSFM | no |
| 168 | 0.99 | 55 | 0.255 | 0.0357 | [-0.0020, 0.0995] | classical | no |

Cells where the interval excludes 0: 8 of 33. Intervals are nominal 95% percentile block-bootstrap intervals, approximate and known to under-cover (STATUS, verified findings); do not quote one as exact.

### Table 4. By dataset (all horizons and taus pooled, Omega `current`)

| dataset | units | TSFM lower (share) | mean Delta | regret always_classical | regret always_tsfm | regret pinball_validation |
|---|---|---|---|---|---|---|
| electricity | 330 | 0.3485 | 0.0315 | 0.0107 | 0.0422 | 0.0058 |
| ett1 | 231 | 0.5671 | -0.0132 | 0.0282 | 0.0150 | 0.0086 |
| ett2 | 231 | 0.5022 | -0.0060 | 0.0160 | 0.0100 | 0.0046 |
| jena_weather | 693 | 0.6017 | -0.0371 | 0.0508 | 0.0138 | 0.0101 |
| solar | 330 | 0.7939 | -0.0792 | 0.0883 | 0.0091 | 0.0024 |

### Table 5. K3 selector deferrals by dataset (threshold 0.45; every deferral is reason omega_out_of_range)

| dataset | units | series | deferred units (`current`) | deferred series (`current`) | deferred units (`last_4096`) | deferred series (`last_4096`) |
|---|---|---|---|---|---|---|
| electricity | 330 | 10 | 66 | 2 | 165 | 5 |
| ett1 | 231 | 7 | 0 | 0 | 0 | 0 |
| ett2 | 231 | 7 | 0 | 0 | 0 | 0 |
| jena_weather | 693 | 21 | 165 | 5 | 165 | 5 |
| solar | 330 | 10 | 0 | 0 | 0 | 0 |

### Table 6. Omega values by variant (series-origin records)

| variant | origin records | in 0.40 to 0.50 | below 0.40 | above 0.50 |
|---|---|---|---|---|
| current | 904 | 100 | 113 | 691 |
| last_4096 | 904 | 138 | 139 | 627 |

## What the tables say (pilot only)

- Picking the family by validation pinball loss has the lowest regret in every horizon (Table 2), 0.0070 pooled. It recovers 43% of the oracle
  value; crps_validation recovers 6%. Always using the TSFM (0.0178) beats always using the classical model (0.0430).
- The static Omega rule (0.0244 at 0.45, 0.0247 at 0.40, 0.0250 at 0.50) sits between always_tsfm and always_classical; it does not beat always_tsfm
  at any of the three thresholds. The 0.45 is a placeholder, not the paper's three-zone definition.
- The K3 selector, with its deferrals priced at the pinball-validation cost, is 0.0209 on `current` and 0.0138 on `last_4096`, against 0.0253 and
  0.0249 for the placeholder D_Omega. It does not beat always_tsfm on `current` (0.0209 against 0.0178), it does on `last_4096` (0.0138 against 0.0178), and it does not beat pinball_validation (0.0070) on either. It defers
  231 units on `current` (jena_weather 165 of 693, electricity 66 of 330) and 330 on `last_4096` (jena_weather 165, electricity 165), all because
  Omega falls outside the range seen in the fitted datasets (Table 5). The decided-only numbers (0.0231, 0.0155) are over fewer units and are not comparable with the other rows.
- The Omega definition matters: 100 of 904 series-origin records sit in 0.40 to 0.50 on `current` and 138 on `last_4096` (Table 6), and the K3 result moves from 0.0209 to 0.0138. This supports keeping the sensitivity analysis; it does not say which definition selects better.
- Which family has the lower cost (Table 3): the TSFM at most taus at horizon 168 (9 of 11 cells have a share above 0.7), at taus 0.01 to 0.40 and 0.91 at horizon 24, and
  the classical model at the top quantile (tau 0.99) in all three horizons and at tau 0.50 for horizons 1 and 24. 8 of 33 intervals exclude zero.
  By dataset (Table 4) the classical model has the lower cost on electricity and the TSFM on jena_weather, ett1 and solar (most clearly solar, share 0.79); ett2 is about even (0.50).
- A weak or absent scalar Omega boundary is a valid outcome. This run does not show one, and it does not rule one out at 55 series.

## Limits that apply to every statement above

- One TSFM against three classical candidates: best-of-family representatives favour the classical side; Moirai's zero-shot status is unresolved.
- Origins of one series overlap, so units are not independent samples. Five dataset groups; the K3 low band rests on two or three datasets.
- Intervals are nominal percentile block-bootstrap intervals, known to under-cover (roughly 64% to 93% for a nominal 95%). Do not quote one as exact.
- The static threshold 0.45 is a placeholder; the paper gives three Omega zones.
- Scale is the in-sample seasonal-naive MAE (season 24) of each series' history before its first origin (#29 item 7a).
- M5 and PJM were not used.
