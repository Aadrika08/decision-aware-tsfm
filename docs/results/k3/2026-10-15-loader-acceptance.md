# K3 input loader: acceptance against the #45 consumer handoff

- **Date:** 2026-10-15
- **Owner:** Abdullah
- **Code:** PR #35 (`analysis/k3/loader.py`), tests in `tests/unit/k3/test_k3_loader.py`
- **Asset:** release `consumer-handoff-45`, `consumer-handoff-45.zip`
- **Evidence kind:** synthetic fixture. Not real data, not the pilot handoff.

## What was run

`load_k3_inputs` over the released bundle, both Omega variants, with naming and
repository validation and the published sidecar hash supplied.

| | current | last_4096 |
|---|---|---|
| partitions listed / hash-verified / unlisted read | 408 / 408 / 0 | 408 / 408 / 0 |
| rows, comparisons | 9,792, 136 | 9,792, 136 |
| series, series-origins | 34, 68 | 34, 68 |
| Omega rows joined, left over | 68, 0 | 68, 0 |
| undefined Omega | 0 | 0 |
| status | `fixture_valid` | `fixture_valid` |

`omega.json` matches the `omega_sha256` published in `handoff.json`. Naming
validates as `dataset-config-v1`: dataset and config separate, no slashes,
`source_repository` carried in the header and consistent across all 408 job
records. Band coverage: 22 series-origins below 0.40, 10 in 0.40-0.50, 36 above
0.50; 29 distinct Omega values over 5 datasets.

## Negative controls

Each is refused: a tampered partition hash; a sidecar hash that does not match
the published value; the bundle loaded without the explicit fixture opt-in (both
a fixture cache and a fixture sidecar); a listed partition that is absent; an
unlisted partition is never read.

## Rules the loader enforces

Hash-verified partitions only, no globbing. Payload validated through
`cache.schema.validate_payload`. Omega joined on
`(dataset, config, series_id, origin)` for one explicit variant, never mixed.
Mandatory provenance on real inputs. `dataset-config-v1` naming and
`source_repository` agreement. Held-out refusal: any dataset or config whose
name contains `m5` or `pjm`, case-insensitive. Omega confined to `[0, 1]`.
A defined Omega must carry `omega_status` `ok` if it carries one at all; an
undefined Omega is null with a non-`ok` status, kept and counted so selectors
defer. One actual per target position `origin + step - 1`, so a `y_true` that
differs across models, seeds or horizons is refused.

## Limits of this acceptance

A format acceptance on a synthetic bundle. It does not exercise:

- undefined Omega: 0 of 136 rows are null, so the null-with-status path is
  covered only by unit tests;
- variant selection: `current` and `last_4096` carry identical values on all 68
  sources, so a wrong-variant join would not change any number here;
- the held-out guard, which no bundle should ever trigger;
- horizons other than 24, overlapping windows, an empty config, non-hourly data.

Also recorded: `fx_tie` has `exp_J_classical == exp_J_tsfm` to full precision at
all 11 taus with `planted_pinball_winner: "classical"` as a tie-break, so those
11 pairs disagree about half the time on finite samples and should not be read
as a defect.

## Reproducing

```
PYTHONPATH=src python -m unittest discover -s tests/unit/k3 -t .
```

102 tests, no skips, passing under unittest and pytest on Python 3.11.
