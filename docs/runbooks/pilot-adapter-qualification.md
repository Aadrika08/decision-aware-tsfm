# Four-model synthetic qualification — #20

[ADR 0005](../decisions/0005-pilot-adapter-qualification.md) records the approved
methods. This command exercises actual model implementations on invented inputs;
it neither reads development datasets nor establishes a production budget.

## Environment

Use Python 3.11 and a separate virtual environment. Do not update the historical
smoke environment in place. The new lock retains every smoke pin and adds the
classical dependencies, including StatsForecast 2.1.1.

```bash
python3.11 -m venv .venv
uv pip sync --python .venv/bin/python --require-hashes requirements/qualification.txt
uv pip check --python .venv/bin/python
```

The full qualification uses a CUDA-capable local runtime for Moirai. It loads the
existing pinned checkpoint from Hugging Face; once cached, set `HF_HUB_OFFLINE=1`
to prevent network access. Classical models always run on CPU. An explicit
`--device cpu` changes Moirai's execution device and records that choice; it does
not count as RTX 4050 evidence.

## Run

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -v
PYTHONPATH=src .venv/bin/python scripts/qualify_pilot_adapters.py artifacts/runs/pilot-qualification --device cuda
```

Choose a fresh output directory: existing evidence cannot be overwritten. The
runner writes synthetic source arrays, every generated quantile array, hashes,
fitted classical specifications, warnings, timings, package/code provenance and
`report.json`. It checkpoints the report after each prediction and records hard
interruptions as failures. Forecast outputs and their summaries stay separate from
the production cache and its timing estimator.

Full coverage is four models × three horizons × two contexts × three attempts:
**72 prediction attempts**, using seed 42 twice and seed 43 once per case. The
contexts are a positive seasonal/trend/noise signal and a constant negative
signal, each 2,048 observations long. All predictions must have eleven finite,
ordered quantiles. The runner checks input preservation and same-seed repeats.
CPU repeats require exact equality; GPU repeats record the maximum difference
against the fixed ADR tolerance, together with sampling warnings.

The command exits 0 only when the complete selected configuration passes all
required coverage. It exits 2 for model failures or partial coverage. For a
bounded diagnostic run:

```bash
PYTHONPATH=src .venv/bin/python scripts/qualify_pilot_adapters.py artifacts/runs/classical-check --models autoarima_native autoets_simulated seasonal_naive_native --horizons 1
```

That subset remains **partial** even if every attempted forecast succeeds. Do not
replace failed evidence or change the synthetic sources to manufacture a pass.
Record numerical limitations and failed fits explicitly.

## Use with the configured planner

`decision_aware_tsfm.cache.pilot.pilot_models(device="cuda")` returns four fresh model
configuration dictionaries. Wrap these in a model document with `schema_version: 1`
and the recorded model `approval_reference`. Feed Dayson's separately approved
workload into the existing `forecast_cache.py plan` command. Real data generation
still requires its workload approval reference and `--real`.

New adapter names are `autoarima_native`, `autoets_simulated`,
`seasonal_naive_native`, and `moirai_pilot`. Planning and read-only audits do not
require StatsForecast to be installed. Loading a classical adapter requires the
exact approved version. The new native SeasonalNaive method is distinct from
`seasonal_naive_smoke`; historical cache evidence retains its original method.

The remaining parent #16 work is the approved stable-ID workload, actual
development pilot, Veer/Abdullah consumer acceptance, and a measured budget using
the final workload and accepted margins. Synthetic qualification does not fulfill
those research dependencies.
