# Decision-Aware Model-Family Selection

Research project studying whether model-family selection for time-series forecasting
should depend on the downstream decision, rather than forecasting error alone.

Given a new time series and a known asymmetric downstream cost, can cheap pre-forecast
properties predict whether a **classical** family or a **time-series foundation model
(TSFM)** family will minimise downstream decision cost — without running expensive
validation across both families? And when the cheap descriptor is insufficient, can the
selector detect that and defer?

Algoverse AI Research · Target: AI4TS @ AAAI-27

## What we are trying to prove

The published selection rule we build on uses spectral predictability (Ω) and was
calibrated against **forecast error** (sMAPE, MSE). Our claim is that the family boundary
moves once the downstream loss is **asymmetric**, and that a minimal Ω- and τ-aware
selector recovers most of the value of full pinball-τ validation at a fraction of the cost.

Where Ω is *not* sufficient, we want to characterise that failure rather than paper over it.

## Current stage

**Five-day feasibility sprint.** Nothing beyond the gates is in scope yet.

We are currently testing:

- **K1** — validate Ω
- **K2a** — reproduce the published relationship on sMAPE / MSE
- **K2b** — bridge to pinball loss at τ = 0.5
- **K3** — determine whether a scalar Ω boundary is meaningful
- small forecasting / cache smoke test
- matched-Ω feasibility

> **Do not run the complete forecasting grid yet.**
> The sprint is restricted to a small development subset. M5 and PJM are held out entirely
> and must not be used for fitting, tuning, or debugging decisions.

A failed gate is a valid result. Do not change an experiment to force a positive outcome.

## Research plan

- [`docs/project-overview.md`](docs/project-overview.md) — start here; the 10-minute version
- [`docs/feasibility-sprint.md`](docs/feasibility-sprint.md) — what each person is doing this week
- [`docs/implementation-plan.pdf`](docs/implementation-plan.pdf) — six-stage execution plan
- [`docs/proposal.pdf`](docs/proposal.pdf) — full research proposal
- [`docs/decisions/`](docs/decisions/) — decision log for anything that changes the experiment

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md).

Work is organised through GitHub issues. Do not start unassigned work without discussing
it first.

Coding agents: read [`AGENTS.md`](AGENTS.md) before touching this repository.
