# AGENTS.md

Instructions for coding agents working in this repository.

## Project

Read before modifying the repository:

1. `README.md`
2. `docs/project-overview.md`
3. `docs/feasibility-sprint.md`
4. the GitHub issue you are working on
5. `CONTRIBUTING.md`

This is a research repository. The output is evidence, not features.

## Source of truth

**The assigned GitHub issue is the execution contract.**

The issue's task list and completion condition define the current task. The PDFs in `docs/`
explain *why*; they do not authorise work beyond the issue.

Do not expand the task because another implementation appears more interesting or
statistically preferable.

Do not silently change:

- datasets
- held-out splits
- horizons
- tau grid
- Omega definition
- evaluation metrics
- experimental gates

If one of these needs changing, **stop** and document the proposed change in the issue.
Once agreed, record it in `docs/decisions/`.

## Research integrity

A negative result is valid. "No stable scalar boundary exists" is a publishable finding and
is an explicitly accepted outcome of K3.

**Do not modify an experiment to make a gate pass.** Not the data subset, not the
preprocessing, not the metric, not the threshold.

**M5 and PJM are held out** and must not be used for fitting, hyperparameter selection,
debugging decisions, or selector tuning. Not "just to check". Not "only to look at one
series". They are the transfer test and touching them destroys it.

Record every result, configuration, seed, runtime, and dataset split. A number that cannot
be regenerated is not a result.

Report uncertainty rather than inventing assumptions. If an input is missing — for example
compatible quantile outputs for K2b — report the missing input precisely; do not relabel it
as a failure of something else.

## Work boundaries

Only work on the assigned issue or its sub-issues.

If additional work is discovered:

1. create or propose a sub-issue
2. link it to the parent
3. explain why it is required
4. continue only if it falls under the parent issue

Do not fix unrelated things you notice in passing. File them.

## Commits

Keep commits scoped and descriptive.

Bad:

- `fix`
- `updates`
- `changes`

Good:

- `omega: add synthetic reference tests`
- `k2: join gift-eval metrics with omega`

## Branches and pull requests

See `CONTRIBUTING.md`. Branches are named `<type>/<issue-number>-<short-description>`.
Every PR links an issue. **PRs close sub-issues; the owner closes the parent issue.**
