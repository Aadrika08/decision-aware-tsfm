# Decision log

Anything that changes an experimental assumption gets a file here.

This exists because a research result is only worth what its provenance is worth. If a
number in the paper depends on a choice, that choice must be findable, dated, and
attributable — not reconstructed from memory in week nine.

## What must be recorded

- datasets and dataset subsets
- development / held-out splits
- forecast horizons
- the tau grid
- the Omega definition, including **levels vs first differences**
- evaluation metrics
- experimental gate criteria and pass thresholds
- the within-family representative rule
- the model roster
- cost models and penalty ratios for M5 and PJM

If you cannot point at the record, the change did not happen.

## What does not go here

Implementation details, refactors, bug fixes, tooling choices. Those live in issues and
pull requests. This directory is for decisions that would change a number in the paper.

## How

One file per decision, named `NNNN-short-slug.md`, numbered sequentially:

```
0001-omega-levels-vs-first-differences.md
0002-development-dataset-manifest.md
```

Use this structure:

```md
# NNNN — Title

- **Date:** YYYY-MM-DD
- **Decided by:**
- **Issue:** #
- **Status:** proposed | accepted | superseded by NNNN

## Context

What forced the decision. What was uncertain.

## Decision

What was chosen. State it precisely enough to implement from.

## Alternatives considered

What else was on the table, and why it lost.

## Consequences

What this constrains downstream. What has to be reported in the paper because of it.
```

Supersede rather than edit. A decision that turned out wrong is itself a finding — mark the
old file `superseded by NNNN` and write a new one.

## Expected early entries

The sprint will produce at least these:

- Omega on levels vs first differences (K1, day 1) — the random-walk trap makes this
  load-bearing
- the development dataset manifest and the sub-1000-timestep exclusion list
- the forecast-cache schema and key definition
- the within-family representative rule, frozen before any held-out evaluation
