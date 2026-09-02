# Contributing

How work is organised, owned, branched, and merged.

Read `README.md` and `docs/project-overview.md` first. Coding agents: also `AGENTS.md`.

---

## 1. Ownership model

Every substantial piece of work starts as a **GitHub issue**.

```
#12 K1: validate Omega
Assignee: Dayson
```

The assignee is the **owner of the issue** — not necessarily the sole person writing every
line, but the person responsible for it landing.

The owner decomposes the issue into **sub-issues**:

```
#12 K1: validate Omega
├── #18 implement Omega statistic
├── #19 synthetic reference signals
├── #20 levels vs differences comparison
└── #21 document K1 conclusion
```

Dayson remains responsible for closing `#12`. **The parent issue closes only once its
completion condition is satisfied** — not when the last sub-issue merges.

### Who creates sub-issues

**The assignee of the parent issue.** Not the project lead. You own the decomposition of
your own work.

```
project lead
    ↓ creates parent issue, assigns owner
owner
    ↓ decomposes it
    ↓ creates and maintains sub-issues
    ↓ coordinates dependencies
    ↓ submits and merges work
    ↓ closes parent when completion condition passes
```

The project lead intervenes only when:

- scope changes
- research design changes
- a dependency crosses an ownership boundary
- someone is blocked
- a completion condition is disputed

---

## 2. Issue hierarchy

Three levels, no more:

```
milestone
    ↓
parent issue
    ↓
sub-issue
```

Worked example for the current milestone:

```
Milestone: Feasibility Sprint

#12 K1 — Validate Omega
    #18 Implement Omega
    #19 Synthetic validation
    #20 Levels vs differences

#13 K2 — Replicate published relationship
    #22 Download/normalize GIFT-Eval results
    #23 Join Omega with metrics
    #24 Run K2a
    #25 Run K2b

#14 Forecast/cache smoke test
    #26 classical baseline
    #27 TSFM smoke run
    #28 parquet cache
    #29 resume test

#15 K3 — Scalar-boundary test
    #30 matched-Omega generator
    #31 threshold analysis
    #32 selector smoke test
```

---

## 3. Branches

Branches are created **from an assigned issue**.

Format:

```
<type>/<issue-number>-<short-description>
```

Examples:

```
feat/12-omega-validation
feat/18-omega-implementation
exp/24-k2-replication
docs/31-cache-contract
fix/42-parquet-resume
```

Types:

| Type     | For                       |
|----------|---------------------------|
| `feat/`  | implementation            |
| `exp/`   | experiment or analysis    |
| `fix/`   | bug                       |
| `docs/`  | documentation             |
| `infra/` | tooling / CI / environment|

### Rule

If you are assigned `#12 K1: validate Omega`, you can either work directly on:

```
exp/12-omega-validation
```

or, once you have broken it into sub-issues:

```
feat/18-omega-implementation
exp/19-synthetic-reference-tests
exp/20-levels-vs-differences
```

**Prefer sub-issue branches** once an issue contains multiple independent changes.

### No personal branches

Not allowed:

```
dayson-work
veer-testing
new-code
final
```

Branches describe **work**, not people.

### Creating a branch

```bash
git checkout main
git pull origin main

git checkout -b feat/18-omega-implementation
```

Before opening the PR:

```bash
git fetch origin
git rebase origin/main
```

---

## 4. Pull requests

Every PR links an issue.

Title format:

```
[K1] implement Omega statistic
```

Description (auto-populated from `.github/pull_request_template.md`):

```md
## Issue

Closes #18

Parent: #12

## What changed

-

## Validation

-

## Research impact

Does this change any experimental assumption or configuration?

- [ ] No
- [ ] Yes — explained below

## Follow-ups

-
```

**PRs close sub-issues. The owner closes the parent issue** after integration, once the
completion condition is met. Do not `Closes #12` from a sub-issue PR.

`main` requires a pull request. No force pushes.

---

## 5. Recording decisions

Anything that changes an experimental assumption — datasets, splits, horizons, the tau grid,
the Omega definition, evaluation metrics, gate criteria — is discussed on the issue and then
recorded in [`docs/decisions/`](docs/decisions/).

If you cannot point at the decision record, the change did not happen.

---

## 6. Non-negotiables during the sprint

- Do not run the complete forecasting grid. Small development subsets only.
- M5 and PJM are held out. No fitting, tuning, or debugging against them.
- Save every result, configuration, seed, runtime, and dataset split.
- A failed gate is a valid result. Do not change the experiment to force a positive outcome.
