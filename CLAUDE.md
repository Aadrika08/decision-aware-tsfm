# CLAUDE.md

Read `AGENTS.md` before doing anything. It is the governing document; this file only adds
Claude-specific working rules.

This repository contains research code.

When working here:

- inspect the assigned GitHub issue first
- understand the parent issue and any linked sub-issues
- do not implement unrelated improvements
- preserve experimental configuration exactly
- do not use held-out datasets (M5, PJM) for development
- report uncertainty rather than inventing assumptions
- prefer small, reviewable changes

Before writing code, state:

1. what issue is being addressed
2. what files are expected to change
3. what completion condition from the issue will be satisfied

Before finishing:

1. run relevant tests
2. summarize what changed
3. mention unresolved limitations
4. identify any follow-up sub-issues
