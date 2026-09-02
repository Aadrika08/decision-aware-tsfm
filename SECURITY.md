# Security policy

## What this repository is

Research code for an academic paper. It is not a deployed service, it is not published as a
package, and it processes only public benchmark time-series data. Nothing here handles user
data, authentication, or network requests to anything other than public dataset and model
hosts.

Scope your expectations accordingly: the realistic risks are a vulnerable dependency and a
credential committed by accident, not a remote exploit.

## Reporting a vulnerability

Open a **private** report rather than a public issue:

1. Email the repository owner directly, or
2. Contact a maintainer through the team channel.

Do not open a public issue for anything that would expose a credential or a private dataset
path until it has been rotated or removed.

Expect an acknowledgement within a few days. This is a student research team on a ten-week
deadline, not a staffed security function.

## Supported versions

Only `main`. There are no releases and no version support commitments.

## Dependency alerts

Dependabot alerts and Dependabot security updates are enabled. There are no dependency
manifests in the repository yet, so nothing is being scanned — alerts will begin to matter
once a `requirements.txt` or `pyproject.toml` lands under the first implementation issue.

Code scanning is not available: GitHub Advanced Security is org-only, and this is a private
repository on a personal account.

## What actually matters here

The failure modes most likely to bite this project are research-integrity and hygiene
issues rather than exploits:

- **Never commit credentials.** No API keys, no Kaggle or HuggingFace tokens, no cloud
  credentials. `.gitignore` excludes `.env`, but that is a safety net, not a control.
- **Never commit data or forecast caches.** `.gitignore` excludes `data/`, `cache/`,
  `outputs/`, and `*.parquet`. Caches are regenerated from the cache contract, not from git.
- **Held-out data stays held out.** M5 and PJM must not appear in this repository in any
  form — not as a sample, not as a debugging fixture, not as a plot. See `AGENTS.md`.

If a credential is committed, rotate it first and rewrite history second. Rotation is the
control; history rewriting is cleanup.
