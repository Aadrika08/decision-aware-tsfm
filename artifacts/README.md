# Artifacts

`reference/` holds the versioned manifest and published-result files used to
reproduce the current feasibility-sprint evidence. Do not edit these files in a
feature branch without an issue that records the provenance change.

`runs/` is intentionally ignored by Git. Each new run must write a small result
record under `docs/results/<area>/` that states the command/configuration,
input-artifact hash, code commit, seed, and conclusion.
