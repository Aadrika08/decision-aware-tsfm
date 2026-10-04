"""Issue #3 consumer schema: existing v1 payload, no inference imports.

This documents the released smoke format, not a frozen production workload.
"""
import numpy as np
import pyarrow as pa

TAUS = (0.01, 0.05, 0.09, 0.25, 0.40, 0.50, 0.60, 0.75, 0.91, 0.95, 0.99)
QCOLS = tuple(f"q_{tau:.2f}" for tau in TAUS)
SCHEMA = pa.schema(
    [(name, pa.string()) for name in
     ("job_id", "dataset", "config", "series_id", "model", "family")]
    + [(name, pa.int64()) for name in ("origin", "horizon", "seed", "step")]
    + [("y_true", pa.float64()), ("timestamp", pa.string())]
    + [(name, pa.float64()) for name in QCOLS]
)


def validate_payload(table):
    """Validate one or more complete jobs; provenance is checked separately.

    Accept row reordering, but reject incomplete jobs and duplicate step keys.
    Arrow v1 fields are nullable in the physical schema; actual nulls are forbidden.
    """
    if not table.schema.remove_metadata().equals(SCHEMA):
        raise ValueError("Expected the exact v1 column names, order and Arrow types")
    if not len(table) or any(table[c].null_count for c in table.column_names):
        raise ValueError("Empty payload or null values")
    values = np.column_stack([table[c].to_numpy() for c in ("y_true",) + QCOLS])
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite actuals or quantiles")
    if (np.diff(values[:, 1:], axis=1) < 0).any():
        raise ValueError("Crossing quantiles")
    jobs = {}
    for row in table.to_pylist():
        if row["family"] not in ("classical", "tsfm"):
            raise ValueError("Unknown family")
        jobs.setdefault(row["job_id"], []).append(row)
    identity = ("dataset", "config", "series_id", "model", "family", "origin", "horizon", "seed")
    for rows in jobs.values():
        first = rows[0]
        horizon = first["horizon"]
        if horizon < 1 or any(any(row[k] != first[k] for k in identity) for row in rows):
            raise ValueError("Inconsistent job identity or horizon")
        if sorted(row["step"] for row in rows) != list(range(1, horizon + 1)):
            raise ValueError("Missing or duplicate job steps")
    return table


def decision_fixture():
    """Invented two-model predictions with known answers, not research evidence."""
    rows = []
    for model, family, prediction in (("fixture_classical", "classical", 8.),
                                      ("fixture_tsfm", "tsfm", 11.)):
        for step in (1, 2):
            row = dict(job_id=f"fixture:{model}", dataset="synthetic_fixture",
                       config="hourly", series_id="fixture_series_0", model=model,
                       family=family, origin=1000, horizon=2, seed=42, step=step,
                       y_true=10., timestamp=f"2020-01-01T0{step-1}:00:00")
            row.update({col: prediction for col in QCOLS})
            rows.append(row)
    table = pa.Table.from_pylist(rows, schema=SCHEMA)
    table = table.replace_schema_metadata({b"fixture": b"true; not generated model forecasts"})
    return validate_payload(table)
