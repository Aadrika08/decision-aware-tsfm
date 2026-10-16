"""Definitions shared by more than one K3 module.

This module exists so the boundary analysis and the selector carry one copy of
the rules they must agree on, rather than the selector importing them from
boundary.py. Two reasons:

  * `MIN_DISTINCT_OMEGA_THRESHOLD` and `ORIENTATIONS` are analysis parameters
    recorded in docs/decisions/0008-k3-boundary-statistics.md. If the two
    modules could drift apart on them, the selector could place a threshold
    the boundary analysis would refuse to fit, or search one orientation while
    the other searched both.
  * `validate_columns` and `distinct_count` decide what counts as malformed
    input and what counts as enough evidence. Those answers have to be the
    same everywhere or the same cache reads as valid in one module and
    invalid in another.

Nothing here knows about costs, forecasts or thresholds. Anything that does
belongs in boundary.py or selector.py.
"""
import numpy as np

# Minimum independent Omega points before any threshold may be fitted.
# Counted in distinct DATASETS when dataset labels are supplied (#32), because
# many series drawn from one dataset share a source and are not independent
# evidence for a boundary. Agreed with Veer on #57 and matched by the decision
# engine's leave-one-dataset-out floor (#61).
MIN_DISTINCT_OMEGA_THRESHOLD = 3

# A threshold can point either way, and which way is an empirical question.
# Both are always searched so the answer is never chosen after seeing results.
#   classical_below -- use the classical family at or below the threshold
#   tsfm_below      -- use the TSFM family at or below the threshold
ORIENTATIONS = ("classical_below", "tsfm_below")


def _as_1d_finite(name, values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    return arr


def validate_columns(omega, *others):
    """Coerce parallel numeric columns, or raise ValueError.

    Every column must be one-dimensional, finite and the same length as the
    first. Malformed input is a different condition from "valid but
    insufficient" and must raise rather than return an insufficient_data
    result, so a bad cache can never be mistaken for a thin one.
    """
    omega = _as_1d_finite("omega", omega)
    out = [omega]
    for i, other in enumerate(others):
        arr = _as_1d_finite(f"input_{i}", other)
        if arr.shape != omega.shape:
            raise ValueError(f"input_{i} length {len(arr)} != omega length {len(omega)}")
        out.append(arr)
    return out


def distinct_count(omega, datasets=None):
    """How many independent Omega points are available.

    With dataset labels, count distinct datasets: many series from one dataset
    share a source and are not independent evidence for a boundary (#32, Veer's
    sign-off condition). Without labels, fall back to distinct Omega values and
    say so in the reported unit.
    """
    if datasets is None:
        return len(np.unique(np.asarray(omega))), "omega_value"
    datasets = np.asarray(datasets)
    if datasets.shape != np.asarray(omega).shape:
        raise ValueError("datasets length must match omega")
    return len(np.unique(datasets)), "dataset"
