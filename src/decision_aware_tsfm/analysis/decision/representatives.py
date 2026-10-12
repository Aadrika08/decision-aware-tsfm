"""One representative model per family, chosen on validation origins only, then frozen (#41).

    PYTHONPATH=src python -m decision_aware_tsfm.analysis.decision.run_representatives CACHE --out FILE

The engine compares exactly one model per family and takes that choice as a
{family: model} dict (cache_io.restrict_to_representatives). This module makes
the dict and writes it to a frozen file. It changes nothing in the engine.

Two rules. Which one is the primary is a team decision, to be recorded before
any real result is seen.
  best_of_family  per family, the model with the lowest mean pinball loss at
                  tau = 0.50 ("mean median pinball loss") over validation origins
  fixed_named     the caller names the models; they are validated against the
                  cache by restrict_to_representatives itself

What best_of_family computes
  1. Roles. costs.assign_roles on every origin in the cache: per (series,
     horizon) the earliest val_fraction of origins is validation, less the
     origins the gap rule removes; the rest is evaluation (`test`). Every row
     of a role other than validation is dropped BEFORE any loss is computed.
  2. Unit loss. Per model and per unit (dataset, config, series_id, horizon):
     J = mean pinball at tau 0.50 over the unit's validation targets, seeds
     averaged as in costs.per_origin_costs, and scale = mean |y| over the same
     targets. These are the engine's J and scale (costs.unit_costs), on
     validation origins instead of test origins.
  3. loss="scaled" (default): a unit counts J / scale, the engine's scaled
     cost, so a series measured in large numbers cannot choose the
     representative on its own. loss="raw": a unit counts J. Both are written
     to the frozen file whichever is used.
  4. Coverage. A model covers a unit when it has every target (origin, step)
     that any model in the cache has for that unit. Its origins are then the
     ones the roles were assigned from, so the window scored here is the
     validation window the engine derives when that model is the
     representative. A family's models are compared on the units ALL of them
     cover (and whose validation scale is not 0); a model cannot win by
     missing a hard series or horizon.
  5. Score. The plain mean of the unit losses over those common units, every
     unit weighing the same.
  6. Choice. The lowest score. Scores within TIE_RTOL (relative, the engine's
     tolerance) of the lowest are a tie, and a tie goes to the model whose
     name sorts first. A family with one model returns it (only_member).

Guards
  * Dataset or config names matching the held-out rule (M5, PJM) are refused.
  * validation_unit_losses refuses any frame that holds a row whose role is
    not validation, so no loss can be computed on an evaluation origin here.

Frozen file. Same layout and checks as E2's frozen specification:
{"payload": ..., "sha256": <hash of the payload>}. The hash guards against
accidental edits; it is not a security control (see experiments/e2.py). The
payload records a fingerprint of the input rows and of the selecting code.
The loader does not re-run the selection, so it does not refuse on a code
change; the command reproduces the file and reports any difference.

representative_robustness is different in kind: it runs the whole engine once
per candidate, so it reads evaluation origins. It is an analysis of results
and must not be used to choose a representative.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ..experiments.e2 import FrozenFileError, _canonical, _read_verified, _sha256
from . import costs
from .cache_io import KEY_COLS, CacheError, load_cache, qcol, restrict_to_representatives
from .costs import SERIES_KEYS, assign_roles, per_origin_costs
from .evaluate import summarize
from .guards import assert_no_held_out
from .metrics import TAU_GRID, check_taus
from .pipeline import build_tables, run
from .schema import FAMILIES
from .selectors import TIE_RTOL

FROZEN_SCHEMA = "representatives-frozen-v1"
RULES = ("best_of_family", "fixed_named")
LOSSES = ("scaled", "raw")
LOSS_COLUMN = {"scaled": "mean_scaled_loss", "raw": "mean_raw_loss"}
SELECTION_TAU = 0.50
SCORED_ROLE = "validation"
REASONS = ("lowest_validation_loss", "tie_first_by_name", "only_member", "named_by_caller")
SCORE_COLUMNS = ("family", "model", "chosen", "mean_scaled_loss", "mean_raw_loss", "loss_minus_best",
                 "n_units_covered", "n_units_scored")
PAYLOAD_KEYS = ("schema", "rule", "representatives", "reasons", "selection", "best_by_loss",
                "val_fraction", "roles_scored", "origins_statement", "candidates", "scores",
                "coverage", "n_origins_by_role", "input", "code_sha256")
ORIGINS_STATEMENT = (
    "Losses were computed on validation origins only. Every row of an evaluation (test) or gap "
    "origin was dropped before any loss was computed; those rows entered only the role assignment, "
    "the coverage check and the input fingerprint.")
AGGREGATION = (
    "per model and per unit (dataset, config, series_id, horizon): mean pinball loss at tau 0.50 over "
    "the unit's validation targets, seeds averaged; scaled = that mean divided by mean |y| over the "
    "same targets, raw = that mean; score = plain mean over the units every candidate of the family "
    "covers, each unit weighing the same")
TIE_RULE = ("lowest score wins; scores within tie_rtol (relative) of the lowest are a tie, and a tie "
            "goes to the model whose name sorts first")


class EvaluationRowsError(RuntimeError):
    """Rows of a role other than validation reached the representative scoring."""


def input_fingerprint(fc: pd.DataFrame, taus=TAU_GRID) -> str:
    """SHA-256 of the canonical forecast rows: cache key, y and every quantile
    column, sorted by cache key, so the same rows in any order give the same
    value. Strings as UTF-8, integers and floats as little-endian 64-bit."""
    d = fc.sort_values(KEY_COLS)
    h = hashlib.sha256()
    for c in KEY_COLS + ["y"] + [qcol(x) for x in check_taus(taus)]:
        h.update(c.encode("utf-8") + b"\x1e")
        if c in ("dataset", "config", "series_id", "model"):
            h.update("\x1f".join(d[c].astype(str)).encode("utf-8"))
        elif c in ("seed", "horizon", "origin", "step"):
            h.update(d[c].to_numpy(dtype="<i8").tobytes())
        else:
            h.update(d[c].to_numpy(dtype="<f8").tobytes())
    return h.hexdigest()


def code_sha256() -> str:
    """SHA-256 over the source text of this module and of costs.py (the role
    rule and the per-origin loss). Code those two call in other modules is not
    covered: the same idea and the same limit as e2.selector_code_sha256."""
    parts = [f"{m.__name__}\x1f{inspect.getsource(m)}" for m in (sys.modules[__name__], costs)]
    return _sha256("\x1e".join(parts))


def _labelled(fc: pd.DataFrame, family_of_model: dict[str, str]) -> pd.DataFrame:
    """Add the family label of each row's model and drop targets with no realised
    value, as restrict_to_representatives does. Fails on a model with no label
    and on a producer family column that disagrees with the mapping."""
    unlabelled = sorted(m for m in fc["model"].unique() if family_of_model.get(m) not in FAMILIES)
    if unlabelled:
        raise CacheError(f"models with no family label in {FAMILIES}: {unlabelled}")
    expected = fc["model"].map(family_of_model)
    if "family" in fc.columns and (fc["family"] != expected).any():
        bad = fc.loc[fc["family"] != expected, ["model", "family"]].drop_duplicates()
        raise CacheError(f"cache family labels disagree with the mapping: {bad.to_dict('records')}")
    out = fc.assign(family=expected)
    return out[out["y"].notna()]


def _covering(fc: pd.DataFrame) -> pd.DataFrame:
    """(unit, model) pairs where the model has every target that any model has
    in that unit. Seeds are not counted: a target is (origin, step)."""
    tg = fc[SERIES_KEYS + ["origin", "step", "model"]].drop_duplicates()
    ref = (tg.drop(columns="model").drop_duplicates().groupby(SERIES_KEYS).size()
           .rename("n_ref").reset_index())
    own = tg.groupby(SERIES_KEYS + ["model"]).size().rename("n_own").reset_index()
    own = own.merge(ref, on=SERIES_KEYS)
    return own.loc[own["n_own"] == own["n_ref"], SERIES_KEYS + ["model"]]


def validation_unit_losses(rows: pd.DataFrame, taus=TAU_GRID) -> pd.DataFrame:
    """One row per (unit, family, model): J = mean pinball at SELECTION_TAU over
    the unit's targets, scale = mean |y| over the same targets, n_origins.

    rows: canonical forecast rows with `family` and `role` columns. EVERY row
    must have role validation; anything else raises EvaluationRowsError, so
    this function cannot compute a loss on an evaluation origin."""
    if "role" not in rows.columns:
        raise EvaluationRowsError("rows carry no role column; assign roles with costs.assign_roles "
                                  "and keep the validation rows")
    other = sorted(set(rows["role"]) - {SCORED_ROLE})
    if other:
        raise EvaluationRowsError(f"representatives are scored on {SCORED_ROLE} origins only; "
                                  f"got rows with role {other}")
    cols = SERIES_KEYS + ["family", "model", "J", "scale", "n_origins"]
    if rows.empty:
        return pd.DataFrame(columns=cols)
    po = per_origin_costs(rows, taus)
    po = po[np.isclose(po["tau"], SELECTION_TAU, rtol=0, atol=1e-9)]
    u = po.groupby(SERIES_KEYS + ["family", "model"], sort=True).agg(
        pinball_sum=("pinball_sum", "sum"), abs_y_sum=("abs_y_sum", "sum"), n=("n", "sum"),
        n_origins=("origin", "nunique")).reset_index()
    u["J"] = u.pinball_sum / u.n
    u["scale"] = u.abs_y_sum / u.n
    return u[cols]


def _pick(score: dict[str, float]) -> tuple[str, str]:
    """Lowest score; a tie (within TIE_RTOL, relative) goes to the first name."""
    best = min(score.values())
    tol = TIE_RTOL * max(1.0, abs(best))
    tied = sorted(m for m, s in score.items() if s <= best + tol)
    return tied[0], ("lowest_validation_loss" if len(tied) == 1 else "tie_first_by_name")


def _select(forecasts, schema_version, family_of_model, rule, named, val_fraction, loss, taus, source):
    if loss not in LOSSES:
        raise ValueError(f"loss must be one of {LOSSES}, got {loss!r}")
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be in (0, 1): the choice needs validation origins")
    grid = check_taus(taus)
    if not np.isclose(grid, SELECTION_TAU, rtol=0, atol=1e-9).any():
        raise ValueError(f"the tau grid has no {SELECTION_TAU}")
    fc = load_cache(forecasts, schema_version, taus)
    assert_no_held_out(fc["dataset"], "representative selection")
    assert_no_held_out(fc["config"], "representative selection")
    # sorted once, so sums, scores and the frozen file do not depend on the row order of the input
    fc = fc.sort_values(KEY_COLS).reset_index(drop=True)
    lab = _labelled(fc, family_of_model)
    present = sorted(lab["model"].unique())
    candidates = {f: [m for m in present if family_of_model[m] == f] for f in FAMILIES}
    for fam, cands in candidates.items():
        if not cands:
            raise CacheError(f"no {fam} model in the cache")
    if rule == "fixed_named":
        restrict_to_representatives(fc, named, family_of_model)      # the engine's own checks

    rows = assign_roles(lab, val_fraction)
    by_role = rows[SERIES_KEYS + ["origin", "role"]].drop_duplicates()["role"].value_counts()
    u = validation_unit_losses(rows[rows["role"] == SCORED_ROLE], taus)
    u = u.merge(_covering(lab), on=SERIES_KEYS + ["model"])

    chosen, reasons, best_by_loss, coverage, scores = {}, {}, {}, {}, []
    for fam in FAMILIES:
        cands = candidates[fam]
        uf = u[u["family"] == fam]
        zero = uf["scale"] <= 0                # scaled loss undefined: listed in the counts, not scored
        n_zero = int(len(uf.loc[zero, SERIES_KEYS].drop_duplicates()))
        uf = uf[~zero]
        common = uf[uf.groupby(SERIES_KEYS)["model"].transform("nunique") == len(cands)]
        common = common.assign(scaled=common["J"] / common["scale"])
        mean = common.groupby("model")[["scaled", "J"]].mean()
        ref = common[common["model"] == cands[0]]
        n_common = int(len(ref))
        by_name = {ls: ({m: float(mean.at[m, col]) for m in cands} if n_common else {})
                   for ls, col in (("scaled", "scaled"), ("raw", "J"))}
        best_by_loss[fam] = {ls: (_pick(s)[0] if s else None) for ls, s in by_name.items()}

        if rule == "fixed_named":
            chosen[fam], reasons[fam] = named[fam], "named_by_caller"
        elif len(cands) == 1:
            chosen[fam], reasons[fam] = cands[0], "only_member"
        elif not n_common:
            raise CacheError(f"no (series, horizon) with validation origins is covered by every "
                             f"{fam} candidate {cands}; a representative cannot be chosen")
        else:
            chosen[fam], reasons[fam] = _pick(by_name[loss])

        coverage[fam] = dict(
            n_candidates=len(cands), n_units_scored=n_common,
            n_series_scored=int(len(ref[SERIES_KEYS[:3]].drop_duplicates())),
            n_datasets_scored=int(ref["dataset"].nunique()),
            n_validation_origins_scored=int(ref["n_origins"].sum()),
            n_units_dropped_zero_scale=n_zero)
        low = min(by_name[loss].values()) if n_common else None
        for m in cands:
            s = {ls: by_name[ls].get(m) for ls in LOSSES}
            scores.append(dict(
                family=fam, model=m, chosen=bool(m == chosen[fam]),
                mean_scaled_loss=s["scaled"], mean_raw_loss=s["raw"],
                loss_minus_best=(s[loss] - low if n_common else None),
                n_units_covered=int((uf["model"] == m).sum()), n_units_scored=n_common))

    payload = dict(
        schema=FROZEN_SCHEMA, rule=rule, representatives=chosen, reasons=reasons,
        selection=dict(tau=SELECTION_TAU, loss=loss, unit=list(SERIES_KEYS), tie_rtol=TIE_RTOL,
                       aggregation=AGGREGATION, tie_rule=TIE_RULE),
        best_by_loss=best_by_loss, val_fraction=float(val_fraction), roles_scored=[SCORED_ROLE],
        origins_statement=ORIGINS_STATEMENT, candidates=candidates, scores=scores, coverage=coverage,
        n_origins_by_role={r: int(by_role.get(r, 0)) for r in costs.ROLES},
        input=dict(sha256=input_fingerprint(fc, taus), cache_schema_version=str(schema_version),
                   n_rows=int(len(fc)), n_rows_missing_y=int(fc["y"].isna().sum()),
                   datasets=sorted(str(x) for x in fc["dataset"].unique()),
                   configs=sorted(str(x) for x in fc["config"].unique()),
                   horizons=sorted(int(x) for x in fc["horizon"].unique()),
                   seeds=sorted(int(x) for x in fc["seed"].unique()),
                   tau_grid=[float(x) for x in grid], source=dict(source or {})),
        code_sha256=code_sha256())
    payload = json.loads(_canonical(payload))       # plain JSON types; raises on NaN
    _check_payload(payload, rule)
    return payload


def best_of_family(forecasts, schema_version, family_of_model, val_fraction=1 / 3, loss="scaled",
                   taus=TAU_GRID, source=None) -> dict:
    """Choose, per family, the model with the lowest mean pinball loss at tau 0.50
    over validation origins (module docstring, steps 1 to 6).

    forecasts, schema_version  as for cache_io.load_cache
    family_of_model            {model: family} for every model in the cache
    val_fraction               the validation share of each (series, horizon)'s origins
    loss                       "scaled" (default) or "raw"
    source                     optional provenance to record with the input, e.g. a bundle hash
    Returns the payload that freeze() writes; payload["representatives"] is the dict
    the engine takes."""
    return _select(forecasts, schema_version, family_of_model, "best_of_family", None,
                   val_fraction, loss, taus, source)


def fixed_named(forecasts, schema_version, family_of_model, representatives, val_fraction=1 / 3,
                loss="scaled", taus=TAU_GRID, source=None) -> dict:
    """Take the caller's {family: model} as the representatives. They are checked
    by cache_io.restrict_to_representatives on this cache, so whatever the engine
    would refuse is refused here. The validation scores of every candidate are
    still recorded, for comparison with best_of_family; they do not affect the choice."""
    return _select(forecasts, schema_version, family_of_model, "fixed_named", representatives,
                   val_fraction, loss, taus, source)


def score_table(payload: dict) -> pd.DataFrame:
    """The frozen score table as a frame, best first within each family."""
    t = pd.DataFrame(payload["scores"])[list(SCORE_COLUMNS)]
    order = t["family"].map({f: i for i, f in enumerate(FAMILIES)}).rename("_family_order")
    col = LOSS_COLUMN[payload["selection"]["loss"]]
    return (t.join(order).sort_values(["_family_order", col, "model"], na_position="last")
            .drop(columns="_family_order").reset_index(drop=True))


# --------------------------------------------------------------------------- frozen file

def _is_sha256(x) -> bool:
    return isinstance(x, str) and len(x) == 64 and all(c in "0123456789abcdef" for c in x)


def _check_payload(p: dict, context: str) -> None:
    """Structural validation. Run when a selection is made, frozen and loaded."""
    if not isinstance(p, dict):
        raise FrozenFileError(f"{context}: payload is not an object")
    missing = [k for k in PAYLOAD_KEYS if k not in p]
    extra = [k for k in p if k not in PAYLOAD_KEYS]
    if missing or extra:
        raise FrozenFileError(f"{context}: payload keys differ. missing={missing} extra={extra}")
    if p["schema"] != FROZEN_SCHEMA:
        raise FrozenFileError(f"{context}: schema {p['schema']!r}, expected {FROZEN_SCHEMA!r}")
    if p["rule"] not in RULES:
        raise FrozenFileError(f"{context}: rule {p['rule']!r} is not one of {RULES}")
    sel = p["selection"]
    if not isinstance(sel, dict) or sel.get("loss") not in LOSSES or sel.get("tau") != SELECTION_TAU:
        raise FrozenFileError(f"{context}: selection must record loss in {LOSSES} and tau {SELECTION_TAU}")
    if p["roles_scored"] != [SCORED_ROLE]:
        raise FrozenFileError(f"{context}: roles_scored is {p['roles_scored']!r}; representatives are "
                              f"scored on {SCORED_ROLE} origins only")
    if not (isinstance(p["val_fraction"], (int, float)) and 0 < p["val_fraction"] < 1):
        raise FrozenFileError(f"{context}: val_fraction must be in (0, 1)")
    inp = p["input"]
    if not (isinstance(inp, dict) and _is_sha256(inp.get("sha256")) and _is_sha256(p["code_sha256"])):
        raise FrozenFileError(f"{context}: input and code fingerprints must be SHA-256 hex digests")
    reps, cands, reasons = p["representatives"], p["candidates"], p["reasons"]
    for name, d in (("representatives", reps), ("candidates", cands), ("reasons", reasons)):
        if not isinstance(d, dict) or set(d) != set(FAMILIES):
            raise FrozenFileError(f"{context}: {name} must name exactly {FAMILIES}")
    if not isinstance(p["scores"], list) or not all(isinstance(r, dict) for r in p["scores"]):
        raise FrozenFileError(f"{context}: scores must be a list of rows")
    for fam in FAMILIES:
        if not (isinstance(reps[fam], str) and reps[fam]):
            raise FrozenFileError(f"{context}: the {fam} representative must be a non-empty model name")
        if reps[fam] not in cands[fam]:
            raise FrozenFileError(f"{context}: representative {reps[fam]!r} is not a {fam} candidate")
        if reasons[fam] not in REASONS:
            raise FrozenFileError(f"{context}: unknown reason {reasons[fam]!r}")
        marked = [r.get("model") for r in p["scores"] if r.get("family") == fam and r.get("chosen")]
        scored = sorted(r.get("model") for r in p["scores"] if r.get("family") == fam)
        if marked != [reps[fam]] or scored != sorted(cands[fam]):
            raise FrozenFileError(f"{context}: the {fam} score table does not match the candidates "
                                  "and the chosen representative")
    # a file that names held-out data as its input is never valid
    assert_no_held_out(inp.get("datasets", []), f"{context}: input datasets")
    assert_no_held_out(inp.get("configs", []), f"{context}: input configs")


def content_sha256(payload: dict) -> str:
    """The hash freeze() records for this payload."""
    return _sha256(_canonical(payload))


def freeze(payload: dict, path) -> str:
    """Validate the selection and write it with its SHA-256 to `path` (UTF-8, LF,
    trailing newline). Returns the hash. Refuses to overwrite: a frozen file is
    written once."""
    _check_payload(payload, "representatives freeze")
    text = _canonical(payload)
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists; a frozen file is not overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"payload": json.loads(text), "sha256": _sha256(text)}
    path.write_text(json.dumps(doc, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")
    return doc["sha256"]


def load_frozen(path) -> dict:
    """Read a frozen file, verify its hash and structure, return the payload.
    A file that was edited, truncated or is missing a field is refused."""
    path = Path(path)
    text, _ = _read_verified(path)
    payload = json.loads(text)
    _check_payload(payload, str(path))
    return payload


def load_representatives(path) -> dict[str, str]:
    """The frozen {family: model} dict, as the engine takes it."""
    return dict(load_frozen(path)["representatives"])


# --------------------------------------------------------------------------- robustness

def representative_robustness(forecasts, schema_version, family_of_model, omega, static_threshold,
                              tsfm, classical=None, by=("selector",),
                              fit_scheme="leave_one_dataset_out", min_fit_datasets=2,
                              **build_kw) -> pd.DataFrame:
    """Does the conclusion depend on the classical representative?

    Runs the engine (pipeline.build_tables and pipeline.run) once per candidate
    classical representative with the TSFM side fixed, and stacks
    evaluate.summarize(by=by) with a classical_representative column.

    classical  the candidates; None means every classical model in the cache.
    build_kw   passed to build_tables (fit_only_datasets, omega_col, omega_variant, ...).

    THIS READS EVALUATION ORIGINS. It is an analysis of results, to run after
    the rule and the primary representatives are frozen. Never choose a
    representative from its output. Oracle and best-fixed are those of each
    pair, so compare selectors within a row group, not costs across groups."""
    if classical is None:
        present = set(load_cache(forecasts, schema_version, build_kw.get("taus", TAU_GRID))["model"])
        classical = [m for m, f in family_of_model.items() if f == FAMILIES[0] and m in present]
    parts = []
    for model in sorted(classical):
        reps = {FAMILIES[0]: model, FAMILIES[1]: tsfm}
        tables = build_tables(forecasts, schema_version, reps, family_of_model, omega, **build_kw)
        res, _ = run(tables, reps, static_threshold, fit_scheme=fit_scheme,
                     min_fit_datasets=min_fit_datasets)
        s = summarize(res, by=by)
        s.insert(0, "classical_representative", model)
        parts.append(s)
    return pd.concat(parts, ignore_index=True)
