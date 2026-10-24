"""E2: fit a selector on development data, freeze it, evaluate it elsewhere.

    python -m decision_aware_tsfm.analysis.experiments.e2 --out artifacts/runs/e2_fixture

The command above runs on the SYNTHETIC FIXTURE. Its output is an interface
check, not a result about real model families.

The intended order is

  fit(fit_tables, ...)   -> FitSpec          in memory, not accepted by evaluate
  freeze(spec, path)     -> writes JSON {payload, sha256}
  load_frozen(path)      -> FrozenSpec       the only thing evaluate accepts
  evaluate(frozen, tables, allow_held_out=False, ledger_path=None) -> EvalResult

WHAT THE HASH IS AND IS NOT. The frozen file carries the SHA-256 of its own
content, with no key. It protects against ACCIDENTAL changes: a file edited by
hand, truncated, or swapped for an older one without also updating the hash is
refused. It is not a security control. Anyone who edits the file and
recomputes the hash, or writes a file by hand that never went through fit,
gets a file this module accepts. The order above is therefore a convention
that the types make hard to break by mistake; someone who sets out to skip
the fit step can. What makes a held-out evaluation credible is the ledger (below) kept
under version control together with the frozen file, so that a reader can see
which frozen specification was evaluated on held-out data, when, and how often.

Rules enforced here
  * fit refuses any dataset, config or series name that matches the held-out
    rule in decision.guards (M5, PJM). There is no flag to turn this off.
  * fit reads the settings the tables were built with from
    fit_tables["settings"] (written by decision.pipeline.build_tables):
    representatives, val_fraction, omega_col, omega_variant, crps_method and
    the tau grid. They are not typed in by the caller. evaluate refuses
    evaluation tables without "settings" and tables whose settings differ from
    the frozen ones in any of those six, and it checks that the models that
    were scored are exactly the frozen representatives.
  * The specification records a hash of the selector's source code (the
    adapter class and the module that defines the selector it builds).
    evaluate recomputes it and refuses when it differs, so a selector kind
    that was edited or registered again with other code after freezing is
    refused. Code that the selector calls from other modules is not covered.
  * freeze validates the specification; load_frozen validates it again.
  * evaluate has no fitting inputs. It receives evaluation tables and builds
    the selector from the frozen parameters only. The selector sees copies of
    the units and of the validation costs, never the test costs.
  * evaluate refuses when an evaluation dataset or an evaluation series_id was
    used in fitting. Names are compared after Unicode normalisation, trimming
    and case folding. The check compares names, not values: fit data renamed
    in every column is not detected.
  * Held-out names may appear only in evaluate, and only with
    allow_held_out=True (the bool, not a truthy value) and a ledger_path. One
    JSON line is appended to the ledger for every evaluation that touched
    held-out names, before any selector runs: time, frozen hash, selector code
    hash, datasets, held-out names. Every EvalResult table carries the frozen
    hash.
  * The tau grid is checked for every series at every horizon, in fit and in
    evaluate. A missing tau and a tau outside the frozen grid both raise.

Selector plug-in shape. The frozen file names a selector `kind`. A kind is an
adapter object with
    kind : str
    fit(fit_units) -> dict                 JSON-serialisable parameters
    build(params, fit_series, fit_datasets) -> selector
where selector has  name, needs_hindsight (must be False)  and
select(SelectorInput) -> decisions, exactly like the engine's comparators.
The one adapter registered here wraps the engine's PLACEHOLDER DOmegaLinear.
The real D_Omega (owner: Abdullah) is added with register_selector_kind.

Nothing in this module reads forecast data. Tables come from the caller.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import inspect
import json
import math
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..decision.evaluate import evaluate as _attach_costs
from ..decision.guards import FitEvalOverlapError, HeldOutLeakError, is_held_out
from ..decision.metrics import TAU_GRID, check_taus
from ..decision.schema import FAMILIES, UNIT_KEYS, validate_selection_frame
from ..decision.selectors import (AlwaysClassical, AlwaysTSFM, CRPSValidation, DOmegaLinear,
                                  HindsightOracle, PinballValidation, SelectorInput,
                                  StaticOmegaRule)

FROZEN_SCHEMA = "e2-frozen-v2"
PAYLOAD_KEYS = ("schema", "selector_kind", "selector_name", "selector_params",
                "selector_code_sha256", "static_threshold", "representatives", "tau_grid",
                "omega_col", "omega_variant", "val_fraction", "crps_method",
                "cache_schema_version", "fit_datasets", "fit_configs", "fit_series", "n_fit_units")
# settings that the evaluation tables must share with the fit tables: payload key -> settings key
CHECKED_SETTINGS = {"representatives": "representatives", "val_fraction": "val_fraction",
                    "omega_col": "omega_col", "omega_variant": "omega_variant",
                    "crps_method": "crps_method", "tau_grid": "taus"}
SETTINGS_KEYS = ("schema_version", "representatives", "val_fraction", "omega_col", "omega_variant",
                 "crps_method", "taus", "models_scored")
BASELINE_NAMES = ("always_classical", "always_tsfm", "static_omega", "pinball_validation",
                  "crps_validation", "hindsight_oracle")
VALIDATION_COMPARATOR = "pinball_validation"
WORST_FRACTION = 0.10
TAU_ATOL = 1e-9
SERIES_HORIZON = ["dataset", "config", "series_id", "horizon"]
TIMING_LABEL = "wall time of select() only, single run on this machine, not reproducible"


class NotFrozenError(RuntimeError):
    """evaluate was given something other than a specification loaded from a frozen file."""


class FrozenFileError(RuntimeError):
    """A frozen file is missing, malformed, or its content does not match its hash."""


class SelectorCodeError(FrozenFileError):
    """The selector's source code cannot be read, or differs from the frozen hash."""


class SettingsMismatchError(ValueError):
    """Evaluation tables were built with settings other than the frozen ones, or record none."""


class TauGridError(ValueError):
    """A table does not carry exactly the frozen tau grid for every series and horizon."""


class LedgerRequiredError(ValueError):
    """allow_held_out was set without a ledger file path."""


# --------------------------------------------------------------------------- names

def normalize_name(x) -> str:
    """Form used to compare names: Unicode NFKC, surrounding white space
    removed, case folded. 'FX_DFIT ' and 'fx_dfit' compare equal."""
    return unicodedata.normalize("NFKC", str(x)).strip().casefold()


def _held_out_names(names) -> list[str]:
    return sorted({str(n) for n in names if is_held_out(n) or is_held_out(normalize_name(n))})


def _assert_no_held_out(names, context: str) -> None:
    hits = _held_out_names(names)
    if hits:
        raise HeldOutLeakError(f"{context}: held-out data present {hits[:5]}")


# --------------------------------------------------------------------------- selector kinds

class DOmegaLinearPlaceholder:
    """Adapter for the engine's placeholder DOmegaLinear. Not the real D_Omega."""
    kind = "d_omega_linear_placeholder"

    def fit(self, fit_units: pd.DataFrame) -> dict:
        s = DOmegaLinear.fit(fit_units)
        return {"omega0": s.omega0, "beta": s.beta}

    def build(self, params: dict, fit_series, fit_datasets):
        return DOmegaLinear(omega0=float(params["omega0"]), beta=float(params["beta"]),
                            fitted_series=frozenset(fit_series),
                            fitted_datasets=tuple(fit_datasets))


SELECTOR_KINDS: dict[str, object] = {}


def register_selector_kind(adapter) -> None:
    """Make a selector kind available to fit and evaluate. See the module docstring.

    Registering a kind again replaces the adapter. A frozen file made with the
    earlier adapter is then refused by evaluate, because the selector code
    hash no longer matches.
    """
    for attr in ("kind", "fit", "build"):
        if not hasattr(adapter, attr):
            raise TypeError(f"selector adapter has no {attr!r}")
    SELECTOR_KINDS[str(adapter.kind)] = adapter


register_selector_kind(DOmegaLinearPlaceholder())


def selector_code_sha256(adapter, selector) -> str:
    """SHA-256 over the source text of (1) the adapter's class and (2) the
    whole module that defines the class of the selector the adapter builds.

    Covers the code most likely to be edited between freezing and evaluating.
    Does not cover code those two call in other modules. Raises
    SelectorCodeError when the source cannot be read (for example a class
    typed into an interactive session): a selector whose code cannot be
    recorded cannot be frozen.
    """
    parts = []
    targets = (("adapter class", type(adapter), f"{type(adapter).__module__}.{type(adapter).__qualname__}"),
               ("selector module", inspect.getmodule(type(selector)), type(selector).__module__))
    for label, obj, name in targets:
        try:
            src = inspect.getsource(obj)
        except (OSError, TypeError) as e:
            raise SelectorCodeError(f"cannot read the source of the {label} {name!r} ({e}); "
                                    "its code cannot be frozen") from e
        parts.append(f"{label}\x1f{name}\x1f{src}")
    return _sha256("\x1e".join(parts))


# --------------------------------------------------------------------------- specification

def _canonical(payload: dict) -> str:
    """One fixed JSON text per payload: sorted keys, no spaces, NaN not allowed."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _check_payload(p: dict, context: str) -> None:
    """Structural validation of a specification. Run by fit, freeze, load_frozen and evaluate."""
    if not isinstance(p, dict):
        raise FrozenFileError(f"{context}: payload is not an object")
    missing = [k for k in PAYLOAD_KEYS if k not in p]
    extra = [k for k in p if k not in PAYLOAD_KEYS]
    if missing or extra:
        raise FrozenFileError(f"{context}: payload keys differ. missing={missing} extra={extra}")
    if p["schema"] != FROZEN_SCHEMA:
        raise FrozenFileError(f"{context}: schema {p['schema']!r}, expected {FROZEN_SCHEMA!r}")
    if p["selector_kind"] not in SELECTOR_KINDS:
        raise FrozenFileError(f"{context}: selector kind {p['selector_kind']!r} is not registered; "
                              f"known kinds {sorted(SELECTOR_KINDS)}")
    if p["selector_name"] in BASELINE_NAMES:
        raise FrozenFileError(f"{context}: selector name {p['selector_name']!r} is a baseline's name")
    code = p["selector_code_sha256"]
    if not (isinstance(code, str) and len(code) == 64 and all(c in "0123456789abcdef" for c in code)):
        raise FrozenFileError(f"{context}: selector_code_sha256 is not a SHA-256 hex digest")
    reps = p["representatives"]
    if not isinstance(reps, dict) or set(reps) != set(FAMILIES):
        raise FrozenFileError(f"{context}: representatives must name exactly {FAMILIES}")
    if not all(isinstance(v, str) and v for v in reps.values()):
        raise FrozenFileError(f"{context}: representatives must be non-empty model names")
    try:
        grid = check_taus(p["tau_grid"])
    except (TypeError, ValueError) as e:
        raise FrozenFileError(f"{context}: bad tau grid ({e})") from e
    if np.any(np.diff(grid) <= 0):
        raise FrozenFileError(f"{context}: tau grid must be strictly increasing")
    if not (isinstance(p["static_threshold"], (int, float)) and math.isfinite(p["static_threshold"])):
        raise FrozenFileError(f"{context}: static_threshold must be a finite number")
    if not (isinstance(p["val_fraction"], (int, float)) and 0 <= p["val_fraction"] < 1):
        raise FrozenFileError(f"{context}: val_fraction must be in [0, 1)")
    series = p["fit_series"]
    if not (isinstance(series, list) and series
            and all(isinstance(s, list) and len(s) == 3 for s in series)):
        raise FrozenFileError(f"{context}: fit_series must be a non-empty list of "
                              "[dataset, config, series_id]")
    if sorted({s[0] for s in series}) != p["fit_datasets"] \
            or sorted({s[1] for s in series}) != p["fit_configs"]:
        raise FrozenFileError(f"{context}: fit_datasets and fit_configs do not match fit_series")
    if not (isinstance(p["n_fit_units"], int) and p["n_fit_units"] >= len(series) * len(grid)):
        raise FrozenFileError(f"{context}: n_fit_units is smaller than fit_series times the tau grid")
    # a specification that names held-out data as fit data is never valid
    _assert_no_held_out(p["fit_datasets"], f"{context}: fit datasets")
    _assert_no_held_out(p["fit_configs"], f"{context}: fit configs")
    _assert_no_held_out([s[2] for s in p["fit_series"]], f"{context}: fit series")


_FIT_TOKEN = object()
_LOAD_TOKEN = object()


class _Spec:
    __slots__ = ("_text",)

    def __setattr__(self, name, value):
        raise AttributeError("a specification cannot be modified")

    def __delattr__(self, name):
        raise AttributeError("a specification cannot be modified")

    @property
    def payload(self) -> dict:
        """A fresh copy of the content on every access; editing it changes nothing."""
        return json.loads(self._text)


class FitSpec(_Spec):
    """Result of fit. In memory only. Created by fit(); evaluate does not accept
    it: freeze it and load the file."""
    __slots__ = ()

    def __init__(self, text: str, _token=None):
        if _token is not _FIT_TOKEN:
            raise NotFrozenError("FitSpec is created only by fit(...)")
        object.__setattr__(self, "_text", text)

    @property
    def sha256(self) -> str:
        return _sha256(self._text)

    def __repr__(self):
        return f"FitSpec(sha256={self.sha256[:12]}...)"


class FrozenSpec(_Spec):
    """A specification read from a frozen file by load_frozen.

    payload   a fresh copy of the frozen content on every access
    sha256    hash of the content, as verified at load time
    path      the file it was loaded from
    """
    __slots__ = ("_hash", "_path")

    def __init__(self, text: str, sha256: str, path: Path, _token=None):
        if _token is not _LOAD_TOKEN:
            raise NotFrozenError("FrozenSpec is created only by load_frozen(path)")
        object.__setattr__(self, "_text", text)
        object.__setattr__(self, "_hash", sha256)
        object.__setattr__(self, "_path", Path(path))

    @property
    def sha256(self) -> str:
        return self._hash

    @property
    def path(self) -> Path:
        return self._path

    def __repr__(self):
        return f"FrozenSpec(sha256={self._hash[:12]}..., path={str(self._path)!r})"


def _settings(tables: dict, context: str) -> dict:
    s = tables.get("settings") if isinstance(tables, dict) else None
    if not isinstance(s, dict):
        raise SettingsMismatchError(
            f"{context}: the tables carry no 'settings' record, so it cannot be checked how they "
            "were built. Build them with decision.pipeline.build_tables")
    missing = [k for k in SETTINGS_KEYS if k not in s]
    if missing:
        raise SettingsMismatchError(f"{context}: tables['settings'] lacks {missing}")
    reps = s["representatives"]
    if not isinstance(reps, dict) or set(reps) != set(FAMILIES):
        raise SettingsMismatchError(f"{context}: settings representatives must name exactly {FAMILIES}")
    if sorted(s["models_scored"]) != sorted(set(reps.values())):
        raise SettingsMismatchError(
            f"{context}: models scored {sorted(s['models_scored'])} are not the representatives "
            f"{sorted(set(reps.values()))}")
    return s


def assert_tau_grid(frame: pd.DataFrame, grid, what: str) -> None:
    """Raise TauGridError unless every (dataset, config, series_id, horizon) of
    `frame` has every tau of `grid` and no other tau."""
    g = np.sort(np.asarray(grid, dtype=float))
    tau = frame["tau"].to_numpy(dtype=float)
    k = np.abs(tau[:, None] - g[None, :]).argmin(axis=1)
    on_grid = np.abs(tau - g[k]) <= TAU_ATOL
    if not on_grid.all():
        raise TauGridError(f"{what}: tau {sorted({float(x) for x in tau[~on_grid]})[:5]} not in the "
                           f"grid {g.tolist()}")
    n = frame[SERIES_HORIZON].assign(_k=k).drop_duplicates().groupby(SERIES_HORIZON)["_k"].nunique()
    short = n[n < g.size]
    if len(short):
        raise TauGridError(f"{what}: {len(short)} (series, horizon) group(s) lack part of the tau grid "
                           f"{g.tolist()}, e.g. {dict(zip(SERIES_HORIZON, short.index[0]))} has "
                           f"{int(short.iloc[0])} of {g.size} taus")


def fit(fit_tables: dict, *, selector_kind: str, static_threshold: float,
        selector_name: str = "d_omega") -> FitSpec:
    """Fit the selector on fit-only units and collect everything needed to select.

    fit_tables  the engine's tables (decision.pipeline.build_tables). Used:
                fit_tables['fit_units'] (unit keys, omega, J_classical, J_tsfm,
                scale, for datasets set aside for fitting) and
                fit_tables['settings'], from which the representatives, the
                validation fraction, the Omega column and variant, the CRPS
                method and the tau grid are RECORDED. They are not arguments,
                so the specification cannot claim settings the tables were not
                built with.
    static_threshold  threshold of the static Omega rule. Not fitted here; it
                is read from the source paper and passed in.
    Held-out names anywhere in fit_units raise HeldOutLeakError.
    """
    settings = _settings(fit_tables, "E2 fit")
    fit_units = fit_tables["fit_units"]
    for col in ("dataset", "config", "series_id"):
        _assert_no_held_out(fit_units[col], f"E2 fit ({col})")
    if selector_kind not in SELECTOR_KINDS:
        raise ValueError(f"unknown selector kind {selector_kind!r}; known {sorted(SELECTOR_KINDS)}")
    if selector_name in BASELINE_NAMES:
        raise ValueError(f"selector name {selector_name!r} is a baseline's name")
    if fit_units.empty:
        raise ValueError("no fit units")
    grid = [float(x) for x in check_taus(settings["taus"])]
    assert_tau_grid(fit_units, grid, "E2 fit units")
    adapter = SELECTOR_KINDS[selector_kind]
    params = json.loads(json.dumps(adapter.fit(fit_units.copy()), allow_nan=False))
    series = (fit_units[["dataset", "config", "series_id"]].drop_duplicates()
              .sort_values(["dataset", "config", "series_id"]))
    fit_series = [[str(a), str(b), str(c)] for a, b, c in series.itertuples(index=False)]
    fit_datasets = sorted({s[0] for s in fit_series})
    selector = adapter.build(params, [s[2] for s in fit_series], fit_datasets)
    _check_selector(selector, selector_name)
    payload = dict(
        schema=FROZEN_SCHEMA, selector_kind=selector_kind, selector_name=selector_name,
        selector_params=params, selector_code_sha256=selector_code_sha256(adapter, selector),
        static_threshold=float(static_threshold),
        representatives={k: str(settings["representatives"][k]) for k in sorted(FAMILIES)},
        tau_grid=grid, omega_col=str(settings["omega_col"]), omega_variant=settings["omega_variant"],
        val_fraction=float(settings["val_fraction"]), crps_method=str(settings["crps_method"]),
        cache_schema_version=str(settings["schema_version"]),
        fit_datasets=fit_datasets, fit_configs=sorted({s[1] for s in fit_series}),
        fit_series=fit_series, n_fit_units=int(len(fit_units)))
    text = _canonical(payload)          # raises on NaN or non-JSON parameters
    _check_payload(json.loads(text), "E2 fit")
    return FitSpec(text, _token=_FIT_TOKEN)


def _check_selector(selector, frozen_name: str) -> None:
    if getattr(selector, "needs_hindsight", True):
        raise ValueError("a frozen selector must not need hindsight")
    if selector.name != frozen_name:
        raise ValueError(f"built selector is named {selector.name!r}, frozen name is {frozen_name!r}")


def freeze(spec: FitSpec, path) -> str:
    """Validate the specification and write it with its SHA-256 to `path`.
    Returns the hash. Refuses to overwrite an existing file: a frozen selector
    is written once. Takes only a FitSpec made by fit()."""
    if not isinstance(spec, FitSpec):
        raise TypeError("freeze takes the FitSpec returned by fit")
    payload = spec.payload
    _check_payload(payload, "E2 freeze")
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists; a frozen specification is not overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"payload": payload, "sha256": spec.sha256}
    path.write_text(json.dumps(doc, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return spec.sha256


def _read_verified(path: Path) -> tuple[str, str]:
    if not path.is_file():
        raise FrozenFileError(f"frozen file not found: {path}")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise FrozenFileError(f"{path}: not valid JSON ({e})") from e
    if not isinstance(doc, dict) or set(doc) != {"payload", "sha256"}:
        raise FrozenFileError(f"{path}: expected an object with exactly 'payload' and 'sha256'")
    try:
        text = _canonical(doc["payload"])
    except (TypeError, ValueError) as e:
        raise FrozenFileError(f"{path}: payload cannot be canonicalised ({e})") from e
    actual = _sha256(text)
    if actual != doc["sha256"]:
        raise FrozenFileError(f"{path}: content hash {actual[:12]}... does not match the recorded "
                              f"{str(doc['sha256'])[:12]}...; the file was modified after freezing")
    return text, actual


def load_frozen(path) -> FrozenSpec:
    """Read a frozen file, verify its hash and structure, return a FrozenSpec.
    The hash shows the file is internally consistent. It does not show that
    the file came from fit (module docstring)."""
    path = Path(path)
    text, sha = _read_verified(path)
    _check_payload(json.loads(text), str(path))
    return FrozenSpec(text, sha, path.resolve(), _token=_LOAD_TOKEN)


# --------------------------------------------------------------------------- metrics

def unit_regret(selection: pd.DataFrame, selector: str) -> pd.DataFrame:
    """R_sel per unit for one selector: regret = J_selected - min_m J_m (cost
    units) and scaled_regret = regret / scale. Undecided units keep NaN."""
    s = selection[selection["selector"] == selector]
    return s[UNIT_KEYS + ["status", "chosen_family", "oracle_family", "realized_cost",
                          "oracle_cost", "regret", "scale", "scaled_regret"]].reset_index(drop=True)


NO_SELECTED = "undefined: no selected units in this group"


def regret_summary(selection: pd.DataFrame, by=("selector", "horizon", "tau")) -> pd.DataFrame:
    """Per group, three regret figures, each under its own name.

    All sums run over the units THE SELECTOR DECIDED in the group (status
    'selected') and use scaled costs (cost / scale):
        SEL         realised cost of the selector's choices
        ORACLE      cost of the hindsight oracle on the same units
        BEST_FIXED  cost, on the same units, of the engine's best single family

      mean_scaled_regret           (SEL - ORACLE) / n_selected
      regret_over_oracle_cost      (SEL - ORACLE) / ORACLE
                                   Regret as a fraction of what the oracle pays.
                                   0 is the oracle. Undefined when ORACLE is 0.
      frac_oracle_value_recovered  (BEST_FIXED - SEL) / (BEST_FIXED - ORACLE)
                                   Share of the gain from selecting at all that the
                                   selector obtained. 1 is the oracle, 0 is the best
                                   fixed family, negative is worse than that.
                                   Undefined when BEST_FIXED equals ORACLE (a
                                   dominance region or an exact tie): there is no
                                   selection value to recover. Same quantity and
                                   same zero rule as evaluate.summarize.

    Which of these the paper calls "normalized regret" is not decided here.
    The overview and plan call the hindsight oracle "the denominator for
    normalised regret" (regret_over_oracle_cost reads that way); the proposal
    lists "fraction of hindsight-oracle selection value recovered" as a
    separate metric. Both are reported so neither is mislabelled.

    BEST_FIXED is the engine's best_fixed_family: one family per (horizon,
    tau), chosen in hindsight over ALL evaluation units of that horizon and
    tau. It is not re-chosen inside smaller groups, so when `by` includes
    dataset the "0 is the best fixed family" reading holds only for the pooled
    (horizon, tau) group, and frac_oracle_value_recovered can be positive for
    an always-one-family selector within a dataset.

    A value that is not defined is NaN, with <metric>_defined False and the
    reason in <metric>_reason. A group in which the selector decided no unit
    has n_selected 0 and every figure undefined for that reason. No division
    by zero happens.
    """
    by = list(by)
    if selection.empty:
        raise ValueError("empty selection frame")
    r = selection[selection["status"] == "selected"]
    sums = pd.DataFrame({"sel": r["realized_cost"] / r["scale"], "orc": r["oracle_cost"] / r["scale"],
                         "bf": r["best_fixed_cost"] / r["scale"]}).groupby([r[c] for c in by]).sum()
    counts = selection.groupby(by).agg(
        n_units=("status", "size"),
        n_selected=("status", lambda x: int((x == "selected").sum())))
    g = counts.join(sums).reset_index()
    some = (g["n_selected"] > 0).to_numpy()
    regret = (g["sel"] - g["orc"]).to_numpy(dtype=float)
    orc = g["orc"].to_numpy(dtype=float)
    gap = (g["bf"] - g["orc"]).to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        orc_ok = some & (orc > 1e-12)
        gap_ok = some & (gap > 1e-12 * np.maximum(1.0, np.abs(g["bf"].to_numpy(dtype=float))))
    g["mean_scaled_regret"] = np.where(some, regret / np.where(some, g["n_selected"], 1), np.nan)
    g["regret_over_oracle_cost"] = np.where(orc_ok, regret / np.where(orc_ok, orc, 1.0), np.nan)
    g["regret_over_oracle_cost_defined"] = orc_ok
    g["regret_over_oracle_cost_reason"] = np.where(
        orc_ok, "", np.where(some, "undefined: oracle cost is zero in this group", NO_SELECTED))
    g["frac_oracle_value_recovered"] = np.where(
        gap_ok, (g["bf"].to_numpy(dtype=float) - g["sel"].to_numpy(dtype=float))
        / np.where(gap_ok, gap, 1.0), np.nan)
    g["frac_oracle_value_recovered_defined"] = gap_ok
    g["frac_oracle_value_recovered_reason"] = np.where(
        gap_ok, "", np.where(some, "undefined: best-fixed family equals the oracle in this group",
                             NO_SELECTED))
    return g.drop(columns=["sel", "orc", "bf"])


def gap_to_validation(selection: pd.DataFrame, selector: str,
                      reference: str = VALIDATION_COMPARATOR,
                      by=("horizon", "tau")) -> pd.DataFrame:
    """Mean over units of (scaled realised cost of `selector`) minus (scaled
    realised cost of `reference`), over the units where BOTH made a choice.
    Units either of them left undecided are left out of the mean, not counted
    as a gap of zero; n_both_selected is the number of units in the mean.

    Positive: the selector costs more than full pinball-tau validation.
    Negative: it costs less. Both face the same oracle, so this also equals
    the difference of their mean scaled regrets on those units.
    """
    by = list(by)
    cols = UNIT_KEYS + ["status", "realized_cost", "scale"]
    a = selection.loc[selection["selector"] == selector, cols]
    b = selection.loc[selection["selector"] == reference, cols]
    if a.empty or b.empty:
        raise ValueError(f"selector {selector!r} or reference {reference!r} not in the results")
    m = a.merge(b, on=UNIT_KEYS, suffixes=("", "_ref"), validate="1:1")
    both = (m["status"] == "selected") & (m["status_ref"] == "selected")
    m["gap"] = np.where(both, (m["realized_cost"] - m["realized_cost_ref"]) / m["scale"], np.nan)
    out = m.groupby(by).agg(n_units=("gap", "size"), n_both_selected=("gap", "count"),
                            gap_to_validation=("gap", "mean")).reset_index()
    out.insert(len(by), "selector", selector)
    out.insert(len(by) + 1, "reference", reference)
    return out


def worst_decile_regret(selection: pd.DataFrame, selector: str, by=("horizon", "tau"),
                        fraction: float = WORST_FRACTION) -> pd.DataFrame:
    """Worst-decile per-series regret.

    Definition. Within each group (default: one horizon and one tau):
      1. per-series regret = the series' mean scaled regret over its units in
         the group. A series is (dataset, config, series_id). With the default
         grouping a series has one unit, so this is that unit's scaled regret.
      2. k = ceil(fraction * number of series), at least 1.
      3. worst_decile_regret = mean of the k LARGEST per-series regrets.
    So it is the average regret of the worst-off tenth of series (an upper
    tail mean), not the 90th percentile. p90_series_regret (linear
    interpolation between order statistics) is reported next to it for
    comparison. Only series the selector decided are included.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    by = list(by)
    s = selection[(selection["selector"] == selector) & (selection["status"] == "selected")]
    if s.empty:
        raise ValueError(f"selector {selector!r} has no selected units")
    if s["scaled_regret"].isna().any():
        raise ValueError("selected units with missing scaled regret")
    per = (s.groupby(by + ["dataset", "config", "series_id"])["scaled_regret"].mean()
           .rename("series_regret").reset_index())
    rows = []
    for key, g in per.groupby(by, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        v = np.sort(g["series_regret"].to_numpy(dtype=float))[::-1]
        k = max(1, int(math.ceil(fraction * v.size - 1e-12)))
        rows.append(dict(zip(by, key), selector=selector, n_series=int(v.size), n_in_worst=k,
                         worst_decile_regret=float(v[:k].mean()),
                         p90_series_regret=float(np.quantile(v, 0.90)),
                         max_series_regret=float(v[0]), mean_series_regret=float(v.mean())))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- evaluate

WORST_DECILE_COLUMNS = ("horizon", "tau", "selector", "n_series", "n_in_worst", "worst_decile_regret",
                        "p90_series_regret", "max_series_regret", "mean_series_regret")
RESULT_TABLES = ("selection", "unit_regret", "regret_summary", "gap_to_validation", "worst_decile",
                 "selector_cost")


@dataclass(frozen=True)
class EvalResult:
    selection: pd.DataFrame          # engine schema, baselines plus the frozen selector
    selector_name: str
    unit_regret: pd.DataFrame        # R_sel per unit for the frozen selector
    regret_summary: pd.DataFrame     # every selector, per (horizon, tau); see regret_summary()
    gap_to_validation: pd.DataFrame
    worst_decile: pd.DataFrame
    selector_cost: pd.DataFrame      # single-run wall seconds of select(); not reproducible
    frozen_sha256: str
    selector_code_sha256: str
    frozen_path: str
    held_out_evaluated: tuple        # held-out names that were present (empty unless allowed)
    ledger_entry: dict | None        # the line appended to the ledger, if any
    settings: dict                   # settings of the evaluation tables, as checked

    def manifest(self) -> dict:
        return dict(experiment="E2", frozen_sha256=self.frozen_sha256,
                    selector_code_sha256=self.selector_code_sha256, frozen_path=self.frozen_path,
                    selector_name=self.selector_name,
                    datasets=sorted(self.selection["dataset"].unique().tolist()),
                    n_units=int(len(self.unit_regret)),
                    held_out_evaluated=list(self.held_out_evaluated),
                    ledger_entry=self.ledger_entry,
                    settings=json.loads(json.dumps(self.settings, default=list)),
                    hash_note="unkeyed SHA-256: guards against accidental edits, not a security control")

    def write(self, out_dir) -> list[str]:
        """Write every table as CSV plus manifest.json. Each file carries the
        frozen hash: as a column in the derived tables, in the manifest, and
        in the file name of the selection frame (whose columns are fixed by
        the engine schema)."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        files = []
        for name in RESULT_TABLES:
            fn = f"selection_{self.frozen_sha256[:12]}.csv" if name == "selection" else f"{name}.csv"
            getattr(self, name).to_csv(out / fn, index=False, lineterminator="\n")
            files.append(fn)
        (out / "manifest.json").write_text(json.dumps(dict(self.manifest(), files=files), indent=2) + "\n", encoding="utf-8")
        return files + ["manifest.json"]


def _check_settings(p: dict, s: dict) -> None:
    for pkey, skey in CHECKED_SETTINGS.items():
        want, got = p[pkey], s[skey]
        if pkey == "tau_grid":
            a, b = np.asarray(want, dtype=float), np.asarray(got, dtype=float)
            same = a.shape == b.shape and bool(np.all(np.abs(a - b) <= TAU_ATOL))
        elif pkey == "val_fraction":
            same = abs(float(want) - float(got)) <= 1e-12
        elif pkey == "representatives":
            same = dict(want) == {k: str(v) for k, v in dict(got).items()}
        else:
            same = want == got
        if not same:
            raise SettingsMismatchError(f"evaluation tables were built with {skey}={got!r} but the "
                                        f"frozen specification says {want!r}")
    if sorted(s["models_scored"]) != sorted(set(p["representatives"].values())):
        raise SettingsMismatchError(
            f"evaluation tables scored models {sorted(s['models_scored'])}; the frozen "
            f"representatives are {sorted(set(p['representatives'].values()))}")


def _append_ledger(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")


def read_ledger(path) -> list[dict]:
    """The ledger as a list of entries, oldest first."""
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def evaluate(frozen: FrozenSpec, tables: dict, allow_held_out: bool = False,
             ledger_path=None) -> EvalResult:
    """Apply a frozen selector to evaluation tables and score it.

    frozen   a FrozenSpec from load_frozen. Anything else raises NotFrozenError.
    tables   the engine's tables for the EVALUATION data: units, val, test,
             n_val and settings. tables['fit_units'] is never read. The test
             table is restricted to the units given, so the engine's
             best-fixed family is chosen on the evaluated units only.
    allow_held_out  must be the bool True for any evaluation dataset, config or
             series whose name matches the held-out rule (M5, PJM). Any other
             non-bool value raises TypeError. It never affects fitting.
    ledger_path  required whenever allow_held_out is True. One JSON line is
             appended per evaluation in which held-out names were present,
             after all checks pass and before any selector runs.

    The function takes no parameters to tune. The selector is rebuilt from the
    frozen parameters, its code hash is compared with the frozen one, it sees
    copies of the units and validation costs only (never test costs), and the
    six baselines run next to it on the same units.

    selector_cost is the wall time of each select() call, measured once. It is
    not reproducible and is not the proposal's selector-compute metric (that is
    forecasting cost avoided, which this module does not see).
    """
    if not isinstance(frozen, FrozenSpec):
        raise NotFrozenError("evaluate needs a specification loaded from a frozen file with "
                             f"load_frozen(path); got {type(frozen).__name__}")
    if not isinstance(allow_held_out, bool):
        raise TypeError(f"allow_held_out must be True or False, got {allow_held_out!r}")
    if allow_held_out and ledger_path is None:
        raise LedgerRequiredError("allow_held_out=True needs ledger_path: every held-out evaluation "
                                  "is recorded")
    text, sha = _read_verified(frozen.path)           # the file must still be there, unchanged
    if sha != frozen.sha256 or text != frozen._text:
        raise FrozenFileError(f"{frozen.path} changed since it was loaded")
    p = frozen.payload
    _check_payload(p, str(frozen.path))

    # the evaluation tables must have been built with the frozen settings
    settings = _settings(tables, "E2 evaluate")
    _check_settings(p, settings)

    units, val, n_val = tables["units"], tables["val"], tables["n_val"]
    if units.empty:
        raise ValueError("no evaluation units")
    test = tables["test"].merge(units[UNIT_KEYS], on=UNIT_KEYS, how="inner", validate="1:1")

    # held-out gate: only here, only with the explicit flag
    names = set(units["dataset"]) | set(units["config"]) | set(units["series_id"].astype(str))
    held = tuple(_held_out_names(names))
    if held and not allow_held_out:
        raise HeldOutLeakError(f"E2 evaluate: held-out data present {list(held)[:5]}; "
                               "pass allow_held_out=True and a ledger_path for the final transfer "
                               "test only")

    # fit / evaluation separation, on normalised names
    fit_ds = {normalize_name(x) for x in p["fit_datasets"]}
    ds_overlap = sorted(d for d in set(units["dataset"]) if normalize_name(d) in fit_ds)
    if ds_overlap:
        raise FitEvalOverlapError(f"evaluation datasets were used in fitting: {ds_overlap[:5]}")
    fit_ids = {normalize_name(s[2]) for s in p["fit_series"]}
    sid_overlap = sorted(s for s in set(units["series_id"].astype(str)) if normalize_name(s) in fit_ids)
    if sid_overlap:
        raise FitEvalOverlapError(f"{len(sid_overlap)} evaluation series were used in fitting, "
                                  f"e.g. {sid_overlap[:3]}")

    # exactly the frozen tau grid, for every series at every horizon
    assert_tau_grid(units, p["tau_grid"], "E2 evaluation units")
    assert_tau_grid(test, p["tau_grid"], "E2 evaluation test costs")
    if len(test) != len(units):
        raise ValueError(f"{len(units) - len(test)} evaluation units have no test costs")

    adapter = SELECTOR_KINDS[p["selector_kind"]]
    selector = adapter.build(p["selector_params"], [s[2] for s in p["fit_series"]], p["fit_datasets"])
    if getattr(selector, "needs_hindsight", True):
        raise ValueError("a frozen selector must not need hindsight")
    if selector.name != p["selector_name"]:
        raise ValueError(f"built selector is named {selector.name!r}, frozen name is "
                         f"{p['selector_name']!r}")
    code = selector_code_sha256(adapter, selector)
    if code != p["selector_code_sha256"]:
        raise SelectorCodeError(
            f"the code of selector kind {p['selector_kind']!r} has hash {code[:12]}..., the frozen "
            f"specification records {p['selector_code_sha256'][:12]}...; the selector was changed "
            "after freezing")

    entry = None
    if held:
        entry = dict(time_utc=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                     frozen_sha256=frozen.sha256, selector_code_sha256=code,
                     frozen_path=str(frozen.path),
                     datasets=sorted(str(d) for d in set(units["dataset"])),
                     held_out_names=list(held), n_units=int(len(units)))
        _append_ledger(Path(ledger_path), entry)

    selectors = [AlwaysClassical(), AlwaysTSFM(), StaticOmegaRule(p["static_threshold"]),
                 PinballValidation(), CRPSValidation(), selector, HindsightOracle()]
    frames, timing = [], []
    for s in selectors:
        # copies: a selector that writes into its input cannot change what the others see
        inp = SelectorInput(units=units.copy(), val=val.copy(),
                            test=test.copy() if s.needs_hindsight else None)
        t0 = time.perf_counter()
        decisions = s.select(inp)
        seconds = time.perf_counter() - t0
        timing.append(dict(selector=s.name, seconds=seconds, n_units=len(units),
                           seconds_per_unit=seconds / len(units), measured=TIMING_LABEL))
        frames.append(_attach_costs(decisions, s.name, test, units, p["representatives"], n_val))
    selection = validate_selection_frame(pd.concat(frames, ignore_index=True))

    if frozen.payload != p or _sha256(_canonical(p)) != frozen.sha256:
        raise FrozenFileError("frozen parameters changed during evaluation")
    name = p["selector_name"]
    tag = lambda d: d.assign(frozen_sha256=frozen.sha256)
    # a selector may defer every unit of a fold (the K3 selector below its floor); then there
    # is no decided series to rank, and the worst-decile table is empty rather than an error
    decided = ((selection["selector"] == name) & (selection["status"] == "selected")).any()
    worst = (worst_decile_regret(selection, name) if decided
             else pd.DataFrame(columns=WORST_DECILE_COLUMNS))
    return EvalResult(selection=selection, selector_name=name,
                      unit_regret=tag(unit_regret(selection, name)),
                      regret_summary=tag(regret_summary(selection)),
                      gap_to_validation=tag(gap_to_validation(selection, name)),
                      worst_decile=tag(worst),
                      selector_cost=tag(pd.DataFrame(timing)),
                      frozen_sha256=frozen.sha256, selector_code_sha256=code,
                      frozen_path=str(frozen.path), held_out_evaluated=held, ledger_entry=entry,
                      settings=dict(settings))


# --------------------------------------------------------------------------- command line

def main(argv=None):
    from ..decision.fixture import (FAMILY_OF_MODEL, FIT_ONLY_DATASETS, FIXTURE_SCHEMA,
                                    FIXTURE_STATIC_THRESHOLD, REPRESENTATIVES, make_fixture,
                                    make_fixture_scales)
    from ..decision.pipeline import build_tables
    from . import bootstrap

    ap = argparse.ArgumentParser(description="E2 on the synthetic fixture (interface check only)")
    ap.add_argument("--out", default="artifacts/runs/e2_fixture")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-boot", type=int, default=200, help="0 skips the bootstrap intervals")
    ap.add_argument("--boot-seed", type=int, default=0)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)

    fc, omega, _ = make_fixture(a.seed)
    tables = build_tables(fc, FIXTURE_SCHEMA, REPRESENTATIVES, FAMILY_OF_MODEL, omega,
                          fit_only_datasets=FIT_ONLY_DATASETS, scales=make_fixture_scales(a.seed))
    spec = fit(tables, selector_kind=DOmegaLinearPlaceholder.kind,
               static_threshold=FIXTURE_STATIC_THRESHOLD)
    path = out / f"frozen_{spec.sha256[:12]}.json"      # named by content, so never overwritten
    if not path.exists():
        freeze(spec, path)
    frozen = load_frozen(path)
    r = evaluate(frozen, tables)
    files = r.write(out)
    p = frozen.payload

    print("SYNTHETIC FIXTURE. Interface check, not a research result.")
    print(f"frozen file {path.name}; sha256 {frozen.sha256}")
    print(f"selector code sha256 {r.selector_code_sha256}")
    print(f"fit: kind {p['selector_kind']}, params {p['selector_params']}, static threshold "
          f"{p['static_threshold']}, {len(p['fit_series'])} fit series, {p['n_fit_units']} fit units, "
          f"fit datasets {p['fit_datasets']}")
    s = r.selection
    print(f"evaluation: {len(r.unit_regret)} units, datasets {sorted(s.dataset.unique())}, "
          f"horizons {sorted(int(h) for h in s.horizon.unique())}, {s.tau.nunique()} taus, "
          f"{s.selector.nunique()} selectors")
    d, st = s[s.selector == r.selector_name], s[s.selector == "static_omega"]
    differ = int((d.chosen_family.to_numpy() != st.chosen_family.to_numpy()).sum())
    print(f"units where {r.selector_name} and static_omega choose differently: {differ} of {len(d)}")
    print(f"minimum regret over all selectors {s.regret.min():.6g}; maximum oracle regret "
          f"{s[s.selector == 'hindsight_oracle'].regret.max():.6g}")
    n = r.regret_summary
    print(f"regret summary groups: {len(n)}; regret_over_oracle_cost defined in "
          f"{int(n.regret_over_oracle_cost_defined.sum())}; frac_oracle_value_recovered defined in "
          f"{int(n.frac_oracle_value_recovered_defined.sum())}")
    cols = ["horizon", "tau", "n_selected", "mean_scaled_regret", "regret_over_oracle_cost",
            "frac_oracle_value_recovered"]
    print(f"\n{r.selector_name}, per (horizon, tau):")
    print(n[n.selector == r.selector_name][cols].round(5).to_string(index=False))
    g, w = r.gap_to_validation, r.worst_decile
    print(f"\ngap to pinball validation over {len(g)} cells: {g.gap_to_validation.min():.6f} to "
          f"{g.gap_to_validation.max():.6f}")
    print(f"worst decile: {sorted(w.n_series.unique())} series per cell, "
          f"{sorted(w.n_in_worst.unique())} in the worst decile, values "
          f"{w.worst_decile_regret.min():.5f} to {w.worst_decile_regret.max():.5f}")
    print("selector timing is written to selector_cost.csv; it is a single run and not reproducible")

    if a.n_boot > 0:
        parts = []
        for h, po in tables["per_origin"].groupby("horizon", sort=True):
            cost = bootstrap.fixed_choice_costs(po, s, r.selector_name)
            for label, stat in (("fixed oracle (default)", bootstrap.stat_mean_scaled_regret),
                                ("recomputed oracle (biased upward, not a CI)",
                                 bootstrap.stat_mean_scaled_regret_recomputed_oracle)):
                ci = bootstrap.bootstrap_ci(cost, lambda x, f=stat: f(x, by="tau"), n_boot=a.n_boot,
                                            seed=a.boot_seed, cluster="dataset+series")
                f = ci.to_frame().rename(columns={"label": "tau", "estimate": "mean_scaled_regret"})
                f.insert(0, "horizon", int(h))
                f.insert(1, "statistic", label)
                f["contains_estimate"] = (f.lower <= f.mean_scaled_regret) & (f.mean_scaled_regret <= f.upper)
                parts.append(f)
        ci = pd.concat(parts, ignore_index=True).assign(selector=r.selector_name,
                                                        frozen_sha256=frozen.sha256)
        ci.to_csv(out / "regret_intervals.csv", index=False, lineterminator="\n")
        files.append("regret_intervals.csv")
        print(f"\nmean scaled regret of {r.selector_name} with a NOMINAL 95% percentile block-bootstrap "
              f"interval (approximate, pointwise; not verified coverage). {a.n_boot} resamples, seed "
              f"{a.boot_seed}, datasets and series as clusters, time blocks per series:")
        for label, f in ci.groupby("statistic", sort=True):
            print(f"  {label}: interval contains the estimate in {int(f.contains_estimate.sum())} of "
                  f"{len(f)} cells; block lengths {sorted(f.block_lengths_used.unique())}")
        show = ci[ci.statistic.str.startswith("fixed")]
        print(show[["horizon", "tau", "mean_scaled_regret", "lower", "upper", "n_boot"]]
              .round(6).to_string(index=False))
    print(f"\nwrote {len(files)} files to {out}")


if __name__ == "__main__":
    main()
