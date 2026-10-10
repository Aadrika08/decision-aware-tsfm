"""Held-out name checks for K2, on top of analysis/decision/guards.py.

guards.py owns the rule (a name containing "m5" or "pjm", ignoring case, is
held out). This module applies that rule to more places and makes every
match visible. It reads names only; no held-out data is opened anywhere.

Two layers:

  strict  The guards.py token rule, plus the manifest's own `held_out_strict`
          entries. An entry matches when its letters and digits appear in the
          name's letters and digits, so "M5_Retail" also matches "m5-retail"
          and "M5 Retail". A strict match is acted on: a manifest row is
          refused, a results row is dropped. Both are printed by the command.

  fuzzy   The tokens looked for after removing everything that is not a letter
          or a digit, so "M-5", "P.J.M" and "pj_m" match. This rule has false
          positives ("item_5" becomes "item5"), so a fuzzy match is reported
          and never acted on.
"""
from __future__ import annotations

from typing import Iterable

from ..decision.guards import HELD_OUT_TOKENS, is_held_out

STRICT_RULE = ("substring match on the tokens (analysis/decision/guards.py), plus the manifest's "
               "held_out_strict entries compared on letters and digits only")
FUZZY_RULE = ("tokens looked for after removing everything that is not a letter or a digit; "
              "reported, never acted on")


class HeldOutListError(ValueError):
    """The manifest's held_out_strict list cannot be used as given."""


def squash(name) -> str:
    """Lower case, letters and digits only. 'M-5_Retail' -> 'm5retail'."""
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def check_listed(listed) -> tuple[str, ...]:
    """Validate a held_out_strict list: strings with at least one letter or digit."""
    if listed is None:
        return ()
    if isinstance(listed, str) or not isinstance(listed, (list, tuple)):
        raise HeldOutListError(f"held_out_strict must be a list of names, got {type(listed).__name__}")
    for entry in listed:
        if not isinstance(entry, str) or not squash(entry):
            raise HeldOutListError(
                f"held_out_strict entry {entry!r} is not a name with letters or digits; "
                "an empty entry would match every series")
    return tuple(listed)


def strict_reasons(name, listed: Iterable[str] = ()) -> list[str]:
    """Why a name is held out under the strict layer; empty if it is not."""
    reasons = []
    if is_held_out(name):
        low = str(name).replace("\\", "/").lower()
        reasons.extend(f"token {tok!r}" for tok in HELD_OUT_TOKENS if tok in low)
    squashed = squash(name)
    reasons.extend(f"held_out_strict entry {entry!r}" for entry in listed
                   if squash(entry) in squashed)
    return reasons


def fuzzy_reasons(name) -> list[str]:
    """Tokens found once punctuation and spacing are removed (reported only)."""
    squashed = squash(name)
    return [f"token {tok!r} after removing punctuation" for tok in HELD_OUT_TOKENS
            if tok in squashed]


def scan_names(names: Iterable, listed: Iterable[str] = ()) -> tuple[dict, dict]:
    """Classify distinct names: ({name: strict reasons}, {name: fuzzy reasons}).

    A name with a strict match is not repeated in the fuzzy result.
    """
    listed = tuple(listed)
    strict, fuzzy = {}, {}
    for name in dict.fromkeys(str(n) for n in names):
        why = strict_reasons(name, listed)
        if why:
            strict[name] = why
            continue
        why = fuzzy_reasons(name)
        if why:
            fuzzy[name] = why
    return strict, fuzzy
