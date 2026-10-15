"""Held-out protection. Same tokens as Stage0Prep's STRICT_HELD_OUT.

Stage0Prep matches the tokens against source folder paths. Here they are matched against
names (dataset, config, series id), where "m5" also turns up inside ordinary words: the
manifest's channel suffix "_dim5", or "item5". So "m5" counts only when no letter comes
straight before it. "pjm" counts anywhere. A run-together name such as "retailm5" is
therefore not caught by this rule.
"""
from __future__ import annotations

import re
from typing import Iterable

HELD_OUT_TOKENS: tuple[str, ...] = ("m5", "pjm")
_HELD_OUT = re.compile(r"(?<![a-z])m5|pjm")


class HeldOutLeakError(RuntimeError):
    """Held-out data reached a fitting, tuning or development path."""


class FitEvalOverlapError(RuntimeError):
    """A series used to fit a selector is also being used to evaluate it."""


def held_out_tokens_in(name: str) -> list[str]:
    """The tokens the rule finds in a name, in HELD_OUT_TOKENS order; empty if none."""
    n = str(name).replace("\\", "/").lower()
    found = set(_HELD_OUT.findall(n))
    return [tok for tok in HELD_OUT_TOKENS if tok in found]


def is_held_out(name: str) -> bool:
    return bool(held_out_tokens_in(name))


def assert_no_held_out(names: Iterable[str], context: str) -> None:
    hits = sorted({str(n) for n in names if is_held_out(n)})
    if hits:
        raise HeldOutLeakError(f"{context}: held-out data present {hits[:5]}")
