"""Held-out protection. Same substring rule as Stage0Prep's STRICT_HELD_OUT."""
from __future__ import annotations

from typing import Iterable

HELD_OUT_TOKENS: tuple[str, ...] = ("m5", "pjm")


class HeldOutLeakError(RuntimeError):
    """Held-out data reached a fitting, tuning or development path."""


class FitEvalOverlapError(RuntimeError):
    """A series used to fit a selector is also being used to evaluate it."""


def is_held_out(name: str) -> bool:
    n = str(name).replace("\\", "/").lower()
    return any(tok in n for tok in HELD_OUT_TOKENS)


def assert_no_held_out(names: Iterable[str], context: str) -> None:
    hits = sorted({str(n) for n in names if is_held_out(n)})
    if hits:
        raise HeldOutLeakError(f"{context}: held-out data present {hits[:5]}")
