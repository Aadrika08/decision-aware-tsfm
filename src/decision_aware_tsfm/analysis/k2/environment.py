"""Record the interpreter and package versions of a run and compare them with the team pins."""
from __future__ import annotations

import platform
import sys
from importlib import metadata

# The team's pinned versions (rules file, 2026-10-04).
TEAM_PINS: dict[str, str] = {
    "python": "3.11.15",
    "numpy": "1.26.4",
    "pandas": "2.2.3",
    "pyarrow": "16.1.0",
    "scipy": "1.11.4",
}


def collect_versions(packages=None) -> dict[str, str | None]:
    """Running versions of the interpreter and the pinned packages (None if not installed).

    Versions are read from the installed package metadata, so nothing is imported.
    """
    packages = [p for p in TEAM_PINS if p != "python"] if packages is None else list(packages)
    out: dict[str, str | None] = {"python": platform.python_version()}
    for name in packages:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def pin_mismatches(versions: dict[str, str | None],
                   pins: dict[str, str] | None = None) -> list[dict]:
    pins = TEAM_PINS if pins is None else pins
    return [{"name": name, "pinned": want, "running": versions.get(name)}
            for name, want in pins.items() if versions.get(name) != want]


def mismatch_warning(mismatches: list[dict]) -> str:
    """Empty string when the running versions equal the pins."""
    if not mismatches:
        return ""
    lines = ["WARNING: the running environment differs from the team pins. The run continues, "
             "but its numbers are not certified to match the recorded ones."]
    for m in mismatches:
        running = m["running"] if m["running"] is not None else "not installed"
        lines.append(f"  {m['name']}: pinned {m['pinned']}, running {running}")
    return "\n".join(lines)


def environment_record() -> dict:
    versions = collect_versions()
    return {
        "python": versions["python"],
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": {k: v for k, v in versions.items() if k != "python"},
        "team_pins": dict(TEAM_PINS),
        "pin_mismatches": pin_mismatches(versions),
    }
