#!/usr/bin/env python3
"""Run from a checkout without installing the package."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from decision_aware_tsfm.cache.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
