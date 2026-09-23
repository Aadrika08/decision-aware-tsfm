"""No metric or selector path may import model-inference code."""
import ast
import os
import pathlib
import subprocess
import sys

import decision_aware_tsfm.analysis.decision as pkg

BANNED = ("torch", "tensorflow", "jax", "gluonts", "chronos", "uni2ts", "timesfm",
          "statsforecast", "transformers", "tsfm_public", "lightning")
PKG_DIR = pathlib.Path(pkg.__file__).parent


def test_source_has_no_banned_imports():
    for f in PKG_DIR.glob("*.py"):
        for node in ast.walk(ast.parse(f.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module]
            for n in names:
                assert n.split(".")[0] not in BANNED, f"{f.name} imports {n}"


def test_importing_every_module_loads_no_inference_library():
    mods = [f"decision_aware_tsfm.analysis.decision.{f.stem}" for f in PKG_DIR.glob("*.py")
            if f.stem != "__init__"]
    code = ("import sys, importlib\n"
            f"for m in {mods!r}: importlib.import_module(m)\n"
            f"bad = [b for b in {BANNED!r} if b in sys.modules]\n"
            "print(','.join(bad))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={**os.environ, "PYTHONPATH": str(PKG_DIR.parents[2])}, check=True)
    assert out.stdout.strip() == "", out.stdout
