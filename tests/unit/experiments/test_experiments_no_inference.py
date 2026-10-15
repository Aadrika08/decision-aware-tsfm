"""No E1, E2, bootstrap or coverage-simulation path may import model-inference code.

Same check as tests/unit/decision/test_decision_no_inference.py, for the
modules directly under analysis/experiments/. The e3/ sub-package has its own
copy of this test in tests/unit/e3/.
"""
import ast
import os
import pathlib
import subprocess
import sys
import unittest

import decision_aware_tsfm.analysis.experiments as pkg

BANNED = ("torch", "tensorflow", "jax", "gluonts", "chronos", "uni2ts", "timesfm",
          "statsforecast", "transformers", "tsfm_public", "lightning")
# not on the team's pinned list either; the brief forbids them in this area
NOT_PINNED = ("sklearn", "statsmodels", "matplotlib")
PKG_DIR = pathlib.Path(pkg.__file__).parent
MODULES = sorted(f for f in PKG_DIR.glob("*.py"))


class NoInferenceImports(unittest.TestCase):
    def test_the_expected_modules_are_scanned(self):
        self.assertLessEqual({"bootstrap.py", "e1.py", "e2.py", "coverage_sim.py"}, {f.name for f in MODULES})

    def test_source_has_no_banned_imports(self):
        for f in MODULES:
            for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    names = [node.module]
                for n in names:
                    self.assertNotIn(n.split(".")[0], BANNED + NOT_PINNED, f"{f.name} imports {n}")

    def test_the_scan_would_see_a_banned_import(self):
        tree = ast.parse("import numpy\nfrom torch import nn\nimport chronos.pipeline as c\n")
        found = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found += [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                found.append(node.module.split(".")[0])
        self.assertEqual(sorted(set(found) & set(BANNED)), ["chronos", "torch"])

    def test_importing_every_module_loads_no_inference_library(self):
        mods = [f"decision_aware_tsfm.analysis.experiments.{f.stem}" for f in MODULES
                if f.stem != "__init__"]
        code = ("import sys, importlib\n"
                f"for m in {mods!r}: importlib.import_module(m)\n"
                f"bad = [b for b in {BANNED + NOT_PINNED!r} if b in sys.modules]\n"
                "print(','.join(bad))")
        src = str(PKG_DIR.parents[2])
        path = os.pathsep.join([src, os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep)
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             env={**os.environ, "PYTHONPATH": path}, check=True)
        self.assertEqual(out.stdout.strip(), "", out.stdout)


if __name__ == "__main__":
    unittest.main()
