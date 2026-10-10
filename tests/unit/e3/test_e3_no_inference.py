"""Nothing in the E3 package may import model-inference code."""
import ast
import os
import pathlib
import subprocess
import sys
import unittest

import decision_aware_tsfm.analysis.experiments.e3 as pkg

BANNED = ("torch", "tensorflow", "jax", "gluonts", "chronos", "uni2ts", "timesfm",
          "statsforecast", "transformers", "tsfm_public", "lightning", "sklearn", "statsmodels")
PKG_DIR = pathlib.Path(pkg.__file__).parent
MODULES = sorted(f.stem for f in PKG_DIR.glob("*.py") if f.stem != "__init__")


class NoInferenceImports(unittest.TestCase):
    def test_package_has_the_expected_modules(self):
        self.assertEqual(MODULES, ["design", "forecasters", "generator", "omega_only",
                                   "omega_reconcile", "population", "run", "scoring", "verify"])

    def test_source_has_no_banned_imports(self):
        for f in PKG_DIR.glob("*.py"):
            for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    names = [node.module]
                for n in names:
                    self.assertNotIn(n.split(".")[0], BANNED, f"{f.name} imports {n}")

    def test_importing_every_module_loads_no_inference_library(self):
        mods = [f"decision_aware_tsfm.analysis.experiments.e3.{m}" for m in MODULES]
        code = ("import sys, importlib\n"
                f"for m in {mods!r}: importlib.import_module(m)\n"
                f"bad = [b for b in {BANNED!r} if b in sys.modules]\n"
                "print(','.join(bad))")
        src = str(PKG_DIR.parents[3])
        path = os.pathsep.join([src, os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep)
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             env={**os.environ, "PYTHONPATH": path}, check=True)
        self.assertEqual(out.stdout.strip(), "", out.stdout)

    def test_omega_is_imported_from_the_team_module_not_copied(self):
        for f in PKG_DIR.glob("*.py"):
            src = f.read_text(encoding="utf-8")
            self.assertNotIn("def compute_omega", src, f.name)
            self.assertNotIn("np.fft", src, f.name)

    def test_held_out_datasets_are_not_referenced(self):
        for f in PKG_DIR.glob("*.py"):
            src = f.read_text(encoding="utf-8").lower()
            for name in ("m5", "pjm"):
                self.assertNotIn(name, src, f.name)


if __name__ == "__main__":
    unittest.main()
