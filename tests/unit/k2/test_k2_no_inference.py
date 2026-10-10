"""K2 uses published scores only: no model-inference library may be imported."""
import ast
import pathlib
import unittest

import decision_aware_tsfm.analysis.k2 as pkg

BANNED = ("torch", "tensorflow", "jax", "gluonts", "chronos", "uni2ts", "timesfm",
          "statsforecast", "transformers", "tsfm_public", "lightning", "sklearn", "statsmodels")
OPTIONAL_INSIDE_FUNCTIONS = ("matplotlib", "requests")
PKG_DIR = pathlib.Path(pkg.__file__).parent


def imports(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name.split(".")[0], node
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module.split(".")[0], node


class NoInferenceImports(unittest.TestCase):
    def test_source_has_no_banned_imports(self):
        for f in sorted(PKG_DIR.glob("*.py")):
            for name, _ in imports(ast.parse(f.read_text(encoding="utf-8"))):
                self.assertNotIn(name, BANNED, f"{f.name} imports {name}")

    def test_optional_packages_are_never_imported_at_module_level(self):
        for f in sorted(PKG_DIR.glob("*.py")):
            tree = ast.parse(f.read_text(encoding="utf-8"))
            top_level = {id(n) for n in tree.body}
            for name, node in imports(tree):
                if name in OPTIONAL_INSIDE_FUNCTIONS:
                    self.assertNotIn(id(node), top_level, f"{f.name} imports {name} at module level")


if __name__ == "__main__":
    unittest.main()
