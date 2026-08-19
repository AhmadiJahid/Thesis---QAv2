"""The generation loop may only consume keys generate() actually returns.

Regression guard for the 2026-08-19 exp-002/exp-003 launch failure: main()'s
per-query dump read gen["raw"] while generate() returns the raw text under
"text", so every real run crashed on its first logged query with
KeyError: 'raw' (run_decomposer.py:1899, runs/exp-002-003-combined.log, all
six runs rc=1). No dry run can reach that branch - a dry run has gen = None,
which short-circuits every gen[...] access - which is why 261 harness checks,
33 smoke stages and six dry-run preflights all passed over the broken line.
This test parses the source instead, so the mismatch is caught with no model.
"""

import ast
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "components" / "decomposer" / "run_decomposer.py"


def _generate_return_keys(tree: ast.Module) -> set[str]:
    """Keys of the dict literal that generate() returns."""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "generate":
            for ret in ast.walk(node):
                if isinstance(ret, ast.Return) and isinstance(ret.value, ast.Dict):
                    return {
                        key.value
                        for key in ret.value.keys
                        if isinstance(key, ast.Constant) and isinstance(key.value, str)
                    }
    raise AssertionError("generate()'s literal return dict was not found")


def _consumed_gen_keys(tree: ast.Module) -> set[str]:
    """Every string key read as gen["<key>"] anywhere in the module."""
    keys = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == "gen"
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ):
            keys.add(node.slice.value)
    return keys


class TestGenerationContract(unittest.TestCase):
    def test_consumed_keys_are_returned_keys(self) -> None:
        tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
        returned = _generate_return_keys(tree)
        consumed = _consumed_gen_keys(tree)
        self.assertTrue(consumed, "no gen[...] consumers found - the guard went blind")
        missing = consumed - returned
        self.assertFalse(
            missing,
            f"main() consumes keys generate() never returns: {sorted(missing)} "
            f"(generate() returns {sorted(returned)})",
        )

    def test_the_exp002_regression_shape_is_caught(self) -> None:
        """The exact broken line of 2026-08-19 must fail this guard."""
        probe = ast.parse('def f(gen):\n    return gen["raw"]\n')
        self.assertEqual(_consumed_gen_keys(probe), {"raw"})


if __name__ == "__main__":
    unittest.main()
