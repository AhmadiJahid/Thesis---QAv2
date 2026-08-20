"""A generation loop may only consume keys its generate function actually returns.

Regression guard for the 2026-08-19 exp-002/exp-003 launch failure: main()'s
per-query dump read gen["raw"] while generate() returns the raw text under
"text", so every real run crashed on its first logged query with
KeyError: 'raw' (run_decomposer.py:1899, runs/exp-002-003-combined.log, all
six runs rc=1). No dry run can reach that branch - a dry run has gen = None,
which short-circuits every gen[...] access - which is why 261 harness checks,
33 smoke stages and six dry-run preflights all passed over the broken line.
This test parses the source instead, so the mismatch is caught with no model.

Two consumer shapes are covered: literal subscripts (gen["text"]), and the
dynamic cost copy ``cost = {k: gen[k] for k in cost}`` - invisible to a
subscript scan, so the keys of main()'s ``cost`` dict literal are checked
against generate()'s return keys directly (review finding I1, 2026-08-19).

Two runners are guarded, because both have the same blind spot: the decomposer
(``generate``) and the MuSiQue answering backend of issue #16
(``components/answerer/run_answerer.py::generate_answer``, whose ``main()``
copies cost the same way). The checks are parameterized by function and
variable name rather than duplicated, so there is one implementation of the
rule; each runner ships its own negative controls, per ADR 0016.

Run::

    .venv/bin/python tests/test_generation_contract.py
    python -m unittest tests.test_generation_contract
"""

import ast
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "components" / "decomposer" / "run_decomposer.py"
ANSWERER = REPO_ROOT / "components" / "answerer" / "run_answerer.py"


def _generate_return_keys(tree: ast.Module, generate_fn: str = "generate") -> set[str]:
    """Keys a consumer may rely on: present in EVERY return dict of ``generate_fn``.

    Today each guarded function has exactly one literal return dict; if an
    early-return path is ever added, a key is only safe when every path returns
    it, so the reduction is the intersection (review nit N2).
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == generate_fn:
            dicts = [
                ret.value
                for ret in ast.walk(node)
                if isinstance(ret, ast.Return) and isinstance(ret.value, ast.Dict)
            ]
            if not dicts:
                raise AssertionError(f"{generate_fn}() has no literal return dict")
            key_sets = [
                {
                    key.value
                    for key in d.keys
                    if isinstance(key, ast.Constant) and isinstance(key.value, str)
                }
                for d in dicts
            ]
            return set.intersection(*key_sets)
    raise AssertionError(f"{generate_fn}() was not found")


def _consumed_gen_keys(tree: ast.Module, var: str = "gen") -> set[str]:
    """Every string key read as ``<var>["<key>"]`` anywhere in the module."""
    keys = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == var
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ):
            keys.add(node.slice.value)
    return keys


def _main_cost_keys(tree: ast.Module, consumer_fn: str = "main") -> set[str]:
    """Keys of the ``cost`` dict literal(s) in the consumer function.

    The consumer copies measurements with ``cost = {k: gen[k] for k in cost}``, a
    variable subscript no literal scan can see - so every key seeded into a
    ``cost`` dict literal must itself be a key the generate function returns.
    """
    keys: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == consumer_fn:
            for stmt in ast.walk(node):
                if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                    targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
                    named_cost = any(
                        isinstance(t, ast.Name) and t.id == "cost" for t in targets
                    )
                    if named_cost and isinstance(stmt.value, ast.Dict):
                        keys.update(
                            key.value
                            for key in stmt.value.keys
                            if isinstance(key, ast.Constant) and isinstance(key.value, str)
                        )
    if not keys:
        raise AssertionError(f"{consumer_fn}()'s cost dict literal was not found")
    return keys


def missing_keys(
    source: str,
    *,
    generate_fn: str = "generate",
    var: str = "gen",
    consumer_fn: str = "main",
) -> set[str]:
    """All keys consumed (literally or via the cost copy) but not returned."""
    tree = ast.parse(source)
    returned = _generate_return_keys(tree, generate_fn)
    consumed = _consumed_gen_keys(tree, var)
    if not consumed:
        raise AssertionError(f"no {var}[...] consumers found - the guard went blind")
    return (consumed | _main_cost_keys(tree, consumer_fn)) - returned


class TestGenerationContract(unittest.TestCase):
    def test_consumed_keys_are_returned_keys(self) -> None:
        missing = missing_keys(RUNNER.read_text(encoding="utf-8"))
        self.assertFalse(
            missing, f"main() consumes keys generate() never returns: {sorted(missing)}"
        )

    def test_the_exp002_regression_is_caught(self) -> None:
        """The exact broken source of 2026-08-19 must fail the full guard."""
        fixed = RUNNER.read_text(encoding="utf-8")
        broken = fixed.replace('+ (gen["text"] if gen else "")', '+ (gen["raw"] if gen else "")')
        self.assertNotEqual(broken, fixed, "the dump line moved; update this control")
        self.assertEqual(missing_keys(broken), {"raw"})

    def test_a_phantom_cost_key_is_caught(self) -> None:
        """A cost field generate() does not return must fail the guard (I1)."""
        fixed = RUNNER.read_text(encoding="utf-8")
        broken = fixed.replace(
            '"prompt_tokens": None,', '"prompt_tokens": None,\n            "gpu_watts": None,'
        )
        self.assertNotEqual(broken, fixed, "the cost initializer moved; update this control")
        self.assertEqual(missing_keys(broken), {"gpu_watts"})


class TestAnswererGenerationContract(unittest.TestCase):
    """Same guard on the MuSiQue answering backend (issue #16).

    Its dry run never generates either - ``gen`` stays None for every step - so
    the per-step dump, the answer it scores and its cost copy are all
    unreachable without a loaded model. ADR 0016: that gets a source-level guard.
    """

    KWARGS = {"generate_fn": "generate_answer", "var": "gen", "consumer_fn": "main"}

    def test_consumed_keys_are_returned_keys(self) -> None:
        missing = missing_keys(ANSWERER.read_text(encoding="utf-8"), **self.KWARGS)
        self.assertFalse(
            missing,
            f"main() consumes keys generate_answer() never returns: {sorted(missing)}",
        )

    def test_a_renamed_answer_key_is_caught(self) -> None:
        """The key the predicted answer is read from is the one that matters most."""
        fixed = ANSWERER.read_text(encoding="utf-8")
        broken = fixed.replace('"answer": answer,', '"final_answer": answer,')
        self.assertNotEqual(broken, fixed, "the return dict moved; update this control")
        self.assertEqual(missing_keys(broken, **self.KWARGS), {"answer"})

    def test_a_phantom_cost_key_is_caught(self) -> None:
        """A cost field generate_answer() does not return must fail the guard."""
        fixed = ANSWERER.read_text(encoding="utf-8")
        broken = fixed.replace(
            '"prompt_tokens": None,', '"prompt_tokens": None,\n                "gpu_watts": None,'
        )
        self.assertNotEqual(broken, fixed, "the cost initializer moved; update this control")
        self.assertEqual(missing_keys(broken, **self.KWARGS), {"gpu_watts"})


if __name__ == "__main__":
    unittest.main()
