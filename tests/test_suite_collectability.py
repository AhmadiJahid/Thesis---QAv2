#!/usr/bin/env python3
"""The test suites stay collectable by pytest. No GPU, no weights, no subprocess.

Why this file exists. Two suites here are *script-style*: they are plain scripts whose
harness functions take arguments (a loaded config) and whose ``main()`` calls them in order.
pytest also imports every ``tests/test_*.py``, and it reads a parameter on a module-level
``test_*`` function as a request for a **fixture** — so one such function turns
``python -m pytest tests/`` into a collection error that has nothing to do with the code under
test. That is exactly what happened (PR #21 review, M-3: five errors), and renaming those
functions ``check_*`` fixed it.

This check is the regression guard for the *next* one. It reproduces the rule with ``ast``
rather than shelling out to pytest, for two measured reasons:

1. ``pytest --collect-only -q`` does **not** catch this shape. Measured on 2026-08-19 against
   a probe module containing exactly ``def test_regression(cfg): pass``:
   ``pytest -q --collect-only`` → *1 test collected*, exit 0; ``pytest -q`` → *1 error*,
   exit 1. The missing fixture is raised when the test is **set up**, not when it is
   collected, so a collect-only gate would pass a suite that a real run errors on.
2. pytest is not installed in the project venv (``.venv``) that runs the other suites, so a
   pytest-shelling check would silently depend on which interpreter invoked it. An ast check
   runs everywhere, including under plain ``unittest``.

Run::

    .venv/bin/python tests/test_suite_collectability.py
    .venv/bin/python -m unittest discover -s tests
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent

#: pytest collects module-level functions whose name starts with this.
PYTEST_FUNCTION_PREFIX = "test_"


def required_parameters(func: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Parameter names pytest would have to supply as fixtures.

    Positional and keyword-only parameters **without** a default. ``*args``/``**kwargs`` and
    defaulted parameters do not make pytest look for a fixture, so they are not reported.
    """
    args = func.args
    positional = args.posonlyargs + args.args
    without_default = positional[: len(positional) - len(args.defaults)]
    kwonly = [
        arg
        for arg, default in zip(args.kwonlyargs, args.kw_defaults)
        if default is None
    ]
    return [arg.arg for arg in without_default + kwonly]


def fixture_requesting_test_functions(path: Path) -> list[tuple[str, list[str]]]:
    """``(function name, required parameters)`` for every collection error in one module."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders: list[tuple[str, list[str]]] = []
    for node in tree.body:  # module level only: methods on a TestCase take self legitimately
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.name.startswith(PYTEST_FUNCTION_PREFIX):
            continue
        params = required_parameters(node)
        if params:
            offenders.append((node.name, params))
    return offenders


class TestEveryTestModuleIsCollectable(unittest.TestCase):
    """No module-level ``test_*`` function takes an argument pytest would call a fixture."""

    def test_the_suites_are_found(self) -> None:
        """A vacuous pass is the one failure mode this check cannot report on its own."""
        modules = sorted(TESTS_DIR.glob("test_*.py"))
        self.assertGreaterEqual(len(modules), 4, f"only found {[m.name for m in modules]}")
        self.assertIn("test_decomposer_conditions.py", [m.name for m in modules])

    def test_no_test_function_requests_a_fixture(self) -> None:
        offenders = {
            path.name: found
            for path in sorted(TESTS_DIR.glob("test_*.py"))
            if (found := fixture_requesting_test_functions(path))
        }
        self.assertEqual(
            offenders,
            {},
            "these module-level test_* functions take arguments, so pytest will treat them as "
            "fixture requests and report collection errors (PR #21 review, M-3): "
            f"{offenders}. A harness function that takes a config belongs under a check_* "
            "name, called from the module's own main().",
        )

    def test_the_rule_catches_the_shape_it_is_meant_to(self) -> None:
        """Hand-computed: the M-3 shape is caught, the legitimate shapes are not."""
        source = (
            "def test_broken(cfg):\n    pass\n"
            "def test_also_broken(a, b=1, *args, **kw):\n    pass\n"
            "def test_kwonly_broken(*, cfg):\n    pass\n"
            "def test_fine():\n    pass\n"
            "def test_defaulted_is_fine(cfg=None):\n    pass\n"
            "def check_helper(cfg):\n    pass\n"
            "class T:\n    def test_method(self):\n        pass\n"
        )
        path = TESTS_DIR / "_collectability_probe.py"
        path.write_text(source, encoding="utf-8")
        try:
            found = fixture_requesting_test_functions(path)
        finally:
            path.unlink()
        self.assertEqual(
            found,
            [
                ("test_broken", ["cfg"]),
                ("test_also_broken", ["a"]),
                ("test_kwonly_broken", ["cfg"]),
            ],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
