#!/usr/bin/env python3
"""Tests for scripts/extend_pool_nested.py (issue #51): nesting, determinism, guards.

Run::

    .venv/bin/python -m unittest tests.test_extend_pool_nested -v
"""
from __future__ import annotations

import importlib.util
import random
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "extend_pool_nested", REPO_ROOT / "scripts" / "extend_pool_nested.py"
)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def _rows(n: int) -> list[dict]:
    hops = ["2hop", "3hop1", "4hop2"]
    return [{"id": f"{hops[i % 3]}__{i:05d}", "question": f"question {i}?"} for i in range(n)]


class ExtendNestedTest(unittest.TestCase):
    def setUp(self) -> None:
        self.input_rows = _rows(300)
        self.base = random.Random(7).sample(self.input_rows, 50)
        self.sizes = [60, 100, 250]

    def test_pools_start_with_base_and_nest(self) -> None:
        pools = mod.extend_nested(self.base, self.input_rows, self.sizes, seed=42)
        self.assertEqual([len(pools[s]) for s in self.sizes], self.sizes)
        for s in self.sizes:
            self.assertEqual(pools[s][:50], self.base)
            self.assertEqual(len({r["id"] for r in pools[s]}), s)
        for a, b in zip(self.sizes, self.sizes[1:]):
            self.assertEqual(pools[b][:a], pools[a])

    def test_deterministic_and_independent_of_input_order(self) -> None:
        shuffled = list(self.input_rows)
        random.Random(1).shuffle(shuffled)
        a = mod.extend_nested(self.base, self.input_rows, self.sizes, seed=42)
        b = mod.extend_nested(self.base, shuffled, self.sizes, seed=42)
        self.assertEqual(a, b)
        c = mod.extend_nested(self.base, self.input_rows, self.sizes, seed=43)
        self.assertNotEqual(a[250], c[250])

    def test_infeasible_size_refused(self) -> None:
        with self.assertRaises(SystemExit):
            mod.extend_nested(self.base, self.input_rows, [301], seed=42)

    def test_base_row_not_from_input_refused(self) -> None:
        base = [dict(self.base[0], question="edited")] + self.base[1:]
        with self.assertRaises(SystemExit):
            mod.extend_nested(base, self.input_rows, self.sizes, seed=42)

    def test_check_pools_passes_and_reports_hop_mix(self) -> None:
        pools = mod.extend_nested(self.base, self.input_rows, self.sizes, seed=42)
        res = mod.check_pools(pools, self.base, {"2hop__dev_1"}, {"not in pool?"})
        self.assertEqual(sum(res[250]["hop_mix"].values()), 250)
        self.assertTrue(all(r["pinned_id_overlap"] == 0 for r in res.values()))

    def test_check_pools_catches_pinned_overlap(self) -> None:
        pools = mod.extend_nested(self.base, self.input_rows, self.sizes, seed=42)
        some = pools[60][55]
        with self.assertRaises(AssertionError):
            mod.check_pools(pools, self.base, {some["id"]}, set())
        with self.assertRaises(AssertionError):
            mod.check_pools(pools, self.base, set(), {mod.norm_question("  " + some["question"].upper())})

    def test_check_pools_catches_broken_nesting(self) -> None:
        pools = mod.extend_nested(self.base, self.input_rows, self.sizes, seed=42)
        pools[100] = pools[100][:59] + pools[250][100:141]  # drops a row of the 60 pool
        with self.assertRaises(AssertionError):
            mod.check_pools(pools, self.base, set(), set())


if __name__ == "__main__":
    unittest.main()
