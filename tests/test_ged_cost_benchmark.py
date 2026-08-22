#!/usr/bin/env python3
"""Checks for ``scripts/ged_cost_benchmark.py``, the tool behind ADR 0026's GED cost table.

The benchmark prints *timings*, which are machine-dependent and therefore not assertable.
What is assertable — and what the PR #44 review actually asked for — is that each shape is
**defined**, not described: a shape's node count, edge count and step texts are fixed by the
code, so the table can be re-derived on any machine. Those are pinned here, together with a
tiny end-to-end run of the CLI (``--max-node-count 8``) so the full path is smoke-testable
without paying for the expensive cells.

Run::

    .venv/bin/python -m unittest tests.test_ged_cost_benchmark -v
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = REPO_ROOT / "scripts" / "ged_cost_benchmark.py"
CONFIG = REPO_ROOT / "configs" / "ged_cost_benchmark.json"


def _import(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BENCH = _import(BENCHMARK, "ged_cost_benchmark")


class TestShapesAreDefinedNotDescribed(unittest.TestCase):
    """Each shape's graph is fixed by the code, so a printed row can be reproduced."""

    GOLD_STEPS = [
        "Who published Quiet Ledger?",
        "Who founded [#1]?",
        "Which college did [#2] attend?",
        "In what year did [#3] open?",
    ]

    def graph(self, shape: str, nodes: int):
        return BENCH._graph_of(BENCH.SHAPES[shape](nodes, self.GOLD_STEPS))

    def test_repeated_step_text_is_n_identical_labels_and_no_edges(self) -> None:
        steps = BENCH.SHAPES["repeated_step_text"](16, self.GOLD_STEPS)
        self.assertEqual(steps, [BENCH._REPEATED_STEP_TEXT] * 16)
        g = self.graph("repeated_step_text", 16)
        self.assertEqual((g.number_of_nodes(), g.number_of_edges()), (16, 0))
        self.assertEqual({g.nodes[n]["label"] for n in g.nodes}, {BENCH._REPEATED_STEP_TEXT})

    def test_gold_step_texts_repeated_cycles_the_gold_verbatim(self) -> None:
        steps = BENCH.SHAPES["gold_step_texts_repeated"](6, self.GOLD_STEPS)
        self.assertEqual(steps, self.GOLD_STEPS + self.GOLD_STEPS[:2])
        # The gold's own references come with the texts: 4 of every 6 steps carry one.
        g = self.graph("gold_step_texts_repeated", 6)
        self.assertEqual((g.number_of_nodes(), g.number_of_edges()), (6, 4))

    def test_chain_shaped_is_a_path(self) -> None:
        steps = BENCH.SHAPES["chain_shaped"](4, self.GOLD_STEPS)
        self.assertEqual(
            steps,
            [
                BENCH._CHAIN_FIRST_STEP,
                "Who founded [#1]?",
                "Who founded [#2]?",
                "Who founded [#3]?",
            ],
        )
        g = self.graph("chain_shaped", 12)
        self.assertEqual((g.number_of_nodes(), g.number_of_edges()), (12, 11))

    def test_all_pairs_referencing_is_the_densest_graph_of_its_size(self) -> None:
        steps = BENCH.SHAPES["all_pairs_referencing"](3, self.GOLD_STEPS)
        self.assertEqual(
            steps,
            [
                BENCH._CHAIN_FIRST_STEP,
                "Which of [#1] is the earliest?",
                "Which of [#1] [#2] is the earliest?",
            ],
        )
        # The edge counts ADR 0026 quotes for this shape.
        for nodes, edges in ((20, 190), (30, 435)):
            with self.subTest(nodes=nodes):
                g = self.graph("all_pairs_referencing", nodes)
                self.assertEqual((g.number_of_nodes(), g.number_of_edges()), (nodes, edges))

    def test_the_committed_config_names_only_known_shapes(self) -> None:
        cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
        for cell in cfg["cost_table_cells"]:
            self.assertIn(cell["shape"], BENCH.SHAPES)
        for cell in cfg["bound_vs_optimizer_cells"]:
            self.assertIn(cell["shape"], BENCH.SHAPES)

    def test_the_bound_is_never_below_the_optimizer_value(self) -> None:
        """The reported fallback is an UPPER bound, which is the whole claim ADR 0026 makes.

        Checked across **every** shape rather than one cell (PR #45 review, the weaker note
        under nit 2), at sizes small enough that the optimizer terminates in milliseconds. A
        bound that came in *under* the optimizer's value would mean the fallback is not a
        valid edit path, and the ADR's "upper bound" wording would be wrong.
        """
        gold_graph = BENCH._graph_of(self.GOLD_STEPS)
        for shape in sorted(BENCH.SHAPES):
            for nodes in (3, 6, 9):
                with self.subTest(shape=shape, nodes=nodes):
                    pred_graph = self.graph(shape, nodes)
                    measured = BENCH._time_optimizer(pred_graph, gold_graph, 60.0)
                    bound = BENCH._bound(pred_graph, gold_graph)
                    self.assertGreaterEqual(bound + 1e-9, measured["ged"])

    def test_the_discriminating_pair_differs_only_in_the_reference(self) -> None:
        """PR #45 review, Important: the 2x2 that separates ties, gold overlap and edges.

        The nonsense pair must share vocabulary and token count and differ in exactly one
        token — the reference — or a cost difference between their rows would be
        attributable to something else. The gold-derived pair must use gold labels: one
        without a reference (step 1 of a MuSiQue gold never has one) and one with.
        """
        no_ref = BENCH.SHAPES["nonsense_text_repeated_no_reference"](16, self.GOLD_STEPS)
        with_ref = BENCH.SHAPES["nonsense_text_repeated_with_reference"](16, self.GOLD_STEPS)
        self.assertEqual(no_ref, [BENCH._NONSENSE_STEP_TEXT] * 16)
        self.assertEqual(with_ref, [BENCH._NONSENSE_STEP_TEXT_WITH_REFERENCE] * 16)
        self.assertEqual(len(no_ref[0].split()), len(with_ref[0].split()))
        self.assertEqual(no_ref[0].split()[:-1], with_ref[0].split()[:-1])
        # No WORD of either nonsense label appears in the gold, so a cost difference
        # against the gold-derived pair cannot be read as similarity to a gold label. The
        # reference token itself is necessarily shared — carrying one is the toggled factor.
        def words(text: str) -> set[str]:
            return {t.lower() for t in text.split() if "#" not in t}

        gold_words = set().union(*(words(step) for step in self.GOLD_STEPS))
        for label in (no_ref[0], with_ref[0]):
            self.assertEqual(gold_words & words(label), set())

        gold_no_ref = BENCH.SHAPES["gold_step_text_repeated_no_reference"](5, self.GOLD_STEPS)
        gold_with_ref = BENCH.SHAPES["gold_step_text_repeated_with_reference"](
            5, self.GOLD_STEPS
        )
        self.assertEqual(gold_no_ref, [self.GOLD_STEPS[0]] * 5)
        self.assertEqual(gold_with_ref, [self.GOLD_STEPS[1]] * 5)
        self.assertNotIn("#", gold_no_ref[0])
        self.assertIn("#1", gold_with_ref[0])

        # Edges are the toggled factor: 0 without a reference, one per step with one (step
        # 1's own '#1' is a self-loop, which networkx counts).
        for shape, edges in (
            ("nonsense_text_repeated_no_reference", 0),
            ("gold_step_text_repeated_no_reference", 0),
            ("nonsense_text_repeated_with_reference", 16),
            ("gold_step_text_repeated_with_reference", 16),
        ):
            with self.subTest(shape=shape):
                g = self.graph(shape, 16)
                self.assertEqual((g.number_of_nodes(), g.number_of_edges()), (16, edges))

    def test_a_gold_with_no_reference_is_refused_not_silently_reshaped(self) -> None:
        """The gold-derived with-reference shape has no fallback."""
        with self.assertRaises(SystemExit):
            BENCH.SHAPES["gold_step_text_repeated_with_reference"](
                4, ["One step, no reference at all?"]
            )


class TestBenchmarkCli(unittest.TestCase):
    def test_a_tiny_run_prints_a_table(self) -> None:
        """`--max-node-count 8`: the full path, in about a second."""
        with tempfile.TemporaryDirectory(prefix="qav2_ged_bench_") as tmp:
            out = Path(tmp) / "bench.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(BENCHMARK),
                    "--max-node-count",
                    "8",
                    "--json",
                    str(out),
                ],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, f"{proc.stdout}\n{proc.stderr}")
            self.assertIn("vs 4-hop gold", proc.stdout)
            self.assertIn("`repeated_step_text`", proc.stdout)
            # The commit is printed, because that is what the ADR records beside the table.
            self.assertIn("- commit: `", proc.stdout)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertTrue(payload["cost_table"])
            for row in payload["cost_table"]:
                self.assertLessEqual(row["nodes"], 8)
                for timing in row["timings"].values():
                    self.assertGreaterEqual(timing["seconds"], 0.0)


if __name__ == "__main__":
    unittest.main()
