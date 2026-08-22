#!/usr/bin/env python3
"""Checks for the MetaQA compile-execute wrapper (``scripts/run_metaqa_compile_execute.py``).

Four things are pinned here, all on the fabricated fixtures under ``tests/fixtures/`` and
with no model and no network:

1. **Fixture equivalence.** The wrapper reproduces the standalone
   ``scripts/evaluate_decompositions.py`` outcomes *exactly* — the same counts, the same
   compile-error taxonomy, the same execution-error categories, entry for entry. The wrapper
   must not have become a second, drifting compiler; the whole point is that it calls the
   one that exists. The golden numbers are also the ones
   ``scripts/smoke_test.py::kg_eval`` has always asserted (5 / 4 / 3 / 1 / 1, KG 15/17), so
   a change that moves them turns two things red, not one.
2. **The gold comparison**, hand-computed from the fixture KB and gold files.
3. **The second denominator** — exact match and Jaccard with compile/execute failures
   counted rather than excluded. The committed predictions fixture cannot exercise it (its
   two failing rows have no gold answer), so that case uses hand-written rows whose
   questions *are* in the fixture gold files.
4. **The GRAG labelling.** GRAG is not wired, and every artifact has to say so. A test
   guards it because the failure mode is silent: a run that looked like a GRAG number would
   be a claim nobody made.

Run::

    .venv/bin/python tests/test_metaqa_compile_execute.py
    .venv/bin/python -m unittest discover -s tests
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_config import PATHS_CONFIG_ENV  # noqa: E402

FIXTURES = REPO_ROOT / "tests" / "fixtures"
PREDICTIONS_FIXTURE = FIXTURES / "predictions" / "decomposer_results_metaqa.json"
SMOKE_PATHS_CONFIG = REPO_ROOT / "configs" / "smoke_paths.json"
WRAPPER = REPO_ROOT / "scripts" / "run_metaqa_compile_execute.py"
STANDALONE = REPO_ROOT / "scripts" / "evaluate_decompositions.py"

#: What the fabricated fixtures produce through the compile-and-execute path. 5 rows: 3
#: compile and execute, 1 has a step whose relation cannot be inferred, 1 names a movie that
#: is not in the fixture KB. Identical to the golden values in scripts/smoke_test.py.
GOLDEN_COVERAGE = {
    "total": 5,
    "compiled_ok": 4,
    "executed_ok": 3,
    "compile_fail": 1,
    "exec_fail": 1,
}
GOLDEN_COMPILE_FAIL_REASONS = {"cannot_infer_relation": 1}
GOLDEN_EXEC_FAIL_REASONS = {"entity_not_in_kb": 1}
GOLDEN_KG = {"kg_entities": 15, "kg_triples": 17}


def _run(script: Path, predictions: Path, run_dir: Path, *extra: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env[PATHS_CONFIG_ENV] = str(SMOKE_PATHS_CONFIG)
    return subprocess.run(
        [
            sys.executable,
            str(script),
            "--predictions",
            str(predictions),
            "--run-dir",
            str(run_dir),
            *extra,
        ],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )


class TestWrapperReproducesTheStandaloneScript(unittest.TestCase):
    """The wrapper wraps: same compiler, same executor, same taxonomies, same numbers."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)

        wrapper_dir = root / "wrapper"
        proc = _run(WRAPPER, PREDICTIONS_FIXTURE, wrapper_dir)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        cls.wrapper_metrics: dict[str, Any] = json.loads(
            (wrapper_dir / "metaqa_e2e_metrics.json").read_text(encoding="utf-8")
        )
        cls.wrapper_dir = wrapper_dir

        standalone_dir = root / "standalone"
        proc = _run(STANDALONE, PREDICTIONS_FIXTURE, standalone_dir)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        cls.standalone_metrics: dict[str, Any] = json.loads(
            (standalone_dir / "kg_eval_metrics.json").read_text(encoding="utf-8")
        )
        cls.standalone_dir = standalone_dir

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_the_counts_match_the_standalone_script(self) -> None:
        for key, expected in GOLDEN_COVERAGE.items():
            self.assertEqual(
                self.wrapper_metrics["coverage"][key],
                self.standalone_metrics[key],
                f"{key}: wrapper and standalone script disagree",
            )
            self.assertEqual(self.wrapper_metrics["coverage"][key], expected, key)

    def test_the_rates_match_the_standalone_script(self) -> None:
        for key in ("compiled_ok_rate", "executed_ok_rate"):
            self.assertAlmostEqual(
                self.wrapper_metrics["coverage"][key], self.standalone_metrics[key], places=9
            )
        self.assertAlmostEqual(self.wrapper_metrics["coverage"]["compiled_ok_rate"], 0.8)
        self.assertAlmostEqual(self.wrapper_metrics["coverage"]["executed_ok_rate"], 0.6)

    def test_the_compile_error_taxonomy_is_preserved_entry_for_entry(self) -> None:
        self.assertEqual(
            self.wrapper_metrics["compile_fail_reasons"],
            self.standalone_metrics["compile_fail_reasons"],
        )
        self.assertEqual(
            self.wrapper_metrics["compile_fail_reasons"], GOLDEN_COMPILE_FAIL_REASONS
        )

    def test_the_execution_error_categories_are_preserved_entry_for_entry(self) -> None:
        self.assertEqual(
            self.wrapper_metrics["exec_fail_reasons"],
            self.standalone_metrics["exec_fail_reasons"],
        )
        self.assertEqual(self.wrapper_metrics["exec_fail_reasons"], GOLDEN_EXEC_FAIL_REASONS)

    def test_the_kg_is_the_same_graph(self) -> None:
        for key, expected in GOLDEN_KG.items():
            self.assertEqual(self.wrapper_metrics[key], self.standalone_metrics[key], key)
            self.assertEqual(self.wrapper_metrics[key], expected, key)

    def test_the_per_item_dumps_are_byte_identical(self) -> None:
        """The dumps are what an error analysis reads, so equivalence has to hold at the item
        level and not only at the aggregate. The wrapper writes them under the analysis
        subdirectory (where the gold comparison looks for success.json); the standalone
        script writes them at the top of its run directory."""
        for name in ("success.json", "compile_fail.json", "exec_fail.json"):
            wrapper_payload = json.loads(
                (self.wrapper_dir / "analysis" / name).read_text(encoding="utf-8")
            )
            standalone_payload = json.loads(
                (self.standalone_dir / name).read_text(encoding="utf-8")
            )
            self.assertEqual(wrapper_payload, standalone_payload, name)


class TestGoldComparison(unittest.TestCase):
    """Exact match and Jaccard against the fabricated MetaQA gold, computed by hand.

    From ``tests/fixtures/data_root/metaqa/kb.txt``: Ada Mireles directed Glass Harbor,
    Winter Marbles and The Tin Compass; their languages are Esperanto, Volapuk and Esperanto.
    So the three executable fixture rows produce ``{Ada Mireles}``,
    ``{Glass Harbor, Winter Marbles, The Tin Compass}`` and ``{Esperanto, Volapuk}``, which
    are exactly the gold answer sets in ``answers_{1,2,3}hop.txt`` — 3 exact matches, mean
    Jaccard 1.0. Coverage per hop is 1 answered of 3 / 2 / 2 gold questions.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        run_dir = Path(cls._tmp.name) / "run"
        proc = _run(WRAPPER, PREDICTIONS_FIXTURE, run_dir)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        cls.metrics: dict[str, Any] = json.loads(
            (run_dir / "metaqa_e2e_metrics.json").read_text(encoding="utf-8")
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_all_three_executed_rows_are_exact_matches(self) -> None:
        executed = self.metrics["answer_accuracy_over_executed_items"]
        self.assertEqual(executed["total_in_success"], 3)
        self.assertEqual(executed["total_with_gold"], 3)
        self.assertEqual(executed["total_exact_match"], 3)
        self.assertEqual(executed["overall_pct_exact"], 100.0)
        self.assertEqual(executed["overall_mean_jaccard"], 1.0)

    def test_per_hop_coverage_and_accuracy(self) -> None:
        per_hop = self.metrics["answer_accuracy_over_executed_items"]["per_hop"]
        self.assertEqual(per_hop["1"]["total_gold_questions"], 3)
        self.assertEqual(per_hop["1"]["answered_count"], 1)
        self.assertEqual(per_hop["1"]["coverage_pct"], 33.33)
        for hop in ("2", "3"):
            self.assertEqual(per_hop[hop]["total_gold_questions"], 2)
            self.assertEqual(per_hop[hop]["answered_count"], 1)
            self.assertEqual(per_hop[hop]["coverage_pct"], 50.0)
        for hop in ("1", "2", "3"):
            self.assertEqual(per_hop[hop]["exact_match_count"], 1)
            self.assertEqual(per_hop[hop]["mean_jaccard"], 1.0)

    def test_the_two_failing_fixture_rows_have_no_gold_so_the_denominators_agree(self) -> None:
        """The fixture's compile-fail and exec-fail rows carry invented questions that are
        in no gold file, so nothing is added to the second denominator here. Pinned because
        it is the premise of the class below."""
        over_all = self.metrics["answer_accuracy_over_all_items_with_gold"]
        self.assertEqual(over_all["items_with_gold_not_executed"], 0)
        self.assertEqual(over_all["items_with_gold"], 3)
        self.assertEqual(over_all["pct_exact"], 100.0)
        self.assertEqual(over_all["mean_jaccard"], 1.0)


class TestFailuresCountedRatherThanExcluded(unittest.TestCase):
    """The second denominator, on hand-written rows whose questions ARE in the fixture gold.

    Three fabricated rows, all with a gold answer in the fixture files:

    - ``who directed Glass Harbor`` decomposes and executes to ``{Ada Mireles}`` — an exact
      match;
    - ``what is the genre of Winter Marbles`` is given a step whose relation cannot be
      inferred, so it is a **compile** failure;
    - ``who wrote Paper Lanterns`` is given a step naming a movie absent from the KB, so it
      is an **execution** failure.

    Over executed items with gold that is 1 of 1 = 100%. Over all 3 items with gold it is 1
    of 3 = 33.33%, and mean Jaccard (1.0 + 0 + 0) / 3 = 0.3333. The gap between the two is
    the point: the first number says nothing about how often a decomposition ran at all.
    """

    ROWS = [
        {
            "question": "who directed Glass Harbor",
            "hop_count": 1,
            "decomposition": "1. Who directed Glass Harbor?",
        },
        {
            "question": "what is the genre of Winter Marbles",
            "hop_count": 1,
            "decomposition": "1. Please summarise the plot.",
        },
        {
            "question": "who wrote Paper Lanterns",
            "hop_count": 1,
            "decomposition": "1. Who wrote Nonexistent Movie?",
        },
    ]

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        predictions = root / "failing_rows_with_gold.json"
        predictions.write_text(json.dumps(cls.ROWS, indent=2), encoding="utf-8")
        proc = _run(WRAPPER, predictions, root / "run")
        assert proc.returncode == 0, proc.stdout + proc.stderr
        cls.metrics: dict[str, Any] = json.loads(
            (root / "run" / "metaqa_e2e_metrics.json").read_text(encoding="utf-8")
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_coverage_separates_the_compile_and_execution_failures(self) -> None:
        coverage = self.metrics["coverage"]
        self.assertEqual(coverage["total"], 3)
        self.assertEqual(coverage["compiled_ok"], 2)
        self.assertEqual(coverage["executed_ok"], 1)
        self.assertEqual(coverage["compile_fail"], 1)
        self.assertEqual(coverage["exec_fail"], 1)
        self.assertEqual(self.metrics["compile_fail_reasons"], {"cannot_infer_relation": 1})
        self.assertEqual(self.metrics["exec_fail_reasons"], {"entity_not_in_kb": 1})

    def test_the_executed_only_denominator_hides_the_failures(self) -> None:
        executed = self.metrics["answer_accuracy_over_executed_items"]
        self.assertEqual(executed["total_with_gold"], 1)
        self.assertEqual(executed["total_exact_match"], 1)
        self.assertEqual(executed["overall_pct_exact"], 100.0)
        self.assertEqual(executed["overall_mean_jaccard"], 1.0)

    def test_the_all_items_denominator_counts_them(self) -> None:
        over_all = self.metrics["answer_accuracy_over_all_items_with_gold"]
        self.assertEqual(over_all["items_with_gold"], 3)
        self.assertEqual(over_all["items_with_gold_executed"], 1)
        self.assertEqual(over_all["items_with_gold_not_executed"], 2)
        self.assertEqual(over_all["exact_match_count"], 1)
        self.assertEqual(over_all["pct_exact"], 33.33)
        self.assertEqual(over_all["mean_jaccard"], 0.3333)
        self.assertEqual(over_all["items_with_an_empty_gold_set"], 0)
        self.assertNotIn("mean_jaccard_note", over_all)


class TestGragIsNotWiredAndSaysSo(unittest.TestCase):
    """The labelling guard. ADR 0006 routes MetaQA end-to-end through the supervisor's GRAG;
    this path does not, and no artifact may leave that ambiguous."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        run_dir = Path(cls._tmp.name) / "run"
        proc = _run(WRAPPER, PREDICTIONS_FIXTURE, run_dir)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        cls.metrics: dict[str, Any] = json.loads(
            (run_dir / "metaqa_e2e_metrics.json").read_text(encoding="utf-8")
        )
        cls.snapshot: dict[str, Any] = json.loads(
            (run_dir / "metaqa_e2e_config.json").read_text(encoding="utf-8")
        )
        cls.note = (run_dir / "metaqa_e2e_notes.md").read_text(encoding="utf-8")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_the_metrics_record_that_grag_is_not_wired(self) -> None:
        self.assertFalse(self.metrics["backend"]["grag_wired"])
        self.assertIn("not wired", self.metrics["backend"]["grag_status"])
        self.assertEqual(
            self.metrics["backend"]["label"], "metaqa_compile_execute_direct_kg"
        )

    def test_the_snapshot_and_the_run_note_record_it_too(self) -> None:
        self.assertFalse(self.snapshot["grag_wired"])
        self.assertIn("GRAG is NOT wired", self.note)

    def test_no_model_is_loaded_in_this_path(self) -> None:
        self.assertFalse(self.metrics["backend"]["model_loaded"])

    def test_the_composition_is_recorded(self) -> None:
        """A reader has to be able to see which existing code produced the numbers."""
        composed = self.metrics["composed_from"]
        self.assertIn("evaluate_decompositions.py", composed["compile_execute"])
        self.assertIn("compare_answer_accuracy.py", composed["gold_comparison"])
        self.assertIn("kg.py", composed["kg"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
