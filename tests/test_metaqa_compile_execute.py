#!/usr/bin/env python3
"""Checks for the MetaQA compile-execute wrapper (``scripts/run_metaqa_compile_execute.py``).

Everything here runs on fabricated fixtures (or hand-written rows and a throwaway data root
built in the test), with no model and no network. Pinned:

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
5. **A measured zero is not a null** (PR #42 review, finding 1) — a run where every item
   fails reports 0.0%, and only a run with no gold at all reports null.
6. **One rule for every item** (finding 2) — an unexecuted item with an empty gold set is
   scored by the block's stated definition on *both* metrics, with the surprise named rather
   than nulled.
7. **Evaluation-set identity and upstream provenance** (finding 3) — the refusal, its
   recorded opt-out, the question-set fingerprint, and the decomposer run's commit/config.
8. **No model is loaded here** (nit 3) — an ADR 0016 source-level guard with negative
   controls, because ``backend.model_loaded: False`` is otherwise an unguarded constant.

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


#: Every fixture and hand-written input here contains at least one question that is in no
#: gold file, which the eval-set assertion refuses for an experiment arm. A fixture run is
#: not an arm, so the recorded opt-out is passed — the same convention the answerer's fixture
#: runs use. ``TestEvaluationSetIdentity`` is where the assertion itself is exercised.
UNPINNED = "--allow-unpinned-eval-set"


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
        proc = _run(WRAPPER, PREDICTIONS_FIXTURE, wrapper_dir, UNPINNED)
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
        proc = _run(WRAPPER, PREDICTIONS_FIXTURE, run_dir, UNPINNED)
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
        proc = _run(WRAPPER, predictions, root / "run", UNPINNED)
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
        proc = _run(WRAPPER, PREDICTIONS_FIXTURE, run_dir, UNPINNED)
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


class TestEveryItemFailing(unittest.TestCase):
    """PR #42 review, finding 1: a run where nothing executes must report 0.0, not null.

    Two fabricated rows, both with a gold answer in the fixture files, both given a step
    whose relation cannot be inferred — so both are **compile** failures and nothing reaches
    the executor. The executed-only block is genuinely unmeasured (0 items, so its mean is
    null by ``compare_answer_accuracy``'s own rule), but the all-items block has 2 items with
    gold and measured them both: exact 0 of 2 = **0.0%**, mean Jaccard (0 + 0) / 2 = **0.0**.
    Reporting null there would say "unmeasured" about a measured floor — the worst kind of
    wrong number, because it hides a total failure as a missing one.
    """

    ROWS = [
        {
            "question": "what is the genre of Winter Marbles",
            "hop_count": 1,
            "decomposition": "1. Please summarise the plot.",
        },
        {
            "question": "who wrote Paper Lanterns",
            "hop_count": 1,
            "decomposition": "1. Please summarise the plot.",
        },
    ]

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        predictions = root / "all_failing.json"
        predictions.write_text(json.dumps(cls.ROWS, indent=2), encoding="utf-8")
        proc = _run(WRAPPER, predictions, root / "run")
        assert proc.returncode == 0, proc.stdout + proc.stderr
        cls.metrics: dict[str, Any] = json.loads(
            (root / "run" / "metaqa_e2e_metrics.json").read_text(encoding="utf-8")
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_nothing_executed(self) -> None:
        coverage = self.metrics["coverage"]
        self.assertEqual(coverage["total"], 2)
        self.assertEqual(coverage["compiled_ok"], 0)
        self.assertEqual(coverage["executed_ok"], 0)
        self.assertEqual(coverage["compile_fail"], 2)
        self.assertEqual(self.metrics["compile_fail_reasons"], {"cannot_infer_relation": 2})

    def test_the_executed_only_block_is_null_because_it_is_genuinely_unmeasured(self) -> None:
        executed = self.metrics["answer_accuracy_over_executed_items"]
        self.assertEqual(executed["total_with_gold"], 0)
        self.assertIsNone(executed["overall_pct_exact"])
        self.assertIsNone(executed["overall_mean_jaccard"])
        self.assertEqual(executed["overall_jaccard_sum"], 0.0)

    def test_the_all_items_block_reports_a_measured_zero(self) -> None:
        over_all = self.metrics["answer_accuracy_over_all_items_with_gold"]
        self.assertEqual(over_all["items_with_gold"], 2)
        self.assertEqual(over_all["items_with_gold_executed"], 0)
        self.assertEqual(over_all["items_with_gold_not_executed"], 2)
        self.assertEqual(over_all["exact_match_count"], 0)
        self.assertEqual(over_all["pct_exact"], 0.0)
        self.assertEqual(over_all["mean_jaccard"], 0.0)
        self.assertEqual(over_all["jaccard_sum"], 0.0)
        self.assertIsNotNone(over_all["pct_exact"], "a measured 0 must not be reported as null")

    def test_null_is_still_reported_when_no_item_has_gold_at_all(self) -> None:
        """The one case that IS unmeasured: nothing to compare against. The fixture's two
        failing rows carry questions in no gold file, so this run measures nothing and both
        numbers are null rather than a fabricated 0."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predictions = root / "no_gold.json"
            predictions.write_text(
                json.dumps(
                    [
                        {
                            "question": "a step whose relation cannot be inferred",
                            "hop_count": 1,
                            "decomposition": "1. Please summarise the plot.",
                        }
                    ],
                    indent=2,
                ),
                encoding="utf-8",
            )
            proc = _run(WRAPPER, predictions, root / "run", UNPINNED)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            metrics = json.loads(
                (root / "run" / "metaqa_e2e_metrics.json").read_text(encoding="utf-8")
            )
        over_all = metrics["answer_accuracy_over_all_items_with_gold"]
        self.assertEqual(over_all["items_with_gold"], 0)
        self.assertIsNone(over_all["pct_exact"])
        self.assertIsNone(over_all["mean_jaccard"])


class TestEmptyGoldSetIsScoredByTheStatedDefinition(unittest.TestCase):
    """PR #42 review, finding 2: the empty-gold guard must not be asymmetric.

    ``jaccard(empty, empty)`` is 1.0, so by this block's own definition — exact match iff
    Jaccard reaches the threshold — an unexecuted item with an empty gold set **is** an exact
    match. The first version guarded only the mean, so a run with 1 executed exact match and
    1 failed row with empty gold reported 50.0% next to a definition that said 100.0%. The
    fix applies one rule to every item; the surprise is named in ``empty_gold_set_note``
    instead of being papered over with a null.

    This needs a gold file with an empty answer set, which the committed fixtures do not
    have (MetaQA gold answers are never empty), so the test builds a throwaway data root and
    a throwaway paths config pointing at it. Hand-computed: exact 2 of 2 = 100.0%, mean
    Jaccard (1.0 + 1.0) / 2 = 1.0.
    """

    ROWS = [
        {
            "question": "who directed Glass Harbor",
            "hop_count": 1,
            "decomposition": "1. Who directed Glass Harbor?",
        },
        {
            "question": "a question whose gold answers are blank",
            "hop_count": 1,
            "decomposition": "1. Please summarise the plot.",
        },
    ]

    @classmethod
    def setUpClass(cls) -> None:
        cls._data = tempfile.TemporaryDirectory()
        # The paths-config override must live inside the repo (run_config.load_paths binds it
        # there deliberately), so it goes in a throwaway directory under configs/.
        cls._cfg = tempfile.TemporaryDirectory(dir=REPO_ROOT / "configs")
        data_root = Path(cls._data.name) / "data_root"
        metaqa = data_root / "metaqa"
        metaqa.mkdir(parents=True)
        (metaqa / "kb.txt").write_text(
            (FIXTURES / "data_root" / "metaqa" / "kb.txt").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        (metaqa / "refined_1hop.txt").write_text(
            "who directed Glass Harbor\na question whose gold answers are blank\n",
            encoding="utf-8",
        )
        # Line 2 is a separator with nothing around it: every part is empty, so the gold set
        # for that question is empty while the line-for-line alignment still holds.
        (metaqa / "answers_1hop.txt").write_text("Ada Mireles\n|\n", encoding="utf-8")

        paths_config = Path(cls._cfg.name) / "paths_probe.json"
        paths_config.write_text(
            json.dumps(
                {
                    "_note": "throwaway paths config for tests/test_metaqa_compile_execute.py",
                    "data_root": str(data_root),
                    "runs_root": str(Path(cls._data.name) / "runs"),
                    "datasets": {
                        "metaqa_kb": "metaqa/kb.txt",
                        "metaqa_questions_template": "metaqa/refined_{hop}hop.txt",
                        "metaqa_answers_template": "metaqa/answers_{hop}hop.txt",
                    },
                    "repo": {},
                }
            ),
            encoding="utf-8",
        )

        predictions = Path(cls._data.name) / "empty_gold.json"
        predictions.write_text(json.dumps(cls.ROWS, indent=2), encoding="utf-8")
        run_dir = Path(cls._data.name) / "run"

        env = os.environ.copy()
        env[PATHS_CONFIG_ENV] = str(paths_config)
        proc = subprocess.run(
            [
                sys.executable,
                str(WRAPPER),
                "--predictions",
                str(predictions),
                "--run-dir",
                str(run_dir),
            ],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        cls.metrics: dict[str, Any] = json.loads(
            (run_dir / "metaqa_e2e_metrics.json").read_text(encoding="utf-8")
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._data.cleanup()
        cls._cfg.cleanup()

    def test_the_empty_gold_row_is_counted(self) -> None:
        over_all = self.metrics["answer_accuracy_over_all_items_with_gold"]
        self.assertEqual(over_all["items_with_gold"], 2)
        self.assertEqual(over_all["items_with_gold_not_executed"], 1)
        self.assertEqual(over_all["items_with_an_empty_gold_set"], 1)

    def test_exact_match_and_jaccard_agree_with_the_stated_definition(self) -> None:
        over_all = self.metrics["answer_accuracy_over_all_items_with_gold"]
        self.assertEqual(over_all["exact_match_count_executed"], 1)
        self.assertEqual(over_all["exact_match_count_not_executed"], 1)
        self.assertEqual(over_all["exact_match_count"], 2)
        self.assertEqual(over_all["pct_exact"], 100.0)
        self.assertEqual(over_all["mean_jaccard"], 1.0)

    def test_the_surprise_is_named_rather_than_hidden(self) -> None:
        over_all = self.metrics["answer_accuracy_over_all_items_with_gold"]
        self.assertIn("empty_gold_set_note", over_all)
        self.assertIn("empty-vs-empty", over_all["empty_gold_set_note"])
        # The old asymmetric behaviour nulled the mean; it must not come back.
        self.assertIsNotNone(over_all["mean_jaccard"])
        self.assertNotIn("mean_jaccard_note", over_all)

    def test_the_asymmetry_with_musique_normalization_is_stated(self) -> None:
        """Finding 4: the element-matching rule here is strip-only and case-sensitive, unlike
        MuSiQue EM's SQuAD normalization. Recorded, not silently changed."""
        definitions = self.metrics["metric_definitions"]
        self.assertIn("answer_normalization", definitions)
        self.assertIn("CASE-SENSITIVE", definitions["answer_normalization"])
        self.assertIn("normalize_answer", definitions["answer_normalization"])
        over_all = self.metrics["answer_accuracy_over_all_items_with_gold"]
        self.assertIn("asymmetry", over_all["definition"])


class TestEvaluationSetIdentity(unittest.TestCase):
    """PR #42 review, finding 3: which evaluation set was scored has to be in the artifact.

    A row whose question is in no gold file is absent from every accuracy denominator, so
    the run silently reports over a smaller set than it read. The MuSiQue side refuses that
    (``run_decomposer.assert_pinned_eval_set``) with a recorded opt-out; this mirrors it.
    """

    def test_a_run_with_ungolded_questions_is_refused_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            proc = _run(WRAPPER, PREDICTIONS_FIXTURE, Path(tmp) / "run")
        self.assertNotEqual(proc.returncode, 0)
        output = proc.stdout + proc.stderr
        self.assertIn("no MetaQA gold answer", output)
        self.assertIn("--allow-unpinned-eval-set", output)

    def test_the_opt_out_is_recorded_in_the_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            proc = _run(WRAPPER, PREDICTIONS_FIXTURE, run_dir, UNPINNED)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            metrics = json.loads(
                (run_dir / "metaqa_e2e_metrics.json").read_text(encoding="utf-8")
            )
        record = metrics["evaluation_set"]
        self.assertFalse(record["pinned"])
        self.assertTrue(record["allow_unpinned_override"])
        self.assertEqual(record["rows_processed"], 5)
        self.assertEqual(record["unmatched_question_count"], 2)
        self.assertEqual(record["rows_per_hop"], {"1": 3, "2": 1, "3": 1})
        self.assertEqual(record["rows_with_gold_per_hop"], {"1": 1, "2": 1, "3": 1})
        self.assertEqual(record["gold_questions_available_per_hop"], {"1": 3, "2": 2, "3": 2})

    def test_a_fully_golded_run_is_pinned_with_no_flag(self) -> None:
        rows = [
            {
                "question": "who directed Glass Harbor",
                "hop_count": 1,
                "decomposition": "1. Who directed Glass Harbor?",
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predictions = root / "pinned.json"
            predictions.write_text(json.dumps(rows), encoding="utf-8")
            proc = _run(WRAPPER, predictions, root / "run")
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            metrics = json.loads(
                (root / "run" / "metaqa_e2e_metrics.json").read_text(encoding="utf-8")
            )
        record = metrics["evaluation_set"]
        self.assertTrue(record["pinned"])
        self.assertEqual(record["unmatched_question_count"], 0)
        self.assertFalse(record["allow_unpinned_override"])

    def test_the_question_set_fingerprint_identifies_the_set(self) -> None:
        """Equal fingerprints mean the same questions were scored; that is what makes two
        MetaQA runs a comparison (CLAUDE.md: same evaluation set, or it is not one)."""
        one = [
            {
                "question": "who directed Glass Harbor",
                "hop_count": 1,
                "decomposition": "1. Who directed Glass Harbor?",
            }
        ]
        # Same question, different decomposition: the same evaluation set, so the same
        # fingerprint. That is the point - the fingerprint identifies the SET, not the arm.
        two = [{**one[0], "decomposition": "1. Who is the director of Glass Harbor?"}]
        three = [
            {
                "question": "what is the genre of Winter Marbles",
                "hop_count": 1,
                "decomposition": "1. What is the genre of Winter Marbles?",
            }
        ]
        digests = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index, rows in enumerate((one, two, three)):
                predictions = root / f"rows{index}.json"
                predictions.write_text(json.dumps(rows), encoding="utf-8")
                proc = _run(WRAPPER, predictions, root / f"run{index}")
                self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
                metrics = json.loads(
                    (root / f"run{index}" / "metaqa_e2e_metrics.json").read_text(
                        encoding="utf-8"
                    )
                )
                digests.append(metrics["evaluation_set"]["question_set_sha256"])
        self.assertEqual(digests[0], digests[1], "same questions must fingerprint the same")
        self.assertNotEqual(digests[0], digests[2], "different questions must differ")

    def test_the_predictions_file_is_content_addressed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            _run(WRAPPER, PREDICTIONS_FIXTURE, run_dir, UNPINNED)
            metrics = json.loads(
                (run_dir / "metaqa_e2e_metrics.json").read_text(encoding="utf-8")
            )
        digest = metrics["evaluation_set"]["predictions_sha256"]
        self.assertEqual(len(digest), 64)
        self.assertEqual(
            digest,
            __import__("hashlib")
            .sha256(PREDICTIONS_FIXTURE.read_bytes())
            .hexdigest(),
        )


class TestUpstreamProvenance(unittest.TestCase):
    """Finding 3, second half: a MetaQA number has to be traceable to the run that produced
    the decompositions, not just to a file path."""

    ROWS = [
        {
            "question": "who directed Glass Harbor",
            "hop_count": 1,
            "decomposition": "1. Who directed Glass Harbor?",
        }
    ]

    def _run_with_sibling(self, sibling: dict[str, Any] | None) -> dict[str, Any]:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            decomposer_run = root / "20260822_101500"
            decomposer_run.mkdir()
            predictions = decomposer_run / "results.json"
            predictions.write_text(json.dumps(self.ROWS), encoding="utf-8")
            if sibling is not None:
                (decomposer_run / "config.json").write_text(
                    json.dumps(sibling), encoding="utf-8"
                )
            proc = _run(WRAPPER, predictions, root / "run")
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            return json.loads(
                (root / "run" / "metaqa_e2e_metrics.json").read_text(encoding="utf-8")
            )

    def test_a_missing_snapshot_is_recorded_not_inferred(self) -> None:
        upstream = self._run_with_sibling(None)["upstream_decomposer_run"]
        self.assertFalse(upstream["found"])
        self.assertIn("no decomposer run snapshot", upstream["note"])

    def test_the_run_id_commit_and_config_are_carried_through(self) -> None:
        snapshot = {
            "script": "run_decomposer.py",
            "run_id": "20260822_101500",
            "component": "decomposer",
            "model": "mistral_7b_instruct",
            "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
            "condition": "unguided",
            "shared_config": "configs/decomposer_musique.json",
            "seed": 42,
            "git": {"commit": "0123456789abcdef", "branch": "main", "dirty": False},
        }
        upstream = self._run_with_sibling(snapshot)["upstream_decomposer_run"]
        self.assertTrue(upstream["found"])
        self.assertEqual(upstream["run_id"], "20260822_101500")
        self.assertEqual(upstream["commit"], "0123456789abcdef")
        self.assertEqual(upstream["branch"], "main")
        self.assertFalse(upstream["dirty"])
        self.assertEqual(upstream["model"], "mistral_7b_instruct")
        self.assertEqual(upstream["shared_config"], "configs/decomposer_musique.json")

    def test_an_unreadable_snapshot_is_reported_not_crashed_on(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            decomposer_run = root / "run_dir"
            decomposer_run.mkdir()
            (decomposer_run / "results.json").write_text(
                json.dumps(self.ROWS), encoding="utf-8"
            )
            (decomposer_run / "config.json").write_text("{not json", encoding="utf-8")
            proc = _run(WRAPPER, decomposer_run / "results.json", root / "out")
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            metrics = json.loads(
                (root / "out" / "metaqa_e2e_metrics.json").read_text(encoding="utf-8")
            )
        self.assertFalse(metrics["upstream_decomposer_run"]["found"])
        self.assertIn("could not read", metrics["upstream_decomposer_run"]["note"])


class TestMissingDumpIsRefused(unittest.TestCase):
    """Nit 2: a missing dump must not be skipped silently.

    Both readers of the dumps refuse rather than continue, because a skipped dump would drop
    rows out of the second denominator with no trace in the output.
    """

    def test_the_readers_refuse_a_missing_dump(self) -> None:
        import run_metaqa_compile_execute as wrapper

        with tempfile.TemporaryDirectory() as tmp:
            empty = Path(tmp)
            with self.assertRaises(SystemExit) as caught:
                wrapper.score_failed_items(empty, gold_by_hop={}, exact_threshold=1.0)
            self.assertIn("compile_fail.json", str(caught.exception))
            with self.assertRaises(SystemExit) as caught:
                wrapper.processed_rows(empty)
            self.assertIn("success.json", str(caught.exception))


class TestNoModelIsLoadedGuard(unittest.TestCase):
    """ADR 0016 source-level guard for ``backend.model_loaded: False`` (nit 3).

    That field is an unguarded constant: nothing in a CPU run can notice if someone later
    adds a model load to this path, and the value would then be a false claim in every
    metrics JSON — while the ~8B ceiling assertion (``src/model_size.py``) would also be
    missing. Only a real run could violate it, so per ADR 0016 the invariant is asserted
    against the source, with negative controls that re-break it in memory.
    """

    #: Names whose presence in this script would mean a model is being loaded.
    LOADER_NAMES = frozenset(
        {"load_model", "from_pretrained", "AutoModelForCausalLM", "AutoTokenizer", "pipeline"}
    )
    #: Top-level modules that only a model-loading path needs.
    MODEL_MODULES = frozenset({"torch", "transformers", "peft", "accelerate", "bitsandbytes"})

    @staticmethod
    def model_loading_sites(source: str) -> set[str]:
        """Every marker in ``source`` that would mean a model is loaded here."""
        import ast

        found: set[str] = set()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root in TestNoModelIsLoadedGuard.MODEL_MODULES:
                        found.add(f"import {root}")
            elif isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".")[0]
                if root in TestNoModelIsLoadedGuard.MODEL_MODULES:
                    found.add(f"from {root}")
            elif isinstance(node, ast.Call):
                func = node.func
                name = (
                    func.attr
                    if isinstance(func, ast.Attribute)
                    else func.id
                    if isinstance(func, ast.Name)
                    else None
                )
                if name in TestNoModelIsLoadedGuard.LOADER_NAMES:
                    found.add(f"call {name}")
        return found

    def test_the_wrapper_loads_no_model(self) -> None:
        self.assertEqual(
            self.model_loading_sites(WRAPPER.read_text(encoding="utf-8")),
            set(),
            "this script reports backend.model_loaded false; a model-loading site here "
            "would make that a false claim and would also need the src/model_size.py "
            "ceiling assertion",
        )

    def test_the_metrics_field_and_the_guard_agree(self) -> None:
        """The guard is only worth having if it protects the field that is actually written."""
        source = WRAPPER.read_text(encoding="utf-8")
        self.assertIn('"model_loaded": False', source)

    def test_a_torch_import_is_caught(self) -> None:
        source = WRAPPER.read_text(encoding="utf-8")
        self.assertEqual(self.model_loading_sites(source), set(), "base source must be clean")
        broken = source + "\nimport torch\n"
        self.assertIn("import torch", broken)
        self.assertIn("import torch", self.model_loading_sites(broken))

    def test_a_loader_call_is_caught(self) -> None:
        source = WRAPPER.read_text(encoding="utf-8")
        anchor = "def main() -> None:"
        self.assertIn(anchor, source, "the negative control's anchor moved; fix the control")
        broken = source.replace(
            anchor, anchor + "\n    tokenizer, model = load_model('x', {}, 'cpu', 'none')", 1
        )
        self.assertNotEqual(broken, source, "the negative control did not modify the source")
        self.assertIn("call load_model", self.model_loading_sites(broken))

    def test_a_from_pretrained_call_is_caught(self) -> None:
        source = WRAPPER.read_text(encoding="utf-8")
        anchor = "def main() -> None:"
        self.assertIn(anchor, source, "the negative control's anchor moved; fix the control")
        broken = source.replace(
            anchor, anchor + "\n    m = SomeClass.from_pretrained('x')", 1
        )
        self.assertNotEqual(broken, source, "the negative control did not modify the source")
        self.assertIn("call from_pretrained", self.model_loading_sites(broken))


if __name__ == "__main__":
    unittest.main(verbosity=2)
