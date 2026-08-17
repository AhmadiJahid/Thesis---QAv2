#!/usr/bin/env python3
"""Hand-computed checks for ``scripts/musique_decompositions_evaluator.py``.

Every expected number below is computed by hand from the **fabricated** fixture gold
(``tests/fixtures/data_root/musique/dev_data/musique_ans_v1.0_dev_clean.jsonl``) and the
metric definitions in the script; the arithmetic is written out in each test's docstring.
The fixture never changes, so a number moving here means a normalization, matching or
aggregation rule changed — which must be a deliberate, reviewed edit (same contract as
the golden metrics in ``scripts/smoke_test.py``).

Predictions are generated into a temp directory and the script is run as a subprocess
with ``QAV2_PATHS_CONFIG=configs/smoke_paths.json``, so the real CLI, the real config and
the real artifact writing are all exercised. Nothing here touches real data.

Run::

    .venv/bin/python -m unittest discover -s tests -v
    .venv/bin/python tests/test_decomposition_metrics.py
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
FIXTURES = REPO_ROOT / "tests" / "fixtures"
GOLD_PATH = FIXTURES / "data_root" / "musique" / "dev_data" / "musique_ans_v1.0_dev_clean.jsonl"
PREDICTIONS_FIXTURE = FIXTURES / "predictions" / "decomposer_results_musique.json"
EVALUATOR = REPO_ROOT / "scripts" / "musique_decompositions_evaluator.py"
SMOKE_PATHS_CONFIG = REPO_ROOT / "configs" / "smoke_paths.json"

PLACES = 9


def _load_gold() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with GOLD_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                rows[obj["id"]] = obj
    return rows


GOLD = _load_gold()


def gold_steps(item_id: str) -> list[str]:
    return [s["question"] for s in GOLD[item_id]["question_decomposition"]]


def as_decomposition(steps: list[str]) -> str:
    """The enumerated string form a decomposer run actually emits."""
    return "\n".join(f"{i}. {s}" for i, s in enumerate(steps, start=1))


def prediction(item_id: str, steps: list[str]) -> dict[str, Any]:
    return {
        "query_id": item_id,
        "question": GOLD[item_id]["question"],
        "hop_count": GOLD[item_id]["hop_count"],
        "decomposition": as_decomposition(steps),
    }


class EvaluatorTestBase(unittest.TestCase):
    """Shared plumbing: write a predictions file, run the script, read the artifacts."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory(prefix="qav2_metrics_test_")
        cls.tmp = Path(cls._tmp.name)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def _run(self, argv: list[str], expect_ok: bool = True) -> subprocess.CompletedProcess:
        env = os.environ.copy()
        env["QAV2_PATHS_CONFIG"] = str(SMOKE_PATHS_CONFIG)
        proc = subprocess.run(
            [sys.executable, str(EVALUATOR), *argv],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
        )
        if expect_ok:
            self.assertEqual(proc.returncode, 0, f"evaluator failed:\n{proc.stdout}\n{proc.stderr}")
        return proc

    def evaluate(self, name: str, predictions: list[dict[str, Any]]) -> tuple[dict[str, Any], Path]:
        """Score ``predictions`` against the fixture gold; return (metrics, per_item path)."""
        preds_path = self.tmp / f"{name}_predictions.json"
        preds_path.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
        run_dir = self.tmp / name
        self._run(["--predictions", str(preds_path), "--run-dir", str(run_dir)])
        metrics = json.loads((run_dir / "eval_metrics.json").read_text(encoding="utf-8"))
        return metrics, run_dir / "eval_per_item.json"

    def assertMetrics(self, metrics: dict[str, Any], expected: dict[str, float]) -> None:
        for key, value in expected.items():
            with self.subTest(metric=key):
                self.assertAlmostEqual(metrics[key], value, places=PLACES)


class TestDirectionalStepCount(EvaluatorTestBase):
    def test_prediction_longer_than_gold(self) -> None:
        """4 predicted steps against the 2-step gold of 2hop__d001_a (over-decomposition).

        Hand computation (1 evaluated row):
          steps      pred 4, gold 2 -> signed +2, |error| 2
          step set   tp 2 -> P 2/4 = 0.5, R 2/2 = 1.0, F1 = 2*0.5*1/1.5 = 2/3
          ordered    2 positional matches / max(4, 2) = 0.5
          ROUGE-L    gold joined = 10 whitespace tokens, pred = 10 + 7 + 3 = 20, and the
                     gold token sequence is a prefix of the pred one, so LCS = 10
                     -> P 10/20 = 0.5, R 10/10 = 1.0, F1 = 2/3
          refs       [#1] in step 2 and [#3] in step 4 are both backward -> 2/2 = 1.0
          hops       gold hop_count 2 vs 4 predicted -> exact 0, |error| 2
          composite  0.4*(2/3) + 0.3*0.5 + 0.2*1.0 + 0.1*max(0, 1 - 2/3) = 0.65
        """
        steps = gold_steps("2hop__d001_a") + [
            "Which city is the union based in?",
            "Who founded [#3]?",
        ]
        metrics, _ = self.evaluate("longer", [prediction("2hop__d001_a", steps)])

        self.assertEqual(metrics["total_evaluated"], 1)
        self.assertMetrics(
            metrics,
            {
                "exact_match_rate": 0.0,
                "step_precision_macro": 0.5,
                "step_recall_macro": 1.0,
                "step_f1_macro": 2 / 3,
                "ordered_step_accuracy_macro": 0.5,
                "rouge_l_precision_macro": 0.5,
                "rouge_l_recall_macro": 1.0,
                "rouge_l_f1_macro": 2 / 3,
                "reference_validity_macro": 1.0,
                "reference_validity_micro": 1.0,
                "step_count_mae": 2.0,
                "step_count_abs_error_mae": 2.0,
                "mean_signed_step_count_error": 2.0,
                "over_decomposition_rate": 1.0,
                "under_decomposition_rate": 0.0,
                "hop_count_exact_match_rate": 0.0,
                "hop_count_abs_error_mae": 2.0,
                "composite_score": 0.65,
            },
        )
        self.assertMetrics(
            metrics["per_gold_hop_metrics"]["2"],
            {
                "step_count_mae": 2.0,
                "mean_signed_step_count_error": 2.0,
                "over_decomposition_rate": 1.0,
                "under_decomposition_rate": 0.0,
            },
        )

    def test_prediction_shorter_than_gold(self) -> None:
        """The first 2 steps of the 4-step gold of 4hop1__d003_c (under-decomposition).

        Hand computation (1 evaluated row):
          steps      pred 2, gold 4 -> signed -2, |error| 2
          step set   tp 2 -> P 2/2 = 1.0, R 2/4 = 0.5, F1 = 2/3
          ordered    2 positional matches / max(2, 4) = 0.5
          ROUGE-L    gold joined = 4 + 3 + 5 + 6 = 18 tokens, pred = 4 + 3 = 7, LCS = 7
                     -> P 7/7 = 1.0, R 7/18, F1 = 2*(7/18)/(1 + 7/18) = 14/25 = 0.56
          refs       [#1] in step 2 is backward -> 1/1 = 1.0
          hops       gold hop_count 4 vs 2 predicted -> exact 0, |error| 2
          composite  0.4*(2/3) + 0.3*0.5 + 0.2*1.0 + 0.1*max(0, 1 - 2/3) = 0.65
        """
        steps = gold_steps("4hop1__d003_c")[:2]
        metrics, _ = self.evaluate("shorter", [prediction("4hop1__d003_c", steps)])

        self.assertEqual(metrics["total_evaluated"], 1)
        self.assertMetrics(
            metrics,
            {
                "exact_match_rate": 0.0,
                "step_precision_macro": 1.0,
                "step_recall_macro": 0.5,
                "step_f1_macro": 2 / 3,
                "ordered_step_accuracy_macro": 0.5,
                "rouge_l_precision_macro": 1.0,
                "rouge_l_recall_macro": 7 / 18,
                "rouge_l_f1_macro": 0.56,
                "reference_validity_micro": 1.0,
                "step_count_mae": 2.0,
                "mean_signed_step_count_error": -2.0,
                "over_decomposition_rate": 0.0,
                "under_decomposition_rate": 1.0,
                "hop_count_exact_match_rate": 0.0,
                "hop_count_abs_error_mae": 2.0,
                "composite_score": 0.65,
            },
        )
        self.assertMetrics(
            metrics["per_gold_hop_metrics"]["4"],
            {
                "step_count_mae": 2.0,
                "mean_signed_step_count_error": -2.0,
                "over_decomposition_rate": 0.0,
                "under_decomposition_rate": 1.0,
            },
        )

    def test_empty_prediction(self) -> None:
        """An empty decomposition string against the 2-step gold of 2hop__d001_a.

        Hand computation (1 evaluated row):
          steps      pred 0, gold 2 -> signed -2, |error| 2, counted as under-decomposition
          step set   pred set empty (gold set is not) -> P 0, R 0, F1 0
          ordered    0 matches / max(0, 2) = 0.0
          ROUGE-L    pred has no tokens, gold has 10 -> P/R/F1 all 0.0
          refs       no [#k] at all -> per-row rate 1.0 and micro 1.0 (0 valid of 0 total):
                     reference validity does NOT punish an empty prediction
          composite  0.4*0 + 0.3*0 + 0.2*1.0 + 0.1*max(0, 1 - 2/3) = 0.2 + 1/30 = 7/30
        """
        pred = prediction("2hop__d001_a", [])
        pred["decomposition"] = ""
        metrics, _ = self.evaluate("empty", [pred])

        self.assertEqual(metrics["total_evaluated"], 1)
        self.assertMetrics(
            metrics,
            {
                "exact_match_rate": 0.0,
                "step_precision_macro": 0.0,
                "step_recall_macro": 0.0,
                "step_f1_macro": 0.0,
                "ordered_step_accuracy_macro": 0.0,
                "rouge_l_precision_macro": 0.0,
                "rouge_l_recall_macro": 0.0,
                "rouge_l_f1_macro": 0.0,
                "reference_validity_macro": 1.0,
                "reference_validity_micro": 1.0,
                "step_count_mae": 2.0,
                "mean_signed_step_count_error": -2.0,
                "over_decomposition_rate": 0.0,
                "under_decomposition_rate": 1.0,
                "hop_count_exact_match_rate": 0.0,
                "composite_score": 7 / 30,
            },
        )
        self.assertEqual(metrics["predicted_hop_distribution"], {"0": 1})


class TestReferenceValidity(EvaluatorTestBase):
    def test_forward_reference(self) -> None:
        """3hop1__d002_b with step 2 pointing forward at [#3] instead of back at [#1].

        Hand computation (1 evaluated row):
          steps      pred 3, gold 3 -> signed 0, so over/under rates are both 0
          refs       step 2 uses [#3] (needs 1 <= 3 < 2: invalid), step 3 uses [#2]
                     (valid) -> per-row rate 1/2, micro 1/2
          step set   step 2 differs -> tp 2, P 2/3, R 2/3, F1 2/3
          ordered    2 positional matches / 3 = 2/3
          ROUGE-L    16 tokens on each side differing in exactly one token -> LCS 15,
                     P = R = F1 = 15/16 = 0.9375
          composite  0.4*(2/3) + 0.3*(2/3) + 0.2*0.5 + 0.1*1.0 = 2/3
        """
        steps = gold_steps("3hop1__d002_b")
        steps[1] = steps[1].replace("[#1]", "[#3]")
        metrics, _ = self.evaluate("forward_ref", [prediction("3hop1__d002_b", steps)])

        self.assertMetrics(
            metrics,
            {
                "exact_match_rate": 0.0,
                "step_f1_macro": 2 / 3,
                "ordered_step_accuracy_macro": 2 / 3,
                "rouge_l_f1_macro": 15 / 16,
                "reference_validity_macro": 0.5,
                "reference_validity_micro": 0.5,
                "step_count_mae": 0.0,
                "mean_signed_step_count_error": 0.0,
                "over_decomposition_rate": 0.0,
                "under_decomposition_rate": 0.0,
                "hop_count_exact_match_rate": 1.0,
                "composite_score": 2 / 3,
            },
        )

    def test_self_reference(self) -> None:
        """2hop__d001_a with step 2 referring to itself as [#2].

        Hand computation (1 evaluated row):
          refs       step 2 uses [#2] (needs 1 <= 2 < 2: invalid) -> rate 0.0, micro 0.0
          step set   step 2 differs -> tp 1, P 1/2, R 1/2, F1 0.5
          ordered    1 positional match / 2 = 0.5
          ROUGE-L    10 tokens each side differing in one token -> LCS 9, P = R = F1 = 0.9
          steps      pred 2, gold 2 -> signed 0; hop count matches -> exact 1.0
          composite  0.4*0.5 + 0.3*0.5 + 0.2*0.0 + 0.1*1.0 = 0.45
        """
        steps = gold_steps("2hop__d001_a")
        steps[1] = steps[1].replace("[#1]", "[#2]")
        metrics, _ = self.evaluate("self_ref", [prediction("2hop__d001_a", steps)])

        self.assertMetrics(
            metrics,
            {
                "exact_match_rate": 0.0,
                "step_f1_macro": 0.5,
                "ordered_step_accuracy_macro": 0.5,
                "rouge_l_precision_macro": 0.9,
                "rouge_l_recall_macro": 0.9,
                "rouge_l_f1_macro": 0.9,
                "reference_validity_macro": 0.0,
                "reference_validity_micro": 0.0,
                "step_count_mae": 0.0,
                "mean_signed_step_count_error": 0.0,
                "hop_count_exact_match_rate": 1.0,
                "composite_score": 0.45,
            },
        )


class TestPairedComparison(EvaluatorTestBase):
    def _per_item_of_fixture(self) -> Path:
        """Score the committed fixture predictions once; reuse the per-item file."""
        run_dir = self.tmp / "compare_base"
        if not (run_dir / "eval_per_item.json").exists():
            self._run(
                ["--predictions", str(PREDICTIONS_FIXTURE), "--run-dir", str(run_dir)]
            )
        return run_dir / "eval_per_item.json"

    def test_identical_inputs_are_never_significant(self) -> None:
        """A file compared with itself: every difference is exactly 0, nothing significant.

        Hand computation: with a = b item by item, every paired bootstrap resample gives
        difference 0, so the percentile interval is [0, 0] — which contains 0, hence not
        significant. McNemar sees no discordant pairs (b = c = 0), so p = 1.0 by the
        convention in _mcnemar_exact_p, hence not significant.
        """
        per_item = self._per_item_of_fixture()
        run_dir = self.tmp / "compare_identical"
        self._run(["--compare", str(per_item), str(per_item), "--run-dir", str(run_dir)])
        metrics = json.loads((run_dir / "compare_metrics.json").read_text(encoding="utf-8"))

        # 4 fixture predictions, 1 without a gold row -> 3 evaluated, so 3 aligned items.
        self.assertEqual(metrics["num_aligned_items"], 3)
        self.assertEqual(sorted(metrics["bootstrap"]), sorted(
            ["rouge_l_f1", "step_f1", "ordered_step_accuracy", "composite_score"]
        ))
        for name, result in metrics["bootstrap"].items():
            with self.subTest(statistic=name):
                self.assertAlmostEqual(result["difference"], 0.0, places=PLACES)
                self.assertAlmostEqual(result["ci_low"], 0.0, places=PLACES)
                self.assertAlmostEqual(result["ci_high"], 0.0, places=PLACES)
                self.assertFalse(result["significant"])
        self.assertEqual(sorted(metrics["mcnemar"]), ["exact_match", "hop_count_exact_match"])
        for name, result in metrics["mcnemar"].items():
            with self.subTest(statistic=name):
                self.assertEqual(result["discordant_pairs"], 0)
                self.assertAlmostEqual(result["p_value"], 1.0, places=PLACES)
                self.assertFalse(result["significant"])

        # The comparison's point estimates must reproduce the scoring run's aggregates:
        # step F1 8/9 and composite 0.9222... over the 3 fixture rows.
        self.assertAlmostEqual(metrics["bootstrap"]["step_f1"]["system_a"], 8 / 9, places=PLACES)
        self.assertAlmostEqual(
            metrics["bootstrap"]["composite_score"]["system_a"], 0.9222222222222222, places=PLACES
        )

    def test_identical_inputs_are_reproducible(self) -> None:
        """Same seed, same inputs -> byte-identical bootstrap results."""
        per_item = self._per_item_of_fixture()
        first = self.tmp / "compare_seed_a"
        second = self.tmp / "compare_seed_b"
        for run_dir in (first, second):
            self._run(["--compare", str(per_item), str(per_item), "--run-dir", str(run_dir)])
        a = json.loads((first / "compare_metrics.json").read_text(encoding="utf-8"))
        b = json.loads((second / "compare_metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(a["bootstrap"], b["bootstrap"])
        self.assertEqual(a["seed"], b["seed"])

    def test_different_evaluation_sets_are_refused(self) -> None:
        """Dropping one item from one side must abort and name the offending id."""
        per_item = self._per_item_of_fixture()
        rows = json.loads(per_item.read_text(encoding="utf-8"))
        dropped = rows[0]["item_id"]
        short_path = self.tmp / "compare_short_per_item.json"
        short_path.write_text(json.dumps(rows[1:], ensure_ascii=False, indent=2), encoding="utf-8")

        proc = self._run(
            [
                "--compare", str(per_item), str(short_path),
                "--run-dir", str(self.tmp / "compare_mismatch"),
            ],
            expect_ok=False,
        )
        self.assertNotEqual(proc.returncode, 0)
        message = proc.stdout + proc.stderr
        self.assertIn(dropped, message)
        self.assertIn("SAME evaluation set", message)


if __name__ == "__main__":
    unittest.main(verbosity=2)
