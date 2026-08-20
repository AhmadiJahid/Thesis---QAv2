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

import ast
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures"
GOLD_PATH = FIXTURES / "data_root" / "musique" / "dev_data" / "musique_ans_v1.0_dev_clean.jsonl"
PREDICTIONS_FIXTURE = FIXTURES / "predictions" / "decomposer_results_musique.json"
EVALUATOR = REPO_ROOT / "scripts" / "musique_decompositions_evaluator.py"
SMOKE_PATHS_CONFIG = REPO_ROOT / "configs" / "smoke_paths.json"

PLACES = 9


def _import_evaluator() -> Any:
    """Import the evaluator as a module, for the checks that call its functions directly."""
    name = "musique_decompositions_evaluator"
    spec = importlib.util.spec_from_file_location(name, EVALUATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered before exec: @dataclass resolves annotations through sys.modules.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


EVAL = _import_evaluator()


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

    def evaluate(
        self, name: str, predictions: list[dict[str, Any]], gold: Path | None = None
    ) -> tuple[dict[str, Any], Path]:
        """Score ``predictions`` against the fixture gold; return (metrics, per_item path)."""
        preds_path = self.tmp / f"{name}_predictions.json"
        preds_path.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
        run_dir = self.tmp / name
        argv = ["--predictions", str(preds_path), "--run-dir", str(run_dir)]
        if gold is not None:
            argv += ["--gold", str(gold)]
        self._run(argv)
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
          exact rate 1 - over(1.0) - under(0.0) = 0.0
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
                "step_count_exact_rate": 0.0,
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
                "step_count_exact_rate": 0.0,
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
                "step_count_exact_rate": 0.0,
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
                "step_count_exact_rate": 0.0,
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
                "step_count_exact_rate": 0.0,
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
                "step_count_exact_rate": 1.0,
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
                "step_count_exact_rate": 1.0,
                "hop_count_exact_match_rate": 1.0,
                "composite_score": 0.45,
            },
        )


class TestStepNormalization(EvaluatorTestBase):
    def test_punctuation_only_difference_is_a_perfect_step_match(self) -> None:
        """2hop__d004_p predicted with punctuation-only changes: '?' -> '.' and '?' dropped.

        This pins the documented normalization rule (lowercase, punctuation stripped except
        '#', whitespace collapsed): with it, the two steps normalize to identical strings.
        Deleting the punctuation-strip line in ``_normalize_step`` turns every step-level
        number below red (exact match 0.0, step F1 0.0, ordered 0.0, composite 0.6).

        Hand computation (1 evaluated row):
          gold       "Which board approved the Rill Valley permit?" / "Who chairs [#1]?"
          pred       "Which board approved the Rill Valley permit." / "Who chairs [#1]"
          normalized both sides -> "which board approved the rill valley permit" and
                     "who chairs #1" -> identical, so EM 1.0, P/R/F1 1.0, ordered 1.0
          ROUGE-L    _tokenize does NOT strip punctuation: 10 tokens each side, differing
                     at "permit?"/"permit." and "[#1]?"/"[#1]" -> LCS 8
                     -> P 8/10 = 0.8, R 0.8, F1 0.8
          refs       [#1] in step 2 is backward -> 1/1 = 1.0
          steps      pred 2, gold 2 -> signed 0, MAE 0, exact rate 1.0, hop exact 1.0
          composite  0.4*1.0 + 0.3*1.0 + 0.2*1.0 + 0.1*max(0, 1 - 0/3) = 1.0
        """
        steps = [
            "Which board approved the Rill Valley permit.",
            "Who chairs [#1]",
        ]
        metrics, _ = self.evaluate("punct_only", [prediction("2hop__d004_p", steps)])

        self.assertEqual(metrics["total_evaluated"], 1)
        self.assertMetrics(
            metrics,
            {
                "exact_match_rate": 1.0,
                "step_precision_macro": 1.0,
                "step_recall_macro": 1.0,
                "step_f1_macro": 1.0,
                "ordered_step_accuracy_macro": 1.0,
                "rouge_l_precision_macro": 0.8,
                "rouge_l_recall_macro": 0.8,
                "rouge_l_f1_macro": 0.8,
                "reference_validity_macro": 1.0,
                "reference_validity_micro": 1.0,
                "step_count_mae": 0.0,
                "mean_signed_step_count_error": 0.0,
                "step_count_exact_rate": 1.0,
                "hop_count_exact_match_rate": 1.0,
                "composite_score": 1.0,
            },
        )


class TestGoldDenominators(EvaluatorTestBase):
    def test_hop_count_disagreeing_with_step_count_is_refused(self) -> None:
        """Gold whose 'hop_count' field contradicts len(question_decomposition) must abort.

        The directional step-count metrics divide by len(gold steps) and the hop-count
        metrics by the 'hop_count' field; a row where those differ would silently make the
        two families measure different things, so the loader refuses and names the row.
        """
        tampered = self.tmp / "gold_hop_mismatch.jsonl"
        rows = [
            json.loads(line)
            for line in GOLD_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        rows[0]["hop_count"] = len(rows[0]["question_decomposition"]) + 1
        tampered.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
        )

        preds_path = self.tmp / "gold_mismatch_predictions.json"
        preds_path.write_text(
            json.dumps([prediction("2hop__d001_a", gold_steps("2hop__d001_a"))], ensure_ascii=False),
            encoding="utf-8",
        )
        proc = self._run(
            [
                "--predictions", str(preds_path),
                "--gold", str(tampered),
                "--run-dir", str(self.tmp / "gold_mismatch"),
            ],
            expect_ok=False,
        )
        self.assertNotEqual(proc.returncode, 0)
        message = proc.stdout + proc.stderr
        self.assertIn("2hop__d001_a", message)
        self.assertIn("hop_count=3, steps=2", message)

    def test_matching_gold_loads(self) -> None:
        """The committed fixture gold agrees on both denominators, so it must load."""
        metrics, _ = self.evaluate(
            "gold_ok", [prediction("2hop__d001_a", gold_steps("2hop__d001_a"))]
        )
        self.assertEqual(metrics["total_evaluated"], 1)


class TestMcNemarPower(unittest.TestCase):
    def test_min_attainable_p_value(self) -> None:
        """b=3, c=1 discordant pairs: p and the smallest p those 4 pairs could give.

        Hand computation (exact two-sided McNemar, m = b + c = 4):
          p     = min(1, 2 * (C(4,0) + C(4,1)) / 2^4) = 2 * 5/16 = 0.625
          min p = the most one-sided outcome, min(b, c) = 0
                = min(1, 2 * C(4,0) / 2^4) = 2/16 = 0.125
          0.125 >= alpha 0.05, so this test cannot reject at 0.05 however the 4 pairs fall:
          "significant: false" here is a statement about n, and 'underpowered' says so.
        """
        def row(exact: float) -> dict[str, Any]:
            return {"exact_match": exact, "hop_count_exact_match": 1.0}

        rows_a = [row(1.0), row(1.0), row(1.0), row(0.0), row(0.0)]
        rows_b = [row(0.0), row(0.0), row(0.0), row(1.0), row(0.0)]
        out = EVAL._mcnemar(rows_a, rows_b, alpha=0.05, underpowered=False)["exact_match"]

        self.assertEqual(out["correct_only_in_a"], 3)
        self.assertEqual(out["correct_only_in_b"], 1)
        self.assertEqual(out["discordant_pairs"], 4)
        self.assertAlmostEqual(out["p_value"], 0.625, places=PLACES)
        self.assertAlmostEqual(out["min_attainable_p_value"], 0.125, places=PLACES)
        self.assertFalse(out["min_attainable_p_reaches_alpha"])
        self.assertFalse(out["significant"])
        self.assertTrue(out["underpowered"])
        self.assertEqual(out["n"], 5)

    def test_six_discordant_pairs_can_reach_alpha(self) -> None:
        """m = 6 is the smallest discordant count whose min p clears 0.05: 2/2^6 = 0.03125."""
        self.assertAlmostEqual(EVAL._mcnemar_exact_p(5, 0), 2 / 32, places=PLACES)  # 0.0625
        self.assertAlmostEqual(EVAL._mcnemar_exact_p(6, 0), 2 / 64, places=PLACES)  # 0.03125


class TestPairedTTest(unittest.TestCase):
    """Known-answer checks for the paired t-test added by issue #30 / ADR 0017.

    The expected p-values come from the closed forms of the Student-t survival function at
    1 and 2 degrees of freedom, not from the library that computes them, so these are
    genuine known answers rather than a restatement of the implementation:

      dof = 1: P(|T| > t) = 1 - (2/pi) * arctan(t)
      dof = 2: P(|T| > t) = 1 - t / sqrt(2 + t^2)
    """

    ALPHA = 0.05

    def _row(self, a: list[float], b: list[float]) -> dict[str, Any]:
        return EVAL._paired_t_test_row(
            np.array(a, dtype=float), np.array(b, dtype=float), self.ALPHA, underpowered=False
        )

    def test_two_items_dof_one(self) -> None:
        """differences [1, 0]: mean 0.5, sd sqrt(0.5), se 0.5, t = 1.0, dof = 1.

        sd = sqrt(((1 - 0.5)^2 + (0 - 0.5)^2) / (2 - 1)) = sqrt(0.5)
        se = sqrt(0.5) / sqrt(2) = 0.5   ->   t = 0.5 / 0.5 = 1.0
        p  = 1 - (2/pi) * arctan(1) = 1 - (2/pi) * (pi/4) = 0.5
        """
        row = self._row([1.0, 0.0], [0.0, 0.0])
        self.assertEqual(row["n"], 2)
        self.assertEqual(row["degrees_of_freedom"], 1)
        self.assertIsNone(row["degenerate"])
        self.assertAlmostEqual(row["difference"], 0.5, places=PLACES)
        self.assertAlmostEqual(row["t_statistic"], 1.0, places=PLACES)
        self.assertAlmostEqual(row["p_value"], 0.5, places=PLACES)
        self.assertFalse(row["significant"])

    def test_three_items_dof_two(self) -> None:
        """differences [1, 1, -1]: t = 0.5 at dof = 2, so p = 1 - 0.5/1.5 = 2/3.

        mean = 1/3; deviations 2/3, 2/3, -4/3
        sd   = sqrt((4/9 + 4/9 + 16/9) / 2) = sqrt(4/3)
        se   = sqrt(4/3) / sqrt(3) = 2/3    ->  t = (1/3) / (2/3) = 0.5
        p    = 1 - 0.5 / sqrt(2 + 0.25) = 1 - 0.5/1.5 = 2/3
        """
        row = self._row([1.0, 1.0, 0.0], [0.0, 0.0, 1.0])
        self.assertEqual(row["degrees_of_freedom"], 2)
        self.assertAlmostEqual(row["difference"], 1 / 3, places=PLACES)
        self.assertAlmostEqual(row["t_statistic"], 0.5, places=PLACES)
        self.assertAlmostEqual(row["p_value"], 2 / 3, places=PLACES)
        self.assertFalse(row["significant"])

    def test_sign_follows_the_difference_direction(self) -> None:
        """Same magnitudes with the systems swapped: t flips sign, p is unchanged."""
        forward = self._row([1.0, 1.0, 0.0], [0.0, 0.0, 1.0])
        reverse = self._row([0.0, 0.0, 1.0], [1.0, 1.0, 0.0])
        self.assertAlmostEqual(reverse["t_statistic"], -forward["t_statistic"], places=PLACES)
        self.assertAlmostEqual(reverse["p_value"], forward["p_value"], places=PLACES)

    def test_significance_is_p_below_alpha(self) -> None:
        """A large, consistent difference at dof = 2: p = 1 - t/sqrt(2+t^2) < 0.05.

        differences [1, 1, 0.9]: mean 0.9666..., sd sqrt(1/300), se sqrt(1/900) = 1/30,
        t = 29.0 -> p = 1 - 29/sqrt(843) = 0.00118..., which is below alpha 0.05.
        """
        row = self._row([1.0, 1.0, 0.9], [0.0, 0.0, 0.0])
        self.assertAlmostEqual(row["t_statistic"], 29.0, places=PLACES)
        self.assertAlmostEqual(row["p_value"], 1 - 29 / math.sqrt(843), places=PLACES)
        self.assertTrue(row["significant"])

    def test_zero_variance_is_degenerate_not_significant(self) -> None:
        """Identical inputs: every difference is 0, t is 0/0, so no claim is made."""
        row = self._row([0.2, 0.4, 0.6], [0.2, 0.4, 0.6])
        self.assertIsNone(row["t_statistic"])
        self.assertIsNone(row["p_value"])
        self.assertFalse(row["significant"])
        self.assertIn("standard deviation", row["degenerate"])

    def test_constant_non_zero_difference_is_degenerate(self) -> None:
        """A constant non-zero difference gives an infinite t: recorded, not claimed.

        Not representable in JSON and not a real case for bounded per-item metrics; the
        bootstrap CI (which would be [0.1, 0.1], excluding 0) is what carries this input.
        """
        row = self._row([0.3, 0.5, 0.7], [0.2, 0.4, 0.6])
        self.assertIsNone(row["t_statistic"])
        self.assertFalse(row["significant"])
        self.assertIsNotNone(row["degenerate"])

    def test_non_finite_input_yields_a_degenerate_row_not_a_nan_in_json(self) -> None:
        """N5: nothing non-finite reaches the metrics JSON.

        `NaN` and `Infinity` are what Python's json writer emits for those floats and are
        **not valid JSON**, so a non-finite result is reported as a degenerate row instead.
        A NaN cannot come from a v1 file any more (the loader type-checks every compared
        column), which is exactly why this guard is tested directly.
        """
        row = self._row([1.0, float("nan"), 0.5], [0.0, 0.0, 0.0])
        self.assertIsNone(row["t_statistic"])
        self.assertIsNone(row["p_value"])
        self.assertFalse(row["significant"])
        self.assertIsNotNone(row["degenerate"])
        # Serialisable, and strictly so: json rejects NaN when told to.
        json.dumps(row, allow_nan=False)

    def test_single_item_is_degenerate(self) -> None:
        """n = 1: dof = 0, no t-test exists."""
        row = self._row([1.0], [0.0])
        self.assertEqual(row["degrees_of_freedom"], 0)
        self.assertIsNone(row["p_value"])
        self.assertIn("at least 2 items", row["degenerate"])

    def test_statistics_covered(self) -> None:
        """The t-test covers every compared metric with a per-item value, and only those."""
        self.assertEqual(
            sorted(EVAL.T_TEST_STATISTICS),
            sorted(
                [
                    "rouge_l_f1",
                    "step_f1",
                    "ordered_step_accuracy",
                    "exact_match",
                    "hop_count_exact_match",
                ]
            ),
        )
        # composite_score is bootstrapped but has no per-item value, so no paired difference
        # to t-test exists.
        self.assertIn("composite_score", EVAL.BOOTSTRAP_STATISTICS)
        self.assertNotIn("composite_score", EVAL.T_TEST_STATISTICS)


class TestBootstrapChunking(unittest.TestCase):
    """The chunked bootstrap must be bit-identical to the single-block draw it replaced."""

    WEIGHTS = {
        "step_f1_macro": 0.4,
        "ordered_step_accuracy_macro": 0.3,
        "reference_validity_micro": 0.2,
        "step_count_error": 0.1,
    }
    SCALE = 3.0
    ITERATIONS = 200
    SEED = 42

    def _arrays(self, offset: float) -> dict[str, np.ndarray]:
        n = 6
        base = np.array([0.1, 0.4, 0.5, 0.8, 0.9, 1.0], dtype=float)
        return {
            "step_f1": np.clip(base + offset, 0.0, 1.0),
            "ordered_step_accuracy": np.clip(base * 0.9 + offset, 0.0, 1.0),
            "rouge_l_f1": np.clip(base * 0.8 + offset, 0.0, 1.0),
            "reference_valid_count": np.array([1, 2, 0, 3, 1, 2], dtype=float),
            "reference_total_count": np.array([2, 2, 0, 3, 2, 2], dtype=float),
            "step_count_abs_error": np.array([0, 1, 2, 0, 1, 3], dtype=float),
        }

    def _run(self, chunk_size: int) -> dict[str, dict[str, float]]:
        return EVAL._paired_bootstrap(
            self._arrays(0.0),
            self._arrays(-0.05),
            n=6,
            iterations=self.ITERATIONS,
            alpha=0.05,
            seed=self.SEED,
            weights=self.WEIGHTS,
            scale=self.SCALE,
            chunk_size=chunk_size,
            underpowered=False,
        )

    def test_chunk_size_does_not_change_the_intervals(self) -> None:
        """Chunk sizes 1, 7, 199 and iterations all give identical CIs for one seed.

        chunk_size == iterations is exactly the pre-chunking code path (one
        ``rng.integers((iterations, n))`` draw), so equality with it is the "identical
        before/after chunking" check, not merely internal consistency.
        """
        reference = self._run(self.ITERATIONS)
        # Non-degenerate: system_a is better by construction, so the CI is not [0, 0].
        self.assertGreater(reference["step_f1"]["difference"], 0.0)
        self.assertNotEqual(reference["step_f1"]["ci_low"], reference["step_f1"]["ci_high"])
        for chunk_size in (1, 7, 199, 10_000):
            with self.subTest(chunk_size=chunk_size):
                self.assertEqual(self._run(chunk_size), reference)

    def test_index_stream_is_order_preserving(self) -> None:
        """The mechanism: chunked int64 draws concatenate to the single-block draw."""
        one = np.random.default_rng(self.SEED).integers(0, 6, size=(10, 6))
        rng = np.random.default_rng(self.SEED)
        chunked = np.concatenate([rng.integers(0, 6, size=(k, 6)) for k in (3, 1, 6)])
        self.assertTrue(np.array_equal(one, chunked))

    def test_non_positive_chunk_size_is_refused(self) -> None:
        with self.assertRaises(SystemExit):
            self._run(0)


#: The four fixture gold ids, in the order the comparison helpers below score them.
FIXTURE_IDS = ("2hop__d001_a", "3hop1__d002_b", "4hop1__d003_c", "2hop__d004_p")


def perfect_predictions() -> list[dict[str, Any]]:
    """Every fixture row predicted exactly as gold."""
    return [prediction(i, gold_steps(i)) for i in FIXTURE_IDS]


def degraded_predictions() -> list[dict[str, Any]]:
    """The same rows, degraded unevenly, so the per-item differences have a spread.

    Two rows are left perfect, one loses its last step's text and one loses everything.
    An even degradation would give a constant per-item difference and hence a degenerate
    t-test, which is not what a real comparison looks like.
    """
    preds = []
    for position, item_id in enumerate(FIXTURE_IDS):
        steps = list(gold_steps(item_id))
        if position == 2:
            steps[-1] = "a step that matches no gold step"
        elif position == 3:
            steps = ["nothing like the gold decomposition"]
        preds.append(prediction(item_id, steps))
    return preds


class TestPairedComparison(EvaluatorTestBase):
    def _arm_a_per_item(self) -> Path:
        _, path = self.evaluate("arm_a_perfect", perfect_predictions())
        return path

    def _arm_b_per_item(self) -> Path:
        _, path = self.evaluate("arm_b_degraded", degraded_predictions())
        return path

    @staticmethod
    def _items(per_item: Path) -> dict[str, dict[str, Any]]:
        payload = json.loads(per_item.read_text(encoding="utf-8"))
        return {row["item_id"]: row for row in payload["items"]}

    def _degraded_comparison(self) -> tuple[Path, dict[str, Any]]:
        """Compare the perfect arm against the degraded one; return (run_dir, metrics)."""
        run_dir = self.tmp / "compare_degraded"
        if not (run_dir / "compare_metrics.json").exists():
            self._run(
                [
                    "--compare", str(self._arm_a_per_item()), str(self._arm_b_per_item()),
                    "--run-dir", str(run_dir),
                ]
            )
        return run_dir, json.loads((run_dir / "compare_metrics.json").read_text(encoding="utf-8"))

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

        # 5 fixture predictions, 1 without a gold row -> 4 evaluated, so 4 aligned items.
        self.assertEqual(metrics["num_aligned_items"], 4)
        self.assertEqual(sorted(metrics["bootstrap"]), sorted(
            ["rouge_l_f1", "step_f1", "ordered_step_accuracy", "composite_score"]
        ))
        for name, result in metrics["bootstrap"].items():
            with self.subTest(statistic=name):
                self.assertAlmostEqual(result["difference"], 0.0, places=PLACES)
                self.assertAlmostEqual(result["ci_low"], 0.0, places=PLACES)
                self.assertAlmostEqual(result["ci_high"], 0.0, places=PLACES)
                self.assertFalse(result["significant"])
                self.assertEqual(result["n"], 4)
        self.assertEqual(sorted(metrics["mcnemar"]), ["exact_match", "hop_count_exact_match"])
        for name, result in metrics["mcnemar"].items():
            with self.subTest(statistic=name):
                self.assertEqual(result["discordant_pairs"], 0)
                self.assertAlmostEqual(result["p_value"], 1.0, places=PLACES)
                self.assertFalse(result["significant"])
                # 0 discordant pairs: the smallest p attainable is 1.0, so this test could
                # not have rejected at any alpha — recorded rather than left implicit.
                self.assertAlmostEqual(result["min_attainable_p_value"], 1.0, places=PLACES)
                self.assertFalse(result["min_attainable_p_reaches_alpha"])
                self.assertTrue(result["underpowered"])

        # The comparison's point estimates must reproduce the scoring run's aggregates:
        # step F1 11/12 and composite 113/120 over the 4 fixture rows.
        self.assertAlmostEqual(metrics["bootstrap"]["step_f1"]["system_a"], 11 / 12, places=PLACES)
        self.assertAlmostEqual(
            metrics["bootstrap"]["composite_score"]["system_a"], 113 / 120, places=PLACES
        )

    def test_n_below_the_floor_is_flagged(self) -> None:
        """n = 4 is below min_items_for_significance_claim (30), so every row is flagged."""
        per_item = self._per_item_of_fixture()
        run_dir = self.tmp / "compare_floor"
        proc = self._run(["--compare", str(per_item), str(per_item), "--run-dir", str(run_dir)])
        metrics = json.loads((run_dir / "compare_metrics.json").read_text(encoding="utf-8"))

        floor = metrics["significance_floor"]
        self.assertEqual(floor["num_items"], 4)
        self.assertEqual(floor["min_items_for_significance_claim"], 30)
        self.assertTrue(floor["below_min_items"])
        self.assertIn("underpowered", floor["warning"])
        for result in metrics["bootstrap"].values():
            self.assertTrue(result["underpowered"])
        self.assertIn("WARNING", (run_dir / "compare_notes.md").read_text(encoding="utf-8"))
        self.assertIn("CI or p", proc.stdout)

    def test_weight_mismatch_between_files_is_refused(self) -> None:
        """Two files scored under different composite weights are not comparable."""
        per_item = self._per_item_of_fixture()
        payload = json.loads(per_item.read_text(encoding="utf-8"))
        self.assertEqual(payload["composite_score_weights"]["step_f1_macro"], 0.4)
        payload["composite_score_weights"]["step_f1_macro"] = 0.5
        other = self.tmp / "compare_other_weights_per_item.json"
        other.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        proc = self._run(
            [
                "--compare", str(per_item), str(other),
                "--run-dir", str(self.tmp / "compare_weights"),
            ],
            expect_ok=False,
        )
        self.assertNotEqual(proc.returncode, 0)
        message = proc.stdout + proc.stderr
        self.assertIn("SAME composite-score weights", message)
        self.assertIn("composite_score_weights", message)

    def test_legacy_bare_list_per_item_file_is_refused(self) -> None:
        """A per-item file with no stamped weights cannot be recomputed against."""
        per_item = self._per_item_of_fixture()
        payload = json.loads(per_item.read_text(encoding="utf-8"))
        legacy = self.tmp / "compare_legacy_per_item.json"
        legacy.write_text(
            json.dumps(payload["items"], ensure_ascii=False, indent=2), encoding="utf-8"
        )

        proc = self._run(
            [
                "--compare", str(legacy), str(legacy),
                "--run-dir", str(self.tmp / "compare_legacy"),
            ],
            expect_ok=False,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("legacy bare-list per-item format", proc.stdout + proc.stderr)

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

    def test_t_test_is_reported_next_to_bootstrap_and_mcnemar(self) -> None:
        """The t-test rows cover the five per-item metrics, on the same pairing.

        Same-pairing check: for every metric that both families report, the t-test's
        difference must equal the bootstrap's (or McNemar's) difference exactly — they are
        the mean of the same per-item differences over the same aligned items.
        """
        run_dir, metrics = self._degraded_comparison()
        self.assertEqual(
            sorted(metrics["t_test"]),
            sorted(
                [
                    "rouge_l_f1",
                    "step_f1",
                    "ordered_step_accuracy",
                    "exact_match",
                    "hop_count_exact_match",
                ]
            ),
        )
        self.assertNotIn("composite_score", metrics["t_test"])
        self.assertEqual(
            metrics["tests_reported"],
            {
                "bootstrap": 4,
                "mcnemar": 2,
                "paired_t_test": 5,
                "headline_protocol": (
                    "bootstrap + McNemar (ADR 0009); the t-test is additive (ADR 0017)"
                ),
                "multiple_comparison_correction": None,
            },
        )
        for name, row in metrics["t_test"].items():
            with self.subTest(statistic=name):
                self.assertEqual(row["n"], metrics["num_aligned_items"])
                self.assertEqual(row["degrees_of_freedom"], metrics["num_aligned_items"] - 1)
                # n = 4 is below the reporting floor, so every row carries the flag (ADR 0011).
                self.assertTrue(row["underpowered"])
                if name in metrics["bootstrap"]:
                    self.assertAlmostEqual(
                        row["difference"], metrics["bootstrap"][name]["difference"], places=PLACES
                    )
                if name in metrics["mcnemar"]:
                    self.assertAlmostEqual(
                        row["difference"], metrics["mcnemar"][name]["difference"], places=PLACES
                    )
                if row["degenerate"] is None:
                    self.assertEqual(row["significant"], row["p_value"] < metrics["alpha"])
                    # A positive difference must carry a positive t: same direction, a - b.
                    self.assertEqual(row["difference"] > 0, row["t_statistic"] > 0)

        note = (run_dir / "compare_notes.md").read_text(encoding="utf-8")
        self.assertIn("paired t-test", note)
        self.assertIn("dof=3", note)
        self.assertIn("headline protocol", note)

    def test_t_test_matches_a_directly_computed_value(self) -> None:
        """The step-F1 t of the degraded comparison, recomputed from its per-item rows.

        Independent of the comparison path: the per-item files are read here, the paired
        differences are formed from them, and t = mean / (sd / sqrt(n)) is evaluated
        directly. This pins that --compare t-tests the per-item column it claims to.
        """
        _, metrics = self._degraded_comparison()
        rows_a = self._items(self._arm_a_per_item())
        rows_b = self._items(self._arm_b_per_item())
        diffs = np.array(
            [rows_a[i]["step_f1"] - rows_b[i]["step_f1"] for i in sorted(rows_a)], dtype=float
        )
        expected_t = float(diffs.mean() / (diffs.std(ddof=1) / math.sqrt(diffs.size)))
        self.assertAlmostEqual(
            metrics["t_test"]["step_f1"]["t_statistic"], expected_t, places=PLACES
        )

    def test_different_evaluation_sets_are_refused(self) -> None:
        """Dropping one item from one side must abort and name the offending id."""
        per_item = self._per_item_of_fixture()
        payload = json.loads(per_item.read_text(encoding="utf-8"))
        dropped = payload["items"][0]["item_id"]
        payload["items"] = payload["items"][1:]
        short_path = self.tmp / "compare_short_per_item.json"
        short_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

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


class TestV1CompareShim(EvaluatorTestBase):
    """``--v1-per-item``: read v1's bare-list per-item files (ADR 0020).

    The v1 format is reconstructed here from a v2 per-item file by dropping ``item_id`` and
    the object header — which is exactly what v1 wrote (verified against
    ``/cta/users/fyilmaz/Thesis---QA/runs/musique_decomposition_eval/*_per_item.json``,
    whose rows carry the same field names and no id). No test reads the v1 tree: it is
    read-only, outside this repo, and a test must not depend on it existing.
    """

    def _v1_file(self, name: str, predictions: list[dict[str, Any]]) -> Path:
        _, per_item = self.evaluate(f"v1src_{name}", predictions)
        payload = json.loads(per_item.read_text(encoding="utf-8"))
        rows = [{k: v for k, v in row.items() if k != "item_id"} for row in payload["items"]]
        path = self.tmp / f"v1_{name}.json"
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def _v1_pair(self) -> tuple[Path, Path]:
        return (
            self._v1_file("perfect", perfect_predictions()),
            self._v1_file("degraded", degraded_predictions()),
        )

    def _compare_v1(self, name: str, extra: list[str] | None = None) -> dict[str, Any]:
        a, b = self._v1_pair()
        run_dir = self.tmp / name
        self._run(
            ["--compare", str(a), str(b), "--v1-per-item", "--run-dir", str(run_dir), *(extra or [])]
        )
        return json.loads((run_dir / "compare_metrics.json").read_text(encoding="utf-8"))

    def test_v1_file_without_the_flag_is_refused_and_names_it(self) -> None:
        """A v1 input is never silently read as a v2 artifact."""
        a, b = self._v1_pair()
        proc = self._run(
            ["--compare", str(a), str(b), "--run-dir", str(self.tmp / "v1_no_flag")],
            expect_ok=False,
        )
        self.assertNotEqual(proc.returncode, 0)
        message = proc.stdout + proc.stderr
        self.assertIn("legacy bare-list per-item format", message)
        self.assertIn("--v1-per-item", message)

    def test_the_flag_on_a_v2_file_is_refused(self) -> None:
        """The opt-in is not a "read anything" switch: a v2 object under it aborts."""
        v2 = self._arm_a_per_item_for_v1_test()
        proc = self._run(
            [
                "--compare", str(v2), str(v2), "--v1-per-item",
                "--run-dir", str(self.tmp / "v1_on_v2"),
            ],
            expect_ok=False,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("expects the v1 bare-list per-item format", proc.stdout + proc.stderr)

    def _arm_a_per_item_for_v1_test(self) -> Path:
        _, per_item = self.evaluate("v1src_perfect", perfect_predictions())
        return per_item

    def test_v1_statistics_equal_the_v2_comparison_of_the_same_rows(self) -> None:
        """The shim changes provenance recording, not arithmetic.

        The same four rows compared as v2 artifacts and as v1 artifacts must give the same
        point estimates, the same McNemar counts and p-values and the same t-tests. Only the
        bootstrap CI edges are excluded: the two paths align on different keys (item_id vs
        normalized question), and the percentile interval depends on the row order the index
        matrix is applied to — the alignment sensitivity ADR 0020 condition 3 exists for.
        """
        v1 = self._compare_v1("v1_vs_v2")
        run_dir = self.tmp / "v2_for_v1_check"
        self._run(
            [
                "--compare", str(self._arm_a_per_item_for_v1_test()),
                str(self._v1_source_of_degraded()),
                "--run-dir", str(run_dir),
            ]
        )
        v2 = json.loads((run_dir / "compare_metrics.json").read_text(encoding="utf-8"))

        self.assertEqual(v1["num_aligned_items"], v2["num_aligned_items"])
        for name, row in v2["bootstrap"].items():
            with self.subTest(bootstrap=name):
                for key in ("system_a", "system_b", "difference"):
                    self.assertAlmostEqual(v1["bootstrap"][name][key], row[key], places=PLACES)
        for family in ("mcnemar", "t_test"):
            for name, row in v2[family].items():
                with self.subTest(family=family, statistic=name):
                    self.assertEqual(sorted(v1[family][name]), sorted(row))
                    for key, value in row.items():
                        got = v1[family][name][key]
                        if isinstance(value, float):
                            self.assertAlmostEqual(got, value, places=PLACES)
                        else:
                            self.assertEqual(got, value)

    def _v1_source_of_degraded(self) -> Path:
        _, per_item = self.evaluate("v1src_degraded", degraded_predictions())
        return per_item

    def test_v1_provenance_is_recorded(self) -> None:
        """ADR 0020: the output states v1 inputs, their hashes, and the no-SHA caveat."""
        metrics = self._compare_v1("v1_provenance")
        block = metrics["v1_format_inputs"]
        self.assertTrue(block["enabled"])
        self.assertTrue(block["prior_work_not_v2_evidence"])
        self.assertIn("NO commit SHA", block["caveat"])
        self.assertEqual(block["alignment"], "normalized_question")
        check = block["same_item_check"]
        self.assertEqual(check["pairs"], metrics["num_aligned_items"])
        # 'question' is the alignment key here, so it witnesses nothing and is recorded as
        # tautological; 'gold_steps' is what actually verified the pairing.
        self.assertEqual(check["verification_fields"], ["gold_steps"])
        self.assertEqual(check["tautological_fields"], ["question"])
        self.assertEqual(
            check["fields_verified_equal"]["gold_steps"], metrics["num_aligned_items"]
        )
        for side, record in block["inputs"].items():
            with self.subTest(side=side):
                path = Path(record["path"])
                self.assertEqual(
                    record["sha256"], hashlib.sha256(path.read_bytes()).hexdigest()
                )
                self.assertEqual(record["rows"], 4)
                self.assertRegex(record["mtime_utc"], r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
        # v1 files stamp no weights, so "do the config's weights match the files'?" has no
        # answer — null, not a bool that would read as "checked and equal".
        self.assertIsNone(metrics["config_weights_match_per_item_files"])
        self.assertIsNone(metrics["per_item_composite_score_weights"])

    def test_v1_caveat_leads_the_note_and_the_stdout(self) -> None:
        """ADR 0020 condition 5: the caveat is the first thing a reader sees."""
        a, b = self._v1_pair()
        run_dir = self.tmp / "v1_caveat"
        proc = self._run(
            ["--compare", str(a), str(b), "--v1-per-item", "--run-dir", str(run_dir)]
        )
        self.assertIn("PRIOR WORK, NOT A v2 MEASUREMENT", proc.stdout)
        lines = [
            line
            for line in (run_dir / "compare_notes.md").read_text(encoding="utf-8").splitlines()
            if line.startswith("- ")
        ]
        self.assertIn("PRIOR WORK, NOT A v2 MEASUREMENT", lines[0])

    def test_duplicate_questions_are_refused_under_question_alignment(self) -> None:
        """Two rows with the same question cannot be paired unambiguously."""
        a, _ = self._v1_pair()
        rows = json.loads(a.read_text(encoding="utf-8"))
        duplicated = self.tmp / "v1_duplicate.json"
        duplicated.write_text(
            json.dumps(rows + [dict(rows[0])], ensure_ascii=False, indent=2), encoding="utf-8"
        )
        proc = self._run(
            [
                "--compare", str(duplicated), str(duplicated), "--v1-per-item",
                "--v1-alignment", "normalized_question",
                "--run-dir", str(self.tmp / "v1_dupe"),
            ],
            expect_ok=False,
        )
        self.assertNotEqual(proc.returncode, 0)
        message = proc.stdout + proc.stderr
        self.assertIn("duplicate alignment key", message)
        self.assertIn("--v1-alignment position", message)

    def test_positional_alignment_pairs_rows_in_file_order(self) -> None:
        """Position mode works, and records that it verified each pair is the same item."""
        metrics = self._compare_v1("v1_position", ["--v1-alignment", "position"])
        block = metrics["v1_format_inputs"]
        self.assertEqual(block["alignment"], "position")
        self.assertIn("file order", block["alignment_definition"])
        check = block["same_item_check"]
        # Position ids say nothing about the item, so both fields witness the pairing here.
        self.assertEqual(check["verification_fields"], ["question", "gold_steps"])
        self.assertEqual(check["tautological_fields"], [])
        self.assertEqual(check["fields_verified_equal"], {"question": 4, "gold_steps": 4})

    def test_positional_alignment_refuses_misaligned_files(self) -> None:
        """Row i of one file holding a different question than row i of the other aborts.

        This is the check that makes positional pairing safe: without it, reversing one
        file's row order would silently produce a comparison between different items.
        """
        a, b = self._v1_pair()
        reversed_b = self.tmp / "v1_reversed.json"
        rows = json.loads(b.read_text(encoding="utf-8"))
        reversed_b.write_text(
            json.dumps(rows[::-1], ensure_ascii=False, indent=2), encoding="utf-8"
        )
        proc = self._run(
            [
                "--compare", str(a), str(reversed_b), "--v1-per-item",
                "--v1-alignment", "position", "--run-dir", str(self.tmp / "v1_misaligned"),
            ],
            expect_ok=False,
        )
        self.assertNotEqual(proc.returncode, 0)
        message = proc.stdout + proc.stderr
        self.assertIn("not the same evaluation item", message)
        self.assertIn("question differs", message)

    def test_alignment_flag_requires_the_v1_flag(self) -> None:
        """--v1-alignment is meaningless for v2 artifacts, which align on item_id."""
        per_item = self._arm_a_per_item_for_v1_test()
        proc = self._run(
            [
                "--compare", str(per_item), str(per_item),
                "--v1-alignment", "position", "--run-dir", str(self.tmp / "v1_flagless"),
            ],
            expect_ok=False,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("requires --v1-per-item", proc.stdout + proc.stderr)

    def test_writing_into_the_prior_work_repo_is_refused(self) -> None:
        """ADR 0020 condition 1: the v1 tree is read-only, enforced in source.

        Checked against the guard directly, with a fabricated root: the point is that no
        run_dir under the configured read-only root is accepted, and a test must not create
        a directory there to prove it.
        """
        root = "/fabricated/prior/work/repo"
        for run_dir in (Path(root), Path(root) / "runs" / "out"):
            with self.subTest(run_dir=str(run_dir)):
                with self.assertRaises(SystemExit) as caught:
                    EVAL._refuse_writing_into_prior_work(run_dir, root)
                self.assertIn("read-only", str(caught.exception))
        # Anywhere else is fine.
        EVAL._refuse_writing_into_prior_work(self.tmp / "somewhere", root)

    # ---- negative controls from the PR #36 Gate-1 review (I-1, N3, N5) ----

    #: A distinctive string standing in for dataset question text, so a test can assert it
    #: never reaches an error message.
    FABRICATED_QUESTION = "ZZQuestionTextThatMustNotAppearInAnyErrorMessage"

    def _synthetic_v1(
        self,
        name: str,
        rows: int,
        *,
        with_question: bool,
        with_gold: bool,
        offset: float = 0.0,
        gold_prefix: str = "step",
        drop_last: bool = False,
        null_metric_row: int | None = None,
    ) -> Path:
        """A v1-shaped bare list built field by field, so a field can be left out.

        The evaluator's own per-item files always carry `question` and `gold_steps`, which is
        exactly why the vacuous case has to be constructed by hand.
        """
        out = []
        for i in range(rows - (1 if drop_last else 0)):
            row: dict[str, Any] = {
                "exact_match": float(i % 2),
                "step_f1": min(1.0, 0.1 * (i % 7) + offset),
                "ordered_step_accuracy": min(1.0, 0.05 * (i % 5) + offset),
                "rouge_l_f1": min(1.0, 0.2 + 0.03 * (i % 9) + offset),
                "reference_valid_count": 0,
                "reference_total_count": 0,
                "step_count_abs_error": i % 3,
                "hop_count_exact_match": float((i + 1) % 2),
            }
            if with_question:
                row["question"] = f"{self.FABRICATED_QUESTION} number {i}?"
            if with_gold:
                row["gold_steps"] = [f"{gold_prefix} {i}.1", f"{gold_prefix} {i}.2"]
            if null_metric_row == i:
                row["step_f1"] = None
            out.append(row)
        path = self.tmp / f"v1syn_{name}.json"
        path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def _refuse(self, argv: list[str], run_dir_name: str) -> str:
        proc = self._run(
            [*argv, "--run-dir", str(self.tmp / run_dir_name)], expect_ok=False
        )
        self.assertNotEqual(proc.returncode, 0, "expected a refusal, got success")
        return proc.stdout + proc.stderr

    def test_unverifiable_pairing_is_refused_not_reported(self) -> None:
        """I-1 negative control: two v1 files with no witness field must not compare.

        The Gate-1 review's demonstration: two unrelated 40-row v1 files carrying neither
        `question` nor `gold_steps` produced rc=0, a full battery, and a run note claiming
        "every pair verified to be the same item ({'question': 0, 'gold_steps': 0})". With no
        id and no witness field there is nothing that says row i of one file is the same
        evaluation item as row i of the other, so the comparison is refused.
        """
        a = self._synthetic_v1("vac_a", 40, with_question=False, with_gold=False)
        b = self._synthetic_v1("vac_b", 40, with_question=False, with_gold=False, offset=0.05)
        message = self._refuse(
            ["--compare", str(a), str(b), "--v1-per-item", "--v1-alignment", "position"],
            "v1_vacuous",
        )
        self.assertIn("nothing can verify that the paired rows are the same", message)
        self.assertIn("missing: ['question', 'gold_steps']", message)
        self.assertFalse((self.tmp / "v1_vacuous" / "compare_metrics.json").exists())

    def test_the_alignment_key_alone_cannot_witness_the_pairing(self) -> None:
        """I-1: under normalized_question, matching questions verify nothing.

        The question IS the key rows were grouped by, so equality is true by construction. A
        file carrying only that field has no independent witness and is refused.
        """
        a = self._synthetic_v1("taut_a", 40, with_question=True, with_gold=False)
        b = self._synthetic_v1("taut_b", 40, with_question=True, with_gold=False, offset=0.05)
        message = self._refuse(
            [
                "--compare", str(a), str(b), "--v1-per-item",
                "--v1-alignment", "normalized_question",
            ],
            "v1_tautological",
        )
        self.assertIn("nothing can verify that the paired rows are the same", message)
        self.assertIn("true by construction", message)

    def test_positional_alignment_requires_the_question_and_the_gold(self) -> None:
        """I-1: positional alignment needs both witness fields, per ADR 0020 condition 3(b).

        That condition (amended 2026-08-20 by PR #35) makes positional alignment admissible
        only when the alignment field *and* the gold are asserted equal on every matched row,
        so a file carrying only the gold is refused rather than compared on half the check.
        """
        a = self._synthetic_v1("wit_a", 40, with_question=False, with_gold=True)
        b = self._synthetic_v1("wit_b", 40, with_question=False, with_gold=True, offset=0.05)
        message = self._refuse(
            ["--compare", str(a), str(b), "--v1-per-item", "--v1-alignment", "position"],
            "v1_one_witness",
        )
        self.assertIn("missing: ['question']", message)
        self.assertIn("ADR 0020 condition 3", message)

    def test_the_gold_is_the_witness_under_question_alignment(self) -> None:
        """I-1 positive control: with the question as the key, the gold verifies the pairing."""
        a = self._synthetic_v1("goldwit_a", 40, with_question=True, with_gold=True)
        b = self._synthetic_v1("goldwit_b", 40, with_question=True, with_gold=True, offset=0.05)
        run_dir = self.tmp / "v1_gold_witness"
        self._run(
            [
                "--compare", str(a), str(b), "--v1-per-item",
                "--v1-alignment", "normalized_question", "--run-dir", str(run_dir),
            ]
        )
        metrics = json.loads((run_dir / "compare_metrics.json").read_text(encoding="utf-8"))
        check = metrics["v1_format_inputs"]["same_item_check"]
        self.assertEqual(check["verification_fields"], ["gold_steps"])
        self.assertEqual(check["tautological_fields"], ["question"])
        self.assertEqual(check["fields_verified_equal"], {"gold_steps": 40})
        note = (run_dir / "compare_notes.md").read_text(encoding="utf-8")
        self.assertIn("all 40 pairs verified to be the same evaluation item on `gold_steps`", note)

    def test_the_verification_sentence_is_built_from_the_counts(self) -> None:
        """I-1: the note never asserts verification unconditionally.

        Checked at the formatter, because the refusal above means the run cannot reach the
        note with nothing verified — and this is the guard that keeps that true if it ever can.
        """
        self.assertIn(
            "NOT VERIFIED",
            EVAL._v1_verification_sentence(
                {"alignment": "position", "pairs": 40, "verification_fields": [],
                 "fields_verified_equal": {}}
            ),
        )
        self.assertIn(
            "NOT VERIFIED",
            EVAL._v1_verification_sentence(
                {"alignment": "position", "pairs": 40, "verification_fields": ["gold_steps"],
                 "fields_verified_equal": {"gold_steps": 39}}
            ),
        )
        self.assertIn(
            "all 40 pairs verified",
            EVAL._v1_verification_sentence(
                {"alignment": "position", "pairs": 40, "verification_fields": ["gold_steps"],
                 "fields_verified_equal": {"gold_steps": 40}}
            ),
        )

    def test_refusals_never_print_dataset_question_text(self) -> None:
        """N3: v1 error messages identify a row by index + key hash, never by content.

        Under `normalized_question` the alignment key *is* a dataset question, and error text
        gets pasted into issues and PRs — which would move data into git (CLAUDE.md). All
        three refusal paths that name ids are covered: id-set mismatch, duplicate key, and a
        same-item mismatch.
        """
        a = self._synthetic_v1("leak_a", 5, with_question=True, with_gold=True)
        short = self._synthetic_v1("leak_short", 5, with_question=True, with_gold=True, drop_last=True)
        other_gold = self._synthetic_v1(
            "leak_gold", 5, with_question=True, with_gold=True, gold_prefix="different"
        )
        duplicated = self.tmp / "v1syn_leak_dup.json"
        rows = json.loads(a.read_text(encoding="utf-8"))
        duplicated.write_text(
            json.dumps(rows + [dict(rows[0])], ensure_ascii=False, indent=2), encoding="utf-8"
        )

        cases = {
            "id_mismatch": (a, short),
            "duplicate_key": (duplicated, duplicated),
            "same_item_mismatch": (a, other_gold),
        }
        for name, (left, right) in cases.items():
            with self.subTest(refusal=name):
                message = self._refuse(
                    [
                        "--compare", str(left), str(right), "--v1-per-item",
                        "--v1-alignment", "normalized_question",
                    ],
                    f"v1_leak_{name}",
                )
                self.assertNotIn(self.FABRICATED_QUESTION, message)
                self.assertIn("sha256:", message)
                self.assertIn("row ", message)

    def test_a_null_metric_field_is_refused_with_a_clear_message(self) -> None:
        """N5: a JSON null in a compared column aborts at load, not inside the statistics."""
        path = self._synthetic_v1(
            "nullmetric", 40, with_question=True, with_gold=True, null_metric_row=7
        )
        message = self._refuse(
            ["--compare", str(path), str(path), "--v1-per-item", "--v1-alignment", "position"],
            "v1_nullmetric",
        )
        self.assertIn("row 7", message)
        self.assertIn("step_f1", message)
        self.assertIn("non-numeric or non-finite", message)
        self.assertNotIn("TypeError", message)

    def test_prior_work_write_guard_is_not_gated_on_the_v1_flag(self) -> None:
        """N1: ADR 0020 condition 1 is unconditional, so the guard cannot sit under the flag.

        Checked on the source (the AST pattern of tests/test_generation_contract.py) rather
        than by pointing a real --run-dir at the read-only repo: a regression in that test
        would itself write into the tree the rule protects.
        """
        tree = ast.parse(EVALUATOR.read_text(encoding="utf-8"))
        compare_fn = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_compare"
        )
        top_level_calls = {
            sub.func.id
            for stmt in compare_fn.body
            for sub in ast.walk(stmt)
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)
            # Only statements that are not conditional: an Expr call in the function body.
            if isinstance(stmt, ast.Expr)
        }
        self.assertIn(
            "_refuse_writing_into_prior_work",
            top_level_calls,
            "_refuse_writing_into_prior_work must be called unconditionally in _compare, "
            "not inside the --v1-per-item branch",
        )

    def test_configured_alignment_default_is_the_analysis_note_alignment(self) -> None:
        """The default reproduces the committed analysis note, whose Task A sorted on it."""
        config = json.loads(
            (REPO_ROOT / "configs" / "musique_eval.json").read_text(encoding="utf-8")
        )
        v1_cfg = config["paired_comparison"]["v1_compat"]
        self.assertEqual(v1_cfg["default_alignment"], "normalized_question")
        self.assertIn(v1_cfg["default_alignment"], EVAL.V1_ALIGNMENTS)
        self.assertTrue(v1_cfg["read_only_prior_work_root"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
