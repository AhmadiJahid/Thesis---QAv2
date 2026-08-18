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

import importlib.util
import json
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
