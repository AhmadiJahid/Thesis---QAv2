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
import copy
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


class TestBreakMetrics(EvaluatorTestBase):
    """Hand-computed checks for the Break-faithful trio and the repaired chain validity.

    Every expected number is derived in the docstring from the official semantics
    (``allenai/break-evaluator``: ``get_exact_match``, ``sari_hook.py``,
    ``graph_matcher.normalized_graph_edit_distance``) and the fabricated fixture gold. The
    conventions these pin — including the two deviations from the official code and the
    fallback policy — are ADR 0026.

    Note the fixture gold writes references as ``[#k]`` while real MuSiQue gold writes bare
    ``#k``. Both are matched by the ``#(\\d+)`` rule these metrics use (Break's own rule), so
    the arithmetic below is the same either way; the bracket characters only ride along as
    part of the step text.
    """

    def test_identical_prediction_scores_the_ceiling_on_every_new_metric(self) -> None:
        """2hop__d001_a predicted exactly as gold: EM 1, SARI 1, GED 0, chain validity 1.

        Hand computation (1 evaluated row):
          break EM   the two ' @@SEP@@ '-joined strings are the same string -> 1.0
          SARI       prediction == target, so for every n: keep tp = selected = relevant
                     (F1 1.0), add tp = selected = relevant (F1 1.0), delete tp = selected
                     (precision 1.0) -> (1 + 1 + 1)/3 = 1.0
          GED        the graphs are identical (2 nodes, edge 2->1, equal labels), so the
                     minimum edit path is empty: 0 / max(3, 3) = 0.0
          chain      gold emits 1 reference so chaining is required; the prediction's [#1]
                     in step 2 is backward -> 1/1 = 1.0
          fallbacks  a 2-node graph is far under the node cap and the optimizer returns in
                     microseconds -> ged_fallback_counts is empty
        """
        item = "2hop__d001_a"
        metrics, per_item = self.evaluate("break_identical", [prediction(item, gold_steps(item))])

        self.assertMetrics(
            metrics,
            {
                "break_exact_match_rate": 1.0,
                "sari_macro": 1.0,
                "ged_macro": 0.0,
                "chain_validity_macro": 1.0,
            },
        )
        self.assertEqual(metrics["ged_fallback_counts"], {})
        # Per gold hop, exactly like every existing metric.
        self.assertMetrics(
            metrics["per_gold_hop_metrics"]["2"],
            {"break_exact_match_rate": 1.0, "sari_macro": 1.0, "ged_macro": 0.0,
             "chain_validity_macro": 1.0},
        )
        row = json.loads(per_item.read_text(encoding="utf-8"))["items"][0]
        self.assertIsNone(row["ged_fallback"])
        self.assertEqual(row["chain_pred_reference_count"], 1)
        self.assertEqual(row["chain_gold_reference_count"], 1)

    def test_reversed_prediction_is_punished_where_step_f1_is_blind(self) -> None:
        """3hop1__d002_b predicted with its gold steps in reverse order.

        This is the survey's order-blindness probe (§3.7): the content is perfect and only
        the order is wrong, so a set-based metric cannot see it at all.

        Hand computation (1 evaluated row):
          step F1    the pred and gold step SETS are identical -> P = R = F1 = 1.0. Wholly
                     blind, which is the point of the comparison below
          ordered    only step 2 sits at its gold index -> 1/3
          break EM   the joined strings differ -> 0.0
          GED        pred graph: step1 has '#2' -> edge (1,2); step2 has '#1' -> edge (2,1);
                     3 nodes, 2 edges. gold graph: edges (2,1) and (3,2); 3 nodes, 2 edges.
                     normalization = max(3+2, 3+2) = 5.
                     Cheapest edit path maps pred 1->gold 3, 2->2, 3->1 (labels then match
                     exactly, so all three substitutions cost 0); the pred edges become
                     (3,2) and (2,3); (3,2) is in the gold, (2,3) is not -> 1 deletion +
                     1 insertion of (2,1) = 2. The identity mapping costs more (2 label
                     substitutions of unrelated steps ~1 each, plus 2 edge edits), so the
                     minimum is 2 -> 2/5 = 0.4
          chain      pred step 1 references #2 (needs 1 <= 2 < 1: invalid), step 2
                     references #1 (valid), step 3 none -> 1/2 = 0.5
        """
        item = "3hop1__d002_b"
        metrics, _ = self.evaluate(
            "break_reversed", [prediction(item, list(reversed(gold_steps(item))))]
        )

        self.assertMetrics(
            metrics,
            {
                "step_f1_macro": 1.0,
                "ordered_step_accuracy_macro": 1 / 3,
                "break_exact_match_rate": 0.0,
                "ged_macro": 0.4,
                "chain_validity_macro": 0.5,
            },
        )
        # SARI is an n-gram bag: reversing the steps moves only the n-grams that straddle a
        # step boundary, so it stays close to 1.0. Asserted as a bound rather than a digit
        # because the exact value is a 1-to-4-gram count over 30+ tokens; the point being
        # pinned is the blindness, not the decimal (survey §3.7: SARI 0.9414 on this probe
        # over the real 600).
        self.assertGreater(metrics["sari_macro"], 0.9)

    def test_two_step_reversal_hits_the_inherited_self_loop_blind_spot(self) -> None:
        """A 2-step reversal scores GED 0.0 — networkx's self-loop pricing, kept on purpose.

        Hand computation (1 evaluated row), and the reason this number is 0.0 and not 2/3:
          pred graph  reversing 2 steps puts '#1' in step 1, so the reference becomes a
                      SELF-LOOP: nodes 1,2, edge (1,1)
          gold graph  nodes 1,2, edge (2,1)
          true GED    map pred 1->gold 2 and pred 2->gold 1 (labels then match, cost 0); the
                      pred edge (1,1) becomes (2,2), which the gold does not have, so 1
                      deletion + 1 insertion of (2,1) = 2 -> 2 / max(3,3) = 2/3
          reported    0.0, because networkx's edit-path search pairs the self-loop (1,1) with
                      the ordinary edge (2,1) at cost 0
        The official Break evaluator computes GED with this same networkx call, so this is
        the ported metric's behaviour and it is pinned rather than "fixed" — fixing it would
        be inventing a metric (ADR 0026). Break EM still catches the reversal, which is why
        the survey's §4 item 2 says an order-sensitive metric has to stay in the reported set.
        """
        item = "2hop__d001_a"
        metrics, _ = self.evaluate(
            "break_reversed_two", [prediction(item, list(reversed(gold_steps(item))))]
        )

        self.assertMetrics(
            metrics,
            {
                "ged_macro": 0.0,
                # EM sees it, and so does the ordered metric: 0 positional matches of 2.
                "break_exact_match_rate": 0.0,
                "exact_match_rate": 0.0,
                "ordered_step_accuracy_macro": 0.0,
                # pred step 1 references #1 (needs 1 <= 1 < 1: invalid), step 2 has none.
                "chain_validity_macro": 0.0,
            },
        )

    def test_over_long_prediction_pays_one_node_deletion(self) -> None:
        """2hop__d001_a predicted as its gold plus one extra, reference-free step.

        Hand computation (1 evaluated row):
          break EM   the joined strings differ by the extra step -> 0.0
          GED        pred graph: 3 nodes, 1 edge (2->1) = 4; gold: 2 nodes, 1 edge = 3;
                     normalization = max(4, 3) = 4. Map pred 1->1 and 2->2 (labels equal,
                     cost 0), delete node 3 (cost 1); the pred edge (2,1) is the gold edge,
                     so it is free. Total 1 -> 1/4 = 0.25. It cannot be cheaper: the node
                     counts differ by 1, so at least one deletion is unavoidable
          chain      1 reference, backward -> 1.0 (over-decomposition is not a chaining
                     error, and this metric does not price length)
          steps      signed +1, so the directional family sees what GED prices as a deletion
        """
        item = "2hop__d001_a"
        steps = gold_steps(item) + ["Which city is the union based in?"]
        metrics, _ = self.evaluate("break_over_long", [prediction(item, steps)])

        self.assertMetrics(
            metrics,
            {
                "break_exact_match_rate": 0.0,
                "ged_macro": 0.25,
                "chain_validity_macro": 1.0,
                "mean_signed_step_count_error": 1.0,
                "over_decomposition_rate": 1.0,
            },
        )

    def test_empty_prediction_is_a_maximal_distance_and_zero_chain_validity(self) -> None:
        """An empty decomposition against the 2-step gold of 2hop__d001_a.

        Hand computation (1 evaluated row):
          break EM   '' != the gold string -> 0.0
          GED        the prediction graph is empty, so the only edit path inserts the gold's
                     2 nodes and 1 edge: 3 / max(0, 3) = 1.0. Computed rather than searched
                     (the optimizer has nothing to align), which is the official formula's
                     value for this degenerate input
          chain      the gold emits a reference and the prediction emits none, so 0.0 — where
                     reference_validity gives the same row 1.0. This is the repair issue #40
                     asked for, and the two numbers are reported side by side
          SARI       source  = the question, 10 tokens ('the' twice), 9 unique unigrams
                     target  = the gold's joined string, 11 tokens, all unique
                     pred    = ''.split(' ') = [''], one token — so there are no pred
                              bigrams/trigrams/4-grams at all
                     keep    tp = 0 with relevant > 0 -> recall 0 -> F1 0, for every n
                     add     tp = 0 with relevant > 0 -> 0, for every n
                     delete  precision only (beta = 0) = |source n-grams NOT in target| /
                             |source n-grams|, since the prediction deletes everything:
                             n=1  1/9 ('that' is the only source unigram absent from gold)
                             n=2  4/9   n=3  5/8   n=4  5/7
                     SARI    ((0) + (0) + (1/9 + 4/9 + 5/8 + 5/7)/4) / 3 = 0.157903...
                     — the floor is well above 0 because SARI rewards deleting question
                     tokens, which is why absolute SARI levels here are not interpretable
        """
        item = "2hop__d001_a"
        pred = prediction(item, [])
        pred["decomposition"] = ""
        metrics, _ = self.evaluate("break_empty", [pred])

        self.assertMetrics(
            metrics,
            {
                "break_exact_match_rate": 0.0,
                "ged_macro": 1.0,
                "chain_validity_macro": 0.0,
                # The old term, unchanged, on the same row: silence still scores 1.0 there.
                "reference_validity_macro": 1.0,
                "reference_validity_micro": 1.0,
                "sari_macro": ((1 / 9 + 4 / 9 + 5 / 8 + 5 / 7) / 4) / 3,
            },
        )

    def test_a_runaway_prediction_uses_the_deterministic_fallback(self) -> None:
        """17 identical junk steps against the 2-step gold: over the node cap, not dropped.

        The config cap is 16 nodes, so the optimizer is never called and the reported value
        is the search-free positional bound. Hand computation (1 evaluated row):
          graphs      pred 17 nodes, 0 edges = 17; gold 2 nodes, 1 edge = 3;
                      normalization = max(17, 3) = 17
          bound       pair nodes in sorted id order: (1,1) and (2,2). 'zzz' shares no token
                      with either gold step, so each substitution costs 1 - 0 = 1 -> 2.
                      Delete the 15 surplus nodes -> 15. No pred edge survives and the gold
                      edge must be inserted -> 1. Total 18 -> 18/17 = 1.058823...
          flag        ged_fallback 'node_cap' on the item and ged_fallback_counts
                      {'node_cap': 1} in the aggregate — the item is reported, never dropped,
                      because a dropped item has no pair for the paired battery (ADR 0026).
                      The cap path is deterministic, so ged_fallback_seconds stays null
        """
        item = "2hop__d001_a"
        metrics, per_item = self.evaluate("break_node_cap", [prediction(item, ["zzz"] * 17)])

        self.assertMetrics(metrics, {"ged_macro": 18 / 17})
        self.assertEqual(metrics["ged_fallback_counts"], {"node_cap": 1})
        row = json.loads(per_item.read_text(encoding="utf-8"))["items"][0]
        self.assertEqual(row["ged_fallback"], "node_cap")
        self.assertIsNone(row["ged_fallback_seconds"])
        self.assertEqual(metrics["ged_policy"]["max_nodes_for_optimizer"], 16)

    def test_the_node_cap_is_a_knob_and_the_optimizer_is_used_under_it(self) -> None:
        """The same two graphs, scored with the cap below and above their size.

        Checked on the function so the knob itself is pinned: with the cap at 2 the 3-node
        prediction takes the positional bound (0.25 here as well — the bound is tight for
        this shape), and with the cap at 30 the optimizer runs and reports no fallback.
        """
        item = "2hop__d001_a"
        pred = EVAL._decomposition_graph(
            EVAL._break_steps(gold_steps(item) + ["Which city is the union based in?"])
        )
        gold = EVAL._decomposition_graph(EVAL._break_steps(gold_steps(item)))

        capped_value, capped_reason, capped_seconds = EVAL._normalized_ged(pred, gold, 2, 20.0)
        self.assertEqual(capped_reason, "node_cap")
        self.assertAlmostEqual(capped_value, 0.25, places=PLACES)
        self.assertIsNone(capped_seconds)

        value, reason, seconds = EVAL._normalized_ged(pred, gold, 16, 20.0)
        self.assertIsNone(reason)
        self.assertIsNone(seconds)
        self.assertAlmostEqual(value, 0.25, places=PLACES)

    def test_break_exact_match_is_stricter_than_the_house_exact_match(self) -> None:
        """2hop__d004_p differs from its gold only by punctuation.

        The house `exact_match` normalizes each step (lowercase, punctuation stripped except
        '#') and scores 1.0; Break's `get_exact_match` lowercases the joined string and
        strips nothing, so it scores 0.0. Both are correct definitions of "exact"; the point
        of pinning them together is that they are not the same metric and a report must not
        read one as the other.
        """
        metrics, _ = self.evaluate(
            "break_punct",
            [
                prediction(
                    "2hop__d004_p",
                    ["Which board approved the Rill Valley permit.", "Who chairs [#1]"],
                )
            ],
        )
        self.assertMetrics(
            metrics, {"exact_match_rate": 1.0, "break_exact_match_rate": 0.0}
        )

    def test_aggregates_are_the_mean_of_the_per_item_columns(self) -> None:
        """The four new aggregates are plain macro averages of their per-item columns.

        Checked over the committed fixture predictions rather than asserted from a docstring:
        an aggregate that drifted from its column is exactly the failure mode that makes a
        per-item metric untestable in the paired battery.
        """
        preds = json.loads(PREDICTIONS_FIXTURE.read_text(encoding="utf-8"))
        metrics, per_item = self.evaluate("break_aggregate", preds)
        items = json.loads(per_item.read_text(encoding="utf-8"))["items"]
        self.assertEqual(len(items), 4)
        for column, aggregate in (
            ("break_exact_match", "break_exact_match_rate"),
            ("sari", "sari_macro"),
            ("ged", "ged_macro"),
            ("chain_validity", "chain_validity_macro"),
        ):
            with self.subTest(metric=column):
                expected = sum(float(row[column]) for row in items) / len(items)
                self.assertAlmostEqual(metrics[aggregate], expected, places=PLACES)


class TestChainValidity(EvaluatorTestBase):
    """The repaired chaining term: bare `#k`, per item, no free credit for silence."""

    def test_a_prediction_that_emits_no_reference_scores_zero_not_one(self) -> None:
        """3hop1__d002_b with every reference written out in words instead.

        Hand computation (1 evaluated row):
          gold refs  2 ([#1] in step 2, [#2] in step 3), so chaining IS required
          pred refs  0 — the steps read 'the river' and 'the country' instead
          chain      0.0 (no free credit)
          old term   reference_validity_macro 1.0 and reference_validity_micro 1.0, because
                     a row with no references scores 1.0 and adds nothing to the micro
                     denominator. Both numbers are reported; neither is changed by this test
        The survey measured this convention flipping a model ranking: 76 of 600 items in one
        arm were paid 1.0 for chaining not at all (§3.2).
        """
        steps = gold_steps("3hop1__d002_b")
        steps[1] = steps[1].replace("[#1]", "the river")
        steps[2] = steps[2].replace("[#2]", "the country")
        metrics, per_item = self.evaluate("chain_no_refs", [prediction("3hop1__d002_b", steps)])

        self.assertMetrics(
            metrics,
            {
                "chain_validity_macro": 0.0,
                "reference_validity_macro": 1.0,
                "reference_validity_micro": 1.0,
            },
        )
        row = json.loads(per_item.read_text(encoding="utf-8"))["items"][0]
        self.assertEqual(row["chain_pred_reference_count"], 0)
        self.assertEqual(row["chain_gold_reference_count"], 2)

    def test_an_invalid_reference_is_priced_by_the_ratio(self) -> None:
        """3hop1__d002_b with step 2 pointing forward at [#3] instead of back at [#1].

        Hand computation: 2 predicted references, step 2's [#3] is not backward (needs
        1 <= 3 < 2) and step 3's [#2] is -> 1/2 = 0.5, the same ratio the old macro term
        reports for this row. The two terms differ only where a prediction is SILENT.
        """
        steps = gold_steps("3hop1__d002_b")
        steps[1] = steps[1].replace("[#1]", "[#3]")
        metrics, _ = self.evaluate("chain_forward_ref", [prediction("3hop1__d002_b", steps)])
        self.assertMetrics(
            metrics, {"chain_validity_macro": 0.5, "reference_validity_macro": 0.5}
        )

    def test_bare_and_bracketed_references_are_both_counted(self) -> None:
        """`#(\\d+)` is Break's own rule: it sees `#1` and the `#1` inside `[#1]`.

        MuSiQue's real gold writes bare `#k` and the fixture writes `[#k]`; a chaining metric
        that saw only one of them would be issue #40 again, in the other direction. Checked
        on the function, on both syntaxes of the same 2-step plan.
        """
        for reference in ("#1", "[#1]"):
            with self.subTest(reference=reference):
                self.assertEqual(
                    EVAL._chain_validity(
                        ["Which union organised the strike?", f"Who leads {reference}?"],
                        ["Which union organised the strike?", "Who leads #1?"],
                    ),
                    (1.0, 1, 1),
                )
        # And the untouched house regex still matches only the bracketed form, which is what
        # issue #40 recorded: this change adds a term, it does not redefine the old one.
        self.assertEqual(EVAL._reference_validity(["a", "b #1"]), (1.0, 0, 0))
        self.assertEqual(EVAL._reference_validity(["a", "b [#1]"]), (1.0, 1, 1))


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
            return {
                "exact_match": exact,
                "hop_count_exact_match": 1.0,
                "break_exact_match": exact,
            }

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

    def _row(
        self, a: list[float], b: list[float], name: str = "step_f1"
    ) -> dict[str, Any]:
        return EVAL._paired_t_test_row(
            np.array(a, dtype=float),
            np.array(b, dtype=float),
            self.ALPHA,
            underpowered=False,
            name=name,
        )

    def test_the_metric_name_is_required_and_sets_the_direction(self) -> None:
        """No default name: a distance must not be labelled higher-is-better by omission.

        The same numbers under two metric names give opposite `favours` verdicts, which is
        precisely why the argument cannot have a default (PR #44 review, nit 7).
        """
        with self.assertRaises(TypeError):
            EVAL._paired_t_test_row(
                np.array([1.0, 1.0, 0.9]), np.array([0.0, 0.0, 0.0]), self.ALPHA, False
            )
        higher = self._row([1.0, 1.0, 0.9], [0.0, 0.0, 0.0], name="step_f1")
        lower = self._row([1.0, 1.0, 0.9], [0.0, 0.0, 0.0], name="ged")
        self.assertEqual(higher["direction"], "higher_is_better")
        self.assertEqual(lower["direction"], "lower_is_better")
        self.assertTrue(higher["significant"] and lower["significant"])
        self.assertEqual(higher["favours"], "system_a")
        self.assertEqual(lower["favours"], "system_b")

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
                    "sari",
                    "ged",
                    "chain_validity",
                    "exact_match",
                    "hop_count_exact_match",
                    "break_exact_match",
                ]
            ),
        )
        # composite_score is bootstrapped but has no per-item value, so no paired difference
        # to t-test exists.
        self.assertIn("composite_score", EVAL.BOOTSTRAP_STATISTICS)
        self.assertNotIn("composite_score", EVAL.T_TEST_STATISTICS)


class TestMetricDirection(unittest.TestCase):
    """GED is a distance, so a comparison has to carry direction, not assume it."""

    def test_ged_is_the_only_lower_is_better_statistic(self) -> None:
        self.assertEqual(EVAL.LOWER_IS_BETTER_STATISTICS, ("ged",))
        self.assertIn("ged", EVAL.BOOTSTRAP_STATISTICS)
        self.assertEqual(EVAL._direction("ged"), "lower_is_better")
        for name in ("step_f1", "sari", "chain_validity", "break_exact_match"):
            with self.subTest(statistic=name):
                self.assertEqual(EVAL._direction(name), "higher_is_better")

    def test_favours_applies_the_direction_to_the_sign(self) -> None:
        """A negative difference favours a on ged and b on everything else.

        This is the misreading the labelling exists to prevent: -0.16 on ged means system a
        is 0.16 closer to the gold graph, i.e. better.
        """
        self.assertEqual(EVAL._favours("ged", -0.16, True), "system_a")
        self.assertEqual(EVAL._favours("ged", +0.16, True), "system_b")
        self.assertEqual(EVAL._favours("step_f1", +0.16, True), "system_a")
        self.assertEqual(EVAL._favours("step_f1", -0.16, True), "system_b")
        # Nothing to favour when the row is not significant, or the difference is 0.
        self.assertIsNone(EVAL._favours("ged", -0.16, False))
        self.assertIsNone(EVAL._favours("ged", 0.0, True))


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
            # The issue #40 columns, so the chunking invariance covers them too.
            "sari": np.clip(base * 0.7 + offset, 0.0, 1.0),
            "ged": np.clip(1.0 - base - offset, 0.0, 2.0),
            "chain_validity": np.clip(base * 0.95 + offset, 0.0, 1.0),
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
            [
                "rouge_l_f1",
                "step_f1",
                "ordered_step_accuracy",
                # The issue #40 additions, each a per-item value and so each bootstrapped.
                "sari",
                "ged",
                "chain_validity",
                "composite_score",
            ]
        ))
        for name, result in metrics["bootstrap"].items():
            with self.subTest(statistic=name):
                self.assertAlmostEqual(result["difference"], 0.0, places=PLACES)
                self.assertAlmostEqual(result["ci_low"], 0.0, places=PLACES)
                self.assertAlmostEqual(result["ci_high"], 0.0, places=PLACES)
                self.assertFalse(result["significant"])
                self.assertEqual(result["n"], 4)
                # Not significant, so there is nothing to favour.
                self.assertIsNone(result["favours"])
        self.assertEqual(
            sorted(metrics["mcnemar"]),
            ["break_exact_match", "exact_match", "hop_count_exact_match"],
        )
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

    def test_a_null_or_nan_in_a_new_column_is_refused_at_load(self) -> None:
        """PR #44 review, I1: the v2 loader's finite-number gate covers the new columns.

        Before the fix the gate iterated the v1 field list, which excludes the issue #40
        columns — so a `"ged": null` died as a raw TypeError inside the statistics and a
        `"ged": NaN` travelled through the entire battery and blew up formatting the run
        note. Both must be refused at load, naming the file, the row and the field. `NaN` is
        written here the way Python's json writer emits it (not valid JSON, which is exactly
        why nothing may rely on a downstream reader catching it).
        """
        per_item = self._per_item_of_fixture()
        payload = json.loads(per_item.read_text(encoding="utf-8"))
        for name, field, value in (
            ("ged_null", "ged", None),
            ("ged_nan", "ged", float("nan")),
            ("sari_string", "sari", "0.9"),
            ("chain_null", "chain_validity", None),
            ("break_em_bool", "break_exact_match", True),
        ):
            with self.subTest(case=name):
                tampered = copy.deepcopy(payload)
                tampered["items"][2][field] = value
                path = self.tmp / f"compare_{name}_per_item.json"
                path.write_text(
                    json.dumps(tampered, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                run_dir = self.tmp / f"compare_{name}"
                proc = self._run(
                    ["--compare", str(path), str(per_item), "--run-dir", str(run_dir)],
                    expect_ok=False,
                )
                self.assertNotEqual(proc.returncode, 0)
                message = proc.stdout + proc.stderr
                self.assertIn("row 2", message)
                self.assertIn(field, message)
                self.assertIn("non-numeric or non-finite", message)
                self.assertNotIn("TypeError", message)
                # Nothing was reported: the refusal is before any statistic is computed.
                self.assertFalse((run_dir / "compare_metrics.json").exists())

    def test_the_v2_numeric_gate_covers_every_compared_column(self) -> None:
        """The gate's field set is the v2 one, not the v1 one — checked on the constants.

        The regression this pins is a set-difference bug, so it is asserted as a set: every
        compared statistic (and every column the composite is rebuilt from) must be in the
        gate, and `item_id` must not be, because it is a string.
        """
        gate = set(EVAL._NUMERIC_PER_ITEM_FIELDS)
        self.assertNotIn("item_id", gate)
        for name in EVAL._ISSUE_40_STATISTICS:
            self.assertIn(name, gate)
        for name in EVAL.MCNEMAR_STATISTICS:
            self.assertIn(name, gate)
        for name in EVAL.BOOTSTRAP_STATISTICS:
            if name != "composite_score":
                self.assertIn(name, gate)
        for name in EVAL._COMPOSITE_INPUT_COLUMNS:
            self.assertIn(name, gate)
        # The v1 gate is the same set minus the columns v1 could not have written.
        self.assertEqual(
            gate - set(EVAL._REQUIRED_V1_PER_ITEM_FIELDS), set(EVAL._ISSUE_40_STATISTICS)
        )

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
                    "sari",
                    "ged",
                    "chain_validity",
                    "exact_match",
                    "hop_count_exact_match",
                    "break_exact_match",
                ]
            ),
        )
        self.assertNotIn("composite_score", metrics["t_test"])
        self.assertEqual(
            metrics["tests_reported"],
            {
                "bootstrap": 7,
                "mcnemar": 3,
                "paired_t_test": 9,
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

    def test_every_row_states_its_direction_and_the_note_says_it_too(self) -> None:
        """No row can be read without its direction, and the run note prints the column.

        The failure this guards is arithmetic-free: `ged` is the only distance in the report,
        so a table that drops the direction turns "system a is 0.16 better" into "0.16
        worse". Every row therefore carries `direction`, every significant row names the
        system it `favours` with the direction already applied, and the note has a `better`
        column plus a sentence saying which way `ged` reads.
        """
        run_dir, metrics = self._degraded_comparison()
        for family in ("bootstrap", "mcnemar", "t_test"):
            for name, row in metrics[family].items():
                with self.subTest(family=family, statistic=name):
                    self.assertEqual(
                        row["direction"],
                        "lower_is_better" if name == "ged" else "higher_is_better",
                    )
                    if not row["significant"]:
                        self.assertIsNone(row["favours"])
                    else:
                        difference = row["difference"]
                        a_better = difference < 0 if name == "ged" else difference > 0
                        self.assertEqual(
                            row["favours"], "system_a" if a_better else "system_b"
                        )
        self.assertEqual(metrics["lower_is_better_statistics"], ["ged"])
        # v2 inputs carry every compared column, so nothing is skipped.
        self.assertEqual(metrics["statistics_not_available_in_inputs"], [])

        note = (run_dir / "compare_notes.md").read_text(encoding="utf-8")
        self.assertIn("| statistic | better |", note)
        self.assertIn("| ged | lower |", note)
        self.assertIn("graph edit **distance**", note)

    def test_the_degraded_arm_is_worse_on_every_new_metric(self) -> None:
        """Direction sanity on real rows: A is perfect, B is degraded.

        Not a significance claim (n = 4 is below the reporting floor) — a check that the
        point estimates move the way the definitions say they must: the perfect arm scores
        higher on SARI, chain validity and Break EM, and LOWER on ged, because ged is a
        distance.
        """
        _, metrics = self._degraded_comparison()
        for name in ("sari", "chain_validity"):
            with self.subTest(statistic=name):
                self.assertGreater(metrics["bootstrap"][name]["difference"], 0.0)
        self.assertLess(metrics["bootstrap"]["ged"]["difference"], 0.0)
        self.assertGreaterEqual(metrics["mcnemar"]["break_exact_match"]["difference"], 0.0)

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

    #: Fields a v1 per-item row cannot have: ``item_id`` (v1 had no concept of it) and every
    #: column added by issue #40 (v1 ran years before those metrics existed). Both are
    #: dropped when a v1 file is reconstructed here, so the shim is exercised on rows the
    #: shape v1 actually wrote.
    NOT_IN_V1 = (
        "item_id",
        "break_exact_match",
        "sari",
        "ged",
        "ged_fallback",
        "chain_validity",
        "chain_pred_reference_count",
        "chain_gold_reference_count",
    )

    def _v1_file(self, name: str, predictions: list[dict[str, Any]]) -> Path:
        _, per_item = self.evaluate(f"v1src_{name}", predictions)
        payload = json.loads(per_item.read_text(encoding="utf-8"))
        rows = [
            {k: v for k, v in row.items() if k not in self.NOT_IN_V1}
            for row in payload["items"]
        ]
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
        # v1 rows predate the issue #40 columns, so the v1 comparison covers a SUBSET of the
        # v2 one. Every statistic it does report must match; the ones it cannot are asserted
        # separately (test_v1_inputs_record_the_metrics_they_cannot_carry).
        for name, row in v1["bootstrap"].items():
            with self.subTest(bootstrap=name):
                for key in ("system_a", "system_b", "difference"):
                    self.assertAlmostEqual(row[key], v2["bootstrap"][name][key], places=PLACES)
        for family in ("mcnemar", "t_test"):
            for name, row in v1[family].items():
                with self.subTest(family=family, statistic=name):
                    self.assertEqual(sorted(row), sorted(v2[family][name]))
                    for key, value in row.items():
                        expected = v2[family][name][key]
                        if isinstance(value, float):
                            self.assertAlmostEqual(value, expected, places=PLACES)
                        else:
                            self.assertEqual(value, expected)

    def test_v1_inputs_record_the_metrics_they_cannot_carry(self) -> None:
        """A v1 file predates sari / ged / chain_validity / break_exact_match.

        Requiring them would refuse every v1 file and retire the ADR 0020 path; computing
        them from a v1 file's stored steps would be a re-score of v1 output rather than a
        comparison of what v1 measured. So they are omitted, named in the metrics JSON and
        named in the run note (ADR 0026 item 10).
        """
        a, b = self._v1_pair()
        run_dir = self.tmp / "v1_missing_metrics"
        self._run(["--compare", str(a), str(b), "--v1-per-item", "--run-dir", str(run_dir)])
        metrics = json.loads((run_dir / "compare_metrics.json").read_text(encoding="utf-8"))

        self.assertEqual(
            metrics["statistics_not_available_in_inputs"],
            sorted(["sari", "ged", "chain_validity", "break_exact_match"]),
        )
        for name in ("sari", "ged", "chain_validity"):
            self.assertNotIn(name, metrics["bootstrap"])
        self.assertNotIn("break_exact_match", metrics["mcnemar"])
        # The legacy battery is untouched: 4 bootstrap intervals, 2 McNemar, 5 t-tests.
        self.assertEqual(metrics["tests_reported"]["bootstrap"], 4)
        self.assertEqual(metrics["tests_reported"]["mcnemar"], 2)
        self.assertEqual(metrics["tests_reported"]["paired_t_test"], 5)
        self.assertIn(
            "NOT COMPARED", (run_dir / "compare_notes.md").read_text(encoding="utf-8")
        )

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


class TestGedConfigKnobs(unittest.TestCase):
    """GED's two cost guards are config, not literals in the source (ADR 0026 item 4)."""

    def test_the_config_declares_both_guards_and_explains_them(self) -> None:
        config = json.loads(
            (REPO_ROOT / "configs" / "musique_eval.json").read_text(encoding="utf-8")
        )
        block = config["break_metrics"]
        self.assertIn("_note", block)
        ged = block["ged"]
        # At or above the capped decomposer arm's 8-step budget, so ordinary predictions are
        # scored by the optimizer and not by the fallback.
        self.assertGreaterEqual(ged["max_nodes_for_optimizer"], 8)
        self.assertGreater(ged["per_item_time_budget_seconds"], 0.0)
        # There is deliberately no on/off switch: the metrics are additive columns.
        self.assertNotIn("enabled", block)

    def test_a_bad_knob_aborts_at_load_naming_the_key(self) -> None:
        """The code path, not just the committed values (PR #44 review, nit 6).

        `_ged_policy` is what the scoring run reads its guards through, so the refusals are
        checked there: a cap below the 8-step floor, a non-integer cap, and a non-positive or
        non-finite budget each abort with the config key in the message. This is the contract
        `gold_validation` already has — a parameter that would quietly change what the metric
        measures is refused at load.
        """
        def cfg(max_nodes: Any, budget: Any) -> dict[str, Any]:
            return {
                "_config_path": "configs/musique_eval.json",
                "break_metrics": {
                    "ged": {
                        "max_nodes_for_optimizer": max_nodes,
                        "per_item_time_budget_seconds": budget,
                    }
                },
            }

        good = EVAL._ged_policy(cfg(16, 20.0))
        self.assertEqual(good["max_nodes_for_optimizer"], 16)
        self.assertEqual(good["per_item_time_budget_seconds"], 20.0)
        # The floor itself is admissible; one below it is not.
        self.assertEqual(EVAL._ged_policy(cfg(8, 0.5))["max_nodes_for_optimizer"], 8)

        for max_nodes, budget, expected_key in (
            (7, 20.0, "max_nodes_for_optimizer"),
            (0, 20.0, "max_nodes_for_optimizer"),
            (16.5, 20.0, "max_nodes_for_optimizer"),
            (True, 20.0, "max_nodes_for_optimizer"),
            ("16", 20.0, "max_nodes_for_optimizer"),
            (16, 0, "per_item_time_budget_seconds"),
            (16, -1.0, "per_item_time_budget_seconds"),
            (16, None, "per_item_time_budget_seconds"),
            (16, float("inf"), "per_item_time_budget_seconds"),
        ):
            with self.subTest(max_nodes=max_nodes, budget=budget):
                with self.assertRaises(SystemExit) as caught:
                    EVAL._ged_policy(cfg(max_nodes, budget))
                self.assertIn(
                    f"break_metrics.ged.{expected_key}", str(caught.exception)
                )

    def test_a_missing_knob_is_a_missing_config_key_not_a_default(self) -> None:
        """No silent default: an absent knob is `require`'s ConfigError (a SystemExit)."""
        with self.assertRaises(SystemExit) as caught:
            EVAL._ged_policy({"_config_path": "x", "break_metrics": {"ged": {}}})
        self.assertIn("max_nodes_for_optimizer", str(caught.exception))

    def test_no_ged_parameter_is_hard_coded_in_the_scoring_path(self) -> None:
        """The scoring path reads both guards through require(), not from a literal."""
        source = EVALUATOR.read_text(encoding="utf-8")
        for key in ("max_nodes_for_optimizer", "per_item_time_budget_seconds"):
            with self.subTest(key=key):
                self.assertIn(f'require(cfg, "break_metrics.ged.{key}")', source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
