#!/usr/bin/env python3
"""The two beside-card terms of ADR 0039 (issue #60): ``step_f1_soft`` and ``plan_connectedness``.

No GPU, no weights. Every expected value is computed by hand in the comment beside it, under
the committed stopword list (NLTK English, normalised the way a step is). One test reads the
real MuSiQue gold and the ADR 0007 pinned 600 ids when they are on disk and is skipped when
they are not.

Run::

    .venv/bin/python -m unittest tests.test_beside_card_metrics
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
sys.path.insert(0, str(REPO_ROOT / "src"))

from run_config import load_config, load_paths, require, resolve_path  # noqa: E402

EVALUATOR = REPO_ROOT / "scripts" / "musique_decompositions_evaluator.py"
FIXTURES = REPO_ROOT / "tests" / "fixtures"
GOLD_FIXTURE = FIXTURES / "data_root" / "musique" / "dev_data" / "musique_ans_v1.0_dev_clean.jsonl"
PREDICTIONS_FIXTURE = FIXTURES / "predictions" / "decomposer_results_musique.json"


def _import_evaluator() -> Any:
    name = "musique_decompositions_evaluator"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, EVALUATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


EVAL = _import_evaluator()
POLICY = EVAL._soft_policy(load_config("musique_eval.json"))
STOP = frozenset(POLICY["stopwords_normalized"])


def soft(pred: list[str], gold: list[str], t: float) -> tuple[float, float, float]:
    return EVAL._soft_step_prf(pred, gold, t, STOP)


class TestSoftPolicy(unittest.TestCase):
    def test_committed_policy(self) -> None:
        self.assertEqual(POLICY["thresholds"], {"050": 0.5, "060": 0.6})
        self.assertEqual(POLICY["stopwords_config_count"], 198)
        # Normalised like a step: the apostrophe goes, so "aren't" is the token "arent".
        self.assertIn("arent", STOP)
        self.assertNotIn("aren't", STOP)
        for w in ("what", "the", "of", "is", "was", "who"):
            self.assertIn(w, STOP)

    def test_bad_thresholds_are_refused(self) -> None:
        base = load_config("musique_eval.json")
        for bad in ([0.555], [0.0], [1.5], [0.5, 0.5], [], ["0.5"], [True]):
            cfg = json.loads(json.dumps(base))
            cfg["beside_report_card"]["step_f1_soft"]["thresholds"] = bad
            with self.subTest(bad=bad), self.assertRaises(SystemExit):
                EVAL._soft_policy(cfg)


class TestTokenF1(unittest.TestCase):
    def test_hand_values(self) -> None:
        self.assertEqual(EVAL._token_f1(["a", "b"], ["a", "b"]), 1.0)
        # overlap 2, lengths 3 + 4 -> 4/7
        self.assertAlmostEqual(EVAL._token_f1(["x", "y", "z"], ["x", "y", "u", "v"]), 4 / 7)
        # multiset: one 'x' on each side overlaps once -> 2*1/(2+1)
        self.assertAlmostEqual(EVAL._token_f1(["x", "x"], ["x"]), 2 / 3)
        self.assertEqual(EVAL._token_f1([], ["x"]), 0.0)
        # exactly 3/5 is the same double as the 0.6 literal, so it clears >= 0.6
        self.assertGreaterEqual(EVAL._token_f1(["a", "b", "c"], ["a", "b", "c", "d", "e", "f", "g"]), 0.6)


class TestStepF1Soft(unittest.TestCase):
    def test_style_difference_matches_at_both_thresholds(self) -> None:
        """Excerpt 5 of the anchor-set note: exact step_f1 0.0, soft 1.0.

        pred1 'cold turkey record label' -> {cold, turkey, record, label}
        gold1 'what was the record label of cold turkey' -> {record, label, cold, turkey}
          F1 = 2*4/8 = 1.0
        pred2 '#1 parent company' -> {#1, parent, company}
        gold2 'what company is #1 part of' -> {company, #1, part}
          F1 = 2*2/6 = 0.667 (>= 0.6)
        cross pairs share no token. tp = 2 -> P = R = F1 = 1.0.
        """
        pred = ["Cold Turkey >> record label", "#1 >> parent company"]
        gold = ["What was the record label of Cold Turkey?", "What company is #1 part of?"]
        self.assertEqual(EVAL._step_prf(pred, gold), (0.0, 0.0, 0.0))
        self.assertEqual(soft(pred, gold, 0.5), (1.0, 1.0, 1.0))
        self.assertEqual(soft(pred, gold, 0.6), (1.0, 1.0, 1.0))

    def test_threshold_separates(self) -> None:
        # {alpha, beta, gamma} vs {alpha, beta, delta, epsilon}: 2*2/7 = 0.571
        pred, gold = ["alpha beta gamma"], ["alpha beta delta epsilon"]
        self.assertEqual(soft(pred, gold, 0.5), (1.0, 1.0, 1.0))
        self.assertEqual(soft(pred, gold, 0.6), (0.0, 0.0, 0.0))

    def test_one_to_one_maximum_matching(self) -> None:
        """Greedy-by-score would stop at 1 match; the maximum matching is 2.

        P1 'alpha beta gamma' == G1 (1.0); P1 vs G2 'alpha beta delta' = 4/6 = 0.667
        P2 'beta gamma' vs G1 = 4/5 = 0.8; P2 vs G2 = 2/5 = 0.4
        Greedy takes P1-G1 first and strands P2. Max matching: P1-G2, P2-G1 -> tp 2.
        Exact step_f1: tp 1 of 2 and 2 -> 0.5.
        """
        pred = ["alpha beta gamma", "beta gamma"]
        gold = ["alpha beta gamma", "alpha beta delta"]
        self.assertEqual(EVAL._step_prf(pred, gold), (0.5, 0.5, 0.5))
        self.assertEqual(soft(pred, gold, 0.6), (1.0, 1.0, 1.0))
        # Order of the inputs cannot change the count.
        self.assertEqual(soft(pred[::-1], gold[::-1], 0.6), (1.0, 1.0, 1.0))

    def test_one_gold_step_cannot_be_matched_twice(self) -> None:
        # Two predictions both close to the one gold step: tp 1. P = 1/2, R = 1, F1 = 2/3.
        pred = ["alpha beta gamma", "alpha beta gamma delta"]
        gold = ["alpha beta gamma"]
        p, r, f = soft(pred, gold, 0.5)
        self.assertEqual((p, r), (0.5, 1.0))
        self.assertAlmostEqual(f, 2 / 3)

    def test_sets_like_step_f1(self) -> None:
        # "Alpha beta?" and "alpha beta" normalise to one step: pred set size 1.
        p, r, f = soft(["Alpha beta?", "alpha beta"], ["alpha beta", "gamma delta"], 0.5)
        self.assertEqual((p, r), (1.0, 0.5))
        self.assertAlmostEqual(f, 2 / 3)
        self.assertEqual(soft([], [], 0.5), (1.0, 1.0, 1.0))
        self.assertEqual(soft([], ["alpha"], 0.5), (0.0, 0.0, 0.0))

    def test_stopword_only_steps(self) -> None:
        # No content tokens: only identical normalised strings match.
        self.assertEqual(soft(["what is it"], ["who is he"], 0.5), (0.0, 0.0, 0.0))
        self.assertEqual(soft(["What is it?"], ["what is it"], 0.5), (1.0, 1.0, 1.0))

    def test_never_below_exact_step_f1_on_the_fixture(self) -> None:
        preds = json.loads(PREDICTIONS_FIXTURE.read_text(encoding="utf-8"))
        gold = EVAL._load_gold(GOLD_FIXTURE, 20)
        rows, _ = EVAL._build_eval_rows(preds, gold)
        self.assertTrue(rows)
        for row in rows:
            exact = EVAL._step_prf(row.pred_steps, row.gold_steps)[2]
            for t in (0.5, 0.6):
                with self.subTest(item=row.item_id, t=t):
                    self.assertGreaterEqual(soft(row.pred_steps, row.gold_steps, t)[2], exact)


class TestPlanConnectedness(unittest.TestCase):
    def test_hand_values(self) -> None:
        cases = [
            (["a", "b #1", "c #2"], (1.0, 2, 2)),
            # The orphan step of the anchor-set note: step 1 is never used.
            (["Cold Turkey >> record label", "x >> parent company"], (0.0, 0, 1)),
            (["a", "b", "c #1"], (0.5, 1, 2)),
            (["a", "b [#1]"], (1.0, 1, 1)),
            (["a", "b", "c #1 #2"], (1.0, 2, 2)),
            # Forward and self references are not 'a later step referencing it'.
            (["a #2", "b #2", "c"], (0.0, 0, 2)),
            (["a"], (None, 0, 0)),
            ([], (None, 0, 0)),
        ]
        for steps, want in cases:
            with self.subTest(steps=steps):
                self.assertEqual(EVAL._plan_connectedness(steps), want)

    def test_aggregate_excludes_and_counts_nulls(self) -> None:
        items = [
            {"plan_connectedness": 1.0, "gold_plan_connectedness": 1.0},
            {"plan_connectedness": 0.0, "gold_plan_connectedness": 1.0},
            {"plan_connectedness": None, "gold_plan_connectedness": 0.5},
        ]
        agg = EVAL._beside_card_aggregate(items)
        self.assertEqual(agg["plan_connectedness_macro"], 0.5)
        self.assertEqual(agg["plan_connectedness_n_items"], 2)
        self.assertEqual(agg["plan_connectedness_null_items"], 1)
        self.assertEqual(agg["gold_plan_connectedness_items_at_1"], 2)
        self.assertEqual(agg["gold_plan_connectedness_items_below_1"], 1)
        # Items that do not carry the columns (junk battery, old files): no keys at all.
        self.assertEqual(EVAL._beside_card_aggregate([{"step_f1": 1.0}]), {})

    def test_gold_is_fully_connected_on_the_pinned_600(self) -> None:
        paths = load_paths("paths.json")
        root = Path(paths["data_root_resolved"])
        gold_path = resolve_path(require(paths, "datasets.musique_dev_gold_clean"), root)
        template = require(paths, "datasets.musique_eval_questions_template")
        pinned = [resolve_path(template.format(hop=h), root) for h in (2, 3, 4)]
        if not gold_path.exists() or not all(p.exists() for p in pinned):
            self.skipTest("MuSiQue gold / pinned-600 files not on disk")
        ids = set()
        for p in pinned:
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    ids.add(json.loads(line)["id"])
        self.assertEqual(len(ids), 600)
        at_one = 0
        for line in gold_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("id") in ids:
                steps = EVAL._decomp_to_steps(obj["question_decomposition"])
                self.assertEqual(EVAL._plan_connectedness(steps)[0], 1.0, obj["id"])
                at_one += 1
        self.assertEqual(at_one, 600)


class TestBesideCardEndToEnd(unittest.TestCase):
    """Scoring and --compare over the fixture: columns, labels, and the battery untouched."""

    def _run(self, args: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(EVALUATOR), *args],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )

    def test_score_then_compare(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "score"
            proc = self._run(
                [
                    "--predictions", str(PREDICTIONS_FIXTURE),
                    "--gold", str(GOLD_FIXTURE),
                    "--run-dir", str(run_dir),
                ]
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            metrics = json.loads((run_dir / "eval_metrics.json").read_text(encoding="utf-8"))
            card_metrics = [t["metric"] for t in metrics["decomposition_report_card"]["terms"]]
            for key in ("step_f1_soft_050_macro", "step_f1_soft_060_macro", "plan_connectedness_macro"):
                self.assertIn(key, metrics)
                self.assertNotIn(key, card_metrics)
            self.assertNotIn("step_f1_soft_050", metrics["decomp_mean_policy"]["components"])
            block = metrics["beside_report_card"]
            self.assertIn("BESIDE the report card", block["label"])
            gold = block["gold_plan_connectedness_check"]
            self.assertEqual(gold["items_at_1"], gold["n_items"])
            notes = (run_dir / "eval_notes.md").read_text(encoding="utf-8")
            self.assertIn("Beside the report card (ADR 0039)", notes)

            per_item = run_dir / "eval_per_item.json"
            cmp_dir = Path(tmp) / "compare"
            proc = self._run(["--compare", str(per_item), str(per_item), "--run-dir", str(cmp_dir)])
            self.assertEqual(proc.returncode, 0, proc.stderr)
            cmp = json.loads((cmp_dir / "compare_metrics.json").read_text(encoding="utf-8"))
            for block_name in ("bootstrap", "mcnemar", "t_test"):
                for name in cmp[block_name]:
                    self.assertNotIn("soft", name)
                    self.assertNotEqual(name, "plan_connectedness")
            beside = cmp["beside_report_card"]
            self.assertEqual(
                sorted(beside["bootstrap"]),
                ["plan_connectedness", "step_f1_soft_050", "step_f1_soft_060"],
            )
            for row in beside["bootstrap"].values():
                self.assertEqual(row["difference"], 0.0)
                self.assertFalse(row["significant"])
            self.assertEqual(beside["not_available_in_inputs"], [])


if __name__ == "__main__":
    unittest.main()
