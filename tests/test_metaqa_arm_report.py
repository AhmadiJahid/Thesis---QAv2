"""Tests for scripts/metaqa_arm_report.py helpers (exp-039, ADR 0045)."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from metaqa_arm_report import failure_reasons, holm, mcnemar_cell, mcnemar_exact_p, repaired_count, score_arm  # noqa: E402


class ArmReportTest(unittest.TestCase):
    def test_holm_matches_hand_computation(self) -> None:
        # sorted p: .01 (x3 = .03), .02 (x2 = .04), .04 (x1 = .04, monotone)
        self.assertEqual([round(x, 6) for x in holm([0.04, 0.01, 0.02])], [0.04, 0.03, 0.04])

    def test_mcnemar_exact(self) -> None:
        self.assertEqual(mcnemar_exact_p(0, 0), 1.0)
        self.assertAlmostEqual(mcnemar_exact_p(5, 0), 2 / 32)

    def test_every_item_is_scored_and_failures_score_zero(self) -> None:
        gold = {"q1": (1, {"A", "B"}), "q2": (1, {"C"}), "q3": (1, {"D"})}
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "success.json").write_text(json.dumps([{"question": "q1", "kg_results": ["A", "X"]}]))
            (p / "compile_fail.json").write_text(json.dumps([{"question": "q2"}]))
            (p / "exec_fail.json").write_text(json.dumps([{"question": "q3"}]))
            rows = score_arm(p, gold)
        self.assertEqual(rows["q1"], {"compiled": 1.0, "executed": 1.0, "em": 0.0,
                                      "jaccard": 1 / 3, "any_hit": 1.0, "rhits1": 0.5})
        self.assertEqual(rows["q2"]["compiled"], 0.0)
        self.assertEqual((rows["q3"]["compiled"], rows["q3"]["executed"], rows["q3"]["em"]), (1.0, 0.0, 0.0))

    def test_a_row_missing_from_the_arm_is_refused(self) -> None:
        gold = {"q1": (1, {"A"}), "q2": (1, {"B"})}
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "success.json").write_text(json.dumps([{"question": "q1", "kg_results": ["A"]}]))
            (p / "compile_fail.json").write_text("[]")
            (p / "exec_fail.json").write_text("[]")
            with self.assertRaises(SystemExit):
                score_arm(p, gold)

    def test_failure_reasons_and_repaired_count_per_hop(self) -> None:
        gold = {"q1": (1, {"A"}), "q2": (2, {"B"}), "q3": (2, {"C"})}
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "success.json").write_text(json.dumps([{"question": "q1", "kg_results": ["A"], "repaired_steps": [2]}]))
            (p / "compile_fail.json").write_text(json.dumps([{"question": "q2", "error_reason": "type_mismatch: step 2"}]))
            (p / "exec_fail.json").write_text(json.dumps([{"question": "q3", "error_reason": "empty_result: x"}]))
            self.assertEqual(failure_reasons(p, gold), {"2": {"compile": {"type_mismatch": 1}, "exec": {"empty_result": 1}}})
            self.assertEqual(repaired_count(p, gold), {"1": 1})

    def test_mcnemar_cell_on_a_subset(self) -> None:
        import numpy as np
        c = mcnemar_cell(np.array([1.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0]))
        self.assertEqual((c["n"], c["only_a"], c["only_b"]), (3, 1, 0))


if __name__ == "__main__":
    unittest.main()
