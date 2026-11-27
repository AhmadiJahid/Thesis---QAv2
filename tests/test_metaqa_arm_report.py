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

from metaqa_arm_report import holm, mcnemar_exact_p, score_arm  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
