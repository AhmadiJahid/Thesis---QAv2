#!/usr/bin/env python3
"""The scaffold-line parser fix and the re-parse tool (issue #60, ADR 0039 item 4 / H3).

No GPU, no weights, no data files. Hand-computed expectations throughout.

Run::

    .venv/bin/python -m unittest tests.test_reparse_decompositions
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from run_config import load_config, require  # noqa: E402
from step_lines import split_step_lines, with_scaffold_truncate_markers  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "reparse_decompositions", REPO_ROOT / "scripts" / "reparse_decompositions.py"
)
rp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rp)

RECORDED = {"strip_think": False, "truncate_at": ["Question:"]}
SCAFFOLD = ["Task:", "Answer:"]
FIXED = with_scaffold_truncate_markers(RECORDED, SCAFFOLD)

#: The two shapes measured in exp-020 / exp-023 (analysis note section 1.2), shortened.
RAW_TASK = (
    "Lauren Bacall >> spouse\n#1 >> which movie did he win his only Oscar for\n\nTask:\n"
    "Question: What is the name of the movie?\nDecomposition:\nyear >> 2004"
)
RAW_ANSWER = (
    "Who founded X?\nWhere was #1 born?\n\nAnswer:\nWho founded X?\nWhere was #1 born?\n"
    "When did #2 end?"
)
RAW_CLEAN = "Who founded X?\nWhere was #1 born?"


class TestScaffoldMarkers(unittest.TestCase):
    def test_markers_are_appended_once_and_input_untouched(self) -> None:
        self.assertEqual(FIXED["truncate_at"], ["Question:", "Task:", "Answer:"])
        self.assertEqual(RECORDED["truncate_at"], ["Question:"])
        again = with_scaffold_truncate_markers(FIXED, SCAFFOLD)
        self.assertEqual(again["truncate_at"], ["Question:", "Task:", "Answer:"])

    def test_bad_marker_lists_are_refused(self) -> None:
        for bad in ("Task:", [""], [1], None):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                with_scaffold_truncate_markers(RECORDED, bad)

    def test_task_line_is_cut_with_question_semantics(self) -> None:
        # Recorded parser: cut at 'Question:', so the 'Task:' label is left as step 3.
        self.assertEqual(
            split_step_lines(rp.parse_raw(RAW_TASK, RECORDED, None)),
            ["Lauren Bacall >> spouse", "#1 >> which movie did he win his only Oscar for", "Task:"],
        )
        # Fixed parser: truncated at 'Task:' - two steps.
        self.assertEqual(
            split_step_lines(rp.parse_raw(RAW_TASK, FIXED, None)),
            ["Lauren Bacall >> spouse", "#1 >> which movie did he win his only Oscar for"],
        )

    def test_answer_label_truncates_everything_after_it(self) -> None:
        # Truncation, not line-dropping: the re-decomposition after 'Answer:' goes too.
        self.assertEqual(len(split_step_lines(rp.parse_raw(RAW_ANSWER, RECORDED, None))), 6)
        self.assertEqual(
            split_step_lines(rp.parse_raw(RAW_ANSWER, FIXED, None)),
            ["Who founded X?", "Where was #1 born?"],
        )

    def test_cap_trim_is_applied_after_truncation(self) -> None:
        self.assertEqual(rp.parse_raw(RAW_ANSWER, FIXED, 1), "Who founded X?")
        self.assertEqual(rp.parse_raw(RAW_ANSWER, RECORDED, 3), "Who founded X?\nWhere was #1 born?\nAnswer:")

    def test_musique_config_carries_the_markers_and_metaqa_does_not(self) -> None:
        self.assertEqual(
            require(load_config("decomposer_musique.json"), "step_parser.scaffold_truncate_at"),
            ["Task:", "Answer:"],
        )
        self.assertEqual(
            require(load_config("decomposer.json"), "step_parser.scaffold_truncate_at"), []
        )

    def test_no_model_folder_truncates_at_answer(self) -> None:
        """The answerer reuses model-folder post_process; 'Answer:' there would empty answers."""
        for cfg_path in sorted((REPO_ROOT / "components" / "decomposer" / "models").glob("*/config.json")):
            with self.subTest(model=cfg_path.parent.name):
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                self.assertEqual(cfg["post_process"]["truncate_at"], ["Question:"])


def _row(qid: str, raw: str, decomposition: str) -> dict:
    return {
        "query_id": qid,
        "question": f"{qid}?",
        "decomposition": decomposition,
        "decomposition_raw": raw,
        "step_lines": len(split_step_lines(decomposition)),
    }


class TestReparseRows(unittest.TestCase):
    def rows(self) -> list[dict]:
        return [
            _row("a", RAW_TASK, rp.parse_raw(RAW_TASK, RECORDED, None)),
            _row("b", RAW_ANSWER, rp.parse_raw(RAW_ANSWER, RECORDED, None)),
            _row("c", RAW_CLEAN, RAW_CLEAN),
            _row("d", "", ""),
        ]

    def test_changed_rows_and_before_after(self) -> None:
        new_rows, changed, not_reproduced, skipped = rp.reparse_rows(
            self.rows(), RECORDED, FIXED, None
        )
        self.assertEqual(not_reproduced, [])
        self.assertEqual(skipped, 1)
        self.assertEqual([c["query_id"] for c in changed], ["a", "b"])
        self.assertEqual(len(changed[0]["pred_steps_before"]), 3)
        self.assertEqual(len(changed[0]["pred_steps_after"]), 2)
        self.assertEqual(len(changed[1]["pred_steps_before"]), 6)
        self.assertEqual(len(changed[1]["pred_steps_after"]), 2)
        self.assertEqual(new_rows[0]["step_lines"], 2)
        self.assertEqual(new_rows[2], self.rows()[2])
        self.assertEqual(new_rows[3], self.rows()[3])
        # Every other field is carried over untouched.
        self.assertEqual(new_rows[1]["decomposition_raw"], RAW_ANSWER)

    def test_a_row_that_does_not_reproduce_is_reported(self) -> None:
        rows = self.rows()
        rows[2]["decomposition"] = "something else"
        _, _, not_reproduced, _ = rp.reparse_rows(rows, RECORDED, FIXED, None)
        self.assertEqual(not_reproduced, ["c"])


class TestReparseCli(unittest.TestCase):
    def _run(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "reparse_decompositions.py"), *args],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )

    def _make_run(self, root: Path) -> Path:
        run = root / "run"
        run.mkdir()
        rows = TestReparseRows().rows()
        (run / "results.json").write_text(json.dumps(rows), encoding="utf-8")
        (run / "config.json").write_text(
            json.dumps({"post_process": RECORDED, "stop_after_step_lines": None}),
            encoding="utf-8",
        )
        return run

    def test_end_to_end_and_refusals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = self._make_run(Path(tmp))
            original = (run / "results.json").read_bytes()
            # In place is refused, and the input is untouched.
            proc = self._run("--results", str(run / "results.json"), "--output-dir", str(run))
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("never writes in place", proc.stderr)
            self.assertEqual((run / "results.json").read_bytes(), original)

            out = Path(tmp) / "out"
            proc = self._run("--results", str(run / "results.json"), "--output-dir", str(out))
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("2/4 rows changed", proc.stdout)
            metrics = json.loads((out / "reparse_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["rows_changed"], 2)
            self.assertEqual(metrics["pred_steps_removed_total"], 5)
            new = json.loads((out / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(new[0]["decomposition"], rp.parse_raw(RAW_TASK, FIXED, None))
            self.assertEqual((run / "results.json").read_bytes(), original)

            # A second run into the same directory is refused rather than overwriting.
            proc = self._run("--results", str(run / "results.json"), "--output-dir", str(out))
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("already exists", proc.stderr)

    def test_non_reproducing_input_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = self._make_run(Path(tmp))
            rows = json.loads((run / "results.json").read_text(encoding="utf-8"))
            rows[2]["decomposition"] = "something else"
            (run / "results.json").write_text(json.dumps(rows), encoding="utf-8")
            out = Path(tmp) / "out"
            proc = self._run("--results", str(run / "results.json"), "--output-dir", str(out))
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("do not reproduce", proc.stderr)
            self.assertFalse((out / "results.json").exists())


if __name__ == "__main__":
    unittest.main()
