#!/usr/bin/env python3
"""The decomposer runner's ``--eval-set`` path, end to end on the fixture tree (issue #56).

Runs ``components/decomposer/run_decomposer.py --dry-run`` (no weights) under
``configs/smoke_paths.json``, where ``musique_dev_full`` resolves to the fabricated 4-row
dev file and its ``expected_total`` is overridden to 4. Retrieval inputs are built in a
temporary directory from the committed fixture row, one row per chosen query id.

Cases:

- pass: exactly the set's 4 ids -> rc 0, ``evaluation_set.matches_named_set`` true,
  ``pinned`` false;
- refuse: one id missing, one extra id, and an unknown set name -> rc != 0;
- opt-out: one id missing with ``--allow-unpinned-eval-set`` -> rc 0, the mismatch recorded;
- the default path is untouched: the same 4-id file WITHOUT ``--eval-set`` is refused by the
  pinned-set assertion, exactly as before.

Run::

    .venv/bin/python -m unittest tests/test_named_eval_set_runner.py -v
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "components" / "decomposer" / "run_decomposer.py"
FIXTURE_ROW = REPO / "tests" / "fixtures" / "retrieval" / "top5_musique_conditions.jsonl"
FIXTURE_DEV = REPO / "tests" / "fixtures" / "data_root" / "musique" / "musique_ans_v1.0_dev.jsonl"


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class RunnerNamedEvalSetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.template = _rows(FIXTURE_ROW)[0]
        cls.dev = {r["id"]: r["question"] for r in _rows(FIXTURE_DEV)}

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def _retrieval(self, name: str, ids_questions: list[tuple[str, str]]) -> Path:
        path = self.root / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for i, (qid, question) in enumerate(ids_questions):
                row = dict(self.template)
                row.update({"query_id": qid, "query_index": i, "query_question": question})
                f.write(json.dumps(row) + "\n")
        return path

    def _run(self, retrieval: Path, *extra: str) -> tuple[int, str, Path]:
        out = self.root / f"out_{retrieval.stem}_{len(extra)}"
        env = dict(os.environ, QAV2_PATHS_CONFIG="configs/smoke_paths.json")
        proc = subprocess.run(
            [sys.executable, str(RUNNER), "--model", "mistral_7b_instruct",
             "--config", "decomposer_musique.json", "--condition", "unguided", "--seed", "42",
             "--retrieval-input", str(retrieval), "--dry-run", "--dry-run-limit", "2",
             "--output-root", str(out), *extra],
            cwd=str(REPO), env=env, capture_output=True, text=True,
        )
        return proc.returncode, proc.stdout + proc.stderr, out

    def _metrics(self, out: Path) -> dict:
        found = sorted(out.glob("*/metrics.json"))
        self.assertTrue(found, f"no metrics.json under {out}")
        return json.loads(found[-1].read_text(encoding="utf-8"))

    def test_pass(self) -> None:
        rc, log, out = self._run(self._retrieval("full", list(self.dev.items())),
                                 "--eval-set", "musique_dev_full")
        self.assertEqual(rc, 0, log[-2000:])
        es = self._metrics(out)["evaluation_set"]
        self.assertTrue(es["matches_named_set"])
        self.assertFalse(es["pinned"])
        self.assertEqual(es["rows_loaded_total"], 4)
        self.assertEqual(es["definition"]["expected_total"], 4)

    def test_refuse_missing_and_extra(self) -> None:
        items = list(self.dev.items())
        rc, log, _ = self._run(self._retrieval("missing", items[:3]), "--eval-set", "musique_dev_full")
        self.assertNotEqual(rc, 0)
        self.assertIn("not the named evaluation set", log)
        extra = items + [("3hop1__zz999_x", "A question that is not in the dev set?")]
        rc, log, _ = self._run(self._retrieval("extra", extra), "--eval-set", "musique_dev_full")
        self.assertNotEqual(rc, 0)
        self.assertIn("not in the set", log)

    def test_refuse_unknown_set(self) -> None:
        rc, log, _ = self._run(self._retrieval("unknown", list(self.dev.items())),
                               "--eval-set", "no_such_set")
        self.assertNotEqual(rc, 0)
        self.assertIn("is not defined", log)

    def test_opt_out_records_mismatch(self) -> None:
        rc, log, out = self._run(self._retrieval("allowed", list(self.dev.items())[:3]),
                                 "--eval-set", "musique_dev_full", "--allow-unpinned-eval-set")
        self.assertEqual(rc, 0, log[-2000:])
        es = self._metrics(out)["evaluation_set"]
        self.assertFalse(es["matches_named_set"])
        self.assertEqual(es["ids_missing_count"], 1)

    def test_default_path_still_asserts_pinned_set(self) -> None:
        rc, log, _ = self._run(self._retrieval("default", list(self.dev.items())))
        self.assertNotEqual(rc, 0)
        self.assertIn("not the pinned evaluation set", log)


if __name__ == "__main__":
    unittest.main()
