#!/usr/bin/env python3
"""``run_decomposer.py --resume``: an interrupted run, resumed, equals an uninterrupted one.

CPU only: the real runner in ``--dry-run`` over the fabricated fixtures (9 rows). A crash is
simulated by keeping the first rows of ``results.partial.jsonl`` plus a half-written line and
deleting the end-of-run artifacts; ``--resume`` must then rebuild a results.json identical to
the uninterrupted run's, and metrics that agree on every row-derived count. Also covered: the
refusals (a changed seed, a finished run, --output-root together with --resume).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from run_config import PATHS_CONFIG_ENV  # noqa: E402

RETRIEVAL = REPO_ROOT / "tests" / "fixtures" / "retrieval" / "top5_musique_conditions.jsonl"
KEEP = 4  # rows that "finished" before the simulated crash


def _run(*extra: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env[PATHS_CONFIG_ENV] = "configs/smoke_paths.json"
    env.setdefault("PYTHONPYCACHEPREFIX", tempfile.gettempdir() + "/qav2-pycache")
    cmd = [
        sys.executable, str(REPO_ROOT / "components" / "decomposer" / "run_decomposer.py"),
        "--model", "mistral_7b_instruct", "--config", "decomposer_musique.json",
        "--condition", "count_first", "--dry-run", "--dry-run-limit", "9",
        "--retrieval-input", str(RETRIEVAL), "--allow-unpinned-eval-set", *extra,
    ]
    return subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=REPO_ROOT)


def _read(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


class TestResume(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = Path(tempfile.mkdtemp(prefix="qav2-resume-"))
        proc = _run("--output-root", str(cls.tmp / "full"))
        assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
        cls.full = next((cls.tmp / "full").iterdir())

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _crashed_copy(self, name: str) -> Path:
        """A run dir as a crash after KEEP rows would leave it."""
        proc = _run("--output-root", str(self.tmp / name))
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        run = next((self.tmp / name).iterdir())
        lines = (run / "results.partial.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 9)
        half = lines[KEEP][: len(lines[KEEP]) // 2]  # a row cut off mid-write
        (run / "results.partial.jsonl").write_text(
            "\n".join(lines[:KEEP]) + "\n" + half, encoding="utf-8"
        )
        for f in ("results.json", "config.json", "metrics.json", "notes.md"):
            (run / f).unlink()
        return run

    def test_partial_file_matches_results(self) -> None:
        rows = [json.loads(x) for x in
                (self.full / "results.partial.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(rows, _read(self.full / "results.json"))

    def test_resumed_equals_uninterrupted(self) -> None:
        run = self._crashed_copy("crashed")
        proc = _run("--resume", str(run))
        self.assertEqual(proc.returncode, 0, proc.stdout[-2000:] + proc.stderr[-2000:])
        self.assertIn(f"{KEEP} of 9 rows already finished", proc.stdout)
        self.assertEqual(_read(run / "results.json"), _read(self.full / "results.json"))
        self.assertEqual(
            (run / "results.json").read_bytes(), (self.full / "results.json").read_bytes()
        )
        m_full, m_res = _read(self.full / "metrics.json"), _read(run / "metrics.json")
        for key in ("total_rows", "rows_loaded_per_hop", "rows_with_empty_decomposition",
                    "anchor_prompts_logged", "anchor_prompts_expected", "few_shot_source_counts",
                    "count_first", "count_first_settings", "evaluation_set"):
            self.assertEqual(m_res[key], m_full[key], key)
        snap = _read(run / "config.json")
        self.assertEqual(snap["resume"]["rows_from_partial"], KEEP)
        self.assertEqual(snap["run_id"], run.name)
        # The partial file now holds every row once.
        ids = [json.loads(x)["query_id"] for x in
               (run / "results.partial.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(sorted(ids), sorted(r["query_id"] for r in _read(run / "results.json")))

    def test_finished_rows_are_not_regenerated(self) -> None:
        run = self._crashed_copy("marked")
        lines = (run / "results.partial.jsonl").read_text(encoding="utf-8").splitlines()
        rows = [json.loads(x) for x in lines[:KEEP]]
        for r in rows:
            r["decomposition"] = "KEPT FROM PARTIAL"
        (run / "results.partial.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
        )
        proc = _run("--resume", str(run))
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        out = _read(run / "results.json")
        self.assertEqual([r["decomposition"] for r in out[:KEEP]], ["KEPT FROM PARTIAL"] * KEEP)
        self.assertEqual(out[KEEP:], _read(self.full / "results.json")[KEEP:])

    def test_refusals(self) -> None:
        run = self._crashed_copy("refusals")
        before = (run / "results.partial.jsonl").read_text(encoding="utf-8")
        proc = _run("--resume", str(run), "--seed", "7")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("--resume refused", proc.stdout + proc.stderr)
        self.assertIn("seed", proc.stdout + proc.stderr)
        self.assertFalse((run / "results.json").exists())
        self.assertEqual((run / "results.partial.jsonl").read_text(encoding="utf-8"), before)

        proc = _run("--resume", str(self.full))
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("already has results.json", proc.stdout + proc.stderr)

        proc = _run("--resume", str(run), "--output-root", str(self.tmp / "x"))
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("exclusive", proc.stdout + proc.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
