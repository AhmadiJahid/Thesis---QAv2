#!/usr/bin/env python3
"""Tests for scripts/preflight_pool_sweep.py (issue #51): match passes, mismatch refuses.

Run::

    .venv/bin/python -m unittest tests.test_preflight_pool_sweep -v
"""
from __future__ import annotations

import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "preflight_pool_sweep", REPO_ROOT / "scripts" / "preflight_pool_sweep.py"
)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class PreflightTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "pools").mkdir()
        (self.root / "retr/size10").mkdir(parents=True)
        self.files = {
            "pools/p10.jsonl": b'{"id": "2hop__1"}\n',
            "retr/size10/pinned600_top20.jsonl": b"top20\n",
            "retr/size10/top5.jsonl": b"top5\n",
        }
        for rel, data in self.files.items():
            (self.root / rel).write_bytes(data)
        self.cfg = {
            "retrieval": {
                "out_dir_template": "retr/size{size}",
                "retrieval_input_template": "retr/size{size}/top5.jsonl",
            },
            "expected_inputs": {
                "10": {
                    "pool_file": "pools/p10.jsonl",
                    "pool_sha256": _sha(self.files["pools/p10.jsonl"]),
                    "top20_sha256": _sha(self.files["retr/size10/pinned600_top20.jsonl"]),
                    "rerank_top5_sha256": _sha(self.files["retr/size10/top5.jsonl"]),
                }
            },
        }

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_all_match(self) -> None:
        self.assertEqual(mod.check(mod.expected_files(self.cfg, 10, self.root)), [])

    def test_mismatch_and_missing_refused(self) -> None:
        (self.root / "retr/size10/top5.jsonl").write_bytes(b"changed\n")
        (self.root / "pools/p10.jsonl").unlink()
        problems = mod.check(mod.expected_files(self.cfg, 10, self.root))
        self.assertEqual(len(problems), 2)
        self.assertTrue(any(p.startswith("pool: missing") for p in problems))
        self.assertTrue(any(p.startswith("rerank_top5: sha256") for p in problems))

    def test_unknown_size_refused(self) -> None:
        with self.assertRaises(SystemExit):
            mod.expected_files(self.cfg, 11, self.root)


if __name__ == "__main__":
    unittest.main()
