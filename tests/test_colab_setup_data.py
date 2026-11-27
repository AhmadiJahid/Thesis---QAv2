#!/usr/bin/env python3
"""scripts/colab_setup_data.py on a fabricated archive. No real data, no network."""
from __future__ import annotations

import io
import os
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "colab_setup_data.py"
DEV = "Thesis---QA/MusiQue/Data/dev_data/"
FILES = {
    DEV + "musique_ans_v1.0_dev_sample_2_hop_200.jsonl": b"{}\n",
    DEV + "musique_ans_v1.0_dev_sample_3_hop_200.jsonl": b"{}\n",
    DEV + "musique_ans_v1.0_dev_sample_4_hop_200.jsonl": b"{}\n",
    DEV + "musique_ans_v1.0_dev_clean.jsonl": b"{}\n",
    "Thesis---QA/MusiQue/Data/emb_cache/x.npy": b"e",
    "Thesis---QA/Data/kb.txt": b"k",
    "Thesis---QAv2/runs/exp-014/retrieval/top5.jsonl": b"r",
    "Thesis---QAv2/runs/exp-032/exp-020/eval/eval_per_item.json": b"p",
    "Thesis---QAv2/runs/exp-031/other.json": b"o",
}


class TestColabSetupData(unittest.TestCase):
    def test_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            archive = tmp / "a.tar.gz"
            with tarfile.open(archive, "w:gz") as tar:
                for name, data in FILES.items():
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))
            data_root, runs_root = tmp / "data", tmp / "runs"
            proc = subprocess.run(
                [sys.executable, str(SCRIPT), "--archive", str(archive),
                 "--data-root", str(data_root), "--runs-root", str(runs_root),
                 "--parts", "musique", "--runs", "exp-014/retrieval/top5.jsonl", "exp-032/exp-020/eval"],
                capture_output=True, text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertTrue((data_root / "musique/dev_data/musique_ans_v1.0_dev_clean.jsonl").exists())
            self.assertFalse((data_root / "musique/emb_cache").exists())  # its own part
            self.assertFalse((data_root / "metaqa").exists())             # not requested
            self.assertEqual((runs_root / "exp-014/retrieval/top5.jsonl").read_bytes(), b"r")
            self.assertTrue((runs_root / "exp-032/exp-020/eval/eval_per_item.json").exists())
            self.assertFalse((runs_root / "exp-031").exists())
            # A requested run path the archive lacks is a refusal, not a silent skip.
            proc = subprocess.run(
                [sys.executable, str(SCRIPT), "--archive", str(archive),
                 "--data-root", str(data_root), "--runs-root", str(runs_root),
                 "--parts", "musique", "--runs", "exp-999/missing"],
                capture_output=True, text=True,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("MISSING", proc.stdout + proc.stderr)


def _archive(path: Path, files: dict[str, bytes]) -> None:
    with tarfile.open(path, "w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def _setup(archive: Path, data_root: Path, runs_root: Path, *runs: str):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--archive", str(archive), "--data-root", str(data_root),
         "--runs-root", str(runs_root), "--parts", "musique", "--runs", *runs],
        capture_output=True, text=True,
    )


class TestColabSetupDataSafety(unittest.TestCase):
    def test_dotdot_and_absolute_members_are_refused(self) -> None:
        for bad in (
            "Thesis---QAv2/runs/exp-014/../../../escape.txt",
            "Thesis---QA/MusiQue/Data/dev_data/../../../../escape.txt",
            "/abs/escape.txt",
            "C:/escape.txt",
        ):
            with self.subTest(member=bad), tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                _archive(tmp / "a.tar.gz", {**FILES, bad: b"x"})
                (tmp / "in").mkdir()
                proc = _setup(tmp / "a.tar.gz", tmp / "in" / "data", tmp / "in" / "runs",
                              "exp-014")
                self.assertNotEqual(proc.returncode, 0)
                self.assertIn("refusing archive member", proc.stdout + proc.stderr)
                self.assertEqual(list(tmp.rglob("escape.txt")), [])

    def test_symlink_below_root_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _archive(tmp / "a.tar.gz", FILES)
            outside = tmp / "outside"
            outside.mkdir()
            runs_root = tmp / "runs"
            runs_root.mkdir()
            try:
                os.symlink(outside, runs_root / "exp-032", target_is_directory=True)
            except (OSError, NotImplementedError) as exc:
                self.skipTest(f"cannot create a symlink here: {exc}")
            proc = _setup(tmp / "a.tar.gz", tmp / "data", runs_root, "exp-032/exp-020/eval")
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("is a symlink", proc.stdout + proc.stderr)
            self.assertEqual(list(outside.rglob("*")), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
