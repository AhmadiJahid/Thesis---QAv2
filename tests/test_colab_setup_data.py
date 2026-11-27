#!/usr/bin/env python3
"""scripts/colab_setup_data.py on a fabricated archive. No real data, no network."""
from __future__ import annotations

import io
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
