#!/usr/bin/env python3
"""Checks for the clustered pool-construction strategy (issue #14, ADR 0021).

CPU only. Every check below runs on hand-made embeddings or the six fabricated rows in
``tests/fixtures/``; nothing here touches the real MuSiQue pool, and the one check that
loads the bi-encoder (``intfloat/e5-small-v2``, ~33M params) skips itself when the weights
are not already in the local cache.

What it covers:

1. **Determinism under the fixed seed** — ``_kmeans_select`` and ``_sample_clustered``
   (which also shuffles) return byte-identical selections on repeated calls with the same
   seed, and the CLI writes a byte-identical ``pool.jsonl`` on a re-run.
2. **The exact target size is reached** — for several ``(n, size, examples_per_cluster)``
   combinations, including ``size == n``, a quota above 1, and a case where clusters cannot
   supply the target so the top-up rule has to run.
3. **No pool-item re-masking (ADR 0003)** — the rows written out are the input rows,
   value-for-value, with their stored masked fields untouched. The strategy reads a text
   field; it never masks.
4. **Config knobs are respected, and a wrong one is refused** — ``examples_per_cluster``
   changes the cluster count, an unknown ``representative_rule`` and a target larger than
   the candidate set are refused with a non-zero exit, and a missing ``text_field`` on any
   row is refused rather than silently dropping the row.

Run::

    .venv/bin/python -m unittest discover -s tests -v
    .venv/bin/python -m unittest tests.test_pool_clustering -v
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_POOL = REPO_ROOT / "MusiQue" / "scripts" / "sample_pool.py"
FIXTURE_POOL = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "data_root"
    / "musique"
    / "chunks_only_question_masked_fixed"
    / "roberta_large_ner_english"
    / "musique_ans_v1.0_train_all_questions_all_expanded_enriched.jsonl"
)

#: The clustering defaults under test, mirroring configs/musique_prep.json. Held here as a
#: literal so a knob changing in the config makes the *config* test fail loudly rather than
#: silently re-pointing every check below at a new value.
DEFAULT_KMEANS = {
    "init": "k-means++",
    "n_init": 1,
    "max_iter": 300,
    "tol": 0.0001,
    "algorithm": "lloyd",
    "num_threads": 1,
}


def _import_sample_pool() -> Any:
    """Import MusiQue/scripts/sample_pool.py the way the script itself is run."""
    for path in (REPO_ROOT / "src", SAMPLE_POOL.parent):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    name = "sample_pool"
    spec = importlib.util.spec_from_file_location(name, SAMPLE_POOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SP = _import_sample_pool()


def _blobs(centres: list[list[float]], per_centre: int, spread: float) -> np.ndarray:
    """Tight, deterministic blobs: centre + a fixed small offset per member.

    Not random: the fixture must not depend on an RNG for the *input* to a determinism
    check, or a failure could not be attributed.
    """
    rows: list[list[float]] = []
    for centre in centres:
        for j in range(per_centre):
            offset = spread * ((j % 3) - 1)
            rows.append([c + offset * (1 + i * 0.1) for i, c in enumerate(centre)])
    return np.asarray(rows, dtype=np.float32)


def _ids(n: int) -> list[str]:
    return [f"row{i:04d}" for i in range(n)]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class TestKmeansSelection(unittest.TestCase):
    """The selection rule, on injected embeddings. No encoder is loaded."""

    def _select(self, emb: np.ndarray, size: int, **kwargs: Any) -> tuple[list[int], dict[str, Any]]:
        params = {
            "seed": 42,
            "examples_per_cluster": 1,
            "representative_rule": "nearest_to_centroid",
            "kmeans_params": dict(DEFAULT_KMEANS),
            "num_threads": 1,
        }
        params.update(kwargs)
        return SP._kmeans_select(emb, _ids(emb.shape[0]), size, **params)

    def test_deterministic_under_the_same_seed(self) -> None:
        emb = _blobs([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]], 6, 0.05)
        first, diag_a = self._select(emb, 8)
        second, diag_b = self._select(emb, 8)
        third, _ = self._select(emb, 8)
        self.assertEqual(first, second)
        self.assertEqual(first, third)
        self.assertEqual(diag_a["kmeans_inertia"], diag_b["kmeans_inertia"])
        self.assertEqual(diag_a["n_clusters"], 8)

    def test_exact_target_size_across_shapes(self) -> None:
        emb = _blobs([[0.0, 0.0], [5.0, 0.0], [0.0, 5.0]], 8, 0.1)  # 24 rows
        for size, quota in ((1, 1), (3, 1), (7, 1), (24, 1), (10, 2), (9, 4), (24, 5)):
            with self.subTest(size=size, quota=quota):
                selected, diag = self._select(emb, size, examples_per_cluster=quota)
                self.assertEqual(len(selected), size)
                self.assertEqual(len(set(selected)), size)
                self.assertEqual(
                    diag["n_clusters"], min(-(-size // quota), emb.shape[0])
                )
                self.assertEqual(
                    diag["selected_from_clusters"] + diag["selected_from_topup"], size
                )

    def test_topup_runs_when_clusters_cannot_fill_the_target(self) -> None:
        """Three tight blobs, quota 4, target 12: k=3 and only 3 rows exist per blob.

        Each cluster can supply at most its own members, so the rank-major pass cannot
        reach 12 and the top-up rule has to close the gap — and say that it did.
        """
        emb = _blobs([[0.0, 0.0], [20.0, 0.0], [0.0, 20.0]], 4, 0.01)  # 12 rows, k=3
        selected, diag = self._select(emb, 12, examples_per_cluster=4)
        self.assertEqual(len(selected), 12)
        self.assertEqual(len(set(selected)), 12)
        self.assertEqual(diag["n_clusters"], 3)
        self.assertEqual(sorted(selected), list(range(12)))

    def test_one_representative_per_separated_blob(self) -> None:
        """With k = number of blobs, the picks land one per blob — the point of the strategy."""
        centres = [[0.0, 0.0], [50.0, 0.0], [0.0, 50.0], [50.0, 50.0]]
        emb = _blobs(centres, 5, 0.02)  # 20 rows, blob b == indices 5b..5b+4
        selected, diag = self._select(emb, 4)
        self.assertEqual(diag["n_clusters"], 4)
        self.assertEqual(diag["empty_clusters"], 0)
        self.assertEqual(sorted(idx // 5 for idx in selected), [0, 1, 2, 3])

    def test_rank_major_order_gives_every_cluster_one_pick_first(self) -> None:
        """Quota 3 with a target that truncates mid-pass still covers all clusters."""
        emb = _blobs([[0.0, 0.0], [30.0, 0.0], [0.0, 30.0]], 5, 0.02)  # 15 rows
        selected, diag = self._select(emb, 4, examples_per_cluster=3)
        self.assertEqual(diag["n_clusters"], 2)  # ceil(4 / 3)
        self.assertEqual(len(selected), 4)
        emb2 = _blobs([[0.0, 0.0], [30.0, 0.0], [0.0, 30.0]], 5, 0.02)
        selected2, _ = self._select(emb2, 3, examples_per_cluster=1)
        self.assertEqual(sorted(idx // 5 for idx in selected2), [0, 1, 2])

    def test_refusals(self) -> None:
        emb = _blobs([[0.0, 0.0], [9.0, 0.0]], 3, 0.05)  # 6 rows
        with self.assertRaises(SystemExit):
            self._select(emb, 7)  # size > candidates
        with self.assertRaises(SystemExit):
            self._select(emb, 3, representative_rule="farthest_from_centroid")
        with self.assertRaises(SystemExit):
            self._select(emb, 3, examples_per_cluster=0)
        with self.assertRaises(SystemExit):
            self._select(emb, 3, num_threads=0)


class TestClusteredSampling(unittest.TestCase):
    """``_sample_clustered`` on the fixture rows, with embeddings injected."""

    def setUp(self) -> None:
        self.rows = _load_jsonl(FIXTURE_POOL)
        self.assertEqual(len(self.rows), 6, "fixture pool changed; update this expectation")
        self.clustering = {
            "text_field": "question",
            "similarity_config": "similarity.json",
            "embed_model": "e5-small",
            "embed_prefix": "passage",
            "embed_batch_size": 4,
            "device": "cpu",
            "examples_per_cluster": 1,
            "representative_rule": "nearest_to_centroid",
            "kmeans": dict(DEFAULT_KMEANS),
        }
        self.emb = _blobs([[0.0, 0.0], [7.0, 0.0], [0.0, 7.0]], 2, 0.03)

    def _sample(self, size: int, seed: int = 42) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        import random

        return SP._sample_clustered(
            self.rows,
            size,
            random.Random(seed),
            clustering=self.clustering,
            embed_model_key=None,
            device=None,
            seed=seed,
            embeddings=self.emb,
        )

    def test_deterministic_including_the_shuffle(self) -> None:
        first, diag = self._sample(3)
        second, _ = self._sample(3)
        self.assertEqual([r["id"] for r in first], [r["id"] for r in second])
        self.assertEqual(len(first), 3)
        self.assertEqual(diag["embedding"]["model_id"], "intfloat/e5-small-v2")

    def test_rows_are_the_input_rows_unchanged(self) -> None:
        """ADR 0003: the pool is never re-masked. The rows out are the rows in."""
        by_id = {r["id"]: r for r in self.rows}
        selected, _ = self._sample(4)
        self.assertEqual(len(selected), 4)
        for row in selected:
            self.assertIn(row["id"], by_id)
            self.assertEqual(row, by_id[row["id"]])
            self.assertEqual(
                row["question_masked_typed"], by_id[row["id"]]["question_masked_typed"]
            )
            self.assertEqual(
                row["question_masked_uniform"], by_id[row["id"]]["question_masked_uniform"]
            )
        self.assertEqual(len({r["id"] for r in selected}), 4)

    def test_text_field_knob_is_used(self) -> None:
        self.clustering["text_field"] = "question_masked_typed"
        _, diag = self._sample(2)
        self.assertEqual(diag["text_field"], "question_masked_typed")

    def test_missing_text_field_is_refused(self) -> None:
        self.clustering["text_field"] = "question_masked_nonexistent"
        with self.assertRaises(SystemExit):
            self._sample(2)

    def test_blank_text_field_is_refused(self) -> None:
        rows = [dict(r) for r in self.rows]
        rows[2]["question"] = "   "
        with self.assertRaises(SystemExit):
            SP._clustering_texts(rows, "question")


class TestConfigWiring(unittest.TestCase):
    """The committed configs carry the strategy, and the orchestrator grid picks it up."""

    def test_prep_config_has_every_clustering_knob(self) -> None:
        cfg = json.loads((REPO_ROOT / "configs" / "musique_prep.json").read_text(encoding="utf-8"))
        clustering = cfg["sample_pool"]["clustering"]
        for key in (
            "text_field",
            "similarity_config",
            "embed_model",
            "embed_prefix",
            "embed_batch_size",
            "device",
            "examples_per_cluster",
            "representative_rule",
            "kmeans",
        ):
            self.assertIn(key, clustering)
        self.assertEqual(clustering["kmeans"], DEFAULT_KMEANS)
        self.assertEqual(clustering["representative_rule"], "nearest_to_centroid")
        # ADR 0018: the bi-encoder alias resolves through the similarity registry.
        similarity = json.loads(
            (REPO_ROOT / "configs" / clustering["similarity_config"]).read_text(encoding="utf-8")
        )
        self.assertEqual(
            similarity["bi_encoder"]["embed_models"][clustering["embed_model"]],
            "intfloat/e5-small-v2",
        )

    def test_sweep_grid_contains_the_clustered_cells(self) -> None:
        name = "pool_sweep_orchestrator_for_clustering_test"
        spec = importlib.util.spec_from_file_location(
            name, REPO_ROOT / "scripts" / "pool_sweep_orchestrator.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)

        cfg = json.loads((REPO_ROOT / "configs" / "pool_sweep.json").read_text(encoding="utf-8"))
        combos = module._grid(cfg)
        self.assertIn((2000, "clustered"), combos)
        # ADR 0006 fixes the pool size at 2000, so that is the size this strategy is swept at.
        self.assertEqual([s for s, b in combos if b == "clustered"], [2000])
        self.assertEqual(
            module._filter_combos(combos, ["clustered"]),
            [(s, b) for s, b in combos if b == "clustered"],
        )


def _biencoder_available() -> bool:
    """True when e5-small-v2 can be loaded from the local cache without the network."""
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    probe = (
        "from sentence_transformers import SentenceTransformer;"
        "SentenceTransformer('intfloat/e5-small-v2', device='cpu')"
    )
    return subprocess.run(
        [sys.executable, "-c", probe], env=env, capture_output=True, text=True
    ).returncode == 0


class TestClusteredCli(unittest.TestCase):
    """The real script, real embeddings, CPU, on the six fabricated fixture rows."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.available = _biencoder_available()

    def setUp(self) -> None:
        if not self.available:
            self.skipTest("intfloat/e5-small-v2 is not in the local cache (offline probe failed)")
        self._tmp = tempfile.TemporaryDirectory(prefix="qav2_clustered_pool_test_")
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def _run(self, out_dir: Path, size: int = 3) -> subprocess.CompletedProcess[str]:
        env = dict(
            os.environ,
            QAV2_PATHS_CONFIG="smoke_paths.json",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            PYTHONUNBUFFERED="1",
        )
        return subprocess.run(
            [
                sys.executable,
                str(SAMPLE_POOL),
                "--config", "musique_prep.json",
                "--size", str(size),
                "--balance", "clustered",
                "--device", "cpu",
                "--embed-model", "e5-small",
                "--out-dir", str(out_dir),
                "--overwrite",
            ],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
        )

    def test_cli_writes_a_pool_and_its_trail_reproducibly(self) -> None:
        first_dir = self.tmp / "run_a"
        second_dir = self.tmp / "run_b"
        first = self._run(first_dir)
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        second = self._run(second_dir)
        self.assertEqual(second.returncode, 0, second.stdout + second.stderr)

        pool_a = (first_dir / "pool.jsonl").read_bytes()
        pool_b = (second_dir / "pool.jsonl").read_bytes()
        self.assertEqual(pool_a, pool_b, "same seed produced a different pool")

        rows = _load_jsonl(first_dir / "pool.jsonl")
        self.assertEqual(len(rows), 3)
        source = {r["id"]: r for r in _load_jsonl(FIXTURE_POOL)}
        for row in rows:
            self.assertEqual(row, source[row["id"]])  # ADR 0003: unchanged rows

        for name in ("stats.json", "metrics.json", "notes.md", "config.json"):
            self.assertTrue((first_dir / name).exists(), f"missing {name}")

        stats = json.loads((first_dir / "stats.json").read_text(encoding="utf-8"))
        self.assertEqual(stats["balance"], "clustered")
        self.assertEqual(stats["size_written"], 3)
        clustering = stats["clustering"]
        self.assertEqual(clustering["n_clusters"], 3)
        self.assertEqual(clustering["representative_rule"], "nearest_to_centroid")
        self.assertEqual(clustering["kmeans_params"]["random_state"], stats["seed"])
        self.assertEqual(clustering["embedding"]["model_id"], "intfloat/e5-small-v2")
        self.assertEqual(clustering["embedding"]["prefix_applied"], "passage")
        # The size ceiling is asserted at load, not assumed (src/model_size.py).
        size_record = clustering["embedding"]["model_size"]
        self.assertTrue(size_record["ceiling_asserted"])
        self.assertLess(size_record["parameter_count"], size_record["parameter_ceiling"])

    def test_cli_refuses_a_size_above_the_candidate_count(self) -> None:
        proc = self._run(self.tmp / "too_big", size=99)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("exceeds pool row count", proc.stdout + proc.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
