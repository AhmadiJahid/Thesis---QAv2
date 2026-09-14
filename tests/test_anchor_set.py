#!/usr/bin/env python3
"""Checks for the frozen 45-item anchor set (supervisor item A4) and its selection path.

Three things are pinned here, and they are the three a reader of
``configs/anchor_set_ids.json`` has to be able to trust:

- the **shape** of the committed list — 45 ids, 15 per hop, distinct, and each id's own hop
  prefix agreeing with the bucket it is filed under;
- its **derivability** — the draw is a pure function of (population, hops, n_per_hop, seed),
  so two calls agree, a different seed disagrees, and the script's ``--verify`` exits
  non-zero on a tampered list (ADR 0027: the list is derived, not asserted);
- the **selection path** — ``scripts/anchor_inspect.py`` returns exactly the anchor items
  from a file that contains them, and refuses when it does not.

The membership check against the real ADR 0007 files is skipped when the data root is not
mounted, so the suite still runs on a box without the dataset; where it ran, it ran against
all 600 ids.

Run::

    .venv/bin/python -m unittest tests.test_anchor_set -v
"""
from __future__ import annotations

import csv
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

BUILD_SCRIPT = REPO_ROOT / "scripts" / "build_anchor_set.py"
INSPECT_SCRIPT = REPO_ROOT / "scripts" / "anchor_inspect.py"
DRAW_CONFIG = REPO_ROOT / "configs" / "anchor_set_draw.json"
IDS_FILE = REPO_ROOT / "configs" / "anchor_set_ids.json"
SMOKE_PATHS_CONFIG = "configs/smoke_paths.json"


def _import(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BUILD = _import(BUILD_SCRIPT, "build_anchor_set")
INSPECT = _import(INSPECT_SCRIPT, "anchor_inspect")

IDS_PAYLOAD = json.loads(IDS_FILE.read_text(encoding="utf-8"))
DRAW_CONFIG_PAYLOAD = json.loads(DRAW_CONFIG.read_text(encoding="utf-8"))
ANCHOR_IDS, ANCHOR_HOP_OF, _ANCHOR_HEADER = INSPECT.load_anchor_ids(IDS_FILE)


def _real_pinned_files() -> list[Path] | None:
    """The three ADR 0007 files, or None when the data root is not on this box."""
    from run_config import load_paths, require, resolve_path  # noqa: PLC0415

    # The real data root, never the fixture tree: this check is about the ADR 0007 files.
    saved = os.environ.pop("QAV2_PATHS_CONFIG", None)
    try:
        paths_cfg = load_paths(DRAW_CONFIG_PAYLOAD["paths_config"])
    finally:
        if saved is not None:
            os.environ["QAV2_PATHS_CONFIG"] = saved
    template = require(paths_cfg, "datasets." + DRAW_CONFIG_PAYLOAD["questions_template_key"])
    data_root = Path(paths_cfg["data_root_resolved"])
    files = [resolve_path(template.format(hop=h), data_root) for h in DRAW_CONFIG_PAYLOAD["hops"]]
    return files if all(f.exists() for f in files) else None


class TestCommittedIdList(unittest.TestCase):
    """The shape the supervisor asked for: 15 per hop, 45 in total, fixed."""

    def test_counts_are_fifteen_per_hop_and_forty_five_total(self) -> None:
        self.assertEqual(IDS_PAYLOAD["counts"], {"2": 15, "3": 15, "4": 15, "total": 45})
        self.assertEqual(
            {hop: len(ids) for hop, ids in IDS_PAYLOAD["ids_by_hop"].items()},
            {"2": 15, "3": 15, "4": 15},
        )
        self.assertEqual(len(ANCHOR_IDS), 45)

    def test_ids_are_distinct(self) -> None:
        self.assertEqual(len(set(ANCHOR_IDS)), 45)

    def test_each_id_sits_under_the_hop_its_own_prefix_declares(self) -> None:
        # MuSiQue ids are '<n>hop...' (2hop__, 3hop1__, 4hop2__ ...). An id filed under the
        # wrong bucket would give a "15 per hop" that is not 15 per hop.
        for qid, hop in ANCHOR_HOP_OF.items():
            self.assertTrue(qid.startswith(f"{hop}hop"), f"{qid} filed under hop {hop}")

    def test_the_committed_file_is_the_parameters_it_was_drawn_with(self) -> None:
        for key in ("seed", "hops", "n_per_hop"):
            self.assertEqual(IDS_PAYLOAD[key], DRAW_CONFIG_PAYLOAD[key], key)
        self.assertEqual(IDS_PAYLOAD["seed"], 42)
        self.assertEqual(IDS_PAYLOAD["schema"], BUILD.ANCHOR_SET_SCHEMA)

    def test_it_carries_ids_and_provenance_only_never_data(self) -> None:
        # CLAUDE.md: data never enters git. The artifact may name files and count ids; it
        # may not carry question text or gold decompositions.
        blob = json.dumps(IDS_PAYLOAD)
        for banned in ("question", "decomposition", "answer", "paragraph"):
            self.assertNotIn(f'"{banned}"', blob, f"the id list carries a {banned!r} field")


class TestDrawIsDeterministic(unittest.TestCase):
    """The draw is a function of (population, hops, n_per_hop, seed) and nothing else."""

    POPULATION = {
        2: [f"2hop__x{i:03d}" for i in range(200)],
        3: [f"3hop1__x{i:03d}" for i in range(200)],
        4: [f"4hop1__x{i:03d}" for i in range(200)],
    }
    HOPS = [2, 3, 4]

    def test_two_calls_agree(self) -> None:
        a = BUILD.select_anchor_ids(self.POPULATION, self.HOPS, 15, 42)
        b = BUILD.select_anchor_ids(self.POPULATION, self.HOPS, 15, 42)
        self.assertEqual(a, b)
        self.assertEqual({h: len(v) for h, v in a.items()}, {"2": 15, "3": 15, "4": 15})

    def test_a_different_seed_draws_a_different_set(self) -> None:
        a = BUILD.select_anchor_ids(self.POPULATION, self.HOPS, 15, 42)
        b = BUILD.select_anchor_ids(self.POPULATION, self.HOPS, 15, 43)
        self.assertNotEqual(a, b)

    def test_source_order_does_not_move_the_draw(self) -> None:
        # The population is sorted before sampling, so a re-ordered source file (or a
        # re-generated one) cannot silently change which items are inspected by hand.
        shuffled = {hop: list(reversed(ids)) for hop, ids in self.POPULATION.items()}
        self.assertEqual(
            BUILD.select_anchor_ids(self.POPULATION, self.HOPS, 15, 42),
            BUILD.select_anchor_ids(shuffled, self.HOPS, 15, 42),
        )

    def test_a_population_smaller_than_the_draw_is_a_refusal(self) -> None:
        with self.assertRaises(SystemExit):
            BUILD.select_anchor_ids({2: ["2hop__a", "2hop__b"]}, [2], 15, 42)


class TestVerifyMode(unittest.TestCase):
    """``--verify`` re-derives the list and exits non-zero when it differs."""

    def _run(self, *args: str, env_paths: str | None = None) -> subprocess.CompletedProcess:
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        if env_paths is None:
            env.pop("QAV2_PATHS_CONFIG", None)
        else:
            env["QAV2_PATHS_CONFIG"] = env_paths
        return subprocess.run(
            [sys.executable, str(BUILD_SCRIPT), *args],
            cwd=str(REPO_ROOT), env=env, capture_output=True, text=True,
        )

    @unittest.skipIf(_real_pinned_files() is None, "the ADR 0007 pinned files are not on this box")
    def test_the_committed_list_reproduces(self) -> None:
        proc = self._run("--verify")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("VERIFIED", proc.stdout)

    @unittest.skipIf(_real_pinned_files() is None, "the ADR 0007 pinned files are not on this box")
    def test_a_tampered_list_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tampered = Path(tmp) / "ids.json"
            payload = json.loads(json.dumps(IDS_PAYLOAD))
            payload["ids_by_hop"]["2"][0] = "2hop__000000_000000"
            tampered.write_text(json.dumps(payload), encoding="utf-8")
            proc = self._run("--verify", "--out", str(tampered))
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("ids_by_hop[2]", proc.stdout + proc.stderr)

    @unittest.skipIf(_real_pinned_files() is None, "the ADR 0007 pinned files are not on this box")
    def test_a_different_seed_fails_against_the_committed_list(self) -> None:
        proc = self._run("--verify", "--seed", "43")
        self.assertNotEqual(proc.returncode, 0)

    def test_write_then_verify_round_trips_on_the_fixtures(self) -> None:
        # The tiny CPU-only path: the fixture tree holds 3 ids per hop, so n_per_hop 2 is a
        # real draw over a real (fabricated) population, and the same command verifies it.
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "anchor_ids.json"
            written = self._run(
                "--write", "--n-per-hop", "2", "--out", str(out), env_paths=SMOKE_PATHS_CONFIG
            )
            self.assertEqual(written.returncode, 0, written.stdout + written.stderr)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["counts"], {"2": 2, "3": 2, "4": 2, "total": 6})
            verified = self._run(
                "--verify", "--n-per-hop", "2", "--out", str(out), env_paths=SMOKE_PATHS_CONFIG
            )
            self.assertEqual(verified.returncode, 0, verified.stdout + verified.stderr)
            # And --write refuses to clobber a list that already exists.
            again = self._run(
                "--write", "--n-per-hop", "2", "--out", str(out), env_paths=SMOKE_PATHS_CONFIG
            )
            self.assertNotEqual(again.returncode, 0)


class TestAnchorIdsAreInThePinned600(unittest.TestCase):
    """Every anchor id is one of the ADR 0007 600 — checked against the files themselves."""

    @unittest.skipIf(_real_pinned_files() is None, "the ADR 0007 pinned files are not on this box")
    def test_membership_and_population_size(self) -> None:
        files = _real_pinned_files()
        assert files is not None
        pinned: set[str] = set()
        for path in files:
            ids = BUILD.load_hop_ids(path, DRAW_CONFIG_PAYLOAD["questions_jsonl"]["id_field"])
            self.assertEqual(len(ids), 200, path)
            pinned.update(ids)
        self.assertEqual(len(pinned), 600)
        self.assertTrue(set(ANCHOR_IDS) <= pinned, sorted(set(ANCHOR_IDS) - pinned))

    @unittest.skipIf(_real_pinned_files() is None, "the ADR 0007 pinned files are not on this box")
    def test_the_recorded_population_fingerprints_still_hold(self) -> None:
        files = _real_pinned_files()
        assert files is not None
        for hop, path in zip(DRAW_CONFIG_PAYLOAD["hops"], files):
            ids = BUILD.load_hop_ids(path, DRAW_CONFIG_PAYLOAD["questions_jsonl"]["id_field"])
            recorded = IDS_PAYLOAD["source"]["per_hop"][str(hop)]
            self.assertEqual(BUILD.ids_sha256(ids), recorded["ids_sha256"], f"hop {hop}")
            self.assertEqual(len(ids), recorded["id_count"], f"hop {hop}")


def _fake_prediction(qid: str) -> dict[str, Any]:
    return {"query_id": qid, "question": f"q for {qid}", "decomposition": "a | b"}


def _fake_per_item(qid: str) -> dict[str, Any]:
    hop = int(ANCHOR_HOP_OF.get(qid, "2"))
    return {
        "item_id": qid,
        "question": f"q for {qid}",
        "pred_steps": ["a", "b"],
        "gold_steps": ["a", "b"],
        "gold_hop_count": hop,
        "break_exact_match": 1.0,
        "sari": 0.5,
        "ged": 0.0,
        "chain_validity": 1.0,
        "hop_count_exact_match": 1.0,
        "step_f1": 1.0,
    }


class TestAnchorFilter(unittest.TestCase):
    """The id filter returns exactly the anchor items — no more, no fewer."""

    def _run(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(INSPECT_SCRIPT), *args],
            cwd=str(REPO_ROOT), env={**os.environ, "PYTHONUNBUFFERED": "1"},
            capture_output=True, text=True,
        )

    def test_filter_returns_exactly_the_forty_five(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "predictions.json"
            decoys = [_fake_prediction(f"2hop__decoy_{i}") for i in range(20)]
            src.write_text(
                json.dumps([_fake_prediction(q) for q in ANCHOR_IDS] + decoys), encoding="utf-8"
            )
            out = Path(tmp) / "anchor_predictions.json"
            proc = self._run("--filter", str(src), "--filter-out", str(out))
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            selected = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(len(selected), 45)
        self.assertEqual([r["query_id"] for r in selected], ANCHOR_IDS)

    def test_filter_keeps_jsonl_as_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "retrieval.jsonl"
            src.write_text(
                "".join(json.dumps(_fake_prediction(q)) + "\n" for q in ANCHOR_IDS + ["2hop__d"]),
                encoding="utf-8",
            )
            out = Path(tmp) / "anchor_retrieval.jsonl"
            proc = self._run("--filter", str(src), "--filter-out", str(out))
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            lines = [l for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
        self.assertEqual(len(lines), 45)

    def test_a_file_missing_an_anchor_item_is_a_refusal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "predictions.json"
            src.write_text(
                json.dumps([_fake_prediction(q) for q in ANCHOR_IDS[:-1]]), encoding="utf-8"
            )
            out = Path(tmp) / "out.json"
            proc = self._run("--filter", str(src), "--filter-out", str(out))
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn(ANCHOR_IDS[-1], proc.stdout + proc.stderr)
            allowed = self._run(
                "--filter", str(src), "--filter-out", str(out), "--allow-missing"
            )
            self.assertEqual(allowed.returncode, 0, allowed.stdout + allowed.stderr)
            self.assertEqual(len(json.loads(out.read_text(encoding="utf-8"))), 44)

    def test_a_duplicate_id_is_a_refusal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "predictions.json"
            rows = [_fake_prediction(q) for q in ANCHOR_IDS] + [_fake_prediction(ANCHOR_IDS[0])]
            src.write_text(json.dumps(rows), encoding="utf-8")
            proc = self._run("--filter", str(src), "--filter-out", str(Path(tmp) / "o.json"))
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("duplicate id", proc.stdout + proc.stderr)


class TestAnchorDump(unittest.TestCase):
    """The human-readable dump: 45 rows, and no metric recomputed."""

    def test_dump_writes_forty_five_rows_and_the_run_trail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "eval_per_item.json"
            src.write_text(
                json.dumps(
                    {
                        "schema": "musique_decomposition_per_item/1",
                        "composite_score_weights": {"step_f1_macro": 1.0},
                        "composite_step_count_error_scale": 3.0,
                        "items": [_fake_per_item(q) for q in ANCHOR_IDS]
                        + [_fake_per_item("2hop__not_an_anchor")],
                    }
                ),
                encoding="utf-8",
            )
            out_dir = Path(tmp) / "dump"
            proc = subprocess.run(
                [
                    sys.executable, str(INSPECT_SCRIPT), "--per-item", str(src),
                    "--out-dir", str(out_dir), "--label", "unit",
                ],
                cwd=str(REPO_ROOT), env={**os.environ, "PYTHONUNBUFFERED": "1"},
                capture_output=True, text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            with (out_dir / "anchor_review.csv").open(encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            metrics = json.loads((out_dir / "anchor_metrics.json").read_text(encoding="utf-8"))
            dumped = json.loads((out_dir / "anchor_per_item.json").read_text(encoding="utf-8"))
            for name in ("anchor_config.json", "anchor_metrics.json", "anchor_notes.md"):
                self.assertTrue((out_dir / name).exists(), name)

        self.assertEqual(len(rows), 45)
        self.assertEqual([r["query_id"] for r in rows], ANCHOR_IDS)
        self.assertEqual(metrics["anchor_items_found"], 45)
        self.assertEqual(metrics["anchor_items_found_per_hop"], {"2": 15, "3": 15, "4": 15})
        self.assertEqual(metrics["anchor_items_missing"], [])
        self.assertEqual(len(dumped["items"]), 45)
        # The values are carried through, not recomputed: the fabricated 0.5 SARI survives.
        self.assertEqual(rows[0]["sari"], "0.5")
        self.assertEqual(rows[0]["gold_decomposition"], "a | b")


if __name__ == "__main__":
    unittest.main()
