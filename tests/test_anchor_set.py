#!/usr/bin/env python3
"""Checks for the frozen 45-item anchor set (supervisor item A4) and its rendering path.

Three things are pinned here, and they are the three a reader of
``configs/anchor_set_45.json`` or of a rendered sheet has to be able to trust:

- the **shape** of the committed list — 45 ids, 15 per hop, distinct, and each id's own hop
  prefix agreeing with the bucket it is filed under;
- its **derivability** — the draw is a pure function of (population, hops, n_per_hop, seed),
  so two runs of the script produce byte-identical ids, a different seed disagrees, and
  ``--verify`` exits non-zero on a tampered list (ADR 0027: the list is derived, not
  asserted);
- the **rendering** — ``scripts/render_anchor_set.py`` post-filters a per-item file down to
  exactly the anchor ids, side by side across arms, and refuses a file that does not cover
  them.

The membership check against the real ADR 0007 files is skipped when the data root is not
mounted, so the suite still runs on a box without the dataset; where it ran, it ran against
all 600 ids. Everything else runs on a synthetic fixture and needs no data and no GPU.

Run::

    .venv/bin/python -m pytest tests/test_anchor_set.py -q
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

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

BUILD_SCRIPT = REPO_ROOT / "scripts" / "build_anchor_set.py"
RENDER_SCRIPT = REPO_ROOT / "scripts" / "render_anchor_set.py"
DRAW_CONFIG = REPO_ROOT / "configs" / "anchor_set_draw.json"
IDS_FILE = REPO_ROOT / "configs" / "anchor_set_45.json"
SMOKE_PATHS_CONFIG = "configs/smoke_paths.json"


def _import(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BUILD = _import(BUILD_SCRIPT, "build_anchor_set")
RENDER = _import(RENDER_SCRIPT, "render_anchor_set")

IDS_PAYLOAD = json.loads(IDS_FILE.read_text(encoding="utf-8"))
DRAW_CONFIG_PAYLOAD = json.loads(DRAW_CONFIG.read_text(encoding="utf-8"))
ANCHOR_IDS, ANCHOR_HOP_OF, ANCHOR_HEADER = RENDER.load_anchor_ids(IDS_FILE)


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


HAVE_PINNED = _real_pinned_files() is not None


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

    def test_provenance_names_the_three_pinned_files_and_fingerprints_them(self) -> None:
        per_hop = IDS_PAYLOAD["source"]["per_hop"]
        self.assertEqual(sorted(per_hop), ["2", "3", "4"])
        for hop, entry in per_hop.items():
            self.assertTrue(entry["file"].endswith(f"_dev_sample_{hop}_hop_200.jsonl"), entry)
            self.assertEqual(entry["id_count"], 200, hop)
            for key in ("ids_sha256", "file_sha256"):
                self.assertRegex(entry[key], r"^[0-9a-f]{64}$", f"hop {hop} {key}")

    def test_the_generated_note_is_templated_from_the_draw_not_typed_in(self) -> None:
        # ADR 0027 in miniature: a number in the record is derived, never retyped. The
        # note quotes the seed and the totals, so the config holds placeholders and a
        # draw at another seed produces a note that says that seed.
        template = DRAW_CONFIG_PAYLOAD["ids_file_note"]
        for placeholder in ("{seed}", "{total}", "{n_per_hop}"):
            self.assertIn(placeholder, template)
        self.assertEqual(
            IDS_PAYLOAD["_note"],
            template.format(seed=42, total=45, n_per_hop=15),
        )
        self.assertNotIn("{", IDS_PAYLOAD["_note"])

    def test_it_records_the_interpreter_that_drew_it(self) -> None:
        self.assertRegex(IDS_PAYLOAD["python_version"], r"^\d+\.\d+\.\d+")
        # Context, not a parameter: --verify must not fail on a different interpreter.
        self.assertNotIn("python_version", BUILD._VERIFIED_FIELDS)

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


class TestBuildScriptCli(unittest.TestCase):
    """The script as a command: two writes agree byte-for-byte, and --verify bites."""

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

    @unittest.skipUnless(HAVE_PINNED, "the ADR 0007 pinned files are not on this box")
    def test_two_full_runs_of_the_script_produce_identical_ids(self) -> None:
        # Determinism of the whole command, not just the sampling function: config load,
        # file read, sort and draw. Only 'generated_utc' may differ between the two.
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "a.json"
            second = Path(tmp) / "b.json"
            for out in (first, second):
                proc = self._run("--write", "--out", str(out))
                self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            a = json.loads(first.read_text(encoding="utf-8"))
            b = json.loads(second.read_text(encoding="utf-8"))
            self.assertEqual(a["ids_by_hop"], b["ids_by_hop"])
            self.assertEqual(a["ids_by_hop"], IDS_PAYLOAD["ids_by_hop"])
            for field in BUILD._VERIFIED_FIELDS:
                self.assertEqual(a[field], b[field], field)

    @unittest.skipUnless(HAVE_PINNED, "the ADR 0007 pinned files are not on this box")
    def test_the_committed_list_reproduces(self) -> None:
        proc = self._run("--verify")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("VERIFIED", proc.stdout)

    @unittest.skipUnless(HAVE_PINNED, "the ADR 0007 pinned files are not on this box")
    def test_a_tampered_list_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tampered = Path(tmp) / "ids.json"
            payload = json.loads(json.dumps(IDS_PAYLOAD))
            payload["ids_by_hop"]["2"][0] = "2hop__000000_000000"
            tampered.write_text(json.dumps(payload), encoding="utf-8")
            proc = self._run("--verify", "--out", str(tampered))
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("ids_by_hop[2]", proc.stdout + proc.stderr)

    @unittest.skipUnless(HAVE_PINNED, "the ADR 0007 pinned files are not on this box")
    def test_a_different_seed_fails_against_the_committed_list(self) -> None:
        proc = self._run("--verify", "--seed", "43")
        self.assertNotEqual(proc.returncode, 0)

    def test_a_draw_at_another_seed_says_that_seed_in_its_own_note(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "seed43.json"
            proc = self._run(
                "--write", "--seed", "43", "--n-per-hop", "2", "--out", str(out),
                env_paths=SMOKE_PATHS_CONFIG,
            )
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            payload = json.loads(out.read_text(encoding="utf-8"))
        self.assertIn("seed 43", payload["_note"])
        self.assertNotIn("seed 42", payload["_note"])
        self.assertIn("The 6 anchor ids", payload["_note"])

    def test_a_different_interpreter_recorded_does_not_fail_verify(self) -> None:
        # python_version is provenance, not a parameter of the draw.
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "ids.json"
            self._run(
                "--write", "--n-per-hop", "2", "--out", str(out), env_paths=SMOKE_PATHS_CONFIG
            )
            payload = json.loads(out.read_text(encoding="utf-8"))
            payload["python_version"] = "3.0.0"
            out.write_text(json.dumps(payload), encoding="utf-8")
            verified = self._run(
                "--verify", "--n-per-hop", "2", "--out", str(out), env_paths=SMOKE_PATHS_CONFIG
            )
        self.assertEqual(verified.returncode, 0, verified.stdout + verified.stderr)

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

    @unittest.skipUnless(HAVE_PINNED, "the ADR 0007 pinned files are not on this box")
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

    @unittest.skipUnless(HAVE_PINNED, "the ADR 0007 pinned files are not on this box")
    def test_the_recorded_population_fingerprints_still_hold(self) -> None:
        files = _real_pinned_files()
        assert files is not None
        for hop, path in zip(DRAW_CONFIG_PAYLOAD["hops"], files):
            ids = BUILD.load_hop_ids(path, DRAW_CONFIG_PAYLOAD["questions_jsonl"]["id_field"])
            recorded = IDS_PAYLOAD["source"]["per_hop"][str(hop)]
            self.assertEqual(BUILD.ids_sha256(ids), recorded["ids_sha256"], f"hop {hop}")
            self.assertEqual(BUILD.file_sha256(path), recorded["file_sha256"], f"hop {hop}")
            self.assertEqual(len(ids), recorded["id_count"], f"hop {hop}")


# --------------------------------------------------------------------------------------
# The renderer, on a synthetic per-item fixture. No data root, no run, no GPU.
# --------------------------------------------------------------------------------------

#: A three-item anchor list in the committed schema. Small enough to read in the assertions.
TINY_ANCHOR = {
    "schema": "musique_anchor_set/1",
    "generated_by": "scripts/build_anchor_set.py",
    "generated_utc": "2026-09-20T00:00:00+00:00",
    "seed": 42,
    "hops": [2, 3],
    "n_per_hop": 2,
    "source": {"description": "synthetic", "per_hop": {}},
    "counts": {"2": 2, "3": 1, "total": 3},
    "ids_by_hop": {"2": ["2hop__aaa", "2hop__bbb"], "3": ["3hop1__ccc"]},
}


def _tiny_row(qid: str, *, pred: list[str], sari: float, question: str | None = None) -> dict[str, Any]:
    return {
        "item_id": qid,
        "question": question if question is not None else f"Question for {qid}?",
        "pred_steps": pred,
        "gold_steps": ["gold one", "#1 >> gold two"],
        "pred_step_count": len(pred),
        "gold_step_count": 2,
        "exact_match": 0.0,
        "step_f1": 0.5,
        "sari": sari,
        "ged_fallback": None,
    }


def _tiny_per_item(rows: list[dict[str, Any]], **header: Any) -> dict[str, Any]:
    payload = {
        "schema": RENDER.PER_ITEM_SCHEMA,
        "created_utc": "2026-09-20T00:00:00+00:00",
        "gold_path": "musique/dev_data/gold.jsonl",
        "composite_score_weights": {"step_f1": 0.5, "sari": 0.5},
        "composite_step_count_error_scale": 4,
        "items": rows,
    }
    payload.update(header)
    return payload


class TestRendererHelpers(unittest.TestCase):
    """The pieces the sheet is built from."""

    def test_arm_spec_takes_a_label_or_falls_back_to_the_parent_directory(self) -> None:
        self.assertEqual(
            RENDER.parse_arm_spec("exp-015 unguided=runs/exp-015/eval/eval_per_item.json"),
            ("exp-015 unguided", Path("runs/exp-015/eval/eval_per_item.json"), None),
        )
        self.assertEqual(
            RENDER.parse_arm_spec("runs/exp-011/exp008_full_train/eval_per_item.json"),
            ("exp008_full_train", Path("runs/exp-011/exp008_full_train/eval_per_item.json"), None),
        )

    def test_a_pipe_in_a_step_cannot_break_the_table(self) -> None:
        rendered = RENDER.cell("a | b", float_format="{:.4f}", empty="-")
        self.assertNotIn("| b", rendered.replace("\\|", ""))
        self.assertIn("\\|", rendered)

    def test_a_newline_in_a_step_is_folded_not_dropped(self) -> None:
        rendered = RENDER.cell("first\nsecond", float_format="{:.4f}", empty="-")
        self.assertEqual(rendered, "first<br>second")

    def test_zero_is_not_an_absence(self) -> None:
        self.assertEqual(RENDER.cell(0.0, float_format="{:.4f}", empty="-"), "0.0000")
        self.assertEqual(RENDER.cell(0, float_format="{:.4f}", empty="-"), "0")

    def test_present_but_null_and_absent_render_differently(self) -> None:
        fmt = {"float_format": "{:.4f}", "empty": "-", "null": "null"}
        self.assertEqual(RENDER.cell(None, **fmt), "null")
        self.assertEqual(RENDER.cell(RENDER.MISSING, **fmt), "-")
        self.assertNotEqual(RENDER.cell(None, **fmt), RENDER.cell(RENDER.MISSING, **fmt))


class TestRendererOnAFixture(unittest.TestCase):
    """End to end over the CLI, on three synthetic anchor items and two arms."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        tmp = Path(self.tmp.name)
        self.anchor = tmp / "anchor.json"
        self.anchor.write_text(json.dumps(TINY_ANCHOR), encoding="utf-8")

        # Arm A covers all three anchor ids plus a non-anchor row that must not appear.
        self.arm_a = tmp / "a" / "eval_per_item.json"
        self.arm_a.parent.mkdir(parents=True)
        self.arm_a.write_text(
            json.dumps(
                _tiny_per_item(
                    [
                        _tiny_row("2hop__aaa", pred=["a one", "a | two"], sari=0.25),
                        _tiny_row("2hop__bbb", pred=["b one"], sari=0.5),
                        _tiny_row("3hop1__ccc", pred=["c one", "c two", "c three"], sari=0.75),
                        _tiny_row("2hop__not_anchor", pred=["x"], sari=1.0),
                    ]
                )
            ),
            encoding="utf-8",
        )

        # Arm B is missing one anchor id, so the strict path must refuse it.
        self.arm_b = tmp / "b" / "eval_per_item.json"
        self.arm_b.parent.mkdir(parents=True)
        self.arm_b.write_text(
            json.dumps(
                _tiny_per_item(
                    [
                        _tiny_row("2hop__aaa", pred=["b-arm one", "b-arm two"], sari=0.125),
                        _tiny_row("3hop1__ccc", pred=["b-arm c"], sari=0.875),
                    ]
                )
            ),
            encoding="utf-8",
        )
        self.out = tmp / "sheet.md"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _run(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(RENDER_SCRIPT), "--anchor", str(self.anchor), *args],
            cwd=str(REPO_ROOT), capture_output=True, text=True,
        )

    def test_one_arm_renders_every_anchor_item_and_nothing_else(self) -> None:
        proc = self._run("--arm", f"armA={self.arm_a}", "--out", str(self.out))
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("armA: 3/3 anchor items", proc.stdout)
        sheet = self.out.read_text(encoding="utf-8")
        for qid in ("2hop__aaa", "2hop__bbb", "3hop1__ccc"):
            self.assertIn(qid, sheet)
        self.assertNotIn("not_anchor", sheet)
        # Anchor order: hops in the committed order, ids sorted within a hop.
        self.assertLess(sheet.index("2hop__aaa"), sheet.index("2hop__bbb"))
        self.assertLess(sheet.index("2hop__bbb"), sheet.index("3hop1__ccc"))

    def test_the_sheet_carries_question_gold_prediction_and_metrics(self) -> None:
        self._run("--arm", f"armA={self.arm_a}", "--out", str(self.out))
        sheet = self.out.read_text(encoding="utf-8")
        self.assertIn("**Question.** Question for 2hop__aaa?", sheet)
        self.assertIn("1. gold one", sheet)
        self.assertIn("2. #1 >> gold two", sheet)
        self.assertIn("a one", sheet)
        self.assertIn("`sari`", sheet)
        self.assertIn("0.2500", sheet)
        self.assertIn("hop 3", sheet)

    def test_a_pipe_in_a_prediction_is_escaped_in_the_table(self) -> None:
        self._run("--arm", f"armA={self.arm_a}", "--out", str(self.out))
        sheet = self.out.read_text(encoding="utf-8")
        self.assertIn("a \\| two", sheet)

    def test_two_arms_appear_as_two_columns_on_the_same_row(self) -> None:
        proc = self._run(
            "--arm", f"armA={self.arm_a}",
            "--arm", f"armB={self.arm_b}",
            "--allow-missing",
            "--out", str(self.out),
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        sheet = self.out.read_text(encoding="utf-8")
        self.assertIn("| metric | armA | armB |", sheet)
        self.assertIn("| a one | b-arm one |", sheet)
        self.assertIn("2/3", sheet)

    def test_an_arm_short_of_the_anchor_set_is_refused_unless_allowed(self) -> None:
        strict = self._run("--arm", f"armB={self.arm_b}", "--out", str(self.out))
        self.assertNotEqual(strict.returncode, 0)
        self.assertIn("do not cover the full anchor set", strict.stdout + strict.stderr)
        self.assertFalse(self.out.exists())

        allowed = self._run(
            "--arm", f"armB={self.arm_b}", "--allow-missing", "--out", str(self.out)
        )
        self.assertEqual(allowed.returncode, 0, allowed.stdout + allowed.stderr)
        sheet = self.out.read_text(encoding="utf-8")
        self.assertIn("is missing 1 anchor item(s)", sheet)
        self.assertIn("`2hop__bbb`", sheet)

    def test_a_file_that_is_not_the_evaluator_schema_is_refused_by_name(self) -> None:
        bad = Path(self.tmp.name) / "bad.json"
        bad.write_text(json.dumps({"schema": "something/else", "items": []}), encoding="utf-8")
        proc = self._run("--arm", f"bad={bad}", "--out", str(self.out))
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn(RENDER.PER_ITEM_SCHEMA, proc.stdout + proc.stderr)

    def test_a_duplicate_item_id_is_refused(self) -> None:
        dup = Path(self.tmp.name) / "dup.json"
        dup.write_text(
            json.dumps(
                _tiny_per_item(
                    [
                        _tiny_row("2hop__aaa", pred=["one"], sari=0.1),
                        _tiny_row("2hop__aaa", pred=["two"], sari=0.2),
                    ]
                )
            ),
            encoding="utf-8",
        )
        proc = self._run("--arm", f"dup={dup}", "--allow-missing", "--out", str(self.out))
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("duplicate item id", proc.stdout + proc.stderr)

    def test_arms_that_disagree_on_the_gold_are_flagged_not_silently_merged(self) -> None:
        other = Path(self.tmp.name) / "other.json"
        row = _tiny_row("2hop__aaa", pred=["one"], sari=0.1)
        row["gold_steps"] = ["a different gold"]
        other.write_text(json.dumps(_tiny_per_item([row])), encoding="utf-8")
        self._run(
            "--arm", f"armA={self.arm_a}",
            "--arm", f"other={other}",
            "--allow-missing",
            "--out", str(self.out),
        )
        sheet = self.out.read_text(encoding="utf-8")
        self.assertIn("disagree on the gold steps", sheet)

    def test_null_in_the_file_and_absent_from_the_file_look_different_in_the_sheet(self) -> None:
        # ged_fallback is present-and-null in every fixture row; 'only_in_a' exists in one
        # arm's rows and not the other's. The sheet must not show them the same way.
        other = Path(self.tmp.name) / "other.json"
        row = _tiny_row("2hop__aaa", pred=["one"], sari=0.1)
        row["only_in_other"] = 7
        other.write_text(json.dumps(_tiny_per_item([row])), encoding="utf-8")
        self._run(
            "--arm", f"armA={self.arm_a}",
            "--arm", f"other={other}",
            "--allow-missing",
            "--out", str(self.out),
        )
        sheet = self.out.read_text(encoding="utf-8")
        self.assertIn("| `ged_fallback` | null | null |", sheet)
        self.assertIn("| `only_in_other` | - | 7 |", sheet)

    def test_the_coverage_table_carries_the_scoring_settings_of_each_arm(self) -> None:
        self._run("--arm", f"armA={self.arm_a}", "--out", str(self.out))
        sheet = self.out.read_text(encoding="utf-8")
        for field in (
            "created_utc",
            "gold_path",
            "composite_score_weights",
            "composite_step_count_error_scale",
        ):
            self.assertIn(field, sheet.split("## 1.")[0], field)
        head = sheet.split("## 1.")[0]
        self.assertIn("musique/dev_data/gold.jsonl", head)
        self.assertIn('{"sari": 0.5, "step_f1": 0.5}', head)
        self.assertIn("2026-09-20T00:00:00+00:00", head)

    def test_arms_scored_under_different_composite_weights_are_flagged(self) -> None:
        other = Path(self.tmp.name) / "weights.json"
        other.write_text(
            json.dumps(
                _tiny_per_item(
                    [_tiny_row("2hop__aaa", pred=["one"], sari=0.1)],
                    composite_score_weights={"step_f1": 1.0},
                    composite_step_count_error_scale=8,
                )
            ),
            encoding="utf-8",
        )
        self._run(
            "--arm", f"armA={self.arm_a}",
            "--arm", f"other={other}",
            "--allow-missing",
            "--out", str(self.out),
        )
        head = self.out.read_text(encoding="utf-8").split("## 1.")[0]
        self.assertIn("scored under different `composite_score_weights`", head)
        self.assertIn("scored under different `composite_step_count_error_scale`", head)

    def test_arms_that_agree_on_the_settings_get_no_banner(self) -> None:
        self._run(
            "--arm", f"armA={self.arm_a}",
            "--arm", f"armB={self.arm_b}",
            "--allow-missing",
            "--out", str(self.out),
        )
        head = self.out.read_text(encoding="utf-8").split("## 1.")[0]
        self.assertNotIn("scored under different", head)

    def test_an_anchor_list_whose_hops_and_buckets_disagree_is_a_written_refusal(self) -> None:
        broken = Path(self.tmp.name) / "broken.json"
        payload = json.loads(json.dumps(TINY_ANCHOR))
        payload["hops"] = [2, 3, 4]  # no '4' bucket in ids_by_hop
        broken.write_text(json.dumps(payload), encoding="utf-8")
        proc = subprocess.run(
            [
                sys.executable, str(RENDER_SCRIPT), "--anchor", str(broken),
                "--arm", f"armA={self.arm_a}", "--allow-missing", "--out", str(self.out),
            ],
            cwd=str(REPO_ROOT), capture_output=True, text=True,
        )
        self.assertNotEqual(proc.returncode, 0)
        combined = proc.stdout + proc.stderr
        self.assertNotIn("KeyError", combined)
        self.assertIn("'ids_by_hop' has no bucket", combined)

    def test_duplicate_column_labels_are_refused(self) -> None:
        proc = self._run(
            "--arm", f"same={self.arm_a}",
            "--arm", f"same={self.arm_b}",
            "--allow-missing",
            "--out", str(self.out),
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("used twice", proc.stdout + proc.stderr)


class TestOutPathCannotEnterGit(unittest.TestCase):
    """--out is refused when the path is inside the repo and git does not ignore it.

    The sheet carries question text and gold decompositions. CLAUDE.md's "data never
    enters git" is behavioural, so the check asks the only authority that cannot drift
    from what ``git add`` would do: ``git check-ignore``.
    """

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        tmp = Path(self.tmp.name)
        self.anchor = tmp / "anchor.json"
        self.anchor.write_text(json.dumps(TINY_ANCHOR), encoding="utf-8")
        self.arm = tmp / "arm" / "eval_per_item.json"
        self.arm.parent.mkdir(parents=True)
        self.arm.write_text(
            json.dumps(
                _tiny_per_item(
                    [
                        _tiny_row("2hop__aaa", pred=["one"], sari=0.1),
                        _tiny_row("2hop__bbb", pred=["two"], sari=0.2),
                        _tiny_row("3hop1__ccc", pred=["three"], sari=0.3),
                    ]
                )
            ),
            encoding="utf-8",
        )
        # Every sheet this class writes goes here, so the repo is left as it was found.
        self.written: list[Path] = []

    def tearDown(self) -> None:
        for path in self.written:
            if path.exists():
                path.unlink()
        self.tmp.cleanup()

    def _render_to(self, out: Path) -> subprocess.CompletedProcess:
        self.written.append(out)
        return subprocess.run(
            [
                sys.executable, str(RENDER_SCRIPT),
                "--anchor", str(self.anchor),
                "--arm", f"armA={self.arm}",
                "--out", str(out),
            ],
            cwd=str(REPO_ROOT), capture_output=True, text=True,
        )

    def test_a_gitignored_path_inside_the_repo_is_allowed(self) -> None:
        # runs/ is gitignored, which is where the tool is meant to write.
        out = REPO_ROOT / "runs" / "anchor" / "_test_out_path_guard.md"
        proc = self._render_to(out)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertTrue(out.exists())

    def test_a_tracked_directory_inside_the_repo_is_refused_and_writes_nothing(self) -> None:
        out = REPO_ROOT / "docs" / "_test_out_path_guard.md"
        proc = self._render_to(out)
        self.assertNotEqual(proc.returncode, 0)
        self.assertFalse(out.exists())
        combined = proc.stdout + proc.stderr
        self.assertIn("NOT", combined)
        self.assertIn("gitignore", combined.lower())
        self.assertIn("data never enters git", combined)

    def test_the_repo_root_itself_is_refused(self) -> None:
        out = REPO_ROOT / "_test_out_path_guard.md"
        proc = self._render_to(out)
        self.assertNotEqual(proc.returncode, 0)
        self.assertFalse(out.exists())

    def test_a_path_that_climbs_back_into_the_repo_is_still_refused(self) -> None:
        # The check resolves the path first, so '..' cannot be used to dodge it.
        out = REPO_ROOT / "runs" / ".." / "docs" / "_test_out_path_guard.md"
        proc = self._render_to(out)
        self.assertNotEqual(proc.returncode, 0)
        self.assertFalse((REPO_ROOT / "docs" / "_test_out_path_guard.md").exists())

    def test_a_path_outside_the_working_tree_is_allowed(self) -> None:
        out = Path(self.tmp.name) / "sheet.md"
        proc = self._render_to(out)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertTrue(out.exists())

    def test_the_guard_refuses_before_any_input_is_read(self) -> None:
        # A bad --out must not be discovered after the per-item files have been loaded:
        # the refusal has to be reachable with inputs that do not even exist.
        proc = subprocess.run(
            [
                sys.executable, str(RENDER_SCRIPT),
                "--anchor", str(REPO_ROOT / "does_not_exist.json"),
                "--arm", "armA=/nonexistent/eval_per_item.json",
                "--out", str(REPO_ROOT / "docs" / "_test_out_path_guard.md"),
            ],
            cwd=str(REPO_ROOT), capture_output=True, text=True,
        )
        self.assertNotEqual(proc.returncode, 0)
        combined = proc.stdout + proc.stderr
        self.assertIn("refusing --out", combined)
        self.assertNotIn("anchor id list not found", combined)


if __name__ == "__main__":
    unittest.main()
