#!/usr/bin/env python3
"""The anchor prompts: which rows get one logged, and how the sheet finds them again.

Jahid's question on 2026-09-20 was *"will I be able to compare 45 examples with their
prompts in each run?"* Before this, no: the decomposer logged every Nth prompt **by
position**, so which 45 items had a prompt depended on where they happened to land in the
row order, and the file name carried the position rather than the id. Two halves close
that, and both are pinned here:

- ``run_decomposer.py`` writes one prompt file per anchor id, named by the id, *in addition
  to* the every-Nth positional sample, whose file names must not move (existing parity
  checks read them);
- ``render_anchor_set.py`` finds those files again by id and joins ``decomposition_raw``
  from the same run's ``results.json``, rendering an explicit marker — never silence — when
  a prompt is not there.

No GPU, no model weights, no data files: the selection is a pure function and the renderer
runs over a synthetic run directory in a temp dir.

Written as ``unittest.TestCase`` methods rather than bare pytest functions on purpose: the
project venv has no pytest (see ``tests/test_suite_collectability.py``).

Run::

    .venv/bin/python -m unittest tests.test_anchor_prompt_logging
    python -m pytest tests/test_anchor_prompt_logging.py
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
sys.path.insert(0, str(REPO_ROOT / "components" / "decomposer"))

import anchor_set as ANCHOR  # noqa: E402
import run_decomposer as rd  # noqa: E402
from run_config import PATHS_CONFIG_ENV  # noqa: E402

RENDER_SCRIPT = REPO_ROOT / "scripts" / "render_anchor_set.py"
RUNNER = REPO_ROOT / "components" / "decomposer" / "run_decomposer.py"
FIXTURE_RETRIEVAL = REPO_ROOT / "tests" / "fixtures" / "retrieval" / "top5_musique_conditions.jsonl"
ANCHOR_IDS_FILE = REPO_ROOT / "configs" / "anchor_set_45.json"


def _import(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RENDER = _import(RENDER_SCRIPT, "render_anchor_set_prompts")

#: Two anchor ids, one of them deliberately NOT on a multiple of the sampling interval.
ANCHORS = frozenset({"2hop__aaa", "3hop1__ccc"})


class TestWhichRowsGetAPromptLogged(unittest.TestCase):
    """``select_prompt_log_paths`` — the one place the two selectors meet."""

    def _paths(self, **kwargs: Any) -> list[str]:
        defaults: dict[str, Any] = {
            "position": 1,
            "gold_hop": 2,
            "query_id": "2hop__not_anchor",
            "prompt_log_every": 5,
            "anchor_ids": ANCHORS,
            "dry_run": False,
        }
        defaults.update(kwargs)
        return [p.name for p in rd.select_prompt_log_paths(Path("/tmp/p"), **defaults)]

    def test_an_anchor_is_logged_at_a_position_the_sampler_skips(self) -> None:
        # Position 7 is not a multiple of 5, so the old code logged nothing for this row.
        names = self._paths(position=7, query_id="2hop__aaa")
        self.assertEqual(names, ["anchor_2hop__aaa_hop2.txt"])

    def test_an_anchor_on_a_sampling_boundary_gets_both_files(self) -> None:
        # The positional file is not replaced by the anchor file: existing parity checks
        # read prompt_idx*.txt by name and the sample must stay exactly what it was.
        names = self._paths(position=10, query_id="2hop__aaa")
        self.assertEqual(names, ["prompt_idx0010_hop2.txt", "anchor_2hop__aaa_hop2.txt"])

    def test_a_non_anchor_off_the_boundary_is_not_logged_at_all(self) -> None:
        self.assertEqual(self._paths(position=7), [])

    def test_a_non_anchor_on_the_boundary_keeps_its_old_single_file(self) -> None:
        self.assertEqual(self._paths(position=10), ["prompt_idx0010_hop2.txt"])

    def test_the_positional_sample_is_unchanged_across_600_rows(self) -> None:
        logged = [
            i
            for i in range(1, 601)
            if any(
                n.startswith("prompt_idx")
                for n in self._paths(position=i, query_id=f"q{i}")
            )
        ]
        self.assertEqual(logged, list(range(5, 601, 5)))
        self.assertEqual(len(logged), 120)

    def test_a_dry_run_logs_every_position_and_still_writes_anchor_files(self) -> None:
        self.assertEqual(self._paths(position=7, dry_run=True), ["prompt_idx0007_hop2.txt"])
        self.assertEqual(
            self._paths(position=7, query_id="2hop__aaa", dry_run=True),
            ["prompt_idx0007_hop2.txt", "anchor_2hop__aaa_hop2.txt"],
        )

    def test_an_empty_anchor_set_disables_anchor_logging(self) -> None:
        # logging.anchor_ids_file: null — the MetaQA config. Nothing anchor-shaped appears.
        self.assertEqual(self._paths(position=7, query_id="2hop__aaa", anchor_ids=frozenset()), [])
        self.assertEqual(
            self._paths(position=10, query_id="2hop__aaa", anchor_ids=frozenset()),
            ["prompt_idx0010_hop2.txt"],
        )

    def test_a_row_with_no_query_id_can_never_be_an_anchor(self) -> None:
        # The MetaQA 'lines' question format carries no ids.
        self.assertEqual(self._paths(position=7, query_id=None), [])

    def test_the_hop_suffix_follows_the_gold_depth(self) -> None:
        names = self._paths(position=3, gold_hop=4, query_id="3hop1__ccc")
        self.assertEqual(names, ["anchor_3hop1__ccc_hop4.txt"])


class TestAnchorIdsFileResolution(unittest.TestCase):
    """``logging.anchor_ids_file``: a path enables, ``null`` disables, junk is refused."""

    def test_null_disables_and_yields_no_ids(self) -> None:
        ids, path = ANCHOR.load_anchor_ids_for_logging(None, REPO_ROOT)
        self.assertEqual(ids, frozenset())
        self.assertIsNone(path)

    def test_the_committed_list_resolves_to_45_ids(self) -> None:
        ids, path = ANCHOR.load_anchor_ids_for_logging("configs/anchor_set_45.json", REPO_ROOT)
        self.assertEqual(len(ids), 45)
        self.assertEqual(path, REPO_ROOT / "configs" / "anchor_set_45.json")

    def test_both_shared_configs_declare_the_key(self) -> None:
        # Required, not defaulted (the few_shot.ordering precedent): a run must not acquire
        # an inspection set by omission, and the MetaQA config has to say 'off' out loud.
        musique = json.loads(
            (REPO_ROOT / "configs" / "decomposer_musique.json").read_text(encoding="utf-8")
        )
        metaqa = json.loads(
            (REPO_ROOT / "configs" / "decomposer.json").read_text(encoding="utf-8")
        )
        self.assertIn("anchor_ids_file", musique.get("logging", {}))
        self.assertIn("anchor_ids_file", metaqa.get("logging", {}))
        self.assertEqual(musique["logging"]["anchor_ids_file"], "configs/anchor_set_45.json")
        self.assertIsNone(metaqa["logging"]["anchor_ids_file"])

    def test_a_non_string_non_null_value_is_refused(self) -> None:
        with self.assertRaises(ANCHOR.AnchorSetError):
            ANCHOR.load_anchor_ids_for_logging(42, REPO_ROOT)
        with self.assertRaises(ANCHOR.AnchorSetError):
            ANCHOR.load_anchor_ids_for_logging("   ", REPO_ROOT)

    def test_an_id_that_cannot_be_a_file_name_is_refused_not_mangled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "anchor.json"
            bad.write_text(
                json.dumps(
                    {
                        "schema": ANCHOR.ANCHOR_SET_SCHEMA,
                        "hops": [2],
                        "ids_by_hop": {"2": ["2hop__ok", "2hop/../escape"]},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ANCHOR.AnchorSetError) as ctx:
                ANCHOR.load_anchor_ids_for_logging(str(bad), REPO_ROOT)
            self.assertIn("file name", str(ctx.exception))


class TestRendererJoinsPromptAndRaw(unittest.TestCase):
    """The reader half: a prompt found by id, a raw generation joined on query_id, and a
    stated marker when either is absent."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        tmp = Path(self.tmp.name)
        self.run_dir = tmp / "run"
        (self.run_dir / "prompts_log").mkdir(parents=True)
        # Only ONE of the two anchor ids has a logged prompt: the other exercises the marker.
        (self.run_dir / "prompts_log" / "anchor_2hop__aaa_hop2.txt").write_text(
            "--- Log Header ---\nFew-shot ordering: as_is (config)\n\n"
            "--- Prompt (plain) ---\nPROMPT BODY FOR aaa\n",
            encoding="utf-8",
        )
        (self.run_dir / "results.json").write_text(
            json.dumps(
                [
                    {"query_id": "2hop__aaa", "decomposition_raw": "RAW FOR aaa"},
                    {"query_id": "3hop1__ccc", "decomposition_raw": "RAW FOR ccc"},
                ]
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _arm(self, *, with_run: bool = True) -> Any:
        results = RENDER.load_results(self.run_dir / "results.json") if with_run else None
        return RENDER.Arm(
            "armA", Path("eval_per_item.json"), {}, {},
            run_dir=self.run_dir if with_run else None,
            results=results,
            prompts_dir=(self.run_dir / "prompts_log") if with_run else None,
            prompts_dir_source="run" if with_run else None,
        )

    def test_the_arm_spec_carries_an_optional_run_directory(self) -> None:
        self.assertEqual(
            RENDER.parse_arm_spec("lbl=a/eval_per_item.json@b/run"),
            ("lbl", Path("a/eval_per_item.json"), Path("b/run")),
        )
        self.assertEqual(
            RENDER.parse_arm_spec("a/eval_per_item.json@b/run"),
            ("a", Path("a/eval_per_item.json"), Path("b/run")),
        )

    def test_a_present_prompt_and_raw_are_both_rendered(self) -> None:
        section = "\n".join(RENDER.render_prompt_section(self._arm(), "2hop__aaa"))
        self.assertIn("PROMPT BODY FOR aaa", section)
        self.assertIn("Few-shot ordering: as_is (config)", section)
        self.assertIn("RAW FOR aaa", section)
        self.assertIn("<details>", section)
        self.assertNotIn("prompt not logged", section)

    def test_an_absent_prompt_renders_a_marker_not_silence(self) -> None:
        section = "\n".join(RENDER.render_prompt_section(self._arm(), "3hop1__ccc"))
        self.assertIn("prompt not logged for this run", section)
        self.assertIn("--prompts-dir", section)
        # The raw generation is still joined: only the prompt is missing.
        self.assertIn("RAW FOR ccc", section)

    def test_no_run_directory_says_so_for_both_halves(self) -> None:
        section = "\n".join(RENDER.render_prompt_section(self._arm(with_run=False), "2hop__aaa"))
        self.assertIn("no run directory was given for this arm", section)
        self.assertNotIn("RAW FOR aaa", section)

    def test_an_id_absent_from_results_json_says_so(self) -> None:
        section = "\n".join(RENDER.render_prompt_section(self._arm(), "2hop__absent"))
        self.assertIn("no `results.json` row for this id", section)

    def test_prompt_coverage_counts_what_is_actually_there(self) -> None:
        prompts, raws = RENDER.prompt_coverage(self._arm(), ["2hop__aaa", "3hop1__ccc"])
        self.assertEqual((prompts, raws), (1, 2))

    def test_a_prompt_containing_a_code_fence_cannot_break_out_of_its_block(self) -> None:
        # Prompt bodies really do contain ``` (few-shot exemplars, model output). A fixed
        # three-backtick fence would be closed early and the rest of the prompt would leak
        # onto the page looking like a rendering quirk rather than a truncated prompt.
        body = "before\n```\ninside\n```\nafter"
        block = "\n".join(RENDER.fenced(body))
        self.assertTrue(block.startswith("````"))
        self.assertIn("inside", block)

    def test_a_duplicate_query_id_in_results_is_refused(self) -> None:
        dup = Path(self.tmp.name) / "dup.json"
        dup.write_text(
            json.dumps([{"query_id": "x", "decomposition_raw": "a"},
                        {"query_id": "x", "decomposition_raw": "b"}]),
            encoding="utf-8",
        )
        with self.assertRaises(SystemExit):
            RENDER.load_results(dup)


class TestARegenerationMustReproduceTheRunItIsShownBeside(unittest.TestCase):
    """``--prompts-dir`` pointed at the wrong regeneration is the one failure that is
    invisible on the page, so it is refused rather than rendered."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        tmp = Path(self.tmp.name)
        self.run_dir = tmp / "run"
        self.regen_dir = tmp / "regen"
        for d in (self.run_dir, self.regen_dir):
            (d / "prompts_log").mkdir(parents=True)
        self.snapshot = {
            "model": "mistral_7b_instruct",
            "condition": "unguided",
            "prompt_sha256": "e9fef279",
            "retrieval": {"input_sha256": "3f95c7dc"},
        }
        self._write(self.run_dir, self.snapshot)
        self._write(self.regen_dir, self.snapshot)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _write(self, d: Path, snapshot: dict[str, Any]) -> None:
        (d / "config.json").write_text(json.dumps(snapshot), encoding="utf-8")

    def _check(self) -> None:
        RENDER.assert_regeneration_matches_run(
            "armA", self.regen_dir / "prompts_log", self.run_dir
        )

    def test_a_matching_regeneration_passes(self) -> None:
        self._check()  # must not raise

    def test_a_different_condition_is_refused_and_names_the_field(self) -> None:
        self._write(self.regen_dir, {**self.snapshot, "condition": "oracle_guided"})
        with self.assertRaises(SystemExit) as ctx:
            self._check()
        self.assertIn("condition", str(ctx.exception))
        self.assertIn("oracle_guided", str(ctx.exception))

    def test_a_different_retrieval_artifact_is_refused(self) -> None:
        self._write(
            self.regen_dir, {**self.snapshot, "retrieval": {"input_sha256": "deadbeef"}}
        )
        with self.assertRaises(SystemExit) as ctx:
            self._check()
        self.assertIn("retrieval.input_sha256", str(ctx.exception))

    def test_a_different_model_or_prompt_is_refused(self) -> None:
        for field, value in (("model", "qwen2_5_7b_instruct"), ("prompt_sha256", "0000")):
            with self.subTest(field=field):
                self._write(self.regen_dir, {**self.snapshot, field: value})
                with self.assertRaises(SystemExit) as ctx:
                    self._check()
                self.assertIn(field, str(ctx.exception))

    def test_a_missing_snapshot_is_refused_not_assumed_good(self) -> None:
        (self.regen_dir / "config.json").unlink()
        with self.assertRaises(SystemExit) as ctx:
            self._check()
        self.assertIn("could not be established", str(ctx.exception))

    def test_the_provenance_phrase_is_words_not_a_flag_name(self) -> None:
        regen = RENDER.Arm(
            "armA", Path("p.json"), {}, {},
            prompts_dir=self.regen_dir / "prompts_log",
            prompts_dir_source=RENDER.PROMPTS_REGENERATED,
            prompts_verified=RENDER.PROMPTS_VERIFIED,
        )
        self.assertIn("regenerated (CPU dry run, not written by this run)", regen.prompts_provenance())
        self.assertIn("verified against the run's config snapshot", regen.prompts_provenance())
        self.assertNotIn("--prompts-dir", regen.prompts_provenance())

        unverified = RENDER.Arm(
            "armB", Path("p.json"), {}, {},
            prompts_dir=self.regen_dir / "prompts_log",
            prompts_dir_source=RENDER.PROMPTS_REGENERATED,
            prompts_verified=RENDER.PROMPTS_UNVERIFIED,
        )
        self.assertIn("unverified against a run", unverified.prompts_provenance())

        from_run = RENDER.Arm(
            "armC", Path("p.json"), {}, {},
            prompts_dir=self.run_dir / "prompts_log",
            prompts_dir_source=RENDER.PROMPTS_FROM_RUN,
        )
        self.assertEqual(from_run.prompts_provenance(), "written by this run")


class TestADuplicateAnchorIdDoesNotDiscardTheRun(unittest.TestCase):
    """The mismatch refusal is deferred until after the artifacts are written.

    A duplicated anchor id means the anchor *naming* drifted. It says nothing about the
    generations, and on a real run those are hours of GPU time. So the run finishes, writes
    everything including the two counts, and only then exits non-zero.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls.tmp.name)

        # One real anchor id, duplicated across two retrieval rows. Real, because the id
        # list the runner reads is the committed one and there is no CLI override for it.
        payload = json.loads(ANCHOR_IDS_FILE.read_text(encoding="utf-8"))
        cls.anchor_id = payload["ids_by_hop"]["2"][0]
        template = json.loads(FIXTURE_RETRIEVAL.read_text(encoding="utf-8").splitlines()[0])

        cls.retrieval = tmp / "duplicate_anchor.jsonl"
        rows = []
        for index in range(2):
            row = dict(template)
            row["query_id"] = cls.anchor_id
            row["query_index"] = index
            rows.append(json.dumps(row, ensure_ascii=False))
        cls.retrieval.write_text("\n".join(rows) + "\n", encoding="utf-8")

        cls.out_root = tmp / "out"
        env = dict(os.environ)
        env[PATHS_CONFIG_ENV] = "configs/smoke_paths.json"
        env.setdefault("PYTHONPYCACHEPREFIX", tempfile.gettempdir() + "/qav2-pycache")
        cls.proc = subprocess.run(
            [
                sys.executable, str(RUNNER),
                "--model", "mistral_7b_instruct",
                "--config", "decomposer_musique.json",
                "--condition", "unguided",
                "--dry-run", "--dry-run-limit", "2",
                "--retrieval-input", str(cls.retrieval),
                "--allow-unpinned-eval-set",
                "--output-root", str(cls.out_root),
            ],
            env=env, cwd=str(REPO_ROOT), capture_output=True, text=True,
        )
        run_dirs = sorted(p for p in cls.out_root.iterdir() if p.is_dir()) if cls.out_root.exists() else []
        cls.run_dir = run_dirs[0] if len(run_dirs) == 1 else None

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_the_run_refuses(self) -> None:
        self.assertNotEqual(self.proc.returncode, 0, self.proc.stdout[-2000:])
        combined = self.proc.stdout + self.proc.stderr
        self.assertIn("anchor prompt file(s)", combined)
        self.assertIn("not to be trusted", combined)

    def test_the_generations_survive_the_refusal(self) -> None:
        self.assertIsNotNone(self.run_dir, "no run directory was written at all")
        assert self.run_dir is not None
        results = json.loads((self.run_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(results), 2)
        self.assertTrue((self.run_dir / "config.json").exists())
        self.assertTrue((self.run_dir / "notes.md").exists())

    def test_the_metrics_record_the_mismatch_that_caused_the_refusal(self) -> None:
        assert self.run_dir is not None
        metrics = json.loads((self.run_dir / "metrics.json").read_text(encoding="utf-8"))
        # Two files were attempted for one distinct id: logged 2, expected 1.
        self.assertEqual(metrics["anchor_prompts_logged"], 2)
        self.assertEqual(metrics["anchor_prompts_expected"], 1)
        self.assertEqual(metrics["anchor_ids_total"], 45)

    def test_the_snapshot_content_addresses_the_anchor_list(self) -> None:
        assert self.run_dir is not None
        snapshot = json.loads((self.run_dir / "config.json").read_text(encoding="utf-8"))
        logging_block = snapshot["logging"]
        self.assertEqual(logging_block["anchor_ids_file"], "configs/anchor_set_45.json")
        self.assertEqual(logging_block["anchor_ids_count"], 45)
        self.assertEqual(
            logging_block["anchor_ids_file_sha256"],
            rd.sha256_file(ANCHOR_IDS_FILE),
        )


class TestFileNamingIsSharedNotDuplicated(unittest.TestCase):
    """The writer and the reader must agree on the file name, or the join silently fails."""

    def test_the_writer_name_is_what_the_reader_globs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            name = ANCHOR.anchor_prompt_filename("2hop__aaa", 2)
            (d / name).write_text("x", encoding="utf-8")
            self.assertEqual(ANCHOR.find_anchor_prompt(d, "2hop__aaa"), d / name)

    def test_the_reader_finds_a_file_filed_under_a_different_hop(self) -> None:
        # The hop in the name is display only; the id is the join key.
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / ANCHOR.anchor_prompt_filename("3hop1__ccc", 4)).write_text("x", encoding="utf-8")
            self.assertIsNotNone(ANCHOR.find_anchor_prompt(d, "3hop1__ccc"))

    def test_a_missing_directory_is_not_an_error(self) -> None:
        self.assertIsNone(ANCHOR.find_anchor_prompt(Path("/nonexistent/prompts_log"), "x"))


if __name__ == "__main__":
    unittest.main()
