#!/usr/bin/env python3
"""Checks for issue #12's three MuSiQue conditions in ``configs/decomposer_musique.json``.

The runner is invoked as a subprocess with ``QAV2_PATHS_CONFIG=configs/smoke_paths.json``
and ``--dry-run``, so the real CLI, the real committed config and the real prompt files
are exercised while nothing is loaded onto a GPU and no real data is read: the retrieval
input resolves to the fabricated fixture under ``tests/fixtures/``.

What is asserted, condition by condition: which prompt file is used, whether the gold hop
count reaches the prompt, and what the metrics JSON records. Plus the two refusals that
protect the comparison (an unknown condition, and ``--guided`` against an unguided one),
and that the MetaQA default config still runs unchanged.

Generation itself is not covered here — a dry run generates nothing. The step-line cap
logic is covered by ``tests/test_step_cap.py``.

Run::

    .venv/bin/python -m unittest discover -s tests -v
    .venv/bin/python tests/test_musique_conditions.py
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
RUNNER = REPO_ROOT / "components" / "decomposer" / "run_decomposer.py"
SMOKE_PATHS_CONFIG = REPO_ROOT / "configs" / "smoke_paths.json"
MUSIQUE_CONFIG = REPO_ROOT / "configs" / "decomposer_musique.json"
FIXTURE_QUESTIONS = (
    REPO_ROOT / "tests" / "fixtures" / "data_root" / "musique" / "dev_data"
)
MODEL_DIR = REPO_ROOT / "components" / "decomposer" / "models" / "mistral_7b_instruct"


def _load_runner():
    """Import run_decomposer.py by path, for the pure helpers it exposes."""
    name = "run_decomposer"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _run(args: list[str], out_root: Path) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["QAV2_PATHS_CONFIG"] = str(SMOKE_PATHS_CONFIG)
    return subprocess.run(
        [sys.executable, str(RUNNER), "--output-root", str(out_root), *args],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )


def _artifacts(out_root: Path) -> tuple[dict[str, Any], dict[str, Any], list[Path]]:
    """(metrics, config snapshot, prompt logs) of the single run under ``out_root``."""
    run_dirs = sorted(p for p in out_root.iterdir() if p.is_dir())
    assert len(run_dirs) == 1, f"expected one run dir under {out_root}, got {run_dirs}"
    run_dir = run_dirs[0]
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    snapshot = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    prompts = sorted((run_dir / "prompts_log").glob("*.txt"))
    return metrics, snapshot, prompts


class _ConditionRun:
    """Run one condition in a temp dir and expose its artifacts."""

    def __init__(self, condition: str, extra: list[str] | None = None) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        out_root = Path(self.tmp.name) / "out"
        self.proc = _run(
            [
                "--model", "mistral_7b_instruct",
                "--config", "decomposer_musique.json",
                "--condition", condition,
                "--dry-run", "--dry-run-limit", "2",
                *(extra or []),
            ],
            out_root,
        )
        if self.proc.returncode == 0:
            self.metrics, self.snapshot, self.prompts = _artifacts(out_root)
            self.prompt_text = "\n".join(p.read_text(encoding="utf-8") for p in self.prompts)

    def close(self) -> None:
        self.tmp.cleanup()


class TestConfigLoads(unittest.TestCase):
    def test_musique_variant_is_the_adr_0007_shape(self) -> None:
        cfg = json.loads(MUSIQUE_CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(cfg["hops"], [2, 3, 4])
        self.assertEqual(cfg["questions_template_key"], "musique_eval_questions_template")
        self.assertEqual(cfg["questions_format"], "jsonl")
        self.assertEqual(cfg["seed"], 42)
        self.assertEqual(
            cfg["retrieval"]["input_key"], "musique_eval_retrieval_rerank_top5"
        )
        self.assertEqual(cfg["retrieval"]["mode"], "typed")
        self.assertEqual(cfg["retrieval"]["k"], 5)
        self.assertEqual(cfg["step_cap"]["max_step_lines"], 8)

    def test_metaqa_defaults_are_not_mutated(self) -> None:
        cfg = json.loads((REPO_ROOT / "configs" / "decomposer.json").read_text(encoding="utf-8"))
        self.assertEqual(cfg["hops"], [1, 2, 3])
        self.assertEqual(cfg["questions_template_key"], "metaqa_questions_template")
        self.assertEqual(cfg["questions_format"], "lines")
        self.assertIsNone(cfg["condition"])
        self.assertFalse(cfg["guided"])

    def test_three_conditions_exactly(self) -> None:
        cfg = json.loads(MUSIQUE_CONFIG.read_text(encoding="utf-8"))
        names = sorted(k for k in cfg["conditions"] if k != "_note")
        self.assertEqual(names, ["oracle_guided", "unguided", "unguided_capped"])
        self.assertEqual(
            {k: v for k, v in cfg["conditions"].items() if k != "_note"},
            {
                "unguided": {"guided": False, "step_cap": False},
                "oracle_guided": {"guided": True, "step_cap": False},
                "unguided_capped": {"guided": False, "step_cap": True},
            },
        )


class TestUnguided(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cond = _ConditionRun("unguided")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.cond.close()

    def test_exits_zero(self) -> None:
        self.assertEqual(self.cond.proc.returncode, 0, self.cond.proc.stdout + self.cond.proc.stderr)

    def test_metrics_record_the_condition(self) -> None:
        self.assertEqual(self.cond.metrics["condition"], "unguided")
        self.assertFalse(self.cond.metrics["guided"])
        self.assertFalse(self.cond.metrics["step_cap"]["enabled"])
        self.assertEqual(self.cond.metrics["total_rows"], 2)

    def test_uses_the_unguided_prompt_file(self) -> None:
        self.assertEqual(self.cond.snapshot["prompt_file"], "prompt_unguided.md")

    def test_no_hop_count_reaches_the_prompt(self) -> None:
        self.assertNotIn("Hop count", self.cond.prompt_text)


class TestOracleGuided(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cond = _ConditionRun("oracle_guided")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.cond.close()

    def test_exits_zero(self) -> None:
        self.assertEqual(self.cond.proc.returncode, 0, self.cond.proc.stdout + self.cond.proc.stderr)

    def test_metrics_record_the_condition(self) -> None:
        self.assertEqual(self.cond.metrics["condition"], "oracle_guided")
        self.assertTrue(self.cond.metrics["guided"])
        self.assertFalse(self.cond.metrics["step_cap"]["enabled"])

    def test_uses_the_guided_prompt_file(self) -> None:
        self.assertEqual(self.cond.snapshot["prompt_file"], "prompt.md")

    def test_gold_hop_count_from_the_id_reaches_the_prompt(self) -> None:
        # The fixture retrieval rows are 2hop__f001_x and 4hop2__f002_y, so the hop counts
        # injected are the gold 2 and 4 parsed from the ids.
        self.assertEqual(len(self.cond.prompts), 2)
        first = self.cond.prompts[0].read_text(encoding="utf-8")
        second = self.cond.prompts[1].read_text(encoding="utf-8")
        self.assertIn("Hop count: 2", first)
        self.assertNotIn("Hop count: 4", first)
        self.assertIn("Hop count: 4", second)
        self.assertNotIn("Hop count: 2", second)


class TestUnguidedCapped(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cond = _ConditionRun("unguided_capped")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.cond.close()

    def test_exits_zero(self) -> None:
        self.assertEqual(self.cond.proc.returncode, 0, self.cond.proc.stdout + self.cond.proc.stderr)

    def test_cap_is_on_and_recorded(self) -> None:
        cap = self.cond.metrics["step_cap"]
        self.assertTrue(cap["enabled"])
        self.assertEqual(cap["max_step_lines"], 8)
        self.assertEqual(self.cond.metrics["rows_truncated_by_step_cap"], 0)
        self.assertIn("step_cap_note", self.cond.metrics)  # dry run: cap could not fire

    def test_token_budget_is_the_models_own_when_config_leaves_it_null(self) -> None:
        model_cfg = json.loads((MODEL_DIR / "config.json").read_text(encoding="utf-8"))
        self.assertIsNone(self.cond.metrics["step_cap"]["max_new_tokens"])
        self.assertEqual(
            self.cond.metrics["step_cap"]["effective_max_new_tokens"],
            model_cfg["generation"]["max_new_tokens"],
        )
        self.assertEqual(
            self.cond.snapshot["generation"]["max_new_tokens"],
            model_cfg["generation"]["max_new_tokens"],
        )

    def test_is_unguided_like_the_uncapped_arm(self) -> None:
        self.assertFalse(self.cond.metrics["guided"])
        self.assertEqual(self.cond.snapshot["prompt_file"], "prompt_unguided.md")
        self.assertNotIn("Hop count", self.cond.prompt_text)


class TestConditionsAreHeldIdentical(unittest.TestCase):
    """Everything except guidance and the cap must match across the three arms."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.runs = {name: _ConditionRun(name) for name in
                    ("unguided", "oracle_guided", "unguided_capped")}

    @classmethod
    def tearDownClass(cls) -> None:
        for cond in cls.runs.values():
            cond.close()

    def test_same_model_seed_retrieval_and_decoding(self) -> None:
        keys = ("model", "model_id", "seed", "quantization", "hops", "generation")
        reference = self.runs["unguided"].snapshot
        for name, cond in self.runs.items():
            for key in keys:
                self.assertEqual(cond.snapshot[key], reference[key], f"{name}.{key}")
            self.assertEqual(cond.snapshot["retrieval"], reference["retrieval"], name)

    def test_same_rows_in_the_same_order(self) -> None:
        reference = None
        for name, cond in self.runs.items():
            ids = [p.name.split("_hop")[0] for p in cond.prompts]
            if reference is None:
                reference = ids
            self.assertEqual(ids, reference, name)


class TestRefusals(unittest.TestCase):
    def test_unknown_condition_is_refused(self) -> None:
        cond = _ConditionRun("guided")  # not a condition name
        try:
            self.assertNotEqual(cond.proc.returncode, 0)
            self.assertIn("unknown condition", cond.proc.stdout + cond.proc.stderr)
        finally:
            cond.close()

    def test_guided_flag_against_an_unguided_condition_is_refused(self) -> None:
        cond = _ConditionRun("unguided", extra=["--guided"])
        try:
            self.assertNotEqual(cond.proc.returncode, 0)
            self.assertIn("contradicts condition", cond.proc.stdout + cond.proc.stderr)
        finally:
            cond.close()


class TestMetaQAPathStillWorks(unittest.TestCase):
    def test_default_config_runs_with_no_condition(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_root = Path(tmp) / "out"
            proc = _run(
                [
                    "--model", "qwen2_5_3b",
                    "--config", "decomposer.json",
                    "--dry-run", "--dry-run-limit", "3",
                ],
                out_root,
            )
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            metrics, snapshot, _ = _artifacts(out_root)
            self.assertIsNone(metrics["condition"])
            self.assertFalse(metrics["step_cap"]["enabled"])
            self.assertFalse(metrics["guided"])
            self.assertEqual(snapshot["hops"], [1, 2, 3])
            self.assertEqual(snapshot["questions_format"], "lines")


class TestMissingUnguidedPrompt(unittest.TestCase):
    """A model with no unguided prompt file cannot give a clean unguided arm."""

    def test_warns_and_records_when_the_unguided_prompt_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_root = Path(tmp) / "out"
            proc = _run(
                [
                    "--model", "qwen2_5_3b",  # no prompt_unguided.md
                    "--config", "decomposer_musique.json",
                    "--condition", "unguided",
                    "--dry-run", "--dry-run-limit", "1",
                ],
                out_root,
            )
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertIn("has no unguided_prompt_file", proc.stdout)
            metrics, snapshot, _ = _artifacts(out_root)
            self.assertTrue(metrics["unguided_prompt_missing"])
            self.assertIn("unguided_prompt_missing_note", metrics)
            self.assertEqual(snapshot["prompt_file"], "prompt.md")

    def test_no_warning_for_a_model_that_has_one(self) -> None:
        cond = _ConditionRun("unguided")  # mistral_7b_instruct
        try:
            self.assertFalse(cond.metrics["unguided_prompt_missing"])
            self.assertNotIn("unguided_prompt_missing_note", cond.metrics)
            self.assertNotIn("has no unguided_prompt_file", cond.proc.stdout)
        finally:
            cond.close()


class TestQuestionRowLoading(unittest.TestCase):
    """The MuSiQue question files are JSONL; MetaQA's are one question per line."""

    def test_jsonl_questions_carry_ids(self) -> None:
        runner = _load_runner()
        rows = runner.load_question_rows(
            FIXTURE_QUESTIONS / "musique_ans_v1.0_dev_sample_4_hop_200.jsonl",
            questions_format="jsonl",
            question_field="question",
            id_field="id",
        )
        self.assertEqual([r["query_id"] for r in rows], ["4hop1__f006_c", "4hop2__f002_y"])
        self.assertTrue(all(r["question"] for r in rows))

    def test_hop_count_parses_from_the_jsonl_ids(self) -> None:
        runner = _load_runner()
        self.assertEqual(runner.parse_hop_from_id("4hop2__f002_y"), 4)
        self.assertEqual(runner.parse_hop_from_id("2hop__f001_x"), 2)
        self.assertEqual(runner.parse_hop_from_id("3hop1__f004_a"), 3)

    def test_line_format_has_no_ids(self) -> None:
        runner = _load_runner()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "refined_2hop.txt"
            path.write_text("First question?\n\nSecond question?\n", encoding="utf-8")
            rows = runner.load_question_rows(
                path, questions_format="lines", question_field="", id_field=""
            )
        self.assertEqual(rows, [
            {"query_id": None, "question": "First question?"},
            {"query_id": None, "question": "Second question?"},
        ])

    def test_unknown_format_is_refused(self) -> None:
        runner = _load_runner()
        with self.assertRaises(SystemExit):
            runner.load_question_rows(
                FIXTURE_QUESTIONS / "musique_ans_v1.0_dev_sample_2_hop_200.jsonl",
                questions_format="parquet",
                question_field="question",
                id_field="id",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
