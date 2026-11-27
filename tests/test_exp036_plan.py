#!/usr/bin/env python3
"""GPU-free checks for the exp-036 plan: the Qwen3.5-9B replication of exp-026 (Refs #49, #50).

exp-036 claims to replicate exp-026 with ONE change, the base model. These tests hold that
claim against the committed artifacts, on CPU, with no weights, no network and no data:

- every training block of configs/finetune_decomposer.json equals what exp-026's own run
  snapshot recorded (experiments/exp-026/config.json), so no hyperparameter moved;
- the exp-036 commands in experiments/exp-036/run_config.json are exp-026's commands plus the
  model folder (and their own output roots), and the input hashes are the ones exp-026 recorded;
- for the qwen3_5_9b folder (a chat-template model, unlike Mistral), the training prompt
  goes through the same messages and the same apply_chat_template arguments the runner uses
  at inference, so a trained adapter sees its training string at evaluation;
- the comparison plan's families have the sizes it states, and --fp16-only drops pairs
  instead of substituting systems into a fixed family.

Run::

    python -m pytest tests/test_exp036_plan.py -q
"""
from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "components" / "decomposer"))

import finetune_data as fd  # noqa: E402
import run_decomposer as rd  # noqa: E402
from run_config import load_config, require  # noqa: E402

EXP_DIR = REPO_ROOT / "experiments" / "exp-036"
RUN_CONFIG = json.loads((EXP_DIR / "run_config.json").read_text(encoding="utf-8"))
EXP026 = json.loads((REPO_ROOT / "experiments" / "exp-026" / "config.json").read_text(encoding="utf-8"))
QWEN_DIR = REPO_ROOT / "components" / "decomposer" / "models" / "qwen3_5_9b"
NOTEBOOK = REPO_ROOT / "notebooks" / "exp-036_qwen_pool_ft_colab.ipynb"


def _flag(argv: list[str], name: str) -> str:
    return argv[argv.index(name) + 1]


class TestOnlyTheModelMoves(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = load_config("finetune_decomposer.json")
        self.snap = EXP026["training_run_config"]

    def test_training_blocks_equal_exp026_snapshot(self) -> None:
        for block in ("lora", "training", "loader"):
            with self.subTest(block=block):
                self.assertEqual(self.cfg[block], self.snap[block])

    def test_prompt_settings_equal_exp026_snapshot(self) -> None:
        snap_prompt = {k: v for k, v in self.snap["prompt"].items()
                       if k not in ("prompt_file", "prompt_path", "prompt_sha256", "prompt_style",
                                    "unguided_hop_placeholder")}
        self.assertEqual(self.cfg["prompt"], snap_prompt)

    def test_arm_spec_equals_exp026_snapshot(self) -> None:
        self.assertEqual(fd.resolve_arm(self.cfg, "pool_2000_imbalanced"), self.snap["arm_spec"])

    def test_seed_is_exp026s(self) -> None:
        self.assertEqual(RUN_CONFIG["seed"], self.snap["seed"])
        self.assertEqual(RUN_CONFIG["seed"], EXP026["decomposer_run_config"]["seed"])
        self.assertEqual(int(_flag(RUN_CONFIG["train"]["argv"], "--seed")), RUN_CONFIG["seed"])

    def test_train_argv_is_exp026s_plus_the_model(self) -> None:
        argv, old = RUN_CONFIG["train"]["argv"], RUN_CONFIG["train"]["exp026_argv"]
        self.assertEqual(_flag(argv, "--model"), "qwen3_5_9b")
        self.assertNotIn("--model", old)
        strip = lambda a: [x for i, x in enumerate(a)  # noqa: E731
                           if x not in ("--model", "--output-root")
                           and (i == 0 or a[i - 1] not in ("--model", "--output-root"))]
        self.assertEqual(strip(argv), strip(old))
        self.assertEqual(_flag(old, "--quantization"), self.snap["quantization"])
        self.assertEqual(_flag(old, "--arm"), self.snap["arm"])

    def test_primary_eval_argv_matches_exp026s_evaluation(self) -> None:
        argv = RUN_CONFIG["eval"]["fp16"]["argv"]
        dec = EXP026["decomposer_run_config"]
        self.assertEqual(_flag(argv, "--quantization"), dec["quantization"])
        self.assertEqual(_flag(argv, "--condition"), dec["condition"])
        self.assertEqual(int(_flag(argv, "--seed")), dec["seed"])
        self.assertIn("--no-few-shot", argv)
        self.assertTrue(dec["no_few_shot"])
        self.assertEqual(_flag(argv, "--config"), Path(dec["shared_config"]).name)
        self.assertTrue(dec["retrieval"]["input_resolved"].endswith(
            _flag(argv, "--retrieval-input").replace("runs/", "", 1)))

    def test_secondary_eval_differs_from_primary_only_in_precision_and_output(self) -> None:
        a, b = RUN_CONFIG["eval"]["fp16"]["argv"], RUN_CONFIG["eval"]["4bit"]["argv"]
        diff = [(x, y) for x, y in zip(a, b) if x != y]
        self.assertEqual(len(a), len(b))
        self.assertEqual(diff, [("none", "4bit"),
                                ("runs/exp-036/decomposer_fp16", "runs/exp-036/decomposer_4bit")])

    def test_input_hashes_are_the_ones_exp026_recorded(self) -> None:
        self.assertEqual(RUN_CONFIG["inputs"]["pool"]["sha256"], self.snap["training_source_sha256"])
        self.assertEqual(RUN_CONFIG["inputs"]["retrieval"]["sha256"],
                         EXP026["decomposer_run_config"]["retrieval"]["input_sha256"])
        self.assertEqual(RUN_CONFIG["inputs"]["pool"]["hop_counts"], self.snap["training_data"]["hop_counts"])
        self.assertTrue(self.snap["training_source"].endswith(
            RUN_CONFIG["inputs"]["pool"]["path_under_runs"]))


class TestQwenFolder(unittest.TestCase):
    def setUp(self) -> None:
        self.model_cfg = load_config(QWEN_DIR / "config.json")

    def test_folder_is_the_model_the_plan_names(self) -> None:
        self.assertEqual(require(self.model_cfg, "model_id"), RUN_CONFIG["model"]["model_id"])
        self.assertEqual(require(self.model_cfg, "prompt_style"), "chat_template")
        self.assertFalse(require(self.model_cfg, "chat_template.enable_thinking"))

    def test_training_prompt_goes_through_the_runners_chat_rendering(self) -> None:
        """A stub tokenizer records exactly what each side hands apply_chat_template."""

        class StubTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return json.dumps({"messages": messages, "kwargs": kwargs}, sort_keys=True)

        tok = StubTokenizer()
        template = (QWEN_DIR / fd.select_prompt_file(self.model_cfg, guided=False)).read_text(encoding="utf-8")
        placeholder = require(load_config("decomposer.json"), "unguided_hop_placeholder")
        marker = require(self.model_cfg, "chat_template.split_marker")
        thinking = bool(require(self.model_cfg, "chat_template.enable_thinking"))
        question = "Who leads the union that organised the strike?"

        trained, is_placeholder = fd.build_prompt(
            template, prompt_style="chat_template", question=question, hop=None,
            few_shot_examples="", unguided_hop_placeholder=placeholder,
            chat_marker=marker, tokenizer=tok, enable_thinking=thinking)
        # run_decomposer.main's chat branch, with --no-few-shot (empty few-shot string).
        inferred = tok.apply_chat_template(
            rd.build_chat_messages(template, marker=marker, question=question, hop_count=None,
                                   few_shot_examples="", unguided_hop_placeholder=placeholder),
            tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
        self.assertFalse(is_placeholder)
        self.assertEqual(trained, inferred)
        self.assertFalse(json.loads(trained)["kwargs"]["enable_thinking"])


class TestComparePlan(unittest.TestCase):
    def setUp(self) -> None:
        spec = importlib.util.spec_from_file_location("exp036_compare", EXP_DIR / "compare.py")
        self.cmp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.cmp)

    def test_family_sizes_are_pairs_x_strata_x_metrics(self) -> None:
        n_metrics = len(RUN_CONFIG["compare"]["report_card_metrics"])
        self.assertEqual(n_metrics, 7)
        for name, fam in RUN_CONFIG["compare"]["families"].items():
            with self.subTest(family=name):
                self.assertEqual(fam["size"], len(fam["pairs"]) * len(fam["strata"]) * n_metrics)
                for p in fam["pairs"]:
                    self.assertTrue(RUN_CONFIG["compare"]["pairs"][p]["in_families"])

    def test_context_pairs_are_in_no_family(self) -> None:
        in_fam = {p for f in RUN_CONFIG["compare"]["families"].values() for p in f["pairs"]}
        for name, p in RUN_CONFIG["compare"]["pairs"].items():
            self.assertEqual(name in in_fam, p["in_families"], name)

    def test_fp16_only_drops_the_4bit_pairs_and_says_so(self) -> None:
        pairs, families = self.cmp.plan(RUN_CONFIG, fp16_only=True)
        for p in pairs.values():
            self.assertNotIn("exp036_4bit", (p["system_a"], p["system_b"]))
        self.assertEqual(families["F1_overall"]["pairs"], ["a_family_contrast"])
        self.assertEqual(families["F1_overall"]["dropped_pairs_4bit_eval_not_run"], ["b_finetune_gain_qwen"])

    def test_full_plan_keeps_every_pair(self) -> None:
        pairs, families = self.cmp.plan(RUN_CONFIG, fp16_only=False)
        self.assertEqual(set(pairs), set(RUN_CONFIG["compare"]["pairs"]))
        self.assertEqual(families["F1_overall"]["pairs"], RUN_CONFIG["compare"]["families"]["F1_overall"]["pairs"])


class TestNotebook(unittest.TestCase):
    def test_notebook_is_valid_json_and_reads_the_plan(self) -> None:
        nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        src = "\n".join("".join(c["source"]) for c in nb["cells"])
        self.assertIn("experiments/exp-036/run_config.json", src)
        self.assertIn("feature/50-qwen-pool-finetune", src)
        self.assertIn(RUN_CONFIG["model"]["hf_revision"], json.dumps(RUN_CONFIG))
        self.assertIn("hf_revision", src)


if __name__ == "__main__":
    unittest.main()
