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
                    if p != "difference_of_differences":
                        self.assertTrue(RUN_CONFIG["compare"]["pairs"][p]["in_families"])

    def test_context_pairs_are_in_no_family(self) -> None:
        in_fam = {p for f in RUN_CONFIG["compare"]["families"].values() for p in f["pairs"]}
        for name, p in RUN_CONFIG["compare"]["pairs"].items():
            self.assertEqual(name in in_fam, p["in_families"], name)

    def test_pair_b_is_the_one_primary_and_heads_the_first_family(self) -> None:
        pairs = RUN_CONFIG["compare"]["pairs"]
        self.assertEqual([k for k, v in pairs.items() if v.get("primary")], ["b_finetune_gain_qwen"])
        b = pairs["b_finetune_gain_qwen"]
        self.assertEqual((b["system_a"], b["system_b"]), ("exp036_4bit", "exp027_qwen35_9b_prompting_4bit"))
        self.assertEqual(RUN_CONFIG["compare"]["families"]["F1_primary_overall"]["pairs"], ["b_finetune_gain_qwen"])
        self.assertIn("PRIMARY", RUN_CONFIG["eval"]["4bit"]["role"])

    def test_dod_is_qwen_gain_minus_mistral_gain_and_is_its_own_family(self) -> None:
        dod = RUN_CONFIG["compare"]["difference_of_differences"]
        b = RUN_CONFIG["compare"]["pairs"]["b_finetune_gain_qwen"]
        self.assertEqual(dod["qwen_gain"], {"system_a": b["system_a"], "system_b": b["system_b"]})
        self.assertEqual(dod["mistral_gain"], {"system_a": "exp026_mistral_pool_ft_fp16",
                                               "system_b": "exp015_mistral_prompting_fp16"})
        files = {**{f"exp036_{k}": v for k, v in RUN_CONFIG["compare"]["exp036_per_item"].items()},
                 **RUN_CONFIG["compare"]["comparators"]}
        for g in ("qwen_gain", "mistral_gain"):
            for s in dod[g].values():
                self.assertIn(s, files)
        self.assertEqual(RUN_CONFIG["compare"]["families"]["F3_cross_family_dod"]["pairs"],
                         ["difference_of_differences"])

    def test_the_4bit_eval_cannot_be_skipped(self) -> None:
        import inspect
        self.assertNotIn("fp16_only", inspect.signature(self.cmp.plan).parameters)
        pairs, families = self.cmp.plan(RUN_CONFIG, fp16_failed_gate=True)
        self.assertIn("b_finetune_gain_qwen", pairs)
        self.assertIn("F3_cross_family_dod", families)

    def test_fp16_failed_gate_drops_the_fp16_pairs_and_says_so(self) -> None:
        pairs, families = self.cmp.plan(RUN_CONFIG, fp16_failed_gate=True)
        for p in pairs.values():
            self.assertNotIn("exp036_fp16", (p["system_a"], p["system_b"]))
        self.assertNotIn("F4_secondary_vs_exp026", families)
        self.assertEqual(families["F1_primary_overall"]["pairs"], ["b_finetune_gain_qwen"])

    def test_full_plan_keeps_every_pair(self) -> None:
        pairs, families = self.cmp.plan(RUN_CONFIG, fp16_failed_gate=False)
        self.assertEqual(set(pairs), set(RUN_CONFIG["compare"]["pairs"]))
        self.assertEqual(set(families), set(RUN_CONFIG["compare"]["families"]))

    def test_dod_arithmetic_on_fabricated_items(self) -> None:
        """Mean DoD and its paired t on four tiny fabricated systems, checked independently."""
        import numpy as np
        from scipy import stats

        metrics = ["sari", "break_exact_match"]

        def system(vals):
            return {f"2hop__{i}": {"gold_hop_count": 2, "sari": v[0], "break_exact_match": v[1]}
                    for i, v in enumerate(vals)}

        q4 = system([(0.9, 1), (0.8, 0), (0.7, 1), (0.6, 1)])
        p27 = system([(0.5, 0), (0.6, 0), (0.5, 1), (0.4, 0)])
        m26 = system([(0.7, 1), (0.7, 0), (0.6, 0), (0.6, 1)])
        p15 = system([(0.6, 0), (0.5, 0), (0.6, 0), (0.5, 1)])
        systems = {"exp036_4bit": q4, "exp027_qwen35_9b_prompting_4bit": p27,
                   "exp026_mistral_pool_ft_fp16": m26, "exp015_mistral_prompting_fp16": p15}
        out = self.cmp.difference_of_differences(RUN_CONFIG["compare"]["difference_of_differences"],
                                                 systems, metrics, ("overall",))
        cell = out["strata"]["overall"]
        for m in metrics:
            d = [(q4[i][m] - p27[i][m]) - (m26[i][m] - p15[i][m]) for i in sorted(q4)]
            self.assertAlmostEqual(cell["bootstrap"][m]["difference"], float(np.mean(d)))
            self.assertAlmostEqual(cell["t_test"][m]["p_value"], float(stats.ttest_1samp(d, 0).pvalue))


def _notebook_cells() -> list[str]:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"]]


def _notebook_function(name: str, env: dict) -> dict:
    """exec one top-level def from the notebook (stdlib only) into env."""
    import re
    src = next(c for c in _notebook_cells() if f"def {name}(" in c)
    start = src.index(f"def {name}(")
    m = re.search(r"\n(?=\S)", src[start:])
    exec(src[start:start + m.start()] if m else src[start:], env)
    return env


class TestNotebook(unittest.TestCase):
    def test_notebook_is_valid_json_and_reads_the_plan(self) -> None:
        src = "\n".join(_notebook_cells())
        self.assertIn("experiments/exp-036/run_config.json", src)
        self.assertIn("feature/50-qwen-pool-finetune", src)
        self.assertIn("hf_revision", src)
        self.assertIn('"huggingface_hub": "1.11.0"', src)

    def test_4bit_eval_is_mandatory_and_runs_before_fp16(self) -> None:
        cells = _notebook_cells()
        self.assertNotIn("RUN_4BIT_EVAL", "\n".join(cells))
        i4 = next(i for i, c in enumerate(cells) if c.strip().startswith('run_eval("4bit")'))
        i16 = next(i for i, c in enumerate(cells) if c.strip().startswith('run_eval("fp16")'))
        self.assertLess(i4, i16)

    def test_commit_is_pinned_and_checked_at_train_and_eval(self) -> None:
        src = "\n".join(_notebook_cells())
        self.assertIn('COMMIT = ""', src)
        self.assertIn("checkout --detach {SHA}", src)
        self.assertIn('"commit": tm["git"]["commit"] == SHA', src)
        self.assertIn('"same_commit_as_training": dc["git"]["commit"] == TRAIN_COMMIT == SHA', src)

    def test_data_extraction_is_decided_from_this_vms_disk(self) -> None:
        src = "\n".join(_notebook_cells())
        self.assertIn('DATA_MARKER = f"{DATA}/.exp036_data_ok"', src)
        self.assertNotIn('done("data")', src)

    def test_eval_done_is_keyed_on_metrics_json(self) -> None:
        self.assertIn('glob.glob(f"{out_root(tag)}/*/metrics.json")', "\n".join(_notebook_cells()))


class TestNotebookRoute(unittest.TestCase):
    """The archive router refuses path traversal and maps the documented layout."""

    def setUp(self) -> None:
        import os
        import re
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        (base / "repo" / "runs").mkdir(parents=True)
        self.env = {"os": os, "re": re, "DATA": str(base / "data"), "RUNS": str(base / "repo" / "runs"),
                    "pinned_runs": [RUN_CONFIG["inputs"]["pool"]["path_under_runs"],
                                    RUN_CONFIG["inputs"]["retrieval"]["path_under_runs"]]}
        _notebook_function("route", self.env)
        self.route = self.env["route"]

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_traversal_and_anchored_names_are_refused(self) -> None:
        for name in ("Thesis---QA/Data/../../../../root/.bashrc",
                     "Thesis---QA/MusiQue/Data/../../../../../etc/passwd",
                     "../Thesis---QA/Data/kb.txt",
                     "/root/Thesis---QA/Data/kb.txt",
                     "C:/Thesis---QA/Data/kb.txt",
                     "Thesis---QA\\Data\\kb.txt"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.route(name)

    def test_documented_layout_is_mapped_under_data_and_runs(self) -> None:
        d, r = self.env["DATA"], self.env["RUNS"]
        pool = RUN_CONFIG["inputs"]["pool"]["path_under_runs"]
        cases = {
            "./Thesis---QA/MusiQue/Data/emb_cache/a.npy": f"{d}/cache/emb_cache/a.npy",
            "Thesis---QA/MusiQue/Data/dev_data/x.jsonl": f"{d}/musique/dev_data/x.jsonl",
            "backup/Thesis---QA/Data/kb.txt": f"{d}/metaqa/kb.txt",
            "Thesis---QA/Pool/p.json": f"{d}/metaqa/pool/p.json",
            f"Thesis---QAv2/runs/{pool}": f"{r}/{pool}",
            "Thesis---QAv2/runs/exp-020/x": None,
            "Thesis---QAv2/Slides/a.pdf": None,
            "Thesis---QA/MusiQue/scripts/a.py": None,
        }
        for name, want in cases.items():
            with self.subTest(name=name):
                self.assertEqual(self.route(name), want)


class TestOutputHealth(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = RUN_CONFIG["eval"]["gate"]["output_health"]
        self.env = {"json": json}
        _notebook_function("output_health", self.env)

    def test_degenerate_and_empty_rows_are_counted(self) -> None:
        import tempfile
        rows = [{"decomposition": "a\nb", "decomposition_raw": "a\nb", "hit_max_new_tokens": False},
                {"decomposition": "", "decomposition_raw": "Question: x", "hit_max_new_tokens": False},
                {"decomposition": "!!!!!!!!!!", "decomposition_raw": "!" * 1024, "hit_max_new_tokens": True}]
        with tempfile.TemporaryDirectory() as t:
            (Path(t) / "results.json").write_text(json.dumps(rows), encoding="utf-8")
            health = self.env["output_health"](t, self.gate)
        self.assertEqual(health, {"rows": 3, "empty": 1, "degenerate": 1, "at_max_new_tokens": 1})

    def test_threshold_is_exp027s_recorded_empty_count(self) -> None:
        self.assertEqual(self.gate["max_rows_with_empty_decomposition"], 1)
        self.assertEqual(self.gate["max_rows_degenerate"], 0)
        m = REPO_ROOT / "runs" / "exp-027" / "decomposer" / "20260920_102503" / "metrics.json"
        if not m.exists():
            self.skipTest("exp-027 run directory is not on this machine (runs/ is gitignored)")
        self.assertEqual(json.loads(m.read_text(encoding="utf-8"))["rows_with_empty_decomposition"],
                         self.gate["max_rows_with_empty_decomposition"])


if __name__ == "__main__":
    unittest.main()
