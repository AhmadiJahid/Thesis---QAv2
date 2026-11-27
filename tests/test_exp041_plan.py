#!/usr/bin/env python3
"""GPU-free checks for exp-041 (training-seed variance of exp-026's pool-only fine-tune).

- Only the seed moves: the training command is exp-026's plus --seed / output root, the
  shared training config equals exp-026's run snapshot, the evaluation command is exp-026's
  protocol and the prompting command is exp-015's.
- The seed reaches every random source train_lora.py has (global seed before LoRA init, trl seed
  and data_seed), and the arm's training SET does not depend on it.
- gates.py and compare.py behave as the plan says, on synthetic files.

Run::

    python -m pytest tests/test_exp041_plan.py -q
"""
from __future__ import annotations

import importlib.util
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
EXP = REPO / "experiments" / "exp-041"
CFG = json.loads((EXP / "run_config.json").read_text(encoding="utf-8"))
EXP026 = json.loads((REPO / "experiments" / "exp-026" / "config.json").read_text(encoding="utf-8"))
EXP015 = json.loads((REPO / "experiments" / "exp-015" / "config.json").read_text(encoding="utf-8"))
RUN_SH = (EXP / "run.sh").read_text(encoding="utf-8")
TRAIN_SRC = (REPO / "components" / "decomposer" / "train_lora.py").read_text(encoding="utf-8")
NOTEBOOK = REPO / "notebooks" / "exp-041_ft_seed_variance_colab.ipynb"

import finetune_data as fd  # noqa: E402
from run_config import load_config  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


G = _load("exp041_gates", EXP / "gates.py")


def _flag(argv, name):
    return argv[argv.index(name) + 1]


class TestOnlyTheSeedMoves(unittest.TestCase):
    def test_training_blocks_equal_exp026_snapshot(self):
        cfg = load_config("finetune_decomposer.json")
        snap = EXP026["training_run_config"]
        for block in ("lora", "training", "loader"):
            self.assertEqual(cfg[block], snap[block], block)
        self.assertEqual(fd.resolve_arm(cfg, "pool_2000_imbalanced"), snap["arm_spec"])

    def test_train_argv_is_exp026s_plus_seed_and_root(self):
        t = CFG["train"]
        for s in CFG["training_seeds"]:
            argv = [a.replace("{seed}", str(s)) for a in t["argv_template"]]
            self.assertEqual(_flag(argv, "--seed"), str(s))
            self.assertEqual(_flag(argv, "--output-root"), f"runs/exp-041/seed{s}/train")
            strip = lambda a: [x for i, x in enumerate(a) if x not in ("--seed", "--output-root")  # noqa: E731
                               and (i == 0 or a[i - 1] not in ("--seed", "--output-root"))]
            self.assertEqual(strip(argv), strip(t["exp026_argv"]))
        self.assertEqual(_flag(t["exp026_argv"], "--quantization"), EXP026["training_run_config"]["quantization"])
        self.assertEqual(_flag(t["exp026_argv"], "--arm"), EXP026["training_run_config"]["arm"])

    def test_seeds(self):
        self.assertEqual(CFG["training_seeds"], [42, 43, 44])
        self.assertEqual(EXP026["training_run_config"]["seed"], 42)
        self.assertEqual(CFG["decode_seed"], EXP026["decomposer_run_config"]["seed"])

    def test_eval_argv_is_exp026s_protocol(self):
        argv, dec = CFG["eval"]["argv_template"], EXP026["decomposer_run_config"]
        self.assertEqual(_flag(argv, "--quantization"), dec["quantization"])
        self.assertEqual(_flag(argv, "--condition"), dec["condition"])
        self.assertEqual(int(_flag(argv, "--seed")), dec["seed"])
        self.assertIn("--no-few-shot", argv)
        self.assertTrue(dec["no_few_shot"])
        self.assertEqual(dec["retrieval"]["input_sha256"], CFG["inputs"]["retrieval"]["sha256"])
        self.assertTrue(dec["retrieval"]["input_resolved"].endswith(_flag(argv, "--retrieval-input").replace("runs/", "", 1)))

    def test_prompted_argv_is_exp015s(self):
        cmd = EXP015["step_1_decompose"].split()[1:]
        self.assertEqual(cmd, CFG["prompted"]["exp015_argv"])
        self.assertEqual(CFG["prompted"]["argv"], cmd[: cmd.index("--output-root")])

    def test_run_sh_uses_these_commands(self):
        self.assertIn("train_lora.py --arm pool_2000_imbalanced --quantization 4bit --seed \"$s\"", RUN_SH)
        self.assertIn("--quantization none --adapter \"$adapter\" --no-few-shot", RUN_SH)
        dec = " ".join(CFG["prompted"]["argv"][1:]).replace(CFG["inputs"]["retrieval"]["path_under_runs"], "")
        flat = re.sub(r"\s+", " ", RUN_SH.replace("\\\n", " "))
        self.assertIn(dec.split(" --retrieval-input")[0], flat)
        self.assertIn(CFG["inputs"]["pool"]["sha256"], RUN_SH)
        self.assertIn(CFG["inputs"]["retrieval"]["sha256"], RUN_SH)

    def test_seed_reaches_init_and_data_order(self):
        # global seed set before the model is built and LoRA-wrapped; trl gets seed and data_seed
        main_src = TRAIN_SRC[TRAIN_SRC.index("def main()"):]
        seeded = main_src.index("seeded = set_global_seed(seed)")
        self.assertLess(seeded, main_src.index("load_base_model("))
        self.assertLess(seeded, main_src.index("attach_lora("))
        self.assertIn("seed=seed,\n        data_seed=seed,", TRAIN_SRC)

    def test_training_set_does_not_depend_on_seed(self):
        arm = fd.resolve_arm(load_config("finetune_decomposer.json"), "pool_2000_imbalanced")
        self.assertIsNone(arm["max_examples"])
        self.assertIsNone(arm["train_hops"])
        exs = [object() for _ in range(50)]
        self.assertEqual(fd.cap_examples(exs, None, stratify_by_hop=False, seed=43),
                         fd.cap_examples(exs, None, stratify_by_hop=False, seed=44))

    def test_notebook(self):
        nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        src = "\n".join("".join(c["source"]) for c in nb["cells"])
        for step in ("canary_prompted", "train", "eval", "prompted", "compare"):
            self.assertIn(f'"{step}"', src)
        self.assertIn("launch_sha.txt", src)
        self.assertIn("colab_setup_data.py", src)
        for c in nb["cells"]:
            if c["cell_type"] == "code":
                compile("".join(c["source"]), "<cell>", "exec")


def _bash_fn(name: str) -> str:
    start = RUN_SH.index(f"{name}() {{")
    return RUN_SH[start: RUN_SH.index("\n}\n", start) + 3]


def _working_bash():
    """A bash that runs (on Windows, `bash` on PATH can be a WSL stub without a distribution)."""
    for cand in (shutil.which("bash"), "C:/Program Files/Git/bin/bash.exe"):
        if cand and Path(cand).exists():
            try:
                r = subprocess.run([cand, "-c", "echo ok"], capture_output=True, text=True, timeout=30)
            except (OSError, subprocess.TimeoutExpired):
                continue
            if r.returncode == 0 and r.stdout.strip() == "ok":
                return cand
    return None


BASH = _working_bash()


@unittest.skipUnless(BASH, "no working bash")
class TestLaunchGates(unittest.TestCase):
    """Behaviour of run.sh's vram_gate / launch_clean, with a stub nvidia-smi and a temp git repo."""

    def _run(self, script: str, cwd: Path, smi_out: str | None = None) -> subprocess.CompletedProcess:
        if smi_out is not None:
            stub = Path(tempfile.mkdtemp())
            (stub / "nvidia-smi").write_text(f"#!/usr/bin/env bash\necho '{smi_out}'\n", encoding="utf-8", newline="\n")
            (stub / "nvidia-smi").chmod(0o755)
            script = f'export PATH="$(cd "{stub.as_posix()}" && pwd):$PATH"\n' + script
        return subprocess.run([BASH, "-c", script], cwd=cwd, capture_output=True, text=True)

    def test_thresholds_match_run_config(self):
        g = CFG["gpu"]["free_vram_gate_gib"]
        self.assertIn(f"VRAM_TRAIN={g['train']}", RUN_SH)
        self.assertIn(f"VRAM_EVAL={g['eval_fp16']}", RUN_SH)
        lines = RUN_SH.splitlines()
        # every GPU launch in run.sh is preceded by both gates
        for launch in ("train_lora.py --arm", '"${DEC[@]}" "$@" --resume', '"${DEC[@]}" "$@" --output-root',
                       '"${DEC[@]}" "$@" --sample-size'):
            i = next(k for k, l in enumerate(lines) if launch in l)
            ctx = "\n".join(lines[max(0, i - 3): i + 1])
            self.assertIn("launch_clean", ctx, launch)
            self.assertIn("vram_gate", ctx, launch)

    def test_vram_gate(self):
        fns = 'log() { echo "$*"; }\n' + _bash_fn("vram_gate")
        free = self._run(fns + "\nvram_gate 15.5 eval", REPO, "24576, 4096")     # 20 GiB free
        self.assertEqual(free.returncode, 0, free.stderr)
        busy = self._run(fns + "\nvram_gate 15.5 eval", REPO, "24576, 10240")    # 14 GiB free
        self.assertEqual(busy.returncode, 4, busy.stderr)
        self.assertIn("Nothing was started", busy.stderr)
        self.assertEqual(self._run(fns + "\nvram_gate 6 train", REPO, "24576, 10240").returncode, 0)

    def test_launch_clean_counts_untracked_files(self):
        repo = Path(tempfile.mkdtemp())
        for cmd in (["git", "init", "-q"], ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q",
                                             "--allow-empty", "-m", "x"]):
            subprocess.run(cmd, cwd=repo, check=True)
        fn = _bash_fn("launch_clean")
        self.assertEqual(self._run(fn + "\nlaunch_clean", repo).returncode, 0)
        (repo / "stray.json").write_text("{}", encoding="utf-8")
        r = self._run(fn + "\nlaunch_clean", repo)
        self.assertEqual(r.returncode, 2)
        self.assertIn("stray.json", r.stderr)


def _train_metrics(seed=43, commit="c" * 40):
    g = CFG["train"]["gate"]
    return {"training_source_sha256": g["training_source_sha256"],
            "dataset": {"num_examples": 2000, "hop_counts": {"2": 1461, "3": 413, "4": 126}},
            "eval_overlap": {"overlap_count": 0}, "quantization": "4bit", "model_id": g["model_id"],
            "train_result": {"global_step": 250}, "seed": seed,
            "base_model_size": {"ceiling_asserted": True}, "lora_model_size": {"ceiling_asserted": True},
            "git": {"commit": commit, "dirty": False}, "dry_run": False}


class TestGates(unittest.TestCase):
    def test_train_gate(self):
        g = CFG["train"]["gate"]
        self.assertTrue(all(G.train_checks(_train_metrics(), g, seed=43, commit="c" * 40).values()))
        self.assertFalse(G.train_checks(_train_metrics(), g, seed=44, commit="c" * 40)["seed"])
        m = _train_metrics(); m["train_result"]["global_step"] = 120
        self.assertFalse(G.train_checks(m, g, seed=43, commit="c" * 40)["global_step"])
        m = _train_metrics(); m["git"]["dirty"] = True
        self.assertFalse(G.train_checks(m, g, seed=43, commit="c" * 40)["clean_tree"])

    def _decode(self, n, empty=0, role="adapter"):
        rows = [{"decomposition": "" if i < empty else "1. x", "hit_max_new_tokens": False} for i in range(n)]
        dm = {"rows_loaded_total": n, "rows_loaded_per_hop": {"2": n // 3, "3": n // 3, "4": n // 3},
              "model_size": {"ceiling_asserted": True}, "anchor_prompts_logged": 45,
              "no_few_shot": role == "adapter", "adapter_prompt_parity": {"checked": True, "mismatches": []}}
        dc = {"retrieval": {"input_sha256": CFG["eval_gate"]["retrieval_input_sha256"]}, "git": {"commit": "c"},
              "quantization": "none"}
        return dm, dc, rows

    def test_decode_gate(self):
        gate = CFG["eval_gate"]
        ok, _ = G.decode_checks(*self._decode(600, 6), gate, kind="full", role="adapter", commit="c", canary_rows=20)
        self.assertTrue(all(ok.values()), ok)
        bad, _ = G.decode_checks(*self._decode(600, 7), gate, kind="full", role="adapter", commit="c", canary_rows=20)
        self.assertFalse(bad["empty_within_threshold"])
        c, _ = G.decode_checks(*self._decode(20, 0, "prompted"), gate, kind="canary", role="prompted", commit="c", canary_rows=20)
        self.assertTrue(all(c.values()), c)
        c, _ = G.decode_checks(*self._decode(20, 20, "prompted"), gate, kind="canary", role="prompted", commit="c", canary_rows=20)
        self.assertFalse(c["not_all_empty"])


def _ids():
    return [f"{h}hop__{i}" for h in (2, 3, 4) for i in range(200)]


def _per_item(path, seed, shift=0.0):
    rng = random.Random(seed)
    items = [{"item_id": q, "gold_hop_count": int(q[0]),
              "break_exact_match": float(rng.random() < 0.3 + shift), "hop_count_exact_match": float(rng.random() < 0.5 + shift),
              "under_decomposition": float(rng.random() < 0.3 - shift), "over_decomposition": float(rng.random() < 0.1),
              "sari": rng.random() + shift, "ged": rng.random() - shift, "chain_validity": rng.random() + shift}
             for q in _ids()]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"items": items}), encoding="utf-8")


class TestCompare(unittest.TestCase):
    def test_end_to_end_on_synthetic_files(self):
        C = _load("exp041_compare", EXP / "compare.py")
        C.R.ITER = 200   # speed only
        root = Path(tempfile.mkdtemp())
        for i, (key, rel) in enumerate(CFG["compare"]["systems"].items()):
            _per_item(root / Path(rel).relative_to("runs"), i, 0.25 if key.startswith(("ft_", "exp026")) else 0.0)
        res = [{"query_id": q, "decomposition": "a"} for q in _ids()]
        for rel in CFG["compare"]["stored_results"].values():
            p = root / Path(rel).relative_to("runs"); p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(res), encoding="utf-8")
        d = root / "exp-041" / "seed42" / "decomposer" / "x"; d.mkdir(parents=True)
        (d / "results.json").write_text(json.dumps(res), encoding="utf-8")
        out = root / "cm.json"
        C.main(["--runs-root", str(root), "--out", str(out)])
        m = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(sorted(m["families"]), sorted(["F_seed42", "F_seed43", "F_seed44", "F_reference",
                                                       "F2_seed42_vs_stored", "F2_seed43_vs_stored", "F2_seed44_vs_stored"]))
        self.assertTrue(all(f["size"] == 28 for f in m["families"].values()))
        ag = m["agreement_primary_vs_reference"]
        self.assertEqual(ag["counts_out_of"], 28)
        # a strong built-in effect: every seed agrees in sign with the reference on the overall sari cell
        sari = next(c for c in ag["cells"] if c["stratum"] == "overall" and c["metric"] == "sari")
        self.assertTrue(all(v["same_sign"] for v in sari["seeds"].values()))
        g = m["spread_primary"]["strata"]["overall"]["sari"]["gain_vs_comparator"]
        self.assertEqual(len(g["values"]), 3)
        self.assertGreater(g["mean"], 0)
        self.assertTrue(m["hardware_control"]["ft_seed42"]["reproduced"])
        self.assertNotIn("prompted_rerun", m["hardware_control"])


if __name__ == "__main__":
    unittest.main()
