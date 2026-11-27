#!/usr/bin/env python3
"""GPU-free checks for exp-040 (raw / uniform / typed masking on the v2 pipeline).

- The arms are exp-020's command with only the retrieval input / mode moved, and the build
  commands are exp-014's with only --mode all, --no-cache and the output paths moved.
- run.sh and the notebook name the same commands and steps the config declares.
- check_retrieval.py: validation, the typed-identity count, and the sha256 pin.
- check_health.py: the output-health thresholds.
- compare.py: the typed-comparator rule, family sizes, Holm, the hardware control.

Run::

    python -m pytest tests/test_exp040_plan.py -q
"""
from __future__ import annotations

import importlib.util
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiments" / "exp-040"
CFG = json.loads((EXP / "config.json").read_text(encoding="utf-8"))
NOTEBOOK = REPO / "notebooks" / "exp-040_masking_v2_colab.ipynb"
RUN_SH = (EXP / "run.sh").read_text(encoding="utf-8")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


CR = _load("exp040_check_retrieval", EXP / "check_retrieval.py")
CH = _load("exp040_check_health", EXP / "check_health.py")


def _strip(argv, flags):
    out, skip = [], False
    for x in argv:
        if skip:
            skip = False
            continue
        if x in flags:
            skip = True
            continue
        out.append(x)
    return out


class TestOneFactor(unittest.TestCase):
    def test_shared_argv_is_exp020s_minus_input_and_output(self):
        a = CFG["arms"]
        self.assertEqual(_strip(a["exp020_argv"], {"--retrieval-input", "--output-root"}), a["shared_argv"])

    def test_exp020_argv_matches_its_committed_record(self):
        rec = json.loads((REPO / "experiments" / "exp-020" / "config.json").read_text(encoding="utf-8"))
        cmd = rec["step_1_decompose"].split()
        self.assertEqual(cmd[1:], CFG["arms"]["exp020_argv"])

    def test_build_is_exp014s_with_three_changes(self):
        rec = json.loads((REPO / "experiments" / "exp-014" / "config.json").read_text(encoding="utf-8"))
        old_sim, old_rr = (c.split() for c in rec["deliverable_2_commands"])
        new_sim = CFG["retrieval_build"]["similarity_argv"]
        drop = {"--out", "--run-dir", "--mode"}
        self.assertEqual(_strip([x for x in new_sim if x != "--no-cache"], drop), _strip(old_sim, drop))
        self.assertEqual(new_sim[new_sim.index("--mode") + 1], "all")
        self.assertIn("--no-cache", new_sim)
        new_rr = CFG["retrieval_build"]["rerank_argv"]
        self.assertEqual(_strip(new_rr, {"--input", "--out", "--run-dir"}),
                         _strip(old_rr, {"--input", "--out", "--run-dir"}))

    def test_input_hashes(self):
        self.assertIn(CFG["inputs"]["pool"]["sha256"], RUN_SH)
        self.assertIn(CFG["inputs"]["exp014_typed_artifact"]["sha256"], RUN_SH)

    def test_arm_modes_in_run_sh(self):
        for arm, spec in CFG["arms"].items():
            if not isinstance(spec, dict) or "retrieval_mode" not in spec:
                continue
            if spec["retrieval_mode"]:
                self.assertIn(f"{arm}) echo \"--retrieval-mode {spec['retrieval_mode']}\"", RUN_SH)
        for a in CFG["retrieval_build"]["similarity_argv"][1:]:
            if a.startswith("--") and a not in ("--pool-file", "--out", "--run-dir", "--query-glob"):
                self.assertIn(a, RUN_SH)

    def test_families_have_declared_sizes(self):
        cc = CFG["comparisons"]
        for fam in cc["holm_families"].values():
            self.assertEqual(len(fam["pairs"]) * len(fam["strata"]) * len(cc["report_card_metrics"]), fam["size"])

    def test_notebook_parses_and_calls_every_step(self):
        nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        src = "\n".join("".join(c["source"]) for c in nb["cells"])
        for step in ("retrieval", "typed_rerun", "raw", "uniform", "typed_rebuilt", "compare", "canary"):
            self.assertIn(f'"{step}"', src)
        self.assertIn("launch_sha.txt", src)
        self.assertIn("colab_setup_data.py", src)
        for c in nb["cells"]:
            if c["cell_type"] == "code":
                code = "".join(c["source"])
                code = "\n".join(l for l in code.splitlines() if not l.lstrip().startswith(("!", "%")))
                compile(code, "<cell>", "exec")


def _row(qid, k=5, ids=None):
    ids = ids or {m: [f"{m}{i}" for i in range(k)] for m in ("raw", "typed", "uniform")}
    return {"query_id": qid, "query_question": "q", **{
        f"{m}_top_k": [{"pool_id": p, "pool_question": "x", "pool_few_shot_decomposition_musique": ["a"]}
                       for p in ids[m]] for m in ("raw", "typed", "uniform")}}


def _ids600():
    return [f"{h}hop__{i}" for h in (2, 3, 4) for i in range(200)]


class TestCheckRetrieval(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.e14 = self.tmp / "e14.jsonl"
        self.art = self.tmp / "art.jsonl"
        base = [{"query_id": q, "typed_top_k": _row(q)["typed_top_k"]} for q in _ids600()]
        self.e14.write_text("\n".join(json.dumps(r) for r in base) + "\n", encoding="utf-8")
        self.sha14 = CR.sha256(self.e14)

    def _write(self, rows):
        self.art.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    def test_identity_and_pin(self):
        rows = [_row(q) for q in _ids600()]
        rows[0]["typed_top_k"] = list(reversed(rows[0]["typed_top_k"]))
        self._write(rows)
        rep = CR.check(self.art, self.e14, k=5, exp014_sha256=self.sha14, expected_rows=600, expected_per_hop=200)
        self.assertTrue(rep["ok"], rep["errors"])
        self.assertEqual(rep["typed_identity"]["identical"], 599)
        self.assertEqual(rep["typed_identity"]["identical_as_set_any_order"], 600)
        pin = self.tmp / "PIN"
        self.assertEqual(CR.apply_pin(pin, rep["artifact_sha256"]), "pinned")
        self.assertEqual(CR.apply_pin(pin, rep["artifact_sha256"]), "verified")
        with self.assertRaises(SystemExit):
            CR.apply_pin(pin, "0" * 64)

    def test_refuses_short_lists_and_wrong_ids(self):
        rows = [_row(q) for q in _ids600()]
        rows[5]["raw_top_k"] = rows[5]["raw_top_k"][:3]
        rows[6]["query_id"] = "2hop__other"
        self._write(rows)
        rep = CR.check(self.art, self.e14, k=5, exp014_sha256=self.sha14, expected_rows=600, expected_per_hop=200)
        self.assertFalse(rep["ok"])
        self.assertEqual(len(rep["errors"]), 2)

    def test_refuses_wrong_exp014(self):
        self._write([_row(q) for q in _ids600()])
        rep = CR.check(self.art, self.e14, k=5, exp014_sha256="f" * 64, expected_rows=600, expected_per_hop=200)
        self.assertFalse(rep["ok"])


class TestHealth(unittest.TestCase):
    gate = CFG["output_health_gate"]

    def test_thresholds(self):
        ok = [{"decomposition": "1. a", "hit_max_new_tokens": False}] * 600
        self.assertTrue(CH.health(ok, self.gate, "full")["ok"])
        six_empty = [{"decomposition": ""}] * 6 + ok[6:]
        self.assertTrue(CH.health(six_empty, self.gate, "full")["ok"])
        seven_empty = [{"decomposition": " "}] * 7 + ok[7:]
        self.assertFalse(CH.health(seven_empty, self.gate, "full")["ok"])
        self.assertFalse(CH.health(ok[:599], self.gate, "full")["ok"])
        self.assertFalse(CH.health([{"decomposition": ""}] * 20, self.gate, "canary")["ok"])
        self.assertTrue(CH.health(ok[:20], self.gate, "canary")["ok"])
        capped = [{"decomposition": "x", "hit_max_new_tokens": True}] * 31 + ok[31:]
        self.assertFalse(CH.health(capped, self.gate, "full")["ok"])


def _per_item(path: Path, seed: int):
    rng = random.Random(seed)
    items = []
    for q in _ids600():
        items.append({"item_id": q, "gold_hop_count": int(q[0]), "question": q,
                      "break_exact_match": float(rng.random() < 0.3), "hop_count_exact_match": float(rng.random() < 0.6),
                      "under_decomposition": float(rng.random() < 0.2), "over_decomposition": float(rng.random() < 0.1),
                      "sari": rng.random(), "ged": rng.random(), "chain_validity": rng.random()})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"items": items}), encoding="utf-8")


class TestCompare(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.C = _load("exp040_compare", EXP / "compare.py")
        cls.C.R.ITER = 200   # speed only; the run uses configs/musique_eval.json's count

    def _tree(self, identical: int, with_rebuilt: bool):
        root = Path(tempfile.mkdtemp())
        run = root / "exp-040"
        for i, arm in enumerate(("typed_rerun", "raw", "uniform") + (("typed_rebuilt",) if with_rebuilt else ())):
            _per_item(run / arm / "eval" / "eval_per_item.json", i)
        _per_item(root / "exp-032" / "exp-020" / "eval" / "eval_per_item.json", 9)
        (run / "retrieval").mkdir(parents=True)
        (run / "retrieval" / "check.json").write_text(json.dumps(
            {"typed_identity": {"compared": 600, "identical": identical}}), encoding="utf-8")
        res = [{"query_id": q, "decomposition": "a"} for q in _ids600()]
        d = run / "typed_rerun" / "decomposer" / "20260101_000000"
        d.mkdir(parents=True)
        (d / "results.json").write_text(json.dumps(res), encoding="utf-8")
        stored = root / "exp-032" / "exp-020" / "reparse"
        stored.mkdir(parents=True)
        res[0] = {"query_id": res[0]["query_id"], "decomposition": "b"}
        (stored / "results.json").write_text(json.dumps(res), encoding="utf-8")
        return root

    def test_identical_typed_uses_rerun(self):
        root = self._tree(600, with_rebuilt=False)
        out = root / "cm.json"
        self.C.main(["--runs-root", str(root), "--out", str(out)])
        m = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(m["typed_comparator"]["system"], "typed_rerun")
        self.assertEqual(m["families"]["F1_primary_overall"]["size"], 14)
        self.assertEqual(m["families"]["F2_primary_per_hop"]["size"], 42)
        self.assertEqual(m["hardware_control"]["identical_decompositions"], 599)
        self.assertFalse(m["hardware_control"]["reproduced"])
        self.assertNotIn("typed_rebuilt_vs_typed_rerun", m["pairs"])
        self.assertEqual(m["pairs"]["typed_vs_raw"]["system_a"], "typed_rerun")

    def test_differing_typed_requires_rebuilt(self):
        root = self._tree(598, with_rebuilt=False)
        with self.assertRaises(SystemExit):
            self.C.main(["--runs-root", str(root), "--out", str(root / "cm.json")])
        root = self._tree(598, with_rebuilt=True)
        self.C.main(["--runs-root", str(root), "--out", str(root / "cm.json")])
        m = json.loads((root / "cm.json").read_text(encoding="utf-8"))
        self.assertEqual(m["typed_comparator"]["system"], "typed_rebuilt")
        self.assertEqual(m["pairs"]["typed_vs_uniform"]["system_a"], "typed_rebuilt")
        self.assertIn("typed_rebuilt_vs_typed_rerun", m["pairs"])


if __name__ == "__main__":
    unittest.main()
