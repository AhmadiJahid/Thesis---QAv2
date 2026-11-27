#!/usr/bin/env python3
"""Checks for the count-first condition (exp-037). No GPU, no model weights.

Covers:

1. ``src/count_first.py::parse_count_first`` - count line present, inline, absent,
   malformed (with and without a readable number), empty output, a count that does not
   match the steps emitted, stray count lines after the first, CRLF and enumerated steps,
   a labelled question ("Hop: Who ...?") kept as a step, non-count lines kept verbatim; and
   ``scripts/reparse_decompositions.py`` on a count-first row.
2. ``summarize_count_first`` - hand-computed accuracy / agreement / status counts.
3. ``insert_instruction_line`` - inserted after the anchor; missing or duplicated anchors
   refused; CRLF kept.
4. ``run_decomposer.format_few_shot_examples`` - the default (``count_first`` off) is the
   pre-change block; on, each exemplar carries its OWN gold step count.
5. ``run_decomposer.resolve_count_first`` refusals (guided, step-line cap, bad block).
6. **Default unchanged, end to end:** seven arms - unguided (Mistral, Qwen3.5-9B,
   Qwen2.5-7B, Qwen2.5-14B) and oracle_guided / unguided_capped / router_guided (Mistral) -
   are re-run with ``--dry-run`` on the fixtures and their rendered prompts
   must hash to the golden values produced by main at 118d777, before count-first existed
   (``tests/fixtures/golden/decomposer_prompts_pre_count_first.json``); results.json rows
   keep the same keys.
7. **The count_first arm, end to end:** its rendered prompt minus the instruction line and
   minus the exemplars' ``[N hops]`` lines is the unguided prompt byte for byte; every
   exemplar's N equals that exemplar's own step count; and the arm is refused with
   ``--no-few-shot``.

Run::

    python tests/test_count_first.py
    python -m unittest discover -s tests
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "components" / "decomposer"))

from count_first import (  # noqa: E402
    format_count_line,
    insert_instruction_line,
    parse_count_first,
    summarize_count_first,
)
from run_config import PATHS_CONFIG_ENV, load_config  # noqa: E402
from step_lines import split_step_lines  # noqa: E402

import run_decomposer as rd  # noqa: E402

GOLDEN = REPO_ROOT / "tests" / "fixtures" / "golden" / "decomposer_prompts_pre_count_first.json"
RETRIEVAL = REPO_ROOT / "tests" / "fixtures" / "retrieval" / "top5_musique_conditions.jsonl"
HOP_PREDICTIONS = REPO_ROOT / "tests" / "fixtures" / "router" / "hop_predictions_musique.jsonl"
ROWS = 9


def _prompt_body(log_text: str) -> str:
    # Same extraction as tests/test_decomposer_conditions.py::_prompt_body.
    after = log_text.split("--- Prompt (", 1)[1]
    after = after.split("---\n", 1)[1]
    return after.split("\n--- Response ---", 1)[0]


def _run_arm(model: str, condition: str, out: Path, *extra: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env[PATHS_CONFIG_ENV] = "configs/smoke_paths.json"
    env.setdefault("PYTHONPYCACHEPREFIX", tempfile.gettempdir() + "/qav2-pycache")
    cmd = [
        sys.executable, str(REPO_ROOT / "components" / "decomposer" / "run_decomposer.py"),
        "--model", model, "--config", "decomposer_musique.json",
        "--condition", condition, "--dry-run", "--dry-run-limit", str(ROWS),
        "--retrieval-input", str(RETRIEVAL), "--allow-unpinned-eval-set",
        "--output-root", str(out), *extra,
    ]
    return subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=REPO_ROOT)


def _artifacts(out: Path) -> dict:
    run_dir = next(out.iterdir())
    return {
        "prompts": {
            p.name: _prompt_body(p.read_text(encoding="utf-8"))
            for p in sorted((run_dir / "prompts_log").glob("prompt_idx*.txt"))
        },
        "results": json.loads((run_dir / "results.json").read_text(encoding="utf-8")),
        "snapshot": json.loads((run_dir / "config.json").read_text(encoding="utf-8")),
        "metrics": json.loads((run_dir / "metrics.json").read_text(encoding="utf-8")),
    }


class TestParser(unittest.TestCase):
    def test_count_line_present(self) -> None:
        p = parse_count_first("[3 hops]\nWho wrote X?\nWhere was [#1] born?\nWhen did [#2] open?")
        self.assertEqual(p.status, "ok")
        self.assertEqual(p.predicted_hops, 3)
        self.assertEqual(p.count_line, "[3 hops]")
        self.assertEqual(
            p.steps_text, "Who wrote X?\nWhere was [#1] born?\nWhen did [#2] open?"
        )
        self.assertEqual(p.extra_count_lines, [])

    def test_count_line_tolerates_case_spacing_singular_and_leading_blank(self) -> None:
        p = parse_count_first("\n  [ 2 HOP ]  \nA?\nB [#1]?")
        self.assertEqual((p.status, p.predicted_hops), ("ok", 2))
        self.assertEqual(p.steps_text, "A?\nB [#1]?")

    def test_count_line_inline_keeps_the_step(self) -> None:
        p = parse_count_first("[2 hops] 1. Who founded X?\n2. Where is [#1]?")
        self.assertEqual((p.status, p.predicted_hops, p.count_line), ("inline", 2, "[2 hops]"))
        self.assertEqual(p.steps_text, "1. Who founded X?\n2. Where is [#1]?")

    def test_count_line_absent_nothing_removed(self) -> None:
        text = "Who founded X?\nWhere is [#1]?"
        p = parse_count_first(text)
        self.assertEqual((p.status, p.predicted_hops, p.count_line), ("absent", None, None))
        self.assertEqual(p.steps_text, text)

    def test_count_line_malformed_with_number(self) -> None:
        for line, n in (("Hops: 3", 3), ("3 hops", 3), ("[three hops]", 3), ("Hop count: 4", 4),
                        ("(2 hops)", 2)):
            with self.subTest(line=line):
                p = parse_count_first(f"{line}\nA?\nB [#1]?")
                self.assertEqual((p.status, p.predicted_hops), ("malformed", n))
                self.assertEqual(p.count_line, line)
                self.assertEqual(p.steps_text, "A?\nB [#1]?")

    def test_count_line_malformed_without_number(self) -> None:
        for line in ("Hops: many", "[N hops]", "[n hops]"):
            with self.subTest(line=line):
                p = parse_count_first(f"{line}\nA?\nB [#1]?")
                self.assertEqual((p.status, p.predicted_hops), ("malformed", None))
                self.assertFalse(p.has_count)
                self.assertEqual(p.steps_text, "A?\nB [#1]?")

    def test_count_mismatches_steps_emitted(self) -> None:
        p = parse_count_first("[4 hops]\nA?\nB [#1]?")
        self.assertEqual((p.status, p.predicted_hops), ("ok", 4))
        self.assertEqual(len(split_step_lines(p.steps_text)), 2)

    def test_empty_output(self) -> None:
        for text in ("", "   \n\n"):
            p = parse_count_first(text)
            self.assertEqual((p.status, p.predicted_hops, p.steps_text), ("empty", None, ""))

    def test_only_count_line_gives_no_steps(self) -> None:
        p = parse_count_first("[3 hops]")
        self.assertEqual((p.status, p.predicted_hops, p.steps_text), ("ok", 3, ""))

    def test_later_count_lines_removed_never_the_prediction(self) -> None:
        p = parse_count_first("[2 hops]\nA?\n[3 hops]\nB [#1]?")
        self.assertEqual(p.predicted_hops, 2)
        self.assertEqual(p.extra_count_lines, ["[3 hops]"])
        self.assertEqual(p.steps_text, "A?\nB [#1]?")
        q = parse_count_first("A?\n[2 hops]\nB [#1]?")
        self.assertEqual((q.status, q.predicted_hops), ("absent", None))
        self.assertEqual(q.extra_count_lines, ["[2 hops]"])
        self.assertEqual(q.steps_text, "A?\nB [#1]?")

    def test_crlf_and_enumerators(self) -> None:
        p = parse_count_first("[2 hops]\r\n1. A?\r\n2. B [#1]?\r\n")
        self.assertEqual(p.predicted_hops, 2)
        self.assertEqual(split_step_lines(p.steps_text), ["A?", "B [#1]?"])

    def test_a_question_mentioning_hops_is_a_step(self) -> None:
        p = parse_count_first("How many hops does the rabbit make?\nWho owns [#1]?")
        self.assertEqual(p.status, "absent")
        self.assertEqual(len(split_step_lines(p.steps_text)), 2)

    def test_labelled_question_is_a_step_not_a_count(self) -> None:
        p = parse_count_first("[2 hops]\nHop: Who founded X?\nWhere is [#1]?")
        self.assertEqual((p.status, p.predicted_hops), ("ok", 2))
        self.assertEqual(p.extra_count_lines, [])
        self.assertEqual(p.steps_text, "Hop: Who founded X?\nWhere is [#1]?")
        q = parse_count_first("Hops: Which river crosses the town?\nB?")
        self.assertEqual(q.status, "absent")

    def test_non_count_lines_kept_verbatim(self) -> None:
        p = parse_count_first("[2 hops]\n  1. A?\n\n   2. B [#1]?  \n")
        self.assertEqual(p.steps_text, "1. A?\n\n   2. B [#1]?")
        q = parse_count_first("  A?\n\n B?")
        self.assertEqual(q.steps_text, "A?\n\n B?")

    def test_format_count_line(self) -> None:
        self.assertEqual(format_count_line(3), "[3 hops]")
        with self.assertRaises(ValueError):
            format_count_line(0)


class TestReparseCountFirst(unittest.TestCase):
    def test_reparse_strips_count_line_and_scaffold(self) -> None:
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        import reparse_decompositions as rp

        recorded = {"strip_think": False, "truncate_at": ["Question:"]}
        fixed = {"strip_think": False, "truncate_at": ["Question:", "Task:", "Answer:"]}
        raw = "\n[2 hops]\nA?\nB [#1]?\nTask:\n"
        stored = rp.parse_raw(raw, recorded, None, count_first=True)
        self.assertEqual(stored, "A?\nB [#1]?\nTask:")
        row = {"query_id": "2hop__x", "decomposition_raw": raw, "decomposition": stored,
               "step_lines": 3, "count_first_status": "ok", "count_first_predicted_hops": 2,
               "count_first_line": "[2 hops]", "count_first_extra_count_lines": []}
        new, changed, not_repro, _ = rp.reparse_rows(
            [row], recorded, fixed, None, count_first=True
        )
        self.assertEqual(not_repro, [])
        self.assertEqual(new[0]["decomposition"], "A?\nB [#1]?")
        self.assertEqual(new[0]["step_lines"], 2)
        self.assertEqual(new[0]["count_first_predicted_hops"], 2)
        self.assertEqual(len(changed), 1)
        # Without count_first the stored (count-stripped) decomposition does not reproduce, so
        # the re-parse refuses rather than silently re-adding the count line.
        _, _, not_repro, _ = rp.reparse_rows([row], recorded, fixed, None)
        self.assertEqual(not_repro, ["2hop__x"])


class TestSummary(unittest.TestCase):
    def test_hand_computed(self) -> None:
        rows = [
            # gold 2, predicted 2, 2 steps emitted -> correct, agrees
            {"hop_count": 2, "count_first_status": "ok", "count_first_predicted_hops": 2,
             "decomposition": "A?\nB?", "count_first_extra_count_lines": []},
            # gold 4, predicted 3, 3 steps -> under, agrees
            {"hop_count": 4, "count_first_status": "ok", "count_first_predicted_hops": 3,
             "decomposition": "A?\nB?\nC?", "count_first_extra_count_lines": []},
            # gold 4, predicted 4, 3 steps -> correct, disagrees (fewer steps)
            {"hop_count": 4, "count_first_status": "malformed", "count_first_predicted_hops": 4,
             "decomposition": "A?\nB?\nC?", "count_first_extra_count_lines": ["[4 hops]"]},
            # gold 3, no count -> miss in all-rows, excluded from parsed
            {"hop_count": 3, "count_first_status": "absent", "count_first_predicted_hops": None,
             "decomposition": "A?\nB?\nC?", "count_first_extra_count_lines": []},
        ]
        s = summarize_count_first(rows)
        self.assertEqual(s["rows"], 4)
        self.assertEqual(s["rows_with_count"], 3)
        self.assertEqual(s["rows_without_count"], 1)
        self.assertEqual(
            s["status_counts"], {"ok": 2, "inline": 0, "malformed": 1, "absent": 1, "empty": 0}
        )
        self.assertEqual(s["rows_with_extra_count_lines"], 1)
        o = s["overall"]
        self.assertEqual(o["count_eq_gold"], 2)
        self.assertAlmostEqual(o["count_eq_gold_rate_all_rows"], 2 / 4)
        self.assertAlmostEqual(o["count_eq_gold_rate_parsed_rows"], 2 / 3)
        self.assertEqual((o["count_below_gold"], o["count_above_gold"]), (1, 0))
        self.assertAlmostEqual(o["mean_signed_count_error_parsed_rows"], -1 / 3)
        self.assertEqual(o["count_eq_steps_emitted"], 2)
        self.assertEqual(
            s["steps_emitted_vs_count"], {"equal": 2, "more_steps": 0, "fewer_steps": 1}
        )
        h4 = s["per_gold_hop"]["4"]
        self.assertEqual((h4["rows"], h4["count_eq_gold"], h4["count_below_gold"]), (2, 1, 1))
        self.assertEqual(s["per_gold_hop"]["3"]["count_eq_gold_rate_parsed_rows"], None)
        self.assertEqual(s["confusion_gold_by_predicted"]["3"], {"none": 1})
        self.assertEqual(s["confusion_gold_by_predicted"]["4"], {"3": 1, "4": 1})


class TestInstructionLine(unittest.TestCase):
    TEMPLATE = "Rules:\n- A.\n- One step per line.\n\nExamples:\n"

    def test_inserted_after_anchor(self) -> None:
        out = insert_instruction_line(
            self.TEMPLATE, instruction="- Count first.", insert_after="- One step per line."
        )
        self.assertEqual(out, "Rules:\n- A.\n- One step per line.\n- Count first.\n\nExamples:\n")
        # Removing the inserted line gives the template back, byte for byte.
        self.assertEqual(out.replace("- Count first.\n", "", 1), self.TEMPLATE)

    def test_crlf_kept(self) -> None:
        t = self.TEMPLATE.replace("\n", "\r\n")
        out = insert_instruction_line(t, instruction="- C.", insert_after="- One step per line.")
        self.assertIn("- One step per line.\r\n- C.\r\n", out)

    def test_refusals(self) -> None:
        with self.assertRaises(ValueError):
            insert_instruction_line(self.TEMPLATE, instruction="- C.", insert_after="- Missing.")
        with self.assertRaises(ValueError):
            insert_instruction_line(
                self.TEMPLATE + "- One step per line.\n", instruction="- C.",
                insert_after="- One step per line.",
            )
        with self.assertRaises(ValueError):
            insert_instruction_line(self.TEMPLATE, instruction="a\nb", insert_after="- A.")

    def test_real_unguided_prompts_have_the_anchor_once(self) -> None:
        block = load_config("decomposer_musique.json")["conditions"]["count_first"]["count_first"]
        for model in ("mistral_7b_instruct", "qwen3_5_9b"):
            text = (REPO_ROOT / "components" / "decomposer" / "models" / model
                    / "prompt_unguided.md").read_text(encoding="utf-8")
            out = insert_instruction_line(
                text, instruction=block["instruction"], insert_after=block["insert_after"]
            )
            self.assertEqual(out.count(block["instruction"]), 1, model)


class TestFormatterAndRefusals(unittest.TestCase):
    EXAMPLES = [
        {"pool_id": "p1", "question": "Q1?", "decomposition": "A?\nB [#1]?"},
        {"pool_id": "p2", "question": "Q2?", "decomposition": "A?\nB [#1]?\nC [#2]?"},
    ]

    def _fmt(self, **kw) -> str:
        return rd.format_few_shot_examples(
            self.EXAMPLES, kw.pop("hop", None), exemplar_hop_mode="exemplar_gold",
            ordering="as_is", **kw,
        )

    def test_default_is_unchanged(self) -> None:
        want = "Question: Q1?\nDecomposition:\nA?\nB [#1]?\n\nQuestion: Q2?\nDecomposition:\nA?\nB [#1]?\nC [#2]?"
        self.assertEqual(self._fmt(), want)
        self.assertEqual(self._fmt(count_first=False), want)

    def test_count_first_uses_each_exemplars_own_count(self) -> None:
        self.assertEqual(
            self._fmt(count_first=True),
            "Question: Q1?\nDecomposition:\n[2 hops]\nA?\nB [#1]?\n\n"
            "Question: Q2?\nDecomposition:\n[3 hops]\nA?\nB [#1]?\nC [#2]?",
        )

    def test_count_first_refused_in_guided_formatting(self) -> None:
        with self.assertRaises(SystemExit):
            self._fmt(hop=3, count_first=True)

    def test_resolve_count_first(self) -> None:
        ok = {"instruction": "- C.", "insert_after": "- One step per line."}
        self.assertIsNone(rd.resolve_count_first("u", {}, guided=False, stop_after_step_lines=None))
        self.assertEqual(
            rd.resolve_count_first("c", {"count_first": ok}, guided=False, stop_after_step_lines=None),
            ok,
        )
        for cond, guided, cap in (
            ({"count_first": ok}, True, None),
            ({"count_first": ok}, False, 8),
            ({"count_first": True}, False, None),
            ({"count_first": {"instruction": "- C."}}, False, None),
            ({"count_first": {**ok, "extra": 1}}, False, None),
        ):
            with self.subTest(cond=cond, guided=guided, cap=cap):
                with self.assertRaises(SystemExit):
                    rd.resolve_count_first("c", cond, guided=guided, stop_after_step_lines=cap)


class TestEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = Path(tempfile.mkdtemp(prefix="qav2-count-first-"))

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _arm(self, model: str, condition: str, *extra: str) -> dict:
        out = self.tmp / f"{condition}__{model}__{len(extra)}"
        proc = _run_arm(model, condition, out, *extra)
        self.assertEqual(proc.returncode, 0, proc.stdout[-1500:] + proc.stderr[-1500:])
        return _artifacts(out)

    def test_default_off_prompts_are_byte_identical_to_main(self) -> None:
        golden = json.loads(GOLDEN.read_text(encoding="utf-8"))["arms"]
        for arm_key, want in golden.items():
            condition, model = arm_key.split("__", 1)
            extra = (
                ("--hop-predictions", str(HOP_PREDICTIONS)) if condition == "router_guided" else ()
            )
            with self.subTest(arm=arm_key):
                got = self._arm(model, condition, *extra)
                hashes = {
                    name: hashlib.sha256(body.encode("utf-8")).hexdigest()
                    for name, body in got["prompts"].items()
                }
                self.assertEqual(hashes, want["prompts_sha256"])
                self.assertEqual(sorted(got["results"][0].keys()), want["results_row_keys"])
                self.assertNotIn("count_first", got["snapshot"])
                self.assertNotIn("count_first", got["metrics"])

    def test_count_first_arm_differs_only_by_count(self) -> None:
        block = load_config("decomposer_musique.json")["conditions"]["count_first"]["count_first"]
        count_rx = re.compile(r"^\[\d+ hops\]\n", re.MULTILINE)
        exemplar_rx = re.compile(
            r"Decomposition:\n\[(\d+) hops\]\n(.*?)(?=\n\nQuestion:|\n\nTask:)", re.DOTALL
        )
        for model in ("mistral_7b_instruct", "qwen3_5_9b"):
            with self.subTest(model=model):
                cf = self._arm(model, "count_first")
                ug = self._arm(model, "unguided")
                self.assertEqual(len(cf["prompts"]), ROWS)
                checked = 0
                for name, body in cf["prompts"].items():
                    # The Qwen (chat_template) dry run logs the messages as JSON.
                    msgs = umsgs = None
                    text = body
                    if model == "qwen3_5_9b":
                        msgs = json.loads(body.split("\n--- Raw generation ---", 1)[0])
                        umsgs = json.loads(
                            ug["prompts"][name].split("\n--- Raw generation ---", 1)[0]
                        )
                        text = msgs[0]["content"] + "\n" + msgs[1]["content"]
                    for n, decomp in exemplar_rx.findall(text):
                        checked += 1
                        self.assertEqual(int(n), len(split_step_lines(decomp)), name)
                    if msgs is not None:
                        # The system half is stripped, so the inserted line is its last.
                        sys_text = msgs[0]["content"].replace("\n" + block["instruction"], "", 1)
                        self.assertEqual(sys_text, umsgs[0]["content"])
                        self.assertEqual(count_rx.sub("", msgs[1]["content"]), umsgs[1]["content"])
                    else:
                        stripped = count_rx.sub("", body.replace(block["instruction"] + "\n", "", 1))
                        self.assertEqual(stripped, ug["prompts"][name], name)
                self.assertGreater(checked, 0)
                self.assertEqual(cf["snapshot"]["count_first"]["instruction"], block["instruction"])
                self.assertIsNone(cf["metrics"]["count_first"])  # dry run: unmeasured
                self.assertIn("count_first_status", cf["results"][0])
                for key in ("seed", "generation", "retrieval", "few_shot", "post_process",
                            "evaluation_set", "prompt_sha256", "quantization"):
                    self.assertEqual(cf["snapshot"][key], ug["snapshot"][key], key)

    def test_count_first_refused_without_few_shot(self) -> None:
        out = self.tmp / "no_few_shot"
        proc = _run_arm("mistral_7b_instruct", "count_first", out, "--no-few-shot")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("count_first needs a few-shot block", proc.stdout + proc.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
