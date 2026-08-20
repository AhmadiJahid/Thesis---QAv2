#!/usr/bin/env python3
"""Hand-computed checks for the MuSiQue answering backend (issue #16).

Three things are pinned here, all without a model and without real data:

1. **``[#k]`` substitution** (``src/step_lines.py::substitute_step_references``) — what makes
   a decomposition executable. Both reference grammars, unresolved references, and the
   single-pass rule that stops an inserted answer from being rescanned.
2. **Answer EM / F1** (``src/answer_metrics.py``) — MuSiQue's official metrics. Every
   expected number is computed by hand in the test's docstring, including the alias
   (max-over-gold-answers) path and SQuAD's empty-side edge case. These are golden values:
   one moving means a normalization or a matching rule changed, which must be a deliberate,
   reviewed edit.
3. **Per-hop aggregation** (``components/answerer/run_answerer.py``) — the reporting shape of
   docs/METRICS.md §2, plus the run's step reading agreeing with the decomposition
   evaluator's.

Run::

    .venv/bin/python tests/test_answer_musique.py
    .venv/bin/python -m unittest discover -s tests
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
sys.path.insert(0, str(REPO_ROOT / "components" / "answerer"))

from answer_metrics import (  # noqa: E402
    compute_exact,
    compute_f1,
    gold_answer_set,
    get_tokens,
    normalize_answer,
    score_answer,
)
from run_config import PATHS_CONFIG_ENV, load_config, require  # noqa: E402
from step_lines import substitute_step_references  # noqa: E402

import run_answerer as ans  # noqa: E402

FIXTURES = REPO_ROOT / "tests" / "fixtures"
PREDICTIONS_FIXTURE = FIXTURES / "predictions" / "decomposer_results_musique.json"
EVALUATOR = REPO_ROOT / "scripts" / "musique_decompositions_evaluator.py"
ANSWERER = REPO_ROOT / "components" / "answerer" / "run_answerer.py"
SMOKE_PATHS_CONFIG = REPO_ROOT / "configs" / "smoke_paths.json"

PLACES = 9


class TestStepReferenceSubstitution(unittest.TestCase):
    """What step k's answer does to step k+1's text."""

    def test_bracketed_reference_is_replaced(self) -> None:
        text, resolved, unresolved = substitute_step_references(
            "Who leads [#1]?", {1: "Harbour Union"}
        )
        self.assertEqual(text, "Who leads Harbour Union?")
        self.assertEqual((resolved, unresolved), ([1], []))

    def test_bare_reference_is_replaced_when_accepted(self) -> None:
        """MuSiQue's own gold convention is bare '#k' (measured 2026-08-20: all 2417 rows of
        musique_ans_v1.0_dev_clean.jsonl use it, 3987 references, none bracketed), so the
        oracle-decomposition ceiling cannot execute without this."""
        text, resolved, _ = substitute_step_references("Who leads #1?", {1: "Harbour Union"})
        self.assertEqual(text, "Who leads Harbour Union?")
        self.assertEqual(resolved, [1])

    def test_bare_reference_is_left_alone_when_not_accepted(self) -> None:
        text, resolved, unresolved = substitute_step_references(
            "Who leads #1?", {1: "Harbour Union"}, accept_bare=False
        )
        self.assertEqual(text, "Who leads #1?")
        self.assertEqual((resolved, unresolved), ([], []))

    def test_two_references_in_one_step(self) -> None:
        text, resolved, _ = substitute_step_references(
            "Did [#1] and [#2] meet?", {1: "Ada", 2: "Bo"}
        )
        self.assertEqual(text, "Did Ada and Bo meet?")
        self.assertEqual(resolved, [1, 2])

    def test_a_forward_reference_is_left_verbatim_and_reported(self) -> None:
        """Step 2 cannot use step 3's answer: the placeholder stays and is counted, because
        dropping it would turn the step into a different, answerable-looking question."""
        text, resolved, unresolved = substitute_step_references(
            "Who leads [#3]?", {1: "Harbour Union"}
        )
        self.assertEqual(text, "Who leads [#3]?")
        self.assertEqual((resolved, unresolved), ([], [3]))

    def test_an_empty_answer_counts_as_unresolved(self) -> None:
        """A failed or empty generation for step 1 must not silently blank the reference."""
        text, _, unresolved = substitute_step_references("Who leads [#1]?", {1: "   "})
        self.assertEqual(text, "Who leads [#1]?")
        self.assertEqual(unresolved, [1])

    def test_an_answer_containing_a_reference_is_not_rescanned(self) -> None:
        """Single pass: the answer '#2 Studio' inserted for step 1 stays literal."""
        text, resolved, unresolved = substitute_step_references(
            "Who founded [#1]?", {1: "#2 Studio", 2: "Ada"}
        )
        self.assertEqual(text, "Who founded #2 Studio?")
        self.assertEqual((resolved, unresolved), ([1], []))

    def test_a_step_with_no_reference_is_unchanged(self) -> None:
        text, resolved, unresolved = substitute_step_references("Who won?", {1: "Ada"})
        self.assertEqual((text, resolved, unresolved), ("Who won?", [], []))


class TestAnswerNormalization(unittest.TestCase):
    """SQuAD's normalize_answer, which every EM and F1 below runs through."""

    def test_case_punctuation_articles_and_whitespace(self) -> None:
        self.assertEqual(normalize_answer("  The   Beatles!  "), "beatles")
        self.assertEqual(normalize_answer("A Study in Scarlet"), "study in scarlet")
        self.assertEqual(normalize_answer("St. Louis, Missouri"), "st louis missouri")

    def test_punctuation_goes_before_articles(self) -> None:
        """'the,' loses its comma first, so the article regex then matches it as a word."""
        self.assertEqual(normalize_answer("the, end"), "end")

    def test_an_article_inside_a_word_survives(self) -> None:
        self.assertEqual(normalize_answer("Theatre"), "theatre")

    def test_tokens_of_the_empty_answer(self) -> None:
        self.assertEqual(get_tokens(""), [])
        self.assertEqual(get_tokens("the"), [])


class TestAnswerExactMatch(unittest.TestCase):
    def test_normalization_makes_these_equal(self) -> None:
        self.assertEqual(compute_exact("Dara Volkov", "dara volkov."), 1.0)
        self.assertEqual(compute_exact("The Harbour Union", "Harbour Union"), 1.0)

    def test_a_different_answer_is_zero(self) -> None:
        self.assertEqual(compute_exact("Dara Volkov", "Rae Solberg"), 0.0)

    def test_a_correct_answer_with_extra_words_is_not_an_exact_match(self) -> None:
        self.assertEqual(compute_exact("Dara Volkov", "Dara Volkov, since 1966"), 0.0)


class TestAnswerF1(unittest.TestCase):
    def test_partial_overlap(self) -> None:
        """gold {harbour, union}; pred {harbour, union, of, marlow}: same=2,
        P=2/4=0.5, R=2/2=1.0, F1=2*0.5*1.0/1.5=0.6666666666666666."""
        self.assertAlmostEqual(
            compute_f1("Harbour Union", "the Harbour Union of Marlow"), 2 / 3, places=PLACES
        )

    def test_no_overlap(self) -> None:
        self.assertEqual(compute_f1("Dara Volkov", "Rae Solberg"), 0.0)

    def test_exact_overlap(self) -> None:
        self.assertEqual(compute_f1("Dara Volkov", "dara volkov"), 1.0)

    def test_multiset_intersection_not_a_set_one(self) -> None:
        """gold tokens [new, york, new, york] (4), pred [new, york] (2): the multiset
        intersection is 2, so P=2/2=1.0, R=2/4=0.5, F1=2*1*0.5/1.5=0.6666666666666666.
        A set intersection would wrongly give R=1.0 and F1=1.0."""
        self.assertAlmostEqual(
            compute_f1("New York New York", "New York"), 2 / 3, places=PLACES
        )

    def test_squad_empty_side_rule(self) -> None:
        """With either side empty the score is 1.0 only when both are empty."""
        self.assertEqual(compute_f1("", ""), 1.0)
        self.assertEqual(compute_f1("Dara Volkov", ""), 0.0)
        self.assertEqual(compute_f1("", "Dara Volkov"), 0.0)
        # "the" normalizes to no tokens, so it is an empty side too.
        self.assertEqual(compute_f1("the", ""), 1.0)


class TestGoldAnswerSetAndMaxOverGroundTruths(unittest.TestCase):
    def test_the_answer_comes_first_and_duplicates_go(self) -> None:
        self.assertEqual(
            gold_answer_set("Dara Volkov", ["D. Volkov", "Dara Volkov"]),
            ["Dara Volkov", "D. Volkov"],
        )

    def test_non_strings_are_ignored_not_coerced(self) -> None:
        self.assertEqual(gold_answer_set("1871", [None, 1871, "  ", "MDCCCLXXI"]), ["1871", "MDCCCLXXI"])

    def test_an_alias_can_carry_the_exact_match(self) -> None:
        """EM 1.0 through the alias 'D. Volkov' (normalizes to 'd volkov'), which the
        primary answer 'Dara Volkov' would score 0.0 on; F1 is the max too: against the
        alias, gold {d, volkov} vs pred {d, volkov} = 1.0."""
        em, f1 = score_answer("d volkov", gold_answer_set("Dara Volkov", ["D. Volkov"]))
        self.assertEqual((em, f1), (1.0, 1.0))

    def test_the_maximum_is_taken_per_metric(self) -> None:
        """pred 'Rae A Solberg': against 'Rae Solberg' EM=0, F1: gold {rae, solberg} vs
        pred {rae, a, solberg} -> the article 'a' is stripped by normalization, so pred is
        {rae, solberg} and F1=1.0. Against the alias 'Rae A. Solberg' EM=1.0."""
        golds = gold_answer_set("Rae Solberg", ["Rae A. Solberg"])
        em, f1 = score_answer("Rae A Solberg", golds)
        self.assertEqual((em, f1), (1.0, 1.0))

    def test_an_empty_gold_set_scores_zero(self) -> None:
        self.assertEqual(score_answer("anything", []), (0.0, 0.0))


class TestAggregation(unittest.TestCase):
    """Macro EM / F1 overall and per gold hop depth (docs/METRICS.md §2 reporting style)."""

    ITEMS: list[dict[str, Any]] = [
        {"gold_hop_count": 2, "answer_em": 1.0, "answer_f1": 1.0},
        {"gold_hop_count": 2, "answer_em": 0.0, "answer_f1": 0.5},
        {"gold_hop_count": 3, "answer_em": 1.0, "answer_f1": 1.0},
        {"gold_hop_count": 4, "answer_em": 0.0, "answer_f1": 0.0},
    ]

    def test_overall_is_the_macro_mean(self) -> None:
        """EM = (1+0+1+0)/4 = 0.5; F1 = (1+0.5+1+0)/4 = 0.625."""
        overall = ans.aggregate_answers(self.ITEMS)
        self.assertEqual(overall["num_items"], 4)
        self.assertAlmostEqual(overall["answer_em"], 0.5, places=PLACES)
        self.assertAlmostEqual(overall["answer_f1"], 0.625, places=PLACES)

    def test_per_hop_blocks(self) -> None:
        """hop 2: EM (1+0)/2 = 0.5, F1 (1+0.5)/2 = 0.75; hop 3: 1.0/1.0; hop 4: 0.0/0.0."""
        per_hop = ans.per_gold_hop_metrics(self.ITEMS)
        self.assertEqual(sorted(per_hop), ["2", "3", "4"])
        self.assertEqual(per_hop["2"]["num_items"], 2)
        self.assertAlmostEqual(per_hop["2"]["answer_em"], 0.5, places=PLACES)
        self.assertAlmostEqual(per_hop["2"]["answer_f1"], 0.75, places=PLACES)
        self.assertEqual((per_hop["3"]["answer_em"], per_hop["3"]["answer_f1"]), (1.0, 1.0))
        self.assertEqual((per_hop["4"]["answer_em"], per_hop["4"]["answer_f1"]), (0.0, 0.0))

    def test_unscored_items_give_null_not_zero(self) -> None:
        """A dry run scores nothing; 0.0 would be a false claim, not a missing one."""
        overall = ans.aggregate_answers([{"gold_hop_count": 2, "answer_em": None}])
        self.assertEqual(overall, {"num_items": 0, "answer_em": None, "answer_f1": None})

    def test_an_item_with_no_gold_hop_is_not_filed_under_a_hop(self) -> None:
        per_hop = ans.per_gold_hop_metrics(
            [{"gold_hop_count": None, "answer_em": 1.0, "answer_f1": 1.0}]
        )
        self.assertEqual(per_hop, {})


class TestStepReadingMatchesTheDecompositionEvaluator(unittest.TestCase):
    """The answerer must execute the steps the quality evaluator scores.

    Two halves of one MuSiQue evaluation (ADR 0006) disagreeing about what "the
    decomposition" is would make their numbers describe different objects.
    """

    @staticmethod
    def _evaluator() -> Any:
        name = "musique_decompositions_evaluator"
        spec = importlib.util.spec_from_file_location(name, EVALUATOR)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    def test_the_two_readings_agree(self) -> None:
        evaluator = self._evaluator()
        cases: list[Any] = [
            "1. Which union organised the strike?\n2. Who leads [#1]?",
            "Which union organised the strike?\nWho leads [#1]?",
            ["Which union?", "Who leads [#1]?"],
            [{"question": "Which union?"}, {"question": "Who leads [#1]?"}],
            "",
            None,
            42,
        ]
        for case in cases:
            with self.subTest(case=case):
                self.assertEqual(
                    ans.steps_from_decomposition(case), evaluator._decomp_to_steps(case)
                )


class TestContextAndPromptGuards(unittest.TestCase):
    def test_every_paragraph_is_in_the_context_in_order(self) -> None:
        paragraphs = [
            {"idx": 0, "title": "A", "paragraph_text": "first", "is_supporting": True},
            {"idx": 1, "title": "B", "paragraph_text": "second", "is_supporting": False},
        ]
        context = ans.format_context(
            paragraphs,
            template="[{idx}] {title}: {text}",
            separator="\n",
            idx_field="idx",
            title_field="title",
            text_field="paragraph_text",
        )
        self.assertEqual(context, "[0] A: first\n[1] B: second")

    def test_a_non_supporting_paragraph_is_kept(self) -> None:
        """ADR 0019 decision 2: the standard answerable setting, not a gold-only oracle."""
        paragraphs = [{"idx": 0, "title": "A", "paragraph_text": "d", "is_supporting": False}]
        self.assertIn(
            "d",
            ans.format_context(
                paragraphs,
                template="{idx} {title} {text}",
                separator="\n",
                idx_field="idx",
                title_field="title",
                text_field="paragraph_text",
            ),
        )

    def test_a_reader_prompt_without_context_is_refused(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            ans.assert_reader_template("Question: {question}", prompt_path=Path("x.md"))
        self.assertIn("{context}", str(caught.exception))

    def test_the_shipped_reader_prompt_passes(self) -> None:
        prompt = REPO_ROOT / "components" / "answerer" / "prompts" / "reader.md"
        ans.assert_reader_template(prompt.read_text(encoding="utf-8"), prompt_path=prompt)

    def test_plain_rendering_drops_the_chat_marker(self) -> None:
        rendered = ans.render_reader_template(
            "SYSTEM\n<<<USER>>>\nUSER {context} {question}",
            prompt_style="plain",
            marker="<<<USER>>>",
        )
        self.assertEqual(rendered, "SYSTEM\n\nUSER {context} {question}")

    def test_chat_rendering_keeps_the_halves_apart(self) -> None:
        self.assertIsNone(
            ans.render_reader_template(
                "SYSTEM\n<<<USER>>>\nUSER", prompt_style="chat_template", marker="<<<USER>>>"
            )
        )
        messages = ans.build_reader_messages(
            "SYSTEM\n<<<USER>>>\nContext:\n{context}\nQuestion: {question}",
            marker="<<<USER>>>",
            context="CTX",
            question="Q?",
        )
        self.assertEqual(messages[0], {"role": "system", "content": "SYSTEM"})
        self.assertEqual(messages[1]["content"], "Context:\nCTX\nQuestion: Q?")


class TestAnswerCleanup(unittest.TestCase):
    def test_first_line_and_prefix(self) -> None:
        answer, truncated = ans.clean_answer(
            "Answer: Dara Volkov\nHe has led the union since 1966.",
            take_first_line=True,
            strip_prefixes=["Answer:"],
            max_answer_chars=200,
        )
        self.assertEqual((answer, truncated), ("Dara Volkov", False))

    def test_truncation_is_reported(self) -> None:
        answer, truncated = ans.clean_answer(
            "x" * 30, take_first_line=True, strip_prefixes=[], max_answer_chars=10
        )
        self.assertEqual((answer, truncated), ("x" * 10, True))

    def test_an_empty_generation_stays_empty(self) -> None:
        self.assertEqual(
            ans.clean_answer(
                "\n\n", take_first_line=True, strip_prefixes=[], max_answer_chars=10
            ),
            ("", False),
        )


class TestGenerationPreflight(unittest.TestCase):
    """Every config key the generation call needs is read before the model is loaded.

    Without this, a missing decoding key or cleanup key is a crash only a real run can
    find — the exp-002/exp-003 failure mode of ADR 0016, in config form.
    """

    GENERATION = {"max_new_tokens": 64, "temperature": 0.0, "top_p": 1.0, "do_sample": False}
    ANSWER = {"take_first_line": True, "strip_prefixes": [], "max_answer_chars": 200}

    def test_a_complete_config_passes(self) -> None:
        ans.assert_generation_preflight(dict(self.GENERATION), dict(self.ANSWER))

    def test_the_shipped_config_and_model_folder_pass(self) -> None:
        """configs/answer_musique.json over its default model folder."""
        cfg = load_config("answer_musique.json")
        model_cfg = load_config(
            REPO_ROOT
            / "components"
            / "decomposer"
            / "models"
            / require(cfg, "reader_model")
            / "config.json"
        )
        generation = ans.rd.apply_generation_overrides(
            dict(require(model_cfg, "generation")),
            cfg.get("generation_overrides"),
            "test",
        )
        ans.assert_generation_preflight(generation, dict(require(cfg, "answer_post_process")))

    def test_a_missing_decoding_key_is_caught(self) -> None:
        broken = {k: v for k, v in self.GENERATION.items() if k != "temperature"}
        with self.assertRaises(SystemExit):
            ans.assert_generation_preflight(broken, dict(self.ANSWER))

    def test_a_missing_cleanup_key_is_caught(self) -> None:
        broken = {k: v for k, v in self.ANSWER.items() if k != "max_answer_chars"}
        with self.assertRaises(SystemExit):
            ans.assert_generation_preflight(dict(self.GENERATION), broken)

    def test_the_declared_keys_are_the_keys_generate_answer_reads(self) -> None:
        """The lists are only as good as their agreement with the source: parse it.

        ``generate_answer`` reads its config through ``require(generation, "...")`` and
        ``require(answer_cfg, "...")``; every such literal must be declared above, or the
        preflight would go blind on the key it did not know about.
        """
        import ast

        tree = ast.parse(ANSWERER.read_text(encoding="utf-8"))
        found: dict[str, set[str]] = {"generation": set(), "answer_cfg": set()}
        for node in ast.walk(tree):
            if not (isinstance(node, ast.FunctionDef) and node.name == "generate_answer"):
                continue
            for call in ast.walk(node):
                if (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Name)
                    and call.func.id == "require"
                    and len(call.args) == 2
                    and isinstance(call.args[0], ast.Name)
                    and call.args[0].id in found
                    and isinstance(call.args[1], ast.Constant)
                ):
                    found[call.args[0].id].add(call.args[1].value)
        self.assertTrue(found["generation"], "no require(generation, ...) calls found")
        self.assertTrue(found["answer_cfg"], "no require(answer_cfg, ...) calls found")
        self.assertEqual(found["generation"] - set(ans.GENERATION_KEYS), set())
        self.assertEqual(found["answer_cfg"] - set(ans.ANSWER_POST_PROCESS_KEYS), set())


class TestDryRunEndToEnd(unittest.TestCase):
    """The real CLI over the fabricated fixtures: no weights, hand-checkable counts."""

    def _run(self, *extra: str) -> dict[str, Any]:
        env = os.environ.copy()
        env[PATHS_CONFIG_ENV] = str(SMOKE_PATHS_CONFIG)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ANSWERER),
                    "--dry-run",
                    "--dry-run-limit",
                    "9",
                    "--allow-unpinned-eval-set",
                    "--output-root",
                    str(out),
                    *extra,
                ],
                cwd=str(REPO_ROOT),
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            metrics = sorted(out.glob("*/answer_metrics.json"))
            self.assertEqual(len(metrics), 1, f"expected one run directory, got {metrics}")
            return json.loads(metrics[0].read_text(encoding="utf-8"))

    def test_predictions_mode_counts(self) -> None:
        """The 5 fixture predictions: 4 join to a MuSiQue item and 1 (2hop__d999_z) does not.
        Steps 2+3+4+2 = 11 sub-questions, of which 1+1+3+1 = 6 carry a [#k] reference. EM/F1
        are null because a dry run generates nothing."""
        metrics = self._run("--predictions", str(PREDICTIONS_FIXTURE))
        counts = metrics["counts"]
        self.assertEqual(counts["items_in_source"], 5)
        self.assertEqual(counts["items_missing_musique_item"], 1)
        self.assertEqual(counts["items_evaluated"], 4)
        self.assertEqual(counts["sub_questions_prepared"], 11)
        self.assertEqual(counts["references_resolved"], 6)
        self.assertEqual(counts["references_unresolved"], 0)
        self.assertEqual(counts["failed_generations"], 0)
        self.assertEqual(metrics["gold_hop_distribution"], {"2": 2, "3": 1, "4": 1})
        self.assertIsNone(metrics["answer_em"])
        self.assertIsNone(metrics["answer_f1"])
        self.assertIn("unmeasured", metrics["answer_metrics_note"])
        self.assertFalse(metrics["model_size"]["ceiling_asserted"])

    def test_oracle_mode_restricts_itself_to_the_pinned_ids(self) -> None:
        """The fixture pins 9 ids (3 per hop); the gold file holds 4 rows, 3 of which are
        pinned ids, so 3 items are executed (2+3+4 = 9 sub-questions) and 6 pinned ids are
        reported missing. 2hop__d004_p is in the gold file but not in the pinned files, so
        the oracle ceiling excludes it: a ceiling is a ceiling for one evaluation set."""
        metrics = self._run("--gold-decompositions")
        counts = metrics["counts"]
        self.assertEqual(metrics["mode"], "gold_decompositions")
        self.assertEqual(counts["pinned_ids_requested"], 9)
        self.assertEqual(counts["pinned_ids_missing_from_gold_file"], 6)
        self.assertEqual(counts["items_evaluated"], 3)
        self.assertEqual(counts["sub_questions_prepared"], 9)
        self.assertEqual(metrics["gold_hop_distribution"], {"2": 1, "3": 1, "4": 1})

    def test_both_modes_at_once_is_refused(self) -> None:
        env = os.environ.copy()
        env[PATHS_CONFIG_ENV] = str(SMOKE_PATHS_CONFIG)
        proc = subprocess.run(
            [
                sys.executable,
                str(ANSWERER),
                "--predictions",
                str(PREDICTIONS_FIXTURE),
                "--gold-decompositions",
                "--dry-run",
            ],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("exactly one of", proc.stdout + proc.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
