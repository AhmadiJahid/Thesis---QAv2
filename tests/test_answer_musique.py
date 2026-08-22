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
4. **The step-level failure taxonomy** (``src/step_failures.py``, issue #16) — which flags
   fire for which step, that the categories are counters rather than a partition, and that
   the three categories this backend cannot produce are declared unavailable with a reason
   instead of reported as zero.

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
from model_size import assert_within_ceiling, ceiling_for, load_limits  # noqa: E402
from run_config import PATHS_CONFIG_ENV, load_config, require  # noqa: E402
from step_failures import (  # noqa: E402
    STEP_FAILURE_CATEGORIES,
    STEP_FAILURE_DEFINITIONS,
    UNAVAILABLE_STEP_FAILURE_CATEGORIES,
    classify_step,
    summarize_step_failures,
)
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

    def test_a_malformed_reference_is_counted_not_ignored(self) -> None:
        """PR #32 review, N-1: '[#1' matched neither alternative, so a truncated reference
        was silently neither substituted nor reported. It is now left verbatim and counted
        unresolved even though step 1 has an answer: the intent is legible, the extent of
        the reference is not."""
        text, resolved, unresolved = substitute_step_references(
            "Who leads [#1?", {1: "Harbour Union"}
        )
        self.assertEqual(text, "Who leads [#1?")
        self.assertEqual((resolved, unresolved), ([], [1]))

    def test_a_malformed_reference_is_caught_in_bracketed_only_mode_too(self) -> None:
        _, resolved, unresolved = substitute_step_references(
            "Who leads [#1?", {1: "Harbour Union"}, accept_bare=False
        )
        self.assertEqual((resolved, unresolved), ([], [1]))

    def test_a_well_formed_bracketed_reference_is_not_read_as_malformed(self) -> None:
        """Alternative order matters: '[#1]' must match the bracketed form, not '[#1' + ']'."""
        text, resolved, unresolved = substitute_step_references("Who leads [#1]?", {1: "HU"})
        self.assertEqual((text, resolved, unresolved), ("Who leads HU?", [1], []))

    def test_a_reference_glued_to_a_word_is_not_a_reference(self) -> None:
        """PR #32 review, N-2: '#1st' parsed as a reference to step 1 followed by 'st'.
        The bare form now requires standalone digits, so it is left alone entirely."""
        text, resolved, unresolved = substitute_step_references(
            "Which #1st edition?", {1: "Harbour Union"}
        )
        self.assertEqual((text, resolved, unresolved), ("Which #1st edition?", [], []))

    def test_a_hash_inside_a_word_is_not_a_reference(self) -> None:
        text, resolved, unresolved = substitute_step_references("Model X#1 sold?", {1: "HU"})
        self.assertEqual((text, resolved, unresolved), ("Model X#1 sold?", [], []))

    def test_a_road_number_is_reported_never_rewritten(self) -> None:
        """The honest limit of the grammar (N-2): 'Route #66' still parses as a reference to
        step 66. There is no step 66, so it is left verbatim and counted unresolved - visible
        in the metrics, never a silent edit. Telling a step reference from a road number needs
        meaning, not a regex; every one of the 3,987 references in MuSiQue's 2,417 gold dev
        decompositions is a well-formed bare '#k' (measured 2026-08-20), so real-data
        behaviour is unchanged by this tightening."""
        text, resolved, unresolved = substitute_step_references(
            "What is on Route #66?", {1: "HU"}
        )
        self.assertEqual(text, "What is on Route #66?")
        self.assertEqual((resolved, unresolved), ([], [66]))

    def test_the_real_gold_grammar_still_resolves(self) -> None:
        """The tightened bare form must still match MuSiQue's own gold shapes, which is all
        the oracle ceiling relies on."""
        for step, expected in (
            ("Who leads #1?", "Who leads HU?"),
            ("Which country contains #1", "Which country contains HU"),
            ("What is the capital of #1 in 1970?", "What is the capital of HU in 1970?"),
            ("#1 was founded by whom?", "HU was founded by whom?"),
        ):
            with self.subTest(step=step):
                text, resolved, _ = substitute_step_references(step, {1: "HU"})
                self.assertEqual(text, expected)
                self.assertEqual(resolved, [1])


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
            ans.assert_reader_template(
                "Question: {question}",
                prompt_path=Path("x.md"),
                prompt_style="plain",
                marker="<<<USER>>>",
            )
        self.assertIn("{context}", str(caught.exception))

    def test_context_only_in_the_system_half_is_refused_for_a_chat_reader(self) -> None:
        """The reviewer's repro (PR #32 review, I-1). build_reader_messages fills the USER
        half only, so '{context}' above the marker is passed through as literal text: the
        reader would run closed-book while the run reported the full-paragraph policy. A
        whole-file check passed this; the filled-half check refuses it."""
        template = "Context:\n{context}\n<<<USER>>>\nQuestion: {question}\nAnswer:"
        # The file does contain both placeholders, which is exactly why the old check passed.
        self.assertIn("{context}", template)
        with self.assertRaises(SystemExit) as caught:
            ans.assert_reader_template(
                template,
                prompt_path=Path("x.md"),
                prompt_style="chat_template",
                marker="<<<USER>>>",
            )
        message = str(caught.exception)
        self.assertIn("{context}", message)
        self.assertIn("<<<USER>>>", message)
        # And the literal proof of the hazard: filling that template leaves the user message
        # with no context at all.
        messages = ans.build_reader_messages(
            template, marker="<<<USER>>>", context="CTX", question="Q?"
        )
        self.assertNotIn("CTX", messages[1]["content"])

    def test_the_same_template_is_fine_for_a_plain_reader(self) -> None:
        """Both halves are joined and filled for a plain folder, so placement is harmless
        there - which is why the guard is per-style, not per-file."""
        ans.assert_reader_template(
            "Context:\n{context}\n<<<USER>>>\nQuestion: {question}",
            prompt_path=Path("x.md"),
            prompt_style="plain",
            marker="<<<USER>>>",
        )

    def test_the_shipped_reader_prompt_passes_in_both_styles(self) -> None:
        prompt = REPO_ROOT / "components" / "answerer" / "prompts" / "reader.md"
        text = prompt.read_text(encoding="utf-8")
        for style in ("plain", "chat_template"):
            with self.subTest(style=style):
                ans.assert_reader_template(
                    text, prompt_path=prompt, prompt_style=style, marker="<<<USER>>>"
                )

    def test_a_chat_reader_prompt_without_the_marker_is_refused(self) -> None:
        with self.assertRaises(SystemExit):
            ans.assert_reader_template(
                "Context: {context}\nQuestion: {question}",
                prompt_path=Path("x.md"),
                prompt_style="chat_template",
                marker="<<<USER>>>",
            )

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

    def test_a_chat_folder_missing_enable_thinking_is_caught(self) -> None:
        """PR #32 review, I-2: enable_thinking is read only on the real chat branch, which
        no dry run reaches - the exp-002/003 crash shape in config form."""
        with self.assertRaises(SystemExit):
            ans.assert_generation_preflight(
                dict(self.GENERATION),
                dict(self.ANSWER),
                model_cfg={"_config_path": "probe", "chat_template": {}},
                prompt_style="chat_template",
            )

    def test_a_chat_folder_with_enable_thinking_passes(self) -> None:
        ans.assert_generation_preflight(
            dict(self.GENERATION),
            dict(self.ANSWER),
            model_cfg={"chat_template": {"enable_thinking": False}},
            prompt_style="chat_template",
        )

    def test_a_plain_folder_does_not_need_the_chat_keys(self) -> None:
        ans.assert_generation_preflight(
            dict(self.GENERATION), dict(self.ANSWER), model_cfg={}, prompt_style="plain"
        )

    def test_every_chat_template_model_folder_in_the_registry_passes(self) -> None:
        """Any folder the reader can be swapped to must survive the preflight, or --model
        would be a flag that only fails after weights load."""
        models_dir = REPO_ROOT / "components" / "decomposer" / "models"
        checked = 0
        for folder in sorted(p for p in models_dir.iterdir() if p.is_dir()):
            model_cfg = load_config(folder / "config.json")
            style = require(model_cfg, "prompt_style")
            generation = ans.rd.apply_generation_overrides(
                dict(require(model_cfg, "generation")), {"max_new_tokens": 64}, "test"
            )
            ans.assert_generation_preflight(
                generation, dict(self.ANSWER), model_cfg=model_cfg, prompt_style=style
            )
            checked += 1
        self.assertGreaterEqual(checked, 4, "the model registry was not found")


class TestReaderCeiling(unittest.TestCase):
    """The reader's ceiling is the standing ~8B one, not the decomposer's ADR 0015 raise."""

    def test_the_answerer_has_its_own_ceiling_key(self) -> None:
        limits = load_limits()
        self.assertEqual(ceiling_for("answerer", limits), 8_000_000_000)

    def test_it_is_tighter_than_the_decomposers(self) -> None:
        """PR #32 review, I-3: ADR 0015 raised default_max_params to 1e10 for ONE role, so
        the reader must not inherit that key."""
        limits = load_limits()
        self.assertEqual(ceiling_for("decomposer", limits), 10_000_000_000)
        self.assertLess(ceiling_for("answerer", limits), ceiling_for("decomposer", limits))

    def test_a_nine_billion_reader_is_refused(self) -> None:
        class _Model:
            def num_parameters(self) -> int:
                return 9_000_000_000

        with self.assertRaises(SystemExit) as caught:
            assert_within_ceiling(
                _Model(), component="answerer", model_id="stub/9b", limits=load_limits()
            )
        self.assertIn("REFUSING TO RUN", str(caught.exception))

    def test_the_default_reader_size_would_pass(self) -> None:
        """Mistral-7B-Instruct-v0.3 is ~7.25B; no weights are loaded to check that here, so
        the ceiling is exercised with the count rather than the model."""

        class _Model:
            def num_parameters(self) -> int:
                return 7_248_023_552

        record = assert_within_ceiling(
            _Model(), component="answerer", model_id="stub/7b", limits=load_limits()
        )
        self.assertTrue(record["ceiling_asserted"])
        self.assertEqual(record["parameter_ceiling"], 8_000_000_000)

    def test_load_limits_requires_every_component_key(self) -> None:
        """A limits config missing a key would otherwise crash only after weights load."""
        with tempfile.TemporaryDirectory(dir=REPO_ROOT / "configs") as tmp:
            probe = Path(tmp) / "limits_probe.json"
            probe.write_text(
                json.dumps({"router_max_params": 1, "default_max_params": 2}), encoding="utf-8"
            )
            with self.assertRaises(SystemExit) as caught:
                load_limits(probe)
        self.assertIn("answerer_max_params", str(caught.exception))

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


class TestStepFailureTaxonomy(unittest.TestCase):
    """``src/step_failures.py``: which flags fire, and what is declared unavailable.

    The categories are deliberately not a partition, so these tests check the flag *sets*
    rather than a single winning label.
    """

    def test_a_clean_executed_step_has_no_flags(self) -> None:
        self.assertEqual(
            classify_step(
                answer="Amador County", unresolved_references=[], error=None, executed=True
            ),
            [],
        )

    def test_an_unresolved_reference_is_a_broken_reference(self) -> None:
        self.assertEqual(
            classify_step(
                answer="something", unresolved_references=[1], error=None, executed=True
            ),
            ["broken_reference"],
        )

    def test_an_empty_answer_on_an_executed_step(self) -> None:
        self.assertEqual(
            classify_step(answer="   ", unresolved_references=[], error=None, executed=True),
            ["empty_answer"],
        )

    def test_a_generation_error_is_not_also_reported_as_an_empty_answer(self) -> None:
        """The error is the cause and the empty answer is its consequence; flagging both
        would double-count one failure."""
        self.assertEqual(
            classify_step(
                answer="", unresolved_references=[], error="RuntimeError: boom", executed=True
            ),
            ["generation_error"],
        )

    def test_a_broken_reference_and_an_empty_answer_are_both_reported(self) -> None:
        """Non-exclusive by design: the step was asked with a placeholder in it AND came
        back with nothing, and an error analysis needs both facts."""
        self.assertEqual(
            classify_step(answer="", unresolved_references=[2], error=None, executed=True),
            ["broken_reference", "empty_answer"],
        )

    def test_a_dry_run_step_is_not_judged_on_its_answer(self) -> None:
        """--dry-run substitutes a stub, so 'empty_answer' would say nothing about a reader
        that was never called; the step is flagged not_executed instead."""
        self.assertEqual(
            classify_step(answer="", unresolved_references=[], error=None, executed=False),
            ["not_executed"],
        )

    def test_a_broken_reference_is_still_real_on_a_dry_run(self) -> None:
        """The substitution chain runs either way, so this flag is meaningful with no
        weights loaded — which is what makes the taxonomy smoke-testable."""
        self.assertEqual(
            classify_step(answer="stub", unresolved_references=[1], error=None, executed=False),
            ["broken_reference", "not_executed"],
        )

    def test_the_flag_order_is_the_declared_category_order(self) -> None:
        flags = classify_step(
            answer="", unresolved_references=[1], error="E: x", executed=False
        )
        self.assertEqual(flags, ["broken_reference", "generation_error", "not_executed"])
        self.assertEqual(
            flags, [c for c in STEP_FAILURE_CATEGORIES if c in flags], "order must be stable"
        )

    def test_the_summary_counts_by_category_and_clean_steps(self) -> None:
        """Four steps: one clean, one broken reference, one empty answer, one with both.
        by_category sums to 4 while only 3 steps carry any flag — the counters are not a
        partition, and steps_clean is what says how many were fine."""
        summary = summarize_step_failures(
            [[], ["broken_reference"], ["empty_answer"], ["broken_reference", "empty_answer"]]
        )
        self.assertEqual(summary["steps"], 4)
        self.assertEqual(summary["steps_clean"], 1)
        self.assertEqual(summary["steps_with_any_flag"], 3)
        self.assertEqual(summary["by_category"]["broken_reference"], 2)
        self.assertEqual(summary["by_category"]["empty_answer"], 2)
        self.assertEqual(summary["by_category"]["generation_error"], 0)
        self.assertEqual(summary["by_category"]["not_executed"], 0)

    def test_every_declared_category_is_present_even_at_zero(self) -> None:
        """A missing key would be indistinguishable from a category this code forgot."""
        summary = summarize_step_failures([])
        self.assertEqual(set(summary["by_category"]), set(STEP_FAILURE_CATEGORIES))
        self.assertEqual(summary["steps"], 0)

    def test_an_unknown_category_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            summarize_step_failures([["retrieval_was_empty"]])

    def test_the_unavailable_categories_are_named_with_a_reason(self) -> None:
        """The three categories issue #16 named that this backend cannot produce are on the
        record with why — absent, never silently zero (CLAUDE.md: unmeasured is unmeasured).
        Each is also disjoint from the categories that DO fire, so no reader sees a
        category both counted and declared unavailable."""
        self.assertEqual(
            set(UNAVAILABLE_STEP_FAILURE_CATEGORIES),
            {"empty_retrieval", "unresolvable_entity", "wrong_intermediate_answer"},
        )
        self.assertEqual(
            set(UNAVAILABLE_STEP_FAILURE_CATEGORIES) & set(STEP_FAILURE_CATEGORIES), set()
        )
        for category, reason in UNAVAILABLE_STEP_FAILURE_CATEGORIES.items():
            self.assertTrue(reason.strip(), f"{category} has no reason recorded")

    def test_every_firing_category_is_defined(self) -> None:
        self.assertEqual(set(STEP_FAILURE_DEFINITIONS), set(STEP_FAILURE_CATEGORIES))


class TestStepFailureTaxonomyEndToEnd(unittest.TestCase):
    """The taxonomy through the real CLI, on a fabricated predictions file.

    The committed MuSiQue predictions fixture happens to have no broken reference (all 6 of
    its ``[#k]`` resolve), so the broken-reference path needs its own hand-written input.
    The ids are the fixture tree's, so the rows still join to a MuSiQue item.
    """

    #: Item 1: step 2's reference is malformed (``[#1`` with no closing bracket), which
    #: ``substitute_step_references`` leaves verbatim and reports unresolved; step 3 is a
    #: forward reference to a step that does not exist. Item 2 is clean.
    ROWS = [
        {
            "query_id": "2hop__d001_a",
            "question": "Which union organised the Marlow Bay strike and who leads it?",
            "hop_count": 2,
            "decomposition": (
                "1. Which union organised the Marlow Bay strike?\n"
                "2. Who leads [#1?\n"
                "3. Where does [#9] live?"
            ),
        },
        {
            "query_id": "3hop1__d002_b",
            "question": "What is the capital of the country the River Anwen rises in?",
            "hop_count": 3,
            "decomposition": (
                "1. Where does the River Anwen rise?\n2. Which country contains [#1]?"
            ),
        },
    ]

    def test_broken_references_are_counted_and_located(self) -> None:
        """5 sub-questions in total. Item 1: step 1 clean, step 2 broken (malformed [#1),
        step 3 broken (no step 9). Item 2: both steps clean. So broken_reference = 2,
        all 5 steps are not_executed (dry run), steps_clean = 0 because not_executed is
        itself a flag, and the final-step block sees 2 steps (one per item) of which item
        1's step 3 is broken."""
        env = os.environ.copy()
        env[PATHS_CONFIG_ENV] = str(SMOKE_PATHS_CONFIG)
        with tempfile.TemporaryDirectory() as tmp:
            predictions = Path(tmp) / "broken_reference_predictions.json"
            predictions.write_text(json.dumps(self.ROWS, indent=2), encoding="utf-8")
            out = Path(tmp) / "out"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ANSWERER),
                    "--predictions",
                    str(predictions),
                    "--dry-run",
                    "--dry-run-limit",
                    "9",
                    "--allow-unpinned-eval-set",
                    "--output-root",
                    str(out),
                ],
                cwd=str(REPO_ROOT),
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            metrics_paths = sorted(out.glob("*/answer_metrics.json"))
            self.assertEqual(len(metrics_paths), 1)
            metrics = json.loads(metrics_paths[0].read_text(encoding="utf-8"))
            per_item = json.loads(
                (metrics_paths[0].parent / "answer_per_item.json").read_text(encoding="utf-8")
            )

        taxonomy = metrics["step_failure_taxonomy"]
        self.assertEqual(taxonomy["all_steps"]["steps"], 5)
        self.assertEqual(taxonomy["all_steps"]["by_category"]["broken_reference"], 2)
        self.assertEqual(taxonomy["all_steps"]["by_category"]["not_executed"], 5)
        self.assertEqual(taxonomy["all_steps"]["by_category"]["empty_answer"], 0)
        self.assertEqual(taxonomy["all_steps"]["by_category"]["generation_error"], 0)
        self.assertEqual(taxonomy["all_steps"]["steps_clean"], 0)
        self.assertEqual(taxonomy["final_step_only"]["steps"], 2)
        self.assertEqual(taxonomy["final_step_only"]["by_category"]["broken_reference"], 1)
        self.assertEqual(taxonomy["items_with_any_step_flag"], 2)
        self.assertEqual(taxonomy["items_with_no_steps_so_no_flags"], 0)
        self.assertEqual(metrics["counts"]["references_unresolved"], 2)

        # The flags are per step in the per-item file, so an error analysis can point at the
        # sub-question rather than at a count.
        first = per_item["items"][0]
        self.assertEqual([s["failure_flags"] for s in first["steps"]][0], ["not_executed"])
        self.assertEqual(
            [s["failure_flags"] for s in first["steps"]][1],
            ["broken_reference", "not_executed"],
        )
        # The malformed reference was never substituted, and never edited away either.
        self.assertIn("[#1?", first["steps"][1]["sub_question"])

    def test_the_committed_fixture_has_no_broken_reference(self) -> None:
        """Pins the premise of the class above: if a future fixture edit introduces a broken
        reference, this test says so instead of the hand-written rows quietly becoming
        redundant."""
        env = os.environ.copy()
        env[PATHS_CONFIG_ENV] = str(SMOKE_PATHS_CONFIG)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ANSWERER),
                    "--predictions",
                    str(PREDICTIONS_FIXTURE),
                    "--dry-run",
                    "--dry-run-limit",
                    "9",
                    "--allow-unpinned-eval-set",
                    "--output-root",
                    str(out),
                ],
                cwd=str(REPO_ROOT),
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            metrics = json.loads(
                sorted(out.glob("*/answer_metrics.json"))[0].read_text(encoding="utf-8")
            )
        taxonomy = metrics["step_failure_taxonomy"]
        self.assertEqual(taxonomy["all_steps"]["steps"], 11)
        self.assertEqual(taxonomy["all_steps"]["by_category"]["broken_reference"], 0)
        self.assertEqual(taxonomy["all_steps"]["by_category"]["not_executed"], 11)
        self.assertEqual(
            set(taxonomy["not_available"]), set(UNAVAILABLE_STEP_FAILURE_CATEGORIES)
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
