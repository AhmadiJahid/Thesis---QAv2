#!/usr/bin/env python3
"""Checks for ``src/step_cap.py``, the step-line cap of issue #12's capped condition.

No GPU, no model download, no real data: the cap is exercised on hand-written text and
on a **fake generation stream** (successive prefixes of a decoded response, which is what
the transformers stopping criteria sees one token at a time).

The last test is the important one for the experiment's integrity: the cap counts step
lines with the same rule the evaluator scores them with. If the two drifted apart, a
decomposition could be capped at 8 steps and then reported as having 9.

Run::

    .venv/bin/python -m unittest discover -s tests -v
    .venv/bin/python tests/test_step_cap.py
"""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from step_cap import (  # noqa: E402
    StepLineBudget,
    count_completed_step_lines,
    count_step_lines,
    split_step_lines,
    truncate_to_step_lines,
)


def _load_evaluator():
    """Import the evaluator by path (it lives in scripts/, not on sys.path)."""
    name = "musique_decompositions_evaluator"
    if name in sys.modules:
        return sys.modules[name]
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Register before exec: the module defines dataclasses, which look themselves up in
    # sys.modules while the class body is being processed.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


#: Shapes a decomposer actually emits: plain lines, enumerated lines, blank lines,
#: trailing whitespace, [#n] references, and a bare enumerator with no content.
STEP_TEXT_CASES = [
    "",
    "\n",
    "Who founded Tinwell Press?",
    "Who founded Tinwell Press?\nIn which town was [#1] born?",
    "Who founded Tinwell Press?\n\n\nIn which town was [#1] born?\n",
    "1. Who founded Tinwell Press?\n2. In which town was [#1] born?",
    "  1.   Who founded Tinwell Press?  \n  2. In which town was [#1] born?  \n",
    "Step one?\n3.\nStep two?",
    "\n".join(f"Sub-question {i}?" for i in range(1, 15)),
]


class TestSplitAndCount(unittest.TestCase):
    def test_blank_lines_and_enumerators_are_not_steps(self) -> None:
        text = "1. Who founded Tinwell Press?\n\n2. In which town was [#1] born?\n"
        self.assertEqual(
            split_step_lines(text),
            ["Who founded Tinwell Press?", "In which town was [#1] born?"],
        )
        self.assertEqual(count_step_lines(text), 2)

    def test_bare_enumerator_line_is_not_a_step(self) -> None:
        self.assertEqual(count_step_lines("Step one?\n3.\nStep two?"), 2)

    def test_empty_text_has_no_steps(self) -> None:
        self.assertEqual(count_step_lines(""), 0)
        self.assertEqual(count_step_lines("\n\n"), 0)


class TestFakeGenerationStream(unittest.TestCase):
    """Drive the budget over a stream, the way the stopping criteria is driven."""

    def test_budget_fires_on_the_newline_that_completes_line_n(self) -> None:
        lines = [f"Sub-question {i}?" for i in range(1, 15)]  # 14 steps: a runaway
        full = "\n".join(lines) + "\n"
        budget = StepLineBudget(8)

        # Feed one character at a time and record the first prefix that trips the budget.
        tripped_at = None
        for cut in range(len(full) + 1):
            if budget.reached(full[:cut]):
                tripped_at = cut
                break
        self.assertIsNotNone(tripped_at)

        prefix = full[:tripped_at]
        # It trips exactly when the 8th line's newline arrives: 8 completed lines, and
        # nothing of line 9 has been generated yet.
        self.assertEqual(count_completed_step_lines(prefix), 8)
        self.assertEqual(count_step_lines(prefix), 8)
        self.assertTrue(prefix.endswith("Sub-question 8?\n"))
        self.assertNotIn("Sub-question 9", prefix)

    def test_budget_does_not_fire_below_the_cap(self) -> None:
        budget = StepLineBudget(8)
        text = ""
        for i in range(1, 8):
            text += f"Sub-question {i}?\n"
            self.assertFalse(budget.reached(text), f"fired early at {i} lines")
        text += "Sub-question 8?\n"
        self.assertTrue(budget.reached(text))

    def test_trailing_partial_line_does_not_count(self) -> None:
        # 8 complete lines would trip a cap of 8; here the 8th is still being written.
        text = "".join(f"Sub-question {i}?\n" for i in range(1, 8)) + "Sub-question 8"
        self.assertEqual(count_completed_step_lines(text), 7)
        self.assertFalse(StepLineBudget(8).reached(text))

    def test_budget_rejects_non_positive_caps(self) -> None:
        for bad in (0, -1):
            with self.assertRaises(ValueError):
                StepLineBudget(bad)


class TestTruncation(unittest.TestCase):
    def test_runaway_is_cut_to_the_cap(self) -> None:
        text = "\n".join(f"Sub-question {i}?" for i in range(1, 15))
        out, truncated = truncate_to_step_lines(text, 8)
        self.assertTrue(truncated)
        self.assertEqual(count_step_lines(out), 8)
        self.assertEqual(split_step_lines(out)[-1], "Sub-question 8?")

    def test_short_decomposition_is_untouched(self) -> None:
        text = "Who founded Tinwell Press?\nIn which town was [#1] born?"
        out, truncated = truncate_to_step_lines(text, 8)
        self.assertFalse(truncated)
        self.assertEqual(out, text)

    def test_exactly_at_the_cap_is_not_truncated(self) -> None:
        text = "\n".join(f"Sub-question {i}?" for i in range(1, 9))
        out, truncated = truncate_to_step_lines(text, 8)
        self.assertFalse(truncated)
        self.assertEqual(count_step_lines(out), 8)

    def test_kept_lines_are_verbatim_including_enumerators(self) -> None:
        text = "\n".join(f"{i}. Sub-question {i}?" for i in range(1, 11))
        out, truncated = truncate_to_step_lines(text, 3)
        self.assertTrue(truncated)
        self.assertEqual(out, "1. Sub-question 1?\n2. Sub-question 2?\n3. Sub-question 3?")

    def test_non_positive_cap_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            truncate_to_step_lines("a\nb", 0)


class _FakeTokenizer:
    """Decodes one token per character, so a token id sequence is a string."""

    def decode(self, ids, skip_special_tokens: bool = True) -> str:  # noqa: ARG002
        return "".join(chr(int(i)) for i in ids)


class TestStoppingCriteriaGlue(unittest.TestCase):
    """The transformers wrapper, driven with a fake tokenizer instead of a model."""

    def setUp(self) -> None:
        try:
            import torch  # noqa: F401
            from transformers import StoppingCriteriaList  # noqa: F401
        except ImportError as exc:  # pragma: no cover - environment guard
            self.skipTest(f"torch/transformers unavailable: {exc}")

    def test_criteria_reports_done_only_after_the_cap(self) -> None:
        import torch

        from step_cap import build_stopping_criteria

        prompt = "PROMPT>"
        response = "".join(f"Step {i}?\n" for i in range(1, 6))
        budget = StepLineBudget(3)
        criteria = build_stopping_criteria(budget, _FakeTokenizer(), len(prompt))

        first_done = None
        for cut in range(len(response) + 1):
            ids = torch.tensor(
                [[ord(c) for c in prompt + response[:cut]]], dtype=torch.long
            )
            done = criteria(ids, None)
            self.assertEqual(tuple(done.shape), (1,))
            self.assertEqual(done.dtype, torch.bool)
            if bool(done[0]) and first_done is None:
                first_done = cut
        self.assertIsNotNone(first_done)
        # Fires on the newline that completes step 3, not before, and not after step 4.
        self.assertTrue(response[:first_done].endswith("Step 3?\n"))
        self.assertEqual(count_step_lines(response[:first_done]), 3)


class TestAgreesWithEvaluator(unittest.TestCase):
    """The cap must count what the evaluator scores, on every shape above."""

    def test_split_matches_the_evaluator(self) -> None:
        evaluator = _load_evaluator()
        for text in STEP_TEXT_CASES:
            with self.subTest(text=text):
                self.assertEqual(
                    split_step_lines(text),
                    evaluator._split_decomposition_text(text),
                )

    def test_truncated_text_scores_at_the_cap(self) -> None:
        evaluator = _load_evaluator()
        text = "\n".join(f"{i}. Sub-question {i}?" for i in range(1, 15))
        out, _ = truncate_to_step_lines(text, 8)
        self.assertEqual(len(evaluator._decomp_to_steps(out)), 8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
