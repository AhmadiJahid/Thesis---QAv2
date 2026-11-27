"""One test per MetaQA executor defect fixed in exp-039 (ADR 0045).

Each test names the defect it pins and fails on the pre-fix code. Toy KB, no data root.

Run::

    .venv/bin/python -m unittest tests.test_metaqa_executor_fixes
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from evaluate_decompositions import (  # noqa: E402
    EmptyResultError,
    build_reverse_index,
    compile_decomposition,
    evaluate_decomposition_rate,
    exec_ops,
)
from kg import build_metaqa_kg  # noqa: E402
from run_config import load_config  # noqa: E402

KB = """Glass Harbor|directed_by|Ada Mireles
Glass Harbor|has_genre|Mystery
Winter Marbles|directed_by|Ada Mireles
Winter Marbles|directed_by|Bo Fenwick
Winter Marbles|has_genre|Drama
Paper Lanterns|written_by|Cy Okafor
Paper Lanterns|starred_actors|Dara Volkov
What Ever Happened to June?|directed_by|Elin Marsh
Lemmy|starred_actors|Lemmy
"""


class ExecutorFixesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        kb = Path(cls.tmp.name) / "kb.txt"
        kb.write_text(KB, encoding="utf-8")
        cls.kg = build_metaqa_kg(kb)
        cls.rev = build_reverse_index(cls.kg)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def run_dec(self, dec: str, exclude_topic: bool = True) -> set:
        ids = exec_ops(self.kg, self.rev, compile_decomposition(dec), exclude_topic=exclude_topic)
        return {self.kg.id_to_entity[i] for i in ids}

    # ---- defect 1: topic entity not excluded at h >= 2 ----
    def test_two_hop_co_director_drops_the_topic(self) -> None:
        dec = "1. What movies did Ada Mireles direct?\n2. Who directed [#1]?"
        self.assertEqual(self.run_dec(dec), {"Bo Fenwick"})
        self.assertEqual(self.run_dec(dec, exclude_topic=False), {"Ada Mireles", "Bo Fenwick"})

    def test_three_hop_drops_the_topic_movie_mid_chain(self) -> None:
        dec = ("1. Who directed Glass Harbor?\n2. What other movies were directed by [#1]?\n"
               "3. What genres are [#2]?")
        self.assertEqual(self.run_dec(dec), {"Drama"})
        self.assertEqual(self.run_dec(dec, exclude_topic=False), {"Mystery", "Drama"})

    def test_one_hop_keeps_the_topic(self) -> None:
        self.assertEqual(self.run_dec("1. What movies did Lemmy star in?"), {"Lemmy"})

    def test_the_shipped_config_turns_exclusion_on(self) -> None:
        self.assertIs(load_config("metaqa_kg_eval.json")["exclude_topic_after_first_step"], True)

    # ---- defect 2: "write" missing from the writer rule ----
    def test_write_infers_written_by(self) -> None:
        ops = compile_decomposition("1. What movies did Cy Okafor write?")
        self.assertEqual((ops[0].op, ops[0].relation), ("MOVIES_WITH_VALUE", "written_by"))
        self.assertEqual(self.run_dec("1. What movies did Cy Okafor write?"), {"Paper Lanterns"})

    # ---- defect 3: reverse step on [#k] compiled as a forward projection ----
    def test_reverse_step_on_a_value_placeholder_is_a_reverse_lookup(self) -> None:
        dec = "1. Who directed Winter Marbles?\n2. What movies did [#1] direct?\n3. What genres are [#2]?"
        ops = compile_decomposition(dec)
        self.assertEqual([o.op for o in ops],
                         ["VALUES_OF_MOVIE", "MOVIES_WITH_VALUE", "PROJECT_VALUES_FROM_MOVIES"])
        # [#1] = {Ada, Bo}; [#2] = {Glass Harbor} once the topic Winter Marbles is dropped
        self.assertEqual(self.run_dec(dec), {"Mystery"})

    def test_projection_on_a_movie_placeholder_is_unchanged(self) -> None:
        ops = compile_decomposition("1. What movies did Ada Mireles direct?\n2. What genres are [#1]?")
        self.assertEqual(ops[1].op, "PROJECT_VALUES_FROM_MOVIES")

    # ---- defect 4: KB names ending in "?" ----
    def test_entity_name_ending_in_a_question_mark_resolves(self) -> None:
        self.assertEqual(self.run_dec("1. Who directed What Ever Happened to June?"), {"Elin Marsh"})

    # ---- empty results are an error class, never a success ----
    def test_empty_final_step_raises(self) -> None:
        # Dara Volkov directed nothing: the chain runs but answers nothing.
        with self.assertRaises(EmptyResultError):
            exec_ops(self.kg, self.rev, compile_decomposition("1. What movies did Dara Volkov direct?"))

    def test_empty_result_is_counted_as_its_own_exec_failure(self) -> None:
        rows = [
            {"question": "q1", "hop_count": 1, "decomposition": "1. What movies did Dara Volkov direct?"},
            {"question": "q2", "hop_count": 1, "decomposition": "1. Who directed Glass Harbor?"},
        ]
        preds = Path(self.tmp.name) / "preds.json"
        preds.write_text(json.dumps(rows), encoding="utf-8")
        m = evaluate_decomposition_rate(self.kg, preds, exclude_topic=True)
        self.assertEqual((m.executed_ok, m.exec_fail), (1, 1))
        self.assertEqual(dict(m.exec_fail_reasons), {"empty_result": 1})


if __name__ == "__main__":
    unittest.main()


class AttributeTemplatesTest(unittest.TestCase):
    """Compiler coverage for genre / year / language / tags steps (exp-039 eval set)."""

    CASES = {
        "1. What genre is The Order?": ("VALUES_OF_MOVIE", "has_genre", "The Order"),
        "1. What is the release year of the movie Election?": ("VALUES_OF_MOVIE", "release_year", "Election"),
        "1. When was Straw Dogs released?": ("VALUES_OF_MOVIE", "release_year", "Straw Dogs"),
        "1. What language is Hamlet in?": ("VALUES_OF_MOVIE", "in_language", "Hamlet"),
        "1. Which words describe movie Solaris?": ("VALUES_OF_MOVIE", "has_tags", "Solaris"),
        "1. Describe Winchester '73.": ("VALUES_OF_MOVIE", "has_tags", "Winchester '73"),
        "1. Who is the director of A Star Is Born?": ("VALUES_OF_MOVIE", "directed_by", "A Star Is Born"),
        "1. What movies are about shakespearean?": ("MOVIES_WITH_VALUE", "has_tags", "shakespearean"),
    }

    def test_each_template(self) -> None:
        for dec, (op, rel, name) in self.CASES.items():
            with self.subTest(dec=dec):
                o = compile_decomposition(dec)[0]
                self.assertEqual((o.op, o.relation, o.movie or o.entity), (op, rel, name))

    def test_no_relation_and_no_template_is_still_cannot_infer_relation(self) -> None:
        with self.assertRaises(ValueError):
            compile_decomposition("1. Please summarise the plot.")
