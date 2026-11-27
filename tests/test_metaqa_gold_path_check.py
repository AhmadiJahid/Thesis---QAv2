"""Tests for scripts/metaqa_gold_path_check.py on a hand-written toy KB.

Run::

    .venv/bin/python -m unittest tests.test_metaqa_gold_path_check
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from evaluate_decompositions import compile_decomposition  # noqa: E402
from kg import build_metaqa_kg  # noqa: E402
from metaqa_gold_path_check import check, entity_mentions, render  # noqa: E402

KB = """Glass Harbor|directed_by|Ada Mireles
Winter Marbles|directed_by|Ada Mireles
Winter Marbles|directed_by|Bo Fenwick
Glass Harbor|has_genre|Mystery
Winter Marbles|has_genre|Drama
Paper Lanterns|directed_by|Bo Fenwick
Paper Lanterns|has_genre|Comedy
"""


class GoldPathCheckTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        kb = Path(cls.tmp.name) / "kb.txt"
        kb.write_text(KB, encoding="utf-8")
        cls.kg = build_metaqa_kg(kb)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_mentions_are_word_bounded_and_longest_first(self) -> None:
        got = entity_mentions("who co-directed with Ada Mireles", list(self.kg.entity_to_id), 2)
        self.assertEqual(got, ["Ada Mireles"])

    def test_co_director_gold_needs_the_topic_dropped(self) -> None:
        # director -> movies -> directors; MetaQA gold excludes the topic person itself
        res = check(self.kg, ["who co-directed with Ada Mireles"], [{"Bo Fenwick"}], 2, 2)
        self.assertEqual(res["reproduced_by_some_path"], 1)
        self.assertEqual(res["per_convention"]["keep_topic"], 0)
        self.assertEqual(res["per_convention"]["drop_topic_final"], 1)

    def test_three_hop_gold_needs_the_topic_dropped_mid_path(self) -> None:
        # movie -> director -> other movies -> genre; the topic movie's own genre is not gold
        res = check(self.kg, ["genres of films by the director of Glass Harbor"], [{"Drama"}], 3, 2)
        self.assertEqual(res["reproduced_by_some_path"], 1)
        self.assertEqual(res["per_convention"]["keep_topic"], 0)
        self.assertEqual(res["per_convention"]["drop_topic_intermediate"], 1)

    def test_unreachable_gold_is_reported_not_hidden(self) -> None:
        res = check(self.kg, ["who directed Glass Harbor"], [{"Bo Fenwick"}], 1, 2)
        self.assertEqual(res["reproduced_by_some_path"], 0)
        self.assertEqual(res["unreproduced"], ["who directed Glass Harbor"])

    def test_compiler_phrasing_compiles_to_the_intended_ops(self) -> None:
        dec = render("Glass Harbor", [("directed_by", "fwd"), ("directed_by", "rev"), ("has_genre", "fwd")], "compiler")
        ops = compile_decomposition(dec)
        self.assertEqual([o.op for o in ops], ["VALUES_OF_MOVIE", "MOVIES_WITH_VALUE", "PROJECT_VALUES_FROM_MOVIES"])
        self.assertEqual([o.relation for o in ops], ["directed_by", "directed_by", "has_genre"])

    def test_natural_phrasing_of_a_reverse_ref_step_compiles_to_a_reverse_lookup(self) -> None:
        # "What movies did [#1] direct?" with [#1] = people: before exp-039's fix this
        # compiled to a forward projection from people and executed to an empty set.
        dec = render("Glass Harbor", [("directed_by", "fwd"), ("directed_by", "rev")], "natural")
        ops = compile_decomposition(dec)
        self.assertEqual(ops[1].op, "MOVIES_WITH_VALUE")
        self.assertEqual(ops[1].ref_step, 1)


if __name__ == "__main__":
    unittest.main()
