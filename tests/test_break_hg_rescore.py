#!/usr/bin/env python3
"""Hand-computed checks for ``scripts/break_hg_rescore.py`` (exp-035, ADR 0041).

Every expected value is worked out by hand from Han & Gardent's fork code
(github.com/hankelvin/complex_question_decomposition @ 9c9a669), quoted in the docstrings.
GED tests pass a whitespace "lemmatizer" so they need no spaCy model; one test loads the
real model and is skipped when spaCy is absent.

Run::

    python -m unittest tests.test_break_hg_rescore -v
"""
from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "break_hg_rescore", REPO_ROOT / "scripts" / "break_hg_rescore.py"
)
hg = importlib.util.module_from_spec(_SPEC)
sys.modules["break_hg_rescore"] = hg
_SPEC.loader.exec_module(hg)


def ws(text: str) -> list[str]:
    return text.split()


class TestStringPath(unittest.TestCase):
    def test_format_qdmr_rewrites_refs_and_strips_return(self):
        # format_qdmr: split on ';', re.sub('return', '', part.strip()), #k -> @@k@@.
        parts = hg.format_qdmr("return the city of #1 ;  when did #2   happen")
        self.assertEqual(parts, [" the city of @@1@@", "when did @@2@@ happen"])

    def test_format_qdmr_return_inside_a_word_is_eaten_like_the_fork(self):
        self.assertEqual(hg.format_qdmr("who returned it"), ["who ed it"])

    def test_format_qdmr_drops_empty_parts(self):
        self.assertEqual(hg.format_qdmr("a ; ; b ;"), ["a", "b"])

    def test_clean_lingering_operators(self):
        # brackets removed, whitespace collapsed, re-joined " ;"
        self.assertEqual(hg.clean_lingering_operators("[select] who is x ;[project]  #1 of y"),
                         "who is x ;#1 of y")
        cot = "Thinking... The sequence of sub-questions should be: a ; b"
        self.assertEqual(hg.clean_lingering_operators(cot), "a ;b")

    def test_fork_strips_only_the_final_question_mark(self):
        _, s = hg.prepare(["Who is X?", "Where is #1?"], "hg_fork")
        self.assertEqual(s, "Who is X? @@SEP@@ Where is @@1@@")

    def test_paper_literal_strips_every_step_and_lowercases(self):
        _, s = hg.prepare(["Who is X?", "Where is #1? "], "hg_paper_literal")
        self.assertEqual(s, "who is x @@SEP@@ where is @@1@@")
        self.assertEqual(hg.prepare_question("Who Is X? ", "hg_paper_literal"), "who is x")
        self.assertEqual(hg.prepare_question("Who Is X? ", "hg_fork"), "Who Is X? ")

    def test_unknown_variant_refused(self):
        with self.assertRaises(ValueError):
            hg.prepare(["a"], "official")


class TestScoreItem(unittest.TestCase):
    def item(self, pred, gold, question="Who is X and where?"):
        return {"item_id": "x", "gold_hop_count": 2, "question": question,
                "pred_steps": pred, "gold_steps": gold}

    def test_em_ignores_final_qmark_and_case_under_fork(self):
        r = hg.score_item(self.item(["who is x", "where is #1"], ["Who is X", "Where is #1?"]),
                          "hg_fork", ws, 180.0)
        self.assertEqual(r["break_exact_match"], 1.0)

    def test_em_differs_on_inner_qmark_fork_vs_literal(self):
        it = self.item(["who is x?", "where is #1"], ["who is x", "where is #1"])
        self.assertEqual(hg.score_item(it, "hg_fork", ws, 180.0)["break_exact_match"], 0.0)
        self.assertEqual(hg.score_item(it, "hg_paper_literal", ws, 180.0)["break_exact_match"], 1.0)

    def test_identical_plans(self):
        r = hg.score_item(self.item(["a b", "c #1"], ["a b", "c #1"]), "hg_fork", ws, 180.0)
        self.assertEqual((r["break_exact_match"], r["ged"]), (1.0, 0.0))


class TestGed(unittest.TestCase):
    def test_extra_referencing_step(self):
        # pred: 2 nodes + edge 2->1 (total 3); gold: 1 node (total 1); normalisation 3.
        # Cheapest path: substitute 1<->1 (cost 0), delete node 2 (1) and its edge (1) = 2.
        ged, note = hg.normalized_ged(["a b", "c @@1@@"], ["a b"], ws, 180.0)
        self.assertTrue(math.isclose(ged, 2 / 3))
        self.assertIsNone(note)

    def test_substitution_cost_is_one_minus_ratio(self):
        # "a b" vs "a c": 1 match of 4 tokens -> ratio 0.5, cost 0.5; normalisation 1.
        ged, _ = hg.normalized_ged(["a b"], ["a c"], ws, 180.0)
        self.assertTrue(math.isclose(ged, 0.5))

    def test_over_budget_item_is_dropped(self):
        ged, note = hg.normalized_ged(["a b"], ["a c"], ws, -1.0)
        self.assertIsNone(ged)
        self.assertEqual(note, "dropped_time_budget")

    def test_both_empty(self):
        self.assertEqual(hg.normalized_ged([], [], ws, 180.0), (0.0, "both_empty"))

    def test_match_ratio_empty(self):
        self.assertEqual(hg.match_ratio([], []), 1.0)
        self.assertEqual(hg.match_ratio(["a"], []), 0.0)

    @unittest.skipUnless(importlib.util.find_spec("en_core_web_sm"), "spaCy model not installed")
    def test_spacy_lemmas_make_inflections_match(self):
        lemmas = hg.make_lemma_tokenizer("en_core_web_sm")
        # "performers" and "performer" share a lemma, so the plans are GED 0 under lemmas
        # and > 0 under whitespace tokens.
        a, b = ["who are the performers"], ["who is the performer"]
        self.assertEqual(hg.normalized_ged(a, b, lemmas, 180.0)[0], 0.0)
        self.assertGreater(hg.normalized_ged(a, b, ws, 180.0)[0], 0.0)


class TestAggregationAndFamily(unittest.TestCase):
    def rows(self, ems, geds, hops):
        return [{"item_id": f"i{k}", "gold_hop_count": h, "break_exact_match": e,
                 "sari": 0.5, "ged": g} for k, (e, g, h) in enumerate(zip(ems, geds, hops))]

    def test_aggregate_drops_none_ged_and_counts_it(self):
        agg = hg.aggregate(self.rows([1, 0, 0], [0.2, None, 0.4], [2, 2, 3]))
        self.assertEqual(agg["overall"]["ged_dropped"], 1)
        self.assertTrue(math.isclose(agg["overall"]["ged"], 0.3))
        self.assertEqual(agg["2"]["n"], 2)
        self.assertEqual(agg["4"]["n"], 0)

    def test_holm(self):
        # m=3, sorted p 0.01, 0.02, 0.04 -> 0.03, 0.04, max(0.04, 0.04) = 0.04
        self.assertEqual([round(x, 10) for x in hg.holm([0.04, 0.01, 0.02])], [0.04, 0.03, 0.04])

    def test_family_size_and_holm_columns(self):
        n = 40
        a = self.rows([1] * n, [0.1] * n, [2, 3, 4, 2] * 10)
        b = self.rows([0] * n, [0.3 + 0.001 * k for k in range(n)], [2, 3, 4, 2] * 10)
        tests = hg.family({"a": a, "b": b}, [["a", "b"]], 0.05, 30)
        self.assertEqual(len(tests), 12)  # 1 pair x 4 strata x 3 metrics
        for t in tests:
            self.assertIn("p_holm", t)

    def test_family_refuses_misaligned_hops(self):
        a = self.rows([1, 0], [0.1, 0.2], [2, 3])
        b = self.rows([1, 0], [0.1, 0.2], [3, 2])
        with self.assertRaises(SystemExit):
            hg.family({"a": a, "b": b}, [["a", "b"]], 0.05, 30)

    def test_apply_limit_is_per_hop(self):
        items = [{"gold_hop_count": h} for h in [2, 2, 2, 3, 3, 4]]
        self.assertEqual([i["gold_hop_count"] for i in hg.apply_limit(items, 1)], [2, 3, 4])
        self.assertIs(hg.apply_limit(items, None), items)


if __name__ == "__main__":
    unittest.main()
