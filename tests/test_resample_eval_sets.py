#!/usr/bin/env python3
"""Checks for the stratified sampler and the named eval-set assertion (issue #56).

No data, no GPU: synthetic ids only.

- stratified draws: per-hop counts, no duplicate within a draw, determinism under a seed,
  a different seed disagrees, and the draws are identical whichever arm reads them (they
  depend only on the population ids, not on their order or on any arm);
- ranks and the pinned percentile follow their documented rules;
- ``assert_named_eval_set`` refuses a missing/extra id and records a match.

Run::

    .venv/bin/python -m unittest tests/test_resample_eval_sets.py -v
"""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from eval_sets import NamedEvalSet, assert_named_eval_set  # noqa: E402


def _load_resampler():
    spec = importlib.util.spec_from_file_location("resample_eval_sets", REPO / "scripts" / "resample_eval_sets.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


rs = _load_resampler()

HOPS = [2, 3, 4]
POP = {
    2: [f"2hop__{i}" for i in range(50)],
    3: [f"3hop1__{i}" for i in range(30)],
    4: [f"4hop1__{i}" for i in range(15)],
}


class StratifiedSamplerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.seeds = rs.draw_seeds(42, 20)
        self.draws = rs.stratified_draws(POP, hops=HOPS, per_hop=10, seeds=self.seeds)

    def test_seed_rule(self) -> None:
        self.assertEqual(self.seeds, list(range(42, 62)))

    def test_counts_and_no_duplicates(self) -> None:
        self.assertEqual(len(self.draws), 20)
        for d in self.draws:
            for hop in HOPS:
                self.assertEqual(len(d[hop]), 10)
                self.assertEqual(len(set(d[hop])), 10)
                self.assertTrue(set(d[hop]) <= set(POP[hop]))

    def test_deterministic_under_seed(self) -> None:
        again = rs.stratified_draws(POP, hops=HOPS, per_hop=10, seeds=self.seeds)
        self.assertEqual(again, self.draws)
        other = rs.stratified_draws(POP, hops=HOPS, per_hop=10, seeds=rs.draw_seeds(7, 20))
        self.assertNotEqual(other, self.draws)

    def test_draws_independent_across_draws(self) -> None:
        self.assertGreater(len({tuple(d[2]) for d in self.draws}), 1)

    def test_identical_across_arms(self) -> None:
        # Two arms present the same population in different orders (as two per-item files
        # can); the draws must not depend on that order.
        shuffled = {h: list(reversed(v)) for h, v in POP.items()}
        self.assertEqual(
            rs.stratified_draws(shuffled, hops=HOPS, per_hop=10, seeds=self.seeds), self.draws
        )

    def test_refuses_small_population_and_duplicates(self) -> None:
        with self.assertRaises(ValueError):
            rs.stratified_draws(POP, hops=HOPS, per_hop=16, seeds=[1])
        dup = dict(POP)
        dup[4] = POP[4] + [POP[4][0]]
        with self.assertRaises(ValueError):
            rs.stratified_draws(dup, hops=HOPS, per_hop=5, seeds=[1])


class RulesTest(unittest.TestCase):
    def test_competition_ranks(self) -> None:
        self.assertEqual(rs.competition_ranks([0.5, 0.7, 0.7, 0.1], False, 1e-12), [3, 1, 1, 4])
        self.assertEqual(rs.competition_ranks([0.5, 0.7, 0.1], True, 1e-12), [2, 3, 1])

    def test_pinned_percentile(self) -> None:
        self.assertEqual(rs.pinned_percentile([1, 2, 3, 4], 2.5, 1e-12), 50.0)
        self.assertEqual(rs.pinned_percentile([1, 2, 3, 4], 2, 1e-12), 37.5)
        self.assertEqual(rs.pinned_percentile([1, 2, 3, 4], 0, 1e-12), 0.0)


class NamedEvalSetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.named = NamedEvalSet(
            name="toy", source_path="toy.jsonl", source_sha256="x",
            ids={"2hop__a", "3hop1__b"}, ids_per_hop={"2": 1, "3": 1}, hops=[2, 3],
            expected_total=2,
        )

    def _assert(self, ids, total=None, allow=False):
        return assert_named_eval_set(
            set(ids), {}, len(ids) if total is None else total, self.named,
            allow_mismatch=allow, source="test", component="test", allow_flag="--allow",
        )

    def test_match(self) -> None:
        rec = self._assert(["2hop__a", "3hop1__b"])
        self.assertTrue(rec["matches_named_set"])
        self.assertFalse(rec["pinned"])

    def test_refusals(self) -> None:
        with self.assertRaises(SystemExit):
            self._assert(["2hop__a"])
        with self.assertRaises(SystemExit):
            self._assert(["2hop__a", "3hop1__b", "4hop1__c"])
        with self.assertRaises(SystemExit):
            self._assert(["2hop__a", "3hop1__b"], total=3)

    def test_allow_records_mismatch(self) -> None:
        rec = self._assert(["2hop__a"], allow=True)
        self.assertFalse(rec["matches_named_set"])
        self.assertEqual(rec["ids_missing_count"], 1)


class NamedPopulationCheckTest(unittest.TestCase):
    """The resampler's refusal, against the fixture tree's musique_dev_full (4 ids)."""

    FIXTURE_IDS = {"2hop__d001_a", "3hop1__d002_b", "4hop1__d003_c", "2hop__d004_p"}

    def setUp(self) -> None:
        from run_config import load_paths

        self.paths = load_paths("smoke_paths.json")

    def _rec(self, label, name="musique_dev_full", matches=True):
        es = None if name is None else {"named_eval_set": name, "matches_named_set": matches}
        return {"label": label, "evaluation_set": es}

    def test_pass(self) -> None:
        rec = rs.check_named_population(
            [self._rec("a"), self._rec("b")], set(self.FIXTURE_IDS), self.paths, allow=False
        )
        self.assertTrue(rec["verified"])
        self.assertEqual(rec["named_eval_set"], "musique_dev_full")

    def test_refusals(self) -> None:
        cases = [
            [self._rec("a"), self._rec("b", name=None)],          # unstamped arm
            [self._rec("a"), self._rec("b", matches=False)],      # recorded mismatch
        ]
        for recs in cases:
            with self.assertRaises(SystemExit):
                rs.check_named_population(recs, set(self.FIXTURE_IDS), self.paths, allow=False)
        with self.assertRaises(SystemExit):  # population is not the set's ids
            rs.check_named_population(
                [self._rec("a")], set(self.FIXTURE_IDS) - {"2hop__d001_a"}, self.paths, allow=False
            )

    def test_opt_out_is_recorded(self) -> None:
        rec = rs.check_named_population(
            [self._rec("a", name=None)], set(self.FIXTURE_IDS), self.paths, allow=True
        )
        self.assertFalse(rec["verified"])
        self.assertTrue(rec["allow_unnamed_population"])


class FpcTest(unittest.TestCase):
    def test_factors(self) -> None:
        f = rs.fpc_sd_factors({"2": 1252, "3": 760, "4": 405}, 200)
        self.assertAlmostEqual(f["2"], 0.9166, places=3)
        self.assertAlmostEqual(f["3"], 0.8584, places=3)
        self.assertAlmostEqual(f["4"], 0.7112, places=3)

    def test_corrected_sd(self) -> None:
        per = {"hop2": {"m": [0.0, 1.0]}, "hop3": {"m": [0.0, 1.0]}, "overall": {"m": [0.0, 1.0]}}
        f = {"2": 0.5, "3": 0.5}
        self.assertAlmostEqual(rs.fpc_corrected_sd(per, "m", "hop2", [2, 3], f), 0.70710678 / 0.5)
        # overall: sqrt(2 * 0.5 / 0.25) / 2
        self.assertAlmostEqual(rs.fpc_corrected_sd(per, "m", "overall", [2, 3], f), 1.0)


if __name__ == "__main__":
    unittest.main()
