#!/usr/bin/env python3
"""Unit tests for the few-shot exemplar ORDERING knob (issue #50, supervisor item A5).

No GPU, no model weights, no data files: these exercise
``components/decomposer/run_decomposer.py``'s ordering functions and the two shared
configs' declared value directly.

What the supervisor asked for on 2026-08-27 was three orderings - highest-similarity-first
(what the pipeline already did), its reverse, and least-complex-to-most-complex. Only the
first two are implemented. ``complexity_asc`` is a reserved name that is *refused*, because
"complexity" was never defined and choosing a definition here would invent methodology
(ADR 0032 item 5). The refusal is tested, not just the two implemented values, so that the
reservation cannot rot into a silent fallback.

The load-bearing property, and the reason the A5 contrast is a contrast at all: an ordering
changes presentation only. The same k exemplars appear under ``as_is`` and ``reversed`` -
same multiset, same length - so a quality difference between the two arms cannot be a
difference in *which* exemplars were shown.

Written as ``unittest.TestCase`` methods rather than bare pytest functions on purpose: the
project venv has no pytest (see ``tests/test_suite_collectability.py``), and that suite's
ast guard reads a parameter on a module-level ``test_*`` function as a fixture request, so
``pytest.mark.parametrize`` is not available here either. ``subTest`` covers the cases
parametrize would. pytest collects and runs this file unchanged.

Run::

    python -m pytest tests/test_few_shot_ordering.py
    .venv/bin/python -m unittest tests.test_few_shot_ordering
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "components" / "decomposer"))

import run_decomposer as rd  # noqa: E402

#: Five exemplars, mirroring the k=5 of the MuSiQue retrieval artifact, in retrieval-rank
#: order (highest similarity first). Distinct questions so order is observable in the output.
EXAMPLES = [
    {"pool_id": f"2hop__e{i}", "question": f"q{i}?", "decomposition": f"1. a{i}?\n2. b{i}?"}
    for i in range(1, 6)
]


class TestOrderFewShotExamples(unittest.TestCase):
    def test_as_is_is_the_input_order(self) -> None:
        out = rd.order_few_shot_examples(EXAMPLES, "as_is")
        self.assertEqual(
            [e["pool_id"] for e in out], [e["pool_id"] for e in EXAMPLES]
        )

    def test_reversed_is_the_input_reversed(self) -> None:
        out = rd.order_few_shot_examples(EXAMPLES, "reversed")
        self.assertEqual(
            [e["pool_id"] for e in out], [e["pool_id"] for e in EXAMPLES][::-1]
        )

    def test_both_orderings_show_the_same_multiset(self) -> None:
        """The A5 contrast's precondition: selection does not move, only presentation."""
        as_is = rd.order_few_shot_examples(EXAMPLES, "as_is")
        rev = rd.order_few_shot_examples(EXAMPLES, "reversed")
        self.assertEqual(len(as_is), len(EXAMPLES))
        self.assertEqual(len(rev), len(EXAMPLES))
        self.assertEqual(
            sorted(e["pool_id"] for e in as_is), sorted(e["pool_id"] for e in rev)
        )
        # Identity, not just equality: the same objects, never rebuilt or re-selected.
        self.assertEqual({id(e) for e in as_is}, {id(e) for e in EXAMPLES})
        self.assertEqual({id(e) for e in rev}, {id(e) for e in EXAMPLES})

    def test_it_orders_the_scored_pairs_the_prompt_log_header_prints(self) -> None:
        """The header's numbered ``sim=...`` list reuses this function, not its own copy.

        It is a list of ``(exemplar, score)`` tuples rather than of exemplars, so the
        function has to be element-type agnostic; if it were not, the header would drift
        back into selection order while the prompt body moved.
        """
        pairs = [(e, 0.9 - 0.1 * i) for i, e in enumerate(EXAMPLES)]
        self.assertEqual(rd.order_few_shot_examples(pairs, "as_is"), pairs)
        self.assertEqual(rd.order_few_shot_examples(pairs, "reversed"), pairs[::-1])

    def test_ordering_does_not_mutate_the_input_list(self) -> None:
        before = list(EXAMPLES)
        rd.order_few_shot_examples(EXAMPLES, "reversed")
        self.assertEqual(EXAMPLES, before)


class TestRefusals(unittest.TestCase):
    def test_complexity_asc_is_reserved_and_refused(self) -> None:
        self.assertIn("complexity_asc", rd.FEW_SHOT_ORDERINGS_RESERVED)
        self.assertNotIn("complexity_asc", rd.FEW_SHOT_ORDERINGS)
        with self.assertRaises(SystemExit) as ctx:
            rd.order_few_shot_examples(EXAMPLES, "complexity_asc")
        msg = str(ctx.exception)
        self.assertIn("complexity", msg)
        self.assertIn("ADR 0032", msg)

    def test_an_unknown_ordering_is_refused(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            rd.order_few_shot_examples(EXAMPLES, "random")
        self.assertIn("random", str(ctx.exception))

    def test_a_non_string_ordering_is_refused(self) -> None:
        for bad in (None, 1, ["as_is"]):
            with self.subTest(value=bad):
                with self.assertRaises(SystemExit):
                    rd.order_few_shot_examples(EXAMPLES, bad)

    def test_the_refusal_names_its_source(self) -> None:
        """A bad value must say whether the config or --ordering has to be fixed."""
        with self.assertRaises(SystemExit) as ctx:
            rd.resolve_few_shot_ordering("complexity_asc", src="--ordering")
        self.assertIn("--ordering", str(ctx.exception))
        with self.assertRaises(SystemExit) as ctx:
            rd.resolve_few_shot_ordering("nope", src="few_shot.ordering in cfg.json")
        self.assertIn("few_shot.ordering in cfg.json", str(ctx.exception))


class TestFormatAppliesTheOrdering(unittest.TestCase):
    """``format_few_shot_examples`` is the one place the ordering is applied."""

    def test_rendered_block_follows_the_requested_order(self) -> None:
        for ordering in rd.FEW_SHOT_ORDERINGS:
            # A seeded ordering needs its seed and the question id; the others refuse a seed.
            seeded = (
                {"ordering_seed": 1, "query_id": "2hop__q1"}
                if ordering in rd.FEW_SHOT_SEEDED_ORDERINGS
                else {}
            )
            expected = [
                e["question"]
                for e in rd.order_few_shot_examples(EXAMPLES, ordering, **seeded)
            ]
            for hop_count in (None, 3):  # unguided and guided
                with self.subTest(ordering=ordering, hop_count=hop_count):
                    block = rd.format_few_shot_examples(
                        EXAMPLES,
                        hop_count,
                        exemplar_hop_mode="exemplar_gold",
                        ordering=ordering,
                        **seeded,
                    )
                    seen = [
                        line[len("Question: ") :]
                        for line in block.splitlines()
                        if line.startswith("Question: ")
                    ]
                    self.assertEqual(seen, expected)

    def test_the_two_orderings_render_the_same_blocks_in_reverse(self) -> None:
        as_is = rd.format_few_shot_examples(
            EXAMPLES, None, exemplar_hop_mode="exemplar_gold", ordering="as_is"
        )
        rev = rd.format_few_shot_examples(
            EXAMPLES, None, exemplar_hop_mode="exemplar_gold", ordering="reversed"
        )
        self.assertNotEqual(as_is, rev)
        self.assertEqual(as_is.split("\n\n")[::-1], rev.split("\n\n"))

    def test_a_reserved_ordering_is_refused_at_the_format_call(self) -> None:
        with self.assertRaises(SystemExit):
            rd.format_few_shot_examples(
                EXAMPLES,
                None,
                exemplar_hop_mode="exemplar_gold",
                ordering="complexity_asc",
            )


class TestSelectionIsNeverReordered(unittest.TestCase):
    def test_examples_from_reranked_row_stays_in_rank_order(self) -> None:
        """Ordering is applied once, downstream.

        If it were applied at selection too, the two applications would cancel on the
        reranked path and the reversed arm would be indistinguishable from the as-is one.
        """
        row = {
            "query_id": "2hop__q",
            "typed_top_k": [
                {
                    "pool_id": e["pool_id"],
                    "pool_question": e["question"],
                    "pool_few_shot_decomposition_musique": e["decomposition"],
                }
                for e in EXAMPLES
            ],
        }
        out, dropped = rd.examples_from_reranked_row(row, "typed", 5)
        self.assertEqual(dropped, 0)
        self.assertEqual(
            [e["pool_id"] for e in out], [e["pool_id"] for e in EXAMPLES]
        )


class TestShuffledOrdering(unittest.TestCase):
    """The seeded 'shuffled' ordering of issue #60 / ADR 0039 item 5 (H4)."""

    def test_it_needs_a_seed_and_a_query_id(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            rd.order_few_shot_examples(EXAMPLES, "shuffled", query_id="q")
        self.assertIn("--ordering-seed", str(ctx.exception))
        for qid in (None, "  "):
            with self.subTest(query_id=qid), self.assertRaises(SystemExit):
                rd.order_few_shot_examples(EXAMPLES, "shuffled", ordering_seed=1, query_id=qid)
        with self.assertRaises(SystemExit):
            rd.order_few_shot_examples(EXAMPLES, "shuffled", ordering_seed="1", query_id="q")

    def test_a_seed_is_refused_with_the_unseeded_orderings(self) -> None:
        for ordering in ("as_is", "reversed"):
            with self.subTest(ordering=ordering), self.assertRaises(SystemExit) as ctx:
                rd.order_few_shot_examples(EXAMPLES, ordering, ordering_seed=1)
            self.assertIn("takes no seed", str(ctx.exception))
        # The CLI-level check refuses the same two cases, naming the flag.
        with self.assertRaises(SystemExit):
            rd.check_ordering_seed("shuffled", None, src="--ordering-seed")
        with self.assertRaises(SystemExit):
            rd.check_ordering_seed("as_is", 3, src="--ordering-seed")
        self.assertIsNone(rd.check_ordering_seed("as_is", None, src="x"))
        self.assertEqual(rd.check_ordering_seed("shuffled", 0, src="x"), 0)

    def test_same_objects_permuted(self) -> None:
        out = rd.order_few_shot_examples(EXAMPLES, "shuffled", ordering_seed=1, query_id="2hop__q1")
        self.assertEqual(len(out), len(EXAMPLES))
        self.assertEqual({id(e) for e in out}, {id(e) for e in EXAMPLES})

    def test_permutation_is_a_stable_function_of_seed_and_id(self) -> None:
        # Golden value, pinned from the implementation (2026-10-14 dry run on the pinned 600,
        # where row 1 of exp-014's retrieval file showed exactly this order): a change to the
        # derivation would silently re-shuffle every H4 arm, so it must fail here first.
        self.assertEqual(rd.shuffled_permutation(5, 1, "2hop__623501_297043"), [4, 1, 0, 3, 2])
        self.assertEqual(
            rd.shuffled_permutation(5, 1, "2hop__623501_297043"),
            rd.shuffled_permutation(5, 1, "2hop__623501_297043"),
        )
        self.assertEqual(sorted(rd.shuffled_permutation(7, 3, "x")), list(range(7)))
        self.assertEqual(rd.shuffled_permutation(0, 1, "x"), [])
        self.assertEqual(rd.shuffled_permutation(1, 1, "x"), [0])

    def test_seed_and_id_both_move_it(self) -> None:
        ids = [f"2hop__q{i}" for i in range(20)]
        self.assertNotEqual(
            [rd.shuffled_permutation(5, 1, q) for q in ids],
            [rd.shuffled_permutation(5, 2, q) for q in ids],
        )
        self.assertGreater(len({tuple(rd.shuffled_permutation(5, 1, q)) for q in ids}), 1)

    def test_independent_of_row_order(self) -> None:
        ids = [f"3hop__q{i}" for i in range(10)]
        forward = {q: rd.shuffled_permutation(5, 7, q) for q in ids}
        backward = {q: rd.shuffled_permutation(5, 7, q) for q in reversed(ids)}
        self.assertEqual(forward, backward)

    def test_independent_of_python_hash_seed(self) -> None:
        code = (
            "import sys; sys.path.insert(0, %r); sys.path.insert(0, %r); "
            "import run_decomposer as rd; "
            "print(rd.shuffled_permutation(5, 11, '4hop__abc'))"
        ) % (str(REPO_ROOT / "src"), str(REPO_ROOT / "components" / "decomposer"))
        outs = set()
        for hash_seed in ("0", "12345"):
            env = dict(os.environ, PYTHONHASHSEED=hash_seed)
            proc = subprocess.run(
                [sys.executable, "-c", code], capture_output=True, text=True, env=env
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            outs.add(proc.stdout.strip())
        self.assertEqual(len(outs), 1)


class TestShuffledDryRunEndToEnd(unittest.TestCase):
    """The real runner, --dry-run on the fabricated fixtures: seed recorded, order permuted."""

    RETRIEVAL = REPO_ROOT / "tests" / "fixtures" / "retrieval" / "top5_musique_conditions.jsonl"

    def _dry_run(self, out: Path, *extra: str) -> Path:
        env = dict(os.environ, QAV2_PATHS_CONFIG="configs/smoke_paths.json")
        proc = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "components" / "decomposer" / "run_decomposer.py"),
                "--model", "mistral_7b_instruct",
                "--config", "decomposer_musique.json",
                "--condition", "unguided",
                "--dry-run", "--dry-run-limit", "4",
                "--retrieval-input", str(self.RETRIEVAL),
                "--allow-unpinned-eval-set",
                "--output-root", str(out),
                *extra,
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        (run_dir,) = sorted(out.iterdir())
        return run_dir

    def test_shuffled_and_as_is(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            shuffled = self._dry_run(
                Path(tmp) / "s", "--ordering", "shuffled", "--ordering-seed", "1"
            )
            as_is = self._dry_run(Path(tmp) / "a", "--ordering", "as_is")
            rows = json.loads((shuffled / "results.json").read_text(encoding="utf-8"))
            metrics = json.loads((shuffled / "metrics.json").read_text(encoding="utf-8"))
            snapshot = json.loads((shuffled / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(snapshot["few_shot_ordering"], "shuffled")
            self.assertEqual(snapshot["few_shot_ordering_seed"], 1)
            self.assertEqual(metrics["few_shot_ordering_seed"], 1)
            tally = metrics["few_shot_shuffle_coincidences"]
            self.assertEqual(
                tally["rows_with_k_at_least_2"] + tally["rows_with_k_below_2"], len(rows)
            )
            for row in rows:
                self.assertEqual(row["few_shot_ordering_seed"], 1)
                self.assertEqual(
                    row["few_shot_permutation"],
                    rd.shuffled_permutation(
                        len(row["few_shot_permutation"]), 1, row["query_id"]
                    ),
                )
            self.assertIn("ordering seed: 1", (shuffled / "notes.md").read_text(encoding="utf-8"))
            a_rows = json.loads((as_is / "results.json").read_text(encoding="utf-8"))
            self.assertNotIn("few_shot_permutation", a_rows[0])
            a_metrics = json.loads((as_is / "metrics.json").read_text(encoding="utf-8"))
            self.assertIsNone(a_metrics["few_shot_ordering_seed"])
            self.assertIsNone(a_metrics["few_shot_shuffle_coincidences"])
            # The prompt body shows the as_is exemplars in permutation order. In a dry run
            # every row is logged as prompt_idx<position>, so sorted names are row order.
            s_logs = sorted((shuffled / "prompts_log").glob("prompt_idx*.txt"))
            a_logs = sorted((as_is / "prompts_log").glob("prompt_idx*.txt"))
            self.assertEqual([p.name for p in s_logs], [p.name for p in a_logs])
            for s_log, a_log, row in zip(s_logs, a_logs, rows):
                def questions(path: Path) -> list[str]:
                    body = path.read_text(encoding="utf-8").split("--- Prompt", 1)[1]
                    body = body.split("--- Raw generation", 1)[0]
                    return [l for l in body.splitlines() if l.startswith("Question: ")]
                q_s, q_a = questions(s_log), questions(a_log)
                perm = row["few_shot_permutation"]
                self.assertEqual(q_s[: len(perm)], [q_a[i] for i in perm])
                self.assertEqual(q_s[len(perm):], q_a[len(perm):])


class TestSharedConfigsDeclareAnOrdering(unittest.TestCase):
    """The runner requires the key rather than defaulting it, so both configs set it."""

    def test_both_shared_configs_declare_as_is(self) -> None:
        for name in ("decomposer_musique.json", "decomposer.json"):
            with self.subTest(config=name):
                cfg = json.loads(
                    (REPO_ROOT / "configs" / name).read_text(encoding="utf-8")
                )
                # exp-015 and every earlier run were highest-similarity-first; the default
                # keeps that, so nothing changes unless a run asks for it.
                self.assertEqual(cfg["few_shot"]["ordering"], "as_is")
                self.assertTrue(cfg["few_shot"]["_ordering_note"].strip())


if __name__ == "__main__":
    unittest.main()
