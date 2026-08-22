#!/usr/bin/env python3
"""Checks for the measurable router (issue #27). CPU only, no weights, no network.

Everything here runs on hand-written synthetic rows (or the fabricated fixtures under
``tests/fixtures/``); the CLI checks use ``--dry-run``, which loads no model.

What it covers:

1. **Predictions keyed by query id** — the router's predictions file is built from the query
   id, and the file it writes is *exactly* what the consumer reads: the round trip through
   ``src/hop_matching.py::load_predicted_hops`` is asserted, so the producer and the two
   consumers (issue #15's ``predictions`` hop source, the decomposer's ``router_guided``
   condition) cannot drift apart silently.
2. **Join integrity, loudly** — a row with no query id and a repeated query id are refused
   on the producing side (before a model is loaded) *and* on the consuming side, and every
   refusal names the offenders. A query with no prediction is refused, never filled in from
   the gold depth.
3. **The few-shot router's prompt construction** — k exemplars per query, each labelled with
   its own gold hop depth parsed from its pool id, with the query itself excluded by id and
   by normalized question text (the decomposer's rule, imported not copied). The exemplar
   block's ``A: <n>`` line is checked to be readable by the run's own response parser, so the
   prompt and the parsing cannot disagree.
4. **The decomposer's router-predictions consumption** — ``hop_source`` resolution and all
   four of its refusals, and that the join moves the *prompt's* hop count while leaving the
   gold depth (and therefore the per-hop reporting and the pinned-set assertion) alone.
5. **End to end on the fixtures** — the few-shot router and the ``router_guided`` decomposer
   arm both run to completion in ``--dry-run``, the routed arm's prompts carry the predicted
   hop where ``oracle_guided`` carries the gold one, and the unchanged MetaQA router path
   still runs.

Run::

    .venv/bin/python -m unittest tests.test_router_predictions -v
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (
    REPO_ROOT / "src",
    REPO_ROOT / "components" / "decomposer",
    REPO_ROOT / "components" / "router",
):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import hop_matching as HM  # noqa: E402
import run_decomposer as rd  # noqa: E402
import run_router as rr  # noqa: E402
from run_config import PATHS_CONFIG_ENV, load_config  # noqa: E402

FIXTURES = REPO_ROOT / "tests" / "fixtures"
RETRIEVAL_FIXTURE = FIXTURES / "retrieval" / "top5_musique_conditions.jsonl"
PREDICTIONS_FIXTURE = FIXTURES / "router" / "hop_predictions_musique.jsonl"
ROUTER_RUNNER = REPO_ROOT / "components" / "router" / "run_router.py"
DECOMPOSER_RUNNER = REPO_ROOT / "components" / "decomposer" / "run_decomposer.py"
SMOKE_PATHS = "configs/smoke_paths.json"

#: The fixture tree holds 3 rows per hop, not the pinned 200 of ADR 0007, so every CLI check
#: here opts out of the pinned-set assertion explicitly. A real arm never passes this.
UNPINNED = "--allow-unpinned-eval-set"

#: mistral_7b_instruct is the decomposer folder used end to end: it ships the parity-guarded
#: unguided prompt configs/decomposer_musique.json requires. qwen2_5_0_5b is the router
#: folder: the smallest one, and no weights are loaded in a dry run anyway.
DECOMPOSER_MODEL = "mistral_7b_instruct"
ROUTER_MODEL = "qwen2_5_0_5b"


def run_cli(cmd: list[str]) -> subprocess.CompletedProcess:
    """Run a runner against the fixture tree, as the smoke test does."""
    env = dict(os.environ)
    env[PATHS_CONFIG_ENV] = SMOKE_PATHS
    return subprocess.run(
        [sys.executable, *[str(c) for c in cmd]],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )


def only_run_dir(root: Path) -> Path:
    """The single timestamped run directory a runner wrote under ``root``."""
    dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if len(dirs) != 1:
        raise AssertionError(f"expected exactly one run dir under {root}, found {dirs}")
    return dirs[0]


def candidate(pool_id: str, question: str, score: float = 0.9) -> dict:
    """One synthetic retrieval candidate, in the shape the retrieval chain writes."""
    return {
        "pool_id": pool_id,
        "pool_index": 0,
        "pool_question": question,
        "pool_few_shot_decomposition_musique": ["Step one?", "Step two?"],
        "score": score,
    }


def retrieval_row(query_id: str, question: str, candidates: list[dict]) -> dict:
    return {"query_id": query_id, "query_question": question, "typed_top_k": candidates}


def inference_row(query_id: str | None, gold_hop: int) -> dict:
    """A decomposer inference row, in the shape run_decomposer builds before the join."""
    return {
        "query_id": query_id,
        "question": f"question for {query_id}",
        "hop_count": gold_hop,
        "gold_hop_count": gold_hop,
        "retrieval_examples": [],
    }


# --------------------------------------------------------------- predictions file


class TestPredictionRows(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            {"query_id": "2hop__a", "question": "q1", "expected_hop": 2},
            {"query_id": "3hop1__b", "question": "q2", "expected_hop": 3},
            {"query_id": "4hop2__c", "question": "q3", "expected_hop": 4},
        ]

    def test_rows_carry_the_query_id_and_the_prediction(self) -> None:
        out = rr.build_prediction_rows(
            self.rows, [2, 2, 4], [True, False, True],
            id_field="query_id", hop_field="predicted_hop",
        )
        self.assertEqual([r["query_id"] for r in out], ["2hop__a", "3hop1__b", "4hop2__c"])
        self.assertEqual([r["predicted_hop"] for r in out], [2, 2, 4])
        self.assertEqual([r["correct"] for r in out], [True, False, True])
        # The middle row's hop came from parsing.default_hop, not from the response.
        self.assertEqual([r["parse_fallback"] for r in out], [False, True, False])

    def test_field_names_come_from_the_config(self) -> None:
        out = rr.build_prediction_rows(
            self.rows, [2, 3, 4], [True, True, True], id_field="id", hop_field="hop",
        )
        self.assertEqual(sorted(out[0]), ["correct", "expected_hop", "hop", "id",
                                          "parse_fallback"])

    def test_no_question_text_in_the_predictions_file(self) -> None:
        # ADR 0022 item 5 rejected keying this join on question text; the text is therefore
        # not carried in the file that gets joined.
        out = rr.build_prediction_rows(
            self.rows, [2, 3, 4], [True, True, True],
            id_field="query_id", hop_field="predicted_hop",
        )
        for row in out:
            self.assertNotIn("question", row)

    def test_misaligned_predictions_are_a_bug_not_a_file(self) -> None:
        with self.assertRaises(AssertionError):
            rr.build_prediction_rows(
                self.rows, [2, 3], [True, True],
                id_field="query_id", hop_field="predicted_hop",
            )

    def test_round_trip_through_the_consumer(self) -> None:
        """What the router writes is what ``load_predicted_hops`` reads. The contract."""
        out = rr.build_prediction_rows(
            self.rows, [2, 4, 4], [True, True, True],
            id_field="query_id", hop_field="predicted_hop",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.jsonl"
            rr.write_predictions_jsonl(path, out)
            loaded = HM.load_predicted_hops(
                path, id_field="query_id", hop_field="predicted_hop"
            )
        self.assertEqual(loaded, {"2hop__a": 2, "3hop1__b": 4, "4hop2__c": 4})

    def test_committed_fixture_is_readable_by_the_consumer(self) -> None:
        loaded = HM.load_predicted_hops(
            PREDICTIONS_FIXTURE, id_field="query_id", hop_field="predicted_hop"
        )
        self.assertEqual(len(loaded), 9)
        # Three of the nine deliberately disagree with the gold depth parsed from the id.
        disagree = [
            qid for qid, hop in loaded.items() if HM.parse_hop_from_id(qid) != hop
        ]
        self.assertEqual(len(disagree), 3, disagree)


class TestQueryIdIntegrity(unittest.TestCase):
    def test_unique_ids_pass(self) -> None:
        rr.assert_query_ids(
            [{"query_id": "2hop__a"}, {"query_id": "3hop1__b"}],
            source="fixture", reason="because",
        )

    def test_a_missing_id_is_refused_by_index(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            rr.assert_query_ids(
                [{"query_id": "2hop__a"}, {"query_id": None}, {"query_id": "  "}],
                source="fixture", reason="because",
            )
        message = str(ctx.exception)
        self.assertIn("2 of 3", message)
        self.assertIn("index 1", message)
        self.assertIn("index 2", message)

    def test_a_duplicate_id_is_refused_and_named(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            rr.assert_query_ids(
                [{"query_id": "2hop__a"}, {"query_id": "2hop__a"}],
                source="fixture", reason="because",
            )
        self.assertIn("2hop__a", str(ctx.exception))

    def test_a_duplicated_router_output_is_refused_on_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.jsonl"
            rr.write_predictions_jsonl(
                path,
                [
                    {"query_id": "2hop__a", "predicted_hop": 2},
                    {"query_id": "2hop__a", "predicted_hop": 3},
                ],
            )
            with self.assertRaises(SystemExit) as ctx:
                HM.load_predicted_hops(
                    path, id_field="query_id", hop_field="predicted_hop"
                )
        self.assertIn("2hop__a", str(ctx.exception))


# ------------------------------------------------------------ few-shot prompting


class TestFewShotPromptConstruction(unittest.TestCase):
    def setUp(self) -> None:
        self.row = retrieval_row(
            "2hop__q1",
            "Who leads the union that organised the strike?",
            [
                candidate("2hop__p1", "Who leads the party that won the election?"),
                candidate("3hop1__p2", "What is the capital of the country?"),
                candidate("4hop2__p3", "What is the population of the town?"),
            ],
        )

    def test_exemplars_are_labelled_with_their_own_hop_depth(self) -> None:
        examples, dropped = rr.examples_from_retrieval_row(
            self.row, mode="typed", k=3, exemplar_hop_source="pool_id"
        )
        self.assertEqual(dropped, 0)
        self.assertEqual([ex["hop_count"] for ex in examples], [2, 3, 4])
        self.assertEqual([ex["pool_id"] for ex in examples],
                         ["2hop__p1", "3hop1__p2", "4hop2__p3"])

    def test_the_query_is_excluded_from_its_own_exemplars_by_id(self) -> None:
        row = retrieval_row(
            "2hop__q1",
            "Who leads the union?",
            [
                candidate("2hop__q1", "A differently worded copy of the query"),
                candidate("2hop__p1", "Who leads the party?"),
                candidate("3hop1__p2", "What is the capital?"),
            ],
        )
        examples, dropped = rr.examples_from_retrieval_row(
            row, mode="typed", k=2, exemplar_hop_source="pool_id"
        )
        self.assertEqual(dropped, 1)
        self.assertEqual([ex["pool_id"] for ex in examples], ["2hop__p1", "3hop1__p2"])

    def test_the_query_is_excluded_by_normalized_question_text(self) -> None:
        row = retrieval_row(
            "2hop__q1",
            "Who leads the union?",
            [
                candidate("2hop__other", "  who LEADS the   union? "),
                candidate("2hop__p1", "Who leads the party?"),
            ],
        )
        examples, dropped = rr.examples_from_retrieval_row(
            row, mode="typed", k=1, exemplar_hop_source="pool_id"
        )
        self.assertEqual(dropped, 1)
        self.assertEqual([ex["pool_id"] for ex in examples], ["2hop__p1"])

    def test_too_few_usable_candidates_is_refused_with_counts(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            rr.examples_from_retrieval_row(
                self.row, mode="typed", k=5, exemplar_hop_source="pool_id"
            )
        message = str(ctx.exception)
        self.assertIn("only 3 usable", message)
        self.assertIn("k=5", message)

    def test_an_unparseable_pool_id_is_refused(self) -> None:
        row = retrieval_row(
            "2hop__q1", "Q?", [candidate("no_hop_prefix", "Some pool question?")]
        )
        with self.assertRaises(SystemExit) as ctx:
            rr.examples_from_retrieval_row(
                row, mode="typed", k=1, exemplar_hop_source="pool_id"
            )
        self.assertIn("no_hop_prefix", str(ctx.exception))

    def test_an_exemplar_without_a_question_is_refused(self) -> None:
        broken = candidate("2hop__p1", "x")
        broken["pool_question"] = "   "
        row = retrieval_row("2hop__q1", "Q?", [broken])
        with self.assertRaises(SystemExit) as ctx:
            rr.examples_from_retrieval_row(
                row, mode="typed", k=1, exemplar_hop_source="pool_id"
            )
        self.assertIn("pool_question", str(ctx.exception))

    def test_an_unknown_exemplar_hop_source_is_refused(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            rr.examples_from_retrieval_row(
                self.row, mode="typed", k=1, exemplar_hop_source="gold_step_count"
            )
        self.assertIn("gold_step_count", str(ctx.exception))

    def test_the_block_shape_is_question_then_hop_count(self) -> None:
        examples, _ = rr.examples_from_retrieval_row(
            self.row, mode="typed", k=2, exemplar_hop_source="pool_id"
        )
        block = rr.format_few_shot_examples(examples)
        self.assertEqual(
            block,
            "Q: Who leads the party that won the election?\nA: 2\n\n"
            "Q: What is the capital of the country?\nA: 3",
        )

    def test_the_prompt_fills_both_placeholders(self) -> None:
        template = (
            REPO_ROOT / "components" / "router" / "models" / "prompt_few_shot_musique.md"
        ).read_text(encoding="utf-8")
        examples, _ = rr.examples_from_retrieval_row(
            self.row, mode="typed", k=2, exemplar_hop_source="pool_id"
        )
        prompt = rr.build_prompt(
            template, "Who leads the union?", rr.format_few_shot_examples(examples)
        )
        self.assertNotIn("{few_shot_examples}", prompt)
        self.assertNotIn("{question}", prompt)
        self.assertIn("A: 2", prompt)
        self.assertIn("Q: Who leads the union?", prompt)
        # The query's own line is last and unanswered: the model has to fill it in.
        self.assertTrue(prompt.rstrip().endswith("A:"), prompt[-40:])

    def test_a_v1_prompt_renders_exactly_as_before(self) -> None:
        self.assertEqual(rr.build_prompt("Q: {question}\nA:", "why?"), "Q: why?\nA:")
        self.assertEqual(rr.build_prompt("Q: {{question}}", "why?"), "Q: why?")
        self.assertEqual(rr.build_prompt("no placeholders", "why?"), "no placeholders")

    def test_the_exemplar_answer_line_is_readable_by_the_run_s_parser(self) -> None:
        """The prompt format and the response parsing must agree, or 4-hop is unreadable."""
        cfg = load_config("router_musique.json")
        model_cfg = load_config(
            REPO_ROOT / "components" / "router" / "models" / ROUTER_MODEL / "config.json"
        )
        parsing = rr.apply_overrides(
            dict(model_cfg["parsing"]),
            cfg["parsing_overrides"],
            block="parsing",
            src="test",
        )
        parsing["hops"] = cfg["hops"]
        for hop in cfg["hops"]:
            self.assertEqual(rr.parse_hop_response(f" {hop}", "q", parsing), (hop, True))
            self.assertEqual(rr.parse_hop_response(f"A: {hop}", "q", parsing), (hop, True))
        # Nothing readable: the configured default, flagged as not parsed.
        hop, parsed = rr.parse_hop_response("no idea", "q", parsing)
        self.assertEqual(hop, int(parsing["default_hop"]))
        self.assertFalse(parsed)

    def test_unknown_override_keys_are_refused(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            rr.apply_overrides(
                {"answer_regex": "x"}, {"answr_regex": "y"}, block="parsing", src="test"
            )
        self.assertIn("answr_regex", str(ctx.exception))


class TestRouterConfigs(unittest.TestCase):
    def test_musique_router_config(self) -> None:
        cfg = load_config("router_musique.json")
        self.assertEqual(cfg["hops"], [2, 3, 4])
        self.assertEqual(cfg["questions_format"], "jsonl")
        self.assertEqual(cfg["questions_template_key"], "musique_eval_questions_template")
        self.assertEqual(cfg["eval_rows_per_hop"], 200)
        self.assertEqual(cfg["seed"], 42)
        self.assertIsNone(cfg["sample_size_per_hop"])
        self.assertTrue(cfg["few_shot"]["enabled"])
        self.assertIn(cfg["few_shot"]["exemplar_hop_source"], rr.EXEMPLAR_HOP_SOURCES)
        self.assertTrue(cfg["predictions"]["enabled"])
        self.assertEqual(cfg["predictions"]["id_field"], "query_id")
        self.assertEqual(cfg["predictions"]["hop_field"], "predicted_hop")
        self.assertEqual(cfg["prompt_file"], "prompt_few_shot_musique.md")
        self.assertEqual(cfg["retrieval"]["input_key"], "musique_few_shot_top5_pinned600")

    def test_the_predictions_field_names_are_the_consumers_defaults(self) -> None:
        router = load_config("router_musique.json")["predictions"]
        similarity = load_config("similarity.json")["hop_match"]
        decomposer = load_config("decomposer_musique.json")["hop_predictions"]
        self.assertEqual(router["id_field"], similarity["predictions_id_field"])
        self.assertEqual(router["hop_field"], similarity["predictions_hop_field"])
        self.assertEqual(router["id_field"], decomposer["id_field"])
        self.assertEqual(router["hop_field"], decomposer["hop_field"])

    def test_metaqa_router_config_unchanged(self) -> None:
        cfg = load_config("router.json")
        self.assertEqual(cfg["hops"], [1, 2, 3])
        self.assertEqual(cfg["questions_format"], "lines")
        self.assertFalse(cfg["few_shot"]["enabled"])
        self.assertFalse(cfg["predictions"]["enabled"])
        self.assertIsNone(cfg["prompt_file"])
        self.assertNotIn("eval_rows_per_hop", cfg)
        self.assertNotIn("parsing_overrides", cfg)


# ------------------------------------------ the decomposer's side of the join


class TestHopSourceResolution(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = load_config("decomposer_musique.json")

    def resolve(self, condition_name, *, guided=True, cli=None):
        name, block = rd.resolve_condition(self.cfg, condition_name)
        return rd.resolve_hop_source(
            name, block, self.cfg, guided=guided, cli_predictions=cli
        )

    def test_hop_source_is_a_permitted_condition_key(self) -> None:
        self.assertIn("hop_source", rd._CONDITION_KEYS)

    def test_oracle_guided_is_gold(self) -> None:
        record = self.resolve("oracle_guided")
        self.assertEqual(record["source"], "gold")
        self.assertIsNone(record["predictions_file"])

    def test_router_guided_takes_the_file_from_the_cli(self) -> None:
        record = self.resolve("router_guided", cli="runs/router/predictions.jsonl")
        self.assertEqual(record["source"], "predictions")
        self.assertEqual(record["predictions_file"], "runs/router/predictions.jsonl")
        self.assertEqual(record["id_field"], "query_id")
        self.assertEqual(record["hop_field"], "predicted_hop")

    def test_router_guided_without_a_file_is_refused(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            self.resolve("router_guided")
        self.assertIn("--hop-predictions", str(ctx.exception))

    def test_predictions_in_an_unguided_arm_is_refused(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            self.resolve("router_guided", guided=False, cli="p.jsonl")
        self.assertIn("unguided", str(ctx.exception))

    def test_a_predictions_file_on_a_gold_arm_is_refused(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            self.resolve("oracle_guided", cli="p.jsonl")
        self.assertIn("would be ignored", str(ctx.exception))

    def test_an_unknown_hop_source_is_refused(self) -> None:
        cfg = dict(self.cfg)
        cfg["hop_source"] = "similarity_vote"
        with self.assertRaises(SystemExit) as ctx:
            rd.resolve_hop_source(None, {}, cfg, guided=True, cli_predictions=None)
        self.assertIn("similarity_vote", str(ctx.exception))

    def test_the_metaqa_config_defaults_to_gold(self) -> None:
        cfg = load_config("decomposer.json")
        record = rd.resolve_hop_source(None, {}, cfg, guided=True, cli_predictions=None)
        self.assertEqual(record["source"], "gold")


class TestJoinPredictedHops(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            inference_row("2hop__a", 2),
            inference_row("3hop1__b", 3),
            inference_row("4hop2__c", 4),
        ]

    def test_the_prompt_hop_moves_and_the_gold_hop_does_not(self) -> None:
        record = rd.join_predicted_hops(
            self.rows,
            {"2hop__a": 3, "3hop1__b": 3, "4hop2__c": 2},
            predictions_path=Path("p.jsonl"),
            id_field="query_id",
        )
        self.assertEqual([r["hop_count"] for r in self.rows], [3, 3, 2])
        self.assertEqual([r["gold_hop_count"] for r in self.rows], [2, 3, 4])
        self.assertEqual(record["rows_joined"], 3)
        self.assertEqual(record["rows_agreeing_with_gold_hop"], 1)
        self.assertEqual(record["rows_disagreeing_with_gold_hop"], 2)
        self.assertEqual(record["predictions_unused"], 0)

    def test_unused_predictions_are_counted(self) -> None:
        record = rd.join_predicted_hops(
            self.rows,
            {"2hop__a": 2, "3hop1__b": 3, "4hop2__c": 4, "2hop__spare": 2},
            predictions_path=Path("p.jsonl"),
            id_field="query_id",
        )
        self.assertEqual(record["predictions_unused"], 1)
        self.assertEqual(record["predictions_in_file"], 4)

    def test_a_missing_prediction_is_refused_and_named(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            rd.join_predicted_hops(
                self.rows,
                {"2hop__a": 2},
                predictions_path=Path("p.jsonl"),
                id_field="query_id",
            )
        message = str(ctx.exception)
        self.assertIn("2 of 3", message)
        self.assertIn("3hop1__b", message)
        self.assertIn("4hop2__c", message)
        # No row may have been silently filled in from the gold depth.
        self.assertEqual([r["hop_count"] for r in self.rows], [2, 3, 4])

    def test_a_row_without_a_query_id_is_refused(self) -> None:
        rows = [inference_row("2hop__a", 2), inference_row(None, 3)]
        with self.assertRaises(SystemExit) as ctx:
            rd.join_predicted_hops(
                rows,
                {"2hop__a": 2},
                predictions_path=Path("p.jsonl"),
                id_field="query_id",
            )
        self.assertIn("no query id", str(ctx.exception))


# ------------------------------------------------------------------ end to end


class TestRouterCliDryRun(unittest.TestCase):
    def test_few_shot_router_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "router"
            proc = run_cli([
                ROUTER_RUNNER, "--model", ROUTER_MODEL,
                "--config", "router_musique.json",
                "--retrieval-input", RETRIEVAL_FIXTURE,
                "--dry-run", "--dry-run-limit", "9", UNPINNED,
                "--output-root", out,
            ])
            self.assertEqual(proc.returncode, 0, proc.stdout[-3000:] + proc.stderr[-3000:])
            run_dir = only_run_dir(out)
            metrics = json.loads((run_dir / "metrics.json").read_text())
            self.assertTrue(metrics["dry_run"])
            self.assertEqual(metrics["total_questions_loaded"], 9)
            self.assertEqual(metrics["questions_per_hop"],
                             {"2hop": 3, "3hop": 3, "4hop": 3})
            self.assertEqual(metrics["prompts_assembled"], 9)
            self.assertTrue(metrics["few_shot"]["enabled"])
            self.assertEqual(metrics["few_shot"]["k"], 5)
            # The two planted self-example rows in the fixture (same id, and the same
            # question text under another id) must both have been dropped.
            self.assertEqual(metrics["few_shot"]["self_examples_dropped"], 2)
            self.assertEqual(metrics["few_shot"]["rows_with_a_self_example_dropped"], 2)
            # A dry run predicts nothing, so it writes no predictions file and says so.
            self.assertIsNone(metrics["predictions_file"])
            self.assertIn("not written", metrics["predictions_note"])
            self.assertIsNone(metrics["accuracy_metrics"])
            self.assertFalse(metrics["model_size"]["ceiling_asserted"])

            snapshot = json.loads((run_dir / "config.json").read_text())
            self.assertTrue(snapshot["predictions"]["enabled"])
            self.assertEqual(snapshot["predictions"]["id_field"], "query_id")
            self.assertIsNotNone(snapshot["retrieval"]["input_sha256"])

            prompt = (run_dir / "prompts_log" / "prompt_idx0001.txt").read_text()
            self.assertIn("Few-shot exemplars (5)", prompt)
            # Every exemplar states a hop count, and the query's own line is unanswered.
            self.assertEqual(prompt.count("\nA: "), 5)
            self.assertTrue(prompt.rstrip().endswith("A:"), prompt[-40:])

    def test_metaqa_router_dry_run_still_works(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "router"
            proc = run_cli([
                ROUTER_RUNNER, "--model", ROUTER_MODEL,
                "--dry-run", "--dry-run-limit", "3",
                "--output-root", out,
            ])
            self.assertEqual(proc.returncode, 0, proc.stdout[-3000:] + proc.stderr[-3000:])
            metrics = json.loads((only_run_dir(out) / "metrics.json").read_text())
            self.assertFalse(metrics["few_shot"]["enabled"])
            self.assertIsNone(metrics["predictions_file"])
            self.assertEqual(metrics["prompts_assembled"], 3)

    def test_a_few_shot_config_without_exemplars_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            proc = run_cli([
                ROUTER_RUNNER, "--model", ROUTER_MODEL,
                "--config", "router_musique.json",
                "--dry-run", UNPINNED, "--output-root", Path(tmp) / "router",
            ])
            # The fixture tree has no file behind retrieval.input_key, so the run refuses
            # rather than silently prompting zero-shot under a few-shot label.
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("retrieval", (proc.stdout + proc.stderr).lower())

    def test_the_v1_prompt_is_refused_by_the_few_shot_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            proc = run_cli([
                ROUTER_RUNNER, "--model", ROUTER_MODEL,
                "--config", "router_musique.json",
                "--prompt-file", "prompt.md",
                "--retrieval-input", RETRIEVAL_FIXTURE,
                "--dry-run", UNPINNED, "--output-root", Path(tmp) / "router",
            ])
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("few_shot_examples", proc.stdout + proc.stderr)


class TestDecomposerRouterGuidedDryRun(unittest.TestCase):
    def decomposer_cmd(self, condition: str, out: Path, *extra) -> list:
        return [
            DECOMPOSER_RUNNER, "--model", DECOMPOSER_MODEL,
            "--config", "decomposer_musique.json", "--condition", condition,
            "--retrieval-input", RETRIEVAL_FIXTURE,
            "--dry-run", "--dry-run-limit", "9", UNPINNED,
            "--output-root", out, *extra,
        ]

    def test_router_guided_uses_the_predicted_hop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            routed_out, oracle_out = Path(tmp) / "routed", Path(tmp) / "oracle"
            routed = run_cli(self.decomposer_cmd(
                "router_guided", routed_out, "--hop-predictions", PREDICTIONS_FIXTURE
            ))
            self.assertEqual(routed.returncode, 0,
                             routed.stdout[-3000:] + routed.stderr[-3000:])
            oracle = run_cli(self.decomposer_cmd("oracle_guided", oracle_out))
            self.assertEqual(oracle.returncode, 0,
                             oracle.stdout[-3000:] + oracle.stderr[-3000:])

            routed_dir, oracle_dir = only_run_dir(routed_out), only_run_dir(oracle_out)
            routed_rows = json.loads((routed_dir / "results.json").read_text())
            oracle_rows = json.loads((oracle_dir / "results.json").read_text())
            predicted = HM.load_predicted_hops(
                PREDICTIONS_FIXTURE, id_field="query_id", hop_field="predicted_hop"
            )

            self.assertEqual(len(routed_rows), 9)
            for row in routed_rows:
                self.assertEqual(row["hop_count_source"], "predictions")
                # The prompt states the prediction; the row stays filed under its gold depth.
                self.assertEqual(row["prompt_hop_count"], predicted[row["query_id"]])
                self.assertEqual(row["hop_count"], HM.parse_hop_from_id(row["query_id"]))
            for row in oracle_rows:
                self.assertEqual(row["hop_count_source"], "gold")
                self.assertEqual(row["prompt_hop_count"], row["hop_count"])

            # The two arms are the same run except for that number: same ids, same order.
            self.assertEqual([r["query_id"] for r in routed_rows],
                             [r["query_id"] for r in oracle_rows])

            metrics = json.loads((routed_dir / "metrics.json").read_text())
            self.assertEqual(metrics["rows_loaded_per_hop"], {"2": 3, "3": 3, "4": 3})
            join = metrics["hop_source"]["join"]
            self.assertEqual(join["rows_joined"], 9)
            self.assertEqual(join["rows_agreeing_with_gold_hop"], 6)
            self.assertEqual(join["rows_disagreeing_with_gold_hop"], 3)
            self.assertIsNotNone(metrics["hop_source"]["predictions_file_sha256"])

            # The prompts differ from the oracle arm's exactly where the prediction does.
            def prompt_of(run_dir: Path, index: int, hop: int) -> str:
                return (
                    run_dir / "prompts_log" / f"prompt_idx{index:04d}_hop{hop}.txt"
                ).read_text()

            self.assertIn("Hop count in prompt: 3 (source: predictions)",
                          prompt_of(routed_dir, 2, 2))
            self.assertIn("Hop count in prompt: 2 (source: gold)",
                          prompt_of(oracle_dir, 2, 2))

    def test_a_predictions_file_missing_a_query_is_refused(self) -> None:
        rows = [
            json.loads(line)
            for line in PREDICTIONS_FIXTURE.read_text().splitlines()
            if line.strip()
        ]
        dropped = rows[-1]["query_id"]
        with tempfile.TemporaryDirectory() as tmp:
            partial = Path(tmp) / "partial.jsonl"
            partial.write_text(
                "".join(json.dumps(r) + "\n" for r in rows[:-1]), encoding="utf-8"
            )
            proc = run_cli(self.decomposer_cmd(
                "router_guided", Path(tmp) / "routed", "--hop-predictions", partial
            ))
            self.assertNotEqual(proc.returncode, 0)
            output = proc.stdout + proc.stderr
            self.assertIn("no prediction", output)
            self.assertIn(dropped, output)

    def test_a_predictions_file_on_the_oracle_arm_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            proc = run_cli(self.decomposer_cmd(
                "oracle_guided", Path(tmp) / "oracle",
                "--hop-predictions", PREDICTIONS_FIXTURE,
            ))
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("would be ignored", proc.stdout + proc.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
