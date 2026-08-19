#!/usr/bin/env python3
"""GPU-free checks for the decomposer LoRA arms (issue #13).

Everything here runs on CPU with no weights and no network: the committed config loads,
the arms select what they say they select, the train/eval overlap assertion fails loudly
and names the offending ids, the prompt/completion formatting matches what the runner will
produce at inference, and the parameter-ceiling gate in ``train_lora.py`` is wired to
``src/model_size.py`` (both directions: within the ceiling and over it).

It also covers the two guards on the adapter *evaluation* path (PR #24 re-review): an
adapter may only be evaluated on the prompt its training run recorded
(``check_adapter_prompt_parity``), and it may only attach to the base model
``adapter_config.json`` names (``assert_adapter_base_model``). Their fixtures are fabricated
JSON in a temp directory - no weights, no run directory, no data.

Run::

    .venv/bin/python -m unittest discover -s tests -v
    .venv/bin/python tests/test_finetune_lora.py
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path
from typing import Any

#: Sentinel for "leave this key out of the fabricated record entirely".
_ABSENT = object()

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "components" / "decomposer"))

import finetune_data as fd  # noqa: E402
import train_lora  # noqa: E402
from model_size import load_limits  # noqa: E402
from musique_decompositions_evaluator import _decomp_to_steps  # noqa: E402
from run_config import load_config, load_paths, require  # noqa: E402

CONFIG_NAME = "finetune_decomposer.json"
ARM_NAMES = ("pool_2000", "full_train", "generalisation_2_3hop")

#: Data paths are read through the smoke paths config, whose data_root is the fabricated
#: fixture tree — so no test here can reach the real dataset location. It is passed
#: explicitly rather than through QAV2_PATHS_CONFIG, which would leak into other modules.
SMOKE_PATHS_CONFIG = "smoke_paths.json"


def _rows(*specs: tuple[str, int]) -> list[dict[str, Any]]:
    """Fabricated source rows: (id, number of steps)."""
    out = []
    for row_id, steps in specs:
        out.append(
            {
                "id": row_id,
                "question": f"question for {row_id}?",
                "question_decomposition": [
                    {"id": i, "question": f"step {i} of {row_id}?"} for i in range(1, steps + 1)
                ],
            }
        )
    return out


DATA_CFG = {
    "id_field": "id",
    "question_field": "question",
    "hop_count_field": "hop_count",
    "step_fields": ["question_decomposition", "few_shot_decomposition_musique"],
}

#: The report cap comes from the committed config, not from a constant in the module under
#: test (there is none any more) and not from a literal here.
MAX_LOAD_IDS = int(require(load_config(CONFIG_NAME), "overlap_check.max_reported_load_ids"))


def _built(rows: list[dict[str, Any]]) -> tuple[list[fd.TrainingExample], dict[str, Any]]:
    """``build_examples`` with the committed report cap."""
    return fd.build_examples(rows, DATA_CFG, max_reported_ids=MAX_LOAD_IDS)


class TestConfigLoads(unittest.TestCase):
    """The committed config has the three arms and the keys every consumer requires."""

    def setUp(self) -> None:
        self.cfg = load_config(CONFIG_NAME)

    def test_three_arms_are_defined(self) -> None:
        arms = require(self.cfg, "arms")
        self.assertEqual(sorted(k for k in arms if k != "_note"), sorted(ARM_NAMES))

    def test_default_arm_resolves(self) -> None:
        fd.resolve_arm(self.cfg, require(self.cfg, "default_arm"))

    def test_unknown_arm_is_refused_and_lists_the_known_ones(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            fd.resolve_arm(self.cfg, "no_such_arm")
        for name in ARM_NAMES:
            self.assertIn(name, str(ctx.exception))

    def test_every_arm_has_the_fields_the_selector_requires(self) -> None:
        for name in ARM_NAMES:
            arm = fd.resolve_arm(self.cfg, name)
            for key in (
                "train_source_key",
                "train_source_path",
                "train_hops",
                "max_examples",
                "stratify_cap_by_hop",
                "eval_hops",
            ):
                self.assertIn(key, arm, f"arm {name} is missing {key}")

    def test_arm_sources_resolve_to_paths_config_keys(self) -> None:
        paths_cfg = load_paths(SMOKE_PATHS_CONFIG)
        for name in ARM_NAMES:
            arm = fd.resolve_arm(self.cfg, name)
            path = fd.arm_source_path(arm, paths_cfg)
            self.assertTrue(str(path).endswith(".jsonl"), path)

    def test_the_two_data_arms_are_the_pool_and_the_full_split(self) -> None:
        pool = fd.resolve_arm(self.cfg, "pool_2000")
        full = fd.resolve_arm(self.cfg, "full_train")
        self.assertEqual(pool["max_examples"], 2000)
        self.assertEqual(pool["train_source_key"], "musique_pool_enriched")
        self.assertIsNone(full["max_examples"])
        self.assertEqual(full["train_source_key"], "musique_train")

    def test_generalisation_arm_trains_on_2_3_hop_and_evaluates_on_4_hop(self) -> None:
        arm = fd.resolve_arm(self.cfg, "generalisation_2_3hop")
        self.assertEqual(arm["train_hops"], [2, 3])
        self.assertEqual(arm["eval_hops"], [4])

    def test_evaluation_block_names_the_string_level_evaluator(self) -> None:
        evaluation = require(self.cfg, "evaluation")
        self.assertEqual(
            require(evaluation, "evaluator_script"),
            "scripts/musique_decompositions_evaluator.py",
        )


class TestOverlapAssertion(unittest.TestCase):
    """Overlap between training data and the evaluation set is a hard failure."""

    def test_disjoint_sets_pass_and_report_counts(self) -> None:
        examples, _ = _built(_rows(("2hop__t1", 2), ("3hop1__t2", 3)))
        record = fd.assert_no_eval_overlap(examples, {"2hop__d1"}, max_reported=20)
        self.assertEqual(record["overlap_count"], 0)
        self.assertEqual(record["checked_training_ids"], 2)
        self.assertTrue(record["asserted"])

    def test_overlap_raises_and_names_the_offending_ids(self) -> None:
        examples, _ = _built(_rows(("2hop__t1", 2), ("2hop__d1", 2), ("4hop1__d2", 4)))
        with self.assertRaises(SystemExit) as ctx:
            fd.assert_no_eval_overlap(examples, {"2hop__d1", "4hop1__d2"}, max_reported=20)
        message = str(ctx.exception)
        self.assertIn("2hop__d1", message)
        self.assertIn("4hop1__d2", message)
        self.assertNotIn("2hop__t1", message)
        self.assertIn("REFUSING TO TRAIN", message)

    def test_the_offending_id_list_is_capped_and_says_how_many_more(self) -> None:
        overlapping = [(f"2hop__d{i}", 2) for i in range(10)]
        examples, _ = _built(_rows(*overlapping))
        eval_ids = {row_id for row_id, _ in overlapping}
        with self.assertRaises(SystemExit) as ctx:
            fd.assert_no_eval_overlap(examples, eval_ids, max_reported=3)
        message = str(ctx.exception)
        self.assertIn("(+7 more)", message)
        self.assertIn("10 training example(s)", message)

    def test_eval_ids_come_from_the_three_adr_0007_hop_files(self) -> None:
        cfg = load_config(CONFIG_NAME)
        paths_cfg = load_paths(SMOKE_PATHS_CONFIG)
        eval_ids, record = fd.load_eval_ids(paths_cfg, require(cfg, "eval_set"))
        self.assertEqual(record["hops"], [2, 3, 4])
        self.assertEqual(len(record["files"]), 3)
        self.assertEqual(record["num_ids"], len(eval_ids))
        # The fixture set carries three ids per hop file; the real set carries 200 (ADR 0007).
        self.assertEqual(sorted(record["ids_per_hop"]), ["2", "3", "4"])
        # ... and the counts it found are the ones the fixture paths config declares, asserted.
        self.assertEqual(record["expected_ids_per_hop"], 3)
        self.assertEqual(record["expected_total_ids"], 9)
        self.assertTrue(record["expected_counts_asserted"])
        self.assertIn("smoke_paths.json", record["expected_counts_source"])

    def test_the_committed_config_declares_the_adr_0007_counts(self) -> None:
        """200 per hop and 600 in total are config values, so they can be asserted."""
        cfg = load_config(CONFIG_NAME)
        expected = require(cfg, "eval_set.expected")
        self.assertEqual(expected["ids_per_hop"], 200)
        self.assertEqual(expected["total_ids"], 600)
        # No paths override: configs/paths.json must not relax the real counts. Read with
        # load_config, not load_paths, so QAV2_PATHS_CONFIG (set when the smoke test runs
        # this suite) cannot substitute the fixture paths config here.
        real_paths = load_config("paths.json")
        self.assertNotIn(fd.PATHS_EVAL_EXPECTED_KEY, real_paths)
        per_hop, total, source = fd.expected_eval_counts(real_paths, require(cfg, "eval_set"))
        self.assertEqual((per_hop, total), (200, 600))
        self.assertIn("eval_set.expected", source)

    def test_a_misresolved_id_field_is_fatal_instead_of_a_vacuous_pass(self) -> None:
        """The C1 failure: zero eval ids used to make the overlap check pass silently."""
        cfg = load_config(CONFIG_NAME)
        paths_cfg = load_paths(SMOKE_PATHS_CONFIG)
        eval_cfg = dict(require(cfg, "eval_set"))
        eval_cfg["id_field"] = "question_id"  # the field this dataset does not use
        with self.assertRaises(SystemExit) as ctx:
            fd.load_eval_ids(paths_cfg, eval_cfg)
        message = str(ctx.exception)
        self.assertIn("REFUSING TO TRAIN", message)
        self.assertIn("0 distinct id(s)", message)
        self.assertIn("question_id", message)

    def test_a_short_hop_file_is_fatal(self) -> None:
        """A file with fewer ids than declared: same class of failure, caught the same way."""
        cfg = load_config(CONFIG_NAME)
        paths_cfg = load_paths(SMOKE_PATHS_CONFIG)
        # The fixture files hold 3 rows each; declaring the real 200/600 must fail here.
        eval_cfg = dict(require(cfg, "eval_set"))
        eval_cfg["expected"] = {"ids_per_hop": 200, "total_ids": 600}
        paths_cfg = {k: v for k, v in paths_cfg.items() if k != fd.PATHS_EVAL_EXPECTED_KEY}
        with self.assertRaises(SystemExit) as ctx:
            fd.load_eval_ids(paths_cfg, eval_cfg)
        message = str(ctx.exception)
        self.assertIn("3 distinct id(s), expected 200", message)

    def test_inconsistent_declared_counts_are_refused(self) -> None:
        cfg = load_config(CONFIG_NAME)
        eval_cfg = dict(require(cfg, "eval_set"))
        eval_cfg["expected"] = {"ids_per_hop": 200, "total_ids": 60}
        with self.assertRaises(SystemExit) as ctx:
            fd.expected_eval_counts({"_config_path": "x"}, eval_cfg)
        self.assertIn("inconsistent", str(ctx.exception))

    def test_an_empty_eval_id_set_is_refused_outright(self) -> None:
        examples, _ = _built(_rows(("2hop__t1", 2)))
        with self.assertRaises(SystemExit) as ctx:
            fd.assert_no_eval_overlap(examples, set(), max_reported=20)
        message = str(ctx.exception)
        self.assertIn("REFUSING TO TRAIN", message)
        self.assertIn("empty", message)

    def test_the_fixture_arms_have_no_overlap_with_the_fixture_eval_set(self) -> None:
        cfg = load_config(CONFIG_NAME)
        paths_cfg = load_paths(SMOKE_PATHS_CONFIG)
        eval_ids, _ = fd.load_eval_ids(paths_cfg, require(cfg, "eval_set"))
        for name in ARM_NAMES:
            arm = fd.resolve_arm(cfg, name)
            rows = fd.load_jsonl(fd.arm_source_path(arm, paths_cfg))
            examples, _ = fd.select_arm_examples(
                arm,
                rows,
                require(cfg, "data"),
                seed=int(require(cfg, "seed")),
                max_reported_ids=MAX_LOAD_IDS,
            )
            record = fd.assert_no_eval_overlap(examples, eval_ids, max_reported=20)
            self.assertEqual(record["overlap_count"], 0, name)


class TestArmSelection(unittest.TestCase):
    """Hop filtering, the seeded cap, and the hop bookkeeping."""

    def test_hop_comes_from_the_id_prefix(self) -> None:
        self.assertEqual(fd.hop_from_id("4hop2__123_456"), 4)
        self.assertEqual(fd.hop_from_id("2hop__1_2"), 2)
        self.assertIsNone(fd.hop_from_id("no-hop-prefix"))

    def test_generalisation_filter_drops_4_hop_rows(self) -> None:
        rows = _rows(("2hop__a", 2), ("3hop1__b", 3), ("4hop1__c", 4), ("4hop2__d", 4))
        examples, _ = _built(rows)
        kept = fd.filter_hops(examples, [2, 3])
        self.assertEqual([ex.row_id for ex in kept], ["2hop__a", "3hop1__b"])
        self.assertEqual(fd.hop_counts(kept), {"2": 1, "3": 1})

    def test_no_hop_filter_keeps_everything(self) -> None:
        examples, _ = _built(_rows(("2hop__a", 2), ("4hop1__c", 4)))
        self.assertEqual(len(fd.filter_hops(examples, None)), 2)

    def test_cap_is_seeded_and_reproducible(self) -> None:
        examples, _ = _built(_rows(*[(f"2hop__r{i}", 2) for i in range(20)]))
        first = fd.cap_examples(examples, 5, stratify_by_hop=False, seed=42)
        again = fd.cap_examples(examples, 5, stratify_by_hop=False, seed=42)
        other = fd.cap_examples(examples, 5, stratify_by_hop=False, seed=7)
        self.assertEqual(len(first), 5)
        self.assertEqual([e.row_id for e in first], [e.row_id for e in again])
        self.assertNotEqual([e.row_id for e in first], [e.row_id for e in other])

    def test_cap_above_the_available_rows_is_a_no_op(self) -> None:
        examples, _ = _built(_rows(("2hop__a", 2), ("3hop1__b", 3)))
        self.assertEqual(len(fd.cap_examples(examples, 2000, stratify_by_hop=True, seed=42)), 2)

    def test_stratified_cap_spreads_across_hop_buckets(self) -> None:
        rows = [(f"2hop__a{i}", 2) for i in range(10)]
        rows += [(f"3hop1__b{i}", 3) for i in range(10)]
        rows += [(f"4hop1__c{i}", 4) for i in range(2)]
        examples, _ = _built(_rows(*rows))
        selected = fd.cap_examples(examples, 9, stratify_by_hop=True, seed=42)
        counts = fd.hop_counts(selected)
        self.assertEqual(len(selected), 9)
        # The 4-hop bucket only has 2 rows, so its shortfall goes to the larger buckets
        # instead of shrinking the total below the cap.
        self.assertEqual(counts["4"], 2)
        self.assertEqual(counts["2"] + counts["3"], 7)

    def test_pool_rows_use_the_pool_step_field(self) -> None:
        pool_row = {
            "id": "2hop__p1",
            "question": "Who directed it?",
            "few_shot_decomposition_musique": ["Who scored it?", "Who directed #1?"],
        }
        examples, stats = _built([pool_row])
        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0].steps, ("Who scored it?", "Who directed #1?"))
        self.assertEqual(stats["dropped_missing_steps"], 0)

    def test_rows_without_steps_or_question_are_dropped_and_counted(self) -> None:
        rows = [
            {"id": "2hop__ok", "question": "q?", "question_decomposition": [{"question": "s?"}]},
            {"id": "2hop__nosteps", "question": "q?", "question_decomposition": []},
            {"id": "2hop__noq", "question": "  ", "question_decomposition": [{"question": "s?"}]},
            {"question": "q?", "question_decomposition": [{"question": "s?"}]},
        ]
        examples, stats = _built(rows)
        self.assertEqual([ex.row_id for ex in examples], ["2hop__ok"])
        self.assertEqual(stats["dropped_missing_steps"], 1)
        self.assertEqual(stats["dropped_missing_question"], 1)
        self.assertEqual(stats["dropped_missing_id"], 1)

    def test_hop_disagreement_is_fatal_for_an_arm_that_filters_on_hops(self) -> None:
        """The generalisation arm's claim rests on the hop labels, so a mismatch is fatal."""
        cfg = load_config(CONFIG_NAME)
        arm = fd.resolve_arm(cfg, "generalisation_2_3hop")
        rows = _rows(("2hop__good", 2), ("3hop1__b", 3))
        # This row says 4-hop in its id and carries 2 steps: the signals disagree.
        rows += [
            {
                "id": "4hop1__liar",
                "question": "q?",
                "question_decomposition": [{"question": "s1?"}, {"question": "s2?"}],
            }
        ]
        with self.assertRaises(SystemExit) as ctx:
            fd.select_arm_examples(
                arm, rows, require(cfg, "data"), seed=42, max_reported_ids=MAX_LOAD_IDS
            )
        message = str(ctx.exception)
        self.assertIn("REFUSING TO TRAIN", message)
        self.assertIn("4hop1__liar", message)
        self.assertIn("train_hops=[2, 3]", message)

    def test_hop_disagreement_is_not_fatal_for_an_arm_without_a_hop_filter(self) -> None:
        cfg = load_config(CONFIG_NAME)
        arm = fd.resolve_arm(cfg, "full_train")
        rows = _rows(("2hop__good", 2))
        rows += [
            {
                "id": "4hop1__liar",
                "question": "q?",
                "question_decomposition": [{"question": "s1?"}, {"question": "s2?"}],
            }
        ]
        examples, record = fd.select_arm_examples(
            arm, rows, require(cfg, "data"), seed=42, max_reported_ids=MAX_LOAD_IDS
        )
        self.assertEqual(len(examples), 2)
        self.assertFalse(record["hop_disagreement_fatal"])
        self.assertEqual(record["source_rows"]["hop_disagreement_count"], 1)
        self.assertEqual(record["source_rows"]["hop_disagreement_ids"], ["4hop1__liar"])

    def test_the_reported_id_cap_comes_from_the_config(self) -> None:
        rows = [
            {
                "id": f"4hop1__liar{i}",
                "question": "q?",
                "question_decomposition": [{"question": "s?"}],
            }
            for i in range(5)
        ]
        _, stats = fd.build_examples(rows, DATA_CFG, max_reported_ids=2)
        self.assertEqual(stats["hop_disagreement_count"], 5)
        self.assertEqual(len(stats["hop_disagreement_ids"]), 2)
        self.assertEqual(stats["hop_disagreement_ids_capped_at"], 2)

    def test_hop_disagreement_is_recorded_not_hidden(self) -> None:
        rows = [{"id": "4hop1__x", "question": "q?", "question_decomposition": [{"question": "s?"}]}]
        examples, stats = _built(rows)
        self.assertEqual(stats["hop_disagreement_count"], 1)
        self.assertEqual(stats["hop_disagreement_ids"], ["4hop1__x"])
        self.assertEqual(examples[0].hop, 4)

    def test_selection_record_reports_what_it_did(self) -> None:
        cfg = load_config(CONFIG_NAME)
        arm = fd.resolve_arm(cfg, "generalisation_2_3hop")
        rows = _rows(("2hop__a", 2), ("3hop1__b", 3), ("4hop1__c", 4))
        examples, record = fd.select_arm_examples(
            arm, rows, require(cfg, "data"), seed=42, max_reported_ids=MAX_LOAD_IDS
        )
        self.assertEqual(record["train_hops"], [2, 3])
        self.assertEqual(record["num_after_hop_filter"], 2)
        self.assertEqual(record["num_selected"], 2)
        self.assertEqual(record["selected_hop_counts"], {"2": 1, "3": 1})
        self.assertEqual(record["source_rows"]["rows_in"], 3)
        self.assertEqual(len(examples), 2)


class TestPromptAndCompletion(unittest.TestCase):
    """The training string is the inference string, and the target parses back to steps."""

    def setUp(self) -> None:
        self.cfg = load_config(CONFIG_NAME)
        self.model_cfg = load_config(
            REPO_ROOT
            / "components"
            / "decomposer"
            / "models"
            / require(self.cfg, "model_folder")
            / "config.json"
        )
        self.model_dir = (
            REPO_ROOT / "components" / "decomposer" / "models" / require(self.cfg, "model_folder")
        )
        self.decomposer_cfg = load_config(require(self.cfg, "decomposer_config"))

    def _template(self, guided: bool) -> str:
        prompt_file = fd.select_prompt_file(self.model_cfg, guided=guided)
        return (self.model_dir / prompt_file).read_text(encoding="utf-8")

    def test_unguided_training_uses_the_unguided_prompt_file(self) -> None:
        self.assertEqual(
            fd.select_prompt_file(self.model_cfg, guided=False),
            require(self.model_cfg, "unguided_prompt_file"),
        )
        self.assertEqual(
            fd.select_prompt_file(self.model_cfg, guided=True),
            require(self.model_cfg, "prompt_file"),
        )

    def test_training_prompt_is_byte_identical_to_the_runners_prompt(self) -> None:
        """The runner's own fill_template, called the way run_decomposer.py calls it."""
        import run_decomposer

        template = self._template(guided=False)
        placeholder = require(self.decomposer_cfg, "unguided_hop_placeholder")
        prompt, is_placeholder = fd.build_prompt(
            template,
            prompt_style=require(self.model_cfg, "prompt_style"),
            question="Who leads the union?",
            hop=None,
            few_shot_examples="",
            unguided_hop_placeholder=placeholder,
        )
        expected = run_decomposer.fill_template(
            template,
            question="Who leads the union?",
            hop_count=None,
            few_shot_examples="",
            unguided_hop_placeholder=placeholder,
        )
        self.assertEqual(prompt, expected)
        self.assertFalse(is_placeholder)
        self.assertIn("Who leads the union?", prompt)
        self.assertTrue(prompt.rstrip().endswith("Decomposition:"))

    def test_zero_shot_training_prompt_contains_no_examples(self) -> None:
        rows, record = fd.build_training_rows(
            [fd.TrainingExample("2hop__a", "Who leads it?", ("Which union?", "Who leads #1?"), 2)],
            template=self._template(guided=False),
            prompt_style=require(self.model_cfg, "prompt_style"),
            prompt_cfg=require(self.cfg, "prompt"),
            unguided_hop_placeholder=require(self.decomposer_cfg, "unguided_hop_placeholder"),
        )
        self.assertFalse(record["few_shot_examples_in_prompt"])
        self.assertEqual(record["dataset_type"], "prompt_completion")
        self.assertEqual(record["num_rows"], 1)
        self.assertNotIn("Hop count:", rows[0]["prompt"])

    def test_completion_is_one_step_per_line_and_parses_back_to_the_steps(self) -> None:
        steps = ("Which union organised it?", "Who leads #1?")
        completion = fd.format_completion(steps, reference_style="as_is", number_lines=False)
        self.assertEqual(completion, "Which union organised it?\nWho leads #1?")
        # The evaluator has to be able to split the trained target back into steps, or the
        # fine-tuned arm cannot be scored on the same footing as the prompting arm.
        self.assertEqual(_decomp_to_steps(completion), list(steps))

    def test_numbered_target_lines_still_parse_back_to_the_steps(self) -> None:
        steps = ("Which union organised it?", "Who leads #1?")
        completion = fd.format_completion(steps, reference_style="as_is", number_lines=True)
        self.assertEqual(completion, "1. Which union organised it?\n2. Who leads #1?")
        self.assertEqual(_decomp_to_steps(completion), list(steps))

    def test_bracketed_reference_style_rewrites_bare_references_only(self) -> None:
        completion = fd.format_completion(
            ("Who leads #1?", "Where is [#2] based?"), reference_style="bracketed", number_lines=False
        )
        self.assertEqual(completion, "Who leads [#1]?\nWhere is [#2] based?")

    def test_unknown_reference_style_is_refused(self) -> None:
        with self.assertRaises(SystemExit):
            fd.format_completion(("a",), reference_style="nonsense", number_lines=False)


class _StubModel:
    """Stands in for a loaded transformers model: only its parameter count matters."""

    def __init__(self, count: int) -> None:
        self._count = count

    def num_parameters(self) -> int:
        return self._count


class TestModelSizeGateWiring(unittest.TestCase):
    """The trainer routes its base model through the committed ceiling, both ways."""

    def setUp(self) -> None:
        cfg = load_config(CONFIG_NAME)
        self.limits = load_limits(require(cfg, "model_limits_config"))
        self.ceiling = int(require(self.limits, "default_max_params"))

    def test_a_model_within_the_ceiling_is_asserted_and_recorded(self) -> None:
        record = train_lora.assert_base_model_within_ceiling(
            _StubModel(self.ceiling - 1), model_id="stub/within", limits=self.limits
        )
        self.assertTrue(record["ceiling_asserted"])
        self.assertEqual(record["component"], "decomposer")
        self.assertEqual(record["parameter_count"], self.ceiling - 1)
        self.assertEqual(record["parameter_ceiling"], self.ceiling)

    def test_a_model_over_the_ceiling_refuses_to_run(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            train_lora.assert_base_model_within_ceiling(
                _StubModel(self.ceiling + 1), model_id="stub/over", limits=self.limits
            )
        self.assertIn("REFUSING TO RUN", str(ctx.exception))

    def test_the_ceiling_is_the_committed_one(self) -> None:
        committed = json.loads(
            (REPO_ROOT / "configs" / "model_limits.json").read_text(encoding="utf-8")
        )
        self.assertEqual(self.ceiling, committed["default_max_params"])
        # 1e10 since ADR 0015 (Qwen3.5-9B admitted, 2026-08-19); reverts to 8e9 if
        # the supervisor reasserts the 8B ceiling.
        self.assertEqual(self.ceiling, 10_000_000_000)

    def test_trainable_parameter_record_counts_only_requires_grad(self) -> None:
        class _Param:
            def __init__(self, n: int, trainable: bool) -> None:
                self._n = n
                self.requires_grad = trainable

            def numel(self) -> int:
                return self._n

        class _Model:
            def parameters(self):
                return [_Param(90, False), _Param(10, True)]

        record = train_lora.trainable_parameter_record(_Model())
        self.assertEqual(record["trainable_parameters"], 10)
        self.assertEqual(record["total_parameters"], 100)
        self.assertAlmostEqual(record["trainable_percent"], 10.0)

    def test_the_total_is_the_model_size_count_not_a_raw_numel_sum(self) -> None:
        """Under 4-bit loading a numel sum halves the model; the record must not.

        ``transformers``' ``num_parameters`` counts a packed ``Params4bit`` as the parameters
        it stores, so ``src/model_size.py``'s ``count_parameters`` is the denominator that
        matches ``base_model_size.parameter_count`` in the same metrics JSON. This stub
        reports the two differently on purpose: 200 from ``num_parameters``, 100 from the
        parameter list.
        """

        class _Param:
            def __init__(self, n: int, trainable: bool) -> None:
                self._n = n
                self.requires_grad = trainable

            def numel(self) -> int:
                return self._n

        class _PackedModel:
            def num_parameters(self) -> int:
                return 200

            def parameters(self):
                return [_Param(90, False), _Param(10, True)]

        record = train_lora.trainable_parameter_record(_PackedModel())
        self.assertEqual(record["trainable_parameters"], 10)
        self.assertEqual(record["total_parameters"], 200)
        self.assertAlmostEqual(record["trainable_percent"], 5.0)
        self.assertIn("count_parameters", record["counting_note"])


def _trl_and_peft_available() -> bool:
    try:
        import peft  # noqa: F401
        import trl  # noqa: F401
    except ImportError:
        return False
    return True


@unittest.skipUnless(
    _trl_and_peft_available(), "peft/trl not installed (see requirements.txt)"
)
class TestTrainerConfigBuildsFromCommittedConfig(unittest.TestCase):
    """No GPU needed to catch a renamed trl/peft field - only a config that still maps.

    This is the cheap version of the failure it prevents: a training run that dies after
    the base model is loaded because ``warmup_ratio`` was removed or ``max_length`` renamed.
    """

    def setUp(self) -> None:
        self.cfg = load_config(CONFIG_NAME)

    def test_sft_config_accepts_every_training_key(self) -> None:
        args = train_lora.build_sft_config(
            require(self.cfg, "training"),
            output_dir=Path("/tmp/qav2-finetune-config-check"),
            seed=int(require(self.cfg, "seed")),
        )
        training = require(self.cfg, "training")
        self.assertEqual(args.max_length, training["max_length"])
        self.assertEqual(args.completion_only_loss, training["completion_only_loss"])
        self.assertEqual(args.seed, int(require(self.cfg, "seed")))
        self.assertEqual(args.data_seed, int(require(self.cfg, "seed")))
        self.assertEqual(args.gradient_accumulation_steps, training["gradient_accumulation_steps"])

    def test_lora_config_accepts_every_lora_key(self) -> None:
        lora = require(self.cfg, "lora")
        built = train_lora.build_lora_config(lora)
        self.assertEqual(built.r, lora["r"])
        self.assertEqual(built.lora_alpha, lora["lora_alpha"])
        self.assertEqual(sorted(built.target_modules), sorted(lora["target_modules"]))

    def test_prompt_completion_rows_load_as_a_dataset(self) -> None:
        from datasets import Dataset

        dataset = Dataset.from_list([{"prompt": "p", "completion": "c"}])
        self.assertEqual(sorted(dataset.column_names), ["completion", "prompt"])


class TestAdapterRequiresZeroShot(unittest.TestCase):
    """``--adapter`` without ``--no-few-shot`` is refused, not warned about."""

    def _check(self, **kwargs):
        import run_decomposer

        return run_decomposer.check_adapter_few_shot_combination(**kwargs)

    def test_adapter_without_no_few_shot_is_refused(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            self._check(adapter="runs/x/adapter", no_few_shot=False, override=False)
        message = str(ctx.exception)
        self.assertIn("REFUSING TO RUN", message)
        self.assertIn("--no-few-shot", message)
        self.assertIn("--adapter-with-few-shot-i-know", message)

    def test_adapter_with_no_few_shot_is_the_fine_tuned_arm(self) -> None:
        record = self._check(adapter="runs/x/adapter", no_few_shot=True, override=False)
        self.assertEqual(record["adapter"], "runs/x/adapter")
        self.assertTrue(record["no_few_shot"])
        self.assertFalse(record["adapter_few_shot_override"])

    def test_the_named_override_is_honoured_and_recorded(self) -> None:
        record = self._check(adapter="runs/x/adapter", no_few_shot=False, override=True)
        self.assertTrue(record["adapter_few_shot_override"])
        self.assertFalse(record["no_few_shot"])

    def test_the_prompting_arm_is_untouched(self) -> None:
        record = self._check(adapter=None, no_few_shot=False, override=False)
        self.assertIsNone(record["adapter"])

    def test_the_override_flag_is_on_the_parser(self) -> None:
        import run_decomposer

        self.assertEqual(
            run_decomposer.ADAPTER_FEW_SHOT_OVERRIDE_FLAG, "--adapter-with-few-shot-i-know"
        )


class TestEvalHopsAreEnforced(unittest.TestCase):
    """An arm's ``eval_hops`` decides which items may be scored, on every side."""

    def setUp(self) -> None:
        import compare_decomposer_arms

        self.mod = compare_decomposer_arms
        self.cfg = load_config(CONFIG_NAME)
        self.tmp = Path(__file__).resolve().parent / "_tmp_eval_hops"
        self.tmp.mkdir(exist_ok=True)

    def tearDown(self) -> None:
        for path in self.tmp.glob("*.json"):
            path.unlink()
        self.tmp.rmdir()

    def _write(self, name: str, payload: Any) -> Path:
        path = self.tmp / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def _rows(self, *ids: str) -> list[dict[str, Any]]:
        return [{"item_id": i, "step_f1": 1.0} for i in ids]

    def test_both_per_item_shapes_yield_the_ids(self) -> None:
        bare = self._write("bare.json", self._rows("2hop__a", "4hop1__b"))
        stamped = self._write(
            "stamped.json",
            {"schema": "musique_decomposition_per_item/1", "items": self._rows("2hop__a")},
        )
        self.assertEqual(self.mod.per_item_ids(bare), ["2hop__a", "4hop1__b"])
        self.assertEqual(self.mod.per_item_ids(stamped), ["2hop__a"])

    def test_items_inside_the_arms_eval_hops_pass_and_are_recorded(self) -> None:
        arm = fd.resolve_arm(self.cfg, "pool_2000")
        record = self.mod.assert_items_within_eval_hops(
            "prompting",
            self.tmp / "x.json",
            ["2hop__a", "3hop1__b", "4hop2__c"],
            require(arm, "eval_hops"),
            max_reported=20,
        )
        self.assertEqual(record["eval_hops"], [2, 3, 4])
        self.assertEqual(record["items_checked"], 3)
        self.assertEqual(record["hop_counts"], {"2": 1, "3": 1, "4": 1})
        self.assertTrue(record["asserted"])

    def test_a_generalisation_comparison_refuses_2_and_3_hop_items(self) -> None:
        arm = fd.resolve_arm(self.cfg, "generalisation_2_3hop")
        with self.assertRaises(SystemExit) as ctx:
            self.mod.assert_items_within_eval_hops(
                "finetuned_2_3hop",
                self.tmp / "x.json",
                ["4hop1__ok", "2hop__nope", "3hop1__nope"],
                require(arm, "eval_hops"),
                max_reported=20,
            )
        message = str(ctx.exception)
        self.assertIn("REFUSING TO COMPARE", message)
        self.assertIn("2hop__nope", message)
        self.assertIn("3hop1__nope", message)
        self.assertNotIn("4hop1__ok", message)

    def test_the_baseline_side_is_checked_too(self) -> None:
        """A 4-hop-only arm against a baseline scored on all 600 is not a comparison."""
        with self.assertRaises(SystemExit) as ctx:
            self.mod.assert_items_within_eval_hops(
                "prompting", self.tmp / "x.json", ["2hop__a"], [4], max_reported=20
            )
        self.assertIn("prompting", str(ctx.exception))

    def test_an_id_without_a_hop_prefix_is_fatal(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            self.mod.assert_items_within_eval_hops(
                "prompting", self.tmp / "x.json", ["no_hop_prefix"], [2, 3, 4], max_reported=20
            )
        message = str(ctx.exception)
        self.assertIn("no MuSiQue hop prefix", message)
        self.assertIn("no_hop_prefix", message)

    def test_no_scored_items_is_fatal_rather_than_a_vacuous_pass(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            self.mod.assert_items_within_eval_hops(
                "prompting", self.tmp / "x.json", [], [2, 3, 4], max_reported=20
            )
        self.assertIn("no scored items", str(ctx.exception))

    def test_the_offending_id_list_is_capped(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            self.mod.assert_items_within_eval_hops(
                "prompting",
                self.tmp / "x.json",
                [f"2hop__x{i}" for i in range(10)],
                [4],
                max_reported=3,
            )
        self.assertIn("(+7 more)", str(ctx.exception))

    def test_the_cap_is_a_config_value(self) -> None:
        self.assertEqual(int(require(self.cfg, "evaluation.max_reported_ids")), 20)


class TestCostReporting(unittest.TestCase):
    """Tokens and latency per query are reported, and never imputed when absent."""

    def test_cost_summary_averages_the_measured_rows(self) -> None:
        import run_decomposer

        rows = [
            {"prompt_tokens": 100, "completion_tokens": 10, "latency_seconds": 1.0},
            {"prompt_tokens": 200, "completion_tokens": 30, "latency_seconds": 3.0},
        ]
        summary = run_decomposer.cost_summary(rows)
        self.assertEqual(summary["rows_measured"], 2)
        self.assertAlmostEqual(summary["mean_prompt_tokens_per_query"], 150.0)
        self.assertAlmostEqual(summary["mean_completion_tokens_per_query"], 20.0)
        self.assertAlmostEqual(summary["mean_total_tokens_per_query"], 170.0)
        self.assertAlmostEqual(summary["mean_latency_seconds_per_query"], 2.0)
        self.assertAlmostEqual(summary["total_generation_seconds"], 4.0)

    def test_a_dry_run_reports_cost_as_unmeasured_not_zero(self) -> None:
        import run_decomposer

        rows = [{"prompt_tokens": None, "completion_tokens": None, "latency_seconds": None}]
        summary = run_decomposer.cost_summary(rows)
        self.assertEqual(summary["rows_measured"], 0)
        self.assertIsNone(summary["mean_prompt_tokens_per_query"])
        self.assertIsNone(summary["mean_latency_seconds_per_query"])
        self.assertIn("unmeasured", summary["note"])


#: The prompt an unguided zero-shot adapter was trained on. Every expectation in the
#: prompt-parity tests is computed against this by hand. ``prompt_sha256`` is deliberately
#: absent: these four fields are the ones every record must carry.
TRAINED_ON = {
    "guided": False,
    "prompt_file": "prompt_unguided.md",
    "prompt_style": "plain",
    "few_shot_examples_in_prompt": False,
}

#: The base model both the fabricated adapter_config.json and the fabricated training records
#: name, so the record-vs-adapter cross-check passes unless a test changes one of them.
FIXTURE_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

#: A fabricated prompt-file hash. Not a real sha256 of anything - the guard compares strings.
FIXTURE_PROMPT_SHA = "a" * 64


class TestAdapterPromptParity(unittest.TestCase):
    """An adapter may only be evaluated on the prompt its training run recorded.

    All fixtures here are fabricated JSON in a temp directory: no adapter weights, no real
    run directory, no data.
    """

    def setUp(self) -> None:
        import run_decomposer

        self.rd = run_decomposer
        self._tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self._tmp.name) / "20260818_222259"
        self.adapter_dir = self.run_dir / "adapter"
        self.adapter_dir.mkdir(parents=True)
        # Every adapter directory carries one; the record-vs-adapter cross-check reads it.
        self._write_adapter_config()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write_adapter_config(self, base_model: str = FIXTURE_MODEL_ID) -> Path:
        path = self.adapter_dir / self.rd.ADAPTER_CONFIG_FILE
        path.write_text(
            json.dumps({self.rd.ADAPTER_BASE_MODEL_FIELD: base_model}), encoding="utf-8"
        )
        return path

    def _write_provenance(
        self, *, payload_overrides: dict[str, Any] | None = None, **prompt_overrides: Any
    ) -> Path:
        """The file train_lora.py writes into the adapter directory."""
        prompt = {**TRAINED_ON, **prompt_overrides}
        payload = {
            "script": "train_lora.py",
            "run_id": "r1",
            "arm": "full_train",
            "model_id": FIXTURE_MODEL_ID,
            "prompt": prompt,
        }
        payload.update(payload_overrides or {})
        payload = {k: v for k, v in payload.items() if v is not _ABSENT}
        path = self.adapter_dir / self.rd.ADAPTER_PROVENANCE_FILE
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def _write_training_snapshot(self, *, script: str = "train_lora.py", **prompt_overrides: Any) -> Path:
        """The training run's config snapshot, next to the adapter dir (the exp-001 shape:
        ``prompt.few_shot_examples`` as an empty string rather than a boolean, and no
        ``prompt_sha256``)."""
        prompt = {
            "guided": TRAINED_ON["guided"],
            "few_shot_examples": "",
            "prompt_file": TRAINED_ON["prompt_file"],
            "prompt_style": TRAINED_ON["prompt_style"],
            **prompt_overrides,
        }
        path = self.run_dir / self.rd.TRAINING_SNAPSHOT_FILE
        path.write_text(
            json.dumps({"script": script, "run_id": "20260818_222259", "arm": "full_train",
                        "model_id": FIXTURE_MODEL_ID, "prompt": prompt}),
            encoding="utf-8",
        )
        return path

    def _check(self, *, run_selection: dict[str, Any], override: bool = False) -> dict[str, Any]:
        return self.rd.check_adapter_prompt_parity(
            adapter=str(self.adapter_dir), run_selection=run_selection, override=override
        )

    def _check_capturing_stdout(self, **kwargs: Any) -> tuple[dict[str, Any], str]:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            record = self._check(**kwargs)
        return record, buffer.getvalue()

    # ---- the record is found ----

    def test_the_provenance_file_in_the_adapter_dir_is_preferred(self) -> None:
        provenance = self._write_provenance()
        self._write_training_snapshot()
        record = self._check(run_selection=dict(TRAINED_ON))
        self.assertTrue(record["checked"])
        self.assertEqual(record["mismatches"], [])
        self.assertEqual(record["training_record"]["source"], "adapter_provenance")
        self.assertEqual(record["training_record"]["path"], str(provenance))
        self.assertEqual(record["trained_on"], TRAINED_ON)

    def test_the_training_run_snapshot_next_to_the_adapter_is_the_fallback(self) -> None:
        """The exp-001 case: an adapter trained before the provenance file existed."""
        snapshot = self._write_training_snapshot()
        record = self._check(run_selection=dict(TRAINED_ON))
        self.assertTrue(record["checked"])
        self.assertEqual(record["training_record"]["source"], "training_run_config")
        self.assertEqual(record["training_record"]["path"], str(snapshot))
        # few_shot_examples "" (the training block) -> few_shot_examples_in_prompt False.
        self.assertEqual(record["trained_on"], TRAINED_ON)

    def test_a_non_empty_few_shot_block_reads_as_trained_with_examples(self) -> None:
        self._write_training_snapshot(few_shot_examples="Q: ...\nSteps: ...")
        record = self._check(
            run_selection={**TRAINED_ON, "few_shot_examples_in_prompt": True}
        )
        self.assertTrue(record["checked"])
        self.assertTrue(record["trained_on"]["few_shot_examples_in_prompt"])

    # ---- the refusals ----

    def test_a_guided_run_against_an_unguided_adapter_is_refused(self) -> None:
        """The finding: --adapter --no-few-shot --guided used to be accepted."""
        self._write_provenance()
        with self.assertRaises(SystemExit) as ctx:
            self._check(
                run_selection={**TRAINED_ON, "guided": True, "prompt_file": "prompt.md"}
            )
        message = str(ctx.exception)
        self.assertIn("REFUSING TO RUN", message)
        self.assertIn("adapter prompt mismatch", message)
        self.assertIn("guided: trained on False, this run True", message)
        self.assertIn("prompt_file: trained on 'prompt_unguided.md', this run 'prompt.md'", message)
        self.assertIn(self.rd.ADAPTER_PROMPT_PARITY_OVERRIDE_FLAG, message)

    def test_exactly_the_mismatching_fields_are_named(self) -> None:
        self._write_provenance()
        with self.assertRaises(SystemExit):
            self._check(run_selection={**TRAINED_ON, "prompt_style": "chat_template"})
        record = self._check(
            run_selection={**TRAINED_ON, "prompt_style": "chat_template"}, override=True
        )
        self.assertEqual([m["field"] for m in record["mismatches"]], ["prompt_style"])
        self.assertEqual(record["mismatches"][0]["trained_on"], "plain")
        self.assertEqual(record["mismatches"][0]["this_run"], "chat_template")

    def test_a_few_shot_prompt_against_a_zero_shot_adapter_is_refused(self) -> None:
        self._write_provenance()
        with self.assertRaises(SystemExit) as ctx:
            self._check(run_selection={**TRAINED_ON, "few_shot_examples_in_prompt": True})
        self.assertIn("few_shot_examples_in_prompt", str(ctx.exception))

    def test_no_training_record_at_all_is_refused(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            self._check(run_selection=dict(TRAINED_ON))
        message = str(ctx.exception)
        self.assertIn("cannot tell what adapter", message)
        self.assertIn(self.rd.ADAPTER_PROVENANCE_FILE, message)
        self.assertIn(self.rd.TRAINING_SNAPSHOT_FILE, message)

    def test_a_config_json_from_another_script_is_not_a_training_record(self) -> None:
        self._write_training_snapshot(script="run_decomposer.py")
        with self.assertRaises(SystemExit) as ctx:
            self._check(run_selection=dict(TRAINED_ON))
        message = str(ctx.exception)
        self.assertIn("'script' is 'run_decomposer.py'", message)
        self.assertIn("not a record written by train_lora.py", message)

    def test_a_record_missing_a_prompt_field_is_not_usable(self) -> None:
        path = self.adapter_dir / self.rd.ADAPTER_PROVENANCE_FILE
        path.write_text(
            json.dumps({"script": "train_lora.py", "prompt": {"guided": False}}),
            encoding="utf-8",
        )
        with self.assertRaises(SystemExit) as ctx:
            self._check(run_selection=dict(TRAINED_ON))
        message = str(ctx.exception)
        self.assertIn("no 'prompt.prompt_file'", message)
        self.assertIn("no 'prompt.prompt_style'", message)

    # ---- the provenance file is validated, not trusted for sitting there ----

    def test_a_provenance_file_with_no_script_field_is_not_a_training_record(self) -> None:
        """The provenance file is a plain JSON file in a directory anyone can copy into."""
        self._write_provenance(payload_overrides={"script": _ABSENT})
        with self.assertRaises(SystemExit) as ctx:
            self._check(run_selection=dict(TRAINED_ON))
        message = str(ctx.exception)
        self.assertIn("'script' is None", message)
        self.assertIn("not a record written by train_lora.py", message)

    def test_a_provenance_file_claiming_another_script_is_refused(self) -> None:
        self._write_provenance(payload_overrides={"script": "run_decomposer.py"})
        with self.assertRaises(SystemExit) as ctx:
            self._check(run_selection=dict(TRAINED_ON))
        self.assertIn("'script' is 'run_decomposer.py'", str(ctx.exception))

    def test_a_provenance_file_for_a_different_base_model_is_refused(self) -> None:
        """A stale or copied provenance file must not pass because it sits beside weights."""
        self._write_provenance(payload_overrides={"model_id": "some/other-base"})
        with self.assertRaises(SystemExit) as ctx:
            self._check(run_selection=dict(TRAINED_ON))
        message = str(ctx.exception)
        self.assertIn("'model_id' is 'some/other-base'", message)
        self.assertIn(FIXTURE_MODEL_ID, message)
        self.assertIn("does not describe this adapter", message)

    def test_the_model_id_cross_check_is_recorded_when_it_passes(self) -> None:
        self._write_provenance()
        record = self._check(run_selection=dict(TRAINED_ON))
        cross = record["training_record"]["model_id_cross_check"]
        self.assertTrue(cross["checked"])
        self.assertEqual(cross["record_model_id"], FIXTURE_MODEL_ID)
        self.assertEqual(cross["adapter_base_model"], FIXTURE_MODEL_ID)

    def test_the_cross_check_says_so_when_it_cannot_be_done(self) -> None:
        """No adapter_config.json: the record is still usable, and the record says why."""
        (self.adapter_dir / self.rd.ADAPTER_CONFIG_FILE).unlink()
        self._write_provenance()
        record = self._check(run_selection=dict(TRAINED_ON))
        cross = record["training_record"]["model_id_cross_check"]
        self.assertTrue(record["checked"])
        self.assertFalse(cross["checked"])
        self.assertIsNone(cross["adapter_base_model"])
        self.assertIn("unchecked", cross["note"])

    def test_a_record_with_no_model_id_is_usable_and_says_it_was_not_cross_checked(self) -> None:
        self._write_provenance(payload_overrides={"model_id": _ABSENT})
        record = self._check(run_selection=dict(TRAINED_ON))
        self.assertTrue(record["checked"])
        cross = record["training_record"]["model_id_cross_check"]
        self.assertFalse(cross["checked"])
        self.assertIn("no 'model_id'", cross["note"])

    # ---- strict types: no coercion of a malformed record ----

    def test_a_string_guided_value_is_malformed_not_true(self) -> None:
        """bool("false") is True, which would invert the condition being claimed."""
        self._write_provenance(guided="false")
        with self.assertRaises(SystemExit) as ctx:
            self._check(run_selection=dict(TRAINED_ON))
        message = str(ctx.exception)
        self.assertIn("'prompt.guided' is 'false' (str), not a boolean", message)
        self.assertIn("cannot tell what adapter", message)

    def test_a_null_prompt_file_is_malformed_rather_than_the_string_none(self) -> None:
        self._write_provenance(prompt_file=None)
        with self.assertRaises(SystemExit) as ctx:
            self._check(run_selection=dict(TRAINED_ON))
        self.assertIn(
            "'prompt.prompt_file' is None (NoneType), not a non-empty string",
            str(ctx.exception),
        )

    def test_a_non_boolean_few_shot_flag_is_malformed(self) -> None:
        self._write_provenance(few_shot_examples_in_prompt="no")
        with self.assertRaises(SystemExit) as ctx:
            self._check(run_selection=dict(TRAINED_ON))
        self.assertIn(
            "'prompt.few_shot_examples_in_prompt' is 'no' (str), not a boolean",
            str(ctx.exception),
        )

    def test_an_absent_field_renders_as_null_not_as_the_string_none(self) -> None:
        self.assertEqual(self.rd._parity_value(None), "null (absent)")
        self.assertEqual(self.rd._parity_value("prompt.md"), "'prompt.md'")

    # ---- content-level parity: prompt_sha256, compared only when recorded ----

    def test_a_matching_prompt_sha256_is_compared_and_passes(self) -> None:
        self._write_provenance(prompt_sha256=FIXTURE_PROMPT_SHA)
        record = self._check(
            run_selection={**TRAINED_ON, "prompt_sha256": FIXTURE_PROMPT_SHA}
        )
        self.assertTrue(record["checked"])
        self.assertIn("prompt_sha256", record["fields_compared"])
        self.assertEqual(record["fields_compared"], list(TRAINED_ON) + ["prompt_sha256"])
        self.assertEqual(record["fields_not_in_training_record"], [])

    def test_an_edited_prompt_file_with_the_same_name_is_refused(self) -> None:
        """The point of the content check: same prompt_file, different bytes."""
        self._write_provenance(prompt_sha256=FIXTURE_PROMPT_SHA)
        with self.assertRaises(SystemExit) as ctx:
            self._check(run_selection={**TRAINED_ON, "prompt_sha256": "b" * 64})
        message = str(ctx.exception)
        self.assertIn("prompt_sha256: trained on", message)
        self.assertIn("adapter prompt mismatch", message)

    def test_a_record_without_a_prompt_sha256_is_not_refused_but_recorded(self) -> None:
        """The exp-001 case: retraining for a field added later is not acceptable."""
        self._write_training_snapshot()  # no prompt_sha256, like exp-001's snapshot
        record = self._check(
            run_selection={**TRAINED_ON, "prompt_sha256": FIXTURE_PROMPT_SHA}
        )
        self.assertTrue(record["checked"])
        self.assertEqual(record["mismatches"], [])
        self.assertEqual(record["fields_compared"], list(TRAINED_ON))
        self.assertEqual(record["fields_not_in_training_record"], ["prompt_sha256"])
        self.assertIn("not a refusal", record["note"])

    def test_a_malformed_prompt_sha256_is_refused_rather_than_skipped(self) -> None:
        self._write_provenance(prompt_sha256="")
        with self.assertRaises(SystemExit) as ctx:
            self._check(run_selection={**TRAINED_ON, "prompt_sha256": FIXTURE_PROMPT_SHA})
        self.assertIn("'prompt.prompt_sha256' is ''", str(ctx.exception))

    # ---- the override, and the prompting arm ----

    def test_the_named_override_runs_and_is_recorded(self) -> None:
        self._write_provenance()
        record = self._check(run_selection={**TRAINED_ON, "guided": True}, override=True)
        self.assertTrue(record["override"])
        # An overridden mismatch is a recorded mismatch, not a passed check.
        self.assertFalse(record["checked"])
        self.assertEqual([m["field"] for m in record["mismatches"]], ["guided"])
        self.assertIn("not the fine-tuned arm", record["note"])

    def test_the_override_also_covers_a_missing_record(self) -> None:
        record = self._check(run_selection=dict(TRAINED_ON), override=True)
        self.assertFalse(record["checked"])
        self.assertIsNone(record["trained_on"])
        self.assertIn("unchecked", record["note"])

    def test_a_run_without_an_adapter_is_not_checked(self) -> None:
        record = self.rd.check_adapter_prompt_parity(
            adapter=None, run_selection=dict(TRAINED_ON), override=False
        )
        self.assertFalse(record["checked"])
        self.assertIn("no --adapter", record["note"])

    def test_the_override_flag_is_named_and_on_the_parser(self) -> None:
        self.assertEqual(
            self.rd.ADAPTER_PROMPT_PARITY_OVERRIDE_FLAG, "--adapter-prompt-mismatch-i-know"
        )
        argv = [
            "run_decomposer.py", "--model", "mistral_7b_instruct", "--adapter", "a",
            "--no-few-shot", self.rd.ADAPTER_PROMPT_PARITY_OVERRIDE_FLAG,
        ]
        with unittest.mock.patch.object(sys, "argv", argv):
            args = self.rd._parse_args()
        self.assertTrue(args.adapter_prompt_mismatch_i_know)

    def test_the_parity_fields_are_the_four_prompt_choices(self) -> None:
        self.assertEqual(
            self.rd.ADAPTER_PROMPT_PARITY_FIELDS,
            ("guided", "prompt_file", "prompt_style", "few_shot_examples_in_prompt"),
        )
        self.assertEqual(self.rd.ADAPTER_PROMPT_PARITY_OPTIONAL_FIELDS, ("prompt_sha256",))

    # ---- a run that proceeds must not print a refusal ----

    def test_an_overridden_mismatch_warns_without_claiming_a_refusal(self) -> None:
        self._write_provenance()
        record, printed = self._check_capturing_stdout(
            run_selection={**TRAINED_ON, "guided": True}, override=True
        )
        self.assertNotIn("REFUSING TO RUN", printed)
        self.assertIn("WARNING:", printed)
        self.assertIn("MISMATCHED", printed)
        self.assertIn("The run proceeds", printed)
        self.assertEqual([m["field"] for m in record["mismatches"]], ["guided"])

    def test_an_overridden_missing_record_warns_without_claiming_a_refusal(self) -> None:
        _record, printed = self._check_capturing_stdout(
            run_selection=dict(TRAINED_ON), override=True
        )
        self.assertNotIn("REFUSING TO RUN", printed)
        self.assertIn("UNCHECKED", printed)
        self.assertIn("The run proceeds", printed)

    def test_the_refusals_still_say_refusing_to_run(self) -> None:
        """The string is reserved for runs that actually stop."""
        self._write_provenance()
        with self.assertRaises(SystemExit) as ctx:
            self._check(run_selection={**TRAINED_ON, "guided": True})
        self.assertIn("REFUSING TO RUN", str(ctx.exception))


class TestAdapterRunNoteStatesWhatWasChecked(unittest.TestCase):
    """notes.md says whether the two adapter guards passed, were overridden, or did not run."""

    def setUp(self) -> None:
        import run_decomposer

        self.rd = run_decomposer

    def _lines(self, **kwargs: Any) -> list[str]:
        defaults: dict[str, Any] = {
            "adapter": "runs/exp-001/x/adapter",
            "no_few_shot": True,
            "parity": {"checked": True, "override": False, "mismatches": []},
            "base_model": None,
            "base_model_note": None,
        }
        return self.rd.adapter_note_lines(**{**defaults, **kwargs})

    def test_the_prompting_arm_gets_one_line_and_no_guard_lines(self) -> None:
        lines = self._lines(adapter=None, no_few_shot=False)
        self.assertEqual(lines, ["- Adapter: none (prompting arm)"])

    def test_a_passed_parity_check_names_the_fields_and_the_record(self) -> None:
        lines = self._lines(
            parity={
                "checked": True,
                "override": False,
                "mismatches": [],
                "fields_compared": list(TRAINED_ON) + ["prompt_sha256"],
                "fields_not_in_training_record": [],
                "training_record": {"path": "/runs/x/config.json", "source": "training_run_config"},
            },
            base_model={
                "adapter_base_model": FIXTURE_MODEL_ID,
                "run_model_id": FIXTURE_MODEL_ID,
            },
        )
        parity_line = next(ln for ln in lines if "prompt parity" in ln)
        self.assertIn("checked OK on 5 field(s)", parity_line)
        self.assertIn("prompt_sha256", parity_line)
        self.assertIn("/runs/x/config.json", parity_line)
        base_line = next(ln for ln in lines if "base model" in ln)
        self.assertIn("asserted", base_line)
        self.assertIn(FIXTURE_MODEL_ID, base_line)

    def test_a_field_the_record_lacks_is_reported_as_not_compared(self) -> None:
        lines = self._lines(
            parity={
                "checked": True,
                "override": False,
                "mismatches": [],
                "fields_compared": list(TRAINED_ON),
                "fields_not_in_training_record": ["prompt_sha256"],
                "training_record": {"path": "/runs/x/config.json", "source": "training_run_config"},
            },
            base_model_note="unasserted: no model was loaded in this run (--dry-run)",
        )
        parity_line = next(ln for ln in lines if "prompt parity" in ln)
        self.assertIn("checked OK on 4 field(s)", parity_line)
        self.assertIn("not compared: prompt_sha256", parity_line)
        base_line = next(ln for ln in lines if "base model" in ln)
        self.assertIn("unasserted", base_line)
        self.assertIn("--dry-run", base_line)

    def test_an_overridden_mismatch_is_stated_with_its_mismatches(self) -> None:
        lines = self._lines(
            parity={
                "checked": False,
                "override": True,
                "mismatches": [
                    {"field": "guided", "trained_on": False, "this_run": True},
                    {"field": "prompt_file", "trained_on": "prompt_unguided.md", "this_run": None},
                ],
            }
        )
        parity_line = next(ln for ln in lines if "prompt parity" in ln)
        self.assertIn("MISMATCHED and overridden", parity_line)
        self.assertIn(self.rd.ADAPTER_PROMPT_PARITY_OVERRIDE_FLAG, parity_line)
        self.assertIn("guided (trained on False, this run True)", parity_line)
        self.assertIn("this run null (absent)", parity_line)
        self.assertIn("not the fine-tuned arm", parity_line)

    def test_an_overridden_missing_record_is_stated_as_unchecked(self) -> None:
        lines = self._lines(parity={"checked": False, "override": True, "mismatches": []})
        parity_line = next(ln for ln in lines if "prompt parity" in ln)
        self.assertIn("UNCHECKED and overridden", parity_line)
        self.assertIn("no usable training record", parity_line)

    def test_the_few_shot_override_line_is_kept(self) -> None:
        lines = self._lines(no_few_shot=False)
        self.assertIn(self.rd.ADAPTER_FEW_SHOT_OVERRIDE_FLAG, lines[0])
        self.assertIn("not the fine-tuned arm", lines[0])


class TestAdapterBaseModelAssertion(unittest.TestCase):
    """An adapter attaches by module name and shape, so its base model is asserted."""

    def setUp(self) -> None:
        import run_decomposer

        self.rd = run_decomposer
        self._tmp = tempfile.TemporaryDirectory()
        self.adapter_dir = Path(self._tmp.name) / "adapter"
        self.adapter_dir.mkdir()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write_adapter_config(self, payload: dict[str, Any]) -> Path:
        path = self.adapter_dir / self.rd.ADAPTER_CONFIG_FILE
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_the_matching_base_model_is_asserted_and_recorded(self) -> None:
        path = self._write_adapter_config(
            {"base_model_name_or_path": "mistralai/Mistral-7B-Instruct-v0.3"}
        )
        record = self.rd.assert_adapter_base_model(
            self.adapter_dir, model_id="mistralai/Mistral-7B-Instruct-v0.3"
        )
        self.assertTrue(record["base_model_asserted"])
        self.assertEqual(record["adapter_base_model"], "mistralai/Mistral-7B-Instruct-v0.3")
        self.assertEqual(record["run_model_id"], "mistralai/Mistral-7B-Instruct-v0.3")
        self.assertEqual(record["adapter_config_path"], str(path))

    def test_a_different_base_model_is_refused_and_both_are_named(self) -> None:
        self._write_adapter_config(
            {"base_model_name_or_path": "mistralai/Mistral-7B-Instruct-v0.2"}
        )
        with self.assertRaises(SystemExit) as ctx:
            self.rd.assert_adapter_base_model(
                self.adapter_dir, model_id="mistralai/Mistral-7B-Instruct-v0.3"
            )
        message = str(ctx.exception)
        self.assertIn("REFUSING TO RUN", message)
        self.assertIn("mistralai/Mistral-7B-Instruct-v0.2", message)
        self.assertIn("mistralai/Mistral-7B-Instruct-v0.3", message)

    def test_an_absent_field_is_refused_rather_than_skipped(self) -> None:
        self._write_adapter_config({"peft_type": "LORA"})
        with self.assertRaises(SystemExit) as ctx:
            self.rd.assert_adapter_base_model(self.adapter_dir, model_id="m")
        self.assertIn("base_model_name_or_path", str(ctx.exception))

    def test_a_missing_adapter_config_is_refused(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            self.rd.assert_adapter_base_model(self.adapter_dir, model_id="m")
        self.assertIn(self.rd.ADAPTER_CONFIG_FILE, str(ctx.exception))

    def test_attach_adapter_asserts_before_loading_anything(self) -> None:
        """The guard is wired into attach_adapter, not only available beside it."""
        self._write_adapter_config({"base_model_name_or_path": "some/other-base"})
        with self.assertRaises(SystemExit) as ctx:
            self.rd.attach_adapter(object(), self.adapter_dir, model_id="m")
        self.assertIn("adapter base-model mismatch", str(ctx.exception))

    def test_the_field_name_is_pefts_own(self) -> None:
        self.assertEqual(self.rd.ADAPTER_CONFIG_FILE, "adapter_config.json")
        self.assertEqual(self.rd.ADAPTER_BASE_MODEL_FIELD, "base_model_name_or_path")

    def test_the_committed_fixture_adapter_names_the_model_folders_base(self) -> None:
        """The smoke test's --adapter fixture must match the model folder it runs against."""
        fixture = json.loads(
            (REPO_ROOT / "tests" / "fixtures" / "adapter" / self.rd.ADAPTER_CONFIG_FILE)
            .read_text(encoding="utf-8")
        )
        model_cfg = load_config(
            REPO_ROOT / "components" / "decomposer" / "models" / "mistral_7b_instruct"
            / "config.json"
        )
        self.assertEqual(
            fixture[self.rd.ADAPTER_BASE_MODEL_FIELD], require(model_cfg, "model_id")
        )


class TestAdapterProvenanceIsWritten(unittest.TestCase):
    """``train_lora.py`` records the training prompt where the evaluation guard reads it."""

    def setUp(self) -> None:
        import run_decomposer

        self.rd = run_decomposer
        self._tmp = tempfile.TemporaryDirectory()
        self.adapter_dir = Path(self._tmp.name) / "adapter"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _snapshot(self, **prompt_overrides: Any) -> dict[str, Any]:
        return {
            "script": "train_lora.py",
            "created_utc": "2026-08-19T00:00:00+00:00",
            "run_id": "20260819_000000",
            "arm": "full_train",
            "model": "mistral_7b_instruct",
            "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
            "prompt": {
                "guided": False,
                "few_shot_examples": "",
                "prompt_file": "prompt_unguided.md",
                "prompt_style": "plain",
                **prompt_overrides,
            },
        }

    def test_a_zero_shot_training_prompt_records_no_examples(self) -> None:
        payload = train_lora.adapter_provenance(self._snapshot())
        self.assertFalse(payload["prompt"]["few_shot_examples_in_prompt"])
        self.assertEqual(payload["run_id"], "20260819_000000")
        self.assertEqual(payload["model_id"], "mistralai/Mistral-7B-Instruct-v0.3")

    def test_a_non_empty_few_shot_block_records_examples(self) -> None:
        payload = train_lora.adapter_provenance(self._snapshot(few_shot_examples="Q: x"))
        self.assertTrue(payload["prompt"]["few_shot_examples_in_prompt"])

    def test_what_training_writes_is_what_the_guard_reads(self) -> None:
        path = train_lora.write_adapter_provenance(self.adapter_dir, self._snapshot())
        self.assertEqual(path.name, self.rd.ADAPTER_PROVENANCE_FILE)
        found = self.rd.read_adapter_training_record(self.adapter_dir)
        self.assertTrue(found["found"])
        self.assertEqual(found["source"], "adapter_provenance")
        self.assertEqual(found["trained_on"], TRAINED_ON)
        # And the run that renders that same prompt passes the guard.
        record = self.rd.check_adapter_prompt_parity(
            adapter=str(self.adapter_dir), run_selection=dict(TRAINED_ON), override=False
        )
        self.assertTrue(record["checked"])
        self.assertEqual(record["mismatches"], [])

    def test_the_two_modules_agree_on_the_filename(self) -> None:
        self.assertEqual(train_lora.ADAPTER_PROVENANCE_FILE, self.rd.ADAPTER_PROVENANCE_FILE)

    def test_the_prompt_content_hash_travels_into_the_provenance(self) -> None:
        payload = train_lora.adapter_provenance(
            self._snapshot(prompt_sha256=FIXTURE_PROMPT_SHA)
        )
        self.assertEqual(payload["prompt"]["prompt_sha256"], FIXTURE_PROMPT_SHA)
        train_lora.write_adapter_provenance(
            self.adapter_dir, self._snapshot(prompt_sha256=FIXTURE_PROMPT_SHA)
        )
        found = self.rd.read_adapter_training_record(self.adapter_dir)
        self.assertEqual(found["trained_on"]["prompt_sha256"], FIXTURE_PROMPT_SHA)

    def test_training_and_inference_hash_a_prompt_file_with_one_function(self) -> None:
        """Content parity is only meaningful if both sides compute the same number."""
        prompt_path = (
            REPO_ROOT / "components" / "decomposer" / "models" / "mistral_7b_instruct"
            / "prompt_unguided.md"
        )
        self.assertTrue(prompt_path.exists(), str(prompt_path))
        self.assertEqual(
            fd.prompt_file_sha256(prompt_path), self.rd.sha256_file(prompt_path)
        )

    def test_the_committed_fixture_provenance_is_readable_by_the_guard(self) -> None:
        found = self.rd.read_adapter_training_record(REPO_ROOT / "tests" / "fixtures" / "adapter")
        self.assertTrue(found["found"])
        self.assertEqual(found["trained_on"], TRAINED_ON)


class TestCostColumnDefinitionsAreReported(unittest.TestCase):
    """The arms note prints what its cost columns measure, taken from the runs themselves."""

    def setUp(self) -> None:
        import compare_decomposer_arms

        self.mod = compare_decomposer_arms
        self.definitions = {"prompt_tokens": "tokens in the rendered prompt"}

    def test_arms_that_agree_are_one_group(self) -> None:
        groups = self.mod.cost_definition_groups(
            {
                "prompting": {"cost_definitions": dict(self.definitions)},
                "finetuned": {"cost_definitions": dict(self.definitions)},
            }
        )
        self.assertEqual(len(groups), 1)
        self.assertEqual(sorted(groups[0][0]), ["finetuned", "prompting"])
        self.assertEqual(groups[0][1], self.definitions)

    def test_arms_that_disagree_are_separate_groups(self) -> None:
        groups = self.mod.cost_definition_groups(
            {
                "prompting": {"cost_definitions": dict(self.definitions)},
                "finetuned": {"cost_definitions": {"prompt_tokens": "something else"}},
            }
        )
        self.assertEqual(len(groups), 2)

    def test_an_arm_without_definitions_is_left_out_rather_than_invented(self) -> None:
        groups = self.mod.cost_definition_groups(
            {
                "prompting": {"cost_definitions": dict(self.definitions)},
                "finetuned": {"cost_definitions": None},
            }
        )
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0][0], ["prompting"])

    def test_the_runner_writes_the_definitions_the_note_reports(self) -> None:
        import run_decomposer

        summary = run_decomposer.cost_summary(
            [{"prompt_tokens": 10, "completion_tokens": 2, "latency_seconds": 0.5}]
        )
        for key in ("prompt_tokens", "completion_tokens", "latency_seconds"):
            self.assertIn(key, summary["definitions"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
