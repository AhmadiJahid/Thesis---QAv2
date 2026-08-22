#!/usr/bin/env python3
"""Checks for the MuSiQue decomposer conditions (issue #12). No GPU, no model weights.

This is a plain script: every check is an assertion with a hand-computed expectation, and the
exit code is the result. It is also importable by pytest, so the harness functions that take
arguments are named ``check_*`` rather than ``test_*``: pytest reads a parameter on a
``test_*`` function as a request for a fixture, and those five functions turned
``pytest tests/`` into five collection errors (PR #21 review, M-3). ``main()`` below is the
order they run in.

What it covers:

1. **Config resolution** - ``configs/decomposer_musique.json`` resolves to the MuSiQue
   question template, hops [2, 3, 4] and the conditions (``unguided``, ``oracle_guided``,
   ``unguided_capped``, and issue #27's ``router_guided``) whose only differences are the hop
   information in the prompt and the step-line budget. The end-to-end checks below cover the
   three arms of issue #12; ``router_guided`` needs a predictions file and is covered end to
   end in ``tests/test_router_predictions.py``.
2. **MetaQA defaults unchanged** - ``configs/decomposer.json`` still has guided=false,
   hops [1, 2, 3], the MetaQA template and no conditions block.
3. **The prompt invariant** - every model folder that ships an unguided prompt has one that
   equals its guided prompt minus the hop-bearing lines, byte for byte. A tampered prompt
   (a rule added, a rule dropped) fails the guard.
4. **The step-line stopping rule** - the shared step normalization in ``src/step_lines.py``
   (the same function the evaluator scores with), ``trim_to_step_lines`` and
   ``StepLineStopper`` against synthetic token streams with hand-computed expectations,
   plus the transformers ``StoppingCriteria`` adapter driven by a fake tokenizer, including
   a ``<think>`` preamble that must not consume the budget.
5. **Self-exclusion** - a retrieved exemplar that is the query itself is dropped on both
   the reranked and the bi-encoder path, and the top-k is still k.
6. **The three arms end to end** - the real runner is executed in ``--dry-run`` against
   the fabricated fixtures once per condition, and the *prompts it wrote* are compared:
   unguided carries no hop count at all, oracle_guided carries the gold hop count of the
   row's id, unguided_capped builds byte-identical prompts to unguided and carries the
   configured cap. The guided prompts equal the unguided ones once hop-bearing lines are
   removed - the mechanical version of "the arms differ in hop information only". Model,
   seed, decoding, retrieval (path *and* sha256) and the question ids must be identical
   across all three. Writes to a temp dir and deletes it.
7. **Refusals** - a model folder without an unguided prompt, a row count that is not the
   pinned per-hop set, and a query id whose hop depth cannot be parsed are each refused
   with a non-zero exit. ``resolve_retrieval_input`` is checked directly for its resolution
   order (``--retrieval-input`` > ``retrieval.input`` > ``retrieval.input_key`` through the
   paths config, ADR 0014) and for its two refusals: both config fields set, and no
   retrieval input at all under ``retrieval.require_input``.
8. **The evaluation set resolves** - the three pinned files of ADR 0007 exist under
   ``data_root`` with ``eval_rows_per_hop`` rows each, and the retrieval artifact behind
   ``retrieval.input_key`` exists with one row per evaluation question. Skipped with
   ``--skip-data-checks`` (the smoke test runs against fabricated fixtures, which are
   neither that set nor that artifact).

Usage::

    python tests/test_decomposer_conditions.py
    python tests/test_decomposer_conditions.py --skip-data-checks
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "components" / "decomposer"))

from run_config import PATHS_CONFIG_ENV, load_config, load_paths, require, resolve_path  # noqa: E402
from step_lines import split_step_lines  # noqa: E402

import run_decomposer as rd  # noqa: E402

CHECKS: list[str] = []

#: The end-to-end checks run the real runner in --dry-run against the fabricated fixtures
#: (3 rows per hop for hops 2/3/4). mistral_7b_instruct is the model folder because it
#: ships an unguided prompt, which the unguided arms require.
E2E_MODEL = "mistral_7b_instruct"
E2E_ROWS = 9
#: configs/decomposer_musique.json points retrieval.input_key at the real v1 artifact (ADR
#: 0014), which is not in the fixture tree, so the fixture runs pass --retrieval-input (it
#: takes precedence); it holds one row per fixture question (9), not the pinned 600, which
#: is why they also pass --allow-unpinned-eval-set.
E2E_RETRIEVAL = REPO_ROOT / "tests" / "fixtures" / "retrieval" / "top5_musique_conditions.jsonl"
MODELS_DIR = REPO_ROOT / "components" / "decomposer" / "models"


def check(name: str, condition: bool, detail: str = "") -> None:
    CHECKS.append(name)
    if not condition:
        raise AssertionError(f"{name} FAILED {detail}".rstrip())
    print(f"  ok  {name}" + (f"  [{detail}]" if detail else ""))


# --------------------------------------------------------------- config resolution


def test_metaqa_defaults_unchanged() -> None:
    print("[metaqa defaults]")
    cfg = load_config("decomposer.json")
    check("metaqa guided is false", cfg["guided"] is False, repr(cfg["guided"]))
    check("metaqa hops are [1, 2, 3]", cfg["hops"] == [1, 2, 3], repr(cfg["hops"]))
    check(
        "metaqa questions template key",
        cfg["questions_template_key"] == "metaqa_questions_template",
        cfg["questions_template_key"],
    )
    check("metaqa questions format is lines", cfg["questions_format"] == "lines")
    check("metaqa config has no conditions block", "conditions" not in cfg)
    check("metaqa config has no generation overrides", "generation_overrides" not in cfg)
    check("metaqa seed is 42", cfg["seed"] == 42, repr(cfg["seed"]))
    # Jahid's 2026-08-19 exemplar-hop decision was taken for the MuSiQue conditions; the
    # MetaQA path keeps v1's behaviour, so this config must still say "query".
    check(
        "metaqa exemplar hop lines still state the query's hop count (v1)",
        cfg["few_shot_exemplar_hop_count"] == "query",
        repr(cfg["few_shot_exemplar_hop_count"]),
    )


def test_musique_config() -> dict:
    print("[musique config]")
    cfg = load_config("decomposer_musique.json")
    check("musique hops are [2, 3, 4]", cfg["hops"] == [2, 3, 4], repr(cfg["hops"]))
    check(
        "musique questions template key",
        cfg["questions_template_key"] == "musique_eval_questions_template",
        cfg["questions_template_key"],
    )
    check("musique questions format is jsonl", cfg["questions_format"] == "jsonl")
    check(
        "musique jsonl fields are id/question",
        require(cfg, "questions_jsonl.question_field") == "question"
        and require(cfg, "questions_jsonl.id_field") == "id",
    )
    check("musique sample_size is null", cfg["sample_size"] is None)
    check("musique seed matches metaqa seed 42", cfg["seed"] == 42, repr(cfg["seed"]))
    check("eval_rows_per_hop is 200", cfg["eval_rows_per_hop"] == 200)
    check(
        "musique exemplar hop lines state the exemplar's own gold hop count",
        cfg["few_shot_exemplar_hop_count"] == "exemplar_gold",
        repr(cfg["few_shot_exemplar_hop_count"]),
    )
    check(
        "max_new_tokens override is present",
        require(cfg, "generation_overrides.max_new_tokens") == 1024,
    )
    return cfg


def check_conditions(cfg: dict) -> None:
    print("[conditions]")
    conditions = require(cfg, "conditions")
    check(
        "the three conditions briefed for issue #12, plus issue #27's router_guided",
        sorted(conditions) == [
            "oracle_guided", "router_guided", "unguided", "unguided_capped"
        ],
        str(sorted(conditions)),
    )

    # Hand-computed expectation per arm: (guided, stop_after_step_lines, hop_source).
    # hop_source is the arm's own key when it sets one, else the config's default; it is what
    # separates oracle_guided (the gold depth) from router_guided (a router's prediction).
    expected = {
        "unguided": (False, None, "gold"),
        "oracle_guided": (True, None, "gold"),
        "unguided_capped": (False, 8, "gold"),
        "router_guided": (True, None, "predictions"),
    }
    for name, (want_guided, want_cap, want_hop_source) in expected.items():
        got_name, block = rd.resolve_condition(cfg, name)
        check(f"condition {name} resolves", got_name == name)
        guided = bool(block["guided"])
        cap = block.get("stop_after_step_lines")
        hop_source = block.get("hop_source") or require(cfg, "hop_source")
        check(f"condition {name} guided={want_guided}", guided is want_guided, repr(guided))
        check(f"condition {name} cap={want_cap}", cap == want_cap, repr(cap))
        check(
            f"condition {name} hop_source={want_hop_source}",
            hop_source == want_hop_source,
            repr(hop_source),
        )
    check("the config's default hop source is gold", require(cfg, "hop_source") == "gold")

    default_name, _ = rd.resolve_condition(cfg, None)
    check("default condition is unguided", default_name == "unguided", str(default_name))

    # No condition may move the model, the seed or decoding: the runner rejects any other key.
    for name, block in conditions.items():
        stray = sorted(set(block) - set(rd._CONDITION_KEYS))
        check(f"condition {name} sets no decoding/model/seed key", not stray, str(stray))

    for bad, why in (
        ("does_not_exist", "unknown condition name"),
        (None, "no name and no default"),
    ):
        broken = dict(cfg)
        if bad is None:
            broken.pop("condition")
        try:
            rd.resolve_condition(broken, bad)
        except SystemExit:
            check(f"rejects {why}", True)
        else:
            check(f"rejects {why}", False)

    drifting = {"_config_path": "<test>", "condition": "x", "conditions": {"x": {"seed": 7}}}
    try:
        rd.resolve_condition(drifting, "x")
    except SystemExit:
        check("rejects a condition that sets seed", True)
    else:
        check("rejects a condition that sets seed", False)

    # generation_overrides apply to every arm identically, and may only touch keys the
    # model's generation block already defines.
    model_generation = {"max_new_tokens": 128, "do_sample": False, "temperature": 0.0, "top_p": 1.0}
    merged = rd.apply_generation_overrides(
        model_generation, require(cfg, "generation_overrides"), "<test>"
    )
    check("override raises max_new_tokens to 1024", merged["max_new_tokens"] == 1024)
    check(
        "override leaves the rest of decoding alone",
        all(merged[k] == model_generation[k] for k in ("do_sample", "temperature", "top_p")),
    )
    try:
        rd.apply_generation_overrides(model_generation, {"beam_search": 4}, "<test>")
    except SystemExit:
        check("rejects an override key the model does not define", True)
    else:
        check("rejects an override key the model does not define", False)


def check_prompt_invariant(cfg: dict) -> None:
    """The unguided prompt of every folder that has one == guided minus hop-bearing lines.

    This is the mechanical form of Jahid's design (plan prompt 4): three conditions on the
    same set, "everything else held identical", unguided = "no hop count in the prompt". A
    residual delta - v1's mistral prompt added "Decompose into the minimal number of atomic
    steps.", v1's qwen3_5_9b prompt dropped the whole 7-rule block - is a second difference
    between the arms, so it fails here.
    """
    print("[prompt invariant]")
    check(
        "the musique config demands the invariant",
        require(cfg, "unguided_prompt_must_equal_guided_minus_hop_lines") is True,
    )
    folders = sorted(
        d.name
        for d in MODELS_DIR.iterdir()
        if d.is_dir() and load_config(d / "config.json").get("unguided_prompt_file")
    )
    check(
        "folders shipping an unguided prompt",
        folders == ["mistral_7b_instruct", "qwen3_5_9b"],
        str(folders),
    )
    for folder in folders:
        model_cfg = load_config(MODELS_DIR / folder / "config.json")
        guided_path = MODELS_DIR / folder / model_cfg["prompt_file"]
        unguided_path = MODELS_DIR / folder / model_cfg["unguided_prompt_file"]
        guided = guided_path.read_bytes().decode("utf-8")
        unguided = unguided_path.read_bytes().decode("utf-8")
        removed = rd.hop_bearing_lines(guided)
        check(f"{folder}: guided prompt has hop lines", len(removed) == 2, str(removed))
        check(
            f"{folder}: unguided == guided minus hop lines (bytes)",
            unguided == rd.derive_unguided_template(guided),
        )
        check(f"{folder}: unguided prompt has no hop line", not rd.hop_bearing_lines(unguided))
        record = rd.assert_unguided_is_guided_minus_hop_lines(
            guided_template=guided,
            unguided_template=unguided,
            guided_path=guided_path,
            unguided_path=unguided_path,
            model=folder,
            config_src="<test>",
        )
        check(f"{folder}: guard passes and records the removed lines", record["checked"] is True)
        # Every other line survives: same line count minus the hop lines, in order.
        kept_guided = [ln for ln in guided.splitlines() if not rd._HOP_LINE_RX.search(ln)]
        check(
            f"{folder}: every non-hop line is preserved in order",
            kept_guided == unguided.splitlines(),
        )
        # ... and a tampered unguided prompt is caught, in both directions.
        for tampered, why in (
            (unguided + "- Decompose into the minimal number of atomic steps.\n", "an added rule"),
            ("\n".join(unguided.splitlines()[:-1]) + "\n", "a dropped line"),
        ):
            try:
                rd.assert_unguided_is_guided_minus_hop_lines(
                    guided_template=guided,
                    unguided_template=tampered,
                    guided_path=guided_path,
                    unguided_path=unguided_path,
                    model=folder,
                    config_src="<test>",
                )
            except SystemExit:
                check(f"{folder}: guard rejects {why}", True)
            else:
                check(f"{folder}: guard rejects {why}", False)
        check(f"{folder}: no guided line mixes hop prose with another instruction",
              not rd.mixed_hop_lines(guided), str(rd.mixed_hop_lines(guided)))


def test_compound_hop_lines_are_refused() -> None:
    """A guided line mixing a hop rule with another instruction cannot be derived from.

    The derivation drops WHOLE lines, so such a line would quietly take its non-hop
    instruction out of the unguided prompt - and the byte-equality guard could not see it,
    because it compares against that same faulty derivation.
    """
    print("[compound hop lines]")
    ok_lines = [
        "- The number of steps MUST equal the hop count.",
        "Hop count: {hop_count}",
        "Hop count: 3",
    ]
    for line in ok_lines:
        check(f"accepted, hop-only line {line!r}", not rd.mixed_hop_lines(line))
    bad_lines = [
        "- The number of steps MUST equal the hop count. Output ONLY the steps.",
        "- Use the hop count and output ONLY the steps.",
        "Hop count: {hop_count}, and do NOT explain",
    ]
    for line in bad_lines:
        check(f"flagged, compound line {line!r}", rd.mixed_hop_lines(line) == [line])

    # The full guard refuses such a guided prompt, telling the editor to split the line.
    guided = (
        "You decompose questions into atomic steps.\n"
        "Rules:\n"
        "- The number of steps MUST equal the hop count. Output ONLY the steps.\n"
        "Question: {question}\n"
    )
    try:
        rd.assert_unguided_is_guided_minus_hop_lines(
            guided_template=guided,
            unguided_template=rd.derive_unguided_template(guided),
            guided_path=Path("<guided>"),
            unguided_path=Path("<unguided>"),
            model="m",
            config_src="<test>",
        )
    except SystemExit as exc:
        check(
            "the invariant guard refuses a compound guided line",
            "mix a hop-count reference" in str(exc) and "Split the line" in str(exc),
        )
    else:
        check("the invariant guard refuses a compound guided line", False)
    # ... and note what the old rule would have done: the instruction just disappears.
    check(
        "the derivation would otherwise have dropped 'Output ONLY the steps.'",
        "Output ONLY the steps." not in rd.derive_unguided_template(guided),
    )


def test_exemplar_hop_lines() -> None:
    """Each exemplar's hop line is its OWN gold step count (Jahid, 2026-08-19, issue #12)."""
    print("[exemplar hop lines]")
    examples = [
        {"pool_id": "2hop__p1", "question": "q1?", "decomposition": "1. a?\n2. b?"},
        {"pool_id": "4hop1__p2", "question": "q2?", "decomposition": "a?\nb?\nc?\nd?"},
    ]
    # Hand-computed: 2 steps and 4 steps, whatever the query's hop count is (here 3).
    block = rd.format_few_shot_examples(examples, 3, exemplar_hop_mode="exemplar_gold")
    hops = [int(m) for m in re.findall(r"Hop count: (\d+)", block)]
    check("exemplar_gold states each exemplar's own step count", hops == [2, 4], str(hops))
    check(
        "exemplar_gold ignores the query's hop count",
        "Hop count: 3" not in block,
    )
    check(
        "the per-exemplar count is the shared splitter's",
        [rd.exemplar_gold_hop_count(e) for e in examples]
        == [len(split_step_lines(e["decomposition"])) for e in examples],
    )

    # v1's behaviour is still available, and is what the MetaQA config selects.
    v1_block = rd.format_few_shot_examples(examples, 3, exemplar_hop_mode="query")
    v1_hops = [int(m) for m in re.findall(r"Hop count: (\d+)", v1_block)]
    check("query mode stamps the query's hop count on every exemplar", v1_hops == [3, 3], str(v1_hops))
    check(
        "the two modes differ only in the hop lines",
        re.sub(r"Hop count: \d+", "Hop count: X", block)
        == re.sub(r"Hop count: \d+", "Hop count: X", v1_block),
    )

    # An unguided prompt has no hop line at all, in either mode.
    for mode in rd.EXEMPLAR_HOP_MODES:
        unguided_block = rd.format_few_shot_examples(examples, None, exemplar_hop_mode=mode)
        check(f"unguided block has no hop line ({mode})", "Hop count" not in unguided_block)

    # A missing or empty gold decomposition is a refusal naming the pool id - no fallback.
    for broken, why in (
        ({"pool_id": "2hop__bad", "question": "q?", "decomposition": ""}, "an empty decomposition"),
        ({"pool_id": "2hop__none", "question": "q?", "decomposition": None}, "a missing decomposition"),
        ({"pool_id": "2hop__blank", "question": "q?", "decomposition": "\n  \n"}, "a blank decomposition"),
    ):
        try:
            rd.format_few_shot_examples([broken], 3, exemplar_hop_mode="exemplar_gold")
        except SystemExit as exc:
            check(
                f"refused: {why}, naming the pool id",
                broken["pool_id"] in str(exc) and "no fallback" in str(exc),
            )
        else:
            check(f"refused: {why}, naming the pool id", False)

    try:
        rd.format_few_shot_examples(examples, 3, exemplar_hop_mode="whatever")
    except SystemExit:
        check("an unknown exemplar hop mode is refused", True)
    else:
        check("an unknown exemplar hop mode is refused", False)


def check_prompt_selection(cfg: dict) -> None:
    """Guided vs unguided picks a different prompt file when the model folder has both."""
    print("[prompt selection]")
    models_dir = MODELS_DIR
    mistral = load_config(models_dir / "mistral_7b_instruct" / "config.json")
    guided_prompt = (models_dir / "mistral_7b_instruct" / mistral["prompt_file"]).read_text()
    unguided_prompt = (
        models_dir / "mistral_7b_instruct" / mistral["unguided_prompt_file"]
    ).read_text()
    check("guided prompt carries {hop_count}", "{hop_count}" in guided_prompt)
    check("unguided prompt carries no {hop_count}", "{hop_count}" not in unguided_prompt)

    # A model folder without an unguided prompt substitutes the placeholder instead.
    filled = rd.fill_template(
        guided_prompt,
        question="q?",
        hop_count=None,
        few_shot_examples="",
        unguided_hop_placeholder=require(cfg, "unguided_hop_placeholder"),
    )
    check("unguided fill substitutes the placeholder", "Hop count: Unknown" in filled)
    filled_guided = rd.fill_template(
        guided_prompt,
        question="q?",
        hop_count=3,
        few_shot_examples="",
        unguided_hop_placeholder="Unknown",
    )
    check("guided fill injects the gold hop count", "Hop count: 3" in filled_guided)

    # ... and "Hop count: Unknown" is exactly why the MuSiQue config forbids running an
    # unguided arm on a prompt that still has the slot.
    check(
        "musique config forbids a hop-count slot in an unguided prompt",
        require(cfg, "unguided_prompt_must_omit_hop_count") is True,
    )
    try:
        rd.assert_unguided_prompt_omits_hop_count(
            guided_prompt, prompt_path=Path("<test>"), model="m", config_src="<test>"
        )
    except SystemExit:
        check("guard rejects a guided prompt used for an unguided arm", True)
    else:
        check("guard rejects a guided prompt used for an unguided arm", False)
    rd.assert_unguided_prompt_omits_hop_count(
        unguided_prompt, prompt_path=Path("<test>"), model="m", config_src="<test>"
    )
    check("guard accepts the unguided prompt", True)

    # The guard is a hop-line check, not a placeholder check: a prompt that hardcodes a hop
    # instruction with no {hop_count} in it is refused too.
    for hardcoded in (
        "Rules:\n- The number of steps MUST equal the hop count.\nQuestion: {question}\n",
        "Task:\nHop count: 3\nQuestion: {question}\n",
        "Task:\nhop_count = 3\nQuestion: {question}\n",
    ):
        try:
            rd.assert_unguided_prompt_omits_hop_count(
                hardcoded, prompt_path=Path("<test>"), model="m", config_src="<test>"
            )
        except SystemExit:
            check(f"guard rejects a hardcoded hop line {hardcoded.splitlines()[1]!r}", True)
        else:
            check(f"guard rejects a hardcoded hop line {hardcoded.splitlines()[1]!r}", False)

    # Model folders that cannot run an unguided arm, stated so the smoke test and the
    # experiment both pick a folder that can. Of the two that ship an unguided prompt,
    # qwen3_5_9b is 9B - above the ceiling in configs/model_limits.json - so as configured
    # exactly ONE folder can run these arms. Whether to add an unguided prompt to another
    # <=8B folder is Jahid's call, not this test's.
    models_with_unguided = sorted(
        d.name
        for d in models_dir.iterdir()
        if d.is_dir() and load_config(d / "config.json").get("unguided_prompt_file")
    )
    check(
        "mistral_7b_instruct and qwen3_5_9b are the folders that ship an unguided prompt",
        models_with_unguided == ["mistral_7b_instruct", "qwen3_5_9b"],
        str(models_with_unguided),
    )
    limits = load_config(require(cfg, "model_limits_config"))
    ceiling = int(require(limits, "default_max_params"))
    check("the parameter ceiling is 1e10 (ADR 0015)", ceiling == 10_000_000_000, str(ceiling))
    check(
        "qwen3_5_9b's own config records the ADR 0015 admission",
        "ADR 0015"
        in load_config(models_dir / "qwen3_5_9b" / "config.json").get("notes", ""),
    )
    check(
        "the musique config records the two runnable folders",
        "TWO model folders: mistral_7b_instruct and qwen3_5_9b"
        in require(cfg, "_runnable_models_note"),
    )


def check_guided_cli_cannot_override_a_condition(cfg: dict) -> None:
    """--guided must not silently relabel an arm."""
    print("[--guided vs condition]")
    conditions = require(cfg, "conditions")
    for name in ("unguided", "unguided_capped"):
        try:
            rd.resolve_guided(True, name, conditions[name], cfg)
        except SystemExit:
            check(f"--guided is refused against condition {name}", True)
        else:
            check(f"--guided is refused against condition {name}", False)
    check(
        "--guided agreeing with oracle_guided is accepted",
        rd.resolve_guided(True, "oracle_guided", conditions["oracle_guided"], cfg) is True,
    )
    # With no conditions block (the MetaQA config) the flag still works as before.
    metaqa = load_config("decomposer.json")
    check("metaqa default is unguided", rd.resolve_guided(None, None, {}, metaqa) is False)
    check("metaqa --guided still works", rd.resolve_guided(True, None, {}, metaqa) is True)

    check("uncapped arms have no step-line cap", rd.resolve_step_line_cap("unguided", {}) is None)
    check(
        "capped arm reads its cap from the config",
        rd.resolve_step_line_cap("unguided_capped", conditions["unguided_capped"]) == 8,
    )
    try:
        rd.resolve_step_line_cap("bad", {"stop_after_step_lines": 0})
    except SystemExit:
        check("a non-positive cap is refused", True)
    else:
        check("a non-positive cap is refused", False)


# ------------------------------------------------------------ step-line stopping


def test_shared_step_normalization() -> None:
    """The decomposer's step counting is the evaluator's, not a private copy.

    The step-line budget, the rows-at-cap counter and the evaluator's step metrics all read
    ``src/step_lines.py::split_step_lines``. If they drifted apart, "a cap of 8" and "8
    steps" in a metrics table would be different numbers.
    """
    print("[shared step normalization]")
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import musique_decompositions_evaluator as ev

    check(
        "the evaluator's splitter IS the shared one",
        ev._split_decomposition_text is split_step_lines,
    )
    cases = [
        ("1. a?\n2. b?", ["a?", "b?"]),
        ("a?\n\n b? \n", ["a?", "b?"]),
        ("", []),
        ("10. a?", ["a?"]),
    ]
    for text, want in cases:
        got = split_step_lines(text)
        check(f"split_step_lines({text!r}) == {want}", got == want, str(got))


def test_completed_step_line_count() -> None:
    print("[completed_step_line_count]")
    from step_lines import completed_step_line_count

    # (text, strip_think, truncate_at, expected). Hand-computed: only lines already
    # terminated by a newline count, blank lines never count, an open <think> block means
    # the model has not started answering, and a tail marker is cut before counting.
    cases = [
        ("", False, [], 0),
        ("1. who directed it?", False, [], 0),      # nothing terminated yet
        ("1. a?\n", False, [], 1),
        ("1. a?\n2. b?", False, [], 1),             # second line still being written
        ("1. a?\n2. b?\n", False, [], 2),
        ("1. a?\n\n2. b?\n", False, [], 2),         # blank lines never count
        ("\n\n", False, [], 0),
        ("1. a?\n2. b?\n3. c?\n4. d?\n", False, [], 4),
        # A <think> preamble must not consume the budget: 0 while it is open, and its
        # content does not count once it is closed.
        ("<think>let me\nsee\n", True, [], 0),
        ("<think>let me\nsee</think>\n1. a?\n2. b?\n", True, [], 2),
        # A "Question:" echo after the steps is cut before counting.
        ("1. a?\n2. b?\nQuestion: next one\nmore\n", False, ["Question:"], 2),
    ]
    for text, strip_think, truncate_at, want in cases:
        got = completed_step_line_count(text, strip_think=strip_think, truncate_at=truncate_at)
        check(f"completed_step_line_count({text!r}) == {want}", got == want, f"got {got}")


def test_trim_to_step_lines() -> None:
    print("[trim_to_step_lines]")
    text = "1. a?\n2. b?\n3. c?\n4. partial"
    check("trim keeps the first 2 lines", rd.trim_to_step_lines(text, 2) == "1. a?\n2. b?")
    check(
        "trim drops the partial tail at the cap",
        rd.trim_to_step_lines(text, 3) == "1. a?\n2. b?\n3. c?",
    )
    check("trim is a no-op below the cap", rd.trim_to_step_lines("1. a?\n", 8) == "1. a?")
    check(
        "trim skips blank lines",
        rd.trim_to_step_lines("1. a?\n\n2. b?\n", 2) == "1. a?\n2. b?",
    )
    # The trim keeps the line as written (the enumerator is a comparison-time
    # normalization, not an edit to the model's output).
    check("trim does not rewrite the kept lines", rd.trim_to_step_lines("1. a?\n", 1) == "1. a?")


def test_step_line_stopper() -> None:
    """Feed a synthetic token stream token by token and pin the stopping index."""
    print("[StepLineStopper]")
    # A toy vocabulary: each id decodes to a piece of text, ids concatenate.
    vocab = {0: "1. a?", 1: "\n", 2: "2. b?", 3: "\n", 4: "3. c?", 5: "\n", 6: "4. d?"}
    decode = lambda ids: "".join(vocab[i] for i in ids)  # noqa: E731
    stream = [0, 1, 2, 3, 4, 5, 6]

    def first_stop(cap: int) -> int | None:
        stopper = rd.StepLineStopper(cap, decode)
        for n in range(1, len(stream) + 1):
            if stopper.should_stop(stream[:n]):
                return n
        return None

    # Hand-computed: the newline at stream index 1 completes line 1, index 3 completes
    # line 2, index 5 completes line 3. So a cap of k first fires after 2k tokens.
    check("cap 1 stops after 2 tokens", first_stop(1) == 2, str(first_stop(1)))
    check("cap 2 stops after 4 tokens", first_stop(2) == 4, str(first_stop(2)))
    check("cap 3 stops after 6 tokens", first_stop(3) == 6, str(first_stop(3)))
    check("cap 8 never fires on this stream", first_stop(8) is None, str(first_stop(8)))

    try:
        rd.StepLineStopper(0, decode)
    except ValueError:
        check("cap must be positive", True)
    else:
        check("cap must be positive", False)

    # A think preamble does not consume the budget: with strip_think the same stream, wrapped
    # in an unterminated <think>, never fires; once closed, only the answer lines count.
    think_vocab = {0: "<think>", 1: "reasoning\n", 2: "more\n", 3: "</think>\n", 4: "1. a?\n"}
    think_decode = lambda ids: "".join(think_vocab[i] for i in ids)  # noqa: E731
    open_stopper = rd.StepLineStopper(1, think_decode, strip_think=True)
    check(
        "an open <think> block never fires the cap",
        not any(open_stopper.should_stop([0, 1, 2][:n]) for n in range(1, 4)),
    )
    closed_stopper = rd.StepLineStopper(1, think_decode, strip_think=True)
    fired = [n for n in range(1, 6) if closed_stopper.should_stop([0, 1, 2, 3, 4][:n])]
    check("the cap fires only on the answer line after </think>", fired == [5], str(fired))


def test_incremental_decoder() -> None:
    """Decoding only the new tail must give the same text as decoding from scratch."""
    print("[IncrementalDecoder]")
    vocab = {0: "1. a?", 1: "\n", 2: "2. b?", 3: "\n"}
    calls: list[int] = []

    def decode(ids: list[int]) -> str:
        calls.append(len(ids))
        return "".join(vocab[i] for i in ids)

    dec = rd.IncrementalDecoder(decode)
    stream = [0, 1, 2, 3]
    for n in range(1, len(stream) + 1):
        check(
            f"incremental text after {n} tokens equals the full decode",
            dec.text(stream[:n]) == "".join(vocab[i] for i in stream[:n]),
        )
    check(
        "each step decoded only the new token (no quadratic re-decode)",
        calls == [1, 1, 1, 1],
        str(calls),
    )
    # A shorter list means a new sequence: state resets rather than mis-appending.
    check("a shorter id list resets the decoder", dec.text([0]) == "1. a?")


def test_stopping_criteria_adapter() -> None:
    """The transformers adapter: same rule, driven through a real StoppingCriteriaList."""
    print("[StoppingCriteria adapter]")
    import torch

    vocab = {0: "P", 1: "1. a?", 2: "\n", 3: "2. b?", 4: "\n"}

    class FakeTokenizer:
        def decode(self, ids, skip_special_tokens=True):  # noqa: ARG002
            return "".join(vocab[int(i)] for i in ids)

    prompt = [0, 0]  # two prompt tokens the criterion must ignore
    criteria, state = rd.make_step_line_stopping_criteria(
        FakeTokenizer(), prompt_len=len(prompt), max_step_lines=2
    )
    check("the cap state starts un-fired", state.fired is False)
    fired_at = None
    generated = [1, 2, 3, 4]
    for n in range(1, len(generated) + 1):
        ids = torch.tensor([prompt + generated[:n]])
        if bool(criteria(ids, None)):
            fired_at = n
            break
    check("adapter fires after the 2nd completed line (4 tokens)", fired_at == 4, str(fired_at))
    # The recorded flag is what distinguishes "the cap cut this off" from "the model happened
    # to emit cap-many steps": it comes from the criterion, not from counting the output.
    check("the cap state records that it fired", state.fired is True)
    check(
        "the cap state records the step count it fired at",
        state.fired_at_step_lines == 2,
        str(state.fired_at_step_lines),
    )
    unfired_criteria, unfired_state = rd.make_step_line_stopping_criteria(
        FakeTokenizer(), prompt_len=len(prompt), max_step_lines=8
    )
    for n in range(1, len(generated) + 1):
        unfired_criteria(torch.tensor([prompt + generated[:n]]), None)
    check(
        "a generation that never reaches the cap leaves the state un-fired",
        unfired_state.fired is False and unfired_state.fired_at_step_lines is None,
    )


# ------------------------------------------------------------------ self-exclusion


def test_self_exclusion() -> None:
    """A retrieved exemplar that IS the query is dropped on every few-shot path."""
    print("[self-exclusion]")
    row = {
        "query_id": "2hop__q1",
        "query_question": "Who leads the union?",
        "typed_top_k": [
            {
                "pool_id": "2hop__q1",
                "pool_question": "Who leads the union?",
                "pool_few_shot_decomposition_musique": ["a?", "b?"],
            },
            {
                "pool_id": "2hop__other",
                "pool_question": " who   LEADS the union? ",  # same text, different id
                "pool_few_shot_decomposition_musique": ["c?", "d?"],
            },
            {
                "pool_id": "2hop__p2",
                "pool_question": "Who founded the press?",
                "pool_few_shot_decomposition_musique": ["e?", "f?"],
            },
            {
                "pool_id": "2hop__p3",
                "pool_question": "Where was he born?",
                "pool_few_shot_decomposition_musique": ["g?", "h?"],
            },
        ],
    }
    examples, dropped = rd.examples_from_reranked_row(
        row, "typed", 2, exclude_query_id="2hop__q1", exclude_question="Who leads the union?"
    )
    check("two self-candidates dropped (by id and by text)", dropped == 2, str(dropped))
    check("k=2 exemplars still assembled", len(examples) == 2, str(len(examples)))
    check(
        "the kept exemplars are the non-self ones",
        [e["question"] for e in examples] == ["Who founded the press?", "Where was he born?"],
        str(examples),
    )
    # Without the exclusion arguments the behaviour is the old one (nothing dropped).
    plain, plain_dropped = rd.examples_from_reranked_row(row, "typed", 2)
    check("no exclusion args means nothing is dropped", plain_dropped == 0)
    check("... and the query itself is then the first exemplar", plain[0]["question"] == "Who leads the union?")
    # Too few usable candidates after exclusion is an error, not a short prompt.
    try:
        rd.examples_from_reranked_row(
            row, "typed", 4, exclude_query_id="2hop__q1", exclude_question="Who leads the union?"
        )
    except ValueError:
        check("k larger than the usable candidates is refused", True)
    else:
        check("k larger than the usable candidates is refused", False)

    # The bi-encoder path: same rule, driven with a fake encoder so no weights are needed.
    import numpy as np

    from pool_embeddings import top_k_similar_decomposer

    items = [
        {"id": "p0", "question": "Who leads the union?", "masked": "Who leads the [ORG]?"},
        {"id": "p1", "question": "Who founded the press?", "masked": "Who founded the [ORG]?"},
        {"id": "p2", "question": "Where was he born?", "masked": "Where was he born?"},
    ]
    embeddings = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])

    class FakeEncoder:
        def encode(self, texts, normalize_embeddings=True):  # noqa: ARG002
            return np.array([[1.0, 0.0]])

    top = top_k_similar_decomposer(
        "Who leads the [ORG]?", items, embeddings, FakeEncoder(),
        model_id="fake/encoder", k=2,
        exclude_question="Who leads the union?", exclude_ids=["p0"],
    )
    check("bi-encoder path returns k=2", len(top) == 2, str(len(top)))
    check(
        "bi-encoder path excluded the query itself",
        [it["id"] for it, _ in top] == ["p1", "p2"],
        str([it["id"] for it, _ in top]),
    )
    unfiltered = top_k_similar_decomposer(
        "Who leads the [ORG]?", items, embeddings, FakeEncoder(), model_id="fake/encoder", k=2
    )
    check(
        "without exclusion args the nearest item is still returned",
        [it["id"] for it, _ in unfiltered] == ["p0", "p1"],
        str([it["id"] for it, _ in unfiltered]),
    )


# --------------------------------------------------------------- end-to-end arms


#: An exemplar block in a rendered guided prompt: its stated hop count and its decomposition.
#: The block ends at the next exemplar or at the "Task:" section (the query's own block).
_EXEMPLAR_BLOCK_RX = re.compile(
    r"Hop count: (\d+)\nQuestion: .*?\nDecomposition:\n(.*?)(?=\n\nHop count: |\n\nTask:)",
    re.DOTALL,
)
#: The query's own hop line, which lives under "Task:" and keeps the query's gold hop count.
_QUERY_HOP_RX = re.compile(r"Task:\nHop count: (\d+)")


def _prompt_body(log_text: str) -> str:
    """The rendered prompt out of a prompts_log file, without header or response."""
    after = log_text.split("--- Prompt (", 1)[1]
    after = after.split("---\n", 1)[1]
    return after.split("\n--- Response ---", 1)[0]


def _runner_env() -> dict:
    env = dict(os.environ)
    # Force the fabricated fixtures: this check is about prompt construction, and it must
    # behave identically inside the smoke test and standalone.
    env[PATHS_CONFIG_ENV] = "configs/smoke_paths.json"
    env.setdefault("PYTHONPYCACHEPREFIX", tempfile.gettempdir() + "/qav2-pycache")
    return env


def _run_runner(extra: list[str]) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "components" / "decomposer" / "run_decomposer.py"),
        "--model", E2E_MODEL,
        "--config", "decomposer_musique.json",
        *extra,
    ]
    return subprocess.run(cmd, env=_runner_env(), capture_output=True, text=True)


def _run_condition(condition: str, out_root: Path) -> dict:
    """Run the real runner in --dry-run for one condition; return its artifacts."""
    proc = _run_runner(
        [
            "--condition", condition,
            "--dry-run", "--dry-run-limit", str(E2E_ROWS),
            "--retrieval-input", str(E2E_RETRIEVAL),
            "--allow-unpinned-eval-set",
            "--output-root", str(out_root / condition),
        ]
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"condition {condition} exited {proc.returncode}\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
        )
    run_dirs = sorted((out_root / condition).iterdir())
    if len(run_dirs) != 1:
        raise AssertionError(f"expected one run dir for {condition}, got {run_dirs}")
    run_dir = run_dirs[0]
    prompts = {
        p.name: _prompt_body(p.read_text(encoding="utf-8"))
        for p in sorted((run_dir / "prompts_log").glob("prompt_idx*.txt"))
    }
    return {
        "snapshot": json.loads((run_dir / "config.json").read_text(encoding="utf-8")),
        "metrics": json.loads((run_dir / "metrics.json").read_text(encoding="utf-8")),
        "results": json.loads((run_dir / "results.json").read_text(encoding="utf-8")),
        "prompts": prompts,
    }


def test_conditions_end_to_end() -> None:
    """Run all three arms and compare the prompts they actually build.

    This is the check that the three conditions are what the brief says they are: it reads
    the prompts the runner wrote, not the config that was supposed to produce them.
    """
    print("[end-to-end arms]")
    out_root = Path(tempfile.mkdtemp(prefix="qav2-conditions-"))
    try:
        arms = {c: _run_condition(c, out_root) for c in ("unguided", "oracle_guided", "unguided_capped")}

        for name, arm in arms.items():
            check(f"{name}: {E2E_ROWS} rows", len(arm["results"]) == E2E_ROWS, str(len(arm["results"])))
            check(f"{name}: {E2E_ROWS} prompts logged", len(arm["prompts"]) == E2E_ROWS)
            check(f"{name}: snapshot records the condition", arm["snapshot"]["condition"] == name)

        # 1. unguided - no hop count anywhere in the prompt (task line or few-shot blocks).
        for fname, body in arms["unguided"]["prompts"].items():
            check(f"unguided prompt {fname} has no hop count", "Hop count" not in body)

        # 2. oracle_guided - the gold hop count of that row, taken from the pinned file the
        #    question was read from, is in the prompt.
        for row, (fname, body) in zip(
            arms["oracle_guided"]["results"], sorted(arms["oracle_guided"]["prompts"].items())
        ):
            want = f"Hop count: {row['hop_count']}"
            check(f"oracle_guided prompt {fname} injects '{want}'", want in body)
        hops_seen = sorted({r["hop_count"] for r in arms["oracle_guided"]["results"]})
        check("oracle_guided covered hops 2/3/4", hops_seen == [2, 3, 4], str(hops_seen))

        # 2b. Every EXEMPLAR hop line states that exemplar's own gold step count (Jahid,
        #     2026-08-19). Read off the rendered prompts, not the formatter: this is the
        #     check that the decision actually reached the prompt. The query's hop line is
        #     the last one (under "Task:") and is excluded here - check 2 covers it.
        exemplar_lines = contradictions = 0
        for fname, body in sorted(arms["oracle_guided"]["prompts"].items()):
            for stated, decomp in _EXEMPLAR_BLOCK_RX.findall(body):
                exemplar_lines += 1
                if int(stated) != len(split_step_lines(decomp)):
                    contradictions += 1
        check(
            f"every exemplar hop line equals its own decomposition's step count "
            f"({exemplar_lines} exemplar lines)",
            exemplar_lines > 0 and contradictions == 0,
            f"contradictions={contradictions}",
        )
        # The query's hop line is untouched by the change: still the row's gold depth, and
        # it can differ from every exemplar's, which is the point.
        query_hops = [
            int(_QUERY_HOP_RX.search(body).group(1))
            for _, body in sorted(arms["oracle_guided"]["prompts"].items())
        ]
        check(
            "the query hop line is still the row's gold hop count",
            query_hops == [r["hop_count"] for r in arms["oracle_guided"]["results"]],
            str(query_hops),
        )
        check(
            "the arms record which exemplar hop rule they ran under",
            {a["snapshot"]["few_shot_exemplar_hop_count"] for a in arms.values()}
            == {"exemplar_gold"},
        )

        # 3. unguided_capped - same prompt as unguided (the cap is a decoding-time rule, not
        #    a prompt change) and the cap from the config is wired through.
        check(
            "unguided_capped builds byte-identical prompts to unguided",
            arms["unguided_capped"]["prompts"] == arms["unguided"]["prompts"],
        )
        check(
            "unguided_capped carries the configured cap of 8",
            arms["unguided_capped"]["snapshot"]["stop_after_step_lines"] == 8,
            str(arms["unguided_capped"]["snapshot"]["stop_after_step_lines"]),
        )
        for name in ("unguided", "oracle_guided"):
            check(
                f"{name} is uncapped",
                arms[name]["snapshot"]["stop_after_step_lines"] is None,
            )

        # 3b. The strong form of "the arms differ in hop information only": strip the
        #     hop-bearing lines out of the *rendered* guided prompt and it becomes the
        #     unguided prompt, byte for byte. This covers the few-shot blocks too (guided
        #     puts a "Hop count: n" line on each exemplar), not just the template.
        for (gname, gbody), (uname, ubody) in zip(
            sorted(arms["oracle_guided"]["prompts"].items()),
            sorted(arms["unguided"]["prompts"].items()),
        ):
            check(f"{gname}: same prompt file index as unguided", gname == uname)
            check(
                f"{gname}: guided minus hop lines == unguided, byte for byte",
                rd.derive_unguided_template(gbody) == ubody,
            )

        # 4. Everything else identical across the three arms - including the retrieval input
        #    by content, not just by path: same bytes or the arms are not comparable.
        for key in ("model_id", "seed", "generation", "hops", "questions_template_key",
                    "retrieval", "quantization", "prompt_style", "few_shot",
                    "post_process", "evaluation_set"):
            values = [json.dumps(arm["snapshot"][key], sort_keys=True) for arm in arms.values()]
            check(f"all three arms share {key}", len(set(values)) == 1, str(set(values)))
        sha = arms["unguided"]["snapshot"]["retrieval"]["input_sha256"]
        check("the retrieval input is content-addressed in the snapshot", bool(sha), str(sha))
        check(
            "the unguided prompt file differs from the guided one, by hash",
            arms["unguided"]["snapshot"]["prompt_sha256"]
            != arms["oracle_guided"]["snapshot"]["prompt_sha256"],
        )
        check(
            "the unguided arms record the prompt invariant as checked",
            all(
                arms[name]["snapshot"]["unguided_prompt_invariant"]["checked"] is True
                for name in ("unguided", "oracle_guided", "unguided_capped")
            ),
        )
        id_sets = [[r["query_id"] for r in arm["results"]] for arm in arms.values()]
        check("all three arms ran the same question ids in the same order", id_sets[0] == id_sets[1] == id_sets[2])
        check(
            "all three arms report the same rows per hop",
            len({json.dumps(a["metrics"]["rows_loaded_per_hop"], sort_keys=True) for a in arms.values()}) == 1,
            str(arms["unguided"]["metrics"]["rows_loaded_per_hop"]),
        )
        check(
            "the fixture run is recorded as NOT the pinned evaluation set",
            arms["unguided"]["metrics"]["evaluation_set"]["pinned"] is False
            and arms["unguided"]["metrics"]["evaluation_set"]["allow_unpinned_flag"] is True,
            json.dumps(arms["unguided"]["metrics"]["evaluation_set"]),
        )
        # Two exclusions, one per mechanism: row 1's planted candidate carries the query's
        # own id, and 4hop1__d003_c's pool candidate 4hop1__t010_j repeats the query's
        # question text under a different id. Both are the query as an exemplar.
        check(
            "the self-exclusion path dropped both self-candidates (by id and by text)",
            arms["unguided"]["metrics"]["few_shot_self_exclusion"]["self_examples_dropped"] == 2
            and arms["unguided"]["metrics"]["few_shot_self_exclusion"][
                "rows_with_a_self_example_dropped"
            ] == 2,
            json.dumps(arms["unguided"]["metrics"]["few_shot_self_exclusion"]),
        )
        check(
            "self-exclusion is reported as enabled when few-shot ran",
            arms["unguided"]["snapshot"]["few_shot_self_exclusion"]["enabled"] is True,
        )
        for name, arm in arms.items():
            check(
                f"{name}: truncation counters are null in a dry run",
                arm["metrics"]["rows_at_max_new_tokens"] is None
                and arm["metrics"]["rows_at_step_line_cap"] is None,
            )
            check(
                f"{name}: max_new_tokens is reported ({arm['metrics']['max_new_tokens']})",
                arm["metrics"]["max_new_tokens"] == 1024,
            )
            check(
                f"{name}: every result row carries the same truncation and cost fields",
                all(
                    set(r) == {
                        "query_id", "question", "hop_count", "decomposition",
                        "few_shot_source", "decomposition_raw", "step_lines",
                        "hit_max_new_tokens", "stopped_at_step_line_cap",
                        # the cost fields of issue #13 (PR #24), same shape in every arm
                        "prompt_tokens", "completion_tokens", "latency_seconds",
                        # issue #27: hop_count stays the GOLD depth in every arm, and these
                        # two say what the prompt actually stated and where it came from
                        # (null / "gold" in these three arms; the prediction in router_guided)
                        "prompt_hop_count", "hop_count_source",
                    }
                    for r in arm["results"]
                ),
                str(sorted(arm["results"][0])),
            )
            check(
                f"{name}: the two step-line-cap counters are defined separately",
                set(arm["metrics"]["truncation_definitions"])
                == {
                    "rows_at_step_line_cap",
                    "rows_stopped_at_step_line_cap",
                    "rows_at_max_new_tokens",
                },
                str(sorted(arm["metrics"]["truncation_definitions"])),
            )
            check(
                f"{name}: cost is recorded as unmeasured in a dry run",
                arm["metrics"]["cost"]["rows_measured"] == 0
                and arm["metrics"]["cost"]["mean_total_tokens_per_query"] is None,
                json.dumps(arm["metrics"]["cost"])[:160],
            )

        # 5. The refusals. Each is a non-zero exit with a message naming the reason.
        refusals = [
            (
                "a model folder without an unguided prompt",
                ["--model", "qwen2_5_3b"],
                "hop-count information",
            ),
        ]
        for label, extra, needle in refusals:
            proc = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "components" / "decomposer" / "run_decomposer.py"),
                    "--config", "decomposer_musique.json",
                    "--condition", "unguided", "--dry-run", "--dry-run-limit", "1",
                    "--retrieval-input", str(E2E_RETRIEVAL), "--allow-unpinned-eval-set",
                    "--output-root", str(out_root / "guard"),
                    *extra,
                ],
                env=_runner_env(), capture_output=True, text=True,
            )
            out = proc.stdout + proc.stderr
            check(
                f"refused: {label}",
                proc.returncode != 0 and needle in out,
                f"rc={proc.returncode} needle={needle!r}",
            )

        # With no --retrieval-input the committed config still has one: retrieval.input_key
        # resolves through the paths config (ADR 0014). Under the fixture paths config that
        # key points at a file the fixture tree does not contain, so the run refuses by
        # NAMING the resolved artifact rather than falling back to MetaQA exemplars.
        proc = _run_runner(
            [
                "--condition", "unguided", "--dry-run", "--dry-run-limit", "1",
                "--allow-unpinned-eval-set", "--output-root", str(out_root / "no_retrieval"),
            ]
        )
        out = proc.stdout + proc.stderr
        check(
            "no --retrieval-input: the config's input_key is resolved, not defaulted away",
            proc.returncode != 0
            and "retrieval input not found" in out
            and "sim_dev_sample600_top20_rerankTop5.jsonl" in out,
            f"rc={proc.returncode} out={out[-200:]!r}",
        )

        # Row counts that are not the pinned set: refused unless the flag is passed.
        proc = _run_runner(
            [
                "--condition", "unguided", "--dry-run", "--dry-run-limit", "1",
                "--retrieval-input", str(E2E_RETRIEVAL),
                "--output-root", str(out_root / "unpinned"),
            ]
        )
        out = proc.stdout + proc.stderr
        check(
            "refused: 3 rows per hop is not the pinned 200 (ADR 0007)",
            proc.returncode != 0 and "not the pinned evaluation set" in out and "hop 2: loaded 3" in out,
            f"rc={proc.returncode}",
        )

        # An id whose hop depth cannot be parsed: a refusal, never hop_fallback.
        bad = out_root / "bad_ids.jsonl"
        rows = [json.loads(l) for l in E2E_RETRIEVAL.read_text().splitlines() if l.strip()]
        rows[0]["query_id"] = "no_hop_prefix__x1"
        bad.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        for condition in ("unguided", "oracle_guided"):
            proc = _run_runner(
                [
                    "--condition", condition, "--dry-run", "--dry-run-limit", "1",
                    "--retrieval-input", str(bad), "--allow-unpinned-eval-set",
                    "--output-root", str(out_root / f"badhop_{condition}"),
                ]
            )
            out = proc.stdout + proc.stderr
            check(
                f"refused ({condition}): an id whose hop depth cannot be parsed",
                proc.returncode != 0
                and "hop depth cannot be parsed" in out
                and "no_hop_prefix__x1" in out,
                f"rc={proc.returncode}",
            )

        # 6. The decoy: right per-hop counts, wrong questions. Counts alone would pass this,
        #    which is why the ids are compared against the pinned files (ADR 0007/0011).
        decoy = out_root / "decoy_ids.jsonl"
        rows = [json.loads(l) for l in E2E_RETRIEVAL.read_text().splitlines() if l.strip()]
        for row in rows:
            hop_prefix = str(row["query_id"]).split("__", 1)[0]
            row["query_id"] = f"{hop_prefix}__decoy_{row['query_index']}"
        decoy.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        proc = _run_runner(
            [
                "--condition", "unguided", "--dry-run", "--dry-run-limit", "9",
                "--retrieval-input", str(decoy),
                "--output-root", str(out_root / "decoy"),
            ]
        )
        out = proc.stdout + proc.stderr
        check(
            "refused: a decoy set with the same per-hop counts but different ids",
            proc.returncode != 0
            and "not in the pinned files" in out
            and "decoy_0" in out
            and "pinned id(s) were not loaded" in out,
            f"rc={proc.returncode}",
        )
        # The same decoy is permitted only with the explicit opt-out, and is then recorded
        # as not the pinned set rather than passing silently.
        proc = _run_runner(
            [
                "--condition", "unguided", "--dry-run", "--dry-run-limit", "9",
                "--retrieval-input", str(decoy), "--allow-unpinned-eval-set",
                "--output-root", str(out_root / "decoy_allowed"),
            ]
        )
        check("the decoy runs only under --allow-unpinned-eval-set", proc.returncode == 0,
              f"rc={proc.returncode}")
        run_dir = sorted((out_root / "decoy_allowed").iterdir())[0]
        decoy_metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        check(
            "... and it is recorded as unpinned with the id mismatch counted",
            decoy_metrics["evaluation_set"]["pinned"] is False
            and decoy_metrics["evaluation_set"]["id_identity_checked"] is True
            and decoy_metrics["evaluation_set"]["ids_unexpected_count"] == 9,
            json.dumps(decoy_metrics["evaluation_set"])[:200],
        )
    finally:
        shutil.rmtree(out_root, ignore_errors=True)


def test_metaqa_guided_prompt_is_unchanged() -> None:
    """The MetaQA path still stamps the QUERY's hop count on every exemplar (v1).

    Jahid's 2026-08-19 exemplar-hop decision was taken for the MuSiQue conditions. The two
    paths share ``format_few_shot_examples``, so the behaviour is selected by
    ``few_shot_exemplar_hop_count`` in each config - and this check is what stops the MuSiQue
    decision from leaking into MetaQA prompts.
    """
    print("[metaqa guided prompt]")
    out_root = Path(tempfile.mkdtemp(prefix="qav2-metaqa-guided-"))
    try:
        proc = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "components" / "decomposer" / "run_decomposer.py"),
                "--model", E2E_MODEL, "--guided", "--dry-run", "--dry-run-limit", "2",
                "--retrieval-input", str(REPO_ROOT / "tests" / "fixtures" / "retrieval" / "top20.jsonl"),
                "--retrieval-mode", "uniform", "--retrieval-k", "5",
                "--output-root", str(out_root),
            ],
            env=_runner_env(), capture_output=True, text=True,
        )
        check("a guided MetaQA dry run still succeeds", proc.returncode == 0, f"rc={proc.returncode}")
        run_dir = sorted(out_root.iterdir())[0]
        snapshot = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        check(
            "the MetaQA run records the v1 exemplar hop rule",
            snapshot["few_shot_exemplar_hop_count"] == "query",
            snapshot["few_shot_exemplar_hop_count"],
        )
        for path in sorted((run_dir / "prompts_log").glob("prompt_idx*.txt")):
            body = _prompt_body(path.read_text(encoding="utf-8"))
            hops = re.findall(r"Hop count: (\d+)", body)
            check(
                f"{path.name}: every hop line is the query's hop count (v1 behaviour)",
                len(hops) > 1 and len(set(hops)) == 1,
                str(hops),
            )
    finally:
        shutil.rmtree(out_root, ignore_errors=True)


# ----------------------------------------------------------------- data resolution


def test_retrieval_input_resolution() -> None:
    """``resolve_retrieval_input``: precedence, and the two refusals it carries.

    The committed MuSiQue config now names its retrieval artifact through
    ``retrieval.input_key`` (ADR 0014), so the ``require_input`` refusal is no longer
    reachable through it. The guard still protects any config with no retrieval input - the
    fallback it prevents is a seeded random draw from the committed MetaQA pool, i.e. a
    different few-shot method under the label of ADR 0006's - so it is checked here directly.
    """
    print("[retrieval input resolution]")
    paths_cfg = {
        "data_root_resolved": "/data/root",
        "datasets": {"artifact": "musique/top5.jsonl"},
    }

    def cfg(input_value: str | None, input_key: str | None, require_input: bool = True) -> dict:
        block: dict = {"input": input_value, "require_input": require_input}
        if input_key is not None:
            block["input_key"] = input_key
        return {"_config_path": "<test>", "retrieval": block}

    def refusal(*args) -> str:
        try:
            rd.resolve_retrieval_input(*args)
        except SystemExit as exc:
            return str(exc)
        return "<no refusal>"

    resolved = rd.resolve_retrieval_input(None, cfg(None, "artifact"), paths_cfg)
    check(
        "input_key resolves against data_root from the paths config",
        resolved == "/data/root/musique/top5.jsonl",
        str(resolved),
    )
    check(
        "--retrieval-input wins over input_key",
        rd.resolve_retrieval_input("cli.jsonl", cfg(None, "artifact"), paths_cfg) == "cli.jsonl",
    )
    check(
        "an explicit retrieval.input path is still honoured",
        rd.resolve_retrieval_input(None, cfg("explicit.jsonl", None), paths_cfg) == "explicit.jsonl",
    )
    both = refusal(None, cfg("explicit.jsonl", "artifact"), paths_cfg)
    check(
        "refused: retrieval.input and retrieval.input_key both set",
        "set exactly one" in both,
        both[:160],
    )
    neither = refusal(None, cfg(None, None), paths_cfg)
    check(
        "refused: no retrieval input at all (retrieval.require_input, ADR 0006)",
        "require_input" in neither and "ADR 0006" in neither,
        neither[:160],
    )
    check(
        "require_input false gives None rather than a refusal",
        rd.resolve_retrieval_input(None, cfg(None, None, require_input=False), paths_cfg) is None,
    )


def check_eval_set_resolves(cfg: dict) -> None:
    print("[evaluation set]")
    paths_cfg = load_paths(require(cfg, "paths_config"))
    template = require(paths_cfg, "datasets." + require(cfg, "questions_template_key"))
    data_root = Path(paths_cfg["data_root_resolved"])
    expected_rows = int(require(cfg, "eval_rows_per_hop"))
    total = 0
    ids: set[str] = set()
    for hop in require(cfg, "hops"):
        path = resolve_path(template.format(hop=hop), data_root)
        check(f"{hop}-hop file exists", path.exists(), str(path))
        items = rd.load_question_items(
            path,
            questions_format=require(cfg, "questions_format"),
            question_field=require(cfg, "questions_jsonl.question_field"),
            id_field=require(cfg, "questions_jsonl.id_field"),
        )
        check(f"{hop}-hop file has {expected_rows} rows", len(items) == expected_rows, str(len(items)))
        check(f"{hop}-hop rows all carry an id", all(i["query_id"] for i in items))
        ids.update(str(i["query_id"]) for i in items)
        total += len(items)
    check(
        f"evaluation set is {expected_rows * 3} questions with distinct ids",
        total == expected_rows * 3 and len(ids) == total,
        f"total={total} distinct_ids={len(ids)}",
    )
    # The retrieval artifact ADR 0014 pins the conditions to, resolved the same way the
    # runner resolves it: through datasets.<retrieval.input_key> in the paths config.
    artifact = Path(rd.resolve_retrieval_input(None, cfg, paths_cfg))
    check("the pinned retrieval artifact exists", artifact.exists(), str(artifact))
    if artifact.exists():
        rows = [line for line in artifact.read_text(encoding="utf-8").splitlines() if line.strip()]
        check(
            f"the retrieval artifact has one row per evaluation question ({total})",
            len(rows) == total,
            f"rows={len(rows)}",
        )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--skip-data-checks",
        action="store_true",
        help="Skip the ADR 0007 evaluation-set check (needs the real data_root).",
    )
    args = p.parse_args()

    test_metaqa_defaults_unchanged()
    cfg = test_musique_config()
    check_conditions(cfg)
    check_prompt_invariant(cfg)
    test_compound_hop_lines_are_refused()
    test_exemplar_hop_lines()
    check_prompt_selection(cfg)
    check_guided_cli_cannot_override_a_condition(cfg)
    test_shared_step_normalization()
    test_completed_step_line_count()
    test_trim_to_step_lines()
    test_step_line_stopper()
    test_incremental_decoder()
    test_stopping_criteria_adapter()
    test_self_exclusion()
    test_retrieval_input_resolution()
    test_conditions_end_to_end()
    test_metaqa_guided_prompt_is_unchanged()
    if args.skip_data_checks:
        print("[evaluation set] skipped (--skip-data-checks)")
    else:
        check_eval_set_resolves(cfg)

    print(f"\n{len(CHECKS)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
