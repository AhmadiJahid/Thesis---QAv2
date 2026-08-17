#!/usr/bin/env python3
"""Checks for the MuSiQue decomposer conditions (issue #12). No GPU, no model weights.

pytest is not in the environment, so this is a plain script: every check is an assertion
with a hand-computed expectation, and the exit code is the result.

What it covers:

1. **Config resolution** - ``configs/decomposer_musique.json`` resolves to the MuSiQue
   question template, hops [2, 3, 4] and three conditions (``unguided``,
   ``oracle_guided``, ``unguided_capped``) whose only differences are the hop information
   in the prompt and the step-line budget.
2. **MetaQA defaults unchanged** - ``configs/decomposer.json`` still has guided=false,
   hops [1, 2, 3], the MetaQA template and no conditions block.
3. **The step-line stopping rule** - ``count_step_lines`` / ``trim_to_step_lines`` /
   ``StepLineStopper`` against synthetic token streams with hand-computed expectations,
   plus the transformers ``StoppingCriteria`` adapter driven by a fake tokenizer.
4. **The three arms end to end** - the real runner is executed in ``--dry-run`` against
   the fabricated fixtures once per condition, and the *prompts it wrote* are compared:
   unguided carries no hop count at all, oracle_guided carries the gold hop count of the
   file each question came from, unguided_capped builds byte-identical prompts to unguided
   and carries the configured cap. Model, seed, decoding, retrieval and the question ids
   must be identical across all three. Also checks that a model folder without an unguided
   prompt is refused for an unguided arm. Writes to a temp dir and deletes it.
5. **The evaluation set resolves** - the three pinned files of ADR 0007 exist under
   ``data_root`` with ``eval_rows_per_hop`` rows each. Skipped with ``--skip-data-checks``
   (the smoke test runs against fabricated fixtures, which are not that set).

Usage::

    python tests/test_decomposer_conditions.py
    python tests/test_decomposer_conditions.py --skip-data-checks
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "components" / "decomposer"))

from run_config import PATHS_CONFIG_ENV, load_config, load_paths, require, resolve_path  # noqa: E402

import run_decomposer as rd  # noqa: E402

CHECKS: list[str] = []

#: The end-to-end checks run the real runner in --dry-run against the fabricated fixtures
#: (3 rows per hop for hops 2/3/4). mistral_7b_instruct is the model folder because it
#: ships an unguided prompt, which the unguided arms require.
E2E_MODEL = "mistral_7b_instruct"
E2E_ROWS = 9


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


def test_musique_config() -> dict:
    print("[musique config]")
    cfg = load_config("decomposer_musique.json")
    check("musique hops are [2, 3, 4]", cfg["hops"] == [2, 3, 4], repr(cfg["hops"]))
    check(
        "musique questions template key",
        cfg["questions_template_key"] == "musique_dev_sample_template",
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
        "max_new_tokens override is present",
        require(cfg, "generation_overrides.max_new_tokens") == 256,
    )
    return cfg


def test_conditions(cfg: dict) -> None:
    print("[conditions]")
    conditions = require(cfg, "conditions")
    check(
        "three conditions, named as briefed",
        sorted(conditions) == ["oracle_guided", "unguided", "unguided_capped"],
        str(sorted(conditions)),
    )

    # Hand-computed expectation per arm: (guided, stop_after_step_lines).
    expected = {
        "unguided": (False, None),
        "oracle_guided": (True, None),
        "unguided_capped": (False, 8),
    }
    for name, (want_guided, want_cap) in expected.items():
        got_name, block = rd.resolve_condition(cfg, name)
        check(f"condition {name} resolves", got_name == name)
        guided = bool(block["guided"])
        cap = block.get("stop_after_step_lines")
        check(f"condition {name} guided={want_guided}", guided is want_guided, repr(guided))
        check(f"condition {name} cap={want_cap}", cap == want_cap, repr(cap))

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
    check("override raises max_new_tokens to 256", merged["max_new_tokens"] == 256)
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


def test_prompt_selection(cfg: dict) -> None:
    """Guided vs unguided picks a different prompt file when the model folder has both."""
    print("[prompt selection]")
    models_dir = REPO_ROOT / "components" / "decomposer" / "models"
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

    # Model folders that cannot run an unguided arm, stated so the smoke test and the
    # experiment both pick a folder that can.
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


def test_guided_cli_cannot_override_a_condition(cfg: dict) -> None:
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


def test_count_step_lines() -> None:
    print("[count_step_lines]")
    cases = [
        ("", 0),
        ("1. who directed it?", 0),          # nothing terminated yet
        ("1. a?\n", 1),
        ("1. a?\n2. b?", 1),                 # the second line is still being written
        ("1. a?\n2. b?\n", 2),
        ("1. a?\n\n2. b?\n", 2),             # blank lines never count
        ("\n\n", 0),
        ("1. a?\n2. b?\n3. c?\n4. d?\n", 4),
    ]
    for text, want in cases:
        got = rd.count_step_lines(text)
        check(f"count_step_lines({text!r}) == {want}", got == want, f"got {got}")


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


def test_stopping_criteria_adapter() -> None:
    """The transformers adapter: same rule, driven through a real StoppingCriteriaList."""
    print("[StoppingCriteria adapter]")
    import torch

    vocab = {0: "P", 1: "1. a?", 2: "\n", 3: "2. b?", 4: "\n"}

    class FakeTokenizer:
        def decode(self, ids, skip_special_tokens=True):  # noqa: ARG002
            return "".join(vocab[int(i)] for i in ids)

    prompt = [0, 0]  # two prompt tokens the criterion must ignore
    criteria = rd.make_step_line_stopping_criteria(
        FakeTokenizer(), prompt_len=len(prompt), max_step_lines=2
    )
    fired_at = None
    generated = [1, 2, 3, 4]
    for n in range(1, len(generated) + 1):
        ids = torch.tensor([prompt + generated[:n]])
        if bool(criteria(ids, None)):
            fired_at = n
            break
    check("adapter fires after the 2nd completed line (4 tokens)", fired_at == 4, str(fired_at))


# --------------------------------------------------------------- end-to-end arms


def _prompt_body(log_text: str) -> str:
    """The rendered prompt out of a prompts_log file, without header or response."""
    after = log_text.split("--- Prompt (", 1)[1]
    after = after.split("---\n", 1)[1]
    return after.split("\n--- Response ---", 1)[0]


def _run_condition(condition: str, out_root: Path) -> dict:
    """Run the real runner in --dry-run for one condition; return its artifacts."""
    env = dict(os.environ)
    # Force the fabricated fixtures: this check is about prompt construction, and it must
    # behave identically inside the smoke test and standalone.
    env[PATHS_CONFIG_ENV] = "configs/smoke_paths.json"
    env.setdefault("PYTHONPYCACHEPREFIX", tempfile.gettempdir() + "/qav2-pycache")
    cmd = [
        sys.executable,
        str(REPO_ROOT / "components" / "decomposer" / "run_decomposer.py"),
        "--model", E2E_MODEL,
        "--config", "decomposer_musique.json",
        "--condition", condition,
        "--dry-run", "--dry-run-limit", str(E2E_ROWS),
        "--output-root", str(out_root / condition),
    ]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
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

        # 4. Everything else identical across the three arms.
        for key in ("model_id", "seed", "generation", "hops", "questions_template_key",
                    "retrieval", "quantization", "prompt_style"):
            values = [json.dumps(arm["snapshot"][key], sort_keys=True) for arm in arms.values()]
            check(f"all three arms share {key}", len(set(values)) == 1, str(set(values)))
        id_sets = [[r["query_id"] for r in arm["results"]] for arm in arms.values()]
        check("all three arms ran the same question ids in the same order", id_sets[0] == id_sets[1] == id_sets[2])
        check(
            "all three arms report the same rows per hop",
            len({json.dumps(a["metrics"]["rows_loaded_per_hop"], sort_keys=True) for a in arms.values()}) == 1,
            str(arms["unguided"]["metrics"]["rows_loaded_per_hop"]),
        )

        # 5. The guard: a model folder with no unguided prompt cannot run an unguided arm.
        env = dict(os.environ)
        env[PATHS_CONFIG_ENV] = "configs/smoke_paths.json"
        proc = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "components" / "decomposer" / "run_decomposer.py"),
                "--model", "qwen2_5_3b", "--config", "decomposer_musique.json",
                "--condition", "unguided", "--dry-run", "--dry-run-limit", "1",
                "--output-root", str(out_root / "guard"),
            ],
            env=env, capture_output=True, text=True,
        )
        check(
            "a model folder without an unguided prompt is refused for an unguided arm",
            proc.returncode != 0 and "{hop_count}" in (proc.stdout + proc.stderr),
            f"rc={proc.returncode}",
        )
    finally:
        shutil.rmtree(out_root, ignore_errors=True)


# ----------------------------------------------------------------- data resolution


def test_eval_set_resolves(cfg: dict) -> None:
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
    test_conditions(cfg)
    test_prompt_selection(cfg)
    test_guided_cli_cannot_override_a_condition(cfg)
    test_count_step_lines()
    test_trim_to_step_lines()
    test_step_line_stopper()
    test_stopping_criteria_adapter()
    test_conditions_end_to_end()
    if args.skip_data_checks:
        print("[evaluation set] skipped (--skip-data-checks)")
    else:
        test_eval_set_resolves(cfg)

    print(f"\n{len(CHECKS)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
