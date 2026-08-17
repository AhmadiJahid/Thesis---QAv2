#!/usr/bin/env python3
"""Decomposer component: split a question into single-hop sub-questions.

One runner for every model. v1 kept four copies of ``decomposer.py``: two were
byte-identical (a plain prompt with inline examples, no retrieval), one added
similarity/reranked few-shot selection and quantization, and one added a chat
template with ``enable_thinking=False`` plus ``<think>`` stripping. Those
differences are now fields in ``components/decomposer/models/<model>/config.json``.
Prompts stay per-model and byte-identical to v1.

Usage::

    python components/decomposer/run_decomposer.py --model mistral_7b_instruct \\
        --retrieval-input runs/pool_sweep/biencoder_top5/<cell>/top5_biencoder.jsonl
    python components/decomposer/run_decomposer.py --model qwen2_5_3b --dry-run
    python components/decomposer/run_decomposer.py --model mistral_7b_instruct \\
        --config decomposer_musique.json --condition unguided_capped

``--config decomposer_musique.json`` runs the pinned MuSiQue evaluation set (ADR 0007) and
carries a ``conditions`` block: ``unguided``, ``oracle_guided`` (gold hop count in the
prompt) and ``unguided_capped`` (no hop count, generation stopped at N step lines). All
three share model, seed and decoding; only the prompt's hop information and the step-line
budget differ. The unguided arms need a model folder that ships an ``unguided_prompt_file``
(the config sets ``unguided_prompt_must_omit_hop_count``), so ``qwen2_5_3b`` and
``phi_4_mini_instruct`` cannot run them.

Every run writes a config snapshot, a metrics JSON and a run note, and asserts the
model's parameter count against the ceiling in ``configs/model_limits.json``.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from model_size import assert_within_ceiling, load_limits, unasserted_note  # noqa: E402
from run_artifacts import now_iso, run_id, write_run_artifacts  # noqa: E402
from run_config import (  # noqa: E402
    load_config,
    load_paths,
    optional,
    require,
    resolve_path,
    runs_path,
)
from seeding import new_rng, set_global_seed  # noqa: E402

_THINK_RX = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_ID_HOP_RX = re.compile(r"^(?P<h>\d+)hop")

#: Keys a ``conditions.<name>`` block may carry. Deliberately short: the conditions of an
#: experiment arm differ only in what the prompt says about the hop count and in the
#: step-line budget. Model, seed and decoding are shared, so no condition can move them.
_CONDITION_KEYS = frozenset({"guided", "stop_after_step_lines", "_note"})


# --------------------------------------------------------------------------- IO


def load_questions(file_path: Path) -> list[str]:
    if not file_path.exists():
        print(f"Warning: {file_path} not found.")
        return []
    with file_path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_question_items(
    file_path: Path, *, questions_format: str, question_field: str, id_field: str
) -> list[dict]:
    """Read one question source file into ``{query_id, question}`` items.

    ``lines`` is the MetaQA plain-text format (one question per line, no ids); ``jsonl``
    is the MuSiQue format (one JSON object per line, with an id that downstream evaluation
    joins on). A malformed JSONL row is an error, not a silent skip: the evaluation set is
    pinned (ADR 0007) and a dropped row would change what a condition was measured on.
    """
    if questions_format == "lines":
        return [{"query_id": None, "question": q} for q in load_questions(file_path)]
    if questions_format != "jsonl":
        raise SystemExit(
            f"unknown questions_format {questions_format!r} (expected 'lines' or 'jsonl')"
        )
    if not file_path.exists():
        raise SystemExit(f"question file not found: {file_path}")
    items: list[dict] = []
    for lineno, row in enumerate(load_jsonl(file_path), start=1):
        question = row.get(question_field)
        if not isinstance(question, str) or not question.strip():
            raise SystemExit(
                f"{file_path}:{lineno} has no usable {question_field!r} field "
                f"(got {row.get(question_field)!r})"
            )
        items.append({"query_id": row.get(id_field), "question": question.strip()})
    if not items:
        raise SystemExit(f"no rows in question file: {file_path}")
    return items


def resolve_condition(cfg: dict, requested: str | None) -> tuple[str | None, dict]:
    """Pick the named condition out of the config's ``conditions`` block.

    A config without a ``conditions`` block (MetaQA's) behaves exactly as before. A config
    with one must name a default in ``condition``; ``--condition`` overrides it.
    """
    conditions = optional(cfg, "conditions")
    name = requested if requested is not None else optional(cfg, "condition")
    src = cfg.get("_config_path", "<config>")

    if conditions is None:
        if name:
            raise SystemExit(
                f"--condition {name!r} was given but {src} has no 'conditions' block"
            )
        return None, {}
    if not isinstance(conditions, dict) or not conditions:
        raise SystemExit(f"'conditions' in {src} must be a non-empty object")
    if not name:
        raise SystemExit(
            f"{src} has a 'conditions' block but no default 'condition'; "
            f"set one or pass --condition (available: {sorted(conditions)})"
        )
    if name not in conditions:
        raise SystemExit(f"unknown condition {name!r} in {src} (available: {sorted(conditions)})")
    block = conditions[name]
    if not isinstance(block, dict):
        raise SystemExit(f"condition {name!r} in {src} must be an object")
    unknown = sorted(set(block) - _CONDITION_KEYS)
    if unknown:
        raise SystemExit(
            f"condition {name!r} in {src} sets {unknown}, which a condition may not set. "
            f"Allowed: {sorted(_CONDITION_KEYS)}. Model, seed and decoding are shared across "
            "conditions on purpose - moving one of them in a single arm would make the arms "
            "incomparable."
        )
    return name, dict(block)


def resolve_guided(
    cli_guided: bool | None, condition_name: str | None, condition: dict, cfg: dict
) -> bool:
    """Resolve the guided flag: CLI, then the condition, then the config default.

    A named condition that fixes ``guided`` may **not** be overridden from the CLI. The
    condition name is what the snapshot, the metrics and the log entry record, so an
    overridden arm would be filed under a label it did not run.
    """
    if cli_guided is not None:
        if "guided" in condition and bool(condition["guided"]) != bool(cli_guided):
            raise SystemExit(
                f"--guided contradicts condition {condition_name!r}, which sets "
                f"guided={bool(condition['guided'])}. Select the condition that encodes the "
                "arm you want instead of overriding it: the run would otherwise be recorded "
                f"as {condition_name!r} while running the other arm's prompt."
            )
        return bool(cli_guided)
    if "guided" in condition:
        return bool(condition["guided"])
    return bool(require(cfg, "guided"))


def resolve_step_line_cap(condition_name: str | None, condition: dict) -> int | None:
    """The condition's step-line budget, or None when the arm is uncapped."""
    cap = condition.get("stop_after_step_lines")
    if cap is None:
        return None
    cap = int(cap)
    if cap <= 0:
        raise SystemExit(
            f"condition {condition_name!r}: stop_after_step_lines must be positive, got {cap}"
        )
    return cap


def assert_unguided_prompt_omits_hop_count(
    template: str, *, prompt_path: Path, model: str, config_src: str
) -> None:
    """Refuse an unguided run whose prompt still carries a hop-count slot.

    A model folder with no ``unguided_prompt_file`` falls back to the guided prompt, where
    ``{hop_count}`` is filled with ``unguided_hop_placeholder``. The prompt then reads
    "Hop count: Unknown" under a rule that says the number of steps must equal the hop
    count - which is neither the unguided condition nor the guided one. A config running a
    guided/unguided comparison sets ``unguided_prompt_must_omit_hop_count`` so this is a
    loud failure rather than a silently mislabelled arm.
    """
    if "{hop_count}" not in template:
        return
    raise SystemExit(
        f"unguided run, but the prompt {prompt_path} still has a {{hop_count}} slot.\n"
        f"Model folder {model!r} has no 'unguided_prompt_file', so the guided prompt was "
        f"used and the hop count would be filled with the placeholder. {config_src} sets "
        "'unguided_prompt_must_omit_hop_count', which forbids that. Use a model folder that "
        "ships an unguided prompt (one without {hop_count}), or add one to this folder."
    )


def apply_generation_overrides(generation: dict, overrides: dict | None, src: str) -> dict:
    """Overlay config-level decoding overrides (same for every condition) on the model's."""
    merged = dict(generation)
    if not overrides:
        return merged
    unknown = sorted(set(overrides) - set(generation))
    if unknown:
        raise SystemExit(
            f"generation_overrides in {src} sets {unknown}, which the model's generation "
            f"block does not define (has: {sorted(generation)})"
        )
    merged.update(overrides)
    return merged


def parse_hop_from_id(qid: str | None) -> int | None:
    if not qid:
        return None
    m = _ID_HOP_RX.match(qid)
    return int(m.group("h")) if m else None


# ------------------------------------------------------------------- few-shot


def decomposition_to_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [v.strip() for v in value if isinstance(v, str) and v.strip()]
        return "\n".join(parts)
    return ""


def examples_from_reranked_row(row: dict, mode: str, k: int) -> list[dict]:
    """Take the first k candidates of ``<mode>_top_k`` as few-shot examples."""
    key = f"{mode}_top_k"
    candidates = row.get(key) or []
    query_id = row.get("query_id") or row.get("id") or "<unknown>"
    if len(candidates) < k:
        raise ValueError(
            f"[decomposer] retrieval row for query_id={query_id!r} has only "
            f"{len(candidates)} candidates under '{key}', need at least k={k}. "
            "This usually means the similarity/rerank step produced short top-k "
            "lists. Rebuild the retrieval input with the correct k."
        )
    out: list[dict] = []
    for idx, cand in enumerate(candidates[:k]):
        if not isinstance(cand, dict):
            raise ValueError(
                f"[decomposer] query_id={query_id!r} candidate {idx} in '{key}' "
                f"is not a dict: type={type(cand).__name__}"
            )
        decomp = decomposition_to_text(cand.get("pool_few_shot_decomposition_musique"))
        if not decomp:
            raise ValueError(
                f"[decomposer] query_id={query_id!r} candidate {idx} in '{key}' "
                f"(pool_id={cand.get('pool_id')!r}) is missing "
                "'pool_few_shot_decomposition_musique'. The pool is corrupt: "
                "re-run enrich_pool_decompositions.py and rebuild similarity/rerank."
            )
        out.append({"question": cand.get("pool_question", ""), "decomposition": decomp})
    if len(out) != k:
        raise AssertionError(
            f"[decomposer] query_id={query_id!r} assembled {len(out)} examples but requested k={k}."
        )
    return out


def format_few_shot_examples(examples: list[dict], hop_count: int | None) -> str:
    """Format (question, decomposition) pairs. Omit the hop line when unguided."""
    blocks = []
    for ex in examples:
        if hop_count is not None:
            block = (
                f"Hop count: {hop_count}\n"
                f"Question: {ex['question']}\n"
                f"Decomposition:\n{ex['decomposition']}"
            )
        else:
            block = f"Question: {ex['question']}\nDecomposition:\n{ex['decomposition']}"
        blocks.append(block)
    return "\n\n".join(blocks)


def all_pool_items(few_shot_data: dict) -> list[dict]:
    out: list[dict] = []
    for key in ("1hop", "2hop", "3hop"):
        out.extend(few_shot_data.get(key, []))
    return out


def sample_few_shot_combined(few_shot_data: dict, n: int, rng) -> list[dict]:
    """Random fallback when similarity selection is unavailable."""
    pool = all_pool_items(few_shot_data)
    if len(pool) <= n:
        return pool
    return rng.sample(pool, n)


# --------------------------------------------------------------------- prompts


def split_chat_template(template: str, marker: str) -> tuple[str, str]:
    if marker not in template:
        raise SystemExit(
            f"prompt template missing required marker {marker!r}. "
            "A chat_template decomposer needs the template split into system/user halves."
        )
    system_part, user_part = template.split(marker, 1)
    return system_part.strip(), user_part.strip()


def fill_template(
    template: str,
    *,
    question: str,
    hop_count: int | None,
    few_shot_examples: str,
    unguided_hop_placeholder: str,
) -> str:
    """Fill only the placeholders the template actually contains."""
    values: dict[str, Any] = {}
    if "{question}" in template:
        values["question"] = question
    if "{few_shot_examples}" in template:
        values["few_shot_examples"] = few_shot_examples
    if "{hop_count}" in template:
        values["hop_count"] = hop_count if hop_count is not None else unguided_hop_placeholder
    if not values:
        return template
    return template.format(**values)


def build_chat_messages(
    template: str,
    *,
    marker: str,
    question: str,
    hop_count: int | None,
    few_shot_examples: str,
    unguided_hop_placeholder: str,
) -> list[dict]:
    system_tmpl, user_tmpl = split_chat_template(template, marker)
    user_msg = fill_template(
        user_tmpl,
        question=question,
        hop_count=hop_count,
        few_shot_examples=few_shot_examples,
        unguided_hop_placeholder=unguided_hop_placeholder,
    )
    return [
        {"role": "system", "content": system_tmpl},
        {"role": "user", "content": user_msg},
    ]


# ------------------------------------------------------- step-line stopping rule


def count_step_lines(text: str) -> int:
    """Count *completed* step lines in a generated continuation.

    A step line is a non-empty line that has already been terminated by a newline; the
    text after the last newline is still being written and does not count yet. So
    ``"a\\nb"`` is 1 completed line, ``"a\\nb\\n"`` is 2, and blank lines never count.
    """
    if "\n" not in text:
        return 0
    completed = text.split("\n")[:-1]
    return sum(1 for line in completed if line.strip())


def trim_to_step_lines(text: str, max_step_lines: int) -> str:
    """Keep the first ``max_step_lines`` non-empty lines, dropping anything after them.

    The stopping criterion fires between tokens, so a capped generation can end with a
    partial ninth line. This removes that tail; it is a companion to the stopping rule,
    not a replacement for it.
    """
    kept: list[str] = []
    for line in text.split("\n"):
        if not line.strip():
            continue
        kept.append(line.rstrip())
        if len(kept) >= max_step_lines:
            break
    return "\n".join(kept)


class StepLineStopper:
    """Decide when a generation has produced ``max_step_lines`` step lines.

    Pure logic with no torch dependency so it is testable against a synthetic token
    stream: ``decode`` turns the generated token ids (prompt excluded) into text.
    """

    def __init__(self, max_step_lines: int, decode: Callable[[list[int]], str]) -> None:
        if max_step_lines <= 0:
            raise ValueError(f"max_step_lines must be positive, got {max_step_lines}")
        self.max_step_lines = int(max_step_lines)
        self.decode = decode

    def should_stop(self, generated_ids) -> bool:
        return count_step_lines(self.decode(list(generated_ids))) >= self.max_step_lines


def make_step_line_stopping_criteria(tokenizer, *, prompt_len: int, max_step_lines: int):
    """Wrap :class:`StepLineStopper` as a transformers ``StoppingCriteriaList``."""
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    stopper = StepLineStopper(
        max_step_lines,
        lambda ids: tokenizer.decode(ids, skip_special_tokens=True),
    )

    class _StepLineCriteria(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            done = [stopper.should_stop(seq[prompt_len:].tolist()) for seq in input_ids]
            return torch.tensor(done, dtype=torch.bool, device=input_ids.device)

    return StoppingCriteriaList([_StepLineCriteria()])


def post_process(response: str, post_cfg: dict) -> str:
    text = response
    if post_cfg.get("strip_think"):
        text = _THINK_RX.sub("", text).strip()
    for marker in post_cfg.get("truncate_at") or []:
        text = text.split(marker)[0]
    return text.strip()


# ----------------------------------------------------------------------- model


def load_model(model_id: str, loader: dict, device: str, quantization: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    trust_remote_code = bool(require(loader, "trust_remote_code"))
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)

    model_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if quantization in ("4bit", "8bit"):
        if device != "cuda":
            raise SystemExit(f"quantization={quantization} requires CUDA but device={device}")
        from transformers import BitsAndBytesConfig

        if quantization == "4bit":
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
        model_kwargs["quantization_config"] = bnb_cfg
        model_kwargs["device_map"] = require(loader, "device_map_cuda")
    elif quantization == "none":
        dtype_name = require(loader, "cuda_dtype") if device == "cuda" else require(loader, "cpu_dtype")
        # `dtype=` is the forward-compatible spelling; transformers 5.x accepts both.
        model_kwargs["dtype"] = getattr(torch, dtype_name)
        if device == "cuda":
            model_kwargs["device_map"] = require(loader, "device_map_cuda")
    else:
        raise SystemExit(f"unknown quantization {quantization!r} (expected none, 4bit or 8bit)")

    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    except Exception as exc:
        raise SystemExit(
            f"[decomposer] failed to load model {model_id}: {exc}\n"
            "Hint: a very new architecture may need a newer transformers release; if 4-bit "
            "loading is the problem, retry with --quantization 8bit or --quantization none."
        ) from exc

    if device != "cuda" and quantization == "none":
        model = model.to(device)
    model.eval()
    return tokenizer, model


def generate(
    prompt_text: str,
    model,
    tokenizer,
    device: str,
    generation: dict,
    *,
    max_step_lines: int | None = None,
) -> str:
    import torch

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]
    stopping_criteria = (
        make_step_line_stopping_criteria(
            tokenizer, prompt_len=prompt_len, max_step_lines=max_step_lines
        )
        if max_step_lines
        else None
    )
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=int(require(generation, "max_new_tokens")),
            temperature=float(require(generation, "temperature")),
            top_p=float(require(generation, "top_p")),
            do_sample=bool(require(generation, "do_sample")),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria,
        )
    text = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip()
    if max_step_lines:
        text = trim_to_step_lines(text, max_step_lines)
    return text.strip()


# ------------------------------------------------------------------------ main


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Model folder under components/decomposer/models/")
    p.add_argument("--config", default="decomposer.json", help="Shared decomposer config")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--guided", action="store_true", default=None, help="Put the hop count in the prompt")
    p.add_argument(
        "--condition",
        default=None,
        help="Named condition from the config's 'conditions' block (e.g. unguided, "
        "oracle_guided, unguided_capped). Overrides the config's default 'condition'.",
    )
    p.add_argument("--sample-size", type=int, default=None)
    p.add_argument("--embed-model", default=None, help="Key in decomposer.json embed_models")
    p.add_argument("--retrieval-input", default=None, help="Reranked/truncated top-k JSONL")
    p.add_argument("--retrieval-mode", default=None, help="Which <mode>_top_k list to use")
    p.add_argument("--retrieval-k", type=int, default=None)
    p.add_argument("--quantization", default=None, choices=["none", "4bit", "8bit"])
    p.add_argument("--output-root", default=None)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Assemble prompts and write artifacts without loading a model or generating.",
    )
    p.add_argument("--dry-run-limit", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    cfg = load_config(args.config)
    paths_cfg = load_paths(require(cfg, "paths_config"))
    limits = load_limits(require(cfg, "model_limits_config"))

    models_root = resolve_path(require(paths_cfg, "repo.decomposer_models_dir"), _REPO_ROOT)
    model_dir = models_root / args.model
    if not model_dir.is_dir():
        raise SystemExit(f"model folder not found: {model_dir}")
    model_cfg = load_config(model_dir / "config.json")

    condition_name, condition = resolve_condition(cfg, args.condition)

    seed = args.seed if args.seed is not None else int(require(cfg, "seed"))
    guided = resolve_guided(args.guided, condition_name, condition, cfg)
    stop_after_step_lines = resolve_step_line_cap(condition_name, condition)
    sample_size = args.sample_size if args.sample_size is not None else require(cfg, "sample_size")
    embed_key = args.embed_model or require(cfg, "embed_model")
    embed_model_id = require(cfg, f"embed_models.{embed_key}")
    retrieval_input = args.retrieval_input or require(cfg, "retrieval.input")
    retrieval_mode = args.retrieval_mode or require(cfg, "retrieval.mode")
    retrieval_k = args.retrieval_k if args.retrieval_k is not None else int(require(cfg, "retrieval.k"))
    retrieval_modes = require(cfg, "retrieval.modes")
    if retrieval_mode not in retrieval_modes:
        raise SystemExit(f"retrieval mode {retrieval_mode!r} not in {retrieval_modes}")
    quantization = args.quantization or require(model_cfg, "loader.quantization")
    hops = [int(h) for h in require(cfg, "hops")]
    unguided_hop_placeholder = require(cfg, "unguided_hop_placeholder")

    prompt_style = require(model_cfg, "prompt_style")
    generation_overrides = optional(cfg, "generation_overrides")
    generation = apply_generation_overrides(
        dict(require(model_cfg, "generation")),
        generation_overrides,
        cfg.get("_config_path", "<config>"),
    )
    loader = dict(require(model_cfg, "loader"))
    few_shot_cfg = dict(require(model_cfg, "few_shot"))
    post_cfg = dict(require(model_cfg, "post_process"))
    prompt_log_every = int(require(model_cfg, "logging.prompt_log_every"))

    device = "cpu"
    if not args.dry_run:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    current_run_id = run_id()
    seeded = set_global_seed(seed)

    output_root = (
        Path(args.output_root)
        if args.output_root is not None
        else runs_path(paths_cfg, require(cfg, "output_subdir"), args.model)
    )
    output_dir = output_root / current_run_id

    # Prompt file: the unguided variant when one exists and the run is unguided.
    unguided_prompt_file = require(model_cfg, "unguided_prompt_file")
    prompt_file = require(model_cfg, "prompt_file")
    if not guided and unguided_prompt_file:
        prompt_file = unguided_prompt_file
    prompt_path = model_dir / prompt_file
    if not prompt_path.exists():
        raise SystemExit(f"prompt file not found: {prompt_path}")
    prompt_template = prompt_path.read_text(encoding="utf-8")
    if not guided and optional(cfg, "unguided_prompt_must_omit_hop_count"):
        assert_unguided_prompt_omits_hop_count(
            prompt_template,
            prompt_path=prompt_path,
            model=args.model,
            config_src=cfg.get("_config_path", "<config>"),
        )

    chat_marker = None
    if prompt_style == "chat_template":
        chat_marker = require(model_cfg, "chat_template.split_marker")
        split_chat_template(prompt_template, chat_marker)  # fail fast on a bad prompt
    elif prompt_style != "plain":
        raise SystemExit(f"unknown prompt_style {prompt_style!r} (expected plain or chat_template)")

    snapshot = {
        "script": Path(__file__).name,
        "created_utc": now_iso(),
        "run_id": current_run_id,
        "component": "decomposer",
        "model": args.model,
        "model_id": require(model_cfg, "model_id"),
        "model_name": require(model_cfg, "model_name"),
        "prompt_style": prompt_style,
        "prompt_file": prompt_file,
        "prompt_path": str(prompt_path),
        "condition": condition_name,
        "condition_settings": condition,
        "guided": guided,
        "stop_after_step_lines": stop_after_step_lines,
        "seed": seed,
        "seeded": seeded,
        "sample_size": sample_size,
        "hops": hops,
        "questions_template_key": require(cfg, "questions_template_key"),
        "questions_format": require(cfg, "questions_format"),
        "device": device,
        "quantization": quantization,
        "embed_model": embed_key,
        "embed_model_id": embed_model_id,
        "retrieval": {
            "input": str(retrieval_input) if retrieval_input else None,
            "mode": retrieval_mode,
            "k": retrieval_k,
        },
        "generation": generation,
        "generation_overrides": generation_overrides,
        "loader": {**loader, "quantization": quantization},
        "few_shot": few_shot_cfg,
        "post_process": post_cfg,
        "shared_config": cfg.get("_config_path"),
        "model_config": model_cfg.get("_config_path"),
        "output_root": str(output_root),
        "dry_run": args.dry_run,
    }
    print(
        f"Starting decomposer run {current_run_id} (condition={condition_name}, "
        f"guided={guided}, stop_after_step_lines={stop_after_step_lines}, "
        f"dry_run={args.dry_run})"
    )
    print(json.dumps(snapshot, indent=2, default=str))

    data_root = Path(paths_cfg["data_root_resolved"])

    # ---- few-shot machinery (only when the prompt actually takes examples) ----
    few_shot_enabled = bool(require(few_shot_cfg, "enabled")) and "{few_shot_examples}" in prompt_template
    few_shot_k = int(require(few_shot_cfg, "k"))
    few_shot_data: dict = {}
    decomposer_items: list[dict] = []
    decomposer_embeddings = None
    embed_model = None
    embed_size_record: dict[str, Any] | None = None
    mask_fn: Callable[[str], str] | None = None
    few_shot_source_mode = "disabled"

    if few_shot_enabled:
        pool_path = resolve_path(require(paths_cfg, "repo." + require(cfg, "few_shot_pool_key")), _REPO_ROOT)
        if pool_path.exists():
            few_shot_data = json.loads(pool_path.read_text(encoding="utf-8"))
        few_shot_source_mode = "reranked" if retrieval_input else "similarity_or_random"

        if not retrieval_input and not args.dry_run and pool_path.exists():
            from entity_masking import build_masker_from_config
            from pool_embeddings import get_decomposer_pool_embeddings

            mask_cfg = load_config(require(cfg, "masking_config"))
            kb_path = resolve_path(
                require(paths_cfg, "datasets." + require(mask_cfg, "kb_path_key")), data_root
            )
            corpus_paths = [
                resolve_path(p, data_root) for p in require(mask_cfg, "corpus_data_paths")
            ] + [resolve_path(p, _REPO_ROOT) for p in require(mask_cfg, "corpus_repo_paths")]
            mask_fn = build_masker_from_config(
                mask_cfg, kb_path=kb_path, corpus_paths=corpus_paths, corpus_root=data_root
            )
            cache_dir = resolve_path(
                require(paths_cfg, "datasets." + require(cfg, "embedding_cache_key")), data_root
            )
            print(f"Loading decomposer pool embeddings ({embed_key}, masked)...")
            decomposer_items, decomposer_embeddings, embed_model = get_decomposer_pool_embeddings(
                pool_path, cache_dir=cache_dir, model_id=embed_model_id
            )
            # The bi-encoder is a loaded model too: assert it against the ceiling, the
            # same way check_pool_coverage.py and test_similarity_router.py do. Without
            # this, the similarity path was the one model load in the pipeline that
            # escaped the check.
            embed_size_record = assert_within_ceiling(
                embed_model, component="retrieval", model_id=embed_model_id, limits=limits
            )

    # ---- inference rows ----
    inference_rows: list[dict] = []
    if retrieval_input:
        retrieval_path = Path(retrieval_input)
        if not retrieval_path.is_absolute():
            retrieval_path = _REPO_ROOT / retrieval_path
        if not retrieval_path.exists():
            raise SystemExit(f"retrieval input not found: {retrieval_path}")
        rows = load_jsonl(retrieval_path)
        if not rows:
            raise SystemExit(f"no rows in retrieval input: {retrieval_path}")
        hop_fallback = int(require(cfg, "retrieval.hop_fallback"))
        for row in rows:
            question = row.get("query_question")
            if not isinstance(question, str) or not question.strip():
                continue
            hop = parse_hop_from_id(row.get("query_id"))
            if hop is None:
                hop = hop_fallback
            examples = (
                examples_from_reranked_row(row, retrieval_mode, retrieval_k)
                if few_shot_enabled
                else []
            )
            inference_rows.append(
                {
                    "query_id": row.get("query_id"),
                    "question": question,
                    "hop_count": hop,
                    "retrieval_examples": examples,
                }
            )
        if not inference_rows:
            raise SystemExit(f"retrieval input has no valid query_question rows: {retrieval_path}")
        if sample_size:
            inference_rows = inference_rows[: int(sample_size)]
        print(
            f"Loaded {len(inference_rows)} retrieval rows from {retrieval_path} "
            f"(mode={retrieval_mode}, k={retrieval_k})"
        )
    else:
        template = require(paths_cfg, "datasets." + require(cfg, "questions_template_key"))
        questions_format = require(cfg, "questions_format")
        question_field = id_field = ""
        if questions_format == "jsonl":
            question_field = require(cfg, "questions_jsonl.question_field")
            id_field = require(cfg, "questions_jsonl.id_field")
        rng = new_rng(seed)
        for hop in hops:
            items = load_question_items(
                resolve_path(template.format(hop=hop), data_root),
                questions_format=questions_format,
                question_field=question_field,
                id_field=id_field,
            )
            if sample_size:
                items = rng.sample(items, min(len(items), int(sample_size)))
            for item in items:
                inference_rows.append(
                    {
                        "query_id": item["query_id"],
                        "question": item["question"],
                        # Guided runs inject this hop count: it is the gold depth of the
                        # file the question was read from, not a model prediction.
                        "hop_count": hop,
                        "retrieval_examples": [],
                    }
                )
        if not inference_rows:
            raise SystemExit(
                f"no questions loaded from {data_root} (expected {template} for hops {hops}); "
                "set data_root in configs/paths.json"
            )
        print(f"Loaded {len(inference_rows)} total questions.")

    # Measured before any dry-run truncation: this is what the arm was actually asked to
    # decompose. Three conditions are only comparable if these numbers match across them
    # (for the pinned MuSiQue set of ADR 0007: 200 per hop for hops 2/3/4, 600 total).
    rows_loaded_total = len(inference_rows)
    rows_loaded_per_hop = {
        str(hop): sum(1 for r in inference_rows if r["hop_count"] == hop)
        for hop in sorted({r["hop_count"] for r in inference_rows})
    }

    if args.dry_run:
        inference_rows = inference_rows[: max(0, args.dry_run_limit)]

    # ---- model ----
    model = tokenizer = None
    size_record = unasserted_note("decomposer", require(model_cfg, "model_id"))
    if not args.dry_run:
        model_id = require(model_cfg, "model_id")
        print(f"Loading model: {model_id} on {device} (quantization={quantization}) ...")
        tokenizer, model = load_model(model_id, loader, device, quantization)
        size_record = assert_within_ceiling(
            model, component="decomposer", model_id=model_id, limits=limits
        )

    # ---- inference ----
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir = output_dir / "prompts_log"
    prompts_dir.mkdir(exist_ok=True)

    results: list[dict] = []
    fallback_rng = new_rng(seed)
    print("Assembling prompts..." if args.dry_run else "Running inference...")
    progress_every = int(require(cfg, "progress_every"))

    for i, row in enumerate(inference_rows):
        question = row["question"]
        hop = row["hop_count"]
        if (i + 1) % progress_every == 0:
            print(f"Processed {i + 1}/{len(inference_rows)}...")

        hop_input = hop if guided else None
        sampled_with_scores: list[tuple[dict, float]] = []
        retrieved = row.get("retrieval_examples") or []
        sampled: list[dict] = []
        source = "none"

        if few_shot_enabled:
            if retrieved:
                sampled = retrieved
                source = "reranked"
            elif decomposer_items and decomposer_embeddings is not None and embed_model and mask_fn:
                from pool_embeddings import top_k_similar_decomposer

                similar = top_k_similar_decomposer(
                    mask_fn(question),
                    decomposer_items,
                    decomposer_embeddings,
                    embed_model,
                    model_id=embed_model_id,
                    k=few_shot_k,
                )
                sampled = [it for it, _ in similar]
                sampled_with_scores = similar
                source = "similarity"
            elif few_shot_data:
                sampled = sample_few_shot_combined(few_shot_data, few_shot_k, fallback_rng)
                source = "random"

        few_shot_str = format_few_shot_examples(sampled, hop_input) if sampled else ""

        if prompt_style == "chat_template":
            messages = build_chat_messages(
                prompt_template,
                marker=chat_marker,
                question=question,
                hop_count=hop_input,
                few_shot_examples=few_shot_str,
                unguided_hop_placeholder=unguided_hop_placeholder,
            )
            if args.dry_run:
                rendered = json.dumps(messages, ensure_ascii=False, indent=2)
                decomposition = ""
            else:
                rendered = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=bool(require(model_cfg, "chat_template.enable_thinking")),
                )
                decomposition = post_process(
                    generate(
                        rendered,
                        model,
                        tokenizer,
                        device,
                        generation,
                        max_step_lines=stop_after_step_lines,
                    ),
                    post_cfg,
                )
        else:
            rendered = fill_template(
                prompt_template,
                question=question,
                hop_count=hop_input,
                few_shot_examples=few_shot_str,
                unguided_hop_placeholder=unguided_hop_placeholder,
            )
            decomposition = (
                ""
                if args.dry_run
                else post_process(
                    generate(
                        rendered,
                        model,
                        tokenizer,
                        device,
                        generation,
                        max_step_lines=stop_after_step_lines,
                    ),
                    post_cfg,
                )
            )

        if args.dry_run or (i + 1) % prompt_log_every == 0:
            log_path = prompts_dir / f"prompt_idx{i + 1:04d}_hop{hop}.txt"
            masked_q = mask_fn(question) if mask_fn else "N/A"
            header = [
                "--- Log Header ---",
                f"Question (original): {question}",
                f"Question (masked): {masked_q}",
                f"Few-shot source: {source} (k={len(sampled)})",
            ]
            for j, (item, score) in enumerate(sampled_with_scores, start=1):
                header.append(
                    f"  {j}. sim={score:.4f} | masked={item.get('masked')} | question={item.get('question')}"
                )
            if not sampled_with_scores:
                header.append("  (no similarity scores available)")
            log_path.write_text(
                "\n".join(header)
                + f"\n\n--- Prompt ({prompt_style}) ---\n"
                + rendered
                + "\n--- Response ---\n"
                + decomposition
                + "\n",
                encoding="utf-8",
            )

        results.append(
            {
                "query_id": row.get("query_id"),
                "question": question,
                "hop_count": hop,
                "decomposition": decomposition,
                "few_shot_source": source,
            }
        )

    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    empty = sum(1 for r in results if not r["decomposition"])
    metrics = {
        "dry_run": args.dry_run,
        "total_rows": len(results),
        "rows_loaded_total": rows_loaded_total,
        "rows_loaded_per_hop": rows_loaded_per_hop,
        "rows_with_empty_decomposition": empty,
        "few_shot_enabled": few_shot_enabled,
        "few_shot_source_mode": few_shot_source_mode,
        "few_shot_source_counts": {
            src: sum(1 for r in results if r["few_shot_source"] == src)
            for src in sorted({r["few_shot_source"] for r in results})
        },
        "condition": condition_name,
        "guided": guided,
        "stop_after_step_lines": stop_after_step_lines,
        # How many decompositions came out at the cap (i.e. the stopping rule bound them).
        "rows_at_step_line_cap": (
            sum(
                1
                for r in results
                if count_step_lines(r["decomposition"] + "\n") >= stop_after_step_lines
            )
            if stop_after_step_lines and not args.dry_run
            else None
        ),
        "seed": seed,
        "model_size": size_record,
        "embedding_model_size": embed_size_record,
        "results_path": str(output_dir / "results.json"),
    }
    if embed_size_record is None:
        metrics["embedding_model_size_note"] = (
            "no bi-encoder was loaded in this run (retrieval input supplied, few-shot "
            "disabled, or --dry-run), so its parameter count is unmeasured"
        )
    if args.dry_run:
        metrics["decomposition_quality"] = None
        metrics["decomposition_quality_note"] = (
            "unmeasured: --dry-run assembles prompts only. Score predictions with "
            "scripts/musique_decompositions_evaluator.py."
        )

    write_run_artifacts(
        output_dir,
        config_snapshot=snapshot,
        metrics=metrics,
        note_title=f"Decomposer {'dry run' if args.dry_run else 'run'} - {current_run_id}",
        note_lines=[
            f"- Model folder: `{args.model}`"
            + ("" if not args.dry_run else " (model not loaded)"),
            f"- Prompt: `{prompt_path}` (style: {prompt_style})",
            f"- Condition: {condition_name or 'none (no conditions block)'}; guided: {guided}; "
            f"step-line cap: {stop_after_step_lines or 'none'}; seed: {seed}",
            f"- Rows: {len(results)} of {rows_loaded_total} loaded "
            f"(per hop: {rows_loaded_per_hop})"
            + (f"; retrieval input: `{retrieval_input}`" if retrieval_input else ""),
            f"- Few-shot: enabled={few_shot_enabled} k={few_shot_k} mode={few_shot_source_mode}",
            (
                f"- Parameters: {size_record['parameter_count']:,} "
                f"(ceiling {size_record['parameter_ceiling']:,})"
                if size_record["ceiling_asserted"]
                else "- Parameter ceiling: not asserted (no model was loaded)."
            ),
            f"- Predictions: `{output_dir / 'results.json'}`",
        ],
    )
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
