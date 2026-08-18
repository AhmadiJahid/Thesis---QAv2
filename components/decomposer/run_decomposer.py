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

    # the fine-tuned arm (issue #13): a LoRA adapter on the same base model, zero-shot.
    # --no-few-shot is not optional here - see check_adapter_few_shot_combination.
    python components/decomposer/run_decomposer.py --model mistral_7b_instruct \\
        --adapter runs/finetune_decomposer/pool_2000/mistral_7b_instruct/<run>/adapter \\
        --no-few-shot --retrieval-input <the evaluation-set query file>

Every run writes a config snapshot, a metrics JSON and a run note, and asserts the
model's parameter count against the ceiling in ``configs/model_limits.json`` (with a LoRA
adapter attached, the count asserted is the base plus the adapter).

Every run also reports **cost next to quality**: per row, the prompt tokens, the completion
tokens and the generation latency; in the metrics JSON, the per-query means and medians.
That is what makes a prompting-versus-fine-tuning comparison a cost/quality comparison
instead of a quality-only one. On a ``--dry-run`` nothing is generated, so those numbers are
recorded as unmeasured rather than as zero.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from model_size import assert_within_ceiling, load_limits, unasserted_note  # noqa: E402
from run_artifacts import now_iso, run_id, write_run_artifacts  # noqa: E402
from run_config import (  # noqa: E402
    load_config,
    load_paths,
    require,
    resolve_path,
    runs_path,
)
from seeding import new_rng, set_global_seed  # noqa: E402

_THINK_RX = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_ID_HOP_RX = re.compile(r"^(?P<h>\d+)hop")


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


def attach_adapter(model, adapter_path: str | Path):
    """Attach a trained LoRA adapter (the fine-tuned arm of issue #13).

    Loaded on top of the same base model the prompting arm uses, so the two arms differ in
    the adapter and nothing else.
    """
    path = Path(adapter_path)
    if not path.exists():
        raise SystemExit(f"[decomposer] adapter not found: {path}")
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise SystemExit(
            "[decomposer] --adapter needs peft (in requirements.txt: peft==0.20.0). "
            f"Import failed: {exc}"
        ) from exc
    model = PeftModel.from_pretrained(model, str(path))
    model.eval()
    return model


#: The loud opt-out for running an adapter with few-shot examples anyway. Named so that it
#: cannot appear in a command line by accident, and so that it is visible in the run's
#: config snapshot.
ADAPTER_FEW_SHOT_OVERRIDE_FLAG = "--adapter-with-few-shot-i-know"


def check_adapter_few_shot_combination(
    *, adapter: str | None, no_few_shot: bool, override: bool
) -> dict[str, Any]:
    """Refuse ``--adapter`` without ``--no-few-shot`` (issue #13), unless overridden.

    Two things go wrong when few-shot examples are injected into an adapter's prompt. The
    adapter was fine-tuned on the zero-shot prompt (``train_lora.py`` renders the same
    template with the few-shot block empty), so it meets a prompt shape it never saw. And the
    examples come from the MuSiQue training pool the adapter was trained on - with
    ``--retrieval-input`` the candidates carry ``pool_few_shot_decomposition_musique`` from
    that pool - so the model can be shown its own training rows at inference.

    Returns a record for the run's config snapshot; raises ``SystemExit`` on the refused
    combination.
    """
    record = {
        "adapter": str(adapter) if adapter else None,
        "no_few_shot": bool(no_few_shot),
        "adapter_few_shot_override": bool(override),
    }
    if not adapter:
        return record
    if not no_few_shot and not override:
        raise SystemExit(
            "[decomposer] REFUSING TO RUN: --adapter without --no-few-shot.\n"
            "The adapter was fine-tuned on the zero-shot prompt (components/decomposer/"
            "train_lora.py builds it from this same template with the few-shot block empty), "
            "and the examples that would be injected come from the MuSiQue training pool the "
            "adapter trained on - so the run would both feed the model a prompt shape it "
            "never saw and risk showing it its own training rows.\n"
            "Pass --no-few-shot (this is the fine-tuned arm as specified), or "
            f"{ADAPTER_FEW_SHOT_OVERRIDE_FLAG} if you deliberately want few-shot on top of "
            "the adapter and will report the run as that, not as the fine-tuned arm."
        )
    if override and not no_few_shot:
        print(
            "WARNING: running --adapter WITH few-shot examples "
            f"({ADAPTER_FEW_SHOT_OVERRIDE_FLAG}). The prompt shape differs from the one the "
            "adapter was trained on, and the examples come from its own training pool: this "
            "run is not the fine-tuned arm of the comparison. It is recorded as "
            "adapter_few_shot_override: true in the config snapshot and the metrics."
        )
    return record


def generate(prompt_text: str, model, tokenizer, device: str, generation: dict) -> dict[str, Any]:
    """Generate one decomposition, returning the text with its token and latency cost.

    The cost fields are measured here rather than estimated later: ``prompt_tokens`` and
    ``completion_tokens`` are the tokenizer's own counts for this call, and
    ``latency_seconds`` is wall clock around ``model.generate`` only (tokenization and
    decoding excluded, so the number is comparable across arms).
    """
    import torch

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    prompt_tokens = int(inputs["input_ids"].shape[1])
    started = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=int(require(generation, "max_new_tokens")),
            temperature=float(require(generation, "temperature")),
            top_p=float(require(generation, "top_p")),
            do_sample=bool(require(generation, "do_sample")),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    if device == "cuda":
        # generate() is synchronous on the returned tensor, but sync explicitly so the
        # timing is the kernel time and not the launch time.
        torch.cuda.synchronize()
    latency_seconds = time.perf_counter() - started
    new_tokens = outputs[0][prompt_tokens:]
    return {
        "text": tokenizer.decode(new_tokens, skip_special_tokens=True).strip(),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": int(new_tokens.shape[0]),
        "latency_seconds": latency_seconds,
    }


def cost_summary(results: list[dict]) -> dict[str, Any]:
    """Per-query token and latency cost over the rows that actually generated.

    Nothing is imputed: rows with no measurement (a dry run) are excluded, and when there
    are none every field is ``None`` with a note, because 0 tokens per query would be a
    false claim rather than a missing one.
    """
    measured = [r for r in results if r.get("latency_seconds") is not None]

    def mean(key: str) -> float | None:
        return statistics.fmean(float(r[key]) for r in measured) if measured else None

    def median(key: str) -> float | None:
        return statistics.median(float(r[key]) for r in measured) if measured else None

    summary: dict[str, Any] = {
        "rows_measured": len(measured),
        "rows_total": len(results),
        "mean_prompt_tokens_per_query": mean("prompt_tokens"),
        "median_prompt_tokens_per_query": median("prompt_tokens"),
        "mean_completion_tokens_per_query": mean("completion_tokens"),
        "median_completion_tokens_per_query": median("completion_tokens"),
        "mean_total_tokens_per_query": (
            statistics.fmean(
                float(r["prompt_tokens"]) + float(r["completion_tokens"]) for r in measured
            )
            if measured
            else None
        ),
        "mean_latency_seconds_per_query": mean("latency_seconds"),
        "median_latency_seconds_per_query": median("latency_seconds"),
        "total_generation_seconds": (
            sum(float(r["latency_seconds"]) for r in measured) if measured else None
        ),
        "definitions": {
            "prompt_tokens": "tokenizer token count of the rendered prompt for that row",
            "completion_tokens": "number of newly generated tokens for that row",
            "latency_seconds": (
                "wall clock around model.generate for that row (CUDA synchronized), "
                "excluding tokenization and decoding"
            ),
            "excluded_rows": "rows without a measurement (e.g. a --dry-run) are excluded",
        },
    }
    if not measured:
        summary["note"] = (
            "unmeasured: no row generated in this run (--dry-run), so tokens per query and "
            "latency per query are unknown, not zero"
        )
    return summary


# ------------------------------------------------------------------------ main


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Model folder under components/decomposer/models/")
    p.add_argument("--config", default="decomposer.json", help="Shared decomposer config")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--guided", action="store_true", default=None, help="Put the hop count in the prompt")
    p.add_argument("--sample-size", type=int, default=None)
    p.add_argument("--embed-model", default=None, help="Key in decomposer.json embed_models")
    p.add_argument("--retrieval-input", default=None, help="Reranked/truncated top-k JSONL")
    p.add_argument("--retrieval-mode", default=None, help="Which <mode>_top_k list to use")
    p.add_argument("--retrieval-k", type=int, default=None)
    p.add_argument("--quantization", default=None, choices=["none", "4bit", "8bit"])
    p.add_argument(
        "--adapter",
        default=None,
        help="Trained LoRA adapter directory (the fine-tuned arm; see train_lora.py).",
    )
    p.add_argument(
        "--no-few-shot",
        action="store_true",
        help="Leave the prompt's few-shot block empty. Required with --adapter, which was "
        "trained on the zero-shot prompt: injecting examples at inference would feed it a "
        "prompt shape it never saw, from the pool it trained on.",
    )
    p.add_argument(
        ADAPTER_FEW_SHOT_OVERRIDE_FLAG,
        action="store_true",
        help="Deliberately run --adapter WITH few-shot examples. Refused by default (see "
        "check_adapter_few_shot_combination); such a run is not the fine-tuned arm of the "
        "comparison and is recorded as an override.",
    )
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

    # Before anything is loaded: an adapter run with few-shot examples is refused here, so a
    # multi-hour evaluation cannot produce a run that is not the arm it claims to be.
    adapter_record = check_adapter_few_shot_combination(
        adapter=args.adapter,
        no_few_shot=bool(args.no_few_shot),
        override=bool(args.adapter_with_few_shot_i_know),
    )

    cfg = load_config(args.config)
    paths_cfg = load_paths(require(cfg, "paths_config"))
    limits = load_limits(require(cfg, "model_limits_config"))

    models_root = resolve_path(require(paths_cfg, "repo.decomposer_models_dir"), _REPO_ROOT)
    model_dir = models_root / args.model
    if not model_dir.is_dir():
        raise SystemExit(f"model folder not found: {model_dir}")
    model_cfg = load_config(model_dir / "config.json")

    seed = args.seed if args.seed is not None else int(require(cfg, "seed"))
    guided = args.guided if args.guided is not None else bool(require(cfg, "guided"))
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
    generation = dict(require(model_cfg, "generation"))
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
        "guided": guided,
        "seed": seed,
        "seeded": seeded,
        "sample_size": sample_size,
        "hops": hops,
        "device": device,
        "quantization": quantization,
        **adapter_record,
        "embed_model": embed_key,
        "embed_model_id": embed_model_id,
        "retrieval": {
            "input": str(retrieval_input) if retrieval_input else None,
            "mode": retrieval_mode,
            "k": retrieval_k,
        },
        "generation": generation,
        "loader": {**loader, "quantization": quantization},
        "few_shot": few_shot_cfg,
        "post_process": post_cfg,
        "shared_config": cfg.get("_config_path"),
        "model_config": model_cfg.get("_config_path"),
        "output_root": str(output_root),
        "dry_run": args.dry_run,
    }
    print(
        f"Starting decomposer run {current_run_id} (guided={guided}, "
        f"adapter={args.adapter}, no_few_shot={bool(args.no_few_shot)}, "
        f"dry_run={args.dry_run})"
    )
    print(json.dumps(snapshot, indent=2, default=str))

    data_root = Path(paths_cfg["data_root_resolved"])

    # ---- few-shot machinery (only when the prompt actually takes examples) ----
    few_shot_enabled = (
        bool(require(few_shot_cfg, "enabled"))
        and "{few_shot_examples}" in prompt_template
        and not args.no_few_shot
    )
    few_shot_k = int(require(few_shot_cfg, "k"))
    few_shot_data: dict = {}
    decomposer_items: list[dict] = []
    decomposer_embeddings = None
    embed_model = None
    embed_size_record: dict[str, Any] | None = None
    mask_fn: Callable[[str], str] | None = None
    few_shot_source_mode = "disabled_by_no_few_shot" if args.no_few_shot else "disabled"
    # A model whose prompt carries its examples inline (few_shot.enabled false, no
    # "{few_shot_examples}" placeholder) cannot have them removed by a flag. Say so rather
    # than let a run be labelled zero-shot when its prompt is not.
    no_few_shot_ineffective = bool(
        args.no_few_shot and "{few_shot_examples}" not in prompt_template
    )
    if no_few_shot_ineffective:
        print(
            f"WARNING: --no-few-shot has no effect for model {args.model!r}: its prompt "
            f"({prompt_file}) has no '{{few_shot_examples}}' placeholder, so any examples "
            "it shows are written into the prompt text itself."
        )

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
        rng = new_rng(seed)
        for hop in hops:
            questions = load_questions(resolve_path(template.format(hop=hop), data_root))
            if sample_size:
                questions = rng.sample(questions, min(len(questions), int(sample_size)))
            for question in questions:
                inference_rows.append(
                    {"query_id": None, "question": question, "hop_count": hop, "retrieval_examples": []}
                )
        if not inference_rows:
            raise SystemExit(
                f"no questions loaded from {data_root} (expected {template} for hops {hops}); "
                "set data_root in configs/paths.json"
            )
        print(f"Loaded {len(inference_rows)} total questions.")

    if args.dry_run:
        inference_rows = inference_rows[: max(0, args.dry_run_limit)]

    # ---- model ----
    model = tokenizer = None
    size_record = unasserted_note("decomposer", require(model_cfg, "model_id"))
    if not args.dry_run:
        model_id = require(model_cfg, "model_id")
        print(f"Loading model: {model_id} on {device} (quantization={quantization}) ...")
        tokenizer, model = load_model(model_id, loader, device, quantization)
        if args.adapter:
            print(f"Attaching LoRA adapter: {args.adapter}")
            model = attach_adapter(model, args.adapter)
            model_id = f"{model_id}+adapter"
        # With an adapter attached this counts base + adapter parameters, which is the
        # thing the ~8B ceiling is about.
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
        cost: dict[str, Any] = {
            "prompt_tokens": None,
            "completion_tokens": None,
            "latency_seconds": None,
        }

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
                generated = generate(rendered, model, tokenizer, device, generation)
                cost = {k: generated[k] for k in cost}
                decomposition = post_process(generated["text"], post_cfg)
        else:
            rendered = fill_template(
                prompt_template,
                question=question,
                hop_count=hop_input,
                few_shot_examples=few_shot_str,
                unguided_hop_placeholder=unguided_hop_placeholder,
            )
            if args.dry_run:
                decomposition = ""
            else:
                generated = generate(rendered, model, tokenizer, device, generation)
                cost = {k: generated[k] for k in cost}
                decomposition = post_process(generated["text"], post_cfg)

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
                **cost,
            }
        )

    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    empty = sum(1 for r in results if not r["decomposition"])
    metrics = {
        "dry_run": args.dry_run,
        "total_rows": len(results),
        "rows_with_empty_decomposition": empty,
        "few_shot_enabled": few_shot_enabled,
        "few_shot_source_mode": few_shot_source_mode,
        "few_shot_source_counts": {
            src: sum(1 for r in results if r["few_shot_source"] == src)
            for src in sorted({r["few_shot_source"] for r in results})
        },
        "guided": guided,
        "seed": seed,
        **adapter_record,
        "no_few_shot_ineffective": no_few_shot_ineffective,
        "model_size": size_record,
        "embedding_model_size": embed_size_record,
        # Cost sits in the same metrics file as quality on purpose: a fine-tuned arm that
        # wins on step F1 while costing three times the tokens is a different conclusion
        # from one that wins for free, and that can only be argued if both are recorded.
        "cost": cost_summary(results),
        "results_path": str(output_dir / "results.json"),
    }
    if no_few_shot_ineffective:
        metrics["no_few_shot_note"] = (
            f"--no-few-shot was passed but the prompt ({prompt_file}) has no "
            "'{few_shot_examples}' placeholder, so this run is not necessarily zero-shot: "
            "any examples the prompt shows are part of its text."
        )
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
            "- Adapter: "
            + (f"`{args.adapter}`" if args.adapter else "none (prompting arm)")
            + (
                f" - run WITH few-shot examples via {ADAPTER_FEW_SHOT_OVERRIDE_FLAG}: this is "
                "not the fine-tuned arm of the comparison."
                if args.adapter and not args.no_few_shot
                else ""
            ),
            f"- Prompt: `{prompt_path}` (style: {prompt_style})",
            f"- Guided: {guided}; seed: {seed}",
            f"- Rows: {len(results)}"
            + (f"; retrieval input: `{retrieval_input}`" if retrieval_input else ""),
            f"- Few-shot: enabled={few_shot_enabled} k={few_shot_k} mode={few_shot_source_mode}",
            (
                f"- Parameters: {size_record['parameter_count']:,} "
                f"(ceiling {size_record['parameter_ceiling']:,})"
                if size_record["ceiling_asserted"]
                else "- Parameter ceiling: not asserted (no model was loaded)."
            ),
            (
                f"- Cost per query: {metrics['cost']['mean_prompt_tokens_per_query']:.1f} prompt "
                f"+ {metrics['cost']['mean_completion_tokens_per_query']:.1f} completion tokens, "
                f"{metrics['cost']['mean_latency_seconds_per_query']:.3f}s "
                f"(means over {metrics['cost']['rows_measured']} rows)"
                if metrics["cost"]["rows_measured"]
                else "- Cost per query: unmeasured (nothing was generated in this run)."
            ),
            f"- Predictions: `{output_dir / 'results.json'}`",
        ],
    )
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
