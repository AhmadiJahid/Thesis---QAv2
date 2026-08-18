#!/usr/bin/env python3
"""Decomposer component: split a question into single-hop sub-questions.

One runner for every model. v1 kept four copies of ``decomposer.py``: two were
byte-identical (a plain prompt with inline examples, no retrieval), one added
similarity/reranked few-shot selection and quantization, and one added a chat
template with ``enable_thinking=False`` plus ``<think>`` stripping. Those
differences are now fields in ``components/decomposer/models/<model>/config.json``.
Guided prompts stay per-model and byte-identical to v1; the two ``prompt_unguided.md``
files are **not** v1's any more - they are derived from their guided sibling by removing the
hop-bearing lines, which is what makes the conditions comparable (see ADR 0012).

Usage::

    python components/decomposer/run_decomposer.py --model mistral_7b_instruct \\
        --retrieval-input runs/pool_sweep/biencoder_top5/<cell>/top5_biencoder.jsonl
    python components/decomposer/run_decomposer.py --model qwen2_5_3b --dry-run
    python components/decomposer/run_decomposer.py --model mistral_7b_instruct \\
        --config decomposer_musique.json --condition unguided_capped \\
        --retrieval-input <the pinned top-5 JSONL over the 600 questions of ADR 0007>

``--config decomposer_musique.json`` runs the pinned MuSiQue evaluation set (ADR 0007) and
carries a ``conditions`` block: ``unguided``, ``oracle_guided`` (gold hop count in the
prompt) and ``unguided_capped`` (no hop count, generation stopped at N step lines). All
three share model, seed, retrieval and decoding; only the prompt's hop information and the
step-line budget differ. That claim is enforced, not asserted in prose: the unguided prompt
must be the guided prompt with its hop-bearing lines removed and nothing else changed
(``unguided_prompt_must_equal_guided_minus_hop_lines``), so the arms cannot differ in a
second way. The unguided arms therefore need a model folder that ships an
``unguided_prompt_file``; ``qwen2_5_3b`` and ``phi_4_mini_instruct`` cannot run them.

Every run writes a config snapshot, a metrics JSON and a run note, and asserts the
model's parameter count against the ceiling in ``configs/model_limits.json``.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
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
from step_lines import (  # noqa: E402
    completed_step_line_count,
    post_process_generation,
    step_line_count,
    trim_to_step_lines,
)

_ID_HOP_RX = re.compile(r"^(?P<h>\d+)hop")
_WS_RX = re.compile(r"\s+")

#: A prompt line that carries hop-count information: the ``{hop_count}`` slot itself, or
#: prose mentioning the hop count (the guided rule "The number of steps MUST equal the hop
#: count."). Removing exactly these lines from the guided prompt is the definition of the
#: unguided prompt (Jahid's plan, prompt 4: three conditions, "everything else held
#: identical", unguided = "no hop count in the prompt").
_HOP_LINE_RX = re.compile(r"\{hop_count\}|hop[\s_-]*count", re.IGNORECASE)

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


def hop_bearing_lines(template: str) -> list[str]:
    """Every line of ``template`` that mentions the hop count (see :data:`_HOP_LINE_RX`)."""
    return [ln for ln in template.splitlines() if _HOP_LINE_RX.search(ln)]


def derive_unguided_template(guided_template: str) -> str:
    """The unguided prompt, by construction: the guided prompt minus its hop-bearing lines.

    Byte-preserving for every other line, line endings included - the two files differ only
    by the removed lines. Nothing is added in their place: an instruction that only the
    unguided arm sees (v1's "Decompose into the minimal number of atomic steps.") would be
    a second difference between the arms, and then a quality gap could not be attributed to
    the hop count.
    """
    return "".join(
        ln for ln in guided_template.splitlines(keepends=True) if not _HOP_LINE_RX.search(ln)
    )


def assert_unguided_prompt_omits_hop_count(
    template: str, *, prompt_path: Path, model: str, config_src: str
) -> None:
    """Refuse an unguided run whose prompt still carries hop-count information.

    Two ways it can: a model folder with no ``unguided_prompt_file`` falls back to the
    guided prompt, where ``{hop_count}`` is filled with ``unguided_hop_placeholder`` (the
    prompt then reads "Hop count: Unknown" under a rule saying the step count must equal
    the hop count - neither arm); or an unguided prompt could hardcode a hop line with no
    placeholder in it, which a ``{hop_count}``-only check would miss. Both are refused when
    the config sets ``unguided_prompt_must_omit_hop_count``.
    """
    offending = hop_bearing_lines(template)
    if not offending:
        return
    raise SystemExit(
        f"unguided run, but the prompt {prompt_path} still carries hop-count information:\n"
        + "\n".join(f"  {ln!r}" for ln in offending)
        + f"\nModel folder {model!r} either has no 'unguided_prompt_file' (so the guided "
        f"prompt was used and the hop count would be filled with the placeholder) or ships "
        f"an unguided prompt that mentions the hop count. {config_src} sets "
        "'unguided_prompt_must_omit_hop_count', which forbids both. Use a model folder whose "
        "unguided prompt is its guided prompt minus the hop-bearing lines."
    )


def assert_unguided_is_guided_minus_hop_lines(
    *,
    guided_template: str,
    unguided_template: str,
    guided_path: Path,
    unguided_path: Path,
    model: str,
    config_src: str,
) -> dict[str, Any]:
    """Refuse when the unguided prompt differs from the guided one by more than hop lines.

    The experiment's whole claim is that the arms differ in hop information only, so any
    other line-level delta - a rule dropped, a rule added, a sentence reworded - has to be
    a loud failure rather than a footnote. Returns the removed lines for the run's snapshot.
    """
    expected = derive_unguided_template(guided_template)
    if unguided_template == expected:
        return {
            "checked": True,
            "guided_prompt": str(guided_path),
            "unguided_prompt": str(unguided_path),
            "hop_lines_removed": hop_bearing_lines(guided_template),
        }
    diff = "\n".join(
        difflib.unified_diff(
            expected.splitlines(),
            unguided_template.splitlines(),
            fromfile=f"{guided_path.name} minus hop lines (expected)",
            tofile=f"{unguided_path.name} (on disk)",
            lineterm="",
        )
    )
    raise SystemExit(
        f"the unguided prompt of model folder {model!r} is not the guided prompt minus its "
        "hop-bearing lines.\n"
        f"  guided:   {guided_path}\n"
        f"  unguided: {unguided_path}\n"
        f"{config_src} sets 'unguided_prompt_must_equal_guided_minus_hop_lines', because the "
        "three conditions are only comparable if the prompt differs in hop information and "
        "nothing else. Residual delta:\n" + diff
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


def normalize_for_self_exclusion(text: str | None) -> str:
    """Lowercased, whitespace-collapsed text, for deciding "this exemplar IS the query"."""
    if not isinstance(text, str):
        return ""
    return _WS_RX.sub(" ", text.strip().lower())


def is_self_example(
    candidate: dict, *, exclude_query_id: str | None, exclude_question: str | None
) -> bool:
    """True when a retrieved exemplar is the query itself (by id or by question text).

    Latent while the few-shot pool is built from MuSiQue *train* and the queries come from
    *dev*, and load-bearing the moment a pool is drawn from the same split: a query
    retrieving its own gold decomposition as an exemplar would be leakage, and it would
    show up as a quality gain, not as an error.
    """
    if exclude_query_id and str(candidate.get("pool_id") or "") == str(exclude_query_id):
        return True
    if exclude_question:
        wanted = normalize_for_self_exclusion(exclude_question)
        if wanted and normalize_for_self_exclusion(candidate.get("pool_question")) == wanted:
            return True
    return False


def examples_from_reranked_row(
    row: dict,
    mode: str,
    k: int,
    *,
    exclude_query_id: str | None = None,
    exclude_question: str | None = None,
) -> tuple[list[dict], int]:
    """Take the first k candidates of ``<mode>_top_k`` as few-shot examples.

    Candidates that are the query itself are dropped first (see :func:`is_self_example`),
    so k exemplars are still assembled from the rest of the ranked list. Returns
    ``(examples, self_excluded_count)``.
    """
    key = f"{mode}_top_k"
    candidates = row.get(key) or []
    query_id = row.get("query_id") or row.get("id") or "<unknown>"
    kept: list[dict] = []
    self_excluded = 0
    for cand in candidates:
        if isinstance(cand, dict) and is_self_example(
            cand, exclude_query_id=exclude_query_id, exclude_question=exclude_question
        ):
            self_excluded += 1
            continue
        kept.append(cand)
    if len(kept) < k:
        raise ValueError(
            f"[decomposer] retrieval row for query_id={query_id!r} has only "
            f"{len(kept)} usable candidates under '{key}' ({len(candidates)} listed, "
            f"{self_excluded} dropped as the query itself), need at least k={k}. "
            "This usually means the similarity/rerank step produced short top-k "
            "lists. Rebuild the retrieval input with the correct k."
        )
    out: list[dict] = []
    for idx, cand in enumerate(kept[:k]):
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
    # The point of the exclusion is that no exemplar is the query; assert it rather than
    # trusting the filter above.
    for ex in out:
        if exclude_question and normalize_for_self_exclusion(
            ex["question"]
        ) == normalize_for_self_exclusion(exclude_question):
            raise AssertionError(
                f"[decomposer] query_id={query_id!r} kept an exemplar identical to the query "
                "after self-exclusion; this is a bug in examples_from_reranked_row."
            )
    return out, self_excluded


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


def sample_few_shot_combined(
    few_shot_data: dict, n: int, rng, *, exclude_question: str | None = None
) -> list[dict]:
    """Random fallback when similarity selection is unavailable.

    Self-exclusion applies here too: the query may not appear as its own exemplar on any
    path, or the same run would leak on one code path and not another.
    """
    pool = all_pool_items(few_shot_data)
    if exclude_question:
        wanted = normalize_for_self_exclusion(exclude_question)
        pool = [
            it for it in pool if normalize_for_self_exclusion(it.get("question")) != wanted
        ]
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


class IncrementalDecoder:
    """Decode a growing generation without re-decoding the whole sequence each step.

    The stopping criterion is called once per generated token; decoding all generated ids
    every time is quadratic in the length of the output. This keeps the decoded prefix and
    only decodes the new tail. Line structure - which is all the step-line rule looks at -
    is unaffected by decoding in chunks. A shorter id list than the last call is taken as a
    new sequence and resets the cache, so one instance per batch row is safe across calls.
    """

    def __init__(self, decode: Callable[[list[int]], str]) -> None:
        self._decode = decode
        self._seen = 0
        self._text = ""

    def text(self, ids: list[int]) -> str:
        if len(ids) < self._seen:
            self._seen, self._text = 0, ""
        if len(ids) > self._seen:
            self._text += self._decode(ids[self._seen :])
            self._seen = len(ids)
        return self._text


class StepLineStopper:
    """Decide when a generation has produced ``max_step_lines`` step lines.

    Pure logic with no torch dependency so it is testable against a synthetic token
    stream: ``decode`` turns the generated token ids (prompt excluded) into text.

    The count comes from ``src/step_lines.py``, i.e. the same normalization the evaluator
    scores with, applied to the *post-processed* text: a ``<think>`` preamble or a trailing
    ``Question:`` echo must not consume part of the budget, or "8 step lines" would not be
    8 of the steps that get scored.
    """

    def __init__(
        self,
        max_step_lines: int,
        decode: Callable[[list[int]], str],
        *,
        strip_think: bool = False,
        truncate_at: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        if max_step_lines <= 0:
            raise ValueError(f"max_step_lines must be positive, got {max_step_lines}")
        self.max_step_lines = int(max_step_lines)
        self.strip_think = bool(strip_think)
        self.truncate_at = list(truncate_at or [])
        self._decoder = IncrementalDecoder(decode)

    def step_lines(self, generated_ids) -> int:
        return completed_step_line_count(
            self._decoder.text(list(generated_ids)),
            strip_think=self.strip_think,
            truncate_at=self.truncate_at,
        )

    def should_stop(self, generated_ids) -> bool:
        return self.step_lines(generated_ids) >= self.max_step_lines


def make_step_line_stopping_criteria(
    tokenizer, *, prompt_len: int, max_step_lines: int, post_cfg: dict | None = None
):
    """Wrap :class:`StepLineStopper` as a transformers ``StoppingCriteriaList``."""
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    post_cfg = post_cfg or {}
    stoppers: dict[int, StepLineStopper] = {}

    def stopper_for(row: int) -> StepLineStopper:
        if row not in stoppers:
            stoppers[row] = StepLineStopper(
                max_step_lines,
                lambda ids: tokenizer.decode(ids, skip_special_tokens=True),
                strip_think=bool(post_cfg.get("strip_think")),
                truncate_at=post_cfg.get("truncate_at") or [],
            )
        return stoppers[row]

    class _StepLineCriteria(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            done = [
                stopper_for(row).should_stop(seq[prompt_len:].tolist())
                for row, seq in enumerate(input_ids)
            ]
            return torch.tensor(done, dtype=torch.bool, device=input_ids.device)

    return StoppingCriteriaList([_StepLineCriteria()])


def post_process(response: str, post_cfg: dict) -> str:
    """Strip ``<think>`` blocks and the configured tail markers (``src/step_lines.py``)."""
    return post_process_generation(
        response,
        strip_think=bool(post_cfg.get("strip_think")),
        truncate_at=post_cfg.get("truncate_at") or [],
    )


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
    post_cfg: dict,
    max_step_lines: int | None = None,
) -> dict[str, Any]:
    """Generate one continuation and report how it ended.

    Returns the raw text, the post-processed text, the final (cap-trimmed) decomposition and
    the two ways a generation can be cut short:

    - ``hit_max_new_tokens`` - the token budget ran out. In the uncapped arms this is the
      only thing that bounds a runaway decomposition, so a truncated unguided output has to
      be distinguishable from a genuinely long one (otherwise the "unguided over-decomposes"
      reading and the "the token cap cut it off" reading are the same number).
    - ``stopped_at_step_line_cap`` - the ``unguided_capped`` arm's step-line budget bound it.

    Post-processing happens **before** the cap is applied, so a ``<think>`` preamble or a
    ``Question:`` echo cannot consume part of the step-line budget.
    """
    import torch

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_new_tokens = int(require(generation, "max_new_tokens"))
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]
    stopping_criteria = (
        make_step_line_stopping_criteria(
            tokenizer,
            prompt_len=prompt_len,
            max_step_lines=max_step_lines,
            post_cfg=post_cfg,
        )
        if max_step_lines
        else None
    )
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=float(require(generation, "temperature")),
            top_p=float(require(generation, "top_p")),
            do_sample=bool(require(generation, "do_sample")),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria,
        )
    generated_ids = outputs[0][prompt_len:]
    generated_tokens = int(generated_ids.shape[-1])
    raw = tokenizer.decode(generated_ids, skip_special_tokens=True)
    processed = post_process(raw, post_cfg)
    decomposition = processed
    if max_step_lines:
        decomposition = trim_to_step_lines(processed, max_step_lines)
    return {
        "raw": raw.strip(),
        "post_processed": processed,
        "decomposition": decomposition.strip(),
        "generated_tokens": generated_tokens,
        "max_new_tokens": max_new_tokens,
        "hit_max_new_tokens": generated_tokens >= max_new_tokens,
        "step_lines": step_line_count(decomposition),
        "stopped_at_step_line_cap": bool(
            max_step_lines and step_line_count(processed) >= max_step_lines
        ),
    }


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
    p.add_argument(
        "--allow-unpinned-eval-set",
        action="store_true",
        help="Permit a run whose loaded row counts do not match the config's "
        "'eval_rows_per_hop' (the pinned ADR 0007 set). For fixture and smoke runs only: "
        "the metrics JSON then records evaluation_set.pinned=false, and such a run is not an "
        "experiment arm.",
    )
    return p.parse_args()


def sha256_file(path: Path) -> str:
    """Content address of a consumed input, so an artifact says which bytes it read."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_retrieval_input(
    cli_value: str | None, cfg: dict
) -> str | None:
    """The retrieval input, refusing the silent-fallback path when the config forbids it.

    ``configs/decomposer_musique.json`` sets ``retrieval.require_input``: the few-shot
    method is fixed by ADR 0006 (bi-encoder top-20 -> cross-encoder top-5, typed masking,
    pool 2000) and is an upstream artifact this runner cannot rebuild. With no retrieval
    input, a few-shot-enabled model folder falls back to the committed **MetaQA** exemplar
    pool via a seeded random draw - a different method, run under the label of the fixed
    one. For the MuSiQue conditions that is a refusal, not a default.
    """
    value = cli_value or require(cfg, "retrieval.input")
    if value:
        return value
    if not optional(cfg, "retrieval.require_input"):
        return None
    src = cfg.get("_config_path", "<config>")
    raise SystemExit(
        f"no retrieval input: pass --retrieval-input, or set 'retrieval.input' in {src}.\n"
        f"{src} sets 'retrieval.require_input', because the few-shot method for this config "
        "is fixed by ADR 0006 (bi-encoder top-20 -> cross-encoder top-5 rerank, typed "
        "masking, pool 2000) and lives in an upstream artifact. Without it the run would "
        "silently fall back to random exemplars from the committed MetaQA pool, which is "
        "neither that method nor a MuSiQue pool - and every condition would be scored under "
        "a method the config does not describe."
    )


def assert_pinned_eval_set(
    rows_per_hop: dict[str, int],
    total: int,
    *,
    cfg: dict,
    hops: list[int],
    allow_unpinned: bool,
    source: str,
) -> dict[str, Any]:
    """Assert the run loaded the pinned evaluation set: ``eval_rows_per_hop`` per hop depth.

    ADR 0007 pins the MuSiQue evaluation set to 600 questions, 200 per hop depth, and ADR
    0011's house stance is that a comparison across different sets is not a comparison. The
    count was previously only checked by a test, so a retrieval input built over some other
    subset would have run happily and produced three arms that were not on the pinned set.
    Now the run refuses. ``--allow-unpinned-eval-set`` is the explicit, recorded opt-out for
    fixture runs.
    """
    expected_per_hop = optional(cfg, "eval_rows_per_hop")
    record: dict[str, Any] = {
        "expected_rows_per_hop": expected_per_hop,
        "expected_hops": hops,
        "rows_source": source,
        "pinned": None,
        "allow_unpinned_flag": allow_unpinned,
    }
    if expected_per_hop is None:
        record["note"] = (
            "this config declares no 'eval_rows_per_hop', so no pinned-set assertion applies"
        )
        return record

    expected_per_hop = int(expected_per_hop)
    expected_total = expected_per_hop * len(hops)
    problems: list[str] = []
    for hop in hops:
        got = int(rows_per_hop.get(str(hop), 0))
        if got != expected_per_hop:
            problems.append(f"hop {hop}: loaded {got}, expected {expected_per_hop}")
    unexpected = sorted(set(rows_per_hop) - {str(h) for h in hops})
    if unexpected:
        problems.append(
            "rows at hop depths the config does not list: "
            + ", ".join(f"{h} ({rows_per_hop[h]} rows)" for h in unexpected)
        )
    if total != expected_total:
        problems.append(f"total: loaded {total}, expected {expected_total}")

    record["pinned"] = not problems
    if not problems:
        return record

    detail = "\n".join(f"  - {p}" for p in problems)
    if allow_unpinned:
        record["note"] = (
            "row counts do not match the config's pinned evaluation set; permitted by "
            "--allow-unpinned-eval-set. This run is not an experiment arm.\n" + detail
        )
        print(
            "[decomposer] WARNING: not the pinned evaluation set (allowed by "
            f"--allow-unpinned-eval-set):\n{detail}"
        )
        return record
    src = cfg.get("_config_path", "<config>")
    raise SystemExit(
        f"the rows loaded are not the pinned evaluation set declared in {src} "
        f"(eval_rows_per_hop={expected_per_hop} for hops {hops}, {expected_total} total).\n"
        f"{detail}\n"
        f"Loaded per hop: {rows_per_hop} (total {total}), source: {source}.\n"
        "ADR 0007 pins that set and ADR 0011's stance is that a comparison across different "
        "evaluation sets is not a comparison, so this is a refusal. Point --retrieval-input "
        "at a file built over exactly those questions, or pass --allow-unpinned-eval-set for "
        "a fixture run that is not an experiment arm."
    )


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
    retrieval_input = resolve_retrieval_input(args.retrieval_input, cfg)
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
    # Universal-newline read, as v1 did: the rendered prompt is unchanged by this fix even
    # for a CRLF prompt file. The invariant below reads the raw bytes instead, so a line
    # ending is part of what it compares.
    prompt_template = prompt_path.read_text(encoding="utf-8")
    config_src = cfg.get("_config_path", "<config>")
    prompt_invariant: dict[str, Any] = {
        "checked": False,
        "note": (
            f"{config_src} does not set 'unguided_prompt_must_equal_guided_minus_hop_lines', "
            "so the unguided/guided prompt relationship is unchecked for this run"
        ),
    }
    if not guided and optional(cfg, "unguided_prompt_must_omit_hop_count"):
        assert_unguided_prompt_omits_hop_count(
            prompt_template,
            prompt_path=prompt_path,
            model=args.model,
            config_src=config_src,
        )
    if optional(cfg, "unguided_prompt_must_equal_guided_minus_hop_lines"):
        guided_path = model_dir / require(model_cfg, "prompt_file")
        if not unguided_prompt_file:
            raise SystemExit(
                f"{config_src} sets 'unguided_prompt_must_equal_guided_minus_hop_lines', but "
                f"model folder {args.model!r} has no 'unguided_prompt_file' to check against "
                f"{guided_path}. The unguided arms cannot run on this folder."
            )
        unguided_path = model_dir / unguided_prompt_file
        if not unguided_path.exists():
            raise SystemExit(f"unguided prompt file not found: {unguided_path}")
        prompt_invariant = assert_unguided_is_guided_minus_hop_lines(
            guided_template=guided_path.read_bytes().decode("utf-8"),
            unguided_template=unguided_path.read_bytes().decode("utf-8"),
            guided_path=guided_path,
            unguided_path=unguided_path,
            model=args.model,
            config_src=config_src,
        )
        print(
            "[decomposer] prompt invariant OK: "
            f"{unguided_path.name} == {guided_path.name} minus "
            f"{len(prompt_invariant['hop_lines_removed'])} hop-bearing line(s)"
        )

    chat_marker = None
    if prompt_style == "chat_template":
        chat_marker = require(model_cfg, "chat_template.split_marker")
        split_chat_template(prompt_template, chat_marker)  # fail fast on a bad prompt
    elif prompt_style != "plain":
        raise SystemExit(f"unknown prompt_style {prompt_style!r} (expected plain or chat_template)")

    # Content-address the retrieval input: the three arms are comparable only if they read
    # the same few-shot exemplars, and "same path" does not prove "same bytes" for a file
    # that lives outside git (ADR 0011 - a comparison claim needs the artifacts to be
    # verifiable from what is committed).
    retrieval_path: Path | None = None
    retrieval_sha256: str | None = None
    if retrieval_input:
        retrieval_path = Path(retrieval_input)
        if not retrieval_path.is_absolute():
            retrieval_path = _REPO_ROOT / retrieval_path
        if not retrieval_path.exists():
            raise SystemExit(f"retrieval input not found: {retrieval_path}")
        retrieval_sha256 = sha256_file(retrieval_path)

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
        "prompt_sha256": sha256_file(prompt_path),
        "unguided_prompt_invariant": prompt_invariant,
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
            "input_resolved": str(retrieval_path) if retrieval_path else None,
            "input_sha256": retrieval_sha256,
            "mode": retrieval_mode,
            "k": retrieval_k,
            "require_input": bool(optional(cfg, "retrieval.require_input")),
        },
        "few_shot_self_exclusion": {
            "enabled": True,
            "by": ["pool_id == query_id", "normalized pool_question == normalized query"],
            "note": (
                "an exemplar that is the query itself is dropped from the ranked list before "
                "the top-k is taken, on both the reranked and the bi-encoder path"
            ),
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
    self_excluded_rows = 0
    self_excluded_examples = 0
    if retrieval_input:
        assert retrieval_path is not None  # resolved and hashed above
        rows = load_jsonl(retrieval_path)
        if not rows:
            raise SystemExit(f"no rows in retrieval input: {retrieval_path}")
        # Declared in both configs; null means "refuse an unparseable hop depth" rather than
        # "guess", so the key is still required to be present.
        hop_fallback = require(cfg, "retrieval.hop_fallback")
        unparseable: list[str] = []
        for row in rows:
            question = row.get("query_question")
            if not isinstance(question, str) or not question.strip():
                continue
            hop = parse_hop_from_id(row.get("query_id"))
            if hop is None:
                # A guided arm would inject `hop_fallback` as if it were the gold hop count,
                # i.e. silently mislabel the oracle. And with a pinned evaluation set every
                # id parses, so an id that does not is a sign the input is not that set.
                if guided or hop_fallback is None:
                    unparseable.append(str(row.get("query_id")))
                    continue
                hop = int(hop_fallback)
            examples: list[dict] = []
            if few_shot_enabled:
                examples, dropped = examples_from_reranked_row(
                    row,
                    retrieval_mode,
                    retrieval_k,
                    exclude_query_id=row.get("query_id"),
                    exclude_question=question,
                )
                if dropped:
                    self_excluded_rows += 1
                    self_excluded_examples += dropped
            inference_rows.append(
                {
                    "query_id": row.get("query_id"),
                    "question": question,
                    "hop_count": hop,
                    "retrieval_examples": examples,
                }
            )
        if unparseable:
            shown = ", ".join(unparseable[:10]) + (" ..." if len(unparseable) > 10 else "")
            raise SystemExit(
                f"{len(unparseable)} row(s) in {retrieval_path} have a query_id whose hop "
                f"depth cannot be parsed (expected an id like '2hop__...'): {shown}\n"
                + (
                    "The gold hop count is what the guided arm injects, so guessing it from "
                    "'retrieval.hop_fallback' would file a wrong hop count as the oracle."
                    if guided
                    else "This config sets 'retrieval.hop_fallback' to null, so an "
                    "unparseable hop depth is a refusal rather than a guess: the hop depth "
                    "is what the per-hop reporting and the pinned-set check are counted on."
                )
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
    distinct_query_ids = len({r["query_id"] for r in inference_rows if r["query_id"] is not None})
    eval_set_record = assert_pinned_eval_set(
        rows_loaded_per_hop,
        rows_loaded_total,
        cfg=cfg,
        hops=hops,
        allow_unpinned=args.allow_unpinned_eval_set,
        source=str(retrieval_path) if retrieval_input else "questions_template_key",
    )
    eval_set_record["rows_loaded_total"] = rows_loaded_total
    eval_set_record["rows_loaded_per_hop"] = rows_loaded_per_hop
    eval_set_record["distinct_query_ids"] = distinct_query_ids
    snapshot["evaluation_set"] = eval_set_record
    print(f"Evaluation set: {json.dumps(eval_set_record, default=str)}")

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
                    # Same self-exclusion as the reranked path: the query may not be its own
                    # few-shot example. Harmless while the pool is MuSiQue train and the
                    # queries are dev; leakage the moment a pool is drawn from the same split.
                    exclude_question=question,
                    exclude_ids=[row.get("query_id")] if row.get("query_id") else None,
                )
                sampled = [it for it, _ in similar]
                sampled_with_scores = similar
                source = "similarity"
            elif few_shot_data:
                sampled = sample_few_shot_combined(
                    few_shot_data, few_shot_k, fallback_rng, exclude_question=question
                )
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
                gen = None
            else:
                rendered = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=bool(require(model_cfg, "chat_template.enable_thinking")),
                )
                gen = generate(
                    rendered,
                    model,
                    tokenizer,
                    device,
                    generation,
                    post_cfg=post_cfg,
                    max_step_lines=stop_after_step_lines,
                )
        else:
            rendered = fill_template(
                prompt_template,
                question=question,
                hop_count=hop_input,
                few_shot_examples=few_shot_str,
                unguided_hop_placeholder=unguided_hop_placeholder,
            )
            gen = (
                None
                if args.dry_run
                else generate(
                    rendered,
                    model,
                    tokenizer,
                    device,
                    generation,
                    post_cfg=post_cfg,
                    max_step_lines=stop_after_step_lines,
                )
            )
        decomposition = gen["decomposition"] if gen else ""

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
            # Identical dump shape in every arm: the raw generation, then the recorded
            # decomposition. Without the raw text, an output that the token budget cut off
            # cannot be told from one the model chose to end - and only one of those is
            # evidence about hop ignorance.
            log_path.write_text(
                "\n".join(header)
                + f"\n\n--- Prompt ({prompt_style}) ---\n"
                + rendered
                + "\n--- Raw generation ---\n"
                + (gen["raw"] if gen else "")
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
                # Same fields in all three arms, whether or not a cap applies.
                "decomposition_raw": gen["raw"] if gen else "",
                "step_lines": gen["step_lines"] if gen else 0,
                "generated_tokens": gen["generated_tokens"] if gen else None,
                "hit_max_new_tokens": gen["hit_max_new_tokens"] if gen else None,
                "stopped_at_step_line_cap": gen["stopped_at_step_line_cap"] if gen else None,
            }
        )

    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    empty = sum(1 for r in results if not r["decomposition"])
    step_line_counts = [step_line_count(r["decomposition"]) for r in results]
    metrics = {
        "dry_run": args.dry_run,
        "total_rows": len(results),
        "rows_loaded_total": rows_loaded_total,
        "rows_loaded_per_hop": rows_loaded_per_hop,
        "distinct_query_ids": distinct_query_ids,
        "evaluation_set": eval_set_record,
        "rows_with_empty_decomposition": empty,
        "few_shot_enabled": few_shot_enabled,
        "few_shot_source_mode": few_shot_source_mode,
        "few_shot_source_counts": {
            src: sum(1 for r in results if r["few_shot_source"] == src)
            for src in sorted({r["few_shot_source"] for r in results})
        },
        "few_shot_self_exclusion": {
            "rows_with_a_self_example_dropped": self_excluded_rows,
            "self_examples_dropped": self_excluded_examples,
            "note": (
                "counted on the reranked retrieval path only (the bi-encoder and random "
                "paths exclude by filtering, without a counter)"
            ),
        },
        "condition": condition_name,
        "guided": guided,
        "stop_after_step_lines": stop_after_step_lines,
        # How many decompositions came out at the cap (the stopping rule bound them),
        # counted with the evaluator's step normalization (src/step_lines.py) so this is
        # the same "number of steps" the metrics will report.
        "rows_at_step_line_cap": (
            sum(1 for n in step_line_counts if n >= stop_after_step_lines)
            if stop_after_step_lines and not args.dry_run
            else None
        ),
        # Token-budget truncation, reported for every arm. In the two uncapped arms
        # max_new_tokens is the only bound on a runaway decomposition, so a row that hit it
        # is a length-truncated output, not a decomposition the model finished.
        "max_new_tokens": int(require(generation, "max_new_tokens")),
        "rows_at_max_new_tokens": (
            sum(1 for r in results if r["hit_max_new_tokens"]) if not args.dry_run else None
        ),
        "rows_stopped_at_step_line_cap": (
            sum(1 for r in results if r["stopped_at_step_line_cap"])
            if stop_after_step_lines and not args.dry_run
            else None
        ),
        "step_lines_per_row_total": sum(step_line_counts) if not args.dry_run else None,
        "seed": seed,
        "model_size": size_record,
        "embedding_model_size": embed_size_record,
        "results_path": str(output_dir / "results.json"),
    }
    if args.dry_run:
        metrics["generation_truncation_note"] = (
            "unmeasured: --dry-run generates nothing, so rows_at_max_new_tokens, "
            "rows_at_step_line_cap and the step-line counts are null"
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
            f"- Prompt: `{prompt_path}` (style: {prompt_style})",
            f"- Condition: {condition_name or 'none (no conditions block)'}; guided: {guided}; "
            f"step-line cap: {stop_after_step_lines or 'none'}; seed: {seed}",
            f"- Rows: {len(results)} of {rows_loaded_total} loaded "
            f"(per hop: {rows_loaded_per_hop}; distinct ids: {distinct_query_ids})",
            f"- Evaluation set pinned: {eval_set_record['pinned']} "
            f"(expected {eval_set_record['expected_rows_per_hop']} rows per hop for "
            f"hops {hops}; source: {eval_set_record['rows_source']})",
            (
                f"- Retrieval input: `{retrieval_path}` (sha256 `{retrieval_sha256}`)"
                if retrieval_input
                else "- Retrieval input: none (questions read from questions_template_key)"
            ),
            f"- Few-shot: enabled={few_shot_enabled} k={few_shot_k} mode={few_shot_source_mode}; "
            f"self-examples dropped: {self_excluded_examples}",
            (
                f"- Truncation: {metrics['rows_at_max_new_tokens']} row(s) at "
                f"max_new_tokens={metrics['max_new_tokens']}, "
                f"{metrics['rows_at_step_line_cap']} at the step-line cap"
                if not args.dry_run
                else "- Truncation: unmeasured (dry run generates nothing)"
            ),
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
