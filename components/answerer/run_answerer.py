#!/usr/bin/env python3
"""MuSiQue answering backend: does a decomposition lead to the right answer? (issue #16)

ADR 0006 gives MuSiQue both decomposition-quality evaluation and **end-to-end** evaluation.
The decomposition-quality half exists (``scripts/musique_decompositions_evaluator.py``, all
string-level); this is the end-to-end half: it *executes* a decomposition and scores the
answer it produces, so a decomposition is judged by where it leads and not only by how it
looks.

What it does, per item:

1. read the item's sub-questions in order (from a decomposer run's predictions dump, or
   from the gold decompositions for the oracle-decomposition ceiling);
2. substitute every ``[#k]`` placeholder with the answer the reader produced for step k
   (``src/step_lines.py::substitute_step_references``);
3. ask the reader model each sub-question over the item's **full paragraph list**;
4. take the **final** step's answer as the predicted answer for the item;
5. score it with MuSiQue's official answer EM and answer F1 against the item's gold answer
   plus its alias list (``src/answer_metrics.py``).

The three methodology choices — reader from the decomposer's registry, full-paragraph
context, official EM/F1 — are Jahid's, made 2026-08-20 in session and pending supervisor
confirmation; they are recorded in ``docs/adr/0019-musique-answering-backend-conventions.md``
and set in ``configs/answer_musique.json``. This script implements exactly them: there is no
gold-supporting-only condition and no retrieval condition, and ``context.policy`` refuses any
value but ``all_paragraphs``. **No model scores, rates or judges anything** — the metrics are
string metrics (CLAUDE.md standing constraint).

Usage::

    # the whole loop on a tiny sample, no weights loaded (what the smoke test runs)
    python components/answerer/run_answerer.py --predictions <decomposer run>/results.json \\
        --dry-run --dry-run-limit 5 --allow-unpinned-eval-set

    # a real run: score one decomposition arm end to end
    python components/answerer/run_answerer.py --predictions runs/<arm>/results.json

    # the oracle-decomposition ceiling: gold plans, same reader, same evaluation set
    python components/answerer/run_answerer.py --gold-decompositions

A real run loads weights, so it holds the **run lock** of ``docs/compute.md`` (the single-GPU
box is shared): acquire ``runs/run.lock`` with the experiment id + ISO timestamp before
launching and release it when the run finishes or fails. ``--dry-run`` loads no weights and
needs no lock.

Guarantees, all of them hard:

- the reader's parameter count is printed and asserted against ``configs/model_limits.json``
  before a single answer is generated — against ``answerer_max_params`` (8e9, the standing
  ceiling), **not** the decomposer's ADR 0015 raise to 1e10, which was granted for one role;
- the reader prompt must carry ``{context}`` and ``{question}`` **in the half that actually
  gets filled**: for a chat-template folder that is the user half, since a placeholder above
  the split marker is passed through as literal text and would leave the reader closed-book;
- the evaluation set is asserted to be the pinned ADR 0007 set **by id**, not only by count
  (the shared assertion in ``run_decomposer.py``), because an end-to-end number on a
  different set is not comparable to anything (ADR 0011). ``--allow-unpinned-eval-set`` is
  the recorded opt-out for fixture runs, which are not experiment arms;
- every item's context is checked to exist **before** the model is loaded, so a scored item
  can never be a silently closed-book one;
- every config value the generation call reads — decoding, answer cleanup, and a
  chat-template folder's ``enable_thinking`` — is read up front, so a missing key fails with
  no weights loaded rather than mid-run on the GPU;
- the run leaves a config snapshot, a metrics JSON and a run note; the metrics JSON carries
  aggregates and counts only (no dataset text), and the per-item file — which does carry
  questions and answers — stays under the gitignored runs root, because data never enters
  git (CLAUDE.md).

On ``--dry-run`` nothing is generated: the join, the context assembly, the placeholder
substitution chain and the artifact writing all run, and EM/F1 are recorded as **unmeasured
rather than zero**. Because a dry run therefore cannot reach any ``gen[...]`` branch, the
generation-result key contract is guarded statically instead, per ADR 0016
(``tests/test_generation_contract.py::TestAnswererGenerationContract``).
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))
#: The reader loads its weights exactly the way the decomposer does, so the loader,
#: quantization, retrieval-free prompt plumbing and the pinned-evaluation-set assertion are
#: *imported* from that runner rather than duplicated here — a second copy of "how a model is
#: loaded" or "what the pinned set is" would be a second thing to keep in step
#: (same reasoning as ``src/finetune_data.py``'s import of it).
sys.path.insert(0, str(_REPO_ROOT / "components" / "decomposer"))

from answer_metrics import (  # noqa: E402
    ANSWER_METRIC_DEFINITIONS,
    gold_answer_set,
    score_answer,
)
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
from seeding import set_global_seed  # noqa: E402
from step_lines import (  # noqa: E402
    post_process_generation,
    split_step_lines,
    substitute_step_references,
)

import run_decomposer as rd  # noqa: E402

_WS_RX = re.compile(r"\s+")

#: The only context policy that may ship (ADR 0019 decision 2).
CONTEXT_POLICIES = ("all_paragraphs",)

PER_ITEM_SCHEMA = "musique_answer_per_item/1"


# ------------------------------------------------------------------ pure helpers


def normalize_question(text: str) -> str:
    """Lowercased, whitespace-collapsed question text — the join fallback key.

    Same rule as ``scripts/musique_decompositions_evaluator.py::_normalize_question``, so a
    prediction that the decomposition evaluator matched to a gold row is matched to the same
    item here.
    """
    return _WS_RX.sub(" ", str(text or "").strip().lower())


def steps_from_decomposition(value: Any) -> list[str]:
    """The sub-questions of a decomposition, in the decomposition evaluator's reading.

    Mirrors ``scripts/musique_decompositions_evaluator.py::_decomp_to_steps``: a string is
    split with the shared splitter (``src/step_lines.py::split_step_lines``), a list takes
    each string, or each element's ``question`` field. Executing a different set of steps
    than the quality evaluator scores would make the two halves of the MuSiQue evaluation
    disagree about what "the decomposition" is; ``tests/test_answer_musique.py`` pins the two
    readings against each other.
    """
    if isinstance(value, str):
        return split_step_lines(value)
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, str):
                if item.strip():
                    out.append(item.strip())
            elif isinstance(item, dict):
                question = item.get("question")
                if isinstance(question, str) and question.strip():
                    out.append(question.strip())
        return out
    return []


def format_context(
    paragraphs: list[dict],
    *,
    template: str,
    separator: str,
    idx_field: str,
    title_field: str,
    text_field: str,
) -> str:
    """Render the item's paragraphs into the reader's context block.

    Every paragraph, in the item's own order, with no filtering — ``is_supporting`` is
    deliberately not read (ADR 0019 decision 2: the standard answerable setting, not a
    gold-supporting-only oracle). ``idx`` falls back to the position when the row has none,
    so the numbering in the prompt is always contiguous.
    """
    blocks: list[str] = []
    for position, para in enumerate(paragraphs, start=1):
        if not isinstance(para, dict):
            continue
        idx = para.get(idx_field, position)
        blocks.append(
            template.format(
                idx=idx,
                title=str(para.get(title_field, "") or "").strip(),
                text=str(para.get(text_field, "") or "").strip(),
            )
        )
    return separator.join(blocks)


def paragraph_list(item: dict, paragraphs_field: str) -> list[dict]:
    """The item's paragraph list, or an empty list when it has none."""
    value = item.get(paragraphs_field)
    return [p for p in value if isinstance(p, dict)] if isinstance(value, list) else []


def render_reader_template(template: str, *, prompt_style: str, marker: str) -> str | None:
    """The reader prompt for a *plain* model folder: the two halves joined, marker gone.

    One committed reader prompt serves every model folder in the registry. A
    ``chat_template`` folder gets the halves as system/user messages (the marker is the
    split), so nothing is joined for it and this returns ``None``. A ``plain`` folder gets
    them concatenated with a blank line, which is the same text a chat model sees, minus the
    chat wrapping.
    """
    if prompt_style == "chat_template":
        return None
    if marker in template:
        system_part, user_part = rd.split_chat_template(template, marker)
        return f"{system_part}\n\n{user_part}"
    return template


def reader_filled_text(template: str, *, prompt_style: str, marker: str) -> str:
    """The part of the reader prompt that placeholders are actually filled into.

    For a ``plain`` folder that is the whole prompt (the halves joined). For a
    ``chat_template`` folder it is the **user** half only: :func:`build_reader_messages`
    passes the system half through verbatim, so a placeholder written above the marker is
    never filled.
    """
    if prompt_style == "chat_template":
        _, user_part = rd.split_chat_template(template, marker)
        return user_part
    return render_reader_template(template, prompt_style=prompt_style, marker=marker) or template


def assert_reader_template(
    template: str, *, prompt_path: Path, prompt_style: str, marker: str
) -> None:
    """Refuse a reader prompt whose *filled* half cannot carry context and sub-question.

    A prompt with no ``{context}`` would score the reader closed-book while the run reported
    the full-paragraph setting, and a prompt with no ``{question}`` would ask nothing at all.
    Both are refusals rather than warnings: either produces numbers that answer a different
    question than the one the run claims.

    Checking the *filled* half rather than the file is the point (PR #32 review, I-1): with
    a ``chat_template`` folder, ``{context}`` written above ``<<<USER>>>`` is passed through
    as literal text in the system message and never filled, so a whole-file check would pass
    a prompt that runs closed-book. What gets filled is what gets checked
    (:func:`reader_filled_text`).
    """
    filled = reader_filled_text(template, prompt_style=prompt_style, marker=marker)
    missing = [slot for slot in ("{context}", "{question}") if slot not in filled]
    if missing:
        where = (
            f"the half after {marker!r} (the only half that is filled for a "
            "chat_template model folder)"
            if prompt_style == "chat_template"
            else "the prompt"
        )
        raise SystemExit(
            f"reader prompt {prompt_path} is missing {', '.join(missing)} in {where}.\n"
            "The reader answers one sub-question over the MuSiQue item's full paragraph "
            "list (ADR 0019 decision 2), so both placeholders are required where they are "
            "filled: without {context} there the run would be closed-book while reporting "
            "the full-paragraph setting."
        )


def fill_reader_template(template: str, *, context: str, question: str) -> str:
    """Fill the reader prompt's placeholders (only the ones the template contains)."""
    values: dict[str, Any] = {}
    if "{context}" in template:
        values["context"] = context
    if "{question}" in template:
        values["question"] = question
    return template.format(**values) if values else template


def build_reader_messages(
    template: str, *, marker: str, context: str, question: str
) -> list[dict]:
    """System/user messages for a ``chat_template`` model folder."""
    system_part, user_part = rd.split_chat_template(template, marker)
    return [
        {"role": "system", "content": system_part},
        {
            "role": "user",
            "content": fill_reader_template(user_part, context=context, question=question),
        },
    ]


def clean_answer(
    text: str,
    *,
    take_first_line: bool,
    strip_prefixes: list[str],
    max_answer_chars: int,
) -> tuple[str, bool]:
    """The reader's answer as scored: ``(answer, was_truncated)``.

    The reader is told to output the answer alone; this is the mechanical cleanup of what it
    actually returns, applied *after* the model folder's own ``post_process`` block. Only
    three things happen, all of them recorded in the config snapshot: the first non-empty
    line is kept (a reader that keeps talking is cut at the line, not paraphrased), a leading
    ``Answer:``-style prefix is dropped, and an answer longer than ``max_answer_chars`` is
    truncated and reported as truncated rather than silently shortened.
    """
    value = str(text or "").strip()
    if take_first_line:
        for line in value.splitlines():
            if line.strip():
                value = line.strip()
                break
        else:
            value = ""
    for prefix in strip_prefixes:
        if prefix and value.lower().startswith(prefix.lower()):
            value = value[len(prefix) :].strip()
            break
    truncated = max_answer_chars > 0 and len(value) > max_answer_chars
    if truncated:
        value = value[:max_answer_chars].strip()
    return value, truncated


def aggregate_answers(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Macro EM / F1 over scored items. ``num_items`` 0 gives nulls, not zeros."""
    scored = [it for it in items if it.get("answer_em") is not None]
    if not scored:
        return {"num_items": 0, "answer_em": None, "answer_f1": None}
    return {
        "num_items": len(scored),
        "answer_em": statistics.fmean(float(it["answer_em"]) for it in scored),
        "answer_f1": statistics.fmean(float(it["answer_f1"]) for it in scored),
    }


def per_gold_hop_metrics(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """The same aggregate per gold hop depth (2 / 3 / 4), in the style of METRICS.md §2."""
    by_hop: dict[int, list[dict[str, Any]]] = {}
    for item in items:
        hop = item.get("gold_hop_count")
        if isinstance(hop, int):
            by_hop.setdefault(hop, []).append(item)
    return {str(hop): aggregate_answers(rows) for hop, rows in sorted(by_hop.items())}


#: Every config value :func:`generate_answer` reads. Checked in ``main()`` instead of being
#: discovered inside the generation call, because a ``--dry-run`` never enters that call: a
#: missing decoding key or a missing cleanup key would otherwise be a crash only a real run
#: (loaded weights, GPU time) could find. Same reasoning as ADR 0016, applied to config keys
#: rather than to result keys.
GENERATION_KEYS = ("max_new_tokens", "temperature", "top_p", "do_sample")
ANSWER_POST_PROCESS_KEYS = ("take_first_line", "strip_prefixes", "max_answer_chars")

#: Model-folder keys read only on the **real** chat-template branch of the loop
#: (``tokenizer.apply_chat_template(..., enable_thinking=...)``); a dry run renders the
#: messages as JSON and never touches them. Same blind spot, so the same treatment: read up
#: front when the folder is a chat-template one (PR #32 review, I-2).
CHAT_TEMPLATE_KEYS = ("chat_template.enable_thinking",)


def assert_generation_preflight(
    generation: dict, answer_cfg: dict, *, model_cfg: dict | None = None, prompt_style: str = "plain"
) -> None:
    """Read every key the generation call will need, now, with no model loaded."""
    for key in GENERATION_KEYS:
        require(generation, key)
    for key in ANSWER_POST_PROCESS_KEYS:
        require(answer_cfg, key)
    if prompt_style == "chat_template":
        if model_cfg is None:
            raise SystemExit(
                "[answerer] internal: prompt_style='chat_template' needs model_cfg in the "
                "generation preflight, so the chat-template keys can be read before the "
                "model loads"
            )
        for key in CHAT_TEMPLATE_KEYS:
            require(model_cfg, key)


def cost_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Token and latency cost per **sub-question generation**, over measured rows only.

    Same discipline as the decomposer's cost block: nothing is imputed — a dry run measures
    no row, so every field is null with a note rather than 0, which would be a false claim
    instead of a missing one.
    """
    measured = [r for r in records if r.get("latency_seconds") is not None]

    def mean(key: str) -> float | None:
        return statistics.fmean(float(r[key]) for r in measured) if measured else None

    def median(key: str) -> float | None:
        return statistics.median(float(r[key]) for r in measured) if measured else None

    summary: dict[str, Any] = {
        "row_unit": "one sub-question generation (a decomposition contributes one row per step)",
        "rows_measured": len(measured),
        "rows_total": len(records),
        "mean_prompt_tokens_per_subquestion": mean("prompt_tokens"),
        "median_prompt_tokens_per_subquestion": median("prompt_tokens"),
        "mean_completion_tokens_per_subquestion": mean("completion_tokens"),
        "median_completion_tokens_per_subquestion": median("completion_tokens"),
        "mean_latency_seconds_per_subquestion": mean("latency_seconds"),
        "median_latency_seconds_per_subquestion": median("latency_seconds"),
        "total_generation_seconds": (
            sum(float(r["latency_seconds"]) for r in measured) if measured else None
        ),
    }
    if not measured:
        summary["note"] = (
            "unmeasured: no sub-question was generated in this run (--dry-run), so tokens "
            "and latency are unknown, not zero"
        )
    return summary


# --------------------------------------------------------------------- generation


def generate_answer(
    prompt_text: str,
    model,
    tokenizer,
    device: str,
    generation: dict,
    *,
    post_cfg: dict,
    answer_cfg: dict,
) -> dict[str, Any]:
    """Answer one sub-question, with its cost and how it ended.

    Mirrors ``run_decomposer.generate``: greedy/sampled decoding straight from the model
    folder's own ``generation`` block (with this config's overrides already applied by the
    caller), ``latency_seconds`` measured around ``model.generate`` alone and CUDA
    synchronized, and ``hit_max_new_tokens`` recorded so a cut-off answer is
    distinguishable from a short one.

    Returns the raw generation, the cleaned answer that gets scored, and the cost fields.
    Every key here is consumed by ``main()``; that contract is asserted statically, because a
    dry run cannot reach the consumer (ADR 0016).
    """
    import torch

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_new_tokens = int(require(generation, "max_new_tokens"))
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    prompt_tokens = int(inputs["input_ids"].shape[1])
    started = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=float(require(generation, "temperature")),
            top_p=float(require(generation, "top_p")),
            do_sample=bool(require(generation, "do_sample")),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    if device == "cuda":
        torch.cuda.synchronize()
    latency_seconds = time.perf_counter() - started
    new_tokens = outputs[0][prompt_tokens:]
    completion_tokens = int(new_tokens.shape[0])
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
    processed = post_process_generation(
        raw,
        strip_think=bool(post_cfg.get("strip_think")),
        truncate_at=post_cfg.get("truncate_at") or [],
    )
    answer, answer_truncated = clean_answer(
        processed,
        take_first_line=bool(require(answer_cfg, "take_first_line")),
        strip_prefixes=list(require(answer_cfg, "strip_prefixes")),
        max_answer_chars=int(require(answer_cfg, "max_answer_chars")),
    )
    return {
        "text": raw.strip(),
        "answer": answer,
        "answer_truncated": answer_truncated,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "latency_seconds": latency_seconds,
        "max_new_tokens": max_new_tokens,
        "hit_max_new_tokens": completion_tokens >= max_new_tokens,
    }


# --------------------------------------------------------------------------- IO


def load_items(path: Path, *, id_field: str, question_field: str) -> tuple[dict, dict]:
    """Index the MuSiQue items by id and by normalized question text.

    Both indexes exist because a decomposer run's predictions dump joins by ``query_id``
    while an older or hand-made predictions file may only carry the question text — the same
    two keys the decomposition evaluator accepts. The id index is authoritative; the question
    index is only the fallback for a row with no id.

    **The question fallback keeps the FIRST row** for a normalized question
    (``setdefault``), so with two dev items sharing a question text an id-less prediction is
    scored against the earlier one. Measured 2026-08-20 on the real dev set: 6
    duplicate-question groups covering 12 of 2417 rows, **all 6 agreeing on the normalized
    gold answer** (2 groups touch the pinned 600), so a mis-scored item is unreachable on
    this data — and every prediction a decomposer run produces carries a ``query_id``
    anyway. The note exists so a dataset swap re-checks it rather than inheriting the
    assumption.
    """
    by_id: dict[str, dict] = {}
    by_question: dict[str, dict] = {}
    for row in rd.load_jsonl(path):
        item_id = row.get(id_field)
        if isinstance(item_id, str) and item_id.strip():
            by_id[item_id.strip()] = row
        question = row.get(question_field)
        if isinstance(question, str) and question.strip():
            by_question.setdefault(normalize_question(question), row)
    if not by_id and not by_question:
        raise SystemExit(f"no usable MuSiQue items in {path}")
    return by_id, by_question


def decomposition_rows_from_predictions(path: Path) -> list[dict[str, Any]]:
    """Rows to execute, from a decomposer run's ``results.json``.

    Reads exactly the fields that dump carries (``query_id`` / ``id``, ``question``,
    ``decomposition``); a row without a usable question is skipped, and a row whose
    decomposition is empty is **kept** — an empty decomposition is a decomposer failure that
    has to score as one, not disappear from the denominator.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise SystemExit(f"--predictions must be a JSON list: {path}")
    rows: list[dict[str, Any]] = []
    for obj in payload:
        if not isinstance(obj, dict):
            continue
        question = obj.get("question")
        if not isinstance(question, str) or not question.strip():
            continue
        item_id = obj.get("query_id") or obj.get("id")
        rows.append(
            {
                "item_id": str(item_id).strip() if item_id else None,
                "question": question.strip(),
                "steps": steps_from_decomposition(obj.get("decomposition")),
            }
        )
    if not rows:
        raise SystemExit(f"no rows with a usable 'question' in {path}")
    return rows


def decomposition_rows_from_gold(
    path: Path,
    *,
    id_field: str,
    question_field: str,
    decomposition_field: str,
    restrict_to_ids: set[str] | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Rows to execute for the **oracle-decomposition ceiling**, from the gold file.

    Only the gold sub-question *text* is read (``steps_from_decomposition`` takes each gold
    step's ``question``); the gold sub-answers that sit next to it in some MuSiQue files are
    never read, or the ceiling would measure a plan that already contains its own answers.

    ``restrict_to_ids`` is the pinned ADR 0007 id set: a ceiling has to be a ceiling *for the
    set the arms ran on*, so the oracle is executed on exactly those ids. Returns
    ``(rows, missing_ids)`` — the pinned ids the gold file has no row for.
    """
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for obj in rd.load_jsonl(path):
        item_id = obj.get(id_field)
        item_id = str(item_id).strip() if item_id else None
        question = obj.get(question_field)
        if not isinstance(question, str) or not question.strip():
            continue
        if restrict_to_ids is not None and (item_id is None or item_id not in restrict_to_ids):
            continue
        if item_id:
            seen.add(item_id)
        rows.append(
            {
                "item_id": item_id,
                "question": question.strip(),
                "steps": steps_from_decomposition(obj.get(decomposition_field)),
            }
        )
    missing = sorted(restrict_to_ids - seen) if restrict_to_ids is not None else []
    if not rows:
        raise SystemExit(
            f"no gold decompositions to execute from {path}"
            + (
                f" (restricted to {len(restrict_to_ids)} pinned id(s), none of which is in "
                "that file)"
                if restrict_to_ids is not None
                else ""
            )
        )
    return rows, missing


# ------------------------------------------------------------------------ main


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", default="answer_musique.json", help="Config (configs/)")
    p.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="A decomposer run's predictions dump (results.json) to execute and score.",
    )
    p.add_argument(
        "--gold-decompositions",
        nargs="?",
        const="",
        default=None,
        metavar="PATH",
        help="Execute the GOLD decompositions instead (the oracle-decomposition ceiling). "
        "With no path, the config's gold_decompositions_key is resolved.",
    )
    p.add_argument("--model", default=None, help="Reader model folder (decomposer registry)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--limit", type=int, default=None, help="Cap the items evaluated.")
    p.add_argument("--quantization", default=None, choices=["none", "4bit", "8bit"])
    p.add_argument("--output-root", default=None)
    p.add_argument("--out-prefix", default=None, help="Override the artifact filename prefix.")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the whole loop — join, context, substitution, artifacts — without loading "
        "a model or generating. EM/F1 are then recorded as unmeasured, not zero.",
    )
    p.add_argument("--dry-run-limit", type=int, default=5)
    p.add_argument(
        "--allow-unpinned-eval-set",
        action="store_true",
        help="Permit a run whose items are not the pinned ADR 0007 set. For fixture and "
        "smoke runs only: the metrics then record evaluation_set.pinned false, and such a "
        "run is not an experiment arm.",
    )
    args = p.parse_args()
    if (args.predictions is None) == (args.gold_decompositions is None):
        p.error(
            "pass exactly one of --predictions (a decomposition run) or "
            "--gold-decompositions (the oracle ceiling)"
        )
    return args


def main() -> None:
    args = _parse_args()

    cfg = load_config(args.config)
    paths_cfg = load_paths(require(cfg, "paths_config"))
    limits = load_limits(require(cfg, "model_limits_config"))
    config_src = cfg.get("_config_path", "<config>")

    seed = args.seed if args.seed is not None else int(require(cfg, "seed"))
    current_run_id = run_id()
    seeded = set_global_seed(seed)

    data_root = Path(paths_cfg["data_root_resolved"])
    fields = dict(require(cfg, "fields"))
    context_cfg = dict(require(cfg, "context"))
    policy = require(cfg, "context.policy")
    if policy not in CONTEXT_POLICIES:
        raise SystemExit(
            f"{config_src} sets context.policy={policy!r}, which is not implemented "
            f"(only {list(CONTEXT_POLICIES)}).\n"
            "ADR 0019 decision 2 ships exactly one context policy: the MuSiQue item's full "
            "paragraph list, the standard answerable setting. A gold-supporting-only or "
            "retrieval context is a different experiment and is not admitted here; adding "
            "one is Jahid's call with his supervisor, not this script's."
        )
    substitution_cfg = dict(require(cfg, "substitution"))
    answer_cfg = dict(require(cfg, "answer_post_process"))
    hops = [int(h) for h in require(cfg, "hops")]
    limit = args.limit if args.limit is not None else require(cfg, "limit")
    out_prefix = args.out_prefix or require(cfg, "out_prefix")

    # ---- reader model folder: the decomposer's registry, unchanged (ADR 0019 decision 1)
    reader_model = args.model or require(cfg, "reader_model")
    models_root = resolve_path(
        require(paths_cfg, "repo." + require(cfg, "reader_models_dir_key")), _REPO_ROOT
    )
    model_dir = models_root / reader_model
    if not model_dir.is_dir():
        raise SystemExit(f"reader model folder not found: {model_dir}")
    model_cfg = load_config(model_dir / "config.json")
    model_id = require(model_cfg, "model_id")
    prompt_style = require(model_cfg, "prompt_style")
    if prompt_style not in ("plain", "chat_template"):
        raise SystemExit(
            f"unknown prompt_style {prompt_style!r} in {model_dir / 'config.json'} "
            "(expected plain or chat_template)"
        )
    quantization = args.quantization or require(model_cfg, "loader.quantization")
    loader = dict(require(model_cfg, "loader"))
    post_cfg = dict(require(model_cfg, "post_process"))
    generation_overrides = optional(cfg, "generation_overrides")
    generation = rd.apply_generation_overrides(
        dict(require(model_cfg, "generation")), generation_overrides, config_src
    )
    assert_generation_preflight(
        generation, answer_cfg, model_cfg=model_cfg, prompt_style=prompt_style
    )

    # ---- reader prompt
    prompt_path = resolve_path(require(cfg, "reader_prompt_file"), _REPO_ROOT)
    if not prompt_path.exists():
        raise SystemExit(f"reader prompt not found: {prompt_path}")
    prompt_template = prompt_path.read_text(encoding="utf-8")
    chat_marker = require(cfg, "chat_split_marker")
    # Splits the prompt itself (so a chat folder with no marker fails here) and checks the
    # half that actually gets filled.
    assert_reader_template(
        prompt_template,
        prompt_path=prompt_path,
        prompt_style=prompt_style,
        marker=chat_marker,
    )
    plain_template = render_reader_template(
        prompt_template, prompt_style=prompt_style, marker=chat_marker
    )
    prompt_sha256 = rd.sha256_file(prompt_path)

    device = "cpu"
    if not args.dry_run:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- what to execute
    mode = "predictions" if args.predictions is not None else "gold_decompositions"
    pinned_ids: set[str] = set()
    pinned_files: list[str] = []
    pinned_id_problems: list[str] = []
    if optional(cfg, "eval_rows_per_hop") is not None:
        pinned_ids, pinned_files, pinned_id_problems = rd.load_pinned_eval_ids(
            paths_cfg, cfg, hops, data_root
        )

    gold_missing_ids: list[str] = []
    if mode == "predictions":
        source_path = args.predictions
        if not source_path.exists():
            raise SystemExit(f"--predictions file not found: {source_path}")
        rows = decomposition_rows_from_predictions(source_path)
    else:
        source_path = (
            Path(args.gold_decompositions)
            if args.gold_decompositions
            else resolve_path(
                require(paths_cfg, "datasets." + require(cfg, "gold_decompositions_key")),
                data_root,
            )
        )
        if not source_path.exists():
            raise SystemExit(f"gold decomposition file not found: {source_path}")
        rows, gold_missing_ids = decomposition_rows_from_gold(
            source_path,
            id_field=require(cfg, "fields.id"),
            question_field=require(cfg, "fields.question"),
            decomposition_field=require(cfg, "fields.gold_decomposition"),
            restrict_to_ids=pinned_ids or None,
        )

    items_path = resolve_path(
        require(paths_cfg, "datasets." + require(cfg, "items_key")), data_root
    )
    if not items_path.exists():
        raise SystemExit(
            f"MuSiQue item file not found: {items_path} (datasets."
            f"{require(cfg, 'items_key')} in the paths config). It carries the context "
            "paragraphs and the gold answer + aliases, so the run cannot proceed without it."
        )
    id_field = require(cfg, "fields.id")
    question_field = require(cfg, "fields.question")
    paragraphs_field = require(cfg, "fields.paragraphs")
    gold_decomposition_field = require(cfg, "fields.gold_decomposition")
    answer_field = require(cfg, "fields.answer")
    aliases_field = require(cfg, "fields.answer_aliases")
    items_by_id, items_by_question = load_items(
        items_path, id_field=id_field, question_field=question_field
    )

    rows_input = len(rows)
    if limit is not None:
        rows = rows[: int(limit)]

    # ---- join every row to its MuSiQue item (context + gold answers)
    eval_rows: list[dict[str, Any]] = []
    missing_items: list[str] = []
    for row in rows:
        item = None
        if row["item_id"]:
            item = items_by_id.get(row["item_id"])
        if item is None:
            item = items_by_question.get(normalize_question(row["question"]))
        if item is None:
            missing_items.append(row["item_id"] or normalize_question(row["question"]))
            continue
        gold_steps = steps_from_decomposition(item.get(gold_decomposition_field))
        eval_rows.append(
            {
                **row,
                "item": item,
                "item_id": row["item_id"] or str(item.get(id_field) or ""),
                # Gold hop depth = the number of steps in the item's gold decomposition, the
                # same denominator the decomposition evaluator's hop-count family uses
                # (docs/METRICS.md §1). The id prefix ("2hop__...") is cross-checked below
                # rather than trusted.
                "gold_hop_count": len(gold_steps) or None,
            }
        )
    if not eval_rows:
        raise SystemExit(
            f"no rows could be joined to a MuSiQue item in {items_path} "
            f"({len(missing_items)} unmatched). Check that the predictions were produced on "
            "the same evaluation set."
        )

    # Context before weights: a scored item may never be a silently closed-book one.
    no_context = [
        r["item_id"] for r in eval_rows if not paragraph_list(r["item"], paragraphs_field)
    ]
    if no_context:
        shown = ", ".join(no_context[:10]) + (" ..." if len(no_context) > 10 else "")
        raise SystemExit(
            f"{len(no_context)} item(s) in {items_path} carry no "
            f"'{paragraphs_field}' list, so they have no context to read: "
            f"{shown}\nThe run is refused rather than scoring them closed-book while "
            "reporting the full-paragraph setting (ADR 0019 decision 2)."
        )

    # Gold hop depth vs the id prefix: reported, so a mis-filed per-hop table is visible.
    hop_prefix_disagreements = [
        r["item_id"]
        for r in eval_rows
        if rd.parse_hop_from_id(r["item_id"]) is not None
        and r["gold_hop_count"] is not None
        and rd.parse_hop_from_id(r["item_id"]) != r["gold_hop_count"]
    ]

    rows_per_hop = {
        str(hop): sum(1 for r in eval_rows if r["gold_hop_count"] == hop)
        for hop in sorted({r["gold_hop_count"] for r in eval_rows if r["gold_hop_count"]})
    }
    loaded_ids = {r["item_id"] for r in eval_rows if r["item_id"]}
    eval_set_record = rd.assert_pinned_eval_set(
        rows_per_hop,
        len(eval_rows),
        cfg=cfg,
        hops=hops,
        allow_unpinned=args.allow_unpinned_eval_set,
        source=str(source_path),
        loaded_ids=loaded_ids,
        pinned_ids=pinned_ids,
        pinned_files=pinned_files,
        pinned_id_problems=pinned_id_problems,
        remedy=(
            "Point --predictions at a decomposition run over exactly those questions (or "
            "--gold-decompositions, which restricts itself to them), or pass "
            "--allow-unpinned-eval-set for a fixture run that is not an experiment arm."
        ),
        component="answerer",
    )
    eval_set_record["rows_loaded_total"] = len(eval_rows)
    eval_set_record["rows_loaded_per_hop"] = rows_per_hop
    eval_set_record["distinct_item_ids"] = len(loaded_ids)

    # The assertion above ran on everything that loaded, which is the point of a dry run as
    # a preflight: "would a real launch be on the pinned set?". But this run then processes
    # only --dry-run-limit rows, and `pinned: true` next to 5 processed items would read as
    # a claim about the run. So the flag is downgraded and the preflight result kept beside
    # it under its own name (PR #32 review, N-4).
    rows_to_process = (
        min(len(eval_rows), max(0, args.dry_run_limit)) if args.dry_run else len(eval_rows)
    )
    eval_set_record["rows_processed"] = rows_to_process
    eval_set_record["truncated_by_dry_run_limit"] = rows_to_process < len(eval_rows)
    if eval_set_record["truncated_by_dry_run_limit"]:
        eval_set_record["pinned_on_full_load"] = eval_set_record["pinned"]
        eval_set_record["pinned"] = False
        eval_set_record["pinned_downgraded_reason"] = (
            f"--dry-run-limit cut the run to {rows_to_process} of {len(eval_rows)} loaded "
            "item(s), so this run did not process the pinned evaluation set even though the "
            "assertion above was applied to everything that loaded "
            f"(pinned_on_full_load={eval_set_record['pinned_on_full_load']})"
        )

    output_root = (
        Path(args.output_root)
        if args.output_root is not None
        else runs_path(paths_cfg, require(cfg, "output_subdir"), reader_model)
    )
    output_dir = output_root / current_run_id

    snapshot = {
        "script": Path(__file__).name,
        "created_utc": now_iso(),
        "run_id": current_run_id,
        "component": "answerer",
        "mode": mode,
        "decomposition_source": str(source_path),
        "decomposition_source_sha256": rd.sha256_file(source_path),
        "items_path": str(items_path),
        "reader_model": reader_model,
        "model_id": model_id,
        "model_name": require(model_cfg, "model_name"),
        "prompt_style": prompt_style,
        "reader_prompt_path": str(prompt_path),
        "reader_prompt_sha256": prompt_sha256,
        "context": context_cfg,
        "substitution": substitution_cfg,
        "answer_post_process": answer_cfg,
        "generation": generation,
        "generation_overrides": generation_overrides,
        "loader": {**loader, "quantization": quantization},
        "post_process": post_cfg,
        "seed": seed,
        "seeded": seeded,
        "hops": hops,
        "limit": limit,
        "device": device,
        "quantization": quantization,
        "evaluation_set": eval_set_record,
        "shared_config": config_src,
        "model_config": model_cfg.get("_config_path"),
        "output_root": str(output_root),
        "out_prefix": out_prefix,
        "dry_run": args.dry_run,
        "scoring": ANSWER_METRIC_DEFINITIONS["source"],
    }
    print(
        f"Starting answerer run {current_run_id} (mode={mode}, reader={reader_model}, "
        f"items={len(eval_rows)}, dry_run={args.dry_run})"
    )
    print(json.dumps(snapshot, indent=2, default=str))
    print(f"Evaluation set: {json.dumps(eval_set_record, default=str)}")

    if args.dry_run:
        eval_rows = eval_rows[: max(0, args.dry_run_limit)]

    # ---- model
    model = tokenizer = None
    size_record = unasserted_note("answerer", model_id)
    if not args.dry_run:
        print(f"Loading reader: {model_id} on {device} (quantization={quantization}) ...")
        tokenizer, model = rd.load_model(model_id, loader, device, quantization)
        size_record = assert_within_ceiling(
            model, component="answerer", model_id=model_id, limits=limits
        )

    # ---- execute
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir = output_dir / "prompts_log"
    prompts_dir.mkdir(exist_ok=True)

    accept_bare = bool(require(substitution_cfg, "accept_bare_references"))
    dry_answer_template = require(substitution_cfg, "dry_run_answer_template")
    paragraph_template = require(context_cfg, "paragraph_template")
    separator = require(context_cfg, "separator")
    expected_paragraphs = int(require(context_cfg, "expected_paragraphs_per_item"))
    progress_every = int(require(cfg, "progress_every"))
    prompt_log_every = int(require(cfg, "prompt_log_every"))

    per_item: list[dict[str, Any]] = []
    generation_records: list[dict[str, Any]] = []
    failed_generations = 0
    steps_prepared = 0
    steps_executed = 0
    resolved_refs = 0
    unresolved_refs = 0
    items_with_unresolved_refs = 0
    items_with_no_steps = 0
    items_with_no_gold_answer = 0
    answers_truncated = 0
    rows_at_max_new_tokens = 0
    paragraph_counts: list[int] = []

    for i, row in enumerate(eval_rows):
        if (i + 1) % progress_every == 0:
            print(f"Processed {i + 1}/{len(eval_rows)} items...")
        item = row["item"]
        paragraphs = paragraph_list(item, paragraphs_field)
        paragraph_counts.append(len(paragraphs))
        context = format_context(
            paragraphs,
            template=paragraph_template,
            separator=separator,
            idx_field=require(cfg, "fields.paragraph_idx"),
            title_field=require(cfg, "fields.paragraph_title"),
            text_field=require(cfg, "fields.paragraph_text"),
        )
        golds = gold_answer_set(item.get(answer_field), item.get(aliases_field))
        if not golds:
            items_with_no_gold_answer += 1

        steps = row["steps"]
        if not steps:
            items_with_no_steps += 1
        answers: dict[int, str] = {}
        step_records: list[dict[str, Any]] = []
        item_unresolved = 0
        log_this_item = args.dry_run or (i + 1) % prompt_log_every == 0

        for k, step_text in enumerate(steps, start=1):
            steps_prepared += 1
            sub_question, resolved, unresolved = substitute_step_references(
                step_text, answers, accept_bare=accept_bare
            )
            resolved_refs += len(resolved)
            unresolved_refs += len(unresolved)
            item_unresolved += len(unresolved)
            cost: dict[str, Any] = {
                "prompt_tokens": None,
                "completion_tokens": None,
                "latency_seconds": None,
            }
            error = None
            gen = None

            if prompt_style == "chat_template":
                messages = build_reader_messages(
                    prompt_template,
                    marker=chat_marker,
                    context=context,
                    question=sub_question,
                )
                rendered = (
                    json.dumps(messages, ensure_ascii=False, indent=2)
                    if args.dry_run
                    else tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=bool(
                            require(model_cfg, "chat_template.enable_thinking")
                        ),
                    )
                )
            else:
                rendered = fill_reader_template(
                    plain_template or prompt_template,
                    context=context,
                    question=sub_question,
                )

            if args.dry_run:
                answer = str(dry_answer_template).format(k=k)
            else:
                steps_executed += 1
                try:
                    gen = generate_answer(
                        rendered,
                        model,
                        tokenizer,
                        device,
                        generation,
                        post_cfg=post_cfg,
                        answer_cfg=answer_cfg,
                    )
                except Exception as exc:  # noqa: BLE001 - one failed step is not a failed run
                    failed_generations += 1
                    error = f"{type(exc).__name__}: {exc}"
                    answer = ""
                    print(
                        f"WARNING: generation failed for item {row['item_id']!r} step {k}: "
                        f"{error}"
                    )
                else:
                    answer = gen["answer"]
                    cost = {key: gen[key] for key in cost}
                    if gen["answer_truncated"]:
                        answers_truncated += 1
                    if gen["hit_max_new_tokens"]:
                        rows_at_max_new_tokens += 1
                    generation_records.append(dict(cost))

            answers[k] = answer
            step_records.append(
                {
                    "step": k,
                    "step_text": step_text,
                    "sub_question": sub_question,
                    "answer": answer,
                    "raw_generation": gen["text"] if gen else None,
                    "resolved_references": resolved,
                    "unresolved_references": unresolved,
                    "error": error,
                    **cost,
                }
            )

            if log_this_item:
                (
                    prompts_dir / f"prompt_item{i + 1:04d}_step{k}.txt"
                ).write_text(
                    "--- Log Header ---\n"
                    f"Item: {row['item_id']}\n"
                    f"Question: {row['question']}\n"
                    f"Step {k}/{len(steps)} as written: {step_text}\n"
                    f"Step after substitution: {sub_question}\n"
                    f"Paragraphs in context: {len(paragraphs)}\n"
                    f"\n--- Prompt ({prompt_style}) ---\n"
                    + rendered
                    + "\n--- Raw generation ---\n"
                    + (gen["text"] if gen else "")
                    + "\n--- Answer ---\n"
                    + answer
                    + "\n",
                    encoding="utf-8",
                )

        if item_unresolved:
            items_with_unresolved_refs += 1

        # The final step's answer is the item's predicted answer.
        predicted_answer = answers.get(len(steps), "") if steps else ""
        if args.dry_run:
            answer_em = answer_f1 = None
        else:
            answer_em, answer_f1 = score_answer(predicted_answer, golds)

        per_item.append(
            {
                "item_id": row["item_id"],
                "question": row["question"],
                "gold_hop_count": row["gold_hop_count"],
                "predicted_answer": predicted_answer,
                "gold_answers": golds,
                "answer_em": answer_em,
                "answer_f1": answer_f1,
                "step_count": len(steps),
                "paragraphs_in_context": len(paragraphs),
                "unresolved_reference_count": item_unresolved,
                "steps": step_records,
            }
        )

    overall = aggregate_answers(per_item)
    per_hop = per_gold_hop_metrics(per_item)
    gold_hop_distribution: dict[str, int] = {}
    for it in per_item:
        key = str(it["gold_hop_count"])
        gold_hop_distribution[key] = gold_hop_distribution.get(key, 0) + 1

    metrics: dict[str, Any] = {
        "script": Path(__file__).name,
        "created_utc": now_iso(),
        "mode": mode,
        "dry_run": args.dry_run,
        "seed": seed,
        "seeded": seeded,
        "reader_model": reader_model,
        "model_id": model_id,
        "model_size": size_record,
        "decomposition_source": str(source_path),
        "items_path": str(items_path),
        "answer_em": overall["answer_em"],
        "answer_f1": overall["answer_f1"],
        "per_gold_hop_metrics": per_hop,
        "gold_hop_distribution": gold_hop_distribution,
        "counts": {
            "items_in_source": rows_input,
            "items_after_limit": len(rows),
            "items_missing_musique_item": len(missing_items),
            "items_evaluated": len(per_item),
            "items_scored": overall["num_items"],
            "items_with_no_steps": items_with_no_steps,
            "items_with_no_gold_answer": items_with_no_gold_answer,
            "pinned_ids_requested": len(pinned_ids),
            "pinned_ids_missing_from_gold_file": len(gold_missing_ids),
            "sub_questions_prepared": steps_prepared,
            "sub_questions_executed": steps_executed,
            "failed_generations": failed_generations,
            "references_resolved": resolved_refs,
            "references_unresolved": unresolved_refs,
            "items_with_unresolved_references": items_with_unresolved_refs,
            "answers_truncated_at_max_chars": answers_truncated,
            "rows_at_max_new_tokens": rows_at_max_new_tokens if not args.dry_run else None,
            "gold_hop_id_prefix_disagreements": len(hop_prefix_disagreements),
        },
        "context_stats": {
            "policy": policy,
            "expected_paragraphs_per_item": expected_paragraphs,
            "min_paragraphs_per_item": min(paragraph_counts) if paragraph_counts else None,
            "max_paragraphs_per_item": max(paragraph_counts) if paragraph_counts else None,
            "mean_paragraphs_per_item": (
                statistics.fmean(paragraph_counts) if paragraph_counts else None
            ),
            "items_below_expected_paragraphs": sum(
                1 for n in paragraph_counts if n < expected_paragraphs
            ),
        },
        "evaluation_set": eval_set_record,
        "cost": cost_summary(generation_records),
        "metric_definitions": ANSWER_METRIC_DEFINITIONS,
        "max_new_tokens": int(require(generation, "max_new_tokens")),
    }
    if args.dry_run:
        metrics["answer_metrics_note"] = (
            "unmeasured: --dry-run generates nothing, so answer_em, answer_f1 and the "
            "per-hop block are null rather than zero. The join, the context assembly and the "
            "[#k] substitution chain did run; substituted answers were the configured "
            "dry-run stub."
        )
    if failed_generations:
        metrics["failed_generation_note"] = (
            f"{failed_generations} sub-question generation(s) raised and were recorded as an "
            "empty answer, which leaves any [#k] that referenced them unresolved. The items "
            "are still scored, so these failures are inside the reported EM/F1, not excluded "
            "from them."
        )
    if hop_prefix_disagreements:
        metrics["gold_hop_note"] = (
            f"{len(hop_prefix_disagreements)} item(s) have a gold step count that disagrees "
            "with the hop depth in their id; the per-hop table is keyed on the gold step "
            f"count. First offenders: {hop_prefix_disagreements[:10]}"
        )

    per_item_path = output_dir / f"{out_prefix}_per_item.json"
    per_item_path.write_text(
        json.dumps(
            {
                "schema": PER_ITEM_SCHEMA,
                "created_utc": metrics["created_utc"],
                "mode": mode,
                "decomposition_source": str(source_path),
                "items_path": str(items_path),
                "reader_model": reader_model,
                "dry_run": args.dry_run,
                "items": per_item,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    write_run_artifacts(
        output_dir,
        config_snapshot=snapshot,
        metrics=metrics,
        note_title=f"MuSiQue answering {'dry run' if args.dry_run else 'run'} - {current_run_id}",
        note_lines=[
            f"- Mode: {mode} (`{source_path}`)",
            f"- Reader: `{reader_model}` ({model_id}), prompt `{prompt_path}` "
            f"(style: {prompt_style})"
            + ("" if not args.dry_run else " - model not loaded"),
            f"- Context: {policy} - every paragraph of the item "
            f"(mean {metrics['context_stats']['mean_paragraphs_per_item']}, "
            f"min {metrics['context_stats']['min_paragraphs_per_item']}, "
            f"{metrics['context_stats']['items_below_expected_paragraphs']} item(s) below "
            f"{expected_paragraphs})",
            f"- Items: {len(per_item)} evaluated of {rows_input} in the source "
            f"(unmatched: {len(missing_items)}; per gold hop: {gold_hop_distribution})",
            f"- Evaluation set pinned: {eval_set_record['pinned']} "
            f"(ids checked: {eval_set_record['id_identity_checked']}; "
            f"source: {eval_set_record['rows_source']})"
            + (
                f" — downgraded: {eval_set_record['pinned_downgraded_reason']}"
                if eval_set_record.get("pinned_downgraded_reason")
                else ""
            ),
            (
                f"- Answer EM: {overall['answer_em']:.4f} / F1: {overall['answer_f1']:.4f} "
                f"over {overall['num_items']} item(s)"
                if overall["answer_em"] is not None
                else "- Answer EM / F1: unmeasured (nothing was generated in this run)"
            ),
            "- Per gold hop: "
            + json.dumps(
                {
                    hop: {
                        "n": block["num_items"],
                        "em": block["answer_em"],
                        "f1": block["answer_f1"],
                    }
                    for hop, block in per_hop.items()
                }
            ),
            f"- Sub-questions prepared: {steps_prepared}; executed: {steps_executed}; "
            f"failed generations: "
            f"{failed_generations}; references resolved/unresolved: "
            f"{resolved_refs}/{unresolved_refs}",
            (
                f"- Parameters: {size_record['parameter_count']:,} "
                f"(ceiling {size_record['parameter_ceiling']:,})"
                if size_record["ceiling_asserted"]
                else "- Parameter ceiling: not asserted (no model was loaded)."
            ),
            (
                f"- Cost per sub-question: "
                f"{metrics['cost']['mean_prompt_tokens_per_subquestion']:.1f} prompt + "
                f"{metrics['cost']['mean_completion_tokens_per_subquestion']:.1f} completion "
                f"tokens, {metrics['cost']['mean_latency_seconds_per_subquestion']:.3f}s "
                f"(means over {metrics['cost']['rows_measured']} generations)"
                if metrics["cost"]["rows_measured"]
                else "- Cost per sub-question: unmeasured (nothing was generated)."
            ),
            f"- Scoring: {ANSWER_METRIC_DEFINITIONS['source']}",
            f"- Per-item: `{per_item_path}`",
        ],
        prefix=f"{out_prefix}_",
    )
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
