#!/usr/bin/env python3
"""Router component: predict a question's hop count.

One runner for every model. v1 kept ten near-identical copies of ``router.py``
(one per model folder); they differed only in the model id, the generation length,
the tokenizer/loader flags and the response-parsing rules. Those differences are
now fields in ``components/router/models/<model>/config.json`` and the code lives
here once. Prompts stay per-model and byte-identical to v1.

Three prompting modes, none of which trains anything. Which one runs is decided by the
prompt file and by the config's ``few_shot`` block:

- **static few-shot** — ``models/<model>/prompt.md``, v1's hand-written MetaQA examples
  (``configs/router.json``, the default);
- **zero-shot** — ``models/prompt_zero_shot.md`` (``--prompt-file``);
- **retrieved few-shot** — ``models/prompt_few_shot_musique.md`` with
  ``configs/router_musique.json`` (issue #27). The exemplars come from the same retrieval
  artifact the decomposer reads, each labelled with its **own** gold hop depth, and the
  query is excluded from its own exemplars by id and by question text (the decomposer's
  rule, imported rather than re-implemented).

Usage::

    python components/router/run_router.py --model qwen2_5_0_5b
    python components/router/run_router.py --model qwen2_5_0_5b --prompt-file prompt_zero_shot.md
    python components/router/run_router.py --model qwen2_5_0_5b --dry-run   # no model load

    # the few-shot-prompted router over a question source that carries ids (issue #27)
    python components/router/run_router.py --model qwen2_5_0_5b \\
        --config router_musique.json --prompt-file prompt_few_shot_musique.md \\
        --retrieval-input <the pinned top-5 JSONL over the 600 questions of ADR 0007>

Every run writes a config snapshot, a metrics JSON and a run note under the
configured runs root, and asserts the model's parameter count against the router
ceiling in ``configs/model_limits.json``.

A run whose question source carries ids (``questions_format: jsonl``) also writes a
**predictions JSONL keyed by query id** — one object per query with the id and the
predicted hop, the shape ADR 0022 item 5 documents. That is what lets the two consumers
join on the id instead of on question text: the retrieval chain's ``predictions`` hop
source (issue #15's router-hop-matched condition) and the decomposer's ``router_guided``
condition. v1's ``detailed_results.json`` was keyed by question text only, which made an
exact-string match load-bearing on the very field the masking modes rewrite.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
#: Three rules this component must not own a second copy of: how a question source with ids
#: is read, what "this exemplar IS the query" means, and what the pinned ADR 0007 evaluation
#: set is (by id, not only by count). They live in the decomposer's runner and are *imported*
#: here rather than duplicated, the same way ``components/answerer/run_answerer.py`` imports
#: the eval-set assertion — a second copy of a leakage rule is a rule that drifts.
#:
#: Inserted BEFORE ``src`` so that ``src`` ends up first on the path: the shared modules must
#: not be shadowed by a same-named file in a component directory (PR #43 review, N3).
sys.path.insert(0, str(_REPO_ROOT / "components" / "decomposer"))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from hop_matching import parse_hop_from_id  # noqa: E402
from model_size import assert_within_ceiling, load_limits, unasserted_note  # noqa: E402
from run_artifacts import now_iso, run_id, write_run_artifacts  # noqa: E402
from run_config import (  # noqa: E402
    data_path,
    load_config,
    load_paths,
    optional,
    require,
    resolve_path,
    runs_path,
)
from seeding import new_rng, set_global_seed  # noqa: E402
from step_lines import post_process_generation  # noqa: E402

import run_decomposer as rd  # noqa: E402

#: Number words, for the response-parsing fallbacks. Covers 1-9 rather than v1's 1-3 so the
#: fallback follows the config's ``hops`` instead of MetaQA's: with 1/2/3 only, a MuSiQue
#: router's "four hops" fell through to ``parsing.default_hop`` and was scored as a
#: prediction (PR #43 review, I1). The upper bound is 9 because the digit-class rule builds a
#: single-character regex class; :func:`assert_parsing_covers_hops` refuses a two-digit hop
#: depth rather than mis-parsing it.
_NUMBER_WORDS = {
    "ONE": 1, "TWO": 2, "THREE": 3, "FOUR": 4, "FIVE": 5,
    "SIX": 6, "SEVEN": 7, "EIGHT": 8, "NINE": 9,
}

#: The largest hop depth the digit-class parsing rule can read (single character class).
MAX_PARSEABLE_HOP = 9

#: How many offending ids / rows an error message lists before it truncates.
_MAX_REPORTED = 10

#: Where a retrieved exemplar's hop label comes from. ``pool_id`` parses the coarse hop
#: depth out of the exemplar's MuSiQue id (``2hop__…``) — the same quantity the router is
#: scored against, since the gold depth of a query is parsed from *its* id the same way. A
#: second source (for instance the step count of the exemplar's gold decomposition, which is
#: what the decomposer's exemplar hop lines state) would teach the prompt a different
#: quantity than the metric measures, so it is not implemented: an unknown value is refused
#: here rather than approximated, and adding one is a visible edit.
EXEMPLAR_HOP_SOURCES = ("pool_id",)


def build_prompt(template: str, question: str, few_shot_examples: str = "") -> str:
    """Fill the placeholders the template actually has.

    ``{question}`` / ``{{question}}`` is v1's; ``{few_shot_examples}`` is the retrieved
    few-shot mode's. Only placeholders present are filled (the decomposer's ``fill_template``
    rule), so a v1 prompt renders exactly as before.
    """
    if "{{question}}" in template:
        return template.replace("{{question}}", question).replace(
            "{{few_shot_examples}}", few_shot_examples
        )
    values: dict[str, str] = {}
    if "{question}" in template:
        values["question"] = question
    if "{few_shot_examples}" in template:
        values["few_shot_examples"] = few_shot_examples
    if not values:
        return template
    return template.format(**values)


def _sample(items: list[str]) -> str:
    shown = items[:_MAX_REPORTED]
    suffix = f" (+{len(items) - len(shown)} more)" if len(items) > len(shown) else ""
    return ", ".join(repr(s) for s in shown) + suffix


# ------------------------------------------------------------------- few-shot


def format_few_shot_examples(examples: list[dict]) -> str:
    """One ``Q:``/``A:`` block per exemplar, whose answer is that exemplar's own hop count.

    The block shape is the one the prompt asks the model to produce and the one the model
    folder's ``parsing.answer_regex`` reads back out, so the exemplars demonstrate the output
    format as well as the task.
    """
    return "\n\n".join(f"Q: {ex['question']}\nA: {ex['hop_count']}" for ex in examples)


def examples_from_retrieval_row(
    row: dict, *, mode: str, k: int, exemplar_hop_source: str
) -> tuple[list[dict], int]:
    """``k`` (question, hop count) exemplars from one retrieval row's ranked candidates.

    The decomposer's rule, applied to a different payload: candidates that **are** the query
    are dropped before the top-k is taken (:func:`run_decomposer.is_self_example`, by pool id
    and by whitespace/case-normalized question text), so the router is never shown the query
    it is about to predict. Returns ``(examples, self_excluded_count)``.

    Every refusal here is loud, because each one would otherwise put a wrong number in the
    prompt: fewer than k usable candidates, an exemplar whose hop depth does not parse from
    its pool id, an exemplar with no question text.
    """
    if exemplar_hop_source not in EXEMPLAR_HOP_SOURCES:
        raise SystemExit(
            f"unknown few_shot.exemplar_hop_source {exemplar_hop_source!r} (expected one of "
            f"{list(EXEMPLAR_HOP_SOURCES)})"
        )
    key = f"{mode}_top_k"
    candidates = row.get(key) or []
    query_id = row.get("query_id") or "<unknown>"
    kept: list[dict] = []
    self_excluded = 0
    for idx, cand in enumerate(candidates):
        if not isinstance(cand, dict):
            raise SystemExit(
                f"[router] query_id={query_id!r} candidate {idx} in {key!r} is not an "
                f"object: type={type(cand).__name__}"
            )
        if rd.is_self_example(
            cand,
            exclude_query_id=row.get("query_id"),
            exclude_question=row.get("query_question"),
        ):
            self_excluded += 1
            continue
        kept.append(cand)
    if len(kept) < k:
        raise SystemExit(
            f"[router] retrieval row for query_id={query_id!r} has only {len(kept)} usable "
            f"candidate(s) under {key!r} ({len(candidates)} listed, {self_excluded} dropped "
            f"as the query itself), need k={k}. Rebuild the retrieval input with the correct "
            f"k, or lower few_shot k in the config."
        )
    examples: list[dict] = []
    query_question = row.get("query_question")
    for cand in kept[:k]:
        hop = parse_hop_from_id(cand.get("pool_id"))
        if hop is None:
            raise SystemExit(
                f"[router] query_id={query_id!r}: exemplar pool_id="
                f"{cand.get('pool_id')!r} carries no parseable hop depth (expected an id "
                f"like '2hop__…'), so its 'A:' line cannot state its own hop count. There is "
                f"deliberately no fallback to the query's hop depth: that would put the "
                f"answer to the query into its own prompt."
            )
        question = cand.get("pool_question")
        if not isinstance(question, str) or not question.strip():
            raise SystemExit(
                f"[router] query_id={query_id!r}: exemplar pool_id="
                f"{cand.get('pool_id')!r} has no usable 'pool_question'."
            )
        examples.append(
            {"pool_id": cand.get("pool_id"), "question": question.strip(), "hop_count": hop}
        )
    # The point of the exclusion is that no exemplar is the query; assert it on the kept set
    # rather than trusting the filter above (the decomposer asserts the same thing after its
    # own filter - PR #43 review, N1).
    wanted = rd.normalize_for_self_exclusion(query_question)
    for ex in examples:
        if wanted and rd.normalize_for_self_exclusion(ex["question"]) == wanted:
            raise AssertionError(
                f"[router] query_id={query_id!r} kept an exemplar identical to the query "
                f"after self-exclusion (pool_id={ex['pool_id']!r}); this is a bug in "
                f"examples_from_retrieval_row."
            )
    return examples, self_excluded


# ---------------------------------------------------------------- query ids


def assert_query_ids(rows: list[dict], *, source: str, reason: str) -> None:
    """Refuse rows whose query id is missing or repeated. Both would break the join.

    A row with no id cannot appear in the predictions file at all (the consumer would have
    no key for it); two rows with the same id would make the file say two different things
    about one query, which is what ADR 0022 item 3 refuses on the reading side. Caught before
    a model is loaded, so a multi-hour run cannot end in an unjoinable predictions file.
    """
    missing = [
        f"index {i}"
        for i, row in enumerate(rows)
        if not isinstance(row.get("query_id"), str) or not str(row["query_id"]).strip()
    ]
    if missing:
        raise SystemExit(
            f"[router] {len(missing)} of {len(rows)} rows from {source} have no usable query "
            f"id: {_sample(missing)}.\n{reason}"
        )
    seen: set[str] = set()
    duplicates: list[str] = []
    for row in rows:
        qid = str(row["query_id"]).strip()
        if qid in seen:
            duplicates.append(qid)
        seen.add(qid)
    if duplicates:
        raise SystemExit(
            f"[router] {len(duplicates)} query id(s) appear more than once in {source}: "
            f"{_sample(sorted(set(duplicates)))}. One row per query id.\n{reason}"
        )


def build_prediction_rows(
    rows: list[dict],
    predictions: list[int],
    parsed_flags: list[bool],
    *,
    id_field: str,
    hop_field: str,
) -> list[dict]:
    """The predictions JSONL rows: a join key, a predicted hop, and how it was obtained.

    Deliberately narrow. ``expected_hop``/``correct`` are the gold depth this prediction is
    scored against, and ``parse_fallback`` says the model's response carried no hop count so
    the model folder's ``parsing.default_hop`` was used — a consumer of a routing decision
    should be able to see that it was a default rather than a prediction.

    No question text: ADR 0022 item 5 rejected keying this join on question text, so the
    text is not carried here to be keyed on by accident. The run's ``detailed_results.json``
    keeps it for debugging.
    """
    if not (len(rows) == len(predictions) == len(parsed_flags)):
        raise AssertionError(
            f"[router] prediction rows misaligned: {len(rows)} queries, "
            f"{len(predictions)} predictions, {len(parsed_flags)} parse flags"
        )
    out: list[dict] = []
    for row, hop, parsed in zip(rows, predictions, parsed_flags):
        out.append(
            {
                id_field: str(row["query_id"]).strip(),
                hop_field: int(hop),
                "expected_hop": row["expected_hop"],
                "correct": int(hop) == row["expected_hop"],
                "parse_fallback": not parsed,
            }
        )
    return out


def write_predictions_jsonl(path: Path, rows: list[dict]) -> None:
    """One JSON object per line, in query order (the shape ADR 0022 item 5 documents)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_response(response: str, truncate_at: list[str] | tuple[str, ...] | None) -> str:
    """The part of a response that is the model's answer, before any hop count is read.

    Two steps, in this order:

    1. ``<think>`` blocks and outer whitespace go, with the shared helper the decomposer
       uses (``src/step_lines.py``), so there is one definition of "generation artifact";
    2. the text is cut at the first of the configured tail markers.

    The order matters and is the reason this is not a single call to
    ``post_process_generation``: that helper truncates *before* stripping outer whitespace,
    so a response beginning with a newline would be cut down to the empty string by a
    ``"\\n"`` marker.

    Why truncation exists at all (PR #43 review, C1). The retrieved-few-shot prompt ends with
    the ``A:`` cue, so the model's own answer is the *first* thing in the response - and a
    model with tokens left over then happily writes a fresh ``Q: … A: …`` pair of its own.
    Read against the untruncated response, an ``A:\\s*(\\d)`` rule matches that regurgitated
    pair instead of the answer, and returns another question's hop count as this query's
    prediction with ``parse_fallback`` false. Cutting the response at its first line ends the
    read at the answer. Markers come from the config, because what counts as the tail depends
    on the prompt: the v1 MetaQA prompts elicit a multi-line ``Trace:``/``A:`` response and
    set no markers at all, so their parsing is unchanged.
    """
    cleaned = post_process_generation(response, strip_think=True, truncate_at=None)
    for marker in truncate_at or []:
        cleaned = cleaned.split(marker)[0]
    return cleaned.strip()


def hop_word_labels(hop: int) -> tuple[str, ...]:
    """The written forms of "<hop> hops" the ``hop_words`` fallback recognises.

    Derived from the hop depth rather than listed per depth, so the fallback follows the
    config's ``hops`` and cannot silently cover 1/2/3 only (PR #43 review, I1).
    """
    word = next((w for w, v in _NUMBER_WORDS.items() if v == hop), None)
    labels = [f"{hop}-hop", f"{hop} hop"]
    if word:
        labels += [f"{word.lower()}-hop", f"{word.lower()} hop"]
    return tuple(labels)


def assert_parsing_covers_hops(parsing: dict, hops: list[int], *, src: str) -> dict[str, Any]:
    """Refuse a config whose parsing rules cannot express every hop depth it may predict.

    A source-level guard for what the configs used to say in prose only (ADR 0016: an
    invariant that matters is asserted, not documented). Three refusals:

    - a hop depth above :data:`MAX_PARSEABLE_HOP`, because the digit-class rule is a
      single-character regex class and a two-digit depth would read as its first digit;
    - a hop depth the configured ``fallback_labels`` table has no word for, which would send
      an otherwise readable response ("four hops") to ``default_hop``;
    - a ``default_hop`` outside ``hops``, which would score every defaulted row against a
      class the run does not evaluate.

    The reason this is a refusal and not a warning (PR #43 review, I1): the default is
    returned *as if it were a prediction*, so an uncovered depth does not look like a
    failure - it looks like a router that always answers ``default_hop``, and on a set with
    200 questions per hop that is a plausible-looking accuracy number.
    """
    if not hops:
        raise SystemExit(f"{src} declares no 'hops', so there is nothing to predict")
    too_deep = [h for h in hops if h < 1 or h > MAX_PARSEABLE_HOP]
    if too_deep:
        raise SystemExit(
            f"{src} declares hop depth(s) {too_deep} outside 1..{MAX_PARSEABLE_HOP}. The "
            f"response-parsing rules read a single digit (a regex character class), so a "
            f"two-digit depth would be read as its first digit. Widening the range means "
            f"changing the parsing rules, not the config."
        )
    fallback = require(parsing, "fallback_labels")
    if fallback == "hop_words":
        uncovered = [h for h in hops if not hop_word_labels(h)]
        covered = {h: list(hop_word_labels(h)) for h in hops}
    elif fallback == "number_words":
        words = {v: w for w, v in _NUMBER_WORDS.items()}
        uncovered = [h for h in hops if h not in words]
        covered = {h: [words[h]] for h in hops if h in words}
    else:
        raise SystemExit(f"{src}: unknown parsing.fallback_labels {fallback!r}")
    if uncovered:
        raise SystemExit(
            f"{src}: the {fallback!r} response-parsing fallback has no label for hop "
            f"depth(s) {uncovered}, so a response naming one in words would fall through to "
            f"parsing.default_hop and be recorded as a prediction. Extend _NUMBER_WORDS in "
            f"{Path(__file__).name} (and its test) rather than accepting the gap."
        )
    default_hop = int(require(parsing, "default_hop"))
    if default_hop not in hops:
        raise SystemExit(
            f"{src}: parsing.default_hop={default_hop} is not one of hops {hops}. Every "
            f"unreadable response is recorded as that depth, so a default outside the "
            f"evaluated classes would be counted as wrong for every query by construction. "
            f"Set it with parsing_overrides if the model folder's value does not fit."
        )
    return {
        "hops": list(hops),
        "fallback_labels": fallback,
        "labels_per_hop": covered,
        "default_hop": default_hop,
        "digit_class": "[" + "".join(str(h) for h in hops) + "]",
        "asserted": True,
    }


def parse_hop_response(response: str, question: str, parsing: dict) -> tuple[int, bool]:
    """Extract a hop count from the model response, per the model's parsing config.

    Mirrors the three parsing variants that existed across v1's router copies:
    an answer-prefix regex (``A:`` or ``Output|A:``), a first-digit fallback, then
    either hop-word labels ("2-hop", "two hop") or number-word labels (ONE/TWO/THREE).

    Returns ``(hop, parsed)``. ``parsed`` is False when nothing in the response could be
    read and ``parsing.default_hop`` was used instead: v1 logged that case to stdout and
    then returned the default as if it were a prediction, so a run's accuracy silently
    mixed predictions with defaults. The flag is what lets the metrics count them, and it
    travels into the predictions file as ``parse_fallback``.
    """
    hops: list[int] = parsing["hops"]
    digit_class = "[" + "".join(str(h) for h in hops) + "]"
    default_hop = int(require(parsing, "default_hop"))

    clean = clean_response(response, require(parsing, "truncate_at"))

    answer_regex = optional(parsing, "answer_regex")
    if answer_regex:
        flags = re.IGNORECASE if parsing.get("answer_regex_ignorecase") else 0
        if match := re.search(answer_regex, clean, flags):
            return int(match.group(1)), True

    if match := re.search(digit_class, clean):
        return int(match.group()), True

    fallback = require(parsing, "fallback_labels")
    if fallback == "hop_words":
        lower = clean.lower()
        for hop in hops:
            if any(w in lower for w in hop_word_labels(hop)):
                return hop, True
    elif fallback == "number_words":
        upper = clean.upper()
        for label, value in _NUMBER_WORDS.items():
            if value in hops and label in upper:
                return value, True
    else:
        raise SystemExit(f"unknown parsing.fallback_labels: {fallback!r}")

    if numbers := re.findall(digit_class, clean):
        return int(numbers[0]), True

    print(
        f"Warning: no valid hop count in response for: '{question[:50]}...'. "
        f"Defaulting to {default_hop}."
    )
    if parsing.get("debug_print_response"):
        print(f"DEBUG: Full model response was: '{response}'")
    return default_hop, False


def classify_hop_count(
    prompt: str,
    question: str,
    model,
    tokenizer,
    device: str,
    generation: dict,
    parsing: dict,
) -> tuple[int, bool]:
    """Generate on one already-assembled prompt and read the hop count out of the response.

    The prompt is assembled by the caller (once per query, before anything is loaded) so
    that the retrieved-few-shot mode, the dry run and the real run all render the *same*
    text, and so a ``--dry-run`` exercises prompt assembly for real.

    ``eos_token_id`` is passed explicitly (as the decomposer does) so a model that has
    finished its answer can stop instead of spending the rest of the token budget writing
    further question/answer pairs of its own. It is the same value ``generate`` would take
    from the model's generation config, so this changes no v1 decoding; it is the stopping
    behaviour that is *stated* rather than assumed. It is not what makes the parsing safe -
    :func:`clean_response` is, because a model can always keep going (PR #43 review, C1).
    """
    import torch

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

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

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()

    try:
        return parse_hop_response(response, question, parsing)
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover - defensive, matches v1 behaviour
        default_hop = int(require(parsing, "default_hop"))
        print(f"Error parsing response for '{question[:50]}...': {exc}. Defaulting to {default_hop}.")
        return default_hop, False


# -------------------------------------------------------------- question sources


def apply_overrides(base: dict, overrides: dict | None, *, block: str, src: str) -> dict:
    """Overlay config-level overrides on a per-model block, refusing unknown keys.

    Same shape and reasoning as the decomposer's ``apply_generation_overrides``: the model
    folders are v1 assets (``prompt.md`` byte-identical, ``parsing`` mirroring v1's code), so
    a config that needs a different value overrides it here instead of editing the asset. A
    key the model's block does not define is refused rather than added, because a typo would
    otherwise be a setting that silently does nothing.
    """
    merged = dict(base)
    if not overrides:
        return merged
    unknown = sorted(set(overrides) - set(base))
    if unknown:
        raise SystemExit(
            f"{block}_overrides in {src} sets {unknown}, which the model's {block!r} block "
            f"does not define (has: {sorted(base)})"
        )
    merged.update(overrides)
    return merged


def resolve_retrieval_input(cli_value: str | None, cfg: dict, paths_cfg: dict) -> str | None:
    """The retrieved-few-shot exemplar source: ``--retrieval-input``, then ``input_key``.

    ``retrieval.input_key`` names a ``datasets.<key>`` in the paths config, which is this
    repo's convention for a path to data outside the tree (no absolute path in a config).
    ``--retrieval-input`` wins, which is how the fixture and smoke runs point at
    ``tests/fixtures/retrieval/``. A few-shot run with no exemplar source is refused rather
    than silently reduced to a zero-shot run under a few-shot label.
    """
    src = cfg.get("_config_path", "<config>")
    explicit = optional(cfg, "retrieval.input")
    input_key = optional(cfg, "retrieval.input_key")
    if explicit and input_key:
        raise SystemExit(
            f"{src} sets both 'retrieval.input' ({explicit!r}) and 'retrieval.input_key' "
            f"({input_key!r}); set exactly one, so the config names one retrieval file."
        )
    value = cli_value or explicit
    if not value and input_key:
        value = str(data_path(paths_cfg, input_key))
    return value or None


def rows_from_questions(
    *,
    paths_cfg: dict,
    cfg: dict,
    hops: list[int],
    data_root: Path,
    seed: int,
    sample_size,
) -> list[dict]:
    """``{query_id, question, expected_hop}`` per query, from the per-hop question files.

    The gold hop depth is the depth of the file the question was read from. ``lines`` files
    (MetaQA) carry no id, so ``query_id`` is None for them and this run writes no predictions
    file; ``jsonl`` files (MuSiQue) carry one.

    Sampling stays exactly v1's, seeded before it draws (ADR 0005): one RNG seeded with the
    run seed, hops drawn in config order, ``min(len(pool), sample_size)`` from each.
    """
    template = require(paths_cfg, "datasets." + require(cfg, "questions_template_key"))
    questions_format = require(cfg, "questions_format")
    question_field = id_field = ""
    if questions_format == "jsonl":
        question_field = require(cfg, "questions_jsonl.question_field")
        id_field = require(cfg, "questions_jsonl.id_field")

    items_by_hop: dict[int, list[dict]] = {}
    for hop in hops:
        path = resolve_path(template.format(hop=hop), data_root)
        if questions_format == "lines" and not path.exists():
            # v1 warned and carried on with an empty hop; the "no questions at all" refusal
            # below is what catches a wholly mis-resolved data root.
            print(f"Warning: {path} not found.")
            items_by_hop[hop] = []
            continue
        items_by_hop[hop] = rd.load_question_items(
            path,
            questions_format=questions_format,
            question_field=question_field,
            id_field=id_field,
        )
    if sample_size:
        rng = new_rng(seed)
        for hop in hops:
            pool = items_by_hop[hop]
            items_by_hop[hop] = rng.sample(pool, min(len(pool), int(sample_size)))

    rows: list[dict] = []
    for hop in hops:
        for item in items_by_hop[hop]:
            rows.append(
                {
                    "query_id": item["query_id"],
                    "question": item["question"],
                    "expected_hop": hop,
                }
            )
    return rows


def rows_from_retrieval(path: Path, *, sample_size) -> list[dict]:
    """``{query_id, question, expected_hop, retrieval_row}`` per query, from a retrieval JSONL.

    The retrieval artifact is the question source *and* the exemplar source, exactly as it is
    for the decomposer: one row per query, carrying the query id, the query question and the
    ranked candidate lists. The gold hop depth is parsed from the query id and an id it
    cannot parse is a refusal, never a guess — the router's accuracy is measured against it.

    ``sample_size_per_hop`` is refused on this path: the artifact is built over a pinned query
    set (ADR 0007/0014), and a sampled subset would produce a predictions file that does not
    cover the queries its consumers ask about.
    """
    if sample_size:
        raise SystemExit(
            f"sample_size_per_hop={sample_size!r} cannot be combined with a retrieval input: "
            f"{path} is built over a pinned query set, and a sampled run would write a "
            f"predictions file that does not cover every query its consumers join on. Set "
            f"sample_size_per_hop to null (or drop --retrieval-input)."
        )
    raw = rd.load_jsonl(path)
    if not raw:
        raise SystemExit(f"no rows in retrieval input: {path}")
    rows: list[dict] = []
    unparseable: list[str] = []
    no_question: list[str] = []
    for idx, row in enumerate(raw):
        question = row.get("query_question")
        if not isinstance(question, str) or not question.strip():
            no_question.append(f"index {idx}: query_id={row.get('query_id')!r}")
            continue
        hop = parse_hop_from_id(row.get("query_id"))
        if hop is None:
            unparseable.append(f"index {idx}: {row.get('query_id')!r}")
            continue
        rows.append(
            {
                "query_id": row.get("query_id"),
                "question": question.strip(),
                "expected_hop": hop,
                "retrieval_row": row,
            }
        )
    if no_question:
        raise SystemExit(
            f"{len(no_question)} of {len(raw)} rows in {path} have no usable "
            f"'query_question': {_sample(no_question)}"
        )
    if unparseable:
        raise SystemExit(
            f"{len(unparseable)} of {len(raw)} rows in {path} have a query_id whose hop depth "
            f"cannot be parsed (expected an id like '2hop__…'): {_sample(unparseable)}. That "
            f"depth is the gold label the router's accuracy is measured against, so it is a "
            f"refusal rather than a guess."
        )
    return rows


#: What the accuracy numbers mean, recorded next to them. A metrics file read on its own has
#: to say what "accuracy" was computed against, and a per-hop row of a routing table is easy
#: to read as precision when it is recall (PR #43 review, I2).
ACCURACY_DEFINITIONS = {
    "gold_hop_count": (
        "the query's GOLD hop depth: parsed from its MuSiQue id ('2hop__…' -> 2) on the "
        "retrieval path, or the depth of the per-hop question file it was read from. Never a "
        "prediction."
    ),
    "overall_accuracy": (
        "exact-hop accuracy: predicted hop == gold hop, over every query in the run. Rows "
        "whose response carried no readable hop count are INCLUDED, scored at "
        "parsing.default_hop (see unparsed_response_rows)."
    ),
    "hop_<h>_accuracy": (
        "within-gold-class recall: of the queries whose GOLD depth is h, the fraction "
        "predicted h. It is not precision - it says nothing about how many other queries "
        "were also predicted h, so the rows do not decompose a confusion matrix."
    ),
    "hop_<h>_total": "how many queries have gold depth h (the denominator above)",
    "unparsed_response_rows": (
        "rows whose response carried no readable hop count, so parsing.default_hop was "
        "recorded as the prediction. They are counted in the accuracies above, which means a "
        "high count inflates the accuracy of whichever class default_hop belongs to; "
        "unparsed_response_rows_per_gold_hop breaks them out."
    ),
}


def unparsed_rows_per_hop(
    all_expected: list[int], parsed_flags: list[bool], hops: list[int]
) -> dict[str, int]:
    """How many defaulted rows sit in each gold hop class.

    Reported because a defaulted row is scored as a prediction: with default_hop = 2, every
    unreadable response is correct for a 2-hop query and wrong for a 3- or 4-hop one, so
    where the defaults fell decides how much of a per-hop number is real (PR #43 review, I1).
    """
    return {
        str(hop): sum(
            1 for expected, ok in zip(all_expected, parsed_flags) if expected == hop and not ok
        )
        for hop in hops
    }


def compute_metrics(
    predictions: list[int],
    all_expected: list[int],
    run_seed: int,
    model_id: str,
    current_run_id: str,
    hops: list[int],
    run_idx: int | None = None,
) -> dict:
    """Metrics for one run. See :data:`ACCURACY_DEFINITIONS` for what they measure."""
    correct = sum(1 for p, e in zip(predictions, all_expected) if p == e)
    accuracy = correct / len(predictions) if predictions else 0.0
    per_hop: dict[str, float | int] = {}
    for h in hops:
        total = all_expected.count(h)
        match = sum(1 for p, e in zip(predictions, all_expected) if e == h and p == e)
        per_hop[f"hop_{h}_accuracy"] = match / total if total > 0 else 0.0
        per_hop[f"hop_{h}_total"] = total
    metrics = {
        "overall_accuracy": accuracy,
        "total_questions": len(predictions),
        "correct_predictions": correct,
        **per_hop,
        "seed": run_seed,
        "model": model_id,
        "run_id": current_run_id,
    }
    if run_idx is not None:
        metrics["run_index"] = run_idx
    return metrics


def load_model(model_id: str, loader: dict, device: str):
    """Load tokenizer + model with this model's loader flags."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if loader.get("print_gpu_info") and device == "cuda":
        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU: {name} (total VRAM: {total:.1f} GB)")

    trust_remote_code = bool(require(loader, "trust_remote_code"))
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        use_fast=bool(require(loader, "use_fast_tokenizer")),
    )

    dtype_name = require(loader, "cuda_dtype") if device == "cuda" else require(loader, "cpu_dtype")
    dtype = getattr(torch, dtype_name)
    # `dtype=` is the forward-compatible spelling; transformers 5.x accepts both and
    # maps the older `torch_dtype=` onto it.
    model_kwargs = {"trust_remote_code": trust_remote_code, "dtype": dtype}
    if device == "cuda":
        model_kwargs["device_map"] = require(loader, "device_map_cuda")

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    if device != "cuda":
        model = model.to(device)
    model.eval()
    return tokenizer, model


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Model folder under components/router/models/")
    p.add_argument("--config", default="router.json", help="Shared router config (default: configs/router.json)")
    p.add_argument("--prompt-file", default=None, help="Override the model's prompt file (e.g. prompt_zero_shot.md)")
    p.add_argument("--seed", type=int, default=None, help="Override the config seed")
    p.add_argument("--num-runs", type=int, default=None, help="Override config num_runs (seeds seed, seed+1, ...)")
    p.add_argument("--sample-size-per-hop", type=int, default=None, help="Override config sample_size_per_hop")
    p.add_argument("--output-root", default=None, help="Override the run output root")
    p.add_argument(
        "--retrieval-input",
        default=None,
        help="Reranked/truncated top-k JSONL: the question source AND the few-shot exemplar "
        "source for the retrieved-few-shot mode. Overrides retrieval.input / "
        "retrieval.input_key in the config.",
    )
    p.add_argument(
        "--allow-unpinned-eval-set",
        action="store_true",
        help="Permit a run whose loaded row counts/ids are not the config's pinned "
        "'eval_rows_per_hop' set (ADR 0007). For fixture and smoke runs only: the metrics "
        "then record evaluation_set.pinned=false, and such a run is not an experiment arm.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Assemble prompts and write artifacts without loading a model or generating.",
    )
    p.add_argument(
        "--dry-run-limit",
        type=int,
        default=5,
        help="Questions to assemble prompts for in --dry-run (default: 5).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    cfg = load_config(args.config)
    paths_cfg = load_paths(require(cfg, "paths_config"))
    limits = load_limits(require(cfg, "model_limits_config"))

    models_root = resolve_path(require(paths_cfg, "repo.router_models_dir"), _REPO_ROOT)
    model_dir = models_root / args.model
    if not model_dir.is_dir():
        raise SystemExit(f"model folder not found: {model_dir}")
    model_cfg = load_config(model_dir / "config.json")

    seed = args.seed if args.seed is not None else int(require(cfg, "seed"))
    num_runs = args.num_runs if args.num_runs is not None else int(require(cfg, "num_runs"))
    sample_size = (
        args.sample_size_per_hop
        if args.sample_size_per_hop is not None
        else require(cfg, "sample_size_per_hop")
    )
    hops = [int(h) for h in require(cfg, "hops")]

    # Retrieved few-shot (issue #27) and the query-id-keyed predictions file: both off in
    # configs/router.json, so the MetaQA path is byte-for-byte the run it was before.
    few_shot_enabled = bool(require(cfg, "few_shot.enabled"))
    predictions_enabled = bool(require(cfg, "predictions.enabled"))
    retrieval_input = resolve_retrieval_input(args.retrieval_input, cfg, paths_cfg)
    if retrieval_input and not few_shot_enabled:
        raise SystemExit(
            f"a retrieval input was given ({retrieval_input}) but 'few_shot.enabled' is false "
            f"in {cfg.get('_config_path', '<config>')}, so no exemplars would be assembled "
            f"from it. Use a config whose few_shot block is enabled, or drop the input."
        )
    if few_shot_enabled and not retrieval_input:
        raise SystemExit(
            f"'few_shot.enabled' is true in {cfg.get('_config_path', '<config>')} but no "
            f"retrieval input was resolved: pass --retrieval-input, or set 'retrieval.input' "
            f"/ 'retrieval.input_key'. The exemplars are an upstream artifact this runner "
            f"cannot rebuild, and a few-shot run with no exemplars is a zero-shot run under a "
            f"few-shot label."
        )

    # Prompt selection: --prompt-file, then the config's own prompt_file (a config whose
    # prompting mode needs a specific prompt names it, so the mode cannot be run with the
    # wrong prompt by forgetting a flag), then the model folder's v1 default.
    prompt_file = (
        args.prompt_file or optional(cfg, "prompt_file") or require(model_cfg, "prompt_file")
    )
    zero_shot = require(cfg, "zero_shot_prompt_marker") in prompt_file
    if args.output_root is not None:
        output_root = Path(args.output_root)
    else:
        # A config declares an output root only for the prompting modes it supports, so a
        # config with no zero-shot arm does not carry a dead zero-shot subdir (PR #43 review,
        # N2). Selecting a prompt that mode does not cover is a refusal naming the key.
        subdir_key = "output_subdir_zero_shot" if zero_shot else "output_subdir_few_shot"
        subdir = optional(cfg, subdir_key)
        if not subdir:
            raise SystemExit(
                f"{cfg.get('_config_path', '<config>')} declares no {subdir_key!r}, so it has "
                f"no output root for the prompt this run selected ({prompt_file!r}). Pass "
                f"--output-root, or use a config that supports this prompting mode."
            )
        output_root = runs_path(paths_cfg, subdir)

    device = "cpu"
    if not args.dry_run:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    generation = dict(require(model_cfg, "generation"))
    loader = dict(require(model_cfg, "loader"))
    config_src = cfg.get("_config_path", "<config>")
    # The model folders are v1 assets (prompts byte-identical, `parsing` mirroring v1's
    # per-copy code), so a config whose hop depths are not v1's 1/2/3 overrides the response
    # parsing here instead of editing the asset.
    parsing = apply_overrides(
        dict(require(model_cfg, "parsing")),
        optional(cfg, "parsing_overrides"),
        block="parsing",
        src=config_src,
    )
    parsing["hops"] = hops
    # Which tail markers end the answer in this prompt's responses. Declared per config
    # because it is a property of the prompt shape, not of the model: the v1 MetaQA prompts
    # set none (their responses are multi-line), the retrieved-few-shot prompt cuts at the
    # first line break (see clean_response).
    parsing["truncate_at"] = list(require(cfg, "response_truncate_at"))
    # Before anything is loaded, and reachable by --dry-run: the parsing rules must be able to
    # express every hop depth this config may predict, and default_hop must be one of them.
    parsing_coverage = assert_parsing_covers_hops(parsing, hops, src=config_src)

    retrieval_mode: str | None = None
    retrieval_k: int | None = None
    exemplar_hop_source: str | None = None
    if few_shot_enabled:
        retrieval_mode = require(cfg, "retrieval.mode")
        retrieval_modes = require(cfg, "retrieval.modes")
        if retrieval_mode not in retrieval_modes:
            raise SystemExit(f"retrieval mode {retrieval_mode!r} not in {retrieval_modes}")
        retrieval_k = int(require(cfg, "retrieval.k"))
        exemplar_hop_source = require(cfg, "few_shot.exemplar_hop_source")
        if exemplar_hop_source not in EXEMPLAR_HOP_SOURCES:
            raise SystemExit(
                f"few_shot.exemplar_hop_source in {config_src} is {exemplar_hop_source!r}; "
                f"expected one of {list(EXEMPLAR_HOP_SOURCES)}"
            )

    # Content-address the exemplar source: two router conditions are comparable only if they
    # read the same exemplars, and "same path" does not prove "same bytes" for a file that
    # lives outside git (the decomposer records the same pair for the same reason).
    retrieval_path: Path | None = None
    retrieval_sha256: str | None = None
    if retrieval_input:
        retrieval_path = Path(retrieval_input)
        if not retrieval_path.is_absolute():
            retrieval_path = _REPO_ROOT / retrieval_path
        if not retrieval_path.exists():
            raise SystemExit(f"retrieval input not found: {retrieval_path}")
        retrieval_sha256 = rd.sha256_file(retrieval_path)

    predictions_file = require(cfg, "predictions.filename")
    predictions_id_field = require(cfg, "predictions.id_field")
    predictions_hop_field = require(cfg, "predictions.hop_field")

    current_run_id = run_id()
    seeded = set_global_seed(seed)

    snapshot = {
        "script": Path(__file__).name,
        "created_utc": now_iso(),
        "run_id": current_run_id,
        "component": "router",
        "model": args.model,
        "model_id": require(model_cfg, "model_id"),
        "model_name": require(model_cfg, "model_name"),
        "prompt_file": prompt_file,
        "zero_shot": zero_shot,
        "seed": seed,
        "seeded": seeded,
        "num_runs": num_runs,
        "sample_size_per_hop": sample_size,
        "hops": hops,
        "questions_template_key": require(cfg, "questions_template_key"),
        "questions_format": require(cfg, "questions_format"),
        "device": device,
        "generation": generation,
        "loader": loader,
        "parsing": {k: v for k, v in parsing.items() if k not in ("hops", "truncate_at")},
        "parsing_overrides": optional(cfg, "parsing_overrides"),
        "response_truncate_at": parsing["truncate_at"],
        "parsing_coverage": parsing_coverage,
        "few_shot": {
            "enabled": few_shot_enabled,
            "k": retrieval_k,
            "exemplar_hop_source": exemplar_hop_source,
            "self_exclusion_by": (
                ["pool_id == query_id", "normalized pool_question == normalized query"]
                if few_shot_enabled
                else None
            ),
        },
        "retrieval": {
            # The config's own literal values; the path this run read is 'input_resolved'.
            "input": optional(cfg, "retrieval.input"),
            "input_key": optional(cfg, "retrieval.input_key"),
            "input_resolved": str(retrieval_path) if retrieval_path else None,
            "input_sha256": retrieval_sha256,
            "mode": retrieval_mode,
            "k": retrieval_k,
        },
        "predictions": {
            "enabled": predictions_enabled,
            "filename": predictions_file,
            "id_field": predictions_id_field,
            "hop_field": predictions_hop_field,
        },
        "shared_config": cfg.get("_config_path"),
        "model_config": model_cfg.get("_config_path"),
        "output_root": str(output_root),
        "dry_run": args.dry_run,
    }
    print(f"Starting router run {current_run_id} (num_runs={num_runs}, dry_run={args.dry_run})")
    print(json.dumps(snapshot, indent=2, default=str))

    # Prompt template: model folder first, then the shared models dir (v1 behaviour).
    prompt_path = model_dir / prompt_file
    if not prompt_path.exists():
        prompt_path = model_dir.parent / prompt_file
    if not prompt_path.exists():
        raise SystemExit(
            f"prompt file not found: {prompt_file} in {model_dir} or {model_dir.parent}"
        )
    prompt_template = prompt_path.read_text(encoding="utf-8")
    snapshot["prompt_path"] = str(prompt_path)

    # ---- queries: the retrieval artifact when there is one, else the question files ----
    data_root = Path(paths_cfg["data_root_resolved"])
    if retrieval_path is not None:
        rows = rows_from_retrieval(retrieval_path, sample_size=sample_size)
        rows_source = str(retrieval_path)
    else:
        rows = rows_from_questions(
            paths_cfg=paths_cfg,
            cfg=cfg,
            hops=hops,
            data_root=data_root,
            seed=seed,
            sample_size=sample_size,
        )
        rows_source = "questions_template_key"

    if not rows:
        raise SystemExit(
            f"no questions loaded from {rows_source} for hops {hops}; "
            f"set data_root in configs/paths.json"
        )

    # Query ids, before anything is loaded: a predictions file is a join key plus a number,
    # so a run that cannot produce one has to say so now rather than after the GPU time.
    if predictions_enabled:
        assert_query_ids(
            rows,
            source=rows_source,
            reason=(
                f"{config_src} sets 'predictions.enabled', so every row needs a unique query "
                f"id to key the predictions file on: that file is what the retrieval chain's "
                f"'predictions' hop source and the decomposer's router-guided condition join "
                f"on (ADR 0022 item 5). A question source with no ids "
                f"(questions_format 'lines') cannot produce one."
            ),
        )

    all_questions = [r["question"] for r in rows]
    all_expected = [r["expected_hop"] for r in rows]
    # Counted over the config's hops *and* whatever depths the rows actually carry: the
    # config's hops keep their key even at zero (v1's shape, and a missing question file is
    # visible as 0 rather than as an absent key), and a row at a depth the config does not
    # list is counted rather than dropped - that is what lets the pinned-set assertion below
    # report it instead of silently ignoring it.
    counted_hops = sorted({r["expected_hop"] for r in rows} | set(hops))
    counts = {
        f"{hop}hop": sum(1 for r in rows if r["expected_hop"] == hop) for hop in counted_hops
    }
    rows_per_hop = {str(hop): counts[f"{hop}hop"] for hop in counted_hops}
    print(f"Loaded {len(rows)} questions {counts}")

    # The pinned evaluation set, by id and not only by count (the decomposer's shared
    # assertion): a router prediction file over a different 600 questions cannot be joined
    # against the arms it is supposed to route (ADR 0007, ADR 0011).
    pinned_ids: set[str] = set()
    pinned_files: list[str] = []
    pinned_id_problems: list[str] = []
    if optional(cfg, "eval_rows_per_hop") is not None:
        pinned_ids, pinned_files, pinned_id_problems = rd.load_pinned_eval_ids(
            paths_cfg, cfg, hops, data_root
        )
    loaded_ids = {str(r["query_id"]) for r in rows if r["query_id"] is not None}
    eval_set_record = rd.assert_pinned_eval_set(
        rows_per_hop,
        len(rows),
        cfg=cfg,
        hops=hops,
        allow_unpinned=args.allow_unpinned_eval_set,
        source=rows_source,
        loaded_ids=loaded_ids,
        pinned_ids=pinned_ids,
        pinned_files=pinned_files,
        pinned_id_problems=pinned_id_problems,
        remedy=(
            "Point --retrieval-input at a file built over exactly those questions (or use the "
            "config's questions_template_key), or pass --allow-unpinned-eval-set for a "
            "fixture run that is not an experiment arm."
        ),
        component="router",
    )
    eval_set_record["rows_loaded_total"] = len(rows)
    eval_set_record["rows_loaded_per_hop"] = rows_per_hop
    eval_set_record["distinct_query_ids"] = len(loaded_ids)
    snapshot["evaluation_set"] = eval_set_record
    print(f"Evaluation set: {json.dumps(eval_set_record, default=str)}")

    # ---- prompts, one per query, before any model is loaded ----
    if few_shot_enabled and "{few_shot_examples}" not in prompt_template:
        raise SystemExit(
            f"'few_shot.enabled' is true in {config_src} but the prompt {prompt_path} has no "
            f"'{{few_shot_examples}}' placeholder, so the retrieved exemplars would never "
            f"reach the model and the run would be zero-shot under a few-shot label. Select a "
            f"prompt that takes exemplars (--prompt-file prompt_few_shot_musique.md)."
        )
    if not few_shot_enabled and "{few_shot_examples}" in prompt_template:
        raise SystemExit(
            f"the prompt {prompt_path} takes '{{few_shot_examples}}' but 'few_shot.enabled' is "
            f"false in {config_src}, so the prompt would be rendered with an empty exemplar "
            f"block. Enable the few-shot block, or select a prompt that carries its own "
            f"examples."
        )

    self_excluded_rows = 0
    self_excluded_examples = 0
    for row in rows:
        examples: list[dict] = []
        if few_shot_enabled:
            # Both resolved above, next to few_shot_enabled; asserted for the type checker.
            assert retrieval_mode is not None and retrieval_k is not None
            examples, dropped = examples_from_retrieval_row(
                row["retrieval_row"],
                mode=retrieval_mode,
                k=retrieval_k,
                exemplar_hop_source=exemplar_hop_source,
            )
            if dropped:
                self_excluded_rows += 1
                self_excluded_examples += dropped
        row["few_shot_examples"] = examples
        row["prompt"] = build_prompt(
            prompt_template, row["question"], format_few_shot_examples(examples)
        )

    few_shot_record = {
        "enabled": few_shot_enabled,
        "k": retrieval_k,
        "exemplar_hop_source": exemplar_hop_source,
        "retrieval_input": str(retrieval_path) if retrieval_path else None,
        "retrieval_input_sha256": retrieval_sha256,
        "retrieval_mode": retrieval_mode,
        "rows_with_a_self_example_dropped": self_excluded_rows,
        "self_examples_dropped": self_excluded_examples,
        "note": (
            "an exemplar that is the query itself is dropped from the ranked list before the "
            "top-k is taken (by pool id and by normalized question text)"
            if few_shot_enabled
            else "no exemplars were retrieved in this run, so there was nothing to exclude"
        ),
    }
    snapshot["few_shot"] = {**snapshot["few_shot"], **few_shot_record}

    output_dir = output_root / current_run_id

    def write_prompt_logs(logged: list[dict]) -> Path:
        prompts_dir = output_dir / "prompts_log"
        prompts_dir.mkdir(parents=True, exist_ok=True)
        for i, row in enumerate(logged, start=1):
            exemplars = "\n".join(
                f"  {j}. hop={ex['hop_count']} pool_id={ex['pool_id']} | {ex['question']}"
                for j, ex in enumerate(row["few_shot_examples"], start=1)
            ) or "  (none)"
            (prompts_dir / f"prompt_idx{i:04d}.txt").write_text(
                f"--- Question ---\n{row['question']}\n"
                f"--- Query id ---\n{row['query_id']}\n"
                f"--- Gold hop ---\n{row['expected_hop']}\n"
                f"--- Few-shot exemplars ({len(row['few_shot_examples'])}) ---\n{exemplars}\n"
                f"\n--- Prompt ---\n{row['prompt']}\n",
                encoding="utf-8",
            )
        return prompts_dir

    if args.dry_run:
        logged = rows[: max(0, args.dry_run_limit)]
        prompts_dir = write_prompt_logs(logged)
        metrics = {
            "dry_run": True,
            "total_questions_loaded": len(rows),
            "questions_per_hop": counts,
            "evaluation_set": eval_set_record,
            "prompts_assembled": len(logged),
            "prompt_chars_mean": (
                sum(len(r["prompt"]) for r in logged) / len(logged) if logged else 0
            ),
            "few_shot": few_shot_record,
            "parsing_coverage": parsing_coverage,
            "model_size": unasserted_note("router", require(model_cfg, "model_id")),
            "accuracy_metrics": None,
            "accuracy_metrics_note": "unmeasured: --dry-run does not load a model or generate",
            # Stated rather than left absent: a dry run predicts nothing, so writing a
            # predictions file would mean fabricating routing decisions.
            "predictions_file": None,
            "predictions_note": (
                "not written: --dry-run generates nothing, so there is no prediction to key "
                "by query id"
            ),
        }
        write_run_artifacts(
            output_dir,
            config_snapshot=snapshot,
            metrics=metrics,
            note_title=f"Router dry run - {current_run_id}",
            note_lines=[
                f"- Model folder: `{args.model}` (model not loaded)",
                f"- Prompt: `{prompt_path}`",
                f"- Questions loaded: {len(rows)} {counts} (source: {rows_source})",
                f"- Evaluation set pinned: {eval_set_record['pinned']} "
                f"(ids checked: {eval_set_record['id_identity_checked']})",
                f"- Few-shot: enabled={few_shot_enabled} k={retrieval_k}; "
                f"self-examples dropped: {self_excluded_examples}",
                f"- Prompts assembled: {len(logged)} (logged under `{prompts_dir}`)",
                "- Accuracy: unmeasured (dry run).",
                "- Predictions file: not written (dry run predicts nothing).",
                "- Parameter ceiling: not asserted (no model was loaded).",
            ],
        )
        print(f"\nDry-run artifacts under: {output_dir}")
        return

    model_id = require(model_cfg, "model_id")
    print(f"Loading model: {model_id} on {device} ...")
    tokenizer, model = load_model(model_id, loader, device)
    size_record = assert_within_ceiling(
        model, component="router", model_id=model_id, limits=limits
    )

    all_runs_metrics: list[dict] = []
    all_runs_predictions: list[list[int]] = []
    all_runs_parsed: list[list[bool]] = []
    progress_every = int(require(cfg, "progress_every"))

    for run_idx in range(num_runs):
        run_seed = seed + run_idx
        set_global_seed(run_seed)
        print(f"\n--- Run {run_idx + 1}/{num_runs} (seed={run_seed}) ---")
        predictions: list[int] = []
        parsed_flags: list[bool] = []
        for i, row in enumerate(rows):
            if (i + 1) % progress_every == 0:
                print(f"Processed {i + 1}/{len(rows)}...")
            hop, parsed = classify_hop_count(
                row["prompt"], row["question"], model, tokenizer, device, generation, parsing
            )
            predictions.append(hop)
            parsed_flags.append(parsed)
        all_runs_predictions.append(predictions)
        all_runs_parsed.append(parsed_flags)
        run_metrics = compute_metrics(
            predictions, all_expected, run_seed, model_id, current_run_id, hops, run_idx=run_idx
        )
        all_runs_metrics.append(run_metrics)
        print(f"Run {run_idx + 1} accuracy: {run_metrics['overall_accuracy']:.4f}")

    if num_runs == 1:
        metrics = dict(all_runs_metrics[0])
    else:
        metrics = {
            "num_runs": num_runs,
            "overall_accuracy_mean": statistics.mean(m["overall_accuracy"] for m in all_runs_metrics),
            "overall_accuracy_std": statistics.stdev(m["overall_accuracy"] for m in all_runs_metrics),
            "model": model_id,
            "run_id": current_run_id,
        }
        for hop in hops:
            accs = [m[f"hop_{hop}_accuracy"] for m in all_runs_metrics]
            metrics[f"hop_{hop}_accuracy_mean"] = statistics.mean(accs)
            metrics[f"hop_{hop}_accuracy_std"] = statistics.stdev(accs)
        metrics["per_run"] = all_runs_metrics

    metrics["model_size"] = size_record
    metrics["questions_per_hop"] = counts
    metrics["evaluation_set"] = eval_set_record
    metrics["few_shot"] = few_shot_record
    metrics["parsing_coverage"] = parsing_coverage
    metrics["accuracy_definitions"] = ACCURACY_DEFINITIONS
    metrics["dry_run"] = False
    # A prediction that fell back to parsing.default_hop is a default, not a routing
    # decision; counted so a reader of the metrics can see how much of the accuracy is
    # actually predicted, and broken out per gold hop because a default is scored as a
    # prediction (default_hop is correct for its own class and wrong for every other).
    # Reported for run 0, which is the run the outputs come from.
    metrics["unparsed_response_rows"] = sum(1 for ok in all_runs_parsed[0] if not ok)
    metrics["unparsed_response_rows_per_gold_hop"] = unparsed_rows_per_hop(
        all_expected, all_runs_parsed[0], hops
    )

    print("\n" + "=" * 30)
    if num_runs == 1:
        print(f"Accuracy: {metrics['overall_accuracy']:.4f}")
        for hop in hops:
            print(f"{hop}-hop: {metrics[f'hop_{hop}_accuracy']:.4f}")
    else:
        print(
            f"Accuracy (mean +/- std): {metrics['overall_accuracy_mean']:.4f} "
            f"+/- {metrics['overall_accuracy_std']:.4f}"
        )
        for hop in hops:
            print(
                f"{hop}-hop: {metrics[f'hop_{hop}_accuracy_mean']:.4f} "
                f"+/- {metrics[f'hop_{hop}_accuracy_std']:.4f}"
            )
    print("=" * 30)

    detailed = [
        {
            # query_id first: v1's detailed_results.json was keyed by question text only,
            # which is the field the masking modes rewrite (ADR 0022 item 5).
            "query_id": row["query_id"],
            "question": row["question"],
            "expected": e,
            "predicted": p,
            "correct": p == e,
            "parse_fallback": not parsed,
        }
        for row, e, p, parsed in zip(
            rows, all_expected, all_runs_predictions[0], all_runs_parsed[0]
        )
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    detailed_name = "detailed_results.json" if num_runs == 1 else "detailed_results_run_0.json"
    (output_dir / detailed_name).write_text(
        json.dumps(detailed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # The query-id-keyed predictions file: run 0's predictions, one object per query. This is
    # the artifact issue #15's router-hop-matched condition and the decomposer's
    # router_guided condition consume; both join on the id.
    predictions_path: Path | None = None
    if predictions_enabled:
        predictions_path = output_dir / predictions_file
        write_predictions_jsonl(
            predictions_path,
            build_prediction_rows(
                rows,
                all_runs_predictions[0],
                all_runs_parsed[0],
                id_field=predictions_id_field,
                hop_field=predictions_hop_field,
            ),
        )
        print(f"Predictions ({len(rows)} rows, keyed by {predictions_id_field}): {predictions_path}")
    metrics["predictions_file"] = str(predictions_path) if predictions_path else None
    metrics["predictions_rows"] = len(rows) if predictions_path else 0
    metrics["predictions_id_field"] = predictions_id_field if predictions_path else None
    metrics["predictions_hop_field"] = predictions_hop_field if predictions_path else None
    if predictions_path is None:
        metrics["predictions_note"] = (
            f"not written: {config_src} sets predictions.enabled false (a question source "
            f"with no ids cannot be keyed by query id)"
        )
    elif num_runs > 1:
        metrics["predictions_note"] = (
            f"predictions are run 0 of {num_runs}, the same run detailed_results_run_0.json "
            f"records; the other runs' predictions are not written"
        )

    headline = (
        f"- Overall accuracy: {metrics['overall_accuracy']:.4f}"
        if num_runs == 1
        else (
            f"- Overall accuracy (mean +/- std over {num_runs} runs): "
            f"{metrics['overall_accuracy_mean']:.4f} +/- {metrics['overall_accuracy_std']:.4f}"
        )
    )
    write_run_artifacts(
        output_dir,
        config_snapshot=snapshot,
        metrics=metrics,
        note_title=f"Router run - {current_run_id}",
        note_lines=[
            f"- Model: `{model_id}` ({size_record['parameter_count']:,} parameters, "
            f"ceiling {size_record['parameter_ceiling']:,})",
            f"- Prompt: `{prompt_path}`",
            f"- Seed: {seed} (runs use seed, seed+1, ...)",
            f"- Questions: {len(all_questions)} {counts} (source: {rows_source})",
            f"- Evaluation set pinned: {eval_set_record['pinned']} "
            f"(ids checked against the pinned files: "
            f"{eval_set_record['id_identity_checked']})",
            f"- Few-shot: enabled={few_shot_enabled} k={retrieval_k}"
            + (
                f" from `{retrieval_path}` (sha256 `{retrieval_sha256}`), mode "
                f"{retrieval_mode}; self-examples dropped: {self_excluded_examples}"
                if few_shot_enabled
                else " (prompt carries its own examples, or none)"
            ),
            headline,
            f"- Responses with no readable hop count (recorded as "
            f"parsing.default_hop={parsing_coverage['default_hop']}, and counted in the "
            f"accuracies above): {metrics['unparsed_response_rows']} of {len(rows)}, per gold "
            f"hop {metrics['unparsed_response_rows_per_gold_hop']}",
            f"- Accuracy is exact-hop against the GOLD depth; the per-hop rows are "
            f"within-gold-class recall (see accuracy_definitions in metrics.json)",
            f"- Detailed predictions: `{output_dir / detailed_name}`",
            (
                f"- Predictions keyed by `{predictions_id_field}`: `{predictions_path}` "
                f"({len(rows)} rows)"
                if predictions_path
                else "- Predictions file: not written (this question source carries no ids)."
            ),
        ],
    )
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
