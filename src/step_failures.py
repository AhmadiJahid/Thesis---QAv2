"""Step-level failure taxonomy for an executed decomposition (issue #16).

The MuSiQue answering backend (``components/answerer/run_answerer.py``) reports answer EM
and F1 for an item, which says *whether* a decomposition led to the right answer but not
*where* it went wrong. This module is the "where": one label set per sub-question, derived
from what the backend already records (the substitution result, the generation error, the
cleaned answer), aggregated into counts a run note can carry.

Nothing here loads a model, reads config or touches the filesystem; it is pure
classification, so the arithmetic is unit-testable without a run.

**Two of the categories issue #16 named cannot be produced by this backend, and are
declared unavailable rather than approximated** — see
:data:`UNAVAILABLE_STEP_FAILURE_CATEGORIES`. The rule from CLAUDE.md applies: what is not
measured is reported as unmeasured, never as zero.

The categories are **not mutually exclusive**: a step can be asked with a broken reference
*and* come back empty, and both facts matter. So the summary is a set of counters plus
``steps_clean`` (steps that carry no flag at all), never a partition that would have to
pick a single "cause" by a precedence rule this repo has not agreed.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

#: What a sub-question can be flagged with, in the order the metrics JSON lists them.
#: Each one is decided by a field the backend already writes to its per-item file, so a
#: reader can re-derive every count from ``answer_per_item.json`` by hand.
STEP_FAILURE_CATEGORIES: tuple[str, ...] = (
    "broken_reference",
    "generation_error",
    "empty_answer",
    "not_executed",
)

#: What each flag means, copied into the metrics JSON so a run note is self-describing.
STEP_FAILURE_DEFINITIONS: dict[str, str] = {
    "broken_reference": (
        "the sub-question still carried at least one unresolved step reference when it was "
        "asked: a forward reference, an out-of-range k, a step whose answer was empty or "
        "failed, or a malformed '[#k' (src/step_lines.py::substitute_step_references leaves "
        "these verbatim rather than guessing them, so the reader saw the placeholder)"
    ),
    "generation_error": "the generation call for this sub-question raised; the answer is empty",
    "empty_answer": (
        "the sub-question was executed without error but the answer was empty after the "
        "model folder's post-processing and the configured answer cleanup"
    ),
    "not_executed": (
        "no generation was attempted for this sub-question (--dry-run), so its answer is the "
        "configured dry-run stub and nothing about the reader is being reported"
    ),
}

#: The categories issue #16 asked for that this backend **cannot** produce, each with the
#: reason. They are emitted in the metrics JSON so the gap is on the record instead of being
#: silently absent — and so nobody later reads a missing key as a zero count.
#:
#: Adding any of them is a methodology change, not an implementation detail: the first two
#: would need a context regime ADR 0019 refuses, the third would need the gold sub-answers
#: that ADR 0019 keeps out of this path. Jahid's call with his supervisor.
UNAVAILABLE_STEP_FAILURE_CATEGORIES: dict[str, str] = {
    "empty_retrieval": (
        "not applicable: this backend runs no retrieval. ADR 0019 decision 2 fixes the "
        "context to the MuSiQue item's full paragraph list and the code refuses any other "
        "policy, so there is no retrieval step that could come back empty. The MetaQA "
        "compile-execute path has the analogous category (an op returning an empty id set)."
    ),
    "unresolvable_entity": (
        "not applicable: this backend resolves no entities. There is no entity linker and no "
        "knowledge base in the MuSiQue path - a sub-question is asked over paragraphs as "
        "text. The MetaQA compile-execute path reports the analogous failure as its "
        "'entity_not_in_kb' execution-error category."
    ),
    "wrong_intermediate_answer": (
        "unmeasured, not zero: judging an intermediate answer needs a gold answer for that "
        "sub-question. MuSiQue ships them (question_decomposition[*].answer), but ADR 0019 "
        "deliberately keeps the gold sub-answers out of this path, and reading them as a "
        "diagnostic would be a new methodology choice rather than an implementation detail. "
        "The final step is the exception and is already scored: an item's answer_em / "
        "answer_f1 IS the correctness of its last step."
    ),
}

STEP_FAILURE_TAXONOMY_NOTE = (
    "Step-level failure flags for the executed sub-questions of this run. The categories are "
    "NOT mutually exclusive, so the counts in by_category can sum to more than "
    "steps_with_any_flag; steps_clean counts sub-questions with no flag at all. "
    "not_available lists the categories issue #16 named that this backend cannot produce, "
    "with the reason for each - they are absent, not zero."
)


def classify_step(
    *,
    answer: str | None,
    unresolved_references: Sequence[int],
    error: str | None,
    executed: bool,
) -> list[str]:
    """The failure flags of one sub-question, in :data:`STEP_FAILURE_CATEGORIES` order.

    ``executed`` is False on a dry run, where no generation was attempted: the answer is a
    stub, so the answer is not judged (no ``empty_answer``) and the step is flagged
    ``not_executed`` instead. A broken reference is still real on a dry run — the
    substitution chain runs either way — and is still flagged.

    An empty list means the step carries no observed failure. That is not a claim that its
    answer is *correct*: see ``wrong_intermediate_answer`` in
    :data:`UNAVAILABLE_STEP_FAILURE_CATEGORIES`.
    """
    flags: set[str] = set()
    if unresolved_references:
        flags.add("broken_reference")
    if error:
        flags.add("generation_error")
    elif executed and not str(answer or "").strip():
        flags.add("empty_answer")
    if not executed:
        flags.add("not_executed")
    return [category for category in STEP_FAILURE_CATEGORIES if category in flags]


def summarize_step_failures(flag_lists: Iterable[Sequence[str]]) -> dict[str, Any]:
    """Aggregate per-step flag lists into the metrics block.

    ``steps`` counts the sub-questions seen, ``steps_clean`` those with no flag,
    ``steps_with_any_flag`` the rest, and ``by_category`` how often each flag fired. Every
    declared category is present with an explicit 0 so a category that never fired is
    distinguishable from one this code does not know about.
    """
    by_category = {category: 0 for category in STEP_FAILURE_CATEGORIES}
    steps = 0
    with_any = 0
    for flags in flag_lists:
        steps += 1
        if flags:
            with_any += 1
        for flag in flags:
            if flag not in by_category:
                raise ValueError(f"unknown step failure category: {flag!r}")
            by_category[flag] += 1
    return {
        "steps": steps,
        "steps_clean": steps - with_any,
        "steps_with_any_flag": with_any,
        "by_category": by_category,
    }
