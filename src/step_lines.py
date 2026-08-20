"""One definition of "a step line", shared by the decomposer and the evaluator.

Why this module exists. A decomposition is a newline-delimited list of sub-questions, and
three places in the pipeline have to agree on how that text is cut into steps:

1. the step-line budget of the ``unguided_capped`` condition (a ``StoppingCriteria`` that
   fires once N step lines exist),
2. the counter that reports how many rows came out at that budget, and
3. ``scripts/musique_decompositions_evaluator.py``, which scores step counts and so
   decides what "8 steps" means in a metric.

They used to disagree: the budget was counted on the *raw* generation (so a ``<think>``
preamble or a trailing ``Question:`` echo consumed part of the budget), the counter on the
post-processed text, and the evaluator on its own private splitter. Three numbers named
"steps" that were not the same number. The splitter below is the evaluator's original one,
moved here unchanged, and the decomposer now derives its budget from it.

Nothing here loads a model or touches config; it is pure text handling.
"""
from __future__ import annotations

import re

#: A ``<think> ... </think>`` block, as emitted by reasoning models, with trailing space.
THINK_BLOCK_RX = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

#: A leading enumerator (``1.``, ``2.`` ...) on a step line. Removed before comparison so
#: an enumerated and a bare decomposition of the same steps score the same.
_ENUMERATOR_RX = re.compile(r"^\s*\d+\.\s*")

#: A reference to the result of step k, in either of the two conventions this repo has:
#: ``[#k]`` — what the decomposer prompt files instruct ("Use [#1], [#2], ... ONLY to refer
#: to previous step results"), the same grammar as
#: ``scripts/musique_decompositions_evaluator.py::_REF_RX``; and bare ``#k`` — MuSiQue's own
#: gold convention and, per ADR 0012 (``target_reference_style: "as_is"``), the style the
#: fine-tuned arm is trained to emit, the same grammar as
#: ``src/finetune_data.py::_BARE_REF_RX``. One regex with two alternatives so substitution is
#: a single pass: a resolved answer that itself contains "#3" must not be rescanned.
_REF_RX = re.compile(r"\[#(?P<bracketed>\d+)\]|(?<!\[)#(?P<bare>\d+)")
_BRACKETED_REF_RX = re.compile(r"\[#(?P<bracketed>\d+)\]")


def split_step_lines(text: str) -> list[str]:
    """Split a decomposition string into steps.

    The single normalization of record: non-empty lines, stripped, with a leading
    ``"<n>. "`` enumerator removed. Moved verbatim from
    ``scripts/musique_decompositions_evaluator.py::_split_decomposition_text`` — the
    metric semantics pinned by the golden values in ``scripts/smoke_test.py`` and
    ``tests/test_decomposition_metrics.py`` depend on it being byte-for-byte this rule.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    cleaned: list[str] = []
    for ln in lines:
        ln = _ENUMERATOR_RX.sub("", ln)
        if ln:
            cleaned.append(ln)
    return cleaned


def has_open_think_block(text: str) -> bool:
    """True while a ``<think>`` block has been opened and not yet closed.

    Mid-generation the closing tag has not arrived, so ``THINK_BLOCK_RX`` cannot strip the
    preamble yet and its newlines would be counted as step lines. The step-line budget
    treats such a prefix as "no steps yet".
    """
    return text.count("<think>") > text.count("</think>")


def strip_generation_artifacts(
    text: str, *, strip_think: bool, truncate_at: list[str] | tuple[str, ...] | None
) -> str:
    """Remove the model's non-answer text: ``<think>`` blocks, then any tail marker.

    ``strip_think`` and ``truncate_at`` come from a model folder's ``post_process`` block.
    Trailing whitespace is *kept*: the step-line budget needs the final newline, because a
    line only counts once it is terminated. :func:`post_process_generation` is the
    whitespace-stripping variant, and it is what gets recorded as the model's output.
    """
    out = text
    if strip_think:
        out = THINK_BLOCK_RX.sub("", out)
    for marker in truncate_at or []:
        out = out.split(marker)[0]
    return out


def post_process_generation(
    text: str, *, strip_think: bool, truncate_at: list[str] | tuple[str, ...] | None
) -> str:
    """What gets recorded as the decomposition: artifacts removed, outer whitespace gone."""
    return strip_generation_artifacts(text, strip_think=strip_think, truncate_at=truncate_at).strip()


def completed_step_line_count(
    text: str, *, strip_think: bool, truncate_at: list[str] | tuple[str, ...] | None
) -> int:
    """How many step lines of ``text`` are *finished*, under the shared normalization.

    Used by the step-line budget, so it is evaluated on a partial generation:

    - a ``<think>`` block that is still open means 0 (the model has not started answering),
    - the text after the last newline is still being written and does not count yet,
    - what remains is counted with :func:`split_step_lines`, i.e. the same rule the
      evaluator uses, so a "cap of 8" is 8 of the steps the evaluator will score.

    Hand-checked: ``"a\\nb"`` -> 1, ``"a\\nb\\n"`` -> 2, ``"\\n\\n"`` -> 0,
    ``"<think>x\\ny\\n"`` -> 0.
    """
    if has_open_think_block(text):
        return 0
    processed = strip_generation_artifacts(text, strip_think=strip_think, truncate_at=truncate_at)
    if "\n" not in processed:
        return 0
    completed, _ = processed.rsplit("\n", 1)
    return len(split_step_lines(completed))


def trim_to_step_lines(text: str, max_step_lines: int) -> str:
    """Keep the first ``max_step_lines`` step lines of ``text``, dropping the rest.

    Companion to the stopping rule, which fires between tokens and so can leave a partial
    line after the cap. Line text is preserved as written (an enumerator is not removed —
    that is a comparison-time normalization, not an edit to the model's output); whether a
    line *is* a step is decided by :func:`split_step_lines`, so the trim and the budget
    agree on the count.
    """
    if max_step_lines <= 0:
        raise ValueError(f"max_step_lines must be positive, got {max_step_lines}")
    kept: list[str] = []
    for line in text.splitlines():
        if not split_step_lines(line):
            continue
        kept.append(line.strip())
        if len(kept) >= max_step_lines:
            break
    return "\n".join(kept)


def step_line_count(text: str) -> int:
    """Number of step lines in a finished decomposition (the evaluator's count)."""
    return len(split_step_lines(text))


def substitute_step_references(
    text: str, answers: dict[int, str], *, accept_bare: bool = True
) -> tuple[str, list[int], list[int]]:
    """Replace each ``[#k]`` (and optionally bare ``#k``) with the answer of step k.

    This is what makes a decomposition *executable*: step 2 of "Which union organised the
    strike? / Who leads [#1]?" can only be asked once step 1 has an answer. Used by the
    MuSiQue answering backend (``components/answerer/run_answerer.py``).

    ``answers`` maps a **1-based** step index to the answer produced for that step.
    Returns ``(substituted_text, resolved_ks, unresolved_ks)``:

    - a reference whose k has a non-empty answer in ``answers`` is replaced by it and k is
      reported in ``resolved_ks`` (once per occurrence);
    - a reference with no answer — a forward reference, an out-of-range k, or a step whose
      generation failed or came back empty — is **left in the text verbatim** and reported
      in ``unresolved_ks``. Leaving it is deliberate: dropping it would silently turn
      "Who leads [#1]?" into a different, answerable-looking question, and the caller
      counts these so a run says how often it happened.

    Substitution is a single left-to-right pass, so an inserted answer that itself contains
    ``#3`` is not rescanned. Which grammars are recognised is the caller's (config's) choice:
    ``accept_bare`` false recognises only the bracketed form the prompts instruct.
    """
    resolved: list[int] = []
    unresolved: list[int] = []

    def _replace(match: re.Match[str]) -> str:
        # groupdict, not group(): the bracketed-only regex has no "bare" group at all.
        groups = match.groupdict()
        raw = groups.get("bracketed") or groups.get("bare")
        k = int(raw)
        answer = answers.get(k)
        if isinstance(answer, str) and answer.strip():
            resolved.append(k)
            return answer.strip()
        unresolved.append(k)
        return match.group(0)

    rx = _REF_RX if accept_bare else _BRACKETED_REF_RX
    return rx.sub(_replace, text), resolved, unresolved
