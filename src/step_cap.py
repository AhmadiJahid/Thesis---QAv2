"""Step-line accounting for the decomposer's length-capped condition.

Issue #12 compares three decomposition conditions on the MuSiQue evaluation set.
The third one, ``unguided_capped``, gets no hop count in the prompt but is stopped
after N step lines (plus the generation's own token budget), because a runaway
14-step decomposition may be the failure mode rather than hop ignorance.

The counting rule here is deliberately the same one the evaluator scores with
(``scripts/musique_decompositions_evaluator.py::_split_decomposition_text``): a
step is a non-blank line, with a leading ``N.`` enumerator stripped. If the two
rules drifted apart, a decomposition could be capped at 8 steps and then scored
as having 9, so ``tests/test_step_cap.py`` asserts the two implementations agree
line for line.

Nothing in this module imports torch or transformers at module level, so the
counting logic is testable without a GPU (or a model download); the stopping
criteria wrapper imports them lazily.
"""
from __future__ import annotations

import re
from typing import Any

#: A leading "1." / " 2. " enumerator, stripped before a line counts as a step.
_ENUMERATOR_RX = re.compile(r"^\s*\d+\.\s*")


def _is_step_line(raw_line: str) -> bool:
    """True when a raw line carries step content (not blank, not a bare "3.")."""
    return bool(_ENUMERATOR_RX.sub("", raw_line.strip()))


def split_step_lines(text: str) -> list[str]:
    """Split generated text into step lines, enumerators stripped."""
    out: list[str] = []
    for raw in (text or "").splitlines():
        line = _ENUMERATOR_RX.sub("", raw.strip())
        if line:
            out.append(line)
    return out


def count_step_lines(text: str) -> int:
    """How many step lines ``text`` contains."""
    return len(split_step_lines(text))


def count_completed_step_lines(text: str) -> int:
    """Step lines that are certainly finished, i.e. everything before the last newline.

    A trailing line with no newline after it is still being generated, so it does not
    count. This is what makes the stopping criteria fire exactly on the newline that
    ends step N rather than after the model has written part of step N+1.
    """
    if "\n" not in (text or ""):
        return 0
    head = text.rsplit("\n", 1)[0]
    return count_step_lines(head)


def truncate_to_step_lines(text: str, max_step_lines: int) -> tuple[str, bool]:
    """Keep at most ``max_step_lines`` step lines. Returns (text, was_truncated).

    Generation can overshoot the cap: the stopping criteria is only consulted at token
    boundaries, and a run may also end on EOS or on the token budget mid-overshoot. So
    the cap is enforced a second time on the decoded text, which is what gets recorded
    and scored. Kept lines are the original lines verbatim (enumerators included).
    """
    if max_step_lines <= 0:
        raise ValueError(f"max_step_lines must be positive, got {max_step_lines}")
    kept: list[str] = []
    total = 0
    for raw in (text or "").splitlines():
        if not _is_step_line(raw):
            continue
        total += 1
        if total <= max_step_lines:
            kept.append(raw.strip())
    if total <= max_step_lines:
        return (text or "").strip(), False
    return "\n".join(kept), True


class StepLineBudget:
    """Decides when a generation has produced enough step lines to stop.

    Pure logic, no torch: ``reached()`` takes the decoded text generated so far, which
    is also how the test drives it over a fake generation stream.
    """

    def __init__(self, max_step_lines: int) -> None:
        if int(max_step_lines) <= 0:
            raise ValueError(f"max_step_lines must be positive, got {max_step_lines}")
        self.max_step_lines = int(max_step_lines)

    def reached(self, generated_text: str) -> bool:
        return count_completed_step_lines(generated_text) >= self.max_step_lines

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"StepLineBudget(max_step_lines={self.max_step_lines})"


def build_stopping_criteria(budget: StepLineBudget, tokenizer: Any, prompt_length: int) -> Any:
    """Wrap ``budget`` as a transformers ``StoppingCriteriaList``.

    ``prompt_length`` is the prompt's token count, so only the newly generated tail is
    decoded and counted. Imports are local: this module must stay importable without
    torch installed.
    """
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    class _StepLineStoppingCriteria(StoppingCriteria):
        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> Any:
            done = [
                budget.reached(
                    tokenizer.decode(row[prompt_length:], skip_special_tokens=True)
                )
                for row in input_ids
            ]
            return torch.tensor(done, dtype=torch.bool, device=input_ids.device)

    return StoppingCriteriaList([_StepLineStoppingCriteria()])
