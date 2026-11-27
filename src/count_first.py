"""Count-first decomposition: the model states its hop count, then writes the steps.

exp-037 (the memo at the repo root, section C1.3 "Hop-count-first structured output"): each
few-shot exemplar's decomposition is shown with a first line ``[N hops]``, N being the step
count of that exemplar's OWN gold decomposition (the same splitter the evaluator scores
with), and the model is asked to write that line before its steps. The count the model writes
is its own prediction - never the gold hop count of the query (that is the oracle arm,
exp-004/005, a different question).

This module is pure text handling (no model, no config IO), so it is unit-testable on CPU:

- :func:`format_count_line` - the line an exemplar carries.
- :func:`insert_instruction_line` - the one rule line the count-first prompt adds to the
  unguided prompt, inserted after an anchor line that must occur exactly once.
- :func:`parse_count_first` - split a post-processed generation into (count line, steps).
  The count line is removed before scoring, so the report card (ADR 0029) scores only the
  steps. A generation whose count line is missing or malformed is NOT dropped: its steps are
  scored like any other row, and its status is counted (:func:`summarize_count_first`).
- :func:`summarize_count_first` - the count's accuracy against the gold hop count and its
  agreement with the number of steps actually emitted.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable

from step_lines import split_step_lines

#: The exemplar / expected count line. Plural for every N, so one exact form is demonstrated.
COUNT_LINE_FORMAT = "[{n} hops]"

#: The exact form, optionally followed on the same line by step text (``[3 hops] 1. Who ...``,
#: the inline shape the memo's illustration uses). Case- and inner-whitespace-tolerant.
_STRICT_RX = re.compile(r"^\[\s*(?P<n>\d+)\s*hops?\s*\]\s*(?P<rest>.*)$", re.IGNORECASE)

#: A line that is ONLY a hop-count statement, in some other shape than the exact one:
#: ``Hops: 3``, ``Hop count: 3``, ``3 hops``, ``(3 hops)``, ``[three hops]``, ``[3]``-free
#: variants. Steps are questions, so a line consisting of nothing but a count statement is
#: not a step; it is removed and the row is labelled ``malformed``.
_NUMBER = r"(?P<n>\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
_LOOSE_RXS = (
    re.compile(rf"^\W*hops?(?:\s*count)?\s*[:=\-]?\s*{_NUMBER}\W*$", re.IGNORECASE),
    re.compile(rf"^\W*{_NUMBER}\s*-?\s*hops?\W*$", re.IGNORECASE),
    # "Hops: <one short word>" - a labelled count whose value is not a number ("Hops: many").
    # One word of at most 10 letters, so a step like "Hop: Who founded X?" stays a step.
    re.compile(r"^\W*hops?(?:\s*count)?\s*[:=]\s*(?P<n>[A-Za-z]{1,10})\W*$", re.IGNORECASE),
    # "[<short word> hops]" - the bracket shape with no number, e.g. the instruction's own
    # placeholder copied verbatim ("[N hops]").
    re.compile(r"^\s*\[\s*(?P<n>[A-Za-z]{0,10})\s*hops?\s*\]\s*$", re.IGNORECASE),
)
_WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

#: Row statuses. ``ok``/``inline`` carry a usable count; ``malformed`` may (``Hops: 3``) or
#: may not; ``absent`` and ``empty`` never do.
STATUSES = ("ok", "inline", "malformed", "absent", "empty")


def format_count_line(n: int) -> str:
    if not isinstance(n, int) or n < 1:
        raise ValueError(f"a hop count must be a positive int, got {n!r}")
    return COUNT_LINE_FORMAT.format(n=n)


def insert_instruction_line(template: str, *, instruction: str, insert_after: str) -> str:
    """``template`` with ``instruction`` as a new line right after the line ``insert_after``.

    The anchor must match one whole line (stripped) exactly once; anything else is a
    ``ValueError`` rather than a guess about where the rule belongs. The line ending of the
    anchor line is reused, so a CRLF prompt file stays CRLF.
    """
    if not instruction.strip() or "\n" in instruction or "\r" in instruction:
        raise ValueError(f"instruction must be one non-empty line, got {instruction!r}")
    lines = template.splitlines(keepends=True)
    hits = [i for i, ln in enumerate(lines) if ln.strip() == insert_after.strip()]
    if len(hits) != 1:
        raise ValueError(
            f"anchor line {insert_after!r} must occur exactly once in the prompt template; "
            f"found {len(hits)}"
        )
    i = hits[0]
    anchor = lines[i]
    ending = anchor[len(anchor.rstrip("\r\n")):] or "\n"
    if not anchor.endswith(("\n", "\r")):
        lines[i] = anchor + ending
    lines.insert(i + 1, instruction + ending)
    return "".join(lines)


def _loose_count(line: str) -> tuple[bool, int | None]:
    for rx in _LOOSE_RXS:
        m = rx.match(line.strip())
        if m:
            raw = m.group("n").lower()
            return True, int(raw) if raw.isdigit() else _WORD_NUMBERS.get(raw)
    return False, None


def _is_count_line(line: str) -> bool:
    m = _STRICT_RX.match(line.strip())
    if m and not m.group("rest").strip():
        return True
    return _loose_count(line)[0]


@dataclass
class CountFirstParse:
    status: str
    predicted_hops: int | None
    count_line: str | None
    steps_text: str
    extra_count_lines: list[str] = field(default_factory=list)

    @property
    def has_count(self) -> bool:
        return self.predicted_hops is not None


def parse_count_first(text: str) -> CountFirstParse:
    """Split a post-processed generation into its count line and its steps.

    - The status is decided on the FIRST non-empty line only (the format is "count first"):
      ``ok`` (exactly ``[N hops]``), ``inline`` (``[N hops]`` followed by step text on the
      same line; the text is kept as the first step), ``malformed`` (a line that is only a
      count statement in another shape; its number is used when one can be read), ``absent``
      (the first line is not a count statement; nothing is removed from it), ``empty``.
    - Any LATER line that is only a count statement is removed too (it is not a step) and
      listed in ``extra_count_lines``; it never supplies the prediction.
    - Non-count lines are returned verbatim (enumerators, blank lines and indentation
      kept); only the outer whitespace is stripped. Scoring normalises them.
    """
    lines = [ln for ln in (text or "").splitlines()]
    first = next((i for i, ln in enumerate(lines) if ln.strip()), None)
    if first is None:
        return CountFirstParse("empty", None, None, "")

    status, predicted, count_line = "absent", None, None
    head = lines[first].strip()
    strict = _STRICT_RX.match(head)
    body = lines[first + 1:]
    kept_head: list[str] = []
    if strict:
        predicted = int(strict.group("n"))
        rest = strict.group("rest").strip()
        status = "inline" if rest else "ok"
        count_line = head[: strict.start("rest")].strip()
        if rest:
            kept_head = [rest]
    else:
        is_loose, n = _loose_count(head)
        if is_loose:
            status, predicted, count_line = "malformed", n, head
        else:
            kept_head = [lines[first]]

    extra: list[str] = []
    kept: list[str] = list(kept_head)
    for ln in body:
        if ln.strip() and _is_count_line(ln):
            extra.append(ln.strip())
            continue
        kept.append(ln)
    # Every non-count line is kept exactly as written (blank lines and indentation
    # included); only the outer whitespace goes, as post_process_generation does for every
    # arm. The evaluator's splitter normalises the rest at scoring time.
    steps_text = "\n".join(kept).strip()
    return CountFirstParse(status, predicted, count_line, steps_text, extra)


def _rate(num: int, den: int) -> float | None:
    return num / den if den else None


def summarize_count_first(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Accuracy of the self-predicted count vs gold, and agreement with the steps emitted.

    Each row needs ``hop_count`` (the gold depth), ``count_first_predicted_hops`` (int or
    None), ``count_first_status`` and ``decomposition`` (the steps that were scored; the
    emitted step count is taken with the evaluator's splitter). Nothing is imputed: a row
    with no usable count is a miss in the ``*_all_rows`` rates and is excluded from the
    ``*_parsed_rows`` rates, and both denominators are stated.
    """
    rows = list(rows)
    n = len(rows)
    status_counts = Counter(r["count_first_status"] for r in rows)
    parsed = [r for r in rows if r.get("count_first_predicted_hops") is not None]
    hops = sorted({int(r["hop_count"]) for r in rows})

    def block(sub: list[dict[str, Any]]) -> dict[str, Any]:
        sub_parsed = [r for r in sub if r.get("count_first_predicted_hops") is not None]
        correct = sum(1 for r in sub_parsed if r["count_first_predicted_hops"] == r["hop_count"])
        under = sum(1 for r in sub_parsed if r["count_first_predicted_hops"] < r["hop_count"])
        over = sum(1 for r in sub_parsed if r["count_first_predicted_hops"] > r["hop_count"])
        agree = sum(
            1 for r in sub_parsed
            if r["count_first_predicted_hops"] == len(split_step_lines(r["decomposition"] or ""))
        )
        errors = [r["count_first_predicted_hops"] - r["hop_count"] for r in sub_parsed]
        return {
            "rows": len(sub),
            "rows_with_count": len(sub_parsed),
            "count_eq_gold": correct,
            "count_eq_gold_rate_all_rows": _rate(correct, len(sub)),
            "count_eq_gold_rate_parsed_rows": _rate(correct, len(sub_parsed)),
            "count_below_gold": under,
            "count_above_gold": over,
            "mean_signed_count_error_parsed_rows": (
                sum(errors) / len(errors) if errors else None
            ),
            "count_eq_steps_emitted": agree,
            "count_eq_steps_emitted_rate_parsed_rows": _rate(agree, len(sub_parsed)),
        }

    confusion: dict[str, dict[str, int]] = {}
    for r in rows:
        pred = r.get("count_first_predicted_hops")
        key = str(pred) if pred is not None else "none"
        confusion.setdefault(str(r["hop_count"]), {}).setdefault(key, 0)
        confusion[str(r["hop_count"])][key] += 1

    steps_vs_count = Counter()
    for r in parsed:
        emitted = len(split_step_lines(r["decomposition"] or ""))
        pred = r["count_first_predicted_hops"]
        steps_vs_count["equal" if emitted == pred else "more_steps" if emitted > pred else "fewer_steps"] += 1

    return {
        "rows": n,
        "status_counts": {s: status_counts.get(s, 0) for s in STATUSES},
        "rows_with_count": len(parsed),
        "rows_without_count": n - len(parsed),
        "rows_with_extra_count_lines": sum(
            1 for r in rows if r.get("count_first_extra_count_lines")
        ),
        "overall": block(rows),
        "per_gold_hop": {str(h): block([r for r in rows if int(r["hop_count"]) == h]) for h in hops},
        "confusion_gold_by_predicted": confusion,
        "steps_emitted_vs_count": {
            k: steps_vs_count.get(k, 0) for k in ("equal", "more_steps", "fewer_steps")
        },
        "definitions": {
            "status": (
                "decided on the first non-empty line of the post-processed generation: ok "
                "= exactly '[N hops]'; inline = '[N hops]' followed by step text on the same "
                "line (kept as the first step); malformed = a line that is only a hop-count "
                "statement in another shape (its number used when readable); absent = first "
                "line is not a count statement (nothing removed); empty = no output"
            ),
            "rows_without_count": (
                "rows whose count could not be read (absent, empty, or malformed without a "
                "number). These rows are NOT dropped: their steps are scored by the evaluator "
                "like every other row"
            ),
            "count_eq_gold_rate_all_rows": "a row without a count counts as a miss",
            "count_eq_gold_rate_parsed_rows": "over rows with a count only",
            "count_eq_steps_emitted": (
                "the count equals the number of scored steps (src/step_lines.py splitter, "
                "count line removed)"
            ),
        },
    }
