"""MuSiQue's official answer metrics: answer EM and answer F1.

Where each definition comes from
--------------------------------
MuSiQue's official evaluation scores a predicted answer with the **SQuAD** answer
metrics; MuSiQue's own ``metrics/answer.py`` (the ``AnswerMetric`` used by its
``evaluate_v1.0.py``) is a thin wrapper around them, taking the maximum over the
item's gold answer set. Each function below names its source:

- :func:`normalize_answer` — the SQuAD official evaluation script's
  ``normalize_answer``: lowercase, remove punctuation (``string.punctuation``),
  remove the articles ``a``/``an``/``the``, collapse whitespace. Applied in that
  order (``white_space_fix(remove_articles(remove_punc(lower(s))))``), which
  matters: punctuation is stripped *before* the article regex, so ``"the,"``
  becomes ``"the"`` and is then removed.
- :func:`get_tokens` — SQuAD's ``get_tokens``: ``normalize_answer(s).split()``,
  with the empty string mapping to no tokens.
- :func:`compute_exact` — SQuAD's ``compute_exact``:
  ``int(normalize_answer(gold) == normalize_answer(pred))``.
- :func:`compute_f1` — SQuAD's ``compute_f1``: bag-of-tokens F1 over
  :func:`get_tokens` with a **multiset** intersection
  (``collections.Counter(gold) & collections.Counter(pred)``), and SQuAD's edge
  case: when either side has no tokens the score is
  ``int(gold_tokens == pred_tokens)`` (1.0 only when both are empty), and when the
  overlap is empty the score is 0.0.
- :func:`score_answer` — MuSiQue's ``metric_max_over_ground_truths``: the score is
  the **maximum** over the gold answer set, which for a MuSiQue item is its
  ``answer`` field plus every entry of ``answer_aliases``
  (:func:`gold_answer_set`).

No model is in the loop anywhere in this module: these are string metrics, which is
what makes them admissible under the standing constraint that no closed commercial
model may score anything in the evaluation loop (CLAUDE.md).

Honesty note about provenance: the definitions above are implemented from the
published SQuAD/MuSiQue evaluation code as documented, not copied from a fetched
copy of ``metrics/answer.py`` — this box has no network access to the MuSiQue
release. A reviewer with the upstream file should diff it against this module
before any number produced here is published; the hand-computed vectors in
``tests/test_answer_musique.py`` pin the behaviour in the meantime.
"""
from __future__ import annotations

import collections
import re
import string
from typing import Any, Iterable

#: SQuAD's article regex, matched as whole words.
_ARTICLES_RX = re.compile(r"\b(a|an|the)\b", re.UNICODE)

#: SQuAD strips exactly Python's ``string.punctuation`` (so ``#`` goes too — an answer is
#: not step text, and the ``[#k]`` grammar that ``src/step_lines.py`` protects does not
#: apply here).
_PUNCTUATION = frozenset(string.punctuation)


def normalize_answer(text: str) -> str:
    """SQuAD ``normalize_answer``: lower, de-punctuate, drop articles, fix whitespace."""
    if text is None:
        return ""
    lowered = str(text).lower()
    depunctuated = "".join(ch for ch in lowered if ch not in _PUNCTUATION)
    de_articled = _ARTICLES_RX.sub(" ", depunctuated)
    return " ".join(de_articled.split())


def get_tokens(text: str) -> list[str]:
    """SQuAD ``get_tokens``: the normalized answer split on whitespace."""
    if not text:
        return []
    return normalize_answer(text).split()


def compute_exact(gold: str, pred: str) -> float:
    """SQuAD ``compute_exact``: 1.0 when the normalized strings are equal, else 0.0."""
    return 1.0 if normalize_answer(gold) == normalize_answer(pred) else 0.0


def compute_f1(gold: str, pred: str) -> float:
    """SQuAD ``compute_f1``: multiset token F1, with SQuAD's empty-side edge case."""
    gold_tokens = get_tokens(gold)
    pred_tokens = get_tokens(pred)
    if not gold_tokens or not pred_tokens:
        # SQuAD: with either side empty, the two can only agree by both being empty.
        return 1.0 if gold_tokens == pred_tokens else 0.0
    common = collections.Counter(gold_tokens) & collections.Counter(pred_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2.0 * precision * recall / (precision + recall)


def gold_answer_set(answer: Any, aliases: Any) -> list[str]:
    """The gold answers of one MuSiQue item: ``answer`` plus its ``answer_aliases``.

    Order is preserved (``answer`` first) and duplicates are dropped, so the returned
    list is what the max is taken over. Non-string entries are ignored rather than
    coerced: ``str(None)`` would put the literal ``"None"`` into the gold set.
    """
    out: list[str] = []
    candidates: list[Any] = [answer]
    if isinstance(aliases, Iterable) and not isinstance(aliases, (str, bytes)):
        candidates.extend(aliases)
    for value in candidates:
        if isinstance(value, str) and value.strip() and value not in out:
            out.append(value)
    return out


def score_answer(pred: str, golds: list[str]) -> tuple[float, float]:
    """``(exact_match, f1)`` — MuSiQue's max over the gold answer set.

    With an empty gold set there is nothing to score against, so both are 0.0; that is
    a broken item rather than a correct prediction, and the caller counts it.
    """
    if not golds:
        return (0.0, 0.0)
    em = max(compute_exact(gold, pred) for gold in golds)
    f1 = max(compute_f1(gold, pred) for gold in golds)
    return (em, f1)


#: Written into every metrics JSON that reports these numbers. A metric named "EM" says
#: nothing without its normalization; the definitions travel with the numbers.
ANSWER_METRIC_DEFINITIONS: dict[str, Any] = {
    "answer_normalization": (
        "SQuAD official normalize_answer: lowercase, remove string.punctuation, remove "
        "the articles a/an/the, collapse whitespace (applied in that order)"
    ),
    "answer_em": (
        "1.0 when the normalized prediction equals a normalized gold answer, else 0.0; "
        "maximum over the gold set (MuSiQue's answer plus its answer_aliases), then "
        "averaged over evaluated items (macro)"
    ),
    "answer_f1": (
        "multiset bag-of-tokens F1 between the normalized prediction and a gold answer "
        "(SQuAD compute_f1, including its empty-side rule: with either side empty the "
        "score is 1.0 only if both are empty); maximum over the gold set, then averaged "
        "over evaluated items (macro)"
    ),
    "gold_answer_set": (
        "the MuSiQue item's 'answer' field plus every string in 'answer_aliases', "
        "de-duplicated, answer first"
    ),
    "not_a_semantic_metric": (
        "string-level only: no model scores, rates or judges anything here (CLAUDE.md "
        "standing constraint). A correct answer worded differently from every gold "
        "string scores 0 on EM and partially on F1"
    ),
    "source": (
        "SQuAD official evaluation script (normalize_answer / get_tokens / compute_exact "
        "/ compute_f1) as adopted by MuSiQue's metrics/answer.py, with the score taken as "
        "the maximum over the gold answer set (metric_max_over_ground_truths). "
        "Implemented from the published definitions; see the provenance note in "
        "src/answer_metrics.py"
    ),
}
