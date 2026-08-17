#!/usr/bin/env python3
"""
Evaluate predicted MuSiQue decompositions against gold dev decompositions.

Scoring techniques (all string-level, no model in the loop):
- Exact match (full decomposition)
- Step-level precision/recall/F1 (unordered)
- Ordered step accuracy
- Step count error, signed (over- vs under-decomposition) and absolute
- Reference validity for [#k] chains
- ROUGE-L precision/recall/F1 (LCS-based)
- A composite score whose weights come from the config

Inputs:
- predictions: JSON list (e.g. a decomposer run's results.json) of items like
    {"question": "...", "decomposition": "..."}
- gold: JSONL with items containing
    {"question": "...", "question_decomposition": [{"question": "..."}, ...]}

Two modes:
- default: score one predictions file against gold, writing ``<prefix>_per_item.json``
  plus the standard config/metrics/notes trail.
- ``--compare A_per_item.json B_per_item.json``: paired significance between two runs
  **on the same evaluation set** (bootstrap CIs + McNemar). It aligns rows by item id
  and refuses to run when the two sets differ, because a comparison across different
  evaluation sets is not a comparison (CLAUDE.md, evidence discipline).

Ported from v1 ``scripts/musique_decompositions_evaluator.py``. Adapted for v2: the
gold path, run directory, seed, limit, the composite-score weights and the paired
comparison parameters come from ``configs/musique_eval.json``; the run writes the
standard config/metrics/notes trail.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from run_artifacts import now_iso, write_run_artifacts  # noqa: E402
from run_config import load_config, load_paths, require, resolve_path, runs_path  # noqa: E402
from seeding import set_global_seed  # noqa: E402

_WS_RX = re.compile(r"\s+")
_PUNCT_KEEP_HASH_RX = re.compile(r"[^\w\s#]")
_REF_RX = re.compile(r"\[#(\d+)\]")

#: Written into every metrics JSON. A number like "step F1 0.42" is meaningless without
#: the normalization and the matching rule behind it, and those live in code — so the
#: definitions travel with the metrics rather than in a doc that can drift.
METRIC_DEFINITIONS: dict[str, Any] = {
    "step_normalization": (
        "each step is lowercased, punctuation is stripped except '#' (so [#k] "
        "references survive), and whitespace is collapsed to single spaces"
    ),
    "step_splitting": (
        "a string decomposition is split on newlines and a leading '<n>. ' enumerator "
        "is removed per line; a list decomposition takes each string, or each item's "
        "'question' field"
    ),
    "question_matching": (
        "a prediction is joined to gold by its question text, lowercased and "
        "whitespace-collapsed (not by id); unmatched predictions are counted in "
        "missing_gold_count and excluded from every metric"
    ),
    "exact_match_rate": (
        "1.0 when the predicted step list equals the gold step list in length and in "
        "normalized text at every position, else 0.0; averaged over evaluated rows"
    ),
    "step_precision_macro": "set-based: |normalized pred steps AND gold steps| / |pred steps|, averaged over rows",
    "step_recall_macro": "set-based: |normalized pred steps AND gold steps| / |gold steps|, averaged over rows",
    "step_f1_macro": (
        "harmonic mean of the per-row set-based precision and recall, averaged over "
        "rows (macro, not computed from the averaged P and R). Unordered and "
        "duplicate-insensitive: identical steps collapse in the set"
    ),
    "ordered_step_accuracy_macro": (
        "positional: matches at the same index / max(len(pred), len(gold)), averaged "
        "over rows; a length mismatch is penalised through the denominator"
    ),
    "rouge_l_macro": (
        "ROUGE-L on the newline-joined step text, LCS over whitespace tokens of the "
        "lowercased text (no punctuation stripping at this stage), averaged over rows"
    ),
    "reference_validity": (
        "a [#k] reference is valid when 1 <= k < its own step index. macro = mean of "
        "per-row valid rates (a row with no references scores 1.0); micro = total valid "
        "references / total references across all rows"
    ),
    "step_count_abs_error_mae": "mean of |len(pred steps) - len(gold steps)|",
    "step_count_mae": (
        "same number as step_count_abs_error_mae, reported again so the directional "
        "step-count family (over/under rates, signed mean, MAE) reads as one block"
    ),
    "step_count_signed_error": (
        "per item: len(pred steps) - len(gold steps). Positive = over-decomposition "
        "(more steps than gold), negative = under-decomposition"
    ),
    "mean_signed_step_count_error": (
        "mean of the per-item signed error. Near zero does NOT mean accurate step "
        "counts: over- and under-decomposition cancel, which is why the two rates and "
        "the MAE are reported next to it"
    ),
    "over_decomposition_rate": "fraction of rows with len(pred steps) > len(gold steps)",
    "under_decomposition_rate": "fraction of rows with len(pred steps) < len(gold steps)",
    "item_id": (
        "the prediction's 'query_id', else its 'id', else its normalized question text; "
        "this is the key --compare aligns two runs on"
    ),
    "hop_count_metrics": (
        "predicted hop count is the number of predicted steps; gold hop count is the "
        "gold 'hop_count' field when present and positive, else the gold step count"
    ),
    "composite_score": (
        "weighted sum of step_f1_macro, ordered_step_accuracy_macro, "
        "reference_validity_micro and a step-count term "
        "max(0, 1 - step_count_abs_error_mae / scale); the weights and scale are "
        "recorded alongside as composite_score_weights and "
        "composite_step_count_error_scale"
    ),
    "not_a_semantic_metric": (
        "every metric here is string-level. Two decompositions that mean the same thing "
        "but word a step differently score as a mismatch"
    ),
}


#: Written into every comparison metrics JSON, for the same reason as METRIC_DEFINITIONS:
#: "significant" is meaningless without the test and the resampling unit behind it.
COMPARISON_DEFINITIONS: dict[str, Any] = {
    "alignment": (
        "the two per-item files are aligned on 'item_id' and must cover exactly the same "
        "ids; any id present in one and not the other aborts the run"
    ),
    "difference_direction": "every reported difference is system_a minus system_b",
    "paired_bootstrap": (
        "percentile bootstrap over items: each of the bootstrap_iterations resamples "
        "draws n item indices with replacement and applies the SAME indices to both "
        "systems (paired), then recomputes each statistic from the resampled items. The "
        "reported interval is the [alpha/2, 1-alpha/2] percentile of the difference"
    ),
    "bootstrap_significance": (
        "significant = the confidence interval of the difference excludes 0. This is a "
        "CI-based decision, not a p-value"
    ),
    "composite_score_in_bootstrap": (
        "recomputed on each resample from the resampled items (its reference-validity "
        "term is a micro rate and its step-count term a MAE, so neither is an average of "
        "per-item values), using the weights and scale of THIS config — which are the "
        "config's, not necessarily the ones the per-item files were produced with"
    ),
    "mcnemar": (
        "exact two-sided McNemar on the discordant pairs of a binary metric: with b = "
        "#(a correct, b wrong) and c = #(a wrong, b correct), p = min(1, 2 * "
        "BinomialCDF(min(b, c); b + c, 0.5)). p = 1.0 when there are no discordant pairs"
    ),
    "mcnemar_significance": "significant = p_value < alpha",
    "multiple_comparisons": (
        "six tests are reported per comparison and none of the p-values or intervals is "
        "corrected for multiple comparisons"
    ),
}


@dataclass
class EvalRow:
    item_id: str
    question: str
    pred_steps: list[str]
    gold_steps: list[str]
    gold_hop_count: int


def _normalize_question(text: str) -> str:
    text = text.strip().lower()
    return _WS_RX.sub(" ", text)


def _normalize_step(text: str) -> str:
    x = text.strip().lower()
    x = _PUNCT_KEEP_HASH_RX.sub("", x)
    return _WS_RX.sub(" ", x)


def _tokenize(text: str) -> list[str]:
    return [t for t in _WS_RX.sub(" ", text.strip().lower()).split(" ") if t]


def _split_decomposition_text(text: str) -> list[str]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    cleaned: list[str] = []
    for ln in lines:
        ln = re.sub(r"^\s*\d+\.\s*", "", ln)
        if ln:
            cleaned.append(ln)
    return cleaned


def _decomp_to_steps(value: Any) -> list[str]:
    if isinstance(value, str):
        return _split_decomposition_text(value)
    if isinstance(value, list):
        out: list[str] = []
        for it in value:
            if isinstance(it, str):
                if it.strip():
                    out.append(it.strip())
            elif isinstance(it, dict):
                q = it.get("question")
                if isinstance(q, str) and q.strip():
                    out.append(q.strip())
        return out
    return []


def _load_gold(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            question = obj.get("question")
            if not isinstance(question, str) or not question.strip():
                continue
            steps = _decomp_to_steps(obj.get("question_decomposition"))
            hop_count = obj.get("hop_count")
            if not isinstance(hop_count, int) or hop_count <= 0:
                hop_count = len(steps)
            out[_normalize_question(question)] = {"steps": steps, "hop_count": hop_count}
    return out


def _load_predictions(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise SystemExit(f"--predictions must be a JSON list: {path}")
    return raw


def _lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        ai = a[i - 1]
        for j in range(1, n + 1):
            if ai == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = dp[i - 1][j] if dp[i - 1][j] >= dp[i][j - 1] else dp[i][j - 1]
    return dp[m][n]


def _rouge_l(pred: str, gold: str) -> tuple[float, float, float]:
    pred_toks = _tokenize(pred)
    gold_toks = _tokenize(gold)
    if not pred_toks and not gold_toks:
        return (1.0, 1.0, 1.0)
    if not pred_toks or not gold_toks:
        return (0.0, 0.0, 0.0)
    lcs = _lcs_len(pred_toks, gold_toks)
    precision = lcs / len(pred_toks)
    recall = lcs / len(gold_toks)
    f1 = 0.0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)
    return (precision, recall, f1)


def _reference_validity(steps: list[str]) -> tuple[float, int, int]:
    """Return (valid_rate, valid_refs, total_refs)."""
    total = 0
    valid = 0
    for idx, step in enumerate(steps, start=1):
        for ref in _REF_RX.findall(step):
            total += 1
            k = int(ref)
            if 1 <= k < idx:
                valid += 1
    rate = 1.0 if total == 0 else valid / total
    return (rate, valid, total)


def _safe_div(a: float, b: float) -> float:
    return 0.0 if b == 0 else a / b


def _step_prf(pred_steps: list[str], gold_steps: list[str]) -> tuple[float, float, float]:
    pset = {_normalize_step(s) for s in pred_steps if s.strip()}
    gset = {_normalize_step(s) for s in gold_steps if s.strip()}
    if not pset and not gset:
        return (1.0, 1.0, 1.0)
    tp = len(pset & gset)
    precision = _safe_div(tp, len(pset))
    recall = _safe_div(tp, len(gset))
    f1 = 0.0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)
    return (precision, recall, f1)


def _ordered_step_accuracy(pred_steps: list[str], gold_steps: list[str]) -> float:
    if not pred_steps and not gold_steps:
        return 1.0
    denom = max(len(pred_steps), len(gold_steps))
    if denom == 0:
        return 0.0
    matches = 0
    for i in range(min(len(pred_steps), len(gold_steps))):
        if _normalize_step(pred_steps[i]) == _normalize_step(gold_steps[i]):
            matches += 1
    return matches / denom


def _exact_decomposition_match(pred_steps: list[str], gold_steps: list[str]) -> float:
    if len(pred_steps) != len(gold_steps):
        return 0.0
    for p, g in zip(pred_steps, gold_steps):
        if _normalize_step(p) != _normalize_step(g):
            return 0.0
    return 1.0


def _join_steps(steps: list[str]) -> str:
    return "\n".join(s.strip() for s in steps if s.strip())


def _item_id(obj: dict[str, Any], question: str) -> str:
    """The key --compare aligns on. Falls back to the normalized question text."""
    for key in ("query_id", "id"):
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
    return _normalize_question(question)


def _build_eval_rows(
    predictions: list[dict[str, Any]],
    gold_by_question: dict[str, dict[str, Any]],
) -> tuple[list[EvalRow], int]:
    rows: list[EvalRow] = []
    missing_gold = 0
    for obj in predictions:
        question = obj.get("question")
        if not isinstance(question, str) or not question.strip():
            continue
        gold_obj = gold_by_question.get(_normalize_question(question))
        if gold_obj is None:
            missing_gold += 1
            continue
        rows.append(
            EvalRow(
                item_id=_item_id(obj, question),
                question=question,
                pred_steps=_decomp_to_steps(obj.get("decomposition")),
                gold_steps=gold_obj["steps"],
                gold_hop_count=int(gold_obj["hop_count"]),
            )
        )
    return rows, missing_gold


_EMPTY_AGGREGATE = {
    "num_rows": 0,
    "exact_match_rate": 0.0,
    "step_precision_macro": 0.0,
    "step_recall_macro": 0.0,
    "step_f1_macro": 0.0,
    "ordered_step_accuracy_macro": 0.0,
    "rouge_l_precision_macro": 0.0,
    "rouge_l_recall_macro": 0.0,
    "rouge_l_f1_macro": 0.0,
    "reference_validity_macro": 0.0,
    "reference_validity_micro": 1.0,
    "step_count_abs_error_mae": 0.0,
    "step_count_mae": 0.0,
    "mean_signed_step_count_error": 0.0,
    "over_decomposition_rate": 0.0,
    "under_decomposition_rate": 0.0,
    "hop_count_exact_match_rate": 0.0,
    "hop_count_abs_error_mae": 0.0,
    "predicted_hop_distribution": {},
}


def _aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
    m = len(items)
    if m == 0:
        return dict(_EMPTY_AGGREGATE)

    ref_valid_num = sum(int(it["reference_valid_count"]) for it in items)
    ref_valid_den = sum(int(it["reference_total_count"]) for it in items)
    pred_dist: dict[str, int] = {}
    for it in items:
        k = str(int(it["predicted_hop_count"]))
        pred_dist[k] = pred_dist.get(k, 0) + 1

    def mean(key: str) -> float:
        return sum(float(it[key]) for it in items) / m

    signed = [int(it["step_count_signed_error"]) for it in items]
    step_count_mae = mean("step_count_abs_error")

    return {
        "num_rows": m,
        "exact_match_rate": mean("exact_match"),
        "step_precision_macro": mean("step_precision"),
        "step_recall_macro": mean("step_recall"),
        "step_f1_macro": mean("step_f1"),
        "ordered_step_accuracy_macro": mean("ordered_step_accuracy"),
        "rouge_l_precision_macro": mean("rouge_l_precision"),
        "rouge_l_recall_macro": mean("rouge_l_recall"),
        "rouge_l_f1_macro": mean("rouge_l_f1"),
        "reference_validity_macro": mean("reference_validity_rate"),
        "reference_validity_micro": 1.0 if ref_valid_den == 0 else ref_valid_num / ref_valid_den,
        "step_count_abs_error_mae": step_count_mae,
        "step_count_mae": step_count_mae,
        "mean_signed_step_count_error": sum(signed) / m,
        "over_decomposition_rate": sum(1 for s in signed if s > 0) / m,
        "under_decomposition_rate": sum(1 for s in signed if s < 0) / m,
        "hop_count_exact_match_rate": mean("hop_count_exact_match"),
        "hop_count_abs_error_mae": mean("hop_count_abs_error"),
        "predicted_hop_distribution": pred_dist,
    }


def _composite_score(overall: dict[str, Any], weights: dict[str, float], scale: float) -> float:
    step_count_term = max(0.0, 1.0 - (overall["step_count_abs_error_mae"] / scale))
    return (
        float(require(weights, "step_f1_macro")) * overall["step_f1_macro"]
        + float(require(weights, "ordered_step_accuracy_macro")) * overall["ordered_step_accuracy_macro"]
        + float(require(weights, "reference_validity_micro")) * overall["reference_validity_micro"]
        + float(require(weights, "step_count_error")) * step_count_term
    )


# --------------------------------------------------------------------------------------
# Paired comparison of two runs (--compare)
# --------------------------------------------------------------------------------------

#: Statistics compared with a paired bootstrap CI. Every one is recomputed from the
#: resampled items, so the point estimate and the interval come from the same code path.
BOOTSTRAP_STATISTICS = ("rouge_l_f1", "step_f1", "ordered_step_accuracy", "composite_score")

#: Binary per-item metrics compared with McNemar.
MCNEMAR_STATISTICS = ("exact_match", "hop_count_exact_match")

#: Per-item fields a comparison needs. A file missing one of them is a file this script
#: did not write (or wrote before these metrics existed), and saying so beats a KeyError.
_REQUIRED_PER_ITEM_FIELDS = (
    "item_id",
    "step_f1",
    "ordered_step_accuracy",
    "rouge_l_f1",
    "reference_valid_count",
    "reference_total_count",
    "step_count_abs_error",
    *MCNEMAR_STATISTICS,
)


def _load_per_item(path: Path) -> dict[str, dict[str, Any]]:
    """Load a ``*_per_item.json`` file into {item_id: row}, failing loudly on duplicates."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise SystemExit(f"--compare expects a per-item JSON list: {path}")
    by_id: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for i, obj in enumerate(raw):
        if not isinstance(obj, dict):
            raise SystemExit(f"{path}: row {i} is not an object")
        missing = [f for f in _REQUIRED_PER_ITEM_FIELDS if f not in obj]
        if missing:
            raise SystemExit(
                f"{path}: row {i} is missing {missing}. --compare takes the "
                f"'<prefix>_per_item.json' files written by this script; re-run the "
                f"evaluation to regenerate them."
            )
        item_id = str(obj["item_id"])
        if item_id in by_id:
            duplicates.append(item_id)
        by_id[item_id] = obj
    if duplicates:
        raise SystemExit(f"{path}: duplicate item_id(s): {sorted(set(duplicates))}")
    return by_id


def _aligned_ids(
    a_by_id: dict[str, dict[str, Any]],
    b_by_id: dict[str, dict[str, Any]],
    path_a: Path,
    path_b: Path,
    max_reported: int,
) -> list[str]:
    """Ids common to both files, or abort naming the offenders.

    A comparison across two different evaluation sets is not a comparison (CLAUDE.md,
    evidence discipline), so a mismatch is fatal rather than an intersection.
    """
    only_a = sorted(set(a_by_id) - set(b_by_id))
    only_b = sorted(set(b_by_id) - set(a_by_id))
    if only_a or only_b:
        def _fmt(ids: list[str]) -> str:
            shown = ids[:max_reported]
            more = "" if len(ids) <= max_reported else f" ... (+{len(ids) - max_reported} more)"
            return ", ".join(shown) + more

        lines = [
            "--compare requires the SAME evaluation set in both files; they differ.",
            f"  a: {path_a} ({len(a_by_id)} items)",
            f"  b: {path_b} ({len(b_by_id)} items)",
        ]
        if only_a:
            lines.append(f"  only in a ({len(only_a)}): {_fmt(only_a)}")
        if only_b:
            lines.append(f"  only in b ({len(only_b)}): {_fmt(only_b)}")
        raise SystemExit("\n".join(lines))
    return sorted(a_by_id)


def _statistic_arrays(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """The per-item columns every compared statistic is built from."""
    def col(key: str) -> np.ndarray:
        return np.array([float(r[key]) for r in rows], dtype=float)

    return {
        "step_f1": col("step_f1"),
        "ordered_step_accuracy": col("ordered_step_accuracy"),
        "rouge_l_f1": col("rouge_l_f1"),
        "reference_valid_count": col("reference_valid_count"),
        "reference_total_count": col("reference_total_count"),
        "step_count_abs_error": col("step_count_abs_error"),
    }


def _statistics_for(
    arrays: dict[str, np.ndarray],
    index: np.ndarray,
    weights: dict[str, float],
    scale: float,
) -> dict[str, np.ndarray]:
    """Evaluate every bootstrap statistic on resamples ``index`` of shape (draws, n).

    Returns one array of length ``draws`` per statistic. The point estimate uses this
    same function with a single "resample" that is the identity, so the observed value
    and the bootstrap distribution can never drift apart.
    """
    step_f1 = arrays["step_f1"][index].mean(axis=1)
    ordered = arrays["ordered_step_accuracy"][index].mean(axis=1)
    rouge = arrays["rouge_l_f1"][index].mean(axis=1)

    valid = arrays["reference_valid_count"][index].sum(axis=1)
    total = arrays["reference_total_count"][index].sum(axis=1)
    # A resample with no [#k] references at all scores 1.0, matching _aggregate().
    ref_micro = np.where(total == 0, 1.0, valid / np.where(total == 0, 1.0, total))

    mae = arrays["step_count_abs_error"][index].mean(axis=1)
    step_count_term = np.maximum(0.0, 1.0 - mae / scale)

    composite = (
        float(require(weights, "step_f1_macro")) * step_f1
        + float(require(weights, "ordered_step_accuracy_macro")) * ordered
        + float(require(weights, "reference_validity_micro")) * ref_micro
        + float(require(weights, "step_count_error")) * step_count_term
    )
    return {
        "rouge_l_f1": rouge,
        "step_f1": step_f1,
        "ordered_step_accuracy": ordered,
        "composite_score": composite,
    }


def _paired_bootstrap(
    arrays_a: dict[str, np.ndarray],
    arrays_b: dict[str, np.ndarray],
    n: int,
    iterations: int,
    alpha: float,
    seed: int,
    weights: dict[str, float],
    scale: float,
) -> dict[str, dict[str, float]]:
    identity = np.arange(n)[None, :]
    point_a = _statistics_for(arrays_a, identity, weights, scale)
    point_b = _statistics_for(arrays_b, identity, weights, scale)

    rng = np.random.default_rng(seed)
    index = rng.integers(0, n, size=(iterations, n))
    draws_a = _statistics_for(arrays_a, index, weights, scale)
    draws_b = _statistics_for(arrays_b, index, weights, scale)

    lo_pct = 100.0 * (alpha / 2.0)
    hi_pct = 100.0 * (1.0 - alpha / 2.0)

    out: dict[str, dict[str, float]] = {}
    for name in BOOTSTRAP_STATISTICS:
        diffs = draws_a[name] - draws_b[name]
        ci_low, ci_high = (float(x) for x in np.percentile(diffs, [lo_pct, hi_pct]))
        out[name] = {
            "system_a": float(point_a[name][0]),
            "system_b": float(point_b[name][0]),
            "difference": float(point_a[name][0] - point_b[name][0]),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "significant": bool(ci_low > 0.0 or ci_high < 0.0),
        }
    return out


def _mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value on discordant counts ``b`` and ``c``."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1))
    return min(1.0, 2.0 * (tail / (2**n)))


def _mcnemar(
    rows_a: list[dict[str, Any]],
    rows_b: list[dict[str, Any]],
    alpha: float,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for name in MCNEMAR_STATISTICS:
        a_vals = [float(r[name]) >= 1.0 for r in rows_a]
        b_vals = [float(r[name]) >= 1.0 for r in rows_b]
        only_a = sum(1 for x, y in zip(a_vals, b_vals) if x and not y)
        only_b = sum(1 for x, y in zip(a_vals, b_vals) if y and not x)
        p_value = _mcnemar_exact_p(only_a, only_b)
        n = len(a_vals)
        out[name] = {
            "system_a_rate": sum(a_vals) / n,
            "system_b_rate": sum(b_vals) / n,
            "difference": (sum(a_vals) - sum(b_vals)) / n,
            "correct_only_in_a": only_a,
            "correct_only_in_b": only_b,
            "discordant_pairs": only_a + only_b,
            "p_value": p_value,
            "significant": bool(p_value < alpha),
        }
    return out


def _compare(
    args: argparse.Namespace,
    cfg: dict[str, Any],
    paths_cfg: dict[str, Any],
    seed: int,
    seeded: dict[str, Any],
    weights: dict[str, float],
    scale: float,
) -> None:
    compare_cfg = require(cfg, "paired_comparison")
    iterations = int(require(compare_cfg, "bootstrap_iterations"))
    alpha = float(require(compare_cfg, "alpha"))
    max_reported = int(require(compare_cfg, "max_reported_id_mismatches"))
    out_prefix = args.out_prefix or require(compare_cfg, "out_prefix")

    path_a, path_b = args.compare
    a_by_id = _load_per_item(path_a)
    b_by_id = _load_per_item(path_b)
    ids = _aligned_ids(a_by_id, b_by_id, path_a, path_b, max_reported)
    if not ids:
        raise SystemExit("--compare: both files are empty, nothing to compare.")

    rows_a = [a_by_id[i] for i in ids]
    rows_b = [b_by_id[i] for i in ids]
    n = len(ids)

    bootstrap = _paired_bootstrap(
        _statistic_arrays(rows_a),
        _statistic_arrays(rows_b),
        n=n,
        iterations=iterations,
        alpha=alpha,
        seed=seed,
        weights=weights,
        scale=scale,
    )
    mcnemar = _mcnemar(rows_a, rows_b, alpha)

    run_dir = args.run_dir if args.run_dir is not None else runs_path(paths_cfg, require(cfg, "run_subdir"))
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "script": Path(__file__).name,
        "mode": "compare",
        "created_utc": now_iso(),
        "seed": seed,
        "seeded": seeded,
        "system_a_path": str(path_a.resolve()),
        "system_b_path": str(path_b.resolve()),
        "num_aligned_items": n,
        "bootstrap_iterations": iterations,
        "alpha": alpha,
        "confidence_level": 1.0 - alpha,
        "difference_direction": "system_a minus system_b",
        "bootstrap": bootstrap,
        "mcnemar": mcnemar,
        "composite_score_weights": weights,
        "composite_step_count_error_scale": scale,
        "comparison_definitions": COMPARISON_DEFINITIONS,
    }

    snapshot = {
        "script": Path(__file__).name,
        "mode": "compare",
        "created_utc": now_iso(),
        "config_path": cfg.get("_config_path"),
        "compare": [str(path_a), str(path_b)],
        "run_dir": str(run_dir),
        "seed": seed,
        "out_prefix": out_prefix,
        "bootstrap_iterations": iterations,
        "alpha": alpha,
        "composite_score_weights": weights,
        "composite_step_count_error_scale": scale,
    }

    note_lines = [
        f"- System a: `{path_a}`",
        f"- System b: `{path_b}`",
        f"- Aligned items: {n} (same evaluation set in both files)",
        f"- Paired bootstrap: {iterations} resamples, seed {seed}, "
        f"{100 * (1 - alpha):.0f}% percentile CI of (a - b)",
        "",
        "| statistic | a | b | a - b | CI | test | significant |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, r in bootstrap.items():
        note_lines.append(
            f"| {name} | {r['system_a']:.4f} | {r['system_b']:.4f} | {r['difference']:+.4f} | "
            f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] | bootstrap | "
            f"{'yes' if r['significant'] else 'no'} |"
        )
    for name, r in mcnemar.items():
        note_lines.append(
            f"| {name} | {r['system_a_rate']:.4f} | {r['system_b_rate']:.4f} | "
            f"{r['difference']:+.4f} | p={r['p_value']:.4g} (b={r['correct_only_in_a']}, "
            f"c={r['correct_only_in_b']}) | McNemar | "
            f"{'yes' if r['significant'] else 'no'} |"
        )
    note_lines.append("")
    note_lines.append(
        "No correction for multiple comparisons is applied to these six tests."
    )

    write_run_artifacts(
        run_dir,
        config_snapshot=snapshot,
        metrics=metrics,
        note_title="MuSiQue decomposition comparison (paired)",
        note_lines=note_lines,
        prefix=f"{out_prefix}_",
    )

    print(f"Compared {n} aligned items ({iterations} bootstrap resamples, seed {seed})")
    for line in note_lines[5:]:
        print(line)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="musique_eval.json", help="Config (default: configs/musique_eval.json)")
    p.add_argument("--predictions", type=Path, default=None, help="Predicted decompositions JSON (list).")
    p.add_argument(
        "--compare",
        type=Path,
        nargs=2,
        default=None,
        metavar=("A_PER_ITEM", "B_PER_ITEM"),
        help="Paired significance between two '<prefix>_per_item.json' files (same eval set).",
    )
    p.add_argument("--gold", type=Path, default=None, help="Override the gold JSONL from config.")
    p.add_argument("--run-dir", type=Path, default=None, help="Override the run directory from config.")
    p.add_argument("--seed", type=int, default=None, help="Override the config seed.")
    p.add_argument("--limit", type=int, default=None, help="Override the config row cap.")
    p.add_argument("--out-prefix", default=None, help="Override the artifact filename prefix.")
    args = p.parse_args()
    if (args.predictions is None) == (args.compare is None):
        p.error("pass exactly one of --predictions (scoring) or --compare A B (comparison)")
    return args


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    paths_cfg = load_paths(require(cfg, "paths_config"))

    seed = args.seed if args.seed is not None else int(require(cfg, "seed"))
    seeded = set_global_seed(seed)
    weights = require(cfg, "composite_score_weights")
    scale = float(require(cfg, "composite_step_count_error_scale"))

    if args.compare is not None:
        _compare(args, cfg, paths_cfg, seed, seeded, weights, scale)
        return

    limit = args.limit if args.limit is not None else require(cfg, "limit")
    out_prefix = args.out_prefix or require(cfg, "out_prefix")

    gold_path = (
        args.gold
        if args.gold is not None
        else resolve_path(
            require(paths_cfg, "datasets." + require(cfg, "gold_key")),
            Path(paths_cfg["data_root_resolved"]),
        )
    )
    run_dir = args.run_dir if args.run_dir is not None else runs_path(paths_cfg, require(cfg, "run_subdir"))
    run_dir.mkdir(parents=True, exist_ok=True)

    if not gold_path.exists():
        raise SystemExit(
            f"gold file not found: {gold_path} (set datasets.{require(cfg, 'gold_key')} "
            "or data_root in configs/paths.json)"
        )

    gold_by_question = _load_gold(gold_path)
    preds = _load_predictions(args.predictions)
    if limit is not None:
        preds = preds[: int(limit)]

    rows, missing_gold = _build_eval_rows(preds, gold_by_question)
    if not rows:
        raise SystemExit("No evaluable rows (no matched questions between predictions and gold).")

    per_item: list[dict[str, Any]] = []
    per_hop_rows: dict[int, list[dict[str, Any]]] = {}
    gold_hop_distribution: dict[str, int] = {}

    for row in rows:
        step_p, step_r, step_f1 = _step_prf(row.pred_steps, row.gold_steps)
        rouge_lp, rouge_lr, rouge_lf = _rouge_l(_join_steps(row.pred_steps), _join_steps(row.gold_steps))
        ref_rate, ref_ok, ref_total = _reference_validity(row.pred_steps)
        pred_hops = len(row.pred_steps)
        gold_hops = row.gold_hop_count

        item_row = {
            "item_id": row.item_id,
            "question": row.question,
            "pred_steps": row.pred_steps,
            "gold_steps": row.gold_steps,
            "pred_step_count": len(row.pred_steps),
            "gold_step_count": len(row.gold_steps),
            "exact_match": _exact_decomposition_match(row.pred_steps, row.gold_steps),
            "step_precision": step_p,
            "step_recall": step_r,
            "step_f1": step_f1,
            "ordered_step_accuracy": _ordered_step_accuracy(row.pred_steps, row.gold_steps),
            "rouge_l_precision": rouge_lp,
            "rouge_l_recall": rouge_lr,
            "rouge_l_f1": rouge_lf,
            "reference_validity_rate": ref_rate,
            "reference_valid_count": ref_ok,
            "reference_total_count": ref_total,
            "step_count_signed_error": len(row.pred_steps) - len(row.gold_steps),
            "step_count_abs_error": abs(len(row.pred_steps) - len(row.gold_steps)),
            "predicted_hop_count": pred_hops,
            "gold_hop_count": gold_hops,
            "hop_count_abs_error": abs(pred_hops - gold_hops),
            "hop_count_exact_match": 1.0 if pred_hops == gold_hops else 0.0,
        }
        per_item.append(item_row)
        per_hop_rows.setdefault(gold_hops, []).append(item_row)
        gold_hop_distribution[str(gold_hops)] = gold_hop_distribution.get(str(gold_hops), 0) + 1

    overall = _aggregate(per_item)
    n = len(per_item)

    metrics = {
        "script": Path(__file__).name,
        "created_utc": now_iso(),
        "seed": seed,
        "seeded": seeded,
        "predictions_path": str(args.predictions.resolve()),
        "gold_path": str(gold_path.resolve()),
        "total_predictions_input": len(preds),
        "total_evaluated": n,
        "missing_gold_count": missing_gold,
        **{k: v for k, v in overall.items() if k != "num_rows"},
        "gold_hop_distribution": gold_hop_distribution,
        "per_gold_hop_metrics": {
            str(h): _aggregate(items) for h, items in sorted(per_hop_rows.items(), key=lambda kv: kv[0])
        },
        "composite_score": _composite_score(overall, weights, scale),
        "composite_score_weights": weights,
        "composite_step_count_error_scale": scale,
        "metric_definitions": METRIC_DEFINITIONS,
    }

    per_item_path = run_dir / f"{out_prefix}_per_item.json"
    per_item_path.write_text(
        json.dumps(per_item, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    snapshot = {
        "script": Path(__file__).name,
        "created_utc": now_iso(),
        "config_path": cfg.get("_config_path"),
        "predictions": str(args.predictions),
        "gold": str(gold_path),
        "run_dir": str(run_dir),
        "seed": seed,
        "limit": limit,
        "out_prefix": out_prefix,
        "composite_score_weights": weights,
        "composite_step_count_error_scale": scale,
    }
    write_run_artifacts(
        run_dir,
        config_snapshot=snapshot,
        metrics=metrics,
        note_title="MuSiQue decomposition evaluation",
        note_lines=[
            f"- Predictions: `{args.predictions}`",
            f"- Gold: `{gold_path}`",
            f"- Evaluated rows: {n} (missing gold matches: {missing_gold})",
            f"- Exact match: {metrics['exact_match_rate']:.4f}",
            f"- Step F1 (macro): {metrics['step_f1_macro']:.4f}",
            f"- Ordered step accuracy: {metrics['ordered_step_accuracy_macro']:.4f}",
            f"- ROUGE-L F1 (macro): {metrics['rouge_l_f1_macro']:.4f}",
            f"- Step count: MAE {metrics['step_count_mae']:.4f}, mean signed "
            f"{metrics['mean_signed_step_count_error']:+.4f} "
            f"(over {metrics['over_decomposition_rate']:.4f} / "
            f"under {metrics['under_decomposition_rate']:.4f})",
            f"- Composite score: {metrics['composite_score']:.4f}",
            f"- Per-item: `{per_item_path}`",
        ],
        prefix=f"{out_prefix}_",
    )

    print(f"Evaluated {n} rows")
    print(f"Wrote {per_item_path}")


if __name__ == "__main__":
    main()
