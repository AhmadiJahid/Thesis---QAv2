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
  (an object: the composite-score weights the rows were scored under, plus ``items``)
  plus the standard config/metrics/notes trail.
- ``--compare A_per_item.json B_per_item.json``: paired significance between two runs
  **on the same evaluation set** (bootstrap CIs + McNemar, plus an additive paired
  t-test). It aligns rows by item id and refuses to run when the two sets differ, or
  when the two files were scored under different composite weights, because neither is a
  comparison (CLAUDE.md, evidence discipline).
  With ``--v1-per-item`` the two inputs are read as **v1 prior-work artifacts** (the
  bare-list format that predates ``musique_decomposition_per_item/1``) under ADR 0020;
  the comparison output then records that its inputs carry no commit SHA and are not v2
  evidence.

Ported from v1 ``scripts/musique_decompositions_evaluator.py``. Adapted for v2: the
gold path, run directory, seed, limit, the composite-score weights and the paired
comparison parameters come from ``configs/musique_eval.json``; the run writes the
standard config/metrics/notes trail.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats as scipy_stats

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from run_artifacts import now_iso, write_run_artifacts  # noqa: E402
from run_config import load_config, load_paths, require, resolve_path, runs_path  # noqa: E402
from seeding import set_global_seed  # noqa: E402
from step_lines import split_step_lines  # noqa: E402

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
        "is removed per line (src/step_lines.py::split_step_lines, shared with the "
        "decomposer's step-line budget so both count the same steps); a list "
        "decomposition takes each string, or each item's 'question' field"
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
    "step_count_exact_rate": (
        "fraction of rows with len(pred steps) == len(gold steps), i.e. exactly "
        "1 - over_decomposition_rate - under_decomposition_rate. NOT the same as "
        "hop_count_exact_match_rate, which compares against the gold 'hop_count' field "
        "instead of the gold step count (see gold_step_count_vs_hop_count)"
    ),
    "gold_step_count_vs_hop_count": (
        "two gold denominators exist: len(gold steps) for the directional step-count "
        "family, and the gold 'hop_count' field for the hop-count family. The gold loader "
        "asserts they agree on every row that carries a positive 'hop_count' and aborts "
        "naming the offending ids, so the two families cannot silently diverge"
    ),
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
    "mcnemar_min_attainable_p_value": (
        "the smallest p this test could have produced given its discordant count m = b + c, "
        "reached when all discordant pairs favour one system: min(1, 2 * 0.5**m), and 1.0 "
        "when m = 0. When it is >= alpha the test cannot reject at alpha no matter how "
        "one-sided the result is, so 'significant: false' says nothing about the effect"
    ),
    "bootstrap_has_no_p_value": (
        "bootstrap rows carry no p-value and therefore no minimum attainable p: their "
        "decision is whether the percentile interval excludes 0. Their resolution is "
        "instead bounded by n (recorded per row as 'n')"
    ),
    "paired_t_test": (
        "two-sided paired t-test (scipy.stats.ttest_rel) over the per-item differences "
        "system_a minus system_b — the same pairing and the same items the bootstrap "
        "resamples. Reports the t statistic, degrees of freedom (n - 1) and the p-value. It "
        "is ADDITIVE (ADR 0009 as amended by ADR 0017, issue #30): the bootstrap CIs and "
        "McNemar remain the headline protocol, and no verdict here replaces one of theirs"
    ),
    "t_test_significance": "significant = p_value < alpha",
    "t_test_statistics_covered": (
        "every compared metric that has a per-item value: the bootstrap statistics except "
        "composite_score, plus the two McNemar metrics. composite_score has no per-item "
        "value (its reference term is a micro rate and its step-count term a MAE), so no "
        "paired difference exists to t-test and only its bootstrap CI is reported"
    ),
    "t_test_degenerate_rows": (
        "a row carries t_statistic = null, p_value = null, significant = false and a "
        "'degenerate' reason when the t statistic is undefined: n < 2, or a zero standard "
        "deviation of the per-item differences (every item differs by exactly the same "
        "amount, including the all-zeros case of comparing a file with itself). No "
        "significance claim is made from such a row; its bootstrap CI still is"
    ),
    "t_test_normality_caveat": (
        "the compared per-item scores are bounded and often exactly 0 or 1, so the "
        "normality-of-differences assumption is doubtful — the reason ADR 0009 chose the "
        "bootstrap and McNemar as the headline. The t-test is reported because the "
        "supervisor asked for it by name (ADR 0017 item 4), next to tests that do not "
        "assume it"
    ),
    "underpowered": (
        "true when n is below paired_comparison.min_items_for_significance_claim, or (for "
        "McNemar) when min_attainable_p_value >= alpha. A 'significant' flag on an "
        "underpowered row is not evidence. It applies to the t-test rows on the same terms "
        "as the rest"
    ),
    "multiple_comparisons": (
        "the headline protocol is six tests per comparison (four bootstrap intervals + two "
        "McNemar p-values, ADR 0009); the paired t-tests are reported alongside them. The "
        "exact counts are in 'tests_reported'. None of the p-values or intervals is "
        "corrected for multiple comparisons, and the t-test rows overlap the other two "
        "families by construction (same items, same metrics) — so they are not six "
        "independent extra tests"
    ),
    "v1_format_inputs": (
        "null for normal (v2) inputs. When --v1-per-item was passed it is an object "
        "recording that both inputs were v1 prior-work per-item files (ADR 0020): their "
        "sha256 and mtime (v1 runs carry no commit SHA), the alignment used, and the caveat "
        "that the result is citable prior work and not a v2 measurement"
    ),
    "composite_score_weights_provenance": (
        "the per-item files stamp the weights and scale they were scored under; --compare "
        "refuses when the two files disagree with each other, and records "
        "config_weights_match_per_item_files when the config it recomputes the bootstrap "
        "composite with differs from them"
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


#: The step splitter now lives in ``src/step_lines.py`` so the decomposer's step-line
#: budget is counted with the same rule this evaluator scores with (issue #12 review).
#: The alias is kept because it is this module's established name for it.
_split_decomposition_text = split_step_lines


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


def _load_gold(path: Path, max_reported_mismatches: int) -> dict[str, dict[str, Any]]:
    """Load gold rows, refusing gold whose two step-count denominators disagree.

    The evaluator measures step counts against ``len(gold steps)`` and hop counts against
    the gold ``hop_count`` field. Those are two denominators for the same quantity, so a
    row where they disagree would make the directional family and the hop-count family
    describe different things without saying so. That is a broken gold file, not a metric
    result, so it aborts here naming the offending ids.
    """
    out: dict[str, dict[str, Any]] = {}
    mismatches: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        question = obj.get("question")
        if not isinstance(question, str) or not question.strip():
            continue
        steps = _decomp_to_steps(obj.get("question_decomposition"))
        hop_count = obj.get("hop_count")
        if isinstance(hop_count, int) and not isinstance(hop_count, bool) and hop_count > 0:
            if hop_count != len(steps):
                mismatches.append(
                    f"{obj.get('id', _normalize_question(question))} "
                    f"(hop_count={hop_count}, steps={len(steps)})"
                )
        else:
            # No usable field: the two denominators are the same number by construction.
            hop_count = len(steps)
        out[_normalize_question(question)] = {"steps": steps, "hop_count": hop_count}

    if mismatches:
        shown = mismatches[:max_reported_mismatches]
        more = (
            ""
            if len(mismatches) <= max_reported_mismatches
            else f"\n  ... (+{len(mismatches) - max_reported_mismatches} more)"
        )
        raise SystemExit(
            f"gold file has {len(mismatches)} row(s) whose 'hop_count' field disagrees with "
            f"len(question_decomposition): {path}\n  "
            + "\n  ".join(shown)
            + more
            + "\nThe step-count metrics use len(question_decomposition) and the hop-count "
            "metrics use 'hop_count'; with these disagreeing they would measure different "
            "things under one report. Fix the gold file (or drop the offending rows)."
        )
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
    # 0.0 rather than 1.0: with no rows there is nothing to be exact about, and the other
    # rates in this block are 0.0 for the same reason.
    "step_count_exact_rate": 0.0,
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
        # Identical to 1 - over - under; counted directly so the three rates come from one
        # pass over the same signed errors.
        "step_count_exact_rate": sum(1 for s in signed if s == 0) / m,
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

#: Statistics a paired t-test is reported on, added alongside the two families above per
#: ADR 0017 item 4 / issue #30 (never as a replacement). It is every compared metric that
#: HAS a per-item value: the bootstrap statistics minus ``composite_score``, whose
#: reference term is a micro rate and step-count term a MAE, so no per-item difference to
#: t-test exists, plus the two binary McNemar metrics (which do have one).
T_TEST_STATISTICS = (
    tuple(name for name in BOOTSTRAP_STATISTICS if name != "composite_score") + MCNEMAR_STATISTICS
)

#: Top-level shape of ``<prefix>_per_item.json``. It is an object rather than a bare list
#: because the composite-score weights the rows were scored under have to travel with them:
#: --compare recomputes the composite, and weights that differ from the ones a file was
#: produced with would otherwise be undetectable.
PER_ITEM_SCHEMA = "musique_decomposition_per_item/1"

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


def _load_per_item(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Load a ``*_per_item.json`` file into ({item_id: row}, header).

    Fails loudly on duplicate ids, on rows missing a compared field, and on the legacy
    bare-list format, which carries no record of the weights it was scored under.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        raise SystemExit(
            f"{path}: this is the legacy bare-list per-item format, which does not record "
            f"the composite-score weights its rows were scored under. --compare recomputes "
            f"the composite, so it needs them; re-run the evaluation to regenerate the file.\n"
            f"If both files are v1 prior-work artifacts (ADR 0020), pass --v1-per-item to "
            f"read them explicitly as v1 format — the comparison output then records that "
            f"its inputs are v1 and carry no commit SHA."
        )
    if not isinstance(payload, dict):
        raise SystemExit(f"--compare expects a per-item JSON object: {path}")
    raw = payload.get("items")
    if not isinstance(raw, list):
        raise SystemExit(f"{path}: missing an 'items' list (schema {PER_ITEM_SCHEMA})")
    header = {
        k: v for k, v in payload.items()
        if k in ("schema", "composite_score_weights", "composite_step_count_error_scale")
    }
    for key in ("composite_score_weights", "composite_step_count_error_scale"):
        if key not in header:
            raise SystemExit(
                f"{path}: missing {key!r}. --compare takes the "
                f"'<prefix>_per_item.json' files written by this script; re-run the "
                f"evaluation to regenerate them."
            )
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
    return by_id, header


# --------------------------------------------------------------------------------------
# v1 prior-work per-item files (--v1-per-item; ADR 0020)
# --------------------------------------------------------------------------------------

#: How the rows of two v1 files are paired. v1 wrote a bare list with no ``item_id``, so
#: the key has to be reconstructed, and which key was used changes the bootstrap CI in the
#: 3rd-4th decimal — hence ADR 0020 condition 3 (alignment must be stated) and hence this
#: is recorded in the comparison output rather than left implicit.
#:
#: - ``normalized_question``: key = the normalized question text, rows processed in sorted
#:   order of it. This is the alignment of
#:   ``docs/analysis/2026-08-20-v1-masking-and-retrieval-significance.md`` §3 (Task A).
#: - ``position``: key = the row's position in the file, so the two files are paired in
#:   file order. Needed for v1 artifacts whose question texts are not unique (that note's
#:   Task B has one question appearing twice), and only sound because every paired row's
#:   question and gold are then asserted equal.
V1_ALIGNMENTS = ("normalized_question", "position")

#: Bound on ``position`` alignment: ids are zero-padded to this width so their sorted order
#: is their file order. A v1 per-item file larger than this does not exist (the largest is
#: 750 rows) and would silently mis-order, so it is refused instead.
_V1_POSITION_ID_WIDTH = 6

#: The per-item fields a v1 file must carry. v1 wrote every one of them except ``item_id``,
#: which it had no concept of.
_REQUIRED_V1_PER_ITEM_FIELDS = tuple(f for f in _REQUIRED_PER_ITEM_FIELDS if f != "item_id")

#: Leads every v1 comparison's run note and travels in its metrics JSON, so a consumer of
#: the output cannot mistake it for a v2 measurement (ADR 0020 conditions 5 and 2).
V1_PRIOR_WORK_CAVEAT = (
    "PRIOR WORK, NOT A v2 MEASUREMENT: both inputs are v1-format per-item files produced "
    "before this repo's rules existed. v1's runs/ is untracked, so these inputs carry NO "
    "commit SHA — they are pinned here only by sha256 and mtime. Gate 2 (committed code + "
    "committed config + fixed seed) is not satisfied by them and there is no "
    "experiments/log.md entry, because nothing was run to produce them. The statistics "
    "below are computed by committed v2 code, but they are statistics ABOUT v1 numbers and "
    "inherit v1's provenance gap (ADR 0005, ADR 0020)."
)


def _file_provenance(path: Path) -> dict[str, Any]:
    """Pin an input by content and mtime — ADR 0020 condition 2 (v1 has no commit SHA)."""
    data = path.read_bytes()
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": stat.st_size,
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
    }


def _load_v1_per_item(
    path: Path,
    alignment: str,
    weights: dict[str, float],
    scale: float,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Load a v1 bare-list per-item file into ({item_id: row}, header, provenance).

    The rows are v1's own scores, untouched; only the ``item_id`` this script aligns on is
    reconstructed, by ``alignment``. The header is synthesized from the CURRENT config
    because v1 stamped no weights — recorded as such in the output, never as if the file
    had stated them.
    """
    if alignment not in V1_ALIGNMENTS:
        raise SystemExit(f"--v1-alignment must be one of {list(V1_ALIGNMENTS)}, got {alignment!r}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise SystemExit(
            f"{path}: --v1-per-item expects the v1 bare-list per-item format, but this file "
            f"is a {type(payload).__name__} — it looks like a v2 "
            f"'{PER_ITEM_SCHEMA}' artifact. Drop --v1-per-item to compare v2 artifacts; "
            f"mixing a v1 file with a v2 one is not supported, because the two were scored "
            f"by different code."
        )
    if len(payload) > 10 ** _V1_POSITION_ID_WIDTH:
        raise SystemExit(
            f"{path}: {len(payload)} rows exceeds the {10 ** _V1_POSITION_ID_WIDTH}-row bound "
            f"of the v1 shim's positional ids."
        )

    by_id: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for i, obj in enumerate(payload):
        if not isinstance(obj, dict):
            raise SystemExit(f"{path}: row {i} is not an object")
        missing = [f for f in _REQUIRED_V1_PER_ITEM_FIELDS if f not in obj]
        if missing:
            raise SystemExit(
                f"{path}: row {i} is missing {missing}. --v1-per-item reads the bare-list "
                f"per-item files v1's evaluator wrote; this file is not one of them."
            )
        if alignment == "normalized_question":
            question = obj.get("question")
            if not isinstance(question, str) or not question.strip():
                raise SystemExit(
                    f"{path}: row {i} has no 'question', so it cannot be aligned by "
                    f"normalized question text. Use --v1-alignment position."
                )
            item_id = _normalize_question(question)
        else:
            item_id = f"row_{i:0{_V1_POSITION_ID_WIDTH}d}"
        if item_id in by_id:
            duplicates.append(item_id)
        row = dict(obj)
        row["item_id"] = item_id
        by_id[item_id] = row

    if duplicates:
        raise SystemExit(
            f"{path}: {len(duplicates)} duplicate alignment key(s) under --v1-alignment "
            f"{alignment}: {sorted(set(duplicates))[:5]}\n"
            "Two rows with the same key cannot be paired unambiguously. If the duplicates "
            "are genuinely duplicated evaluation items, use --v1-alignment position (which "
            "pairs the two files row by row and asserts each pair is the same item)."
        )

    header = {
        # No 'schema': a v1 file carries none, and inventing one would read as if it did.
        "composite_score_weights": weights,
        "composite_step_count_error_scale": scale,
    }
    provenance = {**_file_provenance(path), "rows": len(payload)}
    return by_id, header, provenance


def _assert_v1_pairs_are_the_same_item(
    ids: list[str],
    rows_a: list[dict[str, Any]],
    rows_b: list[dict[str, Any]],
    path_a: Path,
    path_b: Path,
    alignment: str,
    max_reported: int,
) -> dict[str, Any]:
    """Check that each paired v1 row really is the same evaluation item on both sides.

    v1 files have no ids, so the pairing rests on a reconstructed key. Under ``position``
    that key says nothing about the item, so the question text and the gold decomposition
    are the evidence that row i of one file and row i of the other are the same question —
    the check the analysis note performed by hand (§3) and the reason the positional
    alignment is safe there. Fields absent on both sides are skipped, not assumed equal.
    """
    checked = {"question": 0, "gold_steps": 0}
    mismatches: list[str] = []
    for item_id, a, b in zip(ids, rows_a, rows_b):
        if "question" in a or "question" in b:
            qa, qb = a.get("question"), b.get("question")
            same = (
                isinstance(qa, str)
                and isinstance(qb, str)
                and _normalize_question(qa) == _normalize_question(qb)
            )
            if same:
                checked["question"] += 1
            else:
                mismatches.append(f"{item_id} (question differs)")
        if "gold_steps" in a or "gold_steps" in b:
            if a.get("gold_steps") == b.get("gold_steps"):
                checked["gold_steps"] += 1
            else:
                mismatches.append(f"{item_id} (gold_steps differ)")
    if mismatches:
        shown = mismatches[:max_reported]
        more = "" if len(mismatches) <= max_reported else f" ... (+{len(mismatches) - max_reported} more)"
        raise SystemExit(
            f"--v1-per-item --v1-alignment {alignment}: {len(mismatches)} paired row(s) are "
            f"not the same evaluation item on both sides.\n"
            f"  a: {path_a}\n  b: {path_b}\n  " + "\n  ".join(shown) + more + "\n"
            "Pairing rows that hold different questions is not a paired comparison. Check "
            "that the two files were scored on the same evaluation set in the same order."
        )
    return {"alignment": alignment, "pairs": len(ids), "fields_verified_equal": checked}


def _refuse_writing_into_prior_work(run_dir: Path, read_only_root: str) -> None:
    """Refuse an output directory inside the read-only prior-work repo (ADR 0020 cond. 1).

    A source-level guard rather than a convention (ADR 0016): the v1 inputs live in that
    tree, so ``--run-dir`` next to them is an easy mistake, and it would write into a repo
    this project treats as read-only.
    """
    root = Path(read_only_root).expanduser()
    resolved = run_dir.resolve()
    if resolved == root.resolve() or resolved.is_relative_to(root.resolve()):
        raise SystemExit(
            f"refusing to write into the read-only prior-work repo: {resolved}\n"
            f"paired_comparison.v1_compat.read_only_prior_work_root = {root} is read-only "
            f"(ADR 0020 condition 1). Point --run-dir somewhere under this repo's runs/ "
            f"instead."
        )


def _require_matching_weights(
    header_a: dict[str, Any],
    header_b: dict[str, Any],
    path_a: Path,
    path_b: Path,
) -> None:
    """Abort when the two files were scored under different composite weights.

    Same reasoning as the id-mismatch refusal: two composites built from different weights
    are two different quantities, so differencing them is not a comparison.
    """
    keys = ("composite_score_weights", "composite_step_count_error_scale")
    differing = [k for k in keys if header_a.get(k) != header_b.get(k)]
    if not differing:
        return
    lines = [
        "--compare requires both files to have been scored under the SAME composite-score "
        "weights; they differ.",
    ]
    for key in differing:
        lines.append(f"  {key}:")
        lines.append(f"    a ({path_a}): {json.dumps(header_a.get(key), sort_keys=True)}")
        lines.append(f"    b ({path_b}): {json.dumps(header_b.get(key), sort_keys=True)}")
    lines.append(
        "A composite computed with different weights is a different quantity, so the "
        "difference between them is not a comparison. Re-score both runs with one config."
    )
    raise SystemExit("\n".join(lines))


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
    chunk_size: int,
    underpowered: bool,
) -> dict[str, dict[str, float]]:
    """Paired percentile bootstrap, resampled ``chunk_size`` draws at a time.

    Peak memory is O(chunk_size * n) instead of O(iterations * n): only the per-statistic
    difference vectors (length ``iterations``) are kept. Chunking does not change the
    numbers — ``rng.integers`` consumes one draw per index in row-major order, so drawing
    ``(k, n)`` then ``(iterations - k, n)`` is the same stream as one ``(iterations, n)``
    draw. ``TestBootstrapChunking`` pins that invariance across chunk sizes.
    """
    if chunk_size <= 0:
        raise SystemExit(
            f"paired_comparison.bootstrap_chunk_size must be a positive integer, got {chunk_size}"
        )

    identity = np.arange(n)[None, :]
    point_a = _statistics_for(arrays_a, identity, weights, scale)
    point_b = _statistics_for(arrays_b, identity, weights, scale)

    rng = np.random.default_rng(seed)
    diffs = {name: np.empty(iterations, dtype=float) for name in BOOTSTRAP_STATISTICS}
    done = 0
    while done < iterations:
        take = min(chunk_size, iterations - done)
        index = rng.integers(0, n, size=(take, n))
        chunk_a = _statistics_for(arrays_a, index, weights, scale)
        chunk_b = _statistics_for(arrays_b, index, weights, scale)
        for name in BOOTSTRAP_STATISTICS:
            diffs[name][done : done + take] = chunk_a[name] - chunk_b[name]
        done += take

    lo_pct = 100.0 * (alpha / 2.0)
    hi_pct = 100.0 * (1.0 - alpha / 2.0)

    out: dict[str, dict[str, float]] = {}
    for name in BOOTSTRAP_STATISTICS:
        ci_low, ci_high = (float(x) for x in np.percentile(diffs[name], [lo_pct, hi_pct]))
        out[name] = {
            "system_a": float(point_a[name][0]),
            "system_b": float(point_b[name][0]),
            "difference": float(point_a[name][0] - point_b[name][0]),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "significant": bool(ci_low > 0.0 or ci_high < 0.0),
            "n": n,
            "underpowered": underpowered,
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
    underpowered: bool,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for name in MCNEMAR_STATISTICS:
        a_vals = [float(r[name]) >= 1.0 for r in rows_a]
        b_vals = [float(r[name]) >= 1.0 for r in rows_b]
        only_a = sum(1 for x, y in zip(a_vals, b_vals) if x and not y)
        only_b = sum(1 for x, y in zip(a_vals, b_vals) if y and not x)
        p_value = _mcnemar_exact_p(only_a, only_b)
        # The most one-sided outcome this many discordant pairs could have produced. When
        # it is >= alpha, "not significant" is a statement about n, not about the systems.
        min_p = _mcnemar_exact_p(only_a + only_b, 0)
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
            "n": n,
            "min_attainable_p_value": min_p,
            "min_attainable_p_reaches_alpha": bool(min_p < alpha),
            "underpowered": bool(underpowered or min_p >= alpha),
        }
    return out


def _paired_t_test_row(
    values_a: np.ndarray,
    values_b: np.ndarray,
    alpha: float,
    underpowered: bool,
) -> dict[str, Any]:
    """One two-sided paired t-test row over the per-item differences ``a - b``.

    ``scipy.stats.ttest_rel`` computes t and p (scipy is a pinned dependency of this repo).
    The degenerate cases are handled here rather than passed through, because scipy returns
    NaN or an infinite t for them and neither is representable in JSON: a row with an
    undefined t makes no significance claim and says why in ``degenerate``.
    """
    differences = values_a - values_b
    n = int(differences.size)
    row: dict[str, Any] = {
        "system_a": float(values_a.mean()) if n else 0.0,
        "system_b": float(values_b.mean()) if n else 0.0,
        "difference": float(differences.mean()) if n else 0.0,
        "n": n,
        "degrees_of_freedom": n - 1,
        "underpowered": underpowered,
    }

    def degenerate(reason: str) -> dict[str, Any]:
        return {
            **row,
            "t_statistic": None,
            "p_value": None,
            "significant": False,
            "degenerate": reason,
        }

    if n < 2:
        return degenerate(f"n = {n}: a paired t-test needs at least 2 items")
    if float(differences.std(ddof=1)) == 0.0:
        return degenerate(
            "the standard deviation of the per-item differences is 0 (every item differs by "
            "exactly the same amount, e.g. a file compared with itself), so the t statistic "
            "is undefined or infinite; the bootstrap CI for this metric still applies"
        )

    result = scipy_stats.ttest_rel(values_a, values_b)
    p_value = float(result.pvalue)
    return {
        **row,
        "t_statistic": float(result.statistic),
        "p_value": p_value,
        "significant": bool(p_value < alpha),
        "degenerate": None,
    }


def _paired_t_test(
    rows_a: list[dict[str, Any]],
    rows_b: list[dict[str, Any]],
    alpha: float,
    underpowered: bool,
) -> dict[str, dict[str, Any]]:
    """Paired t-tests on the aligned rows, one per :data:`T_TEST_STATISTICS` metric."""
    out: dict[str, dict[str, Any]] = {}
    for name in T_TEST_STATISTICS:
        values_a = np.array([float(r[name]) for r in rows_a], dtype=float)
        values_b = np.array([float(r[name]) for r in rows_b], dtype=float)
        out[name] = _paired_t_test_row(values_a, values_b, alpha, underpowered)
    return out


def _significance_floor(n: int, min_items: int) -> dict[str, Any]:
    """Record how much evidence the comparison actually has, next to its verdicts."""
    below = n < min_items
    return {
        "num_items": n,
        "min_items_for_significance_claim": min_items,
        "below_min_items": below,
        "warning": (
            f"n = {n} is below min_items_for_significance_claim = {min_items}: read every "
            f"'significant' flag in this file as underpowered. It is a reporting guard from "
            f"configs/musique_eval.json, not a statistical standard."
            if below
            else None
        ),
    }


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
    chunk_size = int(require(compare_cfg, "bootstrap_chunk_size"))
    alpha = float(require(compare_cfg, "alpha"))
    min_items = int(require(compare_cfg, "min_items_for_significance_claim"))
    max_reported = int(require(compare_cfg, "max_reported_id_mismatches"))
    out_prefix = args.out_prefix or require(compare_cfg, "out_prefix")
    v1_cfg = require(compare_cfg, "v1_compat")

    run_dir = args.run_dir if args.run_dir is not None else runs_path(paths_cfg, require(cfg, "run_subdir"))

    path_a, path_b = args.compare
    v1_inputs: dict[str, Any] | None = None
    if args.v1_per_item:
        # Checked before any work: the v1 inputs live in a tree this project treats as
        # read-only, so an output directory pointing there must fail immediately.
        _refuse_writing_into_prior_work(run_dir, require(v1_cfg, "read_only_prior_work_root"))
        alignment = args.v1_alignment or require(v1_cfg, "default_alignment")
        a_by_id, header_a, prov_a = _load_v1_per_item(path_a, alignment, weights, scale)
        b_by_id, header_b, prov_b = _load_v1_per_item(path_b, alignment, weights, scale)
    else:
        a_by_id, header_a = _load_per_item(path_a)
        b_by_id, header_b = _load_per_item(path_b)
    _require_matching_weights(header_a, header_b, path_a, path_b)
    ids = _aligned_ids(a_by_id, b_by_id, path_a, path_b, max_reported)
    if not ids:
        raise SystemExit("--compare: both files are empty, nothing to compare.")

    rows_a = [a_by_id[i] for i in ids]
    rows_b = [b_by_id[i] for i in ids]
    n = len(ids)

    if args.v1_per_item:
        same_item_check = _assert_v1_pairs_are_the_same_item(
            ids, rows_a, rows_b, path_a, path_b, alignment, max_reported
        )
        v1_inputs = {
            "enabled": True,
            "prior_work_not_v2_evidence": True,
            "caveat": V1_PRIOR_WORK_CAVEAT,
            "adr": "ADR 0020 (prior-work re-analysis convention)",
            "format": (
                "v1 bare-list per-item file: the same per-item fields this script writes, "
                f"but no item_id and no stamped composite weights (predates {PER_ITEM_SCHEMA})"
            ),
            "alignment": alignment,
            "alignment_definition": (
                "rows keyed by the normalized question text (strip, lowercase, collapse "
                "whitespace) and processed in sorted order of that key"
                if alignment == "normalized_question"
                else "rows keyed by their position in the file, so the two files are paired "
                "in file order (row i with row i)"
            ),
            "same_item_check": same_item_check,
            "composite_score_weights_source": (
                "the config below — v1 per-item files stamp no weights, so the ones the "
                "bootstrap composite is recomputed with cannot be checked against them"
            ),
            "inputs": {"system_a": prov_a, "system_b": prov_b},
        }
    floor = _significance_floor(n, min_items)
    underpowered = bool(floor["below_min_items"])

    bootstrap = _paired_bootstrap(
        _statistic_arrays(rows_a),
        _statistic_arrays(rows_b),
        n=n,
        iterations=iterations,
        alpha=alpha,
        seed=seed,
        weights=weights,
        scale=scale,
        chunk_size=chunk_size,
        underpowered=underpowered,
    )
    mcnemar = _mcnemar(rows_a, rows_b, alpha, underpowered)
    t_test = _paired_t_test(rows_a, rows_b, alpha, underpowered)
    tests_reported = {
        "bootstrap": len(BOOTSTRAP_STATISTICS),
        "mcnemar": len(MCNEMAR_STATISTICS),
        "paired_t_test": len(T_TEST_STATISTICS),
        "headline_protocol": "bootstrap + McNemar (ADR 0009); the t-test is additive (ADR 0017)",
        "multiple_comparison_correction": None,
    }

    # The bootstrap composite is recomputed with THIS config's weights; the files record
    # the ones their rows were scored under. Equal above between a and b, so one check.
    # A v1 file records none, so there is nothing to match against and the answer is null
    # rather than a bool that would read as "checked and equal".
    per_item_weights = None if v1_inputs else header_a.get("composite_score_weights")
    per_item_scale = None if v1_inputs else header_a.get("composite_step_count_error_scale")
    weights_match_config = (
        None if v1_inputs else bool(per_item_weights == weights and per_item_scale == scale)
    )

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
        "bootstrap_chunk_size": chunk_size,
        "alpha": alpha,
        "confidence_level": 1.0 - alpha,
        "difference_direction": "system_a minus system_b",
        "significance_floor": floor,
        "bootstrap": bootstrap,
        "mcnemar": mcnemar,
        "t_test": t_test,
        "tests_reported": tests_reported,
        # Null for v2 inputs; an object (with the prior-work caveat) for v1 inputs.
        "v1_format_inputs": v1_inputs,
        "composite_score_weights": weights,
        "composite_step_count_error_scale": scale,
        "per_item_composite_score_weights": per_item_weights,
        "per_item_composite_step_count_error_scale": per_item_scale,
        "config_weights_match_per_item_files": weights_match_config,
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
        "bootstrap_chunk_size": chunk_size,
        "alpha": alpha,
        "min_items_for_significance_claim": min_items,
        "composite_score_weights": weights,
        "composite_step_count_error_scale": scale,
        "v1_per_item": bool(args.v1_per_item),
        "v1_alignment": (v1_inputs["alignment"] if v1_inputs else None),
    }

    note_lines: list[str] = []
    if v1_inputs:
        # ADR 0020 condition 5: the no-SHA caveat leads.
        note_lines += [
            f"- **{V1_PRIOR_WORK_CAVEAT}**",
            f"- v1 inputs pinned by content (no commit SHA exists): "
            + "; ".join(
                f"{side} `{rec['path']}` sha256 {rec['sha256'][:16]} "
                f"mtime {rec['mtime_utc']} rows {rec['rows']}"
                for side, rec in v1_inputs["inputs"].items()
            ),
            f"- v1 alignment: `{v1_inputs['alignment']}` — {v1_inputs['alignment_definition']}; "
            f"every pair verified to be the same item "
            f"({v1_inputs['same_item_check']['fields_verified_equal']}).",
        ]
    note_lines += [
        f"- System a: `{path_a}`",
        f"- System b: `{path_b}`",
        f"- Aligned items: {n} (same evaluation set in both files)",
        f"- Paired bootstrap: {iterations} resamples (chunk {chunk_size}), seed {seed}, "
        f"{100 * (1 - alpha):.0f}% percentile CI of (a - b)",
        "",
        # The interval column carries a CI for the bootstrap rows and a p-value for the
        # McNemar rows, so it is not labelled "CI".
        "| statistic | a | b | a - b | CI or p | test | significant | underpowered |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, r in bootstrap.items():
        note_lines.append(
            f"| {name} | {r['system_a']:.4f} | {r['system_b']:.4f} | {r['difference']:+.4f} | "
            f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] | bootstrap | "
            f"{'yes' if r['significant'] else 'no'} | "
            f"{'yes' if r['underpowered'] else 'no'} |"
        )
    for name, r in mcnemar.items():
        note_lines.append(
            f"| {name} | {r['system_a_rate']:.4f} | {r['system_b_rate']:.4f} | "
            f"{r['difference']:+.4f} | p={r['p_value']:.4g} (b={r['correct_only_in_a']}, "
            f"c={r['correct_only_in_b']}, min attainable p={r['min_attainable_p_value']:.4g}) | "
            f"McNemar | {'yes' if r['significant'] else 'no'} | "
            f"{'yes' if r['underpowered'] else 'no'} |"
        )
    for name, r in t_test.items():
        if r["degenerate"] is None:
            cell = (
                f"t={r['t_statistic']:+.4f} (dof={r['degrees_of_freedom']}), "
                f"p={r['p_value']:.4g}"
            )
        else:
            # The reason can be a sentence; it stays in the metrics JSON ('degenerate')
            # rather than widening every row of this table.
            cell = (
                f"t=n/a, p=n/a (dof={r['degrees_of_freedom']}; t undefined — see "
                f"'degenerate' in the metrics JSON)"
            )
        note_lines.append(
            f"| {name} | {r['system_a']:.4f} | {r['system_b']:.4f} | {r['difference']:+.4f} | "
            f"{cell} | paired t-test | {'yes' if r['significant'] else 'no'} | "
            f"{'yes' if r['underpowered'] else 'no'} |"
        )
    note_lines.append("")
    note_lines.append(
        f"- n = {n}; the reporting floor is min_items_for_significance_claim = {min_items} "
        f"(configs/musique_eval.json)."
    )
    if floor["warning"]:
        note_lines.append(f"- WARNING: {floor['warning']}")
    if weights_match_config is False:
        note_lines.append(
            "- WARNING: the bootstrap composite was recomputed with this config's weights "
            f"({json.dumps(weights, sort_keys=True)}, scale {scale}), which differ from the "
            f"weights the per-item files were scored under "
            f"({json.dumps(per_item_weights, sort_keys=True)}, scale {per_item_scale})."
        )
    if weights_match_config is None:
        note_lines.append(
            "- NOTE: v1 per-item files stamp no composite weights, so the ones the bootstrap "
            f"composite was recomputed with ({json.dumps(weights, sort_keys=True)}, scale "
            f"{scale}) could not be checked against them; they are this config's."
        )
    note_lines.append(
        f"- The headline protocol is {tests_reported['bootstrap']} bootstrap intervals + "
        f"{tests_reported['mcnemar']} McNemar p-values (ADR 0009); "
        f"{tests_reported['paired_t_test']} paired t-tests are reported alongside them "
        f"(ADR 0017 item 4, issue #30). No correction for multiple comparisons is applied to "
        "any of them, and the t-test rows re-test the same metrics on the same items rather "
        "than adding independent tests."
    )
    note_lines.append(
        "- The paired t-test assumes normally distributed per-item differences, which these "
        "bounded 0/1-heavy scores do not obviously satisfy (ADR 0009); it is reported next to "
        "the bootstrap and McNemar, which do not assume it."
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
    if v1_inputs:
        # Printed first for the same reason it leads the note (ADR 0020 condition 5).
        print(f"[v1] {V1_PRIOR_WORK_CAVEAT}")
    # From the table header on: the preamble lines above it are already in the note file,
    # and the header's index moves with the (optional) v1 caveat block.
    table_start = next(i for i, line in enumerate(note_lines) if line.startswith("| statistic |"))
    for line in note_lines[table_start:]:
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
    p.add_argument(
        "--v1-per-item",
        action="store_true",
        help="Read BOTH --compare inputs as v1 prior-work per-item files (the bare-list "
        "format with no item_id, ADR 0020). Opt-in on purpose: without it a v1 file is "
        "refused, never silently read as a v2 artifact. The comparison output records that "
        "its inputs are v1 and carry no commit SHA.",
    )
    p.add_argument(
        "--v1-alignment",
        choices=V1_ALIGNMENTS,
        default=None,
        help="How the two v1 files are paired (default: "
        "paired_comparison.v1_compat.default_alignment from the config). Recorded in the "
        "output, because CI digits are alignment-dependent (ADR 0020 condition 3).",
    )
    p.add_argument("--gold", type=Path, default=None, help="Override the gold JSONL from config.")
    p.add_argument("--run-dir", type=Path, default=None, help="Override the run directory from config.")
    p.add_argument("--seed", type=int, default=None, help="Override the config seed.")
    p.add_argument("--limit", type=int, default=None, help="Override the config row cap.")
    p.add_argument("--out-prefix", default=None, help="Override the artifact filename prefix.")
    args = p.parse_args()
    if (args.predictions is None) == (args.compare is None):
        p.error("pass exactly one of --predictions (scoring) or --compare A B (comparison)")
    if args.compare is None and (args.v1_per_item or args.v1_alignment is not None):
        p.error("--v1-per-item / --v1-alignment only apply to --compare")
    if args.v1_alignment is not None and not args.v1_per_item:
        p.error("--v1-alignment requires --v1-per-item (v2 artifacts align on item_id)")
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

    gold_by_question = _load_gold(
        gold_path, int(require(cfg, "gold_validation.max_reported_mismatches"))
    )
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
    per_item_payload = {
        "schema": PER_ITEM_SCHEMA,
        "created_utc": metrics["created_utc"],
        "predictions_path": metrics["predictions_path"],
        "gold_path": metrics["gold_path"],
        # Stamped so --compare can tell whether the two files it differences were scored
        # under the same weights as each other and as the config it recomputes with.
        "composite_score_weights": weights,
        "composite_step_count_error_scale": scale,
        "items": per_item,
    }
    per_item_path.write_text(
        json.dumps(per_item_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
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
            f"under {metrics['under_decomposition_rate']:.4f} / "
            f"exact {metrics['step_count_exact_rate']:.4f})",
            f"- Composite score: {metrics['composite_score']:.4f}",
            f"- Per-item: `{per_item_path}`",
        ],
        prefix=f"{out_prefix}_",
    )

    print(f"Evaluated {n} rows")
    print(f"Wrote {per_item_path}")


if __name__ == "__main__":
    main()
