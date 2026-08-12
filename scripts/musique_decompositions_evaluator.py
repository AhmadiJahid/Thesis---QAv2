#!/usr/bin/env python3
"""
Evaluate predicted MuSiQue decompositions against gold dev decompositions.

Scoring techniques (all string-level, no model in the loop):
- Exact match (full decomposition)
- Step-level precision/recall/F1 (unordered)
- Ordered step accuracy
- Step count error
- Reference validity for [#k] chains
- ROUGE-L precision/recall/F1 (LCS-based)
- A composite score whose weights come from the config

Inputs:
- predictions: JSON list (e.g. a decomposer run's results.json) of items like
    {"question": "...", "decomposition": "..."}
- gold: JSONL with items containing
    {"question": "...", "question_decomposition": [{"question": "..."}, ...]}

Ported from v1 ``scripts/musique_decompositions_evaluator.py``. Adapted for v2: the
gold path, run directory, seed, limit and the composite-score weights come from
``configs/musique_eval.json``; the run writes the standard config/metrics/notes trail.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from run_artifacts import now_iso, write_run_artifacts  # noqa: E402
from run_config import load_config, load_paths, require, resolve_path, runs_path  # noqa: E402
from seeding import set_global_seed  # noqa: E402

_WS_RX = re.compile(r"\s+")
_PUNCT_KEEP_HASH_RX = re.compile(r"[^\w\s#]")
_REF_RX = re.compile(r"\[#(\d+)\]")


@dataclass
class EvalRow:
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
        "step_count_abs_error_mae": mean("step_count_abs_error"),
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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="musique_eval.json", help="Config (default: configs/musique_eval.json)")
    p.add_argument("--predictions", type=Path, required=True, help="Predicted decompositions JSON (list).")
    p.add_argument("--gold", type=Path, default=None, help="Override the gold JSONL from config.")
    p.add_argument("--run-dir", type=Path, default=None, help="Override the run directory from config.")
    p.add_argument("--seed", type=int, default=None, help="Override the config seed.")
    p.add_argument("--limit", type=int, default=None, help="Override the config row cap.")
    p.add_argument("--out-prefix", default=None, help="Override the artifact filename prefix.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    paths_cfg = load_paths(require(cfg, "paths_config"))

    seed = args.seed if args.seed is not None else int(require(cfg, "seed"))
    seeded = set_global_seed(seed)
    limit = args.limit if args.limit is not None else require(cfg, "limit")
    out_prefix = args.out_prefix or require(cfg, "out_prefix")
    weights = require(cfg, "composite_score_weights")
    scale = float(require(cfg, "composite_step_count_error_scale"))

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
            f"- Composite score: {metrics['composite_score']:.4f}",
            f"- Per-item: `{per_item_path}`",
        ],
        prefix=f"{out_prefix}_",
    )

    print(f"Evaluated {n} rows")
    print(f"Wrote {per_item_path}")


if __name__ == "__main__":
    main()
