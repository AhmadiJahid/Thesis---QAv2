#!/usr/bin/env python3
"""
Compare the ``kg_results`` in a run's ``analysis/success.json`` with MetaQA gold answers.

Uses Jaccard similarity (set overlap) and reports per-hop and overall statistics; an
"exact match" is a Jaccard of 1.0. Optionally writes visualisations.

Ported from v1 ``scripts/compare_answer_accuracy.py``. Adapted for v2: the gold
question/answer paths, hop list, answer separator and reports directory come from
``configs/answer_accuracy.json`` / ``configs/paths.json``; the run writes the standard
trail.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from run_artifacts import now_iso, write_run_artifacts  # noqa: E402
from run_config import (  # noqa: E402
    load_config,
    load_paths,
    require,
    resolve_path,
    runs_path,
)
from seeding import set_global_seed  # noqa: E402

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    _HAS_MATPLOTLIB = True
except ImportError:  # pragma: no cover - depends on the environment
    _HAS_MATPLOTLIB = False


def load_gold_by_hop(
    data_root: Path,
    questions_template: str,
    answers_template: str,
    hops: list[int],
    separator: str,
) -> tuple[dict[int, dict[str, set[str]]], dict[int, int]]:
    """Return (gold[hop][question] -> answer set, total questions per hop)."""
    gold: dict[int, dict[str, set[str]]] = {hop: {} for hop in hops}
    total_per_hop: dict[int, int] = {hop: 0 for hop in hops}
    for hop in hops:
        q_path = resolve_path(questions_template.format(hop=hop), data_root)
        a_path = resolve_path(answers_template.format(hop=hop), data_root)
        if not q_path.exists() or not a_path.exists():
            continue
        questions = [ln.strip() for ln in q_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        answers = [ln.strip() for ln in a_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if len(questions) != len(answers):
            raise ValueError(
                f"{q_path.name} has {len(questions)} lines but {a_path.name} has {len(answers)}"
            )
        total_per_hop[hop] = len(questions)
        for q, a in zip(questions, answers):
            gold[hop][q] = {x.strip() for x in a.split(separator) if x.strip()}
    return gold, total_per_hop


def jaccard(pred: set[str], gold: set[str]) -> float:
    """|intersection| / |union|; empty vs empty is 1.0."""
    if not pred and not gold:
        return 1.0
    union = pred | gold
    if not union:
        return 1.0
    return len(pred & gold) / len(union)


def run_analysis(
    run_dir: Path,
    *,
    gold_by_hop: dict[int, dict[str, set[str]]],
    total_per_hop: dict[int, int],
    hops: list[int],
    analysis_subdir: str,
    exact_threshold: float,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load success.json and gold; compute per-item Jaccard and aggregates."""
    analysis_dir = run_dir / analysis_subdir
    success_path = analysis_dir / "success.json"
    if not success_path.exists():
        raise FileNotFoundError(f"Not found: {success_path}")

    success_items = json.loads(success_path.read_text(encoding="utf-8"))

    def _empty() -> dict[str, Any]:
        return {"n": 0, "with_gold": 0, "jaccard_sum": 0.0, "exact": 0, "jaccards": []}

    hop_stats: dict[int, dict[str, Any]] = {hop: _empty() for hop in hops}
    per_item: list[dict[str, Any]] = []

    for it in success_items:
        question = (it.get("question") or "").strip()
        hop_count = it.get("hop_count")
        pred_set = {str(x).strip() for x in (it.get("kg_results") or []) if x and str(x).strip()}

        if hop_count not in hop_stats:
            hop_stats[hop_count] = _empty()
        hop_stats[hop_count]["n"] += 1

        gold_set = gold_by_hop.get(hop_count, {}).get(question)
        if gold_set is None:
            per_item.append(
                {
                    "question": question,
                    "hop_count": hop_count,
                    "jaccard": None,
                    "exact_match": None,
                    "gold_found": False,
                }
            )
            continue

        hop_stats[hop_count]["with_gold"] += 1
        j = jaccard(pred_set, gold_set)
        exact = j >= exact_threshold
        hop_stats[hop_count]["jaccard_sum"] += j
        hop_stats[hop_count]["jaccards"].append(j)
        if exact:
            hop_stats[hop_count]["exact"] += 1

        per_item.append(
            {
                "question": question,
                "hop_count": hop_count,
                "jaccard": round(j, 4),
                "exact_match": exact,
                "gold_found": True,
            }
        )

    total_with_gold = sum(s["with_gold"] for s in hop_stats.values())
    total_exact = sum(s["exact"] for s in hop_stats.values())
    total_jaccard_sum = sum(s["jaccard_sum"] for s in hop_stats.values())

    summary: dict[str, Any] = {
        "seed": seed,
        "run_dir": str(run_dir),
        "total_in_success": len(success_items),
        "total_with_gold": total_with_gold,
        "total_exact_match": total_exact,
        "overall_pct_exact": round(100.0 * total_exact / total_with_gold, 2) if total_with_gold else None,
        "overall_mean_jaccard": round(total_jaccard_sum / total_with_gold, 4) if total_with_gold else None,
        # The UNROUNDED sum, so a caller that needs a different denominator (the MetaQA
        # end-to-end wrapper counts compile/execute failures too) can recompute exactly
        # instead of multiplying the 4-dp mean back out (PR #42 review, nit 1). The rounded
        # mean above is unchanged: it is what this script has always reported.
        "overall_jaccard_sum": total_jaccard_sum,
        "per_hop": {},
    }

    for hop in hops:
        s = hop_stats[hop]
        wg = s["with_gold"]
        total_gold = total_per_hop.get(hop, 0)
        entry = {
            "total_gold_questions": total_gold,
            "answered_count": s["n"],
            "coverage_pct": round(100.0 * s["n"] / total_gold, 2) if total_gold else None,
            "with_gold_count": wg,
            "exact_match_count": s["exact"],
            "pct_exact": round(100.0 * s["exact"] / wg, 2) if wg else None,
            "mean_jaccard": round(s["jaccard_sum"] / wg, 4) if wg else None,
        }
        jaccards = sorted(s["jaccards"])
        if jaccards:
            mid = len(jaccards) // 2
            median_j = (
                (jaccards[mid] + jaccards[mid - 1]) / 2 if len(jaccards) % 2 == 0 else jaccards[mid]
            )
            entry["median_jaccard"] = round(median_j, 4)
        else:
            entry["median_jaccard"] = None
        summary["per_hop"][str(hop)] = entry

    return summary, per_item


def load_decomposition_pipeline_stats(analysis_dir: Path) -> dict[str, Any]:
    """Success / compile-fail / exec-fail counts and reason breakdowns for a run."""
    out: dict[str, Any] = {
        "total": 0,
        "success_count": 0,
        "compile_fail_count": 0,
        "exec_fail_count": 0,
        "compile_fail_reasons": {},
        "exec_fail_reasons": {},
    }
    success_path = analysis_dir / "success.json"
    compile_path = analysis_dir / "compile_fail.json"
    exec_path = analysis_dir / "exec_fail.json"
    if success_path.exists():
        out["success_count"] = len(json.loads(success_path.read_text(encoding="utf-8")))
    if compile_path.exists():
        items = json.loads(compile_path.read_text(encoding="utf-8"))
        out["compile_fail_count"] = len(items)
        for it in items:
            r = (it.get("error_reason") or "unknown").strip()
            out["compile_fail_reasons"][r] = out["compile_fail_reasons"].get(r, 0) + 1
    if exec_path.exists():
        items = json.loads(exec_path.read_text(encoding="utf-8"))
        out["exec_fail_count"] = len(items)
        for it in items:
            r = (it.get("error_reason") or "unknown").strip()
            # "entity_not_in_kb: 'x'" -> "entity_not_in_kb"
            if ":" in r:
                r = r.split(":", 1)[0].strip()
            out["exec_fail_reasons"][r] = out["exec_fail_reasons"].get(r, 0) + 1
    out["total"] = out["success_count"] + out["compile_fail_count"] + out["exec_fail_count"]
    return out


def generate_visualizations(
    summary: dict[str, Any],
    per_item: list[dict[str, Any]],
    report_dir: Path,
    hops: list[int],
    dpi: int,
    pipeline_stats: dict[str, Any] | None = None,
) -> list[str]:
    """Write the figures; no-op when matplotlib is unavailable."""
    if not _HAS_MATPLOTLIB:
        return []
    report_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    per_hop = summary.get("per_hop") or {}
    labels = [f"{h}-hop" for h in hops]

    coverage = [per_hop.get(str(h), {}).get("coverage_pct") or 0 for h in hops]
    pct_exact = [per_hop.get(str(h), {}).get("pct_exact") or 0 for h in hops]
    mean_jaccard = [(per_hop.get(str(h), {}).get("mean_jaccard") or 0) * 100 for h in hops]

    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width, coverage, width, label="Coverage %", color="steelblue")
    bars2 = ax.bar(x, pct_exact, width, label="Exact match %", color="seagreen")
    bars3 = ax.bar(x + width, mean_jaccard, width, label="Mean Jaccard (x100)", color="coral")
    for bars in (bars1, bars2, bars3):
        ax.bar_label(bars, fmt="%.1f")
    ax.set_ylabel("Percentage")
    ax.set_title("Answer accuracy by hop")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 105)
    fig.tight_layout()
    out = report_dir / "per_hop_metrics.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close()
    written.append(str(out))

    fig, ax = plt.subplots(figsize=(7, 4))
    jaccards_by_hop: dict[int, list[float]] = {h: [] for h in hops}
    for it in per_item:
        if it.get("jaccard") is not None and it["hop_count"] in jaccards_by_hop:
            jaccards_by_hop[it["hop_count"]].append(it["jaccard"])
    # At least one value per hop so the boxplot renders.
    data = [jaccards_by_hop[h] if jaccards_by_hop[h] else [0.0] for h in hops]
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
    ax.set_ylabel("Jaccard similarity")
    ax.set_title("Jaccard distribution by hop")
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    out = report_dir / "jaccard_by_hop.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close()
    written.append(str(out))

    total_exact = summary.get("total_exact_match") or 0
    total_with_gold = summary.get("total_with_gold") or 1
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(
        ["Exact match", "Non-exact"],
        [total_exact, total_with_gold - total_exact],
        color=["seagreen", "coral"],
    )
    ax.bar_label(bars, fmt="%d")
    ax.set_ylabel("Count")
    ax.set_title(f"Overall answer match (n={total_with_gold})")
    fig.tight_layout()
    out = report_dir / "overall_match.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close()
    written.append(str(out))

    if pipeline_stats and pipeline_stats.get("total", 0) > 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(
            ["Success", "Compile fail", "Exec fail"],
            [
                pipeline_stats.get("success_count") or 0,
                pipeline_stats.get("compile_fail_count") or 0,
                pipeline_stats.get("exec_fail_count") or 0,
            ],
            color=["seagreen", "coral", "indianred"],
        )
        ax.bar_label(bars, fmt="%d")
        ax.set_ylabel("Count")
        ax.set_title(f"Question decomposition pipeline (n={pipeline_stats['total']})")
        fig.tight_layout()
        out = report_dir / "decomposition_pipeline.png"
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        plt.close()
        written.append(str(out))

        for key, title, colour, fname in (
            ("compile_fail_reasons", "Compile / template error reasons", "coral", "compile_fail_reasons.png"),
            ("exec_fail_reasons", "Execution error reasons", "indianred", "exec_fail_reasons.png"),
        ):
            reasons = pipeline_stats.get(key) or {}
            if not reasons:
                continue
            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.barh(list(reasons.keys()), list(reasons.values()), color=colour, alpha=0.8)
            ax.bar_label(bars, fmt="%d")
            ax.set_xlabel("Count")
            ax.set_title(title)
            fig.tight_layout()
            out = report_dir / fname
            fig.savefig(out, dpi=dpi, bbox_inches="tight")
            plt.close()
            written.append(str(out))

    return written


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="answer_accuracy.json")
    p.add_argument("run_dir", type=Path, help="Decomposer run directory (holds analysis/success.json)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--details", action="store_true", help="Write per-question details JSON.")
    p.add_argument("--report", action="store_true", help="Generate visualisations.")
    p.add_argument("--reports-dir", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    paths_cfg = load_paths(require(cfg, "paths_config"))
    data_root = Path(paths_cfg["data_root_resolved"])

    seed = args.seed if args.seed is not None else int(require(cfg, "seed"))
    seeded = set_global_seed(seed)
    hops = [int(h) for h in require(cfg, "hops")]
    analysis_subdir = require(cfg, "analysis_subdir")
    run_dir = args.run_dir.resolve()

    gold_by_hop, total_per_hop = load_gold_by_hop(
        data_root,
        require(paths_cfg, "datasets." + require(cfg, "questions_template_key")),
        require(paths_cfg, "datasets." + require(cfg, "answers_template_key")),
        hops,
        require(cfg, "answer_separator"),
    )
    if not any(gold_by_hop.values()):
        raise SystemExit(
            f"no gold question/answer files found under {data_root}; "
            "set data_root in configs/paths.json"
        )

    summary, per_item = run_analysis(
        run_dir,
        gold_by_hop=gold_by_hop,
        total_per_hop=total_per_hop,
        hops=hops,
        analysis_subdir=analysis_subdir,
        exact_threshold=float(require(cfg, "exact_match_jaccard")),
        seed=seed,
    )

    print("\n" + "=" * 50)
    print("   ANSWER ACCURACY (success.json vs gold)")
    print("=" * 50)
    print(f"Run dir:       {run_dir}")
    print(f"Data root:     {data_root}")
    print(f"Total in success.json: {summary['total_in_success']}")
    print(f"With gold:     {summary['total_with_gold']}")
    print(f"Overall exact match: {summary['total_exact_match']} ({summary['overall_pct_exact']}%)")
    print(f"Overall mean Jaccard: {summary['overall_mean_jaccard']}")
    print("\nPer hop:")
    for hop in hops:
        p = summary["per_hop"][str(hop)]
        cov = f", coverage={p['coverage_pct']}% of {p['total_gold_questions']}" if p["total_gold_questions"] else ""
        print(
            f"  {hop}-hop: answered={p['answered_count']}{cov}, with_gold={p['with_gold_count']}, "
            f"exact={p['exact_match_count']} ({p['pct_exact']}%), "
            f"mean_jaccard={p['mean_jaccard']}, median_jaccard={p['median_jaccard']}"
        )
    print("=" * 50)

    analysis_dir = run_dir / analysis_subdir
    analysis_dir.mkdir(parents=True, exist_ok=True)

    figures: list[str] = []
    if args.report:
        reports_dir = args.reports_dir or runs_path(paths_cfg, require(cfg, "reports_subdir"))
        report_dir = Path(reports_dir) / run_dir.name
        if _HAS_MATPLOTLIB:
            pipeline_stats = (
                load_decomposition_pipeline_stats(analysis_dir) if analysis_dir.exists() else None
            )
            figures = generate_visualizations(
                summary,
                per_item,
                report_dir,
                hops,
                int(require(cfg, "figure_dpi")),
                pipeline_stats=pipeline_stats,
            )
            print(f"Wrote {len(figures)} visualisation(s) to {report_dir}")
        else:
            print("Skipping visualisations: matplotlib not available.")

    if args.details:
        details_path = analysis_dir / "answer_details.json"
        details_path.write_text(
            json.dumps(per_item, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"Wrote {details_path} ({len(per_item)} items)")

    metrics = {
        "script": Path(__file__).name,
        "created_utc": now_iso(),
        "seed": seed,
        "seeded": seeded,
        "data_root": str(data_root),
        "metric": "Jaccard similarity (set overlap); exact = Jaccard >= "
        f"{require(cfg, 'exact_match_jaccard')}",
        "summary": summary,
        "figures": figures,
    }
    write_run_artifacts(
        analysis_dir,
        config_snapshot={
            "script": Path(__file__).name,
            "config_path": cfg.get("_config_path"),
            "run_dir": str(run_dir),
            "data_root": str(data_root),
            "hops": hops,
            "seed": seed,
            "details": args.details,
            "report": args.report,
        },
        metrics=metrics,
        note_title="Answer accuracy (success.json vs gold)",
        note_lines=[
            f"- Run: `{run_dir}`",
            f"- Gold: `{data_root}` (per-hop question and answer files)",
            f"- Total in success: {summary['total_in_success']}, with gold: {summary['total_with_gold']}",
            f"- Overall % exact: {summary['overall_pct_exact']}%, "
            f"mean Jaccard: {summary['overall_mean_jaccard']}",
        ]
        + [
            f"- **{hop}-hop**: answered {summary['per_hop'][str(hop)]['answered_count']}, "
            f"with_gold {summary['per_hop'][str(hop)]['with_gold_count']}, "
            f"exact {summary['per_hop'][str(hop)]['exact_match_count']} "
            f"({summary['per_hop'][str(hop)]['pct_exact']}%), "
            f"mean Jaccard {summary['per_hop'][str(hop)]['mean_jaccard']}"
            for hop in hops
        ],
        prefix="answer_accuracy_",
    )


if __name__ == "__main__":
    main()
