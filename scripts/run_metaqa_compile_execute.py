#!/usr/bin/env python3
"""MetaQA end-to-end backend: compile a decomposition, execute it, score the answer set.

The MetaQA half of issue #16. ADR 0006 gives MetaQA **end-to-end evaluation only** (it has
no gold decompositions), so a MetaQA decomposition is judged by the answer set executing it
produces. Both halves of that already existed as separate scripts run by hand:

1. ``scripts/evaluate_decompositions.py`` compiles each predicted sub-question into a KG op
   and executes the chain, reporting a compile/execute rate with its error taxonomies;
2. ``scripts/compare_answer_accuracy.py`` compares the executed answer set against MetaQA's
   gold answers with Jaccard (exact match = Jaccard 1.0).

This is the **wrapper** that runs them as one command over one predictions file and writes
one metrics JSON: coverage, exact match, Jaccard, with both error taxonomies preserved
verbatim. It **imports** the compiler and the scorer — it does not reimplement either. The
step-template regexes, the relation rules, the compile-error reasons
(``missing_decomposition``, ``unsupported_template``, ``cannot_infer_relation``,
``compile_error_other``) and the execution-error categories (``entity_not_in_kb``,
``bad_reference_or_plan``, ``exec_error_other``) all stay where they are, unchanged;
``tests/test_metaqa_compile_execute.py`` pins that the wrapper reproduces the standalone
script's fixture outcomes exactly.

**This is not GRAG.** ADR 0006 routes MetaQA end-to-end evaluation through the supervisor's
GRAG system. GRAG is external and nothing about its interface exists in this repo or in the
v1 repo, so it is not wired here and is not stubbed: this path is direct execution against
the MetaQA KG built from ``kb.txt`` (``scripts/kg.py``), it says so in ``backend_label`` and
in ``backend.grag_wired: false`` in every metrics JSON it writes, and a number from it is
not a GRAG number. See ``configs/metaqa_compile_execute.json`` (``_grag_note``).

No model is loaded anywhere in this path — the compiler is regexes and the scorer is set
arithmetic — so there is no parameter count to assert and, as everywhere in this repo, no
model scores, rates or judges anything.

Two denominators are reported for exact match and Jaccard, because they answer different
questions and the difference is large:

- **over executed items with gold** — what ``compare_answer_accuracy.py`` has always
  reported: of the decompositions that compiled *and* executed, how good is the answer set.
  It excludes every compile and execution failure, so it is a conditional number.
- **over all items with gold** — every prediction whose question has a gold answer, with a
  compile or execution failure counted as the empty answer set it produced. This is the
  denominator the MuSiQue answering backend uses (a failed item there is inside the reported
  EM/F1, not excluded), so it is the one that can be read against a MuSiQue number for the
  cross-dataset comparison of issue #41. Which of the two is *the* headline MetaQA metric is
  not this script's call; both are reported with their definitions attached.

Usage::

    python scripts/run_metaqa_compile_execute.py --predictions <decomposer run>/results.json
    python scripts/run_metaqa_compile_execute.py --predictions <...> --run-dir runs/probe
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# The two halves, imported rather than reimplemented: one home for the compiler, one home
# for the Jaccard scorer.
from compare_answer_accuracy import jaccard, load_gold_by_hop, run_analysis  # noqa: E402
from evaluate_decompositions import evaluate_decomposition_rate  # noqa: E402
from kg import build_metaqa_kg  # noqa: E402
from run_artifacts import now_iso, write_run_artifacts  # noqa: E402
from run_config import load_config, load_paths, require, resolve_path, runs_path  # noqa: E402
from seeding import set_global_seed  # noqa: E402

#: Recorded in every artifact. GRAG is external and absent (see the module docstring), so
#: this is a constant of the code, not a config knob a flag could flip.
GRAG_WIRED = False
GRAG_STATUS = (
    "not wired: GRAG is the supervisor's system (ADR 0006) and no interface, endpoint, "
    "client, repository or handover for it exists in this repo or in the v1 repo "
    "(Thesis---QA) as of 2026-08-22. This run executed decompositions directly against the "
    "MetaQA KG built from kb.txt instead, which is a different measurement and is labelled "
    "as one. Obtaining GRAG is an external dependency Jahid chases with his supervisor."
)

#: The dumps ``evaluate_decomposition_rate`` writes, and what each one holds.
DUMP_FILES = ("success.json", "compile_fail.json", "exec_fail.json")


def gold_for_item(item: dict, gold_by_hop: dict[int, dict[str, set[str]]]) -> set[str] | None:
    """The gold answer set for one prediction row, or ``None`` when it has no gold.

    Same lookup key as ``compare_answer_accuracy.run_analysis``: the stripped question text
    within the row's ``hop_count``. Kept to one function so the executed and the failed rows
    are matched to gold by exactly the same rule.
    """
    question = str(item.get("question") or "").strip()
    hop_count = item.get("hop_count")
    if not isinstance(hop_count, int):
        return None
    return gold_by_hop.get(hop_count, {}).get(question)


def over_all_items_with_gold(
    analysis_dir: Path,
    *,
    gold_by_hop: dict[int, dict[str, set[str]]],
    executed_exact_matches: int,
) -> dict[str, Any]:
    """Exact match and mean Jaccard with compile/execute failures counted, not excluded.

    A failed item produced no answer set, so its contribution is ``jaccard(set(), gold)``
    — computed with the imported scorer rather than hard-coded, so an item with an empty gold
    set (which would score 1.0 by the empty-vs-empty rule) cannot be silently written off as
    a 0. Such rows are counted separately and, if any exist, the mean is reported as null
    with a note rather than as a number whose meaning is unclear.

    The failed rows are read from the run's own ``compile_fail.json`` / ``exec_fail.json``
    dumps, so "which rows this run processed" is taken from the run's output and never
    re-derived from the predictions file (the row cap would then live in two places).
    """
    failed_with_gold = 0
    failed_jaccard_sum = 0.0
    failed_with_empty_gold = 0
    for name in ("compile_fail.json", "exec_fail.json"):
        path = analysis_dir / name
        if not path.exists():
            continue
        for item in json.loads(path.read_text(encoding="utf-8")):
            gold = gold_for_item(item, gold_by_hop)
            if gold is None:
                continue
            failed_with_gold += 1
            if not gold:
                failed_with_empty_gold += 1
            failed_jaccard_sum += jaccard(set(), gold)
    return {
        "failed_items_with_gold": failed_with_gold,
        "failed_items_with_an_empty_gold_set": failed_with_empty_gold,
        "failed_items_jaccard_sum": round(failed_jaccard_sum, 4),
        "exact_match_count": executed_exact_matches,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", default="metaqa_compile_execute.json")
    p.add_argument(
        "--predictions", type=Path, required=True, help="Decomposer run's results.json"
    )
    p.add_argument("--kb", type=Path, default=None, help="Override the kb path from config.")
    p.add_argument("--max-items", type=int, default=None, help="Override the config row cap.")
    p.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Where the artifacts and the per-item dumps go (default: under the runs root).",
    )
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    cfg = load_config(args.config)
    paths_cfg = load_paths(require(cfg, "paths_config"))
    data_root = Path(paths_cfg["data_root_resolved"])

    # The two composed configs. Every methodology knob is read from them, so the wrapper
    # cannot disagree with the script it wraps about what a number means.
    kg_cfg = load_config(require(cfg, "kg_eval_config"))
    acc_cfg = load_config(require(cfg, "answer_accuracy_config"))

    seed = args.seed if args.seed is not None else int(require(cfg, "seed"))
    seeded = set_global_seed(seed)

    max_items = args.max_items if args.max_items is not None else require(kg_cfg, "max_items")
    hops = [int(h) for h in require(acc_cfg, "hops")]
    analysis_subdir = require(acc_cfg, "analysis_subdir")
    exact_threshold = float(require(acc_cfg, "exact_match_jaccard"))
    out_prefix = require(cfg, "out_prefix")
    backend_label = require(cfg, "backend_label")

    if not args.predictions.exists():
        raise SystemExit(f"predictions not found: {args.predictions}")
    kb_path = args.kb or resolve_path(
        require(paths_cfg, "datasets." + require(kg_cfg, "kb_key")), data_root
    )

    run_dir = (
        args.run_dir
        if args.run_dir is not None
        else runs_path(paths_cfg, require(cfg, "run_subdir"))
    )
    run_dir = Path(run_dir)
    analysis_dir = run_dir / analysis_subdir
    analysis_dir.mkdir(parents=True, exist_ok=True)

    print(f"Backend: {backend_label} (GRAG wired: {GRAG_WIRED})")
    print(f"Loading MetaQA KG from {kb_path} ...")
    kg = build_metaqa_kg(kb_path)

    # ---- half 1: compile + execute (the existing path, called, not reimplemented)
    print(f"Compiling and executing decompositions in {args.predictions} ...")
    rate = evaluate_decomposition_rate(
        kg,
        args.predictions,
        max_items=int(max_items) if max_items is not None else None,
        # Always written: the gold comparison below reads success.json. They hold question
        # and answer text and land under the gitignored runs root, never in git.
        output_dir=analysis_dir,
    )
    coverage = rate.as_dict()

    # ---- half 2: the executed answer sets against MetaQA gold (the existing scorer)
    gold_by_hop, total_per_hop = load_gold_by_hop(
        data_root,
        require(paths_cfg, "datasets." + require(acc_cfg, "questions_template_key")),
        require(paths_cfg, "datasets." + require(acc_cfg, "answers_template_key")),
        hops,
        require(acc_cfg, "answer_separator"),
    )
    if not any(gold_by_hop.values()):
        raise SystemExit(
            f"no MetaQA gold question/answer files found under {data_root}; the executed "
            "answer sets cannot be scored without them (check data_root in the paths config)"
        )

    summary, per_item = run_analysis(
        run_dir,
        gold_by_hop=gold_by_hop,
        total_per_hop=total_per_hop,
        hops=hops,
        analysis_subdir=analysis_subdir,
        exact_threshold=exact_threshold,
        seed=seed,
    )

    # ---- the second denominator: failures counted rather than excluded
    failures = over_all_items_with_gold(
        analysis_dir,
        gold_by_hop=gold_by_hop,
        executed_exact_matches=int(summary["total_exact_match"]),
    )
    items_with_gold = int(summary["total_with_gold"]) + failures["failed_items_with_gold"]
    executed_mean_jaccard = summary["overall_mean_jaccard"]
    if items_with_gold and executed_mean_jaccard is not None:
        overall_exact_rate = round(100.0 * summary["total_exact_match"] / items_with_gold, 2)
        jaccard_sum = executed_mean_jaccard * summary["total_with_gold"]
        overall_mean_jaccard = round(
            (jaccard_sum + failures["failed_items_jaccard_sum"]) / items_with_gold, 4
        )
    else:
        overall_exact_rate = None
        overall_mean_jaccard = None

    over_all = {
        "definition": (
            "every prediction whose question has a gold answer, with a compile or execution "
            "failure counted as the empty answer set it produced (jaccard(empty, gold), "
            "which is 0 for MetaQA's non-empty gold). This is the same convention the "
            "MuSiQue answering backend uses - a failed item is inside the reported metric, "
            "not excluded - so it is the denominator that can be read against a MuSiQue "
            "number (issue #41)."
        ),
        "items_with_gold": items_with_gold,
        "items_with_gold_executed": summary["total_with_gold"],
        "items_with_gold_not_executed": failures["failed_items_with_gold"],
        "exact_match_count": failures["exact_match_count"],
        "pct_exact": overall_exact_rate,
        "mean_jaccard": overall_mean_jaccard,
        "items_with_an_empty_gold_set": failures["failed_items_with_an_empty_gold_set"],
    }
    if failures["failed_items_with_an_empty_gold_set"]:
        over_all["mean_jaccard"] = None
        over_all["mean_jaccard_note"] = (
            f"{failures['failed_items_with_an_empty_gold_set']} failed item(s) have an EMPTY "
            "gold answer set, which the Jaccard rule scores 1.0 against an empty prediction. "
            "A mean that treats those as successes would misstate the run, so it is reported "
            "as unmeasured here; the per-hop and executed-only numbers above are unaffected."
        )

    print("\n" + "=" * 56)
    print(f"   METAQA END-TO-END ({backend_label})")
    print("=" * 56)
    print(f"Items:            {coverage['total']}")
    print(f"Compiled OK:      {coverage['compiled_ok']} ({coverage['compiled_ok_rate']:.2%})")
    print(f"Executed OK:      {coverage['executed_ok']} ({coverage['executed_ok_rate']:.2%})")
    print(f"Compile fail:     {coverage['compile_fail']} {coverage['compile_fail_reasons']}")
    print(f"Exec fail:        {coverage['exec_fail']} {coverage['exec_fail_reasons']}")
    print(
        f"Over EXECUTED items with gold ({summary['total_with_gold']}): "
        f"exact {summary['total_exact_match']} ({summary['overall_pct_exact']}%), "
        f"mean Jaccard {summary['overall_mean_jaccard']}"
    )
    print(
        f"Over ALL items with gold ({items_with_gold}): "
        f"exact {over_all['exact_match_count']} ({over_all['pct_exact']}%), "
        f"mean Jaccard {over_all['mean_jaccard']}"
    )
    for hop in hops:
        block = summary["per_hop"][str(hop)]
        print(
            f"  {hop}-hop: answered {block['answered_count']} of "
            f"{block['total_gold_questions']} gold ({block['coverage_pct']}% coverage), "
            f"with_gold {block['with_gold_count']}, exact {block['exact_match_count']} "
            f"({block['pct_exact']}%), mean Jaccard {block['mean_jaccard']}"
        )
    print("=" * 56)

    details_path = analysis_dir / "answer_details.json"
    details_path.write_text(
        json.dumps(per_item, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    metrics: dict[str, Any] = {
        "script": Path(__file__).name,
        "created_utc": now_iso(),
        "seed": seed,
        "seeded": seeded,
        "backend": {
            "label": backend_label,
            "grag_wired": GRAG_WIRED,
            "grag_status": GRAG_STATUS,
            "model_loaded": False,
            "model_note": (
                "no model is loaded in this path: the compiler is regexes over the step "
                "text and the scorer is set arithmetic, so there is no parameter count to "
                "assert and nothing is scored by a model"
            ),
        },
        "predictions_path": str(args.predictions.resolve()),
        "kb_path": str(Path(kb_path).resolve()),
        "kg_entities": len(kg.id_to_entity),
        "kg_triples": len(kg.triples),
        "data_root": str(data_root),
        "coverage": coverage,
        "compile_fail_reasons": coverage["compile_fail_reasons"],
        "exec_fail_reasons": coverage["exec_fail_reasons"],
        "answer_accuracy_over_executed_items": summary,
        "answer_accuracy_over_all_items_with_gold": over_all,
        "metric_definitions": {
            "coverage": (
                "compiled_ok_rate and executed_ok_rate over the items read from the "
                "predictions file, from scripts/evaluate_decompositions.py; the "
                "compile_fail_reasons and exec_fail_reasons breakdowns are that script's "
                "taxonomies, preserved verbatim"
            ),
            "per_hop_coverage_pct": (
                "answered items / gold questions available for that hop depth, from "
                "scripts/compare_answer_accuracy.py"
            ),
            "exact_match": f"Jaccard >= {exact_threshold} between the executed answer set "
            "and the gold answer set",
            "jaccard": "|intersection| / |union| of the two answer sets; empty vs empty is 1.0",
        },
        "composed_from": {
            "compile_execute": "scripts/evaluate_decompositions.py::evaluate_decomposition_rate",
            "gold_comparison": "scripts/compare_answer_accuracy.py::run_analysis",
            "kg": "scripts/kg.py::build_metaqa_kg",
        },
    }

    snapshot = {
        "script": Path(__file__).name,
        "created_utc": now_iso(),
        "config_path": cfg.get("_config_path"),
        "kg_eval_config_path": kg_cfg.get("_config_path"),
        "answer_accuracy_config_path": acc_cfg.get("_config_path"),
        "backend_label": backend_label,
        "grag_wired": GRAG_WIRED,
        "predictions": str(args.predictions),
        "kb": str(kb_path),
        "data_root": str(data_root),
        "run_dir": str(run_dir),
        "analysis_dir": str(analysis_dir),
        "max_items": max_items,
        "hops": hops,
        "exact_match_jaccard": exact_threshold,
        "answer_separator": require(acc_cfg, "answer_separator"),
        "seed": seed,
        "seeded": seeded,
        "seed_sources": {
            "governing": f"{cfg.get('_config_path')} (or --seed)",
            "unused_component_seeds": {
                str(kg_cfg.get("_config_path")): kg_cfg.get("seed"),
                str(acc_cfg.get("_config_path")): acc_cfg.get("seed"),
            },
            "note": "the composed scripts' functions are called directly, so their configs' "
            "seed keys do not apply to this run; they are recorded for provenance only",
        },
        "dumps_written": [str(analysis_dir / name) for name in DUMP_FILES],
    }

    write_run_artifacts(
        run_dir,
        config_snapshot=snapshot,
        metrics=metrics,
        note_title=f"MetaQA end-to-end ({backend_label})",
        note_lines=[
            f"- **Backend: `{backend_label}` — GRAG is NOT wired.** {GRAG_STATUS}",
            f"- Predictions: `{args.predictions}`",
            f"- KB: `{kb_path}` ({len(kg.id_to_entity)} entities, {len(kg.triples)} triples); "
            "no model loaded",
            f"- Coverage: {coverage['total']} item(s), compiled "
            f"{coverage['compiled_ok']} ({coverage['compiled_ok_rate']:.2%}), executed "
            f"{coverage['executed_ok']} ({coverage['executed_ok_rate']:.2%})",
            f"- Compile fail reasons: {coverage['compile_fail_reasons']}",
            f"- Exec fail reasons: {coverage['exec_fail_reasons']}",
            f"- Over EXECUTED items with gold ({summary['total_with_gold']}): exact "
            f"{summary['total_exact_match']} ({summary['overall_pct_exact']}%), mean Jaccard "
            f"{summary['overall_mean_jaccard']}",
            f"- Over ALL items with gold ({items_with_gold}, failures counted as the empty "
            f"answer set): exact {over_all['exact_match_count']} ({over_all['pct_exact']}%), "
            f"mean Jaccard {over_all['mean_jaccard']}"
            + (
                f" — {over_all['mean_jaccard_note']}"
                if over_all.get("mean_jaccard_note")
                else ""
            ),
        ]
        + [
            f"- **{hop}-hop**: answered {summary['per_hop'][str(hop)]['answered_count']} of "
            f"{summary['per_hop'][str(hop)]['total_gold_questions']} gold "
            f"({summary['per_hop'][str(hop)]['coverage_pct']}% coverage), exact "
            f"{summary['per_hop'][str(hop)]['exact_match_count']} "
            f"({summary['per_hop'][str(hop)]['pct_exact']}%), mean Jaccard "
            f"{summary['per_hop'][str(hop)]['mean_jaccard']}"
            for hop in hops
        ]
        + [f"- Per-item: `{details_path}`, dumps: `{analysis_dir}`"],
        prefix=out_prefix,
    )
    print(f"\nResults saved to: {run_dir}")


if __name__ == "__main__":
    main()
