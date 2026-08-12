#!/usr/bin/env python3
"""
Three-way similarity comparison for the router few-shot pool (hop count unknown).

  A: question NOT masked -> pool NOT masked
  B: question NOT masked -> pool MASKED
  C: question MASKED     -> pool MASKED

Hop prediction is a majority vote over the top ``majority_vote_k`` neighbours. Modes B
and C need the entity masker, so ``configs/masking.json`` and the MetaQA kb are required.

Ported from v1 ``scripts/test_similarity_router.py``. Adapted for v2: pool paths,
question files, embedding model, retrieve/vote k and the sample size come from
``configs/similarity_probe.json`` / ``configs/paths.json``; sampling is seeded; the run
writes the standard trail plus the human-readable report and neighbour dump v1 produced.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from entity_masking import build_masker_from_config  # noqa: E402
from model_size import assert_within_ceiling, load_limits  # noqa: E402
from pool_embeddings import get_router_pool_embeddings, top_k_similar_router  # noqa: E402
from run_artifacts import now_iso, run_id, write_run_artifacts  # noqa: E402
from run_config import (  # noqa: E402
    load_config,
    load_paths,
    require,
    resolve_path,
    runs_path,
)
from seeding import new_rng, set_global_seed  # noqa: E402

MODE_LABELS = {
    "A": "A (raw q, raw pool)",
    "B": "B (raw q, masked pool)",
    "C": "C (masked q, masked pool)",
}


def load_questions(template: str, data_root: Path, hops: list[int]) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for hop in hops:
        path = resolve_path(template.format(hop=hop), data_root)
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").strip().splitlines():
            q = line.strip()
            if q:
                out.append((q, hop))
    return out


def run_mode(
    items: list[dict],
    embeddings: Any,
    model: Any,
    model_id: str,
    questions: list[tuple[str, int]],
    retrieve_k: int,
    majority_k: int,
) -> list[tuple[int, bool, float, list[tuple[dict, float]]]]:
    """Per question: (majority hop, correct, top-1 similarity, neighbours)."""
    results = []
    for query, gold_hop in questions:
        similar = top_k_similar_router(
            query, items, embeddings, model, model_id=model_id, k=retrieve_k
        )
        top_for_vote = similar[:majority_k]
        hop_counts = [it["hop_count"] for it, _ in top_for_vote]
        majority_hop = max(set(hop_counts), key=hop_counts.count) if hop_counts else 0
        top_sim = similar[0][1] if similar else 0.0
        results.append((majority_hop, majority_hop == gold_hop, top_sim, similar))
    return results


def per_hop_stats(
    results: list[tuple[int, bool, float, list]],
    questions: list[tuple[str, int]],
    hops: list[int],
) -> dict[int, dict[str, Any]]:
    by_hop: dict[int, list[bool]] = {}
    for (_, gold_hop), (_, match, _, _) in zip(questions, results):
        by_hop.setdefault(gold_hop, []).append(match)
    out: dict[int, dict[str, Any]] = {}
    for hop in hops:
        lst = by_hop.get(hop, [])
        n = len(lst)
        out[hop] = {
            "correct": sum(lst),
            "total": n,
            "acc_pct": 100 * sum(lst) / n if n else 0,
        }
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="similarity_probe.json")
    p.add_argument("--no-corpus-filter", action="store_true", help="Use the full KB for masking.")
    p.add_argument("--sample-size-per-hop", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--run-dir", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    paths_cfg = load_paths(require(cfg, "paths_config"))
    limits = load_limits("model_limits.json")
    section = require(cfg, "router_probe")
    data_root = Path(paths_cfg["data_root_resolved"])

    seed = args.seed if args.seed is not None else int(require(cfg, "seed"))
    seeded = set_global_seed(seed)
    embed_key = require(section, "embed_model")
    model_id = require(cfg, f"embed_models.{embed_key}")
    hops = [int(h) for h in require(section, "hops")]
    retrieve_k = int(require(section, "retrieve_k"))
    majority_k = int(require(section, "majority_vote_k"))
    preview_rows = int(require(section, "per_question_preview_rows"))
    sample_size = (
        args.sample_size_per_hop
        if args.sample_size_per_hop is not None
        else require(section, "sample_size_per_hop")
    )

    pool_raw = resolve_path(require(paths_cfg, "repo." + require(section, "pool_raw_key")), REPO_ROOT)
    pool_masked = resolve_path(
        require(paths_cfg, "repo." + require(section, "pool_masked_key")), REPO_ROOT
    )
    cache_dir = resolve_path(
        require(paths_cfg, "datasets." + require(cfg, "cache_dir_key")), data_root
    )
    run_dir = args.run_dir or runs_path(paths_cfg, require(section, "run_subdir"))
    run_dir.mkdir(parents=True, exist_ok=True)

    for path, label in ((pool_raw, "raw pool"), (pool_masked, "masked pool")):
        if not path.exists():
            raise SystemExit(f"{label} not found: {path}")

    questions = load_questions(
        require(paths_cfg, "datasets." + require(section, "questions_template_key")), data_root, hops
    )
    if not questions:
        raise SystemExit(
            f"no questions found under {data_root}; set data_root in configs/paths.json"
        )
    if sample_size:
        rng = new_rng(seed)
        by_hop: dict[int, list[str]] = {}
        for q, h in questions:
            by_hop.setdefault(h, []).append(q)
        sampled: list[tuple[str, int]] = []
        for hop in hops:
            pool_qs = by_hop.get(hop, [])
            sampled.extend((q, hop) for q in rng.sample(pool_qs, min(int(sample_size), len(pool_qs))))
        rng.shuffle(sampled)
        questions = sampled

    mask_cfg = load_config(require(section, "masking_config"))
    kb_path = resolve_path(
        require(paths_cfg, "datasets." + require(mask_cfg, "kb_path_key")), data_root
    )
    corpus_paths = None
    if not args.no_corpus_filter:
        corpus_paths = [
            resolve_path(p, data_root) for p in require(mask_cfg, "corpus_data_paths")
        ] + [resolve_path(p, REPO_ROOT) for p in require(mask_cfg, "corpus_repo_paths")]
    mask_fn = build_masker_from_config(
        mask_cfg, kb_path=kb_path, corpus_paths=corpus_paths, corpus_root=data_root
    )

    current_run_id = run_id()
    out_base = run_dir / f"similarity_router_3way_{current_run_id}"
    lines: list[str] = []

    def log(s: str = "") -> None:
        print(s)
        lines.append(s)

    log(f"Questions: {len(questions)} total")
    log(f"Majority vote: top-{majority_k} of top-{retrieve_k}")
    log()

    mode_questions = {
        "A": list(questions),
        "B": list(questions),
        "C": [(mask_fn(q), h) for q, h in questions],
    }
    mode_pool = {"A": pool_raw, "B": pool_masked, "C": pool_masked}

    results: dict[str, list] = {}
    model = None
    size_record = None
    for mode in ("A", "B", "C"):
        log("=" * 80)
        log(f"  {MODE_LABELS[mode]}")
        log("=" * 80)
        items, emb, model = get_router_pool_embeddings(
            mode_pool[mode], cache_dir=cache_dir, model_id=model_id
        )
        if size_record is None:
            size_record = assert_within_ceiling(
                model, component="retrieval", model_id=model_id, limits=limits
            )
        results[mode] = run_mode(
            items, emb, model, model_id, mode_questions[mode], retrieve_k, majority_k
        )
        correct = sum(1 for _, m, _, _ in results[mode] if m)
        at_least_one = sum(
            1
            for (_, gold), r in zip(mode_questions[mode], results[mode])
            if any(it["hop_count"] == gold for it, _ in r[3][:majority_k])
        )
        n = len(questions)
        log(f"Accuracy: {correct}/{n} ({100 * correct / n:.1f}%)")
        log(
            f"At least one top-{majority_k} has correct hop: "
            f"{at_least_one}/{n} ({100 * at_least_one / n:.1f}%)"
        )
        log()

    n = len(questions)
    correct = {m: sum(1 for _, ok, _, _ in results[m] if ok) for m in results}
    at_least_one = {
        m: sum(
            1
            for (_, gold), r in zip(mode_questions[m], results[m])
            if any(it["hop_count"] == gold for it, _ in r[3][:majority_k])
        )
        for m in results
    }
    stats = {m: per_hop_stats(results[m], mode_questions[m], hops) for m in results}
    avg_sim = {m: (sum(r[2] for r in results[m]) / len(results[m]) if results[m] else 0.0) for m in results}

    log("=" * 80)
    log("  COMPARISON & ANALYSIS")
    log("=" * 80)
    log()
    log("Overall accuracy (majority vote):")
    for mode in ("A", "B", "C"):
        log(f"  {MODE_LABELS[mode]}: {correct[mode]}/{n} ({100 * correct[mode] / n:.1f}%)")
    log()
    log(f"At least one top-{majority_k} has correct hop:")
    log(
        "  "
        + "  ".join(
            f"{mode}: {at_least_one[mode]}/{n} ({100 * at_least_one[mode] / n:.1f}%)"
            for mode in ("A", "B", "C")
        )
    )
    log()
    log("Per-hop accuracy:")
    log(f"  {'hop':<5} {'A':<14} {'B':<14} {'C':<14}")
    log(f"  {'-' * 5} {'-' * 14} {'-' * 14} {'-' * 14}")
    for hop in hops:
        cells = []
        for mode in ("A", "B", "C"):
            s = stats[mode][hop]
            cells.append(f"{s['correct']}/{s['total']} ({s['acc_pct']:.0f}%)".ljust(14))
        log(f"  {hop:<5} " + " ".join(cells))
    log()
    best_mode = max(("A", "B", "C"), key=lambda m: correct[m])
    log(f"Best mode: {MODE_LABELS[best_mode]} ({correct[best_mode]}/{n})")
    log()
    log("Avg top-1 similarity:")
    log("  " + "  ".join(f"{mode}: {avg_sim[mode]:.3f}" for mode in ("A", "B", "C")))
    log()

    log(f"Per-question (first {preview_rows}):")
    log(f"  {'#':<4} {'gold':<5} {'A':<8} {'B':<8} {'C':<8}  Question")
    log(f"  {'-' * 4} {'-' * 5} {'-' * 8} {'-' * 8} {'-' * 8}  {'-' * 40}")
    for i in range(min(preview_rows, n)):
        q, gold = questions[i]
        cells = []
        for mode in ("A", "B", "C"):
            maj, ok, _, _ = results[mode][i]
            cells.append(f"{maj}({'y' if ok else 'n'})".ljust(8))
        qs = (q[:38] + "…") if len(q) > 38 else q
        log(f"  {i + 1:<4} {gold:<5} " + " ".join(cells) + f"  {qs}")
    log()

    report_path = out_base.with_suffix(".txt")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report: {report_path}")

    sim_path = run_dir / f"similarity_router_3way_{current_run_id}_similarities.txt"
    with sim_path.open("w", encoding="utf-8") as f:
        f.write("Per-question: query, gold hop, and the top neighbours for A/B/C.\n")
        f.write("=" * 80 + "\n\n")
        for i, (q, gold_hop) in enumerate(questions):
            f.write(f"[{i + 1}] {q}\n")
            f.write(f"    masked: {mask_fn(q)}\n")
            f.write(f"    gold_hop: {gold_hop}\n")
            for mode in ("A", "B", "C"):
                maj, _, _, similar = results[mode][i]
                f.write(f"    {MODE_LABELS[mode]} -> majority: {maj}\n")
                for j, (it, sim) in enumerate(similar[:majority_k], start=1):
                    f.write(f"      {j}. [hop={it['hop_count']}] sim={sim:.3f}  {it['question']}\n")
            f.write("\n")
    print(f"Similarities: {sim_path}")

    metrics = {
        "script": Path(__file__).name,
        "created_utc": now_iso(),
        "run_id": current_run_id,
        "seed": seed,
        "seeded": seeded,
        "embed_model": embed_key,
        "embed_model_id": model_id,
        "model_size": size_record,
        "n_questions": n,
        "retrieve_k": retrieve_k,
        "majority_vote_k": majority_k,
        "corpus_filtered_masking": not args.no_corpus_filter,
        "accuracy": {m: correct[m] / n for m in correct},
        "correct": correct,
        "at_least_one_match": at_least_one,
        "at_least_one_match_pct": {m: at_least_one[m] / n for m in at_least_one},
        "per_hop": {m: {str(h): stats[m][h] for h in hops} for m in stats},
        "avg_top1_sim": avg_sim,
        "best_mode": best_mode,
        "report_txt": str(report_path),
        "similarities_txt": str(sim_path),
    }
    write_run_artifacts(
        run_dir,
        config_snapshot={
            "script": Path(__file__).name,
            "config_path": cfg.get("_config_path"),
            "pool_raw": str(pool_raw),
            "pool_masked": str(pool_masked),
            "embed_model": embed_key,
            "retrieve_k": retrieve_k,
            "majority_vote_k": majority_k,
            "sample_size_per_hop": sample_size,
            "no_corpus_filter": args.no_corpus_filter,
            "seed": seed,
            "run_id": current_run_id,
        },
        metrics=metrics,
        note_title=f"Router similarity three-way probe - {current_run_id}",
        note_lines=[
            f"- Questions: {n}; majority vote over top-{majority_k} of top-{retrieve_k}",
            f"- Embedding model: `{model_id}`",
        ]
        + [
            f"- {MODE_LABELS[mode]}: {correct[mode]}/{n} ({100 * correct[mode] / n:.1f}%)"
            for mode in ("A", "B", "C")
        ]
        + [
            f"- Best mode: {MODE_LABELS[best_mode]}",
            f"- Report: `{report_path}`",
        ],
        prefix=f"router_probe_{current_run_id}_",
    )


if __name__ == "__main__":
    main()
