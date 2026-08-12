#!/usr/bin/env python3
"""
Check similarity between evaluation questions and their hop-specific few-shot pool.

Compares each question to the pool entries for its own hop count. Low similarity is a
sign that the pool lacks coverage for that question type.

Ported from v1 ``scripts/check_pool_coverage.py``. Adapted for v2: the pool path,
question files, embedding model, top-k window, similarity thresholds and sample size
come from ``configs/pool_coverage.json`` / ``configs/paths.json``; sampling is seeded;
the embedding model's parameter count is asserted; the run writes the standard trail.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from model_size import assert_within_ceiling, load_limits  # noqa: E402
from pool_embeddings import (  # noqa: E402
    _needs_e5_prefix,
    get_router_pool_embeddings,
    load_router_pool,
)
from run_artifacts import now_iso, write_run_artifacts  # noqa: E402
from run_config import (  # noqa: E402
    load_config,
    load_paths,
    require,
    resolve_path,
    runs_path,
)
from seeding import new_rng, set_global_seed  # noqa: E402


def load_test_questions(template: str, data_root: Path, hops: list[int]) -> list[tuple[str, str]]:
    """Load (question, hop_key) pairs from the per-hop question files."""
    out: list[tuple[str, str]] = []
    for hop in hops:
        path = resolve_path(template.format(hop=hop), data_root)
        if not path.exists():
            continue
        hop_key = f"{hop}hop"
        for line in path.read_text(encoding="utf-8").strip().splitlines():
            q = line.strip()
            if q:
                out.append((q, hop_key))
    return out


def top_k_in_pool(
    query: str,
    embeddings: np.ndarray,
    model,
    model_id: str,
    k: int,
) -> list[float]:
    """Top-k cosine similarities of query vs a hop-specific pool slice."""
    to_encode = [f"query: {query}"] if _needs_e5_prefix(model_id) else [query]
    q_emb = model.encode(to_encode, normalize_embeddings=True)[0]
    scores = np.dot(embeddings, q_emb)
    return [float(scores[i]) for i in np.argsort(-scores)[:k]]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="pool_coverage.json")
    p.add_argument("--pool", type=Path, default=None, help="Override the few-shot router pool path.")
    p.add_argument("--embed-model", default=None, help="Key in pool_coverage.json embed_models")
    p.add_argument("--sample-size-per-hop", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--run-dir", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    paths_cfg = load_paths(require(cfg, "paths_config"))
    limits = load_limits("model_limits.json")
    data_root = Path(paths_cfg["data_root_resolved"])

    seed = args.seed if args.seed is not None else int(require(cfg, "seed"))
    seeded = set_global_seed(seed)
    embed_key = args.embed_model or require(cfg, "embed_model")
    model_id = require(cfg, f"embed_models.{embed_key}")
    hops = [int(h) for h in require(cfg, "hops")]
    hop_keys = [f"{h}hop" for h in hops]
    top_k = int(require(cfg, "top_k"))
    report_top_k = [int(k) for k in require(cfg, "report_top_k")]
    thresholds = [float(t) for t in require(cfg, "thresholds")]
    sample_size = (
        args.sample_size_per_hop
        if args.sample_size_per_hop is not None
        else require(cfg, "sample_size_per_hop")
    )
    pool_path = args.pool or resolve_path(require(paths_cfg, "repo." + require(cfg, "pool_key")), REPO_ROOT)
    cache_dir = resolve_path(
        require(paths_cfg, "datasets." + require(cfg, "cache_dir_key")), data_root
    )
    run_dir = args.run_dir or runs_path(paths_cfg, require(cfg, "run_subdir"))
    run_dir.mkdir(parents=True, exist_ok=True)

    if not pool_path.exists():
        raise SystemExit(f"pool not found: {pool_path}")

    print(f"Loading pool and embeddings ({embed_key})...")
    pool = load_router_pool(pool_path)
    all_items, all_embeddings, model = get_router_pool_embeddings(
        pool_path, cache_dir=cache_dir, model_id=model_id
    )
    size_record = assert_within_ceiling(
        model, component="retrieval", model_id=model_id, limits=limits
    )

    # Slice the combined pool back into per-hop windows (order: 1hop, 2hop, 3hop).
    offsets = [0]
    for hop_key in hop_keys:
        offsets.append(offsets[-1] + len(pool.get(hop_key, [])))
    hop_data: dict[str, tuple[list[dict], np.ndarray]] = {}
    for i, hop_key in enumerate(hop_keys):
        s, e = offsets[i], offsets[i + 1]
        hop_data[hop_key] = (all_items[s:e], all_embeddings[s:e])
        print(f"  {hop_key}: {len(hop_data[hop_key][0])} items")

    questions = load_test_questions(
        require(paths_cfg, "datasets." + require(cfg, "questions_template_key")), data_root, hops
    )
    if not questions:
        raise SystemExit(
            f"no questions found under {data_root}; set data_root in configs/paths.json"
        )

    if sample_size:
        rng = new_rng(seed)
        by_hop: dict[str, list[str]] = {}
        for q, h in questions:
            by_hop.setdefault(h, []).append(q)
        sampled: list[tuple[str, str]] = []
        for hop_key in hop_keys:
            if hop_key in by_hop:
                pool_qs = by_hop[hop_key]
                chosen = rng.sample(pool_qs, min(int(sample_size), len(pool_qs)))
                sampled.extend((q, hop_key) for q in chosen)
        rng.shuffle(sampled)
        questions = sampled
        print(f"Sampled {len(questions)} questions (seed={seed})")
    else:
        print(f"Loaded {len(questions)} questions")

    results: dict[str, list[dict]] = {hop_key: [] for hop_key in hop_keys}
    for query, hop_key in questions:
        items_h, emb_h = hop_data[hop_key]
        if not items_h:
            continue
        sims = top_k_in_pool(query, emb_h, model, model_id, min(top_k, len(items_h)))
        rec: dict[str, float | str | None] = {
            "question": query[:80] + ("…" if len(query) > 80 else "")
        }
        for k in report_top_k:
            subset = sims[:k] if len(sims) >= k else sims
            rec[f"sim_top{k}"] = round(float(np.mean(subset)), 4) if subset else None
        results[hop_key].append(rec)

    print("\n" + "=" * 70)
    print("  POOL COVERAGE: similarity of questions to their hop-specific pool")
    print("=" * 70)

    first_k = report_top_k[0]
    all_top1: list[float] = []
    per_hop_metrics: dict[str, dict] = {}
    for hop_key in hop_keys:
        recs = results[hop_key]
        if not recs:
            continue
        top1 = [r[f"sim_top{first_k}"] for r in recs if r[f"sim_top{first_k}"] is not None]
        if not top1:
            continue
        all_top1.extend(top1)
        n = len(recs)
        print(f"\n{hop_key} (n={n}):")
        print(
            f"  sim_top{first_k}  min={min(top1):.3f}  mean={np.mean(top1):.3f}  max={max(top1):.3f}"
        )
        for k in report_top_k[1:]:
            vals = [r[f"sim_top{k}"] for r in recs if r[f"sim_top{k}"] is not None]
            if vals:
                print(f"  sim_top{k} mean={np.mean(vals):.3f}")
        entry: dict = {
            f"sim_top{first_k}_min": float(min(top1)),
            f"sim_top{first_k}_mean": float(np.mean(top1)),
            f"sim_top{first_k}_max": float(max(top1)),
        }
        for thresh in thresholds:
            below = sum(1 for s in top1 if s < thresh)
            print(f"  below {thresh}: {below}/{n} ({100 * below / n:.0f}%)")
            entry[f"below_{thresh}"] = below
        per_hop_metrics[hop_key] = entry

    if all_top1:
        print("\nOverall:")
        print(
            f"  sim_top{first_k}  min={min(all_top1):.3f}  "
            f"mean={np.mean(all_top1):.3f}  max={max(all_top1):.3f}"
        )

    metrics = {
        "script": Path(__file__).name,
        "created_utc": now_iso(),
        "seed": seed,
        "seeded": seeded,
        "embed_model": embed_key,
        "embed_model_id": model_id,
        "model_size": size_record,
        "pool_path": str(pool_path),
        "pool_sizes": {hop_key: len(hop_data[hop_key][0]) for hop_key in hop_keys},
        "n_questions": {hop_key: len(results[hop_key]) for hop_key in hop_keys},
        "top_k": top_k,
        "report_top_k": report_top_k,
        "thresholds": thresholds,
        "per_hop": per_hop_metrics,
        f"overall_sim_top{first_k}_mean": float(np.mean(all_top1)) if all_top1 else None,
    }
    write_run_artifacts(
        run_dir,
        config_snapshot={
            "script": Path(__file__).name,
            "config_path": cfg.get("_config_path"),
            "pool": str(pool_path),
            "embed_model": embed_key,
            "sample_size_per_hop": sample_size,
            "seed": seed,
            "run_dir": str(run_dir),
        },
        metrics=metrics,
        note_title="Few-shot pool coverage",
        note_lines=[
            f"- Pool: `{pool_path}`",
            f"- Embedding model: `{model_id}`",
            f"- Questions scored per hop: { {k: len(v) for k, v in results.items()} }",
            f"- Overall mean top-{first_k} similarity: "
            + (f"{float(np.mean(all_top1)):.4f}" if all_top1 else "unmeasured (no rows)"),
        ],
        prefix="coverage_",
    )


if __name__ == "__main__":
    main()
