#!/usr/bin/env python3
"""
Eyeball similarity-based few-shot selection.

For each probe question, print the top-k most similar entries in the committed
few-shot decomposition pool. The first run downloads the embedding model and builds
the cache.

Ported from v1 ``scripts/test_similarity_few_shot.py``. Adapted for v2: the probe
questions, embedding model registry, k and the pool path come from
``configs/similarity_probe.json`` / ``configs/paths.json``; the probe writes the standard
run trail so a printed neighbour list is traceable to a config.

    python scripts/test_similarity_few_shot.py
    python scripts/test_similarity_few_shot.py --compare
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from model_size import assert_within_ceiling, load_limits  # noqa: E402
from pool_embeddings import get_pool_embeddings, top_k_similar  # noqa: E402
from run_artifacts import now_iso, write_run_artifacts  # noqa: E402
from run_config import (  # noqa: E402
    load_config,
    load_paths,
    require,
    resolve_path,
    runs_path,
)
from seeding import set_global_seed  # noqa: E402


def run_probe(
    pool_embeddings: dict[str, tuple[list[dict], Any]],
    probes: list[dict[str, str]],
    model: Any,
    model_id: str,
    top_k: int,
    preview_chars: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    pool_questions: set[str] = set()
    for items, _ in pool_embeddings.values():
        pool_questions |= {it["question"] for it in items}

    for probe in probes:
        hop_key = require(probe, "hop_key")
        query = require(probe, "question")
        print(f"\n>>> PROBE QUESTION ({hop_key}):")
        print(f"    {query}")
        print(f"\n    Top {top_k} most similar from pool:")
        similar = top_k_similar(
            query,
            hop_key,
            pool_embeddings,
            model=model,
            model_id=model_id,
            k=top_k,
            exclude_question=query if query in pool_questions else None,
        )
        neighbours: list[dict[str, Any]] = []
        for i, (item, sim) in enumerate(similar, start=1):
            q = item["question"]
            preview = q[:preview_chars] + ("…" if len(q) > preview_chars else "")
            print(f"    {i}. [{sim:.3f}] {preview}")
            neighbours.append({"rank": i, "similarity": sim, "question": q})
        print()
        out.append({"hop_key": hop_key, "query": query, "neighbours": neighbours})
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="similarity_probe.json")
    p.add_argument("--model", default=None, help="Key in similarity_probe.json embed_models")
    p.add_argument("--compare", action="store_true", help="Run every model in compare_models.")
    p.add_argument("--run-dir", type=Path, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    paths_cfg = load_paths(require(cfg, "paths_config"))
    limits = load_limits("model_limits.json")
    section = require(cfg, "few_shot_probe")
    data_root = Path(paths_cfg["data_root_resolved"])

    seed = args.seed if args.seed is not None else int(require(cfg, "seed"))
    seeded = set_global_seed(seed)
    top_k = int(require(section, "top_k"))
    preview_chars = int(require(section, "question_preview_chars"))
    probes = list(require(section, "probe_questions"))
    pool_path = resolve_path(require(paths_cfg, "repo." + require(section, "pool_key")), REPO_ROOT)
    cache_dir = resolve_path(
        require(paths_cfg, "datasets." + require(cfg, "cache_dir_key")), data_root
    )
    run_dir = args.run_dir or runs_path(paths_cfg, require(section, "run_subdir"))
    run_dir.mkdir(parents=True, exist_ok=True)

    if not pool_path.exists():
        raise SystemExit(f"pool not found: {pool_path}")

    model_keys = (
        list(require(section, "compare_models"))
        if args.compare
        else [args.model or require(section, "embed_model")]
    )

    per_model: dict[str, Any] = {}
    for model_key in model_keys:
        model_id = require(cfg, f"embed_models.{model_key}")
        print(f"\n{'=' * 80}")
        print(f"  MODEL: {model_key} ({model_id})")
        print("=" * 80)
        print("Loading pool and building embeddings...")
        pool_embeddings = get_pool_embeddings(pool_path, cache_dir=cache_dir, model_id=model_id)

        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_id)
        size_record = assert_within_ceiling(
            model, component="retrieval", model_id=model_id, limits=limits
        )
        results = run_probe(pool_embeddings, probes, model, model_id, top_k, preview_chars)
        per_model[model_key] = {
            "model_id": model_id,
            "model_size": size_record,
            "pool_sizes": {k: len(v[0]) for k, v in pool_embeddings.items()},
            "probes": results,
        }

    write_run_artifacts(
        run_dir,
        config_snapshot={
            "script": Path(__file__).name,
            "config_path": cfg.get("_config_path"),
            "pool": str(pool_path),
            "models": model_keys,
            "top_k": top_k,
            "probe_questions": probes,
            "seed": seed,
        },
        metrics={
            "script": Path(__file__).name,
            "created_utc": now_iso(),
            "seed": seed,
            "seeded": seeded,
            "pool_path": str(pool_path),
            "top_k": top_k,
            "num_probes": len(probes),
            "per_model": per_model,
        },
        note_title="Few-shot similarity probe",
        note_lines=[
            f"- Pool: `{pool_path}`",
            f"- Models: {', '.join(model_keys)}",
            f"- Probe questions: {len(probes)}; top-k: {top_k}",
            "- This is a qualitative probe: it prints neighbours, it does not score them.",
        ],
        prefix="probe_",
    )


if __name__ == "__main__":
    main()
