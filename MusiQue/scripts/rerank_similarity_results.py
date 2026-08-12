#!/usr/bin/env python3
"""Cross-encoder reranker for MuSiQue similarity search outputs.

Reads JSONL produced by ``check_question_similarity.py`` (each row has ``raw_top_k``,
``typed_top_k``, ``uniform_top_k`` lists of bi-encoder candidates), reranks each mode's
candidates with a cross-encoder and keeps the top ``--rerank-k``. The bi-encoder score
is preserved as ``bi_encoder_score``.

Pipeline::

    check_question_similarity.py  (top 20)
      -> rerank_similarity_results.py  (20 -> 5)
      -> score_similarity_results.py

Ported from v1. Adapted for v2: the cross-encoder id, rerank-k, device and run
directory come from ``configs/similarity.json``; the cross-encoder's parameter count is
printed and asserted against the ceiling in ``configs/model_limits.json``; the run
writes the standard trail.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _prep_common import load_config, load_paths, require, run_dir_for

from model_size import assert_within_ceiling, load_limits
from run_artifacts import now_iso, write_run_artifacts
from seeding import set_global_seed

QUERY_FIELD: dict[str, str] = {
    "raw": "query_question",
    "typed": "query_question_masked_typed",
    "uniform": "query_question_masked_uniform",
}

POOL_FIELD: dict[str, str] = {
    "raw": "pool_question",
    "typed": "pool_question_masked_typed",
    "uniform": "pool_question_masked_uniform",
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="similarity.json")
    p.add_argument("--input", type=Path, required=True, help="Input JSONL from check_question_similarity.py")
    p.add_argument("--out", type=Path, required=True, help="Output JSONL (same schema, reranked lists)")
    p.add_argument("--cross-encoder", default=None, help="Override the config cross-encoder id.")
    p.add_argument("--rerank-k", type=int, default=None)
    p.add_argument("--device", default=None, help="cpu or cuda")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--run-dir", type=Path, default=None)
    return p.parse_args()


def _rerank_neighbours(
    query_text: str,
    neighbours: list[dict[str, Any]],
    pool_field: str,
    cross_encoder: Any,
    rerank_k: int,
) -> list[dict[str, Any]]:
    """Score (query, neighbour) pairs with the cross-encoder and keep the top rerank_k."""
    if not neighbours or not query_text:
        return neighbours[:rerank_k]

    pairs = [[query_text, nb.get(pool_field, nb.get("pool_question", ""))] for nb in neighbours]
    ce_scores = cross_encoder.predict(pairs, show_progress_bar=False)

    scored = []
    for nb, ce_score in zip(neighbours, ce_scores):
        entry = dict(nb)
        entry["bi_encoder_score"] = entry.get("score")
        entry["score"] = round(float(ce_score), 6)
        scored.append(entry)

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:rerank_k]


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    paths_cfg = load_paths(require(cfg, "paths_config"))
    limits = load_limits("model_limits.json")

    seed = args.seed if args.seed is not None else int(require(cfg, "seed"))
    seeded = set_global_seed(seed)
    modes = tuple(require(cfg, "modes"))
    cross_encoder_id = args.cross_encoder or require(cfg, "rerank.cross_encoder")
    rerank_k = args.rerank_k if args.rerank_k is not None else int(require(cfg, "rerank.rerank_k"))
    device = args.device or require(cfg, "device")
    progress_every = int(require(cfg, "rerank.progress_every"))
    run_dir = args.run_dir or run_dir_for(paths_cfg, require(cfg, "rerank.run_subdir"))
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows = _load_jsonl(args.input)
    print(f"Loaded {len(rows)} rows from {args.input}")
    print(f"Cross-encoder: {cross_encoder_id}  (device={device})")
    print(f"Rerank-k: {rerank_k}")

    from sentence_transformers import CrossEncoder

    cross_encoder = CrossEncoder(cross_encoder_id, device="cpu" if device == "cpu" else device)
    size_record = assert_within_ceiling(
        cross_encoder.model, component="reranker", model_id=cross_encoder_id, limits=limits
    )

    reranked_counts: dict[str, int] = {m: 0 for m in modes}

    with args.out.open("w", encoding="utf-8") as outf:
        for i, row in enumerate(rows):
            result = dict(row)
            for mode in modes:
                top_k_key = f"{mode}_top_k"
                neighbours = row.get(top_k_key)
                if not isinstance(neighbours, list) or not neighbours:
                    continue
                result[top_k_key] = _rerank_neighbours(
                    row.get(QUERY_FIELD[mode], ""),
                    neighbours,
                    POOL_FIELD[mode],
                    cross_encoder,
                    rerank_k,
                )
                reranked_counts[mode] += 1
            outf.write(json.dumps(result, ensure_ascii=False) + "\n")
            if (i + 1) % progress_every == 0 or (i + 1) == len(rows):
                print(f"  [{i + 1}/{len(rows)}] reranked")

    print(f"\nWrote {len(rows)} reranked rows -> {args.out}")

    metrics = {
        "script": Path(__file__).name,
        "created_utc": now_iso(),
        "seed": seed,
        "seeded": seeded,
        "cross_encoder_model": cross_encoder_id,
        "model_size": size_record,
        "rerank_k": rerank_k,
        "device": device,
        "input": str(args.input.resolve()),
        "output": str(args.out.resolve()),
        "num_rows": len(rows),
        "reranked_per_mode": reranked_counts,
    }
    write_run_artifacts(
        run_dir,
        config_snapshot={
            "script": Path(__file__).name,
            "config_path": cfg.get("_config_path"),
            "input": str(args.input),
            "out": str(args.out),
            "cross_encoder": cross_encoder_id,
            "rerank_k": rerank_k,
            "device": device,
            "seed": seed,
        },
        metrics=metrics,
        note_title="Cross-encoder reranking",
        note_lines=[
            f"- Model: `{cross_encoder_id}` ({size_record['parameter_count']:,} parameters)",
            f"- Rerank-k: {rerank_k}; device: {device}",
            f"- Input: `{args.input}`",
            f"- Output: `{args.out}` ({len(rows)} rows)",
            f"- Rows reranked per mode: {reranked_counts}",
        ],
        prefix="rerank_",
    )


if __name__ == "__main__":
    main()
