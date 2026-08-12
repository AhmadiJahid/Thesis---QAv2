#!/usr/bin/env python3
"""
Embed dev questions and a pool JSONL with an E5 model, then report the top-k most
similar pool entries per query (cosine similarity via dot product on L2-normalised
vectors).

Modes (``--mode``)::

    raw      ->  question                 vs  question
    typed    ->  question_masked_typed    vs  question_masked_typed
    uniform  ->  question_masked_uniform  vs  question_masked_uniform
    all      ->  raw + typed + uniform in one pass, side by side

For typed / uniform / all the query questions are NER-masked on the fly unless the
query JSONL already carries the masked fields. Pool embeddings are cached on disk,
keyed by pool content hash + model + mode.

Ported from v1. Adapted for v2: embedding model registry, NER model, top-k, device,
query glob and cache directory come from ``configs/similarity.json`` /
``configs/paths.json``; the embedding model's parameter count is asserted against the
ceiling; the run writes the standard trail.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _prep_common import (
    dataset_path,
    expand_glob,
    load_config,
    load_paths,
    require,
    run_dir_for,
)

import numpy as np

from model_size import assert_within_ceiling, load_limits
from run_artifacts import now_iso, write_run_artifacts
from seeding import set_global_seed


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _cache_stem(pool_path: Path, pool_hash: str, model_id: str, mode: str) -> str:
    key = f"{pool_path.resolve()}::{pool_hash}::{model_id}::{mode}::v1"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    safe_model = model_id.replace("/", "_")
    return f"pool_{safe_model}_{mode}_{digest}"


def _cache_paths(cache_dir: Path, stem: str) -> tuple[Path, Path]:
    return cache_dir / f"{stem}.npy", cache_dir / f"{stem}.json"


def _load_cached_pool_embeddings(
    emb_path: Path,
    meta_path: Path,
    *,
    expected_pool_hash: str,
    expected_model_id: str,
    expected_mode: str,
    expected_rows: int,
) -> np.ndarray | None:
    if not emb_path.exists() or not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("pool_hash") != expected_pool_hash:
            return None
        if meta.get("model_id") != expected_model_id:
            return None
        if meta.get("mode") != expected_mode:
            return None
        if int(meta.get("row_count", -1)) != expected_rows:
            return None
        arr = np.load(emb_path)
        if arr.shape[0] != expected_rows:
            return None
        return np.asarray(arr, dtype=np.float32)
    except Exception:
        return None


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(f"{path.suffix}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _atomic_write_npy(path: Path, array: np.ndarray) -> None:
    tmp = path.with_suffix(f"{path.suffix}.tmp")
    with tmp.open("wb") as f:
        np.save(f, array)
    tmp.replace(path)


def _save_cached_pool_embeddings(
    emb_path: Path,
    meta_path: Path,
    *,
    emb: np.ndarray,
    pool_hash: str,
    model_id: str,
    mode: str,
) -> None:
    emb32 = np.asarray(emb, dtype=np.float32)
    _atomic_write_npy(emb_path, emb32)
    _atomic_write_json(
        meta_path,
        {
            "pool_hash": pool_hash,
            "model_id": model_id,
            "mode": mode,
            "row_count": int(emb32.shape[0]),
            "dim": int(emb32.shape[1]) if emb32.ndim == 2 else None,
            "dtype": str(emb32.dtype),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "format_version": 1,
        },
    )


def _needs_e5_prefix(model_id: str) -> bool:
    return "e5" in model_id.lower()


def _load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _embed(texts: list[str], model: Any, model_id: str, *, prefix: str) -> np.ndarray:
    prepared = [f"{prefix}: {t}" for t in texts] if _needs_e5_prefix(model_id) else texts
    return model.encode(prepared, normalize_embeddings=True, show_progress_bar=True)


def _mask_queries_on_the_fly(
    questions: list[str],
    ner_model_name: str,
    device: str,
    limits: dict[str, Any],
) -> dict[str, list[str]]:
    """Run NER on raw questions and return typed + uniform masked versions."""
    from ner_mask_musique_question_chunks import (
        _load_tokenizer_for_ner,
        _mask_from_entities,
        load_masking_rules,
    )
    from transformers import pipeline as hf_pipeline

    print(f"  NER model: {ner_model_name} (device={device})")
    rules = load_masking_rules()
    tokenizer = _load_tokenizer_for_ner(ner_model_name)
    dev_int = -1 if device == "cpu" else 0
    ner_pipe = hf_pipeline(
        "token-classification",
        model=ner_model_name,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=dev_int,
    )
    assert_within_ceiling(ner_pipe.model, component="ner", model_id=ner_model_name, limits=limits)

    typed: list[str] = []
    uniform: list[str] = []
    for q in questions:
        ents = ner_pipe(q)
        ents = ents if isinstance(ents, list) else []
        typed.append(
            _mask_from_entities(
                ents, q, typed_mode=True, use_regex=True, use_literal_hints=False, rules=rules
            )
        )
        uniform.append(
            _mask_from_entities(
                ents, q, typed_mode=False, use_regex=True, use_literal_hints=False, rules=rules
            )
        )

    return {"question_masked_typed": typed, "question_masked_uniform": uniform}


def _top_k_from_scores(
    scores: np.ndarray,
    pool_rows: list[dict[str, Any]],
    top_k: int,
) -> list[list[dict[str, Any]]]:
    """Top-k neighbours from a (num_queries x num_pool) similarity matrix."""
    all_neighbours: list[list[dict[str, Any]]] = []
    for i in range(scores.shape[0]):
        row_scores = scores[i]
        top_idx = np.argsort(row_scores)[::-1][:top_k]
        neighbours: list[dict[str, Any]] = []
        for j in top_idx:
            prow = pool_rows[int(j)]
            entry: dict[str, Any] = {
                "pool_id": prow.get("id"),
                "pool_index": prow.get("index"),
                "pool_question": prow.get("question"),
                "pool_question_masked_typed": prow.get("question_masked_typed"),
                "pool_question_masked_uniform": prow.get("question_masked_uniform"),
                "pool_few_shot_decomposition_musique": prow.get("few_shot_decomposition_musique"),
                "score": round(float(row_scores[j]), 4),
            }
            neighbours.append({k: v for k, v in entry.items() if v is not None})
        all_neighbours.append(neighbours)
    return all_neighbours


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="similarity.json")
    p.add_argument("--query-file", type=Path, default=None, help="JSONL with query questions (single file).")
    p.add_argument("--query-glob", default=None, help="Override the config query glob (data-root relative).")
    p.add_argument("--pool-file", type=Path, required=True, help="JSONL with pool questions.")
    p.add_argument("--n", type=int, default=None, help="Query rows per file (default: config).")
    p.add_argument("--mode", choices=["raw", "typed", "uniform", "all"], default="all")
    p.add_argument("--embed-model", default=None, help="Key in similarity.json bi_encoder.embed_models")
    p.add_argument("--ner-model", default=None, help="HF NER model for on-the-fly query masking.")
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--device", default=None, help="cpu or cuda")
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--no-cache", action="store_true", help="Disable disk cache reads and writes.")
    p.add_argument("--rebuild-cache", action="store_true", help="Recompute and overwrite cache entries.")
    p.add_argument("--out", type=Path, default=None, help="Output JSONL path (default: stdout).")
    p.add_argument("--run-dir", type=Path, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    paths_cfg = load_paths(require(cfg, "paths_config"))
    limits = load_limits("model_limits.json")

    seed = args.seed if args.seed is not None else int(require(cfg, "seed"))
    seeded = set_global_seed(seed)
    embed_key = args.embed_model or require(cfg, "bi_encoder.embed_model")
    model_id = require(cfg, f"bi_encoder.embed_models.{embed_key}")
    top_k = args.top_k if args.top_k is not None else int(require(cfg, "bi_encoder.top_k"))
    n_per_file = args.n if args.n is not None else int(require(cfg, "bi_encoder.num_queries_per_file"))
    device = args.device or require(cfg, "device")
    ner_model = args.ner_model or require(cfg, "bi_encoder.ner_model")
    mode_fields: dict[str, str] = require(cfg, "mode_fields")
    modes = list(require(cfg, "modes")) if args.mode == "all" else [args.mode]
    cache_enabled = bool(require(cfg, "bi_encoder.cache_enabled")) and not args.no_cache
    cache_dir = args.cache_dir or dataset_path(paths_cfg, require(cfg, "bi_encoder.cache_dir_key"))
    run_dir = args.run_dir or run_dir_for(paths_cfg, require(cfg, "bi_encoder.run_subdir"))

    if args.query_file is not None:
        query_files = [args.query_file.resolve()]
    else:
        pattern = args.query_glob or require(cfg, "bi_encoder.query_glob")
        query_files = expand_glob(paths_cfg, pattern)
        if not query_files:
            raise SystemExit(f"No query files matched {pattern!r} under the data root")

    print(f"Query : {len(query_files)} files  (n={n_per_file} per file)")
    print(f"Pool  : {args.pool_file}")
    print(f"Embed : {model_id}  (device={device})")
    print(f"Modes : {', '.join(modes)}")

    pool_rows = _load_jsonl(args.pool_file)
    print(f"Loaded {len(pool_rows)} pool entries")
    pool_path = args.pool_file.resolve()
    pool_hash = _sha256_file(pool_path) if cache_enabled else ""
    if cache_enabled:
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"Cache : {cache_dir}  (rebuild={args.rebuild_cache})")
    else:
        print("Cache : disabled")

    from sentence_transformers import SentenceTransformer

    embed_model = SentenceTransformer(model_id, device=device)
    embed_size_record = assert_within_ceiling(
        embed_model, component="retrieval", model_id=model_id, limits=limits
    )

    # Pool embeddings once per mode, so multiple dev query files are fast.
    pool_emb_by_mode: dict[str, np.ndarray] = {}
    cache_hits: dict[str, bool] = {}
    for mode in modes:
        field = mode_fields[mode]
        emb_path = meta_path = None
        if cache_enabled:
            stem = _cache_stem(pool_path, pool_hash, model_id, mode)
            emb_path, meta_path = _cache_paths(cache_dir, stem)
            if not args.rebuild_cache:
                cached = _load_cached_pool_embeddings(
                    emb_path,
                    meta_path,
                    expected_pool_hash=pool_hash,
                    expected_model_id=model_id,
                    expected_mode=mode,
                    expected_rows=len(pool_rows),
                )
                if cached is not None:
                    pool_emb_by_mode[mode] = cached
                    cache_hits[mode] = True
                    print(f"  Pool embed ({mode}:{field}) cache hit -> {emb_path.name}")
                    continue
            print(f"  Pool embed ({mode}:{field}) cache miss; computing ...")
        else:
            print(f"  Pool embed ({mode}:{field}) computing ...")

        cache_hits[mode] = False
        pool_texts = [r[field] for r in pool_rows]
        emb = _embed(pool_texts, embed_model, model_id, prefix="passage")
        pool_emb_by_mode[mode] = np.asarray(emb, dtype=np.float32)

        if cache_enabled and emb_path is not None and meta_path is not None:
            _save_cached_pool_embeddings(
                emb_path,
                meta_path,
                emb=pool_emb_by_mode[mode],
                pool_hash=pool_hash,
                model_id=model_id,
                mode=mode,
            )
            print(f"    saved cache -> {emb_path.name}")

    out_handle = args.out.open("w", encoding="utf-8") if args.out else sys.stdout
    total_written = 0
    total_queries = 0
    masked_on_the_fly_files = 0
    try:
        for qpath in query_files:
            print(f"\n=== Query file: {qpath} ===")
            query_rows = _load_jsonl(qpath, limit=n_per_file)
            if not query_rows:
                print("  No queries loaded (empty file or n=0); skipping.")
                continue

            total_queries += len(query_rows)
            raw_questions = [r["question"] for r in query_rows]
            print(f"  Loaded {len(query_rows)} queries")

            masked: dict[str, list[str]] = {}
            if any(m in modes for m in ("typed", "uniform")):
                missing = [
                    m
                    for m in ("typed", "uniform")
                    if m in modes
                    and not all(
                        (mode_fields[m] in r) and (r.get(mode_fields[m]) is not None)
                        for r in query_rows
                    )
                ]
                if missing:
                    print(f"  Masking on-the-fly (missing: {', '.join(missing)}) ...")
                    masked = _mask_queries_on_the_fly(raw_questions, ner_model, device, limits)
                    masked_on_the_fly_files += 1
                else:
                    for m in ("typed", "uniform"):
                        if m in modes:
                            masked[mode_fields[m]] = [r[mode_fields[m]] for r in query_rows]

            query_texts_by_mode: dict[str, list[str]] = {
                "raw": raw_questions,
                "typed": masked.get("question_masked_typed", []),
                "uniform": masked.get("question_masked_uniform", []),
            }

            results_by_mode: dict[str, list[list[dict[str, Any]]]] = {}
            for mode in modes:
                q_texts = query_texts_by_mode[mode]
                if len(q_texts) != len(query_rows):
                    raise SystemExit(
                        f"Internal error: mode={mode} has {len(q_texts)} texts for {len(query_rows)} queries"
                    )
                print(f"  Mode {mode}: embedding {len(q_texts)} queries ...")
                q_emb = _embed(q_texts, embed_model, model_id, prefix="query")
                scores = np.dot(q_emb, pool_emb_by_mode[mode].T)
                results_by_mode[mode] = _top_k_from_scores(scores, pool_rows, top_k)

            for i, qrow in enumerate(query_rows):
                result: dict[str, Any] = {
                    "query_id": qrow.get("id"),
                    "query_index": qrow.get("index"),
                    "query_question": raw_questions[i],
                }
                if "typed" in modes:
                    result["query_question_masked_typed"] = query_texts_by_mode["typed"][i]
                if "uniform" in modes:
                    result["query_question_masked_uniform"] = query_texts_by_mode["uniform"][i]
                for mode in modes:
                    result[f"{mode}_top_k"] = results_by_mode[mode][i]
                out_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                total_written += 1
    finally:
        if args.out:
            out_handle.close()

    dest = str(args.out) if args.out else "stdout"
    print(f"\nWrote {total_written} results ({', '.join(modes)}) from {total_queries} queries -> {dest}")

    metrics = {
        "script": Path(__file__).name,
        "created_utc": now_iso(),
        "seed": seed,
        "seeded": seeded,
        "embed_model": embed_key,
        "embed_model_id": model_id,
        "model_size": embed_size_record,
        "ner_model": ner_model,
        "device": device,
        "modes": modes,
        "top_k": top_k,
        "num_queries_per_file": n_per_file,
        "pool_file": str(pool_path),
        "pool_rows": len(pool_rows),
        "query_files": [str(p) for p in query_files],
        "total_queries": total_queries,
        "total_written": total_written,
        "output": dest,
        "cache_enabled": cache_enabled,
        "cache_hits_per_mode": cache_hits,
        "query_files_masked_on_the_fly": masked_on_the_fly_files,
    }
    write_run_artifacts(
        run_dir,
        config_snapshot={
            "script": Path(__file__).name,
            "config_path": cfg.get("_config_path"),
            "pool_file": str(args.pool_file),
            "query_files": [str(p) for p in query_files],
            "mode": args.mode,
            "top_k": top_k,
            "n": n_per_file,
            "embed_model": embed_key,
            "ner_model": ner_model,
            "device": device,
            "seed": seed,
            "cache_enabled": cache_enabled,
            "out": dest,
        },
        metrics=metrics,
        note_title="Bi-encoder question similarity",
        note_lines=[
            f"- Embedding model: `{model_id}` ({embed_size_record['parameter_count']:,} parameters)",
            f"- Pool: `{pool_path}` ({len(pool_rows)} rows)",
            f"- Query files: {len(query_files)} (n={n_per_file} each)",
            f"- Modes: {', '.join(modes)}; top-k: {top_k}",
            f"- Rows written: {total_written} -> `{dest}`",
            f"- Cache hits per mode: {cache_hits}",
        ],
        prefix="similarity_",
    )


if __name__ == "__main__":
    main()
