#!/usr/bin/env python3
"""Sample a pool JSONL from the merged MuSiQue masked pool file.

Three pool-construction strategies (the ``--balance`` flag, which the sweep reads as
"strategy" — see ADR 0021):

- ``imbalanced``: uniform random draw without replacement over the whole pool, so the
  hop-bucket distribution mirrors the input (the natural imbalance of MuSiQue train).
- ``balanced``: ``size // 3`` rows per coarse hop bucket (2-hop, 3-hop, 4-hop), failing
  fast if a bucket does not have enough rows (4-hop is the tight constraint).
- ``clustered``: seeded k-means over bi-encoder embeddings of the input rows, keeping the
  rows nearest to each centroid (issue #14). Every knob — text field, embedding model,
  cluster count rule, representative rule, k-means parameters — comes from
  ``sample_pool.clustering`` in ``configs/musique_prep.json``. The design is the
  implementer's, not a research decision: ADR 0021 records which parts are arbitrary.

The bucket comes from the row ``id`` prefix (``2hop__``, ``3hop1__``, ``4hop2__``, …),
with fine buckets collapsed into coarse ones.

Clustering reads a **stored** text field off each row and never masks anything, so
ADR 0003 (never re-mask the few-shot pool) holds by construction: the rows written out
are the input rows unchanged.

Ported from v1 (which had the two random strategies only). Adapted for v2: the bucket
list and default input come from ``configs/musique_prep.json`` / ``configs/paths.json``,
the seed goes through ``set_global_seed`` as well as the local RNG, and the run writes
the standard trail alongside the pool.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from _prep_common import dataset_path, load_prep, require

from model_size import assert_within_ceiling, load_limits
from run_artifacts import now_iso, write_run_artifacts
from run_config import load_config
from seeding import set_global_seed

_ID_HOP_RX = re.compile(r"^(?P<h>\d+)hop")

#: The only representative rule implemented. A second rule is a deliberate change to
#: ADR 0021, not a config value someone can set by accident.
_REPRESENTATIVE_RULES = ("nearest_to_centroid",)


def _coarse_bucket(row_id: str | None, buckets: tuple[str, ...]) -> str | None:
    if not row_id:
        return None
    m = _ID_HOP_RX.match(row_id)
    if not m:
        return None
    tag = f"{m.group('h')}hop"
    return tag if tag in buckets else None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _group_by_bucket(
    rows: list[dict[str, Any]], buckets: tuple[str, ...]
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {b: [] for b in buckets}
    skipped = 0
    for r in rows:
        bucket = _coarse_bucket(r.get("id"), buckets)
        if bucket is None:
            skipped += 1
            continue
        grouped[bucket].append(r)
    if skipped:
        print(f"[sample_pool] skipped {skipped} rows with unparseable id", file=sys.stderr)
    return grouped


def _sample_imbalanced(
    rows: list[dict[str, Any]], size: int, rng: random.Random
) -> list[dict[str, Any]]:
    if size > len(rows):
        raise SystemExit(f"[sample_pool] imbalanced size={size} exceeds pool row count {len(rows)}")
    return rng.sample(rows, size)


def _sample_balanced(
    grouped: dict[str, list[dict[str, Any]]],
    size: int,
    rng: random.Random,
    buckets: tuple[str, ...],
) -> list[dict[str, Any]]:
    base = size // len(buckets)
    remainder = size % len(buckets)
    per_bucket = {
        bucket: base + (1 if i < remainder else 0) for i, bucket in enumerate(buckets)
    }
    infeasible = [
        f"{bucket}: need {per_bucket[bucket]}, have {len(grouped.get(bucket, []))}"
        for bucket in buckets
        if per_bucket[bucket] > len(grouped.get(bucket, []))
    ]
    if infeasible:
        raise SystemExit("[sample_pool] balanced sample infeasible:\n  " + "\n  ".join(infeasible))
    out: list[dict[str, Any]] = []
    for bucket in buckets:
        out.extend(rng.sample(grouped[bucket], per_bucket[bucket]))
    rng.shuffle(out)
    return out


# ------- clustered strategy (issue #14, ADR 0021) ------------------------


def _clustering_texts(rows: list[dict[str, Any]], field: str) -> list[str]:
    """The stored text of every row, or a loud failure.

    A missing or blank field is refused rather than skipped: silently dropping rows would
    change the candidate set the clustering ran over without saying so. The field is read
    as-is — no masker runs here (ADR 0003).
    """
    texts: list[str] = []
    missing: list[str] = []
    for i, r in enumerate(rows):
        value = r.get(field)
        if not isinstance(value, str) or not value.strip():
            missing.append(str(r.get("id") or f"index{i}"))
            continue
        texts.append(value)
    if missing:
        raise SystemExit(
            f"[sample_pool] clustering text field {field!r} is missing or blank on "
            f"{len(missing)} row(s), e.g. {missing[:5]}. Point "
            f"sample_pool.clustering.text_field at a field the input pool actually "
            f"carries, or fix the input pool."
        )
    return texts


def _needs_e5_prefix(model_id: str) -> bool:
    """E5 models expect ``query:`` / ``passage:`` prefixes (same rule as the retrieval stage)."""
    return "e5" in model_id.lower()


def _embed_texts(
    texts: list[str],
    *,
    model_id: str,
    device: str,
    batch_size: int,
    prefix: str,
) -> tuple[Any, dict[str, Any]]:
    """Embed ``texts`` with the bi-encoder, L2-normalised, asserting the size ceiling."""
    import numpy as np
    from sentence_transformers import SentenceTransformer

    limits = load_limits("model_limits.json")
    model = SentenceTransformer(model_id, device=device)
    size_record = assert_within_ceiling(
        model, component="retrieval", model_id=model_id, limits=limits
    )
    prepared = [f"{prefix}: {t}" for t in texts] if _needs_e5_prefix(model_id) else list(texts)
    t0 = time.time()
    emb = model.encode(
        prepared,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    emb = np.asarray(emb, dtype=np.float32)
    record = {
        "model_id": model_id,
        "device": device,
        "batch_size": batch_size,
        "prefix_applied": prefix if _needs_e5_prefix(model_id) else None,
        "num_texts": len(texts),
        "dim": int(emb.shape[1]) if emb.ndim == 2 else None,
        "embed_seconds": round(time.time() - t0, 1),
        "model_size": size_record,
    }
    return emb, record


def _kmeans_select(
    embeddings: Any,
    row_ids: list[str],
    size: int,
    *,
    seed: int,
    examples_per_cluster: int,
    representative_rule: str,
    kmeans_params: dict[str, Any],
    num_threads: int,
) -> tuple[list[int], dict[str, Any]]:
    """Pick exactly ``size`` row indices by seeded k-means over ``embeddings``.

    ``k = ceil(size / examples_per_cluster)`` clusters; within each cluster the rows are
    ranked by distance to their centroid (ties broken by row id, then row index, so the
    order never depends on numpy's or sklearn's iteration order). Representatives are
    taken **rank-major**: every cluster contributes its closest row before any cluster
    contributes a second, so truncating at ``size`` cannot starve a cluster.

    Empty clusters (sklearn relocates most, but not by contract) leave a shortfall; it is
    topped up from the unselected rows in the same global ``(distance, id, index)`` order
    and **counted** in the diagnostics rather than passing silently.

    Returns (selected indices in selection order, diagnostics).
    """
    import numpy as np

    n = int(embeddings.shape[0])
    if size > n:
        raise SystemExit(
            f"[sample_pool] clustered size={size} exceeds pool row count {n}"
        )
    if representative_rule not in _REPRESENTATIVE_RULES:
        raise SystemExit(
            f"[sample_pool] unknown clustering.representative_rule "
            f"{representative_rule!r}; implemented: {list(_REPRESENTATIVE_RULES)} "
            f"(adding one is a change to ADR 0021, not a config value)"
        )
    if examples_per_cluster < 1:
        raise SystemExit(
            f"[sample_pool] clustering.examples_per_cluster must be >= 1, got "
            f"{examples_per_cluster}"
        )
    if num_threads < 1:
        raise SystemExit(
            f"[sample_pool] clustering.kmeans.num_threads must be >= 1, got {num_threads}"
        )

    from sklearn.cluster import KMeans
    from threadpoolctl import threadpool_limits

    n_clusters = min(math.ceil(size / examples_per_cluster), n)
    t0 = time.time()
    # Single-threaded by default: sklearn's Lloyd loop reduces over OpenMP chunks, so the
    # thread count is part of the floating-point result. num_threads=1 makes the fit
    # bit-reproducible under the seed; raising it trades that for speed.
    with threadpool_limits(limits=num_threads):
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=seed,
            init=str(require(kmeans_params, "init")),
            n_init=int(require(kmeans_params, "n_init")),
            max_iter=int(require(kmeans_params, "max_iter")),
            tol=float(require(kmeans_params, "tol")),
            algorithm=str(require(kmeans_params, "algorithm")),
        ).fit(np.asarray(embeddings, dtype=np.float32))
    fit_seconds = round(time.time() - t0, 1)

    labels = np.asarray(kmeans.labels_, dtype=int)
    centroids = np.asarray(kmeans.cluster_centers_, dtype=np.float32)
    distances = np.linalg.norm(
        np.asarray(embeddings, dtype=np.float32) - centroids[labels], axis=1
    )

    def sort_key(idx: int) -> tuple[float, str, int]:
        return (float(distances[idx]), row_ids[idx], idx)

    members: list[list[int]] = [[] for _ in range(n_clusters)]
    for idx in range(n):
        members[int(labels[idx])].append(idx)
    for cluster in members:
        cluster.sort(key=sort_key)

    selected: list[int] = []
    for rank in range(examples_per_cluster):
        for cluster in members:
            if rank < len(cluster):
                selected.append(cluster[rank])
                if len(selected) == size:
                    break
        if len(selected) == size:
            break

    from_clusters = len(selected)
    if len(selected) < size:
        chosen = set(selected)
        for idx in sorted((i for i in range(n) if i not in chosen), key=sort_key):
            selected.append(idx)
            if len(selected) == size:
                break

    if len(selected) != size or len(set(selected)) != size:
        raise AssertionError(
            f"[sample_pool] clustered selection produced {len(selected)} indices "
            f"({len(set(selected))} distinct) for size={size}; this is a bug above."
        )

    cluster_sizes = [len(c) for c in members]
    diagnostics = {
        "strategy": "clustered",
        "n_candidates": n,
        "n_clusters": n_clusters,
        "examples_per_cluster": examples_per_cluster,
        "representative_rule": representative_rule,
        "kmeans_params": {
            "init": str(require(kmeans_params, "init")),
            "n_init": int(require(kmeans_params, "n_init")),
            "max_iter": int(require(kmeans_params, "max_iter")),
            "tol": float(require(kmeans_params, "tol")),
            "algorithm": str(require(kmeans_params, "algorithm")),
            "num_threads": num_threads,
            "random_state": seed,
        },
        "kmeans_n_iter": int(kmeans.n_iter_),
        "kmeans_inertia": float(kmeans.inertia_),
        "empty_clusters": int(sum(1 for c in cluster_sizes if c == 0)),
        "cluster_size_min": int(min(cluster_sizes)),
        "cluster_size_max": int(max(cluster_sizes)),
        "selected_from_clusters": from_clusters,
        "selected_from_topup": size - from_clusters,
        "mean_distance_to_centroid_selected": float(
            np.mean(distances[np.asarray(selected, dtype=int)])
        ),
        "kmeans_fit_seconds": fit_seconds,
    }
    return selected, diagnostics


def _sample_clustered(
    rows: list[dict[str, Any]],
    size: int,
    rng: random.Random,
    *,
    clustering: dict[str, Any],
    embed_model_key: str | None,
    device: str | None,
    seed: int,
    embeddings: Any | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Cluster the input rows and return ``size`` representatives plus diagnostics.

    ``embeddings`` is a test seam: pass a precomputed array to exercise the selection
    without loading an encoder. Production always passes ``None`` and embeds here.
    """
    field = str(require(clustering, "text_field"))
    texts = _clustering_texts(rows, field)

    similarity_cfg = load_config(str(require(clustering, "similarity_config")))
    embed_key = embed_model_key or str(require(clustering, "embed_model"))
    model_id = str(require(similarity_cfg, f"bi_encoder.embed_models.{embed_key}"))
    embed_device = device or str(require(clustering, "device"))

    if embeddings is None:
        embeddings, embed_record = _embed_texts(
            texts,
            model_id=model_id,
            device=embed_device,
            batch_size=int(require(clustering, "embed_batch_size")),
            prefix=str(require(clustering, "embed_prefix")),
        )
    else:
        embed_record = {"model_id": model_id, "precomputed": True, "num_texts": len(texts)}

    row_ids = [str(r.get("id") or f"index{i}") for i, r in enumerate(rows)]
    selected, diagnostics = _kmeans_select(
        embeddings,
        row_ids,
        size,
        seed=seed,
        examples_per_cluster=int(require(clustering, "examples_per_cluster")),
        representative_rule=str(require(clustering, "representative_rule")),
        kmeans_params=require(clustering, "kmeans"),
        num_threads=int(require(clustering, "kmeans.num_threads")),
    )
    diagnostics["text_field"] = field
    diagnostics["embedding"] = embed_record
    diagnostics["embed_model_key"] = embed_key

    out = [rows[i] for i in selected]
    # Shuffle for the same reason the balanced draw does: pool order should not encode the
    # strategy (here, distance-to-centroid rank) for anything reading the file in order.
    rng.shuffle(out)
    return out, diagnostics


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="musique_prep.json")
    p.add_argument("--input", type=Path, default=None, help="Input pool JSONL (default: the enriched pool).")
    p.add_argument("--size", type=int, required=True, help="Target pool size.")
    p.add_argument(
        "--balance",
        choices=["imbalanced", "balanced", "clustered"],
        required=True,
        help="Pool-construction strategy (the sweep's 'balance' axis).",
    )
    p.add_argument("--seed", type=int, default=None, help="RNG seed for this trial (default: config seed).")
    p.add_argument("--out-dir", type=Path, required=True, help="Output directory for pool.jsonl + artifacts.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite pool.jsonl if it exists.")
    p.add_argument(
        "--embed-model",
        default=None,
        help="Bi-encoder key from configs/similarity.json (clustered only; default: config).",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Device for the bi-encoder (clustered only; default: config).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg, paths_cfg = load_prep(args.config)
    section = require(cfg, "sample_pool")

    seed = args.seed if args.seed is not None else int(require(cfg, "seed"))
    seeded = set_global_seed(seed)
    buckets = tuple(require(section, "coarse_buckets"))
    input_path = args.input or dataset_path(paths_cfg, "musique_pool_enriched")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pool_path = args.out_dir / "pool.jsonl"
    if pool_path.exists() and not args.overwrite:
        print(f"[sample_pool] pool.jsonl already exists, skipping: {pool_path}")
        return
    if not input_path.exists():
        raise SystemExit(f"[sample_pool] input not found: {input_path}")

    rows = _load_jsonl(input_path)
    print(f"[sample_pool] loaded {len(rows)} rows from {input_path}")
    grouped = _group_by_bucket(rows, buckets)
    input_counts = {b: len(grouped[b]) for b in buckets}
    print(f"[sample_pool] input bucket counts: {input_counts}")

    rng = random.Random(seed)
    clustering_diagnostics: dict[str, Any] | None = None
    if args.balance == "imbalanced":
        sampled = _sample_imbalanced(rows, args.size, rng)
    elif args.balance == "balanced":
        sampled = _sample_balanced(grouped, args.size, rng, buckets)
    else:
        sampled, clustering_diagnostics = _sample_clustered(
            rows,
            args.size,
            rng,
            clustering=require(section, "clustering"),
            embed_model_key=args.embed_model,
            device=args.device,
            seed=seed,
        )

    sampled_counts = Counter(_coarse_bucket(r.get("id"), buckets) or "unknown" for r in sampled)
    sampled_counts_dict = {b: sampled_counts.get(b, 0) for b in buckets}
    if sum(sampled_counts_dict.values()) != len(sampled):
        sampled_counts_dict["unknown"] = sampled_counts.get("unknown", 0)

    _write_jsonl(pool_path, sampled)

    metrics = {
        "script": Path(__file__).name,
        "created_utc": now_iso(),
        "seed": seed,
        "seeded": seeded,
        "input": str(input_path.resolve()),
        "output": str(pool_path.resolve()),
        "size_requested": args.size,
        "size_written": len(sampled),
        "balance": args.balance,
        "coarse_buckets": list(buckets),
        "input_bucket_counts": input_counts,
        "sampled_bucket_counts": sampled_counts_dict,
    }
    if clustering_diagnostics is not None:
        metrics["clustering"] = clustering_diagnostics
    # v1 wrote stats.json next to the pool; keep that filename for compatibility with
    # anything reading a pool directory, and add the standard trail alongside it.
    (args.out_dir / "stats.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_run_artifacts(
        args.out_dir,
        config_snapshot={
            "script": Path(__file__).name,
            "config_path": cfg.get("_config_path"),
            "input": str(input_path),
            "size": args.size,
            "balance": args.balance,
            "seed": seed,
            "out_dir": str(args.out_dir),
            "overwrite": args.overwrite,
            "clustering": (
                require(section, "clustering") if args.balance == "clustered" else None
            ),
            "embed_model_override": args.embed_model,
            "device_override": args.device,
        },
        metrics=metrics,
        note_title=f"Pool sample ({args.balance}, size {args.size}, seed {seed})",
        note_lines=[
            f"- Input: `{input_path}`",
            f"- Output: `{pool_path}` ({len(sampled)} rows)",
            f"- Input bucket counts: {input_counts}",
            f"- Sampled bucket counts: {sampled_counts_dict}",
            *(
                [
                    f"- Clustering (ADR 0021): k={clustering_diagnostics['n_clusters']} "
                    f"x {clustering_diagnostics['examples_per_cluster']} per cluster over "
                    f"`{clustering_diagnostics['text_field']}` embedded with "
                    f"`{clustering_diagnostics['embedding']['model_id']}`; "
                    f"{clustering_diagnostics['selected_from_topup']} row(s) came from the "
                    f"top-up rule, {clustering_diagnostics['empty_clusters']} cluster(s) empty",
                ]
                if clustering_diagnostics is not None
                else []
            ),
        ],
    )

    print(f"[sample_pool] wrote {len(sampled)} rows -> {pool_path}")
    print(f"[sample_pool] bucket counts: {sampled_counts_dict}")
    if clustering_diagnostics is not None:
        print(f"[sample_pool] clustering: {json.dumps(clustering_diagnostics, default=str)}")


if __name__ == "__main__":
    main()
