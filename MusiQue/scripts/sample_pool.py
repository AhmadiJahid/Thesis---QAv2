#!/usr/bin/env python3
"""Sample a pool JSONL from the merged MuSiQue masked pool file.

Two balance strategies:

- ``imbalanced``: uniform random draw without replacement over the whole pool, so the
  hop-bucket distribution mirrors the input (the natural imbalance of MuSiQue train).
- ``balanced``: ``size // 3`` rows per coarse hop bucket (2-hop, 3-hop, 4-hop), failing
  fast if a bucket does not have enough rows (4-hop is the tight constraint).

The bucket comes from the row ``id`` prefix (``2hop__``, ``3hop1__``, ``4hop2__``, …),
with fine buckets collapsed into coarse ones.

Ported from v1. Adapted for v2: the bucket list and default input come from
``configs/musique_prep.json`` / ``configs/paths.json``, the seed goes through
``set_global_seed`` as well as the local RNG, and the run writes the standard trail
alongside the pool.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from _prep_common import dataset_path, load_prep, require

from run_artifacts import now_iso, write_run_artifacts
from seeding import set_global_seed

_ID_HOP_RX = re.compile(r"^(?P<h>\d+)hop")


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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="musique_prep.json")
    p.add_argument("--input", type=Path, default=None, help="Input pool JSONL (default: the enriched pool).")
    p.add_argument("--size", type=int, required=True, help="Target pool size.")
    p.add_argument("--balance", choices=["imbalanced", "balanced"], required=True)
    p.add_argument("--seed", type=int, default=None, help="RNG seed for this trial (default: config seed).")
    p.add_argument("--out-dir", type=Path, required=True, help="Output directory for pool.jsonl + artifacts.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite pool.jsonl if it exists.")
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
    if args.balance == "imbalanced":
        sampled = _sample_imbalanced(rows, args.size, rng)
    else:
        sampled = _sample_balanced(grouped, args.size, rng, buckets)

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
        },
        metrics=metrics,
        note_title=f"Pool sample ({args.balance}, size {args.size}, seed {seed})",
        note_lines=[
            f"- Input: `{input_path}`",
            f"- Output: `{pool_path}` ({len(sampled)} rows)",
            f"- Input bucket counts: {input_counts}",
            f"- Sampled bucket counts: {sampled_counts_dict}",
        ],
    )

    print(f"[sample_pool] wrote {len(sampled)} rows -> {pool_path}")
    print(f"[sample_pool] bucket counts: {sampled_counts_dict}")


if __name__ == "__main__":
    main()
