#!/usr/bin/env python3
r"""Sample a fixed per-hop dev set from the MuSiQue dev question files.

Reads the per-hop dev JSONL files and samples ``--per-hop`` rows per coarse hop
bucket (2-hop, 3-hop, 4-hop), combining the fine-hop files for hops 3 and 4.

Rows are preserved verbatim (including ``id`` and any pre-computed
``question_masked_typed`` / ``question_masked_uniform`` fields) so downstream
similarity search can reuse masked queries instead of re-running NER on the dev
queries for every trial.

The first emitted row is asserted to have an ``id`` matching ``^\d+hop`` so the
decomposer's hop parsing keeps working. If ``--per-hop`` exceeds what a bucket has,
the script caps at what exists and records the reduction.

Ported from v1. Adapted for v2: the dev directory, the per-bucket file lists and the
default per-hop count come from ``configs/musique_prep.json`` / ``configs/paths.json``;
the seed goes through ``set_global_seed`` as well as the local RNG.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

from _prep_common import dataset_path, load_prep, require

from run_artifacts import now_iso, write_run_artifacts
from seeding import set_global_seed

_ID_HOP_RX = re.compile(r"^\d+hop")


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
    p.add_argument("--config", default="musique_prep.json")
    p.add_argument("--dev-dir", type=Path, default=None, help="Override the MuSiQue dev_data directory.")
    p.add_argument("--per-hop", type=int, default=None, help="Rows to sample per coarse hop bucket.")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", type=Path, required=True, help="Output JSONL path.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite the output if it exists.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg, paths_cfg = load_prep(args.config)
    section = require(cfg, "sample_dev")

    seed = args.seed if args.seed is not None else int(require(cfg, "seed"))
    seeded = set_global_seed(seed)
    per_hop = args.per_hop if args.per_hop is not None else int(require(section, "per_hop"))
    dev_dir = args.dev_dir or dataset_path(paths_cfg, require(section, "dev_dir_key"))
    hop_files: dict[str, list[str]] = require(section, "hop_files")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    stats_path = args.out.parent / f"{args.out.stem}_stats.json"

    if args.out.exists() and not args.overwrite:
        print(f"[sample_dev] output already exists, skipping: {args.out}")
        return

    rng = random.Random(seed)

    per_bucket_rows: dict[str, list[dict[str, Any]]] = {}
    available_counts: dict[str, int] = {}
    for bucket, files in hop_files.items():
        collected: list[dict[str, Any]] = []
        for fname in files:
            path = dev_dir / fname
            if not path.exists():
                raise SystemExit(f"[sample_dev] missing dev file: {path}")
            collected.extend(_load_jsonl(path))
        per_bucket_rows[bucket] = collected
        available_counts[bucket] = len(collected)

    print(f"[sample_dev] available per-bucket: {available_counts}")

    sampled_rows: list[dict[str, Any]] = []
    sampled_counts: dict[str, int] = {}
    capped_buckets: list[str] = []

    for bucket in hop_files:
        pool = per_bucket_rows[bucket]
        take = min(per_hop, len(pool))
        if take < per_hop:
            capped_buckets.append(
                f"{bucket}: requested {per_hop}, took {take} (available {len(pool)})"
            )
        sampled_rows.extend(rng.sample(pool, take))
        sampled_counts[bucket] = take

    if not sampled_rows:
        raise SystemExit("[sample_dev] no rows sampled.")

    first_id = sampled_rows[0].get("id", "")
    if not isinstance(first_id, str) or not _ID_HOP_RX.match(first_id):
        raise SystemExit(
            f"[sample_dev] first row id does not match ^\\d+hop: {first_id!r}. "
            "The decomposer's hop parsing would fail."
        )

    with args.out.open("w", encoding="utf-8") as f:
        for r in sampled_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    has_typed = all("question_masked_typed" in r for r in sampled_rows)
    has_uniform = all("question_masked_uniform" in r for r in sampled_rows)

    metrics = {
        "script": Path(__file__).name,
        "created_utc": now_iso(),
        "seed": seed,
        "seeded": seeded,
        "dev_dir": str(dev_dir.resolve()),
        "output": str(args.out.resolve()),
        "per_hop_requested": per_hop,
        "available_counts": available_counts,
        "sampled_counts": sampled_counts,
        "total_sampled": len(sampled_rows),
        "has_question_masked_typed": has_typed,
        "has_question_masked_uniform": has_uniform,
        "capped_buckets": capped_buckets,
    }
    # v1 wrote <out stem>_stats.json; keep it and add the standard trail beside it.
    stats_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_run_artifacts(
        args.out.parent,
        config_snapshot={
            "script": Path(__file__).name,
            "config_path": cfg.get("_config_path"),
            "dev_dir": str(dev_dir),
            "per_hop": per_hop,
            "seed": seed,
            "out": str(args.out),
            "overwrite": args.overwrite,
        },
        metrics=metrics,
        note_title=f"Dev sample ({per_hop} per hop, seed {seed})",
        note_lines=[
            f"- Dev directory: `{dev_dir}`",
            f"- Output: `{args.out}` ({len(sampled_rows)} rows)",
            f"- Sampled counts: {sampled_counts}",
            f"- Masked fields present in all rows: typed={has_typed}, uniform={has_uniform}",
            f"- Capped buckets: {capped_buckets or 'none'}",
        ],
        prefix="sample_dev_",
    )

    print(f"[sample_dev] wrote {len(sampled_rows)} rows -> {args.out}")
    print(f"[sample_dev] sampled counts: {sampled_counts}")
    print(f"[sample_dev] masked fields in all rows: typed={has_typed}, uniform={has_uniform}")
    if capped_buckets:
        print("[sample_dev] WARNING capped buckets:")
        for c in capped_buckets:
            print(f"  {c}")


if __name__ == "__main__":
    main()
