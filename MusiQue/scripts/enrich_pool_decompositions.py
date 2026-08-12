#!/usr/bin/env python3
"""Fill in ``few_shot_decomposition_musique`` on a merged pool JSONL.

Why: when the ``train_0_*_all`` file (which had the field populated for every row)
was combined with 2-hop-only chunk files (which never had it), a large share of the
merged pool ended up with ``few_shot_decomposition_musique = None``. The decomposer
then silently drops those candidates when assembling few-shot prompts.

Fix: the MuSiQue train source has ``question_decomposition`` for every id, so build
an ``id -> [step.question]`` map from it and fill any pool row that is missing the
field.

Guarantees (fail fast): every pool id must be found in the source, every output row
must end up with a non-empty list, all existing fields are preserved.

Ported from v1. Adapted for v2: the source path comes from
``configs/musique_prep.json`` / ``configs/paths.json`` and the run writes the standard
trail next to the per-output stats file.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from _prep_common import dataset_path, load_prep, require, run_dir_for

from run_artifacts import now_iso, write_run_artifacts
from seeding import set_global_seed


def _iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _build_decomp_index(source_path: Path) -> dict[str, list[str]]:
    """Map row id -> list of step-question strings."""
    index: dict[str, list[str]] = {}
    for obj in _iter_jsonl(source_path):
        row_id = obj.get("id")
        if not row_id:
            continue
        step_questions: list[str] = []
        for step in obj.get("question_decomposition") or []:
            if not isinstance(step, dict):
                continue
            q = step.get("question")
            if isinstance(q, str) and q.strip():
                step_questions.append(q.strip())
        if step_questions:
            index[row_id] = step_questions
    return index


def _is_populated(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, str)):
        return bool(value)
    return True


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="musique_prep.json")
    p.add_argument("--pool", type=Path, required=True, help="Merged pool JSONL to enrich.")
    p.add_argument("--source", type=Path, default=None, help="Override the MuSiQue train source JSONL.")
    p.add_argument("--out", type=Path, required=True, help="Output enriched JSONL path.")
    p.add_argument("--run-dir", type=Path, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--overwrite", action="store_true", help="Overwrite the output if it exists.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg, paths_cfg = load_prep(args.config)
    section = require(cfg, "enrich")

    seed = args.seed if args.seed is not None else int(require(cfg, "seed"))
    seeded = set_global_seed(seed)  # logged; this script does not sample
    source = args.source or dataset_path(paths_cfg, require(section, "source_key"))
    run_dir = args.run_dir or run_dir_for(paths_cfg, require(section, "run_subdir"))

    if args.out.exists() and not args.overwrite:
        print(f"[enrich_pool] output already exists, skipping: {args.out}")
        return
    if not args.pool.exists():
        raise SystemExit(f"[enrich_pool] pool file not found: {args.pool}")
    if not source.exists():
        raise SystemExit(f"[enrich_pool] source file not found: {source}")

    print(f"[enrich_pool] building decomposition index from {source} ...")
    decomp_index = _build_decomp_index(source)
    print(f"[enrich_pool] indexed {len(decomp_index)} source ids with decomposition steps")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    pre_existing = 0
    enriched = 0
    still_missing: list[str] = []
    unknown_id: list[str] = []
    bucket_enriched: Counter[str] = Counter()

    with args.out.open("w", encoding="utf-8") as out_f:
        for obj in _iter_jsonl(args.pool):
            total += 1
            row_id = obj.get("id")
            current = obj.get("few_shot_decomposition_musique")

            if _is_populated(current):
                pre_existing += 1
            else:
                if not row_id:
                    still_missing.append("<no-id>")
                    out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    continue
                steps = decomp_index.get(row_id)
                if not steps:
                    unknown_id.append(row_id)
                    out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    continue
                obj["few_shot_decomposition_musique"] = steps
                enriched += 1
                bucket_enriched[row_id.split("_", 1)[0]] += 1

            out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    metrics = {
        "script": Path(__file__).name,
        "created_utc": now_iso(),
        "seed": seed,
        "seeded": seeded,
        "pool_input": str(args.pool.resolve()),
        "source_input": str(source.resolve()),
        "output": str(args.out.resolve()),
        "total_rows": total,
        "pre_existing_decomp": pre_existing,
        "enriched_from_source": enriched,
        "still_missing_rows": len(still_missing) + len(unknown_id),
        "missing_due_to_no_id": len(still_missing),
        "missing_due_to_source_miss": len(unknown_id),
        "bucket_enriched_counts": dict(bucket_enriched),
        "sample_unknown_ids": unknown_id[:20],
    }

    stats_path = args.out.with_suffix(args.out.suffix + ".stats.json")
    stats_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    write_run_artifacts(
        run_dir,
        config_snapshot={
            "script": Path(__file__).name,
            "config_path": cfg.get("_config_path"),
            "pool": str(args.pool),
            "source": str(source),
            "out": str(args.out),
            "seed": seed,
            "overwrite": args.overwrite,
        },
        metrics=metrics,
        note_title="Pool decomposition enrichment",
        note_lines=[
            f"- Pool: `{args.pool}`",
            f"- Source: `{source}`",
            f"- Output: `{args.out}`",
            f"- Rows: {total} (pre-existing {pre_existing}, enriched {enriched}, "
            f"still missing {len(still_missing) + len(unknown_id)})",
            f"- Per-file stats: `{stats_path}`",
        ],
    )

    print(
        f"[enrich_pool] total={total} pre_existing={pre_existing} "
        f"enriched={enriched} still_missing={len(still_missing) + len(unknown_id)}"
    )
    print(f"[enrich_pool] wrote -> {args.out}")

    if unknown_id:
        print(
            f"[enrich_pool] ERROR: {len(unknown_id)} ids not found in source; "
            f"first few: {unknown_id[:5]}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if still_missing:
        print(f"[enrich_pool] ERROR: {len(still_missing)} rows had no id", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
