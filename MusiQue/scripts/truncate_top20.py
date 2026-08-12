#!/usr/bin/env python3
"""Truncate a bi-encoder top-k similarity JSONL to the first k entries per mode.

Input: JSONL rows from ``check_question_similarity.py`` where each row has
``raw_top_k`` / ``typed_top_k`` / ``uniform_top_k`` lists pre-sorted by bi-encoder
cosine score (descending).

Output: the same schema with each ``*_top_k`` truncated to ``--k`` entries. This is the
``biencoder_only`` few-shot selector: take the first k as-is, no cross-encoder, no
re-scoring.

Ported from v1. Adapted for v2: k, the mode list and the run directory come from
``configs/similarity.json``; the run writes the standard trail.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _prep_common import load_config, load_paths, require, run_dir_for

from run_artifacts import now_iso, write_run_artifacts
from seeding import set_global_seed


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="similarity.json")
    p.add_argument("--input", type=Path, required=True, help="Top-k JSONL from check_question_similarity.py.")
    p.add_argument("--out", type=Path, required=True, help="Output JSONL path (truncated).")
    p.add_argument("--k", type=int, default=None, help="Truncation length (default: config truncate.k).")
    p.add_argument("--run-dir", type=Path, default=None, help="Directory for the run trail (default: --out parent).")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    paths_cfg = load_paths(require(cfg, "paths_config"))

    seed = args.seed if args.seed is not None else int(require(cfg, "seed"))
    seeded = set_global_seed(seed)  # logged; this script does not sample
    k = args.k if args.k is not None else int(require(cfg, "truncate.k"))
    modes = tuple(require(cfg, "modes"))
    if args.run_dir is not None:
        run_dir = args.run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
    elif args.out.parent != Path(""):
        run_dir = args.out.parent
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = run_dir_for(paths_cfg, require(cfg, "truncate.run_subdir"))
    args.out.parent.mkdir(parents=True, exist_ok=True)

    truncated_counts: dict[str, int] = {m: 0 for m in modes}
    total_rows = 0
    input_lengths_example: dict[str, int] = {}

    with args.input.open(encoding="utf-8") as fin, args.out.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row: dict[str, Any] = json.loads(line)
            for mode in modes:
                key = f"{mode}_top_k"
                nbrs = row.get(key)
                if isinstance(nbrs, list):
                    input_lengths_example.setdefault(mode, len(nbrs))
                    if len(nbrs) > k:
                        row[key] = nbrs[:k]
                        truncated_counts[mode] += 1
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            total_rows += 1

    metrics = {
        "script": Path(__file__).name,
        "created_utc": now_iso(),
        "seed": seed,
        "seeded": seeded,
        "input": str(args.input.resolve()),
        "output": str(args.out.resolve()),
        "k": k,
        "modes": list(modes),
        "num_rows": total_rows,
        "input_top_k_len_first_row": input_lengths_example,
        "rows_truncated_per_mode": truncated_counts,
    }
    write_run_artifacts(
        run_dir,
        config_snapshot={
            "script": Path(__file__).name,
            "config_path": cfg.get("_config_path"),
            "input": str(args.input),
            "out": str(args.out),
            "k": k,
            "seed": seed,
        },
        metrics=metrics,
        note_title=f"Bi-encoder top-{k} truncation",
        note_lines=[
            f"- Input: `{args.input}`",
            f"- Output: `{args.out}` ({total_rows} rows)",
            f"- Rows truncated per mode: {truncated_counts}",
            f"- First-row input list lengths: {input_lengths_example}",
        ],
        prefix="truncate_",
    )

    print(f"[truncate_top20] wrote {total_rows} rows -> {args.out}")
    print(f"[truncate_top20] truncated counts: {truncated_counts}")


if __name__ == "__main__":
    main()
