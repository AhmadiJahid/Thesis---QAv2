#!/usr/bin/env python3
"""
Stratified n-way split of the MuSiQue train JSONL into chunk files.

Uses id-prefix strata (``2hop``, ``3hop1``, …) because ``hop_count`` is null in the
source.

Ported from v1. Adapted for v2: input path, output directory, prefix, split count
and seed come from ``configs/musique_prep.json`` / ``configs/paths.json``; the seed
is set globally before the shuffle; the run writes the standard trail.

    python MusiQue/scripts/split_musique_train_stratified.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from _prep_common import dataset_path, load_prep, require, run_dir_for

import numpy as np
from sklearn.model_selection import StratifiedKFold

from musique_ids import stratum_from_id
from run_artifacts import now_iso, write_run_artifacts
from seeding import set_global_seed


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="musique_prep.json")
    p.add_argument("--input", type=Path, default=None, help="Override the source train JSONL.")
    p.add_argument("--out-dir", type=Path, default=None, help="Override the chunk output directory.")
    p.add_argument("--prefix", default=None, help="Override the output basename prefix.")
    p.add_argument("--n-splits", type=int, default=None, help="Override the number of chunks.")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--run-dir", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg, paths_cfg = load_prep(args.config)
    section = require(cfg, "split")

    seed = args.seed if args.seed is not None else int(require(cfg, "seed"))
    seeded = set_global_seed(seed)
    prefix = args.prefix or require(section, "prefix")
    n_splits = args.n_splits if args.n_splits is not None else int(require(section, "n_splits"))

    input_path = args.input or dataset_path(paths_cfg, require(section, "input_key"))
    if args.out_dir is not None:
        out_dir = args.out_dir
    elif require(section, "out_dir_key"):
        out_dir = dataset_path(paths_cfg, require(section, "out_dir_key"))
    else:
        out_dir = input_path.parent
    run_dir = args.run_dir or run_dir_for(paths_cfg, require(section, "run_subdir"))

    if not input_path.is_file():
        raise SystemExit(f"input not found: {input_path} (set data_root in configs/paths.json)")

    lines: list[str] = []
    strata: list[str] = []
    with input_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            lines.append(line)
            strata.append(stratum_from_id(obj.get("id", "")))

    n = len(lines)
    if n == 0:
        raise SystemExit("No records in input")

    y = np.array(strata, dtype=object)
    X = np.zeros(n, dtype=np.int8)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    assignments = np.full(n, -1, dtype=np.int8)
    for fold_idx, (_, test_idx) in enumerate(skf.split(X, y)):
        assignments[test_idx] = fold_idx

    if not np.all(assignments >= 0):
        raise SystemExit("Internal error: not all rows assigned to a fold")

    out_dir.mkdir(parents=True, exist_ok=True)

    chunk_stratum_hist: dict[str, dict[str, int]] = {}
    chunk_line_counts: dict[str, int] = {}
    outputs: list[str] = []

    for chunk_id in range(n_splits):
        indices = np.where(assignments == chunk_id)[0]
        indices.sort()
        out_path = out_dir / f"{prefix}_{chunk_id}.jsonl"
        hist: Counter[str] = Counter()
        with out_path.open("w", encoding="utf-8") as wf:
            for i in indices:
                wf.write(lines[int(i)] + "\n")
                hist[strata[int(i)]] += 1
        chunk_stratum_hist[str(chunk_id)] = dict(sorted(hist.items()))
        chunk_line_counts[str(chunk_id)] = int(len(indices))
        outputs.append(str(out_path))

    metrics = {
        "script": Path(__file__).name,
        "created_utc": now_iso(),
        "seed": seed,
        "seeded": seeded,
        "input": str(input_path.resolve()),
        "out_dir": str(out_dir.resolve()),
        "prefix": prefix,
        "n_splits": n_splits,
        "total_lines": n,
        "chunk_line_counts": chunk_line_counts,
        "stratum_histograms_per_chunk": chunk_stratum_hist,
        "global_stratum_histogram": dict(sorted(Counter(strata).items())),
        "outputs": outputs,
    }
    snapshot = {
        "script": Path(__file__).name,
        "config_path": cfg.get("_config_path"),
        "input": str(input_path),
        "out_dir": str(out_dir),
        "prefix": prefix,
        "n_splits": n_splits,
        "seed": seed,
        "run_dir": str(run_dir),
    }
    write_run_artifacts(
        run_dir,
        config_snapshot=snapshot,
        metrics=metrics,
        note_title="MuSiQue train stratified split",
        note_lines=[
            f"- Seed: {seed}",
            f"- Input: `{input_path}`",
            f"- Chunks written under: `{out_dir}`",
            f"- Total lines: {n}",
            f"- Per-chunk counts: {chunk_line_counts}",
        ],
    )
    print(f"Wrote {n_splits} chunks under {out_dir}")


if __name__ == "__main__":
    main()
