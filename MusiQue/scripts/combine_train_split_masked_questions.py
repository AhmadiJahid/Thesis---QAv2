#!/usr/bin/env python3
"""
Combine chunked masked MuSiQue question JSONL files into a single file.

Two modes:

- default: for each NER model subdirectory under ``--input-root``, combine the
  per-stratum chunk files into ``<prefix>_questions_all.jsonl`` (reindexed).
- ``--base-file`` + ``--extra-glob``: start from one file and merge extra globs on
  top, deduping by id.

Ported from v1. Adapted for v2: input root, train split, required fields and run
directory come from ``configs/musique_prep.json``; globs resolve against
``data_root``; the run writes the standard trail.

    python MusiQue/scripts/combine_train_split_masked_questions.py --overwrite
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from _prep_common import dataset_path, load_prep, require, run_dir_for

from run_artifacts import now_iso, write_run_artifacts
from seeding import set_global_seed

_FILENAME_RE = re.compile(
    r"^musique_ans_v1\.0_train_(?P<train>\d+)_questions_(?P<hop>\d+)_hop(?:_(?P<chunk>\d+))?\.jsonl$"
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="musique_prep.json")
    p.add_argument("--input-root", type=Path, default=None, help="Root with model subdirectories.")
    p.add_argument("--train-split", type=int, default=None)
    p.add_argument("--out-name", default=None, help="Output filename (default: <prefix>_questions_all.jsonl).")
    p.add_argument("--run-dir", type=Path, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--glob", default=None, help="Glob override, relative to --input-root.")
    p.add_argument(
        "--hops",
        type=int,
        nargs="*",
        default=None,
        metavar="N",
        help="Only include chunk files whose coarse hop count matches (e.g. 3 4).",
    )
    p.add_argument("--base-file", type=Path, default=None, help="Base JSONL to merge extras onto.")
    p.add_argument(
        "--extra-glob",
        nargs="*",
        default=None,
        help="Extra glob patterns (relative to --input-root) to merge onto --base-file.",
    )
    return p.parse_args()


def _list_inputs(input_root: Path, train_split: int, glob_override: str | None) -> list[Path]:
    pat = glob_override or f"**/musique_ans_v1.0_train_{train_split}_questions_*.jsonl"
    inputs = [p for p in input_root.glob(pat) if not p.name.endswith("_questions_all.jsonl")]
    if not inputs:
        raise SystemExit(f"No input files found under {input_root} (train_split={train_split}).")
    return sorted(inputs)


def _coarse_hop_from_path(path: Path) -> int | None:
    m = _FILENAME_RE.match(path.name)
    return int(m.group("hop")) if m else None


def _hop_chunk_sort_key(path: Path) -> tuple[int, int, str]:
    # 2_hop.jsonl => (2, 0); 3_hop_1.jsonl => (3, 1); 4_hop_3.jsonl => (4, 3)
    m = _FILENAME_RE.match(path.name)
    if not m:
        return (10**9, 10**9, path.name)
    chunk = m.group("chunk")
    return (int(m.group("hop")), int(chunk) if chunk is not None else 0, path.name)


def _iter_valid_rows(path: Path, req: set[str]) -> tuple[list[dict], int]:
    rows: list[dict] = []
    bad_rows = 0
    with path.open(encoding="utf-8") as rf:
        for line in rf:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad_rows += 1
                continue
            if not req.issubset(set(obj.keys())):
                bad_rows += 1
                continue
            rows.append(obj)
    return rows, bad_rows


def _resolve_base_file(path: Path, input_root: Path) -> Path:
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return (input_root / path).resolve()


def main() -> None:
    args = _parse_args()
    cfg, paths_cfg = load_prep(args.config)
    section = require(cfg, "combine")

    seed = args.seed if args.seed is not None else int(require(cfg, "seed"))
    seeded = set_global_seed(seed)  # logged; this script does not sample
    train_split = args.train_split if args.train_split is not None else int(require(section, "train_split"))
    req = set(require(section, "required_fields"))
    input_root = args.input_root or dataset_path(paths_cfg, require(section, "input_root_key"))
    run_dir = args.run_dir or run_dir_for(paths_cfg, require(section, "run_subdir"))
    default_out_name = f"musique_ans_v1.0_train_{train_split}_questions_all.jsonl"

    if args.base_file is not None:
        base_file = _resolve_base_file(args.base_file, input_root)
        if not base_file.exists():
            raise SystemExit(f"--base-file not found: {base_file}")
        if args.extra_glob is None:
            raise SystemExit("--base-file mode requires at least one --extra-glob pattern.")

        base_rows, base_bad_rows = _iter_valid_rows(base_file, req)

        extra_files: list[Path] = []
        for pat in args.extra_glob:
            extra_files.extend(input_root.glob(pat))
        extra_files = sorted({p.resolve() for p in extra_files if p.resolve() != base_file})
        if not extra_files:
            raise SystemExit("No extra files matched --extra-glob patterns.")

        out_path = base_file.parent / (args.out_name or default_out_name)
        if out_path.exists() and out_path.resolve() != base_file.resolve() and not args.overwrite:
            raise SystemExit(f"Output exists and --overwrite not set: {out_path}")

        seen_ids: set[str] = set()
        merged_rows: list[dict] = []
        duplicate_ids = 0
        bad_rows = base_bad_rows
        extra_input_counts: dict[str, int] = {}

        for obj in base_rows:
            rid = obj.get("id")
            if rid in seen_ids:
                duplicate_ids += 1
                continue
            seen_ids.add(rid)
            merged_rows.append(obj)

        for inp in sorted(extra_files, key=_hop_chunk_sort_key):
            rows, inp_bad = _iter_valid_rows(inp, req)
            bad_rows += inp_bad
            extra_input_counts[inp.name] = len(rows)
            for obj in rows:
                rid = obj.get("id")
                if rid in seen_ids:
                    duplicate_ids += 1
                    continue
                seen_ids.add(rid)
                merged_rows.append(obj)

        with out_path.open("w", encoding="utf-8") as wf:
            for idx, obj in enumerate(merged_rows):
                obj["index"] = idx
                wf.write(json.dumps(obj, ensure_ascii=False) + "\n")

        metrics = {
            "script": Path(__file__).name,
            "created_utc": now_iso(),
            "seed": seed,
            "seeded": seeded,
            "input_root": str(input_root.resolve()),
            "train_split": train_split,
            "out_name": out_path.name,
            "mode": "base_plus_extras",
            "base_file": str(base_file),
            "base_rows": len(base_rows),
            "base_bad_rows": base_bad_rows,
            "extra_globs": args.extra_glob,
            "extra_files": [str(p) for p in sorted(extra_files, key=_hop_chunk_sort_key)],
            "extra_rows_per_file": extra_input_counts,
            "written_rows": len(merged_rows),
            "duplicate_ids": duplicate_ids,
            "bad_rows": bad_rows,
            "output": str(out_path.resolve()),
        }
        write_run_artifacts(
            run_dir,
            config_snapshot={
                "script": Path(__file__).name,
                "config_path": cfg.get("_config_path"),
                "mode": "base_plus_extras",
                "input_root": str(input_root),
                "base_file": str(base_file),
                "extra_globs": args.extra_glob,
                "out_name": out_path.name,
                "seed": seed,
                "overwrite": args.overwrite,
            },
            metrics=metrics,
            note_title="Combine masked question chunks (base + extras)",
            note_lines=[
                f"- Seed: {seed}",
                f"- Base file: `{base_file}`",
                f"- Extra globs: `{args.extra_glob}`",
                f"- Output: `{out_path}` ({len(merged_rows)} rows, {duplicate_ids} duplicate ids skipped)",
            ],
        )
        print(f"Wrote {out_path} ({len(merged_rows)} rows)")
        return

    inputs = _list_inputs(input_root, train_split, args.glob)

    hop_filter: set[int] | None = set(args.hops) if args.hops else None
    if hop_filter:
        before = len(inputs)
        inputs = [p for p in inputs if (h := _coarse_hop_from_path(p)) is not None and h in hop_filter]
        if not inputs:
            raise SystemExit(
                f"No inputs left after --hops {sorted(hop_filter)} filter (had {before} files)."
            )

    by_parent_dir: dict[Path, list[Path]] = defaultdict(list)
    for p in inputs:
        by_parent_dir[p.parent].append(p)

    combined_out_name = args.out_name or default_out_name

    metrics: dict = {
        "script": Path(__file__).name,
        "created_utc": now_iso(),
        "seed": seed,
        "seeded": seeded,
        "input_root": str(input_root.resolve()),
        "train_split": train_split,
        "out_name": combined_out_name,
        "mode": "per_model_dir",
        "input_files_count": len(inputs),
        "hops_filter": sorted(hop_filter) if hop_filter else None,
        "per_dir": {},
    }

    for parent_dir, files in sorted(by_parent_dir.items(), key=lambda kv: str(kv[0])):
        ordered = sorted(files, key=_hop_chunk_sort_key)
        out_path = parent_dir / combined_out_name

        if out_path.exists() and not args.overwrite:
            print(f"Skipping existing output: {out_path}")
            metrics["per_dir"][str(parent_dir.resolve())] = {
                "skipped": True,
                "input_files": [p.name for p in ordered],
                "input_files_count": len(ordered),
            }
            continue

        seen_ids: set[str] = set()
        duplicate_ids = 0
        written_rows = 0
        bad_rows = 0

        with out_path.open("w", encoding="utf-8") as wf:
            new_index = 0
            for inp in ordered:
                with inp.open(encoding="utf-8") as rf:
                    for line in rf:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            bad_rows += 1
                            continue
                        if not req.issubset(set(obj.keys())):
                            bad_rows += 1
                            continue

                        rid = obj.get("id")
                        if rid in seen_ids:
                            duplicate_ids += 1
                        else:
                            seen_ids.add(rid)

                        # Reindex sequentially across all combined chunk files so
                        # index values stay unique when inputs reset index per chunk.
                        obj["index"] = new_index
                        new_index += 1

                        wf.write(json.dumps(obj, ensure_ascii=False) + "\n")
                        written_rows += 1

        metrics["per_dir"][str(parent_dir.resolve())] = {
            "skipped": False,
            "input_files": [p.name for p in ordered],
            "input_files_count": len(ordered),
            "written_rows": written_rows,
            "duplicate_ids": duplicate_ids,
            "bad_rows": bad_rows,
            "output": str(out_path.resolve()),
        }
        print(f"Wrote {out_path} ({written_rows} rows)")

    write_run_artifacts(
        run_dir,
        config_snapshot={
            "script": Path(__file__).name,
            "config_path": cfg.get("_config_path"),
            "mode": "per_model_dir",
            "input_root": str(input_root),
            "train_split": train_split,
            "glob": args.glob,
            "hops": args.hops,
            "out_name": combined_out_name,
            "seed": seed,
            "overwrite": args.overwrite,
        },
        metrics=metrics,
        note_title="Combine masked question chunks",
        note_lines=[
            f"- Seed: {seed}",
            f"- Input root: `{input_root}`",
            f"- Train split: {train_split}",
            f"- Hops filter: {sorted(hop_filter) if hop_filter else 'none'}",
            f"- Output filename: `{combined_out_name}`",
            f"- Directories combined: {len(metrics['per_dir'])}",
        ],
    )


if __name__ == "__main__":
    main()
