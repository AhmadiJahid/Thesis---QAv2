#!/usr/bin/env python3
"""
Extract id, question and index from MuSiQue clean JSONL chunks, split by id stratum.

Writes one file per (chunk, stratum), e.g. ``musique_ans_v1.0_train_0_questions_2_hop.jsonl``.

Ported from v1. Adapted for v2: glob, output directory and run directory come from
``configs/musique_prep.json``; the run writes the standard trail.

    python MusiQue/scripts/extract_musique_clean_questions.py
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from _prep_common import (
    dataset_path,
    expand_glob,
    load_prep,
    require,
    run_dir_for,
)

from musique_ids import stratum_from_id, stratum_to_questions_slug
from run_artifacts import now_iso, write_run_artifacts
from seeding import set_global_seed


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="musique_prep.json")
    p.add_argument("--inputs", nargs="*", type=Path, help="Explicit clean JSONL paths (ordered).")
    p.add_argument("--input-glob", default=None, help="Override the config glob (data-root relative).")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--run-dir", type=Path, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--also-write-merged",
        action="store_true",
        help="Also write one ..._questions_all.jsonl per input (debug).",
    )
    return p.parse_args()


def _base_stem_clean(stem: str) -> str:
    return stem[: -len("_clean")] if stem.endswith("_clean") else stem


def main() -> None:
    args = _parse_args()
    cfg, paths_cfg = load_prep(args.config)
    section = require(cfg, "extract_questions")

    seed = args.seed if args.seed is not None else int(require(cfg, "seed"))
    seeded = set_global_seed(seed)
    run_dir = args.run_dir or run_dir_for(paths_cfg, require(section, "run_subdir"))

    if args.inputs:
        inputs = [Path(p).resolve() for p in args.inputs]
    else:
        pattern = args.input_glob or require(section, "input_glob")
        inputs = [p for p in expand_glob(paths_cfg, pattern) if p.stem.endswith("_clean")]
        if not inputs:
            raise SystemExit(
                f"no *_clean inputs matched glob {pattern!r} under the data root; "
                "run clean_musique_train_chunks.py first, or pass --inputs"
            )

    out_dir = args.out_dir or dataset_path(paths_cfg, require(section, "out_dir_key"))
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics: dict = {
        "script": Path(__file__).name,
        "created_utc": now_iso(),
        "seed": seed,
        "seeded": seeded,
        "out_dir": str(out_dir.resolve()),
        "inputs": [str(p) for p in inputs],
        "per_file": {},
        "outputs": [],
    }

    for inp in inputs:
        base = _base_stem_clean(inp.stem)
        by_stratum: dict[str, list[dict]] = defaultdict(list)
        unknown_strata = 0

        with inp.open(encoding="utf-8") as rf:
            for line in rf:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                rid = obj.get("id", "")
                st = stratum_from_id(rid)
                if st == "unknown":
                    unknown_strata += 1
                by_stratum[stratum_to_questions_slug(st)].append(
                    {"id": rid, "question": obj.get("question"), "index": obj.get("index")}
                )

        per_stratum_counts = {k: len(v) for k, v in sorted(by_stratum.items())}
        metrics["per_file"][inp.name] = {
            "per_stratum_counts": per_stratum_counts,
            "total_rows": sum(per_stratum_counts.values()),
            "unknown_stratum_rows": unknown_strata,
        }

        for slug, rows in sorted(by_stratum.items()):
            out_path = out_dir / f"{base}_questions_{slug}.jsonl"
            if out_path.exists() and not args.overwrite:
                raise SystemExit(f"Refusing to overwrite (use --overwrite): {out_path}")
            rows.sort(key=lambda r: (r.get("index") is None, r.get("index", 0)))
            with out_path.open("w", encoding="utf-8") as wf:
                for row in rows:
                    wf.write(json.dumps(row, ensure_ascii=False) + "\n")
            metrics["outputs"].append(str(out_path.resolve()))
            print(f"{inp.name} -> {out_path.name} ({len(rows)} rows)")

        if args.also_write_merged:
            merged_path = out_dir / f"{base}_questions_all.jsonl"
            if merged_path.exists() and not args.overwrite:
                raise SystemExit(f"Refusing to overwrite (use --overwrite): {merged_path}")
            all_rows: list[dict] = []
            for slug in sorted(by_stratum.keys()):
                all_rows.extend(by_stratum[slug])
            all_rows.sort(key=lambda r: (r.get("index") is None, r.get("index", 0)))
            with merged_path.open("w", encoding="utf-8") as wf:
                for row in all_rows:
                    wf.write(json.dumps(row, ensure_ascii=False) + "\n")
            metrics["outputs"].append(str(merged_path.resolve()))
            print(f"{inp.name} -> {merged_path.name} ({len(all_rows)} rows, merged)")

    snapshot = {
        "script": Path(__file__).name,
        "config_path": cfg.get("_config_path"),
        "inputs": [str(p) for p in inputs],
        "out_dir": str(out_dir),
        "seed": seed,
        "overwrite": args.overwrite,
        "also_write_merged": args.also_write_merged,
    }
    write_run_artifacts(
        run_dir,
        config_snapshot=snapshot,
        metrics=metrics,
        note_title="MuSiQue question extract (stratified)",
        note_lines=[
            f"- Seed: {seed}",
            f"- Inputs: {len(inputs)} file(s)",
            f"- Output directory: `{out_dir}`",
            f"- Files written: {len(metrics['outputs'])}",
        ],
    )


if __name__ == "__main__":
    main()
