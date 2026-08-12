#!/usr/bin/env python3
"""
Clean MuSiQue chunk JSONL files: keep id, hop_count (derived), question,
question_decomposition and a per-file index.

Ported from v1. Adapted for v2: the input glob, output directory and suffix come
from ``configs/musique_prep.json``, globs resolve against ``data_root``, and the run
writes the standard trail.

    python MusiQue/scripts/clean_musique_train_chunks.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _prep_common import (
    dataset_path,
    expand_glob,
    load_prep,
    require,
    run_dir_for,
)

from musique_ids import coarse_hop_from_id
from run_artifacts import now_iso, write_run_artifacts
from seeding import set_global_seed


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="musique_prep.json")
    p.add_argument("--inputs", nargs="*", type=Path, help="Explicit input JSONL files (ordered).")
    p.add_argument("--input-glob", default=None, help="Override the config glob (data-root relative).")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--out-suffix", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--run-dir", type=Path, default=None)
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    p.add_argument(
        "--allow-clean-inputs",
        action="store_true",
        help="Include *_clean.jsonl glob matches (default: skip stems ending with _clean).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg, paths_cfg = load_prep(args.config)
    section = require(cfg, "clean")

    seed = args.seed if args.seed is not None else int(require(cfg, "seed"))
    seeded = set_global_seed(seed)
    out_suffix = args.out_suffix or require(section, "out_suffix")
    run_dir = args.run_dir or run_dir_for(paths_cfg, require(section, "run_subdir"))

    if args.inputs:
        inputs = [Path(p).resolve() for p in args.inputs]
    else:
        pattern = args.input_glob or require(section, "input_glob")
        inputs = expand_glob(paths_cfg, pattern)
        if not args.allow_clean_inputs:
            inputs = [p for p in inputs if not p.stem.endswith("_clean")]
        if not inputs:
            raise SystemExit(
                f"no files matched glob {pattern!r} under the data root; "
                "check configs/paths.json data_root, or pass --inputs"
            )

    out_dir = args.out_dir or dataset_path(paths_cfg, require(section, "out_dir_key"))
    out_dir.mkdir(parents=True, exist_ok=True)

    per_file: dict[str, int] = {}
    outputs: list[str] = []
    for inp in inputs:
        out_path = out_dir / f"{inp.stem}{out_suffix}.jsonl"
        if out_path.exists() and not args.overwrite:
            raise SystemExit(f"Refusing to overwrite (use --overwrite): {out_path}")

        idx = 0
        with inp.open(encoding="utf-8") as rf, out_path.open("w", encoding="utf-8") as wf:
            for line in rf:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                rid = obj.get("id", "")
                cleaned = {
                    "id": rid,
                    "hop_count": coarse_hop_from_id(rid),
                    "question": obj.get("question"),
                    "question_decomposition": obj.get("question_decomposition"),
                    "index": idx,
                }
                wf.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
                idx += 1

        per_file[inp.name] = idx
        outputs.append(str(out_path))
        print(f"{inp.name} -> {out_path.name} ({idx} rows)")

    metrics = {
        "script": Path(__file__).name,
        "created_utc": now_iso(),
        "seed": seed,
        "seeded": seeded,
        "inputs": [str(p) for p in inputs],
        "out_dir": str(out_dir.resolve()),
        "out_suffix": out_suffix,
        "rows_per_input": per_file,
        "total_rows": sum(per_file.values()),
        "outputs": outputs,
    }
    snapshot = {
        "script": Path(__file__).name,
        "config_path": cfg.get("_config_path"),
        "inputs": [str(p) for p in inputs],
        "out_dir": str(out_dir),
        "out_suffix": out_suffix,
        "seed": seed,
        "overwrite": args.overwrite,
    }
    write_run_artifacts(
        run_dir,
        config_snapshot=snapshot,
        metrics=metrics,
        note_title="MuSiQue chunk cleaning",
        note_lines=[
            f"- Seed: {seed}",
            f"- Inputs: {len(inputs)} file(s)",
            f"- Output directory: `{out_dir}`",
            f"- Total rows written: {sum(per_file.values())}",
        ],
    )


if __name__ == "__main__":
    main()
