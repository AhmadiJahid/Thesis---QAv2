#!/usr/bin/env python3
"""Extract specific rows (by 1-based line number) from each NER model's masked JSONL.

Used to eyeball the same questions masked by different NER models side by side.

Ported from v1. Adapted for v2: v1 hard-coded absolute cluster paths and a fixed list
of 40 line numbers. Here the base directory, output directory, filename and per-model
subdirectories come from ``configs/musique_prep.json`` / ``configs/paths.json``, and the
line numbers either come from ``extract_sample_rows.line_numbers`` in the config or are
drawn with the config seed (``sample_line_count`` rows) so the selection is reproducible
and recorded.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from _prep_common import dataset_path, load_prep, require, run_dir_for

from run_artifacts import now_iso, write_run_artifacts
from seeding import set_global_seed


def _count_lines(path: Path) -> int:
    n = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="musique_prep.json")
    p.add_argument("--base-dir", type=Path, default=None, help="Root containing the per-model subdirectories.")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--filename", default=None, help="File to sample inside each model subdirectory.")
    p.add_argument("--lines", type=int, nargs="*", default=None, help="Explicit 1-based line numbers.")
    p.add_argument("--count", type=int, default=None, help="How many lines to draw when none are given.")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--run-dir", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg, paths_cfg = load_prep(args.config)
    section = require(cfg, "extract_sample_rows")

    seed = args.seed if args.seed is not None else int(require(cfg, "seed"))
    seeded = set_global_seed(seed)
    base_dir = args.base_dir or dataset_path(paths_cfg, require(section, "base_dir_key"))
    out_dir = args.out_dir or dataset_path(paths_cfg, require(section, "out_dir_key"))
    filename = args.filename or require(section, "filename")
    sources: dict[str, str] = require(section, "sources")
    run_dir = args.run_dir or run_dir_for(paths_cfg, require(section, "run_subdir"))

    source_paths = {tag: base_dir / subdir / filename for tag, subdir in sources.items()}
    missing = [str(p) for p in source_paths.values() if not p.exists()]
    if missing:
        raise SystemExit("source file(s) not found:\n  " + "\n  ".join(missing))

    line_numbers = args.lines if args.lines is not None else list(require(section, "line_numbers"))
    if not line_numbers:
        count = args.count if args.count is not None else int(require(section, "sample_line_count"))
        first_path = next(iter(source_paths.values()))
        total = _count_lines(first_path)
        if total == 0:
            raise SystemExit(f"no rows in {first_path}")
        rng = random.Random(seed)
        line_numbers = sorted(rng.sample(range(1, total + 1), min(count, total)))
        print(f"[extract_sample_rows] drew {len(line_numbers)} line numbers with seed {seed}")

    line_set = set(line_numbers)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_source: dict[str, dict[str, object]] = {}
    for tag, src in source_paths.items():
        rows: dict[int, dict] = {}
        with src.open(encoding="utf-8") as f:
            for lineno, raw in enumerate(f, start=1):
                if lineno in line_set:
                    rows[lineno] = json.loads(raw)
        out_path = out_dir / f"sample_{tag}.jsonl"
        written = 0
        missing_lines: list[int] = []
        with out_path.open("w", encoding="utf-8") as out:
            for ln in line_numbers:
                row = rows.get(ln)
                if row is None:
                    missing_lines.append(ln)
                    print(f"  WARNING: line {ln} not found in {src.name}")
                    continue
                row["_line"] = ln
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
        per_source[tag] = {
            "source": str(src.resolve()),
            "output": str(out_path.resolve()),
            "rows_written": written,
            "missing_lines": missing_lines,
        }
        print(f"Wrote {written} rows -> {out_path}")

    metrics = {
        "script": Path(__file__).name,
        "created_utc": now_iso(),
        "seed": seed,
        "seeded": seeded,
        "base_dir": str(base_dir.resolve()),
        "out_dir": str(out_dir.resolve()),
        "filename": filename,
        "line_numbers": line_numbers,
        "per_source": per_source,
    }
    write_run_artifacts(
        run_dir,
        config_snapshot={
            "script": Path(__file__).name,
            "config_path": cfg.get("_config_path"),
            "base_dir": str(base_dir),
            "out_dir": str(out_dir),
            "filename": filename,
            "sources": sources,
            "line_numbers": line_numbers,
            "seed": seed,
        },
        metrics=metrics,
        note_title="Masked-question sample extract",
        note_lines=[
            f"- Base directory: `{base_dir}`",
            f"- File: `{filename}`",
            f"- Line numbers ({len(line_numbers)}): {line_numbers}",
            f"- Outputs: {[v['output'] for v in per_source.values()]}",
        ],
        prefix="extract_sample_",
    )


if __name__ == "__main__":
    main()
