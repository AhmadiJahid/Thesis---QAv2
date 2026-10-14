#!/usr/bin/env python3
"""Grow a base few-shot pool into larger pools that strictly nest (issue #51, ADR 0035).

The v2 pool-size sweep needs ``size`` to be the only thing that moves between points.
``sample_pool.py --balance imbalanced`` cannot give that past 5,461 rows: Python's
``random.sample`` switches algorithm at that size (for n = 19,938), so its 6k draw does not
contain its 5k draw. This script keeps the base pool as it is and appends rows from the
rest of the input in one seeded order:

1. remainder = every input row whose id is not in the base pool, sorted by id (so the
   order never depends on the input file's line order);
2. ``random.Random(seed).shuffle(remainder)``;
3. pool of size N = base rows (in base-file order) + ``remainder[: N - len(base)]``.

No hop stratification: the extension is a uniform draw from the remainder, the same
"natural proportions" the imbalanced strategy means (ADR 0031 item 1). Rows are written
exactly as they are stored in the input (ADR 0003: nothing is re-masked).

Every pool is checked before anything is written: ids unique, each pool a subset of the
next, and zero overlap with the pinned evaluation set by id and by question text.

Every parameter comes from ``nested_extension`` in ``configs/pool_size_sweep_v2.json``;
``--base-pool`` / ``--out-root`` only relocate files (for running outside the main
checkout).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from run_artifacts import now_iso, write_run_artifacts  # noqa: E402
from run_config import load_config, load_paths, require, resolve_path, runs_path  # noqa: E402
from seeding import set_global_seed  # noqa: E402

_HOP_RX = re.compile(r"^(\d+)hop")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def hop_mix(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for r in rows:
        m = _HOP_RX.match(str(r.get("id") or ""))
        counts[f"{m.group(1)}hop" if m else "unknown"] += 1
    return dict(sorted(counts.items()))


def norm_question(text: Any) -> str:
    return str(text or "").strip().lower()


def extend_nested(
    base_rows: list[dict[str, Any]],
    input_rows: list[dict[str, Any]],
    sizes: list[int],
    seed: int,
) -> dict[int, list[dict[str, Any]]]:
    """Return ``{size: rows}`` where every pool starts with ``base_rows`` and nests."""
    input_ids = [r["id"] for r in input_rows]
    if len(set(input_ids)) != len(input_ids):
        raise SystemExit("[extend_pool_nested] input has duplicate ids")
    by_id = {r["id"]: r for r in input_rows}
    base_ids = [r["id"] for r in base_rows]
    if len(set(base_ids)) != len(base_ids):
        raise SystemExit("[extend_pool_nested] base pool has duplicate ids")
    missing = [i for i in base_ids if i not in by_id]
    if missing:
        raise SystemExit(f"[extend_pool_nested] {len(missing)} base ids not in input, e.g. {missing[:3]}")
    changed = [r["id"] for r in base_rows if by_id[r["id"]] != r]
    if changed:
        raise SystemExit(
            f"[extend_pool_nested] {len(changed)} base rows differ from the input row with "
            f"the same id, e.g. {changed[:3]} - the base was not drawn from this input"
        )
    if sorted(sizes) != list(sizes) or len(set(sizes)) != len(sizes):
        raise SystemExit(f"[extend_pool_nested] sizes must be strictly increasing, got {sizes}")
    base_set = set(base_ids)
    remainder = sorted((r for r in input_rows if r["id"] not in base_set), key=lambda r: r["id"])
    random.Random(seed).shuffle(remainder)
    out: dict[int, list[dict[str, Any]]] = {}
    for size in sizes:
        extra = size - len(base_rows)
        if extra < 0 or extra > len(remainder):
            raise SystemExit(
                f"[extend_pool_nested] size {size} infeasible: base {len(base_rows)}, "
                f"remainder {len(remainder)}"
            )
        out[size] = list(base_rows) + remainder[:extra]
    return out


def check_pools(
    pools: dict[int, list[dict[str, Any]]],
    base_rows: list[dict[str, Any]],
    pinned_ids: set[str],
    pinned_questions: set[str],
) -> dict[int, dict[str, Any]]:
    """Assert nesting and zero pinned overlap; return per-size check results."""
    results: dict[int, dict[str, Any]] = {}
    previous = {r["id"] for r in base_rows}
    for size in sorted(pools):
        rows = pools[size]
        ids = [r["id"] for r in rows]
        id_set = set(ids)
        if len(ids) != size or len(id_set) != size:
            raise AssertionError(f"size {size}: {len(ids)} rows, {len(id_set)} distinct ids")
        if not previous <= id_set:
            raise AssertionError(f"size {size} does not contain the previous pool")
        id_overlap = len(id_set & pinned_ids)
        text_overlap = len({norm_question(r.get("question")) for r in rows} & pinned_questions)
        if id_overlap or text_overlap:
            raise AssertionError(
                f"size {size}: overlap with the pinned eval set (ids {id_overlap}, "
                f"question text {text_overlap})"
            )
        results[size] = {
            "rows": size,
            "contains_previous_pool": True,
            "pinned_id_overlap": id_overlap,
            "pinned_question_text_overlap": text_overlap,
            "hop_mix": hop_mix(rows),
        }
        previous = id_set
    return results


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="pool_size_sweep_v2.json")
    p.add_argument("--base-pool", type=Path, default=None, help="Relocate the base pool file.")
    p.add_argument("--out-root", type=Path, default=None, help="Relocate the pools directory.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    section = require(cfg, "nested_extension")
    paths_cfg = load_paths(require(section, "paths_config"))
    data_root = Path(paths_cfg["data_root_resolved"])

    seed = int(require(section, "seed"))
    seeded = set_global_seed(seed)
    sizes = [int(s) for s in require(section, "sizes")]
    base_path = args.base_pool or runs_path(paths_cfg, require(section, "base_pool"))
    out_root = args.out_root or runs_path(paths_cfg, require(section, "out_root"))
    input_path = resolve_path(require(paths_cfg, f"datasets.{require(section, 'input_key')}"), data_root)
    template = require(paths_cfg, f"datasets.{require(section, 'pinned_eval_template_key')}")

    base_sha = hashlib.sha256(base_path.read_bytes()).hexdigest()
    expected_base_sha = require(section, "base_pool_sha256")
    if base_sha != expected_base_sha:
        raise SystemExit(f"[extend_pool_nested] base pool sha256 {base_sha} != expected {expected_base_sha}")
    base_rows = load_jsonl(base_path)
    if len(base_rows) != int(require(section, "base_size")):
        raise SystemExit(f"[extend_pool_nested] base pool has {len(base_rows)} rows")
    input_rows = load_jsonl(input_path)

    pinned_ids: set[str] = set()
    pinned_questions: set[str] = set()
    for hop in require(section, "pinned_hops"):
        for r in load_jsonl(resolve_path(template.format(hop=hop), data_root)):
            pinned_ids.add(r["id"])
            pinned_questions.add(norm_question(r.get("question")))
    expected_pinned = int(require(section, "pinned_count"))
    if len(pinned_ids) != expected_pinned:
        raise SystemExit(f"[extend_pool_nested] pinned set has {len(pinned_ids)} ids, expected {expected_pinned}")

    pools = extend_nested(base_rows, input_rows, sizes, seed)
    checks = check_pools(pools, base_rows, pinned_ids, pinned_questions)

    dir_template = require(section, "out_dir_template")
    written: dict[str, Any] = {}
    for size, rows in pools.items():
        out_dir = out_root / dir_template.format(size=size)
        pool_path = out_dir / "pool.jsonl"
        if pool_path.exists() and not args.overwrite:
            raise SystemExit(f"[extend_pool_nested] {pool_path} exists; pass --overwrite")
        sha = write_jsonl(pool_path, rows)
        written[str(size)] = {**checks[size], "path": str(pool_path), "sha256": sha}
        print(f"[extend_pool_nested] size {size}: {checks[size]['hop_mix']} sha256 {sha}")

    write_run_artifacts(
        out_root,
        prefix="extend_pool_nested_",
        config_snapshot={
            "script": Path(__file__).name,
            "config_path": cfg.get("_config_path"),
            "nested_extension": section,
            "base_pool": str(base_path),
            "out_root": str(out_root),
            "input": str(input_path),
            "overwrite": args.overwrite,
        },
        metrics={
            "created_utc": now_iso(),
            "seed": seed,
            "seeded": seeded,
            "input_rows": len(input_rows),
            "base_rows": len(base_rows),
            "base_sha256": base_sha,
            "remainder_rows": len(input_rows) - len(base_rows),
            "pinned_ids": len(pinned_ids),
            "pools": written,
        },
        note_title="Nested pool extension (issue #51)",
        note_lines=[
            f"- Base: `{base_path}` ({len(base_rows)} rows, sha256 `{base_sha}`)",
            f"- Remainder: {len(input_rows) - len(base_rows)} input rows, sorted by id, shuffled with seed {seed}",
            *[
                f"- size {s}: {w['hop_mix']}, sha256 `{w['sha256']}`, contains previous pool, "
                f"pinned overlap 0 (ids) / 0 (question text)"
                for s, w in written.items()
            ],
        ],
    )


if __name__ == "__main__":
    main()
