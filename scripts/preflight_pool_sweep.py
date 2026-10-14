#!/usr/bin/env python3
"""Refuse a pool-size-sweep run whose prebuilt inputs do not match the committed hashes.

Issue #51. For one ``--size`` this recomputes the sha256 of the pool file and of the
top-20 and reranked top-5 retrieval files named in ``configs/pool_size_sweep_v2.json``
and compares them with ``expected_inputs[size]``. It exits 0 only if every file exists
and matches; otherwise it prints each mismatch and exits 1. It reads and writes nothing
else, so it can be run immediately before every decompose step.

Paths in the config are relative to the main checkout root; ``--root`` (default: this
repo) relocates them.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from run_config import load_config, require  # noqa: E402

TOP20_NAME = "pinned600_top20.jsonl"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def expected_files(cfg: dict, size: int, root: Path) -> list[tuple[str, Path, str]]:
    """(label, path, expected sha256) for every input the size's run reads."""
    exp = require(cfg, "expected_inputs").get(str(size))
    if exp is None:
        raise SystemExit(f"[preflight] size {size} is not in expected_inputs")
    retrieval_dir = root / require(cfg, "retrieval.out_dir_template").format(size=size)
    top5 = root / require(cfg, "retrieval.retrieval_input_template").format(size=size)
    return [
        ("pool", root / require(exp, "pool_file"), require(exp, "pool_sha256")),
        ("top20", retrieval_dir / TOP20_NAME, require(exp, "top20_sha256")),
        ("rerank_top5", top5, require(exp, "rerank_top5_sha256")),
    ]


def check(files: list[tuple[str, Path, str]]) -> list[str]:
    problems: list[str] = []
    for label, path, want in files:
        if not path.exists():
            problems.append(f"{label}: missing {path}")
            continue
        got = sha256_file(path)
        if got != want:
            problems.append(f"{label}: sha256 {got} != expected {want} ({path})")
        else:
            print(f"[preflight] {label}: OK {got} {path}")
    return problems


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="pool_size_sweep_v2.json")
    p.add_argument("--size", type=int, required=True)
    p.add_argument("--root", type=Path, default=REPO_ROOT)
    args = p.parse_args()
    cfg = load_config(args.config)
    problems = check(expected_files(cfg, args.size, args.root))
    if problems:
        for line in problems:
            print(f"[preflight] REFUSE {line}", file=sys.stderr)
        raise SystemExit(1)
    print(f"[preflight] size {args.size}: all inputs match expected_inputs")


if __name__ == "__main__":
    main()
