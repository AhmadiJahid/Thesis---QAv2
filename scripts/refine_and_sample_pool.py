#!/usr/bin/env python3
"""
Refine MetaQA pool training data and draw a small per-hop sample.

For each hop: read the raw ``qa_train_<h>_hop.txt``, clean brackets
(``[entity]`` -> ``entity``) and drop the tab-separated answers, write the refined
file, then draw ``sample_size_per_hop`` questions into the hop's pool file.

Ported from v1 ``scripts/refine_and_sample_pool.py``. Adapted for v2: the pool
directory, sample size and per-hop file names come from ``configs/pool_refine.json``
/ ``configs/paths.json``; the seed is set once globally and used for a local RNG
(v1 re-seeded ``random`` inside the per-hop loop, so each hop drew from the same
seeded stream); the run writes the standard trail.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from run_artifacts import now_iso, write_run_artifacts  # noqa: E402
from run_config import (  # noqa: E402
    load_config,
    load_paths,
    require,
    resolve_path,
    runs_path,
)
from seeding import new_rng, set_global_seed  # noqa: E402


def clean_brackets(text: str) -> str:
    """Remove square brackets but keep the content inside."""
    return re.sub(r"\[([^\]]*)\]", r"\1", text)


def refine_line(line: str) -> str | None:
    """Extract the question, clean brackets, drop the answer. None for empty lines."""
    line = line.strip()
    if not line:
        return None
    question = line.split("\t", 1)[0]
    cleaned = clean_brackets(question)
    return cleaned.strip() if cleaned else None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="pool_refine.json")
    p.add_argument("--pool-dir", type=Path, default=None, help="Override the pool directory.")
    p.add_argument("--sample-size-per-hop", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--run-dir", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    paths_cfg = load_paths(require(cfg, "paths_config"))
    data_root = Path(paths_cfg["data_root_resolved"])

    seed = args.seed if args.seed is not None else int(require(cfg, "seed"))
    seeded = set_global_seed(seed)
    sample_size = (
        args.sample_size_per_hop
        if args.sample_size_per_hop is not None
        else int(require(cfg, "sample_size_per_hop"))
    )
    pool_dir = args.pool_dir or resolve_path(
        require(paths_cfg, "datasets." + require(cfg, "pool_dir_key")), data_root
    )
    run_dir = args.run_dir or runs_path(paths_cfg, require(cfg, "run_subdir"))
    run_dir.mkdir(parents=True, exist_ok=True)

    rng = new_rng(seed)
    per_hop_metrics: dict[str, dict] = {}

    for entry in require(cfg, "hops"):
        hop = int(require(entry, "hop"))
        src_path = pool_dir / require(entry, "source")
        refined_path = pool_dir / require(entry, "refined")
        pool_path = pool_dir / require(entry, "pool")

        if not src_path.exists():
            raise SystemExit(
                f"input file not found: {src_path} (set data_root / "
                f"datasets.{require(cfg, 'pool_dir_key')} in configs/paths.json)"
            )

        questions: list[str] = []
        with src_path.open(encoding="utf-8") as f:
            for line in f:
                q = refine_line(line)
                if q:
                    questions.append(q)

        refined_path.write_text("\n".join(questions) + "\n", encoding="utf-8")

        n = min(sample_size, len(questions))
        sampled = rng.sample(questions, n)
        pool_path.write_text("\n".join(sampled) + "\n", encoding="utf-8")

        per_hop_metrics[f"{hop}hop"] = {
            "source_path": str(src_path),
            "refined_path": str(refined_path),
            "pool_path": str(pool_path),
            "refined_question_count": len(questions),
            "sampled_count": n,
        }

    metrics = {
        "script": Path(__file__).name,
        "created_utc": now_iso(),
        "seed": seed,
        "seeded": seeded,
        "pool_dir": str(pool_dir),
        "sample_size_per_hop": sample_size,
        "per_hop": per_hop_metrics,
    }
    write_run_artifacts(
        run_dir,
        config_snapshot={
            "script": Path(__file__).name,
            "config_path": cfg.get("_config_path"),
            "pool_dir": str(pool_dir),
            "sample_size_per_hop": sample_size,
            "seed": seed,
        },
        metrics=metrics,
        note_title="MetaQA pool refine and sample",
        note_lines=[
            f"- Seed: {seed}",
            f"- Pool directory: `{pool_dir}`",
            f"- Sample size per hop: {sample_size}",
        ]
        + [
            f"- {hop}: {m['refined_question_count']} refined questions -> "
            f"sampled {m['sampled_count']}"
            for hop, m in per_hop_metrics.items()
        ],
        prefix="refine_",
    )

    print(f"Seed: {seed}")
    for hop, m in per_hop_metrics.items():
        print(f"  {hop}: {m['refined_question_count']} questions -> sampled {m['sampled_count']}")


if __name__ == "__main__":
    main()
