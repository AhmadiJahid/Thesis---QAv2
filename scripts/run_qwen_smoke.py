#!/usr/bin/env python3
"""Decomposer smoke-test runner over a subsample of an existing retrieval file.

Samples ``n_per_hop`` queries per hop from a pool-sweep retrieval JSONL, runs the
decomposer on them, joins the predictions with the gold decompositions and writes one
CSV with columns::

    query_id, hop_count, question, pred_decomposition, gold_decomposition, few_shot_source

The decomposer run's own ``results.json``, ``config.json``, ``metrics.json`` and
``prompts_log/`` are preserved next to the CSV.

Ported from v1 ``scripts/run_qwen_smoke.py``. Adapted for v2: the source file, per-hop
count, hop list, model folder, retrieval settings, quantization and gold path come from
``configs/qwen_smoke.json`` / ``configs/paths.json``; it invokes the consolidated
decomposer runner; ``--dry-run`` builds the subsample only (and can be chained with the
runner's own ``--dry-run`` via ``--decomposer-dry-run``).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
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

DECOMPOSER_RUNNER = REPO_ROOT / "components" / "decomposer" / "run_decomposer.py"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="qwen_smoke.json")
    p.add_argument("--source", type=Path, default=None, help="Retrieval JSONL to subsample from.")
    p.add_argument("--n-per-hop", type=int, default=None)
    p.add_argument("--hops", type=int, nargs="+", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--retrieval-mode", default=None)
    p.add_argument("--retrieval-k", type=int, default=None)
    p.add_argument("--quantization", default=None, choices=["none", "4bit", "8bit"])
    p.add_argument("--model", default=None, help="Decomposer model folder.")
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true", help="Only build the subsample; skip the model run.")
    p.add_argument(
        "--decomposer-dry-run",
        action="store_true",
        help="Run the decomposer with --dry-run (prompts only, no model load).",
    )
    return p.parse_args()


def _hop_of(qid: str | None) -> int | None:
    if not qid:
        return None
    m = re.match(r"^(\d+)hop", qid)
    return int(m.group(1)) if m else None


def _sample_per_hop(source: Path, hops: list[int], n_per_hop: int, seed: int) -> list[dict]:
    buckets: dict[int, list[dict]] = defaultdict(list)
    with source.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            h = _hop_of(row.get("query_id"))
            if h in hops:
                buckets[h].append(row)

    rng = new_rng(seed)
    out: list[dict] = []
    for h in hops:
        pool = buckets.get(h, [])
        if not pool:
            print(f"[smoke] WARNING: no rows found for hop={h} in {source.name}")
            continue
        take = min(n_per_hop, len(pool))
        if take < n_per_hop:
            print(f"[smoke] WARNING: hop={h} only has {take} rows (requested {n_per_hop})")
        out.extend(rng.sample(pool, take))
    return out


def _build_gold_index(gold_path: Path) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    if not gold_path.exists():
        print(f"[smoke] WARNING: gold file not found, CSV will omit gold: {gold_path}")
        return index
    with gold_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = obj.get("id")
            steps = obj.get("question_decomposition") or []
            gold = [s.get("question", "").strip() for s in steps if isinstance(s, dict) and s.get("question")]
            if qid:
                index[qid] = gold
    return index


def _run_decomposer(cmd: list[str]) -> None:
    printable = [str(c) for c in cmd]
    print("[smoke] CMD:", " ".join(printable))
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    rc = subprocess.call(printable, cwd=str(REPO_ROOT), env=env)
    if rc != 0:
        raise SystemExit(f"[smoke] decomposer exited with code {rc}")


def _latest_run_dir(output_root: Path, started: datetime) -> Path:
    """The newest timestamped child of ``output_root`` created after ``started``."""
    candidates = sorted((p for p in output_root.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime)
    for p in reversed(candidates):
        if datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc) >= started:
            return p
    if candidates:
        return candidates[-1]
    raise SystemExit(f"[smoke] no run directory found under {output_root}")


def _write_csv(run_dir: Path, gold_index: dict[str, list[str]]) -> Path:
    results_path = run_dir / "results.json"
    if not results_path.exists():
        raise SystemExit(f"[smoke] results.json missing in {run_dir}")
    results = json.loads(results_path.read_text(encoding="utf-8"))

    csv_path = run_dir / "results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["query_id", "hop_count", "question", "pred_decomposition", "gold_decomposition", "few_shot_source"]
        )
        for r in results:
            qid = r.get("query_id") or ""
            writer.writerow(
                [
                    qid,
                    r.get("hop_count", ""),
                    r.get("question", ""),
                    r.get("decomposition", ""),
                    " | ".join(gold_index.get(qid, [])),
                    r.get("few_shot_source", ""),
                ]
            )
    return csv_path


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    paths_cfg = load_paths(require(cfg, "paths_config"))
    data_root = Path(paths_cfg["data_root_resolved"])

    seed = args.seed if args.seed is not None else int(require(cfg, "seed"))
    seeded = set_global_seed(seed)
    n_per_hop = args.n_per_hop if args.n_per_hop is not None else int(require(cfg, "n_per_hop"))
    hops = args.hops or [int(h) for h in require(cfg, "hops")]
    retrieval_mode = args.retrieval_mode or require(cfg, "retrieval_mode")
    retrieval_k = args.retrieval_k if args.retrieval_k is not None else int(require(cfg, "retrieval_k"))
    quantization = args.quantization or require(cfg, "quantization")
    model_folder = args.model or require(cfg, "decomposer_model_folder")
    source = args.source or runs_path(paths_cfg, require(cfg, "source_relpath"))
    gold_path = resolve_path(require(paths_cfg, "datasets." + require(cfg, "gold_key")), data_root)
    output_root = args.output_root or runs_path(paths_cfg, require(cfg, "output_subdir"))

    if not Path(source).exists():
        raise SystemExit(f"[smoke] source retrieval file not found: {source}")

    output_root.mkdir(parents=True, exist_ok=True)
    input_dir = output_root / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    subset_name = f"subset_{'_'.join(str(h) for h in hops)}hop_n{n_per_hop}_seed{seed}.jsonl"
    subset_path = input_dir / subset_name

    print(
        f"[smoke] sampling {n_per_hop} per hop from {Path(source).name} "
        f"(hops={hops}, seed={seed}) ..."
    )
    rows = _sample_per_hop(Path(source), hops, n_per_hop, seed)
    with subset_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[smoke] wrote {len(rows)} rows -> {subset_path}")

    base_meta = {
        "script": Path(__file__).name,
        "created_utc": now_iso(),
        "seed": seed,
        "seeded": seeded,
        "source": str(Path(source).resolve()),
        "subset": str(subset_path.resolve()),
        "subset_rows": len(rows),
        "n_per_hop": n_per_hop,
        "hops": hops,
        "retrieval_mode": retrieval_mode,
        "retrieval_k": retrieval_k,
        "quantization": quantization,
        "decomposer_model_folder": model_folder,
        "gold_path": str(gold_path),
    }

    if args.dry_run:
        write_run_artifacts(
            output_root,
            config_snapshot={"script": Path(__file__).name, "config_path": cfg.get("_config_path"), **base_meta},
            metrics={**base_meta, "decomposer_launched": False, "csv": None},
            note_title="Decomposer smoke subsample (dry run)",
            note_lines=[
                f"- Source: `{source}`",
                f"- Subset: `{subset_path}` ({len(rows)} rows)",
                "- Decomposer was not launched (--dry-run).",
            ],
            prefix="smoke_",
        )
        print("[smoke] dry run, skipping decomposer launch.")
        return

    started = datetime.now(tz=timezone.utc)
    cmd = [
        sys.executable, "-u", DECOMPOSER_RUNNER,
        "--config", require(cfg, "configs.decomposer"),
        "--model", model_folder,
        "--seed", seed,
        "--output-root", output_root,
        "--retrieval-input", subset_path,
        "--retrieval-mode", retrieval_mode,
        "--retrieval-k", retrieval_k,
        "--quantization", quantization,
    ]
    if args.decomposer_dry_run:
        cmd.extend(["--dry-run", "--dry-run-limit", str(len(rows))])
    _run_decomposer(cmd)

    run_dir = _latest_run_dir(output_root, started)
    print(f"[smoke] decomposer run dir: {run_dir}")

    gold_index = _build_gold_index(gold_path)
    csv_path = _write_csv(run_dir, gold_index)

    metrics = {
        **base_meta,
        "decomposer_launched": True,
        "decomposer_dry_run": args.decomposer_dry_run,
        "run_dir": str(run_dir.resolve()),
        "csv": str(csv_path.resolve()),
        "gold_ids_indexed": len(gold_index),
    }
    write_run_artifacts(
        run_dir,
        config_snapshot={"script": Path(__file__).name, "config_path": cfg.get("_config_path"), **base_meta},
        metrics=metrics,
        note_title="Decomposer smoke run",
        note_lines=[
            f"- Source: `{source}`",
            f"- Subset: `{subset_path}` ({len(rows)} rows)",
            f"- Decomposer model folder: `{model_folder}` (quantization {quantization})",
            f"- Decomposer dry run: {args.decomposer_dry_run}",
            f"- CSV: `{csv_path}`",
            f"- Gold ids indexed: {len(gold_index)}",
        ],
        prefix="smoke_",
    )

    print(f"[smoke] CSV  -> {csv_path}")


if __name__ == "__main__":
    main()
