#!/usr/bin/env python3
"""Orchestrator for the MuSiQue pool sweep.

Reads ``configs/pool_sweep.json`` and drives the full pipeline:

  1. Sample a dev set once (reused for every cell).
  2. For each ``(size, balance, trial)`` — where ``balance`` is the pool-**construction
     strategy** (``imbalanced`` / ``balanced`` / ``clustered``, the last from issue #14
     and ADR 0021, configured under ``clustered_sizes``):
       a. ``sample_pool.py``
       b. ``check_question_similarity.py`` (bi-encoder top-k, mode=all)
       c. ``truncate_top20.py``            -> top5_biencoder.jsonl
          ``rerank_similarity_results.py`` -> top5_ce.jsonl
       d. for each ``(variant, mode)``:
            - ``components/decomposer/run_decomposer.py``
            - stabilise the results path (symlink the timestamped run's results.json
              up to ``<decomposer dir>/<run_key>/results.json``)
            - ``scripts/musique_decompositions_evaluator.py``
            - append one row to ``summary/all_runs.csv``

Features: ``--dry-run`` prints commands only; ``--stage`` restricts to one stage across
the grid; ``--only`` / ``--trials`` / ``--variants`` / ``--modes`` restrict the grid;
per-stage skip-existing unless ``--overwrite``; every subprocess is logged.

Ported from v1. Adapted for v2: dataset paths are keys into ``configs/paths.json``, the
runs root comes from config, the decomposer is invoked through the consolidated runner
by model folder, and each child script is passed its own config path.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from run_artifacts import now_iso, write_run_artifacts  # noqa: E402
from run_config import load_config, load_paths, require, resolve_path, runs_path  # noqa: E402

STAGES = ("sample_pool", "similarity", "truncate", "rerank", "decompose", "eval", "all")

MUSIQUE_SCRIPTS = REPO_ROOT / "MusiQue" / "scripts"
DECOMPOSER_RUNNER = REPO_ROOT / "components" / "decomposer" / "run_decomposer.py"
EVALUATOR = REPO_ROOT / "scripts" / "musique_decompositions_evaluator.py"


# ------- utilities --------------------------------------------------------


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _log_append(log_path: Path, line: str) -> None:
    _ensure_dir(log_path.parent)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line.rstrip("\n") + "\n")


def _run_cmd(
    cmd: list[str],
    *,
    log_path: Path,
    dry_run: bool,
    cwd: Path | None = None,
    extra_env: dict[str, str] | None = None,
) -> int:
    cmd = [str(c) for c in cmd]
    if cmd and cmd[0] == sys.executable and "-u" not in cmd[:3]:
        cmd.insert(1, "-u")

    pretty = " ".join(cmd)
    _log_append(log_path, f"[{now_iso()}] CMD: {pretty}")
    print(f"\n>>> {pretty}", flush=True)
    if dry_run:
        _log_append(log_path, f"[{now_iso()}] DRY_RUN (skipped)")
        return 0

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if extra_env:
        env.update(extra_env)
    t0 = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        bufsize=1,
        text=True,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        print(line, flush=True)
        _log_append(log_path, line)
    proc.wait()
    _log_append(log_path, f"[{now_iso()}] EXIT={proc.returncode} ELAPSED={time.time() - t0:.1f}s")
    return proc.returncode


def _grid(cfg: dict[str, Any]) -> list[tuple[int, str]]:
    """The (size, balance) combinations that actually exist in the config.

    ``balance`` is the pool-**construction strategy** axis, not only a hop-balance flag:
    ``clustered`` (issue #14, ADR 0021) is a third value of it. The name stays as it is so
    every path, run key and ``all_runs.csv`` column keeps its meaning and rows produced
    before the strategy existed stay comparable with rows produced after.
    """
    combos: list[tuple[int, str]] = []
    for s in require(cfg, "imbalanced_sizes"):
        combos.append((int(s), "imbalanced"))
    for s in require(cfg, "balanced_sizes"):
        combos.append((int(s), "balanced"))
    for s in require(cfg, "clustered_sizes"):
        combos.append((int(s), "clustered"))
    return combos


def _trial_seeds(cfg: dict[str, Any]) -> list[int]:
    seeds = require(cfg, "pool_trial_seeds")
    if isinstance(seeds, list) and seeds:
        return [int(x) for x in seeds]
    base = int(require(cfg, "base_pool_seed"))
    return [base + i for i in range(int(require(cfg, "num_trials")))]


def _run_key(size: int, balance: str, trial: int, variant: str, mode: str) -> str:
    return f"size{size}_{balance}_trial{trial}__{variant}__{mode}"


def _cell_key(size: int, balance: str, trial: int) -> str:
    return f"size{size}_{balance}_trial{trial}"


# ------- paths -----------------------------------------------------------


def _paths(cfg: dict[str, Any], paths_cfg: dict[str, Any]) -> dict[str, Path]:
    runs_root = runs_path(paths_cfg, require(cfg, "runs_subdir"))
    return {
        "runs_root": runs_root,
        "dev_sample_dir": runs_root / "dev_sample",
        "pools_dir": runs_root / "pools",
        "similarity_dir": runs_root / "similarity",
        "biencoder_top5_dir": runs_root / "biencoder_top5",
        "rerank_dir": runs_root / "rerank",
        "decomposer_dir": runs_root / "decomposer",
        "eval_dir": runs_root / "eval",
        "summary_dir": runs_root / "summary",
        "log_file": runs_root / "orchestrator.log",
    }


def _dev_sample_path(cfg: dict[str, Any], paths: dict[str, Path]) -> Path:
    seed = int(require(cfg, "dev_seed"))
    per_hop = int(require(cfg, "dev_per_hop"))
    return paths["dev_sample_dir"] / f"dev_sample_{per_hop}per_hop_seed{seed}.jsonl"


def _pool_dir(paths: dict[str, Path], size: int, balance: str, trial: int, seed: int) -> Path:
    return paths["pools_dir"] / f"size{size}_{balance}_trial{trial}_poolseed{seed}"


def _sim_dir(paths: dict[str, Path], size: int, balance: str, trial: int) -> Path:
    return paths["similarity_dir"] / _cell_key(size, balance, trial)


def _bi_top5_dir(paths: dict[str, Path], size: int, balance: str, trial: int) -> Path:
    return paths["biencoder_top5_dir"] / _cell_key(size, balance, trial)


def _ce_dir(paths: dict[str, Path], size: int, balance: str, trial: int) -> Path:
    return paths["rerank_dir"] / _cell_key(size, balance, trial)


def _decomp_dir(paths: dict[str, Path], run_key: str) -> Path:
    return paths["decomposer_dir"] / run_key


def _eval_dir(paths: dict[str, Path], run_key: str) -> Path:
    return paths["eval_dir"] / run_key


# ------- stage implementations -------------------------------------------


def stage_sample_dev(
    cfg: dict[str, Any],
    paths: dict[str, Path],
    log_path: Path,
    dry_run: bool,
    overwrite: bool,
) -> Path:
    dev_path = _dev_sample_path(cfg, paths)
    _ensure_dir(dev_path.parent)
    if dev_path.exists() and not overwrite:
        print(f"[orchestrator] dev sample exists, skipping: {dev_path}")
        return dev_path
    cmd = [
        sys.executable,
        MUSIQUE_SCRIPTS / "sample_dev.py",
        "--config", require(cfg, "configs.musique_prep"),
        "--per-hop", int(require(cfg, "dev_per_hop")),
        "--seed", int(require(cfg, "dev_seed")),
        "--out", dev_path,
    ]
    if overwrite:
        cmd.append("--overwrite")
    rc = _run_cmd(cmd, log_path=log_path, dry_run=dry_run)
    if rc != 0 and not dry_run:
        raise SystemExit(f"sample_dev.py failed with rc={rc}")
    return dev_path


def stage_sample_pool(
    cfg: dict[str, Any],
    paths: dict[str, Path],
    input_pool: Path,
    log_path: Path,
    dry_run: bool,
    overwrite: bool,
    *,
    size: int,
    balance: str,
    trial: int,
    pool_seed: int,
) -> Path:
    out_dir = _pool_dir(paths, size, balance, trial, pool_seed)
    pool_jsonl = out_dir / "pool.jsonl"
    if pool_jsonl.exists() and not overwrite:
        print(f"[orchestrator] pool exists, skipping: {pool_jsonl}")
        return pool_jsonl
    cmd = [
        sys.executable,
        MUSIQUE_SCRIPTS / "sample_pool.py",
        "--config", require(cfg, "configs.musique_prep"),
        "--input", input_pool,
        "--size", size,
        "--balance", balance,
        "--seed", pool_seed,
        "--out-dir", out_dir,
    ]
    if balance == "clustered":
        # The clustered strategy embeds the input pool, so it takes the sweep's own
        # bi-encoder and device rather than the prep config's defaults — one sweep, one
        # embedding model (ADR 0021).
        cmd += [
            "--embed-model", require(cfg, "embed_model"),
            "--device", require(cfg, "device"),
        ]
    if overwrite:
        cmd.append("--overwrite")
    rc = _run_cmd(cmd, log_path=log_path, dry_run=dry_run)
    if rc != 0 and not dry_run:
        raise SystemExit(f"sample_pool.py failed rc={rc} ({size}/{balance}/trial{trial})")
    return pool_jsonl


def stage_similarity(
    cfg: dict[str, Any],
    paths: dict[str, Path],
    log_path: Path,
    dry_run: bool,
    overwrite: bool,
    *,
    dev_path: Path,
    pool_jsonl: Path,
    size: int,
    balance: str,
    trial: int,
    num_queries: int,
) -> Path:
    out_dir = _sim_dir(paths, size, balance, trial)
    _ensure_dir(out_dir)
    top20 = out_dir / "top20.jsonl"
    if top20.exists() and not overwrite:
        print(f"[orchestrator] top20 exists, skipping: {top20}")
        return top20
    cmd = [
        sys.executable,
        MUSIQUE_SCRIPTS / "check_question_similarity.py",
        "--config", require(cfg, "configs.similarity"),
        "--query-file", dev_path,
        "--pool-file", pool_jsonl,
        "--mode", "all",
        "--top-k", int(require(cfg, "similarity_top_k")),
        "--n", num_queries,
        "--device", require(cfg, "device"),
        "--embed-model", require(cfg, "embed_model"),
        "--run-dir", out_dir,
        "--out", top20,
    ]
    rc = _run_cmd(cmd, log_path=log_path, dry_run=dry_run)
    if rc != 0 and not dry_run:
        raise SystemExit(f"check_question_similarity.py failed rc={rc}")
    return top20


def stage_truncate(
    cfg: dict[str, Any],
    paths: dict[str, Path],
    log_path: Path,
    dry_run: bool,
    overwrite: bool,
    *,
    top20: Path,
    size: int,
    balance: str,
    trial: int,
) -> Path:
    out_dir = _bi_top5_dir(paths, size, balance, trial)
    _ensure_dir(out_dir)
    out = out_dir / "top5_biencoder.jsonl"
    if out.exists() and not overwrite:
        print(f"[orchestrator] top5_biencoder exists, skipping: {out}")
        return out
    cmd = [
        sys.executable,
        MUSIQUE_SCRIPTS / "truncate_top20.py",
        "--config", require(cfg, "configs.similarity"),
        "--input", top20,
        "--out", out,
        "--k", int(require(cfg, "retrieval_k")),
        "--run-dir", out_dir,
    ]
    rc = _run_cmd(cmd, log_path=log_path, dry_run=dry_run)
    if rc != 0 and not dry_run:
        raise SystemExit(f"truncate_top20.py failed rc={rc}")
    return out


def stage_rerank(
    cfg: dict[str, Any],
    paths: dict[str, Path],
    log_path: Path,
    dry_run: bool,
    overwrite: bool,
    *,
    top20: Path,
    size: int,
    balance: str,
    trial: int,
) -> Path:
    out_dir = _ce_dir(paths, size, balance, trial)
    _ensure_dir(out_dir)
    out = out_dir / "top5_ce.jsonl"
    if out.exists() and not overwrite:
        print(f"[orchestrator] top5_ce exists, skipping: {out}")
        return out
    cmd = [
        sys.executable,
        MUSIQUE_SCRIPTS / "rerank_similarity_results.py",
        "--config", require(cfg, "configs.similarity"),
        "--input", top20,
        "--out", out,
        "--cross-encoder", require(cfg, "cross_encoder"),
        "--rerank-k", int(require(cfg, "rerank_k")),
        "--device", require(cfg, "device"),
        "--run-dir", out_dir,
    ]
    rc = _run_cmd(cmd, log_path=log_path, dry_run=dry_run)
    if rc != 0 and not dry_run:
        raise SystemExit(f"rerank_similarity_results.py failed rc={rc}")
    return out


def _latest_results_subdir(root: Path) -> Path | None:
    """The most recent timestamped subdirectory containing results.json."""
    if not root.exists():
        return None
    candidates = sorted(
        (p for p in root.iterdir() if p.is_dir() and (p / "results.json").exists()),
        key=lambda p: p.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def _stabilise_results(decomp_out_dir: Path) -> Path | None:
    """Point ``<decomp_out_dir>/results.json`` at the newest timestamped run.

    The decomposer writes ``<output_root>/<run_id>/results.json``; downstream tooling
    wants a stable path, so symlink (or copy when symlinks fail) it one level up.
    """
    sub = _latest_results_subdir(decomp_out_dir)
    if sub is None:
        return None
    src = sub / "results.json"
    dst = decomp_out_dir / "results.json"
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    try:
        # A relative symlink keeps the run folder relocatable.
        dst.symlink_to(os.path.relpath(src, start=decomp_out_dir))
    except OSError:
        shutil.copy2(src, dst)
    return dst


def stage_decompose(
    cfg: dict[str, Any],
    paths: dict[str, Path],
    log_path: Path,
    dry_run: bool,
    overwrite: bool,
    *,
    retrieval_input: Path,
    size: int,
    balance: str,
    trial: int,
    variant: str,
    mode: str,
    pool_seed: int,
) -> Path:
    run_key = _run_key(size, balance, trial, variant, mode)
    out_root = _decomp_dir(paths, run_key)
    _ensure_dir(out_root)
    stable_results = out_root / "results.json"

    if stable_results.exists() and not overwrite:
        print(f"[orchestrator] decomposer results exist, skipping: {stable_results}")
        return stable_results

    cmd = [
        sys.executable,
        DECOMPOSER_RUNNER,
        "--config", require(cfg, "configs.decomposer"),
        "--model", require(cfg, "decomposer_model_folder"),
        "--seed", pool_seed,
        "--output-root", out_root,
        "--retrieval-input", retrieval_input,
        "--retrieval-mode", mode,
        "--retrieval-k", int(require(cfg, "retrieval_k")),
        "--embed-model", require(cfg, "embed_model"),
        "--quantization", require(cfg, "decomposer_quantization"),
    ]
    rc = _run_cmd(cmd, log_path=log_path, dry_run=dry_run, cwd=REPO_ROOT)
    if rc != 0 and not dry_run:
        raise SystemExit(f"run_decomposer.py failed rc={rc} ({run_key})")

    if not dry_run:
        res = _stabilise_results(out_root)
        if res is None:
            raise SystemExit(f"run_decomposer.py produced no results.json under {out_root}")
        return res
    return stable_results


def stage_eval(
    cfg: dict[str, Any],
    paths: dict[str, Path],
    gold_path: Path,
    log_path: Path,
    dry_run: bool,
    overwrite: bool,
    *,
    predictions: Path,
    size: int,
    balance: str,
    trial: int,
    variant: str,
    mode: str,
    pool_seed: int,
) -> Path:
    run_key = _run_key(size, balance, trial, variant, mode)
    out_dir = _eval_dir(paths, run_key)
    _ensure_dir(out_dir)
    metrics_path = out_dir / "eval_metrics.json"
    if metrics_path.exists() and not overwrite:
        print(f"[orchestrator] eval metrics exist, skipping: {metrics_path}")
        return metrics_path
    cmd = [
        sys.executable,
        EVALUATOR,
        "--config", require(cfg, "configs.musique_eval"),
        "--predictions", predictions,
        "--gold", gold_path,
        "--run-dir", out_dir,
        "--out-prefix", "eval",
        "--seed", pool_seed,
    ]
    rc = _run_cmd(cmd, log_path=log_path, dry_run=dry_run)
    if rc != 0 and not dry_run:
        raise SystemExit(f"evaluator failed rc={rc} ({run_key})")
    return metrics_path


# ------- summary csv -----------------------------------------------------


SUMMARY_FIELDS = [
    "run_key", "size", "balance", "trial", "pool_seed", "variant", "mode",
    # Which dev set every row was scored against. Rows from different dev samples are
    # not comparable, and all_runs.csv is exactly the table someone will later read as
    # if one row could be compared with the next.
    "dev_seed", "dev_per_hop", "dev_sample_sha256",
    "num_evaluated", "exact_match_rate",
    "step_precision_macro", "step_recall_macro", "step_f1_macro",
    "ordered_step_accuracy_macro",
    "rouge_l_precision_macro", "rouge_l_recall_macro", "rouge_l_f1_macro",
    "reference_validity_macro", "reference_validity_micro",
    "step_count_abs_error_mae",
    # The directional step-count family. all_runs.csv is where the under-decomposition
    # question gets asked, so these have to reach it and not stop at the per-run metrics
    # JSON. step_count_mae repeats step_count_abs_error_mae by design (see METRIC_DEFINITIONS).
    "step_count_mae", "mean_signed_step_count_error",
    "over_decomposition_rate", "under_decomposition_rate", "step_count_exact_rate",
    "hop_count_exact_match_rate", "hop_count_abs_error_mae",
    "composite_score",
    "eval_metrics_path", "predictions_path",
]

#: Identify the dev set a row was scored against. A mismatch between rows in one table
#: means the table is not a comparison.
DEV_IDENTITY_FIELDS = ("dev_seed", "dev_per_hop", "dev_sample_sha256")

_METRIC_FIELDS = [
    f for f in SUMMARY_FIELDS
    if f not in {"run_key", "size", "balance", "trial", "pool_seed", "variant", "mode",
                 "num_evaluated", "eval_metrics_path", "predictions_path",
                 *DEV_IDENTITY_FIELDS}
]


def _read_metrics(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[orchestrator] failed to read metrics {path}: {exc}")
        return None


def _existing_dev_identity(summary_csv: Path) -> dict[str, str] | None:
    """The dev identity already recorded in the table, or None when it is empty."""
    if not summary_csv.exists():
        return None
    try:
        with summary_csv.open(encoding="utf-8", newline="") as f:
            for rec in csv.DictReader(f):
                return {k: str(rec.get(k, "")) for k in DEV_IDENTITY_FIELDS}
    except Exception as exc:
        print(f"[orchestrator] failed to read {summary_csv} for a dev-identity check: {exc}")
        return None
    return None


def _existing_header(summary_csv: Path) -> list[str] | None:
    """The header already written to the table, or None when there is no file yet."""
    if not summary_csv.exists():
        return None
    try:
        with summary_csv.open(encoding="utf-8", newline="") as f:
            return next(csv.reader(f), None)
    except Exception as exc:
        print(f"[orchestrator] failed to read {summary_csv} for a header check: {exc}")
        return None


def _append_summary(summary_csv: Path, row: dict[str, Any]) -> None:
    """Append one row, refusing to mix dev sets or column sets inside one summary table."""
    _ensure_dir(summary_csv.parent)

    header = _existing_header(summary_csv)
    if header is not None and header != SUMMARY_FIELDS:
        raise SystemExit(
            f"[orchestrator] REFUSING to append to {summary_csv}: its header does not match "
            f"SUMMARY_FIELDS (this file was written by a different version of this script).\n"
            f"  missing from file: {[f for f in SUMMARY_FIELDS if f not in header]}\n"
            f"  only in file:      {[f for f in header if f not in SUMMARY_FIELDS]}\n"
            f"Appending would write values under the wrong columns. Point runs_subdir at a "
            f"fresh sweep root, or re-run the eval stage with --overwrite into a new table."
        )

    existing = _existing_dev_identity(summary_csv)
    incoming = {k: str(row.get(k, "")) for k in DEV_IDENTITY_FIELDS}
    if existing is not None and existing != incoming:
        raise SystemExit(
            f"[orchestrator] REFUSING to append to {summary_csv}: it already holds rows "
            f"scored against a different dev sample.\n"
            f"  existing: {existing}\n"
            f"  incoming: {incoming}\n"
            f"Rows from different dev samples are not comparable, and this file reads as "
            f"a comparison table. Either restore the original dev sample, or point "
            f"runs_subdir at a fresh sweep root so the new dev set gets its own table."
        )

    write_header = not summary_csv.exists()
    with summary_csv.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in SUMMARY_FIELDS})


def _summary_already_contains(summary_csv: Path, run_key: str) -> bool:
    if not summary_csv.exists():
        return False
    try:
        with summary_csv.open(encoding="utf-8", newline="") as f:
            for rec in csv.DictReader(f):
                if rec.get("run_key") == run_key:
                    return True
    except Exception:
        return False
    return False


# ------- main loop -------------------------------------------------------


def _num_queries_from_dev(dev_path: Path) -> int:
    n = 0
    with dev_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _filter_combos(combos: list[tuple[int, str]], only: list[str] | None) -> list[tuple[int, str]]:
    if not only:
        return combos
    tokens = {t.strip() for t in only if t.strip()}
    return [
        (size, balance)
        for size, balance in combos
        if {f"size{size}", balance, f"size{size}_{balance}"} & tokens
    ]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="pool_sweep.json")
    p.add_argument("--stage", choices=STAGES, default="all", help="Restrict to one stage (default: all).")
    p.add_argument(
        "--only",
        default=None,
        help="Comma-separated tokens restricting combos, e.g. size1000,balanced.",
    )
    p.add_argument("--trials", default=None, help="Comma-separated trial indices, e.g. 0,1")
    p.add_argument("--variants", default=None, help="Comma-separated retriever variants")
    p.add_argument("--modes", default=None, help="Comma-separated retrieval modes")
    p.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    p.add_argument("--overwrite", action="store_true", help="Recompute every stage even if outputs exist.")
    return p.parse_args()


def _split_csv(arg: str | None) -> list[str] | None:
    if not arg:
        return None
    return [t.strip() for t in arg.split(",") if t.strip()]


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    paths_cfg = load_paths(require(cfg, "paths_config"))
    data_root = Path(paths_cfg["data_root_resolved"])

    input_pool = resolve_path(
        require(paths_cfg, "datasets." + require(cfg, "input_pool_key")), data_root
    )
    gold_path = resolve_path(
        require(paths_cfg, "datasets." + require(cfg, "gold_key")), data_root
    )

    paths = _paths(cfg, paths_cfg)
    _ensure_dir(paths["runs_root"])
    log_path = paths["log_file"]
    _log_append(log_path, f"[{now_iso()}] === orchestrator start ({args.stage}) ===")

    summary_csv = paths["summary_dir"] / "all_runs.csv"

    dev_path = stage_sample_dev(cfg, paths, log_path, args.dry_run, args.overwrite)
    num_queries = (
        _num_queries_from_dev(dev_path)
        if dev_path.exists()
        else int(require(cfg, "dev_per_hop")) * 3
    )
    dev_identity = {
        "dev_seed": int(require(cfg, "dev_seed")),
        "dev_per_hop": int(require(cfg, "dev_per_hop")),
        "dev_sample_sha256": _sha256_file(dev_path),
    }
    print(f"[orchestrator] dev sample: {dev_path} ({num_queries} queries)")
    print(f"[orchestrator] dev identity: {dev_identity}")

    combos = _filter_combos(_grid(cfg), _split_csv(args.only))
    trial_seeds = _trial_seeds(cfg)
    trial_filter = _split_csv(args.trials)
    variants = _split_csv(args.variants) or list(require(cfg, "retriever_variants"))
    modes = _split_csv(args.modes) or list(require(cfg, "retrieval_modes"))

    print(f"[orchestrator] input pool: {input_pool}")
    print(f"[orchestrator] combos: {combos}")
    print(f"[orchestrator] trials: {trial_seeds} (filter={trial_filter})")
    print(f"[orchestrator] variants x modes: {variants} x {modes}")

    do_stage = {s: (args.stage in ("all", s)) for s in STAGES}
    rows_appended = 0

    for size, balance in combos:
        for t_idx, pool_seed in enumerate(trial_seeds):
            if trial_filter and str(t_idx) not in trial_filter:
                continue
            print(f"\n[orchestrator] === cell {_cell_key(size, balance, t_idx)} (pool_seed={pool_seed}) ===")

            if do_stage["sample_pool"]:
                pool_jsonl = stage_sample_pool(
                    cfg, paths, input_pool, log_path, args.dry_run, args.overwrite,
                    size=size, balance=balance, trial=t_idx, pool_seed=pool_seed,
                )
            else:
                pool_jsonl = _pool_dir(paths, size, balance, t_idx, pool_seed) / "pool.jsonl"

            if do_stage["similarity"]:
                top20 = stage_similarity(
                    cfg, paths, log_path, args.dry_run, args.overwrite,
                    dev_path=dev_path, pool_jsonl=pool_jsonl,
                    size=size, balance=balance, trial=t_idx, num_queries=num_queries,
                )
            else:
                top20 = _sim_dir(paths, size, balance, t_idx) / "top20.jsonl"

            if do_stage["truncate"]:
                top5_bi = stage_truncate(
                    cfg, paths, log_path, args.dry_run, args.overwrite,
                    top20=top20, size=size, balance=balance, trial=t_idx,
                )
            else:
                top5_bi = _bi_top5_dir(paths, size, balance, t_idx) / "top5_biencoder.jsonl"

            if do_stage["rerank"]:
                top5_ce = stage_rerank(
                    cfg, paths, log_path, args.dry_run, args.overwrite,
                    top20=top20, size=size, balance=balance, trial=t_idx,
                )
            else:
                top5_ce = _ce_dir(paths, size, balance, t_idx) / "top5_ce.jsonl"

            variant_to_input = {
                "biencoder_only": top5_bi,
                "biencoder_plus_ce": top5_ce,
            }

            for variant in variants:
                if variant not in variant_to_input:
                    print(f"[orchestrator] unknown variant skipped: {variant}")
                    continue
                retrieval_input = variant_to_input[variant]
                for mode in modes:
                    run_key = _run_key(size, balance, t_idx, variant, mode)

                    if do_stage["decompose"]:
                        predictions = stage_decompose(
                            cfg, paths, log_path, args.dry_run, args.overwrite,
                            retrieval_input=retrieval_input,
                            size=size, balance=balance, trial=t_idx,
                            variant=variant, mode=mode, pool_seed=pool_seed,
                        )
                    else:
                        predictions = _decomp_dir(paths, run_key) / "results.json"

                    if do_stage["eval"]:
                        metrics_path = stage_eval(
                            cfg, paths, gold_path, log_path, args.dry_run, args.overwrite,
                            predictions=predictions,
                            size=size, balance=balance, trial=t_idx,
                            variant=variant, mode=mode, pool_seed=pool_seed,
                        )
                    else:
                        metrics_path = _eval_dir(paths, run_key) / "eval_metrics.json"

                    if args.dry_run:
                        continue

                    metrics = _read_metrics(metrics_path)
                    if metrics is None:
                        continue
                    if _summary_already_contains(summary_csv, run_key) and not args.overwrite:
                        print(f"[orchestrator] summary row already present: {run_key}")
                        continue
                    row: dict[str, Any] = {
                        "run_key": run_key,
                        "size": size,
                        "balance": balance,
                        "trial": t_idx,
                        "pool_seed": pool_seed,
                        "variant": variant,
                        "mode": mode,
                        **dev_identity,
                        "num_evaluated": metrics.get("total_evaluated"),
                        "eval_metrics_path": str(metrics_path),
                        "predictions_path": str(predictions),
                    }
                    for field in _METRIC_FIELDS:
                        row[field] = metrics.get(field)
                    _append_summary(summary_csv, row)
                    rows_appended += 1
                    print(f"[orchestrator] summary row added: {run_key}")

    _log_append(log_path, f"[{now_iso()}] === orchestrator end ===")

    # Sweep-level trail. The per-stage runs each leave their own artifacts, but the
    # grid that produced them lived only in the config and the CLI filters; without
    # this, all_runs.csv has no record of which grid it came from.
    grid_snapshot = {
        "script": Path(__file__).name,
        "created_utc": now_iso(),
        "config_path": cfg.get("_config_path"),
        "stage": args.stage,
        "dry_run": args.dry_run,
        "overwrite": args.overwrite,
        "grid": {
            "imbalanced_sizes": require(cfg, "imbalanced_sizes"),
            "balanced_sizes": require(cfg, "balanced_sizes"),
            "clustered_sizes": require(cfg, "clustered_sizes"),
            "combos_run": [[size, balance] for size, balance in combos],
            "retriever_variants": variants,
            "retrieval_modes": modes,
            "trial_seeds": trial_seeds,
            "trial_filter": trial_filter,
        },
        "dev_identity": dev_identity,
        "dev_sample_path": str(dev_path),
        "dev_num_queries": num_queries,
        "input_pool": str(input_pool),
        "gold": str(gold_path),
        "decomposer_model_folder": require(cfg, "decomposer_model_folder"),
        "decomposer_quantization": require(cfg, "decomposer_quantization"),
        "embed_model": require(cfg, "embed_model"),
        "cross_encoder": require(cfg, "cross_encoder"),
        "similarity_top_k": require(cfg, "similarity_top_k"),
        "rerank_k": require(cfg, "rerank_k"),
        "retrieval_k": require(cfg, "retrieval_k"),
        "device": require(cfg, "device"),
        "child_configs": require(cfg, "configs"),
    }
    expected_cells = len(combos) * len(
        [t for t in range(len(trial_seeds)) if not trial_filter or str(t) in trial_filter]
    )
    write_run_artifacts(
        paths["summary_dir"],
        config_snapshot=grid_snapshot,
        metrics={
            "script": Path(__file__).name,
            "created_utc": now_iso(),
            "stage": args.stage,
            "dry_run": args.dry_run,
            "cells_in_grid": expected_cells,
            "runs_in_grid": expected_cells * len(variants) * len(modes),
            "summary_rows_appended_this_invocation": rows_appended,
            "summary_csv": str(summary_csv),
            "orchestrator_log": str(log_path),
            "dev_identity": dev_identity,
        },
        note_title=f"Pool sweep orchestration ({args.stage})",
        note_lines=[
            f"- Grid: {[[s, b] for s, b in combos]} x {variants} x {modes}",
            f"- Trial seeds: {trial_seeds} (filter={trial_filter})",
            f"- Dev sample: `{dev_path}` ({num_queries} queries), identity {dev_identity}",
            f"- Decomposer: `{require(cfg, 'decomposer_model_folder')}` "
            f"(quantization {require(cfg, 'decomposer_quantization')})",
            f"- Retrieval: {require(cfg, 'embed_model')} top-{require(cfg, 'similarity_top_k')} "
            f"-> k={require(cfg, 'retrieval_k')}; cross-encoder "
            f"`{require(cfg, 'cross_encoder')}` rerank-k {require(cfg, 'rerank_k')}",
            f"- Summary rows appended this invocation: {rows_appended}",
            f"- Summary table: `{summary_csv}`",
            "- Rows in one summary table are only comparable while dev_identity matches; "
            "the appender refuses to mix dev samples.",
        ],
        prefix="sweep_",
    )

    print(f"\n[orchestrator] done. Log: {log_path}")
    print(f"[orchestrator] summary: {summary_csv}")


if __name__ == "__main__":
    main()
