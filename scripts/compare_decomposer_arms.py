#!/usr/bin/env python3
"""Score decomposer arms on the same evaluation set and put cost next to quality.

Glue for the prompting-versus-fine-tuning comparison (issue #13). It does not compute a
single metric of its own: quality comes from ``scripts/musique_decompositions_evaluator.py``
(string-level metrics, no model in the loop, and never a commercial API rating a
decomposition - CLAUDE.md standing constraint), significance from that script's
``--compare`` (paired bootstrap + McNemar), and cost from the ``cost`` block each decomposer
run writes into its own ``metrics.json``. This script runs those in one pass and writes one
table.

Usage::

    python scripts/compare_decomposer_arms.py \\
        --arm prompting=runs/decomposer/mistral_7b_instruct/<run> \\
        --arm finetuned_pool_2000=runs/finetune_decomposer/eval/pool_2000/<run> \\
        --baseline prompting

Each ``--arm NAME=DIR`` names a decomposer run directory holding ``results.json`` (and
``metrics.json``, for the cost block). Every non-baseline arm is compared against the
baseline, with the difference reported as ``arm minus baseline``.

Two things it deliberately does not do:

- **It never parses the evaluator's per-item file.** It only passes the paths through, so
  the per-item schema change of PR #22 (a stamped object instead of a bare list) is the
  evaluator's business and cannot break this script either side of that merge.
- **It never compares runs scored under different settings.** All arms are scored by one
  invocation config, which is also what the evaluator's ``--compare`` requires.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from run_artifacts import now_iso, run_id, write_run_artifacts  # noqa: E402
from run_config import load_config, load_paths, require, runs_path  # noqa: E402


def _parse_arm(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise SystemExit(f"--arm expects NAME=DIR, got {value!r}")
    name, raw = value.split("=", 1)
    name = name.strip()
    if not name:
        raise SystemExit(f"--arm expects a non-empty NAME in {value!r}")
    path = Path(raw.strip())
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return name, path


def _run(cmd: list[str]) -> None:
    printable = " ".join(str(c) for c in cmd)
    print(f"[arms] $ {printable}", flush=True)
    proc = subprocess.run([str(c) for c in cmd], cwd=str(_REPO_ROOT))
    if proc.returncode != 0:
        raise SystemExit(f"[arms] command failed (exit {proc.returncode}): {printable}")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _cost_block(run_dir: Path) -> tuple[dict[str, Any] | None, str | None]:
    """The arm's own cost block, or a note saying why there is none."""
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return None, f"unmeasured: no metrics.json in {run_dir}"
    payload = _read_json(metrics_path)
    cost = payload.get("cost")
    if not isinstance(cost, dict):
        return None, (
            f"unmeasured: {metrics_path} has no 'cost' block (a run from before cost "
            "reporting existed, or a --dry-run)"
        )
    return cost, None


def _fmt(value: Any) -> str:
    if value is None:
        return "unmeasured"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="NAME=DIR",
        help="A decomposer run directory to score. Repeatable.",
    )
    p.add_argument("--baseline", required=True, help="Name of the arm every other arm is compared to")
    p.add_argument("--config", default="finetune_decomposer.json", help="Committed config")
    p.add_argument("--evaluator-config", default=None, help="Override evaluation.evaluator_config")
    p.add_argument("--out-dir", type=Path, default=None, help="Override the output directory")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    paths_cfg = load_paths(require(cfg, "paths_config"))
    eval_cfg = require(cfg, "evaluation")

    evaluator = _REPO_ROOT / require(eval_cfg, "evaluator_script")
    if not evaluator.exists():
        raise SystemExit(f"evaluator script not found: {evaluator}")
    evaluator_config = args.evaluator_config or require(eval_cfg, "evaluator_config")
    evaluator_cfg = load_config(evaluator_config)
    score_prefix = require(evaluator_cfg, "out_prefix")
    compare_prefix = require(evaluator_cfg, "paired_comparison.out_prefix")

    arms = dict(_parse_arm(value) for value in args.arm)
    if args.baseline not in arms:
        raise SystemExit(
            f"--baseline {args.baseline!r} is not one of the arms: {sorted(arms)}"
        )
    if len(arms) < 2:
        raise SystemExit("pass at least two --arm entries: one baseline and one to compare")
    for name, run_dir in arms.items():
        if not (run_dir / "results.json").exists():
            raise SystemExit(f"arm {name!r}: no results.json in {run_dir}")

    current_run_id = run_id()
    out_dir = (
        args.out_dir
        if args.out_dir is not None
        else runs_path(paths_cfg, require(eval_cfg, "out_subdir"), current_run_id)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    quality_keys = list(require(eval_cfg, "quality_keys"))
    cost_keys = list(require(eval_cfg, "cost_keys"))

    # ---- score every arm with one evaluator config ----
    per_arm: dict[str, dict[str, Any]] = {}
    for name, run_dir in arms.items():
        arm_out = out_dir / name
        _run(
            [
                sys.executable,
                evaluator,
                "--config",
                evaluator_config,
                "--predictions",
                run_dir / "results.json",
                "--run-dir",
                arm_out,
            ]
        )
        arm_metrics = _read_json(arm_out / f"{score_prefix}_metrics.json")
        cost, cost_note = _cost_block(run_dir)
        per_arm[name] = {
            "run_dir": str(run_dir),
            "eval_dir": str(arm_out),
            "per_item_path": str(arm_out / f"{score_prefix}_per_item.json"),
            "quality": {key: arm_metrics.get(key) for key in quality_keys},
            "cost": ({key: cost.get(key) for key in cost_keys} if cost else None),
            "cost_note": cost_note,
        }

    # ---- paired significance of every arm against the baseline ----
    comparisons: dict[str, dict[str, Any]] = {}
    for name in sorted(k for k in arms if k != args.baseline):
        compare_dir = out_dir / f"compare_{name}_vs_{args.baseline}"
        _run(
            [
                sys.executable,
                evaluator,
                "--config",
                evaluator_config,
                "--compare",
                per_arm[name]["per_item_path"],
                per_arm[args.baseline]["per_item_path"],
                "--run-dir",
                compare_dir,
            ]
        )
        compare_metrics = _read_json(compare_dir / f"{compare_prefix}_metrics.json")
        comparisons[f"{name}_vs_{args.baseline}"] = {
            "system_a": name,
            "system_b": args.baseline,
            "compare_dir": str(compare_dir),
            "num_aligned_items": compare_metrics.get("num_aligned_items"),
            "significance_floor": compare_metrics.get("significance_floor"),
            "bootstrap": compare_metrics.get("bootstrap"),
            "mcnemar": compare_metrics.get("mcnemar"),
        }

    # ---- one table: quality columns, then cost columns ----
    note_lines = [
        f"- Evaluator: `{evaluator}` with config `{evaluator_cfg.get('_config_path')}`",
        f"- Baseline arm: `{args.baseline}`; differences below are arm minus baseline.",
        "",
        "| arm | " + " | ".join(quality_keys) + " | " + " | ".join(cost_keys) + " |",
        "|---" * (1 + len(quality_keys) + len(cost_keys)) + "|",
    ]
    for name, record in per_arm.items():
        cells = [_fmt(record["quality"].get(k)) for k in quality_keys]
        cells += [
            _fmt(record["cost"].get(k)) if record["cost"] else "unmeasured" for k in cost_keys
        ]
        note_lines.append(f"| {name} | " + " | ".join(cells) + " |")
    note_lines.append("")

    for key, record in comparisons.items():
        note_lines.append(
            f"- `{key}`: {record['num_aligned_items']} aligned items; see "
            f"`{record['compare_dir']}/{compare_prefix}_notes.md` for the full table."
        )
        # 'underpowered' is read with .get: the evaluator gained it in PR #22, and this
        # script must not fall over on a comparison produced before that.
        for stat, row in (record["bootstrap"] or {}).items():
            note_lines.append(
                f"  - bootstrap {stat}: {row['difference']:+.4f} "
                f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] "
                f"significant={row['significant']} "
                f"underpowered={_fmt(row.get('underpowered'))}"
            )
        for stat, row in (record["mcnemar"] or {}).items():
            note_lines.append(
                f"  - McNemar {stat}: {row['difference']:+.4f} p={row['p_value']:.4g} "
                f"significant={row['significant']} "
                f"underpowered={_fmt(row.get('underpowered'))}"
            )
    for name, record in per_arm.items():
        if record["cost_note"]:
            note_lines.append(f"- Cost for `{name}`: {record['cost_note']}")
    note_lines.append(
        "- Quality is string-level throughout (see docs/METRICS.md); no model, and in "
        "particular no commercial API, rates any decomposition in this path."
    )

    metrics = {
        "script": Path(__file__).name,
        "created_utc": now_iso(),
        "run_id": current_run_id,
        "baseline": args.baseline,
        "difference_direction": "arm minus baseline",
        "evaluator_script": str(evaluator),
        "evaluator_config": evaluator_cfg.get("_config_path"),
        "quality_keys": quality_keys,
        "cost_keys": cost_keys,
        "arms": per_arm,
        "comparisons": comparisons,
    }
    snapshot = {
        "script": Path(__file__).name,
        "created_utc": now_iso(),
        "run_id": current_run_id,
        "config": cfg.get("_config_path"),
        "evaluator_config": evaluator_cfg.get("_config_path"),
        "arms": {name: str(path) for name, path in arms.items()},
        "baseline": args.baseline,
        "out_dir": str(out_dir),
    }
    write_run_artifacts(
        out_dir,
        config_snapshot=snapshot,
        metrics=metrics,
        note_title="Decomposer arm comparison (quality and cost)",
        note_lines=note_lines,
        prefix="arms_",
    )
    print("\n".join(note_lines))
    print(f"\n[arms] summary -> {out_dir}")


if __name__ == "__main__":
    main()
