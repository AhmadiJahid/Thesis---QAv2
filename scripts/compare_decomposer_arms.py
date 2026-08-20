#!/usr/bin/env python3
"""Score decomposer arms on the same evaluation set and put cost next to quality.

Glue for the prompting-versus-fine-tuning comparison (issue #13). It does not compute a
single metric of its own: quality comes from ``scripts/musique_decompositions_evaluator.py``
(string-level metrics, no model in the loop, and never a commercial API rating a
decomposition - CLAUDE.md standing constraint), significance from that script's
``--compare`` (paired bootstrap + McNemar, with the additive paired t-test of issue #30
reported next to them), and cost from the ``cost`` block each decomposer
run writes into its own ``metrics.json``. This script runs those in one pass and writes one
table.

Usage::

    python scripts/compare_decomposer_arms.py \\
        --arm prompting=runs/decomposer/mistral_7b_instruct/<run> \\
        --arm finetuned_pool_2000=runs/finetune_decomposer/eval/pool_2000/<run> \\
        --baseline prompting --eval-arm pool_2000

    # the generalisation arm: both sides must be 4-hop only, and this refuses otherwise
    python scripts/compare_decomposer_arms.py \\
        --arm prompting_4hop=runs/decomposer/mistral_7b_instruct/<run> \\
        --arm finetuned_2_3hop=runs/decomposer/mistral_7b_instruct/<run> \\
        --baseline prompting_4hop --eval-arm generalisation_2_3hop

Each ``--arm NAME=DIR`` names a decomposer run directory holding ``results.json`` (and
``metrics.json``, for the cost block). Every non-baseline arm is compared against the
baseline, with the difference reported as ``arm minus baseline``.

``--eval-arm`` names an arm of ``configs/finetune_decomposer.json``, and its ``eval_hops`` is
**enforced**: every scored item's id hop prefix, on every side, must be a hop that arm is
evaluated on. Without that check the arm's ``eval_hops`` is decorative, and a generalisation
run (trained on 2-hop and 3-hop, claimed against 4-hop) could be scored on 2/3-hop rows with
nothing in the output saying so.

Two things it deliberately does not do:

- **It reads the evaluator's per-item file only for item ids.** Nothing else in it is
  parsed: the shape of record is the versioned object of ADR 0011 section 2
  (``{"schema": "musique_decomposition_per_item/1", ..., "items": [...]}``), and what the
  items say is the evaluator's business, not this script's. The id reader also tolerates a
  bare top-level list, but that pre-PR-#22 shape is retired: ``--compare``, which this
  script runs on every non-baseline arm, refuses it, so such an arm cannot reach a
  comparison here - it must be re-scored.
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

from finetune_data import hop_from_id, resolve_arm  # noqa: E402
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


def per_item_ids(path: Path) -> list[str]:
    """The ``item_id`` of every scored item, in file order.

    Reads the versioned object of ADR 0011 section 2
    (``{"schema": "musique_decomposition_per_item/1", ..., "items": [...]}``). A bare
    top-level list is still tolerated *here* (only ids are read, and this check should not
    be the thing that fails on an old file), but that shape is retired: the evaluator's
    ``--compare`` refuses it, so an arm scored under it cannot be compared without re-scoring.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("items") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise SystemExit(
            f"{path}: expected a per-item JSON list, or an object with an 'items' list "
            f"(as written by the evaluator); got {type(rows).__name__}"
        )
    ids: list[str] = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or "item_id" not in row:
            raise SystemExit(f"{path}: per-item row {i} has no 'item_id'")
        ids.append(str(row["item_id"]))
    return ids


def assert_items_within_eval_hops(
    arm_name: str,
    per_item_path: Path,
    item_ids: list[str],
    eval_hops: list[int],
    *,
    max_reported: int,
) -> dict[str, Any]:
    """Refuse when a scored item is not from a hop depth this comparison is defined on.

    ``eval_hops`` comes from the arm in ``configs/finetune_decomposer.json`` named by
    ``--eval-arm``, and it is checked on **every** side, baseline included: a fine-tuned arm
    scored on 4-hop rows against a baseline scored on all 600 is not a comparison, and the
    generalisation claim (learned the task, did not memorise hop patterns) rests entirely on
    the 4-hop restriction actually holding.
    """
    wanted = sorted({int(h) for h in eval_hops})
    if not wanted:
        raise SystemExit(
            f"[arms] arm {arm_name!r}: eval_hops is empty in the config, so there is no hop "
            "restriction to check against. Fix the arm definition."
        )
    if not item_ids:
        raise SystemExit(
            f"[arms] arm {arm_name!r}: {per_item_path} holds no scored items, so the "
            "eval-hop check would pass without checking anything (and there is nothing to "
            "compare). Re-score the arm."
        )

    hop_counts: dict[str, int] = {}
    off_hop: list[str] = []
    unparseable: list[str] = []
    for item_id in item_ids:
        hop = hop_from_id(item_id)
        if hop is None:
            unparseable.append(item_id)
            continue
        hop_counts[str(hop)] = hop_counts.get(str(hop), 0) + 1
        if hop not in wanted:
            off_hop.append(item_id)

    def _listing(ids: list[str]) -> str:
        shown = ids[:max_reported]
        more = "" if len(ids) <= max_reported else f"\n  ... (+{len(ids) - max_reported} more)"
        return "\n  " + "\n  ".join(shown) + more

    if unparseable:
        raise SystemExit(
            f"[arms] REFUSING TO COMPARE: arm {arm_name!r} has {len(unparseable)} scored "
            f"item(s) whose id carries no MuSiQue hop prefix (expected '2hop__...', "
            f"'3hop1__...', '4hop2__...'), so they cannot be checked against eval_hops "
            f"{wanted}. Offending "
            f"ids:{_listing(unparseable)}\n"
            f"  per-item file: {per_item_path}"
        )
    if off_hop:
        raise SystemExit(
            f"[arms] REFUSING TO COMPARE: arm {arm_name!r} was scored on "
            f"{len(off_hop)} item(s) outside eval_hops {wanted} (hop counts in the file: "
            f"{ {k: hop_counts[k] for k in sorted(hop_counts, key=int)} }). Offending "
            f"ids:{_listing(off_hop)}\n"
            f"  per-item file: {per_item_path}\n"
            "Restrict this arm's predictions to the hops the --eval-arm is evaluated on (for "
            "the generalisation arm that is 4-hop only, on BOTH sides), or pass an "
            "--eval-arm whose eval_hops covers what was actually scored. Scoring a "
            "hop-restricted arm on other hops makes the number answer a different question."
        )
    return {
        "eval_hops": wanted,
        "items_checked": len(item_ids),
        "hop_counts": {k: hop_counts[k] for k in sorted(hop_counts, key=int)},
        "asserted": True,
    }


def cost_definition_groups(
    per_arm: dict[str, dict[str, Any]],
) -> list[tuple[list[str], dict[str, Any]]]:
    """Group arm names by the cost definitions their runs recorded.

    The cost columns of the table are per-query aggregates; what they aggregate (a prompt
    token, a latency second) is defined by the decomposer run that measured it, and each run
    writes those definitions into its own ``cost`` block. Printing them next to the numbers
    is what makes the table readable on its own.

    Normally there is exactly one group: every arm was produced by the same runner. More than
    one group is the interesting case and is reported as such - it means a cost column does
    not mean the same thing on both sides of the comparison.
    """
    groups: dict[str, tuple[list[str], dict[str, Any]]] = {}
    for name, record in per_arm.items():
        definitions = record.get("cost_definitions")
        if not isinstance(definitions, dict) or not definitions:
            continue
        key = json.dumps(definitions, sort_keys=True, default=str)
        groups.setdefault(key, ([], definitions))[0].append(name)
    return list(groups.values())


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
    p.add_argument(
        "--eval-arm",
        required=True,
        help="Arm key in the config whose eval_hops every side of this comparison is checked "
        "against (e.g. pool_2000 for hops 2/3/4, generalisation_2_3hop for 4-hop only).",
    )
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
    max_reported_ids = int(require(eval_cfg, "max_reported_ids"))

    eval_arm = resolve_arm(cfg, args.eval_arm)
    eval_hops = [int(h) for h in require(eval_arm, "eval_hops")]
    print(f"[arms] eval-arm={args.eval_arm} eval_hops={eval_hops} (enforced on every side)")

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
        per_item_path = arm_out / f"{score_prefix}_per_item.json"
        # Before any significance test: the items actually scored must be the hops this
        # comparison is defined on. Cheap, and it fails before the bootstrap runs.
        hop_record = assert_items_within_eval_hops(
            name,
            per_item_path,
            per_item_ids(per_item_path),
            eval_hops,
            max_reported=max_reported_ids,
        )
        cost, cost_note = _cost_block(run_dir)
        per_arm[name] = {
            "run_dir": str(run_dir),
            "eval_dir": str(arm_out),
            "per_item_path": str(per_item_path),
            "eval_hops_check": hop_record,
            "quality": {key: arm_metrics.get(key) for key in quality_keys},
            "cost": ({key: cost.get(key) for key in cost_keys} if cost else None),
            # The run's own definitions of what its cost numbers measure, carried through so
            # this comparison's metrics and note state it too.
            "cost_definitions": (cost.get("definitions") if cost else None),
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
            # Added alongside the two above by issue #30 / ADR 0017; read with .get for the
            # same reason 'underpowered' is - a comparison produced before it existed has no
            # such block, and that is not this script's failure.
            "t_test": compare_metrics.get("t_test"),
        }

    # ---- one table: quality columns, then cost columns ----
    note_lines = [
        f"- Evaluator: `{evaluator}` with config `{evaluator_cfg.get('_config_path')}`",
        f"- Baseline arm: `{args.baseline}`; differences below are arm minus baseline.",
        f"- Eval-hop restriction: `--eval-arm {args.eval_arm}` -> hops {eval_hops}, asserted "
        f"on every side ("
        + "; ".join(
            f"{name}: {record['eval_hops_check']['items_checked']} items "
            f"{record['eval_hops_check']['hop_counts']}"
            for name, record in per_arm.items()
        )
        + ")",
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

    # What the cost columns mean, printed where they are reported: the numbers above are
    # per-query aggregates (mean/median over the rows that generated; a row with no
    # measurement, e.g. from a --dry-run, is excluded), and each arm's run recorded what the
    # per-row measurements are.
    note_lines.append(
        "- Cost columns are per-query aggregates of each arm's own `cost` block: "
        "`mean_*_per_query` / `median_*_per_query` over `rows_measured` rows, "
        "`total_generation_seconds` summed over the same rows. The per-row measurements they "
        "aggregate, as defined by the run that measured them:"
    )
    groups = cost_definition_groups(per_arm)
    for names, definitions in groups:
        if len(names) == len(per_arm):
            scope = "all arms"
        else:
            label = "arm" if len(names) == 1 else "arms"
            scope = f"{label} " + ", ".join(f"`{n}`" for n in names)
        note_lines.append(f"  - {scope}:")
        for key in sorted(definitions):
            note_lines.append(f"    - `{key}`: {definitions[key]}")
    if len(groups) > 1:
        note_lines.append(
            "  - NOTE: the arms above do not agree on what their cost numbers measure, so "
            "the cost columns are not comparable across them as they stand."
        )
    undefined = [name for name, record in per_arm.items() if not record.get("cost_definitions")]
    if undefined:
        note_lines.append(
            "  - no cost definitions recorded by: "
            + ", ".join(f"`{n}`" for n in undefined)
            + " (no cost block in that run's metrics.json, so its cost cells are unmeasured)"
        )
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
        for stat, row in (record["t_test"] or {}).items():
            # A degenerate row has no t and no p (e.g. every per-item difference identical);
            # the evaluator says why in 'degenerate' and makes no significance claim.
            stat_cell = (
                f"t={row['t_statistic']:+.4f} (dof={row['degrees_of_freedom']}) "
                f"p={row['p_value']:.4g}"
                if row.get("degenerate") is None
                else (
                    f"t=unmeasured (dof={row['degrees_of_freedom']}; t undefined, reason in "
                    f"the comparison's metrics JSON)"
                )
            )
            note_lines.append(
                f"  - paired t-test {stat}: {row['difference']:+.4f} {stat_cell} "
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
        "eval_arm": args.eval_arm,
        "eval_hops": eval_hops,
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
        "eval_arm": args.eval_arm,
        "eval_arm_spec": eval_arm,
        "eval_hops": eval_hops,
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
