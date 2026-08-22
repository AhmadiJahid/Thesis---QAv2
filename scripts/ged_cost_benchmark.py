#!/usr/bin/env python3
"""
Micro-benchmark behind the GED cost table in ADR 0026.

``break_metrics.ged.max_nodes_for_optimizer`` in ``configs/musique_eval.json`` is a number
picked from measurement: above the cap the evaluator skips networkx's optimizer and reports
a search-free upper bound instead. The measurement that justifies the cap used to live only
as prose in the ADR, with the shapes *named* rather than defined — so nobody could reproduce
a row (PR #44 review, residual finding 1). This script is that measurement: it constructs
every shape explicitly (exact step texts, exact reference pattern), times the optimizer on
it against the committed fixture gold, and prints the table in the form the ADR carries.

It is a **tool, not an experiment**: it scores no system, touches no run, and therefore has
no ``experiments/log.md`` entry. What it prints is a property of this machine and this
implementation, so it prints the commit it was run at (and whether the tree was dirty) —
that SHA is what the ADR records beside the numbers.

Every metric function is imported from ``scripts/musique_decompositions_evaluator.py``, so
the cost measured here is the cost the evaluator actually pays; there is no second copy of
the metric.

Run::

    .venv/bin/python scripts/ged_cost_benchmark.py                  # the full table
    .venv/bin/python scripts/ged_cost_benchmark.py --max-node-count 8   # tiny smoke run
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import networkx as nx  # noqa: E402

from run_artifacts import now_iso  # noqa: E402
from run_config import load_config, require, resolve_path  # noqa: E402
from seeding import set_global_seed  # noqa: E402

EVALUATOR = REPO_ROOT / "scripts" / "musique_decompositions_evaluator.py"

#: Effectively "no cap": this tool measures what the optimizer costs when it is *not*
#: skipped, because that cost is the reason the cap exists.
_NO_NODE_CAP = 10**9


def _import_evaluator() -> Any:
    """Import the evaluator by path, as tests/test_decomposition_metrics.py does."""
    name = "musique_decompositions_evaluator"
    spec = importlib.util.spec_from_file_location(name, EVALUATOR)
    if spec is None or spec.loader is None:  # pragma: no cover - import plumbing
        raise SystemExit(f"cannot import {EVALUATOR}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


EVAL = _import_evaluator()


# --------------------------------------------------------------------------------------
# The shapes. Each one is a function from a node count to a list of step texts, written
# out literally: these texts ARE the measurement's definition, and the ADR's table cannot
# be re-derived from a description of them.
# --------------------------------------------------------------------------------------

#: The single sentence every step of the ``repeated_step_text`` shape carries. Its point is
#: maximal label ambiguity: N identical labels means the optimizer sees N! equally-good node
#: alignments, which is what makes the search branch. The sentence itself is arbitrary; only
#: its being identical across steps, and its token length, matter.
_REPEATED_STEP_TEXT = "Who leads the organisation?"

#: Step 1 of the reference-carrying shapes, so that ``#1`` always names a real step.
_CHAIN_FIRST_STEP = "Who published Quiet Ledger?"


def _shape_repeated_step_text(nodes: int, gold_steps: list[str]) -> list[str]:
    """``nodes`` copies of one sentence, no references.

    Graph: ``nodes`` nodes, 0 edges, every label identical.
    """
    return [_REPEATED_STEP_TEXT] * nodes


def _shape_gold_step_texts_repeated(nodes: int, gold_steps: list[str]) -> list[str]:
    """The gold's own step texts, cycled until there are ``nodes`` of them.

    Verbatim, so the gold's ``[#k]`` references come with them: a 4-hop gold contributes
    ``[#1]``, ``[#2]``, ``[#3]``, which become edges into steps 1-3. This is the shape that
    is hardest for the optimizer in a realistic way — the labels are the ones it is being
    compared against, so every alignment is nearly as good as every other.
    """
    return [gold_steps[i % len(gold_steps)] for i in range(nodes)]


def _shape_chain_shaped(nodes: int, gold_steps: list[str]) -> list[str]:
    """A linear chain: step k references step k-1, so the graph is a path.

    Step 1 is :data:`_CHAIN_FIRST_STEP`; step k (k >= 2) is ``"Who founded [#k-1]?"``. The
    labels differ only in the reference index, which the Break rewrite turns into
    ``@@k-1@@`` — so this shape is *less* ambiguous than the repeated-text ones by exactly
    one token per step. Graph: ``nodes`` nodes, ``nodes - 1`` edges.
    """
    steps = [_CHAIN_FIRST_STEP]
    steps.extend(f"Who founded [#{k - 1}]?" for k in range(2, nodes + 1))
    return steps


def _shape_all_pairs_referencing(nodes: int, gold_steps: list[str]) -> list[str]:
    """Every step references every earlier step: the densest graph of its size.

    Step 1 is :data:`_CHAIN_FIRST_STEP`; step k lists ``[#1] ... [#k-1]``. Graph: ``nodes``
    nodes and ``nodes * (nodes - 1) / 2`` edges (20 -> 190, 30 -> 435). It is in the table
    to show that edge count is *not* what drives the cost — a saturated graph has few
    near-identical alternatives, so the search barely branches.
    """
    steps = [_CHAIN_FIRST_STEP]
    for k in range(2, nodes + 1):
        refs = " ".join(f"[#{j}]" for j in range(1, k))
        steps.append(f"Which of {refs} is the earliest?")
    return steps


SHAPES: dict[str, Callable[[int, list[str]], list[str]]] = {
    "repeated_step_text": _shape_repeated_step_text,
    "gold_step_texts_repeated": _shape_gold_step_texts_repeated,
    "chain_shaped": _shape_chain_shaped,
    "all_pairs_referencing": _shape_all_pairs_referencing,
}


# --------------------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------------------


def _graph_of(steps: list[str]) -> nx.DiGraph:
    """Steps -> the graph the evaluator scores, through the evaluator's own functions."""
    return EVAL._decomposition_graph(EVAL._break_steps(steps))


def _load_gold_columns(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Read the fixture gold rows the table's columns are measured against."""
    path = resolve_path(require(cfg, "gold_fixture"), REPO_ROOT)
    if not path.exists():
        raise SystemExit(f"gold fixture not found: {path}")
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                rows[str(obj["id"])] = obj
    columns = []
    for column in require(cfg, "gold_columns"):
        gold_id = str(require(column, "gold_id"))
        if gold_id not in rows:
            raise SystemExit(f"{path}: no gold row with id {gold_id!r}")
        steps = [s["question"] for s in rows[gold_id]["question_decomposition"]]
        hop_depth = int(require(column, "hop_depth"))
        if len(steps) != hop_depth:
            raise SystemExit(
                f"gold {gold_id!r} has {len(steps)} steps but the config calls it "
                f"{hop_depth}-hop; the table's column header would be wrong."
            )
        columns.append(
            {
                "hop_depth": hop_depth,
                "gold_id": gold_id,
                "steps": steps,
                "graph": _graph_of(steps),
            }
        )
    return columns


def _time_optimizer(
    pred_graph: nx.DiGraph, gold_graph: nx.DiGraph, budget_seconds: float
) -> dict[str, Any]:
    """Time one uncapped GED, exactly as the evaluator computes it.

    ``max_nodes_for_optimizer`` is set out of reach on purpose: the cap is what this
    measurement exists to justify, so the measurement has to be of the un-capped cost.
    ``budget_seconds`` only guarantees termination; a cell that hits it is reported as
    truncated and its seconds are a lower bound.
    """
    started = time.monotonic()
    value, fallback, _ = EVAL._normalized_ged(
        pred_graph, gold_graph, _NO_NODE_CAP, budget_seconds
    )
    seconds = time.monotonic() - started
    return {
        "seconds": seconds,
        "ged": value,
        "truncated": fallback == "time_budget",
        "fallback": fallback,
    }


def _bound(pred_graph: nx.DiGraph, gold_graph: nx.DiGraph) -> float:
    """The search-free upper bound the evaluator reports above the node cap."""
    normalization = max(
        pred_graph.number_of_nodes() + pred_graph.number_of_edges(),
        gold_graph.number_of_nodes() + gold_graph.number_of_edges(),
    )
    return EVAL._positional_edit_cost(pred_graph, gold_graph) / normalization


def _cost_table(
    cfg: dict[str, Any],
    columns: list[dict[str, Any]],
    budget_seconds: float,
    max_node_count: int | None,
) -> list[dict[str, Any]]:
    rows = []
    for cell in require(cfg, "cost_table_cells"):
        shape = str(require(cell, "shape"))
        if shape not in SHAPES:
            raise SystemExit(f"unknown shape {shape!r}; known: {sorted(SHAPES)}")
        for nodes in require(cell, "node_counts"):
            nodes = int(nodes)
            if max_node_count is not None and nodes > max_node_count:
                continue
            row: dict[str, Any] = {"shape": shape, "nodes": nodes, "timings": {}}
            for column in columns:
                pred_graph = _graph_of(SHAPES[shape](nodes, column["steps"]))
                row["pred_nodes"] = pred_graph.number_of_nodes()
                row["pred_edges"] = pred_graph.number_of_edges()
                row["timings"][column["hop_depth"]] = _time_optimizer(
                    pred_graph, column["graph"], budget_seconds
                )
            rows.append(row)
    return rows


def _bound_vs_optimizer(
    cfg: dict[str, Any],
    columns: list[dict[str, Any]],
    budget_seconds: float,
    max_node_count: int | None,
) -> list[dict[str, Any]]:
    """What the cap trades away: the reported bound against the optimizer's own value.

    Measured on the deepest gold column, which is the expensive one and therefore the one
    the ADR quotes.
    """
    column = max(columns, key=lambda c: c["hop_depth"])
    rows = []
    for cell in require(cfg, "bound_vs_optimizer_cells"):
        shape = str(require(cell, "shape"))
        if shape not in SHAPES:
            raise SystemExit(f"unknown shape {shape!r}; known: {sorted(SHAPES)}")
        nodes = int(require(cell, "nodes"))
        if max_node_count is not None and nodes > max_node_count:
            continue
        pred_graph = _graph_of(SHAPES[shape](nodes, column["steps"]))
        measured = _time_optimizer(pred_graph, column["graph"], budget_seconds)
        bound = _bound(pred_graph, column["graph"])
        rows.append(
            {
                "shape": shape,
                "nodes": nodes,
                "gold_hop_depth": column["hop_depth"],
                "optimizer_ged": measured["ged"],
                "bound_ged": bound,
                "gap": bound - measured["ged"],
                "seconds": measured["seconds"],
                "truncated": measured["truncated"],
            }
        )
    return rows


def _commit() -> str:
    """``<sha>`` at HEAD, suffixed ``-dirty`` when the tree is not clean."""
    try:
        sha = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover - no git
        return "unknown"
    return f"{sha}-dirty" if dirty else sha


def _fmt_seconds(t: dict[str, Any]) -> str:
    """``0.98 s``, with a marker when the number is not a plain optimizer timing.

    ``*`` = the per-cell stop fired, so the seconds are a lower bound. ``†`` = networkx
    yielded no approximation at all, so the cell is fast for a reason that is not speed and
    must not be read as one.
    """
    mark = "*" if t["truncated"] else ("†" if t["fallback"] == "no_optimizer_result" else "")
    return f"{t['seconds']:.2f} s{mark}"


def _print_report(
    cfg: dict[str, Any],
    columns: list[dict[str, Any]],
    table: list[dict[str, Any]],
    boundary: list[dict[str, Any]],
    budget_seconds: float,
) -> None:
    depths = [c["hop_depth"] for c in columns]
    print(f"# GED optimizer cost (scripts/ged_cost_benchmark.py)\n")
    print(f"- commit: `{_commit()}`")
    print(f"- config: `{cfg['_config_path']}`")
    print(f"- measured (UTC): {now_iso()}")
    print(
        "- gold columns: "
        + ", ".join(f"{c['hop_depth']}-hop (`{c['gold_id']}`)" for c in columns)
    )
    print(
        f"- per-cell stop: {budget_seconds} s (`*` = the stop fired, so the time is a lower "
        f"bound; `†` = networkx yielded no approximation, so the cell is not a timing)\n"
    )

    header = ["prediction shape", "nodes", "edges"] + [f"vs {d}-hop gold" for d in depths]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for row in table:
        cells = [
            f"`{row['shape']}`",
            str(row["pred_nodes"]),
            str(row["pred_edges"]),
        ] + [_fmt_seconds(row["timings"][d]) for d in depths]
        print("| " + " | ".join(cells) + " |")

    if boundary:
        print("\n## Bound vs optimizer (what the node cap trades away)\n")
        header = ["prediction shape", "nodes", "vs gold", "optimizer", "reported bound", "gap"]
        print("| " + " | ".join(header) + " |")
        print("|" + "---|" * len(header))
        for row in boundary:
            print(
                f"| `{row['shape']}` | {row['nodes']} | {row['gold_hop_depth']}-hop | "
                f"{row['optimizer_ged']:.4f}{'*' if row['truncated'] else ''} | "
                f"{row['bound_ged']:.4f} | {row['gap']:+.4f} |"
            )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="ged_cost_benchmark.json", help="committed config to read")
    p.add_argument(
        "--max-node-count",
        type=int,
        default=None,
        help="skip cells above this node count (a tiny smoke run of the full path)",
    )
    p.add_argument("--json", type=Path, default=None, help="also write the raw numbers here")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    set_global_seed(int(require(cfg, "seed")))
    budget_seconds = float(require(cfg, "timing_budget_seconds"))
    if budget_seconds <= 0.0:
        raise SystemExit("timing_budget_seconds must be positive")

    columns = _load_gold_columns(cfg)
    table = _cost_table(cfg, columns, budget_seconds, args.max_node_count)
    boundary = _bound_vs_optimizer(cfg, columns, budget_seconds, args.max_node_count)
    _print_report(cfg, columns, table, boundary, budget_seconds)

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "script": Path(__file__).name,
                    "commit": _commit(),
                    "created_utc": now_iso(),
                    "config": {k: v for k, v in cfg.items()},
                    "cost_table": table,
                    "bound_vs_optimizer": boundary,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
