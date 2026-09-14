#!/usr/bin/env python3
"""Select the 45 anchor items out of a run's outputs, for the manual inspection of item A4.

The anchor set (``configs/anchor_set_ids.json``, drawn by ``scripts/build_anchor_set.py``)
is a subset of the ADR 0007 pinned 600, so every experiment in this phase already scores all
45 items as part of its normal run. Nothing here re-runs a model and nothing here re-scores:
this script **selects**, which is why it is CPU-only and costs no GPU time.

Two modes, both driven by the same committed id list:

``--per-item <eval_*_per_item.json>``
    The inspection dump. Writes the 45 scored rows back out as a per-item JSON, plus a CSV
    with the question, the gold decomposition, the predicted decomposition and the per-item
    metric values side by side. The CSV follows the convention
    ``scripts/run_qwen_smoke.py`` established for eyeballing decompositions (one row per
    query, steps joined with " | ") rather than inventing a second format.

``--filter <predictions.json|rows.jsonl> --filter-out <path>``
    The id filter. Keeps only the anchor rows of a predictions file (so the evaluator can
    re-score exactly those 45 with no code change: pass the output to ``--predictions``) or
    of a retrieval JSONL (so a generation run can be restricted to them, via the
    decomposer's ``--retrieval-input``; that run is not the pinned set, so it also needs
    ``--allow-unpinned-eval-set`` and it is not an experiment arm).

Rows are emitted in **anchor order** — hops in the committed order, ids sorted within a hop
— so two runs' dumps sit line-by-line beside each other.

A run over the pinned 600 must contain all 45. Fewer is a refusal, because a dump silently
missing items would be read as "the model did this on the anchors". ``--allow-missing``
downgrades it to a recorded warning, for a subset run that was never meant to cover them.

Usage::

    python scripts/anchor_inspect.py --per-item runs/<arm>/eval_per_item.json --label <arm>
    python scripts/anchor_inspect.py --filter runs/<arm>/results.json \\
        --filter-out runs/<arm>/anchor_predictions.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from run_artifacts import now_iso, write_run_artifacts  # noqa: E402
from run_config import load_config, load_paths, require, resolve_path, runs_path  # noqa: E402

#: Id fields a row may carry, in the order they are tried. Same precedence as the
#: evaluator's ``_item_id``, plus ``item_id`` for a per-item file.
_ID_KEYS = ("query_id", "id", "item_id")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", default="anchor_inspect.json")
    p.add_argument("--ids-file", type=Path, default=None, help="Override the committed id list.")
    p.add_argument("--per-item", type=Path, default=None, help="A '*_per_item.json' to dump.")
    p.add_argument("--filter", type=Path, default=None, help="A predictions JSON / rows JSONL.")
    p.add_argument("--filter-out", type=Path, default=None, help="Where --filter writes.")
    p.add_argument("--out-dir", type=Path, default=None, help="Override the dump directory.")
    p.add_argument("--label", default=None, help="Names the dump (e.g. the arm). Default: the input's parent.")
    p.add_argument(
        "--allow-missing",
        action="store_true",
        help="Permit fewer than the full anchor set to be present in the input.",
    )
    args = p.parse_args()
    if (args.per_item is None) == (args.filter is None):
        raise SystemExit("pass exactly one of --per-item or --filter")
    if args.filter is not None and args.filter_out is None:
        raise SystemExit("--filter needs --filter-out")
    return args


def load_anchor_ids(path: Path) -> tuple[list[str], dict[str, str], dict[str, Any]]:
    """``(ids in anchor order, {id: hop}, the file's header)``.

    The header travels into every artifact this script writes, so a dump always states which
    anchor list it was selected with (and at which seed).
    """
    if not path.exists():
        raise SystemExit(
            f"anchor id list not found: {path}\n"
            "Draw it with: python scripts/build_anchor_set.py --write"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    ids_by_hop = payload.get("ids_by_hop")
    if not isinstance(ids_by_hop, dict):
        raise SystemExit(f"{path}: missing an 'ids_by_hop' object (schema musique_anchor_set/1)")
    ordered: list[str] = []
    hop_of: dict[str, str] = {}
    for hop in payload.get("hops", sorted(ids_by_hop)):
        for qid in ids_by_hop[str(hop)]:
            if qid in hop_of:
                raise SystemExit(f"{path}: id {qid!r} appears under more than one hop")
            hop_of[qid] = str(hop)
            ordered.append(qid)
    header = {
        k: payload.get(k)
        for k in ("schema", "seed", "hops", "n_per_hop", "counts", "generated_by", "source")
    }
    header["ids_file"] = str(path)
    return ordered, hop_of, header


def _row_id(row: dict[str, Any]) -> str | None:
    for key in _ID_KEYS:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
    return None


def _index_rows(rows: list[dict[str, Any]], source: Path) -> dict[str, dict[str, Any]]:
    """``{id: row}``. A duplicate id is fatal — which of the two is the anchor item?"""
    by_id: dict[str, dict[str, Any]] = {}
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise SystemExit(f"{source}: row {i} is not an object")
        qid = _row_id(row)
        if qid is None:
            continue
        if qid in by_id:
            raise SystemExit(
                f"{source}: duplicate id {qid!r} (rows carry one of {_ID_KEYS}); "
                "the anchor selection cannot say which row is the anchor item"
            )
        by_id[qid] = row
    return by_id


def _load_rows(path: Path) -> tuple[list[dict[str, Any]], str]:
    """Rows of a predictions JSON list or a JSONL file, plus which format it was."""
    if not path.exists():
        raise SystemExit(f"input not found: {path}")
    if path.suffix == ".jsonl":
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return rows, "jsonl"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("items"), list):
        # A per-item file passed to --filter: keep the caller honest about which mode
        # they wanted rather than half-handling it.
        raise SystemExit(
            f"{path} looks like a '*_per_item.json' (it has an 'items' list). Use "
            "--per-item for that; --filter takes a predictions JSON list or a JSONL."
        )
    if not isinstance(payload, list):
        raise SystemExit(f"{path}: --filter expects a JSON list or a .jsonl file")
    return payload, "json"


def select(
    by_id: dict[str, dict[str, Any]], anchor_ids: list[str]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Anchor rows in anchor order, and the anchor ids the input did not carry."""
    found = [by_id[qid] for qid in anchor_ids if qid in by_id]
    missing = [qid for qid in anchor_ids if qid not in by_id]
    return found, missing


def _coverage_check(missing: list[str], total: int, source: Path, allow_missing: bool) -> str | None:
    if not missing:
        return None
    detail = ", ".join(missing[:10]) + (f" ... (+{len(missing) - 10} more)" if len(missing) > 10 else "")
    message = (
        f"{len(missing)} of {total} anchor items are not in {source}: {detail}"
    )
    if allow_missing:
        print(f"[anchor] WARNING: {message} (allowed by --allow-missing)")
        return message
    raise SystemExit(
        message
        + "\nThe anchor set is a subset of the ADR 0007 pinned 600, so a run over that set "
        "carries all of them. Pass --allow-missing for a subset run that was never meant to."
    )


def _steps(value: Any) -> str:
    """A step list rendered the way scripts/run_qwen_smoke.py renders one, for eyeballing."""
    if isinstance(value, list):
        return " | ".join(str(v) for v in value)
    return "" if value is None else str(value)


def _write_csv(path: Path, rows: list[dict[str, Any]], hop_of: dict[str, str], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["query_id", "anchor_hop", "gold_hop_count", "question", "gold_decomposition",
             "pred_decomposition", *columns]
        )
        for row in rows:
            qid = str(row.get("item_id", ""))
            writer.writerow(
                [
                    qid,
                    hop_of.get(qid, ""),
                    row.get("gold_hop_count", ""),
                    row.get("question", ""),
                    _steps(row.get("gold_steps")),
                    _steps(row.get("pred_steps")),
                    *[row.get(c, "") for c in columns],
                ]
            )


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    paths_cfg = load_paths(require(cfg, "paths_config"))
    ids_path = args.ids_file or resolve_path(require(cfg, "ids_file"), REPO_ROOT)
    anchor_ids, hop_of, anchor_header = load_anchor_ids(ids_path)
    print(f"[anchor] {len(anchor_ids)} anchor ids from {ids_path} ({anchor_header['counts']})")

    if args.filter is not None:
        rows, fmt = _load_rows(args.filter)
        by_id = _index_rows(rows, args.filter)
        selected, missing = select(by_id, anchor_ids)
        _coverage_check(missing, len(anchor_ids), args.filter, args.allow_missing)
        out = args.filter_out
        out.parent.mkdir(parents=True, exist_ok=True)
        if fmt == "jsonl":
            out.write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected),
                encoding="utf-8",
            )
        else:
            out.write_text(
                json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        print(
            f"[anchor] filtered {len(rows)} {fmt} row(s) -> {len(selected)} anchor row(s): {out}"
        )
        return

    # --per-item: the inspection dump.
    payload = json.loads(args.per_item.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise SystemExit(
            f"{args.per_item}: expected a '<prefix>_per_item.json' object with an 'items' "
            "list, as scripts/musique_decompositions_evaluator.py writes."
        )
    by_id = _index_rows(payload["items"], args.per_item)
    selected, missing = select(by_id, anchor_ids)
    coverage_warning = _coverage_check(missing, len(anchor_ids), args.per_item, args.allow_missing)

    label = args.label or args.per_item.parent.name
    out_dir = args.out_dir or runs_path(paths_cfg, require(cfg, "run_subdir"), label)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = require(cfg, "out_prefix")
    columns = [str(c) for c in require(cfg, "csv_metric_columns")]

    per_item_out = out_dir / f"{prefix}_per_item.json"
    per_item_out.write_text(
        json.dumps(
            {
                **{k: v for k, v in payload.items() if k != "items"},
                "anchor_set": anchor_header,
                "selected_from": str(args.per_item.resolve()),
                "items": selected,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    csv_out = out_dir / f"{prefix}_review.csv"
    _write_csv(csv_out, selected, hop_of, columns)

    per_hop_found: dict[str, int] = {}
    for row in selected:
        hop = hop_of.get(str(row.get("item_id", "")), "?")
        per_hop_found[hop] = per_hop_found.get(hop, 0) + 1

    metrics = {
        "script": Path(__file__).name,
        "created_utc": now_iso(),
        "label": label,
        "anchor_set": anchor_header,
        "source_per_item": str(args.per_item.resolve()),
        "source_items": len(payload["items"]),
        "anchor_items_expected": len(anchor_ids),
        "anchor_items_found": len(selected),
        "anchor_items_found_per_hop": per_hop_found,
        "anchor_items_missing": missing,
        "allow_missing": bool(args.allow_missing),
        "coverage_warning": coverage_warning,
        "csv_metric_columns": columns,
        "outputs": {"per_item": str(per_item_out), "csv": str(csv_out)},
        "note": (
            "a selection of rows scored by scripts/musique_decompositions_evaluator.py; "
            "no metric is recomputed here, so every value is the run's own"
        ),
    }
    write_run_artifacts(
        out_dir,
        config_snapshot={
            "script": Path(__file__).name,
            "config_path": cfg.get("_config_path"),
            "paths_config": paths_cfg.get("_config_path"),
            "ids_file": str(ids_path),
            "per_item": str(args.per_item),
            "label": label,
            "out_dir": str(out_dir),
            "allow_missing": bool(args.allow_missing),
        },
        metrics=metrics,
        note_title=f"Anchor-set inspection dump — {label}",
        note_lines=[
            f"- Anchor list: `{ids_path}` (seed {anchor_header.get('seed')}, "
            f"{anchor_header.get('n_per_hop')} per hop, {anchor_header.get('counts')})",
            f"- Scored input: `{args.per_item}` ({len(payload['items'])} items)",
            f"- Anchor items found: {len(selected)} of {len(anchor_ids)} "
            f"(per hop: {per_hop_found})",
            f"- Read side by side: `{csv_out}`",
            "- No metric was recomputed; the values are the ones the evaluator produced.",
        ],
        prefix=f"{prefix}_",
    )
    print(f"[anchor] {len(selected)}/{len(anchor_ids)} anchor items -> {csv_out}")


if __name__ == "__main__":
    main()
