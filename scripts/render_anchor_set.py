#!/usr/bin/env python3
"""Render the 45 anchor items of a finished run (or several, side by side) for hand inspection.

Supervisor item A4 (2026-08-27, recorded in
``docs/plan/2026-09-14-post-supervisor-meeting-phase.md``): *"I want you to pick some anchor
sample that you will check manually by yourself in every experiment… 15 from each hop… those
will be fixed over all the experiments."* Item A7 adds the cross-model half: *"the same
example, check them with Quen and Mistral"*. This script is the reader for both — one
command, one Markdown sheet, the same 45 ids every time.

**It post-filters existing outputs and nothing else.** The input is one or more
``*_per_item.json`` files the evaluator already wrote, which carry the question, the gold
steps, the predicted steps and every per-item metric. No model is loaded, nothing is
re-scored, and no run path is touched: ``run_decomposer.py`` and
``musique_decompositions_evaluator.py`` are untouched by this file, so the pinned-600
assertion and every existing arm read exactly as they did before. That also means an arm can
only be rendered *after* it has been evaluated on the pinned 600 — the anchor 45 are a subset
of it, so the inspection costs no extra inference.

Usage::

    python scripts/render_anchor_set.py \\
        --arm "exp-015 unguided=runs/exp-015/eval/eval_per_item.json" \\
        --arm "exp-008 full_train=runs/exp-011/exp008_full_train/eval_per_item.json" \\
        --out runs/anchor/2026-09-20-baselines.md

    # label defaults to the file's parent directory
    python scripts/render_anchor_set.py --arm runs/exp-015/eval/eval_per_item.json

The sheet goes under ``runs/`` (gitignored) by default and must stay there: it contains
question text and gold decompositions, and data never enters git (CLAUDE.md).

Exit codes: 0 when every arm covers all 45 anchor ids, 1 when one does not (pass
``--allow-missing`` to render a partial sheet anyway — the gaps are then stated in it).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from run_artifacts import now_iso  # noqa: E402
from run_config import load_config, load_paths, require, resolve_path, runs_path  # noqa: E402

#: The per-item schema this script reads. Stated so a file written against a different shape
#: is refused by name instead of half-rendering.
PER_ITEM_SCHEMA = "musique_decomposition_per_item/1"

#: Id fields a per-item row may carry, in the evaluator's own precedence (``_item_id``),
#: plus ``item_id``, which is what the file it writes actually uses.
_ID_KEYS = ("item_id", "query_id", "id")


class _Missing:
    """The field is **not in the file**, as distinct from present and ``null``.

    The two must not render the same. ``ged_fallback: null`` means the evaluator ran the
    exact GED and recorded no fallback; a missing ``decomp_mean`` means the file predates
    that metric. Collapsing both to one marker invites reading the second as the first.
    """

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid only
        return "<missing>"


#: Singleton for "this field is absent from the row".
MISSING = _Missing()


def display_path(path: Path | str) -> str:
    """A path as the record should carry it: relative to the repo root when it is inside it.

    A sheet rendered from a throwaway worktree would otherwise record that worktree's
    absolute path, which tells a later reader nothing about which file was read.
    """
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def assert_out_path_never_enters_git(out: Path) -> None:
    """Refuse an ``--out`` that git would let someone commit.

    The sheet carries question text and gold decompositions. CLAUDE.md is unconditional:
    data never enters git. A path outside the working tree is fine; a path inside it is
    only fine when ``git check-ignore`` says the path is ignored — asking git is the only
    check that cannot drift from what ``git add`` would actually do.

    A git that cannot answer (no git on PATH, not a repository, an internal error) is a
    refusal too: "could not prove it is ignored" is not "it is ignored".
    """
    resolved = Path(out).resolve()
    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError:
        return  # Outside the working tree — git has no say over it.

    try:
        proc = subprocess.run(
            ["git", "check-ignore", "-q", "--", str(resolved)],
            cwd=str(REPO_ROOT), capture_output=True, text=True,
        )
    except OSError as exc:  # pragma: no cover - depends on the box, not on the input
        raise SystemExit(
            f"refusing --out {out}: it is inside the working tree and git could not be "
            f"asked whether it is ignored ({exc}). The sheet carries question text and "
            "gold decompositions, and data never enters git (CLAUDE.md). Write it under "
            "runs/ instead."
        ) from exc

    if proc.returncode == 0:
        return  # git says the path is ignored.
    if proc.returncode == 1:
        raise SystemExit(
            f"refusing --out {out}: that path is inside the repository and is NOT "
            "gitignored, so the sheet could be committed. It carries question text and "
            "gold decompositions, and data never enters git (CLAUDE.md). Write it under "
            "runs/ (gitignored) or outside the working tree."
        )
    raise SystemExit(
        f"refusing --out {out}: git check-ignore exited {proc.returncode} "
        f"({proc.stderr.strip() or 'no message'}), so it could not be established that the "
        "path is ignored. The sheet carries question text and gold decompositions, and "
        "data never enters git (CLAUDE.md)."
    )


class Arm:
    """One evaluated arm: a label, the file it came from, and its rows keyed by item id."""

    def __init__(self, label: str, path: Path, rows: dict[str, dict[str, Any]], header: dict[str, Any]):
        self.label = label
        self.path = path
        self.rows = rows
        self.header = header


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", default="anchor_render.json")
    p.add_argument(
        "--arm",
        action="append",
        default=[],
        metavar="[LABEL=]PER_ITEM_JSON",
        help="An evaluated arm to render. Repeatable; arms appear as columns in the order "
        "given. Without a LABEL the file's parent directory names the column.",
    )
    p.add_argument(
        "--anchor",
        type=Path,
        default=None,
        help="The committed anchor id list. Default: ids_file from the config.",
    )
    p.add_argument("--out", type=Path, default=None, help="Where the Markdown sheet is written.")
    p.add_argument(
        "--allow-missing",
        action="store_true",
        help="Render even when an arm does not cover all 45 anchor ids. The sheet then says "
        "which ids are missing from which arm, so a gap can never be read as a result.",
    )
    args = p.parse_args()
    if not args.arm:
        raise SystemExit("pass at least one --arm [LABEL=]<eval_per_item.json>")
    return args


def parse_arm_spec(spec: str) -> tuple[str, Path]:
    """``"label=path"`` or ``"path"`` -> ``(label, path)``.

    Split on the *first* ``=`` only, so a label may not contain one but a path may.
    A bare path is labelled by its parent directory, which is how the run tree already
    names arms (``runs/exp-011/exp008_full_train/eval_per_item.json`` -> ``exp008_full_train``).
    """
    if "=" in spec:
        label, _, raw = spec.partition("=")
        label = label.strip()
        path = Path(raw.strip())
        if not label or not raw.strip():
            raise SystemExit(f"--arm {spec!r}: expected LABEL=PATH with both sides non-empty")
        return label, path
    path = Path(spec.strip())
    return (path.parent.name or path.stem), path


def load_anchor_ids(path: Path) -> tuple[list[str], dict[str, str], dict[str, Any]]:
    """``(ids in anchor order, {id: hop}, the list's provenance header)``.

    Anchor order is hops in the committed order, ids sorted within a hop — the order the
    draw script writes — so two sheets rendered on different days line up item for item.
    """
    if not path.exists():
        raise SystemExit(
            f"anchor id list not found: {path}\n"
            "Draw it once with: python scripts/build_anchor_set.py --write"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    ids_by_hop = payload.get("ids_by_hop")
    if not isinstance(ids_by_hop, dict):
        raise SystemExit(f"{path}: missing an 'ids_by_hop' object (schema musique_anchor_set/1)")
    ordered: list[str] = []
    hop_of: dict[str, str] = {}
    for hop in payload.get("hops", sorted(ids_by_hop)):
        bucket = ids_by_hop.get(str(hop))
        if bucket is None:
            raise SystemExit(
                f"{path}: 'hops' lists hop {hop} but 'ids_by_hop' has no bucket for it. "
                "The list was hand-edited or written by an older draw; re-derive it with "
                "python scripts/build_anchor_set.py --verify to see what moved."
            )
        if not isinstance(bucket, list):
            raise SystemExit(
                f"{path}: ids_by_hop[{hop}] is {type(bucket).__name__}, expected a list of ids."
            )
        for qid in bucket:
            if qid in hop_of:
                raise SystemExit(f"{path}: id {qid!r} appears under more than one hop")
            hop_of[qid] = str(hop)
            ordered.append(qid)
    if not ordered:
        raise SystemExit(f"{path}: the anchor list is empty")
    header = {
        k: payload.get(k)
        for k in ("schema", "seed", "hops", "n_per_hop", "counts", "generated_by", "generated_utc", "source")
    }
    header["ids_file"] = display_path(path)
    return ordered, hop_of, header


def _row_id(row: dict[str, Any]) -> str | None:
    for key in _ID_KEYS:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
    return None


def load_per_item(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Load an evaluator ``*_per_item.json`` into ``({item_id: row}, header)``.

    A duplicate id is fatal: with two rows for one anchor item the sheet would show one of
    them and there would be no way to tell which.
    """
    if not path.exists():
        raise SystemExit(f"per-item file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise SystemExit(
            f"{path}: not an object with an 'items' list. This script reads the evaluator's "
            f"'{PER_ITEM_SCHEMA}' per-item artifact."
        )
    schema = payload.get("schema")
    if schema != PER_ITEM_SCHEMA:
        raise SystemExit(
            f"{path}: schema is {schema!r}, expected {PER_ITEM_SCHEMA!r}. Re-score the arm "
            "with scripts/musique_decompositions_evaluator.py rather than rendering a file "
            "of unknown shape."
        )
    by_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(payload["items"]):
        if not isinstance(row, dict):
            raise SystemExit(f"{path}: items[{index}] is not an object")
        qid = _row_id(row)
        if qid is None:
            raise SystemExit(f"{path}: items[{index}] carries none of {_ID_KEYS}")
        if qid in by_id:
            raise SystemExit(f"{path}: duplicate item id {qid!r}")
        by_id[qid] = row
    header = {k: v for k, v in payload.items() if k != "items"}
    header["per_item_path"] = display_path(path)
    header["item_count"] = len(by_id)
    return by_id, header


def header_value(arm: "Arm", field: str) -> Any:
    """One arm's header field, or ``MISSING`` when the file does not carry it."""
    return arm.header.get(field, MISSING)


def metric_fields(arms: list[Arm], anchor_ids: list[str], structural: list[str]) -> list[str]:
    """Every non-structural field the anchor rows carry, in first-seen order.

    First-seen across arms in the order the arms were given, which is the evaluator's own
    emission order for the first arm and then anything a later arm adds. The script does not
    curate the list: the sheet shows what the files say.
    """
    seen: list[str] = []
    known = set(structural)
    for arm in arms:
        for qid in anchor_ids:
            row = arm.rows.get(qid)
            if row is None:
                continue
            for key in row:
                if key in known:
                    continue
                known.add(key)
                seen.append(key)
    return seen


def cell(value: Any, *, float_format: str, empty: str, null: str = "null") -> str:
    """One Markdown table cell: pipes escaped, newlines folded, floats formatted.

    ``MISSING`` (the field is absent from the file) and ``None`` (the field is present and
    ``null``) render differently and deliberately — see :class:`_Missing`.
    """
    if isinstance(value, _Missing):
        return empty
    if value is None:
        return null
    if isinstance(value, bool):
        text = "true" if value else "false"
    elif isinstance(value, float):
        text = float_format.format(value)
    elif isinstance(value, (list, tuple)):
        text = " | ".join(str(v) for v in value)
    elif isinstance(value, dict):
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    else:
        text = str(value)
    text = text.replace("\\", "\\\\").replace("|", "\\|")
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br>")
    return text.strip() or empty


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out.extend("| " + " | ".join(r) + " |" for r in rows)
    return out


def _steps(row: dict[str, Any] | None, key: str) -> list[str]:
    if row is None:
        return []
    value = row.get(key)
    if not isinstance(value, list):
        return []
    return [str(v) for v in value]


def render(
    arms: list[Arm],
    anchor_ids: list[str],
    hop_of: dict[str, str],
    anchor_header: dict[str, Any],
    *,
    structural: list[str],
    arm_header_fields: list[str],
    comparability_fields: list[str],
    float_format: str,
    empty: str,
    null: str,
) -> str:
    """The whole Markdown sheet, as one string."""
    fmt = {"float_format": float_format, "empty": empty, "null": null}
    fields = metric_fields(arms, anchor_ids, structural)
    labels = [a.label for a in arms]
    coverage = {a.label: sum(1 for qid in anchor_ids if qid in a.rows) for a in arms}
    total = len(anchor_ids)

    lines: list[str] = []
    lines.append(f"# Anchor set — {total} items, hand inspection")
    lines.append("")
    lines.append(
        f"Rendered {now_iso()} by `scripts/render_anchor_set.py`. This sheet is a "
        "**post-filter of finished evaluations**: every number below is copied from the "
        "per-item file the evaluator wrote for that arm. Nothing here re-scores or re-runs."
    )
    lines.append("")
    lines.append(
        f"Anchor list `{anchor_header.get('ids_file')}` — seed {anchor_header.get('seed')}, "
        f"{anchor_header.get('n_per_hop')} per hop, hops {anchor_header.get('hops')}, "
        f"counts {json.dumps(anchor_header.get('counts'), sort_keys=True)}, drawn "
        f"{anchor_header.get('generated_utc')} by `{anchor_header.get('generated_by')}`."
    )
    lines.append("")
    lines.append("## Arms and coverage")
    lines.append("")
    lines.extend(
        _table(
            ["arm", "per-item file", "rows in file", "anchor items found", *arm_header_fields],
            [
                [
                    cell(a.label, **fmt),
                    "`" + cell(display_path(a.path), **fmt) + "`",
                    cell(a.header.get("item_count", MISSING), **fmt),
                    f"{coverage[a.label]}/{total}",
                    *(cell(header_value(a, field), **fmt) for field in arm_header_fields),
                ]
                for a in arms
            ],
        )
    )
    lines.append("")
    lines.append(
        "The columns after the counts are the arms' own per-item headers, copied verbatim. "
        "They are here so the scoring settings a sheet compares under are visible on the "
        "same page as the comparison."
    )
    lines.append("")

    # The same shape of banner the question / gold checks emit further down: a difference
    # that would make two columns non-comparable is stated, never quietly averaged over.
    for field in comparability_fields:
        values = {a.label: header_value(a, field) for a in arms}
        distinct = {json.dumps(v, sort_keys=True, default=str) for v in values.values()}
        if len(distinct) > 1:
            lines.append(
                f"> **The arms were scored under different `{field}`.** "
                + "; ".join(f"{label}: {cell(value, **fmt)}" for label, value in values.items())
                + ". Any column-to-column reading of a metric that depends on it is not a "
                "comparison until that is explained."
            )
            lines.append("")
    for arm in arms:
        missing = [qid for qid in anchor_ids if qid not in arm.rows]
        if missing:
            lines.append(
                f"- **{arm.label} is missing {len(missing)} anchor item(s)**: "
                + ", ".join(f"`{m}`" for m in missing)
                + ". Those cells are empty below and must not be read as a result."
            )
    if any(coverage[a.label] < total for a in arms):
        lines.append("")

    for index, qid in enumerate(anchor_ids, start=1):
        rows = [arm.rows.get(qid) for arm in arms]
        present = [r for r in rows if r is not None]
        lines.append(f"## {index}. `{qid}` — hop {hop_of.get(qid, '?')}")
        lines.append("")

        questions = {str(r.get("question", "")).strip() for r in present}
        if not present:
            lines.append("_No arm carries this item._")
            lines.append("")
            continue
        if len(questions) > 1:
            lines.append(
                "> **The arms disagree on the question text for this id.** Each arm's own "
                "text is shown under its column heading; do not compare these rows until "
                "that is explained."
            )
            lines.append("")
            for arm, row in zip(arms, rows):
                if row is not None:
                    lines.append(f"- **{arm.label}:** {str(row.get('question', '')).strip()}")
            lines.append("")
        else:
            lines.append(f"**Question.** {questions.pop()}")
            lines.append("")

        gold_variants = {tuple(_steps(r, "gold_steps")) for r in present}
        if len(gold_variants) > 1:
            lines.append(
                "> **The arms disagree on the gold steps for this id** — they were scored "
                "against different gold. The per-arm gold is shown in the steps table."
            )
            lines.append("")
            step_header = ["#", "gold (per arm)"] + labels
        else:
            gold = list(gold_variants.pop())
            lines.append("**Gold steps.**")
            lines.append("")
            if gold:
                lines.extend(
                    f"{i}. {s}" for i, s in enumerate(gold, start=1)
                )
            else:
                lines.append("_(none recorded)_")
            lines.append("")
            step_header = ["#"] + labels

        per_arm_steps = [_steps(r, "pred_steps") for r in rows]
        per_arm_gold = [_steps(r, "gold_steps") for r in rows]
        depth = max([len(s) for s in per_arm_steps] + [len(g) for g in per_arm_gold] + [0])
        lines.append("**Predicted steps.**")
        lines.append("")
        table_rows: list[list[str]] = []
        for i in range(depth):
            row_cells = [str(i + 1)]
            if len(step_header) == len(labels) + 2:
                row_cells.append(
                    " / ".join(
                        cell(g[i] if i < len(g) else MISSING, **fmt)
                        for g in per_arm_gold
                    )
                )
            row_cells.extend(
                cell(s[i] if i < len(s) else MISSING, **fmt)
                for s in per_arm_steps
            )
            table_rows.append(row_cells)
        if table_rows:
            lines.extend(_table(step_header, table_rows))
        else:
            lines.append("_(no predicted steps in any arm)_")
        lines.append("")

        lines.append("**Per-item metrics, as recorded by the evaluator.**")
        lines.append("")
        metric_rows = [
            [f"`{field}`"]
            + [
                cell(row.get(field, MISSING) if row is not None else MISSING, **fmt)
                for row in rows
            ]
            for field in fields
        ]
        if metric_rows:
            lines.extend(_table(["metric"] + labels, metric_rows))
        else:
            lines.append("_(the per-item rows carry no metric fields)_")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    paths_cfg = load_paths(require(cfg, "paths_config"))

    # First, before a single file is read: a sheet that could be committed is not written
    # at all. Cheap, and it fails at the point the mistake was made.
    out = args.out or runs_path(paths_cfg, require(cfg, "out_subdir"), "anchor_set.md")
    assert_out_path_never_enters_git(out)

    ids_file = args.anchor or resolve_path(require(cfg, "ids_file"), REPO_ROOT)
    anchor_ids, hop_of, anchor_header = load_anchor_ids(ids_file)

    arms: list[Arm] = []
    for spec in args.arm:
        label, path = parse_arm_spec(spec)
        if any(a.label == label for a in arms):
            raise SystemExit(f"--arm label {label!r} is used twice; column labels must differ")
        rows, header = load_per_item(path)
        arms.append(Arm(label, path, rows, header))

    total = len(anchor_ids)
    coverage = {a.label: sum(1 for qid in anchor_ids if qid in a.rows) for a in arms}
    for arm in arms:
        print(f"[anchor-render] {arm.label}: {coverage[arm.label]}/{total} anchor items in {arm.path}")

    short = [label for label, n in coverage.items() if n < total]
    if short and not args.allow_missing:
        raise SystemExit(
            "these arms do not cover the full anchor set: "
            + ", ".join(f"{label} ({coverage[label]}/{total})" for label in short)
            + "\nAn arm evaluated on the ADR 0007 pinned 600 contains all of them, so a gap "
            "means the arm was scored on something else. Pass --allow-missing to render it "
            "anyway; the sheet will state the gap."
        )

    sheet = render(
        arms,
        anchor_ids,
        hop_of,
        anchor_header,
        structural=list(require(cfg, "structural_fields")),
        arm_header_fields=list(require(cfg, "arm_header_fields")),
        comparability_fields=list(require(cfg, "comparability_fields")),
        float_format=require(cfg, "float_format"),
        empty=require(cfg, "empty_cell"),
        null=require(cfg, "null_cell"),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(sheet, encoding="utf-8")
    print(f"[anchor-render] wrote {out} ({total} items, {len(arms)} arm(s))")


if __name__ == "__main__":
    main()
