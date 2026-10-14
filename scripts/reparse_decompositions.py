#!/usr/bin/env python3
"""Re-parse a stored decomposer run's raw generations with the current step parser.

Issue #60 / ADR 0039 item 4 (H3). The decomposer now cuts a generation at the scaffold
labels of ``step_parser.scaffold_truncate_at`` (``Task:``, ``Answer:``) with the same
semantics as the model folder's ``Question:`` marker. Runs made before that kept the label
as a scored step. Every row of a ``results.json`` stores the raw generation
(``decomposition_raw``), so the fix can be applied to a stored run with **no inference**:

1. read the run's own config snapshot (``config.json`` beside ``results.json`` unless
   ``--run-config`` says otherwise) for the ``post_process`` block it applied and its
   ``stop_after_step_lines`` cap;
2. **reproduce** every stored ``decomposition`` from ``decomposition_raw`` with that recorded
   block. A row that does not reproduce is a refusal, not a warning: if the recorded parser
   does not give back the recorded decomposition, a re-parse would mix the fix with some
   other, unexplained difference;
3. re-parse with the recorded block plus the scaffold markers
   (``src/step_lines.py::with_scaffold_truncate_markers``, the function the decomposer
   itself uses), applying the same cap trim;
4. write a NEW ``results.json`` into ``--output-dir`` (rows identical to the input except
   ``decomposition`` and ``step_lines`` on the rows that changed), plus a config snapshot,
   a metrics JSON and a run note (prefix ``reparse_``) that list every changed row with its
   ``pred_steps`` before and after, split by the evaluator's own splitter.

It never writes in place: an output directory that is the input's directory, or that
already holds a ``results.json``, is refused. Score the new file with
``scripts/musique_decompositions_evaluator.py --predictions <output-dir>/results.json``.

This is a tool, not an experiment: running it on a committed arm and logging the re-scored
numbers is the experiment-runner's job (ADR 0039 item 4 - exp-020 and exp-023 keep their
recorded values; the re-score is a new experiment).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from run_artifacts import now_iso, write_run_artifacts  # noqa: E402
from run_config import load_config, require  # noqa: E402
from step_lines import (  # noqa: E402
    post_process_generation,
    split_step_lines,
    step_line_count,
    trim_to_step_lines,
    with_scaffold_truncate_markers,
)

#: How many non-reproducing row ids a refusal names.
_MAX_REPORTED_IDS = 20


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_raw(raw: str, post_cfg: dict[str, Any], max_step_lines: int | None) -> str:
    """The decomposition the decomposer records for ``raw`` under ``post_cfg``.

    Mirrors ``components/decomposer/run_decomposer.py::generate``: post-process, then the
    step-line trim when the run had a cap, then strip.
    """
    processed = post_process_generation(
        raw,
        strip_think=bool(post_cfg.get("strip_think")),
        truncate_at=post_cfg.get("truncate_at") or [],
    )
    out = trim_to_step_lines(processed, max_step_lines) if max_step_lines else processed
    return out.strip()


def reparse_rows(
    rows: list[dict[str, Any]],
    recorded_post_cfg: dict[str, Any],
    fixed_post_cfg: dict[str, Any],
    max_step_lines: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], int]:
    """Re-parse every row. Returns (new_rows, changed, not_reproduced_ids, skipped_empty).

    A row with an empty ``decomposition_raw`` (a dry-run row, or a generation that produced
    nothing) is copied unchanged and counted in ``skipped_empty``.
    """
    new_rows: list[dict[str, Any]] = []
    changed: list[dict[str, Any]] = []
    not_reproduced: list[str] = []
    skipped_empty = 0
    for i, row in enumerate(rows):
        if "decomposition_raw" not in row or "decomposition" not in row:
            raise SystemExit(
                f"row {i} has no 'decomposition_raw' / 'decomposition' field; this is not a "
                "results.json written by run_decomposer.py"
            )
        raw = row["decomposition_raw"] or ""
        stored = row["decomposition"] or ""
        row_id = str(row.get("query_id") if row.get("query_id") is not None else f"row{i}")
        if not raw:
            skipped_empty += 1
            new_rows.append(dict(row))
            continue
        if parse_raw(raw, recorded_post_cfg, max_step_lines) != stored:
            not_reproduced.append(row_id)
            new_rows.append(dict(row))
            continue
        fixed = parse_raw(raw, fixed_post_cfg, max_step_lines)
        new_row = dict(row)
        if fixed != stored:
            new_row["decomposition"] = fixed
            if "step_lines" in row:
                new_row["step_lines"] = step_line_count(fixed)
            changed.append(
                {
                    "query_id": row_id,
                    "pred_steps_before": split_step_lines(stored),
                    "pred_steps_after": split_step_lines(fixed),
                }
            )
        new_rows.append(new_row)
    return new_rows, changed, not_reproduced, skipped_empty


def _refuse_overwrite(results_path: Path, output_dir: Path) -> Path:
    """The output path, after refusing anything that could overwrite the input."""
    out_dir = output_dir.resolve()
    in_path = results_path.resolve()
    if out_dir == in_path.parent:
        raise SystemExit(
            f"--output-dir {output_dir} is the input's own directory; this script never "
            "writes in place - pass a new directory"
        )
    out_path = out_dir / "results.json"
    if out_path.exists():
        raise SystemExit(
            f"{out_path} already exists; refusing to overwrite it (pass a new --output-dir)"
        )
    return out_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--results", type=Path, required=True, help="A stored run's results.json.")
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New directory for the re-parsed results.json and its run artifacts.",
    )
    p.add_argument(
        "--run-config",
        type=Path,
        default=None,
        help="The run's config snapshot (default: config.json beside --results).",
    )
    p.add_argument(
        "--config",
        default="decomposer_musique.json",
        help="Shared decomposer config holding step_parser.scaffold_truncate_at "
        "(default: configs/decomposer_musique.json).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    results_path: Path = args.results
    if not results_path.is_file():
        raise SystemExit(f"--results not found: {results_path}")
    out_path = _refuse_overwrite(results_path, args.output_dir)
    run_config_path = args.run_config or results_path.parent / "config.json"
    if not run_config_path.is_file():
        raise SystemExit(f"run config snapshot not found: {run_config_path} (pass --run-config)")

    cfg = load_config(args.config)
    scaffold = require(cfg, "step_parser.scaffold_truncate_at")
    run_cfg = json.loads(run_config_path.read_text(encoding="utf-8"))
    recorded_post_cfg = dict(require(run_cfg, "post_process"))
    max_step_lines = run_cfg.get("stop_after_step_lines")
    if "stop_after_step_lines" not in run_cfg:
        raise SystemExit(f"{run_config_path}: no 'stop_after_step_lines' key")
    fixed_post_cfg = with_scaffold_truncate_markers(recorded_post_cfg, scaffold)

    rows = json.loads(results_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise SystemExit(f"{results_path}: expected a JSON list of rows")
    new_rows, changed, not_reproduced, skipped_empty = reparse_rows(
        rows, recorded_post_cfg, fixed_post_cfg, max_step_lines
    )
    if not_reproduced:
        raise SystemExit(
            f"{len(not_reproduced)} of {len(rows)} rows do not reproduce their stored "
            f"'decomposition' from 'decomposition_raw' under the run's recorded post_process "
            f"{recorded_post_cfg} (cap {max_step_lines}); refusing to re-parse, since the "
            f"result would mix the scaffold fix with an unexplained difference. First ids: "
            f"{not_reproduced[:_MAX_REPORTED_IDS]}. Nothing was written."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(new_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    steps_removed = sum(
        len(c["pred_steps_before"]) - len(c["pred_steps_after"]) for c in changed
    )
    metrics = {
        "script": Path(__file__).name,
        "created_utc": now_iso(),
        "input_results": str(results_path.resolve()),
        "input_results_sha256": _sha256(results_path),
        "run_config": str(run_config_path.resolve()),
        "output_results": str(out_path),
        "output_results_sha256": _sha256(out_path),
        "rows_total": len(rows),
        "rows_reproduced_under_recorded_parser": len(rows) - skipped_empty,
        "rows_skipped_empty_raw": skipped_empty,
        "rows_changed": len(changed),
        "pred_steps_removed_total": steps_removed,
        "recorded_post_process": recorded_post_cfg,
        "fixed_post_process": fixed_post_cfg,
        "stop_after_step_lines": max_step_lines,
        "changed_rows": changed,
    }
    snapshot = {
        "script": Path(__file__).name,
        "created_utc": metrics["created_utc"],
        "results": str(results_path),
        "output_dir": str(args.output_dir),
        "run_config": str(run_config_path),
        "config": cfg.get("_config_path"),
        "scaffold_truncate_at": scaffold,
    }
    note_lines = [
        f"- Input: `{results_path}` (sha256 `{metrics['input_results_sha256']}`)",
        f"- Run config: `{run_config_path}`; recorded post_process {recorded_post_cfg}, "
        f"cap {max_step_lines}",
        f"- Fixed post_process: {fixed_post_cfg} (scaffold markers from "
        f"`{cfg.get('_config_path')}`)",
        f"- Rows: {len(rows)}; reproduced under the recorded parser: "
        f"{metrics['rows_reproduced_under_recorded_parser']}; empty raw (copied): "
        f"{skipped_empty}",
        f"- **Rows whose pred_steps changed: {len(changed)}** "
        f"({steps_removed} step(s) removed in total)",
        f"- Output: `{out_path}` (sha256 `{metrics['output_results_sha256']}`)",
        "- Only `decomposition` and `step_lines` are rewritten. `hit_max_new_tokens`, "
        "`stopped_at_step_line_cap`, token counts and latency are copied as they were, "
        "and describe the original generation.",
        "",
    ]
    for c in changed:
        note_lines.append(
            f"- `{c['query_id']}`: {len(c['pred_steps_before'])} -> "
            f"{len(c['pred_steps_after'])} steps; before {c['pred_steps_before']!r}; "
            f"after {c['pred_steps_after']!r}"
        )
    write_run_artifacts(
        args.output_dir,
        config_snapshot=snapshot,
        metrics=metrics,
        note_title="Re-parse of stored decompositions (issue #60, H3)",
        note_lines=note_lines,
        prefix="reparse_",
    )
    print(f"[reparse] {results_path}: {len(changed)}/{len(rows)} rows changed "
          f"({steps_removed} steps removed); wrote {out_path}")
    for c in changed:
        print(f"  {c['query_id']}: before={c['pred_steps_before']!r}")
        print(f"  {' ' * len(c['query_id'])}  after ={c['pred_steps_after']!r}")


if __name__ == "__main__":
    main()
