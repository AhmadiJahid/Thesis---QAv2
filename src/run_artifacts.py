"""The trail every run leaves: a config snapshot, a metrics JSON, a run note.

CLAUDE.md requires all three for every run. Doing it in one place means a new
script cannot forget one of them, and the file names stay the same across the
pipeline so the log entry can point at a predictable directory.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

CONFIG_SNAPSHOT_NAME = "config.json"
METRICS_NAME = "metrics.json"
NOTES_NAME = "notes.md"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return path


def write_run_artifacts(
    run_dir: str | Path,
    *,
    config_snapshot: dict[str, Any],
    metrics: dict[str, Any],
    note_title: str,
    note_lines: Iterable[str],
    prefix: str = "",
) -> dict[str, str]:
    """Write config snapshot + metrics JSON + run note into ``run_dir``."""
    out = Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)

    config_path = out / f"{prefix}{CONFIG_SNAPSHOT_NAME}"
    metrics_path = out / f"{prefix}{METRICS_NAME}"
    notes_path = out / f"{prefix}{NOTES_NAME}"

    _write_json(config_path, config_snapshot)
    _write_json(metrics_path, metrics)

    body = [f"# {note_title}", ""]
    body.extend(str(line) for line in note_lines)
    body.append("")
    body.append(f"- Config snapshot: `{config_path}`")
    body.append(f"- Metrics: `{metrics_path}`")
    notes_path.write_text("\n".join(body) + "\n", encoding="utf-8")

    print(f"[artifacts] config  -> {config_path}")
    print(f"[artifacts] metrics -> {metrics_path}")
    print(f"[artifacts] notes   -> {notes_path}")
    return {
        "config": str(config_path),
        "metrics": str(metrics_path),
        "notes": str(notes_path),
    }
