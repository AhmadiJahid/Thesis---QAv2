"""The trail every run leaves: a config snapshot, a metrics JSON, a run note.

CLAUDE.md requires all three for every run. Doing it in one place means a new
script cannot forget one of them, and the file names stay the same across the
pipeline so the log entry can point at a predictable directory.

It is also where git provenance is stamped. Gate 2 requires every run to come from
committed code, and the only way to check that later is to record which commit
produced the artifacts and whether the tree was dirty at the time. Doing it at this
choke point means no script can produce an unattributable metrics file.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

CONFIG_SNAPSHOT_NAME = "config.json"
METRICS_NAME = "metrics.json"
NOTES_NAME = "notes.md"

REPO_ROOT = Path(__file__).resolve().parent.parent


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _git(*args: str) -> str | None:
    """Run a git command in the repo; None when git is unavailable or fails."""
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def git_provenance() -> dict[str, Any]:
    """Which commit produced this run, and was the tree dirty when it did.

    ``dirty`` true means the run did **not** come from committed code, so its
    metrics cannot be reproduced from the recorded commit alone. ``dirty_files``
    lists what differed, capped so a large working tree cannot bloat the artifact.
    """
    commit = _git("rev-parse", "HEAD")
    if commit is None:
        return {
            "commit": None,
            "branch": None,
            "dirty": None,
            "note": "git provenance unavailable (not a git repo, or git not installed)",
        }

    status = _git("status", "--porcelain") or ""
    changed = [line.strip() for line in status.splitlines() if line.strip()]
    return {
        "commit": commit,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(changed),
        "dirty_file_count": len(changed),
        "dirty_files": changed[:20],
    }


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
    """Write config snapshot + metrics JSON + run note into ``run_dir``.

    Git provenance is stamped into all three; the caller's dicts are not mutated.
    """
    out = Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)

    config_path = out / f"{prefix}{CONFIG_SNAPSHOT_NAME}"
    metrics_path = out / f"{prefix}{METRICS_NAME}"
    notes_path = out / f"{prefix}{NOTES_NAME}"

    provenance = git_provenance()
    _write_json(config_path, {**config_snapshot, "git": provenance})
    _write_json(metrics_path, {**metrics, "git": provenance})

    body = [f"# {note_title}", ""]
    body.extend(str(line) for line in note_lines)
    body.append("")
    commit = provenance.get("commit")
    if commit:
        dirty = provenance.get("dirty")
        body.append(
            f"- Code: commit `{commit}` on `{provenance.get('branch')}`"
            + (
                f" — **working tree dirty** ({provenance.get('dirty_file_count')} file(s)); "
                "this run did not come from committed code"
                if dirty
                else " (clean tree)"
            )
        )
    else:
        body.append(f"- Code: {provenance.get('note')}")
    body.append(f"- Config snapshot: `{config_path}`")
    body.append(f"- Metrics: `{metrics_path}`")
    notes_path.write_text("\n".join(body) + "\n", encoding="utf-8")

    if provenance.get("dirty"):
        print(
            f"[artifacts] WARNING working tree is dirty "
            f"({provenance.get('dirty_file_count')} file(s)): this run is not reproducible "
            f"from commit {commit} alone."
        )

    print(f"[artifacts] config  -> {config_path}")
    print(f"[artifacts] metrics -> {metrics_path}")
    print(f"[artifacts] notes   -> {notes_path}")
    return {
        "config": str(config_path),
        "metrics": str(metrics_path),
        "notes": str(notes_path),
    }
