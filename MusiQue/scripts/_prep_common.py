"""Shared plumbing for the MuSiQue prep scripts.

Puts the repo's ``src/`` on the path, loads ``configs/musique_prep.json`` plus
``configs/paths.json``, and resolves the data-root-relative paths and globs those
configs hold. v1 built these paths from ``REPO_ROOT`` inside every script, which is
how cluster-specific locations ended up in the code.
"""
from __future__ import annotations

import glob as _glob
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_config import (  # noqa: E402
    load_config,
    load_paths,
    require,
    resolve_path,
    runs_path,
)


def load_prep(config: str | Path = "musique_prep.json") -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (prep config, resolved paths config)."""
    cfg = load_config(config)
    paths_cfg = load_paths(require(cfg, "paths_config"))
    return cfg, paths_cfg


def data_root(paths_cfg: dict[str, Any]) -> Path:
    return Path(paths_cfg["data_root_resolved"])


def dataset_path(paths_cfg: dict[str, Any], key: str) -> Path:
    """Resolve ``datasets.<key>`` against the data root."""
    return resolve_path(require(paths_cfg, f"datasets.{key}"), data_root(paths_cfg))


def under_data_root(paths_cfg: dict[str, Any], rel: str | Path) -> Path:
    return resolve_path(rel, data_root(paths_cfg))


def run_dir_for(paths_cfg: dict[str, Any], subdir: str) -> Path:
    out = runs_path(paths_cfg, subdir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def expand_glob(paths_cfg: dict[str, Any], pattern: str) -> list[Path]:
    """Expand a data-root-relative glob (absolute patterns pass through)."""
    p = Path(pattern)
    full = str(p if p.is_absolute() else data_root(paths_cfg) / p)
    return [Path(m).resolve() for m in sorted(_glob.glob(full, recursive=False))]


__all__ = [
    "REPO_ROOT",
    "data_root",
    "dataset_path",
    "expand_glob",
    "load_config",
    "load_paths",
    "load_prep",
    "require",
    "resolve_path",
    "run_dir_for",
    "runs_path",
    "under_data_root",
]
