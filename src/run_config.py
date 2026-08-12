"""Config loading helpers.

Every script in this repo reads its parameters from a committed JSON config under
``configs/``. These helpers exist so that a missing key is a loud error instead of
a silent hard-coded default: ``require()`` raises, there is no ``.get(key, 42)``
anywhere in the pipeline.

Paths inside a config are interpreted relative to a *root*:

- dataset paths resolve against ``data_root`` from ``configs/paths.json``
  (which defaults to a location outside the working tree, because data never
  enters git),
- repo-internal paths (prompts, committed few-shot exemplars) resolve against the
  repo root,
- output paths resolve against ``runs_root`` (gitignored by default).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"


class ConfigError(SystemExit):
    """Raised (as a SystemExit) when a config is missing a required field."""


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON config. ``path`` may be absolute, or relative to configs/."""
    p = Path(path)
    if not p.is_absolute():
        candidate = Path.cwd() / p
        p = candidate if candidate.exists() else CONFIG_DIR / p.name
    if not p.exists():
        raise ConfigError(f"config not found: {p}")
    with p.open(encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ConfigError(f"config must be a JSON object: {p}")
    cfg["_config_path"] = str(p.resolve())
    return cfg


def require(cfg: dict[str, Any], key: str) -> Any:
    """Fetch ``key`` (dotted path allowed) or fail loudly. No silent defaults."""
    node: Any = cfg
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            src = cfg.get("_config_path", "<config>")
            raise ConfigError(f"missing required config key {key!r} in {src}")
        node = node[part]
    return node


def optional(cfg: dict[str, Any], key: str) -> Any:
    """Fetch ``key`` (dotted path) or return None. Only for genuinely optional fields."""
    node: Any = cfg
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def expand(path_value: str) -> Path:
    """Expand ``~`` and environment variables in a config path value."""
    return Path(os.path.expandvars(str(path_value))).expanduser()


def resolve_path(path_value: str | Path, root: Path) -> Path:
    """Resolve a config path value against ``root`` when it is not absolute."""
    p = expand(str(path_value))
    return p if p.is_absolute() else (root / p)


#: Selects a different *committed* paths config for every script in one go. The only
#: intended use is the synthetic smoke test, which points the whole pipeline at
#: tests/fixtures/ (see scripts/smoke_test.py). It selects a config file; it cannot set
#: individual values, and the file it selects is recorded in every run's config snapshot.
PATHS_CONFIG_ENV = "QAV2_PATHS_CONFIG"


def load_paths(paths_config: str | Path = "paths.json") -> dict[str, Any]:
    """Load configs/paths.json (or the PATHS_CONFIG_ENV override) and resolve its roots."""
    override = os.environ.get(PATHS_CONFIG_ENV)
    if override:
        print(f"[config] {PATHS_CONFIG_ENV}={override} overrides paths config {paths_config!r}")
        paths_config = override
    cfg = load_config(paths_config)
    cfg["data_root_resolved"] = str(resolve_path(require(cfg, "data_root"), REPO_ROOT))
    cfg["runs_root_resolved"] = str(resolve_path(require(cfg, "runs_root"), REPO_ROOT))
    cfg["repo_root_resolved"] = str(REPO_ROOT)
    return cfg


def data_path(paths_cfg: dict[str, Any], key: str) -> Path:
    """Resolve ``datasets.<key>`` from paths.json against ``data_root``."""
    rel = require(paths_cfg, f"datasets.{key}")
    return resolve_path(rel, Path(paths_cfg["data_root_resolved"]))


def repo_path(paths_cfg: dict[str, Any], key: str) -> Path:
    """Resolve ``repo.<key>`` from paths.json against the repo root."""
    rel = require(paths_cfg, f"repo.{key}")
    return resolve_path(rel, REPO_ROOT)


def runs_path(paths_cfg: dict[str, Any], *parts: str) -> Path:
    """Build a path under the configured runs root."""
    out = Path(paths_cfg["runs_root_resolved"])
    for part in parts:
        out = out / part
    return out
