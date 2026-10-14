"""Named, non-pinned evaluation sets, asserted by id (issue #56, ADR 0037).

The ADR 0007 pinned 600 is asserted inside ``components/decomposer/run_decomposer.py``
(:func:`assert_pinned_eval_set`), and that path is left exactly as it was. This module adds
a second, *explicitly selected* kind of evaluation set: a name in ``configs/eval_sets.json``
whose ids come from a source file under ``data_root``. The decomposer runner and the
evaluator both take ``--eval-set <name>`` and call :func:`assert_named_eval_set`, so the two
scripts check the same definition rather than two copies of it.

The assertion is by **id**, as the pinned one is: the loaded ids must be exactly the source
file's answerable ids, and their number must equal ``expected_total`` - so a different file
behind the same key is refused, not silently used. ``allow_mismatch`` is the recorded
opt-out for fixture and smoke runs (a handful of questions); it downgrades the refusal to a
warning and the record says ``matches_named_set: false``.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from run_config import data_path, load_config, require

EVAL_SETS_CONFIG = "eval_sets.json"

#: A paths config may override a named set's ``expected_total`` under this key. Only the
#: synthetic fixture tree uses it (``configs/smoke_paths.json``), the same way it overrides
#: the fine-tune eval-set counts under ``eval_set_expected``: its fabricated dev file holds a
#: handful of rows, so the count is declared there rather than relaxed in code.
#: ``configs/paths.json`` carries no such key, so real runs assert the committed total.
PATHS_EXPECTED_TOTAL_KEY = "named_eval_set_expected_total"

_ID_HOP_RX = re.compile(r"^(?P<h>\d+)hop")


def hop_from_id(qid: str) -> int | None:
    """The hop depth MuSiQue writes into an id ('3hop1__...' -> 3), or None."""
    m = _ID_HOP_RX.match(str(qid))
    return int(m.group("h")) if m else None


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sample(ids: list[str], limit: int = 10) -> str:
    shown = ", ".join(ids[:limit])
    return shown + (f" ... (+{len(ids) - limit} more)" if len(ids) > limit else "")


@dataclass
class NamedEvalSet:
    """A named evaluation set as loaded from its source file."""

    name: str
    source_path: str
    source_sha256: str
    ids: set[str]
    ids_per_hop: dict[str, int]
    hops: list[int]
    expected_total: int
    problems: list[str] = field(default_factory=list)

    def as_record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "ids_total": len(self.ids),
            "ids_per_hop": dict(self.ids_per_hop),
            "hops": list(self.hops),
            "expected_total": self.expected_total,
            "definition_problems": list(self.problems),
        }


def load_named_eval_set(
    name: str,
    paths_cfg: dict[str, Any],
    *,
    eval_sets_config: str | Path = EVAL_SETS_CONFIG,
) -> NamedEvalSet:
    """Load ``name`` from ``configs/eval_sets.json`` and read its ids from the source file.

    Problems with the definition itself (missing file, duplicate ids, unparseable hop,
    a hop the set does not list, a total that is not ``expected_total``) are collected, not
    raised, so the caller's ``allow_mismatch`` decides whether they refuse - the same shape
    as the pinned path's ``load_pinned_eval_ids``.
    """
    cfg = load_config(eval_sets_config)
    if name.startswith("_") or name == "paths_config" or not isinstance(cfg.get(name), dict):
        known = sorted(k for k, v in cfg.items() if isinstance(v, dict) and not k.startswith("_"))
        raise SystemExit(
            f"--eval-set {name!r} is not defined in {cfg.get('_config_path')}; known: {known}"
        )
    spec = cfg[name]
    source_key = require(spec, "source_key")
    id_field = require(spec, "id_field")
    answerable_field = require(spec, "answerable_field")
    hops = [int(h) for h in require(spec, "hops")]
    expected_total = int(require(spec, "expected_total"))
    override = (paths_cfg.get(PATHS_EXPECTED_TOTAL_KEY) or {}).get(name)
    if override is not None:
        expected_total = int(override)

    path = data_path(paths_cfg, source_key)
    problems: list[str] = []
    ids: set[str] = set()
    per_hop: dict[str, int] = {}
    sha = ""
    if not path.exists():
        problems.append(f"source file for eval set {name!r} not found: {path}")
    else:
        sha = _sha256_file(path)
        with path.open(encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get(answerable_field) is not True:
                    continue
                qid = row.get(id_field)
                if not isinstance(qid, str) or not qid:
                    problems.append(f"{path}:{lineno} has no usable {id_field!r}")
                    continue
                if qid in ids:
                    problems.append(f"duplicate id in {path}: {qid}")
                    continue
                hop = hop_from_id(qid)
                if hop is None:
                    problems.append(f"id with no parseable hop depth: {qid}")
                    continue
                if hop not in hops:
                    problems.append(f"id at a hop depth the set does not list ({hop}): {qid}")
                    continue
                ids.add(qid)
                per_hop[str(hop)] = per_hop.get(str(hop), 0) + 1
        if len(ids) != expected_total:
            problems.append(
                f"eval set {name!r} has {len(ids)} answerable ids in {path}, "
                f"expected_total is {expected_total}"
            )
    return NamedEvalSet(
        name=name,
        source_path=str(path),
        source_sha256=sha,
        ids=ids,
        ids_per_hop={k: per_hop[k] for k in sorted(per_hop)},
        hops=hops,
        expected_total=expected_total,
        problems=problems,
    )


def assert_named_eval_set(
    loaded_ids: set[str],
    rows_per_hop: dict[str, int],
    total: int,
    named: NamedEvalSet,
    *,
    allow_mismatch: bool,
    source: str,
    component: str,
    allow_flag: str,
) -> dict[str, Any]:
    """Assert the run loaded exactly ``named``'s ids; return the ``evaluation_set`` record.

    ``total`` is the number of rows loaded (so a duplicated row is caught even when the id
    set matches). ``allow_flag`` names the caller's opt-out flag in the messages.
    """
    problems = list(named.problems)
    missing = sorted(named.ids - loaded_ids)
    extra = sorted(loaded_ids - named.ids)
    if missing:
        problems.append(f"{len(missing)} id(s) of the set were not loaded: {_sample(missing)}")
    if extra:
        problems.append(f"{len(extra)} loaded id(s) are not in the set: {_sample(extra)}")
    if total != len(loaded_ids):
        problems.append(f"{total} rows loaded for {len(loaded_ids)} distinct ids (duplicates)")

    record: dict[str, Any] = {
        "named_eval_set": named.name,
        # The ADR 0007 pinned 600 is a different set: a named-set run is never 'pinned'.
        "pinned": False,
        "named_set_asserted": True,
        "matches_named_set": not problems,
        "allow_mismatch_flag": allow_mismatch,
        "rows_source": source,
        "id_identity_checked": True,
        "ids_missing_count": len(missing),
        "ids_unexpected_count": len(extra),
        "rows_loaded_total": total,
        "rows_loaded_per_hop": dict(rows_per_hop),
        "definition": named.as_record(),
    }
    if not problems:
        return record

    detail = "\n".join(f"  - {p}" for p in problems)
    if allow_mismatch:
        record["note"] = (
            f"the rows are not the named evaluation set {named.name!r}; permitted by "
            f"{allow_flag}. This run is a fixture/smoke run, not an experiment arm.\n" + detail
        )
        print(
            f"[{component}] WARNING: not the named evaluation set {named.name!r} (allowed by "
            f"{allow_flag}):\n{detail}"
        )
        return record
    raise SystemExit(
        f"[{component}] the rows are not the named evaluation set {named.name!r} "
        f"({named.expected_total} ids from {named.source_path}).\n{detail}\n"
        f"Loaded per hop: {rows_per_hop} (total {total}), source: {source}.\n"
        f"A comparison across different evaluation sets is not a comparison (ADR 0011), so "
        f"this is a refusal. Pass {allow_flag} only for a fixture or smoke run."
    )
