"""The frozen anchor id list (ADR 0033), read by everything that has to agree on it.

Three callers need the same 45 ids and, more importantly, the same *file naming* for the
prompts logged against them:

- ``components/decomposer/run_decomposer.py`` writes one prompt file per anchor id;
- ``scripts/render_anchor_set.py`` finds those files again, by id;
- ``tests/`` pins both halves.

Two copies of the "what is an anchor prompt file called" rule is exactly the kind of
divergence that makes a renderer quietly show "prompt not logged" for a prompt that was in
fact logged, so the rule lives here once. ``scripts/build_anchor_set.py`` writes the list
and is deliberately not a caller: it is the producer, this module is the reader.

The list itself holds ids and provenance only - no question text - which is why it is a
committed config rather than a run artifact (CLAUDE.md: data never enters git).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

#: The schema string ``scripts/build_anchor_set.py`` writes. A file that does not carry it
#: is refused by name rather than half-parsed.
ANCHOR_SET_SCHEMA = "musique_anchor_set/1"

#: Prefix of a per-anchor prompt log file. The every-Nth sample keeps its own
#: ``prompt_idx{i:04d}_hop{h}.txt`` name untouched, so existing parity checks over those
#: files see exactly the files they saw before; the anchor files are additional and are
#: named by *id*, because position is not stable across arms and an id is.
ANCHOR_PROMPT_PREFIX = "anchor_"

#: Characters an id may contain and still be used verbatim in a file name. MuSiQue ids
#: (``2hop__623501_297043``) are inside this set. An id outside it is refused at load time
#: rather than mangled: a mangled name could collide with another id's file, and then the
#: renderer would join a prompt onto the wrong question.
_SAFE_ID = re.compile(r"\A[A-Za-z0-9._-]+\Z")


class AnchorSetError(ValueError):
    """The anchor id list is missing, malformed, or carries an unusable id."""


def anchor_prompt_filename(query_id: str, hop: Any) -> str:
    """The prompt-log file name for one anchor item: ``anchor_<query_id>_hop<h>.txt``.

    The hop suffix mirrors the every-Nth file's, so a directory listing groups by hop the
    same way in both halves of the log. The *id* is what the renderer matches on; the hop
    is display only, which is why the lookup below globs the suffix.
    """
    return f"{ANCHOR_PROMPT_PREFIX}{query_id}_hop{hop}.txt"


def anchor_prompt_glob(query_id: str) -> str:
    """The pattern that finds one anchor item's prompt file whatever hop it was filed under."""
    return f"{ANCHOR_PROMPT_PREFIX}{query_id}_hop*.txt"


def find_anchor_prompt(prompts_dir: Path, query_id: str) -> Path | None:
    """The logged prompt file for ``query_id`` in ``prompts_dir``, or ``None``.

    ``None`` means *not logged*, and every caller renders that as an explicit marker rather
    than as an empty section: a silent gap reads as "the prompt was empty".
    """
    if not prompts_dir.is_dir():
        return None
    matches = sorted(prompts_dir.glob(anchor_prompt_glob(query_id)))
    return matches[0] if matches else None


def _validate_id(qid: Any, path: Path) -> str:
    if not isinstance(qid, str) or not qid.strip():
        raise AnchorSetError(f"{path}: an anchor id is {qid!r}; expected a non-empty string")
    qid = qid.strip()
    if not _SAFE_ID.match(qid):
        raise AnchorSetError(
            f"{path}: anchor id {qid!r} contains characters that cannot go in a prompt-log "
            "file name (allowed: letters, digits, '.', '_', '-'). The anchor prompt files "
            "are named by id, so an id that has to be escaped could collide with another "
            "id's file and the renderer would join a prompt onto the wrong question."
        )
    return qid


def load_anchor_ids(path: Path | str) -> tuple[list[str], dict[str, str], dict[str, Any]]:
    """``(ids in anchor order, {id: hop}, the list's provenance header)``.

    Anchor order is hops in the committed order, ids sorted within a hop - the order the
    draw script writes - so two sheets rendered on different days line up item for item.

    Raises :class:`AnchorSetError` on anything it cannot read unambiguously; callers turn
    that into their own refusal (a ``SystemExit`` in the scripts).
    """
    path = Path(path)
    if not path.exists():
        raise AnchorSetError(
            f"anchor id list not found: {path}\n"
            "Draw it once with: python scripts/build_anchor_set.py --write"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    ids_by_hop = payload.get("ids_by_hop")
    if not isinstance(ids_by_hop, dict):
        raise AnchorSetError(
            f"{path}: missing an 'ids_by_hop' object (schema {ANCHOR_SET_SCHEMA})"
        )
    ordered: list[str] = []
    hop_of: dict[str, str] = {}
    for hop in payload.get("hops", sorted(ids_by_hop)):
        bucket = ids_by_hop.get(str(hop))
        if bucket is None:
            raise AnchorSetError(
                f"{path}: 'hops' lists hop {hop} but 'ids_by_hop' has no bucket for it. "
                "The list was hand-edited or written by an older draw; re-derive it with "
                "python scripts/build_anchor_set.py --verify to see what moved."
            )
        if not isinstance(bucket, list):
            raise AnchorSetError(
                f"{path}: ids_by_hop[{hop}] is {type(bucket).__name__}, expected a list of ids."
            )
        for raw in bucket:
            qid = _validate_id(raw, path)
            if qid in hop_of:
                raise AnchorSetError(f"{path}: id {qid!r} appears under more than one hop")
            hop_of[qid] = str(hop)
            ordered.append(qid)
    if not ordered:
        raise AnchorSetError(f"{path}: the anchor list is empty")
    header = {
        k: payload.get(k)
        for k in (
            "schema", "seed", "hops", "n_per_hop", "counts",
            "generated_by", "generated_utc", "source",
        )
    }
    return ordered, hop_of, header


def load_anchor_ids_for_logging(
    value: Any, root: Path
) -> tuple[frozenset[str], Path | None]:
    """Resolve ``logging.anchor_ids_file`` into ``(ids, resolved path)``.

    ``None`` (the key present and null) disables anchor prompt logging and yields an empty
    set - that is what the MetaQA config sets, because its questions come from a ``lines``
    file with no ids at all and the 45 are MuSiQue ids. The key is *required* in both shared
    configs rather than defaulted here, following the ``few_shot.ordering`` precedent: which
    items a run logs a prompt for is a design decision, and a run must not get one by
    omission.
    """
    if value is None:
        return frozenset(), None
    if not isinstance(value, str) or not value.strip():
        raise AnchorSetError(
            f"logging.anchor_ids_file is {value!r}; expected a path to an anchor id list "
            "(schema " + ANCHOR_SET_SCHEMA + ") or null to disable anchor prompt logging"
        )
    path = Path(value.strip())
    if not path.is_absolute():
        path = root / path
    ids, _hop_of, _header = load_anchor_ids(path)
    return frozenset(ids), path
