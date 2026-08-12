"""
Shared helpers for MuSiQue ``id`` strings (e.g. ``2hop__482757_12019``, ``3hop1__...``).

Used by the split/clean/extract/plot scripts so hop and stratum logic stays consistent.
Ported unchanged from v1 ``MusiQue/scripts/musique_ids.py``.
"""

from __future__ import annotations

import re


def stratum_from_id(record_id: object) -> str:
    """Return the prefix before ``__`` (e.g. ``2hop``, ``3hop1``), or ``unknown``."""
    if not isinstance(record_id, str) or "__" not in record_id:
        return "unknown"
    return record_id.split("__", 1)[0] or "unknown"


def coarse_hop_from_id(record_id: object) -> int:
    """
    Derive coarse hop count (2, 3, 4, ...) from the id prefix.
    Returns -1 if the prefix does not start with a digit.
    """
    s = stratum_from_id(record_id)
    if s == "unknown":
        return -1
    if s and s[0].isdigit():
        return int(s[0])
    return -1


def coarse_hop_from_record(obj: dict) -> int:
    """Prefer numeric ``hop_count`` when present; else derive from ``id``."""
    hc = obj.get("hop_count")
    if isinstance(hc, int) and hc >= 0:
        return hc
    return coarse_hop_from_id(obj.get("id", ""))


def stratum_to_questions_slug(stratum: str) -> str:
    """
    Map an id-prefix stratum to a filename fragment, e.g. ``2hop`` -> ``2_hop``,
    ``3hop1`` -> ``3_hop_1``, ``4hop3`` -> ``4_hop_3``.
    Unknown / unexpected values are sanitized for safe paths.
    """
    if stratum == "2hop":
        return "2_hop"
    m = re.fullmatch(r"(\d)hop(\d+)", stratum)
    if m:
        return f"{m.group(1)}_hop_{m.group(2)}"
    if stratum == "unknown":
        return "unknown"
    safe = re.sub(r"[^\w]+", "_", stratum).strip("_")
    return safe or "unknown"
