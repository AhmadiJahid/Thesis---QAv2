"""Hop-matched few-shot retrieval: the candidate filter that binds *before* top-k.

Issue #15 asks whether retrieving few-shot examples of the same hop depth as the query
helps. Three conditions, all produced by the same retrieval chain:

- **mixed** — no hop constraint. ``enabled = false``, which is the retrieval behaviour
  this repo had before this module existed.
- **oracle-hop-matched** — candidates restricted to the pool rows whose hop depth equals
  the query's *gold* hop depth, parsed from the query id (``2hop__…`` / ``3hop1__…`` /
  ``4hop2__…``). ``hop_source = "gold"``.
- **router-hop-matched** — the same restriction, but the hop comes from an external
  predictions file. ``hop_source = "predictions"``. No router exists in v2 yet; this is
  the interface it will write into, not the router.

**The filter is a candidate filter, not a post-filter.** With matching on, the bi-encoder
top-k is computed over the query's hop bucket only, so the whole downstream chain
(``truncate_top20.py``, ``rerank_similarity_results.py``, the decomposer's few-shot block)
sees k in-bucket candidates. Filtering a mixed top-20 afterwards would be a different
method: it would return fewer than k examples for some queries and would make the matched
condition's example count depend on the mixed ranking.

**Nothing here falls back.** A pool or query id that does not parse, a query with no
prediction in the predictions file, or a hop bucket that cannot supply the requested k is a
hard error with counts. A silent fallback to mixed for a subset of queries would make the
matched condition a blend of two conditions and the comparison meaningless.

Design defaults recorded in ADR 0022.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from run_config import optional, require

#: Where the query's hop depth comes from. Any other value is refused at run time rather
#: than silently treated as one of these (the shape ADR 0019/0021 use for policy knobs).
HOP_SOURCES = ("gold", "predictions")

#: ``min_candidates`` sentinel: the query's bucket must hold at least ``top_k`` rows, so a
#: matched query gets the same number of candidates as a mixed one.
MIN_CANDIDATES_TOP_K = "top_k"

#: Ids look like ``2hop__482757_12019`` / ``3hop1__…`` / ``4hop2__…``; the leading digits
#: are the coarse hop depth. Same rule as ``sample_pool.py`` and the decomposer.
_ID_HOP_RX = re.compile(r"^(?P<h>\d+)hop")

#: How many offending ids an error message lists before it truncates.
_MAX_REPORTED = 10


def parse_hop_from_id(value: Any) -> int | None:
    """Coarse hop depth from a MuSiQue row id, or None when the id does not parse."""
    if not isinstance(value, str):
        return None
    m = _ID_HOP_RX.match(value.strip())
    return int(m.group("h")) if m else None


def _sample(items: list[str]) -> str:
    shown = items[:_MAX_REPORTED]
    suffix = f" (+{len(items) - len(shown)} more)" if len(items) > len(shown) else ""
    return ", ".join(repr(s) for s in shown) + suffix


@dataclass(frozen=True)
class HopMatchSettings:
    """The knobs, resolved from config + CLI. Held frozen so no stage mutates them."""

    enabled: bool
    hop_source: str
    predictions_file: Path | None
    predictions_id_field: str
    predictions_hop_field: str
    min_candidates: int | str

    def resolve_min_candidates(self, top_k: int) -> int:
        if self.min_candidates == MIN_CANDIDATES_TOP_K:
            return int(top_k)
        return int(self.min_candidates)

    def as_record(self, top_k: int | None = None) -> dict[str, Any]:
        """The settings as they belong in a metrics JSON / config snapshot."""
        if not self.enabled:
            return {"enabled": False}
        record: dict[str, Any] = {
            "enabled": True,
            "hop_source": self.hop_source,
            "predictions_file": str(self.predictions_file) if self.predictions_file else None,
            "predictions_id_field": self.predictions_id_field,
            "predictions_hop_field": self.predictions_hop_field,
            "min_candidates": self.min_candidates,
        }
        if top_k is not None:
            record["min_candidates_resolved"] = self.resolve_min_candidates(top_k)
        return record


def settings_from_config(
    cfg: dict[str, Any],
    *,
    enabled: bool | None = None,
    hop_source: str | None = None,
    predictions_file: str | Path | None = None,
) -> HopMatchSettings:
    """Read the ``hop_match`` block; CLI arguments (when not None) win over the config.

    ``enabled = None`` means "whatever the config says" — the mixed condition is the
    config default, so a stage that passes no flag keeps today's behaviour.
    """
    src = cfg.get("_config_path", "<config>")
    resolved_enabled = bool(require(cfg, "hop_match.enabled")) if enabled is None else bool(enabled)
    resolved_source = hop_source or require(cfg, "hop_match.hop_source")
    if resolved_source not in HOP_SOURCES:
        raise SystemExit(
            f"hop_match.hop_source={resolved_source!r} is not one of {list(HOP_SOURCES)} "
            f"(config: {src}). A hop source that is not implemented is refused rather than "
            f"treated as 'gold'."
        )
    file_value = predictions_file if predictions_file is not None else optional(
        cfg, "hop_match.predictions_file"
    )
    resolved_file = Path(str(file_value)) if file_value else None
    if resolved_enabled and resolved_source == "predictions" and resolved_file is None:
        raise SystemExit(
            "hop_match.hop_source='predictions' needs a predictions file: pass "
            "--hop-predictions or set 'hop_match.predictions_file' in "
            f"{src}. Without it the router condition would silently become the mixed one."
        )
    min_candidates = require(cfg, "hop_match.min_candidates")
    if min_candidates != MIN_CANDIDATES_TOP_K:
        if not isinstance(min_candidates, int) or isinstance(min_candidates, bool) or min_candidates < 0:
            raise SystemExit(
                f"hop_match.min_candidates={min_candidates!r} must be a non-negative int or "
                f"the string {MIN_CANDIDATES_TOP_K!r} (config: {src})."
            )
    return HopMatchSettings(
        enabled=resolved_enabled,
        hop_source=resolved_source,
        predictions_file=resolved_file,
        predictions_id_field=require(cfg, "hop_match.predictions_id_field"),
        predictions_hop_field=require(cfg, "hop_match.predictions_hop_field"),
        min_candidates=min_candidates,
    )


def load_predicted_hops(
    path: Path, *, id_field: str, hop_field: str
) -> dict[str, int]:
    """Read a predictions JSONL into ``{query_id: predicted_hop}``.

    One row per query id. A duplicate id is refused: two predictions for one query means
    the file does not say which hop the router chose.
    """
    if not path.exists():
        raise SystemExit(f"hop predictions file not found: {path}")
    hops: dict[str, int] = {}
    duplicates: list[str] = []
    bad_rows: list[str] = []
    rows = 0
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno} is not valid JSON: {exc}") from exc
            if not isinstance(row, dict):
                bad_rows.append(f"line {lineno}: not an object")
                continue
            qid = row.get(id_field)
            hop = row.get(hop_field)
            if not isinstance(qid, str) or not qid.strip():
                bad_rows.append(f"line {lineno}: {id_field!r} missing or not a string")
                continue
            if isinstance(hop, bool) or not isinstance(hop, int):
                bad_rows.append(f"line {lineno}: {hop_field!r}={hop!r} is not an int")
                continue
            qid = qid.strip()
            if qid in hops:
                duplicates.append(qid)
                continue
            hops[qid] = hop
    if bad_rows:
        raise SystemExit(
            f"hop predictions file {path} has {len(bad_rows)} unusable row(s): "
            f"{_sample(bad_rows)}. Expected one JSON object per line with "
            f"{id_field!r} (string) and {hop_field!r} (int)."
        )
    if duplicates:
        raise SystemExit(
            f"hop predictions file {path} repeats {len(duplicates)} query id(s): "
            f"{_sample(sorted(set(duplicates)))}. One prediction per query id."
        )
    if not hops:
        raise SystemExit(f"hop predictions file {path} holds no predictions ({rows} rows read)")
    return hops


def group_pool_by_hop(pool_rows: list[dict[str, Any]], *, id_field: str = "id") -> dict[int, list[int]]:
    """``{hop: [pool row indices]}``, index order preserved.

    Every pool row must carry a parseable id: a row that cannot be bucketed could never be
    retrieved under matching, which would quietly shrink the pool for this condition only.
    """
    buckets: dict[int, list[int]] = {}
    unparseable: list[str] = []
    for idx, row in enumerate(pool_rows):
        hop = parse_hop_from_id(row.get(id_field))
        if hop is None:
            unparseable.append(f"index {idx}: {row.get(id_field)!r}")
            continue
        buckets.setdefault(hop, []).append(idx)
    if unparseable:
        raise SystemExit(
            f"hop-matched retrieval: {len(unparseable)} of {len(pool_rows)} pool rows have an "
            f"id whose hop depth does not parse: {_sample(unparseable)}. Ids must look like "
            f"'2hop__…' / '3hop1__…'. Fix the pool rather than dropping the rows: under "
            f"matching an unbucketable row is silently unreachable."
        )
    return buckets


@dataclass(frozen=True)
class HopFilter:
    """Per-query allowed pool indices, plus the counts the run has to record."""

    settings: HopMatchSettings
    top_k: int
    min_candidates: int
    #: allowed[i] = pool row indices the i-th query may retrieve from, ascending.
    allowed: list[list[int]]
    #: query_hops[i] = the hop depth the filter used for the i-th query.
    query_hops: list[int]
    pool_bucket_counts: dict[int, int]
    query_hop_counts: dict[int, int]

    def summary(self) -> dict[str, Any]:
        return {
            **self.settings.as_record(self.top_k),
            "queries": len(self.allowed),
            "pool_bucket_counts": {str(h): n for h, n in sorted(self.pool_bucket_counts.items())},
            "query_hop_counts": {str(h): n for h, n in sorted(self.query_hop_counts.items())},
            "candidates_per_query_min": min((len(a) for a in self.allowed), default=0),
            "candidates_per_query_max": max((len(a) for a in self.allowed), default=0),
        }


def build_hop_filter(
    *,
    query_rows: list[dict[str, Any]],
    pool_buckets: dict[int, list[int]],
    settings: HopMatchSettings,
    top_k: int,
    predicted_hops: dict[str, int] | None = None,
    query_id_field: str = "id",
) -> HopFilter | None:
    """The candidate filter for one query file, or None when matching is off.

    None is the mixed condition: the caller keeps its unfiltered code path, so mixed is
    not "the filter with everything allowed" but literally the code that ran before.
    """
    if not settings.enabled:
        return None
    if not query_rows:
        raise SystemExit("hop-matched retrieval: no query rows to build a filter for")

    min_candidates = settings.resolve_min_candidates(top_k)
    query_hops: list[int] = []
    unparseable: list[str] = []
    missing_prediction: list[str] = []

    for idx, row in enumerate(query_rows):
        qid = row.get(query_id_field)
        if settings.hop_source == "gold":
            hop = parse_hop_from_id(qid)
            if hop is None:
                unparseable.append(f"index {idx}: {qid!r}")
                continue
        else:
            if not isinstance(qid, str) or not qid.strip():
                unparseable.append(f"index {idx}: {qid!r}")
                continue
            hop = (predicted_hops or {}).get(qid.strip())
            if hop is None:
                missing_prediction.append(qid.strip())
                continue
        query_hops.append(hop)

    if unparseable:
        what = (
            "gold hop depth does not parse from the query id"
            if settings.hop_source == "gold"
            else "query id is missing, so no prediction can be looked up"
        )
        raise SystemExit(
            f"hop-matched retrieval ({settings.hop_source}): {len(unparseable)} of "
            f"{len(query_rows)} queries — {what}: {_sample(unparseable)}."
        )
    if missing_prediction:
        raise SystemExit(
            f"hop-matched retrieval (predictions): {len(missing_prediction)} of "
            f"{len(query_rows)} queries have no prediction in "
            f"{settings.predictions_file}: {_sample(missing_prediction)}. The predictions "
            f"file must cover every query; a missing one is not a mixed-condition query."
        )

    query_hop_counts: dict[int, int] = {}
    for hop in query_hops:
        query_hop_counts[hop] = query_hop_counts.get(hop, 0) + 1

    infeasible = [
        f"hop {hop}: {len(pool_buckets.get(hop, []))} pool candidates for "
        f"{query_hop_counts[hop]} quer{'y' if query_hop_counts[hop] == 1 else 'ies'}"
        for hop in sorted(query_hop_counts)
        if len(pool_buckets.get(hop, [])) < min_candidates
    ]
    if infeasible:
        raise SystemExit(
            f"hop-matched retrieval: a hop bucket cannot supply the required "
            f"{min_candidates} candidates (top_k={top_k}, "
            f"min_candidates={settings.min_candidates!r}):\n  "
            + "\n  ".join(infeasible)
            + "\n  pool buckets: "
            + str({h: len(v) for h, v in sorted(pool_buckets.items())})
            + "\nBuild a pool that covers every queried hop depth, or lower "
            "hop_match.min_candidates deliberately — retrieving fewer than k in-bucket "
            "examples is a different method, not this one."
        )

    allowed = [list(pool_buckets[hop]) for hop in query_hops]
    return HopFilter(
        settings=settings,
        top_k=int(top_k),
        min_candidates=min_candidates,
        allowed=allowed,
        query_hops=query_hops,
        pool_bucket_counts={h: len(v) for h, v in pool_buckets.items()},
        query_hop_counts=query_hop_counts,
    )


__all__ = [
    "HOP_SOURCES",
    "MIN_CANDIDATES_TOP_K",
    "HopFilter",
    "HopMatchSettings",
    "build_hop_filter",
    "group_pool_by_hop",
    "load_predicted_hops",
    "parse_hop_from_id",
    "settings_from_config",
]
