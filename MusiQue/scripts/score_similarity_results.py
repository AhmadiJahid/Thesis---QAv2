#!/usr/bin/env python3
"""
Score similarity search outputs for MuSiQue.

Input: JSONL from ``check_question_similarity.py`` (or the rerank/truncate steps), where
each row has ``query_id`` and any subset of ``raw_top_k`` / ``typed_top_k`` /
``uniform_top_k`` neighbour lists.

Scoring, per query and per available mode:

1. Hop-match score. ``q_hops`` is parsed from the ``query_id`` prefix (``2hop__123`` -> 2).
   Walking the mode's neighbour list (sorted by score), the first rung whose window
   contains a neighbour with the query's hop count wins. Only one rung applies. Both the
   point values and the window sizes come from ``configs/similarity.json`` ->
   ``score.hop_match_points`` and ``score.hop_match_windows``; v1 hard-coded 5/3/1/0
   points over windows of 1/3/5.

2. Similarity bonus (cross-mode). For each mode with neighbours, take the mean
   similarity of the top-K scores; the single best mode gets ``similarity_bonus_points``.
   An exact tie for the maximum gives no mode the bonus.

3. ``total_score = hop_match_score + similarity_bonus``.

Output: the input rows plus ``<mode>_hop_score``, ``<mode>_avg_score_K``,
``<mode>_similarity_bonus``, ``<mode>_total_score``, ``best_mode`` and
``best_mode_score``, plus a summary JSON.

Ported from v1. Adapted for v2: top-k, the point ladder and the run directory come from
config; the run writes the standard trail.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from _prep_common import load_config, load_paths, require, run_dir_for

from run_artifacts import now_iso, write_run_artifacts
from seeding import set_global_seed

# Hop ids look like "2hop__646146_5111" or "3hop1__445573_301867_127418".
_HOP_PREFIX_RE = re.compile(r"^(?P<hops>\d+)hop(?:\d+)?__")


@dataclass
class ModeScores:
    hop_score: int = 0
    avg_score: float = 0.0
    similarity_bonus: int = 0

    @property
    def total(self) -> float:
        return float(self.hop_score) + float(self.similarity_bonus)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def parse_hops_from_id(id_str: Any) -> int | None:
    """Parse the hop count from an id like '2hop__646146_5111'; None when it does not match."""
    if not isinstance(id_str, str):
        return None
    m = _HOP_PREFIX_RE.match(id_str.strip())
    if not m:
        return None
    try:
        return int(m.group("hops"))
    except ValueError:
        return None


def _first_n(seq: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    if n <= 0:
        return []
    return list(seq) if len(seq) <= n else seq[:n]


def compute_hop_match_score(
    q_hops: int | None,
    neighbours: list[dict[str, Any]],
    top_k: int,
    points: dict[str, int],
    windows: dict[str, int],
) -> int:
    """The top1 / top3 / top5 / miss ladder for a single mode.

    Both the point values and the window sizes come from config: v1 hard-coded 5/3/1/0
    points *and* the 1/3/5 windows, so "top3" silently meant "the first three".
    """
    if q_hops is None or not neighbours:
        return int(points["miss"])

    neighbours = _first_n(neighbours, top_k)

    def _hop(n: dict[str, Any]) -> int | None:
        return parse_hops_from_id(n.get("pool_id"))

    for rung in ("top1", "top3", "top5"):
        window = int(windows[rung])
        if any(_hop(n) == q_hops for n in _first_n(neighbours, window)):
            return int(points[rung])
    return int(points["miss"])


def compute_avg_score(neighbours: list[dict[str, Any]], top_k: int) -> float:
    """Average similarity score over the top-K neighbours (0.0 when there are none)."""
    if not neighbours or top_k <= 0:
        return 0.0
    scores: list[float] = []
    for n in _first_n(neighbours, top_k):
        try:
            scores.append(float(n.get("score")))
        except (TypeError, ValueError):
            continue
    return float(mean(scores)) if scores else 0.0


def compute_similarity_bonus(avg_by_mode: dict[str, float], bonus_points: int) -> dict[str, int]:
    """Give the unique best mode the bonus; an exact tie gives no one the bonus."""
    out: dict[str, int] = {m: 0 for m in avg_by_mode}
    if not avg_by_mode:
        return out
    max_val = max(avg_by_mode.values())
    best_modes = [m for m, v in avg_by_mode.items() if v == max_val]
    if len(best_modes) == 1:
        out[best_modes[0]] = int(bonus_points)
    return out


def _score_row(
    row: dict[str, Any],
    top_k: int,
    modes: tuple[str, ...],
    points: dict[str, int],
    windows: dict[str, int],
    bonus_points: int,
) -> dict[str, Any]:
    result = dict(row)  # shallow copy; preserve all original fields
    q_hops = parse_hops_from_id(row.get("query_id"))

    mode_scores: dict[str, ModeScores] = {}
    avg_by_mode: dict[str, float] = {}

    for mode in modes:
        neighbours = row.get(f"{mode}_top_k")
        if not isinstance(neighbours, list) or not neighbours:
            continue
        hop_score = compute_hop_match_score(q_hops, neighbours, top_k, points, windows)
        avg_score = compute_avg_score(neighbours, top_k)
        mode_scores[mode] = ModeScores(hop_score=hop_score, avg_score=avg_score)
        avg_by_mode[mode] = avg_score

    for mode, bonus in compute_similarity_bonus(avg_by_mode, bonus_points).items():
        if mode in mode_scores:
            mode_scores[mode].similarity_bonus = bonus

    for mode, ms in mode_scores.items():
        result[f"{mode}_hop_score"] = ms.hop_score
        result[f"{mode}_avg_score_{top_k}"] = ms.avg_score
        result[f"{mode}_similarity_bonus"] = ms.similarity_bonus
        result[f"{mode}_total_score"] = ms.total

    if mode_scores:
        best_mode = max(mode_scores.items(), key=lambda kv: kv[1].total)[0]
        result["best_mode"] = best_mode
        result["best_mode_score"] = mode_scores[best_mode].total

    return result


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="similarity.json")
    p.add_argument("--input", type=Path, required=True, help="Input JSONL from check_question_similarity.py")
    p.add_argument("--out", type=Path, required=True, help="Output JSONL with added scoring fields")
    p.add_argument("--top-k", type=int, default=None, help="Neighbours considered (default: config score.top_k)")
    p.add_argument("--summary-out", type=Path, default=None, help="Summary JSON (default: <out>_summary.json)")
    p.add_argument("--run-dir", type=Path, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    paths_cfg = load_paths(require(cfg, "paths_config"))

    seed = args.seed if args.seed is not None else int(require(cfg, "seed"))
    seeded = set_global_seed(seed)  # logged; this script does not sample
    top_k = args.top_k if args.top_k is not None else int(require(cfg, "score.top_k"))
    modes = tuple(require(cfg, "modes"))
    points = require(cfg, "score.hop_match_points")
    windows = require(cfg, "score.hop_match_windows")
    bonus_points = int(require(cfg, "score.similarity_bonus_points"))
    run_dir = args.run_dir or run_dir_for(paths_cfg, require(cfg, "score.run_subdir"))

    rows = _load_jsonl(args.input)
    summary_out = args.summary_out or args.out.with_name(args.out.stem + "_summary.json")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    totals: dict[str, dict[str, float]] = {
        mode: {
            "num_queries": 0,
            "hop_score_sum": 0.0,
            "similarity_bonus_sum": 0.0,
            "total_score_sum": 0.0,
        }
        for mode in modes
    }
    per_hop: dict[str, dict[str, Any]] = {}

    with args.out.open("w", encoding="utf-8") as outf:
        for row in rows:
            scored = _score_row(row, top_k, modes, points, windows, bonus_points)
            outf.write(json.dumps(scored, ensure_ascii=False) + "\n")

            q_hops = parse_hops_from_id(row.get("query_id"))
            hop_bucket_key = str(q_hops) if q_hops is not None else "unknown"
            if hop_bucket_key not in per_hop:
                per_hop[hop_bucket_key] = {
                    "num_queries": 0,
                    "per_mode": {
                        mode: {
                            "hop_score_sum": 0.0,
                            "similarity_bonus_sum": 0.0,
                            "total_score_sum": 0.0,
                        }
                        for mode in modes
                    },
                }
            per_hop[hop_bucket_key]["num_queries"] += 1

            for mode in modes:
                if f"{mode}_total_score" not in scored:
                    continue
                hop_v = float(scored.get(f"{mode}_hop_score", 0.0))
                bonus_v = float(scored.get(f"{mode}_similarity_bonus", 0.0))
                total_v = float(scored.get(f"{mode}_total_score", 0.0))

                totals[mode]["num_queries"] += 1
                totals[mode]["hop_score_sum"] += hop_v
                totals[mode]["similarity_bonus_sum"] += bonus_v
                totals[mode]["total_score_sum"] += total_v

                bucket = per_hop[hop_bucket_key]["per_mode"][mode]
                bucket["hop_score_sum"] += hop_v
                bucket["similarity_bonus_sum"] += bonus_v
                bucket["total_score_sum"] += total_v

    best_mode = max(modes, key=lambda m: totals[m]["total_score_sum"])
    metrics = {
        "script": Path(__file__).name,
        "created_utc": now_iso(),
        "seed": seed,
        "seeded": seeded,
        "input": str(args.input.resolve()),
        "output_jsonl": str(args.out.resolve()),
        "top_k": top_k,
        "hop_match_points": points,
        "hop_match_windows": windows,
        "similarity_bonus_points": bonus_points,
        "num_rows": len(rows),
        "per_mode": totals,
        "per_hop": per_hop,
        "best_mode_by_total_score_sum": best_mode,
    }
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    write_run_artifacts(
        run_dir,
        config_snapshot={
            "script": Path(__file__).name,
            "config_path": cfg.get("_config_path"),
            "input": str(args.input),
            "out": str(args.out),
            "top_k": top_k,
            "seed": seed,
        },
        metrics=metrics,
        note_title="Similarity result scoring",
        note_lines=[
            f"- Input: `{args.input}` ({len(rows)} rows)",
            f"- Scored JSONL: `{args.out}`",
            f"- Summary JSON: `{summary_out}`",
            f"- Point ladder: {points} over windows {windows} (+{bonus_points} similarity bonus)",
            f"- Best mode by total_score_sum: {best_mode} "
            f"({totals[best_mode]['total_score_sum']:.3f})",
        ],
        prefix="score_",
    )

    print(f"Wrote scored JSONL: {args.out}")
    print(f"Wrote summary JSON: {summary_out}")
    print(
        f"Best mode by total_score_sum: {best_mode} "
        f"({totals[best_mode]['total_score_sum']:.3f})"
    )


if __name__ == "__main__":
    main()
