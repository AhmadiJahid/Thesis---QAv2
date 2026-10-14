#!/usr/bin/env python3
"""Sampling variance of the decomposition report card under random stratified eval sets.

Issue #56 / ADR 0035. Every MuSiQue number so far is one draw: the ADR 0007 pinned 600
(200 per hop). Decoding is greedy and few-shot retrieval is per question, so a question's
decomposition - and hence its per-item score - does not depend on which sample it sits in.
Each arm is therefore run ONCE over a population (``musique_dev_full``, all 2,417
answerable dev questions) and this script scores K random samples of it offline:

- **draws**: K stratified samples, ``per_hop`` ids per hop without replacement within a
  draw, independent across draws (draw ``d`` uses ``random.Random(seed + d)``), and the SAME
  K draws for every arm (paired) - they depend only on the population ids;
- **scores**: each draw is aggregated with the evaluator's own ``_aggregate`` (imported from
  ``scripts/musique_decompositions_evaluator.py``, not reimplemented), overall and per hop,
  for the report-card terms listed in the evaluator's ``DECOMPOSITION_SUITE_TERMS``; the
  evaluator's macro-mean guard is applied to every overall draw;
- **pinned**: the pinned 600 is scored as one more subset and placed in each distribution as
  a mid-rank percentile;
- **ranks**: per draw and metric, the arms are ranked direction-aware (1 = best).

Outputs (``--out-dir``): ``resample_metrics.json``, ``resample_summary.csv``,
``resample_per_draw.csv``, ``resample_ranks.csv``, ``resample_config.json``,
``resample_notes.md`` and three figures (PNG + PDF).

CAVEAT, carried into the notes: draws overlap because the population is finite (a 200-of-405
4-hop draw shares about half its questions with any other), so the spread is a LOWER BOUND
on the spread for truly independent samples, and it measures question-sampling variation
only - with greedy decoding there is no decoding variation in it.

Usage::

    .venv/bin/python scripts/resample_eval_sets.py \\
        --arm "Qwen2.5-7B=runs/exp-029/qwen7b/eval/eval_per_item.json" \\
        --arm "Mistral-7B=runs/exp-029/mistral/eval/eval_per_item.json" \\
        --out-dir runs/exp-029/resample
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from eval_sets import hop_from_id  # noqa: E402
from run_artifacts import now_iso, write_run_artifacts  # noqa: E402
from run_config import load_config, load_paths, require, resolve_path  # noqa: E402
from seeding import set_global_seed  # noqa: E402

SCOPE_OVERALL = "overall"


# ----------------------------------------------------------------------------- evaluator


def load_evaluator(path: Path) -> Any:
    """Import the evaluator module so its aggregation is reused, not reimplemented."""
    spec = importlib.util.spec_from_file_location("musique_decompositions_evaluator", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import the evaluator from {path}")
    module = importlib.util.module_from_spec(spec)
    # Registered before exec: the evaluator defines dataclasses, which look their module up
    # in sys.modules while the class body is processed.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ----------------------------------------------------------------------------- sampling


def draw_seeds(base_seed: int, k: int) -> list[int]:
    """Draw ``d`` (0-based) is seeded with ``base_seed + d``."""
    return [int(base_seed) + d for d in range(int(k))]


def stratified_draws(
    ids_by_hop: dict[int, Sequence[str]],
    *,
    hops: Sequence[int],
    per_hop: int,
    seeds: Sequence[int],
) -> list[dict[int, list[str]]]:
    """K stratified draws: ``per_hop`` ids per hop, without replacement within a draw.

    Each draw has its own ``random.Random(seed)``; hops are drawn in the listed order from
    each hop's **sorted** population, so a draw is a pure function of (population ids, hops,
    per_hop, seed). Nothing here depends on an arm, which is what makes the draws paired.
    """
    pools: dict[int, list[str]] = {}
    for hop in hops:
        population = sorted(set(ids_by_hop.get(hop, ())))
        if len(population) != len(ids_by_hop.get(hop, ())):
            raise ValueError(f"hop {hop}: the population has duplicate ids")
        if len(population) < per_hop:
            raise ValueError(
                f"hop {hop}: population has {len(population)} ids, fewer than per_hop={per_hop}"
            )
        pools[hop] = population
    draws: list[dict[int, list[str]]] = []
    for seed in seeds:
        rng = random.Random(int(seed))
        draws.append({hop: rng.sample(pools[hop], int(per_hop)) for hop in hops})
    return draws


def draw_overlap(draws: list[dict[int, list[str]]], hops: Sequence[int]) -> dict[str, Any]:
    """Measured pairwise overlap between draws, per hop: |D_i ∩ D_j| / |D_i| over i < j."""
    out: dict[str, Any] = {}
    for hop in hops:
        sets = [set(d[hop]) for d in draws]
        fracs = [
            len(sets[i] & sets[j]) / len(sets[i])
            for i in range(len(sets))
            for j in range(i + 1, len(sets))
        ]
        out[str(hop)] = (
            {
                "mean_shared_fraction": float(np.mean(fracs)),
                "min_shared_fraction": float(np.min(fracs)),
                "max_shared_fraction": float(np.max(fracs)),
                "pairs": len(fracs),
            }
            if fracs
            else {"pairs": 0}
        )
    return out


# ----------------------------------------------------------------------------- inputs


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_arm(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """``({item_id: per-item row}, header)`` from an evaluator ``*_per_item.json``."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "items" not in payload:
        raise SystemExit(f"{path}: not an evaluator per-item file (no 'items')")
    rows: dict[str, dict[str, Any]] = {}
    for item in payload["items"]:
        iid = str(item["item_id"])
        if iid in rows:
            raise SystemExit(f"{path}: duplicate item_id {iid}")
        hop = hop_from_id(iid)
        if hop is None or hop != int(item["gold_hop_count"]):
            raise SystemExit(
                f"{path}: item {iid} has gold_hop_count {item['gold_hop_count']} but its id "
                f"says hop {hop}"
            )
        rows[iid] = item
    header = {k: v for k, v in payload.items() if k != "items"}
    return rows, header


def parse_arm_specs(specs: Sequence[str]) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"--arm expects LABEL=PATH, got {spec!r}")
        label, path = spec.split("=", 1)
        label = label.strip()
        if not label or label in {a for a, _ in out}:
            raise SystemExit(f"--arm label {label!r} is empty or repeated")
        out.append((label, Path(path).expanduser()))
    return out


def load_pinned_ids(paths_cfg: dict[str, Any], template_key: str, hops: Sequence[int]) -> set[str]:
    template = require(paths_cfg, f"datasets.{template_key}")
    ids: set[str] = set()
    for hop in hops:
        path = resolve_path(template.format(hop=hop), Path(paths_cfg["data_root_resolved"]))
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    ids.add(str(json.loads(line)["id"]))
    return ids


# ----------------------------------------------------------------------------- scoring


def score_subset(ev: Any, rows: dict[str, dict[str, Any]], ids: Sequence[str], metrics: Sequence[str],
                 hops: Sequence[int], guard: bool) -> dict[str, dict[str, float]]:
    """{scope: {metric: value}} for one subset, with the evaluator's ``_aggregate``."""
    items = [rows[i] for i in ids]
    overall = ev._aggregate(items)
    if guard:
        # The evaluator's own refusal: every suite term must be the macro mean of its column.
        ev._assert_suite_terms_are_macro_means(overall, items)
    out = {SCOPE_OVERALL: {m: float(overall[m]) for m in metrics}}
    for hop in hops:
        hop_items = [it for it in items if int(it["gold_hop_count"]) == hop]
        agg = ev._aggregate(hop_items)
        out[f"hop{hop}"] = {m: float(agg[m]) for m in metrics}
    return out


def pinned_percentile(values: Sequence[float], pinned: float, tol: float) -> float:
    """Mid-rank percentile of ``pinned`` among ``values``: 100*(below + 0.5*tied)/K."""
    arr = np.asarray(values, dtype=float)
    below = int(np.sum(arr < pinned - tol))
    tied = int(np.sum(np.abs(arr - pinned) <= tol))
    return 100.0 * (below + 0.5 * tied) / len(arr)


def summarize(values: Sequence[float], percentiles: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    out = {
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)) if len(arr) > 1 else float("nan"),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }
    for p in percentiles:
        out[f"p{p:g}"] = float(np.percentile(arr, p))
    return out


def competition_ranks(values: Sequence[float], lower_is_better: bool, tol: float) -> list[int]:
    """1 = best; tied arms share the best rank of their tie group (1, 1, 3, ...)."""
    signed = [(-v if not lower_is_better else v) for v in values]
    return [1 + sum(1 for o in signed if o < s - tol) for s in signed]


# ----------------------------------------------------------------------------- figures


def _arrow(direction: str) -> str:
    return "⇓ lower is better" if direction == "lower_is_better" else "⇑ higher is better"


def _spread_panel(ax: Any, arms: list[str], colors: dict[str, str], per_draw: dict[str, list[float]],
                  summary: dict[str, dict[str, float]], pinned: dict[str, float | None],
                  fig_cfg: dict[str, Any], pct: Sequence[float], rng: np.random.Generator) -> None:
    lo_key, hi_key = f"p{pct[0]:g}", f"p{pct[1]:g}"
    jw = float(fig_cfg["jitter_width"])
    dark = fig_cfg["summary_color"]
    for x, arm in enumerate(arms):
        vals = np.asarray(per_draw[arm], dtype=float)
        jitter = rng.uniform(-jw, jw, size=len(vals))
        ax.scatter(x + jitter, vals, s=fig_cfg["dot_size"], color=colors[arm],
                   alpha=fig_cfg["dot_alpha"], linewidths=0, zorder=2)
        s = summary[arm]
        ax.vlines(x, s[lo_key], s[hi_key], color=dark, linewidth=2.0, zorder=3)
        ax.hlines([s[lo_key], s[hi_key]], x - 0.12, x + 0.12, color=dark, linewidth=1.5, zorder=3)
        ax.hlines(s["mean"], x - 0.32, x + 0.32, color=dark, linewidth=3.5, zorder=4)
        if pinned.get(arm) is not None:
            ax.scatter([x + 0.42], [pinned[arm]], marker="D", s=fig_cfg["pinned_marker_size"],
                       facecolor="white", edgecolor="black", linewidth=1.5, zorder=5)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels(arms, rotation=30, ha="right")
    ax.set_xlim(-0.6, len(arms) - 0.4)
    ax.grid(axis="y", alpha=0.3)


def _legend_handles(arms: list[str], colors: dict[str, str], fig_cfg: dict[str, Any],
                    pct: Sequence[float]) -> list[Any]:
    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], marker="o", linestyle="", color=colors[a], alpha=0.8, label=a)
        for a in arms
    ]
    handles += [
        Line2D([], [], color=fig_cfg["summary_color"], linewidth=3.5,
               label=f"mean, whisker {pct[0]:g}-{pct[1]:g} pct"),
        Line2D([], [], marker="D", linestyle="", markerfacecolor="white",
               markeredgecolor="black", label="pinned 600 (ADR 0007)"),
    ]
    return handles


def _save(fig: Any, out_dir: Path, stem: str, fig_cfg: dict[str, Any]) -> list[str]:
    paths = []
    for fmt in fig_cfg["formats"]:
        p = out_dir / f"{stem}.{fmt}"
        fig.savefig(p, dpi=int(fig_cfg["dpi"]), bbox_inches="tight")
        paths.append(str(p))
    return paths


def make_figures(*, out_dir: Path, arms: list[str], metrics: list[str], directions: dict[str, str],
                 per_draw: dict[str, dict[str, dict[str, list[float]]]],
                 summary: dict[str, dict[str, dict[str, dict[str, float]]]],
                 pinned: dict[str, dict[str, dict[str, float | None]]],
                 rank_pct: dict[str, dict[str, list[float]]], hops: Sequence[int],
                 hop_metrics: list[str], per_hop: int, k: int, fig_cfg: dict[str, Any],
                 pct: Sequence[float], seed: int) -> dict[str, list[str]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": fig_cfg["font_size"]})
    palette = list(fig_cfg["palette"])
    if len(arms) > len(palette):
        raise SystemExit(f"{len(arms)} arms but only {len(palette)} palette colors in the config")
    colors = {a: palette[i] for i, a in enumerate(arms)}
    labels = fig_cfg["metric_labels"]
    pw, ph = float(fig_cfg["panel_width"]), float(fig_cfg["panel_height"])
    rng = np.random.default_rng(seed)
    written: dict[str, list[str]] = {}
    n_total = per_hop * len(hops)

    # Fig 1 - one panel per report-card metric, overall.
    ncols = 4
    nrows = math.ceil((len(metrics) + 1) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(pw * ncols, ph * nrows), squeeze=False)
    flat = list(axes.flat)
    for ax, m in zip(flat, metrics):
        _spread_panel(ax, arms, colors,
                      {a: per_draw[a][SCOPE_OVERALL][m] for a in arms},
                      {a: summary[a][SCOPE_OVERALL][m] for a in arms},
                      {a: pinned[a][SCOPE_OVERALL][m] for a in arms}, fig_cfg, pct, rng)
        ax.set_title(labels.get(m, m))
        ax.set_ylabel(_arrow(directions[m]))
    for ax in flat[len(metrics):]:
        ax.axis("off")
    flat[len(metrics)].legend(handles=_legend_handles(arms, colors, fig_cfg, pct), loc="center",
                              frameon=False)
    fig.suptitle(f"Report card over {k} random stratified samples "
                 f"({per_hop} per hop, {n_total} questions each)")
    fig.tight_layout()
    written["fig1_report_card_spread"] = _save(fig, out_dir, "fig1_report_card_spread", fig_cfg)
    plt.close(fig)

    # Fig 2 - rows = hop, columns = the headline metrics.
    fig, axes = plt.subplots(len(hops), len(hop_metrics),
                             figsize=(pw * len(hop_metrics) + 2.2, ph * len(hops)), squeeze=False)
    for r, hop in enumerate(hops):
        scope = f"hop{hop}"
        for c, m in enumerate(hop_metrics):
            ax = axes[r][c]
            _spread_panel(ax, arms, colors,
                          {a: per_draw[a][scope][m] for a in arms},
                          {a: summary[a][scope][m] for a in arms},
                          {a: pinned[a][scope][m] for a in arms}, fig_cfg, pct, rng)
            if r == 0:
                ax.set_title(labels.get(m, m))
            ax.set_ylabel(f"{hop}-hop  {_arrow(directions[m])}")
            if r != len(hops) - 1:
                ax.set_xticklabels([])
    fig.legend(handles=_legend_handles(arms, colors, fig_cfg, pct), loc="center left",
               bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.suptitle(f"Per hop: {k} random samples of {per_hop} questions at each hop depth")
    fig.tight_layout()
    written["fig2_per_hop_spread"] = _save(fig, out_dir, "fig2_per_hop_spread", fig_cfg)
    plt.close(fig)

    # Fig 3 - rank stability heatmaps, arms x rank.
    n = len(arms)
    fig, axes = plt.subplots(nrows, ncols, figsize=(pw * ncols, ph * nrows), squeeze=False)
    flat = list(axes.flat)
    image = None
    for ax, m in zip(flat, metrics):
        mat = np.asarray([rank_pct[m][a] for a in arms], dtype=float)
        image = ax.imshow(mat, cmap=fig_cfg["heatmap_cmap"], vmin=0, vmax=100, aspect="auto")
        for i in range(n):
            for j in range(n):
                v = mat[i, j]
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        color="white" if v >= 60 else "black", fontsize=fig_cfg["font_size"] - 1)
        ax.set_xticks(range(n))
        ax.set_xticklabels([str(r + 1) for r in range(n)])
        ax.set_yticks(range(n))
        ax.set_yticklabels(arms)
        ax.set_xlabel("rank (1 = best)")
        ax.set_title(f"{labels.get(m, m)} {'⇓' if directions[m] == 'lower_is_better' else '⇑'}")
    for ax in flat[len(metrics):]:
        ax.axis("off")
    if image is not None:
        cbar = fig.colorbar(image, ax=flat[len(metrics)], fraction=0.6, location="left")
        cbar.set_label(f"% of the {k} samples")
    fig.suptitle(f"Rank stability: % of {k} samples in which each arm takes each rank")
    fig.tight_layout()
    written["fig3_rank_stability"] = _save(fig, out_dir, "fig3_rank_stability", fig_cfg)
    plt.close(fig)
    return written


# ----------------------------------------------------------------------------- main


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="resample_eval_sets.json")
    p.add_argument("--arm", action="append", required=True,
                   help="LABEL=PATH to an evaluator eval_per_item.json; repeat per arm. The order "
                   "fixes each arm's color.")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--per-hop", type=int, default=None,
                   help="Override per_hop (smoke runs on a small population). Recorded.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    paths_cfg = load_paths(require(cfg, "paths_config"))
    seed = int(require(cfg, "seed"))
    seeded = set_global_seed(seed)
    k = int(require(cfg, "k_draws"))
    per_hop = int(args.per_hop) if args.per_hop is not None else int(require(cfg, "per_hop"))
    hops = [int(h) for h in require(cfg, "hops")]
    pct = [float(x) for x in require(cfg, "percentiles")]
    tol = float(require(cfg, "tie_tolerance"))
    fig_cfg = require(cfg, "figures")
    hop_metrics = list(require(cfg, "per_hop_figure_metrics"))

    ev = load_evaluator(_REPO_ROOT / require(cfg, "evaluator_script"))
    terms = list(ev.DECOMPOSITION_SUITE_TERMS)
    metrics = [t.aggregate_key for t in terms]
    directions = {t.aggregate_key: ev._direction(t.per_item_column) for t in terms}
    unknown = [m for m in hop_metrics if m not in metrics]
    if unknown:
        raise SystemExit(f"per_hop_figure_metrics not in the report card: {unknown}")

    # ---- arms and their common population
    arm_specs = parse_arm_specs(args.arm)
    arms = [label for label, _ in arm_specs]
    rows_by_arm: dict[str, dict[str, dict[str, Any]]] = {}
    arm_records: list[dict[str, Any]] = []
    for label, path in arm_specs:
        rows, header = load_arm(path)
        rows_by_arm[label] = rows
        arm_records.append({
            "label": label, "path": str(path.resolve()), "sha256": _sha256(path),
            "items": len(rows), "evaluation_set": header.get("evaluation_set"),
            "predictions_path": header.get("predictions_path"),
        })
    population = set(rows_by_arm[arms[0]])
    for label in arms[1:]:
        if set(rows_by_arm[label]) != population:
            a, b = population, set(rows_by_arm[label])
            raise SystemExit(
                f"arm {label!r} covers a different item set from {arms[0]!r} "
                f"({len(b - a)} extra, {len(a - b)} missing): paired draws need one population"
            )
    ids_by_hop: dict[int, list[str]] = {h: [] for h in hops}
    for iid in population:
        hop = hop_from_id(iid)
        if hop not in ids_by_hop:
            raise SystemExit(f"population item {iid} is at hop {hop}, not in {hops}")
        ids_by_hop[hop].append(iid)
    population_per_hop = {str(h): len(v) for h, v in ids_by_hop.items()}

    pinned_ids = load_pinned_ids(paths_cfg, require(cfg, "pinned_ids_template_key"), hops)
    pinned_in_population = pinned_ids <= population
    pinned_order = sorted(pinned_ids)

    # ---- draws (arm-independent) and scores
    seeds = draw_seeds(seed, k)
    draws = stratified_draws(ids_by_hop, hops=hops, per_hop=per_hop, seeds=seeds)
    draw_ids = [[i for h in hops for i in d[h]] for d in draws]
    scopes = [SCOPE_OVERALL] + [f"hop{h}" for h in hops]

    per_draw: dict[str, dict[str, dict[str, list[float]]]] = {
        a: {s: {m: [] for m in metrics} for s in scopes} for a in arms
    }
    pinned: dict[str, dict[str, dict[str, float | None]]] = {}
    for a in arms:
        for ids in draw_ids:
            scored = score_subset(ev, rows_by_arm[a], ids, metrics, hops, guard=True)
            for s in scopes:
                for m in metrics:
                    per_draw[a][s][m].append(scored[s][m])
        if pinned_in_population:
            ps = score_subset(ev, rows_by_arm[a], pinned_order, metrics, hops, guard=True)
            pinned[a] = {s: {m: ps[s][m] for m in metrics} for s in scopes}
        else:
            pinned[a] = {s: {m: None for m in metrics} for s in scopes}

    summary: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for a in arms:
        summary[a] = {}
        for s in scopes:
            summary[a][s] = {}
            for m in metrics:
                row = summarize(per_draw[a][s][m], pct)
                pv = pinned[a][s][m]
                row["pinned_value"] = pv
                row["pinned_percentile"] = (
                    pinned_percentile(per_draw[a][s][m], pv, tol) if pv is not None else None
                )
                summary[a][s][m] = row

    # ---- rank stability (overall)
    n = len(arms)
    rank_counts: dict[str, dict[str, list[int]]] = {m: {a: [0] * n for a in arms} for m in metrics}
    for m in metrics:
        lower = directions[m] == "lower_is_better"
        for d in range(k):
            ranks = competition_ranks([per_draw[a][SCOPE_OVERALL][m][d] for a in arms], lower, tol)
            for a, r in zip(arms, ranks):
                rank_counts[m][a][r - 1] += 1
    rank_pct = {m: {a: [100.0 * c / k for c in rank_counts[m][a]] for a in arms} for m in metrics}
    pinned_ranks = (
        {m: dict(zip(arms, competition_ranks(
            [pinned[a][SCOPE_OVERALL][m] for a in arms], directions[m] == "lower_is_better", tol)))
         for m in metrics}
        if pinned_in_population else None
    )

    # ---- outputs
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    lo_key, hi_key = f"p{pct[0]:g}", f"p{pct[1]:g}"
    with (out_dir / "resample_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arm", "scope", "metric", "direction", "n_draws", "per_hop", "mean", "sd",
                    "min", "max", lo_key, hi_key, "pinned_value", "pinned_percentile"])
        for a in arms:
            for s in scopes:
                for m in metrics:
                    r = summary[a][s][m]
                    w.writerow([a, s, m, directions[m], k, per_hop, r["mean"], r["sd"], r["min"],
                                r["max"], r[lo_key], r[hi_key], r["pinned_value"],
                                r["pinned_percentile"]])
    with (out_dir / "resample_per_draw.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["draw", "seed", "arm", "scope", "metric", "value"])
        for d, sd in enumerate(seeds):
            for a in arms:
                for s in scopes:
                    for m in metrics:
                        w.writerow([d, sd, a, s, m, per_draw[a][s][m][d]])
    with (out_dir / "resample_ranks.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "direction", "arm"] + [f"rank{r + 1}_pct" for r in range(n)])
        for m in metrics:
            for a in arms:
                w.writerow([m, directions[m], a] + rank_pct[m][a])

    figures = make_figures(
        out_dir=out_dir, arms=arms, metrics=metrics, directions=directions, per_draw=per_draw,
        summary=summary, pinned=pinned, rank_pct=rank_pct, hops=hops, hop_metrics=hop_metrics,
        per_hop=per_hop, k=k, fig_cfg=fig_cfg, pct=pct, seed=seed,
    )
    overlap = draw_overlap(draws, hops)
    caveat = (
        "Draws overlap because the population is finite: each hop's draws come from the same "
        f"{population_per_hop} questions, so two draws share questions (measured mean shared "
        "fraction per hop: "
        + ", ".join(f"{h}-hop {overlap[str(h)].get('mean_shared_fraction', float('nan')):.2f}"
                    for h in hops)
        + "). The spread is therefore a LOWER BOUND on the spread for truly independent "
        "samples of this size, and it measures question-sampling variation only: decoding is "
        "greedy, so there is no decoding variation in it."
    )
    metrics_json = {
        "script": Path(__file__).name,
        "created_utc": now_iso(),
        "seed": seed,
        "seeded": seeded,
        "k_draws": k,
        "per_hop": per_hop,
        "per_hop_overridden": args.per_hop is not None,
        "hops": hops,
        "draw_seeds": seeds,
        "draw_seed_rule": "draw d (0-based) uses random.Random(seed + d)",
        "percentile_method": "numpy.percentile, linear interpolation",
        "pinned_percentile_rule": "100 * (#draws below + 0.5 * #draws tied) / K",
        "rank_rule": "direction-aware, 1 = best, ties share the best rank (competition ranking)",
        "metrics": metrics,
        "directions": directions,
        "aggregation": f"{require(cfg, 'evaluator_script')}::_aggregate (imported), with "
        "_assert_suite_terms_are_macro_means applied to every overall subset",
        "arms": arm_records,
        "population_total": len(population),
        "population_per_hop": population_per_hop,
        "pinned_ids": len(pinned_ids),
        "pinned_in_population": pinned_in_population,
        "draw_overlap": overlap,
        "summary": summary,
        "rank_stability_pct": rank_pct,
        "pinned_ranks": pinned_ranks,
        "figures": figures,
        "caveat": caveat,
    }
    note_lines = [
        f"- Arms: {', '.join(f'`{a}`' for a in arms)} (population {len(population)} items, per "
        f"hop {population_per_hop}; identical item set in every arm)",
        f"- Draws: K={k}, {per_hop} per hop ({per_hop * len(hops)} per draw), stratified, without "
        f"replacement within a draw, seeds {seeds[0]}..{seeds[-1]}, the same draws for every arm"
        + (" — **per_hop overridden on the command line (smoke run)**" if args.per_hop is not None else ""),
        f"- Pinned 600 in the population: {pinned_in_population}",
        f"- **Caveat.** {caveat}",
        "",
        f"| arm | metric | dir | mean | SD | min | max | {lo_key} | {hi_key} | pinned | pinned pct |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for a in arms:
        for m in metrics:
            r = summary[a][SCOPE_OVERALL][m]
            pv = "n/a" if r["pinned_value"] is None else f"{r['pinned_value']:.4f}"
            pp = "n/a" if r["pinned_percentile"] is None else f"{r['pinned_percentile']:.0f}"
            note_lines.append(
                f"| {a} | {m} | {'⇓' if directions[m] == 'lower_is_better' else '⇑'} | "
                f"{r['mean']:.4f} | {r['sd']:.4f} | {r['min']:.4f} | {r['max']:.4f} | "
                f"{r[lo_key]:.4f} | {r[hi_key]:.4f} | {pv} | {pp} |"
            )
    note_lines += ["", "Figures: " + ", ".join(f"`{p}`" for ps in figures.values() for p in ps)]
    write_run_artifacts(
        out_dir,
        config_snapshot={
            "script": Path(__file__).name,
            "config_path": cfg.get("_config_path"),
            "config": {key: v for key, v in cfg.items() if not key.startswith("_")},
            "per_hop_effective": per_hop,
            "arms": [{"label": a, "path": str(p)} for a, p in arm_specs],
            "out_dir": str(out_dir),
        },
        metrics=metrics_json,
        note_title="Sampling variance of the decomposition report card (issue #56)",
        note_lines=note_lines,
        prefix="resample_",
    )


if __name__ == "__main__":
    main()
