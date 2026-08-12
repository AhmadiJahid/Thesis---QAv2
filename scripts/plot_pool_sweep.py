#!/usr/bin/env python3
"""Plot MuSiQue pool-size sweep results.

Reads the sweep summary CSV produced by ``scripts/pool_sweep_orchestrator.py`` and writes:

- ``metric_vs_pool_size__<metric>.png``: one panel per balance, one line per
  (variant, mode); mean across trials with std error bars and a faint per-trial scatter.
- ``ce_vs_biencoder_delta__<metric>.png``: cross-encoder minus bi-encoder-only delta.
- ``balanced_vs_imbalanced_delta__<metric>.png``: balanced minus imbalanced delta.
- ``per_hop__<metric>.png``: per-gold-hop mean metric vs pool size, read from each run's
  ``eval_metrics.json`` (``per_gold_hop_metrics`` is not in the flat CSV).
- ``metrics_by_cell_mean_std.csv`` plus the standard run trail.

Matplotlib only. Ported from v1; the metric list, axes and dpi now come from
``configs/plot_pool_sweep.json``.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from run_artifacts import now_iso, write_run_artifacts  # noqa: E402
from run_config import load_config, load_paths, require, runs_path  # noqa: E402


def _load_runs_csv(path: Path, metrics: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        raise SystemExit(f"[plot_pool_sweep] summary csv not found: {path}")
    with path.open(encoding="utf-8", newline="") as f:
        for rec in csv.DictReader(f):
            try:
                rec["size"] = int(rec["size"])
            except (KeyError, ValueError):
                continue
            try:
                rec["trial"] = int(rec.get("trial", 0))
            except ValueError:
                rec["trial"] = 0
            for key in metrics:
                v = rec.get(key, "")
                try:
                    rec[key] = float(v) if v not in (None, "") else None
                except (TypeError, ValueError):
                    rec[key] = None
            rows.append(rec)
    return rows


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return (float("nan"), 0.0)
    m = sum(values) / len(values)
    if len(values) == 1:
        return (m, 0.0)
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return (m, var ** 0.5)


def _aggregate(
    rows: list[dict[str, Any]],
    metric: str,
) -> dict[tuple[str, str, str, int], tuple[float, float, list[float]]]:
    """{(balance, variant, mode, size): (mean, std, per-trial values)}."""
    bucket: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    for r in rows:
        v = r.get(metric)
        if v is None:
            continue
        bucket[(r["balance"], r["variant"], r["mode"], int(r["size"]))].append(float(v))
    return {key: (*_mean_std(vals), vals) for key, vals in bucket.items()}


def _plot_metric_vs_size(
    rows: list[dict[str, Any]], metric: str, out_dir: Path, axes_cfg: dict[str, list[str]], dpi: int
) -> Path:
    balances, variants, modes = axes_cfg["balances"], axes_cfg["variants"], axes_cfg["modes"]
    agg = _aggregate(rows, metric)
    fig, axes = plt.subplots(1, len(balances), figsize=(12, 5), sharey=True)
    if len(balances) == 1:
        axes = [axes]

    for ax, balance in zip(axes, balances):
        for variant in variants:
            for mode in modes:
                points = sorted(
                    [
                        (size, agg[(balance, variant, mode, size)])
                        for (b, v, m, size) in agg
                        if b == balance and v == variant and m == mode
                    ],
                    key=lambda x: x[0],
                )
                if not points:
                    continue
                ax.errorbar(
                    [p[0] for p in points],
                    [p[1][0] for p in points],
                    yerr=[p[1][1] for p in points],
                    marker="o",
                    capsize=3,
                    linewidth=1.2,
                    label=f"{variant}/{mode}",
                )
                for size, (_, _, vals) in points:
                    ax.scatter([size] * len(vals), vals, s=10, alpha=0.25)

        ax.set_title(f"{balance}")
        ax.set_xlabel("pool size")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(metric)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02), fontsize=9)
    fig.suptitle(f"{metric} vs pool size (mean +/- std across trials)", y=1.08)
    fig.tight_layout()
    out = out_dir / f"metric_vs_pool_size__{metric}.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out}")
    return out


def _plot_ce_vs_biencoder_delta(
    rows: list[dict[str, Any]], metric: str, out_dir: Path, axes_cfg: dict[str, list[str]], dpi: int
) -> Path:
    balances, modes = axes_cfg["balances"], axes_cfg["modes"]
    agg = _aggregate(rows, metric)
    fig, axes = plt.subplots(1, len(balances), figsize=(12, 5), sharey=True)
    if len(balances) == 1:
        axes = [axes]
    for ax, balance in zip(axes, balances):
        for mode in modes:
            sizes = sorted({s for (b, v, m, s) in agg if b == balance and m == mode})
            xs: list[int] = []
            deltas: list[float] = []
            for size in sizes:
                ce = agg.get((balance, "biencoder_plus_ce", mode, size))
                bi = agg.get((balance, "biencoder_only", mode, size))
                if ce is None or bi is None:
                    continue
                xs.append(size)
                deltas.append(ce[0] - bi[0])
            if xs:
                ax.plot(xs, deltas, marker="o", label=mode)
        ax.axhline(0.0, color="k", linewidth=0.8, alpha=0.5)
        ax.set_title(f"{balance}")
        ax.set_xlabel("pool size")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(f"{metric} delta (CE - bi-encoder)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02), fontsize=9)
    fig.suptitle(f"CE rerank gain over bi-encoder only ({metric})", y=1.08)
    fig.tight_layout()
    out = out_dir / f"ce_vs_biencoder_delta__{metric}.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out}")
    return out


def _plot_balanced_vs_imbalanced_delta(
    rows: list[dict[str, Any]], metric: str, out_dir: Path, axes_cfg: dict[str, list[str]], dpi: int
) -> Path:
    balances, variants, modes = axes_cfg["balances"], axes_cfg["variants"], axes_cfg["modes"]
    agg = _aggregate(rows, metric)
    fig, axes = plt.subplots(1, len(variants), figsize=(12, 5), sharey=True)
    if len(variants) == 1:
        axes = [axes]
    for ax, variant in zip(axes, variants):
        for mode in modes:
            sizes = sorted({s for (b, v, m, s) in agg if v == variant and m == mode and b in balances})
            xs: list[int] = []
            deltas: list[float] = []
            for size in sizes:
                b_v = agg.get(("balanced", variant, mode, size))
                i_v = agg.get(("imbalanced", variant, mode, size))
                if b_v is None or i_v is None:
                    continue
                xs.append(size)
                deltas.append(b_v[0] - i_v[0])
            if xs:
                ax.plot(xs, deltas, marker="o", label=mode)
        ax.axhline(0.0, color="k", linewidth=0.8, alpha=0.5)
        ax.set_title(variant)
        ax.set_xlabel("pool size")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(f"{metric} delta (balanced - imbalanced)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02), fontsize=9)
    fig.suptitle(f"Balanced vs imbalanced pool gain ({metric})", y=1.08)
    fig.tight_layout()
    out = out_dir / f"balanced_vs_imbalanced_delta__{metric}.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out}")
    return out


def _per_hop_metrics(
    rows: list[dict[str, Any]], metric_key: str
) -> dict[tuple[str, str, str, int, int], list[float]]:
    """Walk each run's eval_metrics.json for per-hop numbers."""
    out: dict[tuple[str, str, str, int, int], list[float]] = defaultdict(list)
    for r in rows:
        path = r.get("eval_metrics_path")
        if not path:
            continue
        p = Path(path)
        if not p.exists():
            continue
        try:
            metrics = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        for hop_str, bucket_metrics in (metrics.get("per_gold_hop_metrics") or {}).items():
            try:
                hop = int(hop_str)
            except (TypeError, ValueError):
                continue
            v = bucket_metrics.get(metric_key)
            if not isinstance(v, (int, float)):
                continue
            out[(r["balance"], r["variant"], r["mode"], int(r["size"]), hop)].append(float(v))
    return out


def _plot_per_hop(
    rows: list[dict[str, Any]], metric: str, out_dir: Path, axes_cfg: dict[str, list[str]], dpi: int
) -> Path | None:
    balances, variants, modes = axes_cfg["balances"], axes_cfg["variants"], axes_cfg["modes"]
    per_hop = _per_hop_metrics(rows, metric)
    if not per_hop:
        print(f"[plot] no per-hop data for {metric}, skipping per-hop panel.")
        return None

    hops = sorted({k[4] for k in per_hop})
    fig, axes = plt.subplots(
        len(balances), len(hops),
        figsize=(4 * len(hops), 4 * len(balances)),
        sharey=True, squeeze=False,
    )

    for row_i, balance in enumerate(balances):
        for col_i, hop in enumerate(hops):
            ax = axes[row_i][col_i]
            for variant in variants:
                for mode in modes:
                    sizes = sorted({
                        s for (b, v, m, s, h) in per_hop
                        if b == balance and v == variant and m == mode and h == hop
                    })
                    xs: list[int] = []
                    ys: list[float] = []
                    errs: list[float] = []
                    for size in sizes:
                        vals = per_hop.get((balance, variant, mode, size, hop), [])
                        if not vals:
                            continue
                        m_v, s_v = _mean_std(vals)
                        xs.append(size)
                        ys.append(m_v)
                        errs.append(s_v)
                    if xs:
                        ax.errorbar(
                            xs, ys, yerr=errs, marker="o", capsize=3, linewidth=1.2,
                            label=f"{variant}/{mode}",
                        )
            ax.set_title(f"{balance} | {hop}-hop")
            ax.set_xlabel("pool size")
            ax.grid(True, alpha=0.3)
        axes[row_i][0].set_ylabel(metric)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02), fontsize=9)
    fig.suptitle(f"Per-gold-hop {metric} vs pool size", y=1.04)
    fig.tight_layout()
    out = out_dir / f"per_hop__{metric}.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out}")
    return out


def _write_aggregated_csv(rows: list[dict[str, Any]], metrics: list[str], out_dir: Path) -> Path | None:
    """One row per (balance, variant, mode, size, metric)."""
    rows_out: list[dict[str, Any]] = []
    for metric in metrics:
        for (balance, variant, mode, size), (mean, std, vals) in _aggregate(rows, metric).items():
            rows_out.append(
                {
                    "balance": balance,
                    "variant": variant,
                    "mode": mode,
                    "size": size,
                    "metric": metric,
                    "mean": mean,
                    "std": std,
                    "num_trials": len(vals),
                }
            )
    if not rows_out:
        return None
    out = out_dir / "metrics_by_cell_mean_std.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["balance", "variant", "mode", "size", "metric", "mean", "std", "num_trials"]
        )
        w.writeheader()
        w.writerows(rows_out)
    print(f"[plot] wrote {out}")
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="plot_pool_sweep.json")
    p.add_argument("--summary-csv", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--metrics", default=None, help="Comma-separated metric override.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    paths_cfg = load_paths(require(cfg, "paths_config"))

    sweep_root = runs_path(paths_cfg, require(cfg, "runs_subdir"))
    summary_csv = args.summary_csv or sweep_root / require(cfg, "summary_csv_relpath")
    out_dir = args.out_dir or sweep_root / require(cfg, "out_relpath")
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = (
        [m.strip() for m in args.metrics.split(",") if m.strip()]
        if args.metrics
        else list(require(cfg, "metrics"))
    )
    axes_cfg = {
        "balances": list(require(cfg, "balances")),
        "variants": list(require(cfg, "variants")),
        "modes": list(require(cfg, "modes")),
    }
    dpi = int(require(cfg, "dpi"))

    rows = _load_runs_csv(summary_csv, metrics)
    print(f"[plot] loaded {len(rows)} rows from {summary_csv}")
    if not rows:
        print("[plot] no rows, nothing to plot.")
        return

    figures: list[str] = []
    agg_csv = _write_aggregated_csv(rows, metrics, out_dir)

    for metric in metrics:
        figures.append(str(_plot_metric_vs_size(rows, metric, out_dir, axes_cfg, dpi)))
        figures.append(str(_plot_ce_vs_biencoder_delta(rows, metric, out_dir, axes_cfg, dpi)))
        figures.append(str(_plot_balanced_vs_imbalanced_delta(rows, metric, out_dir, axes_cfg, dpi)))
        per_hop_fig = _plot_per_hop(rows, metric, out_dir, axes_cfg, dpi)
        if per_hop_fig:
            figures.append(str(per_hop_fig))

    write_run_artifacts(
        out_dir,
        config_snapshot={
            "script": Path(__file__).name,
            "config_path": cfg.get("_config_path"),
            "summary_csv": str(summary_csv),
            "out_dir": str(out_dir),
            "metrics": metrics,
            **axes_cfg,
            "dpi": dpi,
        },
        metrics={
            "script": Path(__file__).name,
            "created_utc": now_iso(),
            "summary_csv": str(summary_csv),
            "rows_loaded": len(rows),
            "metrics_plotted": metrics,
            "figures": figures,
            "aggregated_csv": str(agg_csv) if agg_csv else None,
        },
        note_title="Pool sweep plots",
        note_lines=[
            f"- Summary CSV: `{summary_csv}` ({len(rows)} rows)",
            f"- Metrics plotted: {', '.join(metrics)}",
            f"- Figures written: {len(figures)}",
        ],
        prefix="plots_",
    )


if __name__ == "__main__":
    main()
