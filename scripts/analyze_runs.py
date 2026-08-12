#!/usr/bin/env python3
"""
Visualise and summarise completed router runs.

Reads every run directory under ``<runs root>/<runs_subdir>/<component>/``, plots
overall accuracy, per-hop accuracy, per-run confusion matrices and an error-pattern
summary, and writes an HTML report plus the standard run trail.

Ported from v1 ``scripts/analyze_runs.py``. Adapted for v2: the runs root, component,
output directory, hop list and all plot styling come from ``configs/analyze_runs.json``
/ ``configs/paths.json``.

    python scripts/analyze_runs.py --component average_zero_shot
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from run_artifacts import now_iso, write_run_artifacts  # noqa: E402
from run_config import load_config, load_paths, require, runs_path  # noqa: E402

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError as e:  # pragma: no cover - depends on the environment
    print("Error: missing required package. Install with: pip install matplotlib seaborn")
    print(f"Missing: {e.name}")
    sys.exit(1)


def apply_style(style: dict[str, Any]) -> None:
    for candidate in require(style, "matplotlib_styles"):
        try:
            matplotlib.style.use(candidate)
            break
        except OSError:
            continue
    sns.set_palette(require(style, "palette"))
    plt.rcParams["figure.figsize"] = tuple(require(style, "figure_size"))
    plt.rcParams["font.size"] = require(style, "font_size")
    plt.rcParams["axes.labelsize"] = require(style, "axes_labelsize")
    plt.rcParams["axes.titlesize"] = require(style, "axes_titlesize")
    plt.rcParams["xtick.labelsize"] = require(style, "xtick_labelsize")
    plt.rcParams["ytick.labelsize"] = require(style, "ytick_labelsize")


def get_model_short_name(model_id: str | None, suffixes: list[str]) -> str:
    """'Qwen/Qwen2.5-1.5B-Instruct' -> 'Qwen2.5-1.5B'."""
    if not model_id or model_id == "N/A":
        return "N/A"
    parts = model_id.split("/")
    model_name = parts[-1] if len(parts) > 1 else model_id
    for suffix in suffixes:
        if model_name.endswith(suffix):
            model_name = model_name[: -len(suffix)]
    return model_name


def load_run_data(
    component_dir: Path,
    hops: list[int],
    suffixes: list[str],
    *,
    skip_archived: bool,
    archived_marker: str,
) -> list[dict[str, Any]]:
    """Load metrics + config + detailed results for every run under ``component_dir``."""
    if not component_dir.exists():
        print(f"Warning: {component_dir} does not exist")
        return []

    runs_data: list[dict[str, Any]] = []

    for run_dir in component_dir.iterdir():
        if skip_archived and archived_marker in run_dir.name.lower():
            print(f"Skipping archived folder: {run_dir.name}")
            continue
        if not run_dir.is_dir():
            continue

        metrics_file = run_dir / "metrics.json"
        aggregated_file = run_dir / "metrics_aggregated.json"
        config_file = run_dir / "config.json"

        if metrics_file.exists():
            metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
            if "overall_accuracy" not in metrics and "overall_accuracy_mean" in metrics:
                # A multi-run metrics.json from the consolidated runner: normalise the
                # mean to the single-run key names the plots use.
                metrics = _normalise_aggregated(metrics, hops)
        elif aggregated_file.exists():
            metrics = _normalise_aggregated(
                json.loads(aggregated_file.read_text(encoding="utf-8")), hops
            )
        else:
            print(f"Warning: no metrics.json / metrics_aggregated.json, skipping {run_dir.name}")
            continue

        try:
            config = json.loads(config_file.read_text(encoding="utf-8")) if config_file.exists() else {}

            detailed_results: list[dict[str, Any]] = []
            for fname in ("detailed_results.json", "detailed_results_run_0.json"):
                fpath = run_dir / fname
                if fpath.exists():
                    detailed_results = json.loads(fpath.read_text(encoding="utf-8"))
                    break

            model_id = config.get("model_id", "N/A")
            runs_data.append(
                {
                    "run_id": run_dir.name,
                    "metrics": metrics,
                    "config": config,
                    "detailed_results": detailed_results,
                    "run_dir": run_dir,
                    "model_id": model_id,
                    "model_short": get_model_short_name(model_id, suffixes),
                }
            )
        except Exception as e:
            print(f"Error loading {run_dir.name}: {e}")
            continue

    runs_data.sort(key=lambda x: x["run_id"])
    return runs_data


def _normalise_aggregated(raw: dict[str, Any], hops: list[int]) -> dict[str, Any]:
    """Map *_mean keys onto the single-run key names, so plotting stays uniform."""
    metrics: dict[str, Any] = {
        "overall_accuracy": raw.get("overall_accuracy_mean", raw.get("overall_accuracy", 0)),
        "overall_accuracy_std": raw.get("overall_accuracy_std", 0),
        "total_questions": raw.get("total_questions"),
        "correct_predictions": raw.get("correct_predictions"),
        "num_runs": raw.get("num_runs"),
    }
    for hop in hops:
        metrics[f"hop_{hop}_accuracy"] = raw.get(
            f"hop_{hop}_accuracy_mean", raw.get(f"hop_{hop}_accuracy", 0)
        )
        metrics[f"hop_{hop}_accuracy_std"] = raw.get(f"hop_{hop}_accuracy_std", 0)
    return {k: v for k, v in metrics.items() if v is not None}


def plot_overall_accuracy(runs_data: list[dict], output_dir: Path, style: dict) -> str:
    run_ids = [r["run_id"] for r in runs_data]
    accuracies = [r["metrics"]["overall_accuracy"] * 100 for r in runs_data]
    labels = [f"{r['run_id']}\n({r['model_short']})" for r in runs_data]

    fig, ax = plt.subplots(figsize=tuple(require(style, "overall_figure_size")))
    bars = ax.bar(range(len(run_ids)), accuracies, color=sns.color_palette(require(style, "palette"), len(run_ids)))
    for bar, acc in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{acc:.2f}%",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_xlabel("Run ID (Model)")
    ax.set_ylabel("Overall Accuracy (%)")
    ax.set_title("Overall Accuracy Comparison Across Runs", fontweight="bold")
    ax.set_xticks(range(len(run_ids)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "overall_accuracy.png", dpi=require(style, "dpi"), bbox_inches="tight")
    plt.close()
    return "overall_accuracy.png"


def plot_per_hop_accuracy(runs_data: list[dict], output_dir: Path, style: dict, hops: list[int]) -> str:
    labels = [f"{r['run_id']}\n({r['model_short']})" for r in runs_data]
    x = range(len(runs_data))
    width = require(style, "bar_width")

    fig, ax = plt.subplots(figsize=tuple(require(style, "per_hop_figure_size")))
    offset_start = -(len(hops) - 1) / 2
    for i, hop in enumerate(hops):
        values = [r["metrics"].get(f"hop_{hop}_accuracy", 0) * 100 for r in runs_data]
        offsets = [xi + (offset_start + i) * width for xi in x]
        bars = ax.bar(offsets, values, width, label=f"{hop}-hop", alpha=0.8)
        for bar in bars:
            if bar.get_height() > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height(),
                    f"{bar.get_height():.1f}%",
                    ha="center", va="bottom", fontsize=8,
                )

    ax.set_xlabel("Run ID (Model)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Per-Hop Accuracy Comparison Across Runs", fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "per_hop_accuracy.png", dpi=require(style, "dpi"), bbox_inches="tight")
    plt.close()
    return "per_hop_accuracy.png"


def plot_confusion_matrices(
    runs_data: list[dict], output_dir: Path, style: dict, hops: list[int]
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    for run in runs_data:
        detailed_results = run.get("detailed_results") or []
        if not detailed_results:
            continue

        cm: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        for result in detailed_results:
            expected = result.get("expected") or result.get("expected_hop", 0)
            predicted = result.get("predicted") or result.get("predicted_hop", 0)
            cm[expected][predicted] += 1

        matrix = [[cm[i][j] for j in hops] for i in hops]
        max_val = max((max(row) for row in matrix if row), default=1) or 1

        fig, ax = plt.subplots(figsize=tuple(require(style, "confusion_figure_size")))
        sns.heatmap(
            matrix, annot=True, fmt="d", cmap="Reds", ax=ax,
            xticklabels=[f"Predicted {h}" for h in hops],
            yticklabels=[f"Actual {h}" for h in hops],
            cbar_kws={"label": "Count"}, vmin=0, vmax=max_val,
        )
        ax.set_title(f"Confusion Matrix - {run['run_id']}\n({run['model_short']})", fontweight="bold")
        ax.set_xlabel("Predicted Hop Count")
        ax.set_ylabel("Actual Hop Count")

        plt.tight_layout()
        filename = f"confusion_matrix_{run['run_id']}.png"
        plt.savefig(output_dir / filename, dpi=require(style, "dpi"), bbox_inches="tight")
        plt.close()

        out.append(
            {
                "run_id": run["run_id"],
                "model_short": run["model_short"],
                "filename": filename,
                "matrix": matrix,
            }
        )

    return out


def plot_error_patterns(runs_data: list[dict], output_dir: Path, style: dict) -> str | None:
    error_patterns: dict[str, int] = defaultdict(int)

    for run in runs_data:
        for result in run.get("detailed_results") or []:
            if not result.get("correct", True):
                expected = result.get("expected") or result.get("expected_hop", 0)
                predicted = result.get("predicted") or result.get("predicted_hop", 0)
                error_patterns[f"{expected}-hop -> {predicted}-hop"] += 1

    if not error_patterns:
        return None

    sorted_patterns = sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)
    patterns, counts = zip(*sorted_patterns)

    fig, ax = plt.subplots(figsize=tuple(require(style, "error_figure_size")))
    bars = ax.barh(range(len(patterns)), counts, color=sns.color_palette(require(style, "palette"), len(patterns)))
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2.0, f" {count}", ha="left", va="center")

    ax.set_yticks(range(len(patterns)))
    ax.set_yticklabels(patterns)
    ax.set_xlabel("Error Count")
    ax.set_title("Error Pattern Summary (All Runs)", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "error_patterns.png", dpi=require(style, "dpi"), bbox_inches="tight")
    plt.close()
    return "error_patterns.png"


def create_summary_table(runs_data: list[dict], hops: list[int]) -> list[dict[str, Any]]:
    table: list[dict[str, Any]] = []
    for run in runs_data:
        metrics = run["metrics"]
        row = {
            "Run ID": run["run_id"],
            "Overall Accuracy": f"{metrics.get('overall_accuracy', 0) * 100:.2f}%",
            "Model": run.get("model_id", "N/A"),
            "Total Questions": metrics.get("total_questions", 0),
        }
        for hop in hops:
            row[f"{hop}-hop Accuracy"] = f"{metrics.get(f'hop_{hop}_accuracy', 0) * 100:.2f}%"
        table.append(row)
    return table


def generate_html_report(
    runs_data: list[dict],
    plots: dict[str, Any],
    output_dir: Path,
    hops: list[int],
    component: str,
) -> Path:
    summary_table = create_summary_table(runs_data, hops)
    columns = ["Run ID", "Overall Accuracy"] + [f"{h}-hop Accuracy" for h in hops] + [
        "Model", "Total Questions"
    ]

    table_html = ["<table border='1' style='border-collapse: collapse; width: 100%; margin: 20px 0;'>"]
    table_html.append("<tr>" + "".join(f"<th>{c}</th>" for c in columns) + "</tr>")
    for row in summary_table:
        table_html.append("<tr>" + "".join(f"<td>{row[c]}</td>" for c in columns) + "</tr>")
    table_html.append("</table>")

    confusion_html = "".join(
        f"""
        <div style='margin: 30px 0;'>
            <h3>Confusion Matrix - {cm['run_id']} ({cm['model_short']})</h3>
            <img src='{cm['filename']}' alt='Confusion Matrix {cm['run_id']}' style='max-width: 100%;'>
        </div>
        """
        for cm in plots.get("confusion_matrices", [])
    )

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Router Component Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; line-height: 1.6; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        img {{ max-width: 100%; height: auto; margin: 20px 0; border: 1px solid #ddd; border-radius: 4px; }}
        table {{ font-size: 14px; }}
        th {{ background-color: #3498db; color: white; padding: 10px; text-align: left; }}
        td {{ padding: 8px; border: 1px solid #ddd; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .section {{ margin: 40px 0; padding: 20px; background-color: #f9f9f9; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>Router Component Performance Analysis Report</h1>
    <p><strong>Component:</strong> {component}</p>
    <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p><strong>Total Runs Analyzed:</strong> {len(runs_data)}</p>

    <div class="section">
        <h2>Summary Statistics</h2>
        {''.join(table_html)}
    </div>

    <div class="section">
        <h2>Overall Accuracy Comparison</h2>
        <img src='{plots.get('overall_accuracy', '')}' alt='Overall Accuracy Comparison'>
    </div>

    <div class="section">
        <h2>Per-Hop Accuracy</h2>
        <img src='{plots.get('per_hop_accuracy', '')}' alt='Per-Hop Accuracy Comparison'>
    </div>

    <div class="section">
        <h2>Error Pattern Summary</h2>
        <img src='{plots.get('error_patterns', '')}' alt='Error Patterns'>
    </div>

    <div class="section">
        <h2>Confusion Matrices</h2>
        {confusion_html}
    </div>

    <div class="section">
        <h2>Notes</h2>
        <ul>
            <li>Archived run folders are excluded from analysis</li>
            <li>All accuracy values are percentages</li>
            <li>Confusion matrices show actual vs predicted hop counts</li>
        </ul>
    </div>
</body>
</html>
"""
    out = output_dir / "report.html"
    out.write_text(html, encoding="utf-8")
    print(f"HTML report saved to: {out}")
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="analyze_runs.json")
    p.add_argument("--component", default=None, help="Folder under the router runs root to analyse.")
    p.add_argument("--runs-dir", type=Path, default=None, help="Override the router runs root.")
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    paths_cfg = load_paths(require(cfg, "paths_config"))

    component = args.component or require(cfg, "component")
    hops = [int(h) for h in require(cfg, "hops")]
    style = require(cfg, "style")
    suffixes = list(require(cfg, "model_name_suffixes_to_strip"))
    apply_style(style)

    runs_dir = args.runs_dir or runs_path(paths_cfg, require(cfg, "runs_subdir"))
    component_dir = Path(runs_dir) / component
    output_dir = args.output_dir or runs_path(paths_cfg, require(cfg, "reports_subdir"), component)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading run data from {component_dir} ...")
    runs_data = load_run_data(
        component_dir,
        hops,
        suffixes,
        skip_archived=bool(require(cfg, "skip_archived")),
        archived_marker=require(cfg, "archived_marker"),
    )
    if not runs_data:
        print("No run data found!")
        return

    print(f"Found {len(runs_data)} runs")
    print("Generating visualisations...")
    plots: dict[str, Any] = {
        "overall_accuracy": plot_overall_accuracy(runs_data, output_dir, style),
        "per_hop_accuracy": plot_per_hop_accuracy(runs_data, output_dir, style, hops),
        "confusion_matrices": plot_confusion_matrices(runs_data, output_dir, style, hops),
    }
    error_plot = plot_error_patterns(runs_data, output_dir, style)
    if error_plot:
        plots["error_patterns"] = error_plot

    report_path = generate_html_report(runs_data, plots, output_dir, hops, component)

    write_run_artifacts(
        output_dir,
        config_snapshot={
            "script": Path(__file__).name,
            "config_path": cfg.get("_config_path"),
            "runs_dir": str(runs_dir),
            "component": component,
            "output_dir": str(output_dir),
            "hops": hops,
            "style": style,
        },
        metrics={
            "script": Path(__file__).name,
            "created_utc": now_iso(),
            "component": component,
            "runs_analysed": len(runs_data),
            "run_ids": [r["run_id"] for r in runs_data],
            "per_run_overall_accuracy": {
                r["run_id"]: r["metrics"].get("overall_accuracy") for r in runs_data
            },
            "figures": {
                k: v for k, v in plots.items() if k != "confusion_matrices"
            },
            "confusion_matrix_figures": [cm["filename"] for cm in plots["confusion_matrices"]],
            "report_html": str(report_path),
        },
        note_title=f"Router run analysis ({component})",
        note_lines=[
            f"- Runs directory: `{component_dir}`",
            f"- Runs analysed: {len(runs_data)}",
            f"- Report: `{report_path}`",
        ],
        prefix="analysis_",
    )

    print(f"\nAnalysis complete. Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
