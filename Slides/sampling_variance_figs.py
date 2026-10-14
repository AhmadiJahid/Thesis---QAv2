"""Draw the two exp-030 sampling-variance figures for the thesis deck (issue #56).

    .venv/bin/python -I Slides/sampling_variance_figs.py

Reads (never writes) the exp-030 resampler output and writes two vector PDFs into
Slides/figs/, sized for the body of a 16:9 deck slide (142 mm text width):

  sampling_variance_spread.pdf  -- per arm x metric: the 50 draw values (dots), the mean
                                   (tick) with the 2.5-97.5 percentile range (whisker),
                                   and the pinned-600 subset value (open diamond).
  sampling_variance_ranks.pdf   -- per metric: % of the 50 draws in which each arm takes
                                   each rank (1 = best, direction-aware).

Sources:
  experiments/exp-030/resample_metrics.json (committed) -- mean, p2.5, p97.5, pinned
      value, and rank_stability_pct. Every summary mark and every heatmap number comes
      from this file.
  runs/exp-030/resample/resample_per_draw.csv (resampler output, NOT committed; the
      resampler writes it next to resample_metrics.json) -- the per-draw values behind
      the dots only. Before drawing, each arm x metric mean, min and max recomputed from
      the CSV is checked against the committed JSON; any mismatch aborts.

No number is computed here that the slides quote.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
METRICS_JSON = REPO / "experiments" / "exp-030" / "resample_metrics.json"
PER_DRAW_CSV = REPO / "runs" / "exp-030" / "resample" / "resample_per_draw.csv"
OUT_DIR = REPO / "Slides" / "figs"
SEED = 42  # dot jitter only

MM = 1 / 25.4
FIG_W_MM, FIG_A_H_MM, FIG_B_H_MM = 142.0, 43.0, 50.0

# deck palette (Slides/thesis-deck-v2.tex preamble)
INK, INKTWO, MUTED, RULE, PAPER, ACCENT = (
    "#16211F", "#3C4A47", "#687573", "#D7DDD9", "#F5F6F4", "#0D6B60")

# (JSON label, deck label, deck colour); order follows the deck's model lists
ARMS = [
    ("Mistral-7B", "Mistral-7B", "#B07D2A"),          # cMis
    ("Qwen2.5-7B", "Qwen2.5-7B", "#1F5BA8"),          # cQs
    ("Qwen2.5-14B", "Qwen2.5-14B", "#A07ADB"),        # cQl
    ("Qwen3.5-9B", "Qwen3.5-9B", "#C2417F"),          # cQn
    ("Pool-FT (Mistral)", "Fine-tune, pool-only", "#1E6B3C"),  # cFT
]
METRICS = [
    ("break_exact_match_rate", "Break EM"),
    ("sari_macro", "SARI"),
    ("ged_macro", "GED"),
    ("chain_validity_macro", "Chain validity"),
    ("hop_count_exact_match_rate", "Hop-count EM"),
    ("under_decomposition_rate", "Under-decomposition"),
    ("over_decomposition_rate", "Over-decomposition"),
]
UP, DOWN = "⇑", "⇓"  # not in Plex; drawn from the DejaVu Sans fallback

FS_TITLE, FS_TICK, FS_CELL = 7.0, 6.5, 6.5


def setup_fonts() -> str:
    """Register IBM Plex Sans (system, $PLEX_SANS_DIR, or the tectonic cache)."""
    candidates = []
    if os.environ.get("PLEX_SANS_DIR"):
        candidates += glob.glob(os.path.join(os.environ["PLEX_SANS_DIR"], "IBMPlexSans-*.otf"))
    candidates += glob.glob(os.path.expanduser(
        "~/.cache/tectonic/bundles/data/*/IBMPlexSans-*.otf"))
    for path in candidates:
        if os.path.basename(path).split("-")[1].split(".")[0] in ("Regular", "Bold", "Medium"):
            font_manager.fontManager.addfont(path)
    names = {f.name for f in font_manager.fontManager.ttflist}
    family = "IBM Plex Sans" if "IBM Plex Sans" in names else "DejaVu Sans"
    if family != "IBM Plex Sans":
        print("WARNING: IBM Plex Sans not found; falling back to DejaVu Sans", file=sys.stderr)
    plt.rcParams.update({
        "font.family": [family, "DejaVu Sans"],
        "pdf.fonttype": 42,
        "font.size": FS_TICK,
        "text.color": INK,
        "axes.edgecolor": RULE,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2,
        "ytick.major.size": 2,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "savefig.transparent": True,
    })
    return family


def load() -> tuple[dict, dict]:
    data = json.loads(METRICS_JSON.read_text(encoding="utf-8"))
    per_draw: dict = {}
    with PER_DRAW_CSV.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["scope"] != "overall":
                continue
            per_draw.setdefault(row["arm"], {}).setdefault(row["metric"], {})[
                int(row["draw"])] = float(row["value"])
    k = data["k_draws"]
    out: dict = {}
    for arm, _, _ in ARMS:
        for m, _ in METRICS:
            d = per_draw[arm][m]
            if sorted(d) != list(range(k)):
                sys.exit(f"per-draw CSV: {arm}/{m} has draws {sorted(d)[:3]}..., expected 0..{k - 1}")
            vals = np.array([d[i] for i in range(k)])
            s = data["summary"][arm]["overall"][m]
            for name, got in (("mean", vals.mean()), ("min", vals.min()), ("max", vals.max())):
                if abs(got - s[name]) > 1e-12:
                    sys.exit(f"per-draw CSV disagrees with committed JSON: {arm}/{m} {name} "
                             f"{got} vs {s[name]}")
            out.setdefault(arm, {})[m] = vals
    return data, out


def title(ax, metric_key: str, label: str, data: dict) -> None:
    arrow = UP if data["directions"][metric_key] == "higher_is_better" else DOWN
    ax.set_title(f"{label} {arrow}", fontsize=FS_TITLE, color=INK, loc="left", pad=3,
                 fontweight="medium")


def fmt_tick(v: float, span: float) -> str:
    return f"{v:.3f}" if span < 0.03 else f"{v:.2f}"  # spans here are all >= 0.03


def fig_spread(data: dict, per_draw: dict, path: Path) -> None:
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(2, 4, figsize=(FIG_W_MM * MM, FIG_A_H_MM * MM))
    fig.subplots_adjust(left=0.055, right=0.995, top=0.92, bottom=0.04, wspace=0.42, hspace=0.5)
    for idx, (m, label) in enumerate(METRICS):
        ax = axes.flat[idx]
        lo_all, hi_all = [], []
        for i, (arm, _, colour) in enumerate(ARMS):
            vals = per_draw[arm][m]
            s = data["summary"][arm]["overall"][m]
            x = rng.uniform(-0.17, 0.17, size=vals.size) + i - 0.12
            ax.scatter(x, vals, s=2.2, color=colour, alpha=0.38, linewidths=0, zorder=2)
            xs = i + 0.26
            ax.plot([xs, xs], [s["p2.5"], s["p97.5"]], color=INK, lw=0.7, zorder=3,
                    solid_capstyle="butt")
            ax.plot([xs - 0.13, xs + 0.13], [s["mean"]] * 2, color=INK, lw=1.3, zorder=4,
                    solid_capstyle="butt")
            ax.scatter([xs], [s["pinned_value"]], marker="D", s=9, facecolors="white",
                       edgecolors=colour, linewidths=0.8, zorder=5)
            lo_all += [vals.min(), s["pinned_value"]]
            hi_all += [vals.max(), s["pinned_value"]]
        lo, hi = min(lo_all), max(hi_all)
        pad = 0.08 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlim(-0.55, len(ARMS) - 0.45)
        ax.set_xticks([])
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt_tick(v, hi - lo)))
        ax.tick_params(axis="y", pad=1.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.grid(axis="y", color=RULE, lw=0.4, zorder=0)
        ax.set_axisbelow(True)
        title(ax, m, label, data)
    leg = axes.flat[7]
    leg.axis("off")
    handles = [Line2D([], [], marker="o", ls="", ms=3.4, color=c, label=lab)
               for _, lab, c in ARMS]
    handles += [
        Line2D([], [], marker="_", ls="", ms=6, mew=1.3, color=INK, label="mean, 95% range"),
        Line2D([], [], marker="D", ls="", ms=3.2, mfc="white", mec=INK, mew=0.8,
               label="pinned 600"),
    ]
    # anchored at the top so the seven rows run up into the empty gap above the cell
    leg.legend(handles=handles, loc="upper left", frameon=False, fontsize=FS_TICK,
               handlelength=1.0, handletextpad=0.5, labelspacing=0.22, borderaxespad=0,
               labelcolor=INKTWO, bbox_to_anchor=(-0.05, 1.30))
    fig.savefig(path)
    plt.close(fig)


def fig_ranks(data: dict, path: Path) -> None:
    cmap = LinearSegmentedColormap.from_list("paper_accent", [PAPER, ACCENT])
    fig, axes = plt.subplots(2, 4, figsize=(FIG_W_MM * MM, FIG_B_H_MM * MM))
    fig.subplots_adjust(left=0.162, right=0.995, top=0.925, bottom=0.075, wspace=0.12, hspace=0.5)
    n = len(ARMS)
    for idx, (m, label) in enumerate(METRICS):
        ax = axes.flat[idx]
        grid = np.array([data["rank_stability_pct"][m][arm] for arm, _, _ in ARMS])
        ax.pcolormesh(np.arange(-0.5, n), np.arange(-0.5, n), grid, cmap=cmap, vmin=0,
                      vmax=100, edgecolors="white", linewidth=0.6)
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(n - 0.5, -0.5)
        for r in range(n):
            for c in range(n):
                v = grid[r, c]
                if v == 0:
                    continue
                ax.text(c, r, f"{v:.0f}", ha="center", va="center", fontsize=FS_CELL,
                        color="white" if v >= 55 else INK)
        ax.set_xticks(range(n))
        ax.set_xticklabels([str(i + 1) for i in range(n)])
        ax.tick_params(axis="both", length=0, pad=1.5)
        ax.set_yticks(range(n))
        if idx % 4 == 0:
            ax.set_yticklabels([lab for _, lab, _ in ARMS])
            for tl in ax.get_yticklabels():
                tl.set_color(INKTWO)
        else:
            ax.set_yticklabels([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        title(ax, m, label, data)
    key = axes.flat[7]
    key.axis("off")
    cax = key.inset_axes([0.08, 0.50, 0.84, 0.11])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 100))
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=[0, 50, 100])
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, pad=1.5, labelsize=FS_TICK)
    key.text(0.08, 0.78, "% of 50 draws", fontsize=FS_TICK, color=INKTWO, transform=key.transAxes)
    key.text(0.08, 0.10, "rank 1 = best; blank = 0", fontsize=FS_TICK, color=MUTED,
             transform=key.transAxes)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    family = setup_fonts()
    data, per_draw = load()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_spread(data, per_draw, OUT_DIR / "sampling_variance_spread.pdf")
    fig_ranks(data, OUT_DIR / "sampling_variance_ranks.pdf")
    print(f"font: {family}; wrote {OUT_DIR}/sampling_variance_spread.pdf, "
          f"{OUT_DIR}/sampling_variance_ranks.pdf; per-draw CSV checked against "
          f"{METRICS_JSON.relative_to(REPO)} for {len(ARMS)} arms x {len(METRICS)} metrics")


if __name__ == "__main__":
    main()
