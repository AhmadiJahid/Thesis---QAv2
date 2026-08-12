#!/usr/bin/env python3
"""
Plot hop / stratum counts per MuSiQue chunk JSONL (raw split or cleaned).

Writes the figures plus the standard run trail under the configured runs root.

Ported from v1. Adapted for v2: the input glob, output directory, dpi and figure
format come from ``configs/musique_prep.json``; globs resolve against ``data_root``.

    python MusiQue/scripts/plot_musique_chunk_stats.py
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

from _prep_common import expand_glob, load_prep, require, run_dir_for

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from musique_ids import coarse_hop_from_record, stratum_from_id  # noqa: E402
from run_artifacts import now_iso, write_run_artifacts  # noqa: E402
from seeding import set_global_seed  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="musique_prep.json")
    p.add_argument("--inputs", nargs="*", type=Path, help="Ordered chunk JSONL paths.")
    p.add_argument("--input-glob", default=None, help="Override the config glob (data-root relative).")
    p.add_argument("--out-dir", type=Path, default=None, help="Directory for figures and the run trail.")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--dpi", type=int, default=None)
    p.add_argument("--format", default=None, choices=("png", "pdf", "svg"))
    p.add_argument("--no-stratum-figure", action="store_true", help="Skip the id-prefix stratum heatmap.")
    p.add_argument(
        "--allow-clean-inputs",
        action="store_true",
        help="Include *_clean.jsonl glob matches (default: skip stems ending with _clean).",
    )
    return p.parse_args()


def _chunk_label(path: Path) -> str:
    m = re.search(r"_(\d+)(?:_clean)?\.jsonl$", path.name)
    return m.group(1) if m else path.stem


def main() -> None:
    args = _parse_args()
    cfg, paths_cfg = load_prep(args.config)
    section = require(cfg, "plot_chunk_stats")

    seed = args.seed if args.seed is not None else int(require(cfg, "seed"))
    seeded = set_global_seed(seed)  # logged; this script does not sample
    dpi = args.dpi if args.dpi is not None else int(require(section, "dpi"))
    fmt = args.format or require(section, "format")
    stratum_figure = require(section, "stratum_figure") and not args.no_stratum_figure

    if args.inputs:
        inputs = [Path(p).resolve() for p in args.inputs]
    else:
        pattern = args.input_glob or require(section, "input_glob")
        inputs = expand_glob(paths_cfg, pattern)
        if not args.allow_clean_inputs:
            inputs = [p for p in inputs if not p.stem.endswith("_clean")]
        if not inputs:
            raise SystemExit(f"no files matched glob {pattern!r} under the data root")

    out_dir = args.out_dir or run_dir_for(paths_cfg, require(section, "run_subdir"))
    out_dir.mkdir(parents=True, exist_ok=True)

    chunk_keys = [_chunk_label(p) for p in inputs]
    coarse_per_chunk: dict[str, Counter[int]] = {k: Counter() for k in chunk_keys}
    stratum_per_chunk: dict[str, Counter[str]] = {k: Counter() for k in chunk_keys}

    for path, ck in zip(inputs, chunk_keys):
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                ch = coarse_hop_from_record(obj)
                if ch >= 0:
                    coarse_per_chunk[ck][ch] += 1
                stratum_per_chunk[ck][stratum_from_id(obj.get("id", ""))] += 1

    hops_sorted = sorted({h for c in coarse_per_chunk.values() for h in c.keys()})

    # --- Figure 1: grouped bars, coarse hop per chunk ---
    x = range(len(chunk_keys))
    width = 0.22
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, h in enumerate(hops_sorted):
        offsets = [xi + (i - (len(hops_sorted) - 1) / 2) * width for xi in x]
        heights = [coarse_per_chunk[ck][h] for ck in chunk_keys]
        ax.bar(offsets, heights, width=width, label=str(h))
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"chunk {k}" for k in chunk_keys])
    ax.set_ylabel("count")
    ax.set_title("Coarse hop counts per chunk")
    ax.legend(title="hop")
    fig.tight_layout()
    p1 = out_dir / f"hop_counts_per_chunk.{fmt}"
    fig.savefig(p1, dpi=dpi)
    plt.close(fig)

    # --- Figure 2: chunk x stratum heatmap ---
    p2 = None
    if stratum_figure:
        strata_sorted = sorted({s for c in stratum_per_chunk.values() for s in c.keys()})
        mat = [[stratum_per_chunk[ck][s] for ck in chunk_keys] for s in strata_sorted]
        fig2, ax2 = plt.subplots(
            figsize=(max(8, len(chunk_keys) * 1.2), max(4, len(strata_sorted) * 0.35))
        )
        im = ax2.imshow(mat, aspect="auto", cmap="Blues")
        ax2.set_xticks(range(len(chunk_keys)))
        ax2.set_xticklabels([f"c{k}" for k in chunk_keys])
        ax2.set_yticks(range(len(strata_sorted)))
        ax2.set_yticklabels(strata_sorted)
        ax2.set_xlabel("chunk")
        ax2.set_ylabel("id stratum")
        ax2.set_title("Stratum counts per chunk (heatmap)")
        fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        fig2.tight_layout()
        p2 = out_dir / f"stratum_heatmap_per_chunk.{fmt}"
        fig2.savefig(p2, dpi=dpi)
        plt.close(fig2)

    metrics = {
        "script": Path(__file__).name,
        "created_utc": now_iso(),
        "seed": seed,
        "seeded": seeded,
        "input_files": [str(p) for p in inputs],
        "chunk_keys": chunk_keys,
        "coarse_hop_counts_per_chunk": {
            k: {str(h): coarse_per_chunk[k][h] for h in hops_sorted} for k in chunk_keys
        },
        "stratum_counts_per_chunk": {
            k: dict(sorted(stratum_per_chunk[k].items())) for k in chunk_keys
        },
        "figures": {
            "hop_counts_per_chunk": str(p1),
            **({"stratum_heatmap_per_chunk": str(p2)} if p2 else {}),
        },
    }
    write_run_artifacts(
        out_dir,
        config_snapshot={
            "script": Path(__file__).name,
            "config_path": cfg.get("_config_path"),
            "inputs": [str(p) for p in inputs],
            "out_dir": str(out_dir),
            "dpi": dpi,
            "format": fmt,
            "stratum_figure": stratum_figure,
            "seed": seed,
        },
        metrics=metrics,
        note_title="MuSiQue chunk stats plots",
        note_lines=[
            f"- Seed: {seed}",
            f"- Inputs: {len(inputs)} file(s)",
            f"- Figures: `{p1}`" + (f", `{p2}`" if p2 else ""),
        ],
        prefix="plot_",
    )
    print(f"Wrote {p1}")
    if p2:
        print(f"Wrote {p2}")


if __name__ == "__main__":
    main()
