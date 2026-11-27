#!/usr/bin/env python3
"""Per-arm metrics and the pre-declared paired comparisons for exp-039 (ADR 0045).

Reads each arm's compile-execute run (the per-item dumps scripts/run_metaqa_compile_execute.py
writes under ``<run>/analysis/``) and the MetaQA gold of the evaluation set, and scores every
item of the set - a row that failed to compile or execute scores 0 on every answer metric
(the all-items denominator; nothing is dropped).

Per item:
  - ``compiled`` / ``executed``: the stage the row reached;
  - ``em``: exact set match (Jaccard 1.0) - the thesis's answer metric;
  - ``jaccard``: |P & G| / |P | G|;
  - ``any_hit``: P & G non-empty (the "hit rate" some LLM-KGQA papers report);
  - ``rhits1``: |P & G| / |P|, the expected Hits@1 of a uniformly random pick from the
    unranked answer set (Interactive-KBQA's RHits@1 in expectation). Symbolic execution
    returns a set, not a ranking, so ranked Hits@1 is undefined here; this is the closest
    honest equivalent (ADR 0045).

Per arm and hop: each rate with a seeded percentile-bootstrap 95% CI.
Per declared comparison and hop: exact McNemar on ``em`` (the primary family, Holm over all
declared comparisons x hops, ADR 0036) and a paired percentile bootstrap CI plus paired-t p on
``jaccard`` (secondary family, Holm likewise). Holm is reported beside the uncorrected p.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy import stats

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from run_artifacts import now_iso, write_run_artifacts  # noqa: E402
from run_config import load_config, load_paths, require, resolve_path, runs_path  # noqa: E402
from seeding import set_global_seed  # noqa: E402

METRICS = ("compiled", "executed", "em", "jaccard", "any_hit", "rhits1")


def load_gold(paths_cfg: dict, cfg: dict, data_root: Path) -> Dict[str, tuple]:
    sep = require(cfg, "answer_separator")
    gold: Dict[str, tuple] = {}
    for hop in require(cfg, "hops"):
        def _lines(key: str) -> List[str]:
            f = resolve_path(require(paths_cfg, "datasets." + require(cfg, key)).format(hop=hop), data_root)
            return [ln.strip() for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]
        qs, ans = _lines("questions_template_key"), _lines("answers_template_key")
        if len(qs) != len(ans):
            raise SystemExit(f"hop {hop}: question/answer line counts differ")
        for q, a in zip(qs, ans):
            if q in gold:
                raise SystemExit(f"question text repeated in the evaluation set: {q!r}")
            gold[q] = (int(hop), {x.strip() for x in a.split(sep) if x.strip()})
    return gold


def score_arm(analysis_dir: Path, gold: Dict[str, tuple]) -> Dict[str, Dict[str, float]]:
    rows: Dict[str, Dict[str, float]] = {}

    def _load(name: str) -> List[dict]:
        return json.loads((analysis_dir / name).read_text(encoding="utf-8"))

    for it in _load("success.json"):
        p = set(it["kg_results"])
        g = gold[it["question"]][1]
        inter = len(p & g)
        rows[it["question"]] = {"compiled": 1.0, "executed": 1.0, "em": float(p == g),
                                "jaccard": inter / len(p | g) if p | g else 1.0,
                                "any_hit": float(inter > 0), "rhits1": inter / len(p) if p else 0.0}
    for name, compiled in (("compile_fail.json", 0.0), ("exec_fail.json", 1.0)):
        for it in _load(name):
            rows[it["question"]] = {"compiled": compiled, "executed": 0.0, "em": 0.0,
                                    "jaccard": 0.0, "any_hit": 0.0, "rhits1": 0.0}
    missing = set(gold) - set(rows)
    extra = set(rows) - set(gold)
    if missing or extra:
        raise SystemExit(f"{analysis_dir}: {len(missing)} evaluation-set questions have no row, "
                         f"{len(extra)} rows are not in the evaluation set - not the same set")
    return rows


def boot_ci(x: np.ndarray, rng: np.random.Generator, iters: int, alpha: float) -> List[float]:
    idx = rng.integers(0, len(x), size=(iters, len(x)))
    means = x[idx].mean(axis=1)
    return [float(v) for v in np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])]


def mcnemar_exact_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2.0 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


def holm(pvals: List[float]) -> List[float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="metaqa_arm_report.json")
    ap.add_argument("--arm", action="append", default=[], metavar="NAME=RUN_DIR",
                    help="An arm's compile-execute run dir (holding analysis/). Repeatable; "
                         "every arm named in the config's comparisons must be given.")
    ap.add_argument("--run-dir", type=Path, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths_cfg = load_paths(require(cfg, "paths_config"))
    data_root = Path(paths_cfg["data_root_resolved"])
    seed = int(require(cfg, "seed"))
    set_global_seed(seed)
    iters, alpha = int(require(cfg, "bootstrap_iterations")), float(require(cfg, "alpha"))
    analysis_subdir = require(cfg, "analysis_subdir")
    hops = [int(h) for h in require(cfg, "hops")]

    gold = load_gold(paths_cfg, cfg, data_root)
    arms = dict(a.split("=", 1) for a in args.arm)
    scored = {name: score_arm(Path(d) / analysis_subdir, gold) for name, d in arms.items()}

    per_arm: Dict[str, Any] = {}
    for name, rows in scored.items():
        per_arm[name] = {}
        for hop in hops:
            qs = sorted(q for q in gold if gold[q][0] == hop)
            rng = np.random.default_rng(seed)
            cell = {"n": len(qs)}
            for m in METRICS:
                x = np.array([rows[q][m] for q in qs])
                cell[m] = {"mean": float(x.mean()), "ci95": boot_ci(x, rng, iters, alpha)}
            per_arm[name][str(hop)] = cell

    comparisons = []
    for a, b in require(cfg, "comparisons"):
        if a not in scored or b not in scored:
            print(f"comparison {a} vs {b} skipped: arm not supplied")
            comparisons.append({"a": a, "b": b, "skipped": "arm not supplied"})
            continue
        for hop in hops:
            qs = sorted(q for q in gold if gold[q][0] == hop)
            ea = np.array([scored[a][q]["em"] for q in qs]); eb = np.array([scored[b][q]["em"] for q in qs])
            only_a, only_b = int(((ea == 1) & (eb == 0)).sum()), int(((ea == 0) & (eb == 1)).sum())
            ja = np.array([scored[a][q]["jaccard"] for q in qs]); jb = np.array([scored[b][q]["jaccard"] for q in qs])
            rng = np.random.default_rng(seed)
            d = ja - jb
            idx = rng.integers(0, len(d), size=(iters, len(d)))
            ci = [float(v) for v in np.percentile(d[idx].mean(axis=1), [100 * alpha / 2, 100 * (1 - alpha / 2)])]
            t_p = float(stats.ttest_rel(ja, jb).pvalue) if np.any(d != d[0]) else None
            comparisons.append({
                "a": a, "b": b, "hop": hop, "n": len(qs),
                "em": {"a": float(ea.mean()), "b": float(eb.mean()), "difference": float(ea.mean() - eb.mean()),
                       "only_a": only_a, "only_b": only_b, "mcnemar_exact_p": mcnemar_exact_p(only_a, only_b)},
                "jaccard": {"a": float(ja.mean()), "b": float(jb.mean()), "difference": float(d.mean()),
                            "paired_bootstrap_ci95": ci, "paired_t_p": t_p},
            })
    done = [c for c in comparisons if "skipped" not in c]
    for key, getter in (("em", lambda c: c["em"]["mcnemar_exact_p"]),
                        ("jaccard", lambda c: c["jaccard"]["paired_t_p"])):
        ps = [getter(c) if getter(c) is not None else 1.0 for c in done]
        for c, h in zip(done, holm(ps)):
            c[key]["holm_p"] = h
            c[key]["holm_family_size"] = len(ps)
            c[key]["significant_holm"] = bool(h < alpha)

    out = {"script": Path(__file__).name, "created_utc": now_iso(), "seed": seed,
           "evaluation_set_size": len(gold), "arms": {k: str(v) for k, v in arms.items()},
           "per_arm": per_arm, "comparisons": comparisons,
           "declared_family_size": len(require(cfg, "comparisons")) * len(hops)}
    run_dir = args.run_dir or runs_path(paths_cfg, require(cfg, "run_subdir"))
    run_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for name, cells in per_arm.items():
        for hop, c in cells.items():
            lines.append(f"- {name} hop {hop}: " + ", ".join(
                f"{m} {c[m]['mean']:.3f} [{c[m]['ci95'][0]:.3f}, {c[m]['ci95'][1]:.3f}]" for m in METRICS))
    for c in done:
        lines.append(f"- {c['a']} vs {c['b']} hop {c['hop']}: EM {c['em']['difference']:+.3f} "
                     f"({c['em']['only_a']}:{c['em']['only_b']}, p={c['em']['mcnemar_exact_p']:.3g}, "
                     f"Holm p={c['em']['holm_p']:.3g}); Jaccard {c['jaccard']['difference']:+.3f} "
                     f"CI {c['jaccard']['paired_bootstrap_ci95']}")
    print("\n".join(lines))
    write_run_artifacts(run_dir, config_snapshot={"script": Path(__file__).name, "config_path": cfg.get("_config_path"),
                                                  "arms": out["arms"], "seed": seed},
                        metrics=out, note_title="exp-039 MetaQA arm report", note_lines=lines, prefix="arm_report_")


if __name__ == "__main__":
    main()
