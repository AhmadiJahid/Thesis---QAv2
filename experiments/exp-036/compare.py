"""exp-036 - CPU-side paired comparison of the Qwen3.5-9B pool-only fine-tune (Refs #49, #50).

Runs on the laptop after the Colab run's per-item files are copied back into runs/exp-036/.
No GPU, no model. Reuses exp-029's re-analysis machinery unchanged (experiments/exp-029/
reanalysis.py: ``run_pair`` -> the evaluator's own ``_paired_bootstrap`` / ``_mcnemar`` /
``_paired_t_test_row`` / ``_significance_floor``, and its ``holm``), so the statistics are the
repo's paired bootstrap + exact McNemar (ADR 0009) with Holm-Bonferroni reported beside the
uncorrected verdict (ADR 0036). Pairs, strata, metrics, the difference-of-differences and the
Holm families are read from experiments/exp-036/run_config.json, committed before any exp-036
p-value existed.

The PRIMARY pair is (b): the Qwen fine-tune at 4-bit vs Qwen prompting at 4-bit (exp-027); the
4-bit eval is mandatory. The cross-family claim rests on family F3, a paired bootstrap of the
per-item difference of differences (Qwen gain - Mistral gain) on the shared pinned 600.

Usage (repo root)::

    python experiments/exp-036/compare.py
    python experiments/exp-036/compare.py --fp16-failed-gate   # fp16 eval failed its gate / did not run
    python experiments/exp-036/compare.py --runs-root ../Thesis---QAv2/runs   # from a worktree

Writes experiments/exp-036/compare_metrics.json and prints a summary table.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "experiments" / "exp-029"))
import reanalysis as R  # noqa: E402  (exp-029; importing runs no analysis)

E = R.E  # scripts/musique_decompositions_evaluator.py, as exp-029 imported it
RUN_CONFIG = HERE / "run_config.json"
DOD = "difference_of_differences"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_items(path: Path) -> dict[str, dict]:
    d = json.loads(path.read_text(encoding="utf-8"))
    items = d["items"]
    by = {str(r["item_id"]): r for r in items}
    if len(by) != len(items):
        raise SystemExit(f"duplicate item ids in {path}")
    return by


def means(rows: dict[str, dict], metrics: list[str]) -> dict[str, float]:
    return {m: sum(float(r[m]) for r in rows.values()) / len(rows) for m in metrics}


def holm_p(cell: dict, metric: str) -> tuple[float, str]:
    if metric in cell["mcnemar"]:
        return cell["mcnemar"][metric]["p_value"], "mcnemar_exact"
    if metric in cell["t_test"]:
        return cell["t_test"][metric]["p_value"], "paired_t (bootstrap emits no p)"
    raise SystemExit(f"{metric}: no p-value in this cell")


def verdict_row(cell: dict, metric: str) -> dict:
    return cell["mcnemar"].get(metric) or cell["bootstrap"][metric]


def plan(cfg: dict, fp16_failed_gate: bool) -> tuple[dict, dict]:
    """Pairs and families to run. The primary pair and the DoD never drop: the 4-bit eval is
    mandatory. --fp16-failed-gate drops every pair that needs the fp16 eval and SAYS so; it
    never substitutes another system into a fixed family."""
    pairs = dict(cfg["compare"]["pairs"])
    families = {k: dict(v) for k, v in cfg["compare"]["families"].items()}
    if fp16_failed_gate:
        pairs = {k: v for k, v in pairs.items() if "exp036_fp16" not in (v["system_a"], v["system_b"])}
        for fam in families.values():
            kept = [p for p in fam["pairs"] if p in pairs or p == DOD]
            fam["dropped_pairs_fp16_eval_failed_gate"] = sorted(set(fam["pairs"]) - set(kept))
            fam["pairs"] = kept
        families = {k: v for k, v in families.items() if v["pairs"]}
    primary = [k for k, v in pairs.items() if v.get("primary")]
    if primary != ["b_finetune_gain_qwen"]:
        raise SystemExit(f"run_config must declare exactly pair (b) primary, found {primary}")
    return pairs, families


def per_item_gain(a: dict, b: dict, ids: list[str], metrics: list[str]) -> dict[str, np.ndarray]:
    return {m: np.array([float(a[i][m]) - float(b[i][m]) for i in ids]) for m in metrics}


def difference_of_differences(spec: dict, systems: dict, metrics: list[str], strata: tuple) -> dict:
    """Per-item (Qwen gain - Mistral gain) on the shared 600: bootstrap CI + paired t per metric."""
    q, m = spec["qwen_gain"], spec["mistral_gain"]
    keys = [q["system_a"], q["system_b"], m["system_a"], m["system_b"]]
    ids = sorted(systems[keys[0]])
    for k in keys[1:]:
        if set(systems[k]) != set(ids):
            raise SystemExit(f"DoD: {k} does not cover the same item ids as {keys[0]} - refusing")
    for i in ids:
        if len({int(systems[k][i]["gold_hop_count"]) for k in keys}) != 1:
            raise SystemExit(f"DoD: {i}: gold hop differs across the four systems")
    out = {"qwen_gain": q, "mistral_gain": m, "ids_identical": True, "n_total": len(ids),
           "difference_direction": "mean(Qwen gain) - mean(Mistral gain); for ged/under/over a negative value means the Qwen gain is the larger",
           "note": "bootstrap rows' 'favours'/'direction' fields are the evaluator's per-metric labels and are not meaningful for a difference of differences; read the sign with difference_direction",
           "strata": {}}
    for st in strata:
        sel = ids if st == "overall" else [i for i in ids if int(systems[keys[0]][i]["gold_hop_count"]) == int(st)]
        gq = per_item_gain(systems[q["system_a"]], systems[q["system_b"]], sel, metrics)
        gm = per_item_gain(systems[m["system_a"]], systems[m["system_b"]], sel, metrics)
        n = len(sel)
        floor = E._significance_floor(n, R.MIN_ITEMS)
        under = bool(floor["below_min_items"])
        boot = E._paired_bootstrap(gq, gm, n=n, iterations=R.ITER, alpha=R.ALPHA, seed=R.SEED,
                                   weights=R.WEIGHTS, scale=R.SCALE, chunk_size=R.CHUNK,
                                   underpowered=under, statistics=tuple(metrics))
        tt = {s: E._paired_t_test_row(gq[s], gm[s], R.ALPHA, under, name=s) for s in metrics}
        out["strata"][str(st)] = {"n": n, "significance_floor": floor, "bootstrap": boot,
                                  "mcnemar": {}, "t_test": tt}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", type=Path, default=REPO / "runs")
    ap.add_argument("--fp16-failed-gate", action="store_true",
                    help="the secondary fp16 eval failed its gate or did not run; drops pair (a) and the context pairs")
    ap.add_argument("--out", type=Path, default=HERE / "compare_metrics.json")
    args = ap.parse_args()

    cfg = json.loads(RUN_CONFIG.read_text(encoding="utf-8"))
    cc = cfg["compare"]
    metrics = list(cc["report_card_metrics"])
    pairs, families = plan(cfg, args.fp16_failed_gate)
    dod_spec = cc[DOD]

    files = {f"exp036_{k}": v for k, v in cc["exp036_per_item"].items()}
    files.update(cc["comparators"])
    needed = {s for p in pairs.values() for s in (p["system_a"], p["system_b"])}
    needed |= {s for g in ("qwen_gain", "mistral_gain") for s in dod_spec[g].values()}
    systems, inputs = {}, {}
    for key in sorted(needed):
        path = args.runs_root / Path(files[key]).relative_to("runs")
        if not path.exists():
            raise SystemExit(f"missing per-item file for {key}: {path}"
                             + ("  (fp16 eval failed its gate? pass --fp16-failed-gate)" if key == "exp036_fp16" else ""))
        systems[key] = load_items(path)
        inputs[key] = {"path": str(path), "sha256": sha256(path), "n_items": len(systems[key]),
                       "report_card_means": means(systems[key], metrics)}
        if len(systems[key]) != 600:
            raise SystemExit(f"{key}: {len(systems[key])} items, expected the pinned 600")

    strata = ("overall", "2", "3", "4")
    results = {name: R.run_pair(p["system_a"], p["system_b"], strata,
                                a=systems[p["system_a"]], b=systems[p["system_b"]])
               for name, p in pairs.items()}
    for name, p in pairs.items():
        results[name].update(framing=p["framing"], in_families=p["in_families"], primary=p.get("primary", False))
    results[DOD] = difference_of_differences(dod_spec, systems, metrics, strata)
    results[DOD].update(framing=dod_spec["_note"], in_families=True, primary=False)

    fam_out = {}
    for fam, spec in families.items():
        members = []
        for pk in spec["pairs"]:
            for st in spec["strata"]:
                cell = results[pk]["strata"][st]
                for m in metrics:
                    p, basis = holm_p(cell, m)
                    members.append({"pair": pk, "stratum": st, "metric": m, "p": p, "p_basis": basis})
        adj = R.holm([x["p"] for x in members])
        for x, a in zip(members, adj):
            x["holm_adjusted_p"] = a
            x["holm_significant"] = bool(a < R.ALPHA)
            verdict_row(results[x["pair"]]["strata"][x["stratum"]], x["metric"]).setdefault("holm", {})[fam] = {
                "family_size": len(members), "p_basis": x["p_basis"], "p": x["p"],
                "adjusted_p": a, "significant": bool(a < R.ALPHA)}
        fam_out[fam] = {**spec, "metrics": metrics, "size": len(members),
                        "size_planned": cc["families"][fam]["size"],
                        "uncorrected_significant": sum(
                            1 for x in members
                            if verdict_row(results[x["pair"]]["strata"][x["stratum"]], x["metric"])["significant"]),
                        "holm_significant": sum(1 for x in members if x["holm_significant"]),
                        "members": members}

    git = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    ev = REPO / "scripts" / "musique_decompositions_evaluator.py"
    out = {
        "exp_id": "exp-036", "kind": "CPU paired comparison of stored per-item files",
        "git_head_at_compute": git,
        "evaluator": {"path": "scripts/musique_decompositions_evaluator.py", "sha256": sha256(ev)},
        "run_config_sha256": sha256(RUN_CONFIG),
        "settings": {"seed": R.SEED, "bootstrap_iterations": R.ITER, "alpha": R.ALPHA,
                     "min_items_for_significance_claim": R.MIN_ITEMS,
                     "difference_direction": "system_a minus system_b",
                     "primary_pair": "b_finetune_gain_qwen",
                     "precision_asymmetry": "Qwen gain (pair b) is 4-bit vs 4-bit; the Mistral reference gain (exp-026 vs exp-015) is fp16 vs fp16",
                     "holm": "within each family of run_config.json compare.families; evaluator verdicts stay uncorrected (ADR 0036)",
                     "fp16_failed_gate": args.fp16_failed_gate},
        "inputs": inputs, "pairs": results, "families": fam_out,
    }
    args.out.write_text(json.dumps(out, indent=2, default=float) + "\n", encoding="utf-8")

    print(f"wrote {args.out}")
    fam_of = {p: f for f, s in families.items() for p in s["pairs"] if "overall" in s["strata"]}
    for name, r in results.items():
        cell = r["strata"]["overall"]
        tag = "[PRIMARY]" if r.get("primary") else ("" if r["in_families"] else "[CONTEXT ONLY]")
        print(f"\n== {name} (overall, n={cell['n']}) {tag}")
        for m in metrics:
            row = verdict_row(cell, m)
            diff = row.get("difference", row.get("observed_difference"))
            h = row.get("holm", {}).get(fam_of.get(name, ""))
            hs = "" if h is None else f"  holm_p={h['adjusted_p']:.3g}{' *' if h['significant'] else ''}"
            print(f"  {m:24s} diff={diff!s:>24}  sig_uncorrected={row['significant']}{hs}")
    for fam, f in fam_out.items():
        print(f"\n{fam}: size {f['size']} (planned {f['size_planned']}), "
              f"uncorrected significant {f['uncorrected_significant']}, Holm significant {f['holm_significant']}")


if __name__ == "__main__":
    main()
