"""exp-041 - training-seed variance of exp-026's pool-only fine-tune: spread + per-seed verdicts.

CPU, no model. Reuses exp-029's machinery unchanged (experiments/exp-029/reanalysis.py
``run_pair`` -> the evaluator's own paired bootstrap / exact McNemar / paired t, and ``holm``)
and exp-036's family helpers (experiments/exp-036/compare.py). Everything it computes is fixed in
experiments/exp-041/run_config.json ``compare``, committed before any exp-041 p-value existed:

- spread: per report-card term x stratum, each seed's mean and the mean / SD / min / max / range
  across seeds, for the fine-tune and for its gain over the prompting comparator;
- per seed: fine-tune minus the same-machine prompting re-run (primary; one Holm family of 28 per
  seed) and minus the stored exp-015 (secondary); the stored exp-026 vs exp-015 reading is
  recomputed as the reference, and every seed's cell is read against it (sign, uncorrected and
  Holm verdict);
- context: seed-vs-seed pairs and the two hardware controls (uncorrected).

Usage (repo root)::

    python experiments/exp-041/compare.py
    python experiments/exp-041/compare.py --runs-root C:/Users/suuser/thesis/Thesis---QAv2/runs
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUN_CONFIG = HERE / "run_config.json"
N_ITEMS = 600


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


R = _load_module("exp029_reanalysis", REPO / "experiments" / "exp-029" / "reanalysis.py")
C36 = _load_module("exp036_compare", REPO / "experiments" / "exp-036" / "compare.py")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def stratum_means(rows: dict, metrics: list[str], strata: list[str]) -> dict:
    out = {}
    for st in strata:
        sel = [r for r in rows.values() if st == "overall" or int(r["gold_hop_count"]) == int(st)]
        out[st] = {m: sum(float(r[m]) for r in sel) / len(sel) for m in metrics}
    return out


def describe(vals: list[float]) -> dict:
    return {"values": vals, "mean": statistics.fmean(vals),
            "sd_ddof1": statistics.stdev(vals) if len(vals) > 1 else None,
            "min": min(vals), "max": max(vals), "range": max(vals) - min(vals)}


def spread(systems: dict, members: list[str], comparator: str, metrics, strata) -> dict:
    m = {k: stratum_means(systems[k], metrics, strata) for k in members + [comparator]}
    out = {"members": members, "gain_comparator": comparator, "strata": {}}
    for st in strata:
        out["strata"][st] = {
            met: {"fine_tune": describe([m[k][st][met] for k in members]),
                  "gain_vs_comparator": describe([m[k][st][met] - m[comparator][st][met] for k in members])}
            for met in metrics}
    return out


def family(results: dict, name: str, pair: str, metrics, strata) -> dict:
    members = []
    for st in strata:
        cell = results[pair]["strata"][st]
        for met in metrics:
            p, basis = C36.holm_p(cell, met)
            members.append({"stratum": st, "metric": met, "p": p, "p_basis": basis})
    for x, a in zip(members, R.holm([x["p"] for x in members])):
        row = C36.verdict_row(results[pair]["strata"][x["stratum"]], x["metric"])
        x.update(holm_adjusted_p=a, holm_significant=bool(a < R.ALPHA),
                 difference=float(row["difference"]), significant_uncorrected=bool(row["significant"]))
        row.setdefault("holm", {})[name] = {"family_size": len(members), "p_basis": x["p_basis"], "p": x["p"],
                                            "adjusted_p": a, "significant": bool(a < R.ALPHA)}
    return {"pair": pair, "size": len(members),
            "uncorrected_significant": sum(x["significant_uncorrected"] for x in members),
            "holm_significant": sum(x["holm_significant"] for x in members), "members": members}


def sign(x: float) -> int:
    return (x > 0) - (x < 0)


def agreement(fams: dict, ref: str, seeds: list[str]) -> dict:
    ref_cells = {(c["stratum"], c["metric"]): c for c in fams[ref]["members"]}
    cells, counts = [], {s: {"same_sign": 0, "same_uncorrected": 0, "same_holm": 0} for s in seeds}
    for key, rc in ref_cells.items():
        row = {"stratum": key[0], "metric": key[1], "reference": {k: rc[k] for k in
               ("difference", "significant_uncorrected", "holm_significant", "holm_adjusted_p")}, "seeds": {}}
        for s in seeds:
            c = next(x for x in fams[s]["members"] if (x["stratum"], x["metric"]) == key)
            same = {"same_sign": sign(c["difference"]) == sign(rc["difference"]),
                    "same_uncorrected": c["significant_uncorrected"] == rc["significant_uncorrected"],
                    "same_holm": c["holm_significant"] == rc["holm_significant"]}
            for k, v in same.items():
                counts[s][k] += int(v)
            row["seeds"][s] = {"difference": c["difference"], "significant_uncorrected": c["significant_uncorrected"],
                               "holm_significant": c["holm_significant"], "holm_adjusted_p": c["holm_adjusted_p"], **same}
        row["holds_across_all_seeds"] = all(
            row["seeds"][s]["same_sign"] and row["seeds"][s]["holm_significant"] for s in seeds) and rc["holm_significant"]
        cells.append(row)
    return {"reference_family": ref, "cells": cells, "counts_out_of": len(cells), "counts": counts,
            "cells_holding_across_all_seeds": sum(r["holds_across_all_seeds"] for r in cells),
            "rule": "a cell 'holds across all seeds' iff the reference is Holm-significant and every seed's primary pair has the same sign and is Holm-significant in its own family"}


def identical(a: Path, b: Path) -> dict:
    da = {str(r["query_id"]): r["decomposition"] for r in json.loads(a.read_text(encoding="utf-8"))}
    db = {str(r["query_id"]): r["decomposition"] for r in json.loads(b.read_text(encoding="utf-8"))}
    if set(da) != set(db):
        raise SystemExit(f"hardware control: id sets differ ({a} vs {b})")
    same = sum(da[q] == db[q] for q in da)
    return {"items": len(da), "identical_decompositions": same, "reproduced": same == N_ITEMS == len(da),
            "a": str(a), "b": str(b)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", type=Path, default=REPO / "runs")
    ap.add_argument("--out", type=Path, default=HERE / "compare_metrics.json")
    args = ap.parse_args(argv)
    cfg = json.loads(RUN_CONFIG.read_text(encoding="utf-8"))
    cc = cfg["compare"]
    metrics, strata = list(cc["report_card_metrics"]), list(cc["strata"])

    def under(p: str) -> Path:
        return args.runs_root / Path(p).relative_to("runs")

    systems, inputs = {}, {}
    for key, rel in cc["systems"].items():
        path = under(rel)
        if not path.exists():
            raise SystemExit(f"missing per-item file for {key}: {path}")
        systems[key] = C36.load_items(path)
        if len(systems[key]) != N_ITEMS:
            raise SystemExit(f"{key}: {len(systems[key])} items, expected {N_ITEMS}")
        inputs[key] = {"path": str(path), "sha256": sha256(path)}

    groups = {"primary": cc["primary_pairs"], "reference": cc["reference_pair"],
              "secondary": cc["secondary_pairs"], "context": cc["context_pairs"]}
    results, fams = {}, {}
    for grp, pairs in groups.items():
        for name, p in pairs.items():
            results[name] = R.run_pair(p["system_a"], p["system_b"], tuple(strata),
                                       a=systems[p["system_a"]], b=systems[p["system_b"]])
            results[name].update(group=grp, note=p.get("note"))
            if "family" in p:
                fams[p["family"]] = family(results, p["family"], name, metrics, strata)
                if fams[p["family"]]["size"] != cc["family_size_each"]:
                    raise SystemExit(f"{p['family']}: size {fams[p['family']]['size']} != {cc['family_size_each']}")

    seed_fams = [p["family"] for p in cc["primary_pairs"].values()]
    stored_fams = [p["family"] for p in cc["secondary_pairs"].values()]
    ref_fam = next(iter(cc["reference_pair"].values()))["family"]
    sp = cc["spread"]
    run = args.runs_root / "exp-041"
    hw = {}
    for arm, (glob_root, stored) in {"ft_seed42": ("seed42/decomposer", "exp026_stored"),
                                     "prompted_rerun": ("prompted_exp015_rerun/decomposer", "exp015_stored")}.items():
        res = sorted((run / glob_root).glob("*/results.json"))
        if res:
            hw[arm] = {"vs": stored, **identical(res[0], under(cc["stored_results"][stored]))}

    git = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    out = {
        "exp_id": "exp-041", "kind": "CPU spread + paired comparisons of per-item files",
        "git_head_at_compute": git, "run_config_sha256": sha256(RUN_CONFIG),
        "evaluator_sha256": sha256(REPO / "scripts" / "musique_decompositions_evaluator.py"),
        "settings": {"seed": R.SEED, "bootstrap_iterations": R.ITER, "alpha": R.ALPHA,
                     "difference_direction": "system_a minus system_b",
                     "holm": "within each family (one pair x 7 terms x 4 strata = 28); evaluator verdicts stay uncorrected (ADR 0036)"},
        "inputs": inputs,
        "spread_primary": spread(systems, sp["primary_set"], "prompted_rerun", metrics, strata),
        "spread_secondary": spread(systems, sp["secondary_set"], "exp015_stored", metrics, strata),
        "agreement_primary_vs_reference": agreement(fams, ref_fam, seed_fams),
        "agreement_secondary_vs_reference": agreement(fams, ref_fam, stored_fams),
        "families": fams, "hardware_control": hw, "pairs": results,
        "wording": "ADR 0036 item 4: non-significance is 'not distinguishable', never 'equal'. n = 3 seeds: the spread is descriptive.",
    }
    args.out.write_text(json.dumps(out, indent=2, default=float) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    for k, v in hw.items():
        print(f"hardware control {k} vs {v['vs']}: {v['identical_decompositions']}/{v['items']} identical")
    ag = out["agreement_primary_vs_reference"]
    print(f"cells holding across all seeds: {ag['cells_holding_across_all_seeds']}/{ag['counts_out_of']}")
    for s, c in ag["counts"].items():
        print(f"  {s}: same sign {c['same_sign']}, same uncorrected verdict {c['same_uncorrected']}, same Holm verdict {c['same_holm']} (of {ag['counts_out_of']})")
    for fam, f in fams.items():
        print(f"{fam}: uncorrected significant {f['uncorrected_significant']}/{f['size']}, Holm significant {f['holm_significant']}")
    print("overall spread of the gain (primary set):")
    for met, d in out["spread_primary"]["strata"]["overall"].items():
        g = d["gain_vs_comparator"]
        print(f"  {met:24s} gain mean {g['mean']:+.4f} sd {g['sd_ddof1']:.4f} range [{g['min']:+.4f}, {g['max']:+.4f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
