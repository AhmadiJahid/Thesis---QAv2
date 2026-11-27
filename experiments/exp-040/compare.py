"""exp-040 - CPU-side paired comparison of the three masking arms (raw / uniform / typed).

Runs after the decodes (in the notebook, or on the laptop once runs/exp-040/ is copied back).
No GPU, no model. Reuses exp-029's re-analysis machinery unchanged (experiments/exp-029/
reanalysis.py ``run_pair``: the evaluator's own ``_paired_bootstrap`` / ``_mcnemar`` /
``_paired_t_test_row``, and its ``holm``) and exp-036's family helpers (experiments/exp-036/
compare.py ``holm_p`` / ``verdict_row`` / ``load_items`` / ``means``), so the statistics are the
repo's paired bootstrap + exact McNemar + paired t with Holm beside the uncorrected verdict
(ADR 0009, 0036). Pairs, families and the typed-comparator rule are read from
experiments/exp-040/config.json, committed before any exp-040 p-value existed.

Usage (repo root)::

    python experiments/exp-040/compare.py
    python experiments/exp-040/compare.py --runs-root ../Thesis---QAv2/runs   # from a worktree

Writes experiments/exp-040/compare_metrics.json and prints a summary.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CONFIG = HERE / "config.json"
N_ITEMS = 600
STRATA = ("overall", "2", "3", "4")


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


def typed_comparator(check: dict | None, have_rebuilt: bool) -> dict:
    """config.json typed_comparator_rule: typed_rerun iff the rebuild reproduced exp-014's typed
    exemplars (ordered pool ids) for every query; otherwise typed_rebuilt, which must exist."""
    if check is None:
        raise SystemExit("runs/exp-040/retrieval/check.json missing - the retrieval step did not finish")
    ti = check["typed_identity"]
    identical = ti["identical"] == ti["compared"] == N_ITEMS
    if identical:
        return {"system": "typed_rerun", "typed_identity": ti, "reason": "rebuilt typed exemplars == exp-014's for 600/600 queries"}
    if not have_rebuilt:
        raise SystemExit(f"typed exemplars differ from exp-014's for {ti['compared'] - ti['identical']} queries, "
                         "so the pre-declared typed comparator is typed_rebuilt - run `run.sh typed_rebuilt` first")
    return {"system": "typed_rebuilt", "typed_identity": ti,
            "reason": f"rebuilt typed exemplars differ from exp-014's for {ti['compared'] - ti['identical']} queries"}


def hardware_control(results_a: Path, results_b: Path, rule: str) -> dict:
    a = {str(r["query_id"]): r["decomposition"] for r in json.loads(results_a.read_text(encoding="utf-8"))}
    b = {str(r["query_id"]): r["decomposition"] for r in json.loads(results_b.read_text(encoding="utf-8"))}
    if set(a) != set(b) or len(a) != N_ITEMS:
        raise SystemExit(f"hardware control: id sets differ ({len(a)} vs {len(b)})")
    same = sum(a[q] == b[q] for q in a)
    return {"items": len(a), "identical_decompositions": same, "reproduced": same == N_ITEMS, "rule": rule,
            "results_a": str(results_a), "results_b": str(results_b)}


def holm_families(results: dict, families: dict, metrics: list[str]) -> dict:
    out = {}
    for fam, spec in families.items():
        members = []
        for pk in spec["pairs"]:
            for st in spec["strata"]:
                cell = results[pk]["strata"][st]
                for m in metrics:
                    p, basis = C36.holm_p(cell, m)
                    members.append({"pair": pk, "stratum": st, "metric": m, "p": p, "p_basis": basis})
        adj = R.holm([x["p"] for x in members])
        for x, a in zip(members, adj):
            x["holm_adjusted_p"] = a
            x["holm_significant"] = bool(a < R.ALPHA)
            row = C36.verdict_row(results[x["pair"]]["strata"][x["stratum"]], x["metric"])
            x["significant_uncorrected"] = bool(row["significant"])
            row.setdefault("holm", {})[fam] = {"family_size": len(members), "p_basis": x["p_basis"], "p": x["p"],
                                               "adjusted_p": a, "significant": bool(a < R.ALPHA)}
        out[fam] = {**spec, "metrics": metrics, "size": len(members), "size_planned": spec["size"],
                    "uncorrected_significant": sum(x["significant_uncorrected"] for x in members),
                    "holm_significant": sum(x["holm_significant"] for x in members), "members": members}
        if len(members) != spec["size"]:
            raise SystemExit(f"{fam}: {len(members)} tests, the plan declared {spec['size']}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", type=Path, default=REPO / "runs")
    ap.add_argument("--out", type=Path, default=HERE / "compare_metrics.json")
    args = ap.parse_args(argv)

    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    cc = cfg["comparisons"]
    metrics = list(cc["report_card_metrics"])
    run = args.runs_root / "exp-040"

    def per_item(arm: str) -> Path:
        return run / arm / "eval" / "eval_per_item.json"

    check_p = run / "retrieval" / "check.json"
    check = json.loads(check_p.read_text(encoding="utf-8")) if check_p.exists() else None
    tc = typed_comparator(check, per_item("typed_rebuilt").exists())

    files = {arm: per_item(arm) for arm in ("typed_rerun", "raw", "uniform", "typed_rebuilt")}
    files["exp020_stored"] = args.runs_root / Path(cc["stored_comparators"]["exp020_stored"]["per_item"]).relative_to("runs")
    systems, inputs = {}, {}
    for key, path in files.items():
        if not path.exists():
            if key == "typed_rebuilt":
                continue
            raise SystemExit(f"missing per-item file for {key}: {path}")
        systems[key] = C36.load_items(path)
        if len(systems[key]) != N_ITEMS:
            raise SystemExit(f"{key}: {len(systems[key])} items, expected the pinned {N_ITEMS}")
        inputs[key] = {"path": str(path), "sha256": sha256(path), "n_items": N_ITEMS,
                       "report_card_means": C36.means(systems[key], metrics)}
    alias = {"typed": tc["system"]}

    pairs = {}
    for name, p in cc["primary_pairs"].items():
        pairs[name] = {**p, "primary": True}
    for name, p in cc["secondary_pairs"].items():
        if name == "typed_rebuilt_vs_typed_rerun" and "typed_rebuilt" not in systems:
            continue
        pairs[name] = {**p, "primary": False}
    results = {}
    for name, p in pairs.items():
        a, b = alias.get(p["system_a"], p["system_a"]), alias.get(p["system_b"], p["system_b"])
        results[name] = R.run_pair(a, b, STRATA, a=systems[a], b=systems[b])
        results[name].update(primary=p["primary"], declared_as=f"{p['system_a']} minus {p['system_b']}",
                             note=p.get("note"))

    fam_out = holm_families(results, cc["holm_families"], metrics)

    stored_res = args.runs_root / Path(cc["stored_comparators"]["exp020_stored"]["results"]).relative_to("runs")
    rerun_res = sorted((run / "typed_rerun" / "decomposer").glob("*/results.json"))
    hw = hardware_control(rerun_res[0], stored_res, cc["hardware_control"]["reproduced_if"]) if rerun_res else None

    git = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    out = {
        "exp_id": "exp-040", "kind": "CPU paired comparison of per-item files",
        "git_head_at_compute": git,
        "evaluator": {"path": "scripts/musique_decompositions_evaluator.py",
                      "sha256": sha256(REPO / "scripts" / "musique_decompositions_evaluator.py")},
        "config_sha256": sha256(CONFIG),
        "settings": {"seed": R.SEED, "bootstrap_iterations": R.ITER, "alpha": R.ALPHA,
                     "min_items_for_significance_claim": R.MIN_ITEMS,
                     "difference_direction": "system_a minus system_b",
                     "holm": "within each family of config.json comparisons.holm_families; evaluator verdicts stay uncorrected (ADR 0036)"},
        "typed_comparator": tc, "retrieval_check": check, "hardware_control": hw,
        "inputs": inputs, "pairs": results, "families": fam_out,
        "wording": cc["wording"],
    }
    args.out.write_text(json.dumps(out, indent=2, default=float) + "\n", encoding="utf-8")

    print(f"wrote {args.out}; typed comparator = {tc['system']} ({tc['reason']})")
    if hw:
        print(f"hardware control: {hw['identical_decompositions']}/{hw['items']} identical to stored exp-020 "
              f"-> {'reproduced' if hw['reproduced'] else 'NOT reproduced'}")
    for name, r in results.items():
        cell = r["strata"]["overall"]
        print(f"\n== {name} (overall, n={cell['n']}) {'[PRIMARY]' if r['primary'] else '[secondary, uncorrected]'}")
        for m in metrics:
            row = C36.verdict_row(cell, m)
            diff = row.get("difference", row.get("observed_difference"))
            h = row.get("holm", {}).get("F1_primary_overall")
            hs = "" if h is None else f"  holm_p={h['adjusted_p']:.3g}{' *' if h['significant'] else ''}"
            print(f"  {m:24s} diff={diff!s:>24}  sig_uncorrected={row['significant']}{hs}")
    for fam, f in fam_out.items():
        print(f"\n{fam}: size {f['size']}, uncorrected significant {f['uncorrected_significant']}, "
              f"Holm significant {f['holm_significant']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
