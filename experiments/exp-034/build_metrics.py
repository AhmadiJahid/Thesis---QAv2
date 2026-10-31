"""Build experiments/exp-034/metrics.json from runs/exp-034/ (read-only).

exp-034 (issue #51, ADR 0039): exp-031's 9 stored pool-size arms re-parsed with the PR #61
scaffold-label parser and re-scored with the merged evaluator (method of exp-032), then exp-031's
8 compares (sizeN vs size2000) re-run with both sides re-parsed. Holm-Bonferroni (ADR 0036) over
exp-031's pre-fixed 64-test family, reported BESIDE the evaluator's uncorrected verdict. The
reproducibility pair (size2000 vs exp-020, here exp-032's re-parsed exp-020) is outside the family.
Blocker check: size2000 re-parsed metrics must equal exp-032's re-parsed exp-020 reading.
Usage: python3 experiments/exp-034/build_metrics.py  (paths absolute below)
"""
import glob
import json
import math
from pathlib import Path

ROOT = "/cta/users/fyilmaz/Thesis---QAv2"
RUNS = f"{ROOT}/runs"
R = f"{RUNS}/exp-034"
OUT = Path(__file__).parent / "metrics.json"
SIZES = [2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
ALPHA = 0.05
FAMILY = {
    "break_exact_match_rate": ("mcnemar", "break_exact_match", None, "higher_is_better"),
    "sari_macro": ("t_test", "sari", "sari", "higher_is_better"),
    "ged_macro": ("t_test", "ged", "ged", "lower_is_better"),
    "chain_validity_macro": ("t_test", "chain_validity", "chain_validity", "higher_is_better"),
    "hop_count_exact_match_rate": ("mcnemar", "hop_count_exact_match", None, "higher_is_better"),
    "under_decomposition_rate": ("mcnemar", "under_decomposition", None, "lower_is_better"),
    "over_decomposition_rate": ("mcnemar", "over_decomposition", None, "lower_is_better"),
    "step_f1_macro": ("t_test", "step_f1", "step_f1", "higher_is_better"),
}
BESIDE = ["step_f1_soft_050_macro", "step_f1_soft_060_macro", "plan_connectedness_macro",
          "plan_connectedness_null_items"]


def load(p):
    with open(p) as f:
        return json.load(f)


def holm(ps):
    # Same step-down implementation as experiments/exp-031/build_metrics.py.
    m = len(ps)
    order = sorted(range(m), key=lambda k: ps[k])
    adj = [0.0] * m
    running = 0.0
    for rank, k in enumerate(order):
        running = max(running, min(1.0, (m - rank) * ps[k]))
        adj[k] = running
    return adj


def numeric_leaves(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(numeric_leaves(v, f"{prefix}/{k}"))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            out.update(numeric_leaves(v, f"{prefix}[{i}]"))
    elif isinstance(d, (int, float)) and not isinstance(d, bool):
        out[prefix] = d
    return out


# per-size re-parse + eval
per_size, run_checks = {}, {}
for n in SIZES:
    rp = load(f"{R}/size{n}/reparse/reparse_metrics.json")
    ev = load(f"{R}/size{n}/eval/eval_metrics.json")
    old = load(f"{RUNS}/exp-031/size{n}/eval/eval_metrics.json")
    run_checks[f"size{n}"] = {
        "input_results": rp["input_results"], "input_results_sha256": rp["input_results_sha256"],
        "output_results_sha256": rp["output_results_sha256"],
        "reparse_rows_total": rp["rows_total"], "reparse_rows_changed": rp["rows_changed"],
        "reparse_steps_removed": rp["pred_steps_removed_total"],
        "eval_total_evaluated": ev["total_evaluated"], "eval_missing_gold_count": ev["missing_gold_count"],
        "eval_git_commit": ev["git"]["commit"], "eval_git_dirty": ev["git"].get("dirty"),
    }
    per_size[f"size{n}"] = {
        "reparsed": {k: ev[k] for k in FAMILY},
        "exp031_old_parser": {k: old[k] for k in FAMILY},
        "reparsed_minus_old": {k: ev[k] - old[k] for k in FAMILY},
        "beside_card_reparsed": {k: ev.get(k) for k in BESIDE},
    }


def cell(cm, sec, key, bkey, direction):
    r = cm[sec][key]
    out = {"test": "exact McNemar" if sec == "mcnemar" else "paired t-test",
           "system_a": r.get("system_a", r.get("system_a_rate")),
           "system_b": r.get("system_b", r.get("system_b_rate")),
           "difference": r["difference"], "direction": direction, "p_value": r["p_value"],
           "significant_uncorrected": r["significant"], "favours_uncorrected": r["favours"]}
    if sec == "mcnemar":
        out["discordant_only_in_a"] = r["discordant_only_in_a"]
        out["discordant_only_in_b"] = r["discordant_only_in_b"]
    if bkey is not None:
        b = cm["bootstrap"][bkey]
        out["bootstrap_ci_low"] = b["ci_low"]
        out["bootstrap_ci_high"] = b["ci_high"]
        out["bootstrap_significant"] = b["significant"]
        out["bootstrap_vs_t_disagree"] = b["significant"] != r["significant"]
    return out


old31 = load(f"{ROOT}/experiments/exp-031/metrics.json")["holm_family"]["pairs"]
family, cells = {}, []
for n in SIZES[1:]:
    pair = f"size{n}_vs_size2000"
    cm = load(f"{R}/compare/{pair}/compare_metrics.json")
    assert cm["num_aligned_items"] == 600
    assert cm["system_b_path"].endswith("exp-034/size2000/eval/eval_per_item.json")
    family[pair] = {"path": f"{R}/compare/{pair}/compare_metrics.json",
                    "difference_direction": cm["difference_direction"],
                    "num_aligned_items": cm["num_aligned_items"],
                    "git_commit": cm["git"]["commit"], "cells": {}}
    for k, (sec, key, bkey, direction) in FAMILY.items():
        c = cell(cm, sec, key, bkey, direction)
        o = old31[pair]["cells"][k]
        c["exp031_old_parser"] = {"difference": o["difference"], "p_value": o["p_value"],
                                  "significant_uncorrected": o["significant_uncorrected"],
                                  "holm_adjusted_p": o["holm_adjusted_p"]}
        family[pair]["cells"][k] = c
        cells.append(c)
assert len(cells) == 64
adj = holm([c["p_value"] for c in cells])
for c, a in zip(cells, adj):
    c["holm_adjusted_p"] = a
    c["significant_holm"] = a < ALPHA

it = [(p, k, c) for p in family for k, c in family[p]["cells"].items()]
uncorr_sig = [f"{p}:{k}" for p, k, c in it if c["significant_uncorrected"]]
holm_sig = [f"{p}:{k}" for p, k, c in it if c["significant_holm"]]
disagree = [f"{p}:{k}" for p, k, c in it if c.get("bootstrap_vs_t_disagree")]
verdict_changes = [{"cell": f"{p}:{k}", "exp031": c["exp031_old_parser"]["significant_uncorrected"],
                    "exp034": c["significant_uncorrected"], "exp031_p": c["exp031_old_parser"]["p_value"],
                    "exp034_p": c["p_value"]}
                   for p, k, c in it if c["significant_uncorrected"] != c["exp031_old_parser"]["significant_uncorrected"]]
min_cell = min(it, key=lambda t: t[2]["holm_adjusted_p"])

# identity / blocker check: size2000 re-parsed vs exp-032's re-parsed exp-020
e20 = load(f"{RUNS}/exp-032/exp-020/eval/eval_metrics.json")
s2 = load(f"{R}/size2000/eval/eval_metrics.json")
skip = ("git", "predictions_path", "run_dir", "timestamp", "generated_at", "paths", "inputs", "elapsed")
na = {k: v for k, v in numeric_leaves({k: v for k, v in s2.items() if k not in skip}).items()}
nb = {k: v for k, v in numeric_leaves({k: v for k, v in e20.items() if k not in skip}).items()}
common = sorted(set(na) & set(nb))
mism = [k for k in common if not math.isclose(na[k], nb[k], rel_tol=0, abs_tol=1e-12)]
ra = {r["query_id"]: r for r in load(f"{R}/size2000/reparse/results.json")}
rb = {r["query_id"]: r for r in load(f"{RUNS}/exp-032/exp-020/reparse/results.json")}
rp2 = load(f"{R}/size2000/reparse/reparse_metrics.json")
rp20 = load(f"{RUNS}/exp-032/exp-020/reparse/reparse_metrics.json")
identity = {
    "reference": f"{RUNS}/exp-032/exp-020/eval/eval_metrics.json",
    "numeric_values_compared": len(common), "mismatches": mism,
    "only_in_size2000": sorted(set(na) - set(nb)), "only_in_exp020": sorted(set(nb) - set(na)),
    "family_terms": {k: {"size2000_reparsed": s2[k], "exp032_exp020_reparsed": e20[k]} for k in FAMILY},
    "reparsed_results_same_query_ids": set(ra) == set(rb),
    "reparsed_rows_decomposition_differs": sum(ra[q]["decomposition"] != rb[q]["decomposition"] for q in ra),
    "reparse_rows_changed_size2000_vs_exp020": [rp2["rows_changed"], rp20["rows_changed"]],
    "reparse_steps_removed_size2000_vs_exp020": [rp2["pred_steps_removed_total"], rp20["pred_steps_removed_total"]],
}
identity["passed"] = (not mism and identity["reparsed_results_same_query_ids"]
                      and identity["reparsed_rows_decomposition_differs"] == 0
                      and rp2["rows_changed"] == rp20["rows_changed"])

rcp = f"{R}/compare/size2000_vs_exp020_reparsed/compare_metrics.json"
rcm = load(rcp)
repro = {"path": rcp, "system_a_path": rcm["system_a_path"], "system_b_path": rcm["system_b_path"],
         "num_aligned_items": rcm["num_aligned_items"],
         "differences": {k: rcm[FAMILY[k][0]][FAMILY[k][1]]["difference"] for k in FAMILY},
         "significant_uncorrected_any_section": sorted(
             f"{sec}.{k}" for sec in ("bootstrap", "mcnemar", "t_test")
             for k, v in rcm[sec].items() if v.get("significant"))}

out = {
    "exp_id": "exp-034", "issue": 51, "adr": "0039 (items 2-4), 0036 (Holm), 0035",
    "code_commit": run_checks["size2000"]["eval_git_commit"],
    "parser": "new (PR #61 scaffold-label truncation), via scripts/reparse_decompositions.py",
    "model": "none loaded (stored exp-031 predictions of mistralai/Mistral-7B-Instruct-v0.3, 4-bit)",
    "eval_set": "MuSiQue ADR 0007 pinned 600 (200/hop)", "seed": 42, "sizes": SIZES,
    "run_checks": run_checks, "per_size": per_size,
    "per_size_note": "descriptive own-eval values, no direction asserted; beside-card terms never ranked or tallied (ADR 0039 item 2)",
    "identity_check_size2000_vs_exp032_exp020": identity,
    "holm_family": {
        "definition": "exp-031's pre-fixed family (experiments/exp-031/config.json compare.holm_family): 8 pairs sizeN vs size2000 x 8 metrics, overall only",
        "family_size": len(cells), "alpha": ALPHA, "pairs": family,
        "count_significant_uncorrected": len(uncorr_sig), "significant_uncorrected": uncorr_sig,
        "count_significant_holm": len(holm_sig), "significant_holm": holm_sig,
        "min_holm_adjusted_p": {"cell": f"{min_cell[0]}:{min_cell[1]}", "value": min_cell[2]["holm_adjusted_p"]},
        "bootstrap_vs_t_disagreements": disagree,
        "uncorrected_verdict_changes_vs_exp031": verdict_changes,
    },
    "reproducibility_pair_size2000_vs_exp020_reparsed": repro,
}
OUT.write_text(json.dumps(out, indent=1) + "\n")
print("identity passed", identity["passed"], "compared", len(common), "mismatches", mism)
print("changed rows", {s: run_checks[s]["reparse_rows_changed"] for s in run_checks})
print("uncorrected sig", len(uncorr_sig), uncorr_sig)
print("holm sig", len(holm_sig), holm_sig, "min", out["holm_family"]["min_holm_adjusted_p"])
print("disagree", disagree)
print("verdict changes", verdict_changes)
