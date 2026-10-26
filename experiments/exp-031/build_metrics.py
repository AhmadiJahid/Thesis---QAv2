"""Build experiments/exp-031/metrics.json from the exp-031 run outputs (read-only).

Analysis plan as fixed in the exp-031 log row and configs/pool_size_sweep_v2.json
(compare.holm_family) before the run: measurement only, no direction asserted.
Holm-Bonferroni (ADR 0036) over the pre-fixed 64-test family, reported BESIDE the evaluator's
uncorrected verdict. The reproducibility pair (size2000 vs exp-020) is outside the family and
reported uncorrected.
Usage: python3 experiments/exp-031/build_metrics.py  (paths are absolute below)
"""
import glob
import json
import re

ROOT = "/cta/users/fyilmaz/Thesis---QAv2"
RUNS = f"{ROOT}/runs"
OUT = f"{ROOT}/experiments/exp-031/metrics.json"
CFG = f"{ROOT}/configs/pool_size_sweep_v2.json"
ALPHA = 0.05

# Holm family metrics: metrics key -> (p-value section, compare key, bootstrap key or None, direction)
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


def load(p):
    with open(p) as f:
        return json.load(f)


def holm(ps):
    # Same step-down implementation as experiments/exp-029/reanalysis.py.
    m = len(ps)
    order = sorted(range(m), key=lambda k: ps[k])
    adj = [0.0] * m
    running = 0.0
    for rank, k in enumerate(order):
        running = max(running, min(1.0, (m - rank) * ps[k]))
        adj[k] = running
    return adj


cfg = load(CFG)
sizes = cfg["sizes"]
assert cfg["compare"]["holm_family"]["family_size"] == 64

# wall clock per size from run.log
wall = {}
with open(f"{RUNS}/exp-031/run.log") as f:
    for line in f:
        mm = re.search(r"size (\d+) decompose rc=(\d+) wall_s=(\d+) peak_vram_mib=(\d+)", line)
        if mm:
            wall[int(mm.group(1))] = {"decompose_rc": int(mm.group(2)), "decompose_wall_s": int(mm.group(3)),
                                      "peak_vram_mib": int(mm.group(4))}

# per-size run checks and own-eval values
run_checks, per_size = {}, {}
for n in sizes:
    (d,) = glob.glob(f"{RUNS}/exp-031/size{n}/decomposer/*/")
    c, m, r = load(d + "config.json"), load(d + "metrics.json"), load(d + "results.json")
    ev = load(f"{RUNS}/exp-031/size{n}/eval/eval_metrics.json")
    headers = set()
    for p in glob.glob(d + "prompts_log/*.txt"):
        with open(p) as f:
            for line in f:
                if line.startswith("Few-shot source:"):
                    headers.add(line.strip())
                    break
    exp = cfg["expected_inputs"][str(n)]
    run_checks[f"size{n}"] = {
        "decomposer_dir": d,
        "results_rows": len(r),
        "rows_loaded_per_hop": m["rows_loaded_per_hop"],
        "rows_with_empty_decomposition": m["rows_with_empty_decomposition"],
        "rows_at_max_new_tokens": m["rows_at_max_new_tokens"],
        "few_shot_source_counts": m["few_shot_source_counts"],
        "few_shot_ordering": m["few_shot_ordering"],
        "prompts_log_files": len(glob.glob(d + "prompts_log/*.txt")),
        "prompts_log_few_shot_headers": sorted(headers),
        "config_few_shot_k_mislabel_issue_53": c["few_shot"]["k"],
        "retrieval_input_sha256": c["retrieval"]["input_sha256"],
        "retrieval_input_sha256_matches_expected": c["retrieval"]["input_sha256"] == exp["rerank_top5_sha256"],
        "quantization": c["quantization"],
        "condition": c["condition"],
        "seed": c["seed"],
        "model_id": c["model_id"],
        "decomposer_git": m["git"],
        "eval_git_commit": ev["git"]["commit"],
        "eval_git_dirty": ev["git"].get("dirty"),
        "eval_total_evaluated": ev["total_evaluated"],
        "eval_missing_gold_count": ev["missing_gold_count"],
        **wall.get(n, {}),
    }
    per_size[f"size{n}"] = {k: ev[k] for k in FAMILY}
    per_size[f"size{n}"]["pool_rows"] = exp["pool_rows"]
    per_size[f"size{n}"]["pool_hop_mix"] = exp["pool_hop_mix"]


def cell(cm, sec, key, bkey, direction):
    r = cm[sec][key]
    out = {
        "test": "exact McNemar" if sec == "mcnemar" else "paired t-test",
        "system_a": r.get("system_a", r.get("system_a_rate")),
        "system_b": r.get("system_b", r.get("system_b_rate")),
        "difference": r["difference"],
        "direction": direction,
        "p_value": r["p_value"],
        "significant_uncorrected": r["significant"],
        "favours_uncorrected": r["favours"],
    }
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


# the 64-test family
family, cells = {}, []
for n in sizes[1:]:
    pair = f"size{n}_vs_size2000"
    cm = load(f"{RUNS}/exp-031/compare/{pair}/compare_metrics.json")
    assert cm["num_aligned_items"] == 600
    assert cm["system_b_path"].endswith("exp-031/size2000/eval/eval_per_item.json")
    family[pair] = {"path": f"{RUNS}/exp-031/compare/{pair}/compare_metrics.json",
                    "difference_direction": cm["difference_direction"],
                    "num_aligned_items": cm["num_aligned_items"],
                    "git_commit": cm["git"]["commit"], "cells": {}}
    for k, (sec, key, bkey, direction) in FAMILY.items():
        c = cell(cm, sec, key, bkey, direction)
        family[pair]["cells"][k] = c
        cells.append(c)
assert len(cells) == 64
adj = holm([c["p_value"] for c in cells])
for c, a in zip(cells, adj):
    c["holm_adjusted_p"] = a
    c["significant_holm"] = a < ALPHA

uncorr_sig = [(p, k) for p in family for k, c in family[p]["cells"].items() if c["significant_uncorrected"]]
holm_sig = [(p, k) for p in family for k, c in family[p]["cells"].items() if c["significant_holm"]]
disagree = [(p, k) for p in family for k, c in family[p]["cells"].items() if c.get("bootstrap_vs_t_disagree")]

# reproducibility pair (outside the family, uncorrected)
rp = f"{RUNS}/exp-031/compare/size2000_vs_exp020/compare_metrics.json"
rcm = load(rp)
repro = {"path": rp, "system_a_path": rcm["system_a_path"], "system_b_path": rcm["system_b_path"],
         "num_aligned_items": rcm["num_aligned_items"], "difference_direction": rcm["difference_direction"],
         "git_commit": rcm["git"]["commit"], "cells": {}}
for k, (sec, key, bkey, direction) in FAMILY.items():
    repro["cells"][k] = cell(rcm, sec, key, bkey, direction)
repro["significant_uncorrected_any_section"] = sorted(
    f"{sec}.{k}" for sec in ("bootstrap", "mcnemar", "t_test") for k, v in rcm[sec].items() if v["significant"])
e20 = load(f"{RUNS}/exp-020/eval/eval_metrics.json")
s2 = load(f"{RUNS}/exp-031/size2000/eval/eval_metrics.json")
repro["own_eval_identical_to_exp020"] = {k: s2[k] == e20[k] for k in FAMILY}
(ra,) = glob.glob(f"{RUNS}/exp-031/size2000/decomposer/*/results.json")
(rb,) = glob.glob(f"{RUNS}/exp-020/decomposer/*/results.json")
da = {r["query_id"]: r for r in load(ra)}
db = {r["query_id"]: r for r in load(rb)}
repro["results_json"] = {
    "size2000": ra, "exp020": rb,
    "same_query_id_set": set(da) == set(db), "rows": len(da),
    "rows_decomposition_raw_differs": sum(da[k]["decomposition_raw"] != db[k]["decomposition_raw"] for k in da),
    "rows_decomposition_differs": sum(da[k]["decomposition"] != db[k]["decomposition"] for k in da),
}

out = {
    "exp_id": "exp-031",
    "issue": 51,
    "adr": "0035 (+ 2026-10-14 amendment), 0036",
    "code_commit": "3bd3595e48ca8976e1748e566eaadfc03f3ae967",
    "parser": "old (pre-PR #61 scaffold-label fix); ADR 0039: exp-031 is an old-parser run compared only with old-parser runs (its own size2000 and exp-020's stored eval)",
    "model": "mistralai/Mistral-7B-Instruct-v0.3, 4-bit",
    "eval_set": "MuSiQue ADR 0007 pinned 600 (200/hop)",
    "few_shot_exemplars_in_prompt": "5 (prompts_log header 'Few-shot source: reranked (k=5)' on every logged prompt of every size; config.json few_shot.k=3 is the issue #53 mislabel)",
    "sizes": sizes,
    "chain": {"status_file": f"{RUNS}/exp-031/chain_status.txt", "result": "chain done CFAIL=0 2026-10-14T18:50:18+0300",
              "total_wall_s": 21339, "lock": "acquired 2026-10-14T12:54:39+0300, released by the waiter at exit rc=0"},
    "run_checks": run_checks,
    "per_size": per_size,
    "per_size_note": "own-eval values from runs/exp-031/sizeN/eval/eval_metrics.json; descriptive, no direction asserted",
    "holm_family": {
        "definition": cfg["compare"]["holm_family"],
        "family_size": len(cells),
        "alpha": ALPHA,
        "pairs": family,
        "count_significant_uncorrected": len(uncorr_sig),
        "significant_uncorrected": [f"{p}:{k}" for p, k in uncorr_sig],
        "count_significant_holm": len(holm_sig),
        "significant_holm": [f"{p}:{k}" for p, k in holm_sig],
        "bootstrap_vs_t_disagreements": [f"{p}:{k}" for p, k in disagree],
    },
    "reproducibility_pair_size2000_vs_exp020": repro,
}
with open(OUT, "w") as f:
    json.dump(out, f, indent=1)
print("wrote", OUT)
print("uncorrected sig", len(uncorr_sig), uncorr_sig)
print("holm sig", len(holm_sig), holm_sig)
print("disagree", disagree)
