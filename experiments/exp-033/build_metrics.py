"""Build experiments/exp-033/metrics.json from the exp-033 run outputs (read-only).

Analysis plan (a)-(d) as fixed in experiments/exp-033/notes.md before the run.
Usage: python experiments/exp-033/build_metrics.py  (paths are absolute below)
"""
import glob
import json
import statistics

RUNS = "/cta/users/fyilmaz/Thesis---QAv2/runs"
OUT = "/cta/users/fyilmaz/Thesis---QAv2/experiments/exp-033/metrics.json"

# Report card (ADR 0029 item 1): metrics key -> (compare section, compare key, direction)
CARD = {
    "break_exact_match_rate": ("mcnemar", "break_exact_match", "higher_is_better"),
    "sari_macro": ("bootstrap", "sari", "higher_is_better"),
    "ged_macro": ("bootstrap", "ged", "lower_is_better"),
    "chain_validity_macro": ("bootstrap", "chain_validity", "higher_is_better"),
    "hop_count_exact_match_rate": ("mcnemar", "hop_count_exact_match", "higher_is_better"),
    "under_decomposition_rate": ("mcnemar", "under_decomposition", "lower_is_better"),
    "over_decomposition_rate": ("mcnemar", "over_decomposition", "lower_is_better"),
}
BESIDE = ["step_f1_soft_050_macro", "step_f1_soft_060_macro", "plan_connectedness_macro"]


def load(p):
    with open(p) as f:
        return json.load(f)


def seed_decomposer_dir(s):
    (d,) = glob.glob(f"{RUNS}/exp-033/seed{s}/decomposer/*/")
    return d


# Per-ordering eval files. Report-card values of exp-021/exp-024 come from their own committed
# evals; exp-032 re-scored both at 5c4e442 with 0 changed rows and identical report-card values,
# and is the only source of their beside-card terms (the original evals predate them).
ORDER = {
    "as_is (exp-021)": {"card": f"{RUNS}/exp-021/eval/eval_metrics.json",
                         "beside": f"{RUNS}/exp-032/exp-021/eval/eval_metrics.json"},
    "reversed (exp-024)": {"card": f"{RUNS}/exp-024/eval/eval_metrics.json",
                            "beside": f"{RUNS}/exp-032/exp-024/eval/eval_metrics.json"},
}
for s in (1, 2, 3):
    p = f"{RUNS}/exp-033/seed{s}/eval/eval_metrics.json"
    ORDER[f"shuffled seed {s} (exp-033)"] = {"card": p, "beside": p}

per_ordering = {}
for name, src in ORDER.items():
    card, beside = load(src["card"]), load(src["beside"])
    row = {k: card[k] for k in CARD}
    # consistency: where both sources exist, report-card values must agree exactly
    for k in CARD:
        assert beside[k] == card[k], (name, k)
    for k in BESIDE:
        row[k] = beside[k]
    row["plan_connectedness_null_items"] = beside["plan_connectedness_null_items"]
    row["_sources"] = src
    per_ordering[name] = row


def spread(vals):
    return {
        "mean": statistics.mean(vals),
        "sd_sample_n_minus_1": statistics.stdev(vals),
        "min": min(vals),
        "max": max(vals),
        "range": max(vals) - min(vals),
        "n_orderings": len(vals),
    }


spread_table = {k: spread([per_ordering[o][k] for o in per_ordering]) for k in list(CARD) + BESIDE}

# (a) compares
compares = {}
for s in (1, 2, 3):
    m = load(f"{RUNS}/exp-033/compare/seed{s}_vs_exp021_as_is/compare_metrics.json")
    rows = {}
    for sec in ("bootstrap", "mcnemar", "t_test"):
        for k, v in m[sec].items():
            rows[f"{sec}.{k}"] = {kk: v.get(kk) for kk in
                                  ("difference", "ci_low", "ci_high", "p_value", "significant", "favours")}
    rows_sig = sorted(k for k, v in rows.items() if v["significant"])
    compares[f"seed{s}_vs_exp021_as_is"] = {
        "path": f"{RUNS}/exp-033/compare/seed{s}_vs_exp021_as_is/compare_metrics.json",
        "difference_direction": m["difference_direction"],
        "num_aligned_items": m["num_aligned_items"],
        "tests_reported": m["tests_reported"],
        "beside_card_compared": False,
        "beside_card_note": "not compared: exp-021's per-item file (runs/exp-021/eval) predates the beside-card columns; plan (a) names that file",
        "significant_tests_uncorrected": rows_sig,
        "rows": rows,
    }

# (c) family / size differences (exp-032 re-parsed compares, ADR 0039 cross-parser rule)
REF = {
    "family_exp021_minus_exp020": f"{RUNS}/exp-032/compare/exp-021_vs_exp-020/compare_metrics.json",
    "size_exp022_minus_exp021": f"{RUNS}/exp-032/compare/exp-022_vs_exp-021/compare_metrics.json",
}


def classify(rng, diff):
    # Descriptive convention (stated in notes.md): ordering range vs |difference|.
    d = abs(diff)
    if d == 0:
        return "difference is 0"
    if rng < 0.5 * d:
        return "smaller"
    if rng > 2 * d:
        return "larger"
    return "comparable"


comparison_c = {}
for k, (sec, ck, direction) in CARD.items():
    entry = {"direction": direction, "ordering_spread": spread_table[k]}
    for label, p in REF.items():
        r = load(p)[sec][ck]
        a = r.get("system_a", r.get("system_a_rate"))
        b = r.get("system_b", r.get("system_b_rate"))
        entry[label] = {
            "system_a": a, "system_b": b, "difference": r["difference"],
            "abs_difference": abs(r["difference"]),
            "test": sec, "p_value": r.get("p_value"), "ci_low": r.get("ci_low"),
            "ci_high": r.get("ci_high"), "significant": r["significant"],
            "ordering_range_over_abs_difference": (spread_table[k]["range"] / abs(r["difference"])
                                                   if r["difference"] else None),
            "ordering_sd_over_abs_difference": (spread_table[k]["sd_sample_n_minus_1"] / abs(r["difference"])
                                                if r["difference"] else None),
            "ordering_range_vs_difference": classify(spread_table[k]["range"], r["difference"]),
        }
    comparison_c[k] = entry

# (d) coincidences + run checks
runs_check, coincidences = {}, {}
for s in (1, 2, 3):
    d = seed_decomposer_dir(s)
    c, m, r = load(d + "config.json"), load(d + "metrics.json"), load(d + "results.json")
    ev = load(f"{RUNS}/exp-033/seed{s}/eval/eval_metrics.json")
    runs_check[f"seed{s}"] = {
        "decomposer_dir": d,
        "few_shot_ordering": m["few_shot_ordering"],
        "few_shot_ordering_seed": m["few_shot_ordering_seed"],
        "config_few_shot_ordering": c["few_shot_ordering"],
        "config_few_shot_ordering_seed": c["few_shot_ordering_seed"],
        "rows": len(r),
        "rows_with_empty_decomposition": m["rows_with_empty_decomposition"],
        "eval_total_evaluated": ev["total_evaluated"],
        "eval_missing_gold_count": ev["missing_gold_count"],
    }
    coincidences[f"seed{s}"] = m["few_shot_shuffle_coincidences"]

out = {
    "exp_id": "exp-033",
    "issue": 60,
    "adr": "0039 item 5 (H4)",
    "code_commit": "5c4e442aee149561a4bb3c7a8b21045016fc2284",
    "model": "Qwen/Qwen2.5-7B-Instruct 4-bit NF4 (one model; transfer to other models is an assumption)",
    "eval_set": "MuSiQue ADR 0007 pinned 600 (200/hop)",
    "orderings": list(per_ordering),
    "run_checks": runs_check,
    "per_ordering": per_ordering,
    "b_spread_table": spread_table,
    "b_spread_note": "SD is the sample SD (n-1) over the 5 orderings; beside-card rows are descriptive only (no ranking, no k-of-n tally)",
    "a_compares": compares,
    "c_spread_vs_family_and_size": comparison_c,
    "c_classification_rule": "ordering range (max-min over 5 orderings) vs |difference|: 'smaller' if range < 0.5*|diff|, 'larger' if range > 2*|diff|, else 'comparable'. A descriptive convention chosen by the runner at analysis time, not part of the pre-fixed plan; the raw ratios are given beside it.",
    "c_sources": REF,
    "d_few_shot_shuffle_coincidences": coincidences,
}
with open(OUT, "w") as f:
    json.dump(out, f, indent=1)
print("wrote", OUT)
