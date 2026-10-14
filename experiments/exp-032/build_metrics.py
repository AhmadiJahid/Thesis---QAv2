#!/usr/bin/env python3
"""Assemble experiments/exp-032/metrics.json from runs/exp-032/ (issue #60, ADR 0039).

Reads only evaluator / reparse outputs already on disk; computes nothing new except the
old-vs-new differences and the verdict-change list. No ranking or "k of n" tally is taken
from the beside-card terms (ADR 0039 item 2).
"""
import json, math, sys
from pathlib import Path

M = Path("/cta/users/fyilmaz/Thesis---QAv2")
R = M / "runs/exp-032"
ARMS = ["exp-008", "exp-015", "exp-020", "exp-021", "exp-022", "exp-023", "exp-024",
        "exp-025", "exp-026", "exp-027", "exp-028"]
CHANGED = ["exp-015", "exp-020", "exp-023"]
CARD = ["break_exact_match_rate", "sari_macro", "ged_macro", "chain_validity_macro",
        "hop_count_exact_match_rate", "under_decomposition_rate", "over_decomposition_rate"]
BESIDE = ["step_f1_soft_050_macro", "step_f1_soft_060_macro", "plan_connectedness_macro",
          "plan_connectedness_n_items", "plan_connectedness_null_items",
          "gold_plan_connectedness_items_at_1", "gold_plan_connectedness_items_below_1",
          "gold_plan_connectedness_null_items"]
# report-card term -> (headline test block, statistic name)  [ADR 0009 headline: bootstrap + McNemar]
CARD_TESTS = {"break_exact_match_rate": ("mcnemar", "break_exact_match"), "sari_macro": ("bootstrap", "sari"),
              "ged_macro": ("bootstrap", "ged"), "chain_validity_macro": ("bootstrap", "chain_validity"),
              "hop_count_exact_match_rate": ("mcnemar", "hop_count_exact_match"),
              "under_decomposition_rate": ("mcnemar", "under_decomposition"),
              "over_decomposition_rate": ("mcnemar", "over_decomposition"), "step_f1_macro": ("bootstrap", "step_f1")}
T_NAME = {k: v[1] for k, v in CARD_TESTS.items()}
PAIRS = [("015", "008"), ("020", "008"), ("020", "015"), ("021", "008"), ("021", "020"), ("022", "008"),
         ("022", "020"), ("022", "021"), ("023", "020"), ("024", "021"), ("025", "022"), ("026", "008"),
         ("026", "015"), ("026", "020"), ("027", "008"), ("027", "020"), ("027", "021"), ("027", "022"),
         ("028", "027")]
COMMITTED = {  # pair -> committed compare dirs (relative to runs/), and whether A/B are reversed
    ("015", "008"): [("exp-015/compare_vs_exp008_full_train", False),
                     ("exp-026/arm_comparison/compare_finetuned_full_train_exp008_vs_prompting_fp16_exp015", True)],
    ("020", "008"): [("exp-020/compare_vs_exp008_full_train", False)],
    ("020", "015"): [("exp-020/compare_vs_exp015_unguided_fp16", False)],
    ("021", "008"): [("exp-021/compare_vs_exp008_full_train", False)],
    ("021", "020"): [("exp-021/compare_vs_exp020_mistral_4bit", False)],
    ("022", "008"): [("exp-022/compare_vs_exp008_full_train", False)],
    ("022", "020"): [("exp-022/compare_vs_exp020_mistral_4bit", False)],
    ("022", "021"): [("exp-022/compare_vs_exp021_qwen7b_4bit", False)],
    ("023", "020"): [("exp-023/compare_vs_exp020_as_is", False)],
    ("024", "021"): [("exp-024/compare_vs_exp021_as_is", False)],
    ("025", "022"): [("exp-025/compare_vs_exp022_as_is", False)],
    ("026", "008"): [("exp-026/compare_vs_exp008_full_train", False),
                     ("exp-026/compare_vs_exp008_full_train_rescored", False)],
    ("026", "015"): [("exp-026/compare_vs_exp015_prompting_fp16", False)],
    ("026", "020"): [("exp-026/compare_vs_exp020_prompting_4bit", False)],
    ("027", "008"): [("exp-027/compare_vs_exp008_full_train", False)],
    ("027", "020"): [("exp-027/compare_vs_exp020_mistral_4bit", False)],
    ("027", "021"): [("exp-027/compare_vs_exp021_qwen2_5_7b_4bit", False)],
    ("027", "022"): [("exp-027/compare_vs_exp022_qwen2_5_14b_4bit", False)],
    ("028", "027"): [("exp-028/compare_vs_exp027_as_is", False)],
}


def load(p):
    return json.loads(Path(p).read_text())


def sig(cm, term, which):
    block, stat = CARD_TESTS[term] if which == "headline" else ("t_test", T_NAME[term])
    v = cm.get(block, {}).get(stat)
    if v is None:
        return None
    return {"significant": v.get("significant"), "difference": v.get("difference"),
            "p_value": v.get("p_value"), "ci": [v.get("ci_low"), v.get("ci_high")] if "ci_low" in v else None,
            "favours": v.get("favours")}


out = {"_note": ("exp-032 (issue #60, ADR 0039): CPU re-score of 11 stored arms with the scaffold-label "
                 "parser fix and the beside-card terms, code at 5c4e442. Beside-card terms are reported "
                 "BESIDE the report card: no arm ranking and no 'k of n significant' tally is taken from them "
                 "(ADR 0039 item 2). exp-020 and exp-023 keep their recorded values (ADR 0039 item 4); these "
                 "are the new, re-parsed readings."),
       "code_commit": "5c4e442aee149561a4bb3c7a8b21045016fc2284", "seed": 42,
       "eval_set": "MuSiQue ADR 0007 pinned 600", "per_arm": {}, "identity_check": {},
       "old_vs_new_changed_arms": {}, "compares": {}, "verdict_changes": []}

for a in ARMS:
    em = load(R / a / "eval/eval_metrics.json")
    rp = load(R / a / "reparse/reparse_metrics.json")
    assert em["git"]["commit"].startswith("5c4e442") and not em["git"]["dirty"], a
    row = {"reparse_rows_changed": rp["rows_changed"], "reparse_rows_total": rp["rows_total"],
           "reparse_steps_removed": rp["pred_steps_removed_total"],
           "input_results": rp["input_results"], "input_results_sha256": rp["input_results_sha256"],
           "output_results_sha256": rp["output_results_sha256"],
           "total_evaluated": em["total_evaluated"], "step_f1_macro": em["step_f1_macro"]}
    row.update({k: em[k] for k in BESIDE})
    row["report_card"] = {k: em[k] for k in CARD}
    out["per_arm"][a] = row

# identity check results come from the scratch checker (re-run inline here for self-containment)
sys.path.insert(0, str(Path(__file__).parent))
from identity_check import run as identity_run  # noqa: E402
out["identity_check"] = identity_run()

for a in CHANGED:
    new = load(R / a / "eval/eval_metrics.json")
    stored = load(R / a / "eval_stored_same_code/eval_metrics.json")
    orig = load(M / "runs" / a / "eval/eval_metrics.json")
    terms = {}
    for k in CARD + ["step_f1_macro"]:
        terms[k] = {"committed_original_eval": orig.get(k), "stored_file_same_code": stored.get(k),
                    "reparsed_new": new[k], "new_minus_stored_same_code": new[k] - stored[k]}
    out["old_vs_new_changed_arms"][a] = {
        "note": ("committed_original_eval = runs/<id>/eval/eval_metrics.json as committed; "
                 "stored_file_same_code = the stored (un-reparsed) results.json scored by the 5c4e442 "
                 "evaluator, so new_minus_stored_same_code isolates the parser fix"),
        "terms": terms,
        "beside_card_new": {k: new[k] for k in BESIDE},
        "beside_card_stored_same_code": {k: stored[k] for k in BESIDE}}

for A, B in PAIRS:
    name = f"exp-{A}_vs_exp-{B}"
    cm = load(R / "compare" / name / "compare_metrics.json")
    entry = {"run_dir": f"runs/exp-032/compare/{name}", "num_aligned_items": cm["num_aligned_items"],
             "beside_report_card": {"bootstrap": cm["beside_report_card"]["bootstrap"],
                                    "plan_connectedness_pairing": cm["beside_report_card"].get("plan_connectedness_pairing")},
             "report_card_new": {}, "committed": {}}
    for k in CARD_TESTS:
        entry["report_card_new"][k] = {"headline": sig(cm, k, "headline"), "t_test": sig(cm, k, "t")}
    for d, rev in COMMITTED[(A, B)]:
        oc = load(M / "runs" / d / "compare_metrics.json")
        cv = {}
        for k in CARD_TESTS:
            cv[k] = {"headline": sig(oc, k, "headline"), "t_test": sig(oc, k, "t")}
            for which in ("headline", "t_test"):
                o, n = cv[k][which], entry["report_card_new"][k][which]
                if o is None or n is None:
                    continue
                if bool(o["significant"]) != bool(n["significant"]):
                    out["verdict_changes"].append({"pair": name, "committed_compare": f"runs/{d}",
                                                   "a_b_reversed_in_committed": rev, "term": k, "test": which,
                                                   "committed": "significant" if o["significant"] else "n.s.",
                                                   "new": "significant" if n["significant"] else "n.s.",
                                                   "committed_difference": o["difference"], "new_difference": n["difference"],
                                                   "committed_p": o["p_value"], "new_p": n["p_value"]})
        entry["committed"][f"runs/{d}"] = {"a_b_reversed": rev, "report_card": cv}
    out["compares"][name] = entry

out["verdict_change_scope_check"] = sorted({v["pair"] for v in out["verdict_changes"]})
print(json.dumps({"verdict_changes": len(out["verdict_changes"]), "pairs": out["verdict_change_scope_check"]}))
(Path(__file__).parent / "metrics.json").write_text(json.dumps(out, indent=1) + "\n")
