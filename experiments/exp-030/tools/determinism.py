import json, glob, sys
PIN = {"qwen2_5_7b": "exp-021", "pool_ft": "exp-026", "qwen2_5_14b": "exp-022", "qwen3_5_9b": "exp-027", "mistral_7b": "exp-020"}
K3 = {"2hop__226204_69048", "2hop__263638_69048", "4hop2__161602_474028_88460_21034"}
TIE11 = {"2hop__226204_69048", "2hop__401710_456439", "2hop__115515_779396", "2hop__818692_113442", "2hop__263638_69048", "2hop__76402_653666", "2hop__554167_451128", "2hop__765180_123956", "2hop__323534_810083", "2hop__168816_144857", "4hop2__161602_474028_88460_21034"}
TOP5 = {"2hop__115515_779396", "2hop__554167_451128", "2hop__765180_123956", "2hop__168816_144857"}
out = {}
for arm, exp in PIN.items():
    old_f = glob.glob(f"runs/{exp}/decomposer/*/results.json"); assert len(old_f) == 1, old_f
    new_f = sorted(glob.glob(f"runs/exp-030/{arm}/decomposer/*/results.json"))
    if not new_f: print(arm, "no new results"); continue
    old = {r["query_id"]: r for r in json.load(open(old_f[0]))}
    new = {r["query_id"]: r for r in json.load(open(new_f[-1]))}
    assert set(old) <= set(new), (arm, len(set(old) - set(new)))
    mism = [q for q in old if old[q]["decomposition"] != new[q]["decomposition"]]
    raw_mism = [q for q in old if old[q].get("decomposition_raw") != new[q].get("decomposition_raw")]
    out[arm] = {"pinned_run": exp, "old": old_f[0], "new": new_f[-1], "n": len(old),
                "decomposition_match": len(old) - len(mism), "raw_match": len(old) - len(raw_mism),
                "mismatch_ids": mism, "mismatch_in_k3_tie_set": sorted(set(mism) & K3),
                "mismatch_in_top5_tie_set": sorted(set(mism) & TOP5),
                "mismatch_in_tie11_set": sorted(set(mism) & TIE11), "identical_prompt_rows_match": f"{len(old)-len(TIE11)-len(set(mism)-TIE11)}/{len(old)-len(TIE11)}", "mismatch_unexplained": sorted(set(mism) - TIE11)}
    print(arm, exp, f"{out[arm]['decomposition_match']}/{len(old)} decomposition, {out[arm]['raw_match']}/{len(old)} raw;",
          "mismatches:", mism, "| identical-prompt rows:", out[arm]["identical_prompt_rows_match"], "| outside tie-11:", out[arm]["mismatch_unexplained"])
json.dump(out, open("runs/exp-030/determinism_check.json", "w"), indent=2)
