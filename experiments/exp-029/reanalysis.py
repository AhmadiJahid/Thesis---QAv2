"""exp-029 — per-hop and end-to-end significance re-analysis (ADR 0035 item 5; Refs #50, #16).

Re-analysis, not a run: reads existing per-item files under runs/, imports the evaluator's own
test functions UNCHANGED (_paired_bootstrap / _statistics_for, _mcnemar, _paired_t_test_row,
_significance_floor) and calls them on hop-stratified, id-aligned rows. The only arithmetic
added here is Holm-Bonferroni over the p-values those functions emit.

Usage (repo root):  .venv/bin/python experiments/exp-029/reanalysis.py
Writes experiments/exp-029/config.json and experiments/exp-029/metrics.json.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import musique_decompositions_evaluator as E  # noqa: E402

OUT = Path(__file__).resolve().parent
CFG = json.loads((REPO / "configs" / "musique_eval.json").read_text())
PC = CFG["paired_comparison"]
SEED = int(CFG["seed"])
ITER = int(PC["bootstrap_iterations"])
CHUNK = int(PC["bootstrap_chunk_size"])
ALPHA = float(PC["alpha"])
MIN_ITEMS = int(PC["min_items_for_significance_claim"])
WEIGHTS = CFG["composite_score_weights"]
SCALE = float(CFG["composite_step_count_error_scale"])

# Full battery = everything the evaluator compares, minus the legacy composite (ADR 0035 item 4).
BOOT_ALL = tuple(s for s in E.BOOTSTRAP_STATISTICS if s != "composite_score")
MCN_ALL = tuple(E.MCNEMAR_STATISTICS)

F = {
    # A7 — exp-009, re-scored with the suite columns at exp-016 (additive re-score)
    "exp009_mixed": "runs/exp-016/exp009_mixed/eval_per_item.json",
    "exp009_oracle_hop_matched": "runs/exp-016/exp009_oracle_hop_matched/eval_per_item.json",
    # B1 — exp-004 (Mistral-7B fp16) / exp-005 (Qwen3.5-9B 4-bit), exp-016 re-scores
    "exp004_unguided": "runs/exp-016/exp004_unguided/eval_per_item.json",
    "exp004_oracle_guided": "runs/exp-016/exp004_oracle_guided/eval_per_item.json",
    "exp005_unguided": "runs/exp-016/exp005_unguided/eval_per_item.json",
    "exp005_oracle_guided": "runs/exp-016/exp005_oracle_guided/eval_per_item.json",
    # D2 — 4-bit prompting arms and the two fine-tunes
    "exp020_mistral7b_4bit": "runs/exp-020/eval/eval_per_item.json",
    "exp021_qwen25_7b_4bit": "runs/exp-021/eval/eval_per_item.json",
    "exp022_qwen25_14b_4bit": "runs/exp-022/eval/eval_per_item.json",
    "exp027_qwen35_9b_4bit": "runs/exp-027/eval/eval_per_item.json",
    "exp008_full_train": "runs/exp-016/exp008_full_train/eval_per_item.json",
    "exp026_pool_only_ft": "runs/exp-026/eval/eval_per_item.json",
    # D3 — exp-013 generalisation fine-tune (hop=4 slice written by exp-013 itself)
    "exp013_gen_ft_hop4": "runs/exp-013/eval/eval_per_item_hop4_only.json",
    # E1 — exp-017 end-to-end answers
    "exp017_oracle": "runs/exp-017/oracle/20260823_070959/answer_per_item.json",
    "exp017_full_train": "runs/exp-017/full_train/20260823_073434/answer_per_item.json",
    "exp017_unguided": "runs/exp-017/unguided/20260823_075723/answer_per_item.json",
    "exp017_no_decomposition_control": "runs/exp-017/no_decomposition_control/20260823_082452/answer_per_item.json",
}
# Files read only to prove the re-scored copies above carry the same predictions/scores.
XCHECK = {
    "exp011_exp009_mixed": "runs/exp-011/exp009_mixed/eval_per_item.json",
    "exp011_exp009_oracle_hop_matched": "runs/exp-011/exp009_oracle_hop_matched/eval_per_item.json",
    "exp011_exp004_unguided": "runs/exp-011/exp004_unguided/eval_per_item.json",
    "exp011_exp004_oracle_guided": "runs/exp-011/exp004_oracle_guided/eval_per_item.json",
    "exp011_exp005_unguided": "runs/exp-011/exp005_unguided/eval_per_item.json",
    "exp011_exp005_oracle_guided": "runs/exp-011/exp005_oracle_guided/eval_per_item.json",
    "exp011_exp008_full_train": "runs/exp-011/exp008_full_train/eval_per_item.json",
    "exp026_arm_comparison_pool_only": "runs/exp-026/arm_comparison/finetuned_pool_2000_imbalanced_exp026/eval_per_item.json",
    "exp013_baseline_unguided_4hop": "runs/exp-013/baseline_unguided_4hop_per_item.json",
    "exp013_baseline_exp008_4hop": "runs/exp-013/baseline_exp008_4hop_per_item.json",
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(key: str, table=F) -> dict[str, dict]:
    d = json.loads((REPO / table[key]).read_text())
    items = d["items"]
    by = {str(r["item_id"]): r for r in items}
    assert len(by) == len(items), f"duplicate ids in {key}"
    return by


def assert_same_scores(a: dict, b: dict, label: str, cols=None) -> dict:
    """Same id set; per item, every shared metric column and pred_steps identical."""
    assert set(a) == set(b), f"{label}: id sets differ"
    first = next(iter(a.values()))
    shared = cols or [c for c in first if c in next(iter(b.values()))
                      and c not in ("ged_fallback_seconds",)]
    for i in a:
        for c in shared:
            assert a[i][c] == b[i][c], f"{label}: {i} column {c} differs"
    return {"pair": label, "ids": len(a), "columns_checked": sorted(shared), "identical": True}


def holm(ps: list[float]) -> list[float]:
    m = len(ps)
    order = sorted(range(m), key=lambda k: ps[k])
    adj = [0.0] * m
    running = 0.0
    for rank, k in enumerate(order):
        running = max(running, min(1.0, (m - rank) * ps[k]))
        adj[k] = running
    return adj


def compare(rows_a: list[dict], rows_b: list[dict], boot: tuple, mcn: tuple) -> dict:
    n = len(rows_a)
    floor = E._significance_floor(n, MIN_ITEMS)
    under = bool(floor["below_min_items"])
    arr_a = {s: np.array([float(r[s]) for r in rows_a]) for s in boot}
    arr_b = {s: np.array([float(r[s]) for r in rows_b]) for s in boot}
    out = {"n": n, "significance_floor": floor, "bootstrap": {}, "mcnemar": {}, "t_test": {}}
    if boot:
        out["bootstrap"] = E._paired_bootstrap(arr_a, arr_b, n=n, iterations=ITER, alpha=ALPHA,
                                               seed=SEED, weights=WEIGHTS, scale=SCALE,
                                               chunk_size=CHUNK, underpowered=under,
                                               statistics=boot)
        for s in boot:
            out["t_test"][s] = E._paired_t_test_row(arr_a[s], arr_b[s], ALPHA, under, name=s)
    if mcn:
        out["mcnemar"] = E._mcnemar(rows_a, rows_b, ALPHA, under, statistics=mcn)
    return out


def available(rows_a, rows_b, names):
    return tuple(s for s in names if all(s in r for r in rows_a) and all(s in r for r in rows_b))


def run_pair(a_key, b_key, strata, a=None, b=None, boot=BOOT_ALL, mcn=MCN_ALL):
    a = a if a is not None else load(a_key)
    b = b if b is not None else load(b_key)
    if set(a) != set(b):
        raise SystemExit(f"{a_key} vs {b_key}: id sets differ ({len(a)} vs {len(b)}) — refusing to pair")
    ids = sorted(a)
    for i in ids:
        assert a[i]["gold_hop_count"] == b[i]["gold_hop_count"], f"{i}: gold hop differs"
        if "question" in a[i] and "question" in b[i]:
            assert a[i]["question"] == b[i]["question"], f"{i}: question text differs"
    res = {"system_a": a_key, "system_b": b_key, "ids_identical": True, "n_total": len(ids),
           "difference_direction": "system_a minus system_b", "strata": {}}
    for st in strata:
        sel = ids if st == "overall" else [i for i in ids if int(a[i]["gold_hop_count"]) == int(st)]
        ra, rb = [a[i] for i in sel], [b[i] for i in sel]
        bs = available(ra, rb, boot)
        ms = available(ra, rb, mcn)
        r = compare(ra, rb, bs, ms)
        r["statistics_not_available_in_inputs"] = sorted(set(boot + mcn) - set(bs + ms))
        res["strata"][str(st)] = r
    return res


def main() -> None:
    HOPS = ("2", "3", "4")
    ALL = ("overall",) + HOPS

    # ---- provenance cross-checks (assert, then record) ----
    xc = []
    for k16, k11 in [("exp009_mixed", "exp011_exp009_mixed"),
                     ("exp009_oracle_hop_matched", "exp011_exp009_oracle_hop_matched"),
                     ("exp004_unguided", "exp011_exp004_unguided"),
                     ("exp004_oracle_guided", "exp011_exp004_oracle_guided"),
                     ("exp005_unguided", "exp011_exp005_unguided"),
                     ("exp005_oracle_guided", "exp011_exp005_oracle_guided"),
                     ("exp008_full_train", "exp011_exp008_full_train")]:
        xc.append(assert_same_scores(load(k16), load(k11, XCHECK), f"{F[k16]} == {XCHECK[k11]}"))
    xc.append(assert_same_scores(load("exp026_pool_only_ft"), load("exp026_arm_comparison_pool_only", XCHECK),
                                 f"{F['exp026_pool_only_ft']} == {XCHECK['exp026_arm_comparison_pool_only']}"))
    g4 = load("exp013_gen_ft_hop4")
    ung4 = {i: r for i, r in load("exp004_unguided").items() if int(r["gold_hop_count"]) == 4}
    ft4 = {i: r for i, r in load("exp008_full_train").items() if int(r["gold_hop_count"]) == 4}
    xc.append(assert_same_scores(ung4, load("exp013_baseline_unguided_4hop", XCHECK),
                                 "exp-016 exp004_unguided hop=4 == exp-013 baseline_unguided_4hop"))
    xc.append(assert_same_scores(ft4, load("exp013_baseline_exp008_4hop", XCHECK),
                                 "exp-016 exp008_full_train hop=4 == exp-013 baseline_exp008_4hop"))

    pairs = {}
    # A7
    pairs["A7|exp009_mixed_vs_oracle_hop_matched"] = run_pair("exp009_mixed", "exp009_oracle_hop_matched", ALL)
    # B1
    pairs["B1|mistral7b_exp004_unguided_vs_oracle_guided"] = run_pair("exp004_unguided", "exp004_oracle_guided", ALL)
    pairs["B1|qwen35_9b_exp005_unguided_vs_oracle_guided"] = run_pair("exp005_unguided", "exp005_oracle_guided", ALL)
    # D2: prompting (a) vs fine-tune (b), per hop
    for p in ("exp020_mistral7b_4bit", "exp021_qwen25_7b_4bit", "exp022_qwen25_14b_4bit", "exp027_qwen35_9b_4bit"):
        for ft in ("exp008_full_train", "exp026_pool_only_ft"):
            pairs[f"D2|{p}_vs_{ft}"] = run_pair(p, ft, ALL)
    # D3: generalisation FT (a) vs baselines (b), hop=4 only
    pairs["D3|exp013_gen_ft_vs_exp004_unguided_hop4"] = run_pair("exp013_gen_ft_hop4", "exp004_unguided_hop4", ("4",), a=g4, b=ung4)
    pairs["D3|exp013_gen_ft_vs_exp008_full_train_hop4"] = run_pair("exp013_gen_ft_hop4", "exp008_full_train_hop4", ("4",), a=g4, b=ft4)
    # E1: all 6 pairs of the 4 sources, higher-ranked source as system_a
    src = ("exp017_oracle", "exp017_full_train", "exp017_unguided", "exp017_no_decomposition_control")
    for i in range(len(src)):
        for j in range(i + 1, len(src)):
            pairs[f"E1|{src[i]}_vs_{src[j]}"] = run_pair(src[i], src[j], ALL,
                                                         boot=("answer_f1",), mcn=("answer_em",))

    # ---- Holm families: one per slide, the metrics and strata the slide's claim rests on ----
    B1_METRICS = ("sari", "ged", "chain_validity", "step_f1", "ordered_step_accuracy", "rouge_l_f1",
                  "exact_match", "hop_count_exact_match", "break_exact_match")
    families = {
        "A7": {"metrics": ("step_f1", "ordered_step_accuracy", "hop_count_exact_match", "exact_match"), "strata": ALL},
        "B1": {"metrics": B1_METRICS, "strata": ALL},
        "D2": {"metrics": ("step_f1", "under_decomposition"), "strata": HOPS},
        "D3": {"metrics": ("chain_validity", "under_decomposition", "hop_count_exact_match", "step_f1"), "strata": ("4",)},
        "E1": {"metrics": ("answer_em", "answer_f1"), "strata": ALL},
    }
    fam_out = {}
    for fam, spec in families.items():
        members = []
        for pk, pr in pairs.items():
            if not pk.startswith(fam + "|"):
                continue
            for st in spec["strata"]:
                r = pr["strata"][st]
                for m in spec["metrics"]:
                    if m in r["mcnemar"]:
                        p, basis = r["mcnemar"][m]["p_value"], "mcnemar_exact"
                    elif m in r["t_test"]:
                        p, basis = r["t_test"][m]["p_value"], "paired_t (bootstrap emits no p)"
                    else:
                        raise SystemExit(f"{fam} {pk} {st} {m}: not available")
                    members.append({"pair": pk, "stratum": st, "metric": m, "p": p, "p_basis": basis})
        adj = holm([x["p"] for x in members])
        for x, a in zip(members, adj):
            x["holm_adjusted_p"] = a
            x["holm_significant"] = bool(a < ALPHA)
            r = pairs[x["pair"]]["strata"][x["stratum"]]
            row = r["mcnemar"].get(x["metric"]) or r["bootstrap"][x["metric"]]
            row["holm"] = {"family": fam, "family_size": len(members), "p_basis": x["p_basis"],
                           "p": x["p"], "adjusted_p": a, "significant": bool(a < ALPHA)}
        fam_out[fam] = {"metrics": list(spec["metrics"]), "strata": list(spec["strata"]),
                        "size": len(members),
                        "uncorrected_significant": sum(
                            1 for x in members
                            if (pairs[x["pair"]]["strata"][x["stratum"]]["mcnemar"].get(x["metric"])
                                or pairs[x["pair"]]["strata"][x["stratum"]]["bootstrap"][x["metric"]])["significant"]),
                        "holm_significant": sum(1 for x in members if x["holm_significant"])}

    git = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    settings = {"seed": SEED, "bootstrap_iterations": ITER, "bootstrap_chunk_size": CHUNK, "alpha": ALPHA,
                "min_items_for_significance_claim": MIN_ITEMS, "source": "configs/musique_eval.json paired_comparison",
                "bootstrap_seeding": "fresh default_rng(seed) per pair x stratum, exactly as one --compare call on the stratum's rows",
                "excluded": "composite_score (legacy, ADR 0035 item 4); decomp_mean is computed in the full battery but is in no family",
                "holm": "Holm-Bonferroni within each slide family; p basis = exact McNemar for binary metrics, paired t for continuous (the percentile bootstrap emits a CI, not a p). Evaluator verdicts are uncorrected; holm is a separate field."}
    config = {"exp_id": "exp-029", "kind": "re-analysis, not a run", "git_head_at_compute": git,
              "evaluator": {"path": "scripts/musique_decompositions_evaluator.py",
                            "sha256": sha256(REPO / "scripts/musique_decompositions_evaluator.py"),
                            "functions_used_unchanged": ["_paired_bootstrap", "_statistics_for (via _paired_bootstrap)",
                                                         "_mcnemar", "_paired_t_test_row", "_significance_floor"]},
              "config": {"path": "configs/musique_eval.json", "sha256": sha256(REPO / "configs/musique_eval.json")},
              "settings": settings,
              "inputs": {k: {"path": v, "sha256": sha256(REPO / v)} for k, v in F.items()},
              "cross_check_inputs": {k: {"path": v, "sha256": sha256(REPO / v)} for k, v in XCHECK.items()},
              "families": {k: {"metrics": list(v["metrics"]), "strata": list(v["strata"])} for k, v in families.items()}}
    metrics = {"exp_id": "exp-029", "settings": settings, "provenance_cross_checks": xc,
               "holm_families": fam_out, "pairs": pairs}
    (OUT / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=1) + "\n")
    print(json.dumps(fam_out, indent=1))


if __name__ == "__main__":
    main()
