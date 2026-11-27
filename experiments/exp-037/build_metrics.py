"""Build experiments/exp-037/metrics.json from runs/exp-037/ (read-only on runs/).

exp-037: the count-first arm (configs/decomposer_musique.json condition ``count_first``) vs the
unguided arm of the same model, everything else equal. Run from the repo root after
``bash experiments/exp-037/run.sh compare``:

    python experiments/exp-037/build_metrics.py

Pre-declared in experiments/exp-037/config.json BEFORE any run (ADR 0036 item 5):

- PRIMARY pairs: count-first minus the SAME-MACHINE unguided re-run of the same model
  (mistral_count_first_vs_rerun, qwen_count_first_vs_rerun). Both arms of a primary pair ran
  on the same GPU, environment and commit.
- F1 (Holm): primary pairs x the 8 report-card metrics, overall (n=600). p = exact McNemar
  (binary) / paired t (continuous), from the evaluator's own compare output.
- F2 (Holm): primary pairs x {hop_count_exact_match, under_decomposition} x gold hop {2,3,4}
  (n=200 each), exact McNemar via the evaluator's own ``_mcnemar_exact_p``.
- SECONDARY, uncorrected and labelled: count-first minus the STORED baseline (exp-020 re-parsed
  by exp-032; exp-027), and the hardware control (same-machine re-run minus stored baseline).
- "Reproduced" (hardware control): the same-machine unguided re-run's recorded
  ``decomposition`` equals the stored baseline's for 600/600 query ids, compared as exact
  strings against the stored re-parsed results.json. Anything less is "not reproduced", and
  the secondary pair of that model is then labelled hardware-confounded.

The evaluator's verdicts are never overwritten (ADR 0036 item 1). Nothing is imputed: a pair
whose inputs are missing is listed under ``pairs_not_run``.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from musique_decompositions_evaluator import _mcnemar_exact_p  # noqa: E402

R = ROOT / "runs" / "exp-037"
CONFIG = Path(__file__).resolve().parent / "config.json"
OUT = Path(__file__).resolve().parent / "metrics.json"
ALPHA = 0.05
N_ITEMS = 600
ARMS = ["mistral_count_first", "mistral_unguided_rerun", "qwen_count_first", "qwen_unguided_rerun"]
STORED = {
    "exp020_reparsed": {"eval": ROOT / "runs/exp-032/exp-020/eval",
                        "results": ROOT / "runs/exp-032/exp-020/reparse/results.json"},
    "exp027_reparsed": {"eval": ROOT / "runs/exp-032/exp-027/eval",
                        "results": ROOT / "runs/exp-032/exp-027/reparse/results.json"},
}
PRIMARY = {
    "mistral_count_first_vs_rerun": ("mistral_count_first", "mistral_unguided_rerun"),
    "qwen_count_first_vs_rerun": ("qwen_count_first", "qwen_unguided_rerun"),
}
SECONDARY = {
    "mistral_count_first_vs_exp020": ("count-first minus stored exp-020 (re-parsed)", "mistral_unguided_rerun"),
    "qwen_count_first_vs_exp027": ("count-first minus stored exp-027", "qwen_unguided_rerun"),
    "mistral_rerun_vs_exp020": ("hardware control: same-machine re-run minus stored exp-020 (re-parsed)", "mistral_unguided_rerun"),
    "qwen_rerun_vs_exp027": ("hardware control: same-machine re-run minus stored exp-027", "qwen_unguided_rerun"),
}
CONTROLS = {"mistral_unguided_rerun": "exp020_reparsed", "qwen_unguided_rerun": "exp027_reparsed"}
# Same 8 metrics, tests and directions as exp-031/exp-034's families.
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
PER_HOP = ("hop_count_exact_match", "under_decomposition")
HOPS = (2, 3, 4)


def load(p: Path):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def holm(ps: list[float]) -> list[float]:
    # Same step-down implementation as experiments/exp-031 and exp-034/build_metrics.py.
    m = len(ps)
    order = sorted(range(m), key=lambda k: ps[k])
    adj = [0.0] * m
    running = 0.0
    for rank, k in enumerate(order):
        running = max(running, min(1.0, (m - rank) * ps[k]))
        adj[k] = running
    return adj


def eval_dir(name: str) -> Path:
    return STORED[name]["eval"] if name in STORED else R / name / "eval"


def cell(cm: dict, sec: str, key: str, bkey: str | None, direction: str) -> dict:
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
        out.update(bootstrap_ci_low=b["ci_low"], bootstrap_ci_high=b["ci_high"],
                   bootstrap_significant=b["significant"],
                   bootstrap_vs_t_disagree=b["significant"] != r["significant"])
    return out


def compare(pair: str) -> dict | None:
    p = R / "compare" / pair / "compare_metrics.json"
    if not p.exists():
        return None
    cm = load(p)
    assert cm["num_aligned_items"] == N_ITEMS, (pair, cm["num_aligned_items"])
    cm["_path"] = str(p.relative_to(ROOT))
    return cm


def per_item(name: str) -> dict[str, dict]:
    return {it["item_id"]: it for it in load(eval_dir(name) / "eval_per_item.json")["items"]}


def results_by_id(path: Path) -> dict[str, str]:
    return {str(r["query_id"]): r["decomposition"] for r in load(path)}


declared = load(CONFIG)["hardware_control"]["reproduced_if"]

# ---- per-arm readings
arms = {}
for arm in ARMS:
    res = sorted(glob.glob(str(R / arm / "decomposer" / "*" / "metrics.json")))
    ev = R / arm / "eval" / "eval_metrics.json"
    if not res or not ev.exists():
        continue
    dm, em = load(Path(res[0])), load(ev)
    arms[arm] = {
        "decomposer_metrics_path": str(Path(res[0]).relative_to(ROOT)),
        "eval_metrics_path": str(ev.relative_to(ROOT)),
        "report_card": {k: em[k] for k in FAMILY},
        "eval_total_evaluated": em["total_evaluated"],
        "eval_missing_gold_count": em["missing_gold_count"],
        "eval_git": em.get("git"),
        "rows_with_empty_decomposition": dm["rows_with_empty_decomposition"],
        "rows_at_max_new_tokens": dm["rows_at_max_new_tokens"],
        "evaluation_set_pinned": dm["evaluation_set"].get("pinned"),
        "cost": {k: dm["cost"].get(k) for k in ("mean_prompt_tokens_per_query",
                                                 "mean_completion_tokens_per_query",
                                                 "mean_latency_seconds_per_query",
                                                 "total_generation_seconds")},
        # The self-predicted count: accuracy vs gold, agreement with steps emitted, status
        # counts (rows without a readable count are kept and scored, never dropped).
        "count_first": dm.get("count_first"),
    }

# ---- hardware control: reproduced = 600/600 identical decompositions (declared in config.json)
controls = {}
for arm, base in CONTROLS.items():
    res = sorted(glob.glob(str(R / arm / "decomposer" / "*" / "results.json")))
    if not res:
        continue
    a, b = results_by_id(Path(res[0])), results_by_id(STORED[base]["results"])
    assert set(a) == set(b) and len(a) == N_ITEMS, (arm, len(a), len(b))
    same = sum(a[q] == b[q] for q in a)
    controls[arm] = {"stored_baseline": base, "items": len(a), "identical_decompositions": same,
                     "reproduced": same == N_ITEMS, "rule": declared}

# ---- F1: primary pairs x 8 metrics, overall
f1_cells, f1, not_run = [], {}, []
for pair in PRIMARY:
    cm = compare(pair)
    if cm is None:
        not_run.append(pair)
        continue
    f1[pair] = {"path": cm["_path"], "difference_direction": cm["difference_direction"],
                "num_aligned_items": cm["num_aligned_items"], "git": cm.get("git"), "cells": {}}
    for k, (sec, key, bkey, direction) in FAMILY.items():
        c = cell(cm, sec, key, bkey, direction)
        f1[pair]["cells"][k] = c
        f1_cells.append(c)
for c, a in zip(f1_cells, holm([c["p_value"] for c in f1_cells])):
    c["holm_adjusted_p"], c["significant_holm"] = a, a < ALPHA

# ---- F2: primary pairs x {hop EM, under} x hop, McNemar per stratum
f2_cells, f2 = [], {}
for pair, (a, b) in PRIMARY.items():
    if pair in not_run:
        continue
    ia, ib = per_item(a), per_item(b)
    assert set(ia) == set(ib) and len(ia) == N_ITEMS, pair
    f2[pair] = {}
    for h in HOPS:
        ids = [i for i in ia if int(ia[i]["gold_hop_count"]) == h]
        assert all(int(ib[i]["gold_hop_count"]) == h for i in ids)
        for metric in PER_HOP:
            av = [float(ia[i][metric]) >= 1.0 for i in ids]
            bv = [float(ib[i][metric]) >= 1.0 for i in ids]
            only_a = sum(x and not y for x, y in zip(av, bv))
            only_b = sum(y and not x for x, y in zip(av, bv))
            c = {"hop": h, "metric": metric, "n": len(ids),
                 "system_a_rate": sum(av) / len(ids), "system_b_rate": sum(bv) / len(ids),
                 "difference": (sum(av) - sum(bv)) / len(ids),
                 "discordant_only_in_a": only_a, "discordant_only_in_b": only_b,
                 "p_value": _mcnemar_exact_p(only_a, only_b)}
            c["significant_uncorrected"] = c["p_value"] < ALPHA
            f2[pair][f"hop{h}:{metric}"] = c
            f2_cells.append(c)
for c, a in zip(f2_cells, holm([c["p_value"] for c in f2_cells])):
    c["holm_adjusted_p"], c["significant_holm"] = a, a < ALPHA

# ---- secondary: vs stored baseline and the hardware control (uncorrected, labelled)
secondary = {}
for pair, (what, control_arm) in SECONDARY.items():
    cm = compare(pair)
    if cm is None:
        continue
    ctl = controls.get(control_arm)
    secondary[pair] = {
        "what": what, "path": cm["_path"], "num_aligned_items": cm["num_aligned_items"],
        "hardware_confounded": None if ctl is None else not ctl["reproduced"],
        "cells": {k: cell(cm, *v) for k, v in FAMILY.items()},
    }

out = {
    "exp_id": "exp-037",
    "declared_in": "experiments/exp-037/config.json (primary_pairs, holm_families, hardware_control)",
    "arms": arms,
    "stored_baselines": {k: {kk: str(vv.relative_to(ROOT)) for kk, vv in v.items()}
                         for k, v in STORED.items()},
    "hardware_control": controls,
    "holm_family_F1_primary_overall": {
        "size": len(f1_cells), "pairs": f1, "pairs_not_run": not_run,
        "count_significant_uncorrected": sum(c["significant_uncorrected"] for c in f1_cells),
        "count_significant_holm": sum(c["significant_holm"] for c in f1_cells),
    },
    "holm_family_F2_per_hop": {
        "size": len(f2_cells), "pairs": f2,
        "count_significant_uncorrected": sum(c["significant_uncorrected"] for c in f2_cells),
        "count_significant_holm": sum(c["significant_holm"] for c in f2_cells),
    },
    "secondary_uncorrected": secondary,
    "alpha": ALPHA,
    "wording": "ADR 0036 item 4: non-significance is 'not distinguishable', never 'equal'.",
}
OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
print(f"wrote {OUT}: arms={sorted(arms)} F1={len(f1_cells)} F2={len(f2_cells)} "
      f"not_run={not_run} controls={ {k: v['reproduced'] for k, v in controls.items()} }")
