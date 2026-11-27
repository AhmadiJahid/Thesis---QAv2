"""exp-038 - paired tests on the pinned 600, overall and per gold hop, with Holm (ADR 0042).

Reads the evaluator's per-item files of the five arms run through
``run_decomposer.py --weight-edit`` (baseline = merged, unedited; edit; random-direction
control; length-matched EOS-row control; norm-only control) and calls the evaluator's own
test functions UNCHANGED (``_paired_bootstrap``, ``_mcnemar``, ``_paired_t_test_row``,
``_significance_floor``), exactly as exp-029 does. The only arithmetic added is Holm-Bonferroni
(``src/weight_space.py::holm``, ADR 0036) within each pair's family.

Usage (repo root)::

    python experiments/exp-038/analysis.py --arms runs/exp-038/eval/arms.json \
        --out experiments/exp-038

``arms.json`` maps arm name -> per-item file, plus optional ``exp026_stored`` (the committed
exp-026 per-item file) for the reproduction check of the in-notebook baseline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import musique_decompositions_evaluator as E  # noqa: E402
from weight_space import holm  # noqa: E402

CFG = json.loads((REPO / "configs" / "musique_eval.json").read_text())
PC = CFG["paired_comparison"]
SEED = int(CFG["seed"])
ITER = int(PC["bootstrap_iterations"])
CHUNK = int(PC["bootstrap_chunk_size"])
ALPHA = float(PC["alpha"])
MIN_ITEMS = int(PC["min_items_for_significance_claim"])
WEIGHTS = CFG["composite_score_weights"]
SCALE = float(CFG["composite_step_count_error_scale"])

BOOT_ALL = tuple(s for s in E.BOOTSTRAP_STATISTICS if s != "composite_score")
MCN_ALL = tuple(E.MCNEMAR_STATISTICS)
#: The seven report-card terms (ADR 0029), by per-item column - every Holm family's metrics.
CARD = ("break_exact_match", "sari", "ged", "chain_validity", "hop_count_exact_match",
        "under_decomposition", "over_decomposition")
STRATA = ("overall", "2", "3", "4")
#: (family, system_a, system_b). a minus b. The first is the primary question; the second is
#: the length-bias test ("does the edit beat a control that lengthens plans as much?").
PAIRS = (
    ("P_edit_vs_baseline", "edit", "baseline"),
    ("L_edit_vs_eos_control", "edit", "eos_control"),
    ("R_edit_vs_random", "edit", "random"),
    ("C_eos_control_vs_baseline", "eos_control", "baseline"),
    ("C_random_vs_baseline", "random", "baseline"),
    ("C_norm_control_vs_baseline", "norm_control", "baseline"),
)


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(path: Path) -> dict[str, dict]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))["items"]
    by = {str(r["item_id"]): r for r in rows}
    assert len(by) == len(rows), f"duplicate ids in {path}"
    return by


def compare(ra: list[dict], rb: list[dict], boot: tuple, mcn: tuple) -> dict:
    n = len(ra)
    floor = E._significance_floor(n, MIN_ITEMS)
    under = bool(floor["below_min_items"])
    out = {"n": n, "significance_floor": floor, "bootstrap": {}, "mcnemar": {}, "t_test": {}}
    arr_a = {s: np.array([float(r[s]) for r in ra]) for s in boot}
    arr_b = {s: np.array([float(r[s]) for r in rb]) for s in boot}
    if boot:
        out["bootstrap"] = E._paired_bootstrap(arr_a, arr_b, n=n, iterations=ITER, alpha=ALPHA, seed=SEED,
                                               weights=WEIGHTS, scale=SCALE, chunk_size=CHUNK,
                                               underpowered=under, statistics=boot)
        for s in boot:
            out["t_test"][s] = E._paired_t_test_row(arr_a[s], arr_b[s], ALPHA, under, name=s)
    if mcn:
        out["mcnemar"] = E._mcnemar(ra, rb, ALPHA, under, statistics=mcn)
    return out


def run_pair(a: dict, b: dict, label: str) -> dict:
    if set(a) != set(b):
        raise SystemExit(f"{label}: id sets differ ({len(a)} vs {len(b)}) - refusing to pair")
    if len(a) != 600:
        raise SystemExit(f"{label}: {len(a)} items, the pinned set has 600")
    ids = sorted(a)
    for i in ids:
        assert a[i]["gold_hop_count"] == b[i]["gold_hop_count"], f"{i}: gold hop differs"
    res = {"difference_direction": "system_a minus system_b", "strata": {}}
    for st in STRATA:
        sel = ids if st == "overall" else [i for i in ids if int(a[i]["gold_hop_count"]) == int(st)]
        ra, rb = [a[i] for i in sel], [b[i] for i in sel]
        bs = tuple(s for s in BOOT_ALL if all(s in r for r in ra + rb))
        ms = tuple(s for s in MCN_ALL if all(s in r for r in ra + rb))
        res["strata"][st] = compare(ra, rb, bs, ms)
    return res


def reproduction_check(baseline: dict, stored: dict) -> dict:
    """In-notebook merged baseline vs exp-026's stored 3090 predictions, same 600 ids."""
    if set(baseline) != set(stored):
        raise SystemExit("reproduction check: id sets differ")
    same = sum(1 for i in baseline if baseline[i]["pred_steps"] == stored[i]["pred_steps"])
    return {"n": len(baseline), "identical_pred_steps": same,
            "note": "merged fp16 on Colab vs unmerged adapter on the 3090: numeric differences can flip greedy tokens; "
                    "the paired baseline of every test is the in-notebook one, never exp-026's stored file"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    arms_cfg = json.loads(args.arms.read_text(encoding="utf-8"))
    arms = {k: load(Path(v)) for k, v in arms_cfg.items() if k != "exp026_stored"}
    pairs = {fam: {"system_a": a, "system_b": b, **run_pair(arms[a], arms[b], fam)} for fam, a, b in PAIRS}

    families = {}
    for fam, pr in pairs.items():
        members = []
        for st in STRATA:
            r = pr["strata"][st]
            for m in CARD:
                if m in r["mcnemar"]:
                    p, basis = r["mcnemar"][m]["p_value"], "mcnemar_exact"
                elif m in r["t_test"]:
                    p, basis = r["t_test"][m]["p_value"], "paired_t (bootstrap emits no p)"
                else:
                    raise SystemExit(f"{fam} {st} {m}: not available in the per-item files")
                if p is None:
                    # The evaluator emits no p when there is nothing to test (no discordant
                    # pairs, or zero-variance differences - e.g. two identical arms). Holm
                    # takes it as 1.0, and the row says so.
                    p, basis = 1.0, basis + "; evaluator emitted no p (nothing to test), taken as 1.0"
                members.append({"stratum": st, "metric": m, "p": p, "p_basis": basis})
        adj = holm([x["p"] for x in members])
        for x, q in zip(members, adj):
            x["holm_adjusted_p"] = q
            x["holm_significant"] = bool(q < ALPHA)
        families[fam] = {"size": len(members), "holm_significant": sum(x["holm_significant"] for x in members),
                         "members": members}

    repro = None
    if "exp026_stored" in arms_cfg:
        repro = reproduction_check(arms["baseline"], load(Path(arms_cfg["exp026_stored"])))

    git = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    settings = {"seed": SEED, "bootstrap_iterations": ITER, "alpha": ALPHA, "min_items": MIN_ITEMS,
                "families": "one Holm family per pair: 7 report-card terms x 4 strata (overall, hop 2/3/4) = 28 tests",
                "p_basis": "exact McNemar for binary terms, paired t for continuous (the bootstrap emits a CI, not a p)"}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "analysis_config.json").write_text(json.dumps({
        "exp_id": "exp-038", "git_head_at_compute": git, "settings": settings,
        "evaluator_sha256": sha256(REPO / "scripts/musique_decompositions_evaluator.py"),
        "inputs": {k: {"path": v, "sha256": sha256(Path(v))} for k, v in arms_cfg.items()}}, indent=2) + "\n")
    (args.out / "analysis_metrics.json").write_text(json.dumps({
        "exp_id": "exp-038", "settings": settings, "reproduction_check": repro,
        "holm_families": families, "pairs": pairs}, indent=1) + "\n")
    print(json.dumps({k: {"size": v["size"], "holm_significant": v["holm_significant"]} for k, v in families.items()}, indent=1))


if __name__ == "__main__":
    main()
