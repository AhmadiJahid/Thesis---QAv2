#!/usr/bin/env python3
"""exp-035 parity gate: scripts/break_hg_rescore.py vs Han & Gardent's own fork code.

Runs the fork's functions themselves (not a re-reading of them) on the same items and
compares per item with the ``hg_fork`` rows of a break_hg_rescore run:

- ``evaluation/evaluate_break_suite.py::clean_lingering_operators_cot``
- ``tools/break_evaluator/scripts/evaluate_predictions.py::format_qdmr``,
  ``get_exact_match``, ``get_sari_score`` and the ``strip_qmark_whitespace`` regex
  (copied from ``evaluate()``, the one piece that is inline there)
- ``tools/break_evaluator/evaluation/graph_matcher.py::GraphMatchScorer`` (spaCy lemmas via
  the fork's ``SequenceMatchScorer``, the ``edit-distance`` package, ``@exit_after(180)``)

Modules the measured path never calls (pandas, sacrebleu, datasets, matplotlib, the NormEM
normalizer) are stubbed so the fork imports under Python 3.13. The fork is NOT vendored into
this repo: fetch it at the commit below into any directory and pass it as --fork-root.

    https://github.com/hankelvin/complex_question_decomposition @ 9c9a6699ec440175192004a12fb2526c1d7a40c8

Usage::

    python experiments/exp-035/parity_check.py --fork-root <dir> \\
        --per-item <run-dir>/per_item.json --runs-root <runs> \\
        --config configs/break_hg_rescore_exp035.json --ged-items-per-arm 200 --out <json>
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
import types
from pathlib import Path

FORK_COMMIT = "9c9a6699ec440175192004a12fb2526c1d7a40c8"


def _stub(name: str, **attrs) -> None:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod


def load_fork(root: Path):
    _stub("pandas")
    _stub("sacrebleu")
    _stub("datasets")
    plt = types.ModuleType("matplotlib.pyplot")
    _stub("matplotlib", pyplot=plt)
    sys.modules["matplotlib.pyplot"] = plt
    _stub("normal_form")
    _stub("normal_form.normalized_graph_matcher", NormalizedGraphMatchScorer=object)
    _stub("normal_form.normalization_rules")
    sys.path.insert(0, str(root / "tools" / "break_evaluator" / "scripts"))
    sys.path.insert(0, str(root / "evaluation"))
    cwd = os.getcwd()
    try:
        ep = importlib.import_module("evaluate_predictions")
        suite = importlib.import_module("evaluate_break_suite")
    finally:
        os.chdir(cwd)
    gm = sys.modules["graph_matcher"]
    return ep, suite, gm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fork-root", required=True, type=Path)
    ap.add_argument("--per-item", required=True, type=Path)
    ap.add_argument("--runs-root", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--ged-items-per-arm", type=int, default=200)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    import re

    ep, suite, gm = load_fork(args.fork_root.resolve())
    scorer = gm.GraphMatchScorer()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    ours_all = json.loads(args.per_item.read_text(encoding="utf-8"))["hg_fork"]

    report = {"fork_commit": FORK_COMMIT, "arms": {}}
    for arm, rel in cfg["arms"].items():
        items = json.loads((args.runs_root / rel).read_text(encoding="utf-8"))["items"]
        ours = {r["item_id"]: r for r in ours_all[arm]}
        items = [it for it in items if it["item_id"] in ours]
        preds, golds, questions = [], [], []
        for it in items:
            preds.append(ep.format_qdmr(suite.clean_lingering_operators_cot(" ;".join(it["pred_steps"]))))
            golds.append(ep.format_qdmr(suite.clean_lingering_operators_cot(" ;".join(it["gold_steps"]))))
            questions.append(it["question"])
        d_str = [re.sub(r"\?+$", "", d.to_string().strip()).strip() for d in preds]
        g_str = [re.sub(r"\?+$", "", g.to_string().strip()).strip() for g in golds]
        em = ep.get_exact_match(d_str, g_str)
        sari = ep.get_sari_score(d_str, g_str, questions)
        n_ged = min(args.ged_items_per_arm, len(items))
        # Spread the GED sample over the file (which is ordered by hop) deterministically.
        step = max(1, len(items) // n_ged)
        ged_idx = list(range(0, len(items), step))[:n_ged]
        ged = scorer.get_edit_distance_match_scores(
            [preds[i].to_graph() for i in ged_idx], [golds[i].to_graph() for i in ged_idx]
        )
        em_mismatch = sari_max = 0
        for i, it in enumerate(items):
            o = ours[it["item_id"]]
            em_mismatch += int(float(em[i]) != o["break_exact_match"])
            sari_max = max(sari_max, abs(float(sari[i]) - o["sari"]))
        ged_max, ged_none_mismatch = 0.0, 0
        for j, i in enumerate(ged_idx):
            o = ours[items[i]["item_id"]]["ged"]
            if ged[j] is None or o is None:
                ged_none_mismatch += int((ged[j] is None) != (o is None))
                continue
            ged_max = max(ged_max, abs(float(ged[j]) - o))
        report["arms"][arm] = {
            "items_em_sari": len(items),
            "em_mismatches": em_mismatch,
            "sari_max_abs_diff": sari_max,
            "items_ged": len(ged_idx),
            "ged_max_abs_diff": ged_max,
            "ged_dropped_disagreements": ged_none_mismatch,
        }
        print(arm, report["arms"][arm], flush=True)
    tol = 1e-9
    report["passed"] = all(
        a["em_mismatches"] == 0 and a["sari_max_abs_diff"] <= tol and a["ged_max_abs_diff"] <= tol
        and a["ged_dropped_disagreements"] == 0
        for a in report["arms"].values()
    )
    report["tolerance"] = tol
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("PASSED" if report["passed"] else "FAILED")
    if not report["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
