"""Reproduce every number on the metric-example slides of Slides/thesis-deck-v2.tex.

Learnings (non-obvious, for whoever edits these slides next):
- Every per-plan number comes from the evaluator's own ``score_item`` (the function a real
  scoring run calls), with the GED guards and blend settings read from
  configs/musique_eval.json. The intermediate quantities (SARI's per-n terms, GED's graph
  sizes and edit path, OA positions, step-set sizes, LCS and token counts, chain links) are
  rebuilt from the evaluator's own helpers and asserted to reproduce the matching
  score_item value, so a drift in the evaluator breaks this script instead of the deck.
- The example plans use the bracketed ``[#k]`` form shown on the slides; the evaluator's
  bare-``#k`` regex reads both, but Break EM / SARI see the brackets as text.

Run from the repo root:  .venv/bin/python Slides/score_metric_examples.py
No metric is reimplemented here; nothing is written to disk.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import networkx as nx

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import musique_decompositions_evaluator as E  # noqa: E402
from answer_metrics import compute_exact, compute_f1, get_tokens, normalize_answer  # noqa: E402

CONFIG_PATH = REPO / "configs" / "musique_eval.json"

QUESTION = "What languages are spoken in films that share a director with The Matrix?"
GOLD = [
    "Who directed The Matrix?",
    "Which other films did [#1] direct?",
    "What languages are spoken in [#2]?",
]
PLANS = {
    "reversed": [GOLD[2], GOLD[1], GOLD[0]],
    "merged": [
        "Who directed The Matrix?",
        "What languages are spoken in films [#1] directed?",
    ],
    "unlinked": [
        "Who directed The Matrix?",
        "Which other films did that director direct?",
        "What languages are spoken in those films?",
    ],
    "extra": [
        "Who directed The Matrix?",
        "Which other films did [#1] direct?",
        "What are the titles of those films?",
        "What languages are spoken in [#2]?",
    ],
    "punct_only": ["Who directed The Matrix.", GOLD[1], GOLD[2]],
}
ANSWER_GOLD = "English"
ANSWER_PRED = "the English language"


def r(x: float, k: int = 3) -> float:
    return round(float(x), k)


def sari_terms(question: str, pred_string: str, gold_string: str) -> dict:
    """SARI's keep/add/delete terms per n, from the evaluator's own n-gram and F helpers."""
    src_t, pred_t, tgt_t = question.split(" "), pred_string.split(" "), gold_string.split(" ")
    rows = []
    for n in range(1, E._SARI_MAX_GRAM_SIZE + 1):
        s, p, t = E._ngram_set(src_t, n), E._ngram_set(pred_t, n), E._ngram_set(tgt_t, n)
        sp, st = s & p, s & t
        keep = E._fbeta(len(sp & st), len(sp), len(st))
        snp, snt = s - p, s - t
        dele = E._fbeta(len(snp & snt), len(snp), len(snt), beta=E._SARI_DELETION_BETA)
        added = p - s
        add = E._fbeta(len(added & t), len(added), len(t - s))
        rows.append(dict(n=n, keep=keep, keep_counts=(len(sp & st), len(sp), len(st)),
                         add=add, add_counts=(len(added & t), len(added), len(t - s)),
                         delete=dele, delete_counts=(len(snp & snt), len(snp), len(snt))))
    k = sum(x["keep"] for x in rows) / len(rows)
    a = sum(x["add"] for x in rows) / len(rows)
    d = sum(x["delete"] for x in rows) / len(rows)
    return dict(per_n=rows, avg_keep=k, avg_add=a, avg_delete=d, sari=(k + a + d) / 3.0)


def ged_detail(pred_steps: list[str], gold_steps: list[str]) -> dict:
    gp = E._decomposition_graph(E._break_steps(pred_steps))
    gg = E._decomposition_graph(E._break_steps(gold_steps))
    size_p = gp.number_of_nodes() + gp.number_of_edges()
    size_g = gg.number_of_nodes() + gg.number_of_edges()
    # One optimal edit path, for the slide's explanation only (same cost function).
    paths, cost = nx.optimal_edit_paths(gp, gg, node_subst_cost=E._node_subst_cost)
    node_path, edge_path = paths[0]
    node_costs = []
    for u, v in node_path:
        if u is not None and v is not None:
            node_costs.append((u, v, r(E._node_subst_cost(gp.nodes[u], gg.nodes[v]))))
        else:
            node_costs.append((u, v, 1.0))
    return dict(pred_nodes=gp.number_of_nodes(), pred_edges=gp.number_of_edges(),
                gold_nodes=gg.number_of_nodes(), gold_edges=gg.number_of_edges(),
                pred_size=size_p, gold_size=size_g, normalisation=max(size_p, size_g),
                optimal_cost=r(cost), node_path=node_costs,
                edge_path=[e for e in edge_path if e[0] is None or e[1] is None])


def main() -> None:
    cfg = json.loads(CONFIG_PATH.read_text())
    ged_policy = E._ged_policy(cfg)
    blend = E._decomp_mean_policy(cfg)
    print(f"config: {CONFIG_PATH.relative_to(REPO)}  ged guards: "
          f"max_nodes={ged_policy['max_nodes_for_optimizer']} "
          f"budget={ged_policy['per_item_time_budget_seconds']}s")
    print(f"question: {QUESTION}")
    gold_tokens = E._tokenize(E._join_steps(GOLD))
    gold_break = E._break_string(E._break_steps(GOLD))
    print(f"gold break string: {gold_break!r}")
    print(f"gold tokens (ROUGE-L): {len(gold_tokens)}  gold refs: "
          f"{sum(len(E._BARE_REF_RX.findall(s)) for s in GOLD)}")
    print()

    matrix = {}
    for name, steps in PLANS.items():
        row = E.EvalRow(item_id=name, question=QUESTION, pred_steps=steps,
                        gold_steps=GOLD, gold_hop_count=len(GOLD))
        it = E.score_item(row, ged_policy["max_nodes_for_optimizer"],
                          ged_policy["per_item_time_budget_seconds"],
                          tuple(blend["components"]), blend["ged_clamp"])
        assert it["ged_fallback"] is None, it["ged_fallback"]
        pred_tokens = E._tokenize(E._join_steps(steps))
        lcs = E._lcs_len(pred_tokens, gold_tokens)
        pset = {E._normalize_step(s) for s in steps}
        gset = {E._normalize_step(s) for s in GOLD}
        positions = [E._normalize_step(p) == E._normalize_step(g) for p, g in zip(steps, GOLD)]
        valid_links = [(i, int(k), 1 <= int(k) < i) for i, s in enumerate(steps, 1)
                       for k in E._BARE_REF_RX.findall(s)]
        pred_break = E._break_string(E._break_steps(steps))
        st = sari_terms(QUESTION, pred_break, gold_break)
        assert math.isclose(st["sari"], it["sari"], abs_tol=1e-12), (st["sari"], it["sari"])
        gd = ged_detail(steps, GOLD)
        assert math.isclose(gd["optimal_cost"] / gd["normalisation"], it["ged"], abs_tol=1e-3)
        # Every other displayed intermediate, checked against its score_item counterpart.
        assert math.isclose(sum(positions) / max(len(steps), len(GOLD)),
                            it["ordered_step_accuracy"], abs_tol=1e-12)
        assert math.isclose(len(pset & gset) / len(pset), it["step_precision"], abs_tol=1e-12)
        assert math.isclose(len(pset & gset) / len(gset), it["step_recall"], abs_tol=1e-12)
        assert math.isclose(lcs / len(pred_tokens), it["rouge_l_precision"], abs_tol=1e-12)
        assert math.isclose(lcs / len(gold_tokens), it["rouge_l_recall"], abs_tol=1e-12)
        assert len(valid_links) == it["chain_pred_reference_count"]
        assert (sum(len(E._BARE_REF_RX.findall(s)) for s in GOLD)
                == it["chain_gold_reference_count"])
        links_score = (sum(ok for _, _, ok in valid_links) / len(valid_links)
                       if valid_links else 0.0)  # gold chains here, so silence scores 0
        assert math.isclose(links_score, it["chain_validity"], abs_tol=1e-12)

        print(f"=== {name}")
        print(f"  steps: {steps}")
        print(f"  exact_match={it['exact_match']}  normalised-position matches={positions}")
        print(f"  step P={r(it['step_precision'])} R={r(it['step_recall'])} "
              f"F1={r(it['step_f1'])}  |P∩G|={len(pset & gset)} |P|={len(pset)} |G|={len(gset)}")
        print(f"  ordered_step_accuracy={r(it['ordered_step_accuracy'])}  "
              f"matches={sum(positions)} / max({len(steps)},{len(GOLD)})")
        print(f"  ROUGE-L P={r(it['rouge_l_precision'])} R={r(it['rouge_l_recall'])} "
              f"F1={r(it['rouge_l_f1'])}  |LCS|={lcs} pred_tokens={len(pred_tokens)} "
              f"gold_tokens={len(gold_tokens)}")
        print(f"  hop_count_exact_match={it['hop_count_exact_match']}  "
              f"pred_hops={it['predicted_hop_count']} gold_hops={it['gold_hop_count']}  "
              f"under={it['under_decomposition']} over={it['over_decomposition']}")
        print(f"  chain_validity={r(it['chain_validity'])}  emitted={it['chain_pred_reference_count']} "
              f"gold_refs={it['chain_gold_reference_count']}  links(step,k,valid)={valid_links}")
        print(f"  break_exact_match={it['break_exact_match']}  pred break string: {pred_break!r}")
        print(f"  SARI={r(it['sari'])}  avg keep={r(st['avg_keep'])} add={r(st['avg_add'])} "
              f"delete={r(st['avg_delete'])}")
        for x in st["per_n"]:
            print(f"     n={x['n']}: keep={r(x['keep'])} {x['keep_counts']}  add={r(x['add'])} "
                  f"{x['add_counts']}  delete={r(x['delete'])} {x['delete_counts']}")
        print(f"  GED={r(it['ged'])}  pred graph {gd['pred_nodes']}n+{gd['pred_edges']}e="
              f"{gd['pred_size']}  gold graph {gd['gold_nodes']}n+{gd['gold_edges']}e="
              f"{gd['gold_size']}  cost={gd['optimal_cost']} / max={gd['normalisation']}")
        print(f"     node path (pred,gold,cost): {gd['node_path']}")
        print(f"     edge inserts/deletes: {gd['edge_path']}")
        print()
        matrix[name] = it

    print("=== score matrix (rounded to 2 dp)")
    cols = ["exact_match", "step_f1", "ordered_step_accuracy", "rouge_l_f1",
            "hop_count_exact_match", "under_decomposition", "over_decomposition",
            "chain_validity", "break_exact_match", "sari", "ged"]
    print("  plan       " + " ".join(f"{c[:8]:>8}" for c in cols))
    for name, it in matrix.items():
        print(f"  {name:<10} " + " ".join(f"{it[c]:>8.2f}" for c in cols))
    print()

    empty = E._sari(QUESTION, E._break_string(E._break_steps([])), gold_break)
    print(f"=== empty plan SARI (pred_steps=[]): {r(empty)}")
    print()

    print("=== answer metrics")
    print(f"  normalised gold={normalize_answer(ANSWER_GOLD)!r} pred={normalize_answer(ANSWER_PRED)!r}")
    gt, pt = get_tokens(ANSWER_GOLD), get_tokens(ANSWER_PRED)
    print(f"  tokens gold={gt} pred={pt}")
    print(f"  EM={compute_exact(ANSWER_GOLD, ANSWER_PRED)}  F1={r(compute_f1(ANSWER_GOLD, ANSWER_PRED))}")
    print()

    pc = cfg["paired_comparison"]
    print("=== statistics (config values and the evaluator's McNemar)")
    print(f"  seed={cfg['seed']} bootstrap_iterations={pc['bootstrap_iterations']} "
          f"alpha={pc['alpha']} min_items_for_significance_claim={pc['min_items_for_significance_claim']}")
    for m in range(1, 8):
        print(f"  McNemar min attainable p with m={m} discordant pairs: "
              f"{r(E._mcnemar_exact_p(m, 0), 4)}")
    print(f"  McNemar p(b=3, c=0)={r(E._mcnemar_exact_p(3, 0), 4)}  "
          f"p(b=8, c=2)={r(E._mcnemar_exact_p(8, 2), 4)}")


if __name__ == "__main__":
    main()
