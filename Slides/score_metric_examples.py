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
- The statistics slides' toy examples go through the evaluator's own ``_mcnemar``,
  ``_paired_bootstrap``, ``_statistics_for`` and ``_paired_t_test_row`` with the config's
  seed, iterations and alpha. The three example resamples are the first three rows of the
  stream ``_paired_bootstrap`` draws (one ``rng.integers`` call per chunk, row-major).
- The "real run" lines are READ from a committed metrics JSON (exp-005), never recomputed.

Run from the repo root:  .venv/bin/python Slides/score_metric_examples.py
No metric is reimplemented here; nothing is written to disk.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import networkx as nx
import numpy as np

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
# Statistics toys.  McNemar: 10 questions, 1 = correct (A, B are two systems).
TOY_MCNEMAR_A = [1, 1, 1, 0, 1, 1, 1, 1, 0, 1]
TOY_MCNEMAR_B = [0, 0, 1, 1, 0, 0, 1, 0, 0, 0]
# Bootstrap and t-test: 5 questions, a per-item score (step F1) for systems A and B.
TOY_SCORES_A = [0.8, 0.6, 1.0, 0.5, 0.7]
TOY_SCORES_B = [0.6, 0.5, 0.7, 0.3, 0.6]
TOY_SHOWN_RESAMPLES = 3
REAL_RUN = REPO / "experiments" / "exp-005" / "metrics.json"
REAL_COMPARISON = "unguided_vs_oracle_guided"
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

    up = E._tokenize(E._join_steps(PLANS["unlinked"]))
    lcs_u = E._lcs_len(up, gold_tokens)
    only_p = [t for t in up if t not in set(gold_tokens)]
    only_g = [t for t in gold_tokens if t not in set(up)]
    # Here the words outside the LCS are exactly the words the other side lacks.
    assert len(only_p) == len(up) - lcs_u and len(only_g) == len(gold_tokens) - lcs_u
    print(f"=== ROUGE-L unlinked: words outside the LCS  pred-only={only_p}  gold-only={only_g}")
    q1 = E._ngram_set(QUESTION.split(" "), 1)
    g1 = E._ngram_set(gold_break.split(" "), 1)
    print(f"=== SARI unigrams, gold vs question: kept={sorted(x[0] for x in q1 & g1)}")
    print(f"    added={sorted(x[0] for x in g1 - q1)}")
    print(f"    deleted={sorted(x[0] for x in q1 - g1)}")
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
    alpha, seed, iters = pc["alpha"], cfg["seed"], pc["bootstrap_iterations"]
    print("=== statistics (config values and the evaluator's McNemar)")
    print(f"  seed={seed} bootstrap_iterations={iters} "
          f"alpha={alpha} min_items_for_significance_claim={pc['min_items_for_significance_claim']}")
    for m in range(1, 8):
        print(f"  McNemar min attainable p with m={m} discordant pairs: "
              f"{r(E._mcnemar_exact_p(m, 0), 4)}")
    print(f"  McNemar p(b=3, c=0)={r(E._mcnemar_exact_p(3, 0), 4)}  "
          f"p(b=8, c=2)={r(E._mcnemar_exact_p(8, 2), 4)}")
    print()

    print("=== toy McNemar (evaluator's _mcnemar on 10 binary items)")
    rows_a = [{"exact_match": float(x)} for x in TOY_MCNEMAR_A]
    rows_b = [{"exact_match": float(x)} for x in TOY_MCNEMAR_B]
    n_toy = len(rows_a)
    mc = E._mcnemar(rows_a, rows_b, alpha, n_toy < pc["min_items_for_significance_claim"],
                    statistics=("exact_match",))["exact_match"]
    b, c = mc["discordant_only_in_a"], mc["discordant_only_in_b"]
    tail_terms = [math.comb(b + c, k) for k in range(min(b, c) + 1)]
    assert math.isclose(mc["p_value"], min(1.0, 2 * sum(tail_terms) / 2 ** (b + c)))
    print(f"  A={TOY_MCNEMAR_A}\n  B={TOY_MCNEMAR_B}")
    print(f"  b={b} c={c} n_discordant={b + c} tail terms C(n,k)={tail_terms} "
          f"sum={sum(tail_terms)} 2^n={2 ** (b + c)}")
    print(f"  p={r(mc['p_value'], 4)} significant={mc['significant']} "
          f"min_attainable_p={r(mc['min_attainable_p_value'], 4)} underpowered={mc['underpowered']}")
    print()

    print("=== toy paired bootstrap (evaluator's _paired_bootstrap / _statistics_for)")
    a_arr, b_arr = np.array(TOY_SCORES_A), np.array(TOY_SCORES_B)
    n5 = a_arr.size
    under5 = n5 < pc["min_items_for_significance_claim"]
    bs = E._paired_bootstrap({"step_f1": a_arr}, {"step_f1": b_arr}, n5, iters, alpha, seed,
                             {}, 1.0, pc["bootstrap_chunk_size"], under5,
                             statistics=("step_f1",))["step_f1"]
    # The first rows of the evaluator's own resample stream, re-drawn for display.
    full = np.random.default_rng(seed).integers(0, n5, size=(iters, n5))
    first = np.random.default_rng(seed).integers(0, n5, size=(TOY_SHOWN_RESAMPLES, n5))
    assert (full[:TOY_SHOWN_RESAMPLES] == first).all()
    deltas = (E._statistics_for({"step_f1": a_arr}, full, {}, 1.0, ("step_f1",))["step_f1"]
              - E._statistics_for({"step_f1": b_arr}, full, {}, 1.0, ("step_f1",))["step_f1"])
    lo, hi = np.percentile(deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    assert math.isclose(lo, bs["ci_low"]) and math.isclose(hi, bs["ci_high"])
    print(f"  A={TOY_SCORES_A} mean={r(bs['system_a'])}\n  B={TOY_SCORES_B} mean={r(bs['system_b'])}")
    print(f"  observed delta=A-B={r(bs['difference'])}")
    for k in range(TOY_SHOWN_RESAMPLES):
        idx = [int(i) + 1 for i in first[k]]
        print(f"  resample {k + 1}: questions {idx}  "
              f"A mean={r(a_arr[first[k]].mean())} B mean={r(b_arr[first[k]].mean())} "
              f"delta*={r(deltas[k])}")
    print(f"  {iters} resamples: CI=[{r(bs['ci_low'])}, {r(bs['ci_high'])}] "
          f"significant={bs['significant']} underpowered={bs['underpowered']}")
    print()

    print("=== toy paired t-test (evaluator's _paired_t_test_row -> scipy ttest_rel)")
    tt = E._paired_t_test_row(a_arr, b_arr, alpha, under5, name="step_f1")
    d = a_arr - b_arr
    sd = float(d.std(ddof=1))
    assert math.isclose(tt["t_statistic"], d.mean() / (sd / math.sqrt(n5)))
    print(f"  d={[r(x, 2) for x in d]} d_bar={r(d.mean())} s_d={r(sd, 4)} "
          f"s_d/sqrt(n)={r(sd / math.sqrt(n5), 4)} n={n5}")
    print(f"  t={r(tt['t_statistic'])} df={tt['degrees_of_freedom']} p={r(tt['p_value'], 4)} "
          f"significant={tt['significant']} underpowered={tt['underpowered']}")
    print()

    print(f"=== real run, READ from {REAL_RUN.relative_to(REPO)} "
          f"comparisons.{REAL_COMPARISON} (not recomputed)")
    real = json.loads(REAL_RUN.read_text())["comparisons"][REAL_COMPARISON]
    print(f"  system_a={real['system_a_path']}\n  system_b={real['system_b_path']}")
    print(f"  n={real['num_aligned_items']} seed={real['seed']} "
          f"bootstrap_iterations={real['bootstrap_iterations']} alpha={real['alpha']}")
    m = real["mcnemar"]["exact_match"]
    print(f"  McNemar exact_match: b={m['correct_only_in_a']} c={m['correct_only_in_b']} "
          f"p={m['p_value']} rates A={r(m['system_a_rate'])} B={r(m['system_b_rate'])}")
    assert math.isclose(E._mcnemar_exact_p(m["correct_only_in_a"], m["correct_only_in_b"]),
                        m["p_value"])  # sanity only: the slide quotes the JSON's value
    bt = real["bootstrap"]["step_f1"]
    print(f"  bootstrap step_f1: A={r(bt['system_a'])} B={r(bt['system_b'])} "
          f"delta={bt['difference']} CI=[{bt['ci_low']}, {bt['ci_high']}] "
          f"significant={bt['significant']}")
    tr = real["t_test"]["step_f1"]
    print(f"  t-test step_f1: t={tr['t_statistic']} df={tr['degrees_of_freedom']} "
          f"p={tr['p_value']} significant={tr['significant']}")

if __name__ == "__main__":
    main()
