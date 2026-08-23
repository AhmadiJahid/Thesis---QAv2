"""Truncation-confound audit of exp-010. Read-only; no GPU, no generation."""
import json, glob, os, sys, math
import numpy as np

ROOT = "/cta/users/fyilmaz/Thesis---QAv2"
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src"))

CELLS = []
for bal in ["imbalanced", "balanced", "clustered"]:
    for var in ["biencoder_only", "biencoder_plus_ce"]:
        for mode in ["raw", "typed", "uniform"]:
            CELLS.append(f"size2000_{bal}_trial0__{var}__{mode}")


def load_dec(cell):
    p = f"{ROOT}/runs/pool_sweep/decomposer/{cell}/results.json"
    rows = json.load(open(p))
    return {r["query_id"]: r for r in rows}


def load_eval(cell):
    p = f"{ROOT}/runs/pool_sweep/eval/{cell}/eval_per_item.json"
    d = json.load(open(p))
    return {r["item_id"]: r for r in d["items"]}


def cfg_max_new(cell):
    g = glob.glob(f"{ROOT}/runs/pool_sweep/decomposer/{cell}/*/config.json")
    assert len(g) == 1, (cell, g)
    c = json.load(open(g[0]))
    return c["generation"]["max_new_tokens"], c.get("quantization")


DEC = {c: load_dec(c) for c in CELLS}
EV = {c: load_eval(c) for c in CELLS}

print("=== 0. sanity ===")
for c in CELLS:
    mn, q = cfg_max_new(c)
    d = DEC[c]
    assert len(d) == 750 and len(EV[c]) == 750, (c, len(d), len(EV[c]))
    assert set(d) == set(EV[c]), c
    # flag consistency: hit_max_new_tokens == (completion_tokens >= max_new_tokens)
    bad = [k for k, r in d.items() if r["hit_max_new_tokens"] != (r["completion_tokens"] >= mn)]
    caplines = sum(1 for r in d.values() if r["stopped_at_step_line_cap"])
    ct = [r["completion_tokens"] for r in d.values()]
    print(f"{c:60s} max_new={mn} quant={q} flag_mismatch={len(bad)} "
          f"step_line_cap_fired={caplines} max_ct={max(ct)} n_ct_eq_cap={sum(1 for x in ct if x >= mn)}")

print()
print("=== 1. per-cell cap-hit rate vs hop_count_EM ===")
rows = []
for c in CELLS:
    d, e = DEC[c], EV[c]
    ids = sorted(d)
    hit = np.array([d[i]["hit_max_new_tokens"] for i in ids], dtype=bool)
    hem = np.array([e[i]["hop_count_exact_match"] for i in ids], dtype=float)
    gold = np.array([e[i]["gold_hop_count"] for i in ids], dtype=int)
    bal = c.split("_")[1].split("2000")[-1] if False else c.split("size2000_")[1].split("_trial0")[0]
    var, mode = c.split("__")[1], c.split("__")[2]
    rows.append(dict(cell=c, balance=bal, variant=var, mode=mode,
                     n_hit=int(hit.sum()), cap_rate=float(hit.mean()),
                     hop_em=float(hem.mean()),
                     hop_em_hit=float(hem[hit].mean()) if hit.sum() else float("nan"),
                     hop_em_nohit=float(hem[~hit].mean()),
                     n_nohit=int((~hit).sum())))
    print(f"{bal:11s} {var:18s} {mode:8s} n_hit={hit.sum():3d} cap={hit.mean():.4f} "
          f"hopEM={hem.mean():.4f} hopEM|hit={rows[-1]['hop_em_hit']:.4f} hopEM|nohit={rows[-1]['hop_em_nohit']:.4f}")

cap = np.array([r["cap_rate"] for r in rows])
hem = np.array([r["hop_em"] for r in rows])
from scipy import stats as st
pr, pp = st.pearsonr(cap, hem)
sr, sp = st.spearmanr(cap, hem)
print(f"\n18-cell Pearson r={pr:.4f} p={pp:.4f} ; Spearman rho={sr:.4f} p={sp:.4f}")

HEAD = {b: f"size2000_{b}_trial0__biencoder_plus_ce__typed" for b in ["imbalanced", "balanced", "clustered"]}
print("\nheadline 3 cells:")
for b, c in HEAD.items():
    r = [x for x in rows if x["cell"] == c][0]
    print(f"  {b:11s} n_hit={r['n_hit']:3d} cap={r['cap_rate']:.4f} hopEM={r['hop_em']:.4f}")
hc = np.array([[x for x in rows if x["cell"] == HEAD[b]][0]["cap_rate"] for b in HEAD])
hh = np.array([[x for x in rows if x["cell"] == HEAD[b]][0]["hop_em"] for b in HEAD])
print("  3-point Pearson r =", round(float(np.corrcoef(hc, hh)[0, 1]), 4), "(n=3, descriptive only)")

print()
print("=== 2. cap-hit by gold hop depth (all 18 cells) ===")
for c in CELLS:
    d, e = DEC[c], EV[c]
    ids = sorted(d)
    gold = np.array([e[i]["gold_hop_count"] for i in ids])
    hit = np.array([d[i]["hit_max_new_tokens"] for i in ids], dtype=bool)
    parts = " ".join(f"h{h}={int(hit[gold == h].sum()):3d}/{int((gold == h).sum())}" for h in [2, 3, 4])
    print(f"{c:60s} {parts}  tot={int(hit.sum())}")

print("\nheadline cells: chi-square of cap-hit x gold hop")
for b, c in HEAD.items():
    d, e = DEC[c], EV[c]
    ids = sorted(d)
    gold = np.array([e[i]["gold_hop_count"] for i in ids])
    hit = np.array([d[i]["hit_max_new_tokens"] for i in ids], dtype=bool)
    tab = np.array([[int(hit[gold == h].sum()), int((~hit)[gold == h].sum())] for h in [2, 3, 4]])
    chi2, p, dof, _ = st.chi2_contingency(tab)
    print(f"  {b:11s} table(hit,nohit by hop 2/3/4)={tab.tolist()} chi2={chi2:.3f} p={p:.4g}")

print("\npooled over 3 headline cells (union of cells, not paired):")
tot = np.zeros((3, 2), dtype=int)
for b, c in HEAD.items():
    d, e = DEC[c], EV[c]
    ids = sorted(d)
    gold = np.array([e[i]["gold_hop_count"] for i in ids])
    hit = np.array([d[i]["hit_max_new_tokens"] for i in ids], dtype=bool)
    for j, h in enumerate([2, 3, 4]):
        tot[j, 0] += int(hit[gold == h].sum()); tot[j, 1] += int((~hit)[gold == h].sum())
chi2, p, dof, _ = st.chi2_contingency(tot)
print(f"  {tot.tolist()} rates={[round(tot[j,0]/(tot[j,0]+tot[j,1]),4) for j in range(3)]} chi2={chi2:.3f} p={p:.4g}")

print()
print("=== 3. decisive test: restricted paired comparisons (ADR 0009 protocol) ===")
from musique_decompositions_evaluator import _mcnemar, _paired_bootstrap, _paired_t_test_row  # noqa

ALPHA, ITERS, CHUNK, SEED, MINN = 0.05, 10000, 1000, 42, 30


def compare(a_cell, b_cell, keep_ids, label):
    ea, eb = EV[a_cell], EV[b_cell]
    ids = sorted(keep_ids)
    ra = [ea[i] for i in ids]
    rb = [eb[i] for i in ids]
    n = len(ids)
    under = n < MINN
    mc = _mcnemar(ra, rb, ALPHA, under, statistics=("exact_match", "hop_count_exact_match"))
    h = mc["hop_count_exact_match"]
    va = np.array([float(r["hop_count_exact_match"]) for r in ra])
    vb = np.array([float(r["hop_count_exact_match"]) for r in rb])
    tt = _paired_t_test_row(va, vb, ALPHA, under, "hop_count_exact_match")
    # paired bootstrap on the hop-count rate (mean of the binary vector), same index matrix
    rng = np.random.default_rng(SEED)
    diffs = np.empty(ITERS)
    done = 0
    while done < ITERS:
        take = min(CHUNK, ITERS - done)
        idx = rng.integers(0, n, size=(take, n))
        diffs[done:done + take] = va[idx].mean(axis=1) - vb[idx].mean(axis=1)
        done += take
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"{label:44s} n={n:3d} A={h['system_a_rate']:.4f} B={h['system_b_rate']:.4f} "
          f"diff={h['difference']:+.4f} b={h['correct_only_in_a']} c={h['correct_only_in_b']} "
          f"McNemar p={h['p_value']:.4f} sig={h['significant']} minp={h['min_attainable_p_value']:.4g} "
          f"| boot95=[{lo:+.4f},{hi:+.4f}] sig={bool(lo>0 or hi<0)} "
          f"| t p={tt.get('p_value')}")
    return dict(label=label, n=n, a=h["system_a_rate"], b=h["system_b_rate"], diff=h["difference"],
                b_disc=h["correct_only_in_a"], c_disc=h["correct_only_in_b"], mcnemar_p=h["p_value"],
                mcnemar_sig=h["significant"], min_p=h["min_attainable_p_value"],
                boot_lo=float(lo), boot_hi=float(hi), boot_sig=bool(lo > 0 or hi < 0),
                t_p=tt.get("p_value"))


ALL = sorted(EV[HEAD["imbalanced"]])
out = {}
for pair in [("imbalanced", "balanced"), ("clustered", "balanced"), ("imbalanced", "clustered")]:
    A, B = HEAD[pair[0]], HEAD[pair[1]]
    hitA = {i for i in ALL if DEC[A][i]["hit_max_new_tokens"]}
    hitB = {i for i in ALL if DEC[B][i]["hit_max_new_tokens"]}
    union = hitA | hitB
    inter = hitA & hitB
    keep_union = [i for i in ALL if i not in union]
    print(f"\n-- {pair[0]} vs {pair[1]}: hitA={len(hitA)} hitB={len(hitB)} union={len(union)} inter={len(inter)}")
    out[f"{pair[0]}_vs_{pair[1]}_full"] = compare(A, B, ALL, f"{pair[0]} vs {pair[1]} FULL")
    out[f"{pair[0]}_vs_{pair[1]}_nocap"] = compare(A, B, keep_union, f"{pair[0]} vs {pair[1]} NO-CAP(union-drop)")
    # truncated-only subset, for direction
    if union:
        out[f"{pair[0]}_vs_{pair[1]}_caponly"] = compare(A, B, sorted(union), f"{pair[0]} vs {pair[1]} CAP-HIT-ONLY")

print()
print("=== 4. power check: random subsets of the same size as the no-cap set ===")
for pair in [("imbalanced", "balanced"), ("clustered", "balanced")]:
    A, B = HEAD[pair[0]], HEAD[pair[1]]
    hitA = {i for i in ALL if DEC[A][i]["hit_max_new_tokens"]}
    hitB = {i for i in ALL if DEC[B][i]["hit_max_new_tokens"]}
    keep_n = 750 - len(hitA | hitB)
    rng = np.random.default_rng(SEED)
    sig = 0; ps = []
    for _ in range(1000):
        idx = rng.choice(750, size=keep_n, replace=False)
        ids = [ALL[i] for i in idx]
        ra = [EV[A][i] for i in ids]; rb = [EV[B][i] for i in ids]
        mc = _mcnemar(ra, rb, ALPHA, False, statistics=("hop_count_exact_match",))["hop_count_exact_match"]
        ps.append(mc["p_value"]); sig += mc["significant"]
    ps = np.array(ps)
    print(f"{pair[0]} vs {pair[1]}: random n={keep_n} subsets (1000 draws, seed 42): "
          f"McNemar sig in {sig}/1000 ({sig/10:.1f}%), median p={np.median(ps):.4f}, "
          f"p25={np.percentile(ps,25):.4f} p75={np.percentile(ps,75):.4f}")

print()
print("=== 5. does truncation actually change hop count? (mechanism) ===")
for b, c in HEAD.items():
    d, e = DEC[c], EV[c]
    ids = sorted(d)
    hit = np.array([d[i]["hit_max_new_tokens"] for i in ids], dtype=bool)
    pred = np.array([e[i]["predicted_hop_count"] for i in ids])
    gold = np.array([e[i]["gold_hop_count"] for i in ids])
    err = pred - gold
    print(f"{b:11s} hit: n={hit.sum()} mean_pred_steps={pred[hit].mean():.2f} mean_gold={gold[hit].mean():.2f} "
          f"mean_signed_err={err[hit].mean():+.2f} over_rate={(err[hit]>0).mean():.3f} under_rate={(err[hit]<0).mean():.3f}")
    print(f"{'':11s} nohit: n={(~hit).sum()} mean_pred_steps={pred[~hit].mean():.2f} mean_gold={gold[~hit].mean():.2f} "
          f"mean_signed_err={err[~hit].mean():+.2f} over_rate={(err[~hit]>0).mean():.3f} under_rate={(err[~hit]<0).mean():.3f}")

json.dump({"per_cell": rows, "comparisons": out}, open(os.path.join(os.path.dirname(__file__), "trunc_out.json"), "w"), indent=1)
print("\nwrote trunc_out.json")
