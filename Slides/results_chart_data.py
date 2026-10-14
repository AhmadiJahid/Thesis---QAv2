#!/usr/bin/env python3
"""Print every number plotted or quoted in the results section of
Slides/thesis-deck-v2.tex, with the exp-id and JSON key it comes from.

Reads committed files: experiments/<exp-id>/metrics.json.  One exception,
labelled where it happens: the C2 Holm family needs the evaluator's paired-t
p-values for three compares whose committed metrics.json does not carry its
`t_test` block; those are read from the runs/ compare_metrics.json that the
committed metrics.json names under `source_files` (compute box only; the
reader cross-checks its bootstrap differences and McNemar p-values against
the committed block before using it).  No computation beyond reading,
rounding, counting the evaluator's own `significant` / `holm.significant`
flags, and Holm-Bonferroni over the evaluator's own p-values (differences are
the evaluator's `difference` fields, never re-derived).  Run from anywhere:

    python3 -I Slides/results_chart_data.py

Each block names the slide, the exp-id and the key it reads, so a number in
the .tex can be traced by searching this output.  Exit status is 0 only if
every key was found.

Highlight rule used by the deck (ADR 0037, stated on the results divider):
  accent *  significantly better than the slide's named baseline
  red *     significantly worse than it
  test      the evaluator's UNCORRECTED verdict: paired bootstrap 95% CI
            excludes 0, or exact McNemar p < 0.05
  words     titles/callouts claim only what survives Holm over the slide's
            family (ADR 0036: McNemar p for binary, paired-t p for continuous)
The `verdict` printed below applies the star rule; `holm` columns the words rule.
Exception: the exp-029 slides (A7, B1, D2, D3, E1) mark Holm survivors, as
their source lines say.

Not printable by this script (no experiments/<exp-id>/ directory exists):
  pool construction slide  -> exp-019, numbers in its experiments/log.md row
  reranking slide          -> exp-018, numbers in its experiments/log.md row
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOWER_IS_BETTER = {"ged", "under_decomposition", "over_decomposition"}
_cache = {}


def load(exp):
    if exp not in _cache:
        _cache[exp] = json.loads((ROOT / "experiments" / exp / "metrics.json").read_text())
    return _cache[exp]


def _norm(v):
    """Normalise the three schema generations of a --compare stat block."""
    diff = next(v[k] for k in ("difference", "difference_a_minus_b", "diff_a_minus_b") if k in v)
    if "ci" in v:
        lo, hi = v["ci"]
    else:
        lo, hi = v.get("ci_low"), v.get("ci_high")
    a = next((v[k] for k in v if k.startswith("system_a") and isinstance(v[k], float)), v.get("a", v.get("a_rate")))
    b = v.get("system_b", v.get("b", v.get("b_rate", v.get("system_b_rate"))))
    holm = v.get("holm", {})
    return dict(a=a, b=b, d=diff, lo=lo, hi=hi, p=v.get("p_value"), sig=v["significant"],
                holm_sig=holm.get("significant"), holm_p=holm.get("adjusted_p"))


def holm(tests, alpha=0.05):
    """Holm-Bonferroni step-down over [(label, p)].  Returns rows sorted by p:
    (label, p, adjusted_p, survives)."""
    order = sorted(tests, key=lambda t: t[1])
    m, out, running, stop = len(order), [], 0.0, False
    for i, (lab, p) in enumerate(order):
        running = max(running, min(1.0, (m - i) * p))
        stop = stop or p > alpha / (m - i)
        out.append((lab, p, running, not stop))
    return out


def print_holm(name, tests):
    rows = holm(tests)
    print(f"   Holm over the {name} family: size={len(rows)}; first threshold 0.05/{len(rows)}={0.05 / len(rows):.4f}")
    for lab, p, adj, ok in rows:
        print(f"     {lab:58s} p={p:.3g} holm_p={adj:.3g} survives={ok}")
    print(f"   {name}: {sum(r[3] for r in rows)} of {len(rows)} survive Holm")
    return rows


def tp(exp, key, metric):
    """Evaluator's paired-t p for a continuous metric, from the committed block."""
    return load(exp)[key]["t_test"][metric]["p_value"]


def tp_runs(exp, key, src_key, metric):
    """Paired-t p from the runs/ compare file the committed metrics.json names
    under source_files[src_key] (used only where the committed block has no
    t_test).  Cross-checks the runs file against the committed block first."""
    rel = load(exp)["source_files"][src_key]
    r = json.loads((ROOT / rel).read_text())
    c = load(exp)[key]
    for m, v in c["bootstrap"].items():
        if m in r["bootstrap"]:
            assert abs(_norm(v)["d"] - _norm(r["bootstrap"][m])["d"]) < 1e-9, (exp, m)
    for m, v in c["mcnemar"].items():
        if m in r["mcnemar"]:
            assert abs(_norm(v)["p"] - r["mcnemar"][m]["p_value"]) < 1e-12, (exp, m)
    return r["t_test"][metric]["p_value"], rel


def verdict(metric, d, sig):
    """'better' / 'worse' for system_a against system_b, or 'n.s.'."""
    if not sig:
        return "n.s."
    good = d < 0 if metric in LOWER_IS_BETTER else d > 0
    return "better" if good else "worse"


def boot(exp, key, metric):
    return _norm(load(exp)[key]["bootstrap"][metric])


def mcn(exp, key, metric):
    return _norm(load(exp)[key]["mcnemar"][metric])


def e29(pair, stratum, metric):
    s = load("exp-029")["pairs"][pair]["strata"][stratum]
    kind = "mcnemar" if metric in s["mcnemar"] else "bootstrap"
    return _norm(s[kind][metric]), kind


def e29line(pair, stratum, metric, star_for="a"):
    """star_for='b' when the slide's baseline is system_a (the star sits on b)."""
    s, kind = e29(pair, stratum, metric)
    sign = 1 if star_for == "a" else -1
    test = (f"ci=[{s['lo']:+.4f},{s['hi']:+.4f}]" if kind == "bootstrap" else f"p={s['p']:.2g}")
    return (s, f"a={s['a']:.4f} b={s['b']:.4f} d={s['d']:+.4f} {test} unc={s['sig']} "
               f"holm_p={s['holm_p']:.2g} holm={s['holm_sig']} star({star_for}, holm)={verdict(metric, sign * s['d'], s['holm_sig'])}")


# ------------------------------------------------------------ A5 / A6 ordering
ORDER_PAIRS = [  # (label, reversed exp, compare key) -- as-is twin is system_b
    ("Mistral-7B", "exp-023", "compare_vs_exp020_as_is"),
    ("Qwen2.5-7B", "exp-024", "compare_vs_exp021_as_is"),
    ("Qwen2.5-14B", "exp-025", "compare_vs_exp022_as_is"),
    ("Qwen3.5-9B", "exp-028", "compare_vs_exp027_as_is"),
]
CARD_BOOT = ("sari", "ged", "chain_validity")
CARD_MCN = ("break_exact_match", "hop_count_exact_match", "under_decomposition", "over_decomposition")


def slide_a5_a6():
    print("== A5 ordering forest: reversed minus as-is, paired bootstrap 95% CI (system_a = reversed)")
    for metric in ("sari", "ged", "chain_validity", "step_f1"):
        print(f"  -- {metric}")
        for lab, exp, key in ORDER_PAIRS:
            s = boot(exp, key, metric)
            print(f"   {lab:12s} {exp}.{key}.bootstrap.{metric}: d={s['d']:+.4f} "
                  f"ci=[{s['lo']:+.4f},{s['hi']:+.4f}] sig={s['sig']} verdict={verdict(metric, s['d'], s['sig'])}")
    n = sig = 0
    for lab, exp, key in ORDER_PAIRS:
        for m in CARD_BOOT:
            n += 1; sig += boot(exp, key, m)["sig"]
        for m in CARD_MCN:
            n += 1; sig += mcn(exp, key, m)["sig"]
    print(f"   A5 title count: report-card tests significant = {sig} of {n} (7 terms x 4 models, composite excluded)")
    fam = []
    for lab, exp, key in ORDER_PAIRS:
        for m in CARD_BOOT:
            fam.append((f"{lab} {m} (paired-t, {exp}.{key}.t_test)", tp(exp, key, m)))
        for m in CARD_MCN:
            fam.append((f"{lab} {m} (McNemar)", mcn(exp, key, m)["p"]))
    print_holm("A5", fam)
    print("== A6 ordering dumbbell: as-is rate -> reversed rate, exact McNemar")
    big = (0, "")
    nsig = 0
    for metric in ("hop_count_exact_match", "under_decomposition", "over_decomposition", "break_exact_match"):
        print(f"  -- {metric}")
        for lab, exp, key in ORDER_PAIRS:
            s = mcn(exp, key, metric)
            nsig += s["sig"]
            if abs(s["d"]) > big[0]:
                big = (abs(s["d"]), f"{lab} {metric}")
            print(f"   {lab:12s} {exp}.{key}.mcnemar.{metric}: as-is={s['b']:.4f} reversed={s['a']:.4f} "
                  f"d={s['d']:+.4f} p={s['p']:.3f} sig={s['sig']}")
    print(f"   A6: significant = {nsig} of 16; largest |shift| = {big[0]:.4f} ({big[1]})")


# ------------------------------------------------------------ A7 hop-matched retrieval
A7 = "A7|exp009_mixed_vs_oracle_hop_matched"


def slide_a7():
    print("== A7 hop-matched retrieval: exp-029.pairs[A7].strata.<s>; a = mixed (exp-009), b = oracle hop-matched")
    print("   Holm family A7 (16 tests); star is for b (hop-matched) against the baseline a (mixed)")
    for metric in ("hop_count_exact_match", "step_f1", "ordered_step_accuracy", "exact_match"):
        print(f"  -- {metric}")
        for st in ("overall", "2", "3", "4"):
            _, line = e29line(A7, st, metric, star_for="b")
            print(f"   {st:7s} {line}")
    f = load("exp-029")["holm_families"]["A7"]
    print(f"   exp-029.holm_families.A7: size={f['size']} holm_significant={f['holm_significant']}")


# ------------------------------------------------------------ B1 oracle hop count
B1 = [("Mistral-7B fp16 (exp-004)", "B1|mistral7b_exp004_unguided_vs_oracle_guided"),
      ("Qwen3.5-9B 4-bit (exp-005)", "B1|qwen35_9b_exp005_unguided_vs_oracle_guided")]
B1_METRICS = ("sari", "ged", "chain_validity", "step_f1", "ordered_step_accuracy", "rouge_l_f1",
              "exact_match", "hop_count_exact_match", "break_exact_match")


def slide_b1():
    print("== B1 oracle hop count: exp-029.pairs[B1|...].strata.<s>; a = unguided (baseline), b = oracle-guided; Holm (72)")
    for lab, pair in B1:
        print(f"  -- {lab}: hop_count_exact_match by stratum")
        for st in ("overall", "2", "3", "4"):
            _, line = e29line(pair, st, "hop_count_exact_match", star_for="b")
            print(f"   {st:7s} {line}")
        unc = holm = 0
        names = []
        for m in B1_METRICS:
            s, _ = e29(pair, "overall", m)
            unc += s["sig"]; holm += bool(s["holm_sig"])
            if s["holm_sig"]:
                names.append(m)
        print(f"   overall: {holm} of 9 survive Holm ({', '.join(names)}); {unc} of 9 uncorrected")
        for m in ("ged", "step_f1", "ordered_step_accuracy"):
            for st in ("overall", "2", "3", "4"):
                _, line = e29line(pair, st, m, star_for="b")
                print(f"   {m:22s} {st:7s} {line}")
    f = load("exp-029")["holm_families"]["B1"]
    print(f"   exp-029.holm_families.B1: size={f['size']} holm_significant={f['holm_significant']}")


# ------------------------------------------------------------ C1 / C2
def card(exp):
    d = load(exp)
    if "eval" in d and "decomposition_report_card_seven_terms" in d["eval"]:
        return {**d["eval"]["decomposition_report_card_seven_terms"], **d["eval"]["other_reported"]}, "eval.decomposition_report_card_seven_terms/other_reported"
    if "eval" in d and "decomposition_report_card_six_terms" in d["eval"]:
        return {**d["eval"]["decomposition_report_card_six_terms"], **d["eval"]["other_reported"]}, "eval.decomposition_report_card_six_terms/other_reported"
    if exp == "exp-026":
        return d["eval"], "eval"
    if exp == "exp-015":
        return d["prompting_arm_full_600"], "prompting_arm_full_600"
    if exp == "exp-008":
        return d["eval_metrics"], "eval_metrics"
    raise KeyError(exp)


ARMS = [("Mistral-7B", "exp-020"), ("Qwen2.5-7B", "exp-021"), ("Qwen2.5-14B", "exp-022"),
        ("Qwen3.5-9B", "exp-027"), ("FT pool-only", "exp-026"), ("FT full", "exp-008")]


def slide_c1():
    print("== C1 length errors: under / over rate, pinned 600")
    for lab, exp in ARMS:
        c, where = card(exp)
        print(f"   {lab:12s} {exp}.{where}: under={c['under_decomposition_rate']:.4f} over={c['over_decomposition_rate']:.4f}")


def slide_c2():
    print("== C2 one-factor contrasts (system_a minus system_b; b = the row's reference arm; uncorrected)")
    rows = [("precision 4-bit - fp16 (Mistral)", "exp-020", "compare_vs_exp015_unguided_quantization_none"),
            ("family Qwen2.5-7B - Mistral-7B", "exp-021", "compare_vs_exp020_mistral_7b_4bit"),
            ("size Qwen2.5-14B - Qwen2.5-7B", "exp-022", "compare_vs_exp021_qwen2_5_7b_4bit"),
            ("CONFOUNDED Qwen3.5-9B - Qwen2.5-7B", "exp-027", "compare_vs_exp021_qwen2_5_7b_4bit")]
    print("  plotted panels: under / over / break EM (McNemar, dot only) and chain validity (bootstrap CI)")
    for metric in ("under_decomposition", "over_decomposition", "break_exact_match", "hop_count_exact_match", "exact_match"):
        print(f"  -- {metric}")
        for lab, exp, key in rows:
            s = mcn(exp, key, metric)
            print(f"   {lab:36s} {exp}.{key}.mcnemar.{metric}: d={s['d']:+.4f} p={s['p']:.3g} "
                  f"sig={s['sig']} verdict={verdict(metric, s['d'], s['sig'])}")
    print("  source-line claim: SARI, GED, step F1 n.s. on the three one-factor rows")
    for metric in ("chain_validity", "sari", "ged", "step_f1"):
        print(f"  -- {metric}")
        for lab, exp, key in rows:
            s = boot(exp, key, metric)
            print(f"   {lab:36s} {exp}.{key}.bootstrap.{metric}: d={s['d']:+.4f} "
                  f"ci=[{s['lo']:+.4f},{s['hi']:+.4f}] sig={s['sig']} verdict={verdict(metric, s['d'], s['sig'])}")
    # Holm family = every test the slide reads (ADR 0036 item 2): the four plotted
    # panels on all four rows (16) + the source line's EM, hop-count EM, SARI, GED,
    # step F1 on the three one-factor rows (15) = 31.
    runs_src = {"exp-020": "compare_vs_exp015", "exp-021": "compare_vs_exp020", "exp-022": "compare_vs_exp021"}

    def t_p(exp, key, m):
        if "t_test" in load(exp)[key]:
            return tp(exp, key, m), f"{exp}.{key}.t_test"
        return tp_runs(exp, key, runs_src[exp], m)

    fam = []
    for i, (lab, exp, key) in enumerate(rows):
        mcn_metrics = ["under_decomposition", "over_decomposition", "break_exact_match"]
        t_metrics = ["chain_validity"]
        if i < 3:
            mcn_metrics += ["exact_match", "hop_count_exact_match"]
            t_metrics += ["sari", "ged", "step_f1"]
        for m in mcn_metrics:
            fam.append((f"{lab.split()[0]} {m} (McNemar)", mcn(exp, key, m)["p"]))
        for m in t_metrics:
            p, where = t_p(exp, key, m)
            fam.append((f"{lab.split()[0]} {m} (paired-t, {where.split('/')[1] if where.startswith('runs/') else 'committed'})", p))
    print("  paired-t source: exp-027 committed t_test; exp-020/021/022 from runs/<exp>/.../compare_metrics.json"
          " named in the committed source_files (cross-checked)")
    print_holm("C2", fam)


# ------------------------------------------------------------ D1 / D2 / D3
def slide_d1():
    print("== D1 fine-tune vs prompting: absolute values + test vs exp-008 full_train (verdict for the arm vs exp-008)")
    vs008 = {"exp-015": "compare_vs_exp008_full_train", "exp-020": "compare_vs_exp008_full_train",
             "exp-021": "compare_vs_exp008_full_train", "exp-022": "compare_vs_exp008_full_train",
             "exp-027": "compare_vs_exp008_full_train", "exp-026": "compare_vs_exp008_full_train_rescored"}
    for lab, exp in [("Mistral-7B fp16", "exp-015")] + ARMS:
        c, where = card(exp)
        line = f"   {lab:16s} {exp}.{where}: step_f1={c['step_f1_macro']:.4f} hop_em={c['hop_count_exact_match_rate']:.4f}"
        if exp in vs008:
            k = vs008[exp]
            sb, sm = boot(exp, k, "step_f1"), mcn(exp, k, "hop_count_exact_match")
            line += (f" | {exp}.{k}: step_f1 verdict={verdict('step_f1', sb['d'], sb['sig'])}"
                     f" hop_em verdict={verdict('hop_count_exact_match', sm['d'], sm['sig'])} p={sm['p']:.2g}")
        print(line)
    print(f"   callout: exp-026.training.wall_clock_seconds={load('exp-026')['training']['wall_clock_seconds']:.1f}"
          f" num_examples={load('exp-026')['training']['dataset']['num_examples']}")
    print(f"            exp-001.wall_clock_seconds={load('exp-001')['wall_clock_seconds']:.1f}"
          f" num_examples={load('exp-001')['dataset']['num_examples']}  (exp-008 evaluates the exp-001 adapter)")


D2_ARMS = [("Mistral-7B", "exp020_mistral7b_4bit"), ("Qwen2.5-7B", "exp021_qwen25_7b_4bit"),
           ("Qwen2.5-14B", "exp022_qwen25_14b_4bit"), ("Qwen3.5-9B", "exp027_qwen35_9b_4bit")]
D2_FTS = ("exp008_full_train", "exp026_pool_only_ft")


def slide_d2():
    print("== D2 by gold hop depth (values: same keys as C1/D1 per_gold_hop_*); tests exp-029 D2, Holm (48)")
    for lab, exp in ARMS:
        d = load(exp)
        if exp == "exp-008":
            ph, where = d["eval_metrics"]["per_gold_hop_metrics"], "eval_metrics.per_gold_hop_metrics"
        elif exp == "exp-026":
            ph, where = d["eval"]["per_gold_hop_metrics"], "eval.per_gold_hop_metrics"
        else:
            ph, where = d["eval"]["per_gold_hop_report_card"], "eval.per_gold_hop_report_card"
        sf = "/".join(f"{ph[h]['step_f1_macro']:.4f}" for h in "234")
        ud = "/".join(f"{ph[h]['under_decomposition_rate']:.3f}" for h in "234")
        print(f"   {lab:12s} {exp}.{where}.<2|3|4>: step_f1={sf} under={ud}")
    print("  star = verdict vs BOTH fine-tunes under Holm (prompting arm = system_a)")
    for metric in ("step_f1", "under_decomposition"):
        n_holm = 0
        for lab, arm in D2_ARMS:
            for h in "234":
                vs = []
                for ft in D2_FTS:
                    s, _ = e29(f"D2|{arm}_vs_{ft}", h, metric)
                    n_holm += bool(s["holm_sig"])
                    vs.append(verdict(metric, s["d"], s["holm_sig"]))
                star = vs[0] if vs[0] == vs[1] else "none (" + "/".join(vs) + ")"
                print(f"   {metric:20s} {lab:12s} hop {h}: vs full={vs[0]:6s} vs pool-only={vs[1]:6s} -> star: {star}")
        print(f"   {metric}: {n_holm} of 24 survive Holm")


D3 = [("vs exp-004 unguided", "D3|exp013_gen_ft_vs_exp004_unguided_hop4"),
      ("vs exp-008 full FT", "D3|exp013_gen_ft_vs_exp008_full_train_hop4")]


def slide_d3():
    print("== D3 generalisation, hop=4 (n=200): exp-029.pairs[D3|...].strata.4; a = exp-013 gen-FT; Holm (8)")
    for metric in ("chain_validity", "under_decomposition", "hop_count_exact_match", "step_f1"):
        for lab, pair in D3:
            _, line = e29line(pair, "4", metric)
            print(f"   {metric:22s} {lab:20s} {line}")
    s, _ = e29(D3[0][1], "4", "under_decomposition")
    print(f"   169/200 check: under rate {s['a']:.3f} x 200 = {s['a'] * 200:.0f}")


# ------------------------------------------------------------ E1 end to end
E1_CELLS = ("no_decomposition_control", "unguided", "full_train", "oracle")
E1_PAIRS = [("gold - prompted", "E1|exp017_oracle_vs_exp017_unguided"),
            ("gold - none", "E1|exp017_oracle_vs_exp017_no_decomposition_control"),
            ("gold - fine-tuned", "E1|exp017_oracle_vs_exp017_full_train"),
            ("fine-tuned - prompted", "E1|exp017_full_train_vs_exp017_unguided"),
            ("fine-tuned - none", "E1|exp017_full_train_vs_exp017_no_decomposition_control"),
            ("prompted - none", "E1|exp017_unguided_vs_exp017_no_decomposition_control")]


def slide_e1():
    print("== E1 end to end (exp-017), answer EM by gold hop")
    cells = load("exp-017")["cells"]
    for cell in E1_CELLS:
        ph = cells[cell]["per_gold_hop_metrics"]
        em = "/".join(f"{ph[h]['answer_em']:.3f}" for h in "234")
        print(f"   {cell:26s} exp-017.cells.{cell}: overall={cells[cell]['answer_em']:.4f} by hop={em}")
    print("  survivors matrix: exp-029.pairs[E1|...].strata.<s>.{mcnemar.answer_em, bootstrap.answer_f1}.holm.significant")
    total = 0
    for lab, pair in E1_PAIRS:
        row = []
        for st in ("overall", "2", "3", "4"):
            em, _ = e29(pair, st, "answer_em")
            f1, _ = e29(pair, st, "answer_f1")
            total += bool(em["holm_sig"]) + bool(f1["holm_sig"])
            row.append(f"{st}: EM {em['d']:+.3f}{'*' if em['holm_sig'] else ' '} F1 {f1['d']:+.3f}{'*' if f1['holm_sig'] else ' '}")
        print(f"   {lab:22s} " + " | ".join(row))
    h4 = sum(bool(e29(p, "4", m)[0]["holm_sig"]) for _, p in E1_PAIRS for m in ("answer_em", "answer_f1"))
    print(f"   * = survives Holm; total {total} of 48; at 4 hops {h4} of 12")


if __name__ == "__main__":
    slide_a5_a6(); slide_a7(); slide_b1(); slide_c1(); slide_c2()
    slide_d1(); slide_d2(); slide_d3(); slide_e1()
