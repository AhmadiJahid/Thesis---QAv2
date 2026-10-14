# Results section of the Beamer deck — redesign plan

**Date:** 2026-10-14 · **Refs:** #50 · **Status:** proposal for Jahid. The deck is untouched and nothing is committed.
**Prototype:** `Slides/proposals/results-charts-prototype.pdf` (8 pages). Every plotted number is printed by
`python3 Slides/results_chart_data.py`, which reads `experiments/<exp-id>/metrics.json` (plus, labelled, three runs/ compare files for C2's paired-t p-values).

Scope: deck frames 19–35 (`Slides/thesis-deck-v2.tex` l.1147–1545). The audit checked every number against the JSON or the log.

---

## 1. Audit of the current results slides

"✓" means every number on the slide matches its source at the precision printed.

| # | Slide | Claim | Source | Numbers | Highlighting | Eval set / model |
|---|---|---|---|---|---|---|
| 20 | Masking | Typed leads on every plan metric | v1 re-analysis `docs/analysis/2026-08-20-v1-masking-and-retrieval-significance.json` `task_a_masking` | ✓ table. **Callout ✗**: "2.1 points of EM" is **2.2** (`typed_vs_raw` diff +0.0217) | Bold on every typed cell. No rule is stated, and the bold reads as "wins". Typed vs **uniform** is n.s. on all 5 metrics. Typed vs raw is significant on EM, step F1, ordered accuracy and ROUGE-L, and n.s. on hop EM. | Pinned 600 (v1 ids, ADR 0007), Mistral-7B, **v1 prior work** |
| 20 | (callout) | "No significance test has been run on typed vs uniform" | same JSON | **✗ wrong.** It was tested: EM p=0.85, step F1 CI [−0.006, +0.029], all 5 n.s. | — | — |
| 21 | By hop (v1) | Quality falls with every hop | same JSON `per_gold_hop` (system_a/b) | ✓ (18 cells) | Bold = best in column. The rule is stated and applied correctly. | v1 600 |
| 21 | (callout) | v2 typed re-run "0.710/0.515/0.300" | `exp-004.eval_metrics.unguided.per_gold_hop_metrics.*.hop_count_exact_match_rate` | ✓, but these are **hop-count EM**, while the sentence before them is about step F1. The metric needs a label. | — | Mixes v1 and v2 runs in one sentence |
| 22 | Pool size | 1,000 is too small; nothing above separates | `docs/analysis/2026-08-20-v1-pool-size-significance.md` §4.1 cell means | ✓ (0.1897/0.1841/0.1816/0.1698). **Caption ✗**: "Exact match stays flat across all cells" is wrong. EM cell means rise 0.0316→0.0469, and EM 2000 vs 1000 (imbal/BE/typed, +0.024) **survives Holm** (adj p 0.0036 t / 0.0048 McNemar), §4.3 | 1,000 is grey and the rest are accent. No rule is stated. The bars are cell means with no CI. | **v1 750 sample** (seed 42, 250/hop). Whether these ids match v2's pool_sweep 750 is unverified. |
| 23 | Pool construction | Every stage differs, but no metric moves | exp-019 log row | ✓ (Jaccard 0.0529, 201/2000, 0/750, 0/150). "92%" is a range of **688–700/750 = 92–93%** | Gold on 92%. No rule is stated. | **pool_sweep 750** (405/600 overlap with pinned). The footer flags this correctly. |
| 24 | Reranking | +CE moves overlap and structure | exp-018 log row | ✓ (8 sign counts, 8 Holm counts) | **Inconsistent**: SARI is accent with 0 Holm survivors, and hop-count is grey with 8/9 positive and 0 Holm. Under the proposed rule, accent goes only to metrics with ≥1 Holm survivor (ROUGE-L 3, GED 4, step F1 1, ordered 1). | pool_sweep 750 |
| 25 | Hop guidance | A perfect count helps one model far more | exp-004/005 `eval_metrics`, `exp-014.deliverable_1_compare.pairs` | ✓ (12 values; "6 of 9" = 7/10 minus legacy composite) | Gold+bold = significant vs unguided, and it is applied correctly (Mistral hop EM p=0.0002; Qwen step F1, hop EM, EM p=0.039). But the section uses **bold** for this on other slides. | **Callout ✗ "same conditions"**: exp-004 is fp16 (`quantization: none`) and exp-005 is 4-bit, so the model contrast also moves precision. exp-005 (Qwen3.5-9B) uses the v1 retrieval artifact. exp-027 (also Qwen3.5-9B, unguided, 4-bit) uses the exp-014 artifact, and its numbers differ (hop EM 0.520 vs 0.570). **Never show these two as one arm.** |
| 26 | Hop-matched retrieval | Moves count, not quality | `exp-009.compare_mixed_vs_oracle_hop_matched`, exp-014 | ✓ | Gold on 0.600 = significant ✓ (McNemar p=0.0006). **Test mix ✗**: the ordered-accuracy row quotes the **paired t-test** p=0.950. The other rows quote the bootstrap/McNemar result, so this row should be CI [−0.016, +0.017]. | Pinned 600, Mistral fp16 |
| 27 | By hop, hop-matched | Hop-matching hurts at 4 hops | `exp-009.per_hop` | ✓ (18 cells) | **✗**: gold+bold on oracle hop EM at every depth, and **bad** red on 4-hop 0.071/0.060. No per-hop test exists (the footer says so). The colours assert verdicts that were never tested. | Pinned 600 |
| 28 | Fine-tuning | Fine-tuning beats prompting at its best | `exp-015.compare_vs_exp008_full_train` | ✓ (0.189/0.545/0.341/0.665) | Bold = best ✓. "10/10, seven bootstrap" counts the **retired composite_score**. The report-card count is 9/9. | Pinned 600, both fp16 |
| 29 | Generalisation | Transfers form, not depth | `exp-013.per_gold_hop_metrics`, `comparison_A…` | ✓ (1.000, 169/200, 14.5 vs 30.0 McNemar p=4.7e-5, 0.198 vs 0.117) | Bad on 14.5% = significantly worse ✓ | 4-hop 200. Omitted: vs full FT at 4 hops, hop EM is **14.5 vs 41.0** (p=5.9e-10, `comparison_B…`) |
| 30 | End to end | Better plans give better answers up to 3 hops | `exp-017.cells.*` | ✓ (20 values) | **✗**: gold+bold on the gold-plan cells and red on 13.5. Yet "no significance tests between these cells" is stated on the same slide, and the headline is an untested claim. | Pinned 600, reader Mistral-7B |
| 32 | Gap in evidence | "All but one arm is Mistral-7B" | — | **Stale.** exp-021/022/024/025/027/028 are Qwen. | — | — |
| 33 | Also queued | 4 things queued | — | **Stale.** Pool-only FT (exp-026), the anchor set and reversed order (exp-023/024/025/028) are done. Complexity order did not run. | — | — |
| 34 | Honest boundaries | "Typed vs uniform untested" | v1 JSON | **✗** It was tested (n.s.). | — | — |

**Not in the deck at all:** the whole #50 block (exp-020 … exp-028). The block report's §0 line "biggest movement anywhere 2.5 pp" is
**superseded**: exp-028 vs exp-027 moved chain validity **+0.0349** (CI [+0.0041, +0.0652], significant). Use §9 of the report, not §0.

**Mixed evaluation sets.** These must never share an axis: the pinned 600 (everything except 22–24), the pool_sweep 750 (exp-010/018/019),
and the v1 750 (slide 22). **Mixed models/precision in one comparison:** slide 25 (fp16 Mistral vs 4-bit Qwen), and fine-tunes
(fp16 eval) vs 4-bit prompting arms (exp-020/021/022/027). The second is stated on the slide; the first is not.

---

## 2. Proposed section structure (ordered by question, not by exp number)

The prototype page is given in brackets. Every slide states its eval set and its named baseline in the source line.

| # | Question → slide | One-sentence claim | Chart | Data series (source) | Highlighted, and by what rule |
|---|---|---|---|---|---|
| A1 | Prompt: masking | Typed beats raw on content (4 of 5 metrics); typed vs uniform is not separable | Δ-forest vs raw (typed−raw, uniform−raw), 5 metrics | v1 JSON `task_a_masking.pairs.*` diff + CI | * where the CI excludes 0 vs raw. Badge: "v1 prior work". |
| A2 | Prompt: pool size | Only contrasts with 1,000 separate | Line, step F1 (and EM) cell mean vs size 1k→8k | v1 pool-size §4.1 table | * only on Holm-surviving contrasts (§4.3). Badge: "v1 750 sample". |
| A3 | Prompt: pool construction | A real null: every stage differs, no metric moves | Keep the stat row. Fix "92–93%". | exp-019 log | No colour verdicts. Badge: "750". |
| A4 | Prompt: reranking | +CE moves ROUGE-L and GED in 9/9 pairs | Dot strip: 9 pair Δs per metric, zero line | `docs/analysis/2026-08-27-pool-construction-slide-facts.json` | * = metric with ≥1 Holm survivor. Badge: "750". |
| A5 | Prompt: example order (1/2) [p.2] | Reversing order moves 2 of 28 report-card tests | Δ-forest, 4 models × SARI/GED/chain/step F1 | exp-023/024/025/028 `compare_vs_*_as_is.bootstrap` | * = CI excludes 0 vs the as-is twin |
| A6 | Prompt: example order (2/2) [p.3] | The model sets the length error; the order does not | Dumbbell, as-is → reversed, hop EM/under/over/Break EM | same, `.mcnemar` | None significant (exact McNemar). "Up to 10%" shown as the prior. |
| A7 | Prompt: hop-matched retrieval | Count moves, quality does not | Dumbbell overall + per-hop lines (no colour) | exp-009 `overall`, `per_hop` | * only on the pooled-600 tests. Per-hop shown plain. |
| B1 | Guidance: oracle hop count | Helps Qwen3.5-9B on content and count; Mistral on count, ROUGE-L and GED only | Dumbbell, unguided → oracle, 2 models | exp-004/005, exp-014 pairs | * vs unguided. The source line states the fp16/4-bit and retrieval differences. |
| C1 | Model: plan length [p.4] | Prompting runs long or short; fine-tunes almost never run long | Diverging bars, under ← → over | exp-020/021/022/027/026/008 rates | None (magnitudes). The tests are on C2. |
| C2 | Model: one factor at a time [p.5] | Family and size move chaining, not wording | Δ-forest, 4-bit−fp16 / family / size / confounded row | exp-020/021/022/027 `compare_vs_*` | * vs the row's reference arm. The confounded row is grey. |
| D1 | Fine-tuning vs prompting [p.6] | Every prompting model trails both fine-tunes | Bars from zero, step F1 + hop EM | 7 arms. Tests from `compare_vs_exp008_full_train*` | * = differs from full FT |
| D2 | Fine-tuning by depth [p.7] | FT leads at every depth but stops short at 4 hops | Lines over 2/3/4 hops, step F1 + under | `per_gold_hop_*` (6 arms) | None. Descriptive, with no per-hop test. |
| D3 | Fine-tuning: unseen depth | Transfers form, not depth | Keep the stat trio. Add the vs-full-FT 4-hop row. | exp-013 comparisons A and B | `bad` on 14.5% (significantly worse vs both baselines) |
| E1 | End to end [p.8] | Better plans lift answers at 2–3 hops; at 4 they converge | Lines over hops, 4 sources | exp-017 `cells.*.per_gold_hop_metrics` | None. "No test run" is stated. |
| F1 | What is still open | Replaces slides 32 and 33 | Text list | #50 report §6 | — |
| — | Honest boundaries / To agree on | Keep. Fix "typed vs uniform untested". | — | — | — |

16 result slides plus 3 closing slides, against 11 + 4 today.

---

## 3. One highlighting rule for the whole section (stated once, on the divider)

> **An accent mark (an accent number in a table, an accent \* beside a chart mark) means the difference is significant against
> the baseline named on that slide. Bold means best in its column, and is used only in tables. Nothing else is coloured for emphasis.
> `bad` (red) appears only on a value that is significantly worse. Entity colours identify arms and carry no judgement.**

- **Tests** (ADR 0009, as emitted by `scripts/musique_decompositions_evaluator.py --compare`): a paired bootstrap 95% percentile CI
  (10,000 resamples, seed 42) for continuous metrics, and exact McNemar (α 0.05) for EM, hop-count EM, Break EM, under- and
  over-decomposition. The paired t-test is never the basis for a highlight.
- **No multiplicity correction** unless the slide says *Holm*. Only A2 (v1 pool size) and A4 (rerank) use Holm, as their sources did.
- **Per-hop numbers are never highlighted**, because no per-hop test exists for v2 arms. The exception is A1, where the v1 JSON carries per-hop tests.
- **The legacy composite_score is not counted** in any "k of n significant" tally (see Q4).
- **Entity colours**, fixed order, in the prototype preamble: Mistral-7B `B07D2A`, Qwen2.5-7B `1F5BA8`, Qwen2.5-14B `A07ADB`,
  Qwen3.5-9B `C2417F`, fine-tune `1E6B3C`. These were checked with `validate_palette.py` (python, `--mode light --surface #F5F6F4 --pairs all`):
  every check passed, with one CVD WARN (7.8, Qwen3.5-9B↔Qwen2.5-7B). Every series therefore also carries a label or a distinct marker.
  Node is not installed on the box, so the `.js` validator was not run. The `.py` twin was.

---

## 4. Open questions for Jahid

1. **Length.** The structure has 16 result slides against 11 today. Should A3, A4 and A7 (construction, rerank, hop-matched) go into one
   "retrieval knobs that did not matter" slide, or into an appendix?
2. **Eval sets.** Slides 22–24 sit on 750-question samples that are not the pinned 600. Should they stay in the section behind an "other eval set"
   badge, or move to an appendix? They can never share an axis with the 600-question results.
3. **v1 prior work.** Masking and pool size are v1 runs that were never re-run in v2. Keep them as cited prior work, or drop them until re-run?
4. **Legacy composite.** Should it be removed from every count? exp-015's "10/10" would become 9/9 (report-card plus beside-card), and the
   ordering tally would become 3 of 48 instead of 4 of 52. The prototype headline uses the report-card-only count, "2 of 28".
5. **Untested claims.** Slides D2 and E1 currently claim per-hop and end-to-end differences with no test behind them. Do you want a CPU-only
   re-analysis of existing per-item files first (per-hop paired bootstrap, McNemar on answer EM between exp-017 cells)? Otherwise the slides
   stay worded as descriptive.

*Noted, not asked:* exp-022, exp-025, exp-027 and exp-028 run above 8B. They load only under the decomposer exception (1.5e10), which is
still pending the supervisor's confirmation (CLAUDE.md). If he reasserts 8B, those rows leave A5, A6, C1, C2 and D1.
