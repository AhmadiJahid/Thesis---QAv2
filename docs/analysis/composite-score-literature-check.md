# Is v1's composite decomposition-quality score defensible? A literature and bias check

**Date:** 2026-08-20 · **Author:** analyst agent session · **Base commit:** `b0a9ce8` ·
**Scope:** issue [#29](https://github.com/AhmadiJahid/Thesis---QAv2/issues/29); feeds issue #6
item 5 (thesis-primary metric) · **Runs:** none — no model was run, so there is no
`experiments/log.md` entry. The measurements below are a re-analysis of v1 per-item artifacts
under [ADR 0020](../adr/0020-prior-work-re-analysis-convention.md).

Origin of the task: the supervisor's own words at the 2026-08-12 meeting, [46:18] — *"This
composite score is handmade… Maybe it's biased. Let's put a note"*
([`docs/meetings/2026-08-12-supervisor-meeting-transcript-crosscheck.md`](../meetings/2026-08-12-supervisor-meeting-transcript-crosscheck.md)
item 4, triaged into #29 by [ADR 0017](../adr/0017-triage-of-the-2026-08-12-transcript-cross-check.md)).

---

## Read this before quoting any number below

- **This note recommends; it does not decide.** §5 is a *ranked* list of options with expected
  impact, and it does take positions — it ranks answer EM/F1 plus a no-blend panel highest, and it
  recommends **against** keeping the composite unchanged as thesis-primary. What it does not do is
  settle anything: the thesis-primary-metric choice is issue #6 item 5 and is **explicitly deferred
  to Jahid and his supervisor**, whose call overrides every ranking here. Every finding in §4 is a
  property of the metric's formula and of v1's numbers; none of it authorises a change to the
  pipeline.
- **Every measured input is a v1 artifact** from `/cta/users/fyilmaz/Thesis---QA` (read-only
  here, unmodified). Those files carry **no commit SHA**; Gate 2 is not satisfied; they are
  citable *prior work*, never v2 measurements. All six are pinned by content in §0.1 below (ADR
  0020 condition 2).
- **What is new here is the arithmetic**, computed in this session by a session-local harness
  that **imports the v2 evaluator's own functions** (`_step_prf`, `_ordered_step_accuracy`,
  `_reference_validity`, `_composite_score`, `_REF_RX`) so no formula is re-implemented. The
  harness is **not committed** (analyst output is analysis, not code), so — as with the two
  earlier v1 notes — the verifiable artifact is the machine-readable companion beside this file:
  [`composite-score-literature-check.json`](composite-score-literature-check.json) (statistics
  only, no dataset content, no question or prediction text).
- **Validation of the harness (ran; observed).** Recomputing the composite from the per-item
  columns reproduces v1's own published `composite_score` **bit-identically for all 3 runs**:
  typed 0.3606219114, uniform 0.3553568422, raw 0.1888136724, each matching
  `eval_*_unguided_metrics.json` to < 1e-9 (`probe_A_recomputation.*.recompute_matches_published_to_1e-9`
  = true, 3/3). So the numbers below describe the metric v1 actually reported.
- **Every literature claim in §3 was fetched in this session**, and the URL fetched is quoted
  next to it. §6 separates verified-fetched citations from recalled-but-unverified ones.

### 0.1 Inputs, pinned by content (ADR 0020 condition 2)

All six live in `Thesis---QA/runs/musique_decomposition_eval/` — untracked in v1, hence no commit
SHA. sha256 shown to the first 16 hex; **mtimes are UTC** (the masking note's table gives the same
instants in local time, +03:00).

| role | file | sha256[:16] | mtime (UTC) | bytes |
|---|---|---|---|---|
| per-item, typed | `eval_typed_unguided_per_item.json` | `53ab0a08f380db9a` | 2026-04-13 10:07:57 | 632325 |
| per-item, uniform | `eval_uniform_unguided_per_item.json` | `a0348260dd4a7d73` | 2026-04-13 10:08:14 | 635252 |
| per-item, raw | `eval_raw_unguided_per_item.json` | `80e64a38b7934c52` | 2026-04-13 10:07:52 | 636752 |
| metrics, typed | `eval_typed_unguided_metrics.json` | `c4c9634b9287e354` | 2026-04-13 10:07:57 | 3714 |
| metrics, uniform | `eval_uniform_unguided_metrics.json` | `4cff2acebaa9eac2` | 2026-04-13 10:08:14 | 3807 |
| metrics, raw | `eval_raw_unguided_metrics.json` | `c4a4093bdd479aab` | 2026-04-13 10:07:52 | 3843 |

The three per-item and three metrics hashes are **identical to the ones tabulated in**
[`2026-08-20-v1-masking-and-retrieval-significance.md`](2026-08-20-v1-masking-and-retrieval-significance.md)
§2 — the two notes analyse byte-identical inputs, so §4's figures and that note's significance
verdicts describe the same three runs. The metrics files are used only for the §0 cross-check, not
as a source of any statistic.

The one v2 input is **`configs/musique_eval.json`**, from which the weights and scale are read:
sha256 `7b37a2e6fc7667f9e73c67b7b5271cdd537d155764ea931438a6c9dbb0c02c9c`, pinned by the base
commit `b0a9ce8` (verified equal to the git blob content at that commit), not by an mtime.

---

## 1. The composite, exactly as implemented

Source of truth: `scripts/musique_decompositions_evaluator.py::_composite_score` (lines 563–570 at
`b0a9ce8`), weights from `configs/musique_eval.json`, documented in
[`docs/METRICS.md`](../METRICS.md) §4.

```
step_count_term = max(0.0, 1.0 - step_count_abs_error_mae / scale)

composite = 0.4 * step_f1_macro
          + 0.3 * ordered_step_accuracy_macro
          + 0.2 * reference_validity_micro
          + 0.1 * step_count_term          scale = 3.0
```

Five properties of the construction matter for everything below, and all five are read off the
code rather than inferred:

1. **It is an aggregate of aggregates.** `_composite_score` takes the *overall* metric block, not
   per-item values. There is therefore **no per-item composite**, which is why the repo's own
   comparison battery gives it a bootstrap CI but **no paired t-test and no McNemar**
   (`T_TEST_STATISTICS` excludes it; `docs/METRICS.md` §5). It is the only headline metric that
   cannot take two of the three tests in the house protocol.
2. **Three of the four terms are macro averages; the reference term is a micro rate** pooled over
   all rows (`reference_validity_micro`). The composite therefore mixes two aggregation regimes.
   Two computations that both call themselves an average of F1-like quantities can diverge
   substantially and even re-rank systems — the general point of *Macro F1 and Macro F1*
   (fetched: https://arxiv.org/abs/1911.03347).
3. **`step_f1_macro` is set-based and unordered** (`_step_prf`: sets of normalized steps,
   duplicates collapse), while `ordered_step_accuracy_macro` is positional. The unordered term
   carries the larger weight (0.4 vs 0.3).
4. **The length term is direction-blind by construction**: it consumes
   `step_count_abs_error_mae`, an MAE, so over- and under-decomposition are penalised identically.
   `docs/METRICS.md` §4 already flags this; ADR 0017 records the supervisor's contrary judgment
   ([33:23]) that over-decomposition is tolerable and under-decomposition is not.
5. **Reference validity scores an item 1.0 when it emits no `[#k]` at all**, and contributes
   nothing to the micro denominator (`_reference_validity` returns `rate = 1.0` when
   `total == 0`; the micro rate is 1.0 when there are no references anywhere).

The weights are a **choice, not a result** — hard-coded literals in v1, promoted to config in v2
so a run records them (`configs/musique_eval.json` `_note`). No document in either repo records a
derivation, a calibration, or a validation of 0.4 / 0.3 / 0.2 / 0.1 or of scale 3.0. **Whether
the composite correlates with human judgment of decomposition quality is unmeasured**, in v1 and
in v2.

---

## 2. What the composite is being asked to do

Two different questions get conflated in "is it defensible":

- **As a reporting convenience** (one number to skim a 33-cell sweep with) it is unobjectionable,
  provided the underlying block is reported beside it. That is roughly how v1 used it.
- **As the thesis-primary metric** — the number a claim of improvement rests on, and the number a
  supervisor and an external examiner read — it has to survive the biases in §4. That is the
  question #6 item 5 asks.

§4 is written against the second use.

---

## 3. What published work actually uses (all citations fetched this session)

### 3.1 Multi-hop QA benchmarks: the gold decomposition is *supervision*, and the reported metric is the *answer*

- **MuSiQue** — *MuSiQue: Multihop Questions via Single-hop Question Composition* (Trivedi et al.,
  TACL 2022). Fetched: https://arxiv.org/abs/2108.00573 and
  https://ar5iv.labs.arxiv.org/html/2108.00573. Its Metrics paragraph, verbatim: *"For ♫-Ans, HQ,
  and 2W, we report the standard F1 based metrics for answer (An) and support identification (Sp);
  see Yang et al. 2018 for details."* The gold decomposition `G_Q` is described as something
  *"which can be leveraged during training"* — **not** as a scored output. **MuSiQue's own paper
  defines no intrinsic decomposition-quality metric at all.** This repo's answering backend
  implements those official answer metrics (`src/answer_metrics.py`, ADR 0019).
- **HotpotQA** (Yang et al., EMNLP 2018). Fetched: https://arxiv.org/abs/1809.09600 and
  https://ar5iv.labs.arxiv.org/html/1809.09600. Reports answer EM/F1, supporting-fact EM/F1, and
  **joint** EM/F1 — and the joint metric is **multiplicative and per-example**:
  `P_joint = P_ans · P_sup`, `R_joint = R_ans · R_sup`, then the harmonic mean; *"Joint EM is 1
  only if both tasks achieve an exact match"*; *"Intuitively, these metrics penalize systems that
  perform poorly on either task. All metrics are evaluated example-by-example, and then averaged
  over examples."* This is the closest published thing to "combine two components into one
  score", and it differs from v1's composite on all three axes: **product not sum, per-example
  not aggregate-of-aggregates, and no free credit for an unattempted component.**
- **StrategyQA** (Geva et al., TACL 2021). Fetched: https://arxiv.org/abs/2101.02235 and
  https://ar5iv.labs.arxiv.org/html/2101.02235. Decompositions are annotated, and their quality is
  evaluated **extrinsically**: QA accuracy with predicted vs gold vs no decomposition, plus
  **Recall@10** of the gold evidence paragraphs (*"We compute Recall@10, i.e., the fraction of the
  gold paragraphs retrieved in the top-10 results of each method"*), plus a qualitative read of
  predicted decompositions (their example: a decomposition that is *"grammatical and
  well-structured"* but *"uses a wrong strategy"*). **No intrinsic string-overlap score of the
  decomposition is reported anywhere in it.** That last observation cuts both ways for this
  thesis and is worth carrying into the write-up: a purely string-level metric of the kind
  `docs/METRICS.md` documents cannot see the failure mode StrategyQA's authors found most common.

### 3.2 Question-decomposition parsing (Break / QDMR): a *panel* of metrics, and a single canonicalized EM as the official one

- **Break / QDMR** — *Break It Down: A Question Understanding Benchmark* (Wolfson et al., TACL
  2020). Fetched: https://arxiv.org/abs/2001.11770 and https://ar5iv.labs.arxiv.org/html/2001.11770.
  It names the exact problem v1's composite tries to paper over — *"Figure 7 lists various
  properties by which question decompositions may differ, such as granularity … ordering … and
  wording"* — and answers it with **four metrics reported side by side, never blended**: Exact
  Match; **SARI** (added / kept / deleted n-gram F1, following Xu et al. 2016); **GED**, a
  normalized graph edit distance over the QDMR graph with substitution cost `1 − Align(u,v)`; and
  **GED+**, adding node merge/split operations (computed only for graphs with ≤ 5 nodes, *"covering
  75.2% of the examples"*). SARI's origin is verified separately: *Optimizing Statistical Machine
  Translation for Text Simplification* (Xu et al., TACL 2016), fetched:
  https://aclanthology.org/Q16-1029/.
- **Question Decomposition with Dependency Graphs** (Hasson & Berant, \*SEM 2021). Fetched:
  https://arxiv.org/abs/2104.08647 and https://ar5iv.labs.arxiv.org/html/2104.08647. Two facts
  from it are directly load-bearing here. First, *"The official evaluation metric for QDMR … is
  normalized EM (NormEM), where the predicted and gold QDMR structures are normalized using a
  rule-based procedure, and then exact string match is computed"* — **the field's official
  decomposition metric is a single canonicalized binary EM, not a blend.** Second, they propose
  LF-EM and **validate it against human judgment**: *"We manually evaluate the metrics normalized
  EM and LF-EM on 50 random development set examples … both (binary) metrics have perfect
  precision … LF-EM on this sample is 52.0, while normalized EM is 40.0."* Their canonicalization
  explicitly neutralises benign step reordering — *"there are multiply [sic] ways to order the
  steps … We then re-order steps by layer and then lexicographically"*. That is the literature's answer to
  the ordering problem: **canonicalize, then match — not add an ordering term with a weight.**

### 3.3 Prompted decomposition methods: the reported number is downstream accuracy

Every prompted-decomposition method checked reports task performance, not a decomposition score:
**Least-to-Most Prompting** (fetched: https://arxiv.org/abs/2205.10625), **Decomposed Prompting**
(*"outperform prior work on few-shot prompting"* — fetched: https://arxiv.org/abs/2210.02406),
**Self-Ask / the compositionality gap** (fetched: https://arxiv.org/abs/2210.03350), and **IRCoT**
(fetched: https://arxiv.org/abs/2212.10509).

### 3.4 Reasoning-chain evaluation: suites of scores, meta-evaluated against humans

- **ROSCOE** (Golovneva et al., ICLR 2023). Fetched: https://arxiv.org/abs/2212.07919 and
  https://ar5iv.labs.arxiv.org/html/2212.07919. A *suite* of interpretable scores, deliberately
  **not** collapsed into one number, and each one **meta-evaluated against human and synthetic
  scores with Somers' D** (*"We use Somers' D … to meta-evaluate each scorer against synthetic and
  human scores"*).
- **ReCEval** (Prasad et al., EMNLP 2023). Fetched: https://arxiv.org/abs/2304.10703. Argues the
  opposite of a pure extrinsic metric: *"Most existing methods focus solely on whether the
  reasoning chain leads to the correct conclusion, but this answer-oriented view may confound
  reasoning quality with other spurious shortcuts to predict the answer."* This is the strongest
  published argument for keeping an intrinsic decomposition metric **alongside** an answer metric
  rather than replacing it.
- **HELM** (Liang et al., 2022). Fetched: https://arxiv.org/abs/2211.09110. *"we adopt a
  multi-metric approach"* … *"This ensures metrics beyond accuracy don't fall to the wayside, and
  that trade-offs are clear"* — i.e. report the panel, make the trade-off visible, do not blend.

### 3.5 On blending metrics into one score

- **The Benchmark Lottery** (Dehghani et al., 2021). Fetched: https://arxiv.org/abs/2107.07002 and
  https://ar5iv.labs.arxiv.org/html/2107.07002. §3.2 *Score and rank aggregation* is squarely on
  point: averaging *"a model that performs poorly on any of the bundled tasks may be set up for
  failure"*; *"the `and` operator is an inductive bias"*; aggregation implicitly *"upweighting
  tasks with more headroom"*; and rankings demonstrably change with the aggregation choice. Their
  suggested mitigations are geometric mean, macro-grouping, and rank aggregation.
- **Dynaboard / the Dynascore** (Ma et al., NeurIPS 2021). Fetched: https://arxiv.org/abs/2106.06052
  and https://ar5iv.labs.arxiv.org/html/2106.06052. **The one verified published precedent for a
  weighted-average composite** — and note what it does that v1's does not: it **converts every
  metric into common units first**, via an estimated average marginal rate of substitution
  (*"By dividing M by AMRS(M, perf), we can convert it to units of performance"*), it states a
  principled default (*"our default weighting places half the weight on the canonical performance
  metric and splits the remaining half among the others"*), and it makes the weights
  **user-adjustable with live re-ranking** so the weight-dependence of a conclusion is visible
  rather than hidden.

### 3.6 Bottom line on precedent

**In this check, no published work uses a hand-weighted linear blend of decomposition sub-metrics
as its primary decomposition-quality metric.** The dominant published patterns are (i) report the
answer/downstream metric (MuSiQue, StrategyQA, all four prompting papers), (ii) report a panel of
decomposition metrics separately (Break, ROSCOE, HELM), and (iii) where a single decomposition
number is wanted, use a **canonicalized exact match** (NormEM / LF-EM) validated against human
judgment. The nearest thing to a blend that is published (HotpotQA's joint F1) is multiplicative
and per-example; the nearest thing to a weighted average (Dynascore) unitises its inputs and
exposes its weights. **v1's composite therefore has no direct precedent — it is not thereby
wrong, but it does mean it carries the full burden of justification itself,** and §4 is what that
burden looks like.

---

## 4. Biases and degenerate cases, each with a measured example

All figures from the JSON companion; n = 600 throughout (v1's MuSiQue clean-dev eval set, 200 each
of 2/3/4 hop). "Degenerate systems" are constructed from the `gold_steps` column of
`eval_typed_unguided_per_item.json` and scored against that same gold, so every comparison is
within one item set.

### 4.1 The 0.2 reference term is decided by a handful of observations

Measured emission counts (`probe_C_reference_emission`): items emitting at least one `[#k]` are
**3/600 (typed), 4/600 (uniform), 1/600 (raw)**; total references 8, 9 and 4. So a term carrying
**20% of the primary metric** is determined by **8, 9 and 4 observations** respectively, while the
other 592–599 items contribute nothing to it. The macro variant is a mirage for the same reason —
0.9967, 0.9967, 0.9983 — as `docs/prior-work.md` §3 already notes.

### 4.2 Not attempting the chaining scores strictly better than attempting it imperfectly

Strip every `[#k]` from the predictions and rescore (`probe_B_strip_references`): the composite
**rises** in all three runs — typed **0.3606 → 0.4106 (+0.0500)**, uniform **0.3554 → 0.3996
(+0.0443)**, raw **0.1888 → 0.3888 (+0.2000)**.

The sharpest version (`probe_G_sensitivity.one_invalid_reference_on_one_item_of_600`): take the
reference-free typed run (composite 0.4106) and inject **one invalid `[#k]` into one item out of
600**. The composite drops to **0.2106 — a loss of exactly 0.2000**, because that single item now
owns the entire micro denominator. In step-F1 currency that loss equals a
`step_f1_macro` drop of 0.5, i.e. **300 of 600 items losing a perfect step F1**
(`probe_G_sensitivity.exchange_rates`). A metric in which one item's placeholder syntax outweighs
half the evaluation set is not measuring what its name says.

This is exactly what happened to v1's raw run: its single reference-emitting item
(4 references, 0 valid) cost it 0.2 composite points, whereas the entire typed-vs-raw **step F1**
gap (0.0232) is worth **0.0093** composite points — a **21.6×** ratio
(`probe_G_sensitivity.v1_typed_vs_raw`). The already-committed significance note reached the same
conclusion from the other direction (the gap is not significant and is 87% one fragile term:
[`2026-08-20-v1-masking-and-retrieval-significance.md`](2026-08-20-v1-masking-and-retrieval-significance.md)
§4).

### 4.3 The floor is not zero, and a real system can rank below a null one

Measured (`probe_D_degenerate_systems`): **empty prediction → 0.2000**; **one fixed junk step →
0.2333**; **echo the question back as a single step → 0.2333**. All three collect the full 0.2
reference term for emitting nothing, plus a slice of the length term.

v1's raw unguided run scored **0.1888** — **below all three null systems.** Under this metric,
predicting nothing at all would have beaten a real 600-item decomposition run by 0.011.

### 4.4 A content-perfect but maximally mis-ordered plan keeps 73% of the oracle score

Because the unordered set-F1 term (0.4) outweighs the positional term (0.3), reversing every gold
decomposition scores **0.7333** (step F1 1.0000, ordered accuracy 0.1111,
`probe_D_degenerate_systems.gold_order_reversed`) against the oracle's 1.0000. For a thesis whose
subject is *sequential* decomposition with `[#k]` chaining, the metric weights order-insensitivity
above order. Note the literature's alternative (§3.2): Break/QDMR treats benign reordering by
**canonicalizing** the structure before matching, not by trading one term off against another.

### 4.5 Duplicate padding is cheaper than adding a new step, because the set collapses duplicates

`gold + duplicate of the last step` → **0.8883**; `gold + one junk step` → **0.8278**. Both have
step-count MAE 1.0 and identical ordered accuracy (0.7389); the duplicate keeps `step_f1_macro` at
**1.0000** because `_step_prf` deduplicates, while the junk step drops it to 0.8487. A decomposer
that pads by repeating a step is penalised **less** than one that pads with new content.

### 4.6 The length term cannot express the asymmetry the supervisor stated

`probe_E_direction_blindness`: over-decomposition (duplicate the last gold step) and
under-decomposition (drop the last gold step) both give MAE 1.0 and therefore **identical
step-count-term contributions of 0.0666667** (`step_count_term_contributions_identical` = true).
The two composites do differ (0.8883 vs 0.7682) — but only through the other terms, and only
because dropping a step also destroys step F1 and ordered accuracy. Stated honestly: **the
composite is not globally direction-blind, but the term designed to price length error is**, and
nothing in the composite encodes ADR 0017's "over is tolerable, under is not". The directional
family in `docs/METRICS.md` §2 exists precisely because the composite cannot carry it.

### 4.7 The length term is nearly a constant offset at the observed operating point

Term contributions across the three v1 runs: **0.0766 / 0.0741 / 0.0704** — a spread of 0.0062
across three systems whose composites span 0.17. So 10% of the metric's weight buys ~3% of its
observed variation. `scale = 3.0` also saturates: MAE ≥ 3 zeroes the term, which the empty
prediction hits exactly (MAE 3.0000 on this gold). Neither 3.0 nor the saturation point is
documented as derived from anything.

### 4.8 A headline v1 conclusion is a function of the weights

Sweeping all weightings on the 4-simplex that sum to 1 (`probe_H_weight_simplex`), recomputing
each run's composite from its own aggregate block:

| ordering | grid step 0.1 (286 weightings) | grid step 0.05 (1771 weightings) |
|---|---|---|
| typed > raw | 286/286 (100.0%) | 1771/1771 (100.0%) |
| uniform > raw | 286/286 (100.0%) | 1771/1771 (100.0%) |
| **typed > uniform** | **206/286 (72.0%)** | **1292/1771 (73.0%)** |

So "typed beats raw" is weight-robust, while **"typed beats uniform" fails in ~27–28% of
weightings** — the ranking of v1's two masked retrieval modes is partly a property of the chosen
0.4/0.3/0.2/0.1. This is the Benchmark Lottery §3.2 effect (§3.5) reproduced on this metric. It is
also consistent with the committed note's finding that typed vs uniform decides nothing at n=600
by significance either.

### 4.9 The composite cannot take two of the house protocol's three tests

Because it is an aggregate of aggregates, `--compare` gives it a bootstrap CI but **no paired
t-test and no McNemar** (`docs/METRICS.md` §5; `T_TEST_STATISTICS` excludes it). A primary metric
that is the least testable of the reported metrics is an awkward position to defend in November,
and it is the mechanical reason the two committed v1 notes had to invent the
`composite_no_ref_renorm` diagnostic (`docs/METRICS.md` §4.1) to say anything about it.

### 4.10 Where headroom actually lives

At v1's typed operating point (`probe_F_headroom_at_v1_typed`), remaining headroom per term is:
`step_f1` **0.3197**, `ordered` **0.2462**, `reference_micro` **0.0500**, `step_count` **0.0234**
(total 0.6393). So **89%** of the achievable improvement sits in the two step-level terms — while, per §4.2, the
metric's largest single available *swing* is in the reference term. The metric's sensitivity and
its headroom point in different directions.

### 4.11 What is **not** wrong with it

For balance, three things this check did not find:
- It is correctly bounded: the oracle (gold verbatim) scores exactly **1.0000**.
- It is faithfully implemented: v1's published composites reproduce bit-identically (§0).
- The weights are recorded per run and enforced across compared runs (`configs/musique_eval.json`,
  `_require_matching_weights`), so a composite built from different weights cannot be silently
  compared — a real improvement v2 made over v1.

**Unmeasured, and stated as such:** the composite's correlation with human judgment (never
measured, in v1 or v2); its correlation with the answer EM/F1 the answering backend produces (no
run has produced both on the same eval set); and whether any of §4's degenerate cases reproduce on
a v2 run (no v2 decomposition run has completed — exp-004/exp-005 are PENDING in
`experiments/log.md`).

---

## 5. Ranked options for the thesis-primary metric — for Jahid and his supervisor to decide

Ranked by **expected impact on defensibility per unit of work**, with the direction of impact and
the reason. **This is a list, not a choice.** Issue #6 item 5 is explicitly deferred to the
supervisor meeting, and nothing below should be read as adopted.

Excluded up front, and permanently: **closed commercial models as judges/scorers of decomposition
quality.** The supervisor rejected that as not scientifically defensible (CLAUDE.md standing
constraint; `docs/prior-work.md` §8). It is not an option and is not costed here.

1. **Answer EM / F1 from the answering backend as thesis-primary, reported overall and per gold
   hop depth, with the intrinsic panel (§option 2) reported beside it.**
   *Expected direction:* largest gain in defensibility. *Why:* it is **MuSiQue's own official
   metric** (§3.1) and matches what every prompted-decomposition paper checked reports (§3.3), so
   it is comparable to published numbers instead of only to itself; it has a per-item value, so the
   full ADR 0009 battery (bootstrap + McNemar + t-test) applies; and it is already implemented and
   convention-pinned (`components/answerer/run_answerer.py`, `src/answer_metrics.py`, ADR 0019).
   *Costs and risks:* one answering run per condition (GPU, and `docs/compute.md` records
   contention); it **confounds decomposer quality with answerer quality** — exactly ReCEval's
   objection (§3.4), which is why it is paired with the intrinsic panel rather than replacing it;
   and it changes what "primary" means relative to every v1 number.

2. **Report a small fixed panel of intrinsic metrics jointly, with no blend, and a stated
   tie-break rule.** Candidate panel, all already computed: `step_f1_macro`,
   `ordered_step_accuracy_macro`, `rouge_l_f1_macro`, `hop_count_exact_match_rate`, and the
   over/under-decomposition rates **kept separate** per ADR 0017, plus reference validity **always
   quoted with its emission denominator** (e.g. "6/8 references over 3/600 items").
   *Expected direction:* removes every §4 bias that comes from *combining* (4.2, 4.3, 4.7, 4.8) at
   **zero implementation cost** — the evaluator already emits all of it. *Why:* it is the dominant
   published pattern (Break's four metrics, ROSCOE's suite, HELM's explicit multi-metric stance,
   §3.2/§3.4). *Costs:* no single number to rank a 33-cell sweep on, so the sweep needs a stated
   objective; and "we report five numbers" is weaker rhetorically in a thesis abstract than one.

3. **Keep a composite, but repair the construction, and report it under a weight sweep.** Three
   independent repairs, in descending measured impact: **(a)** replace `reference_validity_micro`
   with a **per-item** chaining score that penalises *missing* chaining where the gold requires it
   (an item whose gold uses `[#k]` and whose prediction emits none scores 0, not 1) — this alone
   removes the free 0.2 (§4.2, §4.3) and gives the composite a per-item value, which makes the
   McNemar and t-test in §4.9 applicable; **(b)** make the length term **directional** per ADR 0017
   so under-decomposition costs more than over-decomposition (§4.6); **(c)** publish the composite
   with a **weight-sensitivity sweep** rather than a single weighting, as Dynaboard does with
   user-adjustable weights (§3.5) — §4.8 shows this changes a conclusion.
   *Expected direction:* converts the composite from indefensible to arguable while keeping one
   headline number. *Costs:* it is a **shared-pipeline change** (Gate 1 review), it **invalidates
   comparability with every v1 composite** including the whole pool sweep, and the weights are
   still ultimately handmade — the supervisor's original objection survives (a), (b) and (c).

4. **Adopt a canonicalized exact match (NormEM / LF-EM style) as the single intrinsic primary.**
   *Expected direction:* strongest *literature* backing of any intrinsic option — it is the
   **official** QDMR metric, and LF-EM was validated against human judgment (§3.2). It is binary
   and per-item, so McNemar applies directly. *Costs:* real implementation work — MuSiQue steps are
   free-form natural language, not QDMR logical forms, so the canonicalizer (step reordering by
   dependency layer, `[#k]` normalisation, lexical normalisation) has to be built and its own
   validity argued; and v1's **unadjusted exact-match metric is 0.037–0.058 across the three retrieval
   variants** (`docs/prior-work.md` §3 — not to be confused with the variant *named* `raw`, whose
   exact match alone is 0.0367), so a floor effect is likely, which
   suppresses the very comparisons the thesis needs to make.

5. **Replace the weighted sum with a multiplicative per-example joint score**, in HotpotQA's form:
   compute components per item, multiply, then average over items (§3.1).
   *Expected direction:* removes the free-credit and single-item-swing pathologies structurally
   (a component that is 0 cannot be bought back by another), and restores a per-item value.
   *Costs:* which two or three components to join is a new arbitrary choice; the score floors hard
   at v1's operating point (step F1 ≈ 0.20 × ordered ≈ 0.18 → ≈ 0.036), which may be too
   compressed to discriminate; and no published precedent exists for applying it to
   *decompositions* specifically — the precedent is answer × support.

6. **Whichever of 1–5 is chosen, validate it once against human judgment on a small annotated
   sample** — the ROSCOE and LF-EM pattern (§3.2, §3.4): a few dozen items, rank correlation
   (Somers' D or Kendall τ) between the metric and Jahid's own quality judgments.
   *Expected direction:* this is the **cheapest single action that answers the supervisor's actual
   question** ("maybe it's biased"), because it converts an argument about construction into a
   measurement. LF-EM's own validation used **50 examples**. *Costs:* human annotation time, and it
   is a companion to options 1–5 rather than a metric in itself. *Note:* the annotation must be
   human — a closed commercial model doing it is the excluded method above.

**Not recommended, and stated plainly:** keeping the composite unchanged as the thesis-primary
metric. §4.2, §4.3 and §4.8 are the reasons — a metric under which one item's placeholder syntax
outweighs half the evaluation set, under which predicting nothing beats a real run, and under
which a headline retrieval conclusion depends on the weight choice, is a metric an examiner can
dismantle in one question. Retaining it as a **secondary reporting convenience** beside a panel
(option 2) carries none of that risk, because the burden it has to bear drops accordingly.

---

## 6. Citations — verified-fetched vs recalled-unverified

**Verified-fetched this session (16).** Each was retrieved in this session and the fetched URL is
quoted at the point of use in §3; abstracts were read for all, and full text for the eight marked ✚
(from which the verbatim metric definitions in §3 are quoted).

| # | Work | Fetched URL(s) |
|---|---|---|
| 1 ✚ | MuSiQue (Trivedi et al., TACL 2022) | https://arxiv.org/abs/2108.00573 · https://ar5iv.labs.arxiv.org/html/2108.00573 |
| 2 ✚ | HotpotQA (Yang et al., EMNLP 2018) | https://arxiv.org/abs/1809.09600 · https://ar5iv.labs.arxiv.org/html/1809.09600 |
| 3 ✚ | StrategyQA (Geva et al., TACL 2021) | https://arxiv.org/abs/2101.02235 · https://ar5iv.labs.arxiv.org/html/2101.02235 |
| 4 ✚ | Break / QDMR (Wolfson et al., TACL 2020) | https://arxiv.org/abs/2001.11770 · https://ar5iv.labs.arxiv.org/html/2001.11770 |
| 5 ✚ | Question Decomposition with Dependency Graphs (Hasson & Berant, 2021) | https://arxiv.org/abs/2104.08647 · https://ar5iv.labs.arxiv.org/html/2104.08647 |
| 6 | SARI (Xu et al., TACL 2016) | https://aclanthology.org/Q16-1029/ |
| 7 ✚ | ROSCOE (Golovneva et al., 2022) | https://arxiv.org/abs/2212.07919 · https://ar5iv.labs.arxiv.org/html/2212.07919 |
| 8 | ReCEval (Prasad et al., 2023) | https://arxiv.org/abs/2304.10703 |
| 9 | HELM (Liang et al., 2022) | https://arxiv.org/abs/2211.09110 |
| 10 ✚ | The Benchmark Lottery (Dehghani et al., 2021) | https://arxiv.org/abs/2107.07002 · https://ar5iv.labs.arxiv.org/html/2107.07002 |
| 11 ✚ | Dynaboard / Dynascore (Ma et al., 2021) | https://arxiv.org/abs/2106.06052 · https://ar5iv.labs.arxiv.org/html/2106.06052 |
| 12 | Macro F1 and Macro F1 (Opitz & Burst, 2019) | https://arxiv.org/abs/1911.03347 |
| 13 | Least-to-Most Prompting (Zhou et al., 2022) | https://arxiv.org/abs/2205.10625 |
| 14 | Decomposed Prompting (Khot et al., 2022) | https://arxiv.org/abs/2210.02406 |
| 15 | Self-Ask / compositionality gap (Press et al., 2022) | https://arxiv.org/abs/2210.03350 |
| 16 | IRCoT (Trivedi et al., 2022) | https://arxiv.org/abs/2212.10509 |

Titles, dates, abstracts **and author lists** came from the arXiv Atom API
(`https://export.arxiv.org/api/query?id_list=...`) and, for #6, from the ACL Anthology page — so the
first-author attributions above are fetched, not recalled (e.g. #5's entry lists exactly
`['Matan Hasson', 'Jonathan Berant']`, #12's `['Juri Opitz', 'Sebastian Burst']`, #6's anthology
citation line names Wei Xu, Courtney Napoles, Ellie Pavlick, Quanze Chen, Chris Callison-Burch).
**Venue labels are the conventional ones for these IDs and were only spot-checked** against the
fetched pages (e.g. #3's and #6's TACL lines); a venue label is not load-bearing for any argument
in §3.

**Recalled but not verified in this session (0).** Nothing was cited from memory. Two things
deliberately *omitted* for that reason, and named so a later pass can pick them up: the Break
leaderboard's current official-metric page (the NormEM definition here comes from #5's quotation
of it, not from the leaderboard itself), and any WMT-metrics-shared-task methodology for
meta-evaluating a metric against human judgment (the pattern in option 6 is cited to #5 and #7,
which were fetched).

---

## 7. Reproducing §4

The harness is session-local and uncommitted, so — following
[`2026-08-20-v1-pool-size-significance.md`](2026-08-20-v1-pool-size-significance.md) — the
verifiable artifact is [`composite-score-literature-check.json`](composite-score-literature-check.json).
It records the base commit, the composite construction actually used (weights + scale read from
`configs/musique_eval.json`, pinned by that commit), the sha256 / mtime / size of all six v1 input
files — also inlined in §0.1 per ADR 0020 condition 2 — the alignment statement, and every number quoted in §4 under the probe key it came from
(`probe_A_recomputation` … `probe_H_weight_simplex`). Independent recomputation is the check, as it
was for the two earlier v1 notes; §0's bit-identical reproduction of v1's three published
composites is the harness's own validation. Deriving §4 from committed code would need the
evaluator to accept a synthetic prediction file constructed from gold — which is a pipeline change
and therefore not this lane's to make. This probe family is recorded as the sixth harness-only
family in [ADR 0020](../adr/0020-prior-work-re-analysis-convention.md)'s Consequences.
