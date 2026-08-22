# Where the fine-tuned decomposer actually wins, and what the composite score is really measuring

Error analysis of **exp-004** (Mistral-7B prompting, 3 conditions), **exp-005** (Qwen3.5-9B
prompting, 3 conditions) and **exp-008** (exp-001's `full_train` LoRA on Mistral, zero-shot) on the
**same** eval set: the ADR 0007 pinned 600 MuSiQue questions, 200 per gold hop depth, seed 42.

Statistics companion beside this note:
[`2026-08-22-finetuned-vs-prompting-error-analysis.json`](2026-08-22-finetuned-vs-prompting-error-analysis.json).
Every number quoted here is in it, keyed by the section that uses it, together with the sha256/size/mtime
of all seven per-item inputs and the evaluator's own weights and seed.

**This note recommends and ranks. It does not decide.** Nothing below is a research direction, a
metric replacement, or a router keep/drop call — those are Jahid's with his supervisor.

---

## 0. Read this before quoting any number

- **Sources.** Aggregates come from the committed `experiments/exp-00{4,5,8}/metrics.json`. Everything
  per-item comes from the seven evaluator per-item files under `runs/` (paths and hashes in the
  companion's `provenance.inputs_pinned_by_content`). `runs/` is ignored by git, so the companion
  pins those inputs by content.
- **Recomputation used the evaluator's own functions.** `_step_prf`, `_ordered_step_accuracy`,
  `_exact_decomposition_match`, `_reference_validity`, `_rouge_l`, `_statistic_arrays`,
  `_statistics_for` were imported from `scripts/musique_decompositions_evaluator.py`. No metric was
  reimplemented.
- **The bootstrap here is not a re-run — it is the committed one, decomposed.** With seed 42, 10000
  iterations, chunk 1000, items aligned by sorted `item_id`, the reproduction is **bit-identical** to
  `experiments/exp-008/metrics.json`: `full_train_vs_unguided` composite difference
  `0.313721990971991`, CI `[0.10547500797813295, 0.33070602055352055]`;
  `oracle_guided_vs_unguided` difference `0.2098428964553965`, CI
  `[0.005844228977041489, 0.21888995490620491]`. So §3's mode decomposition decomposes the published
  intervals, not a lookalike.
- **`composite_no_ref_renorm`** is the named house *diagnostic* of `docs/METRICS.md` §4.1 (drop the
  reference term, renormalize the other three to 1: 0.5 / 0.375 / 0.125). Per that section it is
  reported *beside* the house composite, never instead of it, and no thesis claim may rest on it
  alone. Where the two disagree, the disagreement is the finding.
- **Two things below are explicitly simulations, not runs:** the cap sweep (§5) and the fuzzy-EM
  probe (§6). Both truncate or fuzzy-compare *already-generated* text. A generation-time cap would
  also change what the model decodes, so the sweep bounds a post-hoc edit, not a real condition.
- **Unmeasured, and stated as such:** downstream answer accuracy for any arm; whether any hypothesis
  in §8 actually moves a metric; `pool_2000` and `generalisation_2_3hop` (never trained, per exp-008's
  log row).

---

## 1. Headline: the composite's 0.2 reference term is decided by two items, and it is doing most of the work

Decomposing each arm's committed composite into its four weighted terms
(companion `finding_1`):

| arm (log row) | step F1 ×0.4 | ordered ×0.3 | **ref micro ×0.2** | length ×0.1 | **house composite** | `no_ref_renorm` |
|---|---|---|---|---|---|---|
| exp-004 `unguided` | 0.0816 | 0.0541 | **0.0000** | 0.0742 | **0.2098** | 0.2623 |
| exp-004 `oracle_guided` | 0.0825 | 0.0556 | **0.2000** | 0.0817 | **0.4197** | 0.2746 |
| exp-004 `unguided_capped` | 0.0816 | 0.0542 | **0.0000** | 0.0771 | **0.2128** | 0.2661 |
| exp-005 `unguided` | 0.0852 | 0.0573 | **0.2000** | 0.0786 | **0.4212** | 0.2765 |
| exp-005 `oracle_guided` | 0.0947 | 0.0661 | **0.2000** | 0.0821 | **0.4429** | 0.3037 |
| exp-005 `unguided_capped` | 0.0852 | 0.0573 | **0.2000** | 0.0789 | **0.4215** | 0.2768 |
| exp-008 `full_train` | 0.1364 | 0.0971 | **0.2000** | 0.0900 | **0.5236** | 0.4045 |

`reference_validity_micro` is `1.0` for six of the seven arms and `0.0` for exp-004's `unguided` and
`unguided_capped`. The reason (companion `finding_2`): the evaluator's
`_REF_RX = re.compile(r"\[#(\d+)\]")` matches **bracketed** `[#k]` only, and
`reference_validity_micro` is defined as `1.0` when the denominator is zero. Across all seven arms
the bracketed-reference denominator is:

| arm | items emitting a bracketed `[#k]` | valid / total refs | micro |
|---|---|---|---|
| exp-004 `unguided` | **2** / 600 | 0 / **2** | **0.0** |
| exp-004 `unguided_capped` | **2** / 600 | 0 / **2** | **0.0** |
| all five other arms | **0** / 600 | 0 / **0** | **1.0** (empty denominator) |

The two items are `3hop1__86965_488922_4093` and `4hop1__161765_53706_795904_580996`. Each emitted a
single leading `[#1]` — a forward reference in step 1, so invalid — and between them they own the
entire micro denominator of the run. Two items out of 600 move 20% of the primary metric from 1.0 to 0.0.

**The metric this term is named after is not being measured at all.** The gold decompositions of this
eval set use bare `#k`, never brackets: **1200** bare `#k` references across 600 items, **0**
bracketed (companion `finding_2b`). So on this dataset the reference term is a constant 1.0 that
occasionally detonates.

### 1.1 The exchange rates this creates

Companion `finding_3`:

- One invalid bracketed `[#k]` on 1 of 600 items in an otherwise reference-free run: **−0.2000**
  composite — equivalent to **300 of 600 items each losing a perfect step F1**.
- exp-004 `unguided`'s worst runaway, `4hop2__71753_158279_70784_79935`, predicted **39** steps
  against a gold of 4. Its *entire* contribution to the composite, via the length term, is
  **0.00194**. Ratio: **one stray bracket is worth 103 of that runaway.**
- **The step-count term is not unbounded below.** It is `max(0, 1 − MAE/3.0)` — clamped at 0, built
  from the *aggregate* MAE, and direction-blind. It reaches its floor only at MAE ≥ 3.0; observed
  MAE across all seven arms is **0.300 to 0.775**, so it never clamped in any run. And there is no
  per-item composite at all: the reference term is a micro rate and the length term a MAE, which is
  exactly why `T_TEST_STATISTICS` excludes composite and it gets a bootstrap CI but no t-test and no
  McNemar (PR #38 §4.9).

So the brief's candidate mechanism — per-item length error unbounded below for runaways — is
**falsified**. The mechanism is the reference term.

### 1.2 Null systems on this eval set

PR #38 measured null-system floors on v1 artifacts. Recomputed here on the pinned 600 so the
comparison is same-eval-set (companion `finding_6`; gold step counts are exactly 200/200/200 at
2/3/4, mean 3.0):

| system | house composite | `no_ref_renorm` |
|---|---|---|
| empty prediction | **0.2000** | 0.0000 |
| echo the question back as one step | **0.2333** | 0.0417 |
| one fixed junk step | **0.2333** | 0.0417 |
| three fixed junk steps | **0.2778** | 0.0972 |
| gold decomposition, order reversed | 0.7333 | 0.6667 |
| — exp-004 `unguided` (measured) | **0.2098** | 0.2623 |
| — exp-004 `unguided_capped` (measured) | **0.2128** | 0.2661 |

Under the house composite, exp-004's **deployable baseline scores below a system that echoes the
question back**, and below one that emits three fixed junk steps. Under the diagnostic the ordering
is sane (nulls 0.00–0.10, real arms 0.26–0.40). PR #38 §4.3 predicted exactly this on v1; **it has
now happened in v2's headline numbers.**

---

## 2. The Mistral composite anomaly, answered

`unguided` → `oracle_guided` composite rises **+0.2098** (0.2098 → 0.4197) while step F1 moves
**+0.0022** and ordered accuracy **+0.0048** (exp-004 log row; both CIs straddle zero, both t-tests
not significant). Term by term, from §1's table:

| term | contribution to the +0.2098 | share |
|---|---|---|
| **reference validity micro** | **+0.2000** | **95.3%** |
| length | +0.0075 | 3.6% |
| step F1 | +0.0009 | 0.4% |
| ordered accuracy | +0.0014 | 0.7% |
| *(diagnostic total, ref term dropped)* | *+0.0123* | — |

The doubling is **95% two items' bracket syntax**. Oracle guidance's real, measured effect on
Mistral-7B is hop-count EM (+0.0817, McNemar p = 2.41e-4, exp-004 log row) and +0.0123 on the
diagnostic — not a doubling of decomposition quality.

The same artifact contaminates a **cross-model** read: exp-005 `unguided` (0.4212) appears to double
exp-004 `unguided` (0.2098), a gap of 0.2114. On the diagnostic the same two arms are
**0.2765 vs 0.2623 — a gap of 0.0142**. **93% of the apparent Qwen-over-Mistral composite gap is the
bracket artifact.** Both runs are on the same pinned 600, so this is a legitimate comparison; it is
the metric, not the eval set, that is misleading.

---

## 3. The bootstrap CI asymmetry, diagnosed: it is bimodality, not heavy tails and not clipping

Companion `finding_4`. Each resample draws 600 item indices with replacement. exp-004 `unguided`'s
bracketed-reference denominator lives in **2** items, so a resample that happens to draw neither has
an *empty* denominator, and the empty-denominator rule flips its reference term from 0.0 to **1.0**
— changing the baseline's composite by 0.2 mid-bootstrap. Probability a resample contains neither:
`(1 − 2/600)^600 = 0.1349`. **Observed: 0.1373 of 10000 resamples.**

That splits the bootstrap difference distribution into two modes:

| comparison | mode | share | mean diff | range |
|---|---|---|---|---|
| `full_train − unguided` | baseline ref-den > 0 | 86.3% | **+0.3138** | [+0.2801, +0.3485] |
| | baseline ref-den = 0 | **13.7%** | **+0.1137** | [+0.0871, +0.1448] |
| `oracle_guided − unguided` | baseline ref-den > 0 | 86.3% | **+0.2099** | [+0.1947, +0.2277] |
| | baseline ref-den = 0 | **13.7%** | **+0.0098** | [−0.0056, +0.0254] |

This explains both published oddities exactly. The **2.5th percentile falls inside the lower mode**,
so the CI's lower bound (+0.1055, +0.0058) is not a tail of sampling noise on decomposition quality —
it is *the answer to a different question* ("what if this run had emitted no bracket at all?").
The **97.5th percentile sits inside the upper mode**, only slightly above that mode's mean, which is
why the point estimate lands just under the upper bound (56.5th and 56.8th percentile of the
distribution — not at the bound, but near it because 86% of the mass is in a narrow upper mode).

Dropping the reference term removes the bimodality entirely:

| comparison | house composite | `no_ref_renorm` diagnostic |
|---|---|---|
| `full_train − unguided` | +0.3137, CI [+0.1055, +0.3307], **width 0.2252**, pt at 56.5% | +0.1422, CI [+0.1207, +0.1640], **width 0.0433**, pt at 49.8% |
| `oracle_guided − unguided` | +0.2098, CI [+0.0058, +0.2189], **width 0.2130**, pt at 56.8% | +0.0123, CI [+0.0011, +0.0240], **width 0.0229**, pt at 49.8% |
| `unguided_capped − unguided` | +0.0030, CI [+0.0004, +0.0072] | +0.0038, CI [+0.0006, +0.0090] |
| exp-005 `oracle − unguided` | +0.0217, CI [+0.0070, +0.0348] | +0.0272, CI [+0.0087, +0.0435] |

Interval width falls **5.2×** and **9.3×**, and the point estimate returns to the middle of its
interval. exp-005's comparison, where neither arm emitted a bracket, shows no asymmetry at all —
the control that confirms the mechanism.

### Is the composite safe as a headline number?

**On the measured evidence, no — not without the diagnostic beside it.** Specifically, on this eval
set the house composite: (a) ranks exp-004's deployable baseline below a question-echoing null
system (§1.2); (b) assigns 95% of the Mistral oracle-guidance "doubling" and 93% of the apparent
Qwen-over-Mistral gap to two items' bracket syntax (§2); (c) produces a CI whose lower bound answers
a counterfactual rather than bounding sampling error (§3); and (d) is the only reported metric that
can take neither the paired t-test nor McNemar (PR #38 §4.9). Meanwhile `step_f1` and
`ordered_step_accuracy` have per-item values, take the full ADR 0009 battery, carry tight symmetric
intervals (`step_f1` +0.1372 CI [+0.1150, +0.1601], exp-008 log row), and hold **89%** of the
achievable headroom (PR #38 §4.10).

That is an argument that the step-level metrics are the better-behaved carriers of the exp-008
result. **It is not a decision to replace the composite** — that is issue #6 item 5, deferred to
Jahid and his supervisor, and PR #38 §5 already lays out the ranked options. What is new here is that
the failure mode PR #38 demonstrated with synthetic probes on v1 has now **fired in v2's committed
headline numbers**, which raises it from hypothetical to observed.

---

## 4. Where fine-tuning wins, per hop depth

Companion `finding_7.per_hop` and `finding_8`. exp-008 `full_train` vs exp-004's arms, same 600:

| gold hop | metric | exp-004 `unguided` | exp-004 `oracle_guided` | exp-008 `full_train` |
|---|---|---|---|---|
| 2 | step F1 | 0.3268 | 0.3403 | **0.4810** |
| | exact match | 0.1300 | 0.1300 | **0.2450** |
| | max predicted steps | **14** | 4 | **3** |
| 3 | step F1 | 0.1677 | 0.1693 | **0.2930** |
| | max predicted steps | **15** | 7 | **4** |
| 4 | step F1 | 0.1172 | 0.1087 | **0.2494** |
| | max predicted steps | **39** | 10 | **5** |

**The failure modes it fixes.**

1. **Runaway step counts — eliminated.** Predicted-step-count support collapses from
   `{1..9, 14, 15, 39}` (exp-004 `unguided`) to `{1, 2, 3, 4, 5}` (exp-008), with **0/600** rows
   above 5 steps. exp-005 `oracle_guided` is worse than its own unguided arm here (9 rows above 8
   steps, max **60**, vs 2 rows and max 11) — oracle guidance made most items exact while producing a
   small set of catastrophic runaways.
2. **Step-count accuracy — the largest single win.** Rows with the count exactly right go
   142/103/60 (exp-004, hop 2/3/4) to **195/149/82**; hop-count EM 0.5083 → 0.7100 (log rows).
3. **Placeholder chaining — improved, but it was never badly broken.** The evaluator does not measure
   this (its term needs brackets), so it was measured here on bare `#k`, valid meaning `1 ≤ k < own
   step index` (companion `finding_9`). Gold control: 1200/1200 valid. Arms:

   | arm | bare refs | validity | items with ≥1 broken ref |
   |---|---|---|---|
   | exp-004 `unguided` | 1352 | 0.9704 | **29** / 600 |
   | exp-005 `unguided` | 1147 | 0.9817 | 13 / 600 |
   | exp-008 `full_train` | 1028 | **0.9961** | **4** / 600 |

   Every one of the 4 remaining exp-008 breakages is a **self-reference** (`#k` in step `k`).
   Broken-chain rate rises with depth for prompting (4/11/14 at hop 2/3/4) but not for the tuned
   model (1/0/3). **Chaining is a 3% problem for prompting and a 0.4% problem after tuning — it is
   not where the loss is**, and it is not what the 0.2-weighted term is capturing.
4. **Not paraphrase.** ROUGE-L is over joined steps and `step_f1` is exact normalized set matching, so
   the "ROUGE rewards paraphrase that step F1 punishes" story is testable: rows with ROUGE ≥ 0.8 but
   step F1 = 0 number **2/600** (exp-004 `unguided`) and **8/600** (exp-008), against 349 and 213 rows
   at step F1 = 0. Mean ROUGE on the step-F1-zero rows is 0.4398 and 0.5200. Those rows are
   *substantively different decompositions*, not near-paraphrases. Paraphrase is a real but **minor**
   effect at the whole-decomposition level (it is larger per-step — see §6).

**Where its under-decomposition hurts: yes, the 4-hop stratum is paying for it.** exp-008's mean
signed step-count error is −0.233 overall but **−0.540 at hop 4**, with under-rate 0.550 (log row).
Conditioning on its own predicted count at hop 4 (companion `finding_8`):

| exp-008 4-hop rows | n | step F1 | recall | precision | ordered | EM |
|---|---|---|---|---|---|---|
| predicted 3 steps (under) | **104** | 0.1538 | 0.1346 | 0.1795 | 0.1178 | 0.0000 |
| predicted 4 steps (correct) | 82 | **0.3902** | 0.3902 | 0.3902 | 0.3841 | 0.0488 |
| predicted 2 steps | 6 | 0.0556 | 0.0417 | 0.0833 | 0.0417 | 0.0000 |

When it emits 4 steps it scores **0.3902** — nearly as well as its own 2-hop stratum, and **1.87×**
exp-004 `unguided`'s 4-step 4-hop rows (0.2083). The 4-hop aggregate of 0.2494 is dragged down by the
**52% of 4-hop items where it stops at 3 steps**. **Caveat, and it matters: this conditions on the
model's own prediction, so the items it counts correctly may simply be the easier items.** The
association is strong and consistent across all three arms; the causal claim is **not established**
and would need a run, not an analysis.

**A striking inversion in the oracle arm.** At hop 4, exp-004 `oracle_guided`'s count-correct rows
score step F1 **0.0845** while its count-*wrong* rows score **0.1221** — being told the answer is 4
steps makes its 4 steps *worse*. 55 of its 71 count-correct 4-hop rows have step F1 exactly **0**.
This is the mechanism behind exp-004's headline puzzle: oracle guidance buys hop-count EM and pays
for it in step content at depth, which is why step F1 stayed flat on Mistral while every metric moved
on Qwen (exp-005 log row).

---

## 5. `unguided_capped` was a non-test — and runaway length is not a real failure mode

The cap at 8 fired on 12/600 rows (7 truncated) for Mistral and 4/600 (2 truncated) for Qwen (log
rows), so the arm measured almost nothing. Two separate questions follow.

**What cap would bind?** Gold step counts on this eval set never exceed **4** (exactly 200/200/200 at
2/3/4). So 4 is the largest cap that can never truncate a row whose gold is longer, and it is the
most aggressive *defensible* cap. Rows exceeding each candidate (companion `finding_7`):

| cap | exp-004 `unguided` rows truncated | exp-005 `unguided` | exp-008 `full_train` |
|---|---|---|---|
| 8 | 7 (1.2%) | 2 | **0** |
| 6 | 16 (2.7%) | 13 | **0** |
| 5 | 37 (6.2%) | 35 | **0** |
| **4** | **84 (14.0%)** | **112 (18.7%)** | 8 (1.3%) |
| 3 | 195 (32.5%) | 245 | 98 |

**Does it buy anything?** Simulated by truncating generated step lists and rescoring with the
evaluator's own functions — **a simulation, not a run** (companion `finding_7.cap_simulation_is_a_simulation`):

| exp-004 `unguided` | step F1 | ordered | step-count MAE | EM |
|---|---|---|---|---|
| measured baseline | 0.2039 | 0.1804 | 0.775 | 0.0617 |
| cap 8 (the arm that ran) | 0.2040 | 0.1805 | 0.687 | 0.0617 |
| **cap 4 (maximally binding)** | **0.2050** | **0.1829** | **0.438** | 0.0617 |
| cap 3 | 0.2027 | 0.1796 | 0.513 | 0.0533 |

Even the maximally binding cap, firing on 84/600 rows, moves step F1 by **+0.0011** and ordered
accuracy by **+0.0025**, and leaves EM **unchanged**. It halves the step-count MAE — which is exactly
why it would move the composite's length term while changing nothing about decomposition quality.
Cap 3 makes things worse.

**Conclusion the data supports: runaway length is not a quality failure mode for these models, it is
a length-metric artifact.** A 39-step prediction and a 5-step prediction on a 4-step gold both have
step F1 ≈ 0; truncating the first to 8 or to 4 does not make its content correct. The errors are in
step *content*, not step *count* — which is also why exp-008's large step-count win (§4) translates
into only part of its step-F1 win. exp-004's original hypothesis ("if `unguided_capped` closes most of
the gap, the problem was runaway length") can now be answered: **it is not**, and no cap value in the
defensible range would have changed that verdict.

---

## 6. exp-008's exact match of 0.1083: near-misses are mostly surface form

Companion `finding_10`. EM requires **all n** steps to match verbatim after normalization, so it
decays geometrically with depth: 0.2450 / 0.0600 / 0.0200 at hop 2/3/4.

**The near-miss ladder** — among rows where exp-008 got the step *count* right, how many steps
matched verbatim:

| gold hop | count-correct rows | 0 right | 1 | 2 | 3 | 4 | EM needs |
|---|---|---|---|---|---|---|---|
| 2 | 195 | 54 | **92** | 49 | — | — | 2 |
| 3 | 149 | 62 | 45 | **30** | 12 | — | 3 |
| 4 | 82 | 12 | 28 | **32** | 6 | 4 | 4 |

**128 of 600 items** have the right count and every step but one verbatim correct. Those single
misses are small: of exp-008's 1105 positionally-aligned non-matching step pairs, median character
similarity is **0.537**, with **13.5%** at ≥ 0.9 and **23.1%** at ≥ 0.8 (exp-004 `unguided`: 0.448
median, 6.5% and 12.8%). A fuzzy-EM probe accepting ≥ 0.9 character similarity per step — **a
simulation and an illustration, not a proposed metric** — moves EM from 0.1083 to **0.1717**
(103/600) for exp-008 and 0.0617 to 0.0983 for exp-004.

**What the misses actually are.** The gold is stylistically heterogeneous: **655** of its 1800 steps
use the `X >> relation` template and **1145** are natural-language questions, and **381 of 600 items
mix both styles inside one decomposition**. Verbatim hit rates split hard along that seam:

| arm | on `>>` template gold steps | on natural-language gold steps |
|---|---|---|
| exp-004 `unguided` | 238/633 = **0.3760** | 53/1035 = **0.0512** |
| exp-008 `full_train` | 381/620 = **0.6145** | 154/1020 = **0.1510** |

Fine-tuning lifts the template hit rate **1.6×** and the natural-language rate **2.9×**, but the gap
between the two styles survives tuning: **7.3×** for exp-004 `unguided`, still **4.1×** for exp-008.
The template steps are near-deterministic and learnable; the natural-language steps require
reproducing an annotator's exact phrasing.

Representative near-misses (ids in companion `finding_11`; short snippets only). `2hop__557496_57594`
loses EM on one word — pred `what gun does #1 use in westworld` against gold
`what gun does #1 used in westworld`, i.e. **the gold's own grammatical error**. `2hop__252521_80650`
misses on `santa claus 3` vs the gold's `the santa clause 3`. `2hop__825727_584042` matches step 1
verbatim and expresses step 2 as a question (`What league was #1 ?`) where the gold uses the template
(`#1 >> league`) — same meaning, zero credit. `2hop__628752_538661` is the mirror case: the model
writes the natural-language form where gold used the template.

**Does EM belong in the results section?** What the data supports: EM as reported is a **conjunction
of n exact string matches against a stylistically inconsistent reference that contains typos**, so it
measures format conformance to the annotator alongside decomposition correctness, and it compresses
the 4-hop stratum to 0.0200 where 82 rows had the structure right. It is also the metric with the
clearest precedent — Break/QDMR reports a *canonicalized* EM as its official number (PR #38 §3.2),
and this pipeline's EM is **not** canonicalized. Both facts are relevant to the write-up; **which way
to report it is Jahid's call with his supervisor**, and §8 ranks the options rather than choosing.

---

## 7. Suspected defects in the evaluation itself — all stated as hypotheses

Flagged for a decision, not acted on. None of these is a research-direction call; each is a
measurable claim about the harness.

1. **`_REF_RX` matches a placeholder syntax this dataset does not use.** *Established as measured:*
   gold carries 1200 bare `#k` and 0 bracketed `[#k]` references across 600 items, so the
   0.2-weighted reference term has an empty denominator on five of seven arms and scores them 1.0 for
   emitting nothing. Whether this is a bug or an intended v1-carried definition is **unknown to me** —
   `docs/METRICS.md` §4 documents the micro-rate fragility but not a syntax mismatch. A measured
   alternative exists (§4, bare-`#k` validity 0.9704 → 0.9961) but is **not a house metric** and was
   computed only in this note.
2. **The empty-denominator convention rewards silence and creates the bimodal CI.** *Established:*
   `1.0` for zero references means "never chained" outranks "chained imperfectly", and it is the
   mechanism behind §3's mode split. A run that emits one invalid reference is scored *worse than a
   run that emits none*.
3. **The composite ranks two real arms below null systems on its own eval set.** *Established by
   measurement* (§1.2), using the evaluator's own functions on the pinned 600.
4. **`ordered_step_accuracy` is positional against a reference whose step order is not canonical.**
   *Suggestive.* Set-matched-but-mis-positioned steps cost **41** step matches in exp-004 `unguided`
   (12.3% of its 332 set matches, 38 items) and **23** in exp-008 (4.1%, 19 items). Item
   `4hop2__9988_261673_70784_79935` shows the pattern: the model's steps 1 and 2 are the gold's 2 and
   1. Whether that reordering is *benign* (a valid alternative plan) or a genuine error is a
   **semantic judgement no metric here makes**, and I did not adjudicate it item by item — so the
   41 and 23 are upper bounds on the penalty, not established errors.
5. **Gold reference quality bounds EM from above.** *Suggestive, and not quantified.* Two of five
   sampled near-misses fail on a gold typo or article (`used` for `use`, `the santa clause 3`). I did
   **not** audit the gold, so the size of this ceiling is **unmeasured**; §6's fuzzy-EM probe (0.1083
   → 0.1717) bounds *all* surface-form effects together, not typos specifically.
6. **The composite is the least testable reported metric.** *Established* and already recorded in
   PR #38 §4.9: no per-item value, so no t-test and no McNemar.

---

## 8. Ranked hypotheses

Ranked by **expected impact on what the thesis can defend, per unit of work**. Each states the
expected direction and the reason. **This is a list, not a choice** — and per PR #38 §5, the
primary-metric question is issue #6 item 5, already deferred.

1. **Report the `no_ref_renorm` diagnostic beside every composite already published, and treat the
   reference term as unmeasured on this dataset until the syntax mismatch is adjudicated.**
   *Direction:* removes the largest known distortion in the numbers now in the log; shrinks the
   exp-008 headline from +0.3137 to +0.1422 and the Mistral oracle effect from +0.2098 to +0.0123,
   both with 5–9× tighter intervals. *Why:* measured, §1–§3, and it needs no run — the diagnostic is
   already a named house quantity (`docs/METRICS.md` §4.1). *Risk:* §4.1 forbids any thesis claim
   resting on it alone, so it is a reporting change, not a metric swap, and the disagreement between
   it and the house composite is itself the finding.
2. **Decide whether `_REF_RX` should match bare `#k`, then re-score existing predictions.**
   *Direction:* would make the 0.2 term measure the thing it is named after (a real 0.9704→0.9961
   signal exists) and would remove the bimodality at the source. *Why:* §1, §4.3, §7.1. *Cost:* CPU
   re-scoring only — no generation. *Risk:* this is a **pipeline-code change to a metric definition**,
   so it changes every published composite and needs Gate-1 review; and whether the bracket form was
   intended is a question for Jahid, not something to infer.
3. **Attack 4-hop under-decomposition in the fine-tuned arm.** *Direction:* the largest measured
   quality headroom — 104/200 4-hop items stop at 3 steps and score 0.1538 where its own 4-step rows
   score 0.3902. *Why:* §4. *Risk:* the conditional comparison is confounded (the model may under-count
   precisely the harder items), so the headroom is **suggestive, not established**; and any fix is a
   training-side change, which is a run, not an analysis.
4. **Report a canonicalized or per-step EM alongside strict EM.** *Direction:* would separate
   decomposition correctness from annotator-format conformance; strict EM 0.1083 vs a 0.9-similarity
   probe at 0.1717, and 128/600 items are one step short. *Why:* §6, plus the Break/QDMR precedent in
   PR #38 §3.2 where canonicalized EM *is* the official metric. *Risk:* any canonicalization is a new
   metric definition and needs to be justified rather than tuned; the 0.1717 figure is a probe, not a
   proposal.
5. **Retire `unguided_capped` as an arm, or re-run it at cap 4 knowing it will not move quality.**
   *Direction:* saves compute and removes a misleading "significant" composite row (+0.0030, CI
   excludes 0) that is driven entirely by the length term. *Why:* §5 — the maximally binding cap moves
   step F1 by +0.0011. *Risk:* the sweep is a post-hoc truncation simulation; a generation-time cap
   also changes decoding, so "will not move quality" is a **strongly supported expectation, not a
   measured result** for a real cap-4 run.

Lower-ranked and listed for completeness: auditing gold step order for benign reorderings (§7.4,
bounded at 41 and 23 step matches); auditing gold surface quality (§7.5, unquantified).

---

## 9. Reproducing this note

```
.venv/bin/python  # then, per section, the evaluator's own functions:
#   from scripts/musique_decompositions_evaluator.py:
#   _step_prf, _ordered_step_accuracy, _exact_decomposition_match, _reference_validity,
#   _rouge_l, _statistic_arrays, _statistics_for, _normalize_step
```

Inputs are the seven `eval_*_per_item.json` files pinned by sha256 in the companion's
`provenance.inputs_pinned_by_content`; they live under `runs/` (git-ignored) on the compute box.
Aggregates are read from the committed `experiments/exp-00{4,5,8}/metrics.json`. The bootstrap uses
seed 42, 10000 iterations, chunk 1000, alignment by sorted `item_id`, which reproduces
`experiments/exp-008/metrics.json`'s composite rows bit-identically (§0). No model was run and no
pipeline code was changed to produce this note.
