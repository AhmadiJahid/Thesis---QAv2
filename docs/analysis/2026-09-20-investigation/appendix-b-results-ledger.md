<!-- Lane ledger produced by a read-only investigator agent on 2026-09-20 for the lead's state-of-the-work investigation (`docs/analysis/2026-09-20-state-of-the-work-investigation.md`). Every claim cites a file, commit, issue or log row. Numbers are quoted from committed artifacts, never re-computed. -->

# Results ledger — the experiment record (exp-001 … exp-028)

Read-only investigation, 2026-09-20. Sources: `experiments/log.md` (28 real rows; a 29th `| exp-001 |`
line is the schema's illustrative example, inside an HTML comment), `experiments/exp-*/{config,metrics,notes}`,
`docs/METRICS.md`, all of `docs/analysis/`, `docs/prior-work.md`, `gh issue list`.
Every number below is quoted from the log row or committed metrics JSON named beside it. Where a value
is not in a committed artifact it is marked **unmeasured**.

Two eval sets exist and they are not interchangeable:
- **P600** = ADR 0007 pinned 600 MuSiQue dev (200 each at 2/3/4 hops).
- **D750** = pool_sweep's own dev sample (`dev_seed 42`, 250/hop = 750). Intersection with P600 is **405/600**
  (exp-018, exp-019 rows) — numbers from the two sets may never be placed on one axis.

Two few-shot pools exist: **v1 pool** (9,156 rows, ADR 0014, sha256 `e5c418a9…`) used by exp-004/005/008/009/012,
and **size2000_imbalanced** (2,000 rows, sha256 `cc364662…`, retrieval artifact `3f95c7dc…`) used by
exp-014/015/020–028. Arms across the two pools are not comparable.

---

## 1. One row per experiment

| id | date | question it answered | model (+quant) | eval set | headline metrics | status |
|---|---|---|---|---|---|---|
| exp-001 | 08-19 | produce the `full_train` LoRA adapter | Mistral-7B-Instruct-v0.3, 4-bit QLoRA r=16 | — (eval deferred) | train loss 0.140; 2 epochs / 2,494 steps / 19,938 ex; 15,497.5 s; peak GPU 5.05 GiB; quality **unmeasured** | training-only |
| exp-002 | 08-19 | guided vs unguided (Mistral) | Mistral-7B | P600 | rc=1 all 3 arms, `KeyError: 'raw'`; zero output | aborted → relaunched as exp-004 |
| exp-003 | 08-19 | guided vs unguided (Qwen3.5-9B) | Qwen3.5-9B | P600 | same crash | aborted → exp-005 |
| exp-004 | 08-19 | does an oracle hop count help decomposition? | Mistral-7B, fp16, v1 pool | P600 | unguided step_f1 .2039 / ord .1804 / hop_EM .5083 / composite .2098; oracle_guided .2061/.1852/.5900/.4197; capped .2040/.1805/.5083/.2128. Only hop_EM + composite move significantly | complete |
| exp-005 | 08-19 | same, on Qwen3.5-9B | Qwen3.5-9B, fp16, v1 pool | P600 | unguided .2131/.1911/.5200/.4212; oracle .2368/.2205/.8733/.4429; capped .2131/.1911/.5200/.4215. **Every** quality metric moves for oracle | complete |
| exp-006 | 08-20 | build 3 pool-construction strategies at size 2000 (CPU) | none (e5-small-v2 + MiniLM CE) | D750 | 3×2,000-row pools (imb 1461/413/126, bal 667/667/666, clu 1396/495/109); k-means 11 iters, inertia 1803.46, 0 empty clusters; decomposition quality deferred | complete (artifact only) |
| exp-007 | 08-21 | build mixed vs oracle-hop-matched retrieval (CPU) | none | P600 | 600/600 rows both arms, 200/200/200; 6 artifact sha256 recorded; quality deferred to exp-009 | complete (artifact only) |
| exp-008 | 08-22 | does LoRA fine-tuning beat prompting, same eval set? | Mistral-7B + `full_train` LoRA, fp16 base | P600 | step_f1 **.3411**, ord .3237, hop_EM .7100, EM .1083, composite .5236; signed step error −0.233; 151.9 tok / 1.387 s per query vs unguided's 499.8 tok / 1.879 s. **Every** metric significant vs exp-004 unguided | complete |
| exp-009 | 08-22 | does hop-matched few-shot retrieval help? | Mistral-7B, fp16, v1 pool | P600 | mixed vs oracle: step_f1 .2035/.1931 n.s.; ord .1804/.1799 n.s.; hop_EM .5150/.6000 **sig** (McNemar p=6.2e-4) | complete |
| exp-010 | 08-22 | imbalanced vs balanced vs clustered pool construction | Mistral-7B, **4-bit**, 128-tok cap | **D750** | headline cell (+CE, typed): step_f1 .1900/.1926/.1939, hop_EM .5133/.4667/.5133; only hop_EM significant (imb & clu > bal); imb ≡ clu | complete |
| exp-011 | 08-22 | back-fill Break-faithful columns on 9 older arms | none (re-score) | P600 | all 9 reproduce prior values exactly; adds break_EM/SARI/GED/chain_validity. chain_validity separates by base model: Qwen arms .8605–.8697 vs Mistral .9686–.9942 | complete (coverage re-score) |
| exp-012 | 08-23 | does a prompted router earn its place? | Mistral-7B, fp16 | P600 | router accuracy **.3833** (230/600) vs 3-class chance .333, **394/600 responses unparsed** and defaulted to hop 2; unguided vs router_guided **0 of 10 statistics significant**; oracle vs router_guided: hop_EM .5900 vs .4917 sig | complete |
| exp-013 | 08-23 | does a 2/3-hop-only fine-tune generalise to 4 hops? | Mistral-7B + `generalisation_2_3hop` LoRA | P600 (claim restricted to 200 hop-4) | at hop 4: hop_EM **.145** vs baseline .300 (McNemar p=4.7e-5, favours baseline) and vs exp-008 .410 (p=5.9e-10); under-decomp .845; break_EM 0/200; chain_validity 1.000 | complete |
| exp-014 | 08-23 | paired significance on the new columns + build the S2 retrieval artifact | none | P600 | 8/8 compares rc=0. Headline exp-004 unguided vs exp-008: break_EM .0533→.1017, SARI .5850→.6823, GED .4715→.3094, chain .9686→.9942 — all significant for the fine-tune | complete |
| exp-015 | 08-23 | prompting at its **best pool** vs the fine-tune | Mistral-7B, **fp16**, size2000 pool | P600 | step_f1 .1894, ord .1662, hop_EM .5400, break_EM .0350, SARI .5785, GED .4714, chain .9694, under .2317, over .2283; vs exp-008 **10/10 significant, all for the fine-tune** | complete |
| exp-016 | 08-23 | back-fill the six-term report card on 12 arms | none (re-score) | P600 | 12/12 reproduce exactly; first log entry carrying the report card per arm | complete (coverage re-score) |
| exp-017 | 08-23 | does better decomposition give better **answers**? | Mistral-7B reader, all-paragraph context | P600 | answer_EM/F1: oracle **.2800/.3730** > exp-008 full_train **.2417/.3283** > exp-004 unguided **.2133/.2952** > no-decomposition control **.1800/.2723** (monotone) | complete (cell 4 pending sign-off) |
| exp-018 | 08-27 | does cross-encoder rerank move anything beyond the legacy composite? | none (re-score of exp-010) | **D750** | ROUGE-L 9/9 pairs positive (3 survive Holm); GED 9/9 improve (4 Holm); SARI 9/9 positive (0 Holm); EM/hop_EM/break_EM/chain_validity null; composite swings −0.1967…+0.2093 | complete (re-analysis) |
| exp-019 | 09-14 | is the imbalanced≡clustered tie a plumbing bug? | none (re-analysis of exp-010) | **D750** | **No defect.** 688–700 of 750 predictions differ; pools share 201/2000 ids (Jaccard .0529); 0/750 identical top-5 lists; 18/18 distinct generation SHAs. The exact tie is a 129/129 McNemar split in 1 of 6 cells | complete (re-analysis) |
| exp-020 | 09-20 | what does 4-bit cost Mistral? | Mistral-7B, **4-bit** | P600 | break_EM .0400, SARI .5766, GED .4846, chain .9626, hop_EM .5117, under .1683, over .3200. vs exp-015: **10 of 13 n.s.**; significant: over +.0917 (worse), under −.0633 (better), legacy composite. 2.3× slower than fp16; peak 4,818 MiB | complete |
| exp-021 | 09-20 | does model **family** matter at 7B? | Qwen2.5-7B-Instruct, 4-bit | P600 | break_EM .0550, SARI .5720, GED .4694, chain .9210, hop_EM .5533, under .2933, over .1533. vs exp-020: 6/13 sig, both directions; all wording/graph/overlap terms n.s.; signed error −0.150 vs Mistral +0.585; 3.0× faster | complete |
| exp-022 | 09-20 | does **size** matter inside Qwen2.5? | Qwen2.5-14B-Instruct, 4-bit | P600 | break_EM .0333, SARI .5795, GED .4605, chain **.9838**, hop_EM **.6167**, under **.1017**, over .2817. vs exp-021: 7/13 sig — 14B better on under/chain/hop_EM, 7B on over/break_EM/EM; all wording terms n.s. vs exp-008: **9/10** (chain_validity falls out) | complete |
| exp-023 | 09-20 | does reversing exemplar order matter (Mistral)? | Mistral-7B, 4-bit, `reversed` | P600 | break_EM .0483, SARI .5831, GED .4813, chain .9629, hop_EM .5267, under .1783, over .2950. vs exp-020: **0 of 13 significant**; largest move 2.5 pp | complete |
| exp-024 | 09-20 | same, Qwen2.5-7B | Qwen2.5-7B, 4-bit, `reversed` | P600 | break_EM .0583, SARI .5771, GED .4728, chain .9282, hop_EM .5400, under .2900, over .1700. vs exp-021: **1 of 13** — step_f1 +.0152 (CI +.0013…+.0292), not a report-card term; report card 0/7 | complete |
| exp-025 | 09-20 | same, Qwen2.5-14B | Qwen2.5-14B, 4-bit, `reversed` | P600 | break_EM .0333, SARI .5832, GED .4670, chain .9767, hop_EM .5967, under .1150, over .2883. vs exp-022: **0 of 13**; largest move 2.0 pp | complete |
| exp-026 | 09-20 | what does training on the 2,000-question **pool alone** buy? | Mistral-7B + `pool_2000_imbalanced` LoRA (4-bit train, fp16 infer) | P600 | break_EM .0967, SARI .6804, GED .3191, chain .9750, hop_EM .7033, under .2583, over .0383; step_f1 .3185, EM .1083. vs exp-008 **4 of 13 sig, all ≤2.7 pp, only chain_validity from the card**; vs exp-015 **11/13 for the fine-tune**. 250 steps / 1,568.8 s ≈ 10% of exp-001's compute | complete |
| exp-027 | 09-20 | where does Qwen3.5-9B sit on the block axis? | Qwen3.5-9B, 4-bit | P600 | break_EM .0550, SARI **.5984**, GED .4838, chain **.8471**, hop_EM .5700, under .1350, over .2950; step_f1 **.2135**, ROUGE-L **.5625**. Best of the four on wording, worst on chaining; vs exp-008 10/10 for the fine-tune | complete (all model comparisons confounded) |
| exp-028 | 09-20 | reversed order on Qwen3.5-9B | Qwen3.5-9B, 4-bit, `reversed` | P600 | break_EM .0517, SARI .6023, GED .4645, chain .8821, hop_EM .5717, under .1383, over .2900. vs exp-027: **3 of 13 sig, all favouring reversed** — chain_validity +.0349 (CI +.0041…+.0652), GED −.0194, legacy composite. **The only G3 arm where a report-card term moves** | complete |

---

## 2. The metric suite as it stands (`docs/METRICS.md`)

**The reported primary is the six/seven-term `decomposition_report_card`** (§2.3), always printed together:

1. `break_exact_match_rate` ⇑ — whole plan character-identical to gold (Break's official EM).
2. `sari_macro` ⇑ — n-gram edit quality of the plan against the question. **Floor on this data ≈0.29, not 0.**
3. `ged_macro` ⇓ — graph edit distance between predicted and gold plan graphs (wording + chaining + count in one).
4. `chain_validity_macro` ⇑ — house repair of reference validity: gold chains but prediction emits no `#k` ⇒ 0.0.
5. `hop_count_exact_match_rate` ⇑ — right number of steps.
6a/6b. `under_decomposition_rate` / `over_decomposition_rate` ⇓ — never summed (ADR 0017: over tolerable, under not).

**Reported beside the card, not in it:** `exact_match_rate`, `step_precision/recall/f1_macro` (set-based, unordered),
`ordered_step_accuracy_macro`, `rouge_l_f1_macro` (LCS over the flattened token stream), `step_count_mae`,
`mean_signed_step_count_error`, `predicted_hop_distribution`, `decomp_mean_macro` (unweighted blend of 5 terms,
contingency only), `per_gold_hop_metrics`.

**Significance** (`--compare`, §5): paired bootstrap 10,000 resamples on 8 statistics, McNemar exact on 5 binary
ones, paired t-test on 12; α=0.05; alignment by `item_id` with a hard refusal on any id mismatch.
**No multiple-comparison correction anywhere** (ADR 0009) — this bites exp-024's single hit and exp-028's three.

### Known defects

- **`composite_score` — LEGACY, frozen, do not lead with it (issue #40).** Its 0.2-weight
  `reference_validity_micro` term uses `_REF_RX = r"\[#(\d+)\]"` while MuSiQue gold writes bare `#k`
  (1,200 bare / 0 bracketed across P600), so the term never measured reference validity here. Measured
  consequence (2026-08-22 error analysis §1.2): on P600 the deployable exp-004 baseline (0.2098) scores
  **below a system that echoes the question back** (0.2333) and below three fixed junk steps (0.2778).
  Deliberately not repaired so committed numbers stay quotable.
- **`reference_validity_macro` gives free credit for silence** — an item with no references scores 1.0.
- **`chain_validity` cannot see an orphan step.** It checks that each `#k` points backwards, never that every
  step is *used*: 85 of 86 dangling-step item×arm pairs score 1.0 (anchor note §1.3); replicated 76 of 83 on
  the seed-43 draw. All 45 golds are fully chained in both draws.
- **Exact-string step matching punishes correct plans.** On 13 of 45 anchors (8 of 45 on seed-43) *all eight arms*
  score `step_f1 = 0.0`, including items where all eight write the same plausible plan. Token-overlap re-scoring
  lifts mean step-F1 from 0.155–0.305 to 0.50–0.68. (Scratch diagnostic, not a registered metric.)
- **GED caveats:** node-cap fallback substitutes a non-tight bound (up to +0.0588 worse) — read `ged_fallback_counts`
  beside every GED comparison (largest in the record: 6/600 on exp-005 oracle_guided); GED is order-blind on
  2-step plans; no spaCy lemmatizer, so absolute values are **not** comparable to published Break numbers.
- **Issue #53:** the decomposer run note prints the model-config `few_shot.k = 3` on reranked runs that actually
  showed **5** exemplars. Affects the prose of every reranked run's `notes.md` (exp-015, exp-020–028); the prompt
  log header is correct. No metric is affected.
- **Issue #54:** `metrics.json` stamps `git.commit` at *write* time, not execution time — quote the launch SHA
  recorded by hand in the log row.
- **Issue #55:** `print_model_params.py` (9,653,104,368, from the checkpoint's `mtp` head) and the runtime
  ceiling assertion (8,953,803,264) disagree by 0.7B on Qwen3.5-9B.
- **`statistics_not_available_in_inputs`:** the exp-011 stored per-item files predate `over_decomposition`,
  `under_decomposition` and `decomp_mean`, so **every `--compare` against exp-008 via exp-011 omits those three**
  (exp-015/020/021/022/027 all carry this flag). Those rates are reported descriptively, not paired-tested.

---

## 3. Which comparisons are legitimate today

Legitimate = same eval set, same pool, same seed, one factor moved, both sides carrying the compared columns.

| group | arms | the one factor | evidence |
|---|---|---|---|
| Hop guidance, Mistral | exp-004 unguided / oracle_guided / unguided_capped | hop count given or not | exp-004, re-tested exp-014 |
| Hop guidance, Qwen3.5-9B | exp-005 triple | same | exp-005, exp-014 |
| Prompting vs fine-tuning (v1 pool) | exp-004 unguided ↔ exp-008 | training vs prompting | exp-008, exp-014 |
| Retrieval hop-matching | exp-009 mixed ↔ oracle | hop-matched exemplars | exp-009, exp-014 |
| Router | exp-012 unguided ↔ router_guided ↔ oracle_guided | source of the hop count | exp-012 |
| Precision (Mistral only) | exp-015 ↔ exp-020 | fp16 vs 4-bit | exp-020 |
| Family at 7B | exp-020 ↔ exp-021 | Mistral vs Qwen2.5 | exp-021 |
| Size inside Qwen2.5 | exp-021 ↔ exp-022 | 7.62B vs 14.77B | exp-022 |
| Exemplar ordering | exp-023↔020, exp-024↔021, exp-025↔022, exp-028↔027 | `as_is` vs `reversed`, ordering invariant verified 120/120 on artifacts | exp-023/024/025/028 |
| Prompting vs fine-tuning (block pool) | exp-015/020/021/022/027 ↔ exp-008 | prompting vs fine-tuning | those rows |
| Pool-only fine-tune | exp-026 ↔ exp-015 (precision-matched prompting) | training vs prompting | exp-026 |
| Held-out depth | exp-013 hop-4 ↔ exp-004 unguided hop-4; exp-013 hop-4 ↔ exp-008 hop-4 | training hop coverage | exp-013 |
| Pool construction (D750 only) | exp-010 imbalanced / balanced / clustered | construction strategy | exp-010, exp-019 |
| Reranker (D750 only) | exp-018's 9 matched +CE vs bi-encoder-only pairs | cross-encoder rerank | exp-018 |
| End-to-end answering | exp-017's 4 cells (oracle / full_train / unguided / no-decomposition) | decomposition source | exp-017 |

### Not legitimate, and why

- **exp-004/005 ↔ exp-020–028.** Different few-shot pool (v1 9,156 vs size2000) *and* different precision.
  exp-027's row states it outright: **exp-005 is not a comparator for exp-027** despite being the same model.
- **exp-026 ↔ exp-008** is *three* factors (training identity, volume 2,000 vs 19,938, hop mix); the clean
  same-size-different-draw control `pool_2000` has never been trained.
- **exp-026 ↔ exp-020** moves precision as well — the row labels it context only.
- **exp-022 ↔ exp-020** moves family *and* size; exp-027 ↔ exp-021/022 moves model generation *and* parameter
  count. None separates the two.
- **Any D750 number placed beside a P600 number** (exp-006/010/018/019 vs everything else) — 405/600 id overlap.
- **`biencoder_plus_ce__typed` quoted alone** from exp-010 — that is the one cell of six with the exact 129/129
  tie, and quoting it alone is what produced the supervisor's suspected-bug report (exp-019).
- **Any ranking off `composite_score`** — junk baselines outrank real arms on it.
- **Any arm-vs-arm ordering read off a 45-item anchor sheet** — no power, and the draws are difficulty-biased
  (per-hop step_f1 shifts up to 0.158, consistent in sign across all eight arms).
- **over/under-decomposition in the exp-008 comparisons** — not paired-tested (see §2, `NOT COMPARED`).

---

## 4. What is established well enough for the thesis

Each is ran-X-observed-Y with counts, on one stated eval set.

1. **Fine-tuning beats prompting on decomposition quality, on the same 600 questions, for five prompting arms
   and four models.** exp-008 (`full_train` LoRA) beats exp-004 unguided on every metric (exp-008, exp-014);
   and exp-015 (fp16, best pool), exp-020 (Mistral 4-bit), exp-021 (Qwen2.5-7B), exp-027 (Qwen3.5-9B) each lose
   **10 of 10** headline statistics to it, exp-022 (Qwen2.5-14B) **9 of 10** (chain_validity ties). Gap sizes:
   step_f1 +0.13–0.16, hop_EM +0.09–0.20, GED −0.15–0.18.
2. **And it is cheaper to serve:** 151.9 tokens / 1.387 s per query (exp-008) and 152.3 / 1.404 s (exp-026)
   against 471.6 / 1.810 s for prompting (exp-015) — about 32% of the tokens, because no exemplars are in the prompt.
3. **Training on only the 2,000-question pool reaches full-split fine-tuning.** exp-026 is indistinguishable from
   exp-008 on 9 of 13 statistics; the 4 that differ favour exp-008 by ≤2.7 pp and only one (chain_validity) is a
   report-card term — at 250 optimizer steps / 1,568.8 s versus 2,494 steps / 15,497 s (≈10% of the compute).
4. **A perfect oracle hop count moves step *count*, not step *content*** — on Mistral (exp-004: hop_EM +0.0817,
   McNemar p=2.4e-4, while step_f1 and ordered accuracy have CIs straddling zero). The same shape appears on an
   unrelated axis: hop-matched retrieval (exp-009, hop_EM .515→.600 sig, step quality n.s.) and pool construction
   (exp-010, hop_EM the only significant axis of four). On Qwen3.5-9B (exp-005) oracle guidance moves everything —
   so this finding is model-dependent, which is what issue #49 flags.
5. **A prompted few-shot router does not earn its place as built.** exp-012: router accuracy 0.3833 (230/600)
   against 3-class chance 0.333, with **394/600 responses unparsed** and silently defaulted to hop 2 (so the
   0.875 hop-2 recall is inflated — only 52/77 of genuinely parsed hop-2 rows were right). Unguided vs
   router_guided is **0 of 10 statistics significant**, including hop_count_EM itself.
6. **Exemplar order does not move the report card on three of four models.** exp-023 0/13, exp-024 1/13
   (step_f1 only, not a card term), exp-025 0/13, exp-028 3/13 (chain_validity +0.0349, GED −0.0194, legacy
   composite) — 4 of 52 statistics across four pairs, largest single movement 3.5 pp against the supervisor's
   "up to 10%". Scoped to: unguided, 4-bit, k=5 reranked exemplars from size2000_imbalanced, P600, greedy, seed 42.
   **The per-item picture is different and must be stated with it:** reversing rewrites 82% of plans
   (37/37/41 of 45 anchors; 491/496/492 of 600) and the per-item swings reach ±0.5 — the null is **cancellation,
   not insensitivity**.
7. **Model family and size move length behaviour, not wording.** Mistral over-decomposes (signed error +0.585),
   Qwen2.5-7B under-decomposes (−0.150), Qwen2.5-14B lands at +0.285 with the block's best step_count_MAE 0.5117;
   every n-gram/graph/step-overlap term is statistically indistinguishable in both the family pair (exp-021 vs 020)
   and the size pair (exp-022 vs 021). What size buys is granularity and chaining: under-decomposition
   0.2933→0.1017 (p=7.3e-23) and chain_validity 0.9210→0.9838.
8. **A 2/3-hop-only fine-tune does not generalise to 4 hops on step count.** exp-013 at hop 4: hop_EM 0.145
   (29/200) vs 0.300 for the prompting baseline (p=4.7e-5) and 0.410 for the all-hop fine-tune (p=5.9e-10);
   under-decomposition 0.845 (169/200); break_EM 0/200. But chain *well-formedness* survives (chain_validity
   200/200), so the gap is length/granularity, not malformed chains.
9. **Pool construction is a measured null with a verified mechanism.** exp-010: no step-quality metric separates
   imbalanced/balanced/clustered; exp-019 proved the arms really were contrasted — pools share 201/2000 ids
   (Jaccard 0.0529), 0/750 identical top-5 exemplar lists, 0/150 identical prompts, 688–700/750 different
   generations — so the null is a finding, not a plumbing defect.
10. **Cross-encoder rerank does move quality**, contrary to the circulating slide: exp-018, 9/9 matched pairs
    positive on ROUGE-L (3 survive Holm) and 9/9 improve on GED (4 survive Holm), while EM/hop_EM/chain_validity
    are null. This is the positive control that makes items 6 and 9 credible rather than a dead harness.
11. **Better decompositions do give better answers, in ordering.** exp-017, same reader and context on P600:
    answer EM 0.1800 (no decomposition) < 0.2133 (prompting) < 0.2417 (fine-tuned) < 0.2800 (gold decompositions).
    Descriptive ordering — no paired significance test was run across the four cells.
12. **Runaway generation length is not a quality failure mode.** The `unguided_capped` arm fired on 12/600 rows
    (exp-004) and 4/600 (exp-005); a post-hoc simulation at the maximally defensible cap of 4 (84/600 rows)
    moves step_f1 by +0.0011 and leaves EM unchanged (2026-08-22 error analysis §5).

---

## 5. What is UNMEASURED that the README's idea implies

The README's pipeline is router → decomposer → jury, on MetaQA-then-MuSiQue.

- **End-to-end answer accuracy for almost everything.** Only exp-017's 4 cells exist, all Mistral-reader,
  all sourced from exp-004/exp-008. **No answer-side number exists for any Qwen arm, for exp-015/020–028,
  for exp-026, for exp-013, or for the router.** Issue #16 is the open home.
- **The router's effect on the pipeline.** exp-012 measured decomposition only. Router-hop-matched *retrieval*
  is not even runnable (today's router writes no query id — exp-007/exp-009, ADR 0022). The parse-failure
  defect is issue #47; a regressor router is issue #23; neither has run.
- **The jury** — never implemented or used (README).
- **MetaQA** — zero v2 experiments. All 28 rows are MuSiQue. The MetaQA compile-execute wrapper is issue #16;
  the MetaQA+MuSiQue training mix is issue #28, undecided. The only MetaQA numbers anywhere are v1
  (`docs/prior-work.md` §1–2) and may not be cited as v2.
- **Cross-dataset generalisation** — nothing. exp-013 tests generalisation across *hop depth only*, inside MuSiQue.
- **Variance across seeds.** Every arm ran once, greedy, seed 42. There is no repeat of any arm and therefore
  no variance estimate; every "tied within noise" statement is about sampling noise over the 600 items only
  (block report §6.3).
- **Qwen coverage.** No Qwen model has ever been fine-tuned — every LoRA arm is Mistral (exp-001/008/013/026).
  No Qwen guided/oracle/router arm exists on the block pool. exp-005 is the lone Qwen row on the v1 pool and it
  contradicts ADR 0030's Mistral-only oracle-ceiling premise — issue #49.
- **Precision axis beyond Mistral.** exp-015↔exp-020 is the only fp16-vs-4-bit pair. Qwen2.5-7B and Qwen3.5-9B
  were never run unquantized on the block pool; the 14B cannot fit uncompressed on the 3090 at all.
- **The `pool_2000` seeded hop-stratified control** — never trained, so exp-026 cannot separate training identity,
  volume and hop mix (issue #13).
- **The third exemplar ordering `complexity_asc`** — reserved and refused at load pending a definition (ADR 0032 item 5).
- **Pool-size sweep at 1k granularity** (2k…8k) — deferred, issue #51.
- **Per-hop significance.** Every per-hop table in the record is descriptive; all tests run on the pooled 600.
- **What exp-010's 18 cells would score at a 1024-token cap** (the audit bounds only the direction), and any
  effect of the 128-token cap on step-level metrics.
- **exp-015's and exp-008's peak VRAM** — never sampled, so no memory saving is claimed for 4-bit anywhere.
- **Whether the primary metric is adopted** — issue #6 item 5; `docs/METRICS.md` describes what is reported, not what is settled.

---

## 6. Model coverage matrix

Condition axes: prompting unguided / oracle- or router-guided / fine-tuned / 4-bit / full precision.

| model | prompting unguided | guided (oracle/router/capped) | fine-tuned | 4-bit | fp16 / full precision |
|---|---|---|---|---|---|
| **Mistral-7B-Instruct-v0.3** | exp-004 (v1 pool), exp-009 (both retrieval arms), exp-015 (size2000), exp-020, exp-023 | exp-004 oracle_guided + unguided_capped; exp-012 router_guided (+ oracle re-score) | exp-001 train → exp-008 eval; exp-013 (2/3-hop only); exp-026 (pool-only) | exp-010 (18 cells), exp-020, exp-023; exp-001/026 train in 4-bit QLoRA | exp-004, exp-008, exp-009, exp-012, exp-013, exp-015, exp-026 inference, exp-017 reader |
| **Qwen3.5-9B** | exp-005 (v1 pool), exp-027 (size2000), exp-028 (reversed) | exp-005 oracle_guided + unguided_capped | **none** | exp-027, exp-028 | exp-005 |
| **Qwen2.5-7B-Instruct** | exp-021, exp-024 (reversed) | **none** | **none** | exp-021, exp-024 | **none** |
| **Qwen2.5-14B-Instruct** | exp-022, exp-025 (reversed) | **none** | **none** | exp-022, exp-025 | **none** (does not fit on the 3090) |
| e5-small-v2 bi-encoder + ms-marco-MiniLM-L-12-v2 cross-encoder | retrieval only — exp-006, exp-007, exp-010, exp-014, exp-018 | — | — | — | CPU |

Parameter counts on record: Mistral 7,248,023,552 (base) / 7,289,966,592 (+LoRA); Qwen2.5-7B 7,615,616,512;
Qwen2.5-14B 14,770,033,664 (**98.5% of the raised 1.5e10 decomposer ceiling — pending supervisor confirmation;
if 8B or 10B is reasserted, exp-022 and exp-025 are withdrawn**); Qwen3.5-9B 8,953,803,264 runtime /
9,653,104,368 checkpoint (issue #55), also above the standing 8B.

---

## 7. Confounds and caveats the analyses themselves raised

1. **Truncation confound (exp-010's 128-token cap) — raised and dismissed** (2026-08-23 audit): the gaps survive
   on cap-free rows and strengthen, and cap-hit rows *over*-decompose, the opposite of the hypothesis. What those
   cells would score at 1024 tokens remains unmeasured.
2. **Precision axis mismatch.** exp-004/005/008/015 are fp16; exp-010 and exp-020–028 are 4-bit; exp-026 trains in
   4-bit and infers in fp16. Any cross-block comparison silently moves precision unless it is exp-015↔exp-020.
3. **Pool mismatch.** The v1 9,156 pool and the size2000_imbalanced pool are different retrieval universes; arms
   across them are not comparable (explicit in exp-012's and exp-027's rows).
4. **Eval-set mismatch.** D750 ∩ P600 = 405/600 (exp-018, exp-019).
5. **`few_shot.k` reporting (issue #53).** Reranked run notes say `k=3` where 5 exemplars were shown.
6. **Ordering nulls are cancellation.** 82% of plans are rewritten under reversal with ±0.5 per-item swings that
   average out (anchor note §3, corrected into block report §9.2).
7. **`chain_validity` is blind to orphan steps** (86 and 83 item×arm pairs across the two 45-item draws).
8. **Exact-string step matching zeroes plainly correct plans** (13/45 and 8/45 all-arm zeros).
9. **Legacy composite defect (issue #40)** — junk outranks real arms; never headline it.
10. **GED:** node-cap substitution bias up to +0.0588 (largest count 6/600, exp-005 oracle); order-blind on 2-step
    plans; absolute values not externally comparable (no spaCy). **SARI floor ≈0.29**, so absolute levels are
    uninterpretable — differences on the same data are not.
11. **No multiple-comparison correction across the 13 statistics** (ADR 0009). exp-024's lone hit has a CI lower
    bound of +0.0013; two of exp-028's three sit near the boundary.
12. **Prompt-style asymmetry.** Mistral uses `prompt_style: plain`, the Qwen folders `chat_template` — the literal
    strings differ and cannot not differ. How much of the family effect is the template is unmeasured.
13. **exp-009's hop-4 confound** — the 4-hop pool bucket is the smallest (1,175/9,156), which is where oracle
    hop-matching loses on step quality.
14. **ADR 0021's two self-recorded pool confounds** — `num_trials: 1` (strategy mixed with draw noise), and
    construction shares the e5-small-v2 embedding space with retrieval.
15. **exp-012's router hop-2 recall is inflated** by 394 defaulted rows; the drift-check reuse of exp-004's
    unguided arm is an argument from targeted diffs, not proof by re-generation.
16. **Commit provenance (issue #54)** and the Qwen3.5-9B parameter-count disagreement (issue #55).
17. **exp-017 cell 4 (`no_decomposition_control`) is lead-added** and still pending Jahid's and the supervisor's
    acceptance as a standing arm.
18. **Anchor-sheet draws are difficulty-biased** in a direction consistent across all arms; read anchors for
    mechanism, the 600 for numbers. Arm-vs-arm orderings read off 45 items do not replicate (4 of 40 findings
    failed, notably exp-008 vs exp-026, which flipped from even to a clean exp-008 sweep).
19. **Two arms' validity is conditional** on the pending 1.5e10 ceiling confirmation (exp-022, exp-025; and
    exp-005/027/028 sit above the standing 8B under ADR 0015).
20. **exp-011-sourced comparators omit three columns** (`over`/`under`/`decomp_mean`) from every paired test
    against exp-008.
