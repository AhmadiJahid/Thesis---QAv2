# Are v1's masking and retrieval comparisons statistically significant?

**Date:** 2026-08-20 · **Author:** analyst agent session · **Scope:** issue #6 item 4 (decomposer
retrieval default) · **Runs:** none — this is a re-analysis of existing v1 artifacts, so there is
no `experiments/log.md` entry.

---

## Prior-work caveat — read this before quoting any number below

**Every input to this note is a v1 artifact** produced in the predecessor repo
(`/cta/users/fyilmaz/Thesis---QA`, treated as read-only here) **before v2's rules existed**.

- v1's `runs/` is gitignored and **untracked** — `git ls-files runs/` in v1 returns **0 files**
  (ran; observed 0). The per-item files this note analyses therefore **carry no commit SHA**;
  their only provenance is path + mtime + content hash (§2). This is the ADR 0005 situation: v1
  runs are not tied to committed code.
- Consequently **these are citable prior work, not v2 experimental claims.** Nothing here may be
  reported as a v2 measurement, and none of it satisfies Gate 2 (committed code + committed config
  + fixed seed). Any of these comparisons that the thesis wants to *assert* has to be re-run in v2
  first (ADR 0001; which baselines and when is Jahid's call).
- What *is* new and reproducible here is the **statistics**: the significance battery was computed
  in this session from the v1 per-item scores, using v2's committed protocol code (§1). The
  significance verdicts are properties of v1's numbers; they inherit v1's provenance gap.

---

## 1. Method — what was run

The house protocol is ADR 0009 as amended 2026-08-20 (paired bootstrap + exact McNemar, with a
paired t-test added alongside per issue #30), implemented in
`scripts/musique_decompositions_evaluator.py` `--compare` and documented in `docs/METRICS.md` §5.

v1's per-item files are **bare JSON lists** with **no `item_id`** and no stamped composite
weights, so v2's `--compare` refuses them by design (`_load_per_item` / `_require_matching_weights`).
The battery was therefore run from a session-local harness that **imports v2's own functions**
(`_paired_bootstrap`, `_statistic_arrays`, `_mcnemar_exact_p`) and re-implements only the alignment
step. Parameters are v2's (`configs/musique_eval.json`): 10,000 bootstrap resamples, chunk size
1,000, α = 0.05, seed 42, `numpy.random.default_rng`, one index matrix applied to both systems,
percentile CI of (A − B), composite recomputed on every resample with weights
0.4 / 0.3 / 0.2 / 0.1 and step-count scale 3.0. **Those weights and that scale are identical in v1**
(`Thesis---QA/scripts/musique_decompositions_evaluator.py:411-416`, read this session), so the
composite compared here is the same quantity v1 reported.

**Validation (ran; observed):** for the four statistics v2's protocol covers (`rouge_l_f1`,
`step_f1`, `ordered_step_accuracy`, `composite_score`), the harness's difference, CI low and CI high
were compared to v2's `_paired_bootstrap` called directly on the same rows, with **zero tolerance**
(`abs_tol=0`). Result: **bit-identical on both a Task-A pair and a Task-B pair** (2/2 checks OK).
Two statistics were added to the same bootstrap (`exact_match`, `hop_count_exact_match` means) plus
one diagnostic (§4); adding statistics does not perturb the RNG stream, so the four house numbers
are exactly what `--compare` would print.

- **Exact McNemar** (two-sided, integer binomial on discordant pairs) for the two binary per-item
  metrics: `exact_match`, `hop_count_exact_match`.
- **Paired t-test**: `scipy.stats.ttest_rel` (**scipy 1.17.1 is present** in `.venv` — checked; no
  hand-rolled fallback was needed) over per-item differences, for all five per-item metrics. The
  **composite has no per-item value** (its reference term is a micro rate, its step-count term a
  MAE), so no t-test and no McNemar is reported for it — only the bootstrap, which recomputes it.
- **Per gold hop** (2/3/4): the whole battery re-run on each hop subset, seed 42 on that subset.
  All strata have n ≥ 200, above `min_items_for_significance_claim` = 30.
- **No multiple-comparison correction in the headline**, per ADR 0009. Holm-Bonferroni was computed
  separately within stated families and is reported where it changes the reading (§5, §7).

**Reproducibility gap, stated plainly:** the harness is not committed (analyst output is analysis,
not code). The four house statistics are re-derivable today by re-stamping the v1 per-item files
into v2's schema and running `--compare`; the EM / hop-EM bootstrap columns and the diagnostic in §4
are not, until such a shim exists. See recommendation R5.

## 2. Inputs — paths and content hashes

All paths under `/cta/users/fyilmaz/Thesis---QA` (read-only; not modified). v1 HEAD is `a020fd6`
with a clean tree, but as stated above the run files are untracked, so that SHA does **not** pin
them.

**Task A** — `runs/musique_decomposition_eval/` (mtime 2026-04-13; sha256, first 16 hex):

| variant | per-item file | sha256[:16] | mtime |
|---|---|---|---|
| typed | `eval_typed_unguided_per_item.json` | `53ab0a08f380db9a` | 2026-04-13 13:07:57 |
| uniform | `eval_uniform_unguided_per_item.json` | `a0348260dd4a7d73` | 2026-04-13 13:08:14 |
| raw | `eval_raw_unguided_per_item.json` | `80e64a38b7934c52` | 2026-04-13 13:07:52 |

Companion metrics files used for cross-checking: `eval_typed_unguided_metrics.json`
`c4c9634b9287e354`, `eval_uniform_unguided_metrics.json` `4cff2acebaa9eac2`,
`eval_raw_unguided_metrics.json` `c4a4093bdd479aab`. The three eval configs all record
`seed: 42`, `limit: null`, gold `MusiQue/Data/dev_data/musique_ans_v1.0_dev_clean.jsonl`, and
prediction paths `runs/decomposer_raw_unguided/{typed,masked,raw}/results.json` — note **"uniform"
is the `masked` predictions path**, as `docs/prior-work.md` §3 says.

**Task B** — `runs/pool_sweep/eval/<cell>/eval_per_item.json`, 33 cells, mtimes 2026-04-20 05:05 →
2026-04-21 00:51. sha256[:12] per cell, in `ls` order:
`ee64da7b422f`, `43231fdbb102`, `5792aa706a92`, `2e5530974c1e`, `f911796f96a4`, `4c3fc18b7138`,
`7144c097bfe4`, `e353c76ae359`, `9547be04e6ea`, `648a213520d0`, `bd2f1d6fd499`, `c1e6f9759385`,
`70e58b63076c`, `5fad3c5f009e`, `dc5abb6406e0`, `b2bfd5eda5d8`, `b054f418dd52`, `543f4cfc77c7`,
`e06ffc80a9c1`, `fd64acc34395`, `21fd8dd22295`, `a53aabd9f3cc`, `3a0fb90b6f0c`, `6ac7af550531`,
`1ab9623bce72`, `5444b2e2a60e`, `18451b8cfe7b`, `0c5bf23b087a`, `2a2dd118ee32`, `18fcac5e0213`,
`fb5399e1ddc2`, `a0df3f17bbd7`, `c5ccdb504f23`
(cell order: `size1000_balanced` {only,plus_ce} × {raw,typed,uniform}, `size1000_imbalanced` …,
`size2000_balanced__biencoder_only` …, `size2000_imbalanced` …, `size4000_imbalanced` …,
`size8000_imbalanced` …). Aggregate CSV `runs/pool_sweep/summary/all_runs.csv` sha256
`89c5e923ddbb2a8737c12bda160e12b7df7d4f1ed7add3767604b6419ba51e6e` (used only to confirm the cell
inventory; **no statistic in this note is computed from aggregates**).

## 3. Item-set verification

**Task A (ran; observed).** All three files hold **600 rows each, 600 distinct normalized questions
each, 0 duplicates**. The three question sets are **identical** (`typed == uniform == raw`, True), so
the comparison is on the full 600 — **no intersection fallback was needed**. Gold is identical
across the three files for every item (`gold_steps` and `gold_hop_count`, 600/600 match). Gold hop
distribution: **200 / 200 / 200** for hops 2 / 3 / 4 — the ADR 0007 pinned shape.

**Aggregate cross-check.** Re-aggregating the per-item rows reproduced each variant's committed
`_metrics.json` to a maximum absolute delta of **3.3e-16** (typed), **2.2e-16** (uniform),
**3.3e-16** (raw) across EM, step F1, ordered accuracy, ROUGE-L F1, hop EM, composite,
`reference_validity_micro` and `step_count_abs_error_mae`. The reference counts stated in
`docs/prior-work.md` §3 are confirmed exactly: **typed 6/8, uniform 7/9, raw 0/4** valid/total
`[#k]` references over 600 predictions.

**Task B (ran; observed).** 33 eval cells, **750 rows each**, and the **normalized question
sequence is identical across all 33 cells, order included** (True). Hop distribution 250 / 250 / 250.
One question text appears twice (rows 370 and 376) with **identical gold** — a genuinely duplicated
eval item, which is why keying by question is not usable and why v2's `--compare` would abort on
these files. Alignment is therefore **positional**, and for every matched pair each row's normalized
question **and** `gold_steps` were asserted equal between the two cells (verified True for all 15
pairs). This is a documented deviation from v2's id-based alignment, forced by v1's file format; it
is safe here only because the sequences are literally identical.

## 4. Task A — masking (typed / uniform / raw), n = 600, all three pairwise

`*` and **yes** mark a 95% bootstrap CI that excludes 0. "b/c" are McNemar's discordant counts
(right only in A / right only in B).

| pair | metric | A | B | diff | 95% CI | CI sig | McNemar b/c, p | t (dof=599), p |
|---|---|---|---|---|---|---|---|---|
| typed vs raw | EM | 0.0583 | 0.0367 | +0.0217 | [+0.0033, +0.0400] | **yes** | 21/8, p=0.0241 | t=2.424, p=0.0157 |
| typed vs raw | step F1 | 0.2006 | 0.1775 | +0.0232 | [+0.0053, +0.0412] | **yes** | n/a (not binary) | t=2.545, p=0.0112 |
| typed vs raw | ord acc | 0.1794 | 0.1581 | +0.0213 | [+0.0039, +0.0388] | **yes** | n/a (not binary) | t=2.385, p=0.0174 |
| typed vs raw | ROUGE-L F1 | 0.5442 | 0.5315 | +0.0126 | [+0.0001, +0.0250] | **yes** | n/a (not binary) | t=1.997, p=0.0463 |
| typed vs raw | hop EM | 0.5217 | 0.4850 | +0.0367 | [-0.0083, +0.0817] | no | 107/85, p=0.1294 | t=1.590, p=0.1124 |
| typed vs raw | composite | 0.3606 | 0.1888 | +0.1718 | [-0.1842, +0.2259] | **no** | n/a | n/a (no per-item value) |
| typed vs uniform | EM | 0.0583 | 0.0550 | +0.0033 | [-0.0133, +0.0200] | no | 15/13, p=0.8506 | t=0.378, p=0.7058 |
| typed vs uniform | step F1 | 0.2006 | 0.1896 | +0.0111 | [-0.0064, +0.0290] | no | n/a | t=1.225, p=0.2211 |
| typed vs uniform | ord acc | 0.1794 | 0.1662 | +0.0132 | [-0.0038, +0.0303] | no | n/a | t=1.515, p=0.1302 |
| typed vs uniform | ROUGE-L F1 | 0.5442 | 0.5431 | +0.0011 | [-0.0107, +0.0127] | no | n/a | t=0.178, p=0.8589 |
| typed vs uniform | hop EM | 0.5217 | 0.5050 | +0.0167 | [-0.0283, +0.0617] | no | 96/86, p=0.5048 | t=0.741, p=0.4590 |
| typed vs uniform | composite | 0.3606 | 0.3554 | +0.0053 | [-0.1841, +0.1868] | no | n/a | n/a |
| uniform vs raw | EM | 0.0550 | 0.0367 | +0.0183 | [+0.0033, +0.0333] | **yes** | 16/5, p=0.0266 | t=2.410, p=0.0163 |
| uniform vs raw | step F1 | 0.1896 | 0.1775 | +0.0121 | [-0.0038, +0.0279] | no | n/a | t=1.484, p=0.1383 |
| uniform vs raw | ord acc | 0.1662 | 0.1581 | +0.0081 | [-0.0075, +0.0236] | no | n/a | t=1.032, p=0.3025 |
| uniform vs raw | ROUGE-L F1 | 0.5431 | 0.5315 | +0.0115 | [-0.0010, +0.0244] | no | n/a | t=1.785, p=0.0747 |
| uniform vs raw | hop EM | 0.5050 | 0.4850 | +0.0200 | [-0.0283, +0.0683] | no | 112/100, p=0.4500 | t=0.824, p=0.4103 |
| uniform vs raw | composite | 0.3554 | 0.1888 | +0.1665 | [-0.1875, +0.2143] | no | n/a | n/a |

**The three tests agree everywhere they overlap.** Every metric flagged by the CI has t-test
p < 0.05, and every metric the CI does not flag has t-test p > 0.05; both McNemar verdicts match
their CI verdict. There is no case in the overall table where the bootstrap and the t-test disagree.

### Per gold hop (n = 200 per stratum)

| pair | hop | EM | step F1 | ord acc | ROUGE-L F1 | hop EM | composite | composite w/o ref term |
|---|---|---|---|---|---|---|---|---|
| typed vs raw | 2 | +0.0300 | +0.0397* | +0.0392 | +0.0183 | +0.0400 | +0.0298* | +0.0372* |
| typed vs raw | 3 | +0.0150 | +0.0241 | +0.0254* | +0.0185 | +0.0350 | -0.1769 | +0.0289* |
| typed vs raw | 4 | +0.0200* | +0.0057 | -0.0008 | +0.0010 | +0.0350 | +0.0126 | +0.0157 |
| typed vs uniform | 2 | -0.0100 | +0.0019 | +0.0072 | -0.0054 | +0.0250 | +0.2029 | +0.0036 |
| typed vs uniform | 3 | +0.0050 | +0.0145 | +0.0242* | +0.0056 | +0.0300 | +0.0137 | +0.0171 |
| typed vs uniform | 4 | +0.0150 | +0.0168 | +0.0081 | +0.0030 | -0.0050 | -0.1842 | +0.0198 |
| uniform vs raw | 2 | +0.0400 | +0.0378* | +0.0320 | +0.0237 | +0.0150 | -0.1731 | +0.0336* |
| uniform vs raw | 3 | +0.0100 | +0.0096 | +0.0012 | +0.0129 | +0.0050 | -0.1906 | +0.0117 |
| uniform vs raw | 4 | +0.0050 | -0.0111 | -0.0089 | -0.0020 | +0.0400 | +0.1967 | -0.0041 |

Per-hop McNemar and t-tests were computed too. Across all 60 Task-A comparisons where both the
bootstrap CI and the t-test apply (3 pairs × 5 metrics × {overall, hop 2, hop 3, hop 4}) the two
verdicts disagree in **exactly one** cell, named below. Notable per-hop t-tests:
typed vs raw step F1 at hop 2 t=2.013 p=0.0455 (dof=199); typed vs raw ordered accuracy at hop 3
t=2.115 p=0.0357; typed vs raw EM at hop 4 t=2.015 p=0.0452 but **McNemar p=0.125 on 4/0 discordant
pairs** — with only four discordant items the exact test cannot reach 0.05 (its minimum attainable
p is 0.125), so that cell is a power artifact, not a finding. typed vs uniform ordered accuracy at
hop 3 has CI [+0.0002, +0.0492] (nominally significant) while t p=0.0563 — a genuinely borderline
cell that should not be leaned on.

**Direction of the typed advantage:** typed > raw is concentrated in **hops 2 and 3**. At hop 4 the
step-level gaps collapse (step F1 +0.0057, ordered accuracy −0.0008, ROUGE-L +0.0010, none
significant). The masking benefit, such as it is, is a shallow-question benefit in these runs.

### The typed-vs-raw composite gap is almost entirely one term (Task A.3)

Exact decomposition of the composite difference into its four weighted terms (ran; the four
contributions sum to the total to 1e-6):

| pair | total Δcomposite | 0.4·Δstep F1 | 0.3·Δord acc | 0.2·Δref_micro | 0.1·Δstep-count term |
|---|---|---|---|---|---|
| typed vs raw | **+0.1718** | +0.0093 (5.4%) | +0.0064 (3.7%) | **+0.1500 (87.3%)** | +0.0062 (3.6%) |
| uniform vs raw | +0.1665 | +0.0048 (2.9%) | +0.0024 (1.5%) | **+0.1556 (93.4%)** | +0.0037 (2.2%) |
| typed vs uniform | +0.0053 | +0.0044 | +0.0040 | −0.0056 | +0.0024 |

`reference_validity_micro` is **0.75 for typed, 0.7778 for uniform, 0.0000 for raw**, and those
rates rest on **8 references from 3 items (typed), 9 from 4 items (uniform), 4 from 1 item (raw)**
out of 600 predictions. So **87.3%** of the headline typed-vs-raw composite gap that
`docs/prior-work.md` §3 reports is produced by a 0.2-weighted micro rate decided by **four
predictions in total**. `docs/prior-work.md`'s caveat is confirmed and now quantified.

Three consequences, all measured:

1. **The composite gap is not significant.** typed vs raw composite CI is
   **[−0.1842, +0.2259]** — width 0.41 on a 0-to-1 scale, straddling 0. The bootstrap resamples the
   handful of reference-emitting items in or out, so the term flips between 0 and 0.75 across
   resamples. The same happens for uniform vs raw ([−0.1875, +0.2143]) and typed vs uniform
   ([−0.1841, +0.1868]). **No composite comparison among the three masking variants is significant
   at n = 600.**
2. **The step-level gap does survive.** With the reference term removed and the remaining weights
   renormalized (a diagnostic quantity constructed for this note, **not** a house metric), typed vs
   raw is **+0.0273, CI [+0.0107, +0.0440] — significant**, and it stays significant within hop 2
   (+0.0372) and hop 3 (+0.0289) separately. uniform vs raw on the same diagnostic is +0.0137,
   CI [−0.0015, +0.0289] — **not** significant.
3. **Per hop the composite term outright reverses the ranking.** At hop 3 typed's ref_micro is 0.00
   while raw's is 1.00 (raw emitted no references in that stratum, which scores 1.0 by the micro
   convention when the denominator is empty), so the hop-3 composite reads **−0.1769 in raw's
   favour** while every step-level metric in that stratum favours typed. The composite is not a
   usable ranking signal at these reference volumes.

### Power at n = 600

Minimum detectable difference (paired, α = 0.05 two-sided, 80% power, from the observed
per-item difference SDs), and the n the observed difference would need:

| pair | metric | observed diff | sd(diff) | MDE at n=600 | n needed for observed diff |
|---|---|---|---|---|---|
| typed − raw | step F1 | +0.0232 | 0.2228 | 0.0255 | 727 |
| typed − raw | ord acc | +0.0213 | 0.2184 | 0.0250 | 828 |
| typed − raw | EM | +0.0217 | 0.2190 | 0.0250 | 802 |
| typed − raw | hop EM | +0.0367 | 0.5650 | 0.0646 | 1,863 |
| typed − uniform | step F1 | +0.0111 | 0.2212 | 0.0253 | 3,139 |
| typed − uniform | ord acc | +0.0132 | 0.2129 | 0.0244 | 2,051 |
| uniform − raw | step F1 | +0.0121 | 0.1996 | 0.0228 | 2,137 |

Read: n = 600 buys ~2.5 points of step F1 / ordered accuracy. The typed-vs-raw effects sit **right
at** that threshold — they are significant, and they are marginal. **typed vs uniform would need
roughly 2,000–3,000 items** to resolve at the observed effect size; declaring it a tie at n = 600 is
a statement about power, not about the two modes being equal.

## 5. Task B — bi-encoder vs bi-encoder + cross-encoder rerank

**Per-item data survives** — `runs/pool_sweep/eval/<cell>/eval_per_item.json` exists for all 33
cells, so the "no per-item data ⇒ no test computable" fallback does **not** apply. **15 matched
pairs** exist (same size, balance, trial and mask mode, differing only in retrieval variant):
size1000 balanced and imbalanced, size2000 imbalanced, size4000 imbalanced, size8000 imbalanced,
each × {raw, typed, uniform}. The three `size2000_balanced_trial0__biencoder_only__*` cells have
**no `biencoder_plus_ce` counterpart** and are excluded — that is why 33 cells yield 15 pairs, not 16.
Each pair is n = 750 on the identical, identically-ordered eval sequence.

Differences are **(+CE) − (bi-encoder only)**; `*` = 95% CI excludes 0.

| pool cell (size_balance__mask) | EM | step F1 | ord acc | ROUGE-L F1 | hop EM | composite |
|---|---|---|---|---|---|---|
| size1000_balanced__raw | -0.0053 | +0.0119 | +0.0045 | +0.0079 | -0.0027 | +0.1883 |
| size1000_balanced__typed | -0.0040 | +0.0131 | +0.0111 | +0.0072 | +0.0187 | +0.0947 |
| size1000_balanced__uniform | -0.0080 | +0.0079 | +0.0044 | +0.0129* | -0.0120 | +0.0345 |
| size1000_imbalanced__raw | +0.0040 | +0.0156* | +0.0173* | +0.0049 | +0.0160 | +0.0422 |
| size1000_imbalanced__typed | +0.0053 | +0.0146* | +0.0151* | +0.0199* | +0.0360 | +0.0141 |
| size1000_imbalanced__uniform | +0.0013 | +0.0086 | +0.0099 | +0.0116* | +0.0893* | +0.0035 |
| size2000_imbalanced__raw | -0.0093 | +0.0011 | +0.0006 | +0.0167* | +0.0000 | -0.1970 |
| size2000_imbalanced__typed | -0.0040 | +0.0015 | +0.0014 | +0.0161* | +0.0520* | -0.0199 |
| size2000_imbalanced__uniform | +0.0053 | +0.0038 | +0.0033 | +0.0095 | +0.0307 | +0.2063* |
| size4000_imbalanced__raw | -0.0053 | +0.0124 | +0.0112 | +0.0177* | +0.0573* | -0.1899 |
| size4000_imbalanced__typed | +0.0067 | +0.0114 | +0.0178* | +0.0172* | +0.0173 | -0.0864 |
| size4000_imbalanced__uniform | -0.0053 | +0.0105 | +0.0090 | +0.0101 | +0.0493* | -0.1889 |
| size8000_imbalanced__raw | +0.0040 | +0.0057 | +0.0075 | +0.0071 | +0.0120 | +0.0063 |
| size8000_imbalanced__typed | +0.0027 | +0.0165* | +0.0211* | +0.0136* | +0.0213 | +0.1361 |
| size8000_imbalanced__uniform | -0.0053 | -0.0069 | -0.0075 | +0.0079 | -0.0147 | +0.0348 |

Counts across the 15 pairs (CI-significant / t-significant / McNemar-significant, all α = 0.05,
uncorrected):

| metric | Δ > 0 | Δ < 0 | Δ = 0 | CI sig | t sig | McNemar sig |
|---|---|---|---|---|---|---|
| exact match | 7 | 8 | 0 | **0** | **0** | **0** |
| step F1 | 14 | 1 | 0 | 3 | 3 | n/a |
| ordered step accuracy | 14 | 1 | 0 | 4 | 4 | n/a |
| ROUGE-L F1 | **15** | **0** | 0 | **8** | **8** | n/a |
| hop-count EM | 11 | 3 | 1 | 4 | 4 | 4 |
| composite | 10 | 5 | 0 | 1 | n/a | n/a |

The CI and t-test verdicts agree **75/75** on the overall rows where both apply (five metrics ×
15 pairs; the composite has no t-test), and McNemar agrees with the CI on **both** binary metrics in
all 15 pairs. Including the per-hop strata (300 comparisons), the CI and t-test disagree in 2 cells,
both borderline: `size1000_imbalanced__uniform` hop 3 hop EM (CI [+0.0000, +0.1520], t p = 0.0486)
and `size4000_imbalanced__typed` hop 2 ordered accuracy (CI [+0.0003, +0.0683], t p = 0.0502).

**Multiplicity matters here** (15 pairs are a family). Holm-Bonferroni within each 15-test family:

- **ROUGE-L F1: 5 of 15 survive** — `size1000_imbalanced__typed` (adj p = 0.0030),
  `size4000_imbalanced__raw` (0.0123), `size2000_imbalanced__typed` (0.0250),
  `size4000_imbalanced__typed` (0.0259), `size2000_imbalanced__raw` (0.0302).
- **hop-count EM: 1 survives** — `size1000_imbalanced__uniform` (t adj p = 0.0004; McNemar adj
  p = 0.0005, discordant 162/95).
- **ordered accuracy: 1 survives** — `size8000_imbalanced__typed` (adj p = 0.0370).
- **step F1: 0 survive. Exact match: 0 survive.**

The 15 pairs are **not independent** (identical 750 eval items; nested pools), so the sign counts
above are descriptive; no across-pair test was run and none is claimed. The strongest statement the
data supports: **+CE raises ROUGE-L F1 in 15/15 pairs, significantly in 8 uncorrected and 5 after
Holm, at a magnitude of +0.005 to +0.020**, and **never moves exact match** (0/15 by any test, with
signs split 7/8). Composite is uninformative for the same reason as §4 — **14 of its 15 CIs straddle 0**,
with widths from 0.2111 to 0.4083.

## 6. What contradicts, and what confirms, `docs/prior-work.md`

- **Confirmed:** typed's advantage over raw on step F1 / ordered accuracy / EM / ROUGE-L is real at
  n = 600 (uncorrected) — FINDINGS' "prefer typed (or at least masked) over raw" is supported on the
  step-level metrics.
- **Confirmed and quantified:** the composite gap 0.3606 vs 0.1888 is 87.3% one micro term resting
  on four predictions, and it is **not statistically significant**.
- **Not supported as stated:** §4's aggregate reading that "cross-encoder rerank is not a free win
  on mean composite" is true of the *mean composite* but understates ROUGE-L, where +CE wins
  15/15 pairs. Conversely, §4's "helps … especially hop-count EM / step F1 at larger pools" is only
  partly borne out: hop EM's single Holm-surviving win is at **size 1000** (imbalanced, uniform),
  and step F1's three uncorrected wins are at sizes 1000 and 8000 — the size pattern is not clean.
- **Unmeasured here:** pool size. Every pair in §5 holds size fixed, so **no claim in this note
  bears on the "~2000–4000" half of issue #6 item 4.** Cross-size matched pairs exist on the same
  750 items and the same battery would apply.

## 7. Ranked takeaways for the masking-default decision (#6 item 4)

Ranked by how much each would change the decision. These are options and evidence, **not** a chosen
direction — the call is Jahid's with his supervisor.

1. **Do not default to typed on the strength of the composite.** The composite typed-vs-raw gap is
   not significant (CI [−0.1842, +0.2259]) and is 87.3% one 0.2-weighted micro rate decided by four
   predictions. Any masking default justified by "0.3606 vs 0.1888" is justified by an artifact.
   This is the highest-impact item because it is the number most likely to be quoted.
2. **Typed over raw is defensible on the step-level metrics, and only just.** step F1 +0.0232
   (CI [+0.0053, +0.0412], t p=0.0112), ordered accuracy +0.0213, EM +0.0217 (McNemar p=0.0241),
   ROUGE-L +0.0126 (CI lower bound +0.0001 — the weakest of the four). All four sit at the n = 600
   detection threshold (MDE ≈ 0.025). If any multiplicity correction is applied across the 15-test
   Task-A family, **nothing survives** (smallest Holm-adjusted p = 0.1676); within the typed-vs-raw
   five-metric family alone the smallest is 0.0559. So: significant as ADR 0009 reports it
   (uncorrected), fragile under correction. Whether to correct is a supervisor question and is
   already flagged in ADR 0009.
3. **typed vs uniform is undecidable at n = 600 — do not record a winner.** Nothing is significant
   in either direction on any metric (largest effect: ordered accuracy +0.0132, t p=0.13). Resolving
   it at the observed effect size needs ~2,000–3,000 items. If the choice between typed and uniform
   matters to the thesis, the honest options are (a) enlarge the eval set, or (b) record the choice
   as made on non-statistical grounds (e.g. masking-pipeline simplicity) and say so.
4. **Cross-encoder rerank: adopt-or-not turns on which metric is primary.** +CE is the most
   consistent effect anywhere in this note on **ROUGE-L F1** (15/15 pairs positive, 8 uncorrected /
   5 Holm-surviving) and is **flat on exact match** (0/15, signs 7/8). That makes it a decision that
   cannot be taken before #6 item 5 (thesis-primary metric) — worth surfacing as a dependency rather
   than deciding twice.
5. **The reference-validity term needs a decision before the composite is used anywhere.** At v1's
   reference volumes (4–9 references per 600 predictions) the term is noise with a 0.2 weight, and
   its "no references ⇒ 1.0" convention makes a reference-free system beat a partly-correct one
   (visible at hop 3 in §4, where the composite reverses the ranking). This is direct input to
   issue #29 (composite bias check) and to #6 item 5.
6. **Two cheap follow-ups the same battery answers, both unmeasured today:** (a) pool-size pairs
   (2000 vs 4000 etc., matched on balance/mode/retrieval, same 750 items) — the other half of
   #6 item 4; (b) a `--compare` shim that accepts v1-format per-item files, which would make every
   number in §4–§5 reproducible from committed code instead of from a session-local harness.

## 8. Unmeasured / out of scope

Stated so it is not read into the note: pool size significance; balanced vs imbalanced pools; any
guided (hop-forced) condition; MetaQA; any v2 run. No model was run and no v1 file was modified.
The v1 predictions' own provenance (which decoder, which prompt revision, whether the 600 and the
750 were drawn with a seed) is not established by this note — see the caveat up top and ADR 0005.
