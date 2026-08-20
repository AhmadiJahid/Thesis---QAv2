# Did v1's few-shot pool size measurably affect decomposition quality?

**Date:** 2026-08-20 · **Author:** analyst agent session · **Scope:** issue #6 item 4, the *pool-size*
half left unmeasured by
[`2026-08-20-v1-masking-and-retrieval-significance.md`](2026-08-20-v1-masking-and-retrieval-significance.md)
§6 · **Runs:** none — this is a re-analysis of existing v1 artifacts under
[ADR 0020](../adr/0020-prior-work-re-analysis-convention.md), so there is no `experiments/log.md`
entry.

---

## Prior-work caveat — read this before quoting any number below

**Every input to this note is a v1 artifact** produced in the predecessor repo
(`/cta/users/fyilmaz/Thesis---QA`, read-only here and unmodified) **before v2's rules existed**.

- v1's `runs/` is gitignored and **untracked** — `git ls-files runs/` in v1 returns **0 files** (ran;
  observed 0). The 33 per-item files this note analyses therefore **carry no commit SHA**; their only
  provenance is path + mtime + content hash (§2). This is the ADR 0005 situation.
- Consequently **these are citable prior work, not v2 experimental claims.** Nothing here may be
  reported as a v2 measurement, and none of it satisfies Gate 2 (committed code + committed config +
  fixed seed). Anything the thesis wants to *assert* has to be re-run in v2 first (ADR 0001); which
  baselines and when is Jahid's call with his supervisor.
- What is new and reproducible here is the **statistics**: the battery was computed in this session
  from v1's per-item scores using v2's committed protocol code (§1). The verdicts are properties of
  v1's numbers and inherit v1's provenance gap.
- **The harness that produced these figures is not committed** (analyst output is analysis, not code),
  so the figures are **not re-derivable from committed code** until the `--compare` shim in §9 item 6(b)
  lands. Every statistic below is emitted machine-readably beside this note in
  [`2026-08-20-v1-pool-size-significance.json`](2026-08-20-v1-pool-size-significance.json)
  (statistics only — no dataset content, no per-item rows), which also carries the sha256 + mtime of
  every input file, the alignment order, the seed/resample parameters and the Holm-adjusted p-values.
- **This note decides nothing.** ADR 0006 already fixed pool size at 2000; §9 states what v1's own
  sweep does and does not say about that dimension, as conditionals.

---

## 1. Method — what was run

House protocol is ADR 0009 as amended 2026-08-20: **paired bootstrap + exact McNemar**, implemented
in `scripts/musique_decompositions_evaluator.py` `--compare` and documented in `docs/METRICS.md` §5,
**plus a paired t-test reported alongside** per the ADR 0009 amendment (issue #30). Be precise about
the second one: at this note's base SHA `50daea7` the t-test is **not** in the committed evaluator and
is not in `docs/METRICS.md` — its implementation was in progress in a separate lane at the time of
writing, and every t-test figure below comes from `scipy.stats.ttest_rel` in this session's harness
(§1, last paragraph). Parameters are v2's own
(`configs/musique_eval.json`): **10,000 bootstrap resamples, chunk size 1,000, α = 0.05, seed 42**,
`numpy.random.default_rng`, one index matrix applied to both systems, percentile CI of (A − B),
composite recomputed on every resample with weights 0.4 / 0.3 / 0.2 / 0.1 and step-count scale 3.0.
Those weights and that scale are identical in v1
(`Thesis---QA/scripts/musique_decompositions_evaluator.py:411-416`), so the composite compared here is
the same quantity v1 reported.

v1's per-item files are bare JSON lists with no `item_id` and no stamped composite weights, so v2's
`--compare` refuses them by design (`_load_per_item` / `_require_matching_weights`). The battery was
run from a session-local harness that **imports v2's own functions** (`_statistic_arrays`,
`_paired_bootstrap`, `_mcnemar_exact_p`) and re-implements only the alignment step plus two extra
bootstrap columns and one diagnostic.

**Validation (ran; observed).** For the four statistics v2's protocol covers (`rouge_l_f1`,
`step_f1`, `ordered_step_accuracy`, `composite_score`) the harness's difference, CI low and CI high
were compared against v2's `_paired_bootstrap` called directly on the same rows, at **zero tolerance**
(exact float equality). Result: **bit-identical on 3/3 spot-checked pairs, 12 fields each** — one
pool-size pair in the balanced arm, one balance pair, one pool-size pair in the imbalanced arm
(`validation_against_committed_paired_bootstrap` in the JSON). Adding statistics to the same bootstrap
does not perturb the RNG stream, so those four columns are exactly what `--compare` would print.

- **Bootstrap columns:** the four house statistics, plus the means of `exact_match` and
  `hop_count_exact_match`, plus one **diagnostic** — `composite_no_ref_renorm`, the composite with the
  `reference_validity_micro` term dropped and the remaining weights renormalized to sum to 1
  (0.5 / 0.375 / 0.125). It is **constructed for this note and is not a house metric**; §4.4 is why it
  is here.
- **Exact McNemar** (two-sided, integer binomial on discordant pairs) on the two binary per-item
  metrics `exact_match` and `hop_count_exact_match`.
- **Paired t-test:** `scipy.stats.ttest_rel` (scipy 1.17.1 present; no fallback needed) over per-item
  differences, for those same five per-item metrics. The composite and the composite-without-reference
  diagnostic **have no per-item value** (a micro rate and an MAE term), so neither a t-test nor
  McNemar is reported for them — only the bootstrap, which recomputes them.
- **Per gold hop (2/3/4):** the whole battery re-run on each 250-item subset, seed 42 on that subset.
  All strata are far above `min_items_for_significance_claim` = 30.
- **No multiplicity correction in the headline**, per ADR 0009. Holm-Bonferroni is computed within the
  stated families and reported in §4.3 and §5.

## 2. Inputs — paths and content hashes

All paths under `/cta/users/fyilmaz/Thesis---QA` (read-only; not modified). v1 HEAD is `a020fd6` with
a clean tree, but the run files are untracked, so that SHA does **not** pin them.

**Every statistic in §4, §5 and §7 is computed from the 33
`runs/pool_sweep/eval/<cell>/eval_per_item.json` files** below (sha256 first 12 hex; full hashes in
the JSON companion). **§6 is the one exception:** its nesting overlaps come from the six
`runs/pool_sweep/pools/*/pool.jsonl` files (pool item ids) and its exemplar counts from the six
`stats.json` beside them — all twelve hashed in the companion under
`inputs.pool_jsonl_files_used_for_nesting_only` and
`inputs.pool_stats_files_used_for_composition_only`.

| cell | sha256[:12] | mtime (UTC) |
|---|---|---|
| `size1000_balanced_trial0__biencoder_only__raw` | `ee64da7b422f` | 2026-04-20T18:24:03Z |
| `size1000_balanced_trial0__biencoder_only__typed` | `43231fdbb102` | 2026-04-20T18:49:53Z |
| `size1000_balanced_trial0__biencoder_only__uniform` | `5792aa706a92` | 2026-04-20T19:16:08Z |
| `size1000_balanced_trial0__biencoder_plus_ce__raw` | `2e5530974c1e` | 2026-04-20T19:41:03Z |
| `size1000_balanced_trial0__biencoder_plus_ce__typed` | `f911796f96a4` | 2026-04-20T20:05:27Z |
| `size1000_balanced_trial0__biencoder_plus_ce__uniform` | `4c3fc18b7138` | 2026-04-20T20:30:46Z |
| `size1000_imbalanced_trial0__biencoder_only__raw` | `7144c097bfe4` | 2026-04-20T02:32:52Z |
| `size1000_imbalanced_trial0__biencoder_only__typed` | `e353c76ae359` | 2026-04-20T02:58:15Z |
| `size1000_imbalanced_trial0__biencoder_only__uniform` | `9547be04e6ea` | 2026-04-20T02:05:33Z |
| `size1000_imbalanced_trial0__biencoder_plus_ce__raw` | `648a213520d0` | 2026-04-20T03:23:36Z |
| `size1000_imbalanced_trial0__biencoder_plus_ce__typed` | `bd2f1d6fd499` | 2026-04-20T03:47:31Z |
| `size1000_imbalanced_trial0__biencoder_plus_ce__uniform` | `c1e6f9759385` | 2026-04-20T04:12:07Z |
| `size2000_balanced_trial0__biencoder_only__raw` | `70e58b63076c` | 2026-04-20T20:57:38Z |
| `size2000_balanced_trial0__biencoder_only__typed` | `5fad3c5f009e` | 2026-04-20T21:24:18Z |
| `size2000_balanced_trial0__biencoder_only__uniform` | `dc5abb6406e0` | 2026-04-20T21:51:10Z |
| `size2000_imbalanced_trial0__biencoder_only__raw` | `b2bfd5eda5d8` | 2026-04-20T04:37:19Z |
| `size2000_imbalanced_trial0__biencoder_only__typed` | `b054f418dd52` | 2026-04-20T05:02:19Z |
| `size2000_imbalanced_trial0__biencoder_only__uniform` | `543f4cfc77c7` | 2026-04-20T05:27:50Z |
| `size2000_imbalanced_trial0__biencoder_plus_ce__raw` | `e06ffc80a9c1` | 2026-04-20T05:53:28Z |
| `size2000_imbalanced_trial0__biencoder_plus_ce__typed` | `fd64acc34395` | 2026-04-20T06:17:55Z |
| `size2000_imbalanced_trial0__biencoder_plus_ce__uniform` | `21fd8dd22295` | 2026-04-20T06:42:23Z |
| `size4000_imbalanced_trial0__biencoder_only__raw` | `a53aabd9f3cc` | 2026-04-20T07:07:32Z |
| `size4000_imbalanced_trial0__biencoder_only__typed` | `3a0fb90b6f0c` | 2026-04-20T07:33:32Z |
| `size4000_imbalanced_trial0__biencoder_only__uniform` | `6ac7af550531` | 2026-04-20T07:58:55Z |
| `size4000_imbalanced_trial0__biencoder_plus_ce__raw` | `1ab9623bce72` | 2026-04-20T08:23:25Z |
| `size4000_imbalanced_trial0__biencoder_plus_ce__typed` | `5444b2e2a60e` | 2026-04-20T08:48:29Z |
| `size4000_imbalanced_trial0__biencoder_plus_ce__uniform` | `18451b8cfe7b` | 2026-04-20T09:12:47Z |
| `size8000_imbalanced_trial0__biencoder_only__raw` | `0c5bf23b087a` | 2026-04-20T09:38:25Z |
| `size8000_imbalanced_trial0__biencoder_only__typed` | `2a2dd118ee32` | 2026-04-20T10:04:36Z |
| `size8000_imbalanced_trial0__biencoder_only__uniform` | `18fcac5e0213` | 2026-04-20T10:29:56Z |
| `size8000_imbalanced_trial0__biencoder_plus_ce__raw` | `fb5399e1ddc2` | 2026-04-20T10:54:19Z |
| `size8000_imbalanced_trial0__biencoder_plus_ce__typed` | `a0df3f17bbd7` | 2026-04-20T11:19:06Z |
| `size8000_imbalanced_trial0__biencoder_plus_ce__uniform` | `c5ccdb504f23` | 2026-04-20T11:44:19Z |

The 33 `eval_metrics.json` and 33 `eval_config.json` files beside them are used **only** for
provenance and cross-checking; their hashes are in the JSON companion. Also hashed there: the six
`pool.jsonl` files and the six pool `stats.json` files behind §6; the eval
sample `runs/pool_sweep/dev_sample/dev_sample_250per_hop_seed42.jsonl`
(sha256 `3ec876260e74c274…`, mtime 2026-04-20T01:37:46Z) and its stats file; and both copies of the
aggregate CSV — `runs/pool_sweep/summary/all_runs.csv` and
`handoff/results_analysis/pool_sweep_summary/all_runs.csv`, which are **byte-identical** (sha256
`89c5e923ddbb…`, verified) and are used **only as a cross-check**. **No statistic in this note is
computed from an aggregate CSV.**

Each cell's `eval_config.json` records `seed: 42`, `limit: null`, gold
`MusiQue/Data/dev_data/musique_ans_v1.0_dev_clean.jsonl`, and a predictions path
`runs/pool_sweep/decomposer/<cell>/results.json`.

## 3. Grid map, pairability, and item-set verification

**The grid is incomplete, and the shape matters for what is comparable.** The full
`size{1000,2000,4000,8000} × {balanced,imbalanced} × {biencoder_only,biencoder_plus_ce} ×
{raw,typed,uniform}` cross is 48 cells; **33 exist** (ran; observed):

- **imbalanced arm: complete** — all 4 sizes × both retrieval variants × 3 mask modes = 24 cells.
- **balanced arm: 9 cells** — size 1000 with both retrieval variants (6), size 2000 with
  `biencoder_only` only (3). There is **no balanced cell at 4000 or 8000** and no
  `size2000_balanced__biencoder_plus_ce`. The 15 absent cells are listed in the JSON (`grid.cells_absent`).
- **`trial0` only** — the sweep has no repeated trial, so **pool-draw variance is unmeasured** and no
  statistic below separates a size effect from a single-draw effect.

**Pairability (ran; observed).** All 33 cells hold **750 rows each**, and the **normalized question
sequence is identical across all 33 cells, order included** (True), as are `gold_steps` (True) and
`gold_hop_count` (True). Gold hop distribution is **250 / 250 / 250** for hops 2 / 3 / 4. The
sequence is **identical, order included, to the committed eval sample**
`runs/pool_sweep/dev_sample/dev_sample_250per_hop_seed42.jsonl` (750 rows) — so the cells do share the
eval sample the brief names, and the comparison is a genuine paired comparison on the full 750.

**Alignment (ADR 0020 condition 3).** One normalized question appears **twice** (rows 370 and 376,
with identical gold) — a genuinely duplicated eval item, which is why keying by question text is not
usable and why v2's `--compare` would abort on these files. Alignment is therefore **positional (row
index), in v1 file order**, and for **every one of the 750 rows of every one of the 36 pairs** the
row's normalized question **and** `gold_steps` were asserted equal between the two cells (all
assertions passed). This is a documented deviation from v2's id-based alignment, forced by v1's file
format; it is safe here only because the 33 sequences are literally identical.

**Aggregate cross-check (ran; observed).** Re-aggregating the per-item rows of all 33 cells reproduced
each cell's committed `eval_metrics.json` to a maximum absolute delta of **1.2e-15** across EM, step
F1, ordered accuracy, ROUGE-L F1, hop EM, `reference_validity_micro`, `step_count_abs_error_mae` and
composite. The same recomputation agrees with `all_runs.csv` to the same 1.2e-15, with **0
disagreements** over 33 cells. v1's per-size mean-composite figures quoted in `docs/prior-work.md` §4
(1000: 0.232, 2000: 0.271, 4000: 0.311, 8000: 0.230) recompute as **0.2322 / 0.2707 / 0.3105 /
0.2295** — they are correct as stated. §4.4 is about what they mean, not whether they are right.

**Read CI bounds to three significant figures, not to the last digit.** The bootstrap resamples row
indices, so endpoints depend on the order the aligned matrices were built in. Re-running one battery
in reversed row order (recorded in the JSON under `row_order_sensitivity`) gave point estimates
identical to ≤ **1.1e-16**, **every significance verdict identical (7/7 statistics)**, and CI endpoints
moving in the third or fourth decimal (e.g. exact match [−0.0173, +0.0133] in file order vs
[−0.0173, +0.0147] reversed).

## 4. Primary — pool-size pairs, n = 750 each

**27 matched pool-size pairs** exist, holding balance, retrieval variant and mask mode fixed and
varying only size: 24 in the imbalanced arm (6 fixed combinations × 4 contrasts) and 3 in the balanced
arm (1000 vs 2000, `biencoder_only`, 3 mask modes — the only cross-size contrast the balanced arm
supports). Differences are **(larger pool) − (smaller pool)**, so **positive favours the larger pool**.
`*` = 95% bootstrap CI excludes 0. `retr` is the retrieval variant: **BE** = `biencoder_only`,
**CE** = `biencoder_plus_ce`. `compNR` is the §1 diagnostic, not a house metric.

| contrast | balance | retr | mask | EM | step F1 | ord acc | ROUGE-L | hop EM | composite | compNR |
|---|---|---|---|---|---|---|---|---|---|---|
| 2000 vs 1000 | balanced | BE | raw | −0.0013 | +0.0013 | −0.0013 | −0.0039 | −0.0107 | −0.0018 | −0.0023 |
| 2000 vs 1000 | balanced | BE | typed | −0.0107 | −0.0041 | −0.0068 | −0.0050 | −0.0147 | +0.0714 | −0.0059 |
| 2000 vs 1000 | balanced | BE | uniform | −0.0107 | −0.0031 | −0.0061 | +0.0035 | −0.0200 | +0.1954 | −0.0057 |
| 2000 vs 1000 | imbal | BE | raw | +0.0160 | **+0.0181\*** | **+0.0164\*** | −0.0020 | +0.0107 | **+0.2112\*** | +0.0140 |
| 2000 vs 1000 | imbal | BE | typed | **+0.0240\*** | **+0.0241\*** | **+0.0247\*** | **+0.0134\*** | −0.0053 | **+0.0411\*** | **+0.0202\*** |
| 2000 vs 1000 | imbal | BE | uniform | +0.0053 | +0.0097 | +0.0084 | +0.0042 | +0.0280 | −0.0695 | +0.0103 |
| 2000 vs 1000 | imbal | CE | raw | +0.0027 | +0.0036 | −0.0003 | +0.0099 | −0.0053 | −0.0280 | +0.0007 |
| 2000 vs 1000 | imbal | CE | typed | **+0.0147\*** | +0.0110 | +0.0110 | +0.0096 | +0.0107 | +0.0072 | +0.0090 |
| 2000 vs 1000 | imbal | CE | uniform | +0.0093 | +0.0048 | +0.0018 | +0.0021 | −0.0307 | +0.1333 | −0.0001 |
| 4000 vs 2000 | imbal | BE | raw | −0.0013 | −0.0018 | +0.0004 | +0.0051 | −0.0107 | +0.0021 | +0.0027 |
| 4000 vs 2000 | imbal | BE | typed | −0.0040 | −0.0046 | −0.0069 | −0.0023 | +0.0027 | +0.1695 | −0.0069 |
| 4000 vs 2000 | imbal | BE | uniform | +0.0093 | −0.0001 | +0.0010 | −0.0009 | −0.0240 | +0.1982 | −0.0023 |
| 4000 vs 2000 | imbal | CE | raw | +0.0027 | +0.0095 | +0.0110 | +0.0060 | **+0.0467\*** | +0.0093 | +0.0116 |
| 4000 vs 2000 | imbal | CE | typed | +0.0067 | +0.0053 | +0.0094 | −0.0012 | −0.0320 | +0.1030 | +0.0038 |
| 4000 vs 2000 | imbal | CE | uniform | −0.0013 | +0.0067 | +0.0067 | −0.0003 | −0.0053 | −0.1970 | +0.0038 |
| 8000 vs 4000 | imbal | BE | raw | −0.0013 | +0.0089 | +0.0052 | +0.0067 | +0.0267 | −0.1970 | +0.0038 |
| 8000 vs 4000 | imbal | BE | typed | +0.0013 | +0.0082 | +0.0071 | +0.0016 | +0.0213 | −0.1936 | +0.0079 |
| 8000 vs 4000 | imbal | BE | uniform | +0.0013 | +0.0092 | +0.0074 | −0.0004 | +0.0413 | −0.1735 | +0.0123 |
| 8000 vs 4000 | imbal | CE | raw | +0.0080 | +0.0022 | +0.0016 | −0.0039 | −0.0187 | −0.0008 | −0.0010 |
| 8000 vs 4000 | imbal | CE | typed | −0.0027 | +0.0132 | +0.0104 | −0.0020 | +0.0253 | +0.0289 | +0.0111 |
| 8000 vs 4000 | imbal | CE | uniform | +0.0013 | −0.0082 | −0.0091 | −0.0026 | −0.0227 | +0.0502 | −0.0087 |
| 8000 vs 1000 | imbal | BE | raw | +0.0133 | **+0.0251\*** | **+0.0221\*** | +0.0098 | +0.0267 | +0.0164 | **+0.0205\*** |
| 8000 vs 1000 | imbal | BE | typed | **+0.0213\*** | **+0.0277\*** | **+0.0249\*** | **+0.0127\*** | +0.0187 | +0.0170 | **+0.0212\*** |
| 8000 vs 1000 | imbal | BE | uniform | **+0.0160\*** | **+0.0187\*** | **+0.0168\*** | +0.0028 | **+0.0453\*** | −0.0448 | **+0.0203\*** |
| 8000 vs 1000 | imbal | CE | raw | +0.0133 | +0.0152 | +0.0123 | **+0.0120\*** | +0.0227 | −0.0195 | +0.0113 |
| 8000 vs 1000 | imbal | CE | typed | **+0.0187\*** | **+0.0296\*** | **+0.0308\*** | +0.0064 | +0.0040 | +0.1390 | **+0.0238\*** |
| 8000 vs 1000 | imbal | CE | uniform | +0.0093 | +0.0033 | −0.0006 | −0.0008 | **−0.0587\*** | −0.0136 | −0.0050 |

### 4.1 The pattern: the 1000-vs-8000 extremes separate; the adjacent steps above 1000 do not

Counting how many of the pairs in each contrast family show **any** CI-significant difference on the
five real per-item metrics (EM, step F1, ordered accuracy, ROUGE-L, hop EM):

| contrast | pairs | pairs with ≥1 CI-significant metric | step F1 sign (+/−) | step F1 CI-sig | step F1 Holm survivors |
|---|---|---|---|---|---|
| 1000 vs 2000 | 9 | **3** (all three in the imbalanced arm) | 7 / 2 | 2 | 1 |
| 2000 vs 4000 | 6 | **1** (hop EM only, CE + raw) | 3 / 3 | 0 | 0 |
| 4000 vs 8000 | 6 | **0** | 5 / 1 | 0 | 0 |
| 1000 vs 8000 | 6 | **6** | **6 / 0** | 4 | **4** |

The third column counts **CI-significance only**, as its heading says. Read with the t-test the one
change is 4000 vs 8000, where 1 of the 30 metric-cells is t-significant uncorrected (hop EM,
p = 0.0476) on a CI whose lower bound is exactly 0.0 — §4.2 and §4.4. It survives no correction, and
no 4000-vs-8000 cell is McNemar-significant.

Across all 27 size pairs: step F1 favours the larger pool in **21/27** (6 CI-significant, 6
t-significant), ordered accuracy in **20/27** (6/6), EM in **19/27** (5 CI-sig, 6 t-sig, 3
McNemar-sig), ROUGE-L in **15/27** (3/3), hop EM in **14/27** (3 CI-sig, 4 t-sig, 3 McNemar-sig).

Cell-mean absolute values line up with that reading. Averaged over the **6 imbalanced cells at each
size** (a cell-mean, not a per-item statistic, and carrying no CI):

| size | mean step F1 | mean EM | mean hop EM | mean compNR |
|---|---|---|---|---|
| 1000 | 0.1698 | 0.0316 | 0.4836 | 0.2312 |
| 2000 | 0.1816 | 0.0436 | 0.4849 | 0.2402 |
| 4000 | 0.1841 | 0.0456 | 0.4811 | 0.2423 |
| 8000 | 0.1897 | 0.0469 | 0.4933 | 0.2465 |

Step F1, EM and the reference-free diagnostic are **weakly monotone increasing in pool size**; hop EM
is **not** monotone (4000 < 1000 < 2000 < 8000). The increments above 1000 are 0.002–0.006 step F1 per
doubling — below what n = 750 can resolve (§4.5).

The two largest single effects in the whole table are both **`typed` at 8000 vs 1000**: step F1
**+0.0277** with `biencoder_only` (CI [+0.0122, +0.0431], t = 3.496, p = 0.0005) and **+0.0296** with
`biencoder_plus_ce` (CI [+0.0141, +0.0457], t = 3.683, p = 0.0002), with ordered accuracy +0.0249 and
+0.0308 respectively. The largest EM effect is **+0.0240** at 2000 vs 1000, imbalanced / BE / typed
(0.0507 vs 0.0267; CI [+0.0107, +0.0373]; t = 3.557, p = 0.0004; **McNemar 22/4 discordant,
p = 0.0005**).

### 4.2 The three tests agree almost everywhere

- **Overall rows, CI vs paired t:** 180 comparisons (36 pairs × 5 metrics), **3 disagreements**, all
  borderline — `size2000_imbal_BE_raw` vs `size1000` EM (CI [0.0000, +0.0320], t p = 0.0454);
  `size8000_imbal_BE_uniform` vs `size4000` hop EM (CI [0.0000, +0.0827], t p = 0.0476);
  `size2000_balanced_BE_uniform` vs `size2000_imbalanced` step F1 (CI [−0.0312, −0.0000], t p = 0.0566).
- **Overall rows, CI vs exact McNemar:** 72 comparisons, **3 disagreements**, all EM with McNemar
  p just above α (0.0501, 0.0614, 0.0522 on 32 / 29 / 27 discordant pairs).
- **Per-hop rows, CI vs t:** 540 comparisons, **4 disagreements**, each with a t-test p between
  **0.0453 and 0.0508**; two of the four have a CI endpoint of **exactly 0.0** and the other two have
  an endpoint **7.6e-05** and **2.6e-04** from 0.

**Why these near-misses matter for reading the tables.** `significant` in the bootstrap column is
`ci_low > 0 or ci_high < 0` — a strict inequality, so a CI endpoint that lands on exactly 0.0 reads
as *not* significant. Three of the seven disagreements above are that knife-edge (endpoint exactly
0.0), including the one 4000-vs-8000 cell §4.4 and §7 name explicitly. Where a claim below turns on
this, the test basis is stated rather than the word "significant" alone.

None of these disagreements changes a conclusion in §9; each is a cell that should not be leaned on.
Full lists are in the JSON under `verdict_agreement`.

### 4.3 Under Holm correction, only the 1000-vs-8000 family holds up

Holm-Bonferroni within each **contrast × metric** family (family = every fixed combination on which
that contrast exists — 9 tests for 1000 vs 2000, 6 for the others):

| family | metric | family size | survivors at α = 0.05 | smallest adjusted p |
|---|---|---|---|---|
| 1000 vs 2000 | EM (t) | 9 | 1 (imbal BE typed, 0.0036) | 0.0036 |
| 1000 vs 2000 | EM (McNemar) | 9 | 1 (imbal BE typed, 0.0048) | 0.0048 |
| 1000 vs 2000 | step F1 | 9 | 1 (imbal BE typed, 0.0096) | 0.0096 |
| 1000 vs 2000 | ord acc | 9 | 1 (imbal BE typed, 0.0045) | 0.0045 |
| 1000 vs 2000 | ROUGE-L / hop EM | 9 | 0 | 0.1034 / 1.0000 |
| 2000 vs 4000 | all five metrics | 6 | **0** | 0.1076 (hop EM) |
| 4000 vs 8000 | all five metrics | 6 | **0** | 0.2855 (hop EM) |
| 1000 vs 8000 | step F1 | 6 | **4** (BE raw 0.0143, BE typed 0.0025, BE uniform 0.0443, CE typed 0.0015) | 0.0015 |
| 1000 vs 8000 | ord acc | 6 | **3** (BE raw 0.0319, BE typed 0.0061, CE typed 0.0008) | 0.0008 |
| 1000 vs 8000 | EM (t) | 6 | 1 (BE typed 0.0147) | 0.0147 |
| 1000 vs 8000 | EM (McNemar) | 6 | 1 (BE typed 0.0223) | 0.0223 |
| 1000 vs 8000 | ROUGE-L / hop EM | 6 | 0 | 0.1967 / 0.0508 |

Within-pair Holm across each pair's own five-metric family is also in the JSON
(`holm_per_pair_5metric_family`): the 2000-vs-1000 imbalanced/BE/typed pair keeps **4 of 5** metrics
(smallest adjusted p 0.0020), and both typed 8000-vs-1000 pairs keep **3 of 5** (smallest 0.0025 and
0.0006). So the headline effects here are, unlike v1's masking effects, **not fragile under
correction**.

### 4.4 The composite is not usable for reading the size trend — and v1's "8000 drops vs 4000" is the reference term

The composite's `reference_validity_micro` term is decided by a handful of `[#k]` references per cell.
Measured over the 33 cells (750 predictions each): **median 4 references per cell** (min 0, max 29),
**median 2 items** emitting any reference (min 0, max 5), and **3 cells emit none at all** — which
score 1.0 on that term by the micro convention when the denominator is empty.

Consequences, all measured:

- Over the 27 size pairs, the **median share of |Δcomposite| produced by the 0.2-weighted reference
  term is 100.1%**; the term exceeds 50% of the difference in **20/27** pairs and exceeds 90% in
  **16/27**. (The four weighted term contributions sum to Δcomposite exactly — max residual
  5.6e-17.)
- **25 of 27 composite CIs straddle 0**, with widths from 0.0206 to 0.4163 (median 0.2162) on a
  0-to-1 scale. The reference-free diagnostic's CIs are 0.0258–0.0323 wide (median 0.0286) — an order
  of magnitude tighter, because it is not resampling three-item terms.
- **v1's headline size trend is that term.** `docs/prior-work.md` §4 reports mean composite by size
  with "size 4000 highest at 0.311 … 8000: 0.230" and reads it as *larger is not always better*.
  Decomposing that 4000→8000 cell-mean drop of **−0.0810** (imbalanced-only, 6 cells each side):
  reference term **−0.0844 = 104.2% of it**, step F1 **+0.0022**, ordered accuracy **+0.0011**,
  step-count term **+0.0000**. Mean `reference_validity_micro` falls 0.5833 → 0.1615 between those
  size groups — on **13 total references at 4000 and 37 at 8000**. Per-item, across the six
  4000-vs-8000 pairs, **0 of 30 metric-cells are CI-significant, 0 of 12 are McNemar-significant, and
  nothing survives Holm** (smallest Holm-adjusted p **0.2855**); **1 of 30 is t-significant
  uncorrected** — `biencoder_only` / `uniform` hop EM, **+0.0413**, t = 1.984, **p = 0.0476**, whose CI
  **[0.0000, +0.0827]** misses only on the strict knife-edge of §4.2 and whose McNemar is 138/107,
  p = 0.0551. Step F1 favours 8000 in 5 of 6.
- Reading the same size axis on the reference-free diagnostic **reverses the direction**: cell means
  rise monotonically 0.2312 → 0.2402 → 0.2423 → 0.2465 from 1000 to 8000 (§4.1).

### 4.5 Per-hop, and where the size effect lives

Per gold hop (n = 250 each), across all 27 size pairs × 3 strata: CI-significant cells are
**EM 10/81, step F1 10/81, ordered accuracy 9/81, ROUGE-L 8/81, hop EM 4/81, composite 10/81,
compNR 12/81**. Excluding the composite column, that is 53 significant cells, of which **7 favour the
smaller pool** and 46 the larger. The 1000-vs-8000 family holds **23 of the 53**, concentrated in
**hop 2** (e.g. imbalanced BE raw: step F1 +0.0604 CI [+0.0235, +0.0986]; imbalanced BE typed: EM
+0.0520 CI [+0.0160, +0.0920], step F1 +0.0511) and, for **typed only**, also in **hop 4** (BE typed:
step F1 +0.0369, hop EM +0.0920 CI [+0.0160, +0.1720]; CE typed: step F1 +0.0339, ordered accuracy
+0.0329). Within this family the only significant hop-4 effect **against** the larger pool is
`8000 vs 1000, imbalanced CE uniform` hop EM **−0.0880** (CI [−0.1680, −0.0080]); across all 27 pairs
there are three more negative significant hop-4 cells, all in other families and all listed in the
JSON.

Suggestive, not established: the imbalanced pool's 4-hop exemplar count grows 81 → 126 → 245 → 495
across the four sizes (pool `stats.json`, §6), and the only sizeable hop-4 gains appear with `typed`
masking. That is consistent with "more deep exemplars help deep questions when the retrieval text is
structural", but three significant cells out of 81 do not establish it.

### 4.6 Power at n = 750

Minimum detectable difference (paired, α = 0.05 two-sided, 80% power, from the observed per-item
difference SDs), median over the 27 size pairs. The constant used is **2.8016**, a rounding of
z₀.₉₇₅ + z₀.₈ = 1.95996 + 0.84162 = 2.80158 (also disclosed in the companion under `power.formula`);
at these SDs the rounding moves an MDE by less than 1e-5.

| metric | median sd(diff) | median MDE at n = 750 |
|---|---|---|
| exact match | 0.1863 | 0.0191 |
| step F1 | 0.2044 | 0.0209 |
| ordered step accuracy | 0.2002 | 0.0205 |
| ROUGE-L F1 | 0.1455 | 0.0149 |
| hop-count EM | 0.5678 | 0.0581 |

n = 750 buys about **2 points** of step F1 / ordered accuracy and about **6 points** of hop EM. The
adjacent-step differences above 1000 are 0.000–0.013, i.e. **below the detection threshold**: the n
required for 80% power at each observed 2000-vs-4000 or 4000-vs-8000 step-level difference ranges from
**1,890** (8000 vs 4000, CE typed, step F1) to over **20 million** (4000 vs 2000, BE uniform, step F1
+0.0001), with most between 3,000 and 15,000 (all 24 values in the JSON under
`power.adjacent_step_pairs_n_needed`). **"2000, 4000 and 8000 are indistinguishable here" is a
statement about power at n = 750, not a demonstration that they are equal.**

## 5. Secondary — balanced vs imbalanced pools, where both exist

**9 matched pairs**: size 1000 × both retrieval variants × 3 mask modes (6), size 2000 ×
`biencoder_only` × 3 mask modes (3). Differences are **balanced − imbalanced**.

| pair | EM | step F1 | ord acc | ROUGE-L | hop EM | composite | compNR |
|---|---|---|---|---|---|---|---|
| 1000, BE, raw | +0.0080 | +0.0014 | +0.0057 | −0.0016 | −0.0147 | −0.0012 | −0.0015 |
| 1000, BE, typed | **+0.0147\*** | +0.0026 | +0.0026 | +0.0047 | −0.0360 | +0.0647 | −0.0024 |
| 1000, BE, uniform | +0.0120 | −0.0026 | −0.0041 | −0.0062 | −0.0093 | −0.0820 | −0.0052 |
| 1000, CE, raw | −0.0013 | −0.0022 | −0.0071 | +0.0014 | −0.0333 | +0.1449 | −0.0105 |
| 1000, CE, typed | +0.0053 | +0.0010 | −0.0014 | −0.0080 | **−0.0533\*** | +0.1453 | −0.0058 |
| 1000, CE, uniform | +0.0027 | −0.0033 | −0.0097 | −0.0050 | **−0.1107\*** | **−0.0509\*** | −0.0148 |
| 2000, BE, raw | −0.0093 | −0.0154 | −0.0120 | −0.0035 | −0.0360 | **−0.2142\*** | **−0.0178\*** |
| 2000, BE, typed | **−0.0200\*** | **−0.0257\*** | **−0.0289\*** | **−0.0136\*** | **−0.0453\*** | +0.0950 | **−0.0285\*** |
| 2000, BE, uniform | −0.0040 | **−0.0154\*** | **−0.0186\*** | −0.0070 | **−0.0573\*** | +0.1830 | **−0.0213\*** |

**Hop-count EM favours the imbalanced pool in 9/9 pairs** — 4 CI-significant, 4 t-significant, 4
McNemar-significant; the largest is `1000 CE uniform` at **−0.1107** (0.4227 vs 0.5333; CI [−0.1547,
−0.0667]; t = −4.941, p ≈ 0.0000; McNemar 104/187, p ≈ 0.0000), which survives Holm within its
6-test family (adjusted p ≈ 0.0000 by both tests). The reference-free diagnostic also favours the
imbalanced pool in **9/9** (3 CI-significant). At size 2000 the imbalanced pool wins on the step-level
metrics as well: the `typed` pair is significant on **all five** metrics uncorrected, and **survives
Holm on all five within its own five-metric family** (adjusted p 0.0019–0.0400). Under the
across-mask family instead (3 tests per metric, comparing the three mask modes at size 2000) it keeps
**3 of 5** — EM 0.0334, step F1 0.0050, ordered accuracy 0.0011 — while ROUGE-L (0.0600) and hop EM
(0.0670) fall out and hop EM's surviving cell in that family is `uniform` instead (0.0238; McNemar
0.0284). Which family is the right one is the open question ADR 0009 already flags.

This is the direction that needs stating carefully. The eval set is **hop-balanced** (250/250/250),
while the imbalanced pool follows the MuSiQue train distribution (about 71% 2-hop) and so holds far
fewer deep exemplars — 81 4-hop items at size 1000 against 333 in the balanced pool. **The pool with
fewer deep exemplars nonetheless predicts hop count better on a hop-balanced eval set, in 9 of 9
matched pairs.** These runs measure that; they do not explain it, and the balanced arm is only 9 cells
at two sizes with non-nested pools (§6), so it is a **suggestive pattern, not an established
mechanism**.

## 6. Confounds in the pool construction itself

From the six pool `stats.json` files (hashed in the JSON companion), all `poolseed42`, all drawn from
the same masked MuSiQue **train** file, with input buckets 2hop 14376 / 3hop 4387 / 4hop 1175:

- **"balanced"** = equal 2/3/4-hop exemplars (e.g. 334/333/333 at size 1000); **"imbalanced"** = the
  train distribution (709/210/81 at 1000; 5788/1717/495 at 8000).
- **Nesting, by pool item id (ran; observed):** imbalanced **1000 ⊂ 2000 ⊂ 4000** exactly
  (1000/1000 and 2000/2000 overlap), but **4000 ⊄ 8000** — only **3484 of 4000** items survive into
  the 8000 pool, and only **982 of 1000** of the size-1000 items. The balanced pools are **not**
  nested at all (682/1000 overlap between 1000 and 2000). Balanced and imbalanced pools at the same
  size barely overlap (**50/1000** at size 1000, **198/2000** at size 2000).
- Therefore: the **1000→2000 and 2000→4000 imbalanced contrasts add exemplars and keep the old ones**
  — a clean size manipulation. The **4000→8000 and 1000→8000 contrasts also change which exemplars
  are available**, and the **balanced 1000→2000 contrast is a different draw** as well as a larger one.
  The balanced arm's three cross-size pairs (all non-significant, all slightly negative on the
  step-level metrics) should not be read as evidence that a larger balanced pool does not help.
- The 27 size pairs and 9 balance pairs share the **identical 750 eval items** and largely nested
  pools, so they are **not independent**. Sign counts in §4.1 and §5 are descriptive; **no across-pair
  test was run and none is claimed.**

## 7. What this confirms and what it does not support in `docs/prior-work.md`

- **Confirmed as arithmetic:** the per-size mean composites (0.232 / 0.271 / 0.311 / 0.230) and the
  best-cell table recompute exactly from the per-item files (§3).
- **Not supported as an interpretation:** FINDINGS' "larger pool is not always better (8000 drops vs
  4000)". That drop is **104.2% the reference-validity term** on 13 vs 37 references (§4.4); across
  the 6 matched 4000-vs-8000 pairs no metric is CI- or McNemar-significant and none survives Holm
  (smallest adjusted p 0.2855, with one cell t-significant uncorrected at p = 0.0476 — §4.4); and on
  the reference-free diagnostic the size trend is monotone **upward** through 8000. The "8000 drops"
  reading is an artifact of the composite, not a pool-size finding.
- **Confirmed, and now with a significance verdict:** "exact match stays ~3–5% across the board — pool
  size alone does not solve gold-plan matching." EM ranges 0.0267–0.0533 across the 33 cells, and the
  largest matched-pair EM gain anywhere is **+0.0240** (2000 vs 1000, imbalanced BE typed). Even the
  1000→8000 eightfold increase moves EM by at most +0.0213.
- **Newly measured (the §6 gap of the masking/retrieval note):** the pool-size axis. **1000 is
  measurably worse than 8000** on step-level metrics (6/6 pairs positive on step F1, 4 CI-significant,
  4 Holm-surviving), and worse than 2000 in the imbalanced/typed cell (Holm-surviving on 4 of 5
  metrics). **Above 1000, no adjacent doubling survives correction on anything** — 2000 vs 4000:
  1 of 30 metric-cells CI-significant, 0 Holm survivors; 4000 vs 8000: **0 of 30 CI-significant,
  1 of 30 t-significant uncorrected** (p = 0.0476), 0 Holm survivors.
- **Also newly measured:** balanced vs imbalanced (§5), which `docs/prior-work.md` does not discuss.

## 8. Unmeasured / out of scope

Stated so it is not read into the note: any **v2** run (nothing here was executed on v2 code or v2
models); **within-cell seed variance** — only `trial0` exists, so a "size effect" here is one pool draw
per size; pool sizes outside {1000, 2000, 4000, 8000}; **balanced pools at 4000 and 8000** and
`size2000_balanced__biencoder_plus_ce` (cells absent); any guided (hop-forced) condition; MetaQA;
end-to-end answer accuracy. The v1 predictions' own provenance (which decoder, which prompt revision)
is not established by this note. No model was run and no v1 file was modified.

## 9. Ranked takeaways for issue #6 item 4 — options, not decisions

Ranked by how much each would change how the pool-size dimension is written up or re-run. These are
**conditionals**; ADR 0006 already fixed pool size at 2000, and which baselines the thesis leans on
is Jahid's call with his supervisor.

1. **If the thesis states a pool-size conclusion from v1, the defensible one is "1000 is too small;
   2000 and above are indistinguishable at n = 750" — not "4000 is best".** Measured: 6/6 pairs
   favour 8000 over 1000 on step F1 with 4 CI-significant and 4 Holm-surviving; **0 of 6 pairs show a
   CI- or McNemar-significant 4000-vs-8000 difference and none survives Holm** (one cell is
   t-significant uncorrected, p = 0.0476 — §4.4); 1 of 6 shows a CI-significant 2000-vs-4000
   difference, which does not survive Holm either (hop EM, CE + raw). This ranks first because "4000 was best, 8000 dropped" is the sentence most likely to be
   carried into the thesis from `docs/prior-work.md` §4, and §4.4 shows it is the reference term.
2. **If any composite number is quoted on the size axis, it needs the §4.4 caveat attached or it
   should not be quoted.** The reference term produces a median **100.1%** of |Δcomposite| across the
   27 pairs, 25 of 27 composite CIs straddle 0 (widths up to 0.4163), and the 4000→8000 aggregate drop
   decomposes to **104.2%** that term on 13 vs 37 references. This is direct input to issue #29 and to
   #6 item 5, and it is the same defect the masking note found on a different axis — so it is a
   property of the metric, not of one comparison.
3. **ADR 0006's choice of 2000 is not contradicted by v1's own sweep, and the evidence for it is
   "on the plateau", not "optimal".** No metric in any matched pair shows 4000 or 8000 significantly
   beating 2000; the cell-mean step F1 gain from 2000 to 8000 is +0.008 against a median MDE of
   0.021. If the thesis wants to *claim* 2000 is sufficient rather than merely chosen, that claim
   needs either a v2 re-run or the power statement in §4.6 stated alongside it — the options the
   evidence leaves open, not a recommendation between them.
4. **The balanced-vs-imbalanced result is the most surprising thing in this note and it is currently
   undecided in the repo.** Hop-count EM favours the *imbalanced* (train-distribution) pool in **9/9**
   matched pairs — 4 significant by all three tests, one surviving Holm at p ≈ 0.0000 with 104/187
   discordant pairs — on a hop-balanced eval set, and at size 2000 the imbalanced pool wins on all
   five metrics uncorrected (`typed`; 3 of 5 survive Holm across the mask family, all 5 across the
   metric family — §5). If v2's few-shot pool construction is going to be
   described in the thesis, this is a decision point that has evidence attached; whether to act on it
   is a supervisor question, and the balanced arm's 9 cells at 2 sizes with non-nested pools (§6) are
   thin ground for acting.
5. **If a v2 pool-size re-run is ever scheduled, n = 750 will not settle the 2000-vs-4000-vs-8000
   question.** The observed adjacent-step differences need roughly **2,000–15,000** items for 80%
   power (§4.6). The cheap alternative the data supports is to re-run only the **extreme** contrast
   (smallest vs largest), which is the one v1 can already detect.
6. **Two follow-ups the same battery would answer, both unmeasured today:** (a) whether the size
   effect is a *pool-draw* effect — no repeated trial exists in the sweep, so a second draw at one
   size would separate the two, and until then every §4 number is one draw per size; (b) the
   `--compare` shim that accepts v1-format per-item files (carried over from the masking note's
   §7.6(b)), which would make §4 and §5 re-derivable from committed code instead of from a
   session-local harness. As of `main` at `50daea7` no such shim is committed; uncommitted
   work-in-progress toward one was present in the shared working tree while this note was written,
   and it is another lane's to land, so this note claims nothing about it beyond that.
