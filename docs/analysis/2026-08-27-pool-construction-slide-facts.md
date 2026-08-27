# Pool construction — the slide-ready facts

**Date:** 2026-08-27 · **Author:** lead session (compilation) · **Runs:** none.

This note **measures nothing new**. It gathers numbers that already exist in committed
artifacts into one page, because they are currently spread across a 39 KB analysis note, two
`experiments/<id>/metrics.json` files and three ADRs, and the deck presents one of them
wrongly. Every figure below cites the artifact it came from. Written at Jahid's request so the
deck can be corrected from a single source.

Related: [ADR 0006](../adr/0006-drop-the-jury-fix-dataset-roles-and-the-few-shot-method.md)
(size fixed at 2000), [ADR 0021](../adr/0021-clustering-pool-construction-strategy.md)
(the clustering design), [ADR 0028](../adr/0028-jahid-2026-08-23-delegations-pool-choice-router-call-composite-authorized.md)
(the pool choice), [ADR 0007](../adr/0007-musique-evaluation-set-reuses-v1-600-questions-200-per-hop.md)
(the pinned 600), issue [#14](https://github.com/AhmadiJahid/Thesis---QAv2/issues/14).

---

## 1. Pool size — the "4000 is best" slide is not supportable, and there was no v2 rerun

**There has been no v2 pool-size sweep.** v2 has only ever run pool size **2000** (exp-006
precompute, exp-010 evaluation). The 1000-versus-the-rest result is a **re-analysis of v1's
33-cell sweep**, performed on 2026-08-20 under
[ADR 0020](../adr/0020-prior-work-re-analysis-convention.md), and recorded in
[`2026-08-20-v1-pool-size-significance.md`](2026-08-20-v1-pool-size-significance.md) §7. It is
**prior work, not a v2 measurement** — v1's `runs/` is untracked and carries no commit SHA, so
it does not satisfy Gate 2 and must be labelled as v1 on any slide that shows it.

**What the re-analysis found** (27 matched pool-size pairs, n = 750 per pair, holding balance /
retrieval variant / mask mode fixed and varying only size; paired bootstrap 95 % CI + McNemar +
paired t, Holm-corrected within each contrast × metric family):

| contrast | pairs | pairs with ≥1 CI-significant metric | step-F1 sign (+/−) | Holm survivors (step F1) |
|---|---|---|---|---|
| 1000 vs 2000 | 9 | 3 | 7 / 2 | 1 |
| 2000 vs 4000 | 6 | 1 (hop EM only) | 3 / 3 | 0 |
| 4000 vs 8000 | 6 | **0** | 5 / 1 | 0 |
| 1000 vs 8000 | 6 | **6** | **6 / 0** | **4** |

- **1000 is measurably the worst.** 1000 vs 8000: step F1 favours the larger pool in 6/6 pairs,
  4 CI-significant, **4 surviving Holm**. Largest single effects, both `typed` at 8000 vs 1000:
  step F1 **+0.0277** (CI [+0.0122, +0.0431], t = 3.496, p = 0.0005) and **+0.0296**
  (CI [+0.0141, +0.0457], t = 3.683, p = 0.0002).
- **Above 1000, no adjacent doubling survives correction on anything.** 2000 vs 4000: 1 of 30
  metric-cells CI-significant, 0 Holm survivors. 4000 vs 8000: **0 of 30 CI-significant**, 1 of
  30 t-significant uncorrected (p = 0.0476, on a CI whose lower bound is exactly 0.0000),
  0 Holm survivors, smallest Holm-adjusted p **0.2855**.

**Why the deck's "quality peaks at 4000" is wrong.** That reading comes from v1's mean composite
by size (1000: 0.232, 2000: 0.271, **4000: 0.311**, 8000: 0.230). Those four numbers recompute
exactly — the arithmetic is fine, the *interpretation* is not. Decomposing the 4000 → 8000 drop
of **−0.0810**: the 0.2-weighted `reference_validity_micro` term contributes **−0.0844 = 104.2 %
of it**, while step F1 moves **+0.0022** and ordered accuracy **+0.0011**. That term flips on
**13 total `[#k]` references at size 4000 versus 37 at size 8000**, across cells of 750
predictions each (median 4 references per cell; 3 cells emit none at all and score 1.0 by the
micro convention on an empty denominator). Read on a reference-free diagnostic instead, the size
trend **reverses and rises monotonically**: 0.2312 → 0.2402 → 0.2423 → **0.2465** from 1000 to
8000. This is the same defect as issue [#40](https://github.com/AhmadiJahid/Thesis---QAv2/issues/40),
which froze the composite as legacy on 2026-08-23.

**Defensible sentence for the slide:**

> Re-analysis of the v1 33-cell sweep (27 matched pairs, n = 750, Holm-corrected) shows a pool of
> 1000 is measurably worse than 8000 on step-level metrics (6/6 pairs, 4 surviving correction),
> and **no doubling above 1000 is significant on any metric**. The earlier "quality peaks at
> 4000" reading is an artifact of the legacy composite's reference-validity term — 104 % of that
> drop, on 13 vs 37 references — and reverses on a reference-free metric. *(v1 prior work,
> re-analysed 2026-08-20; not re-run in v2.)*

**What this does and does not license.** It does **not** say 2000 is optimal — it says the
evidence cannot separate 2000 from 4000 or 8000, which is why ADR 0006's supervisor-fixed 2000 is
not contradicted by the data. It does say 1000 would have been a bad choice. Confirmed alongside:
exact match stays **0.0267–0.0533** across all 33 cells and the largest matched-pair EM gain
anywhere is **+0.0240** — pool size alone does not solve gold-plan matching.

---

## 2. How the three pools are built, with the numbers

All three draw from the **same candidate set**: the enriched MuSiQue **train** questions,
**19,938 rows** (`musique_pool_enriched`). All three produce **exactly 2000 rows**, seed **42**,
one trial (`pool_trial_seeds: [42]`). Built in exp-006 on CPU in 823 s total.
Source: `experiments/exp-006/metrics.json`.

### 2.1 Candidate set and resulting pools

| | rows | 2-hop | 3-hop | 4-hop |
|---|---|---|---|---|
| **input candidates** (train) | 19,938 | 14,376 (72.10 %) | 4,387 (22.00 %) | 1,175 (5.89 %) |
| **imbalanced** | 2,000 | 1,461 (73.05 %) | 413 (20.65 %) | 126 (6.30 %) |
| **balanced** | 2,000 | 667 (33.35 %) | 667 (33.35 %) | 666 (33.30 %) |
| **clustered** | 2,000 | 1,396 (69.80 %) | 495 (24.75 %) | 109 (5.45 %) |

Read: **imbalanced** reproduces the corpus skew almost exactly (73.05 % vs 72.10 % 2-hop).
**balanced** forces a flat 1/3 per bucket. **clustered** ignores hop buckets entirely and lands
*near* imbalanced but slightly flatter — it keeps proportionally more 3-hop (11.28 % of available
3-hop rows retained, vs imbalanced's 9.41 %) and fewer 4-hop (9.28 % vs 10.72 %). Clustered is the
**diversity-driven counterpart of imbalanced**, not a third point between imbalanced and balanced.

### 2.2 The three construction rules

- **imbalanced** — seeded uniform random draw over all 19,938 rows. No stratification.
- **balanced** — seeded random draw with an equal quota per coarse hop bucket (667/667/666).
- **clustered** — seeded k-means over bi-encoder embeddings, keep the row nearest each centroid
  (issue #14, ADR 0021). Concretely:
  1. Embed all 19,938 candidate questions with **`intfloat/e5-small-v2`** (33,360,000 parameters,
     the same encoder retrieval uses), `passage:` prefix, L2-normalised. Text is the **raw stored
     `question`** field — no masker runs, so ADR 0003 holds by construction.
  2. Fit `sklearn.cluster.KMeans`, **k = ceil(2000 / examples_per_cluster) = 2000** clusters
     (`examples_per_cluster = 1`), `random_state = 42`, `k-means++`, `n_init = 1`,
     `max_iter = 300`, `tol = 1e-4`, Lloyd, **single-threaded** inside `threadpool_limits`
     (sklearn's Lloyd loop reduces over OpenMP chunks, so thread count is part of the float
     result; 1 thread buys bit-reproducibility under the seed).
  3. Take the row **nearest each centroid** by Euclidean distance, ties broken by
     `(row_id, row_index)` so nothing depends on library iteration order. Selection is
     **rank-major** — every cluster gives its closest row before any gives a second.
  4. Shortfall (empty clusters, duplicate points) is topped up from unselected rows in the same
     global order and **counted**, never silent.
  5. Shuffle with the trial's seeded RNG before writing, so file order does not encode
     distance-to-centroid rank.

### 2.3 Clustering diagnostics as measured (exp-006)

| quantity | value |
|---|---|
| candidates clustered | 19,938 |
| clusters (k) | 2,000 |
| k-means iterations to convergence | 11 |
| k-means inertia | 1803.46 |
| **empty clusters** | **0** |
| cluster size, min / max | 1 / 56 |
| selected from clusters / from top-up | **2000 / 0** |
| clusters represented | 2,000 (all) |
| mean distance to centroid of selected rows | 0.2351 |
| embed time / k-means fit time | 33.4 s / 25.4 s (CPU) |

Every cluster was non-empty and contributed exactly one row — the top-up path never fired, so the
pool is a clean one-representative-per-cluster cover.

### 2.4 Leakage check

Pools are drawn from **train**; the evaluation questions come from **dev**. Measured
independently: **0 / 2000 id overlap between each of the three size-2000 pools and the evaluation
set** (`2026-08-23-s2-feasibility-prompting-vs-finetuned.md` §method). Leak-free by construction
and by measurement.

### 2.5 Two confounds ADR 0021 records against itself

State these on the slide if the clustered arm is shown as a contribution rather than an ablation:

1. **One trial per arm.** `num_trials: 1`. The clustered arm is effectively deterministic
   (reseeding moves only the k-means init over the same candidate set), while imbalanced and
   balanced are each a *single random draw* from a distribution of possible pools. A one-trial
   clustered-minus-random delta therefore mixes strategy effect with draw noise, and the two
   cannot be separated from these numbers.
2. **Same embedding space for construction and retrieval.** The clustered pool is a
   coverage-optimal cover of exactly the space the bi-encoder ranks in; the random arms have no
   such alignment. A clustered advantage cannot be attributed to "better examples" as distinct
   from "examples spread out in the retriever's own metric".

---

## 3. Where the 750 evaluation questions come from

There are **two different MuSiQue evaluation sets** in this project and they must not be mixed on
a slide.

| | the pinned **600** | the pool-sweep **750** |
|---|---|---|
| source split | MuSiQue **dev** | MuSiQue **dev** |
| construction | 200 per hop depth (2/3/4) | 250 per coarse hop bucket (2hop/3hop/4hop) |
| provenance | v1's exact question ids, reused verbatim | drawn by `MusiQue/scripts/sample_dev.py`, `dev_seed 42`, `dev_per_hop 250` |
| record | ADR 0007 | `configs/pool_sweep.json`; `experiments/exp-006/metrics.json` |
| used by | exp-002..005, exp-008, exp-009, exp-011, exp-015 | **exp-006, exp-010 (the pool sweep only)** |

Both are **dev**, not test — MuSiQue's own test split ships no gold decompositions, so dev is the
only labelled data that can score a decomposition. Neither overlaps the training pool (§2.4).

**They are not interchangeable.** Measured id intersection is **405 of 600** — 195 of the pinned
600 were never decomposed in exp-010, so exp-010 cannot be subset down to the pinned set and no
aligned 600-item pairing exists to recover
(`2026-08-23-s2-feasibility-prompting-vs-finetuned.md` §2). **Consequence for the deck: exp-010's
absolute numbers may never be placed on the same axis as exp-004/005/008/009/015.** The pool
sweep's comparison is internally valid — all 18 cells share one config and one dev sample — and
externally incomparable.

The 750 is the *older* of the two in usage terms only in that it belongs to the ported v1 pool
sweep; the 600 (ADR 0007, decided 2026-08-17) is the project's standing evaluation set. The pool
sweep kept its own 750 because it was ported wholesale from v1 with `dev_per_hop: 250` already in
the config, not because 750 was chosen over 600 for a reason anyone recorded.

---

## 4. What is still open

- Issue [#14](https://github.com/AhmadiJahid/Thesis---QAv2/issues/14) remains **open** although
  exp-010 satisfies its "done when". The clustered arm **is** evaluated — any slide saying
  "pools built, not yet evaluated" is stale as of 2026-08-22.
- No v2 pool-size sweep exists and none is planned; §1 is v1 evidence carried forward.
