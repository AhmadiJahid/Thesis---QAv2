# Are the imbalanced and clustered pool arms the same run? — no.

**Date:** 2026-09-14 · **Author:** analyst agent · **Runs:** none (read-only over exp-010's committed
artifacts; no GPU, no generation, `runs/run.lock` not acquired) · **Repo commit:** `13d2677`

Tracking issue: [#52](https://github.com/AhmadiJahid/Thesis---QAv2/issues/52). Bears on
[#14](https://github.com/AhmadiJahid/Thesis---QAv2/issues/14). Logged as **exp-019**.

Written to answer the question Jahid's supervisor raised on 2026-08-27:

> "There's something wrong in the experiment. […] you try clustering with and without clustering
> and the result is matching exactly."

The prompt for this note was issue [#46](https://github.com/AhmadiJahid/Thesis---QAv2/issues/46) §4.1,
which records that in exp-010 the imbalanced and clustered arms **tied at 0.5133** on
`hop_count_exact_match` while balanced sat at 0.4667.

Raw counts: [`2026-09-14-pool-arm-identity-investigation.json`](2026-09-14-pool-arm-identity-investigation.json).
All figures below are recomputed from `runs/pool_sweep/` — untracked heavy output on the compute box,
per the reproducibility rule; paths are given so every number can be re-derived.

---

## 0. The answer

**It is (a): a real null result on the imbalanced-vs-clustered contrast, not a plumbing defect.**
The two arms are physically separate at every stage of the pipeline, and their outputs differ on
**690 of 750** items in the cell the supervisor was shown. The **0.5133 = 0.5133 tie is an aggregate
coincidence in one cell out of six**: 129 items where clustered gets the hop count right and
imbalanced does not, 129 items the other way, cancelling to exactly zero.

There is **no defect to name** — question (4)'s "if (b)" branch does not apply. What there *is*
is a much weaker claim than the deck has been making: the arms were genuinely contrasted and the
contrast produced nothing.

---

## 1. Is it actually identical, or equal-to-4-decimals? — equal-to-4-decimals, in one cell only

Compared the committed per-item outputs
`runs/pool_sweep/eval/size2000_{imbalanced,clustered}_trial0__<variant>__<mode>/eval_per_item.json`,
field `pred_steps`, joined on `item_id`. All 18 cells carry the same 750 ids (verified by set
equality), so the pairing is exact.

| cell (variant · mask) | n | identical `pred_steps` | **differing** |
|---|---|---|---|
| biencoder_only · raw | 750 | 50 | **700** |
| biencoder_only · typed | 750 | 62 | **688** |
| biencoder_only · uniform | 750 | 59 | **691** |
| biencoder_plus_ce · raw | 750 | 54 | **696** |
| **biencoder_plus_ce · typed** *(the 0.5133 cell)* | 750 | **60** | **690** |
| biencoder_plus_ce · uniform | 750 | 54 | **696** |

**6.7 %–8.3 % of items agree; 92 %–93 % differ.** For calibration, the imbalanced-vs-**balanced**
pair — which nobody suspects of being the same run — agrees on **35–54** items in the same cells.
Clustered is no closer to imbalanced than balanced is.

### 1.1 Where the tie comes from

`hop_count_exact_match` is binary per item, so the mean is a rate over 750. McNemar tables,
imbalanced vs clustered:

| cell | imbalanced only correct | clustered only correct | discordant | mean imb | mean clu | exact p |
|---|---|---|---|---|---|---|
| biencoder_only · raw | 123 | 129 | 252 | 0.4667 | 0.4747 | 0.7529 |
| biencoder_only · typed | 125 | 138 | 263 | 0.4707 | 0.4880 | 0.4594 |
| biencoder_only · uniform | 127 | 130 | 257 | 0.4640 | 0.4680 | 0.9007 |
| biencoder_plus_ce · raw | 126 | 141 | 267 | 0.4840 | 0.5040 | 0.3916 |
| **biencoder_plus_ce · typed** | **129** | **129** | **258** | **0.5133** | **0.5133** | 1.0000 |
| biencoder_plus_ce · uniform | 141 | 133 | 274 | 0.4880 | 0.4773 | 0.6725 |

**Only one of six cells ties.** In that cell 258 of 750 items disagree and the disagreement splits
129/129. The other five cells differ by 0.0040 to 0.0200 — small, but not equal, and none of the six
is significant (McNemar exact p 0.39–1.00).

Even the *number of steps produced* differs: predicted step count matches on only **336–369 of 750**
items (45 %–49 %). The arms are not producing the same decompositions and are not even producing the
same shapes.

**So the supervisor's observation is correct as an observation and wrong as a diagnosis.** The number
he was shown really is identical to four decimals; it is identical because 258 differing items
cancelled, not because one run was reported twice.

---

## 2. Where does the identity begin? — it never begins; the arms diverge at stage 1 and stay diverged

Walked the chain forwards. Every stage has its own artifact tree per arm.

### 2.1 Pools — different sets, ~90 % disjoint

`runs/pool_sweep/pools/size2000_<strategy>_trial0_poolseed42/pool.jsonl`, distinct SHA-256 for all
three (recorded in the companion JSON). Question-id overlap:

| pair | size A | size B | intersection | Jaccard |
|---|---|---|---|---|
| imbalanced vs clustered | 2000 | 2000 | **201** | **0.0529** |
| imbalanced vs balanced | 2000 | 2000 | 198 | 0.0521 |
| clustered vs balanced | 2000 | 2000 | 201 | 0.0529 |

Each pool is 2000 unique ids. **Clustered shares 201 of 2000 rows with imbalanced — 10.05 % — which
is indistinguishable from what imbalanced shares with balanced (198).** This is the number that kills
hypothesis (3c) at the pool level: clustered selection has *not* degenerated into a near-copy of the
imbalanced draw.

The k-means diagnostics confirm the selection was non-degenerate
(`pools/size2000_clustered_trial0_poolseed42/metrics.json`): 19,938 candidates → **k = 2000**,
`kmeans_n_iter = 11`, `kmeans_inertia = 1803.46`, **0 empty clusters**, cluster size 1–56,
2000 selected from clusters and **0 from top-up**, `representative_rule: nearest_to_centroid`,
mean distance to centroid 0.2351. The brief asked whether one representative per cluster with
k = 2000 over a pool "of that size" is effectively a near-identity selection — it is not: k = 2000 is
over the **19,938-row candidate set**, a 10:1 reduction, not over the 2000-row pool. Every cluster
contributed exactly one row and the top-up path never fired.

**What clustered and imbalanced *do* share is the hop-mix, not the rows.** 69.80 % / 24.75 % / 5.45 %
2/3/4-hop for clustered against 73.05 % / 20.65 % / 6.30 % for imbalanced, while balanced is forced to
33/33/33 (`2026-08-27-pool-construction-slide-facts.md` §2.1). Hold onto this — §3(c).

### 2.2 Retrieval — different exemplars for every single query

`runs/pool_sweep/biencoder_top5/size2000_<strategy>_trial0/top5_biencoder.jsonl` and
`runs/pool_sweep/rerank/size2000_<strategy>_trial0/top5_ce.jsonl`, distinct SHA-256 for all six files.
Compared the ordered top-5 `pool_id` lists per `query_id`, per mask mode.

| stage · pair | queries | identical top-5 lists | queries sharing ≥1 exemplar | mean shared of 5 |
|---|---|---|---|---|
| bi-encoder · imb vs clu (raw/typed/uniform) | 750 | **0 / 0 / 0** | 307 / 295 / 294 | 0.484 / 0.441 / 0.467 |
| bi-encoder · imb vs bal | 750 | **0 / 0 / 0** | 224 / 200 / 161 | 0.332 / 0.300 / 0.223 |
| +CE rerank · imb vs clu | 750 | **0 / 0 / 0** | 287 / 255 / 268 | 0.443 / 0.389 / 0.415 |
| +CE rerank · imb vs bal | 750 | **0 / 0 / 0** | 196 / 193 / 195 | 0.284 / 0.281 / 0.280 |

**0 of 750 queries receive the same top-5 exemplar list in any arm pair, in any mask mode, at either
retrieval stage.** The maximum overlap observed anywhere — all 18 pair × mode combinations, both stages — is **3 of 5**,
on 2–4 queries per cell. On
**59–66 %** of queries the two arms share **zero** exemplars (443/455/456 for the bi-encoder and
463/495/482 after rerank, of 750).

### 2.3 Prompts — different on every logged prompt

`runs/pool_sweep/decomposer/<cell>/<timestamp>/prompts_log/*.txt` — the harness logs every 5th prompt,
**150 per cell**, on the same index stride in every arm, so the files pair by name.

**0 of 150 logged prompts are identical** between imbalanced and clustered, in every one of the six
cells (and likewise for the other two arm pairs). As a whole-set check that does not depend on the
sampling stride, the recorded `prompt_tokens` field matches on only **3–15 of 750** items per cell —
i.e. even prompt *length* almost never coincides.

### 2.4 Generations — 18 separate inference runs

`results.json` in each of the 18 decomposer cells resolves through its own timestamp directory to a
file with a **distinct SHA-256 (18/18 unique)**. The timestamps are sequential and non-overlapping
(`20260822_112418` … `20260822_190500`, ~26 min apart), which is the signature of 18 real generation
passes, not of a cache being re-read. Decoding is deterministic (`do_sample: false`,
`temperature: 0.0`, `seed: 42`, `mistralai/Mistral-7B-Instruct-v0.3`), so identical output implies
identical prompt — and §2.3 shows the prompts are not identical.

All 18 `eval_per_item.json` files carry a `predictions_path` pointing into **their own** decomposer
cell — checked programmatically, **18/18 correct, 0 cross-wired**.

---

## 3. Which of (a), (b), (c) is it?

### (a) — yes, with counts

| link in the chain | do the arms differ? | evidence |
|---|---|---|
| pool rows | **yes** — 201/2000 shared (Jaccard 0.053) | `pools/*/pool.jsonl`, distinct SHA |
| retrieved exemplars | **yes** — 0/750 identical top-5 lists, mean 0.39–0.48 of 5 shared | `biencoder_top5/`, `rerank/` |
| prompts | **yes** — 0/150 logged prompts identical; 3–15/750 even share a token count | `decomposer/*/prompts_log/` |
| generations | **yes** — 690–700 of 750 differ | `decomposer/*/results.json`, 18/18 distinct SHA |
| **aggregate metrics** | **no** — 5/6 cells differ by ≤0.02, 1/6 ties exactly | `eval/*/eval_metrics.json` |

**The contrast was really run. The model is close to insensitive to which of these two pools its
exemplars came from.** That is a finding about the decomposer, not a bug in the harness.

### (b) — ruled out, on counts

No shared pool file, no shared retrieval file, no shared prompt, no shared generation file, no
cross-wired `predictions_path`. Every candidate mechanism for a plumbing defect was checked and each
one is negative. **I found no defect to name**, so question (4)'s "if (b), name the exact defect"
has no answer to give.

### (c) — the pool-level form is ruled out; a *mask-level* form is live and is the real story

Clustered does **not** degenerate into imbalanced by construction (§2.1: 201/2000 shared rows, 0
empty clusters, 0 top-up). But there is a weaker version of (c) that the data supports, and it is the
part worth telling the supervisor.

**The one property the retrieved exemplars share is their step-count distribution.** Counting the gold
decomposition length of all 3,750 retrieved exemplar slots (750 queries × 5) per cell:

| cell | imbalanced | clustered | balanced | imb−clu gap | imb−bal gap |
|---|---|---|---|---|---|
| biencoder_only · raw | 2.4224 | 2.4693 | 2.9843 | 0.047 | **0.562** |
| biencoder_only · typed | 2.2704 | 2.3123 | 2.6920 | 0.042 | **0.422** |
| biencoder_only · uniform | 2.2437 | 2.2931 | 2.7195 | 0.049 | **0.476** |
| biencoder_plus_ce · raw | 2.4523 | 2.5059 | 3.0005 | 0.054 | **0.548** |
| biencoder_plus_ce · typed | 2.3941 | 2.4501 | 2.9720 | 0.056 | **0.578** |
| biencoder_plus_ce · uniform | 2.3955 | 2.4885 | 3.0307 | 0.093 | **0.635** |

*(mean gold steps per retrieved exemplar)*

**Imbalanced and clustered sit 0.04–0.09 apart; balanced sits 0.42–0.64 away — roughly ten times
further.** `hop_count_exact_match` is precisely the metric that measures "did the model emit the right
number of steps", and the few-shot exemplars' step counts are the most direct signal the prompt
carries about that. So the pattern issue #46 reports — *hop-count EM is the only metric that separates
balanced from the other two, and imbalanced/clustered track each other* — has an exemplar-level
mechanism sitting right under it.

This is **suggestive, not established**: I measured the correlation between the exemplar step-count
mix and the hop-EM ordering across 6 cells; I did not run an intervention that holds exemplar identity
fixed and varies only the step-count mix, which is what would establish it. It is a hypothesis with a
measured antecedent.

---

## 4. How insensitive is the model? — total, within the range these pools produce

Cross-tabulating, per query, the number of shared exemplars between the imbalanced and clustered
top-5 against whether the two arms produced a byte-identical generation:

| shared exemplars (of 5) | queries (typed · +CE) | identical generations | rate |
|---|---|---|---|
| 0 | 495 | 42 | **8.5 %** |
| 1 | 222 | 14 | **6.3 %** |
| 2 | 29 | 3 | 10.3 % |
| 3 | 4 | 1 | 25.0 % |
| 4–5 | **0** | — | — |

The same flat pattern holds in all six cells (companion JSON,
`insensitivity_shared_exemplars_vs_identical_generation`). **Sharing more exemplars does not make the
outputs agree more.** At 0 shared exemplars the arms still agree 6.6 %–8.8 % of the time; at 2 shared
they agree at essentially the same rate. The 4- and 5-shared bins are empty everywhere, so the
question "how similar must the exemplar sets be before output stabilises?" **cannot be answered from
these artifacts** — the range was never sampled. That is a gap, and it is a gap in the experiment
design, not in this analysis.

What the agreeing items *are*, though, is clear: they are the short and easy ones. Mean generation
length on the agreeing subset is **72–84 characters / 2.1–2.4 step lines**, against **156–167
characters / 3.5–3.7 step lines** on the disagreeing subset — roughly half the length and a full step
shorter. The ~7 % agreement is not exemplar-driven convergence; it is the floor of items where any
reasonable prompt yields the same two-step answer.

**So the honest statement of the insensitivity is:** across a contrast that changed 90 % of the pool
rows, 100 % of the top-5 exemplar lists and 100 % of the prompts, the arms still land within 0.02 of
each other on hop-count EM in every cell, and issue #46 records every step-quality metric as
non-significant across all three strategies. The output *text* moves constantly — 690/750 items — but
the *quality* does not move in any direction the metrics can detect.

---

## 5. What this implies for issues #52 and #14

- **Issue #14's conclusion is not invalidated by a plumbing defect** — there isn't one. The clustered
  arm was built, retrieved from, prompted with and generated from independently of the imbalanced arm,
  and all 18 cells are correctly wired.
- **But #14 cannot claim clustering helps, and it should stop being presented as a contribution.**
  The measured result is: clustering changes the pool (Jaccard 0.053), changes the exemplars (0/750
  identical lists), changes the generated text (690/750 items) and **changes no reported metric by a
  detectable amount** (hop EM |Δ| ≤ 0.0200, McNemar p 0.39–1.00, six cells; every step-quality metric
  non-significant per #46). That is an **ablation with a null outcome**, which is a legitimate and
  reportable thesis result — but it is a different claim from the one on the slide.
- **The two confounds ADR 0021 already records against itself still bind** and should be stated
  wherever this null is reported: one trial per arm (`num_trials: 1`), so clustered-minus-random mixes
  strategy with draw noise; and construction and retrieval share the e5-small-v2 embedding space.
- **The `biencoder_plus_ce · typed` cell should never again be cited alone.** It is the one cell of six
  where the tie is exact, and quoting it as "the result" is what made a 129/129 coincidence look like
  a duplicated run. Report all six cells or report the range.
- **The supervisor's underlying worry is not fully answered by this note.** "Many changes, no effect in
  the result, not even a drop" is a pattern across the sweep, not only in these two arms. This note
  establishes that *these* two arms were really contrasted. Whether the same holds for the other
  knobs — masking modes, retriever variant, pool size — is **unmeasured here**, and issue #46 §4.1's
  non-significance across all three strategies suggests the question generalises. The +CE contrast is
  the one knob already shown to move something (ROUGE-L 9/9, GED 9/9,
  `2026-08-27-pool-construction-slide-facts.md` §5.2), which is evidence the harness *can* detect an
  effect when one exists — a useful positive control for the supervisor's bug hypothesis.

---

## 6. Log row — **exp-019**

Approved by the lead on 2026-09-14 and appended to [`experiments/log.md`](../../experiments/log.md)
as **exp-019**. exp-018 set the precedent that a re-analysis of committed *v2* artifacts producing new
reportable statistics earns a row (ADR 0020 covers *v1* re-analysis and requires none), and this note
produces several that exist nowhere else: the 688–700/750 per-item divergence, the 129/129 McNemar
decomposition of the 0.5133 tie, the 0/750 identical-top-5 counts, the 0.0529 pool Jaccard and the
exemplar step-count mix table. Cite **exp-019** for any of them. No GPU was used, no generation was
run, `runs/run.lock` was never acquired; the inputs are exp-010's artifacts at repo commit `13d2677`.

## 7. Method and limits

- Everything above is recomputed from `runs/pool_sweep/`, which is **untracked** per the
  reproducibility rule (heavy output stays on the compute box). The counts are reproducible from those
  artifacts; they are not reproducible from a fresh clone without them.
- Pool-arm comparisons pair on `id` / `query_id` / `item_id`; all pairings were verified as exact
  bijections over 750 (eval, retrieval, generation) or 2000 (pools) before any count was taken.
- McNemar p-values are **exact binomial**, two-sided, uncorrected — six cells, one metric. No
  multiplicity correction was applied because the finding is a null and correction would only widen it.
- Prompt comparison rests on the harness's **every-5th-prompt** log (150 of 750 per cell). The
  whole-set `prompt_tokens` check is a weaker but complete cross-check; full-text prompts for all 750
  are not retained and **cannot** be compared from these artifacts.
- The step-count-mix mechanism in §3(c) is **correlational across 6 cells**. It is not established.
- No GPU was touched, no generation was run, `runs/run.lock` was not acquired, no pipeline code was
  modified.
