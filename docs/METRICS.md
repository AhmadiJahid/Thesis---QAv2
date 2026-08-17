# Decomposition metrics — what the evaluator actually computes

Source of truth: [`scripts/musique_decompositions_evaluator.py`](../scripts/musique_decompositions_evaluator.py)
and [`configs/musique_eval.json`](../configs/musique_eval.json). Every statement below was
read off that code, not off a plan; the function that implements each metric is named so a
reader can check it. If the code and this file disagree, the code is right and this file is
a bug.

Every metric here is **string-level**: no model is in the loop (`METRIC_DEFINITIONS`
`not_a_semantic_metric`). Two decompositions that mean the same thing but word a step
differently score as a mismatch.

## 1. How rows are built

- **Predictions** (`--predictions`): a JSON list of objects with `question` and
  `decomposition`. A string `decomposition` is split on newlines with a leading `<n>. `
  enumerator removed per line (`_split_decomposition_text`); a list `decomposition` takes
  each string, or each element's `question` field (`_decomp_to_steps`).
- **Gold**: JSONL rows with `question` and `question_decomposition`, keyed by the question
  text lowercased and whitespace-collapsed (`_load_gold`, `_normalize_question`).
- **Joining is by question text, not by id** (`_build_eval_rows`). A prediction whose
  question has no gold row is counted in `missing_gold_count` and excluded from every
  metric.
- `limit` (config, or `--limit`) truncates the prediction list *before* matching.
- **`item_id`** is written into each per-item row: the prediction's `query_id`, else its
  `id`, else the normalized question text (`_item_id`). It is the key `--compare` aligns
  two runs on; it is not used for gold matching.

Two normalizations exist and they are **not** the same:

| | used by | rule |
|---|---|---|
| `_normalize_step` | exact match, step P/R/F1, ordered accuracy | lowercase, strip punctuation **except `#`** (so `[#k]` survives), collapse whitespace |
| `_tokenize` | ROUGE-L only | lowercase, collapse whitespace, split on spaces — **punctuation is not stripped** |

So `strike?` and `strike` are one token apart for ROUGE-L but identical for the step-level
metrics. This asymmetry is inherited from v1 and is deliberate only in the sense that
nobody has changed it; it is worth knowing before reading a ROUGE-L number closely.

## 2. The metrics

Unless stated otherwise, an aggregate is the **macro average**: the metric is computed per
item and the per-item values are averaged over evaluated rows (`_aggregate`). The single
exception is `reference_validity_micro`, which pools counts across all rows.

| metrics JSON key | definition (function) |
|---|---|
| `exact_match_rate` | 1.0 when pred and gold step lists have equal length and equal normalized text at every position, else 0.0; averaged (`_exact_decomposition_match`) |
| `step_precision_macro` / `step_recall_macro` / `step_f1_macro` | **set-based, unordered**: P = \|pred ∩ gold\| / \|pred\|, R = \|pred ∩ gold\| / \|gold\| over sets of normalized steps; F1 is the harmonic mean **per item**, then averaged (so it is not the F1 of the averaged P and R). Duplicate steps collapse in the set (`_step_prf`) |
| `ordered_step_accuracy_macro` | positional matches / `max(len(pred), len(gold))` — a length mismatch is penalised through the denominator (`_ordered_step_accuracy`) |
| `rouge_l_precision_macro` / `_recall_macro` / `_f1_macro` | see §3 |
| `reference_validity_macro` | per item: valid `[#k]` references / total references, where valid means `1 <= k < the step's own 1-based index`; an item with **no** references scores 1.0. Averaged over items (`_reference_validity`) |
| `reference_validity_micro` | total valid references / total references pooled over all rows; 1.0 when there are no references anywhere |
| `step_count_abs_error_mae` | mean of \|len(pred) − len(gold)\| |
| `step_count_mae` | the same number, reported again so the directional family reads as one block |
| `mean_signed_step_count_error` | mean of len(pred) − len(gold). **Over- and under-decomposition cancel here**, which is why the two rates below are reported next to it |
| `over_decomposition_rate` | fraction of rows with len(pred) > len(gold) |
| `under_decomposition_rate` | fraction of rows with len(pred) < len(gold) |
| `hop_count_exact_match_rate`, `hop_count_abs_error_mae` | predicted hop count = number of predicted steps; gold hop count = the gold row's `hop_count` when it is a positive int, else the gold step count |
| `predicted_hop_distribution`, `gold_hop_distribution` | counts of items per hop count |
| `per_gold_hop_metrics` | the **entire** aggregate block above, recomputed per gold hop depth (2, 3, 4, …) — including the four directional step-count metrics |

Per item, the same quantities are written to `<prefix>_per_item.json`, plus
`step_count_signed_error` (`len(pred) − len(gold)`), `pred_steps`, `gold_steps` and
`item_id`.

Reference validity checks **predicted** decompositions only; gold is never checked. Note
the sharp edge stated in `METRIC_DEFINITIONS`: a prediction with no `[#k]` references at
all scores 1.0 macro and contributes nothing to the micro denominator — a near-perfect
reference-validity number can mean "no references were emitted", not "references were
correct". (v1 saw exactly this; see `docs/prior-work.md`.)

## 3. ROUGE-L — the answer to the supervisor's question

Read off `_rouge_l`, `_join_steps` and `_tokenize`, verbatim behaviour:

- **The steps are joined into one string** before scoring: `_join_steps` concatenates the
  steps with `\n`, and the same is done to gold. There is no per-step alignment and no
  step-by-step ROUGE.
- **It is LCS-based.** `_lcs_len` is a standard dynamic-programming longest common
  subsequence over whitespace tokens of the lowercased joined text (the `\n` separator is
  collapsed to a space by `_tokenize`, so it is ROUGE-L over the flattened token stream,
  not summary-level ROUGE-L / ROUGE-Lsum).
- **It reports precision, recall and F1**: P = LCS / \|pred tokens\|, R = LCS / \|gold
  tokens\|, F1 = the unweighted harmonic mean (β = 1). All three are in the metrics JSON.
- **It is macro-averaged across items**: computed per item, then averaged over evaluated
  rows in `_aggregate`.
- Edge cases: both sides empty → (1.0, 1.0, 1.0); exactly one side empty → (0.0, 0.0, 0.0).
- No stemming, no stopword removal, no punctuation stripping, no external ROUGE package —
  this is the repo's own implementation, so it is not numerically comparable to a
  `rouge-score` / `pyrouge` number without checking their tokenizers.

## 4. Composite score

`_composite_score` combines **aggregate** values (not per-item ones):

```
composite = w_step_f1            * step_f1_macro
          + w_ordered            * ordered_step_accuracy_macro
          + w_reference_micro    * reference_validity_micro
          + w_step_count         * max(0, 1 - step_count_abs_error_mae / scale)
```

Weights and scale, from `configs/musique_eval.json` (they land in every run's config
snapshot and metrics JSON):

| config key | value |
|---|---|
| `composite_score_weights.step_f1_macro` | 0.4 |
| `composite_score_weights.ordered_step_accuracy_macro` | 0.3 |
| `composite_score_weights.reference_validity_micro` | 0.2 |
| `composite_score_weights.step_count_error` | 0.1 |
| `composite_step_count_error_scale` | 3.0 |

The weights are a **choice, not a result**: they were hard-coded literals in v1 and were
promoted to config in v2 so a run records them. Jahid's supervisor flagged this score as
handmade and possibly biased and asked for it to be checked against standard methods
(`docs/meetings/2026-08-12-supervisor-meeting-transcript-crosscheck.md`, open item 4). Two
consequences worth carrying into the write-up: the 0.2 reference-validity term is a micro
rate that can swing the composite on a handful of `[#k]` references, and the step-count
term is direction-blind (it uses the MAE, so over- and under-decomposition are penalised
identically). The directional metrics in §2 are reported separately precisely because the
composite cannot express that difference.

## 5. Paired comparison (`--compare A_per_item.json B_per_item.json`)

Compares two runs **on the same evaluation set**. Parameters live in
`configs/musique_eval.json` under `paired_comparison`: `bootstrap_iterations` 10000,
`alpha` 0.05, `max_reported_id_mismatches` 20, `out_prefix` `compare`. The seed is the
config's top-level `seed` (42) or `--seed`.

- **Alignment** (`_load_per_item`, `_aligned_ids`): rows are keyed by `item_id`. A
  duplicate id inside one file, or any id present in one file and not the other, aborts the
  run with the offending ids listed (capped at `max_reported_id_mismatches`). There is no
  silent intersection — a comparison across different evaluation sets is not a comparison
  (CLAUDE.md, evidence discipline). Aligned ids are processed in sorted order, so the
  bootstrap is reproducible.
- **Paired bootstrap CIs** (`_paired_bootstrap`, `_statistics_for`) for `rouge_l_f1`,
  `step_f1`, `ordered_step_accuracy` and `composite_score`. Each of the 10000 resamples
  draws n item indices with replacement (`numpy.random.default_rng(seed)`) and applies the
  **same** indices to both systems; every statistic is recomputed from the resampled items.
  The interval is the [alpha/2, 1−alpha/2] percentile of the difference **system_a minus
  system_b**, and `significant` is true when that interval excludes 0. The point estimate
  runs through the same function with the identity resample, so the observed value and the
  distribution cannot drift apart. `composite_score` is genuinely recomputed on each
  resample (its reference term is a micro rate and its step-count term a MAE, neither of
  which is an average of per-item values) using **this config's** weights and scale, which
  need not be the ones the per-item files were produced with.
- **McNemar** (`_mcnemar`, `_mcnemar_exact_p`) for the two binary metrics `exact_match` and
  `hop_count_exact_match`: with b = #(a correct, b wrong) and c = #(a wrong, b correct),
  the exact two-sided p-value is `min(1, 2 * BinomialCDF(min(b, c); b + c, 0.5))`, computed
  with exact integer arithmetic. With no discordant pairs p = 1.0. `significant` is
  `p < alpha`.
- **Six tests are reported and none is corrected for multiple comparisons.** The run note
  says so too.
- This substitutes paired bootstrap + McNemar for the "t-test" the supervisor asked for
  ([31:33] in the meeting cross-check); the substitution is recorded there as open item 8.

Output: `<out_prefix>_config.json`, `<out_prefix>_metrics.json` and `<out_prefix>_notes.md`
in the run directory, with a Markdown table of every statistic, its difference, its
interval or p-value, and its significant/not annotation.

## 6. "Composite score" names two different things — proposed fix

The name currently collides in Jahid's work:

1. **this decomposition-quality composite** — §4, computed by
   `scripts/musique_decompositions_evaluator.py`, key `composite_score` in its metrics
   JSON, and the same key aggregated by `scripts/pool_sweep_orchestrator.py` and plotted by
   `scripts/plot_pool_sweep.py`;
2. **a rank-weighted retrieval score** used on the retrieval / re-ranking side of the work.
   No implementation of it exists in this repository as of this commit — it is not defined
   here, and this file deliberately does not guess its formula.

Two different quantities under one name is a write-up hazard in November and a review
hazard before that. **Proposal: keep "composite score" for (1) and rename (2) to
"retrieval rank score."** Renaming (2) rather than (1) leaves every committed metrics JSON,
config key and plot in this repo valid. This is a naming proposal, not a decision: the
retrieval-side quantity is Jahid's, and the rename should be confirmed by him before it is
applied anywhere.

## 7. Running it

```bash
# score one run against gold
.venv/bin/python scripts/musique_decompositions_evaluator.py \
    --predictions runs/<...>/results.json

# paired significance between two scored runs (same eval set)
.venv/bin/python scripts/musique_decompositions_evaluator.py \
    --compare runs/<a>/eval_per_item.json runs/<b>/eval_per_item.json
```

The hand-computed checks for all of the above are in
[`tests/test_decomposition_metrics.py`](../tests/test_decomposition_metrics.py) (also run
as the `decomposition_metric_tests` stage of `scripts/smoke_test.py`); each test's
docstring shows the arithmetic behind its expected numbers against the fabricated fixture.
