# Decomposition metrics — what the evaluator actually computes

Source of truth: [`scripts/musique_decompositions_evaluator.py`](../scripts/musique_decompositions_evaluator.py)
and [`configs/musique_eval.json`](../configs/musique_eval.json). Every statement below was
read off that code, not off a plan; the function that implements each metric is named so a
reader can check it. If the code and this file disagree, the code is right and this file is
a bug.

Every metric here is **string-level**: no model is in the loop (`METRIC_DEFINITIONS`
`not_a_semantic_metric`). Two decompositions that mean the same thing but word a step
differently score as a mismatch.

**This file is the decomposition-*quality* half of MuSiQue evaluation only.** The end-to-end
half — execute the decomposition and score the answer it produces — is
[`components/answerer/run_answerer.py`](../components/answerer/run_answerer.py), whose answer
EM / answer F1 are defined in [`src/answer_metrics.py`](../src/answer_metrics.py) (MuSiQue's
official metrics) and whose conventions are recorded in ADR
[0019](adr/0019-musique-answering-backend-conventions.md). Those numbers are reported
overall and per gold hop depth, in the same style as `per_gold_hop_metrics` below, and they
are string-level too — no model scores anything on either side. The two halves share one
reading of "a step" (`src/step_lines.py`), pinned against each other by
`tests/test_answer_musique.py::TestStepReadingMatchesTheDecompositionEvaluator`.

## 1. How rows are built

- **Predictions** (`--predictions`): a JSON list of objects with `question` and
  `decomposition`. A string `decomposition` is split on newlines with a leading `<n>. `
  enumerator removed per line (`split_step_lines` in `src/step_lines.py`, aliased here as
  `_split_decomposition_text`); a list `decomposition` takes each string, or each element's
  `question` field (`_decomp_to_steps`). That splitter is shared with the decomposer: the
  `unguided_capped` condition's step-line budget and its rows-at-cap counter are defined on
  the same function, so "a cap of 8 step lines" and "8 steps" in this report are the same
  count.
- **Gold**: JSONL rows with `question` and `question_decomposition`, keyed by the question
  text lowercased and whitespace-collapsed (`_load_gold`, `_normalize_question`).
- **Gold carries two step-count denominators, and they are asserted equal at load.**
  `len(question_decomposition)` is the denominator of the directional step-count family;
  the `hop_count` field is the denominator of the hop-count family. `_load_gold` aborts,
  naming the offending ids (capped at `gold_validation.max_reported_mismatches`), on any
  row that has a positive `hop_count` disagreeing with its step count — otherwise the two
  families would describe different things inside one report. A row with no usable
  `hop_count` falls back to the step count, so the two are equal by construction. (PR #17's
  review reported 0 disagreements over 2417 real gold rows scanned — issue #20, finding 1;
  that scan is not re-run here. The assertion exists so a future gold file cannot change
  that silently.)
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
item and the per-item values are averaged over evaluated rows (`_aggregate`). Three keys are
not macro averages: `reference_validity_micro` pools counts across all rows;
`predicted_hop_distribution` / `gold_hop_distribution` are item counts per hop count; and
`composite_score` is computed from the aggregate values, not averaged from per-item ones
(§4).

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
| `step_count_exact_rate` | fraction of rows with len(pred) == len(gold), i.e. exactly 1 − `over_decomposition_rate` − `under_decomposition_rate`. **Not** `hop_count_exact_match_rate`: this one is against len(gold steps), that one against the gold `hop_count` field (§1). The two are equal whenever the gold passes the load-time assertion, but they are different definitions and only one of them belongs next to the over/under rates |
| `hop_count_exact_match_rate`, `hop_count_abs_error_mae` | predicted hop count = number of predicted steps; gold hop count = the gold row's `hop_count` when it is a positive int, else the gold step count |
| `predicted_hop_distribution`, `gold_hop_distribution` | counts of items per hop count |
| `per_gold_hop_metrics` | the **entire** aggregate block above, recomputed per gold hop depth (2, 3, 4, …) — including the five directional step-count metrics |

When reading the two direction rates, note that the errors are **not equally bad**: the
supervisor's judgment (2026-08-12 meeting, [33:23] — "a three-hop question, it's fine to
make it a four-hop, but it's not fine to make it a two-hop"; recorded per ADR 0017) is that
**over-decomposition is tolerable, under-decomposition is not**. So never collapse the two
rates into a single "wrong-length" figure, and read `mean_signed_step_count_error`'s sign
with this asymmetry in mind.

Per item, the same quantities are written to `<prefix>_per_item.json`, plus
`step_count_signed_error` (`len(pred) − len(gold)`), `pred_steps`, `gold_steps` and
`item_id`. That file is a JSON **object**, not a bare list: `schema`, `created_utc`,
`predictions_path`, `gold_path`, `composite_score_weights`,
`composite_step_count_error_scale`, then `items` (the per-item rows). The weights are
stamped there because `--compare` recomputes the composite (§5) and needs to know what the
rows were scored under. A file in the old bare-list format is refused by `--compare` with
an instruction to re-run the evaluation.

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
(`docs/meetings/2026-08-12-supervisor-meeting-transcript-crosscheck.md`, item 4; that
check is tracked as issue #29). Two
consequences worth carrying into the write-up: the 0.2 reference-validity term is a micro
rate that can swing the composite on a handful of `[#k]` references, and the step-count
term is direction-blind (it uses the MAE, so over- and under-decomposition are penalised
identically). The directional metrics in §2 are reported separately precisely because the
composite cannot express that difference.

### 4.1 Non-house diagnostics — `composite_no_ref_renorm`

Analysis notes sometimes need to ask *"how much of this composite difference is the
reference term?"*. The quantity they use for it is the composite with the
`reference_validity_micro` term **dropped** and the remaining weights **renormalized to sum
to 1**:

| term | house weight | renormalized weight |
|---|---|---|
| `step_f1_macro` | 0.4 | **0.5** |
| `ordered_step_accuracy_macro` | 0.3 | **0.375** |
| `step_count_error` (same `max(0, 1 - mae/scale)`, scale 3.0) | 0.1 | **0.125** |
| `reference_validity_micro` | 0.2 | **dropped** |

Rules, because this has now been used by two notes and it changes a headline in both:

- It is a **diagnostic, not a house metric.** The evaluator does not compute it; it is not in
  `configs/musique_eval.json`; it never appears in a run's `eval_metrics.json`. A note that
  reports it says so at the point of use.
- **No thesis claim may rest on it alone.** Its job is to show what the 0.2-weighted micro rate
  is doing to a comparison, so it is reported *beside* the house composite and the directional
  metrics in §2, never instead of them. Where it and the house composite disagree, that
  disagreement is the finding — not a licence to pick the friendlier number.
- Like the house composite it is built from **aggregate** values, so it has **no per-item
  value**: bootstrap only, no paired t-test and no McNemar.

Its two instances, both prior-work re-analyses under
[ADR 0020](adr/0020-prior-work-re-analysis-convention.md):
[`docs/analysis/2026-08-20-v1-masking-and-retrieval-significance.md`](analysis/2026-08-20-v1-masking-and-retrieval-significance.md)
§4, "the typed-vs-raw composite gap is almost entirely one term" (house composite not
significant, CI [−0.1842, +0.2259]; diagnostic +0.0273, CI [+0.0107, +0.0440], significant) and
[`docs/analysis/2026-08-20-v1-pool-size-significance.md`](analysis/2026-08-20-v1-pool-size-significance.md)
§4.4 (the pool-size trend **reverses direction** between the two). Both are input to issue #29.

Like the rest of this section's conventions, these rules are agent-side reporting documentation
(recorded 2026-08-20, PR #35) — Jahid's and his supervisor's to revise, not settled methodology.

## 5. Paired comparison (`--compare A_per_item.json B_per_item.json`)

Compares two runs **on the same evaluation set**. Parameters live in
`configs/musique_eval.json` under `paired_comparison`: `bootstrap_iterations` 10000,
`bootstrap_chunk_size` 1000, `alpha` 0.05, `min_items_for_significance_claim` 30,
`max_reported_id_mismatches` 20, `out_prefix` `compare`. The seed is the config's top-level
`seed` (42) or `--seed`.

- **Alignment** (`_load_per_item`, `_aligned_ids`): rows are keyed by `item_id`. A
  duplicate id inside one file, or any id present in one file and not the other, aborts the
  run with the offending ids listed (capped at `max_reported_id_mismatches`). There is no
  silent intersection — a comparison across different evaluation sets is not a comparison
  (CLAUDE.md, evidence discipline). Aligned ids are processed in sorted order, so the
  bootstrap is reproducible.
- **Same weights in both files** (`_require_matching_weights`): the two per-item files must
  carry the same `composite_score_weights` and `composite_step_count_error_scale`, or the
  run aborts printing both — a composite built from different weights is a different
  quantity. The config's weights (the ones the bootstrap composite is recomputed with) may
  still differ from the stamped ones; that no longer passes unnoticed, because the metrics
  JSON records `per_item_composite_score_weights`,
  `per_item_composite_step_count_error_scale` and
  `config_weights_match_per_item_files`, and the run note prints a WARNING line when they
  disagree.
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
  The resamples are drawn `bootstrap_chunk_size` rows at a time, so peak memory is
  O(`bootstrap_chunk_size` × n) rather than O(`bootstrap_iterations` × n); only the
  per-statistic difference vectors are kept. This does not change any number — `integers`
  consumes one draw per index in row-major order, so chunked draws concatenate to the
  single-block draw (pinned by `TestBootstrapChunking`, which also asserts equality with
  `chunk_size == iterations`, i.e. the un-chunked path).
- **McNemar** (`_mcnemar`, `_mcnemar_exact_p`) for the two binary metrics `exact_match` and
  `hop_count_exact_match`: with b = #(a correct, b wrong) and c = #(a wrong, b correct),
  the exact two-sided p-value is `min(1, 2 * BinomialCDF(min(b, c); b + c, 0.5))`, computed
  with exact integer arithmetic. With no discordant pairs p = 1.0. `significant` is
  `p < alpha`.
- **How much evidence there is, recorded next to the verdicts.** Every row carries `n` and
  `underpowered`; each McNemar row also carries `min_attainable_p_value` — the smallest p
  its discordant count m = b + c could have produced, `min(1, 2 · 0.5^m)`, reached when all
  discordant pairs favour one system — and `min_attainable_p_reaches_alpha`. With m = 3 that
  floor is 0.25, so "not significant" is a statement about n, not about the systems; m = 6
  is the smallest count that can reach p < 0.05 at all (0.03125). Bootstrap rows have no
  p-value and therefore no minimum p — their decision is whether the interval excludes 0.
  A top-level
  `significance_floor` block records n, `min_items_for_significance_claim`,
  `below_min_items` and a `warning` string (null when n is at or above the floor), and the
  run note prints the warning. `min_items_for_significance_claim` is a **reporting guard
  chosen in config, not a statistical standard** — it exists so a tiny evaluation set
  cannot print a bare `significant: true`; its value is Jahid's and his supervisor's to set.
- **Paired t-test** (`_paired_t_test`, `_paired_t_test_row`, `scipy.stats.ttest_rel`) over
  the same per-item differences, the same pairing and the same aligned items, for the five
  compared metrics that **have** a per-item value: `rouge_l_f1`, `step_f1`,
  `ordered_step_accuracy`, `exact_match`, `hop_count_exact_match`
  (`T_TEST_STATISTICS`). Each row carries `t_statistic`, `degrees_of_freedom` (n − 1) and
  `p_value`, with `significant` = `p_value < alpha` and the same `underpowered` flag as the
  rest. `composite_score` gets **no** t-test: its reference term is a micro rate and its
  step-count term a MAE, so there is no per-item composite difference to test — only its
  bootstrap CI, which recomputes it per resample. A row whose t is undefined (n < 2, or a
  zero standard deviation of the differences, e.g. a file compared with itself, or a
  non-finite result from the test) carries `t_statistic: null`, `p_value: null`,
  `significant: false` and a `degenerate` reason string, and makes no claim; its bootstrap
  CI still applies. Nothing non-finite is written to the metrics JSON: `NaN` and `Infinity`
  are not valid JSON, so they are turned into a degenerate row instead. The t-test is **additive**:
  bootstrap + McNemar remain the headline protocol (ADR 0009), and this covers the
  supervisor's literal ask for "a t-test" (ADR
  [0017](adr/0017-triage-of-the-2026-08-12-transcript-cross-check.md) item 4, issue #30).
  Note the assumption it makes and the others do not: these bounded, 0/1-heavy per-item
  scores are not obviously normal in their differences, which is why ADR 0009 chose the
  bootstrap and McNemar as the headline. The run note prints that caveat.
- **The headline protocol is six tests** (four bootstrap intervals + two McNemar p-values)
  **and the five t-tests are reported alongside them**; the counts are in the metrics JSON
  under `tests_reported`. **None** is corrected for multiple comparisons, and the t-test
  rows re-test the same metrics on the same items rather than adding independent tests. The
  run note says so too.
- **v1 prior-work inputs** (`--v1-per-item`, `--v1-alignment`; ADR
  [0020](adr/0020-prior-work-re-analysis-convention.md)). `--compare` can read v1's
  bare-list per-item files — the format that predates
  `musique_decomposition_per_item/1`, carrying the same per-item fields but no `item_id`
  and no stamped weights. It is **opt-in**: without the flag such a file is refused (the
  error names the flag), so a v1 artifact is never silently treated as a v2 one; with the
  flag on a v2 object the run aborts too, and mixing one of each is not supported. The
  `item_id` is reconstructed by `--v1-alignment`, either `normalized_question` (key = the
  normalized question text, rows in sorted order of it — the default from
  `paired_comparison.v1_compat.default_alignment`, and the alignment of the committed
  analysis note `docs/analysis/2026-08-20-v1-masking-and-retrieval-significance.md`) or
  `position` (row i paired with row i, for v1 files whose question texts are not unique).
  Either way the pairing has to be **witnessed**, and the witness fields are **required, not
  optional**: under `position` both `question` and `gold_steps` must be present on every row
  of both files and equal on every pair (ADR
  [0020](adr/0020-prior-work-re-analysis-convention.md) condition 3(b) — the alignment field
  *and* the gold), and under `normalized_question` the question is the alignment key, so its
  equality holds by construction and witnesses nothing while `gold_steps` is the required
  independent witness. A missing witness field **refuses the comparison**: a verification
  that checked nothing would report "same item" having established nothing (the case the
  PR #36 Gate-1 review demonstrated). A mismatch on a present witness aborts too, so a
  misordered file cannot be compared as if aligned. The output records which fields
  witnessed the pairing and on how many pairs
  (`v1_format_inputs.same_item_check.verification_fields` / `fields_verified_equal`), and
  the run note's sentence is built from those counts rather than asserted. Refusal messages on this path name a row by its index
  and a short hash of its key, never by the question text — an error pasted into an issue
  must not move dataset content into git. The output
  records a `v1_format_inputs` block: the prior-work caveat (these inputs carry **no commit
  SHA**, Gate 2 is not satisfied, the result is citable prior work and not a v2
  measurement), each input's sha256, mtime and row count, the alignment used and its
  definition, and that the composite weights are the config's because the files stamp none
  (`config_weights_match_per_item_files` is then `null`, not a bool). The caveat leads the
  run note and the stdout. Every compared field is type-checked at load: a JSON `null`, a
  string or a non-finite number in a metric column aborts naming the file, row and field,
  rather than surfacing as a `TypeError` inside the statistics.
  `paired_comparison.v1_compat.read_only_prior_work_root` names the v1 repo, and **any**
  `--run-dir` inside it is refused, v1 or not: v1 is read-only (ADR 0020 condition 1).

Output: `<out_prefix>_config.json`, `<out_prefix>_metrics.json` and `<out_prefix>_notes.md`
in the run directory, with a Markdown table of every statistic, its difference, its
interval **or** p-value (the column is headed `CI or p`, because the bootstrap rows carry an
interval while the McNemar and t-test rows carry a p-value), and its
significant/underpowered annotations. One row per test, with a `test` column naming which
family it came from.

### 5.1 Reporting power alongside a null result

When a comparison reports **no significant difference**, the note states the **minimum
detectable difference** at that n (paired, α = 0.05 two-sided, 80% power, from the observed
per-item difference SD: `MDE = 2.8016 * sd(diff) / sqrt(n)`, the constant being
z₀.₉₇₅ + z₀.₈ = 2.80158 rounded) and, where the observed difference is non-zero, the **n it
would need** for 80% power: `n = (2.8016 * sd(diff) / diff)^2`. The reason is that "not
significant" and "equal" are different claims, and only the first is supported — the two v1
notes cited in §4.1 both had to say so about a tie the reader would otherwise read as a null
result.
These are report-time figures computed from the same per-item differences as the tests; they
are not produced by `--compare` today. Like §4.1, this is an agent-side reporting convention
(recorded 2026-08-20, PR #35) — Jahid's and his supervisor's to revise, not settled methodology.

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

# the same battery over two v1 prior-work per-item files (ADR 0020; opt-in, and the
# output records that its inputs are v1 and carry no commit SHA)
.venv/bin/python scripts/musique_decompositions_evaluator.py \
    --compare <v1>/eval_typed_unguided_per_item.json <v1>/eval_raw_unguided_per_item.json \
    --v1-per-item --v1-alignment normalized_question --run-dir runs/<out>
```

The hand-computed checks for all of the above are in
[`tests/test_decomposition_metrics.py`](../tests/test_decomposition_metrics.py) (also run
as the `decomposition_metric_tests` stage of `scripts/smoke_test.py`); each test's
docstring shows the arithmetic behind its expected numbers against the fabricated fixture.
One fixture prediction (`2hop__d004_p`) differs from its gold **only** by punctuation, so
the `_normalize_step` rule at the top of §1 is itself pinned by golden values in both the
unit tests and the `musique_eval` stage of the smoke test: delete the punctuation-strip
line and both go red.
