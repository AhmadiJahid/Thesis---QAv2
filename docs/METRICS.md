# Decomposition metrics — what the evaluator actually computes

Source of truth: [`scripts/musique_decompositions_evaluator.py`](../scripts/musique_decompositions_evaluator.py)
and [`configs/musique_eval.json`](../configs/musique_eval.json). Every statement below was
read off that code, not off a plan; the function that implements each metric is named so a
reader can check it. If the code and this file disagree, the code is right and this file is
a bug.

Every metric here is **string-level**: no model is in the loop (`METRIC_DEFINITIONS`
`not_a_semantic_metric`). Two decompositions that mean the same thing but word a step
differently score as a mismatch. That still holds for the metrics added in §2.1 and §2.2:
the graph edit distance is computed by `networkx`, a graph library, and every candidate that
would have put a *model* in the scoring loop is left out — whether the supervisor's
"no closed commercial model may judge decomposition quality" extends to open-weight scorers
is his question, recorded as such in
[`docs/analysis/2026-08-22-metric-candidates.md`](analysis/2026-08-22-metric-candidates.md) §6.

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
| `_tokenize` | ROUGE-L, and GED's node substitution cost (§2.2) | lowercase, collapse whitespace, split on spaces — **punctuation is not stripped** |
| `_break_steps` / `_break_string` | Break EM, SARI, GED (§2.2) | per step: collapse whitespace, rewrite `#k` → `@@k@@`; steps joined with `" @@SEP@@ "`. **No lowercasing** (EM lowercases the joined string itself) and **no punctuation stripping** |

So `strike?` and `strike` are one token apart for ROUGE-L but identical for the step-level
metrics. This asymmetry is inherited from v1 and is deliberate only in the sense that
nobody has changed it; it is worth knowing before reading a ROUGE-L number closely.

## 2. The metrics

Unless stated otherwise, an aggregate is the **macro average**: the metric is computed per
item and the per-item values are averaged over evaluated rows (`_aggregate`). Four keys are
not macro averages: `reference_validity_micro` pools counts across all rows;
`predicted_hop_distribution` / `gold_hop_distribution` are item counts per hop count;
`ged_fallback_counts` is an item count per fallback reason (§2.2); and
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
| `chain_validity_macro` | see §2.1 |
| `break_exact_match_rate`, `sari_macro`, `ged_macro`, `ged_fallback_counts` | see §2.2 |
| `per_gold_hop_metrics` | the **entire** aggregate block above, recomputed per gold hop depth (2, 3, 4, …) — including the five directional step-count metrics and everything in §2.1 / §2.2 |

When reading the two direction rates, note that the errors are **not equally bad**: the
supervisor's judgment (2026-08-12 meeting, [33:23] — "a three-hop question, it's fine to
make it a four-hop, but it's not fine to make it a two-hop"; recorded per ADR 0017) is that
**over-decomposition is tolerable, under-decomposition is not**. So never collapse the two
rates into a single "wrong-length" figure, and read `mean_signed_step_count_error`'s sign
with this asymmetry in mind.

Per item, the same quantities are written to `<prefix>_per_item.json`, plus
`step_count_signed_error` (`len(pred) − len(gold)`), `pred_steps`, `gold_steps` and
`item_id`, plus the §2.1 / §2.2 columns: `chain_validity`,
`chain_pred_reference_count`, `chain_gold_reference_count`, `break_exact_match`, `sari`,
`ged` and `ged_fallback`. That file is a JSON **object**, not a bare list: `schema`, `created_utc`,
`predictions_path`, `gold_path`, `composite_score_weights`,
`composite_step_count_error_scale`, then `items` (the per-item rows). The weights are
stamped there because `--compare` recomputes the composite (§5) and needs to know what the
rows were scored under. A file in the old bare-list format is refused by `--compare` with
an instruction to re-run the evaluation.

Reference validity checks **predicted** decompositions only; gold is never checked. Note
the sharp edge stated in `METRIC_DEFINITIONS`: a prediction with no `[#k]` references at
all scores 1.0 macro and contributes nothing to the micro denominator — a near-perfect
reference-validity number can mean "no references were emitted", not "references were
correct". (v1 saw exactly this; see `docs/prior-work.md`.) That edge, plus the syntax
mismatch of [issue #40](https://github.com/AhmadiJahid/Thesis---QAv2/issues/40) (`_REF_RX`
matches `[#k]`, MuSiQue's gold writes bare `#k`), is why `chain_validity` exists beside it
— §2.1. **Neither `reference_validity_*` nor `composite_score` was changed:** the two terms
are reported side by side so both can be read, and issue #40's open question ("is the
bracketed syntax a defect or an intended v1-carried definition?") is Jahid's, not answered
by the code.

### 2.1 `chain_validity` — the repaired chaining term (per item, no free credit)

Read off `_chain_validity`, and **additive**: it is a new column, it does not enter
`composite_score`, and no number that existed before it moved.

Per item, in this order:

| case | value |
|---|---|
| the **gold** emits no `#k` reference (chaining is not required) | 1.0 |
| the gold chains and the prediction emits **no** reference at all | **0.0** — the repair |
| otherwise | valid predicted references / total predicted references, valid = `1 <= k <` the step's own 1-based index |

Two differences from `reference_validity_*`: references are matched by `#(\d+)`, which sees
the `#1` in `#1` **and** in `[#1]` (this is Break's own rule — `format_qdmr` rewrites
`#(\d+)` → `@@\1@@`); and silence is not free. The second is not cosmetic:
[`docs/analysis/2026-08-22-metric-candidates.md`](analysis/2026-08-22-metric-candidates.md)
§3.2 measured the free-credit convention **flipping** a model ranking, with 76 of 600 items
in one arm paid 1.0 for chaining not at all. It has a per-item value, so the whole §5
battery applies to it. It is **not** a published metric — it is PR #38 §5 option 3(a) made
concrete, and it is labelled that way wherever it is defined.

Per item the file also carries `chain_pred_reference_count` and
`chain_gold_reference_count`, so a 0.0 can be told apart from a 0.0: an item that emitted
nothing versus one whose references were all invalid.

### 2.2 Break's own metrics — EM, SARI, GED

Ported from the **official** Break leaderboard evaluator (`allenai/break-evaluator` at
master: `scripts/evaluate_predictions.py`, `evaluation/sari_hook.py`,
`evaluation/graph_matcher.py`, `evaluation/decomposition.py`,
`evaluation/sequence_matcher.py`), read file by file — the survey note §1.2 and §7 record the
same reading. Additive, like §2.1: none of the three enters `composite_score`. The
conventions and the deviations are ADR
[0026](adr/0026-break-faithful-metrics-the-implementers-conventions.md).

All three score the **`" @@SEP@@ "`-joined** decomposition (`_break_steps` collapses
whitespace per step and rewrites `#k` → `@@k@@`; `_break_string` joins).

| metrics JSON key | definition (function) |
|---|---|
| `break_exact_match_rate` | official `get_exact_match`: `pred.lower() == gold.lower()` on the joined string, **no punctuation stripping** (`_break_exact_match`). Strictly harder than `exact_match_rate`, which normalizes each step — on the fixture row that differs from its gold only by punctuation, `exact_match` is 1.0 and this is 0.0, and both are correct |
| `sari_macro` | official SARI (`_sari`): `(keep-F1 + add-F1 + delete-precision) / 3`, averaged over n-grams n = 1…4, with the **question** as the source and the joined decompositions as prediction and single target. `_get_fbeta_score`'s 0/0 = 1 convention and `BETA_FOR_SARI_DELETION_F_MEASURE = 0` are reproduced |
| `ged_macro` | official `normalized_graph_edit_distance` (`_normalized_ged`): the last value `networkx.optimize_graph_edit_distance` yields for (prediction, gold), with the lexical node substitution cost and unit edge/node insert-delete costs, divided by `max(nodes + edges)` of the two graphs. **LOWER IS BETTER** and it can exceed 1.0 |
| `ged_fallback_counts` | how many items did not get the optimizer's own value, by reason (`node_cap`, `time_budget`, `time_budget_no_yield`). `{}` means every value is the optimizer's |
| `ged_policy` | the two guards in force for the run, plus the direction and the node-cost convention, copied into the metrics JSON and the config snapshot |

**The graph** (`_decomposition_graph`, official `Decomposition.to_graph`): node `i` is the
i-th step (1-based) labelled with its text; an edge runs **from** the referencing step **to**
the step it references (official `(i+1, ref)`); a reference to a step that does not exist
creates that node with an empty label, so a `#7` in a 3-step plan pays for the dangling node.

**The node substitution cost** (`_node_subst_cost`, `_match_score`) is
`1 - 2·matches / (len(a) + len(b))` over the two step texts, where `matches` is the number of
equal pairs on a minimum-edit-distance alignment (`_alignment_matches`, a port of
`edit_distance.edit_distance` including its tie-break order — the number of matches on a
*minimum-cost* path is **not** the LCS length).

**Three named deviations from the official code**, so nothing here is read as a leaderboard
number:

1. **No spaCy.** Break lemmatizes the substitution cost's tokens with `en_core_web_sm`; this
   implementation uses lowercased whitespace tokens (`_tokenize`), because spaCy is not a
   dependency and its model needs a download. **Absolute GED values are therefore not
   comparable to published Break GED.** Within one evaluation every system is scored
   identically, so comparisons on this data are valid.
2. **`norm_EM` is not ported.** Break's fourth leaderboard metric normalizes with ~14
   QDMR-*operation-specific* rewrite rules over spaCy parses plus a 16-way operation
   classifier; MuSiQue sub-questions are free-form natural language, so porting it means
   writing a new canonicalizer whose validity is argued from scratch (survey §1.3).
3. **No `return`-stripping and no `;`-splitting** from `format_qdmr`: the first would eat the
   substring inside ordinary English words ("returned" → "ed"), the second splits QDMR's step
   separator where `src/step_lines.py` already split ours. Whitespace collapsing and the
   reference rewrite **are** applied.

**Cost, and what happens when GED runs out of it.** Break wraps the optimizer in
`@exit_after(180)` and its aggregator **drops** an item that times out. Dropping is not
available here — a dropped item has no pair, and the whole §5 battery is paired — so an item
over budget is reported with a documented **upper bound** and flagged in `ged_fallback`.
Two guards, both in `configs/musique_eval.json` under `break_metrics.ged`:

- `max_nodes_for_optimizer` (30): above it the optimizer is not called and the value is the
  cost of one concrete edit path (pair nodes in sorted id order, substitute each pair,
  delete/insert the surplus, keep the edges whose endpoints are both paired, delete/insert
  the rest), normalized the same way. Deterministic — the same number on every machine.
- `per_item_time_budget_seconds` (20.0): a wall-clock backstop under the cap, checked between
  the optimizer's successive approximations, keeping the last one. This is the **one
  machine-dependent path** in the report, which is why each firing is counted.

**Two sharp edges to know before quoting a number.** (a) *SARI's floor is not 0 on this
data*: every decomposition shares the `@@SEP@@` and template boilerplate, which inflates the
keep and add terms, and SARI rewards deleting question tokens — the survey measured 0.29 for
an empty prediction over the real 600. **Absolute SARI levels here are not interpretable;
differences on the same data are.** (b) *GED is order-light, and on a 2-step plan it is
order-blind*: reversing a 2-step decomposition turns its `#1` into a self-loop, and
networkx's edit-path search prices that self-loop against an ordinary edge at 0, so a
reversed 2-step plan scores **GED 0.0** (a 3-step reversal is priced correctly). The official
evaluator computes GED through the same networkx call, so this is kept and pinned by a test
rather than "fixed" — fixing it would be inventing a metric. `break_exact_match` catches the
reversal in both cases, which is the survey's §4 item 2 in one sentence: an order-sensitive
metric has to stay in the reported set.

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

**The composite is unchanged by the metrics of §2.1 and §2.2.** None of `chain_validity`,
`break_exact_match`, `sari` or `ged` enters this formula, and neither
`reference_validity_micro` nor its `[#k]`-only regex was touched, so every committed
`composite_score` still has the value it was published with. Whether the composite keeps
leading — or is repaired, or retired — is issue #6 item 5, Jahid's with his supervisor; the
code measures, it does not adopt.

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
  `step_f1`, `ordered_step_accuracy`, `sari`, `ged`, `chain_validity` and `composite_score`
  (`BOOTSTRAP_STATISTICS`). Each of the 10000 resamples
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
- **Direction, carried in the data** (`LOWER_IS_BETTER_STATISTICS`, `_direction`,
  `_favours`). `ged` is a **distance** — lower is better — and it is the only one in the
  report, so a bare `-0.16` on that row means the *opposite* of what it means on every other
  row. Every bootstrap, McNemar and t-test row therefore carries `direction`
  (`higher_is_better` / `lower_is_better`) and, when it is significant, `favours`
  (`system_a` / `system_b`) with the direction already applied (null otherwise); the metrics
  JSON also lists `lower_is_better_statistics`; and the run note's table has a `better`
  column plus a sentence saying which way `ged` reads, with each significant verdict
  annotated `yes (favours a|b)`.
- **McNemar** (`_mcnemar`, `_mcnemar_exact_p`) for the three binary metrics `exact_match`,
  `hop_count_exact_match` and `break_exact_match`: with b = #(a correct, b wrong) and c = #(a wrong, b correct),
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
  the same per-item differences, the same pairing and the same aligned items, for the nine
  compared metrics that **have** a per-item value: `rouge_l_f1`, `step_f1`,
  `ordered_step_accuracy`, `sari`, `ged`, `chain_validity`, `exact_match`,
  `hop_count_exact_match`, `break_exact_match`
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
- **The headline protocol is the bootstrap intervals plus the McNemar p-values** (ten of
  them on v2 inputs today: seven bootstrap + three McNemar) **and the t-tests are reported
  alongside them** (nine). The count grew with the additive metrics of §2.1 / §2.2, so read
  the exact numbers off `tests_reported` in the metrics JSON rather than off this sentence.
  **None** is corrected for multiple comparisons, and the t-test
  rows re-test the same metrics on the same items rather than adding independent tests. The
  run note says so too.
- **A compared metric the inputs do not carry is skipped and named**, not silently omitted:
  `statistics_not_available_in_inputs` in the metrics JSON, plus a `NOT COMPARED` line in the
  run note. On v2 artifacts it is empty (the §2.1 / §2.2 columns are **required** — a
  per-item file written before them gets the existing "re-run the evaluation to regenerate
  them" refusal). It is non-empty only for `--v1-per-item` inputs, which predate those
  columns; computing them there from the stored steps would be a *re-score of v1 output*
  rather than a comparison of what v1 measured.
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
