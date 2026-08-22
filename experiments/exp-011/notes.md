# exp-011 — nine-arm additive-metric re-score (Refs #40)

## What this run is

PR #44 (merged `a99b573`) added four additive columns to
`scripts/musique_decompositions_evaluator.py`: the Break-leaderboard trio
(`break_exact_match_rate`, `sari_macro`, `ged_macro` — GED is **lower is
better**, `ged_fallback_counts` records how many rows hit the search-free
upper-bound path) plus `chain_validity_macro` (the repaired per-item chaining
term). exp-010's 18 cells were evaluated at/after `a99b573` and already carry
these columns. Nine older committed arms did not: exp-004 (`unguided`,
`oracle_guided`, `unguided_capped`), exp-005 (same three conditions, Qwen3.5-9B),
exp-008 (`full_train`, the LoRA fine-tuned arm), exp-009 (`mixed`,
`oracle_hop_matched`).

This run re-scores each arm's already-committed `results.json` predictions at
current `main` (`f4a1a19`) — same predictions, same gold
(`musique_ans_v1.0_dev_clean.jsonl`), same eval set (ADR 0007 pinned 600,
200/hop, verified 600/600 rows and 0 missing gold on every arm before and
after). **No pipeline code or config changed** — `configs/musique_eval.json`
is byte-identical to the version PR #44 shipped, and the evaluator's own
docstring states the four new columns are strictly additive (they do not
enter `composite_score`, and no pre-existing metric changes value — verified:
every re-scored arm's `step_f1_macro` / `ordered_step_accuracy_macro` /
`hop_count_exact_match_rate` / `composite_score` reproduces its original log
row exactly). No new decomposition or inference run, CPU-only, no GPU touched.

**This is a coverage run, not a comparison.** No `--compare` was invoked here.
Cross-arm "improved/beat" claims already exist in exp-004/005/008/009's own
rows on their own eval sets and are not re-derived or extended here.

## Caveat carried from issue #40

`composite_score` is recorded per schema in `experiments/exp-011/metrics.json`
for every arm but is **not** headlined: issue #40 documents a known
`reference_validity` defect in its 0.2-weight term (a regex/bare-vs-bracketed-
`#k` mismatch) that predates and is unrelated to this run's new columns. The
headline metrics below are `step_f1_macro` / `ordered_step_accuracy_macro` /
`hop_count_exact_match_rate` (all unchanged from each arm's original row) plus
the four new additive columns.

## Descriptive results across the nine arms

| arm | source | break_EM | SARI (macro) | GED (macro, lower better) | GED fallback (node_cap) | chain_validity (macro) |
|---|---|---|---|---|---|---|
| exp004_unguided | exp-004 unguided (Mistral-7B) | 0.0533 | 0.5850 | 0.4715 | 1/600 | 0.9686 |
| exp004_oracle_guided | exp-004 oracle_guided (Mistral-7B) | 0.0500 | 0.5890 | 0.4414 | 0/600 | 0.9712 |
| exp004_unguided_capped | exp-004 unguided_capped (Mistral-7B) | 0.0533 | 0.5849 | 0.4708 | 0/600 | 0.9686 |
| exp005_unguided | exp-005 unguided (Qwen3.5-9B) | 0.0517 | 0.5979 | 0.4910 | 0/600 | 0.8605 |
| exp005_oracle_guided | exp-005 oracle_guided (Qwen3.5-9B) | 0.0600 | 0.6033 | 0.4241 | 6/600 | 0.8697 |
| exp005_unguided_capped | exp-005 unguided_capped (Qwen3.5-9B) | 0.0517 | 0.5978 | 0.4909 | 0/600 | 0.8605 |
| exp008_full_train | exp-008 full_train (Mistral-7B + LoRA) | 0.1017 | 0.6823 | 0.3094 | 0/600 | 0.9942 |
| exp009_mixed | exp-009 mixed (Mistral-7B) | 0.0533 | 0.5851 | 0.4713 | 1/600 | 0.9686 |
| exp009_oracle_hop_matched | exp-009 oracle_hop_matched (Mistral-7B) | 0.0483 | 0.5879 | 0.4540 | 1/600 | 0.9686 |

Full per-arm tables (including `per_gold_hop_metrics`, provenance —
predictions sha256, evaluator git commit at scoring, run dir) are in
`experiments/exp-011/metrics.json`.

## Reading (descriptive only — no significance testing run here)

- `exp008_full_train` (the LoRA fine-tuned arm) leads every new metric by a
  wide margin: highest `break_exact_match_rate` (0.102, roughly double the
  next-highest arm), highest `sari_macro` (0.682), lowest (best) `ged_macro`
  (0.309), and the highest `chain_validity_macro` (0.994). This is the same
  direction as exp-008's already-logged significant advantage on
  `step_f1`/`ordered_step_accuracy`/`composite_score` over the Mistral
  prompting baselines — the new columns do not contradict that reading, they
  add detail to it. No significance test was re-run here (no `--compare`
  invoked), so "leads" is descriptive, not a tested claim.
- `chain_validity_macro` separates by **model**, not by prompting condition:
  the three exp-005 arms (Qwen3.5-9B) sit at 0.860–0.870, visibly lower than
  every Mistral-7B arm (0.969–0.994, exp-008's fine-tuned arm highest). This
  is despite exp-005's own log row recording Qwen3.5-9B ahead of Mistral-7B
  on `step_f1`/`ordered_step_accuracy` (exp-004 vs exp-005) — i.e. on this
  additive chaining-validity axis the ranking does not obviously follow the
  ranking on the pre-existing quality metrics. This is an observation, not a
  tested comparison (the two experiments were never run through `--compare`
  against each other, on this axis or any other, in any log row); it is
  flagged here for whoever picks this up next.
- Within a triple (`unguided` / `oracle_guided` / `unguided_capped`, same
  base model), `oracle_guided` is consistently the best on all four new
  metrics vs its own model's `unguided`/`unguided_capped` (exp-004:
  `break_EM` 0.0500 vs 0.0533/0.0533, `SARI` 0.589 vs 0.585/0.585, `GED` 0.441
  vs 0.471/0.471; exp-005: `break_EM` 0.060 vs 0.0517/0.0517, `SARI` 0.603 vs
  0.598/0.598, `GED` 0.424 vs 0.491/0.491) — the same direction as those
  arms' already-logged (in some cases significant, in some not) advantage on
  the older metrics. `unguided` and `unguided_capped` are effectively
  identical on every new metric in both triples, matching the original
  finding that the step-line cap fired on very few rows.
- `GED` fallback (`ged_fallback_counts.node_cap`, the >16-node search-free
  upper-bound path per `configs/musique_eval.json`'s `break_metrics.ged`
  block) fired on at most 6/600 rows in any arm (exp005_oracle_guided) and 0
  or 1/600 elsewhere — negligible, consistent with `configs/musique_eval.json`'s
  own note that the machine-dependent optimizer path is "effectively
  unreachable for graphs under the cap" for MuSiQue-depth gold.
- exp-009's two arms (`mixed` vs `oracle_hop_matched`, few-shot retrieval
  source, same base model/condition as exp-004's `unguided`) land within a
  point or two of exp-004's `unguided`/`unguided_capped` on every new metric
  (`break_EM` 0.053/0.048 vs 0.053; `SARI` 0.585/0.588 vs 0.585; `GED`
  0.471/0.454 vs 0.471) — small movement, no significance test run to
  characterise it (exp-009's own log row already ran `--compare` between its
  two arms on the *old* metrics and found only `hop_count_exact_match_rate`
  significant; that test is not re-run here for the new columns).

## Provenance / reproducibility

- Evaluator: `scripts/musique_decompositions_evaluator.py` at `main` commit
  `f4a1a19e68a518f41857f3de8cc8904650f11d04` (clean tree), config
  `configs/musique_eval.json` (unedited, byte-identical to the PR #44 version
  every arm's original composite-score weights were computed under).
  `configs/musique_eval.json`'s own `_note` on the additive `break_metrics.ged`
  block, plus the definitions of every new column, are embedded verbatim in
  `experiments/exp-011/metrics.json`'s `metric_definitions`.
- Predictions: each arm's already-committed/on-box `results.json` (paths and
  sha256 in `experiments/exp-011/metrics.json.arms.<arm>`), unchanged by this
  run.
- Gold: `/cta/users/fyilmaz/thesis-qav2-data/musique/dev_data/musique_ans_v1.0_dev_clean.jsonl`
  (same file every source arm was scored against; `gold_key
  musique_dev_gold_clean` in `configs/musique_eval.json`, unedited).
- Seed 42 throughout (the evaluator's own scoring is deterministic; seed
  matters for reproducibility record-keeping, not for any bootstrap here since
  no `--compare` was run).
- `runs/exp-011/<arm>/{eval_metrics.json,eval_per_item.json,eval_notes.md,
  eval_config.json,eval.log}` hold the full per-arm evaluator trail
  (gitignored, on-box). `experiments/exp-011/{config.json,metrics.json,
  notes.md}` are the committed summary.
- Original artifacts under `experiments/exp-004/`, `experiments/exp-005/`,
  `experiments/exp-008/`, `experiments/exp-009/` were not read for writing
  and are untouched by this run (verified via `git status` before commit).
- `runs/run.lock` was acquired (`exp-011 <ISO timestamp>`) before the first
  evaluator invocation and released after this note and
  `experiments/exp-011/{config.json,metrics.json}` were committed. GPU state
  was irrelevant (CPU-only) and was not checked or waited on beyond the
  routine idle confirmation at lock acquisition.
