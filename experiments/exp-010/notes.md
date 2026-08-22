# exp-010 -- pool sweep GPU follow-up: clustered vs balanced vs imbalanced (Refs #14)

## What this run is

The deferred GPU half of issue #14 / ADR 0021's pool-construction sweep. exp-006 produced
the CPU precompute (pool construction + bi-encoder similarity + CE rerank, pool size 2000,
all 3 construction strategies) against a shared 750-query dev sample. This run consumes
that precompute unedited and runs the 18 decompose+eval cells
(`{imbalanced, balanced, clustered} x {biencoder_only, biencoder_plus_ce} x {raw, typed, uniform}`)
via `scripts/pool_sweep_orchestrator.py --stage decompose` then `--stage eval`, both restricted
to `--only size2000_imbalanced,size2000_balanced,size2000_clustered`.

## Pre-launch check (required by the brief): typed/uniform are not a silent fallback

The dev sample carries no `question_masked_typed`/`question_masked_uniform` fields
(exp-006's own flag). `check_question_similarity.py`'s docstring says it NER-masks queries
on the fly when those fields are missing. Verified directly on exp-006's committed
`top20.jsonl` artifacts (all 3 balances): `typed_top_k`/`uniform_top_k` candidate-id lists
differ from `raw_top_k` on **750/750 rows in every balance** (0 rows identical across all
three modes). So `typed`/`uniform` are genuinely distinct retrieval conditions here -- no
cells were cut. Issue #14's own comment independently records the same finding.

Also traced (not a blocker): `configs/pool_sweep.json` points `configs.decomposer` at the
MetaQA-flavoured `configs/decomposer.json`, not `decomposer_musique.json`. With
`--retrieval-input` supplied (always, for this sweep), the question-loading branch that
reads `hops`/`questions_template_key` is never reached (hop comes from the retrieval file's
own `query_id` prefix), and the pinned-eval-set assertion is a no-op because
`decomposer.json` declares no `eval_rows_per_hop`. Confirmed live in the launch log: `"this
config declares no 'eval_rows_per_hop', so no pinned-set assertion applies"`, 750/750 rows
loaded, 250/250/250 per hop.

## Rc verification

All 18 decompose cells: rc=0 (`runs/pool_sweep/orchestrator.log`, elapsed 1515-1792s each,
~8h10m total for decompose). All 18 eval cells: rc=0 (elapsed 5-9s each). Overall `overall_rc=0`
in `runs/exp-010/run.log`. Every eval cell independently confirms 750/750 rows evaluated,
0 missing gold. 0 empty decompositions on the 3 headline cells;
`rows_at_max_new_tokens` (cap 128, `configs/decomposer.json`'s unedited default) 50/48/73 out
of 750 for imbalanced/balanced/clustered respectively -- the same cap applied uniformly, not
a confound between strategies.

## A shared-checkout complication, resolved with evidence

This run's wall clock (2026-08-22T11:23:00Z start -> 19:33:08Z end, ~8h10m) spans three PRs
that self-merged to `main` during the run: `fe4b32f` (#43, router readiness), `a940376`
(#42, end-to-end backends), `a99b573` (#44, additive Break-faithful metrics), plus two
docs-only commits (`ae29d38`, `f920fb8`). Each of the orchestrator's 36 subprocess
invocations (18 decompose, 18 eval) is spawned fresh and reads the file on disk at that
moment, so different cells of this one sweep genuinely ran under different `HEAD` commits --
verified by reading `git.commit` out of each decompose cell's own run config
(`experiments/exp-010/config.json`'s `commit_provenance_per_cell`): imbalanced ran entirely
on the pre-merge/docs-only commits, balanced split across docs-only and `a940376`, clustered
ran entirely on `a99b573`. The eval stage ran as one contiguous burst strictly after
`a99b573` landed, so all 18 `eval_metrics.json`/`eval_per_item.json` carry its four new
additive columns (`chain_validity_macro`, `break_exact_match_rate`, `sari_macro`,
`ged_macro`).

Checked with `git diff` restricted to every file this sweep's decompose+eval path reads
(full detail in `experiments/exp-010/config.json`): `configs/pool_sweep.json`,
`configs/similarity.json`, `configs/musique_prep.json`, the mistral model config and its
prompt files are byte-identical across the whole range. `run_decomposer.py`'s only change
(`resolve_hop_source`, from #43) is gated on `guided=true` or an explicit `hop_source`
condition key -- neither of which this sweep's unguided, no-conditions-block config sets --
so it is a documented no-op here, matching the pre-existing "gold" behaviour exactly.
`musique_decompositions_evaluator.py`'s change (#44) is additive by its own docstring and by
`_statistics_available()`'s handling of missing columns: `composite_score_weights` is
byte-identical in `configs/musique_eval.json`, and no metric this row headlines
(`step_f1`, `ordered_step_accuracy`, `hop_count_exact_match_rate`, `rouge_l_f1`,
`exact_match_rate`, `composite_score`) changes definition or value. Conclusion: predictions
are behaviourally identical regardless of which decompose-stage commit produced them, and
the eval numbers this row reports are not a re-score under the new metrics -- per the
coordinator's instruction, the nine-arm re-score under the new metrics is a separate,
later assignment, and this lane's job was to complete exp-010 as designed at its own
committed SHA, which this note demonstrates it did (mid-flight code drift notwithstanding,
because none of it touches a value this row reports).

## Headline comparison

`biencoder_plus_ce` + `typed` (ADR 0006 item 4's stated default; applicable here since the
pre-launch check confirms `typed` is genuinely distinct from `raw`). Pairwise `--compare`
(bootstrap 95% CI + McNemar + paired t-test, seed 42, n=750 aligned) among the 3 strategies:

- **step_f1 / ordered_step_accuracy / rouge_l_f1 / exact_match: not significant in any
  pairwise comparison.** All bootstrap CIs cross 0.
- **hop_count_exact_match_rate is the one significant axis.** imbalanced (0.5133) and
  clustered (0.5133, exactly tied) both significantly beat balanced (0.4667): McNemar
  p=0.044 and p=0.038, paired t-test p=0.038 and p=0.033. imbalanced vs clustered are
  statistically indistinguishable (identical rate, McNemar p=1.0).
- **composite_score**: recorded per schema (imbalanced 0.2001, balanced 0.1974, clustered
  0.2038, no pairwise difference significant) but **not headlined** -- issue #40 flags its
  reference-validity term as regex-artifact-suspect.

Per-hop (headline cell, step_f1/ordered_step_accuracy/hop_count_EM):
- hop=2: imbalanced 0.302/0.290/0.756, balanced 0.282/0.268/0.584, clustered 0.320/0.310/0.716
  -- the hop-count gap is concentrated here (imbalanced and clustered both well above balanced).
- hop=3: imbalanced 0.151/0.127/0.472, balanced 0.174/0.152/0.504, clustered 0.152/0.129/0.552
  -- clustered leads hop-count here; step-level metrics stay flat across strategies.
- hop=4: imbalanced 0.117/0.095/0.312, balanced 0.122/0.093/0.312, clustered 0.110/0.093/0.272
  -- all three converge to similarly low accuracy (the smallest pool bucket in every strategy:
  126/109/1175-of-9938 candidates for imbalanced/clustered/balanced respectively per exp-006).

**Reading**: pool-construction strategy moves whether the model states the right step COUNT
(hop_count_exact_match), not whether the steps it emits are the right ones (step_f1,
ordered_step_accuracy both flat) -- the same shape exp-005 (oracle hop guidance) and exp-009
(hop-matched retrieval) found on unrelated axes. No claim that clustering beats imbalanced:
they are statistically indistinguishable here, and both beat balanced only on the hop-count
axis, not on step quality.

## Pool composition (recap, measured by exp-006, not remeasured here)

imbalanced 2000 rows (1461/413/126 by hop), balanced 2000 rows (667/667/666), clustered 2000
rows (1396/495/109; k-means k=2000, 11 iters, inertia 1803.46, 0 empty clusters, 0 rows from
top-up).

## Artifacts

- Log entry: `experiments/log.md`, `exp-010` row (pre-launch commit `dddf30c`, completed in
  this commit).
- Config snapshot: `experiments/exp-010/config.json` (full commit-provenance verification).
- Metrics: `experiments/exp-010/metrics.json` (all 18 cells + headline per-hop + pairwise compare).
- Raw run log: `runs/exp-010/run.log`; launcher `runs/exp-010/run_exp010.sh`.
- Pairwise compare outputs: `runs/exp-010/compare/{imbalanced_vs_balanced,imbalanced_vs_clustered,balanced_vs_clustered}/compare_metrics.json`.
- Decompose/eval artifacts (gitignored, stay on the box): `runs/pool_sweep/{decomposer,eval}/...`,
  summary table `runs/pool_sweep/summary/all_runs.csv`.
