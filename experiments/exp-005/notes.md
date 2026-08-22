# exp-005 — Guided vs unguided decomposition, Qwen/Qwen3.5-9B (issue #12)

Relaunch of exp-003 at the fixed SHA (`gen["raw"]` -> `gen["text"]`, PR #26). Three
conditions on `Qwen/Qwen3.5-9B` (admitted above the ~8B ceiling by ADR 0015,
pending supervisor confirmation), MuSiQue ADR 0007 pinned 600 (200/hop), retrieval
artifact per ADR 0014 (v1 9,156-example pool, sha256 `e5c418a9b25f...`). Generation
completed overnight 2026-08-21/22, rc=0 all three arms, 600/600 rows each, 0 empty
decompositions. Evaluated via `scripts/musique_decompositions_evaluator.py` at
commit `0068842` (pipeline code unchanged since generation at `6fc4bba`; the one
intervening commit only gitignores an untracked personal note file).

Run dirs: `runs/exp-005/20260821_233524` (unguided), `runs/exp-005/20260822_000025`
(oracle_guided), `runs/exp-005/20260822_003156` (unguided_capped, step-line cap 8;
4/600 rows hit the step-line cap, 2 of those actually cut off by the
StoppingCriteria).

## Headline (overall, n=600)

| condition | composite | step_f1 | ordered_step_acc | hop_count_EM | exact_match | step_count_MAE | mean_signed_err | over_rate | under_rate |
|---|---|---|---|---|---|---|---|---|---|
| unguided | 0.4212 | 0.2131 | 0.1911 | 0.5200 | 0.0617 | 0.642 | +0.398 | 0.365 | 0.115 |
| oracle_guided | 0.4429 | 0.2368 | 0.2205 | 0.8733 | 0.0733 | 0.538 | +0.472 | 0.093 | 0.033 |
| unguided_capped | 0.4215 | 0.2131 | 0.1911 | 0.5200 | 0.0617 | 0.633 | +0.390 | 0.365 | 0.115 |

## Per gold-hop-depth (over_decomposition_rate / under_decomposition_rate / step_count_mae / mean_signed_step_count_error)

| condition | hop=2 | hop=3 | hop=4 |
|---|---|---|---|
| unguided | over 0.280 / under 0.005 / mae 0.335 / signed +0.325 | over 0.400 / under 0.130 / mae 0.695 / signed +0.435 | over 0.415 / under 0.210 / mae 0.895 / signed +0.435 |
| oracle_guided | over 0.025 / under 0.000 / mae 0.025 / signed +0.025 | over 0.105 / under 0.060 / mae 0.745 / signed +0.625 | over 0.150 / under 0.040 / mae 0.845 / signed +0.765 |
| unguided_capped | over 0.280 / under 0.005 / mae 0.335 / signed +0.325 | over 0.400 / under 0.130 / mae 0.695 / signed +0.435 | over 0.415 / under 0.210 / mae 0.870 / signed +0.410 |

hop_count_exact_match_rate by hop: unguided 0.715 / 0.470 / 0.375 (hop 2/3/4);
oracle_guided 0.975 / 0.835 / 0.810 (large gain at every depth, largest at hop=4);
unguided_capped 0.715 / 0.470 / 0.375 (identical to unguided; the cap barely fired
on this model, 4/600 rows).

Full per-hop metrics: `experiments/exp-005/metrics.json` -> `eval_metrics.<condition>.per_gold_hop_metrics`.

## Pairwise comparisons (--compare, paired bootstrap 95% CI + McNemar + paired t-test, n=600 aligned items)

**unguided vs oracle_guided** (`experiments/exp-005/metrics.json` -> `comparisons.unguided_vs_oracle_guided`):
- step_f1: -0.0237, CI [-0.0351, -0.0123], significant; t=-4.09, p=4.8e-05 — significant
- ordered_step_accuracy: -0.0293, CI [-0.0403, -0.0188], significant; t=-5.34, p=1.3e-07 — significant
- rouge_l_f1: -0.0122, CI [-0.0214, -0.0032], significant; t=-2.63, p=0.0087 — significant
- composite_score: -0.0217, CI [-0.0348, -0.0070], significant
- exact_match (McNemar): -0.0117, p=0.0391, significant; t=-2.34, p=0.0195 — significant
- hop_count_exact_match (McNemar): -0.3533, p=5.0e-48, significant; t=-16.10, p=1.0e-48 — significant

Unlike Mistral (exp-004), oracle guidance on Qwen3.5-9B moves **every** quality
metric significantly in its favour, not just hop-count and composite — step_f1,
ordered_step_accuracy and rouge_l_f1 all show significant gains under oracle
guidance on this model.

**unguided vs unguided_capped**:
- Statistically indistinguishable on every metric (differences 0.0000-0.0003,
  composite_score CI [-0.0007,+0.0000] not significant); exact_match and
  hop_count_exact_match identical between the two runs (0 discordant pairs,
  McNemar p=1, t-test degenerate). The step-line cap fired on only 4/600 rows for
  this model (2 actually cut off), so it had almost no room to move any metric.

**oracle_guided vs unguided_capped**:
- Mirrors unguided vs oracle_guided in sign and magnitude (oracle_guided ahead on
  every metric): step_f1 +0.0237 significant (p=4.8e-05); ordered_step_acc +0.0293
  significant (p=1.3e-07); composite_score +0.0215 significant; hop_count_exact_match
  +0.3533 significant (McNemar p=5.0e-48, t p=1.0e-48); exact_match +0.0117
  significant (McNemar p=0.0391, t p=0.0195).

Full comparison JSONs (all 3 bootstrap stats, both McNemar rows, all 5 paired
t-test rows): `experiments/exp-005/metrics.json` -> `comparisons.*`.

## Cost per query (mean over 600 rows)

| condition | prompt tok | completion tok | total tok | latency (s) | rows at max_new_tokens |
|---|---|---|---|---|---|
| unguided | 398.4 | 34.8 | 433.2 | 2.479 | 0 |
| oracle_guided | 446.4 | 44.2 | 490.5 | 3.129 | 6 |
| unguided_capped | 398.4 | 34.8 | 433.1 | 2.474 | 0 |

## Reading against the issue #12 hypothesis

- On Qwen3.5-9B, oracle hop count materially improves decomposition quality
  across the board: step_f1, ordered_step_accuracy, rouge_l_f1, exact_match and
  hop_count_exact_match all move significantly in oracle_guided's favour (bootstrap
  CI excludes zero, McNemar and paired t-test both p<0.05), at a cost of +13%
  total tokens and +26% latency per query. This is a stronger and broader signal
  than the same comparison on Mistral-7B (exp-004), where only hop-count and
  composite moved significantly and step-level quality did not.
- `unguided_capped` is statistically indistinguishable from `unguided` on this
  model too — the step-line cap fired on only 4/600 rows, so this run cannot
  separate "runaway generation length" from "hop ignorance" as the dominant
  failure mode for Qwen3.5-9B either.
- Qwen3.5-9B arms remain out-of-scope-pending if the supervisor reasserts the 8B
  parameter ceiling (ADR 0015). No router-design recommendation is made here;
  the router go/no-go decision is Jahid's/the supervisor's call.

## Reproducibility

- Code: run-time commit `6fc4bba`; evaluation commit `0068842` (no pipeline code
  changed in between — the only intervening commit gitignores an untracked
  personal note file). Config: `configs/decomposer_musique.json`,
  `configs/musique_eval.json`. Seed 42 throughout (generation and evaluator).
- Predictions (`results.json`, per-item JSONs) and raw per-query text stay on the
  box under `runs/exp-005/`; only configs and metrics are committed here.
