# exp-004 — Guided vs unguided decomposition, Mistral-7B-Instruct-v0.3 (issue #12)

Relaunch of exp-002 at the fixed SHA (`gen["raw"]` -> `gen["text"]`, PR #26). Three
conditions on `mistralai/Mistral-7B-Instruct-v0.3`, MuSiQue ADR 0007 pinned 600
(200/hop), retrieval artifact per ADR 0014 (v1 9,156-example pool, sha256
`e5c418a9b25f...`). Generation completed overnight 2026-08-21/22, rc=0 all three
arms, 600/600 rows each, 0 empty decompositions. Evaluated via
`scripts/musique_decompositions_evaluator.py` at commit `0068842` (pipeline code
unchanged since generation at `6fc4bba`; the one intervening commit only
gitignores an untracked personal note file).

Run dirs: `runs/exp-004/20260821_222907` (unguided), `runs/exp-004/20260821_224805`
(oracle_guided), `runs/exp-004/20260821_231641` (unguided_capped, step-line cap 8;
12/600 rows hit the step-line cap, 7 of those actually cut off by the
StoppingCriteria).

## Headline (overall, n=600)

| condition | composite | step_f1 | ordered_step_acc | hop_count_EM | exact_match | step_count_MAE | mean_signed_err | over_rate | under_rate |
|---|---|---|---|---|---|---|---|---|---|
| unguided | 0.2098 | 0.2039 | 0.1804 | 0.5083 | 0.0617 | 0.775 | +0.335 | 0.290 | 0.202 |
| oracle_guided | 0.4197 | 0.2061 | 0.1852 | 0.5900 | 0.0567 | 0.550 | +0.427 | 0.348 | 0.062 |
| unguided_capped | 0.2128 | 0.2040 | 0.1805 | 0.5083 | 0.0617 | 0.687 | +0.247 | 0.290 | 0.202 |

Composite jumps for oracle_guided almost entirely through `reference_validity_micro`
(weight 0.2) and the step-count-error term, not through step_f1/ordered_step_acc,
which move within noise between unguided and oracle_guided (see per-hop table and
comparisons below).

## Per gold-hop-depth (over_decomposition_rate / under_decomposition_rate / step_count_mae / mean_signed_step_count_error)

| condition | hop=2 | hop=3 | hop=4 |
|---|---|---|---|
| unguided | over 0.280 / under 0.010 / mae 0.450 / signed +0.430 | over 0.290 / under 0.195 / mae 0.705 / signed +0.315 | over 0.300 / under 0.400 / mae 1.170 / signed +0.260 |
| oracle_guided | over 0.150 / under 0.000 / mae 0.170 / signed +0.170 | over 0.370 / under 0.065 / mae 0.560 / signed +0.430 | over 0.525 / under 0.120 / mae 0.920 / signed +0.680 |
| unguided_capped | over 0.280 / under 0.010 / mae 0.420 / signed +0.400 | over 0.290 / under 0.195 / mae 0.665 / signed +0.275 | over 0.300 / under 0.400 / mae 0.975 / signed +0.065 |

hop_count_exact_match_rate by hop: unguided 0.71 / 0.515 / 0.30 (hop 2/3/4);
oracle_guided 0.85 / 0.565 / 0.355; unguided_capped 0.71 / 0.515 / 0.30 (identical
to unguided at hop 2/3, only hop=4's directional error narrows under the cap).

Full per-hop metrics: `experiments/exp-004/metrics.json` -> `eval_metrics.<condition>.per_gold_hop_metrics`.

## Pairwise comparisons (--compare, paired bootstrap 95% CI + McNemar + paired t-test, n=600 aligned items)

**unguided vs oracle_guided** (`experiments/exp-004/metrics.json` -> `comparisons.unguided_vs_oracle_guided`):
- step_f1: -0.0022, CI [-0.0129, +0.0085], not significant; t=-0.41, p=0.686 — not significant
- ordered_step_accuracy: -0.0048, CI [-0.0155, +0.0061], not significant; t=-0.88, p=0.378 — not significant
- rouge_l_f1: -0.0084, CI [-0.0165, -0.0004], significant; t=-2.03, p=0.042 — significant
- composite_score: -0.2098, CI [-0.2189, -0.0058], significant; t not computed for composite (only exact/hop-EM get McNemar)
- exact_match (McNemar): +0.0050, p=0.581, not significant; t=+0.83, p=0.406 — not significant
- hop_count_exact_match (McNemar): -0.0817, p=0.000241, significant; t=-3.77, p=0.000182 — significant

**unguided vs unguided_capped**:
- All quality metrics indistinguishable (step_f1 diff 0.0000-0.0001, CIs straddling
  zero or degenerate at n=0 discordant pairs); composite_score: -0.0030, CI
  [-0.0072, -0.0004], significant (driven by the step-count-error term, not by
  step_f1/ordered_step_acc/hop-EM, all of which tie exactly). exact_match and
  hop_count_exact_match are literally identical between the two runs (0 discordant
  pairs, McNemar p=1, t-test degenerate/undefined).

**oracle_guided vs unguided_capped**:
- Mirrors unguided vs oracle_guided in sign (oracle_guided still ahead): step_f1
  +0.0022 not significant (CI [-0.0086,+0.0128]); ordered_step_acc +0.0047 not
  significant; composite_score +0.2068, CI [+0.0033,+0.2146], significant;
  hop_count_exact_match +0.0817, McNemar p=0.000241, significant, t=+3.77,
  p=0.000182 significant; exact_match not significant.

Full comparison JSONs (all 3 bootstrap stats, both McNemar rows, all 5 paired
t-test rows): `experiments/exp-004/metrics.json` -> `comparisons.*`.

## Cost per query (mean over 600 rows)

| condition | prompt tok | completion tok | total tok | latency (s) | rows at max_new_tokens |
|---|---|---|---|---|---|
| unguided | 420.4 | 79.3 | 499.8 | 1.879 | 23 |
| oracle_guided | 475.4 | 120.5 | 595.9 | 2.844 | 43 |
| unguided_capped | 420.4 | 77.4 | 497.9 | 1.857 | 22 |

## Reading against the issue #12 hypothesis

- Oracle hop count moves `hop_count_exact_match_rate` a lot (+0.082 overall,
  McNemar and paired-t both significant, p<0.001) and moves `composite_score`
  (driven by reference-validity + step-count-error terms), but does **not**
  move step-level quality (`step_f1_macro`, `ordered_step_accuracy_macro`) by a
  significant margin at n=600 — the CIs straddle zero and the paired t-tests are
  not significant on those two metrics. Read literally: oracle guidance helps the
  model land on the *right number* of steps more often, at higher token/latency
  cost (+19% total tokens, +51% latency), without a measured improvement in
  whether those steps are the *right* steps.
- `unguided_capped` vs `unguided` is statistically indistinguishable on every
  quality metric except a small composite-score shift attributable to the
  step-count-error term at hop=4 (mean_signed_step_count_error narrows from
  +0.26 to +0.065 there) — the step-line cap fired on only 12/600 rows (7 actually
  cut off), so it had limited room to close any gap. This does not by itself
  distinguish "runaway generation length" from "hop ignorance" as the dominant
  failure mode, because the capped condition barely differs from unguided at all
  on this run's cap setting (step-line cap 8).
- No router-design recommendation is made here; these are the measured numbers
  for issue #12's stated hypothesis. The router go/no-go decision is Jahid's/
  the supervisor's call.

## Reproducibility

- Code: run-time commit `6fc4bba`; evaluation commit `0068842` (no pipeline code
  changed in between — the only intervening commit gitignores an untracked
  personal note file). Config: `configs/decomposer_musique.json`,
  `configs/musique_eval.json`. Seed 42 throughout (generation and evaluator).
- Predictions (`results.json`, per-item JSONs) and raw per-query text stay on the
  box under `runs/exp-004/`; only configs and metrics are committed here.
