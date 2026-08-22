# exp-008 — Fine-tuned decomposer (LoRA, full_train) vs prompting (issue #13)

Inference-only run of the exp-001 `full_train` LoRA adapter (QLoRA r=16 on
`mistralai/Mistral-7B-Instruct-v0.3`, trained 2026-08-19) on the ADR 0007
pinned 600 (200/hop). No training happened in this experiment. Command:

```
components/decomposer/run_decomposer.py --model mistral_7b_instruct \
  --config decomposer_musique.json --condition unguided \
  --adapter runs/exp-001/20260818_222259/adapter --no-few-shot \
  --output-root runs/exp-008
```

`--condition unguided` renders the same prompt shape the adapter was trained
on (guided=false, `prompt_unguided.md`, plain style); `--no-few-shot` empties
the few-shot block to match training. Dry-run and real-run both reported
`adapter_prompt_parity.mismatches: []` against the training record at
`runs/exp-001/20260818_222259/config.json`, and `assert_adapter_base_model`
confirmed both name `mistralai/Mistral-7B-Instruct-v0.3`. Base+adapter
parameter count: 7,289,966,592 (72.9% of the 10B ceiling, ADR 0015).

Run dir: `runs/exp-008/20260822_100745`. rc=0, 600/600 rows, 0 rows at
max_new_tokens. Launched in tmux (`runs/exp-008/run.log`), GPU free at launch
(confirmed via `nvidia-smi`), completed in ~832s of generation.

## Headline (overall, n=600)

| arm | composite | step_f1 | ordered_step_acc | hop_count_EM | exact_match | step_count_MAE | mean_signed_err | over_rate | under_rate |
|---|---|---|---|---|---|---|---|---|---|
| full_train (fine-tuned) | 0.5236 | 0.3411 | 0.3237 | 0.7100 | 0.1083 | 0.300 | -0.233 | 0.033 | 0.257 |
| unguided (prompting baseline, exp-004) | 0.2098 | 0.2039 | 0.1804 | 0.5083 | 0.0617 | 0.775 | +0.335 | 0.290 | 0.202 |
| oracle_guided (exp-004) | 0.4197 | 0.2061 | 0.1852 | 0.5900 | 0.0567 | 0.550 | +0.427 | 0.348 | 0.062 |
| unguided_capped (exp-004) | 0.2128 | 0.2040 | 0.1805 | 0.5083 | 0.0617 | 0.687 | +0.247 | 0.290 | 0.202 |

Note the fine-tuned arm's error is directionally different from all three
prompting arms: it under-decomposes (mean_signed -0.233, under_rate 0.257,
over_rate only 0.033), where every prompting arm over-decomposes on average.

## Per gold-hop-depth, full_train

| hop | n | step_f1 | hop_count_EM | over_rate | under_rate | step_count_MAE | mean_signed_error |
|---|---|---|---|---|---|---|---|
| 2 | 200 | 0.4810 | 0.9750 | 0.020 | 0.005 | 0.025 | +0.015 |
| 3 | 200 | 0.2930 | 0.7450 | 0.040 | 0.215 | 0.255 | -0.175 |
| 4 | 200 | 0.2494 | 0.4100 | 0.040 | 0.550 | 0.620 | -0.540 |

Full per-hop metrics for all four arms: `experiments/exp-008/metrics.json` ->
`eval_metrics.per_gold_hop_metrics` (full_train) and
`experiments/exp-004/metrics.json` -> `eval_metrics.<condition>.per_gold_hop_metrics`
(the three prompting arms).

## Cost per query (mean over 600 rows)

| arm | prompt tok | completion tok | total tok | latency (s) |
|---|---|---|---|---|
| full_train (fine-tuned) | 122.5 | 29.4 | 151.9 | 1.387 |
| unguided (prompting baseline) | 420.4 | 79.3 | 499.8 | 1.879 |
| oracle_guided | 475.4 | 120.5 | 595.9 | 2.844 |
| unguided_capped | 420.4 | 77.4 | 497.9 | 1.857 |

The fine-tuned arm's prompt is ~3.4x shorter than any prompting arm's (no
few-shot block, no retrieval context needed at inference), for a 70% cut in
total tokens/query and a 26% cut in latency/query versus the unguided
baseline.

## Phase C comparison (`scripts/compare_decomposer_arms.py --eval-arm full_train`)

Baseline: `unguided` (the deployable dynamic-prompting condition — brief's
choice, `configs/finetune_decomposer.json` does not designate a different
baseline protocol for this comparison). `oracle_guided` and `unguided_capped`
are included and labeled as they use gold hop counts unavailable at
inference / a step-line cap, respectively — not the deployable baseline.
`--eval-arm full_train` enforced eval_hops [2,3,4] on every side (asserted:
600 items each, 200/200/200 per hop, all four arms). Full command output and
per-arm cost table: `experiments/exp-008/metrics.json` -> `arm_comparison`.

**full_train vs unguided** (n=600 aligned items):
- step_f1: +0.1372, CI [+0.1150, +0.1601], significant; t=+11.82, p=4.0e-29 — significant
- ordered_step_accuracy: +0.1433, CI [+0.1205, +0.1666], significant; t=+12.34, p=2.4e-31 — significant
- rouge_l_f1: +0.1218, CI [+0.1065, +0.1369], significant; t=+15.92, p=7.7e-48 — significant
- composite_score: +0.3137, CI [+0.1055, +0.3307], significant
- exact_match (McNemar): +0.0467, p=1.75e-04, significant; t=+3.85, p=1.29e-04 — significant
- hop_count_exact_match (McNemar): +0.2017, p=1.12e-15, significant; t=+8.33, p=5.5e-16 — significant

Every quality metric moves significantly in the fine-tuned arm's favour over
the deployable prompting baseline, at roughly 1/3 the token cost and lower
latency.

**oracle_guided vs unguided** (same numbers as exp-004's within-model
comparison, included here for context): step_f1/ordered_step_accuracy not
significant; composite and hop_count_exact_match significant. See
`experiments/exp-004/notes.md` for the full breakdown — oracle guidance does
not close the gap to the fine-tuned arm's step-level quality (full_train's
step_f1 0.3411 vs oracle_guided's 0.2061).

**unguided_capped vs unguided**: statistically indistinguishable (as in
exp-004) — the step-line cap is not a substitute for fine-tuning either.

## Reading against the issue #13 hypothesis

- On this evaluation set, LoRA fine-tuning on the full MuSiQue train split
  (19,938 examples, 2 epochs) beats the deployable few-shot dynamic-prompting
  baseline (`unguided`) by a significant margin on every quality metric
  measured here (step_f1, ordered_step_accuracy, rouge_l_f1, exact_match,
  hop_count_exact_match — all bootstrap CI excludes zero, McNemar p<0.001,
  paired t-test p<0.001), while costing about 30% of the tokens and ~74% of
  the latency per query. This also beats `oracle_guided`, which is not a
  deployable arm (it needs the gold hop count at inference) and which the
  fine-tuned arm still exceeds on step_f1 (0.3411 vs 0.2061) and
  ordered_step_accuracy (0.3237 vs 0.1852).
- The fine-tuned arm's error pattern is qualitatively different: it
  under-decomposes at higher hop depths (under_rate 0.550 at hop=4, mean
  signed error -0.540) where the prompting arms over-decompose. hop_count_EM
  degrades sharply with depth for the fine-tuned arm too (0.975 / 0.745 /
  0.410 at hop 2/3/4) — better than unguided prompting's 0.71/0.515/0.30 at
  every depth, but still far from solved at hop=4.
- This is a single evaluation (one adapter, one eval set, one seed) — it
  measures the `full_train` arm specifically; `pool_2000` remains blocked on
  the best-pool decision and `generalisation_2_3hop` is untrained, so this
  does not generalize to those arms. No deployment recommendation is made
  here; that is Jahid's/the supervisor's call.

## Reproducibility

- Code: commit `374f862` (log entry) / launch and eval both ran at this SHA
  (working tree clean; no pipeline code changed during this experiment).
  Config: `configs/decomposer_musique.json`, `configs/finetune_decomposer.json`
  arm `full_train`, `configs/musique_eval.json`. Seed 42 throughout.
- Predictions (`results.json`, per-item JSONs) stay on the box under
  `runs/exp-008/`; only configs and metrics are committed here.
- `runs/run.lock` released after this entry is committed.
