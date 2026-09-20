# exp-028 — Qwen3.5-9B, unguided, 4-bit, **reversed** exemplar order, ADR 0007 pinned 600

Issue #50 GPU lane, the fourth G3 arm (ADR 0032 item 3 and its 2026-09-20 amendment; ADR 0034
decision 1, extended to the model Jahid added). Ran 2026-09-20, 13:54–14:19 +03.
Launch SHA `b9c0c3f`.

## What was run

exp-027's command sequence with `--ordering reversed` and a new output root, nothing else:

```
.venv/bin/python components/decomposer/run_decomposer.py --model qwen3_5_9b \
  --config decomposer_musique.json --condition unguided --seed 42 --quantization 4bit \
  --ordering reversed \
  --retrieval-input runs/exp-014/retrieval/s2_pinned600_imbalanced_rerank_top5.jsonl \
  --output-root runs/exp-028/decomposer
```

then the evaluator on the 600 results, then **one** `--compare`, against exp-027 and nothing else.
Configs unedited, model folder unedited. rc=0 at every step.

## Launch gate — 7/7 PASS, checked twice

| precondition | observed |
|---|---|
| retrieval `input_sha256` | `3f95c7dc18c14fc6bea3473f1be5c0d7fb21b7b2f2b2a1175dbfaebb5092fdb0` |
| `quantization` | `"4bit"` |
| `few_shot_ordering` | `"reversed"` (source: `--ordering`) |
| `evaluation_set.pinned` | `true` — 600 rows, 200/200/200, 0 missing, 0 unexpected |
| `model_size.ceiling_asserted` | `true` — 8,953,803,264 / 15,000,000,000 (59.7%) |
| `logging.anchor_ids_file` | `configs/anchor_set_45.json`, sha256 `c03aa351…7111`, 45 ids |
| `anchor_prompts_logged` | 45 of 45 expected |

## Ordering invariant, verified on the artifacts — 120/120

120 positional prompt files shared with exp-027; **120/120 identical exemplar set**, **120/120
exactly reversed order** (`runs/check_ordering_invariant.py`). Both sides are real runs of the same
`chat_template` folder, so the logged prompt bodies are directly comparable — no dry-run
regeneration is involved anywhere in this pair. Mean prompt tokens per query is 375.018 in both
arms, to three decimals, which is what the same exemplars in a different order should cost.

## Cost

| | exp-028 reversed | exp-027 as_is |
|---|---|---|
| wall clock, total | **1,508 s** (25 min 8 s) | 1,685 s |
| decompose | 1,501 s | 1,672 s |
| `model.generate` over 600 | 1,483.9 s (mean 2.473 s, median 2.300 s) | 1,655.1 s (mean 2.759 s) |
| tokens per query | 375.0 prompt / 34.4 completion | 375.0 / 38.4 |
| peak VRAM | **8,024 MiB** | 8,026 MiB |

## Report card (n=600; break_EM ⇑ / SARI ⇑ / GED ⇓ / chain_validity ⇑ / hop_EM ⇑ / under ⇓ / over ⇓)

| term | exp-028 reversed | exp-027 as_is |
|---|---|---|
| break_exact_match_rate | 0.0517 | 0.0550 |
| sari_macro | 0.6023 | 0.5984 |
| ged_macro | **0.4645** | 0.4838 |
| chain_validity_macro | **0.8821** | 0.8471 |
| hop_count_exact_match_rate | 0.5717 | 0.5700 |
| under_decomposition_rate | 0.1383 | 0.1350 |
| over_decomposition_rate | 0.2900 | 0.2950 |

Beside the suite: exact_match_rate 0.0600 (36/600), step_f1_macro 0.2207, ordered_step_accuracy
0.2000, rouge_l_f1_macro 0.5700, step_precision 0.2165, step_recall 0.2279, step_count_exact_rate
0.5717, step_count_mae 0.5700 (exp-027: 0.9850), mean_signed_step_count_error **+0.2800**
(exp-027: +0.6917), decomp_mean_macro 0.5291, legacy composite_score 0.4293.

Predicted hop distribution `{2:191, 3:182, 4:131, 5:61, 6:23, 7:5, 1:3, 8:2, 9:1, 10:1}` — **no
runaway tail**: 0 rows at `max_new_tokens`, `ged_fallback_counts {}`, longest plan 10 steps, and
0 empty decompositions, against exp-027's 2 / `{node_cap: 2}` / 141 / 1.

Per gold hop, chain_validity 0.955 / 0.892 / 0.800 and hop_EM 0.780 / 0.510 / 0.425 at 2 / 3 / 4
hops. The monotone-with-depth shape exp-027 showed survives the reversal; the gain is spread
across depths, largest at 3 hops (0.892 vs 0.821).

## The ordering contrast — 3 of 13 significant, and 2 of them are report-card terms

`--compare runs/exp-028/eval/eval_per_item.json runs/exp-027/eval/eval_per_item.json`, n=600
aligned, `statistics_not_available_in_inputs` empty. All three favour the **reversed** arm:

| statistic | difference | evidence |
|---|---|---|
| chain_validity | **+0.0349** | 0.8821 vs 0.8471, bootstrap CI **+0.0041…+0.0652**, t p=0.0258 |
| ged (lower is better) | **−0.0194** | 0.4645 vs 0.4838, CI **−0.0358…−0.0031**, t p=0.0228 |
| legacy composite_score | +0.0176 | 0.4293 vs 0.4117, CI +0.0002…+0.0398 (not a report-card term, issue #40) |

Not significant (10), by absolute difference: decomp_mean +0.0111 (t p=0.0991), rouge_l_f1 +0.0076,
step_f1 +0.0072, exact_match −0.0050 (McNemar p=0.607, 6 vs 9), over_decomposition −0.0050
(p=0.848, 53 vs 56), sari +0.0039, break_EM −0.0033 (p=0.774, 5 vs 7), under_decomposition +0.0033
(p=0.896, 30 vs 28), ordered_step_accuracy +0.0028, hop_count_EM +0.0017 (p=1.0, 67 vs 66).

**Is it the two runaway rows?** No. Recomputing the two significant report-card means over the 598
rows that exclude `3hop1__691197_15840_36002` and `4hop3__387712_132409_146285_35031` —
descriptively, with no re-test — gives chain_validity **+0.0334** (0.8817 vs 0.8483) and ged
**−0.0188** (0.4633 vs 0.4821), against +0.0349 and −0.0194 on all 600. The differences survive
without those two items.

## Reading

**This is the first G3 arm where a report-card term moves.** exp-023 (Mistral-7B) came back 0/13,
exp-024 (Qwen2.5-7B) 1/13 on step_f1 only, exp-025 (Qwen2.5-14B) 0/13 — all three with 0 of 7
report-card terms significant. On Qwen3.5-9B, **2 of 7 report-card terms are significant**
(chain_validity +3.5 pp, GED −0.019), both favouring the reversed order, plus the legacy composite.

Two things to keep beside that. First, the **magnitude is still far below the supervisor's "up to
10%"**: 3.5 percentage points on chain_validity is the largest ordering effect measured anywhere in
this block, and it is about a third of his figure. Second, **no multiple-comparison correction is
applied across the thirteen statistics** (ADR 0009); the GED CI upper bound is −0.0031 and the
composite's lower bound is +0.0002, i.e. two of the three sit close to the boundary. chain_validity
is the most solid of them (CI +0.0041…+0.0652).

**Across all four G3 pairs, 39+13 = 52 statistics, 4 are significant** (exp-024's step_f1 and this
arm's three). Ordering is not nothing on this model, and it is nothing on the other three.

## Artifacts

`experiments/exp-028/{config.json,metrics.json,notes.md}`; heavy output under the gitignored
`runs/exp-028/`. Anchor sheet covering this arm: `runs/anchor/2026-09-20-all-arms-plus-9b.md`.
