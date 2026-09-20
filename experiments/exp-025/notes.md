# exp-025 — G3c: Qwen2.5-14B-Instruct, unguided, 4-bit, **reversed** exemplar order, ADR 0007 pinned 600

Issue #50 GPU lane, run G3c (ADR 0034 decision 1; ADR 0032 item 3 and its 2026-09-20 amendment).
Ran 2026-09-20, 05:16–05:43 +03. Third and last G3 arm.

## What was run

exp-022's command sequence, unchanged except for `--ordering reversed` and the output root.
Condition `unguided`, seed 42, `--quantization 4bit`, k=5 exemplars from
`runs/exp-014/retrieval/s2_pinned600_imbalanced_rerank_top5.jsonl` (sha256 `3f95c7dc…fdb0`).
Configs unedited. Then the evaluator on the 600 results, then **one** `--compare`, against exp-022
and nothing else. rc=0 at every step.

## Launch gate — 7/7 PASS, checked twice

| precondition | observed |
|---|---|
| retrieval `input_sha256` | `3f95c7dc18c14fc6bea3473f1be5c0d7fb21b7b2f2b2a1175dbfaebb5092fdb0` |
| `quantization` | `"4bit"` |
| `few_shot_ordering` | `"reversed"` (source: `--ordering`) |
| `evaluation_set.pinned` | `true` — 600 rows, 200/200/200, 0 missing, 0 unexpected |
| `model_size.ceiling_asserted` | `true` — 14,770,033,664 / 15,000,000,000 (**98.5%**) |
| `logging.anchor_ids_file` | `configs/anchor_set_45.json`, sha256 `c03aa351…7111`, 45 ids |
| `anchor_prompts_logged` | 45 of 45 expected |

**Size-cap flag.** This arm loads only because `decomposer_max_params` is 1.5e10 (ADR 0031 item 4,
confirmed by Jahid in ADR 0032 item 2), still *pending supervisor confirmation*, while
`default_max_params` stays at the supervisor's standing 8e9. If he reasserts 8B or 10B, this arm is
refused at load and the result is withdrawn — the same caveat exp-022 carries.

## Ordering invariant, verified on the artifacts — 120/120

120 positional prompt files shared with exp-022; **120/120 identical exemplar set**, **120/120
exactly reversed order** (`runs/check_ordering_invariant.py`).

## Report card (n=600; break_EM ⇑ / SARI ⇑ / GED ⇓ / chain_validity ⇑ / hop_EM ⇑ / under ⇓ / over ⇓)

| term | exp-025 reversed | exp-022 as_is |
|---|---|---|
| break_exact_match_rate | 0.0333 | 0.0333 |
| sari_macro | 0.5832 | 0.5795 |
| ged_macro | 0.4670 | 0.4605 |
| chain_validity_macro | 0.9767 | 0.9838 |
| hop_count_exact_match_rate | 0.5967 | 0.6167 |
| under_decomposition_rate | 0.1150 | 0.1017 |
| over_decomposition_rate | 0.2883 | 0.2817 |

Beside the suite: exact_match 0.0417, step_f1 0.1973, ordered_step_accuracy 0.1744, rouge_l_f1
0.5473, step_count_mae 0.5267, mean signed step-count error +0.280, decomp_mean 0.5446, legacy
composite 0.4137. 600/600 evaluated, 0 missing gold, 0 empty decompositions, **0 rows at
`max_new_tokens`**, 1,968 step lines, all 600 few-shot sets from the reranked artifact. Per gold
hop, chain_validity 0.9700 / 0.9700 / 0.9901 and hop_EM 0.795 / 0.510 / 0.485 at 2 / 3 / 4 hops —
the near-uniform chaining exp-022 showed survives the reversal.

## The compare — exp-025 vs exp-022, n=600: **0 of 13 headline statistics significant**

All 8 bootstrap CIs straddle 0; all 5 McNemar tests are far from α; all 12 paired t-tests are
non-significant (smallest p = 0.249).

| statistic | reversed | as_is | difference | test |
|---|---|---|---|---|
| hop_count_exact_match | 0.5967 | 0.6167 | −0.0200 | McNemar p=0.349 (63 vs 75) |
| under_decomposition | 0.1150 | 0.1017 | +0.0133 | McNemar p=0.312 (28 vs 20) |
| chain_validity | 0.9767 | 0.9838 | −0.0071 | CI −0.0200…+0.0053 |
| over_decomposition | 0.2883 | 0.2817 | +0.0067 | McNemar p=0.775 (57 vs 53) |
| ged | 0.4670 | 0.4605 | +0.0065 | CI −0.0079…+0.0209 |
| decomp_mean | 0.5446 | 0.5507 | −0.0060 | CI −0.0166…+0.0048 |
| sari | 0.5832 | 0.5795 | +0.0038 | CI −0.0054…+0.0131 |
| ordered_step_accuracy | 0.1744 | 0.1778 | −0.0033 | CI −0.0146…+0.0080 |
| exact_match | 0.0417 | 0.0450 | −0.0033 | McNemar p=0.727 (3 vs 5) |
| rouge_l_f1 | 0.5473 | 0.5453 | +0.0020 | CI −0.0063…+0.0102 |
| composite (legacy) | 0.4137 | 0.4150 | −0.0013 | CI −0.0097…+0.0070 |
| step_f1 | 0.1973 | 0.1969 | +0.0005 | CI −0.0114…+0.0122 |
| break_exact_match | 0.0333 | 0.0333 | ±0.0000 | McNemar p=1.0 (5 vs 5) |

## The one-line answer

**For Qwen2.5-14B-Instruct, reversing the exemplar order does not change the report card.** Nothing
of thirteen is significant; the largest movement is 2.0 pp (hop_count_EM, slightly favouring
`as_is`) and every other term moves by ≤1.3 pp. Against the supervisor's *"it can change up to
10%"*, the measured effect at the largest model the ceiling admits is again an order of magnitude
smaller than that and indistinguishable from zero at n=600. Capacity does not turn the ordering
knob on: the 14B is as insensitive to exemplar order as the two 7B models.

## Cost

Wall clock 1,624 s total (27 min 4 s), of which decompose 1,617 s; `model.generate` 1,602.2 s over
600 queries (mean latency 2.670 s, median 2.582 s; mean 349.0 prompt / 33.9 completion tokens).
Peak VRAM **10,416 MiB** — identical to exp-022. Generation was 2.8% faster than exp-022's
1,648.0 s; one unreplicated timing pair, no effect claimed.

## Files

`runs/exp-025/` (gitignored): `decomposer/20260920_021624/` (results, config, metrics, 120
positional + 45 anchor prompts), `eval/`, `compare_vs_exp022_as_is/`, `run.log`. Committed here:
`config.json` (the run's own snapshot, copied verbatim), `metrics.json`, this note.
