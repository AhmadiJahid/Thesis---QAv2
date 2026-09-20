# exp-024 — G3b: Qwen2.5-7B-Instruct, unguided, 4-bit, **reversed** exemplar order, ADR 0007 pinned 600

Issue #50 GPU lane, run G3b (ADR 0034 decision 1; ADR 0032 item 3 and its 2026-09-20 amendment).
Ran 2026-09-20, 04:59–05:13 +03.

## What was run

exp-021's command sequence, unchanged except for `--ordering reversed` and the output root.
Condition `unguided`, seed 42, `--quantization 4bit`, k=5 exemplars from
`runs/exp-014/retrieval/s2_pinned600_imbalanced_rerank_top5.jsonl` (sha256 `3f95c7dc…fdb0`).
Configs unedited. Then the evaluator on the 600 results, then **one** `--compare`, against exp-021
and nothing else. rc=0 at every step.

**The ordering contrast moves one factor**: model, condition, precision, eval set, retrieval
artifact (hence the same five exemplars per query), prompt derivation, seed and decoding are all
held; only the presentation order is inverted to lowest-similarity-first.

## Launch gate — 7/7 PASS, checked twice

| precondition | observed |
|---|---|
| retrieval `input_sha256` | `3f95c7dc18c14fc6bea3473f1be5c0d7fb21b7b2f2b2a1175dbfaebb5092fdb0` |
| `quantization` | `"4bit"` |
| `few_shot_ordering` | `"reversed"` (source: `--ordering`) |
| `evaluation_set.pinned` | `true` — 600 rows, 200/200/200, 0 missing, 0 unexpected |
| `model_size.ceiling_asserted` | `true` — 7,615,616,512 / 15,000,000,000 (50.8%) |
| `logging.anchor_ids_file` | `configs/anchor_set_45.json`, sha256 `c03aa351…7111`, 45 ids |
| `anchor_prompts_logged` | 45 of 45 expected |

## Ordering invariant, verified on the artifacts — 120/120

120 positional prompt files shared with exp-021; **120/120 identical exemplar set**, **120/120
exactly reversed order** (`runs/check_ordering_invariant.py`).

## Report card (n=600; break_EM ⇑ / SARI ⇑ / GED ⇓ / chain_validity ⇑ / hop_EM ⇑ / under ⇓ / over ⇓)

| term | exp-024 reversed | exp-021 as_is |
|---|---|---|
| break_exact_match_rate | 0.0583 | 0.0550 |
| sari_macro | 0.5771 | 0.5720 |
| ged_macro | 0.4728 | 0.4694 |
| chain_validity_macro | 0.9282 | 0.9210 |
| hop_count_exact_match_rate | 0.5400 | 0.5533 |
| under_decomposition_rate | 0.2900 | 0.2933 |
| over_decomposition_rate | 0.1700 | 0.1533 |

Beside the suite: exact_match 0.0700, step_f1 0.2081, ordered_step_accuracy 0.1905, rouge_l_f1
0.5431, step_count_mae 0.6000, mean signed step-count error −0.1267 (exp-021: −0.150 — this arm
still under-shoots on length, the Qwen2.5-7B failure mode exp-021 found), decomp_mean 0.5262,
legacy composite 0.4204. 600/600 evaluated, 0 missing gold, 0 empty decompositions, **0 rows at
`max_new_tokens`**, 1,724 step lines, all 600 few-shot sets from the reranked artifact. Per gold
hop, chain_validity 0.8825 / 0.9500 / 0.9522 and hop_EM 0.820 / 0.470 / 0.330 at 2 / 3 / 4 hops —
the 2-hop chain-validity dip exp-021 showed (0.8775) is still there at 0.8825.

## The compare — exp-024 vs exp-021, n=600: **1 of 13 headline statistics significant**

| statistic | reversed | as_is | difference | test |
|---|---|---|---|---|
| **step_f1** | **0.2081** | **0.1929** | **+0.0152** | **bootstrap CI +0.0013…+0.0292, favours reversed** (t-test p=0.032) |
| over_decomposition | 0.1700 | 0.1533 | +0.0167 | McNemar p=0.326 (47 vs 37) |
| hop_count_exact_match | 0.5400 | 0.5533 | −0.0133 | McNemar p=0.573 (73 vs 81) |
| ordered_step_accuracy | 0.1905 | 0.1771 | +0.0133 | CI −0.0005…+0.0271 |
| chain_validity | 0.9282 | 0.9210 | +0.0072 | CI −0.0157…+0.0304 |
| exact_match | 0.0700 | 0.0633 | +0.0067 | McNemar p=0.541 (14 vs 10) |
| sari | 0.5771 | 0.5720 | +0.0051 | CI −0.0047…+0.0146 |
| ged | 0.4728 | 0.4694 | +0.0034 | CI −0.0125…+0.0200 |
| break_exact_match | 0.0583 | 0.0550 | +0.0033 | McNemar p=0.824 (11 vs 9) |
| under_decomposition | 0.2900 | 0.2933 | −0.0033 | McNemar p=0.927 (58 vs 60) |
| rouge_l_f1 | 0.5431 | 0.5412 | +0.0019 | CI −0.0087…+0.0122 |
| decomp_mean | 0.5262 | 0.5264 | −0.0003 | CI −0.0143…+0.0135 |
| composite (legacy) | 0.4204 | 0.4111 | +0.0093 | CI −0.0009…+0.0195 |

**The one significant statistic is not a report-card term.** The seven-term decomposition report
card (break_EM, SARI, GED, chain_validity, hop_EM, under, over) is **0 of 7** significant; step_f1
is a step-overlap term reported beside the suite. No correction for multiple comparisons is applied
to the 13 (ADR 0009), and one marginal hit out of thirteen at α=0.05 with CI lower bound +0.0013 is
the weakest form the evidence can take while still being called significant. It is recorded as
what it is: a small positive step-overlap difference favouring the reversed order, on one model.

## The one-line answer

**For Qwen2.5-7B-Instruct, reversing the exemplar order leaves the report card unchanged and moves
one non-report-card statistic by 1.5 points.** step_f1 +0.0152 (CI +0.0013…+0.0292) is the only
significant result of thirteen; all seven report-card terms are non-significant and every one of
them moves by ≤1.7 pp. Against the supervisor's *"it can change up to 10%"*, the measured effect on
this model is ~1.5 points on one step-overlap term and nothing on the report card — far below 10%.

## Cost

Wall clock 851 s total (14 min 11 s), of which decompose 844 s; `model.generate` 832.3 s over 600
queries (mean latency 1.387 s, median 1.329 s; mean 349.0 prompt / 28.8 completion tokens). Peak
VRAM **6,082 MiB** — identical to exp-021, as expected for the same weights at the same precision.
Generation was 2.4% slower than exp-021's 812.9 s; a single unreplicated timing pair, no effect
claimed.

## Files

`runs/exp-024/` (gitignored): `decomposer/20260920_015937/` (results, config, metrics, 120
positional + 45 anchor prompts), `eval/`, `compare_vs_exp021_as_is/`, `run.log`. Committed here:
`config.json` (the run's own snapshot, copied verbatim), `metrics.json`, this note.
