# exp-023 — G3a: Mistral-7B-Instruct-v0.3, unguided, 4-bit, **reversed** exemplar order, ADR 0007 pinned 600

Issue #50 GPU lane, run G3a (ADR 0034 decision 1; ADR 0032 item 3 and its 2026-09-20 amendment).
Ran 2026-09-20, 04:17–04:56 +03.

## What was run

exp-020's command sequence, unchanged except for `--ordering reversed` and the output root.
Condition `unguided`, seed 42, `--quantization 4bit`, k=5 exemplars from
`runs/exp-014/retrieval/s2_pinned600_imbalanced_rerank_top5.jsonl` (sha256 `3f95c7dc…fdb0`,
verified before launch). Configs unedited. Then the evaluator on the 600 results, then **one**
`--compare`, against exp-020 and nothing else: against any other arm more than one factor would
move and it would not be an ordering reading. rc=0 at every step.

**The ordering contrast moves one factor.** Against exp-020 this arm holds the model, the
condition, the precision, the eval set, the retrieval artifact (hence the same five exemplars per
query), the prompt derivation, the seed and the decoding. Only the order the five exemplars are
*presented* in differs: lowest-similarity-first instead of highest-similarity-first.

## Launch gate — 7/7 PASS, checked twice

Checked from the startup dump in `run.log` 10 s after launch, before a token was generated and with
a kill wired to a mismatch, and again from the written `config.json` / `metrics.json` before the
evaluator was allowed to run (`runs/precheck_g3_arm.py`, the predecessor's A1 gate with the
expected ordering changed to `reversed` and two checks added).

| precondition | observed |
|---|---|
| retrieval `input_sha256` | `3f95c7dc18c14fc6bea3473f1be5c0d7fb21b7b2f2b2a1175dbfaebb5092fdb0` |
| `quantization` | `"4bit"` |
| `few_shot_ordering` | `"reversed"` (source: `--ordering`) |
| `evaluation_set.pinned` | `true` — 600 rows, 200/200/200, 0 missing, 0 unexpected |
| `model_size.ceiling_asserted` | `true` — 7,248,023,552 / 15,000,000,000 (48.3%) |
| `logging.anchor_ids_file` | `configs/anchor_set_45.json`, sha256 `c03aa351…7111`, 45 ids |
| `anchor_prompts_logged` | 45 of 45 expected (first arm in the block to log them natively) |

The gate was exercised both ways before launch: it passes the ordering and anchor checks on a
3-row CPU dry run with `--ordering reversed`, and fails them on exp-022's run log (`as_is`, no
anchor block).

## Ordering invariant, verified on the artifacts — 120/120

ADR 0032's amendment says presentation order is applied at one funnel and selection is untouched,
so the two arms must show the *same* five exemplars. That is checked rather than trusted
(`runs/check_ordering_invariant.py`): for each of the **120** positional prompt files exp-023 and
exp-020 share by name, the exemplar questions were pulled out of the prompt body in the order they
appear.

- checked: **120** of 120
- identical exemplar set: **120/120**
- exactly reversed order: **120/120**

Sanity control: the same checker on exp-021 vs exp-022 (two `as_is` arms over the same artifact)
reports 120/120 identical sets and **0/120** reversed, which is the correct reading there.

## Report card (n=600; break_EM ⇑ / SARI ⇑ / GED ⇓ / chain_validity ⇑ / hop_EM ⇑ / under ⇓ / over ⇓)

| term | exp-023 reversed | exp-020 as_is |
|---|---|---|
| break_exact_match_rate | 0.0483 | 0.0400 |
| sari_macro | 0.5831 | 0.5766 |
| ged_macro | 0.4813 | 0.4846 |
| chain_validity_macro | 0.9629 | 0.9626 |
| hop_count_exact_match_rate | 0.5267 | 0.5117 |
| under_decomposition_rate | 0.1783 | 0.1683 |
| over_decomposition_rate | 0.2950 | 0.3200 |

Beside the suite: exact_match 0.0550, step_f1 0.1958, ordered_step_accuracy 0.1708, rouge_l_f1
0.5413, step_count_mae 1.1267, mean signed step-count error +0.7133, decomp_mean 0.5280, legacy
composite 0.1920 (not headlined, issue #40). 600/600 evaluated, 0 missing gold, 0 empty
decompositions, **22 rows at `max_new_tokens`** (exp-020 is the comparator for that too — see the
run metrics), 2,228 step lines, all 600 few-shot sets from the reranked artifact.

## The compare — exp-023 vs exp-020, n=600: **0 of 13 headline statistics significant**

Nothing reaches significance: 8 bootstrap CIs all straddle 0 and all 5 McNemar tests are far from
α. The 12 paired t-tests reported alongside are likewise all non-significant (smallest p = 0.061,
step_f1).

| statistic | reversed | as_is | difference | test |
|---|---|---|---|---|
| over_decomposition | 0.2950 | 0.3200 | **−0.0250** | McNemar p=0.228 (60 vs 75) |
| hop_count_exact_match | 0.5267 | 0.5117 | +0.0150 | McNemar p=0.531 (86 vs 77) |
| step_f1 | 0.1958 | 0.1823 | +0.0134 | CI −0.0003…+0.0277 |
| exact_match | 0.0550 | 0.0450 | +0.0100 | McNemar p=0.238 (12 vs 6) |
| under_decomposition | 0.1783 | 0.1683 | +0.0100 | McNemar p=0.556 (39 vs 33) |
| break_exact_match | 0.0483 | 0.0400 | +0.0083 | McNemar p=0.302 (10 vs 5) |
| ordered_step_accuracy | 0.1708 | 0.1632 | +0.0076 | CI −0.0056…+0.0216 |
| decomp_mean | 0.5280 | 0.5213 | +0.0067 | CI −0.0052…+0.0194 |
| sari | 0.5831 | 0.5766 | +0.0065 | CI −0.0018…+0.0153 |
| ged | 0.4813 | 0.4846 | −0.0034 | CI −0.0193…+0.0127 |
| rouge_l_f1 | 0.5413 | 0.5385 | +0.0028 | CI −0.0067…+0.0124 |
| chain_validity | 0.9629 | 0.9626 | +0.0003 | CI −0.0144…+0.0151 |
| composite (legacy) | 0.1920 | 0.1898 | +0.0022 | CI −0.0238…+0.2016 |

## The one-line answer

**For Mistral-7B-Instruct-v0.3 on this eval set, reversing the exemplar order does not change the
report card.** No term moves significantly; the largest report-card movement is 2.5 percentage
points (over-decomposition, and in the direction that favours the reversed arm), and every other
term moves by ≤1.5 pp. Against the supervisor's *"it can change up to 10%"*, the measured effect on
this model is an order of magnitude smaller than that and statistically indistinguishable from
zero at n=600. This is one model's answer; exp-024 and exp-025 ask the same question of Qwen2.5-7B
and Qwen2.5-14B, whose failure modes differ.

## Cost

Wall clock 2,324 s total (38 min 44 s), of which decompose 2,316 s; `model.generate` 2,304.0 s over
600 queries (mean latency 3.840 s, median 1.719 s; mean 395.2 prompt / 80.4 completion tokens).
Peak VRAM **4,818 MiB**, sampled from `nvidia-smi` every 5 s filtered to this run's own pid — the
same figure exp-020 measured, as expected for the same weights at the same precision. The
generation is 6.7% *faster* than exp-020's 2,469.5 s; that is one unreplicated timing pair on a
shared box and is not offered as an ordering effect.

## Files

`runs/exp-023/` (gitignored): `decomposer/20260920_011726/` (results, config, metrics, 120
positional + 45 anchor prompts), `eval/`, `compare_vs_exp020_as_is/`, `run.log`. Committed here:
`config.json` (the run's own snapshot, copied verbatim), `metrics.json`, this note.
