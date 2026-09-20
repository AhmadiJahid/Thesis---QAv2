# exp-027 — Qwen3.5-9B, unguided, 4-bit, `as_is` exemplar order, ADR 0007 pinned 600

Issue #50 GPU lane, the arm Jahid added on 2026-09-20 by reversing ADR 0032 item 6
(*"I need you to run Qwen 9B parameter one. Why haven't you run that one?"*).
Ran 2026-09-20, 13:25–13:53 +03. Launch SHA `ebb5307`.

## What was run

exp-021/exp-022's A1 command sequence with only the model folder and output root changed:

```
.venv/bin/python components/decomposer/run_decomposer.py --model qwen3_5_9b \
  --config decomposer_musique.json --condition unguided --seed 42 --quantization 4bit \
  --retrieval-input runs/exp-014/retrieval/s2_pinned600_imbalanced_rerank_top5.jsonl \
  --output-root runs/exp-027/decomposer
```

then the evaluator on the 600 results, then **four** `--compare` runs, each in its own directory.
Configs unedited, including `components/decomposer/models/qwen3_5_9b/config.json`
(`trust_remote_code: true`, `chat_template.enable_thinking: false`, `post_process.strip_think:
true` all left as committed). rc=0 at every step.

## The confound, stated once and carried everywhere

ADR 0031 item 5: **Qwen3.5-9B fits neither one-factor contrast in this block.** Against
Qwen2.5-7B the model *generation* and the *parameter count* both move; against Qwen2.5-14B they
both move again in the other direction. Against Mistral-7B family, generation and size all move.
So every comparison below places this model on the block's axis and **none of them attributes a
difference to family or to size**. exp-005 (this model on the v1 9,156-example pool, unquantized)
is a different axis and is **not** a comparator here.

## Launch gate — 6/6 PASS, checked twice

| precondition | observed |
|---|---|
| retrieval `input_sha256` | `3f95c7dc18c14fc6bea3473f1be5c0d7fb21b7b2f2b2a1175dbfaebb5092fdb0` |
| `quantization` | `"4bit"` |
| `few_shot_ordering` | `"as_is"` (source: config) |
| `evaluation_set.pinned` | `true` — 600 rows, 200/200/200, 0 missing, 0 unexpected |
| `model_size.ceiling_asserted` | `true` — 8,953,803,264 / 15,000,000,000 (59.7%) |
| `logging.anchor_ids_file` | `configs/anchor_set_45.json`, sha256 `c03aa351…7111`, 45 ids |
| `anchor_prompts_logged` (artifacts pass) | 45 of 45 expected |

### The parameter count the gate had to be corrected to

The first launch (13:21 +03) was **killed by this gate at 115 s, before a token was generated**,
because it was armed with **9,653,104,368** — the figure `scripts/print_model_params.py --offline`
reads from safetensors metadata. The runtime counter `src/model_size.py` uses
(`model.num_parameters()` on the instantiated `Qwen3_5ForConditionalGeneration`) reports
**8,953,803,264**. The 699,301,104 difference is a **15-tensor `mtp` multi-token-prediction head
the checkpoint carries and the runtime model does not instantiate**. 8,953,803,264 is exactly what
`experiments/exp-005/metrics.json` recorded for this same `model_id` on 2026-08-21, so the gate was
re-armed with it and the identity check keeps its teeth. Nothing about the run changed — same
command, same config, same seed — and the aborted attempt produced no generations (its log is kept
at `runs/exp-027/run.log.aborted-param-count-gate`, its empty output directory deleted).
**The two counting methods disagreeing by 0.7B on a model that is being run against a ceiling is a
tooling observation the block should carry, not something this lane fixed.**

## Cost

| | |
|---|---|
| wall clock, total | **1,685 s** (28 min 5 s) |
| of which decompose | 1,672 s |
| `model.generate` over 600 queries | 1,655.1 s (mean 2.759 s, median 2.339 s) |
| tokens per query | mean 375.0 prompt / 38.4 completion |
| peak VRAM | **8,026 MiB** (nvidia-smi sampled every 5 s, this run's pid only) |

Beside the block, `model.generate` seconds over the same 600 and sampled peak VRAM:
exp-021 Qwen2.5-7B 812.9 s / 6,082 MiB, exp-022 Qwen2.5-14B 1,648.0 s / 10,416 MiB,
exp-020 Mistral-7B 2,469.5 s / 4,818 MiB. This arm sits between the two Qwen2.5 sizes on both.

## Run-level observations

- **600/600 rows evaluated, 0 missing gold.**
- **1 row with an empty decomposition** (`2hop__92590_43786`) — the only empty row anywhere in the
  block. The model emitted `Question: <the query>` before its steps and the committed
  `post_process.truncate_at: ["Question:"]` rule cut the whole output. Config untouched.
- **2 rows at `max_new_tokens`** (`3hop1__691197_15840_36002`, 141 step lines;
  `4hop3__387712_132409_146285_35031`, 91 step lines) — the same two rows drive
  `ged_fallback_counts {"node_cap": 2}`. A runaway tail of the Mistral kind, two orders rarer.
- **`<think>` handling: 0 of 600** raw generations contain `<think>`/`</think>`, so this folder's
  `strip_think: true` never fired; `enable_thinking: false` suppressed thinking at the template
  level. Measured by scanning `decomposition_raw`.
- 2,215 step lines total; all 600 few-shot sets came from the reranked artifact; 45/45 anchor
  prompts written natively.

## Report card (n=600; break_EM ⇑ / SARI ⇑ / GED ⇓ / chain_validity ⇑ / hop_EM ⇑ / under ⇓ / over ⇓)

| term | exp-027 Qwen3.5-9B | exp-021 Qwen2.5-7B | exp-022 Qwen2.5-14B | exp-020 Mistral-7B |
|---|---|---|---|---|
| break_exact_match_rate | **0.0550** | 0.0550 | 0.0333 | 0.0400 |
| sari_macro | **0.5984** | 0.5720 | 0.5795 | 0.5766 |
| ged_macro | **0.4838** | 0.4694 | 0.4605 | 0.4846 |
| chain_validity_macro | **0.8471** | 0.9210 | 0.9838 | 0.9626 |
| hop_count_exact_match_rate | **0.5700** | 0.5533 | 0.6167 | 0.5117 |
| under_decomposition_rate | **0.1350** | 0.2933 | 0.1017 | 0.1683 |
| over_decomposition_rate | **0.2950** | 0.1533 | 0.2817 | 0.3200 |

Beside the suite: exact_match_rate 0.0650 (39/600), step_f1_macro 0.2135, ordered_step_accuracy
0.1972, rouge_l_f1_macro 0.5625, step_precision 0.2091, step_recall 0.2207, step_count_exact_rate
0.5700, step_count_mae 0.9850, **mean_signed_step_count_error +0.6917**, decomp_mean_macro 0.5180,
legacy composite_score 0.4117 (not headlined, issue #40).

Predicted hop distribution `{2:192, 3:174, 4:132, 5:52, 6:24, 7:14, 1:4, 8:3, 9:1, 10:1, 91:1,
141:1, 0:1}`. Per gold hop, chain_validity 0.945 / 0.821 / 0.775 and hop_EM 0.800 / 0.510 / 0.400
at 2 / 3 / 4 hops — chaining degrades monotonically with depth, the opposite of exp-021's pattern
(2-hop dip).

## The four comparisons

**1. vs exp-021 Qwen2.5-7B (CONFOUNDED) — 7 of 13 significant.** Favouring Qwen3.5-9B:
under_decomposition −0.1583 (0.1350 vs 0.2933, McNemar p=7.5e-16, 26 vs 121), sari +0.0263,
rouge_l_f1 +0.0213, step_f1 +0.0206, ordered_step_accuracy +0.0201. Favouring Qwen2.5-7B:
over_decomposition +0.1417 (0.2950 vs 0.1533, p=6.0e-12, 121 vs 36), chain_validity −0.0739.
Not significant: hop_EM +0.0167 (p=0.535), ged +0.0144, decomp_mean −0.0084, exact_match +0.0017,
composite +0.0006, break_EM ±0.0000.

**2. vs exp-022 Qwen2.5-14B (CONFOUNDED) — 10 of 13 significant.** Favouring Qwen3.5-9B on every
n-gram/step-overlap term plus both exact-match terms: break_EM +0.0217, exact_match +0.0200,
ordered_step_accuracy +0.0194, sari +0.0189, rouge_l_f1 +0.0172, step_f1 +0.0166. Favouring
Qwen2.5-14B: chain_validity −0.1366 (0.8471 vs 0.9838), decomp_mean −0.0327, ged +0.0233,
under_decomposition +0.0333. Not significant: hop_EM −0.0467 (McNemar p=0.0547), over_decomposition
+0.0133, composite −0.0033.

**3. vs exp-020 Mistral-7B — 8 of 13 significant.** Favouring Qwen3.5-9B: composite +0.2219,
hop_EM +0.0583, ordered_step_accuracy +0.0340, step_f1 +0.0312, rouge_l_f1 +0.0240, sari +0.0218,
exact_match +0.0200. Favouring Mistral: chain_validity −0.1155. Not significant: under
−0.0333 (p=0.0798), over −0.0250, break_EM +0.0150 (p=0.0931), decomp_mean −0.0033, ged −0.0008.

**4. vs exp-008 `full_train` — 10 of 10 significant, every one favouring the fine-tuned arm**:
ged +0.1744, chain_validity −0.1470, hop_EM −0.1400, step_f1 −0.1276, ordered_step_accuracy
−0.1265, composite −0.1118, rouge_l_f1 −0.1022, sari −0.0839, break_EM −0.0467, exact_match
−0.0433. As for exp-015/020/021/022, the exp-011 comparator file predates the
over/under_decomposition and decomp_mean columns, so those three are **NOT COMPARED**;
descriptively, exp-027 under 0.1350 / over 0.2950 vs exp008_full_train under 0.2567 / over 0.0333.

## Reading

Qwen3.5-9B is **not uniformly better or worse than any arm in the block**. It is the best of the
four prompting arms on the n-gram and step-overlap terms (SARI 0.5984, ROUGE-L 0.5625, step_f1
0.2135 are all the highest measured here) and significantly ahead of the two Qwen2.5 arms on them.
It is the **worst of the four on chaining** by a wide margin (chain_validity 0.8471 against
0.9210 / 0.9626 / 0.9838), and that gap is significant against all three. Its length error flips
back to **over**-decomposition (+0.6917 mean signed error, the block's largest after Mistral's
+0.585), so against Qwen2.5-7B it wins on under and loses on over — the same opposite-direction
pattern exp-021 vs exp-020 showed, now on a third model. Under ADR 0017's asymmetry (over
tolerable, under not) this row does not adjudicate between them.

**Prompting still does not reach fine-tuning:** this is the fourth 10/10 against exp-008
`full_train` (exp-015, exp-020, exp-021, exp-022 — and now a fourth model generation). Changing
the prompting model does not close that gap.

**Every one of those readings is on a confounded axis where the Qwen2.5 arms are concerned** —
generation and size move together — so none of it separates "Qwen3.5 is a better generation" from
"9B is a better size". Saying which would need a Qwen3.5-7B or a Qwen2.5-9B, and neither is in
this block.

## Artifacts

`experiments/exp-027/{config.json,metrics.json,notes.md}`; heavy output under the gitignored
`runs/exp-027/`. Anchor sheet covering this arm: `runs/anchor/2026-09-20-all-arms-plus-9b.md`.
