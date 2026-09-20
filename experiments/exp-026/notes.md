# exp-026 — G2: the pool-only fine-tune (`pool_2000_imbalanced`), 4-bit QLoRA train + fp16 eval on the ADR 0007 pinned 600

Issue #50 GPU lane, run **G2** — supervisor item A2, issue #13; ADR 0034 decisions 2–3 and its
2026-09-20 amendment. Ran 2026-09-20, 05:45–06:28 +03.

**The supervisor's question, in his framing:** *"just let's see the effect of the pool"* — does a
decomposer trained on **only the 2,000-question pool** the prompting arms retrieve their exemplars
from land between prompting and full-split fine-tuning, and how far each way?

## The framing that binds every number below (ADR 0034 amendment)

- **vs exp-008 `full_train`:** training-set **identity, volume AND hop mix** move together —
  **2,000 vs 19,938 examples**, **250 vs 2,494 optimizer steps** (same 2 epochs, so ~1/10 the
  compute), pool hop mix **1461 / 413 / 126** (73% two-hop) instead of the split's. Everything
  else is held: same base model, same LoRA config and hyperparameters, same seed, same prompt,
  same inference command, same evaluation set.
- **vs exp-015:** Mistral prompting at **fp16** over the **same pool** — the precision-matched
  prompting comparator.
- **vs exp-020:** the block's **4-bit** Mistral prompting arm — **context only, precision differs**
  (4-bit generation vs this arm's fp16 generation).
- The clean same-size-different-draw control (`pool_2000`, seeded hop-stratified) has **never been
  trained** (issue #13) and is not in this block, so "2,000 examples" here is inseparable from
  "*these* 2,000 examples, in *this* hop mix".

## What was run

Step 1, **train** (4-bit QLoRA, exactly as exp-001 trained `full_train`):

```
.venv/bin/python components/decomposer/train_lora.py --arm pool_2000_imbalanced \
  --quantization 4bit --seed 42 --output-root runs/exp-026
```

Step 2, **evaluate** (exp-008's command, plus the explicit `--retrieval-input` the ADR 0034
amendment requires; fp16 base per ADR 0034 decision 3, `--no-few-shot` per ADR 0012 item 7):

```
.venv/bin/python components/decomposer/run_decomposer.py --model mistral_7b_instruct \
  --config decomposer_musique.json --condition unguided --seed 42 --quantization none \
  --adapter runs/exp-026/20260920_024535/adapter --no-few-shot \
  --retrieval-input runs/exp-014/retrieval/s2_pinned600_imbalanced_rerank_top5.jsonl \
  --output-root runs/exp-026/decomposer
```

Then the evaluator on the 600 results, then four `--compare`s (each in its own directory) and
`scripts/compare_decomposer_arms.py --eval-arm pool_2000_imbalanced`. Configs **unedited**; no
shared pipeline code touched by this lane. Seed 42 throughout. rc=0 at every step.

## Launch gates — train 7/7 then 8/8, eval 6/6 then 8/8

Training gate, read from the two `[finetune]` lines printed **before any weights load**, and
re-verified against the written `metrics.json`:

| precondition | observed |
|---|---|
| `training_source_sha256` | `cc3646625d759a1b58179b18c25f83d255db24f05acc08b666985af2e58d6646` |
| source rows / selected examples | 2000 / 2000 |
| hop counts | `{2: 1461, 3: 413, 4: 126}` |
| eval overlap | `overlap_count=0`, `asserted=true`, 2,000 training ids vs 600 eval ids |
| `quantization` | `"4bit"` |
| `arm` | `"pool_2000_imbalanced"` |
| ceiling asserted (base / base+LoRA) | 7,248,023,552 and 7,289,966,592 against 15,000,000,000 (48.3% / 48.6%) |

Evaluation gate, read from the startup dump before generation, and re-verified against the written
artifacts:

| precondition | observed |
|---|---|
| retrieval `input_sha256` | `3f95c7dc18c14fc6bea3473f1be5c0d7fb21b7b2f2b2a1175dbfaebb5092fdb0` |
| `quantization` | `"none"` |
| adapter + `no_few_shot` | `runs/exp-026/20260920_024535/adapter`, `true` |
| `adapter_prompt_parity` | `checked: true`, `override: false`, **0 mismatches**, **5 fields** compared against this adapter's own `training_provenance.json` (`guided`, `prompt_file`, `prompt_style`, `few_shot_examples_in_prompt`, `prompt_sha256`) — exp-008 could only compare 4, because exp-001's record predates `prompt_sha256` |
| `evaluation_set.pinned` | `true` — 600 rows, 200/200/200, 0 missing, 0 unexpected |
| `model_size.ceiling_asserted` | `true` — 7,289,966,592 / 15,000,000,000 (48.6%) |
| `anchor_prompts_logged` | 45 of 45 expected |
| `rows_with_empty_decomposition` | 0 |

## Training cost

250 optimizer steps (2 epochs × 2,000 examples, batch 1 × grad-accum 16), **1,568.8 s** (26 min
9 s), 6.3 s/step. Final `train_loss` 0.3575 (the run's mean over logged steps). Loss curve
endpoints: **1.62 at step 10 → 0.1361 at step 250**; mean token accuracy 0.6849 → 0.9583. Peak GPU
**5.035 GiB** by `torch.cuda.max_memory_allocated` (5,648 MiB by per-pid `nvidia-smi` sampling).
Trainable 41,943,040 / 7,289,966,592 parameters (0.5754%). For scale, exp-001's `full_train` took
15,497 s for 2,494 steps — this arm is **~10% of that compute**.

## Inference cost

854 s of wall clock for the step, of which `model.generate` **842.3 s** over 600 queries. Mean
1.404 s/query (median 1.358 s), mean 122.5 prompt + 29.8 completion = 152.3 tokens/query. Peak VRAM
**14,438 MiB** (fp16 base + adapter). 600/600 rows, 0 at `max_new_tokens`, 0 empty, 1,659 step
lines.

| arm | prompt tok | completion tok | total tok | latency (s) |
|---|---|---|---|---|
| exp-026 pool-only fine-tune | 122.5 | 29.8 | 152.3 | 1.404 |
| exp-008 `full_train` fine-tune | 122.5 | 29.4 | 151.9 | 1.387 |
| exp-015 prompting, fp16, same pool | 395.2 | 76.4 | 471.6 | 1.810 |

The two fine-tuned arms are indistinguishable in cost (the prompt is the same zero-shot string);
both cost about **32% of the prompting arm's tokens** and ~78% of its latency.

## Report card (n=600; break_EM ⇑ / SARI ⇑ / GED ⇓ / chain_validity ⇑ / hop_EM ⇑ / under ⇓ / over ⇓)

| term | exp-026 pool-only FT | exp-008 full_train FT | exp-015 prompting fp16 |
|---|---|---|---|
| break_exact_match_rate | 0.0967 | 0.1017 | 0.0350 |
| sari_macro | 0.6804 | 0.6823 | 0.5785 |
| ged_macro | 0.3191 | 0.3094 | 0.4714 |
| chain_validity_macro | 0.9750 | 0.9942 | 0.9694 |
| hop_count_exact_match_rate | 0.7033 | 0.7100 | 0.5400 |
| under_decomposition_rate | 0.2583 | 0.2567 | 0.2317 |
| over_decomposition_rate | 0.0383 | 0.0333 | 0.2283 |

Beside the suite: exact_match 0.1083, step_f1 0.3185, ordered_step_accuracy 0.2969, rouge_l_f1
0.6607, decomp_mean 0.6272, legacy composite 0.5059, step_count_mae 0.3183, mean signed step-count
error −0.235, reference_validity_micro 1.0. 600/600 evaluated, 0 missing gold.

Per gold hop (step_f1 / chain_validity / hop_EM / over / under / MAE / signed):

| hop | step_f1 | chain_validity | hop_EM | over | under | MAE | signed |
|---|---|---|---|---|---|---|---|
| 2 | 0.4495 | 0.9850 | 0.965 | 0.020 | 0.015 | 0.040 | +0.010 |
| 3 | 0.2899 | 0.9750 | 0.645 | 0.060 | 0.295 | 0.360 | −0.230 |
| 4 | 0.2162 | 0.9649 | 0.500 | 0.035 | 0.465 | 0.555 | −0.485 |

The pool-only adapter reproduces the **direction** of exp-008's error: it under-decomposes at depth
(under_rate 0.465 at hop 4) where every prompting arm over-decomposes. Note it is *less* extreme
than exp-008 at hop 4 (under 0.465 vs 0.550, hop_EM 0.500 vs 0.410) and weaker at hop 3 (hop_EM
0.645 vs 0.745) — the two do not differ significantly on hop_count_exact_match overall.

## The headline compare — exp-026 vs exp-008 `full_train`, n=600: **4 of 13 significant, all favouring full_train, all small**

Both sides re-scored by one evaluator invocation
(`runs/exp-026/compare_vs_exp008_full_train_rescored`), so all 13 statistics are available. The
brief's compare against exp-011's stored per-item file
(`runs/exp-026/compare_vs_exp008_full_train`) carries 10 — that file predates `decomp_mean`,
`over_decomposition` and `under_decomposition` — and the two **agree on every statistic they
share**.

| statistic | exp-026 (2,000 ex / 250 steps) | exp-008 (19,938 ex / 2,494 steps) | difference | test |
|---|---|---|---|---|
| **ordered_step_accuracy** | **0.2969** | **0.3237** | **−0.0268** | **CI [−0.0464, −0.0067], favours full_train** |
| **step_f1** | **0.3185** | **0.3411** | **−0.0226** | **CI [−0.0421, −0.0027], favours full_train** |
| **chain_validity** | **0.9750** | **0.9942** | **−0.0192** | **CI [−0.0306, −0.0086], favours full_train** |
| **composite (legacy)** | **0.5059** | **0.5236** | **−0.0177** | **CI [−0.0313, −0.0037], favours full_train** |
| ged | 0.3191 | 0.3094 | +0.0097 | CI [−0.0051, +0.0242] |
| decomp_mean | 0.6272 | 0.6357 | −0.0085 | CI [−0.0207, +0.0040] |
| hop_count_exact_match | 0.7033 | 0.7100 | −0.0067 | McNemar p=0.789 |
| break_exact_match | 0.0967 | 0.1017 | −0.0050 | McNemar p=0.766 |
| over_decomposition | 0.0383 | 0.0333 | +0.0050 | McNemar p=0.728 |
| rouge_l_f1 | 0.6607 | 0.6647 | −0.0039 | CI [−0.0155, +0.0077] |
| sari | 0.6804 | 0.6823 | −0.0019 | CI [−0.0139, +0.0102] |
| under_decomposition | 0.2583 | 0.2567 | +0.0017 | McNemar p=1.0 |
| exact_match | 0.1083 | 0.1083 | 0.0000 | McNemar p=1.0 |

**Of the seven report-card terms, exactly one (chain_validity, −1.9 pp) is significant.** The other
three significant statistics are the two step-overlap terms and the legacy composite. Every
significant difference is ≤ 2.7 percentage points. `exact_match` is *identical* at 0.1083 (25
items right only in exp-026, 25 right only in exp-008 — a different 25, at the same rate). No
correction for multiple comparisons is applied (ADR 0009).

## exp-026 vs exp-015 (prompting, fp16, same pool) — **11 of 13 significant, all favouring the fine-tune**

composite +0.1926 CI [+0.1412, +0.2994]; over_decomposition −0.1900 McNemar p=1.0e-26;
hop_count_EM +0.1633 p=7.1e-12; ged −0.1523 CI [−0.1706, −0.1337]; ordered_step_accuracy +0.1307;
step_f1 +0.1291; rouge_l_f1 +0.1161; sari +0.1019; decomp_mean +0.0969; exact_match +0.0633
p=4.5e-07; break_exact_match +0.0617 p=1.2e-07 — all favouring exp-026. Not significant:
under_decomposition (+0.0267, p=0.205 — the fine-tune is *slightly worse* here, not significantly)
and chain_validity (+0.0056). **Six of the seven report-card terms move significantly in the
fine-tune's favour**, at ~32% of the token cost.

## exp-026 vs exp-020 (prompting, 4-bit) — **12 of 13 significant; CONTEXT ONLY, precision differs**

composite +0.3160; over_decomposition −0.2817 p=1.2e-42; hop_count_EM +0.1917 p=8.9e-14; ged
−0.1655; step_f1 +0.1362; ordered_step_accuracy +0.1338; rouge_l_f1 +0.1222; decomp_mean +0.1059;
sari +0.1038; exact_match +0.0633; break_exact_match +0.0567 — all favouring exp-026. The one
statistic favouring exp-020 is **under_decomposition +0.0900 (p=8.0e-06)**: the 4-bit prompting arm
under-decomposes less, because it over-decomposes far more (0.3200 vs 0.0383). chain_validity
+0.0124 is not significant. **This pair moves both training method and precision** (exp-020
generates at 4-bit, exp-026 at fp16) and is reported as the confounded pair it is.

## `compare_decomposer_arms.py --eval-arm pool_2000_imbalanced` — accepted, nothing refused

Run with three arms — `prompting_fp16_exp015` (baseline), `finetuned_pool_2000_imbalanced_exp026`,
`finetuned_full_train_exp008` — it **accepted all three** and enforced the arm's `eval_hops`
[2, 3, 4] on every side (600 scored items each, all hop prefixes admissible; ADR 0012 item 11). It
re-scored all three run directories under one evaluator invocation, produced the two
arm-vs-baseline comparisons and the cost table above, rc=0. **It refused nothing.** Output:
`runs/exp-026/arm_comparison/`. Its exp-008-vs-exp-015 comparison, included for context, comes out
at **12 of 13 significant, all favouring `full_train`** (the exception is `under_decomposition`,
+0.0250, p=0.273) — the same shape as exp-026's own 11-of-13 against the same baseline.

## The one-line answer to "what did training on the pool only do"

**Training on only the 2,000-question pool gets essentially all of the way to full-split
fine-tuning, and stays far above prompting.** Against `full_train` — with **2,000 vs 19,938
examples and 250 vs 2,494 optimizer steps**, i.e. about a tenth of the training compute — 9 of 13
statistics are statistically indistinguishable, the 4 that differ all favour `full_train` by ≤ 2.7
pp, and only one of the seven report-card terms (chain_validity, −1.9 pp) is among them. Against
the precision-matched prompting arm on the same pool (exp-015), 11 of 13 favour the fine-tune, six
of them report-card terms, by 6–19 pp. So on this evaluation set the pool-only adapter does **not**
land midway: it lands next to full training and well above prompting. What it does *not* separate
is which of identity, volume or hop mix carries that — the untrained `pool_2000` control (issue
#13) is what would.

## Caveats a later reader needs

- One adapter, one evaluation set, one seed. No claim beyond `pool_2000_imbalanced` on the ADR 0007
  pinned 600.
- `chain_validity` is the one report-card term where full training measurably wins (0.9942 vs
  0.9750). If chain well-formedness is the property that matters downstream, the gap is real even
  though it is small.
- The exp-020 comparison confounds training method with precision and is context only.
- Run scaffolding incident, recorded because it is in the log: the first waiter's **evaluation**
  gate misfired — it armed on `[model_size]` appearing in the run log, which the *training* step
  had already written, then failed to parse a decomposer startup dump that did not exist yet and
  killed the decomposer **15 s in, before any token was generated** (`rc=143`). No training was
  repeated, no hyperparameter changed; the decomposer was relaunched with the identical command
  from `runs/g2_eval.sh`, which gives the decomposer its own log so the gate can only see
  decomposer output. Both the aborted attempt and the successful one are in
  `runs/exp-026/run.log`. The scaffolding lives under the gitignored `runs/` tree; no pipeline code
  or config was touched.

## Reproducibility

- Launch SHA: the run's own snapshots stamp **`f2fe7d2`** — snapshots stamp the commit at *write*
  time, not execution time (issue #54), and the G3 lane landed two commits between this lane's
  pre-run row and this lane's launch. This lane's **pre-run row** is `02ba422`. The full range
  `02ba422..f2fe7d2` is exactly two commits: `2129313` (exp-025 G3c completion — its committed
  artifacts and its log row) and `f2fe7d2` (docs only, 8 added lines in
  `components/decomposer/README.md`). **No pipeline code and no config differs across that range**
  (`git diff --stat 02ba422 f2fe7d2`: `components/decomposer/README.md`,
  `experiments/exp-025/{config,metrics,notes}`, `experiments/log.md`), so the code this run
  executed is identical at both SHAs.
- Working tree dirty flag in the snapshots is the three untracked `Slides/*.pdf`, the same three
  exp-023/024/025 recorded. No tracked file was modified during the run.
- Config: `configs/finetune_decomposer.json` arm `pool_2000_imbalanced`,
  `configs/decomposer_musique.json`, `configs/musique_eval.json`. Seed 42 throughout.
- Adapter, checkpoints, predictions and per-item files stay on the box under `runs/exp-026/`; only
  the config snapshot, the metrics JSON and this note are committed.
- `runs/run.lock` acquired at 05:45:35 +03 (after the exp-025 holder released it — this lane waited
  1,140 s in a detached poller and never contended), re-acquired for the resumed evaluation at
  06:13:49, released at 06:28:29. No process was ever killed except this run's own decomposer by
  its own gate.
