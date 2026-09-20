# exp-021 — G1b: Qwen2.5-7B-Instruct, unguided, **4-bit**, ADR 0007 pinned 600

Issue #50 GPU lane, run G1b (`docs/plan/2026-09-20-next-step.md` §2; ADR 0031 item 5 and its
2026-09-14 precision amendment; ADR 0032 items 1 and 8). Ran 2026-09-20, 03:17–03:31 +03.

## What was run

exp-020's command sequence, unchanged except for the model folder (`qwen2_5_7b_instruct`) and the
output root. Condition `unguided`, seed 42, `--quantization 4bit`, few-shot k=3 from the reranked
top-5 artifact `runs/exp-014/retrieval/s2_pinned600_imbalanced_rerank_top5.jsonl` (sha256
`3f95c7dc…fdb0`, verified before launch). `configs/decomposer_musique.json` and
`configs/musique_eval.json` unedited; `--ordering` was **not** passed, so `few_shot.ordering`
resolved to its default `as_is` — byte-identically the exemplar order exp-015 and exp-020 used.
Then the evaluator on the 600 results, then two `--compare` runs. rc=0 at all five steps.

**The family contrast moves one factor.** Against exp-020, this arm holds the eval set, the
retrieval artifact (hence the same k exemplars), the prompt derivation (the parity-guarded
`prompt_unguided.md`), the condition, the seed, the decoding and the precision constant. The model
family is the only thing that differs.

## Launch gate (issue #50, Gate 1 finding I-3)

All five preconditions **PASS**, checked twice: from the startup dump in `run.log` 10 seconds after
launch — before a token was generated, with a kill wired to a mismatch — and again from the written
`config.json` / `metrics.json` before the evaluator was allowed to run.

| precondition | observed |
|---|---|
| retrieval `input_sha256` | `3f95c7dc18c14fc6bea3473f1be5c0d7fb21b7b2f2b2a1175dbfaebb5092fdb0` |
| `quantization` | `"4bit"` |
| `few_shot_ordering` | `"as_is"` (source: config) |
| `evaluation_set.pinned` | `true` — 600 rows, 200/200/200, 0 missing, 0 unexpected |
| `model_size.ceiling_asserted` | `true` |

`[model_size] decomposer model=Qwen/Qwen2.5-7B-Instruct parameters=7,615,616,512
ceiling=15,000,000,000 (50.8% of ceiling)`

**Note for the next runner.** The brief assumed `config.json` appears at launch. It does not:
`run_decomposer.py` writes the snapshot only at the *end* of a run (`src/run_artifacts.py:109`).
The same fields are printed to stdout at startup (`run_decomposer.py:2433`, the `Evaluation set:`
line, and `[model_size]` from `src/model_size.py:93`), so the gate reads the run log — and that
requires `PYTHONUNBUFFERED=1`, because Python block-buffers stdout into a file and the first launch
attempt (03:06) sat 11 minutes with the model loaded and nothing in the log. That attempt was
killed and the run relaunched unbuffered; no generated output from it was kept.

## Report card (n=600, 0 missing gold)

| term | exp-021 Qwen2.5-7B | exp-020 Mistral-7B | direction |
|---|---|---|---|
| break_exact_match_rate | **0.0550** | 0.0400 | ⇑ |
| sari_macro | 0.5720 | 0.5766 | ⇑ |
| ged_macro | 0.4694 (`ged_fallback_counts {}`) | 0.4846 (`{node_cap: 2}`) | ⇓ |
| chain_validity_macro | **0.9210** | 0.9626 | ⇑ |
| hop_count_exact_match_rate | 0.5533 (332/600) | 0.5117 (307/600) | ⇑ |
| under_decomposition_rate | **0.2933** (176/600) | 0.1683 (101/600) | ⇓ |
| over_decomposition_rate | **0.1533** (92/600) | 0.3200 (192/600) | ⇓ |

Beside the suite: exact_match_rate 0.0633 (38/600), step_f1_macro 0.1929, ordered_step_accuracy
0.1771, rouge_l_f1_macro 0.5412, step_count_exact_rate 0.5533, step_count_mae 0.5767, mean signed
step-count error **−0.150**; `decomp_mean_macro` 0.5264 (contingency blend, not in the suite);
legacy `composite_score` 0.4111 (not headlined, issue #40).

## Does the model family change the report card? Yes — 6 of 13, in both directions

`--compare` vs exp-020, n=600 aligned, **6 of 13 headline statistics significant** (8 bootstrap +
5 McNemar):

- **Favouring Qwen2.5-7B (4):** `over_decomposition` −0.1667 (0.1533 vs 0.3200, p=1.25e-15, 132
  items worse only under Mistral vs 32 under Qwen), `exact_match` +0.0183 (p=0.0192, 15 vs 4),
  `break_exact_match` +0.0150 (p=0.0352, 12 vs 3), legacy `composite_score` +0.2212 (CI
  +0.2006..+0.2432).
- **Favouring Mistral-7B (2):** `chain_validity` −0.0416 (0.9210 vs 0.9626, CI −0.0658..−0.0188),
  `under_decomposition` +0.1250 (0.2933 vs 0.1683, p=2.85e-09, 118 vs 43).
- **Not significant (7):** ROUGE-L +0.0027, step_f1 +0.0106, ordered_step_accuracy +0.0139, SARI
  −0.0046, GED −0.0152 (CI −0.0341..+0.0036), decomp_mean +0.0051, and — the one most worth
  noticing — `hop_count_exact_match` +0.0417 (0.5533 vs 0.5117, **p=0.108**, 124 vs 99).

**The mechanism is a sign flip in length error, not a quality gap.** Mistral over-decomposes
(mean signed step-count error **+0.585**); Qwen under-decomposes (**−0.150**). Each model is
significantly better than the other on the error it does not make. Net granularity accuracy
(hop_count_EM) is **not** distinguishable, and neither is any n-gram / graph / step-overlap quality
term. Under ADR 0017's asymmetry — over-decomposition tolerable, under-decomposition not — the two
significant length moves point in opposite normative directions and this note does not adjudicate
between them. Per gold hop the flip is monotone: under-decomposition Qwen/Mistral is 0.105/0.000 at
2 hops, 0.320/0.220 at 3, 0.455/0.285 at 4, while over-decomposition runs the other way
(0.085/0.230, 0.155/0.315, 0.220/0.415).

**Qwen's plans have no runaway tail.** Predicted step counts span 1–8 (`{2:262, 3:168, 4:97, 5:31,
1:24, 6:12, 7:5, 8:1}`); exp-020 emitted single plans of 13, 14, 15, 19 and **126** steps and hit
`max_new_tokens` on 26 rows. exp-021 hit it on **0** rows and needed no GED node-cap fallback
(exp-020 needed 2). The legacy composite is the only blended number that moves, and it moves a lot
(+0.2212), which is consistent with its step-count-error term being sensitive to exactly this —
an argument for issue #40's demotion, not a quality claim.

**Where Qwen is worse is chaining.** chain_validity 0.9210 vs 0.9626 is significant and is
concentrated at 2 hops (0.8775 vs 0.9869); at 4 hops Qwen is slightly ahead (0.9538 vs 0.9428).
Shorter plans with fewer `#k` references is a plausible mechanism and is **not** tested here.

## Against the fine-tuned arm

`--compare` vs `runs/exp-011/exp008_full_train/eval_per_item.json` (LoRA `full_train`, exp-008, not
re-run): **10 of 10 headline statistics significant, every one favouring the fine-tuned arm** —
ROUGE-L −0.1234, step_f1 −0.1482, ordered_step_accuracy −0.1466, SARI −0.1103, GED +0.1600 (worse),
chain_validity −0.0732, composite −0.1125 (bootstrap, all CIs exclude 0); exact_match −0.0450
(p=1.98e-04), hop_count_EM −0.1567 (p=1.57e-10), break_EM −0.0467 (p=9.02e-05). This is the third
10/10 on this comparator (exp-015, exp-020, exp-021): **the prompting-vs-fine-tuning gap is not a
Mistral idiosyncrasy** — changing the prompting model's family does not close it. Caveat carried
over: the exp-011 comparator file predates the over/under-decomposition and `decomp_mean` columns,
so those three are not in this paired test.

## Cost

| | exp-021 Qwen2.5-7B | exp-020 Mistral-7B |
|---|---|---|
| total wall clock (decompose+eval+compares) | **833 s** (13 min 53 s) | 2,537 s (42 min 17 s) |
| decompose wall clock | 824 s | 2,526 s |
| `model.generate` total, 600 queries | **812.9 s** | 2,469.5 s |
| mean / median latency per query | 1.355 s / 1.332 s | 4.116 s / 1.739 s |
| peak VRAM (own-pid sample) | **6,082 MiB** | 4,818 MiB |
| mean completion tokens per query | 28.2 | (see exp-020) |

Qwen2.5-7B generated the same 600 decompositions **~3.0× faster** than Mistral-7B at the same
precision on this 3090, for ~1.26× the sampled VRAM footprint. Both figures are sampled
process-reserved maxima filtered to the run's own pid, not `torch.cuda.max_memory_allocated`.

## Machine / process

Preflight: `main` **feef949**, working tree clean of tracked changes (three untracked `Slides/*.pdf`),
`runs/run.lock` absent, GPU 4,752/24,576 MiB held by another user's two processes (never touched),
`/cta/users` 339 G free, retrieval artifact sha256 verified independently, Qwen2.5-7B weights 4/4
shards in the HF cache. Pre-run log row committed as **4c2375c** — that is the **launch SHA**, and
it is recorded by hand because `metrics.json` stamps the commit at *write* time: this run's
`metrics.json` says `37dae1a`, a commit that landed on `main` during the 14-minute run and whose
entire diff is `tests/test_model_params.py` (+6/−2), i.e. no pipeline code and no config. Lock
`exp-021 2026-09-20T03:17:39+0300` acquired before launch, released by the waiter at 03:31:32.
Launched detached in tmux `exp-021` via `runs/exp-021/waiter.sh`, log `runs/exp-021/run.log`,
rc-checked at every step. One CPU-only build lane (B4) was working in `.worktrees/b4` concurrently
and committed nothing to `main`.
