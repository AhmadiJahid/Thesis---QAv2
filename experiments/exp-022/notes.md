# exp-022 — G1c: Qwen2.5-14B-Instruct, unguided, **4-bit**, ADR 0007 pinned 600

Issue #50 GPU lane, run G1c (`docs/plan/2026-09-20-next-step.md` §2; ADR 0031 item 5 and its
2026-09-14 precision amendment; ADR 0032 items 1, 2 and 8). Ran 2026-09-20, 03:38–04:06 +03.

## What was run

exp-021's command sequence, unchanged except for the model folder (`qwen2_5_14b_instruct`) and the
output root. Condition `unguided`, seed 42, `--quantization 4bit`, few-shot k=3 from
`runs/exp-014/retrieval/s2_pinned600_imbalanced_rerank_top5.jsonl` (sha256 `3f95c7dc…fdb0`,
verified before launch). Configs unedited; `--ordering` not passed, so `few_shot.ordering` resolved
to its default `as_is`. Then the evaluator on the 600 results, then three `--compare` runs. rc=0 at
all six steps.

**The size contrast moves one factor.** Against exp-021 this arm holds the family (Qwen2.5), the
precision (4-bit), the eval set, the retrieval artifact (hence the same k exemplars), the exemplar
order, the prompt derivation, the condition, the seed and the decoding. Parameter count —
7,615,616,512 → 14,770,033,664 — is the only thing that differs. ADR 0031's precision amendment is
what makes this possible: the 14B's bf16 weights (29.54 GB) do not fit this 3090 at all, so pinning
every arm to 4-bit is what stops size and precision moving together.

## Launch gate (issue #50, Gate 1 finding I-3)

All five preconditions **PASS**, checked twice — from the startup dump in `run.log` 15 s after
launch, before a token was generated and with a kill wired to a mismatch, and again from the
written `config.json` / `metrics.json` before the evaluator was allowed to run.

| precondition | observed |
|---|---|
| retrieval `input_sha256` | `3f95c7dc18c14fc6bea3473f1be5c0d7fb21b7b2f2b2a1175dbfaebb5092fdb0` |
| `quantization` | `"4bit"` |
| `few_shot_ordering` | `"as_is"` (source: config) |
| `evaluation_set.pinned` | `true` — 600 rows, 200/200/200, 0 missing, 0 unexpected |
| `model_size.ceiling_asserted` | `true` |

`[model_size] decomposer model=Qwen/Qwen2.5-14B-Instruct parameters=14,770,033,664
ceiling=15,000,000,000 (98.5% of ceiling)`

**98.5% of the ceiling.** This arm loads only because `decomposer_max_params` is 1.5e10 (ADR 0031
item 4, confirmed by Jahid in ADR 0032 item 2), which is still *pending supervisor confirmation*;
`default_max_params` stays at the supervisor's standing 8e9. If he reasserts 8B or 10B, this arm is
refused at load and this result is withdrawn.

## Report card (n=600, 0 missing gold)

| term | exp-022 Qwen2.5-14B | exp-021 Qwen2.5-7B | exp-020 Mistral-7B | direction |
|---|---|---|---|---|
| break_exact_match_rate | 0.0333 | 0.0550 | 0.0400 | ⇑ |
| sari_macro | 0.5795 | 0.5720 | 0.5766 | ⇑ |
| ged_macro | 0.4605 | 0.4694 | 0.4846 | ⇓ |
| chain_validity_macro | **0.9838** | 0.9210 | 0.9626 | ⇑ |
| hop_count_exact_match_rate | **0.6167** (370/600) | 0.5533 | 0.5117 | ⇑ |
| under_decomposition_rate | **0.1017** (61/600) | 0.2933 | 0.1683 | ⇓ |
| over_decomposition_rate | 0.2817 (169/600) | 0.1533 | 0.3200 | ⇓ |

`ged_fallback_counts {}` (empty, as exp-021; exp-020 needed `{node_cap: 2}`). Beside the suite:
exact_match_rate 0.0450 (27/600), step_f1_macro 0.1969, ordered_step_accuracy 0.1778,
rouge_l_f1_macro 0.5453, step_count_exact_rate 0.6167, step_count_mae 0.5117, mean signed
step-count error **+0.285**; `decomp_mean_macro` 0.5507 (contingency blend, not in the suite);
legacy `composite_score` 0.4150 (not headlined, issue #40). 0 rows at `max_new_tokens`, 0 empty
decompositions, 1,971 step lines, predicted step counts spanning 1–8.

## Does doubling parameters inside Qwen2.5 change the report card? Yes — 7 of 13

`--compare` vs exp-021, n=600 aligned, **7 of 13 headline statistics significant**:

- **Favouring the 14B (4):** `under_decomposition` −0.1917 (0.1017 vs 0.2933, p=7.32e-23, 133 items
  worse only at 7B vs 18 at 14B — the largest single effect anywhere in this block),
  `chain_validity` +0.0628 (0.9838 vs 0.9210, CI +0.0410..+0.0851), `hop_count_exact_match` +0.0633
  (0.6167 vs 0.5533, p=0.0120, 128 vs 90), `decomp_mean` +0.0242 (CI +0.0085..+0.0397).
- **Favouring the 7B (3):** `over_decomposition` +0.1283 (0.2817 vs 0.1533, p=1.87e-10, 113 vs 36),
  `break_exact_match` −0.0217 (p=0.0072, 4 vs 17), `exact_match` −0.0183 (p=0.0433, 7 vs 18).
- **Not significant (6):** every n-gram / graph / step-overlap quality term — ROUGE-L +0.0041,
  step_f1 +0.0039, ordered_step_accuracy +0.0006, SARI +0.0075, GED −0.0089, legacy composite
  +0.0039.

**What size buys is granularity and chaining, not wording.** The 14B is the only arm in the block
whose net granularity accuracy is significantly above another arm's (hop_count_EM 0.6167), it
almost eliminates under-decomposition (0.1017, from the 7B's 0.2933), and it chains near-perfectly
(0.9838, and uniformly across hops: 0.9800 / 0.9838 / 0.9875 at 2 / 3 / 4 hops, where the 7B falls
to 0.8775 at 2 hops). What it gives back is over-decomposition (0.2817 vs 0.1533) and the two
verbatim-identity metrics, `break_exact_match` and `exact_match`, both of which drop — a longer,
more elaborated plan is less likely to be character-identical to the gold. Mean signed step-count
error moves from the 7B's −0.150 to +0.285: the 14B swings back across zero toward Mistral's +0.585
without reaching it, and step_count_mae is the block's best at 0.5117. Under ADR 0017's asymmetry
(over-decomposition tolerable, under-decomposition not) the 14B's trade is the favourable one, but
this note does not adjudicate — that reading is Jahid's and his supervisor's.

## Against the Mistral point (confounded: family and size both move)

`--compare` vs exp-020: **6 of 13 significant, all six favouring the 14B** — hop_count_EM +0.1050
(p=1.72e-05), under_decomposition −0.0667 (p=9.31e-05), GED −0.0241 (CI −0.0417..−0.0064),
chain_validity +0.0211, decomp_mean +0.0293, legacy composite +0.2252. Not significant: ROUGE-L,
step_f1, ordered_step_accuracy, SARI, exact_match (p=1.0), break_EM (p=0.503) and — worth noting —
`over_decomposition` (0.2817 vs 0.3200, p=0.094). This pair moves **both** factors at once and is
therefore not one of the supervisor's contrasts; it is recorded because it is the cheapest way to
say where the largest admissible prompting model sits against the block's Mistral baseline.

## Against the fine-tuned arm — the first crack

`--compare` vs `runs/exp-011/exp008_full_train/eval_per_item.json`: **9 of 10 significant, all nine
favouring the fine-tuned arm** — ROUGE-L −0.1194, step_f1 −0.1443, ordered_step_accuracy −0.1460,
SARI −0.1028, GED +0.1511, composite −0.1086 (bootstrap); exact_match −0.0633 (p=1.38e-07),
hop_count_EM −0.0933 (p=3.64e-04), break_EM −0.0683 (p=5.81e-09).

**The tenth is `chain_validity`, and for the first time in this block it is NOT significant**:
0.9838 vs 0.9942, CI −0.0219..+0.0006. exp-015, exp-020 and exp-021 each went 10/10; a 14B
prompting model closes the chaining term to a statistical tie. Every wording and step-overlap term
still favours fine-tuning by a wide margin, so **the gap is not closed** — one term of ten moved
out of the significant column. Caveat carried over: the exp-011 comparator file predates the
over/under-decomposition and `decomp_mean` columns, so those three are not in this paired test.
Descriptively (each side's own `eval_metrics.json`, not a paired test): exp-022 under 0.1017 / over
0.2817 vs exp008_full_train under 0.2567 / over 0.0333.

## Cost

| | exp-022 14B | exp-021 7B | exp-020 Mistral-7B |
|---|---|---|---|
| total wall clock | 1,674 s (27 min 54 s) | 833 s | 2,537 s |
| decompose wall clock | 1,663 s | 824 s | 2,526 s |
| `model.generate`, 600 queries | 1,648.0 s | 812.9 s | 2,469.5 s |
| mean / median latency per query | 2.747 s / 2.649 s | 1.355 s / 1.332 s | 4.116 s / 1.739 s |
| peak VRAM (own-pid sample) | **10,416 MiB** | 6,082 MiB | 4,818 MiB |
| mean completion tokens per query | 33.9 | 28.2 | — |

Doubling parameters cost **2.03× the generation time** and **1.71× the sampled VRAM** inside
Qwen2.5. The 14B is still **1.50× faster** than 4-bit Mistral-7B on the same 600 queries. VRAM
figures are sampled process-reserved maxima filtered to each run's own pid, not
`torch.cuda.max_memory_allocated`.

## Machine / process

Preflight: `main` **761374e**, working tree clean of tracked changes (three untracked `Slides/*.pdf`),
`runs/run.lock` absent, GPU **10 MiB / 24,576 MiB** — the other user's two processes had exited, so
this arm had the card to itself; nothing of anyone's was touched at any point. `/cta/users` 339 G
free. Qwen2.5-14B weights 8/8 shards in the HF cache (28 GB). Pre-run log row committed as
**2416a8b** — the **launch SHA**.

**Commit provenance — read before quoting a SHA.** `metrics.json` stamps `3cce79c`, which is the
commit at *write* time. Two commits landed on `main` during the 28-minute run: `78b7f06` (ADR 0034,
docs) at 03:55 and `3cce79c` (the B4 anchor-prompt-logging lane) at 04:04, two minutes before the
decomposer finished. `3cce79c` touches `run_decomposer.py` (+155) and adds a required
`logging.anchor_ids_file` key to `configs/decomposer_musique.json`. **It did not reach this
process**, and the evidence is in the artifacts, not in an argument from timing: the run's startup
config dump contains no `anchor_ids_file` (0 matches in `run.log`), the written `config.json` has
no `logging` block, `metrics.json` has no `anchor_prompts_logged` field, and `prompts_log/` holds
only the 120 positional `prompt_idx*.txt` samples with no `anchor_*.txt` files. The evaluator and
the three `--compare` runs executed after `3cce79c`, but
`git diff 2416a8b 3cce79c -- scripts/musique_decompositions_evaluator.py configs/musique_eval.json`
is **empty**, so scoring is byte-identical to the pre-run state.

Lock `exp-022 2026-09-20T03:38:10+0300` acquired before launch, released by the waiter at 04:06:04
(verified absent afterwards). Launched detached in tmux `exp-022` via `runs/exp-022/waiter.sh`, log
`runs/exp-022/run.log`, rc-checked at every step, peak-VRAM sampler scoped to this run's own pid.
