# exp-020 — G1a: Mistral-7B-Instruct-v0.3, unguided, **4-bit**, ADR 0007 pinned 600

Issue #50 GPU lane, run G1a (`docs/plan/2026-09-20-next-step.md` §2; ADR 0031 item 5 and its
2026-09-14 precision amendment; ADR 0032 items 1 and 8). Ran 2026-09-20, 01:39–02:21 +03.

## What was run

exp-015's command sequence, unchanged except for `--quantization 4bit` and the output root.
Mistral-7B-Instruct-v0.3, condition `unguided`, seed 42, few-shot k=3 from the reranked top-5
artifact `runs/exp-014/retrieval/s2_pinned600_imbalanced_rerank_top5.jsonl` (sha256 verified against
exp-015's committed snapshot before launch). `configs/decomposer_musique.json` and
`configs/musique_eval.json` unedited; the 4-bit pinning is on the command line, so the model
folder default stays `quantization: "none"` — where ADR 0031's amendment says it belongs.
Then the evaluator on the 600 results, then two `--compare` runs. rc=0 at all four steps.

## Report card (n=600, 0 missing gold)

| term | value | direction |
|---|---|---|
| break_exact_match_rate | 0.0400 | ⇑ |
| sari_macro | 0.5766 | ⇑ |
| ged_macro | 0.4846 (`ged_fallback_counts {node_cap: 2}`) | ⇓ |
| chain_validity_macro | 0.9626 | ⇑ |
| hop_count_exact_match_rate | 0.5117 (307/600) | ⇑ |
| under_decomposition_rate | 0.1683 (101/600) | ⇓ |
| over_decomposition_rate | 0.3200 (192/600) | ⇓ |

Beside the suite: exact_match_rate 0.0450, step_f1_macro 0.1823, ordered_step_accuracy_macro
0.1632, rouge_l_f1_macro 0.5385, step_count_exact_rate 0.5117, mean signed step-count error +0.585;
`decomp_mean_macro` 0.5213 (contingency blend, not in the suite); legacy `composite_score` 0.1898
(not headlined, issue #40).

## Does 4-bit move Mistral's numbers on this eval set?

**On quality, no; on the direction of its length errors, yes.** Against exp-015 — the identical
arm at `quantization: none`, same 600 items, same retrieval artifact, same prompt, same seed —
**3 of 13 headline statistics are significant** (8 bootstrap + 5 McNemar):

- **Not significant (10/13):** every quality term. break_EM +0.0050 (p=0.375), SARI −0.0019
  (CI −0.0089..+0.0052), GED +0.0132 (CI −0.00003..+0.0266 — the CI touches zero), chain_validity
  −0.0067, hop_count_EM −0.0283 (p=0.149), exact_match ±0.0000 (p=1.0), ROUGE-L −0.0062,
  step_f1 −0.0070, ordered_step_accuracy −0.0030, decomp_mean −0.0091.
- **Significant (3/13):** `over_decomposition` +0.0917 (0.3200 vs 0.2283, p=1.96e-09, 71 items
  worse only under 4-bit vs 16 better) — favours fp16; `under_decomposition` −0.0633 (0.1683 vs
  0.2317, p=1.21e-06, 50 vs 12) — favours **4-bit**; legacy `composite_score` −0.1234
  (CI −0.1739..−0.0068) — favours fp16, and it is legacy/not headlined precisely because it is a
  blend that the step-count-error term makes sensitive to exactly this shift.

So 4-bit pushes the same model toward **longer** plans without changing how good they are:
significantly more over-decomposition, significantly less under-decomposition, and a *net*
granularity accuracy (hop_count_EM 0.5117 vs 0.5400) that is **not** statistically distinguishable.
Under ADR 0017's asymmetry — over-decomposition tolerable, under-decomposition not — the two
significant length moves point in opposite normative directions, and this note does not adjudicate
between them. The tail is visible in the hop distribution: exp-020 emitted single plans of 13, 14,
15, 19 and **126** steps where exp-015's longest was 12.

The other measured cost of 4-bit is **time**: 2,469.5 s of `model.generate` against exp-015's
1,086.2 s for the same 600 queries (mean latency 4.12 s vs 1.81 s), i.e. **~2.3× slower** on this
3090. What it buys is VRAM headroom — **4,818 MiB peak** for this run. exp-015's peak was never
measured, so no VRAM ratio is claimed here.

## Against the fine-tuned arm

`--compare` vs `runs/exp-011/exp008_full_train/eval_per_item.json` (LoRA `full_train`, exp-008, not
re-run): **10 of 10 headline statistics significant, every one favouring the fine-tuned arm** —
ROUGE-L −0.1262, step_f1 −0.1588, ordered_step_accuracy −0.1605, SARI −0.1057, GED +0.1752 (worse),
chain_validity −0.0315, composite −0.3337 (bootstrap, all CIs exclude 0); exact_match −0.0633
(p=6.97e-08), hop_count_EM −0.1983 (p=6.16e-15), break_EM −0.0617 (p=5.73e-08). This reproduces
exp-015's 10/10 result on the 4-bit axis: quantizing the prompting arm does not close the gap.
Caveat carried over from exp-015: the exp-011 comparator file predates the over/under-decomposition
and `decomp_mean` columns, so those three are **not** in this paired test.

## Commit provenance — read this before quoting the commit

The run's own `metrics.json` stamps `cc13bee`, but that is the commit at the moment it **wrote**
the file (02:21), not the one it executed. Generation ran on **990be38** (= `b703100` for all
pipeline code; 990be38 adds only this run's pre-run log row), loaded at 01:39:02. Four build-lane
commits landed on `main` during the 42-minute generation (`a6e439c`, `e95e550` B2 anchor set,
`72538e1`, `cc13bee` B3 ordering knob). None reached this process: Python had already read
`run_decomposer.py` and both configs, and the run log's startup config dump contains **no
`ordering` key** — direct evidence it used the pre-merge config. The evaluator and both compares
ran at `cc13bee`, but `git diff 990be38 cc13bee -- scripts/musique_decompositions_evaluator.py
configs/musique_eval.json` is empty, so scoring is byte-identical to the pre-run state.

## Machine / process

Preflight: `main` b703100, lock absent, GPU 269/24,576 MiB used by other users, `/cta/users` 348 G
free, retrieval artifact sha256 `3f95c7dc…fdb0` matching exp-015's snapshot, 600 rows, 200/200/200.
Lock `exp-020 2026-09-20T01:38:58+0300` acquired before launch and released by the runner at
02:21:19. Launched detached in tmux `exp-020` via `runs/exp-020/waiter.sh`, log `runs/exp-020/run.log`,
rc-checked at every step. Three `ml-engineer` build lanes shared the GPU with smoke tests during
the run (a second 2,660 MiB process was visible throughout); per the brief that is expected and was
not treated as contention — the VRAM sampler is filtered to this run's own pid so their usage is
excluded from the 4,818 MiB figure.
