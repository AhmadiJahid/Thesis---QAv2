# exp-038 — weight-space repair of under-decomposition at depth (PENDING)

Method, leakage guard and controls: [ADR 0042](../../docs/adr/0042-weight-space-repair-track-exp-038.md).
**A separate track outside the few-shot pool study.** Nothing here has been run on a GPU yet.
No number below is a result of this experiment.

## Target and the defect (numbers cited from committed files)

The exp-026 pool-only fine-tune (Mistral-7B-Instruct-v0.3 + LoRA `pool_2000_imbalanced`), merged
in fp16. On the pinned 600 (`experiments/exp-026/notes.md`, per-hop table):

| gold hop | hop-count EM | under | over |
|---|---|---|---|
| 2 | 0.965 | 0.015 | 0.020 |
| 3 | 0.645 | 0.295 | 0.060 |
| 4 | 0.500 | 0.465 | 0.035 |

It stops too early at depth and almost never runs long, so the error is one-sided. The prompted
Mistral-7B arm at 4 hops has under 0.285 and over 0.415 on the pinned subset
(`experiments/exp-030/resample_summary.csv`), so its error is two-sided. exp-013's 0.845 at 4 hops
(`experiments/exp-013/notes.md`) comes from leaving 4-hop rows out of training.

## Steps (the notebook does all of them; each is resumable and mirrored to Drive)

| # | stage | where | what | estimate (not measured) |
|---|---|---|---|---|
| 0–3 | setup | Colab | mount Drive, clone `main`, install pinned deps (Colab's torch kept, recorded), extract only the needed files from `thesis-backup.tar.gz` (sha256-checked), GPU check | ~15 min (tarball copy + extract) |
| 1 | `split` | CPU | locate 400/hop, val_select 100/hop, val_confirm 100/hop from MuSiQue **train**, minus the adapter's 2,000 rows; 16 disjointness checks against dev / pinned 600 / pool | seconds |
| 2 | `locate` | GPU | greedy generation on all three splits, batched 16, and residual stream after every layer at every step boundary of locate and val_select | 15–25 min |
| 3 | `directions` | CPU | per-layer stratum-matched difference-of-means (two kinds), val AUROC, cosines, proposed sweep | a few min |
| — | **PAUSE 1** | Claude | inspect `directions_table.md` etc.; optional `candidates_override.json`; reasoning in `choices.md`; commit and push | — |
| 4 | `sweep` | GPU | val_select under baseline, 12 orthogonalisation edits (2 kinds × 3 layers × α 0.5/1), 2 steering diagnostics, 6 additive EOS-logit biases (+ up to 3 refinement rounds of 3 until one matches the edit's mean step count within 0.05), the norm-only sanity check | 30–75 min |
| 5 | `choose` | CPU | ADR 0042 default rule, length-matched EOS-bias control and its achieved gap → `proposed_choice.json` | seconds |
| — | **PAUSE 2** | Claude | commit `choice.json` (exactly once) + `choices.md`; `freeze` then records every spec hash in `specs_frozen.json`, never rewritten | — |
| 6 | `make-specs`, `freeze`, `equivalence`, `confirm` | GPU | five spec files; freeze; hook-vs-weight check (stops unless every argmax agrees and the largest logit difference is ≤ 0.5); one run of the five on val_confirm | ~15 min |
| 7 | pinned 600 × 5 arms | GPU | baseline / edit / random / eos_control / norm_control through `run_decomposer.py --weight-edit`, unbatched as exp-026, then the evaluator | ~75–90 min |
| 8 | `analysis.py` | CPU | bootstrap + McNemar + paired-t, overall and per hop, Holm within each pair's family of 28; the headline rule (hop-4 under-decomposition Holm-significant for the edit in BOTH family P and family L); length gap and per-hop step histograms on the pinned 600; size of the random control's weight change vs the edit's | ~5–10 min |

**Total GPU estimate: ~2.5–3.5 h on an A100 40 GB, ~4–6 h on an L4.** These are estimates, not
measurements. The per-query basis is exp-026's measured 1.404 s/query unbatched on the RTX 3090
(`experiments/exp-026/notes.md`), scaled by guesswork. On top of that come ~9 model loads
(~2 min each from the HF cache, plus one ~14.5 GB first download).

## Hardware: A100 40 GB recommended, L4 24 GB works, T4 does not

- exp-026's fp16 evaluation peaked at **14,438 MiB** (`experiments/exp-026/notes.md`), which is too
  close to a T4's 15 GB, and the T4 has no bf16. The notebook refuses below 22 GiB.
- Estimated peak VRAM (unmeasured; the run records it): ~15–18 GB. That is fp16 weights (~14.5 GB),
  batched greedy generation with short outputs, and one float32 copy of the matrix being edited
  (≤ 235 MB for a `down_proj`). An L4 fits; an A100 is roughly twice as fast per stage.
- System RAM: the sweep keeps CPU copies of every edited matrix, ~5.1 GB for all `o_proj` +
  `down_proj` + embedding in fp16. Activations take ~1 GB. Use the high-RAM runtime.
- Drive: ~1.5 GB (activations, generations, specs, results).

## Claude's two choices

Claude writes `choices.md`, `candidates_override.json` (optional) and `choice.json` in this folder,
in Jahid's terminal, from the statistics the notebook saved to `MyDrive/thesis-exp038/`. Each
choice is committed before the notebook proceeds. The pinned 600 are never shown before
`choice.json` is committed (ADR 0042 items 5–6). Jahid approved this role on 2026-11-27 and does not consider it a
closed-model judge (ADR 0042, "Decided by Jahid"). `choice.json` gets exactly one commit; a
second one makes the freeze refuse.

## What gets committed here after the run

The notebook's last cell lists the files: environment and GPU records, `target.json`, the split
record, the per-split locate summaries, the direction statistics, the sweep, confirm and choice
records, `specs/*.json` (the direction `.npy` files stay on Drive; their sha256 is in each spec),
`specs_frozen.json`, each arm's decomposer `config.json`/`metrics.json` and evaluator
`eval_metrics.json`, and `analysis_{config,metrics}.json`. After that come the log row's
metrics and a run note. Activations, generations and per-item files stay on Drive.

## Before running

1. The notebook clones `main` (the repo is public, so no GitHub token is used). Put `HF_TOKEN` in Colab secrets.
2. The lead reserves exp-038 and ADR 0042 on `main` before merge (ADR 0040 item 2).
3. Code review of the `--weight-edit` runner change (Gate 1), since it touches shared pipeline code.
