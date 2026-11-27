# exp-041 — training-seed variance of the headline fine-tune

**Status: PENDING, not run.** Nothing here is a result yet. Decisions: ADR 0047. Plan, commands,
gates and families: `run_config.json` (committed before any run).

## Why

The headline "fine-tuning beats prompting" rests on one training run per adapter: exp-026 vs
exp-015, with 11 of 13 terms significant (experiments/exp-026/notes.md). exp-041 retrains exp-026
under three training seeds. It asks how much the report card moves from seed to seed, and whether
each seed's comparison against prompting keeps exp-026's sign and significance (writing audit (d),
rank 3).

## What moves, what is held

**Only `--seed` on `train_lora.py` moves: 42, 43 and 44.** No pipeline code changes.

- `train_lora.py` already takes `--seed`. It sets the global seed before the base model loads and
  LoRA is attached, which fixes the LoRA init.
- It passes the seed to trl as `seed` and `data_seed`, which fix the data order and dropout.
- The arm has `max_examples: null`, so the training set is the whole pinned pool under every seed
  (`tests/test_exp041_plan.py` and the training gate check this).

Everything else is held, all from the unedited `configs/finetune_decomposer.json`, which the test
asserts equals exp-026's snapshot:

- the pool (sha256 `cc364662…6646`, 2,000 rows, 1461/413/126);
- 4-bit QLoRA with r=16;
- 250 steps;
- exp-026's evaluation: fp16 + adapter, `--no-few-shot`, unguided, decode seed 42, pinned 600.

Seed 42 is retrained on the same Colab GPU, so the three-seed spread carries no hardware term.
Stored exp-026 (3090) serves as the hardware reference.

The prompting comparator is exp-015's exact command, re-run on the same GPU (primary). exp-015 is
the precision-matched arm that exp-026 was compared with. The stored exp-015, read through
exp-032's re-parse, is the secondary comparator.

## What is reported (`compare.py` → `compare_metrics.json`)

**Spread.** For each of the 7 report-card terms (break EM, SARI, GED, chain validity, hop-count EM,
under- and over-decomposition) × {overall, hop 2/3/4}, `compare.py` reports:

- each seed's value, with the mean, SD, min, max and range across seeds;
- the same statistics for each seed's gain over the prompting comparator.

It does this for {42-Colab, 43, 44}, and as a secondary set for {exp-026 stored, 43, 44}. With
n = 3 seeds the spread is descriptive only.

**Per-seed verdicts.** Each seed is compared against the prompting re-run. Each comparison is one
Holm family of 28 tests (7 terms × 4 strata).

- The stored exp-026 vs exp-015 reading is recomputed by the same code as the reference family.
- Each cell reports whether it has the same sign, the same uncorrected verdict and the same Holm
  verdict as the reference.
- "Holds across seeds" is claimed only for cells where the reference and all three seeds are
  Holm-significant with the same sign.
- Secondary families repeat this against the stored exp-015.

Holm's p is exact McNemar for binary terms and paired t for continuous ones, with the bootstrap CI
beside it (ADR 0009, 0036).

**Context (uncorrected).** Seed-vs-seed pairs, plus two hardware controls: seed 42 vs stored
exp-026, and the prompting re-run vs stored exp-015. A control counts as "reproduced" only when
600/600 decompositions are identical strings.

## How to run

**On Colab** (the runner session or Jahid): open `notebooks/exp-041_ft_seed_variance_colab.ipynb`
on a GPU runtime with ≥ 16 GiB and bf16 (L4, A100, H100 or G4) and run all cells. It needs `HF_TOKEN` (Colab secret or environment) and
`MyDrive/thesis-backup.tar.gz`. It follows the exp-036/037 conventions:

- clones the public repo, and pins the commit on Drive at first launch (`launch_sha.txt`);
- builds a Python 3.13.9 uv venv with the `requirements.txt` pins and refuses on drift;
- extracts data with `scripts/colab_setup_data.py`;
- links `runs/exp-041` in the Colab clone to Drive.

Order of the run:

1. A 20-question canary of the prompting command.
2. For seeds 43, 44, then 42: train, then the training gate. The gate checks the pool sha, 2,000
   rows, hop mix, overlap 0, 250 steps, the seed, the launch commit, a clean tree and the ceiling.
3. For each seed, a 20-question canary with the adapter, then the full 600. The decode gate checks
   600 = 200/hop, the retrieval sha, prompt parity 0, 45 anchors, the commit, and output health
   (≤ 1% empty, ≤ 5% at `max_new_tokens`). The evaluator runs after the gate.
4. The prompting re-run.
5. `compare`.

Before any GPU job launches, `run.sh` runs two more checks:

- **Clean tree, untracked files included.** It uses the full `git status --porcelain`, the same
  definition the training gate uses through `src/run_artifacts.py`. A stray file fails here, not
  after training.
- **Free VRAM.** It checks total minus used memory (`run_config.json` `gpu.free_vram_gate_gib`):
  training needs ≥ 6 GiB free and evaluation ≥ 15.5 GiB, the exp-026 peak plus a margin. The GPU
  may be shared with queued jobs, and a step that does not fit exits 4 with nothing started.
  Re-run it when the GPU is free.

If the runtime dies, run all cells again. An interrupted decode resumes. An interrupted training
restarts at step 0 with the same seed, and the partial directory is kept under `train_incomplete/`.

**On any CUDA box:**

```bash
bash experiments/exp-041/run.sh canary_prompted
for s in 43 44 42; do bash experiments/exp-041/run.sh train $s; bash experiments/exp-041/run.sh eval $s; done
bash experiments/exp-041/run.sh prompted
bash experiments/exp-041/run.sh compare     # -> experiments/exp-041/compare_metrics.json
```

## GPU time: ESTIMATES, not measured

These estimates scale exp-026's 3090 measurements:

- training: 1,568.8 s;
- fp16 eval: 842.3 s;
- exp-015: 1.81 s/query.

The scaling factors are A100 ×0.5–0.7 for training and ×0.5–0.6 for decoding.

| step | A100 (estimate) | L4 (estimate) |
|---|---|---|
| setup | 15–25 min | 15–25 min |
| prompting canary | 3–6 min | 5–10 min |
| each seed: training | 13–18 min | 34–65 min |
| each seed: canary + full eval + evaluator | 10–16 min | 40–55 min |
| prompting re-run + evaluator | 12–17 min | 50–62 min |
| compare (CPU) | 5–15 min | 5–15 min |
| **total** | **~1.7–2.8 h** | **~5.0–7.9 h** |

The L4 factors (exp-036's) are training ×1.3–2.5 and fp16 decoding ×2.5–3.1 relative to the 3090.
The code reviewer's independent guess for the L4 was ~4–6 h.

Memory peaks measured on the 3090 were 5.0 GiB for training and 14.1 GiB for the fp16 eval. The
notebook requires ≥ 16 GiB and bf16. A GPU shared with another job is slower. Measured times are
written to `runs/exp-041/colab_steps.jsonl`.

## After the run

1. Copy `MyDrive/thesis-exp-041/runs/exp-041/` to the laptop's `runs/exp-041/` and
   `compare_metrics.json` to `experiments/exp-041/`. To recompute it, run
   `python experiments/exp-041/compare.py --runs-root <absolute path to runs>`.
2. Copy each seed's training `config.json` / `metrics.json` and decomposer `config.json` here as
   snapshots, and write `notes.md`.
3. Complete the exp-041 row in `experiments/log.md` from `compare_metrics.json`. Never type a
   number by hand.
