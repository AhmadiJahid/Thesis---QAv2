# exp-037 — count-first decomposition arm

**Status: PENDING, not run.** No metrics exist yet; `metrics.json` is written by
`build_metrics.py` after the runs. Approved by Jahid in session on 2026-11-27 (memo C1.3);
see ADR 0043.

## What it tests

The memo at the repo root (section C1.3, "Hop-count-first structured output") proposes making
the decomposer state its hop count before writing steps, aimed at under-decomposition at
4 hops. exp-037 runs that as the `count_first` condition in `configs/decomposer_musique.json`:

- each of the 5 few-shot exemplars shows `[N hops]` as the first line of its decomposition,
  where N is the step count of that exemplar's own gold decomposition;
- one rule line is added to the unguided prompt:
  `- Before the steps, write the number of steps on its own line as [N hops].`;
- the first line the model writes is parsed as its self-predicted count
  (`src/count_first.py`) and removed before the decomposition is recorded, so the report
  card scores only the steps. Rows whose count line is missing or malformed are kept and
  scored, and counted in `metrics.json` under `count_first.status_counts`.

The count is the model's own prediction. Oracle guidance was exp-004/005 and answers a
different question.

Everything else is the unguided arm's: 4-bit NF4, the exp-014 retrieval artifact (the same
exemplars), `as_is` ordering, greedy decoding with `max_new_tokens` 1024, seed 42, the ADR 0039
step parser, and the pinned 600. Full details are in `config.json`.

## Comparators (declared before any run)

| model | PRIMARY comparator | SECONDARY comparator |
|---|---|---|
| Mistral-7B-Instruct-v0.3 | `mistral_unguided_rerun`: exp-020's exact command, re-run on the same Colab GPU | stored exp-020, read through exp-032's re-parse (ADR 0039 cross-parser rule) |
| Qwen3.5-9B | `qwen_unguided_rerun`: exp-027's exact command, re-run on the same Colab GPU | stored exp-027 (exp-032's re-parse changed 0/600 rows) |

Why these baselines: exp-020 and exp-027 are the 4-bit unguided arms on this retrieval
artifact. exp-015 is Mistral at fp16, and exp-005 is Qwen3.5-9B on the v1 pool, unquantized;
either would change a second factor.

Why the re-run is primary: the stored baselines ran on an RTX 3090, while exp-037 runs on
Colab. Greedy 4-bit decoding is not guaranteed to be identical across GPUs or library builds.
In the primary pair, both arms run on the same machine, environment and commit, so only the
condition differs.

**Hardware control.** Both re-runs are mandatory. A re-run counts as **reproduced** only if
its recorded `decomposition` equals the stored baseline's re-parsed `decomposition` as an
exact string for **600/600** query ids. Anything less is not reproduced, and that model's
secondary pair is then labelled hardware-confounded. The primary pairs are unaffected either
way.

**Statistics** (ADR 0009/0036, the exp-031/034 method). The evaluator's `--compare` runs
the paired bootstrap, exact McNemar and paired t. `build_metrics.py` applies Holm-Bonferroni
over the families fixed in `config.json`:

- F1: the 2 primary pairs × the 8 report-card metrics, overall (16 tests);
- F2: the 2 primary pairs × {hop-count EM, under-decomposition} × hops 2/3/4 (12 tests).

Primary endpoints are declared in advance: under-decomposition and hop-count EM.

## How to run

**On Colab** (Jahid): open `notebooks/exp-037_count_first_colab.ipynb` on an L4 runtime and
run all cells. The first cell lists what to set up (HF token, branch pushed, archive on
Drive).

`runs/exp-037` is a symlink into `MyDrive/thesis-exp-037/`, so every finished row reaches
Drive as it completes. If the runtime dies, start a new runtime and run all cells again: each
decode step resumes where it stopped.

**On any CUDA box with the repo venv**, from the repo root of a clean checkout:

```bash
python scripts/colab_setup_data.py --archive <thesis-backup.tar.gz> --parts musique \
  --runs exp-014/retrieval/s2_pinned600_imbalanced_rerank_top5.jsonl \
         exp-032/exp-020/eval exp-032/exp-020/reparse exp-032/exp-027/eval exp-032/exp-027/reparse
bash experiments/exp-037/run.sh mistral_count_first
bash experiments/exp-037/run.sh mistral_unguided_rerun
bash experiments/exp-037/run.sh qwen_count_first
bash experiments/exp-037/run.sh qwen_unguided_rerun
bash experiments/exp-037/run.sh compare
bash experiments/exp-037/run.sh metrics        # -> experiments/exp-037/metrics.json
```

`run.sh` refuses a tree whose tracked files differ from the commit. It also checks the
retrieval artifact's sha256 before every decode. Each decode step resumes an interrupted run
directory with `run_decomposer.py --resume`, which refuses unless the commit, effective
prompt, seed, retrieval sha256, decoding and query ids all match. Finished decodes and evals
are never redone.

## Wall clock (ESTIMATE, not measured)

These are the measured baseline decodes on the RTX 3090:

- exp-020 (Mistral): 2,526 s;
- exp-027 (Qwen3.5-9B): 1,672 s.

4-bit decoding at batch size 1 is memory-bandwidth bound. An L4 has about a third of the
3090's bandwidth, which gives these **estimates**:

| step | L4 (estimate) |
|---|---|
| setup: pip install, archive copy, extract | ~15–25 min |
| each Mistral run (count-first, re-run) | ~1.5–2.5 h |
| each Qwen run (count-first, re-run) | ~1–1.5 h |
| compare + metrics (CPU) | ~5 min |
| **all four runs** | **~5–8 h** |

An A100 should be at or below the 3090 times (~40–45 min per Mistral run), at a higher
compute-unit rate. The real numbers are written to `runs/exp-037/colab_steps.jsonl`.

## After the run

Copy `MyDrive/thesis-exp-037/runs/exp-037/` to the laptop's `runs/exp-037/` and
`MyDrive/thesis-exp-037/metrics.json` to `experiments/exp-037/metrics.json`. Then add each
arm's decomposer `config.json` snapshot and a `notes.md` here, and complete the exp-037 row
in `experiments/log.md`.
