# exp-037 — count-first decomposition arm

**Status: PENDING, not run.** No metrics exist yet; `metrics.json` is written by
`build_metrics.py` after the runs.

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

Everything else is the baseline's: 4-bit NF4, the exp-014 retrieval artifact (the same
exemplars), `as_is` ordering, greedy decoding with `max_new_tokens` 1024, seed 42, the ADR 0039
step parser, and the pinned 600. Full details are in `config.json`.

| model | paired baseline | why this one |
|---|---|---|
| Mistral-7B-Instruct-v0.3 | exp-020, read through exp-032's re-parse (`runs/exp-032/exp-020/eval`) | the 4-bit unguided arm on this artifact. exp-015 is fp16. exp-020 was scored on the old parser, so the comparison has to use the re-parsed reading (ADR 0039). |
| Qwen3.5-9B | exp-027 (`runs/exp-032/exp-027/eval`, 0/600 rows changed by the re-parse) | the 4-bit unguided Qwen3.5-9B arm on this artifact. exp-005 uses the v1 pool, unquantized. |

**Hardware control.** The baselines ran on an RTX 3090 and exp-037 runs on Colab. Greedy
4-bit decoding is not guaranteed to be identical across GPUs or library builds, so the
baseline's own command is re-run on the same Colab GPU (`mistral_unguided_rerun`, and
optionally `qwen_unguided_rerun`). The decision rule was declared before any run: if the
control does not reproduce the stored baseline, the primary result is reported as
confounded by hardware, alongside the count-first vs re-run sensitivity pair.

**Statistics** (ADR 0009/0036, the exp-031/034 method). The evaluator's `--compare` runs
the paired bootstrap, exact McNemar and paired t. `build_metrics.py` applies Holm-Bonferroni
over the families fixed in `config.json`:

- F1: the primary pairs × the 8 report-card metrics, overall;
- F2: the primary pairs × {hop-count EM, under-decomposition} × hops 2/3/4.

Primary endpoints are declared in advance: under-decomposition and hop-count EM.

## How to run

**On Colab** (Jahid): open `notebooks/exp-037_count_first_colab.ipynb` on an L4 runtime and
run all cells. The first cell lists what to set up (HF token, branch pushed, archive on Drive).

**On any CUDA box with the repo venv**, from the repo root:

```bash
python scripts/colab_setup_data.py --archive <thesis-backup.tar.gz> --parts musique \
  --runs exp-014/retrieval/s2_pinned600_imbalanced_rerank_top5.jsonl exp-032/exp-020/eval exp-032/exp-027/eval
bash experiments/exp-037/run.sh mistral_count_first
bash experiments/exp-037/run.sh mistral_unguided_rerun     # hardware control
bash experiments/exp-037/run.sh qwen_count_first
bash experiments/exp-037/run.sh qwen_unguided_rerun        # optional
bash experiments/exp-037/run.sh compare
bash experiments/exp-037/run.sh metrics                    # -> experiments/exp-037/metrics.json
```

`run.sh` checks the retrieval artifact's sha256 before every decode and will not overwrite an
existing arm.

## Wall clock (ESTIMATE, not measured)

These are the measured baseline decodes on the RTX 3090:

- exp-020 (Mistral): 2,526 s;
- exp-027 (Qwen3.5-9B): 1,672 s.

4-bit decoding at batch size 1 is memory-bandwidth bound. An L4 has about a third of the
3090's bandwidth, which gives these **estimates**:

| step | L4 (estimate) |
|---|---|
| setup: pip install, archive copy, extract | ~15–25 min |
| Mistral run (count-first or re-run) | ~1.5–2.5 h each |
| Qwen run | ~1–1.5 h each |
| compare + metrics (CPU) | ~5 min |
| **default plan** (Mistral count-first + Mistral re-run + Qwen count-first) | **~4–6 h** |

An A100 should be at or below the 3090 times (~40–45 min per Mistral run), at a higher
compute-unit rate. Count-first adds about 5 tokens per prompt exemplar and per output, which
is negligible. The real numbers are written to `runs/exp-037/colab_steps.jsonl`.

## After the run

Copy `MyDrive/thesis-exp-037/runs/exp-037/` to the laptop's `runs/exp-037/` and
`MyDrive/thesis-exp-037/metrics.json` to `experiments/exp-037/metrics.json`. Then add each
arm's decomposer `config.json` snapshot and a `notes.md` here, and complete the exp-037 row
in `experiments/log.md`.
