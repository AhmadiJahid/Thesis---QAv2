# exp-040 — query masking (raw / uniform / typed) on the v2 pipeline

**Status: PENDING, not run.** Nothing here is a result yet. Decisions: ADR 0046. Plan, commands,
thresholds and Holm families: `config.json` (committed before any run).

## Why

The thesis's masking result (typed > uniform > raw) comes from v1. It was never re-run in v2, and
none of its 15 tests survives Holm (smallest adjusted p 0.168,
`docs/analysis/2026-08-20-v1-masking-and-retrieval-significance.json`). exp-040 measures the three
masking modes on the current fixed method, so the claim can be stated or withdrawn using v2 numbers
(writing audit (d), rank 1).

## What moves, what is held

**One factor moves:** which form of the question the few-shot retriever matches on
(`--retrieval-mode`):

- `raw` matches the raw question;
- `uniform` replaces each NER entity with `[MASK]`;
- `typed` replaces each entity with its type, e.g. `[PLACE]`.

Queries are masked at lookup time and the pool's masked fields are precomputed (ADR 0003). The
prompt always shows the raw query and the raw exemplar questions, so the only thing masking
changes is **which 5 exemplars** the model sees.

**Held from exp-020:**

- Mistral-7B-Instruct-v0.3, 4-bit NF4, `unguided`;
- pool size2000_imbalanced (sha256 `cc364662…6646`);
- e5-small-v2 top-20, then ms-marco-MiniLM-L-12-v2 top-5, on CPU;
- `as_is` ordering, greedy decoding, `max_new_tokens` 1024, seed 42;
- the ADR 0039 parser and the pinned 600.

## Artifacts

exp-014's artifact (`3f95c7dc…fdb0`, the input to exp-020) was built with `--mode typed` and holds
only `typed_top_k`. The notebook's **retrieval stage** reruns exp-014's two commands with three
changes: `--mode all`, `--no-cache`, and exp-040 output paths. `check_retrieval.py` then checks the
result: 600 ids equal to exp-014's, 200 per hop, and ≥ 5 candidates per mode. It also **pins the
sha256** on Drive (`runs/exp-040/retrieval/PINNED_SHA256`), and every later step refuses a
different file.

The check also reports for how many queries the rebuild's typed exemplars equal exp-014's (ordered
pool ids). That count decides the typed comparator, a rule declared before the run:

- **600/600:** `typed_rerun` (exp-020's exact command on exp-014's artifact) is the typed arm. Its
  prompts are byte-identical to typed mode on the rebuilt artifact.
- **Otherwise:** the notebook also runs `typed_rebuilt` (typed mode on the rebuilt artifact). That
  arm becomes the typed side of the primary pairs, and `typed_rerun` is kept as context and as the
  hardware control.

## Comparisons (fixed before any run)

The primary comparator is a same-machine re-run, following the exp-037 / ADR 0043 item 7
convention. Greedy 4-bit output can differ across GPUs, so all arms run on the same Colab GPU and
the stored exp-020 is only a hardware control.

| family | pairs | strata | size |
|---|---|---|---|
| F1_primary_overall | typed − uniform, typed − raw | overall | 14 |
| F2_primary_per_hop | typed − uniform, typed − raw | hops 2/3/4 | 42 |

Each family covers 7 report-card terms: break EM, SARI, GED, chain validity, hop-count EM, under-
and over-decomposition. Holm's p is exact McNemar for the binary terms and paired t for the
continuous ones. The bootstrap CI is reported beside it, and Holm sits beside the uncorrected
verdict (ADR 0009, 0036). No direction is pre-declared.

The following are reported uncorrected and in no family:

- uniform − raw;
- typed_rerun − stored exp-020 (read through exp-032's re-parse). This is the hardware control:
  "reproduced" only if 600/600 decompositions are identical strings;
- typed_rebuilt − typed_rerun, if typed_rebuilt ran.

## How to run

**On Colab** (the runner session or Jahid): open `notebooks/exp-040_masking_v2_colab.ipynb` on an
A100 and run all cells. It needs the Colab secret `HF_TOKEN` (or `HF_TOKEN` set in the environment)
and `MyDrive/thesis-backup.tar.gz`.

- It clones the public repo, and the first launch pins the commit in
  `MyDrive/thesis-exp-040/launch_sha.txt`.
- It builds the exp-026 environment (Python 3.13.9 via uv, `requirements.txt` pins) and refuses on
  version drift.
- It extracts data with `scripts/colab_setup_data.py` from a private, atomically copied archive.
- It links `runs/exp-040` in the Colab clone to Drive.
- Before each arm's full decode it runs a **20-question canary** and the output-health gate. The
  gate is checked again on the full 600 before evaluation, with these thresholds: rows = expected,
  at most 1% empty, not all empty, at most 5% at `max_new_tokens`. A failed gate stops the arm and
  writes `HEALTH_FAILED.json`.
- Decodes resume via `run_decomposer.py --resume` from `results.partial.jsonl`. If the runtime
  dies, run all cells again.

**On any CUDA box**, from a clean checkout with the stored runs restored (see the `run.sh` header):

```bash
bash experiments/exp-040/run.sh retrieval
for a in typed_rerun raw uniform; do bash experiments/exp-040/run.sh canary $a; bash experiments/exp-040/run.sh $a; done
# only if runs/exp-040/retrieval/check.json typed_identity.identical < 600:
bash experiments/exp-040/run.sh canary typed_rebuilt; bash experiments/exp-040/run.sh typed_rebuilt
bash experiments/exp-040/run.sh compare      # -> experiments/exp-040/compare_metrics.json
```

## GPU time: ESTIMATE for an A100, not measured

The bases are measured on the 3090:

- exp-020 decode: 2,526 s, peak 4.8 GiB;
- exp-014 typed build: 62.5 s + 51.5 s on 64 CPU threads.

The rebuild covers 3 modes on Colab's ~12 vCPUs.

| step | A100 (estimate) |
|---|---|
| setup (venv, archive, model download) | 15–25 min |
| retrieval rebuild (CPU) | 10–25 min |
| 3 canaries | 6–12 min |
| each full decode | 30–45 min |
| each evaluation (CPU) | 2–6 min |
| **total, three arms** | **~2.3–3.4 h** |
| + typed_rebuilt, only if needed | +0.55–0.85 h |

A GPU shared with another notebook is slower. Measured times are written to
`runs/exp-040/colab_steps.jsonl`.

## After the run

1. Copy `MyDrive/thesis-exp-040/runs/exp-040/` to the laptop's `runs/exp-040/` and
   `compare_metrics.json` to `experiments/exp-040/`. To recompute on the laptop, run
   `python experiments/exp-040/compare.py --runs-root <absolute path to runs>`.
2. Copy each arm's decomposer `config.json` here as a snapshot and write `notes.md`.
3. Complete the exp-040 row in `experiments/log.md` from `compare_metrics.json`. Never type a
   number by hand.
