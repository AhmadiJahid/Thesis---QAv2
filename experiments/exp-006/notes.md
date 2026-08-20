# exp-006 -- pool sweep CPU precompute for the clustering strategy (Refs #14)

## What this run is, and is not

Issue #14's "done when" is the clustering strategy's **decomposition metrics** sitting next
to `imbalanced` and `balanced` in `experiments/log.md`, at pool size 2000, on the same dev
sample. That needs a GPU (the `mistral_7b_instruct` decomposer). At launch time the GPU was
unavailable for this work: another user's `ollama` held ~21.3 GiB of 24 GiB, and
`runs/run.lock` was held by `exp-004`/`exp-005`, whose detached waiter
(`runs/rearmed_wait_and_launch.sh`) owns the next GPU window (deadline 2026-08-21 09:00).

So this run covers **only the CPU-feasible half** of the plan: pool construction, bi-encoder
similarity, bi-encoder-only truncation, and cross-encoder rerank -- for all three
construction strategies (`imbalanced`, `balanced`, `clustered`) at size 2000, against one
shared dev sample. **No decomposition-quality metric exists yet.** The GPU half is queued as
a follow-up, not started here.

## Minimal run set (per the brief)

`(size=2000, balance) x (variant, mode)` for `balance in {imbalanced, balanced, clustered}`,
`variant in {biencoder_only, biencoder_plus_ce}`, `mode in {raw, typed, uniform}` -- 18
decompose+eval cells total, one trial each (`pool_trial_seeds: [42]`), all scored against the
same dev sample (`dev_seed 42`, `dev_per_hop 250`, 750 queries). Verified in `experiments/log.md`
before starting: no `pool_sweep` cell, in any strategy, has ever produced v2 metrics -- the
sweep has only ever `--dry-run`'d in this repo.

## How the CPU stages were run

Not through `scripts/pool_sweep_orchestrator.py` directly for the similarity/rerank/clustered
sample_pool stages -- confirmed via its own `--dry-run` that it forces `--device cuda` onto
those (from `configs/pool_sweep.json`'s top-level `"device": "cuda"`), unconditionally for
similarity/rerank and for the clustered pool. Running any of that on the GPU right now would
compete for the ~3 GiB of headroom that `exp-004`/`exp-005`'s waiter is watching for, which is
exactly what this run was told not to do.

Instead, the underlying child scripts (`MusiQue/scripts/sample_dev.py`, `sample_pool.py`,
`check_question_similarity.py`, `truncate_top20.py`, `rerank_similarity_results.py`) were
invoked directly, at their own committed defaults (`configs/musique_prep.json`,
`configs/similarity.json` -- verified equal to `pool_sweep.json`'s intended values for
`embed_model`, `top_k`, `k`/`rerank_k`, `cross_encoder`), with `--device cpu` explicit. This
is not an invented value: `configs/similarity.json`'s own `"device"` key, and
`configs/musique_prep.json`'s `sample_pool.clustering.device`, already default to `"cpu"` --
the orchestrator is what overrides them to `cuda`, not the other way around. ADR 0021 item 7
also notes that the *only* verified bit-reproducible k-means fit in this repo is the CPU one;
GPU embedding determinism is explicitly unverified. CPU is the more defensible device here,
not a fallback of convenience.

Every output landed at the exact path `pool_sweep_orchestrator.py`'s own `_pool_dir` /
`_sim_dir` / `_bi_top5_dir` / `_ce_dir` / `_dev_sample_path` helpers produce -- checked
against the orchestrator's own `--dry-run` printout for
`--only size2000_imbalanced,size2000_balanced,size2000_clustered`. A later
`--stage decompose` (or `--stage all`) run will find all of this precompute already present
and skip it via the orchestrator's existing skip-if-exists logic, going straight to the GPU
decomposer stage.

## What ran, and what it produced

All three cells completed with rc=0 on every stage (`fail=0` at the end of
`runs/exp-006/cpu_stages.log`), ~13.7 minutes wall clock (2026-08-20 21:09:17 -> 21:23:00
local time):

| balance | pool rows | pool hop split (2/3/4) | top20/top5-bi/top5-ce rows |
|---|---|---|---|
| imbalanced | 2000 | 1461 / 413 / 126 | 750 / 750 / 750 |
| balanced | 2000 | 667 / 667 / 666 | 750 / 750 / 750 |
| clustered | 2000 | 1396 / 495 / 109 | 750 / 750 / 750 |

Clustered pool diagnostics (from its own `stats.json`): k-means over 19,938 candidate rows at
k=2000, 11 iterations to converge, inertia 1803.46, **0 empty clusters, 0 rows from top-up**
(every cluster supplied its own representative), cluster sizes 1--56, mean distance to
centroid of the selected row 0.235. Embedding (`intfloat/e5-small-v2`, CPU) took 33.4 s; the
k-means fit itself took 25.4 s (single-threaded, per ADR 0021's reproducibility note).

The clustered pool's hop distribution (1396/495/109) falls out of the embedding geometry --
it is not stratified by hop (ADR 0021 item 10) -- and lands closer to `imbalanced`'s shape
than to `balanced`'s, which is worth remembering when reading the eventual decomposition
numbers: `clustered` and `imbalanced` will have shared some of the same hop-distribution
confound that `balanced` does not.

One flag for whoever runs the GPU follow-up: `sample_dev.py` reported
`typed=False, uniform=False` for every row of this dev sample -- it carries no
`question_masked_typed` / `question_masked_uniform` fields. This run did not evaluate whether
that degrades the `typed`/`uniform` retrieval modes; `check_question_similarity.py` ran
`--mode all` without erroring, but whether the typed/uniform outputs it produced are
meaningful (as opposed to falling back on empty/raw text) has not been checked and should be
before trusting those two modes' eventual decomposition numbers.

GPU memory was checked immediately before launch and immediately after completion:
21279 MiB used / 2972 MiB free both times, identical -- this run never touched the GPU.
`runs/run.lock` was not read, written, or otherwise touched.

## What remains (the GPU follow-up, not started here)

`scripts/pool_sweep_orchestrator.py --stage decompose --only size2000_imbalanced,size2000_balanced,size2000_clustered`
followed by `--stage eval` (or one `--stage all` pass) once `exp-004`/`exp-005` release
`runs/run.lock` and the GPU has real headroom again. That drives 18 `mistral_7b_instruct`
decomposer runs (3 balances x 2 retriever variants x 3 retrieval modes) plus their evaluator
passes, appending 18 rows to `runs/pool_sweep/summary/all_runs.csv`. Those rows -- not
anything in this note -- are what closes issue #14's "done when".

## Artifacts

- Log entry: `experiments/log.md`, `exp-006` (this commit and the follow-up completion edit).
- Config snapshot: `experiments/exp-006/config.json`.
- Metrics (precompute-only; no decomposition-quality numbers): `experiments/exp-006/metrics.json`.
- Raw run log: `runs/exp-006/cpu_stages.log`; launcher script `runs/exp-006/run_cpu_stages.sh`.
- Precompute artifacts (gitignored, stay on the box): `runs/pool_sweep/{dev_sample,pools,similarity,biencoder_top5,rerank}/...`.
