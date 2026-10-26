# exp-030: launch commands for the issue #56 full-dev runs (not yet run)

This file contains commands only. No GPU run below has happened. Write the `experiments/log.md`
row before launching (CLAUDE.md). See ADR 0038 for the design.

Run every command from the repo root of the checkout that holds the merged code. All paths are
relative to that root. Each arm's command is copied from its committed record:
`experiments/exp-0NN/config.json` for exp-020 and exp-026, and `experiments/exp-0NN/notes.md` for
exp-021, exp-022 and exp-027. Three things change in every command, and nothing else:

- `--retrieval-input runs/exp-030/retrieval/dev_full_imbalanced_rerank_top5.jsonl` replaces the
  exp-014 pinned-600 artifact.
- `--eval-set musique_dev_full` is added.
- `--output-root` moves under `runs/exp-030/<arm>/`.

Prefix every command with `PYTHONUNBUFFERED=1`. Without it the startup dump does not reach a
redirected log until the run ends (exp-021 note).

## The retrieval artifact (built, CPU)

| item | value |
|---|---|
| builder | `.venv/bin/python scripts/build_dev_full_retrieval.py` (`configs/dev_full_retrieval.json`) |
| artifact | `runs/exp-030/retrieval/dev_full_imbalanced_rerank_top5.jsonl`, sha256 `98e4f7201b0b11cabfd9d1df9ee154464ebc230b7296c27be98ae02cd1efa114` |
| rows | 2,417 (1,252 / 760 / 405); ids match `musique_dev_full` exactly |
| pool | `size2000_imbalanced`, sha256 `cc3646625d75…6646`, 2,000 rows |
| pool ∩ dev | **0** ids, 0 normalized question texts |
| equivalence vs exp-014 (`3f95c7dc…fdb0`) | **589/600** rows have the same 5 exemplar ids in the same order. All 11 mismatches have an identical rerank-score sequence, so every difference comes from exact ties. 7 of the 11 only reorder tied exemplars within the same top-5 set. 4 have a different top-5 set because a tie straddles the rank-5 cutoff: `2hop__115515_779396`, `2hop__554167_451128`, `2hop__765180_123956` and `2hop__168816_144857`. The lead accepted this as is (no splicing). Masked query text matches 600/600. The first 3 ids match 597/600 (not the prompt's count: reranked prompts show k=5 exemplars per the prompts_log header; the run notes' `k=3` is the issue #53 mislabel); the 3 that differ are `2hop__226204_69048`, `2hop__263638_69048` and `4hop2__161602_474028_88460_21034` |
| wall clock | similarity 290.5 s, rerank 172.7 s (CPU, built at commit `8ba9b5a`, the pre-rebase SHA of this lane's first commit; read from the build's stdout) |

The builder exits non-zero because its equivalence verdict is strict (589 is not 600).

**Where the build trail lives.** The artifact was built under `runs/exp-029/` and moved to
`runs/exp-030/` when this lane was renumbered. The sha256 is unchanged after the move. The build's
own trail is the two sub-runs' trails:

- `runs/exp-030/retrieval/similarity/similarity_{config,metrics,notes}`: 2,417 rows written, commit
  `8ba9b5a`, clean tree;
- `runs/exp-030/retrieval/rerank/rerank_{config,metrics,notes}`: 2,417 rows.

They still name the old `runs/exp-029/...` paths, because they were written before the move. The
top-level `build_*` record was overwritten by an earlier `--verify-only` pass. That bug is fixed:
`--verify-only` now writes `verify_*`. The overwritten record is kept as `verify_at_d13ea6e_*`. The
current verification is `runs/exp-030/retrieval/verify_metrics.json`.

## Decompose (GPU), fastest first

The time estimates are each arm's committed `cost.total_generation_seconds` on 600 questions,
multiplied by 2417/600 = 4.03. They are estimates, not measurements.

```bash
# 1. Qwen2.5-7B 4-bit (exp-021)  ~55 min
PYTHONUNBUFFERED=1 .venv/bin/python components/decomposer/run_decomposer.py --model qwen2_5_7b_instruct \
  --config decomposer_musique.json --condition unguided --seed 42 --quantization 4bit \
  --retrieval-input runs/exp-030/retrieval/dev_full_imbalanced_rerank_top5.jsonl \
  --eval-set musique_dev_full --output-root runs/exp-030/qwen2_5_7b/decomposer

# 2. Pool-only fine-tune (exp-026): fp16 base + adapter, NO few-shot block (as exp-026)  ~57 min
PYTHONUNBUFFERED=1 .venv/bin/python components/decomposer/run_decomposer.py --model mistral_7b_instruct \
  --config decomposer_musique.json --condition unguided --seed 42 --quantization none \
  --adapter runs/exp-026/20260920_024535/adapter --no-few-shot \
  --retrieval-input runs/exp-030/retrieval/dev_full_imbalanced_rerank_top5.jsonl \
  --eval-set musique_dev_full --output-root runs/exp-030/pool_ft/decomposer

# 3. Qwen2.5-14B 4-bit (exp-022)  ~1 h 51 min
PYTHONUNBUFFERED=1 .venv/bin/python components/decomposer/run_decomposer.py --model qwen2_5_14b_instruct \
  --config decomposer_musique.json --condition unguided --seed 42 --quantization 4bit \
  --retrieval-input runs/exp-030/retrieval/dev_full_imbalanced_rerank_top5.jsonl \
  --eval-set musique_dev_full --output-root runs/exp-030/qwen2_5_14b/decomposer

# 4. Qwen3.5-9B 4-bit (exp-027)  ~1 h 51 min
PYTHONUNBUFFERED=1 .venv/bin/python components/decomposer/run_decomposer.py --model qwen3_5_9b \
  --config decomposer_musique.json --condition unguided --seed 42 --quantization 4bit \
  --retrieval-input runs/exp-030/retrieval/dev_full_imbalanced_rerank_top5.jsonl \
  --eval-set musique_dev_full --output-root runs/exp-030/qwen3_5_9b/decomposer

# 5. Mistral-7B 4-bit (exp-020)  ~2 h 46 min
PYTHONUNBUFFERED=1 .venv/bin/python components/decomposer/run_decomposer.py --model mistral_7b_instruct \
  --config decomposer_musique.json --condition unguided --seed 42 --quantization 4bit \
  --retrieval-input runs/exp-030/retrieval/dev_full_imbalanced_rerank_top5.jsonl \
  --eval-set musique_dev_full --output-root runs/exp-030/mistral_7b/decomposer
```

Total estimate: about 9 h 20 min.

Before each launch, check the startup dump in the log. Expected: `Evaluation set: {"named_eval_set":
"musique_dev_full", ... "matches_named_set": true ... "rows_loaded_per_hop": {"2": 1252, "3": 760,
"4": 405}`, the retrieval `input_sha256` `98e4f720…a114`, and `few_shot_ordering` `as_is`.

## Evaluate (CPU), per arm

```bash
.venv/bin/python scripts/musique_decompositions_evaluator.py \
  --predictions runs/exp-030/<arm>/decomposer/<run_id>/results.json \
  --run-dir runs/exp-030/<arm>/eval --out-prefix eval --seed 42 --eval-set musique_dev_full
```

## Resample and plot (CPU)

```bash
.venv/bin/python scripts/resample_eval_sets.py \
  --arm "Qwen2.5-7B=runs/exp-030/qwen2_5_7b/eval/eval_per_item.json" \
  --arm "Pool-FT (Mistral)=runs/exp-030/pool_ft/eval/eval_per_item.json" \
  --arm "Qwen2.5-14B=runs/exp-030/qwen2_5_14b/eval/eval_per_item.json" \
  --arm "Qwen3.5-9B=runs/exp-030/qwen3_5_9b/eval/eval_per_item.json" \
  --arm "Mistral-7B=runs/exp-030/mistral_7b/eval/eval_per_item.json" \
  --committed-pinned "Qwen2.5-7B=runs/exp-021/eval/eval_metrics.json" \
  --committed-pinned "Pool-FT (Mistral)=runs/exp-026/eval/eval_metrics.json" \
  --committed-pinned "Qwen2.5-14B=runs/exp-022/eval/eval_metrics.json" \
  --committed-pinned "Qwen3.5-9B=runs/exp-027/eval/eval_metrics.json" \
  --committed-pinned "Mistral-7B=runs/exp-020/eval/eval_metrics.json" \
  --out-dir runs/exp-030/resample
```

Do not pass `--allow-unnamed-population` here. Without it the resampler refuses unless every arm
was scored with `--eval-set musique_dev_full` (`matches_named_set` true) and the population equals
the set's 2,417 ids. The diamond in the figures is the pinned-600 subset recomputed from these
runs. `committed_pinned` in `resample_metrics.json` gives each arm's committed pinned value and the
difference from the recomputed one.

Commit `resample_metrics.json`, `resample_summary.csv`, `resample_notes.md` and the figures under
`experiments/exp-030/`. Keep `resample_per_draw.csv` and the decomposer outputs in `runs/`.

## Known caveats to carry into the log row

- **Determinism check on the pinned 600.** The full-dev outputs for the pinned ids can be compared
  with each arm's committed pinned run. Before counting agreement, note that 3/600 prompts differ
  because of exact rerank ties (see above). Only the pool-FT arm, which uses no
  few-shot block, has identical prompts on all 600.
- **The SD is shrunk by the finite-population correction.** Draws are without replacement from
  finite hop strata, so the SD is scaled by sqrt(1 − n/N) ≈ 0.92 / 0.86 / 0.71 at 2 / 3 / 4 hops.
  `sd_fpc_corrected` reports the SD with that correction removed. Overlap between draws only
  correlates the K draws (ADR 0038 item 4).
- **The gold join is by question text (an existing evaluator behaviour, unchanged here; issue #57).** The dev
  split has 6 pairs of rows (12 rows) with identical question text. In 5 of those pairs the gold
  decompositions differ, and the first row of each pair is scored against its twin's gold. One of these
  rows is in the pinned 600 (`3hop2__127483_125104_10557`, scored against
  `3hop2__125104_127483_10557`'s gold). It affects every committed pinned run equally.
