# exp-029: launch commands for the issue #56 full-dev runs (not yet run)

This file contains commands only. No GPU run below has happened. Write the `experiments/log.md`
row before launching (CLAUDE.md). See ADR 0035 for the design.

Run every command from the repo root of the checkout that holds the merged code. All paths are
relative to that root. Each arm's command is copied from its committed record:
`experiments/exp-0NN/config.json` for exp-020 and exp-026, and `experiments/exp-0NN/notes.md` for
exp-021, exp-022 and exp-027. Three things change in every command, and nothing else:

- `--retrieval-input runs/exp-029/retrieval/dev_full_imbalanced_rerank_top5.jsonl` replaces the
  exp-014 pinned-600 artifact.
- `--eval-set musique_dev_full` is added.
- `--output-root` moves under `runs/exp-029/<arm>/`.

Prefix every command with `PYTHONUNBUFFERED=1`. Without it the startup dump does not reach a
redirected log until the run ends (exp-021 note).

## The retrieval artifact (built, CPU)

| item | value |
|---|---|
| builder | `.venv/bin/python scripts/build_dev_full_retrieval.py` (`configs/dev_full_retrieval.json`) |
| artifact | `runs/exp-029/retrieval/dev_full_imbalanced_rerank_top5.jsonl`, sha256 `98e4f7201b0b11cabfd9d1df9ee154464ebc230b7296c27be98ae02cd1efa114` |
| rows | 2,417 (1,252 / 760 / 405); ids match `musique_dev_full` exactly |
| pool | `size2000_imbalanced`, sha256 `cc3646625d75…6646`, 2,000 rows |
| pool ∩ dev | **0** ids, 0 normalized question texts |
| equivalence vs exp-014 (`3f95c7dc…fdb0`) | **589/600** rows have the same 5 exemplar ids in the same order. All 11 mismatches have an identical rerank-score sequence, so the only change is the order of exactly tied exemplars. Masked query text matches 600/600. The first 3 ids (what a k=3 prompt shows) match 597/600; the 3 that differ are `2hop__226204_69048`, `2hop__263638_69048` and `4hop2__161602_474028_88460_21034` |
| wall clock | similarity 290.5 s, rerank 172.7 s (CPU, built at commit `8ba9b5a`) |

The builder exits non-zero because its equivalence verdict is strict (589 is not 600). That
verdict is recorded in `runs/exp-029/retrieval/build_metrics.json`.

## Decompose (GPU), fastest first

The time estimates are each arm's committed `cost.total_generation_seconds` on 600 questions,
multiplied by 2417/600 = 4.03. They are estimates, not measurements.

```bash
# 1. Qwen2.5-7B 4-bit (exp-021)  ~55 min
PYTHONUNBUFFERED=1 .venv/bin/python components/decomposer/run_decomposer.py --model qwen2_5_7b_instruct \
  --config decomposer_musique.json --condition unguided --seed 42 --quantization 4bit \
  --retrieval-input runs/exp-029/retrieval/dev_full_imbalanced_rerank_top5.jsonl \
  --eval-set musique_dev_full --output-root runs/exp-029/qwen2_5_7b/decomposer

# 2. Pool-only fine-tune (exp-026): fp16 base + adapter, NO few-shot block (as exp-026)  ~57 min
PYTHONUNBUFFERED=1 .venv/bin/python components/decomposer/run_decomposer.py --model mistral_7b_instruct \
  --config decomposer_musique.json --condition unguided --seed 42 --quantization none \
  --adapter runs/exp-026/20260920_024535/adapter --no-few-shot \
  --retrieval-input runs/exp-029/retrieval/dev_full_imbalanced_rerank_top5.jsonl \
  --eval-set musique_dev_full --output-root runs/exp-029/pool_ft/decomposer

# 3. Qwen2.5-14B 4-bit (exp-022)  ~1 h 51 min
PYTHONUNBUFFERED=1 .venv/bin/python components/decomposer/run_decomposer.py --model qwen2_5_14b_instruct \
  --config decomposer_musique.json --condition unguided --seed 42 --quantization 4bit \
  --retrieval-input runs/exp-029/retrieval/dev_full_imbalanced_rerank_top5.jsonl \
  --eval-set musique_dev_full --output-root runs/exp-029/qwen2_5_14b/decomposer

# 4. Qwen3.5-9B 4-bit (exp-027)  ~1 h 51 min
PYTHONUNBUFFERED=1 .venv/bin/python components/decomposer/run_decomposer.py --model qwen3_5_9b \
  --config decomposer_musique.json --condition unguided --seed 42 --quantization 4bit \
  --retrieval-input runs/exp-029/retrieval/dev_full_imbalanced_rerank_top5.jsonl \
  --eval-set musique_dev_full --output-root runs/exp-029/qwen3_5_9b/decomposer

# 5. Mistral-7B 4-bit (exp-020)  ~2 h 46 min
PYTHONUNBUFFERED=1 .venv/bin/python components/decomposer/run_decomposer.py --model mistral_7b_instruct \
  --config decomposer_musique.json --condition unguided --seed 42 --quantization 4bit \
  --retrieval-input runs/exp-029/retrieval/dev_full_imbalanced_rerank_top5.jsonl \
  --eval-set musique_dev_full --output-root runs/exp-029/mistral_7b/decomposer
```

Total estimate: about 9 h 20 min.

Before each launch, check the startup dump in the log. Expected: `Evaluation set: {"named_eval_set":
"musique_dev_full", ... "matches_named_set": true ... "rows_loaded_per_hop": {"2": 1252, "3": 760,
"4": 405}`, the retrieval `input_sha256` `98e4f720…a114`, and `few_shot_ordering` `as_is`.

## Evaluate (CPU), per arm

```bash
.venv/bin/python scripts/musique_decompositions_evaluator.py \
  --predictions runs/exp-029/<arm>/decomposer/<run_id>/results.json \
  --run-dir runs/exp-029/<arm>/eval --out-prefix eval --seed 42 --eval-set musique_dev_full
```

## Resample and plot (CPU)

```bash
.venv/bin/python scripts/resample_eval_sets.py \
  --arm "Qwen2.5-7B=runs/exp-029/qwen2_5_7b/eval/eval_per_item.json" \
  --arm "Pool-FT (Mistral)=runs/exp-029/pool_ft/eval/eval_per_item.json" \
  --arm "Qwen2.5-14B=runs/exp-029/qwen2_5_14b/eval/eval_per_item.json" \
  --arm "Qwen3.5-9B=runs/exp-029/qwen3_5_9b/eval/eval_per_item.json" \
  --arm "Mistral-7B=runs/exp-029/mistral_7b/eval/eval_per_item.json" \
  --out-dir runs/exp-029/resample
```

Commit `resample_metrics.json`, `resample_summary.csv`, `resample_notes.md` and the figures under
`experiments/exp-029/`. Keep `resample_per_draw.csv` and the decomposer outputs in `runs/`.

## Known caveats to carry into the log row

- **Determinism check on the pinned 600.** The full-dev outputs for the pinned ids can be compared
  with each arm's committed pinned run. Before counting agreement, note that 3/600 prompts differ
  because exactly tied exemplars are reordered (see above). Only the pool-FT arm, which uses no
  few-shot block, has identical prompts on all 600.
- **The gold join is by question text (an existing evaluator behaviour, unchanged here).** The dev
  split has 6 pairs of rows (12 rows) with identical question text. In 5 of those pairs the gold
  decompositions differ, and the first row of each pair is scored against its twin's gold. One of these
  rows is in the pinned 600 (`3hop2__127483_125104_10557`, scored against
  `3hop2__125104_127483_10557`'s gold). It affects every committed pinned run equally.
