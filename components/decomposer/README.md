# Decomposer Component

Turns a multi-hop question into ordered single-hop sub-questions, chained with `[#k]`
placeholders that refer to earlier answers. Ported from v1, with the four per-model
`decomposer.py` copies consolidated into one runner.

- `run_decomposer.py` — the runner, for every model.
- `models/<model_name>/prompt.md` (and `prompt_unguided.md` where v1 had one) — **byte-identical to v1**.
- `models/<model_name>/config.json` — model id, prompt style, generation, loader/quantization,
  few-shot and post-processing settings.
- `configs/decomposer.json` — seed, guided flag, embedding-model registry, retrieval defaults.

```bash
# few-shot examples from a reranked/truncated retrieval file
python components/decomposer/run_decomposer.py --model mistral_7b_instruct \
    --retrieval-input runs/pool_sweep/biencoder_top5/<cell>/top5_biencoder.jsonl \
    --retrieval-mode uniform --retrieval-k 5

# prompts only, no model load (works without a GPU)
python components/decomposer/run_decomposer.py --model qwen2_5_3b --dry-run
```

Two prompt styles, set per model by `prompt_style`:

- `plain` — the template is formatted directly.
- `chat_template` — the template is split on `chat_template.split_marker` into a system and a
  user half and rendered with the tokenizer's chat template (`enable_thinking` from config).

Few-shot examples come from the reranked retrieval file when `--retrieval-input` is given,
otherwise from masked-similarity selection over the committed exemplar pool, otherwise from a
seeded random draw. Models whose prompt carries its examples inline set `few_shot.enabled: false`.

Every run writes `results.json`, `config.json`, `metrics.json`, `notes.md` and a `prompts_log/`
under the configured runs root. The runner prints and asserts the parameter count against
`default_max_params` in `configs/model_limits.json` (~8B, per CLAUDE.md).

Every run also records **cost next to quality**: per row `prompt_tokens`, `completion_tokens`
and `latency_seconds` in `results.json`, and their per-query means/medians in the metrics
JSON under `cost`. A `--dry-run` generates nothing, so those are reported as unmeasured
rather than as zero.

Score the predictions with `scripts/musique_decompositions_evaluator.py` (MuSiQue gold) or
`scripts/evaluate_decompositions.py` (MetaQA KG execution).

## The fine-tuned arm (LoRA / QLoRA, issue #13)

`train_lora.py` trains a LoRA adapter on MuSiQue; `configs/finetune_decomposer.json` holds
everything else. Three training-data arms, selected with `--arm`:

| `--arm` | training data | evaluated on |
|---|---|---|
| `pool_2000` | 2000 examples, seeded draw spread across hop buckets from `datasets.musique_pool_enriched` | hops 2, 3, 4 |
| `full_train` | the full MuSiQue training split (`datasets.musique_train`) | hops 2, 3, 4 |
| `generalisation_2_3hop` | 2-hop and 3-hop rows only | hop 4 only |

```bash
# the whole data + prompt path on the fixtures, no weights (what the smoke test runs)
python components/decomposer/train_lora.py --arm pool_2000 --dry-run

# a real run (QLoRA 4-bit by default; adapters land under the gitignored runs/ root)
python components/decomposer/train_lora.py --arm pool_2000

# inference with the adapter — zero-shot, because that is the prompt it was trained on
python components/decomposer/run_decomposer.py --model mistral_7b_instruct \
    --adapter runs/finetune_decomposer/pool_2000/mistral_7b_instruct/<run>/adapter \
    --no-few-shot --retrieval-input <the 600-question evaluation-set file>

# quality + significance + cost for both arms, in one table
python scripts/compare_decomposer_arms.py \
    --arm prompting=runs/decomposer/mistral_7b_instruct/<run> \
    --arm finetuned_pool_2000=runs/decomposer/mistral_7b_instruct/<run> \
    --baseline prompting
```

Guarantees that are hard failures, not warnings:

- **Training ids are asserted disjoint from the ADR 0007 evaluation ids** (all 600, whatever
  hops the arm is evaluated on), naming the offenders.
- **The base model is asserted against the ~8B ceiling** before any step, and the
  LoRA-wrapped model again after the adapter is attached. `run_decomposer.py --adapter`
  asserts base + adapter.
- Adapters, checkpoints and prediction dumps stay under `runs/`; only config, metrics and the
  run note enter git.

The training prompt is the model's own `prompt_unguided.md` with the few-shot block empty,
built through `run_decomposer.py`'s template helpers so the trained string and the inference
string cannot drift. Evaluation is always
`scripts/musique_decompositions_evaluator.py` — string-level, no model in the loop, and no
commercial API rating a decomposition (CLAUDE.md).
