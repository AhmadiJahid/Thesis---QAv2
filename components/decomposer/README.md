# Decomposer Component

Turns a multi-hop question into ordered single-hop sub-questions, chained with `[#k]`
placeholders that refer to earlier answers. Ported from v1, with the four per-model
`decomposer.py` copies consolidated into one runner.

- `run_decomposer.py` — the runner, for every model.
- `models/<model_name>/prompt.md` (and `prompt_unguided.md` where v1 had one) — **byte-identical to v1**.
- `models/<model_name>/config.json` — model id, prompt style, generation, loader/quantization,
  few-shot and post-processing settings.
- `configs/decomposer.json` — MetaQA: seed, guided flag, embedding-model registry, retrieval defaults.
- `configs/decomposer_musique.json` — the MuSiQue variant (issue #12): the pinned 600-question
  evaluation set of ADR 0007, hops 2/3/4, and the three run conditions.

```bash
# few-shot examples from a reranked/truncated retrieval file
python components/decomposer/run_decomposer.py --model mistral_7b_instruct \
    --retrieval-input runs/pool_sweep/biencoder_top5/<cell>/top5_biencoder.jsonl \
    --retrieval-mode uniform --retrieval-k 5

# prompts only, no model load (works without a GPU)
python components/decomposer/run_decomposer.py --model qwen2_5_3b --dry-run

# one arm of the guided-vs-unguided comparison
python components/decomposer/run_decomposer.py --model mistral_7b_instruct \
    --config decomposer_musique.json --condition oracle_guided
```

## Run conditions (`configs/decomposer_musique.json`)

`--condition` selects a named block from the config's `conditions`; the config's
`condition` key is the default when the flag is omitted. A condition may set only `guided`
and `stop_after_step_lines` — model, seed and decoding are shared, and the runner rejects a
condition that tries to move them, so the arms cannot drift apart.

| condition | prompt | generation |
|---|---|---|
| `unguided` | no hop count | `generation_overrides.max_new_tokens` only |
| `oracle_guided` | gold hop count of the question's source file | same |
| `unguided_capped` | no hop count | stops after `stop_after_step_lines` step lines (default 8) |

The cap is a `StoppingCriteria` on completed newline-delimited step lines, with a trim for
the partial line the criterion can leave behind. The unguided arms require a model folder
with an `unguided_prompt_file`: without one the guided prompt is reused and `{hop_count}`
is filled with the placeholder, which is not the unguided condition. The config sets
`unguided_prompt_must_omit_hop_count`, so that combination is refused rather than run.
`configs/decomposer.json` does not set it and the MetaQA behaviour is unchanged.

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

Score the predictions with `scripts/musique_decompositions_evaluator.py` (MuSiQue gold) or
`scripts/evaluate_decompositions.py` (MetaQA KG execution).
