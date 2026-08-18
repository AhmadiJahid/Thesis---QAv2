# Decomposer Component

Turns a multi-hop question into ordered single-hop sub-questions, chained with `[#k]`
placeholders that refer to earlier answers. Ported from v1, with the four per-model
`decomposer.py` copies consolidated into one runner.

- `run_decomposer.py` — the runner, for every model.
- `models/<model_name>/prompt.md` (and `prompt_unguided.md` where v1 had one) — **byte-identical to v1**.
- `models/<model_name>/config.json` — model id, prompt style, generation, loader/quantization,
  few-shot and post-processing settings.
- `configs/decomposer.json` — seed, guided flag, embedding-model registry, retrieval defaults;
  the MetaQA default (hops 1–3, question files one per line).
- `configs/decomposer_musique.json` — the MuSiQue variant for issue #12: hops 2–4, the MuSiQue
  question files (JSONL), the ADR 0007 evaluation set through bi-encoder top-20 + cross-encoder
  rerank to top-5, and the three named conditions.

```bash
# few-shot examples from a reranked/truncated retrieval file
python components/decomposer/run_decomposer.py --model mistral_7b_instruct \
    --retrieval-input runs/pool_sweep/biencoder_top5/<cell>/top5_biencoder.jsonl \
    --retrieval-mode uniform --retrieval-k 5

# prompts only, no model load (works without a GPU)
python components/decomposer/run_decomposer.py --model qwen2_5_3b --dry-run

# issue #12: one arm of the MuSiQue comparison (unguided | oracle_guided | unguided_capped)
python components/decomposer/run_decomposer.py --model mistral_7b_instruct \
    --config decomposer_musique.json --condition oracle_guided
```

## Conditions (issue #12)

`--condition` picks a named entry of the config's `conditions` block; everything else — model,
decoding, seed, retrieval input — comes from the same config, so the arms differ only in the
thing under test.

| condition | hop count in the prompt | step-line cap |
|---|---|---|
| `unguided` | no | off |
| `oracle_guided` | yes, the gold hop count parsed from the query id | off |
| `unguided_capped` | no | on: `step_cap.max_step_lines` (8), plus the token budget |

The cap stops generation once that many step lines are complete and truncates the decoded text
to the same number, counting step lines the way the evaluator does (`src/step_cap.py`).
`step_cap.max_new_tokens: null` keeps the model's own `max_new_tokens`, so decoding stays
identical across the three arms. Passing `--guided` against a `guided: false` condition is
refused rather than merged, so an arm cannot be mislabelled. Every run records its condition,
cap and `rows_truncated_by_step_cap` in `metrics.json`.

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
