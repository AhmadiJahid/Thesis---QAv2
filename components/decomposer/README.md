# Decomposer Component

Turns a multi-hop question into ordered single-hop sub-questions, chained with `[#k]`
placeholders that refer to earlier answers. Ported from v1, with the four per-model
`decomposer.py` copies consolidated into one runner.

- `run_decomposer.py` — the runner, for every model.
- `models/<model_name>/prompt.md` — the guided prompt, **byte-identical to v1**.
- `models/<model_name>/prompt_unguided.md` — **derived**, not ported: the guided prompt with its
  hop-bearing lines removed and nothing else changed. It is therefore *not* byte-identical to
  v1's file of the same name; v1's differed in more than the hop count, which would have made
  the guided/unguided arms incomparable (ADR 0012).
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

# one arm of the guided-vs-unguided comparison (the SAME retrieval file in all three arms)
python components/decomposer/run_decomposer.py --model mistral_7b_instruct \
    --config decomposer_musique.json --condition oracle_guided \
    --retrieval-input <the pinned top-5 JSONL over the 600 questions of ADR 0007>
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

The cap is a `StoppingCriteria` during decoding, plus a trim of the partial line the
criterion can leave behind once it fires. Step lines are counted **after** `<think>`/tail
post-processing and with the evaluator's own step normalization
(`src/step_lines.py::split_step_lines`), so a cap of 8 is 8 of the steps the metrics score,
and `metrics.json` reports `rows_at_step_line_cap` on the same count.

**The arms differ in hop information and nothing else — enforced, not asserted.** The
config sets two guards, both refusals:

- `unguided_prompt_must_omit_hop_count` — an unguided arm may not run on a prompt that
  mentions the hop count. Without an `unguided_prompt_file` the guided prompt is reused and
  `{hop_count}` is filled with the placeholder ("Hop count: Unknown" under a rule saying the
  step count must equal the hop count), which is neither arm; a hardcoded hop line is caught
  too.
- `unguided_prompt_must_equal_guided_minus_hop_lines` — the unguided prompt must be the
  guided prompt with the hop-bearing lines removed, byte for byte. A residual delta (a rule
  added, a rule dropped, a sentence reworded) fails loudly with a diff.

Two more refusals for this config: it requires `--retrieval-input` (`retrieval.require_input`,
so a run cannot silently fall back to random exemplars from the committed MetaQA pool — ADR
0006 fixes the method), and it asserts `eval_rows_per_hop` (200 per hop, 600 total — the
pinned set of ADR 0007) on whatever rows were actually loaded. `--allow-unpinned-eval-set` is
the explicit opt-out for fixture runs, recorded in the metrics as
`evaluation_set.pinned: false`. `configs/decomposer.json` sets none of this and the MetaQA
behaviour is unchanged.

**Model availability.** As configured, exactly one model folder can run these arms:
`mistral_7b_instruct`. `qwen3_5_9b` also ships an unguided prompt but is 9B, above the ~8B
ceiling in `configs/model_limits.json`, so it is refused at load; `qwen2_5_3b` and
`phi_4_mini_instruct` are within the ceiling but ship no unguided prompt. Whether to add an
unguided prompt to another ≤8B folder is Jahid's decision.

Two prompt styles, set per model by `prompt_style`:

- `plain` — the template is formatted directly.
- `chat_template` — the template is split on `chat_template.split_marker` into a system and a
  user half and rendered with the tokenizer's chat template (`enable_thinking` from config).

Few-shot examples come from the reranked retrieval file when `--retrieval-input` is given,
otherwise from masked-similarity selection over the committed exemplar pool, otherwise from a
seeded random draw. Models whose prompt carries its examples inline set `few_shot.enabled: false`.
On every one of those paths an exemplar that **is** the query (same id, or the same question
text after normalization) is dropped before the top-k is taken, and the count of drops is
recorded in `metrics.json`.

Every run writes `results.json`, `config.json`, `metrics.json`, `notes.md` and a `prompts_log/`
under the configured runs root. The snapshot content-addresses what the run read (`prompt_sha256`,
`retrieval.input_sha256`) so two arms' comparability is checkable after the fact, and every arm
reports `rows_at_max_new_tokens` — in the uncapped arms the token budget is the only bound on a
runaway decomposition, so a truncated output has to be distinguishable from a finished one.
The runner prints and asserts the parameter count against `default_max_params` in
`configs/model_limits.json` (~8B, per CLAUDE.md).

Score the predictions with `scripts/musique_decompositions_evaluator.py` (MuSiQue gold) or
`scripts/evaluate_decompositions.py` (MetaQA KG execution).
