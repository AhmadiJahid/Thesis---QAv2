# Decomposer Component

Turns a multi-hop question into ordered single-hop sub-questions, chained with `[#k]`
placeholders that refer to earlier answers. Ported from v1, with the four per-model
`decomposer.py` copies consolidated into one runner.

- `run_decomposer.py` — the runner, for every model.
- `models/<model_name>/prompt.md` — the guided prompt, **byte-identical to v1**.
- `models/<model_name>/prompt_unguided.md` — **derived**, not ported: the guided prompt with its
  hop-bearing lines removed and nothing else changed. It is therefore *not* byte-identical to
  v1's file of the same name; v1's differed in more than the hop count, which would have made
  the guided/unguided arms incomparable (ADR 0013).
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
and `metrics.json` reports `rows_at_step_line_cap` on the same count. Next to it,
`rows_stopped_at_step_line_cap` counts the generations the `StoppingCriteria` actually cut off
(read from the criterion's state) — a model that ends on exactly the cap by itself is not a
capped generation, and only the second number is causal. Both are defined in
`metrics.json`'s `truncation_definitions`.

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
0006 fixes the method), and it asserts the pinned evaluation set of ADR 0007 on whatever rows
were actually loaded — both the per-hop counts (`eval_rows_per_hop`, 200 per hop / 600 total)
**and** the question ids, which must be exactly the ids of the three pinned files. Counts alone
are not identity: a decoy 600 has the right arithmetic and the wrong questions.
`--allow-unpinned-eval-set` is the explicit opt-out for fixture runs, recorded in the metrics as
`evaluation_set.pinned: false` with the id-mismatch counts. `configs/decomposer.json` sets none of this and the MetaQA
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

# a real run (QLoRA 4-bit by default; adapters land under the gitignored runs/ root).
# Hold the runs/run.lock of docs/compute.md for the duration: one GPU, shared box.
python components/decomposer/train_lora.py --arm pool_2000

# inference with the adapter — zero-shot, because that is the prompt it was trained on
python components/decomposer/run_decomposer.py --model mistral_7b_instruct \
    --adapter runs/finetune_decomposer/pool_2000/mistral_7b_instruct/<run>/adapter \
    --no-few-shot --retrieval-input <retrieval rows for the evaluation-set ids>

# quality + significance + cost for both arms, in one table
python scripts/compare_decomposer_arms.py \
    --arm prompting=runs/decomposer/mistral_7b_instruct/<run> \
    --arm finetuned_pool_2000=runs/decomposer/mistral_7b_instruct/<run> \
    --baseline prompting --eval-arm pool_2000
```

### The evaluation input, and what each arm is scored on

The ADR 0007 evaluation set is **three per-hop files** — `musique_ans_v1.0_dev_sample_{2,3,4}_hop_200.jsonl`,
200 rows each, 600 distinct ids — resolved through `datasets.musique_eval_questions_template`.
There is no single 600-question file in this pipeline: `train_lora.py` reads all three (that is
what the train/eval overlap assertion checks against, and it asserts the 200-per-hop / 600-total
counts declared in `eval_set.expected` so a mis-resolved id field cannot leave it checking
nothing), and `run_decomposer.py` takes its MuSiQue questions from `--retrieval-input`, a JSONL
keyed by those same `query_id`s.

Which of the three an arm is scored on is the arm's `eval_hops`, and it is **enforced** by
`compare_decomposer_arms.py --eval-arm <arm>`: every scored item's id hop prefix, on every
side, must be a hop that arm is evaluated on, or the comparison is refused. So for
`generalisation_2_3hop` (`eval_hops: [4]`) **both** the fine-tuned arm and the prompting
baseline must be run over the 4-hop rows only — the 200 ids of the 4-hop file, and nothing
else. `run_decomposer.py` has no hop filter of its own: the restriction is a property of the
`--retrieval-input` file handed to both arms, and the two arms must be handed the same one.
The other two arms (`eval_hops: [2, 3, 4]`) are scored on all 600.

Guarantees that are hard failures, not warnings:

- **Training ids are asserted disjoint from the ADR 0007 evaluation ids** (all 600, whatever
  hops the arm is evaluated on), naming the offenders — and the evaluation id set is itself
  asserted to be the declared size first, because an overlap check against an empty set
  passes while proving nothing.
- **The base model is asserted against the ~8B ceiling** before any step, and the
  LoRA-wrapped model again after the adapter is attached. `run_decomposer.py --adapter`
  asserts base + adapter.
- **`--adapter` without `--no-few-shot` is refused.** The adapter was trained on the
  zero-shot prompt, and the few-shot examples come from the same MuSiQue pool it trained on.
  `--adapter-with-few-shot-i-know` overrides it deliberately, prints a warning and is
  recorded in the run's metrics; such a run is not the fine-tuned arm of the comparison.
- **A hop-signal disagreement is fatal for an arm that filters on hops** (`train_hops` not
  null): the generalisation claim rests on those labels being right.
- Adapters, checkpoints and prediction dumps stay under `runs/`; only config, metrics and the
  run note enter git.

The training prompt is the model's own `prompt_unguided.md` with the few-shot block empty,
built through `run_decomposer.py`'s template helpers so the trained string and the inference
string cannot drift. Evaluation is always
`scripts/musique_decompositions_evaluator.py` — string-level, no model in the loop, and no
commercial API rating a decomposition (CLAUDE.md).

A training run and an adapter evaluation both occupy the single GPU, so both are held under
the **run lock** convention in [`docs/compute.md`](../../docs/compute.md): take
`runs/run.lock` (experiment id + timestamp) before launching, release it when the run ends,
and never clear someone else's lock on age alone. `--dry-run` loads no weights and needs no
lock.
