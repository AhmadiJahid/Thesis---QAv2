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
  evaluation set of ADR 0007, hops 2/3/4, and the run conditions (including `router_guided`,
  issue #27's with-router arm).

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
`condition` key is the default when the flag is omitted. A condition may set only `guided`,
`hop_source` and `stop_after_step_lines` — model, seed and decoding are shared, and the runner
rejects a condition that tries to move them, so the arms cannot drift apart.

| condition | prompt | generation |
|---|---|---|
| `unguided` | no hop count | `generation_overrides.max_new_tokens` only |
| `oracle_guided` | the query's gold hop count, and each exemplar's own gold hop count | same |
| `router_guided` | the **router's predicted** hop count for that query id, exemplars unchanged | same |
| `unguided_capped` | no hop count | stops after `stop_after_step_lines` step lines (default 8) |

**The with-router arm (`router_guided`, issue #27).** Identical to `oracle_guided` except where
the query's hop count comes from: a router predictions JSONL (`--hop-predictions`, written by
`components/router/run_router.py`), joined on the **query id**, whose prediction overrides the
gold depth even when the two disagree — that disagreement is the behaviour under test. Nothing
falls back: a query with no prediction in the file is refused, naming the offending ids, because
filling it in from the gold depth would make the arm a blend of two conditions. The per-hop
reporting and the pinned-set assertion stay on the **gold** depth, so a prediction cannot move a
row into another hop's table: `results.json` carries `hop_count` (gold, unchanged meaning),
`prompt_hop_count` (what the prompt stated) and `hop_count_source`, and `metrics.json`'s
`hop_source` block carries the predictions path, its sha256 and how often the prediction agreed
with the gold depth.

```bash
python components/decomposer/run_decomposer.py --model mistral_7b_instruct \
    --config decomposer_musique.json --condition router_guided \
    --hop-predictions runs/router_musique/few_shot/<run_id>/predictions.jsonl \
    --retrieval-input <the pinned top-5 JSONL over the 600 questions of ADR 0007>
```

The cap is a `StoppingCriteria` during decoding, plus a trim of the partial line the
criterion can leave behind once it fires. Step lines are counted **after** `<think>`/tail
post-processing and with the evaluator's own step normalization
(`src/step_lines.py::split_step_lines`), so a cap of 8 is 8 of the steps the metrics score,
and `metrics.json` reports `rows_at_step_line_cap` on the same count. Next to it,
`rows_stopped_at_step_line_cap` counts the generations the `StoppingCriteria` actually cut off
(read from the criterion's state) — a model that ends on exactly the cap by itself is not a
capped generation, and only the second number is causal. Both are defined in
`metrics.json`'s `truncation_definitions`.

**Hop lines in a guided prompt are gold, and there are two kinds** (Jahid's decision of
2026-08-19, issue #12; ADR 0013 item 8). The query's hop line states the query's gold hop count;
each few-shot exemplar's `Hop count:` line states *that exemplar's own* gold hop count — the
number of steps in its own gold decomposition, counted with the shared splitter. An exemplar
with no usable gold decomposition is refused, naming its pool id, rather than falling back to
the query's hop count. This is selected by `few_shot_exemplar_hop_count` per config: the MuSiQue
config sets `exemplar_gold`, and `configs/decomposer.json` keeps v1's `query` so MetaQA guided
prompts are unchanged (verified byte-identical). v1 stamped the query's hop count on every
exemplar, so v1's guided numbers are not directly comparable to v2's.

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

**Model availability.** Four model folders ship a parity-guarded unguided prompt and can run
these arms: `mistral_7b_instruct` (7.25B), `qwen2_5_7b_instruct` (7.62B), `qwen3_5_9b` (9.65B) and
`qwen2_5_14b_instruct` (14.77B). The last two are above the ~8B standing ceiling and load only
because **`decomposer_max_params`** in `configs/model_limits.json` was raised — to `1e10` for
Qwen3.5-9B (ADR 0015) and to `1.5e10` for Qwen2.5-14B-Instruct (ADR 0031), **both pending
supervisor confirmation**; if the 8B ceiling is reasserted, both are refused at load. That is the
decomposer's *own* key: `default_max_params` stays at the supervisor's standing 8e9, so a raise
for this role cannot lift the ceiling of a role no ADR admits.
`qwen2_5_3b` and `phi_4_mini_instruct` are within the ceiling but ship no unguided prompt.
Whether to add an unguided prompt to another folder is Jahid's decision.

All four are in the local HF cache: `qwen2_5_14b_instruct`'s 8 shards (28 GB) landed on
2026-09-20 (`runs/downloads/qwen2_5_14b_instruct.log`, `EXIT 0`). Its `config.json` `notes`
field keeps the exact fetch command as recovery instructions, for a fresh box or a cache that
has been cleared. `scripts/print_model_params.py` prints every registered model's parameter
count against the committed ceiling, from safetensors metadata only — no weights loaded,
nothing downloaded (ADR 0027). On a sharded snapshot it counts the files
`model.safetensors.index.json` names, not every `*.safetensors` in the directory, so a
`consolidated.safetensors` left beside the shards cannot double the reported total.

**Which pool an arm runs on is a flag, not a default.** `retrieval.input_key` in
`configs/decomposer_musique.json` resolves to the v1 9,156-example-pool artifact that ADR 0014
pinned for issue #12. An issue #50 / A1 arm is pinned by ADR 0031 item 1 to the
`size2000_imbalanced` pool instead and **must** pass `--retrieval-input` explicitly; the exact
path and sha256 are in that config's `_runnable_models_note`. An arm that runs off the default
is on a different pool from the Mistral baseline it is being compared against, which makes it
not a comparison.

Two prompt styles, set per model by `prompt_style`:

- `plain` — the template is formatted directly.
- `chat_template` — the template is split on `chat_template.split_marker` into a system and a
  user half and rendered with the tokenizer's chat template (`enable_thinking` from config).

**Adding a model folder: generate `prompt_unguided.md`, never hand-write it.** The parity guard
(`tests/test_decomposer_conditions.py`, `check_prompt_invariant`) demands that the unguided
prompt equal the guided one minus its hop-bearing lines, *byte for byte* — that is what makes the
guided and unguided arms differ in the hop count and nothing else (ADR 0013). Hand-editing a copy
reintroduces exactly the residual deltas v1 had. So derive it with the same function the guard
checks against, and the guard passes by construction:

```python
import run_decomposer as rd  # components/decomposer/
guided = (folder / "prompt.md").read_bytes().decode("utf-8")
(folder / "prompt_unguided.md").write_bytes(rd.derive_unguided_template(guided).encode("utf-8"))
```

Both Qwen2.5 folders were built this way. Match the prompt of a folder in the same model family
(Qwen folders use the `<<<USER>>>` chat-template form, `mistral_7b_instruct` the plain one)
rather than inventing wording: a prompt difference between arms is a second moving factor.

**The family contrast holds prompt *content*, not prompt *bytes*.** `mistral_7b_instruct`'s
`prompt.md` and `prompt_unguided.md` have **CRLF** line terminators (19 CR in the guided file);
all three Qwen folders' are **LF**. Strip the `<<<USER>>>` marker line and the two are the same
text line for line, which is what the Mistral-vs-Qwen2.5-7B contrast rests on — but they are not
the same bytes, so `prompt_sha256` differs between those arms by construction and is not evidence
of a prompt edit. The line endings are **deliberately left alone**: `mistral_7b_instruct/prompt.md`
is the exact file exp-015 and every earlier Mistral arm ran on, and normalising it now would make
this block's Mistral baseline a different artifact from the one its own history was measured with.
The parity guard is unaffected — it compares each folder's unguided prompt against *its own*
guided one, never across folders.

Few-shot examples come from the reranked retrieval file when `--retrieval-input` is given,
otherwise from masked-similarity selection over the committed exemplar pool, otherwise from a
seeded random draw. Models whose prompt carries its examples inline set `few_shot.enabled: false`.
On every one of those paths an exemplar that **is** the query (same id, or the same question
text after normalization) is dropped before the top-k is taken, and the count of drops is
recorded in `metrics.json`.

Every run writes `results.json`, `config.json`, `metrics.json`, `notes.md` and a `prompts_log/`
under the configured runs root.

### What lands in `prompts_log/`

Two selectors, additive — a row can produce one file, two, or none:

| file | written for | controlled by |
|---|---|---|
| `prompt_idx{position:04d}_hop{h}.txt` | every Nth row **by position** (`position = i + 1`), and every row on a `--dry-run` | `logging.prompt_log_every` in `components/decomposer/models/<model>/config.json` |
| `anchor_{query_id}_hop{h}.txt` | every row whose `query_id` is in the frozen 45-item anchor set, **whatever position it sits at** | `logging.anchor_ids_file` in the shared config (`configs/decomposer_musique.json`; `null` in `configs/decomposer.json`, which turns it off) |

The two answer different questions and neither replaces the other. The positional sample
says what a typical prompt looked like. The anchor files answer *"show me these same 45
items' prompts in every run"* (supervisor item A4, ADR 0033), which position cannot: a run
over a different retrieval artifact orders the rows differently, so the every-Nth sample
catches a different subset each time. An anchor row that also lands on the Nth boundary is
written to **both** names, byte-identical, so existing parity checks over `prompt_idx*.txt`
see exactly the files they saw before. `metrics.json` records `anchor_prompts_logged` and
`anchor_prompts_expected` (the anchor ids actually among this run's rows), and the runner
refuses to finish if they disagree.

The prompt text is deliberately **not** copied into `results.json` — that file keeps its
schema and its size, and `prompts_log/` is the prompt's home. `scripts/render_anchor_set.py`
joins the two on `query_id`.

### Recovering the prompts of a run that finished before anchor logging existed

Every run up to and including exp-020 logged only the positional sample, so most anchor
items have no prompt file. Those prompts are recoverable **on CPU, with no GPU and no model
weights**: `--dry-run` rebuilds each prompt from the committed config snapshot and the same
retrieval artifact, and it writes the anchor files too. Read the finished run's own
`config.json` snapshot for `model`, `condition` and `retrieval.input_resolved`, then:

> **`chat_template` arms (the Qwen folders): one extra step.** A `--dry-run` loads no tokenizer, so
> for a `prompt_style: chat_template` model it logs the chat *messages as JSON*, whereas the real
> run logged `apply_chat_template(...)` of those same messages. The two are not byte-equal as text.
> Load the arm's own tokenizer on CPU, apply its chat template to the regenerated messages, and
> compare that — measured 2026-09-20 for exp-021 and exp-022 at 120/120 byte-identical each
> (exp-025's log row). For `plain` arms such as Mistral the dry-run bodies compare directly
> (exp-020: 120/120).

```bash
# exp-020 (Mistral-7B-Instruct-v0.3, unguided, ADR 0007 pinned 600).
# --quantization is irrelevant here: a dry run loads no model. The seed comes from the config.
.venv/bin/python components/decomposer/run_decomposer.py \
    --model mistral_7b_instruct --config decomposer_musique.json \
    --condition unguided \
    --retrieval-input runs/exp-014/retrieval/s2_pinned600_imbalanced_rerank_top5.jsonl \
    --dry-run --dry-run-limit 600 \
    --output-root runs/anchor/exp-020-prompts
```

That writes all 600 positional prompts plus the 45 `anchor_*.txt`. Measured 2026-09-20 on
exp-020, exp-015 and exp-008 `full_train`: **120/120** of each run's own logged prompt
bodies are byte-identical to the regeneration (the regenerated *header* additionally carries
the `Few-shot ordering:` line those runs predate). Point the renderer at the regenerated
directory while keeping the real run as the source of `decomposition_raw`:

```bash
python scripts/render_anchor_set.py \
    --arm "exp-020=runs/exp-020/eval/eval_per_item.json@runs/exp-020/decomposer/<run id>" \
    --prompts-dir "exp-020=runs/anchor/exp-020-prompts/<run id>/prompts_log" \
    --out runs/anchor/<date>-exp020.md
```

The sheet states, per arm and in words, whether a prompt was *written by this run* or is
*regenerated (CPU dry run, not written by this run)*, and renders an explicit *prompt not
logged for this run* marker where there is none. When an arm carries both `--prompts-dir`
and `@<run dir>`, the renderer compares the regeneration's config snapshot against the
run's on `model`, `condition`, `prompt_sha256` and `retrieval.input_sha256` and **refuses**
on any mismatch: pointing at the wrong regeneration would put one arm's prompt beside
another arm's generation, and the page would look correct. A `--prompts-dir` with no run
directory to check against is marked *unverified against a run*. The snapshot content-addresses what the run read (`prompt_sha256`,
`retrieval.input_sha256`) so two arms' comparability is checkable after the fact, and every arm
reports `rows_at_max_new_tokens` — in the uncapped arms the token budget is the only bound on a
runaway decomposition, so a truncated output has to be distinguishable from a finished one.
The runner prints and asserts the parameter count against `decomposer_max_params` in
`configs/model_limits.json` (1.5e10; the standing constraint in CLAUDE.md is ~8B, and every role
without its own key still inherits that).

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
  **That flag alone is not enough**: few-shot examples are also a prompt difference from the
  training prompt, so the parity guard below refuses as well and
  `--adapter-prompt-mismatch-i-know` has to be passed with it. Two flags, deliberately — one
  says "I want examples", the other says "I accept the off-training prompt", and both land in
  the config snapshot and the metrics.
- **An adapter run whose prompt is not the prompt it was trained on is refused.** The
  training run's own record decides — `training_provenance.json` inside the adapter directory
  (written by `train_lora.py`), or the training run's `config.json` next to it for adapters
  trained before that file existed. A record counts only if it says `script: train_lora.py`
  **and** its `model_id` matches the base model `adapter_config.json` names, so a provenance
  file copied in from another run does not pass. Compared fields: `guided`, the prompt file,
  the prompt style and whether the prompt carries few-shot examples — so
  `--adapter --no-few-shot --guided` (a guided template with a hop line, which training never
  rendered) no longer runs — plus `prompt_sha256`, the content of the rendered prompt file,
  **when the record carries it** (recorded from 2026-08-19 onwards; a record without it is
  reported as not compared, not refused, because exp-001's predates it). No usable training
  record is also a refusal. `--adapter-prompt-mismatch-i-know` overrides it, warns without
  claiming a refusal, and is recorded; such a run is not the fine-tuned arm either.
- **An adapter attaches only to the base model it names.** `adapter_config.json`'s
  `base_model_name_or_path` is compared against the run's `model_id` before
  `PeftModel.from_pretrained`, because LoRA weights attach by module name and shape: a
  different fine-tune of the same architecture would load silently.
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
