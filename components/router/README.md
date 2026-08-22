# Router Component

Predicts a question's hop count. Ported from v1, with the ten near-identical per-model
`router.py` copies consolidated into one runner. Nothing here is trained — the router is
prompted.

- `run_router.py` — the runner, for every model.
- `models/<model_name>/prompt.md` — the hop-count prompt (static few-shot), **byte-identical to v1**.
- `models/<model_name>/config.json` — model id, generation length, loader flags, response parsing.
- `models/prompt_zero_shot.md` — shared zero-shot prompt (no few-shot examples).
- `models/prompt_few_shot_musique.md` — shared **retrieved**-few-shot prompt: an instruction, a
  `{few_shot_examples}` block, the query.
- `configs/router.json` — the MetaQA path: everything that is not model-specific (seed, hops,
  run counts, output roots).
- `configs/router_musique.json` — the few-shot-prompted router of issue #27: the pinned MuSiQue
  evaluation set (ADR 0007), hops 2/3/4, retrieved exemplars, and a predictions file keyed by
  query id.

```bash
python components/router/run_router.py --model qwen2_5_0_5b
python components/router/run_router.py --model qwen2_5_0_5b --prompt-file prompt_zero_shot.md
python components/router/run_router.py --model qwen2_5_0_5b --dry-run   # prompts only, no model

# the few-shot-prompted router over the pinned MuSiQue set (issue #27)
python components/router/run_router.py --model qwen2_5_0_5b --config router_musique.json
```

Output goes under the runs root from `configs/paths.json`: `router/average_zero_shot/<run_id>/`
for the zero-shot prompt, `router/average_few_shot/<run_id>/` otherwise (and
`router_musique/...` for the MuSiQue config). Every run writes `config.json`, `metrics.json`,
`notes.md` and `detailed_results*.json`.

**Predictions keyed by query id.** A **real** run whose config sets `predictions.enabled`
writes `predictions.jsonl` — one object per query with `query_id` and `predicted_hop` (field
names from the config), plus the gold depth it is scored against and whether the hop was read
out of the response or fell back to `parsing.default_hop`. Two gates, both explicit: the knob
is on only in `configs/router_musique.json` (a `questions_format: lines` source such as
MetaQA's has no id to key on, so `configs/router.json` sets it false), and a `--dry-run`
writes **no** predictions file at all — it generates nothing, so there is no prediction to
record and the metrics say so rather than reporting an empty or invented file. The shape is
the one ADR 0022 item 5 documents, and it is what the two consumers join on:

```bash
# issue #15's router-hop-matched retrieval condition
python MusiQue/scripts/check_question_similarity.py --hop-match --hop-source predictions \
    --hop-predictions runs/router_musique/few_shot/<run_id>/predictions.jsonl

# the decomposer's with-router arm (issue #27)
python components/decomposer/run_decomposer.py --model mistral_7b_instruct \
    --config decomposer_musique.json --condition router_guided \
    --hop-predictions runs/router_musique/few_shot/<run_id>/predictions.jsonl
```

A missing or repeated query id is refused before a model is loaded; a query the predictions
file does not cover is refused on the consuming side. Nothing falls back to the gold depth.

**Retrieved few-shot exemplars.** With `few_shot.enabled`, the exemplars for a query are the
candidates the retrieval artifact already holds for it (the same artifact the decomposer
reads), each labelled with **its own** gold hop depth parsed from its pool id. A candidate that
is the query itself is dropped — by id and by normalized question text, using the decomposer's
own rule — before the top-k is taken, so no query is shown its own answer. A dry run assembles
all of this and writes the prompt log; it predicts nothing, so it writes no predictions file.

**Reading the hop count out of a response.** The retrieved-few-shot prompt ends with the `A:`
cue, so the model's answer is the *first* thing in its response and anything after the first
line break is the model writing a fresh question of its own. `response_truncate_at` in the
config cuts the response there before it is parsed, and that config drops the `A:`-prefix
regex, so the answer is read as the leading digit; the v1 MetaQA prompts set no markers and
keep their own regex, so their parsing is unchanged. Every run — dry runs included — asserts
that the parsing rules can express every depth in `hops` and that `parsing.default_hop` is one
of them, and records the result as `parsing_coverage`. A response with no readable hop count is
recorded as `default_hop` with `parse_fallback` true, **counted in the accuracies**, and broken
out per gold hop as `unparsed_response_rows_per_gold_hop` — a defaulted row is correct for
whichever class the default belongs to, so that split is what says how much of a per-hop number
is a prediction. `accuracy_definitions` in `metrics.json` states what each accuracy measures
(exact-hop against the gold depth; the per-hop rows are within-gold-class recall, not
precision).

`--num-runs N` repeats inference with seeds seed, seed+1, … and reports mean ± std. The
predictions file is written from run 0, the same run `detailed_results_run_0.json` records.

**Parameter ceiling.** The runner prints the loaded model's parameter count and asserts it
against `router_max_params` in `configs/model_limits.json` — 8e9 since ADR 0008 lifted the
former ~600M router-specific cap to the overall ~8B ceiling (the per-model `notes` fields that
still mention ~600M predate that ADR). A model above the ceiling is refused at load time;
raising the number is Jahid's decision with his supervisor, not an agent's.

Plots and an HTML report over completed runs:

```bash
python scripts/analyze_runs.py --config configs/analyze_runs.json --component average_zero_shot
```
