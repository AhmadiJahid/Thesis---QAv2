# 0024. Router Readiness: Query-Id-Keyed Predictions, a Retrieved-Few-Shot Router, and the Router-Guided Decomposer Arm

- **Status**: Accepted (design agent-authored, pending Jahid)
- **Date**: 2026-08-22

## Context

Issue #27 asks for a few-shot-prompted router and a with-router versus without-router
evaluation on the pinned MuSiQue set; ADR
[0023](./0023-jahid-2026-08-22-direction-metric-pipeline-completion-generalisation.md) item 2.2
makes that comparison the instrument for Jahid's next pipeline decision ("use the router or omit
it and decompose anyway"). Two things blocked it, both recorded rather than guessed:

- ADR [0022](./0022-hop-matched-retrieval-the-implementers-design.md) item 5 recorded that the
  router writes `detailed_results.json` keyed by **question text with no query id**, so the
  `predictions` hop source it documented had no producer — issue #15's router-hop-matched
  condition was blocked on a file in that shape.
- `run_decomposer.py`'s guided condition took the hop count from the query's **gold** depth only
  (parsed from the id, or the per-hop file the question came from). There was no way to feed it a
  prediction, so the with-router arm did not exist as a runnable configuration.

**That these things get built and measured is Jahid's decision** (issue #27, ADR 0023). How they
are implemented was not decided by anyone; this record states the implementer's choices so the
November write-up does not mistake one for a research decision.

**Nothing in this record is a measurement.** No router has been run on the evaluation set, no
with-router/without-router comparison exists, and no claim about router accuracy or about the
router's effect on decomposition quality may be made until an `experiments/log.md` entry exists.
What *was* executed is under "What was verified".

ADR [0010](./0010-keep-the-router-as-a-hop-count-regressor-prioritize-fine-tuning.md) reframed
the router as a **regressor** and left its target encoding, loss, rounding rule and headline
metric explicitly open for Jahid and his supervisor. This record does not answer any of them: it
adds a *prompted* router (nothing is trained) whose output is an integer hop count, and it scores
that the way the MetaQA path already did (exact-hop accuracy, per hop). A trained regressor can
later write the same predictions file and drop into the same two consumers unchanged.

## Decision (the implementer's, in three lines)

1. **The router's predictions are a JSONL keyed by query id** — `{query_id, predicted_hop, …}`,
   one object per query, exactly the shape ADR 0022 item 5 documented for its consumers.
2. **The few-shot-prompted router is a prompting mode of the existing runner**, whose exemplars
   are the query's own retrieved candidates from the artifact the decomposer already reads, each
   labelled with its own gold hop depth, with the query self-excluded.
3. **The decomposer gains a `hop_source` condition key**: `gold` (every arm that existed before)
   or `predictions`, and `router_guided` is the condition that sets the latter.

## Implementation conventions (agent-authored)

1. **The predictions file is a join key plus a number, and it carries no question text.**
   `query_id` and `predicted_hop` are the field names (configurable, defaulting to the values
   `configs/similarity.json` and `configs/decomposer_musique.json` already expect). Alongside
   them: `expected_hop`, `correct`, and `parse_fallback` — a flag saying the model's response
   carried no readable hop count, so the model folder's `parsing.default_hop` was used. v1
   returned that default silently, so a run's accuracy mixed predictions with defaults and
   nothing recorded which was which; the flag and the `unparsed_response_rows` count are what
   make that visible. Question text is deliberately absent: ADR 0022 item 5 rejected keying the
   join on it, and a field that is present gets keyed on. `detailed_results.json` keeps it, and
   now also carries `query_id`.

1a. **The response is cut at the answer before a hop count is read, and the parsing rules are
   asserted to cover the config's hop depths.** *(Added 2026-08-22 in the PR #43 review fix
   pass — findings C1 and I1. Implementer design, pending Jahid, like the rest of this record.)*

   The retrieved-few-shot prompt ends with the `A:` cue, so the model's answer is the **first**
   thing in the response and everything after the first line break is the model writing a fresh
   `Q:`/`A:` pair of its own. Two consequences, both now encoded:

   - **Truncation before parsing.** `response_truncate_at` (per config, because it is a property
     of the prompt shape) cuts the response at its first line break for the MuSiQue prompt and
     is empty for the v1 MetaQA prompts, whose responses are multi-line by design. And that
     config sets `answer_regex` to null: a rule requiring an `A:` prefix can only match a
     regurgitated exemplar when the prompt consumed the cue itself, so the answer is read as the
     leading digit by the rule whose digit class is derived from `hops`. Measured on the
     committed configs: the response `"2\n\nQ: …\nA: 3\n\nQ:"` parsed to **(3, True)** under the
     first version of this change — another question's answer, returned as this query's
     prediction with `parse_fallback` false — and parses to **(2, True)** now. `eos_token_id` is
     also passed to `generate` so a finished model can stop, but that is not what makes the read
     safe: a model can always keep going, so truncation is the guarantee.
   - **A coverage assertion, not a prose requirement.** `assert_parsing_covers_hops` refuses, at
     config-load time and therefore on a dry run too: a hop depth the fallback label table
     cannot name (v1's tables stopped at three, so a MuSiQue router's "four hops" fell through
     to `default_hop`), a hop depth above 9 (the digit-class rule is a single-character regex
     class), and a `default_hop` outside `hops`. The label tables are now derived from the hop
     depth rather than listed, so they follow the config. This is ADR 0016's rule applied to a
     config invariant: the earlier version of this record stated the requirement in a config
     `_note`, and a `_note` does not fail a run.

   **How a defaulted hop is scored, stated once here because it shapes every router number.** An
   unreadable response is recorded as `parsing.default_hop` **as if it were a prediction** and is
   **counted in the accuracies** — it is not dropped, and the denominator does not change. That
   is deliberate (a router that cannot answer has still routed the query, and the pipeline will
   use that hop), but it means the default's own class is flattered: with `default_hop` 2, every
   defaulted row is correct for a 2-hop query and wrong for a 3- or 4-hop one. So the count is
   reported overall *and* per gold hop (`unparsed_response_rows_per_gold_hop`), and
   `accuracy_definitions` in the metrics says that the per-hop rows are within-gold-class recall
   rather than precision. Any router accuracy claim in the write-up has to carry that split;
   whether a defaulted row should instead be excluded, or scored as a refusal, is Jahid's call
   and is not decided here.

2. **A dry run writes no predictions file.** It generates nothing, so a predictions file from a
   dry run would be fabricated routing decisions. The metrics say `predictions_file: null` with a
   note, in the "unmeasured, not zero" style of ADR
   [0016](./0016-real-run-only-invariants-get-source-level-guards.md). Everything *up to*
   generation is exercised by the dry run: the query-id integrity check, the exemplar assembly,
   the prompt. The file's own shape is therefore pinned by a unit test that round-trips it
   through the consumer (`load_predicted_hops`), not by the CLI.

3. **Missing and duplicated ids fail on both sides.** The router refuses, before a model is
   loaded, a question source with a blank/absent id or a repeated one (naming the offenders); the
   consumer refuses a duplicate id (`load_predicted_hops`, already) and a query the file does not
   cover (`join_predicted_hops`, naming up to ten offending ids). No fallback to the gold depth
   for the uncovered queries — ADR 0022 item 3's reasoning, on the decomposer's side of the same
   join: a per-query fallback would make the routed arm a blend of two conditions.

4. **The few-shot exemplars are the retrieval artifact's own candidates, and the labels come from
   the pool id.** The router reads the same JSONL the decomposer reads (`--retrieval-input`, or
   `retrieval.input_key`), takes the first k candidates of the configured `<mode>_top_k`, and
   states each exemplar's coarse hop depth parsed from *its* pool id. Two reasons for the id
   rather than a step count: it is the same quantity the router is scored against (a query's gold
   depth is parsed from its id the same way), and it needs no gold decomposition, so a pool row
   with a missing decomposition cannot silently change the label. `few_shot.exemplar_hop_source`
   names it, and the only implemented value is `pool_id`; a second one (for instance the step
   count of the exemplar's gold decomposition, which is what the *decomposer's* exemplar hop
   lines state per ADR [0013](./0013-guided-vs-unguided-condition-conventions-on-the-musique-set.md))
   is refused rather than approximated, because choosing it is a research decision.

5. **Self-exclusion is the decomposer's rule, imported, not a second copy.** `is_self_example`
   (pool id equal to the query id, or whitespace/case-normalized pool question equal to the
   query's) is imported from `run_decomposer.py`, along with the question-source reader and the
   pinned-eval-set assertion — the way `run_answerer.py` already imports the last of those. A
   leakage rule with two copies is a rule that drifts, and ADR 0022 item 6 already records four
   copies of "hop from id" as this repo's live example of that risk. The cost is a real import
   dependency from the router component onto the decomposer's runner; the alternative was moving
   three functions into `src/`, which is a wider edit to a file two queued experiments read.

6. **The router asserts the pinned evaluation set by id.** `eval_rows_per_hop` in
   `configs/router_musique.json` runs the same assertion the decomposer and the answerer use, so
   a predictions file over a different 600 questions is refused rather than joined against arms it
   does not cover (ADR [0007](./0007-musique-evaluation-set-reuses-v1-600-questions-200-per-hop.md),
   ADR [0011](./0011-comparison-artifact-conventions-and-the-significance-claim-floor.md)).
   `--allow-unpinned-eval-set` is the recorded opt-out for fixture runs, exactly as elsewhere.
   `sample_size_per_hop` is refused on the retrieval path for the same reason: a sampled run
   writes a predictions file that does not cover every query its consumers ask about.

7. **`hop_count` in the decomposer's outputs keeps its meaning; the prompt's number is a new
   field.** The join sets a row's *prompt* hop count from the prediction and leaves
   `gold_hop_count` alone, and the per-hop counts and the pinned-set assertion are computed on
   the gold depth — otherwise a router predicting 3 for a 2-hop query would move that query into
   the 3-hop row of the results table and the pinned-set assertion would fail on a correctly
   loaded set. `results.json` therefore carries `hop_count` (gold, unchanged for every existing
   consumer), `prompt_hop_count` (null in an unguided arm) and `hop_count_source`. The prompt-log
   file names stay on the gold depth too, so the arms' logs line up file by file; the predicted
   number is stated in the log header.

8. **The condition block is where the hop source lives.** `hop_source` joins `guided` and
   `stop_after_step_lines` as the only keys a condition may set, because it says *which* hop count
   the prompt carries — the single difference between `oracle_guided` and `router_guided`. Model,
   seed, retrieval and decoding stay shared and un-overridable, so the two arms cannot differ in a
   second way. **Five** refusals, all of them a run that would otherwise be filed under the wrong
   label: an unimplemented source (an explicit `"hop_source": null` in a condition included — it
   names no source, so it may not quietly inherit the default); `predictions` in an unguided arm
   (no hop count reaches the prompt at all there); `predictions` together with
   `few_shot_exemplar_hop_count: "query"`, which would stamp the *prediction* on all k exemplars
   and so resurrect ADR 0013's defect with a prediction where the gold depth used to be —
   unreachable from the committed configs, which is exactly why it is a guard and not a fix
   *(added 2026-08-22, PR #43 review I3)*; `predictions` with no file named; and
   `--hop-predictions` passed to a `gold` arm, where the file would be ignored while the operator
   believed the run was routed.

9. **The predictions path is a flag, not a committed config value.**
   `hop_predictions.file` is null in both decomposer configs. The file is a *run output* under the
   gitignored runs root, not a pinned artifact, so a path baked into a committed config would
   silently route a later run with an older router's decisions. The run records the resolved path
   **and its sha256**, next to the retrieval input's, so "which routing decisions produced this
   arm" is answerable from the run trail.

10. **Every knob is in a committed config; the MetaQA path is off by default.**
    `configs/router.json` gains `questions_format: "lines"`, `few_shot.enabled: false` and
    `predictions.enabled: false` — explicit rather than absent, in this repo's no-silent-default
    style — so that path renders and runs exactly as before. The v1 model folders are untouched:
    the new prompt is a *shared* file (`models/prompt_few_shot_musique.md`, named by the MuSiQue
    config so the mode cannot be run with a MetaQA prompt by forgetting a flag), and MuSiQue's
    2/3/4 answer digits come from a `parsing_overrides` block rather than an edit to a v1 asset.

## What was verified (and what was not)

Ran 2026-08-22, CPU only, no weights, on this branch. Counts are from the **PR #43 review fix
pass**; the first version of this record cited 44 / 354, and the earlier count that mattered most
was also *wrong about what it checked* — it asserted the response shape `A: <n>`, which the
retrieved-few-shot prompt cannot produce, and that is the mistake finding C1 exposed. The
response-shape checks below are written against the shape the prompt actually elicits.

- `tests/test_router_predictions.py` — **62 tests, all passing**: the predictions rows and their
  round trip through `load_predicted_hops`, the committed fixture read by the consumer, every id
  and join refusal, the exemplar assembly (labels, both self-exclusion paths, four refusals, and
  the post-filter assertion), the prompt rendering including that a v1 prompt renders unchanged,
  the **response parsing** (the answer is the leading digit; a response that answers and then
  regurgitates a `Q:`/`A:` pair parses to the answer; a leading newline does not truncate the
  answer away; a `<think>` preamble is stripped; a hop named in words is read rather than
  defaulted; v1 multi-line parsing unchanged with no markers configured), the parsing-coverage
  assertion and its four refusals, `hop_source` resolution with all five refusals, the accuracy
  definitions and the per-gold-hop default counts, and the join's effect on the prompt hop versus
  the gold hop.
- Full suite `python -m unittest discover -s tests` — **372 tests, OK** (310 before this change).
- `scripts/smoke_test.py` — **39/39 stages**, including the two new ones
  (`router_dry_run_musique_few_shot`, `decomposer_dry_run_musique_router_guided`).
- `python tests/test_decomposer_conditions.py --skip-data-checks` — **258 checks passed**.
- **The C1 regression, measured on the committed configs**: response
  `"2\n\nQ: Who founded the press in Marlow Bay?\nA: 3\n\nQ:"` → `(3, True)` with the first
  version's parsing (an `A:`-prefix regex over the untruncated response), `(2, True)` now.
- **The routed arm does something different from the oracle arm, on the same nine fixture
  queries**: with a fabricated predictions file in which three of nine predictions disagree with
  the gold depth, `router_guided` put the *predicted* hop in all nine prompts (verified row by row
  against the file) while `oracle_guided` put the gold one, the two arms produced the same query
  ids in the same order, and the routed run's per-hop counts stayed 3/3/3 on the gold depth.
- The MetaQA router dry run still exits 0 and reports `few_shot.enabled` false with no
  predictions file.

Not verified, and not claimed: **any router accuracy number** (no model has been loaded — a dry
run asserts no parameter count either, the ADR 0016 caveat), **any effect of routing on
decomposition quality**, and the feasibility of the real 600-query artifact under this path (no
router predictions file over the pinned set exists yet).

## Consequences

- Issue #27's with/without-router comparison is now a runnable pair of configurations:
  `oracle_guided` (or `unguided`) against `router_guided`, same model, seed, retrieval artifact and
  decoding. Producing the numbers is a separate experiment lane with its own log entry; this record
  produces none.
- Issue #15's third regime is unblocked: `check_question_similarity.py --hop-source predictions`
  now has a producer, and the field names match with no adapter.
- The router component now imports the decomposer's runner (item 5). If those three functions
  ever move to `src/`, both importers change together.
- A trained hop predictor (ADR 0010's regressor) inherits the interface rather than defining a new
  one: whatever produces `{query_id, predicted_hop}` rows drives both consumers. The regressor's
  own open questions — target encoding, loss, rounding, headline metric — are untouched here.
- `configs/router.json` and both decomposer configs now require keys they did not before
  (`questions_format`, `few_shot`, `predictions`; `hop_source`, `hop_predictions`). A config
  without them is refused loudly by `require()`, which is the intended behaviour and matches what
  ADR 0022 item 4 did to `configs/similarity.json`.
- The router's response parsing now reports how often it fell back to a default, overall and per
  gold hop. If that count is high on a real run, the router's "accuracy" is partly the default's
  accuracy — a caveat the write-up must carry, and a number that now exists to carry it.
- A new prompt for this component brings an obligation with it: state its
  `response_truncate_at`, and check that its answer position matches the parsing rules. The
  coverage assertion catches a hop-depth gap automatically; it cannot catch a prompt whose answer
  is not where the parsing looks for it. That check is a test against the response shape the
  prompt elicits (`TestResponseParsing`), and it is the shape of test the next prompt needs.

## Alternatives considered

- **Key the predictions on question text** (what today's `detailed_results.json` would allow).
  Rejected in ADR 0022 item 5 and not revisited: the masking modes rewrite that field.
- **Write a small adapter from `detailed_results.json` to the documented shape.** Rejected: it
  would leave the router's own output unjoinable and put the id back on the text it was keyed by,
  a level of indirection that hides the very field the join needs.
- **A fixed, hand-written MuSiQue exemplar set for the router** (the shape of v1's MetaQA
  `prompt.md`). Not implemented: the repo's few-shot method is retrieval (ADR 0006), the retrieval
  artifact already exists over exactly these queries, and a hand-written set would be a new
  research asset nobody asked for. It remains the obvious cheaper variant if Jahid wants a
  static-prompt arm.
- **Fabricate a predictions file on a dry run so the CLI covers the whole path.** Rejected; see
  item 2. A unit test round-trip covers the shape without inventing predictions.
- **Let `hop_count` in `results.json` become the predicted hop** (fewer fields). Rejected; see
  item 7: it changes the meaning of a field several scripts read, and the pinned-set assertion
  would fail on a correctly loaded set.
- **Sampling and per-hop subsets on the router's retrieval path.** Refused rather than supported:
  a partial predictions file is a refusal on the consuming side anyway, so allowing it upstream
  would only move the failure later.
- **Consolidating `is_self_example` / the question reader / the eval-set assertion into `src/`.**
  Deferred, not rejected: it is the right cleanup, and it touches a file that queued experiments
  read. Same reasoning as ADR 0022 item 6's parked de-duplication.
