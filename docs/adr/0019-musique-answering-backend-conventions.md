# 0019. MuSiQue Answering-Backend Conventions: Reader, Context and Scoring

- **Status**: Accepted (Jahid, 2026-08-20, in session; pending supervisor confirmation)
- **Date**: 2026-08-20

## Context

ADR [0006](./0006-drop-the-jury-fix-dataset-roles-and-the-few-shot-method.md) gives MuSiQue
**both** decomposition-quality evaluation and **end-to-end** evaluation. Only the first half
existed: `scripts/musique_decompositions_evaluator.py` scores how a decomposition *looks*
(exact match, step F1, ROUGE-L, `[#k]` validity — all string-level, `docs/METRICS.md`).
Nothing scored whether a decomposition *leads to the right answer*, which is half of what
ADR 0006 specifies and the half that a decomposition method is ultimately judged on
(issue #16).

Building that half needs three choices that the README does not settle and that an agent must
not invent: which model reads the paragraphs, what context it is given, and how the answer is
scored. Jahid made all three in session on 2026-08-20. This record states them and adds
nothing to them.

## Decision

**1. The reader is a model from the decomposer's own registry.** The answering backend loads
its model from `components/decomposer/models/<model>/config.json` — the same `model_id`,
loader, quantization, decoding block and `post_process` block the decomposer uses — default
`mistral_7b_instruct`, swappable through `reader_model` in `configs/answer_musique.json` or
`--model`. **No new model family enters the pipeline for this.** The reader's parameter count
is asserted against `configs/model_limits.json` like every other model load
(`src/model_size.py`, component `answerer`).

**The reader's ceiling is 8e9 — the standing ~8B constraint — under its own
`answerer_max_params` key.** ADR [0015](./0015-admit-qwen3.5-9b-to-the-decomposer-despite-the-8b-ceiling.md)
raised `default_max_params` to 1e10 for **one role**, Qwen3.5-9B as a *decomposer*; letting
the reader inherit that key would have extended the exception to a role no ADR admits. The
default reader (Mistral-7B-Instruct-v0.3, ~7.25B) passes; a 9B reader is refused at load time
until Jahid records an extension for it, and swapping `reader_model` to `qwen3_5_9b` for a
real run is therefore a decision, not a flag. (`--dry-run` loads no weights, so a dry run with
any folder reports `ceiling_asserted: false` and asserts nothing — including the smoke stage
that exercises the chat-template rendering path with `qwen3_5_9b`.)

Only the *prompt* is the answerer's own (`components/answerer/prompts/reader.md`): a
decomposition prompt cannot answer a question. It is one committed file for the whole
registry, and the runner refuses a reader prompt that lacks `{context}` or `{question}`.

**2. The context is the MuSiQue item's full paragraph list — the standard answerable
setting.** Every paragraph of the item, in the item's own order, with `is_supporting`
deliberately **not** read. There is **no** gold-supporting-only condition and **no** retrieval
condition: `context.policy` accepts exactly `all_paragraphs` and refuses anything else, so a
second context regime cannot arrive without this record changing.

MuSiQue dev is "20 paragraphs" only as a near-rule: measured 2026-08-20 over
`musique_ans_v1.0_dev.jsonl`, 2401 of 2417 rows carry 20 paragraphs and 16 carry 17–19. The
policy is therefore "all of the item's paragraphs", and the shortfall is **reported**
(`context_stats.items_below_expected_paragraphs`) rather than refused — refusing would drop
real evaluation items for a property of the dataset.

**3. Scoring is MuSiQue's official answer EM and answer F1.** SQuAD's `normalize_answer`
(lowercase, strip punctuation, drop `a`/`an`/`the`, collapse whitespace), SQuAD's
`compute_exact` and multiset-token `compute_f1` including its empty-side rule, and MuSiQue's
`metric_max_over_ground_truths` — the score is the **maximum over the item's `answer` plus
every entry of its `answer_aliases`**. Implemented in `src/answer_metrics.py`, where each
function names its source, and the aggregates are the macro means overall and **per gold hop
depth** (2/3/4), in the reporting style of `docs/METRICS.md` §2.

Aliases are not decoration: measured 2026-08-20, 680 of 2417 dev rows carry a non-empty alias
list, and 136 of the pinned 600 evaluation items do (51 at 2-hop, 53 at 3-hop, 32 at 4-hop).
Scoring against `answer` alone would understate every arm.

## Implementation conventions (agent-authored)

Everything above is Jahid's. The following came with the implementation (PR #32) and is
recorded here so the write-up does not later mistake it for a research decision. None of it
was reviewed as methodology by the supervisor; all of it is changeable without touching the
three decisions.

- **The reader prompt** `components/answerer/prompts/reader.md`. Its wording is the
  implementer's, including the **no-abstention line** — "If the context does not state the
  answer, answer with the most likely span from the context" (line 8), which forces a guess
  rather than allowing an empty answer. That is a scoring-relevant choice: MuSiQue's
  answerable setting has an answer for every item, and an abstention would score 0 on EM and
  F1 anyway, but a forced guess can also pick up partial F1 credit. **Jahid read the prompt
  and approved the wording as-is on 2026-08-20 in session** (pending supervisor confirmation,
  like the rest of this record).
- **One reader prompt for the whole registry**, not one per model folder: a reader prompt is
  not model-specific the way the decomposition prompts are (those are byte-identical to v1 and
  must stay so). The single file is split on `chat_split_marker` (`<<<USER>>>`) into
  system/user messages for a `chat_template` folder and joined with a blank line for a `plain`
  one, so both styles see the same text. `reader_prompt_file` in the config is where a
  per-model prompt would go if one is ever needed.
- **Answer cleanup**: the first non-empty line of the generation, a leading `Answer:`-style
  prefix dropped, and a truncation at `max_answer_chars` (200) that is counted rather than
  silent. Mechanical, and configured in `answer_post_process`.
- **A 64-token completion budget** for a sub-question (`generation_overrides.max_new_tokens`),
  against the model folders' decomposition-sized default. A MuSiQue answer is a short span;
  rows that reach the budget are reported as `rows_at_max_new_tokens`, so the choice is
  observable rather than assumed.
- **Both step-reference grammars are executed**, and a malformed reference (`[#1` with no
  closing bracket) is left verbatim and counted unresolved rather than guessed at.

## Consequences

- End-to-end answer accuracy is measurable for any decomposition run, and the same code path
  measures the **oracle-decomposition ceiling** (`--gold-decompositions`, gold plans executed
  by the same reader) — the reference a prompting or fine-tuned arm is read against.
- Because the reader is fixed and shared, an arm-to-arm difference is a difference in
  *decompositions*, not in readers. Changing `reader_model` changes what the numbers mean, so
  the model folder and its `model_id` are recorded in every metrics JSON and run note.
- **No commercial model is anywhere in this path**, and no model scores anything: EM/F1 are
  string metrics. This is the standing constraint from Jahid's supervisor (CLAUDE.md), and it
  is why an "LLM judges the answer" variant is not offered here.
- A gold-supporting-only or retrieval-context experiment is now a *deliberate* change: it
  needs a new policy value, this record amended, and Jahid's call.
- The end-to-end number is a **joint** measurement of the decomposition and the reader. A
  wrong answer can come from a bad plan or from a reader that cannot find the span, and this
  design cannot separate them; the per-step answers are written to the per-item file so an
  error analysis can. The oracle ceiling is what bounds the reader's contribution.
- Both grammars of step reference are executed (`[#k]` as the prompts instruct, bare `#k` as
  MuSiQue's gold and — per ADR
  [0012](./0012-fine-tuning-arm-conventions-for-the-decomposer.md) — the fine-tuned arm's
  `as_is` targets). Measured 2026-08-20: all 3,987 references across the 2,417 gold dev rows
  are bare, so without the bare form the oracle ceiling could not be executed at all.
- **A shared assertion now carries its caller's voice.** `assert_pinned_eval_set` lives in
  `run_decomposer.py` and is used by both runners, so its refusal takes a `remedy` sentence
  and its warning a `component` tag from the caller. The convention for any further sharing
  out of that module: parameterize the caller-specific text rather than duplicating the
  assertion, because two copies of "what the pinned set is" would be two things to keep in
  step — and a refusal that names the wrong flag sends the reader to the wrong script.
- **The upstream metric definitions are verified, not assumed.** `src/answer_metrics.py`
  carries the provenance: a 20,000-pair differential test against the official SQuAD script
  (0 mismatches) and a direct read of MuSiQue's `metrics/answer.py` + `evaluate_v1.0.py`
  (fetched from `github.com/StonyBrookNLP/musique` @ `main` on 2026-08-20, archived outside
  git with sha256s). Two upstream details are recorded rather than mirrored blindly: the
  argument-order wart in `metric_max_over_ground_truths` (no numeric effect — both metrics are
  symmetric) and this repo's defensive de-duplication of the gold set (unreachable on this
  dataset: 0 blank, 0 non-string, 0 duplicate aliases over 2417 dev rows).
- Status is **pending supervisor confirmation**, like ADRs 0014 and 0015. If the supervisor
  prefers a different reader, a gold-only context or a different answer metric, that
  supersedes this record and the runs made under it are re-run rather than reinterpreted.

## Alternatives considered

- **A separate reader model family** (an extractive QA model, or a larger instruct model).
  Rejected by decision 1: it would add a model to the pipeline, a second parameter-ceiling
  conversation, and a component whose behaviour is not shared with the arm under test.
- **Gold-supporting-paragraphs-only context.** It would raise every number and measure the
  decomposition in a setting no deployment has. Not chosen; explicitly out of scope until
  Jahid decides otherwise (decision 2). The code refuses it rather than allowing it silently.
- **A retrieval condition** (retrieve per sub-question over the corpus). Closer to a real
  system and a plausible later experiment, but it introduces a retriever whose quality would
  be confounded with the decomposition's. Not shipped.
- **Answer scoring by a model** (a commercial API, or a local judge). The commercial variant
  is forbidden (CLAUDE.md); a local judge was not requested and would need its own validation
  before it could carry a claim. MuSiQue ships official metrics, so the defensible choice is
  to use them.
- **Scoring against `answer` only, ignoring aliases.** Simpler and wrong: it is not MuSiQue's
  official metric, and 136 of the pinned 600 items have aliases.
