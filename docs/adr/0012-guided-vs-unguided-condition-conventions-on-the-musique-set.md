# 0012. Guided-vs-Unguided Condition Conventions on the MuSiQue Set

- **Status**: Accepted (two items are open questions for Jahid, marked below)
- **Date**: 2026-08-19

Records the conventions established by PR #21 (issue #12) for the guided-versus-unguided
comparison. Everything here is sourced either to **Jahid's plan, prompt 4** (his statement of
the experimental design) or to an existing committed config; nothing in it is a research
decision taken by an agent. No success criteria, no thresholds, no recommendation about the
router — those are Jahid's with his supervisor.

## Context

Jahid's plan, prompt 4, states the design: *"Build and run three conditions on the same
MuSiQue evaluation set, everything else held identical (same model, same decoding config,
same seed, same retrieval: bi-encoder top-20, cross-encoder to top-5, typed masking, pool
2000)"*, with `unguided` = *"no hop count in the prompt"*, `oracle_guided` = *"gold hop count
injected"*, `unguided_capped` = *"no hop count, but generation stops after N step lines, N in
config, default 8, plus a max-token cap"*.

The harness for those three arms is code, and the way it encodes "everything else held
identical" is a set of conventions that will be visible in every artifact the experiment
produces. PR #21's independent review found the first implementation broke the design in ways
that were invisible from the config: the two `prompt_unguided.md` files (ported from v1)
differed from their guided siblings by more than the hop count, the default config would have
run off ADR 0006's fixed retrieval method, and the pinned row count was only checked by a
test. This ADR records the conventions the fix pass established, so that in November the
reason each guard exists is on the record rather than in a diff.

## Decision

**1. Three conditions, exactly as Jahid's plan states them.** `configs/decomposer_musique.json`
carries a `conditions` block with `unguided`, `oracle_guided` and `unguided_capped`, selected
with `--condition`. The gold hop count injected by `oracle_guided` is the depth of the pinned
evaluation file the question was read from, or the depth parsed from its id on the retrieval
path — never a prediction.

**2. `_CONDITION_KEYS` whitelist plus `--guided`-contradiction refusal — the anti-drift
mechanism.** A condition block may set only `guided`, `stop_after_step_lines` and `_note`. Any
other key (a seed, a decoding parameter, a model) is refused at load: the arms are only
comparable if model, seed, decoding and retrieval are shared, and a config is where such a
difference would hide. Separately, `--guided` may not contradict a named condition: the
condition name is what the snapshot, the metrics and the log entry record, so an overridden
arm would be filed under a label it did not run.

**3. Unguided = guided minus hop-bearing lines, byte for byte.** The unguided prompt of a
model folder must equal its guided prompt with every hop-bearing line removed
(`{hop_count}`, or prose mentioning the hop count) and nothing else changed — no replacement
instruction, no dropped rule. `unguided_prompt_must_equal_guided_minus_hop_lines` in the
config makes a residual line-level delta a loud failure with a diff, and the two existing
`prompt_unguided.md` files were **regenerated** by that rule.

The consequence is deliberate and worth stating plainly: those two files are **no longer
byte-identical to v1**, which ADR [0001](./0001-v1-to-v2-migration-scope-and-method.md)'s
porting convention otherwise requires for prompts. v1's `mistral_7b_instruct` unguided prompt
replaced the hop rule with *"Decompose into the minimal number of atomic steps."*, and v1's
`qwen3_5_9b` unguided prompt dropped the entire 7-rule format block and substituted *"Follow
the given examples below."* Under either, a quality difference between arms could not be
attributed to the hop count — the prompts also differed in what they instructed. Jahid's
stated design ("everything else held identical") governs, and the deviation from the porting
convention is recorded here, in `configs/decomposer_musique.json` and in
`components/decomposer/README.md`.

**4. The step-line cap default is 8, and two numbers are on the record.** Jahid's plan states
*"N in config, default 8"*, and 8 is the value in the config. The supervisor's in-meeting
example was a cap of **6** lines
(`docs/meetings/2026-08-12-supervisor-meeting-transcript-crosscheck.md`, which records the
length cap as confirmed and the example as 6). Both facts are recorded; this ADR does not
choose between them, and N is a one-line config edit.

**5. One definition of "a step line".** The cap, the rows-at-cap counter and the evaluator all
use `src/step_lines.py::split_step_lines` (moved there unchanged from
`scripts/musique_decompositions_evaluator.py`), and the cap is counted **after** `<think>` and
tail post-processing. Before this, the budget was counted on the raw generation, the counter on
the post-processed text, and the evaluator with its own private splitter — three numbers named
"steps" that were not the same number.

**6. `max_new_tokens: 256` is an unconfirmed plumbing default.** It is the largest value across
the per-model configs (`qwen3_5_9b`), applied identically to all three arms via
`generation_overrides`. It was not measured and is **an open question for Jahid** (below).
Because it is the only bound on a runaway decomposition in the two uncapped arms, every run
reports `rows_at_max_new_tokens`: a length-truncated unguided output must be distinguishable
from a decomposition the model chose to end.

**7. Refuse rather than silently substitute.** Four load-time refusals, each pointing at the
decision it protects:

- **No retrieval input** (`retrieval.require_input`) — ADR
  [0006](./0006-drop-the-jury-fix-dataset-roles-and-the-few-shot-method.md) fixes the few-shot
  method (bi-encoder top-20 → cross-encoder top-5, typed masking, pool 2000) and it lives in an
  upstream artifact. Without it the run would fall back to random exemplars from the committed
  MetaQA pool — a different method under the label of the fixed one.
- **Row counts that are not the pinned set** (`eval_rows_per_hop: 200`, hops 2/3/4) — ADR
  [0007](./0007-musique-evaluation-set-reuses-v1-600-questions-200-per-hop.md) pins 600
  questions, 200 per hop, and ADR
  [0011](./0011-comparison-artifact-conventions-and-the-significance-claim-floor.md)'s stance is
  that a comparison across different evaluation sets is not a comparison. Asserted on the rows
  actually loaded, from either source. `--allow-unpinned-eval-set` is the explicit opt-out for
  fixture and smoke runs; it records `evaluation_set.pinned: false`, and such a run is not an
  experiment arm.
- **A query id whose hop depth cannot be parsed** — `retrieval.hop_fallback` is `null` for this
  config. In the `oracle_guided` arm a guessed depth would be injected as if it were the gold
  hop count; and every id in the pinned set parses, so one that does not is a sign the input is
  not that set. (`configs/decomposer.json` keeps the MetaQA fallback of 2 for unguided runs.)
- **An unguided prompt that mentions the hop count** (`unguided_prompt_must_omit_hop_count`),
  including a hardcoded hop line with no placeholder in it.

**8. Self-exclusion on every few-shot path.** An exemplar that is the query itself — same id,
or the same question text after normalization — is dropped from the ranked list before the
top-k is taken, on the reranked, bi-encoder and random-fallback paths alike, and the drop count
is recorded in the metrics. This is latent while the pool comes from MuSiQue train and the
queries from dev; it is leakage the moment a pool is drawn from the same split, and it would
read as a quality gain rather than as an error.

**9. Comparison artifacts are content-addressed.** Each run's snapshot records `prompt_sha256`
and `retrieval.input_sha256`. The retrieval file lives outside git (data never enters git), so
"the same path" does not prove "the same bytes"; per ADR 0011's spirit, the three arms'
comparability has to be checkable from the committed artifacts alone.

## Consequences

- The three arms cannot silently diverge: a second difference between the prompts, a different
  retrieval file, a different row count or a mislabelled arm is a non-zero exit, not a
  footnote in a metrics file.
- `prompt_unguided.md` is now a **derived** artifact. Editing a guided prompt requires
  regenerating its unguided sibling by the same rule, or the invariant fails.
- The hop-bearing-line rule is a regex over prompt lines (`{hop_count}`, or prose matching
  `hop[\s_-]*count`). A future prompt that expresses the hop count some other way ("split into
  three steps") would not be caught by it, and would need the rule extended.
- The smoke test and the fixture-based conditions test must pass `--retrieval-input` and
  `--allow-unpinned-eval-set`; a real arm passes neither of those opt-outs.
- Per-row results now carry `decomposition_raw`, `step_lines`, `generated_tokens`,
  `hit_max_new_tokens` and `stopped_at_step_line_cap` in every arm, so the dumps are shaped
  identically whether or not a cap applies. `results.json` is bigger for it.
- Two things this ADR deliberately does **not** contain: any threshold or success criterion for
  the comparison, and any statement about whether the router should be kept (recorded as open by
  ADR 0006 item 6, resolved separately by ADR
  [0010](./0010-keep-the-router-as-a-hop-count-regressor-prioritize-fine-tuning.md)).

## Open questions for Jahid

1. **The comparison currently rests on one model.** As configured, exactly one model folder can
   run these arms: `mistral_7b_instruct`. `qwen3_5_9b` ships an unguided prompt but is 9B —
   above the ~8B ceiling in `configs/model_limits.json` — so it is refused at load;
   `qwen2_5_3b` and `phi_4_mini_instruct` are within the ceiling but ship no unguided prompt.
   Whether to add an unguided prompt to another ≤8B folder (making the comparison two-model) is
   Jahid's decision; no agent should take it.
2. **Is `max_new_tokens: 256` the intended token cap?** It is a plumbing default carried from
   the largest per-model value, unmeasured. It bounds the uncapped arms, so it is part of the
   experiment's definition rather than an implementation detail.

## Alternatives considered

- **Keep v1's `prompt_unguided.md` byte-identical and note the difference in the write-up.**
  Rejected: it makes the arms differ in instruction content as well as hop information, which
  is exactly the confound the experiment exists to avoid. The porting convention loses to
  Jahid's stated design, and the deviation is recorded instead of hidden.
- **Fall back to random MetaQA exemplars when no retrieval input is given** (the previous
  behaviour). Rejected: it runs a different method under the label of ADR 0006's fixed one.
- **Check the pinned row count in a test only** (the previous behaviour). Rejected: the test
  checks the *files*, not what a given run actually loaded, so a retrieval input built over
  some other subset would have run happily.
- **Guess an unparseable hop depth from `hop_fallback`** (the previous behaviour). Rejected for
  this config: in the guided arm that mislabels the oracle.
- **Add replacement wording to the unguided prompt** (e.g. "use the minimal number of steps").
  Rejected: it is an instruction only one arm sees, i.e. a second difference between the arms.
