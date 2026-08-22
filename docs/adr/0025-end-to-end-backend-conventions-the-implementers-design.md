# 0025. End-to-End Backend Conventions: The Implementer's Design

- **Status**: Accepted (design agent-authored, pending Jahid)
- **Date**: 2026-08-22

## Context

Issue #16 asked for two end-to-end backends — a MuSiQue answering backend and a MetaQA
compile-execute wrapper — so that a decomposition is judged by the answer it leads to and
not only by how it looks. The MuSiQue reader, its context regime and its answer metric are
Jahid's decisions and are recorded in ADR
[0019](./0019-musique-answering-backend-conventions.md); the MetaQA dataset role is his
supervisor's and is recorded in ADR
[0006](./0006-drop-the-jury-fix-dataset-roles-and-the-few-shot-method.md) §2.

Building the rest of issue #16 (PR #42) forced three conventions that **no ADR settles and
that an agent must not present as research decisions**. Gate-1 review asked for them to be
captured before merge, because each one changes how a number in the thesis reads. They are
recorded here, in the style of ADR
[0022](./0022-hop-matched-retrieval-the-implementers-design.md): this is the implementer's
design, it is changeable without touching any of Jahid's decisions, and where a choice is
genuinely his it says so and stops.

Nothing here is a measured result. No experiment was run for it; the numbers quoted are from
the fabricated fixtures under `tests/fixtures/` and say only that the code paths run.

## Decision

### 1. MetaQA end-to-end reports two denominators, and picks neither

`scripts/run_metaqa_compile_execute.py` reports exact match and mean Jaccard twice:

- **over executed items with gold** — of the decompositions that compiled *and* executed,
  how good is the answer set. This is what `scripts/compare_answer_accuracy.py` has always
  reported. It is a **conditional** number: every compile and execution failure is outside
  its denominator.
- **over all items with gold** — every prediction whose question has a gold answer, with a
  compile or execution failure **counted as the empty answer set it produced**.

The second exists because the first is not comparable with a MuSiQue number. The MuSiQue
answering backend puts a failed item *inside* its reported EM/F1 (ADR 0019; the metrics JSON
says so in `failed_generation_note`), so a MetaQA number that excludes failures and a MuSiQue
number that includes them are not measuring the same thing — and issue #41's cross-dataset
comparison needs them to be. Coverage is reported alongside both, so the gap between them is
always visible.

Three rules make the second denominator honest rather than clever:

- **A failure's contribution is computed, not assumed.** It is `jaccard(∅, gold)` through the
  same scorer, and it is an exact match iff that reaches the same `exact_match_jaccard`
  threshold the executed items are scored against. For MetaQA's non-empty gold that is 0 on
  both metrics; the code does not hard-code the 0.
- **One rule for every item, including the surprising one.** `jaccard(∅, ∅)` is 1.0, so an
  unexecuted item whose gold answer set is empty *is* an exact match by the definition
  printed next to the number. An earlier version guarded the mean for that case and left the
  exact-match count unguarded, which made the reported percentage contradict its own
  definition text (Gate-1 finding 2). The fix was to apply one rule uniformly and **name**
  the edge case in `empty_gold_set_note` when it fires, not to null it away. MetaQA gold
  answers are non-empty, so a non-zero count there points at a malformed gold line.
- **A measured zero is reported as zero.** A run where every decomposition fails reports
  0.0% and 0.0. Null is reserved for the one genuinely unmeasured case: no item in the run
  has a gold answer at all. Reporting null for total failure would hide a measured floor as
  a missing measurement (Gate-1 finding 1).

**Which of the two is *the* headline MetaQA metric is Jahid's call, not this record's.** The
script reports both with their definitions attached and refuses to choose.

**Element matching stays strip-only and case-sensitive** — the rule
`compare_answer_accuracy.py` has always used — and is therefore *not* MuSiQue EM's SQuAD
`normalize_answer` (ADR 0019 decision 3). The asymmetry is **recorded, not removed**:
`metric_definitions.answer_normalization` states it and the over-all-items definition repeats
it, because the two numbers are comparable in denominator but not in string strictness.
Aligning them would move every MetaQA number ever reported and is a scoring-rule decision for
Jahid with his supervisor.

### 2. A failure category that cannot be produced is reported as unavailable, with a reason — never as zero

Issue #16 named four step-level failure categories for the MuSiQue backend. Two of them
(`empty_retrieval`, `unresolvable_entity`) cannot exist in a design that has no retriever and
no knowledge base — ADR 0019 decision 2 fixes the context to the item's full paragraph list
and the code refuses any other policy. A third (`wrong_intermediate_answer`) would need the
gold sub-answers ADR 0019 deliberately keeps out of that path.

The convention: such a category is **emitted in the metrics JSON under `not_available` with
the reason for each**, and never as a zero count. A missing key reads as "the code forgot
this"; a zero reads as "this never happened"; and both are false when the truth is "this
cannot be measured here". This is CLAUDE.md's reproducibility rule — *if it is not measured,
say it is unmeasured* — applied to a taxonomy rather than to a metric.

The firing categories are counters, **not a partition**: a step can be asked with an
unresolved reference still in it *and* come back empty, and both facts matter to an error
analysis. Picking one "cause" per step would need a precedence rule nobody has agreed, so
`steps_clean` carries the "nothing wrong here" count instead. `src/step_failures.py` is the
one home for the category names and the classification rule; they are code, the same way the
MetaQA compile-error taxonomy is code.

**What this does not claim:** a step with no flags is not a step with a *correct* answer. Only
the final step's correctness is measured, and it is measured by the item's `answer_em` /
`answer_f1`. Extending `wrong_intermediate_answer` into a real per-step judgement means
reading MuSiQue's gold sub-answers, which amends ADR 0019 — Jahid's call.

### 3. A path that is not GRAG says so in every artifact it writes

ADR 0006 §2 routes MetaQA end-to-end evaluation through the supervisor's **GRAG** system.
GRAG is external. Searched on 2026-08-22 in this repo and in the v1 repo
(`/cta/users/fyilmaz/Thesis---QA`): there is **no interface, endpoint, client, repository or
handover** for it anywhere — only prose in ADRs 0006/0017/0023 and the 2026-08-12 meeting
cross-check, which records "no handover date, repo, or access mechanism was discussed".

The convention for this and any future substitute path: it is **neither stubbed as GRAG nor
presented as GRAG**. The compile-execute path carries a `backend_label`
(`metaqa_compile_execute_direct_kg`), a hard-coded `backend.grag_wired: false`, and a
`grag_status` sentence naming what is missing — in the metrics JSON, the config snapshot and
the run note alike. `grag_wired` is a **constant of the code, not a config knob**, so no flag
can flip the label without a source change and a review. A test asserts the labelling,
because the failure mode is silent: a number that looked like a GRAG number would be a claim
nobody made.

The seven things Jahid must obtain before GRAG can be wired are listed in PR #42 and are not
duplicated here.

### 4. Evidence conventions carried across from the MuSiQue side

- **Evaluation-set identity is asserted, not trusted.** A processed MetaQA question with no
  gold answer cannot be scored and would silently shrink every denominator, so the run is
  refused unless `--allow-unpinned-eval-set` is passed — the same shape as the MuSiQue
  backend's flag, for the same reason (ADR 0011: a number on a different set is not a
  comparison). MetaQA has no ADR-pinned id subset the way ADR 0007 pins MuSiQue's 600, so
  identity is carried by `evaluation_set.question_set_sha256`, a fingerprint over the sorted
  `hop<TAB>question` lines. **Equal fingerprints mean the same questions were scored**; that
  is the artifact-level check a comparison claim cites. It is a hash, so no dataset text
  leaves the run.
- **A number is traceable to the run that produced its decompositions.** The predictions
  file is content-addressed, and the decomposer run's sibling `config.json` supplies run id,
  commit, branch, dirty flag and config paths. Absence is recorded as `found: false` with a
  reason, never inferred.
- **A wrapper composes configs rather than copying their values.** `metaqa_compile_execute.json`
  owns no methodology knob: every knob is read from `metaqa_kg_eval.json` and
  `answer_accuracy.json`, so a value has exactly one home and cannot drift between the
  wrapper and the script it wraps. The composed configs' `seed` keys are recorded as unused,
  because the wrapper calls their functions directly under one seed.
- **An invariant only a real run could break gets a source-level guard** (ADR
  [0016](./0016-real-run-only-invariants-get-source-level-guards.md)).
  `backend.model_loaded: false` is such an invariant: no CPU test can notice a model load
  being added later, and the ~8B ceiling assertion would then be missing too. An AST guard
  with negative controls asserts the MetaQA path has no model-loading site.

## Consequences

- A MetaQA end-to-end number can be read against a MuSiQue one **in denominator**, which is
  what issue #41 needs, while the string-strictness difference stays on the record so the
  write-up does not overclaim the comparison.
- Every MetaQA artifact states which evaluation set it scored and which decomposer run it
  came from, so "same evaluation set" is checkable from files rather than from memory.
- The GRAG dependency stays visibly open. If Jahid obtains GRAG, it becomes a **second**
  labelled MetaQA path; its numbers do not retroactively reinterpret the compile-execute
  ones, because the two measure different systems.
- Three decisions are queued for Jahid and are *not* taken here: the headline MetaQA
  denominator, whether MetaQA element matching adopts SQuAD normalization, and whether
  `wrong_intermediate_answer` is implemented from gold sub-answers.
- If Jahid or his supervisor prefers different conventions, this record is superseded and the
  runs made under it are re-run rather than reinterpreted — the same clause ADR 0019 carries.

## Alternatives considered

- **One MetaQA denominator only.** Executed-only is what existed and is the smaller change,
  but it silently rewards a method that fails to compile (a failure leaves the denominator),
  and it is not comparable with MuSiQue. All-items-only was the other option; it discards the
  conditional number that says how good an answer set is *when* a plan runs, which is the
  useful diagnostic. Reporting both, labelled, was chosen over picking for Jahid.
- **Guarding the empty-gold case with a null.** Shipped first, caught in review: it made the
  reported percentage contradict its own printed definition. Rejected in favour of one
  uniform rule plus a note.
- **Switching MetaQA to SQuAD `normalize_answer`** so both datasets score strings alike.
  Defensible and possibly right, but it changes every MetaQA number and is a scoring rule, so
  it is surfaced rather than taken.
- **Omitting the unavailable failure categories** instead of emitting them with reasons.
  Cheaper, and it is exactly how a gap becomes invisible: a later reader cannot distinguish
  "not applicable" from "forgotten".
- **Stubbing GRAG** behind an interface so the MetaQA path "works". Rejected outright: a stub
  that produced numbers under GRAG's name would be a fabricated result, and ADR 0023 already
  records that the dependency stays flagged, not stubbed.
