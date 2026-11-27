# 0043 — The count-first condition (exp-037): a self-stated hop count, stripped before scoring

- **Status:** accepted. **Running the arm is Jahid's decision**: he approved it in session on
  2026-11-27 (the memo's C1.3). Items 2–7 are the implementer's conventions, written with
  exp-037 under the lead's brief and its code review of `ed01caa`. They are Jahid's to
  reverse.
- **Date:** 2026-11-27.
- **Amends** [0013](./0013-guided-vs-unguided-condition-conventions-on-the-musique-set.md)
  item 2 and its rejected alternative "an instruction only one arm sees", and
  [0024](./0024-router-readiness-query-id-keyed-predictions-and-the-router-guided-arm.md)
  item 8 (the condition-key whitelist). **Departs from** ADR 0041 item 1 ("Tier 0 only, no new
  arms", Jahid 2026-10-31) for this one arm, on Jahid's 2026-11-27 approval.

## Context

The external memo at the repo root (*"Jev, State of the Art, and a Plan to Improve Hop-Aware
Question Decomposition on MuSiQue"*) has a section C1.3, "Hop-count-first structured output
(self-guided oracle)". It proposes annotating exemplars as `[k hops]` and having the model emit
`[k hops]` before its steps. The idea targets the known under-decomposition at depth (for
example, exp-013's hop-count EM was 0.145 at 4 hops). On 2026-10-31 Jahid limited the memo's
follow-ups to re-analysis (ADR 0041 item 1). On 2026-11-27 he approved running this one arm
as exp-037.

The arm has to sit inside the conditions machinery of ADR 0013/0024. There, a condition may
differ from `unguided` only in what the prompt says about the hop count. Every other
difference is refused.

## Decision

1. **exp-037 runs** (Jahid, 2026-11-27): Mistral-7B and Qwen3.5-9B, 4-bit, on the pinned
   600, with the exp-014 retrieval artifact, against the unguided arm of the same model. It is
   one arm and is not a reopening of the memo's other proposals. ADR 0041 item 1 stands for
   everything else.
2. **`count_first` joins the `_CONDITION_KEYS` whitelist** next to `guided`,
   `stop_after_step_lines` and `hop_source`. The reason is the same as for those keys: it
   changes only the hop information in the prompt (the model is asked to state its own count)
   and how that statement is read back. The arm is `conditions.count_first` in
   `configs/decomposer_musique.json`. Absent, which is every condition that existed before,
   means off. `tests/test_count_first.py` proves that the rendered prompts of seven pre-existing
   arms hash byte-identically to `118d777`. Model, seed, decoding and retrieval stay shared and
   un-overridable.
3. **A departure from ADR 0013's rejected alternative "an instruction only one arm sees".**
   The count-first prompt adds exactly one rule line that the unguided prompt does not have:
   `- Before the steps, write the number of steps on its own line as [N hops].` It is inserted
   after `- One step per line.`, and the anchor line must occur exactly once. Each exemplar's
   decomposition also gains a first line `[N hops]`, with N being that exemplar's own gold
   step count. The rule line and the exemplar lines are the arm: asking for the count *is* the
   treatment, so the arms differ in one respect (the hop-count statement) by construction.
   The test reconstructs the unguided prompt byte for byte from the count-first one by removing
   the inserted line and the `[N hops]` lines. ADR 0013's objection was to wording that changes
   something other than the hop information, and that objection still holds for every other
   condition.
4. **The count is self-predicted and stripped before scoring.** `src/count_first.py` reads the
   first line of the post-processed generation as the model's count and removes it, along with
   any later line that is only a count statement. Only then is `decomposition` recorded, so the
   report card (ADR 0029) scores only the steps. Rows whose count line is absent, malformed or
   empty are **kept and scored**. Their number is reported (`count_first.status_counts`,
   `rows_without_count`) together with the count's accuracy against the gold hop count and its
   agreement with the number of steps emitted. `scripts/reparse_decompositions.py` applies the
   same strip to a count-first run, using the `count_first` block in the run's snapshot.
5. **`prompt_sha256` no longer identifies the rendered prompt of a count-first run.** The
   snapshot's `prompt_sha256` remains the hash of the prompt *file* (the unguided file). The
   template actually rendered is hashed in `count_first.effective_template_sha256`, and that
   field is the one to compare across count-first runs. Every other arm is unaffected, because
   its file is its template.
6. **Refusals.** `count_first` is refused with a guided arm (the prompt would carry an outside
   count as well), with a step-line cap (the stopper would count the count line as a step),
   with no few-shot block (the format would never be demonstrated), and with `--adapter`.
7. **The primary comparator is the same-machine re-run**, declared before any run in
   `experiments/exp-037/config.json`. exp-037 runs on Colab, while the stored baselines ran on
   an RTX 3090. Each model's unguided arm is therefore re-run on the same GPU, and
   count-first vs that re-run is the primary pair (Holm families F1/F2, ADR 0036). Count-first
   vs the stored baseline is secondary. The re-run counts as having *reproduced* the stored
   baseline only if 600/600 decompositions are identical strings.

## Consequences

- The conditions whitelist has four keys. ADR 0013 item 2 and ADR 0024 item 8 carry a dated
  note pointing here.
- Decoder runs are now resumable (`run_decomposer.py --resume`, `results.partial.jsonl`,
  `resume_key.json`), which every arm can use. A resume refuses unless the commit, the
  effective template, the seed, the retrieval sha256, decoding and the query ids all match.
- A reader comparing count-first prompts across runs must use `effective_template_sha256`,
  not `prompt_sha256`.
- This branch skips 0042 in its index because that number is taken on the exp-038 branch;
  the index is gap-free once both land.

## Alternatives considered

- **The exemplars alone demonstrate the format, with no rule line.** Rejected: nothing would
  ask the model to write the count, and a missing count line would then be indistinguishable
  from a model that ignored an unstated convention. One line is the minimum that makes the
  arm's request explicit.
- **The memo's alternative, injecting a predicted count as a hint ("likely 3 hops").**
  Rejected for this arm: it needs a trained hop classifier (memo C1.1), which is not built,
  and it would be an outside count, which is ADR 0024's router-guided arm.
- **The `Hops: N` format.** Rejected in favour of the memo's `[N hops]`, which the memo
  specifies.
- **The stored baseline as the primary comparator.** Rejected after code review: a GPU or
  library difference could then pass for a condition effect.
