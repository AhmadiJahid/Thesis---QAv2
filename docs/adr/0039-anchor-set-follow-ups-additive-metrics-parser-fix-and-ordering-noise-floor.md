# 0039 — Anchor-set follow-ups: two additive terms, a parser fix, and an ordering noise floor

- **Status:** accepted (Jahid, 2026-10-14, in session; operational details decided by the lead and
  his to reverse).
- **Context:** `docs/analysis/2026-09-20-anchor-set-qualitative-analysis.md` §8 ranked five
  hypotheses (H1–H5). The lead reviewed the note on 2026-10-14, approved its counts (the seed-43
  replication had reproduced them exactly), and corrected five sentences (§10 of the note). Jahid
  chose which hypotheses to act on. Tracked in issue #60.

## Decisions

1. **`step_f1` is not changed.** Its definition, code and every recorded value stay as they are.
   Jahid asked; the answer recorded here is no.
2. **H1: a soft step-match term is added beside the report card** (*"I agree with first two as
   recommended"*). A predicted step matches a gold step when their token-overlap F1 reaches a
   threshold. It is reported at **both 0.50 and 0.60** as a sensitivity pair, with the stopword
   list committed. No arm ranking or "k of n significant" tally is taken from it.
3. **H2: a `plan_connectedness` term is added beside the report card**: the fraction of non-final
   steps that some later step references. It measures the dangling step that `chain_validity`
   cannot see (86 anchor cases, 85 scoring `chain_validity` 1.0).
4. **H3: the step parser stops counting bare `Task:` / `Answer:` scaffold lines as steps.** Stored
   predictions are re-parsed from `decomposition_raw`, with no inference. The re-scored numbers are
   a **new** experiment, and exp-020 and exp-023 keep their recorded values.
5. **H4 runs** (*"lets run h4 as well"*). exp-021's arm (Qwen2.5-7B 4-bit, unguided, pinned 600,
   seed 42) is re-run under **3 seeded shuffles** of the few-shot exemplars, with everything else
   fixed. The spread of each report-card term over the 5 orderings (as_is = exp-021,
   reversed = exp-024, 3 shuffles) is the empirical ordering noise floor. It is read against the
   family and size differences, and is complementary to #56's question-sampling spread.
   About 45 GPU-minutes, queued behind exp-030 and exp-031 via the lock.
6. **H5 (gold hop count given to the fine-tune) is not pursued** (*"I do not need this one"*).

## Consequences

- The evaluator and decomposer changes go through Gate 1 on one branch. Their merge into the main
  checkout waits until exp-030's chain has finished, because that chain re-executes
  `components/decomposer/run_decomposer.py` and the evaluator from the main checkout for each arm.
- The H4 noise floor comes from one model. Applying it to the other models is an assumption,
  and is stated as one.

## Implementation choices (lead, 2026-10-14, PR #61)

- **The parser truncates at the scaffold label; it does not just drop that one line.** This mirrors the
  existing `Question:` rule. Text after `Task:` / `Answer:` is the model running on into a new
  example block, and in the `Answer:` rows it is often a second copy of the plan. On the stored runs,
  24/600 exp-020 rows lose 57 steps and 31/600 exp-023 rows lose 175. No row is left empty, and all
  other arms are unchanged. The analysis note's 0.3200→0.3050 / 0.2950→0.2817 assumed line-dropping,
  so the logged re-score supersedes those figures.
- **The beside-card terms get a paired bootstrap only, with no t-test** (ADR 0037).
- The answerer's model-folder `post_process` is untouched, so the reader never truncates at `Answer:`.
- **Item 4 is amended:** the parser *truncates the generation at the first `Task:` / `Answer:`
  anywhere in the text*, the same substring rule as `Question:`. It does not merely stop counting
  the scaffold line. So a real step containing one of these strings mid-line would be cut there.
  No #50-block arm has such a step; exp-005 has one.
- **`step_f1_soft_*`:** the NLTK English stopword list (198 entries; source and sha256 in
  `configs/musique_eval.json`), each entry normalised the way a step is normalised. Matching is
  one-to-one by maximum bipartite matching, so the result does not depend on order.
- **`plan_connectedness`:** null for plans with fewer than 2 steps. Those rows are excluded from
  the mean and counted, so the denominator depends on what the model emitted. Gold scores 1.0 on
  600/600.
- **Cross-parser comparisons (review I-3):** after the merge, every new Mistral MuSiQue run is
  parsed with the fixed parser. Older Mistral-family runs are not 0-affected (exp-015 7/600, exp-004
  10/3/10, exp-005 1, exp-009 10/11, exp-012 1, v1 pool-sweep 29–47/750 per arm). **A comparison
  between a new-parser run and any of these uses the old run's re-parsed file**, never its stored
  one. This does not touch exp-030's Mistral arm (it runs before the merge) or exp-031 (it runs from
  a worktree pinned at `3bd3595`). Both are old-parser runs compared only with old-parser runs.
  Qwen and the fine-tunes re-parse to 0 changed rows.
