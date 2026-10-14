# 0035 — Sampling variance is measured by one full-dev run per arm and offline stratified resampling

- **Status:** Accepted (lead decision under Jahid's delegation, 2026-10-14).
- **Context:** Issue #56. The supervisor asked how much the MuSiQue decomposition metrics would move
  on random 200-per-hop eval samples other than the ADR 0007 pinned 600, with plots. Until now every
  MuSiQue number came from that one draw.

## Decisions

1. **The population is the full answerable dev split.** The study's population is
   `musique_dev_full` (`configs/eval_sets.json`): every question of
   `musique/musique_ans_v1.0_dev.jsonl`, 2,417 rows, which is 1,252 / 760 / 405 at 2 / 3 / 4 hops.
   The runner and the evaluator select it explicitly with `--eval-set musique_dev_full` and assert
   it by id (`src/eval_sets.py`). Without that flag both scripts behave exactly as before.
2. **Each arm runs once over the population. Samples are scored offline.**
   `scripts/resample_eval_sets.py` draws K = 50 stratified samples of 200 per hop, seed 42. Draw
   d uses `random.Random(42 + d)`. Ids are drawn without replacement within a draw, and draws are
   independent of each other. The same 50 draws are reused for every arm, so the samples are
   paired. Each draw is scored with the evaluator's own `_aggregate`, overall and per hop.
3. **Why this is valid.** Decoding is greedy (`do_sample: false`). Each question's few-shot
   exemplars are retrieved for that question alone. So a question's decomposition, and its per-item
   score, does not depend on which sample the question sits in. One run over the population
   therefore gives the per-item scores from which any sample can be scored, with no more GPU time.
   The retrieval artifact for the population is built by exp-014's exact method
   (`scripts/build_dev_full_retrieval.py`).
4. **The overlap caveat goes with every result.** The dev split is finite, so draws overlap. A
   4-hop draw takes 200 of only 405 questions, so any two 4-hop draws share about half their
   questions. The spread is therefore a **lower bound** on the spread for truly independent samples.
   It measures **question-sampling variation only**: greedy decoding contributes no decoding
   variation.
5. **The pinned 600 stays the headline eval set.** It is a subset of the population. It is scored
   as one more subset and reported as a value inside each distribution, with its percentile.
   Nothing here replaces it or changes ADR 0007.

## Consequences

- Five GPU reruns over 2,417 questions (exp-029; commands in `experiments/exp-029/README-commands.md`).
- The full-dev retrieval artifact reproduces exp-014's exemplar ids and order for 589 of the 600
  pinned questions. All 11 differences reorder exemplars whose cross-encoder scores tie exactly.
  For 597 of the 600, the first 3 exemplars (the ones a k=3 prompt shows) are identical. So an
  arm's full-dev outputs on the pinned 600 are not guaranteed to equal its committed pinned run on
  those 3 questions. The few-shot-free fine-tuned arm is unaffected.
