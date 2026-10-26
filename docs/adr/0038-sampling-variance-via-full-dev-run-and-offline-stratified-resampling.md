# 0038 — Sampling variance is measured by one full-dev run per arm and offline stratified resampling

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
   The evaluator stamps its full eval-set record into the per-item file, including
   `matches_named_set`. The resampler refuses to run unless every arm carries the same named set,
   matched, and the population equals that set's ids. Its one opt-out is
   `--allow-unnamed-population`, for smoke runs only, and its use is recorded in the outputs.
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
4. **The caveat goes with every result.** This wording is the lead's correction.
   - Each draw takes n = 200 questions per hop **without replacement from a finite hop stratum**
     (N = 1,252 / 760 / 405). The across-draw spread is therefore smaller than it would be for
     independent samples from an unlimited population.
   - The size of that shrinkage is the **finite-population correction**: the SD is scaled by
     sqrt(1 − n/N) ≈ **0.92 / 0.86 / 0.71** at 2 / 3 / 4 hops.
   - `resample_metrics.json` and `resample_summary.csv` report an FPC-corrected SD
     (`sd_fpc_corrected`), clearly labelled, beside the raw SD. The figures plot the raw spread.
   - Overlap between draws does not shrink the band. It only correlates the K draws with each
     other, which limits how precisely K = 50 draws estimate the band.
   - The spread measures **question-sampling variation only**: greedy decoding contributes no
     decoding variation.
5. **The pinned 600 stays the headline eval set.** It is a subset of the population. It is
   recomputed from each full-dev run as one more subset and reported as a value inside each
   distribution, with its percentile. Each arm's committed pinned-run value, and its difference
   from the recomputed one, is recorded beside it. Nothing here replaces the pinned 600 or changes
   ADR 0007.

## Consequences

- Five GPU reruns over 2,417 questions (exp-030; commands in `experiments/exp-030/README-commands.md`).
- The full-dev retrieval artifact matches exp-014's exemplar ids and order for **589 of the 600**
  pinned questions. The lead accepted this as is, with no splicing.
  - All 11 mismatches have an identical rerank-score sequence: the differences come from exact ties.
  - 7 of the 11 only reorder tied exemplars within the same top-5 set.
  - 4 have a different top-5 set, because a tie straddles the rank-5 cutoff: `2hop__115515_779396`,
    `2hop__554167_451128`, `2hop__765180_123956` and `2hop__168816_144857`.
  - The first 3 exemplars match for 597/600. That figure does not describe the prompt: reranked
    prompts show **k = 5** exemplars (prompts_log header `Few-shot source: reranked (k=5)`; the
    run notes' `k=3` is the mislabel tracked in issue #53), so the prompt differs on all 11.
  - So an arm's full-dev outputs on the pinned 600 are not guaranteed to equal its committed pinned
    run on those 11 questions *(corrected 2026-10-26: 3→11 / k=5, issues #53, #62)*. The few-shot-free
    fine-tuned arm is unaffected.
- The evaluator joins gold by question text, and some dev questions share their text with another
  row whose gold differs. That flaw is tracked in issue #57 and is left unchanged here. It affects
  the full-dev runs and the committed pinned runs alike.
