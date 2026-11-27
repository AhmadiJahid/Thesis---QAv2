# 0047 — exp-041: training-seed variance of the pool-only fine-tune (seeds, comparators, families)

- **Status:** accepted. The lead decided this on 2026-11-27 under Jahid's grant of full authority over technical choices. The items below are the implementer's conventions under that brief. Jahid may reverse them.
- **Date:** 2026-11-27.
- **Context:** The thesis's claim that "fine-tuning beats prompting" rests on one training run per adapter: exp-026 vs exp-015. Thesis Ch6 lists this as the "one run per arm" limit, and the writing audit ranks it (d) 3. The prompting arms decode greedily, so the training seed is their only source of run-to-run variance. Pool draws are a separate question. exp-041 measures the training-seed term for the headline adapter.

## Decision

1. **Only the training seed moves: 42, 43, 44.** 43 and 44 are the next integers after the repo seed, fixed before any run. `train_lora.py --seed` already threads the seed in three places:
   - the global seed, set before model load and LoRA attach, which fixes the LoRA init;
   - trl's `seed` and `data_seed`, which fix data order and dropout;
   - the arm's selection. This arm has no cap, so the training set is the whole pool under every seed.

   **No pipeline code changes.** The rest is exp-026's training and evaluation, from unedited configs (tested).
2. **Seed 42 is retrained on the same GPU.** The primary spread is therefore three seeds on one machine. The stored exp-026 (RTX 3090) sits in a secondary set and serves as a hardware reference. Without this, a GPU or library difference would inflate the apparent seed spread (the ADR 0043 item 7 convention).
3. **The prompting comparator is exp-015's command, re-run on the same GPU** (primary). exp-015 is the precision-matched arm exp-026 was compared with. The stored exp-015, re-parsed by exp-032, is the secondary comparator. exp-020 (4-bit) stays context only, as it was for exp-026, and is not repeated.
4. **Families (ADR 0036), fixed before any run:**
   - One family of 28 per seed: seed vs the prompting re-run × 7 report-card terms × 4 strata. One family per seed fits because each seed stands in for "exp-026 had been run with this seed".
   - The stored exp-026 vs exp-015 reading is recomputed by the same code as the reference family.
   - Every seed cell is read against the reference for sign, uncorrected verdict and Holm verdict.
   - "The headline holds across seeds" is claimed only for cells where the reference and all three seeds are Holm-significant in the same direction.
   - The spread (mean, SD, range across seeds, for the fine-tune and for its gain) is descriptive at n = 3.
5. **Gates:**
   - Training gate: exp-036's gate plus the seed.
   - Decode gate: row counts, retrieval sha, prompt parity, anchors, commit.
   - Output health: at most 1% empty and at most 5% at `max_new_tokens`. These limits are arbitrary and stated as such.
   - A 20-question canary runs before the first GPU-heavy step and before each adapter's full evaluation.
   - An interrupted training restarts from step 0 (ADR 0044 item 5).

## Consequences

- The thesis can give the training-seed spread of every report-card term. It can also say for which cells the fine-tune-vs-prompting verdict is stable across seeds, or that a cell's verdict is seed-dependent.
- Pool-draw variance (writing audit (d) rank 2) is not addressed here.

## Alternatives considered

- **Two new seeds only, against the stored exp-026 and exp-015.** Rejected as the primary design. It mixes a hardware term into the spread and into each comparison. It is kept as the secondary set, which costs nothing extra.
- **One pooled Holm family over all seeds.** Rejected. It answers "is any seed significant", which is not the question. The question is whether each single-run reading would have reproduced exp-026's verdict.
- **More seeds.** The defence deadline sets the budget. Three seeds give a range and an SD, but not a distribution, and the report says so.
