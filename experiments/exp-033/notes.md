# exp-033 — H4 exemplar-ordering noise floor (Qwen2.5-7B, 3 seeded shuffles)

Issue #60, ADR 0039 item 5. Status at launch (2026-10-14): **queued** behind exp-031 (which holds
`runs/run.lock`). The waiter `waiter.sh` runs in detached tmux `exp033` from the clean worktree at
5c4e442 and writes `runs/exp-033/run.log` and `runs/exp-033/chain_status.txt`. The chain finishes
when `chain_status.txt` says `chain done`. A failure shows as `seedS DECOMPOSE_FAIL` /
`EVAL_FAIL` plus `exit rc=N`.

**Arm.** exp-021's committed command (Qwen2.5-7B-Instruct, 4-bit, unguided, seed 42, pinned 600,
exp-014 reranked top-5 retrieval, `configs/decomposer_musique.json` unedited). The only change is
`--ordering shuffled --ordering-seed S` for S in {1, 2, 3}, into `runs/exp-033/seedS/`. Each
seed is evaluated with exp-021's evaluator invocation (`--out-prefix eval --seed 42`).

**Analysis plan (fixed before the run; done by a fresh agent when the chain ends):**
- (a) Run each shuffle vs exp-021 with the evaluator `--compare` battery:
  `--compare runs/exp-033/seedS/eval/eval_per_item.json runs/exp-021/eval/eval_per_item.json
  --run-dir runs/exp-033/compare/seedS_vs_exp021_as_is --out-prefix compare --seed 42`.
- (b) For every report-card term, take the mean / SD / min / max across the 5 orderings: exp-021
  as_is, exp-024 reversed, and shuffle seeds 1-3.
- (c) Set that spread against the exp-021-vs-exp-020 (family) and exp-022-vs-exp-021 (size)
  differences.
- (d) Tally `few_shot_shuffle_coincidences` per seed (from each seed's decomposer `metrics.json`:
  rows whose shuffle equals the as_is order or the reversed order).

**Parser.** exp-021 and exp-024 re-parse to 0 changed rows (exp-032), so all 5 orderings share one
parser (ADR 0039). The family and size differences in (c) involve exp-020, which re-parses to
24/600 changed rows. Use exp-032's re-parsed compares (`runs/exp-032/compare/exp-021_vs_exp-020`,
`runs/exp-032/compare/exp-022_vs_exp-021`) per ADR 0039's cross-parser rule, and state which file
was used.

**Caveat.** The noise floor comes from one model. Applying it to other models is an assumption
(ADR 0039 Consequences).
