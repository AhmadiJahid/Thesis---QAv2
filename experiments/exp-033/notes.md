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

## Results (completed 2026-10-14, CPU analysis after the GPU chain)

**Chain.** `runs/exp-033/chain_status.txt`: seed1/2/3 OK, `chain done`, `exit rc=0` at
2026-10-14T19:36:09+0300, HEAD 5c4e442. Lock released by the waiter. Checks per seed (decomposer
`config.json` + `metrics.json`): `few_shot_ordering: shuffled`, `few_shot_ordering_seed` = 1 / 2 / 3,
600 rows, 0 empty decompositions, git dirty 0; evals 600/600 evaluated, 0 missing gold.

**Sources.** Report-card values of exp-021 and exp-024 come from `runs/exp-021/eval` and
`runs/exp-024/eval`. Their beside-card values come from exp-032's re-scores
(`runs/exp-032/exp-021/eval`, `runs/exp-032/exp-024/eval`, 5c4e442, 0 changed rows). The original
evals predate those columns. The script checks that both sources give identical report-card values,
and they do. (c) uses `runs/exp-032/compare/exp-021_vs_exp-020` (family) and
`runs/exp-032/compare/exp-022_vs_exp-021` (size).
Built by `experiments/exp-033/build_metrics.py` -> `experiments/exp-033/metrics.json`.

### (b) Per ordering, and spread over the 5 orderings (SD = sample SD, n-1)

| term | as_is 021 | reversed 024 | shuf s1 | shuf s2 | shuf s3 | mean | SD | min | max | range |
|---|---|---|---|---|---|---|---|---|---|---|
| break EM (up) | 0.0550 | 0.0583 | 0.0567 | 0.0500 | 0.0583 | 0.0557 | 0.0035 | 0.0500 | 0.0583 | 0.0083 |
| SARI (up) | 0.5720 | 0.5771 | 0.5724 | 0.5768 | 0.5723 | 0.5741 | 0.0026 | 0.5720 | 0.5771 | 0.0051 |
| GED (down) | 0.4694 | 0.4728 | 0.4728 | 0.4766 | 0.4676 | 0.4719 | 0.0035 | 0.4676 | 0.4766 | 0.0090 |
| chain validity (up) | 0.9210 | 0.9282 | 0.9086 | 0.9100 | 0.9342 | 0.9204 | 0.0112 | 0.9086 | 0.9342 | 0.0256 |
| hop-count EM (up) | 0.5533 | 0.5400 | 0.5300 | 0.5233 | 0.5350 | 0.5363 | 0.0113 | 0.5233 | 0.5533 | 0.0300 |
| under-decomp (down) | 0.2933 | 0.2900 | 0.3167 | 0.3317 | 0.3050 | 0.3073 | 0.0172 | 0.2900 | 0.3317 | 0.0417 |
| over-decomp (down) | 0.1533 | 0.1700 | 0.1533 | 0.1450 | 0.1600 | 0.1563 | 0.0093 | 0.1450 | 0.1700 | 0.0250 |
| *beside:* soft step-F1 0.50 | 0.7237 | 0.7386 | 0.7228 | 0.7356 | 0.7409 | 0.7323 | 0.0085 | 0.7228 | 0.7409 | 0.0181 |
| *beside:* soft step-F1 0.60 | 0.5965 | 0.6131 | 0.6042 | 0.6213 | 0.6146 | 0.6099 | 0.0097 | 0.5965 | 0.6213 | 0.0249 |
| *beside:* plan_connectedness [nulls] | 0.8843 [24] | 0.8864 [25] | 0.8776 [27] | 0.8746 [28] | 0.8900 [21] | 0.8826 | 0.0063 | 0.8746 | 0.8900 | 0.0154 |

Beside-card rows are descriptive only. No ranking and no k-of-n tally is taken from them.

### (c) Ordering range vs the family and size differences (re-parsed exp-032 compares)

Differences are system_a minus system_b. Significance is the compare battery's own verdict
(bootstrap for SARI/GED/chain, McNemar for the rate terms), uncorrected. The class column uses a
descriptive convention chosen at analysis time and **not** part of the pre-fixed plan: "smaller" if
range < 0.5 x |diff|, "larger" if range > 2 x |diff|, otherwise "comparable". The ratio is given so
the reader can apply a different cut.

| term | ordering range | family 021-020 (sig?) | range/abs(fam) | class | size 022-021 (sig?) | range/abs(size) | class |
|---|---|---|---|---|---|---|---|
| break EM | 0.0083 | +0.0150 (sig, p=0.035) | 0.56 | comparable | -0.0217 (sig, p=0.007) | 0.38 | smaller |
| SARI | 0.0051 | -0.0047 (n.s.) | 1.08 | comparable | +0.0075 (n.s.) | 0.68 | comparable |
| GED | 0.0090 | -0.0122 (n.s.) | 0.74 | comparable | -0.0089 (n.s.) | 1.01 | comparable |
| chain validity | 0.0256 | -0.0414 (sig) | 0.62 | comparable | +0.0628 (sig) | 0.41 | smaller |
| hop-count EM | 0.0300 | +0.0317 (n.s., p=0.224) | 0.95 | comparable | +0.0633 (sig, p=0.012) | 0.47 | smaller |
| under-decomp | 0.0417 | +0.1200 (sig) | 0.35 | smaller | -0.1917 (sig) | 0.22 | smaller |
| over-decomp | 0.0250 | -0.1517 (sig) | 0.16 | smaller | +0.1283 (sig) | 0.19 | smaller |

Description. The ordering range is below both differences for under- and over-decomposition (ratio
0.16-0.35). It is below the size difference, and comparable to the family difference, for break EM,
chain validity and hop-count EM. For SARI and GED, the ordering range is about as large as both
differences (ratio 0.68-1.08), and none of those four differences was significant in its own compare.
Under this convention no term has an ordering range larger than either difference.

### (a) Each shuffle vs exp-021 as_is (`--compare`, `runs/exp-033/compare/seedS_vs_exp021_as_is`)

25 tests per seed (8 bootstrap + 5 McNemar + 12 paired t), no multiple-comparison correction.
- seed 1: 0/25 significant.
- seed 2: 2/25 significant, both on under-decomposition (diff +0.0383, so the shuffle under-decomposes
  more): McNemar p=0.045, t-test p=0.036. Every other test n.s.
- seed 3: 0/25 significant.
The beside-card terms were **not compared**. Plan (a) names exp-021's original per-item file
(`runs/exp-021/eval`), which predates those columns, and the evaluator reports them as not compared.
seed1's `composite_score` bootstrap reads -0.196: one row with an invalid reference sets the legacy
micro-pooled `reference_validity_micro` to 0.0 (ADR 0029: excluded from the card). It is not a
report-card term.

### (d) `few_shot_shuffle_coincidences` (rows out of 600; all 600 have k >= 2 exemplars)

| seed | equal to as_is | equal to reversed |
|---|---|---|
| 1 | 7 | 7 |
| 2 | 9 | 6 |
| 3 | 5 | 9 |

**Caveats.** This is one model (Qwen2.5-7B-Instruct 4-bit) and five orderings. Treating this spread as
the ordering noise for other models is an assumption (ADR 0039 Consequences). Two of the five orderings
are fixed rather than random (as_is, reversed), so this is not a sample from one shuffle distribution.
