# exp-013 — generalisation_2_3hop LoRA arm vs unguided baseline vs full_train fine-tune (Refs #41)

## What this run is

Issue #41 instrument 1 / ADR 0023 item 3 / issue #46 S3: a decomposer LoRA fine-tuned on 2-hop
and 3-hop MuSiQue training data only (`generalisation_2_3hop` arm), evaluated on the full ADR
0007 pinned 600 (200/hop). The substantive question is whether this arm generalises to the
held-out depth (hop=4, never seen in training).

Waiter (`runs/exp-013/waiter.sh`, gitignored) ran end to end once the run lock freed: train the
adapter, decode the full pinned 600 (`unguided`/`--no-few-shot`, same convention as exp-008),
evaluate, filter to the 200 hop=4 items, and run one paired `--compare` against a pre-built
hop=4-only baseline file. **This update adds a second `--compare`** (Task 1 below) that the
original armed row did not yet have.

## Commit discrepancy (recorded, not silently adopted)

This row was armed at `6679fef`. The waiter actually ran — training, decoding, evaluation, and
both comparisons — at `6a15158` (current `main` HEAD, clean tree throughout), because PR #48
(`e3725c7`, Refs #40, the `decomposition_report_card` six-term suite) landed while this run
queued behind `exp-012`'s GPU lock and then its own wait. `git diff --stat 6679fef 6a15158 --
scripts/ configs/ components/` shows 4 shared-pipeline files touched
(`scripts/musique_decompositions_evaluator.py`, `scripts/decomposition_junk_battery.py`,
`configs/musique_eval.json`, `configs/decomposition_junk_battery.json`). This is not silently
adopted: exp-016 (Refs #40) independently re-scored nine other arms across this exact commit
range and confirmed every pre-existing metric reproduces exactly (additive-only change) — so
the six-term suite showing up in this row's numbers is new coverage, not a changed comparison
protocol.

## Full pinned 600 — decomposition report card (the reported primary, ADR 0029)

| term | value |
|---|---|
| break_exact_match_rate | 0.0850 (51/600) |
| sari_macro | 0.6640 |
| ged_macro (lower better) | 0.3450 |
| chain_validity_macro | 0.9942 |
| hop_count_exact_match_rate | 0.6217 (373/600) |
| under_decomposition_rate (lower better) | 0.3617 (217/600) |
| over_decomposition_rate (lower better) | 0.0167 (10/600) |

Also: exact_match_rate 0.0950, step_f1_macro 0.3292, ordered_step_accuracy_macro 0.3078,
rouge_l_f1_macro 0.6473. Legacy composite_score 0.5110 (frozen byte-identical, not headlined,
issue #40).

## Per-gold-hop breakdown (n=200/hop) — the clearest generalisation result in the study so far

| gold hop | hop_count_EM | under_decomp | break_EM | GED (lower better) | chain_validity |
|---|---|---|---|---|---|
| 2 | 0.960 (192/200) | 0.000 (0/200) | 0.200 (40/200) | 0.2065 | 0.995 (per-item sum 199.0/200) |
| 3 | 0.760 (152/200) | 0.240 (48/200) | 0.055 (11/200) | 0.3136 | 0.9875 (per-item sum 197.5/200 — a few items score 0.5, not strictly binary) |
| 4 (held out) | 0.145 (29/200) | 0.845 (169/200) | 0.000 (0/200) | 0.5149 | 1.000 (200/200, exact) |

Reading (counts, not adjectives): on the unseen depth the arm keeps emitting well-formed chains
(chain_validity 200/200 = 1.000) that are systematically too short (under-decomposition
169/200 = 0.845 items), and exact structural agreement is 0/200 (break_exact_match). It is not
that the model produces garbage on hop=4 — it produces valid but too-short chains.

## Comparison A — vs exp-004 unguided prompting baseline, hop=4 only (pre-existing)

System a = this arm's 200 hop=4 items; system b = exp-004 `unguided` re-scored at exp-011,
filtered to its 200 hop=4 items. n=200, paired bootstrap (10000 resamples) + McNemar + paired
t-test (ADR 0009/0011).

| statistic | a | b | a − b | test | significant |
|---|---|---|---|---|---|
| rouge_l_f1 | 0.5551 | 0.4758 | +0.0793 | bootstrap | yes, favours a |
| step_f1 | 0.1982 | 0.1172 | +0.0810 | bootstrap | yes, favours a |
| ordered_step_accuracy | 0.1557 | 0.0846 | +0.0712 | bootstrap | yes, favours a |
| sari | 0.6051 | 0.5554 | +0.0496 | bootstrap | yes, favours a |
| ged (lower better) | 0.5149 | 0.5731 | −0.0581 | bootstrap | yes, favours a |
| chain_validity | 1.0000 | 0.9568 | +0.0432 | bootstrap | yes, favours a |
| composite (legacy) | 0.3962 | 0.1332 | +0.2629 | bootstrap | yes, favours a |
| exact_match | 0.0000 | 0.0250 | −0.0250 | McNemar (b=0, c=5) | no, underpowered |
| hop_count_exact_match | 0.1450 | 0.3000 | −0.1550 | McNemar p=4.71e-05 (b=13, c=44) | **yes, favours b (baseline)** |
| break_exact_match | 0.0000 | 0.0250 | −0.0250 | McNemar (b=0, c=5) | no, underpowered |

Full table: `runs/exp-013/compare_vs_unguided_4hop/compare_metrics.json`.

## Comparison B — vs exp-008 full_train fine-tune, hop=4 only (Task 1, this update)

The question comparison A does not answer: is the fine-tuned win dataset-bound? System a = this
arm's 200 hop=4 items (trained on 2/3-hop only); system b = exp-008 `full_train` re-scored at
exp-016 (`e3725c7`), filtered to its 200 hop=4 items (trained on ALL hops). Same base model,
same LoRA recipe, differing only in whether hop=4 examples were in the training mix. Item ids
verified to intersect exp-013's hop4 file at exactly 200/200 (zero symmetric difference) before
running. n=200, same protocol as comparison A.

| statistic | a (2/3-hop only) | b (full_train) | a − b | test | significant |
|---|---|---|---|---|---|
| rouge_l_f1 | 0.5551 | 0.6033 | −0.0482 | bootstrap | yes, favours b |
| step_f1 | 0.1982 | 0.2494 | −0.0513 | bootstrap | yes, favours b |
| ordered_step_accuracy | 0.1557 | 0.2270 | −0.0712 | bootstrap | yes, favours b |
| sari | 0.6051 | 0.6439 | −0.0388 | bootstrap | yes, favours b |
| ged (lower better) | 0.5149 | 0.3995 | +0.1155 | bootstrap | yes, favours b |
| chain_validity | 1.0000 | 0.9942 | +0.0058 | bootstrap | no (CI touches 0) |
| decomp_mean (contingency) | 0.4470 | 0.5337 | −0.0867 | bootstrap | yes, favours b |
| composite (legacy) | 0.3962 | 0.4472 | −0.0511 | bootstrap | yes, favours b |
| exact_match | 0.0000 | 0.0200 | −0.0200 | McNemar (b=0, c=4) | no, underpowered |
| hop_count_exact_match | 0.1450 | 0.4100 | −0.2650 | McNemar p=5.92e-10 (b=12, c=65) | **yes, favours b** |
| break_exact_match | 0.0000 | 0.0200 | −0.0200 | McNemar (b=0, c=4) | no, underpowered |
| over_decomposition (lower better) | 0.0100 | 0.0400 | −0.0300 | McNemar (b=2, c=8) | no |
| under_decomposition (lower better) | 0.8450 | 0.5500 | +0.2950 | McNemar p=4.63e-11 (b=72, c=13) | **yes, favours b (worse for a)** |

Full table: `runs/exp-013/compare_vs_exp008_hop4/compare_metrics.json`.

**Reading (Task 1's answer, counts not adjectives, no keep/drop verdict asserted beyond what the
tests support):** on the held-out depth, the arm that never saw hop=4 (169/200 under-decompose,
29/200 exact hop count) is significantly worse than the arm that did see it in training
(110/200 under-decompose, 82/200 exact hop count) on step count, step quality, and edit
distance — but the two arms are statistically indistinguishable on chain well-formedness
(chain_validity +0.0058, CI touches 0, 200/200 vs per-item sum 198.83/200). The generalisation
gap issue #41 asked about is present here, and on this evidence it reads as a
length/granularity gap specifically, not a chain-validity gap.

- Code: commit `6a15158` on `main` (clean tree) for all steps of this row, including the
  original armed run (the wait outlasted the pre-run row's recorded commit).
- Config snapshot: `experiments/exp-013/config.json`
- Metrics: `experiments/exp-013/metrics.json`
- Full per-item / per-arm detail (gitignored, stays on the box): `runs/exp-013/eval/`,
  `runs/exp-013/compare_vs_unguided_4hop/`, `runs/exp-013/compare_vs_exp008_hop4/`
