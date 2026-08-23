# exp-015 — S2 GPU feasibility cell: best-pool prompting vs fine-tuned (Refs #46)

## What this run is

Issue #46 S2 GPU cell / ADR 0023 item 2.3, per
`docs/analysis/2026-08-23-s2-feasibility-prompting-vs-finetuned.md` §3 and ADR 0028 item 1: is
dynamic few-shot prompting (Mistral-7B-Instruct-v0.3, `unguided`) at its **best pool**
(`size2000_imbalanced`, the lead's ADR 0028 item 1 delegated choice) different from the
fine-tuned decomposer (exp-008 `full_train`) on the ADR 0007 pinned 600 — same eval set, same
decoding, same condition as exp-004/exp-008, with only the few-shot pool behind the retrieval
artifact moved (exp-014's reranked top-5 artifact over `size2000_imbalanced`). exp-008 needed no
re-run (`no_few_shot: true`; the pool decision cannot touch it).

## Commit discrepancy (recorded, not silently adopted)

This row was armed at `c69ef5f`. The waiter actually ran at `6a15158` (current `main` HEAD,
clean tree throughout), because PR #48 (`e3725c7`, Refs #40, the `decomposition_report_card`
six-term suite) landed while this run queued behind `exp-013`'s GPU-lock hold.
`git diff --stat c69ef5f 6a15158 -- scripts/ configs/ components/` shows the same 4
shared-pipeline files as exp-013's row, confirmed additive-only by exp-016's independent
coverage re-score (Refs #40). This row's `exp008_full_train` comparator is exp-016's
post-`e3725c7` re-score of exp-008's own already-committed predictions (byte-identical
predictions, more columns) — not a re-generation.

## Prompting arm, full pinned 600 (best pool, reranked top-5)

| statistic | value |
|---|---|
| exact_match_rate | 0.0450 |
| step_f1_macro | 0.1894 |
| ordered_step_accuracy_macro | 0.1662 |
| rouge_l_f1_macro | 0.5447 |
| hop_count_exact_match_rate | 0.5400 (324/600) |
| break_exact_match_rate | 0.0350 (21/600) |
| sari_macro | 0.5785 |
| ged_macro (lower better) | 0.4714 |
| chain_validity_macro | 0.9694 |
| under_decomposition_rate (lower better) | 0.2317 (139/600) |
| over_decomposition_rate (lower better) | 0.2283 (137/600) |
| composite_score (legacy, not headlined) | 0.3132 |

## `--compare` vs exp-008 `full_train` (n=600 aligned, same eval set as exp-008's own row)

All 7 bootstrap statistics + all 3 McNemar statistics = **10/10 significant, every one
favouring the fine-tuned arm.**

| statistic | prompting (a) | full_train (b) | a − b | test | significant |
|---|---|---|---|---|---|
| rouge_l_f1 | 0.5447 | 0.6647 | −0.1200 | bootstrap | yes, favours b |
| step_f1 | 0.1894 | 0.3411 | −0.1518 | bootstrap | yes, favours b |
| ordered_step_accuracy | 0.1662 | 0.3237 | −0.1575 | bootstrap | yes, favours b |
| sari | 0.5785 | 0.6823 | −0.1038 | bootstrap | yes, favours b |
| ged (lower better) | 0.4714 | 0.3094 | +0.1620 | bootstrap | yes, favours b |
| chain_validity | 0.9694 | 0.9942 | −0.0248 | bootstrap | yes, favours b |
| composite (legacy) | 0.3132 | 0.5236 | −0.2103 | bootstrap | yes, favours b |
| exact_match | 0.0450 | 0.1083 | −0.0633 | McNemar p=6.97e-08 (b=7, c=45) | yes, favours b |
| hop_count_exact_match | 0.5400 | 0.7100 | −0.1700 | McNemar p=1.94e-11 (b=66, c=168) | yes, favours b |
| break_exact_match | 0.0350 | 0.1017 | −0.0667 | McNemar p=1.03e-08 (b=6, c=46) | yes, favours b |

n=600, tests per ADR 0009/0011. Full table: `runs/exp-015/compare_vs_exp008_full_train/compare_metrics.json`.

## The non-obvious detail: asymmetry of length errors

The paired `--compare` above does **not** carry `under_decomposition`/`over_decomposition`,
because its comparator side (`runs/exp-011/exp008_full_train/eval_per_item.json`, per this row's
own approach column) was scored before those columns existed — flagged `NOT COMPARED` in
`compare_notes.md`. Reported here descriptively instead, from each side's own `eval_metrics.json`
(not a paired test):

| arm | under_decomposition | over_decomposition |
|---|---|---|
| prompting (this run) | 0.2317 (139/600) | 0.2283 (137/600) — near-symmetric |
| exp008_full_train (exp-016 re-score) | 0.2567 (154/600) | 0.0333 (20/600) — asymmetric |

Prompting under-decomposes marginally **less** than the fine-tuned arm (0.2317 vs 0.2567), but
over-decomposes about **6.9x more** (0.2283 vs 0.0333, ratio 6.85 ≈ seven times). So the
fine-tuned arm's length errors skew almost entirely toward the tolerable direction
(over-decomposition, ADR 0017), while prompting spreads its length errors roughly evenly across
both the tolerable and the not-tolerable direction.

## Reading

Every one of the 10 headline statistics favours the fine-tuned arm and every one is
significant (n=600) — best-pool reranked prompting does not close the gap to the fine-tuned
decomposer on this eval set. This is the S2 feasibility cell's answer: the pool choice does not
make prompting competitive.

- Code: commit `6a15158` on `main` (clean tree) for the decompose/evaluate/compare steps of this
  row (the wait outlasted the pre-run row's recorded commit `c69ef5f`).
- Config snapshot: `experiments/exp-015/config.json`
- Metrics: `experiments/exp-015/metrics.json`
- Full per-item detail (gitignored, stays on the box): `runs/exp-015/eval/`,
  `runs/exp-015/compare_vs_exp008_full_train/`
