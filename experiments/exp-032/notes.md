# exp-032 — CPU re-score with the scaffold-label parser fix and the beside-card terms

Issue #60, ADR 0039 (items 2-4 and the 2026-10-14 implementation choices). Code 5c4e442, run
from a clean detached worktree; seed 42; no GPU, no lock. Config: `config.json`; numbers:
`metrics.json` (assembled by `build_metrics.py` from `runs/exp-032/`).

**What ran.** The stored `results.json` of 11 arms (exp-008 full_train, exp-015, exp-020..028)
was re-parsed with `scripts/reparse_decompositions.py` (no inference), re-evaluated, and the
19 distinct arm pairs the earlier compares covered were re-run with both sides re-parsed.

**Re-parse.** Changed rows: exp-020 24/600 (57 steps removed), exp-023 31/600 (175), exp-015
7/600 (21), every other arm 0/600. This matches PR #61's counts exactly. Every row of every arm
reproduced its stored decomposition under the recorded parser before the fix was applied.

**Identity check.** For the 8 zero-change arms, every committed own-eval value present in
`experiments/<id>/metrics.json` equals the re-score exactly (32-33 report-card values per arm,
52-115 values in total; 0 mismatches). For exp-008 the check also covers exp-011's, exp-016's and
exp-015's rescored blocks. Scoring the *stored* exp-015/020/023 files with the same code gives
back their committed values, so every old->new difference below comes from the parser fix alone.

**Old -> new (committed -> re-parsed), report card + step_f1.**

| term | exp-015 | exp-020 | exp-023 |
|---|---|---|---|
| break EM | 0.0350 -> 0.0350 | 0.0400 -> 0.0400 | 0.0483 -> 0.0483 |
| SARI | 0.5785 -> 0.5786 | 0.5766 -> 0.5768 | 0.5831 -> 0.5834 |
| GED (lower better) | 0.4714 -> 0.4699 | 0.4846 -> 0.4816 | 0.4813 -> 0.4744 |
| chain validity | 0.9694 -> 0.9694 | 0.9626 -> 0.9624 | 0.9629 -> 0.9625 |
| hop-count EM | 0.5400 -> 0.5433 | 0.5117 -> 0.5217 | 0.5267 -> 0.5383 |
| under-decomp | 0.2317 -> 0.2333 | 0.1683 -> 0.1733 | 0.1783 -> 0.1867 |
| over-decomp | 0.2283 -> 0.2233 | 0.3200 -> 0.3050 | 0.2950 -> 0.2750 |
| step_f1 | 0.1894 -> 0.1894 | 0.1823 -> 0.1828 | 0.1958 -> 0.1973 |

exp-020 and exp-023 keep their recorded values (ADR 0039 item 4); these are the new readings.
The analysis note's predicted over-decomposition (exp-020 0.3050, exp-023 0.2817) assumed line
dropping; the measured truncation values (0.3050, 0.2750) supersede them (ADR 0039).

**Verdict changes vs the committed compares (5, all in pairs touching exp-020/023):**
- exp-023 vs exp-020, step_f1: bootstrap n.s. -> significant (diff 0.0134 -> 0.0145); t-test n.s. -> significant (p 0.061 -> 0.045).
- exp-027 vs exp-020, hop-count EM: McNemar significant -> n.s. (diff 0.0583 -> 0.0483, p 0.021 -> 0.058).
- exp-027 vs exp-020, under-decomposition: McNemar n.s. -> significant (diff -0.0333 -> -0.0383, p 0.080 -> 0.042); t-test n.s. -> significant (p 0.066 -> 0.033).
No verdict changed for any pair touching exp-015 or for any other pair. exp-026 vs exp-008 and
015 vs 008 were checked against both of their committed compares. All verdicts are uncorrected
(no multiple-comparison family is defined for this re-score).

**Beside-card terms** (`step_f1_soft_050/060`, `plan_connectedness` + null counts,
`gold_plan_connectedness`) are in `metrics.json` per arm and per compare. Gold plan connectedness
is 1.0 on 600/600 for every arm. Per ADR 0039 they are reported beside the card: no ranking and no
"k of n" tally is taken from them.
