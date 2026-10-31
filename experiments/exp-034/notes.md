# exp-034 — exp-031 pool-size sweep re-parsed onto the PR #61 parser (CPU re-score)

Issue #51, ADR 0039 items 2-4 (cross-parser rule), ADR 0036 (Holm), ADR 0035. No inference, no GPU,
no lock (CPU only, exp-032 precedent). Code 57859f0, from the clean detached worktree
`/cta/users/fyilmaz/worktrees/exp-034`. The parser and evaluator code are the same as exp-032's
(`git diff 5c4e442..57859f0 -- scripts components configs src` is empty). Seed 42. Config:
`config.json`; numbers: `metrics.json`, built by `build_metrics.py`; driver: `run.sh`.

**Method (exp-032's).** For each of exp-031's 9 stored arms (Mistral-7B-Instruct-v0.3 4-bit, pinned 600):
`reparse_decompositions.py`, then `musique_decompositions_evaluator.py`, then exp-031's 8 compares
(sizeN vs size2000) with both sides re-parsed. Every step returned rc=0. Every eval and compare
records git 57859f0, dirty=False, 600/600 evaluated, 0 missing gold, and 600 aligned items.

## Re-parse: changed rows per arm (of 600)

| size | 2k | 3k | 4k | 5k | 6k | 7k | 8k | 9k | 10k |
|---|---|---|---|---|---|---|---|---|---|
| rows changed | 24 | 31 | 31 | 35 | 34 | 28 | 22 | 22 | 31 |
| steps removed | 57 | 59 | 75 | 52 | 79 | 50 | 46 | 31 | 54 |

## Identity check (blocker gate): PASSED

The re-parsed size2000 eval equals exp-032's re-parsed exp-020 eval (`runs/exp-032/exp-020/eval/eval_metrics.json`)
on all 257 numeric values: 0 mismatches and no key present on only one side. The re-parsed
`decomposition` strings are identical on 600/600 query ids, and both re-parses change 24 rows and
remove 57 steps. The reproducibility compare (size2000 vs re-parsed exp-020, outside the family) gives
a difference of 0.0 on all 8 family metrics, with 0 tests significant.

## Re-parsed own-eval values per size (descriptive, no direction asserted)

| size | break EM | SARI | GED (down) | chain validity | hop-count EM | under (down) | over (down) | step F1 |
|---|---|---|---|---|---|---|---|---|
| 2000 | 0.0400 | 0.5768 | 0.4816 | 0.9624 | 0.5217 | 0.1733 | 0.3050 | 0.1828 |
| 3000 | 0.0383 | 0.5875 | 0.4742 | 0.9587 | 0.5150 | 0.2000 | 0.2850 | 0.1961 |
| 4000 | 0.0517 | 0.5909 | 0.4703 | 0.9622 | 0.4867 | 0.1600 | 0.3533 | 0.1976 |
| 5000 | 0.0500 | 0.5879 | 0.4739 | 0.9598 | 0.5150 | 0.1817 | 0.3033 | 0.1940 |
| 6000 | 0.0450 | 0.5838 | 0.4793 | 0.9658 | 0.5017 | 0.1783 | 0.3200 | 0.1929 |
| 7000 | 0.0500 | 0.5861 | 0.4823 | 0.9723 | 0.4817 | 0.1950 | 0.3233 | 0.2005 |
| 8000 | 0.0500 | 0.5873 | 0.4784 | 0.9688 | 0.5133 | 0.1717 | 0.3150 | 0.2008 |
| 9000 | 0.0483 | 0.5861 | 0.4764 | 0.9617 | 0.5100 | 0.1967 | 0.2933 | 0.2010 |
| 10000 | 0.0483 | 0.5887 | 0.4713 | 0.9577 | 0.5283 | 0.1950 | 0.2767 | 0.2050 |

Beside-card terms (soft050, soft060, plan_connectedness) are in `metrics.json` `per_size`. They are
not ranked or tallied (ADR 0039 item 2). Old-parser values and their differences are in
`per_size.*.exp031_old_parser` / `reparsed_minus_old`.

## Holm family (exp-031's pre-fixed 64: 8 pairs sizeN vs size2000 × 8 metrics, alpha 0.05)

- **Uncorrected: 11 / 64 significant.** These are the same 11 cells as exp-031, and no uncorrected
  verdict changed. SARI at 3k (+0.0108, p=0.0162), 4k (+0.0141, p=0.0063), 5k (+0.0111, p=0.0346)
  and 10k (+0.0120, p=0.0300). Step F1 at 3k (+0.0133, p=0.0392), 4k (+0.0147, p=0.0414), 7k (+0.0177,
  p=0.0251), 8k (+0.0179, p=0.0211), 9k (+0.0182, p=0.0232) and 10k (+0.0221, p=0.0067).
  Over-decomposition at 4k (+0.0483, p=0.0261). Differences are sizeN minus size2000.
- **Holm-corrected: 0 / 64 significant.** The smallest Holm-adjusted p is 0.4001 (4k SARI), then
  0.4196 (10k step F1). exp-031's old-parser values were 0.2585 and 0.4410. After the family
  correction, no size is distinguishable from 2k on any of the 8 metrics.
- **Bootstrap CI vs paired t:** 0 disagreements across the 32 continuous cells.

**Caveats.** One model (Mistral-7B 4-bit), one seed, one nested pool chain. A non-significant
result means "not distinguishable", not "equal" (ADR 0036 item 4). No direction is asserted. These
are new-parser numbers, so they are comparable to exp-032's re-parsed Mistral arms. exp-031 keeps its
recorded old-parser values.
A first pass of the same driver was discarded, because the evaluator stamped `dirty=True` after
this build script was written into the worktree mid-run. Its outputs are kept under
`runs/exp-034/attempt1_dirty/` and are unused.
