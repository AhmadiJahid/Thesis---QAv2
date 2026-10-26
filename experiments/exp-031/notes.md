# exp-031 — v2 pool-size sweep, 2k–10k (measurement only)

Issue #51, ADR 0035 (+ its 2026-10-14 amendment), ADR 0036. Code 3bd3595, run from the clean
detached worktree `/cta/users/fyilmaz/worktrees/exp-031`; seed 42. Config: `config.json` (snapshot
of `configs/pool_size_sweep_v2.json`); numbers: `metrics.json`, built by `build_metrics.py`
(CPU, read-only) from `runs/exp-031/`. Launcher: `waiter.sh`.

**Arm.** exp-020's arm with one factor moving: the exemplar pool size, 2000→10000 in 1k steps, one
nested chain (each pool an exact prefix of the next; 2k = the adopted `cc364662…6646` pool).
Mistral-7B-Instruct-v0.3 4-bit, unguided, as_is ordering, ADR 0007 pinned 600, imbalanced
construction, e5-small-v2 top-20 → cross-encoder rerank → top-5. **Exemplars in the prompt: 5**, read
off the prompts_log header (`Few-shot source: reranked (k=5)`) on all 165 logged prompts of every
size; the decomposer `config.json` `few_shot.k: 3` is the issue #53 mislabel.

Run-note sentence (required by the log row): *The pinned-600 query files were rewritten on
2026-09-14 (same ids, different order), so this sweep's retrieval sha256s differ from exp-014's by
row order only; the 2k-re-run vs exp-020 comparison is the reproducibility check.*

**Parser.** Old parser (pre-PR #61). Per ADR 0039 exp-031 is an old-parser run compared only with
old-parser runs (its own size2000, and exp-020's stored eval). These values are not comparable to
re-parsed (new-parser) Mistral numbers such as exp-032's exp-020 re-score.

## Chain

`runs/exp-031/chain_status.txt`: all 9 sizes OK, `chain done CFAIL=0` at 2026-10-14T18:50:18+0300,
total_wall_s 21339, `exit rc=0`, lock released by the waiter. Preflight matched every pool / top-20 /
top-5 sha256 against `expected_inputs` for all 9 sizes. Per size: 600 results rows (200/200/200),
0 empty decompositions, 600 reranked few-shot sources, git 3bd3595 clean; eval 600/600 evaluated,
0 missing gold; compares 600 aligned items, 3bd3595 clean. Decompose wall_s per size:
2445 / 2707 / 2255 / 2586 / 2405 / 2336 / 2064 / 2215 / 2256 (2k→10k). Peak VRAM 4818 MiB at
every size.

## Own-eval values per size (full precision in `metrics.json` `per_size`)

| size | break EM | SARI | GED (down) | chain validity | hop-count EM | under (down) | over (down) | step F1 |
|---|---|---|---|---|---|---|---|---|
| 2000 | 0.0400 | 0.5766 | 0.4846 | 0.9626 | 0.5117 | 0.1683 | 0.3200 | 0.1823 |
| 3000 | 0.0383 | 0.5876 | 0.4776 | 0.9587 | 0.5100 | 0.1900 | 0.3000 | 0.1953 |
| 4000 | 0.0517 | 0.5915 | 0.4751 | 0.9622 | 0.4767 | 0.1567 | 0.3667 | 0.1970 |
| 5000 | 0.0500 | 0.5878 | 0.4778 | 0.9603 | 0.5100 | 0.1717 | 0.3183 | 0.1930 |
| 6000 | 0.0450 | 0.5836 | 0.4823 | 0.9663 | 0.5017 | 0.1633 | 0.3350 | 0.1918 |
| 7000 | 0.0500 | 0.5862 | 0.4854 | 0.9728 | 0.4817 | 0.1850 | 0.3333 | 0.1999 |
| 8000 | 0.0500 | 0.5871 | 0.4813 | 0.9701 | 0.5050 | 0.1650 | 0.3300 | 0.2001 |
| 9000 | 0.0483 | 0.5857 | 0.4791 | 0.9617 | 0.5017 | 0.1900 | 0.3083 | 0.2005 |
| 10000 | 0.0483 | 0.5883 | 0.4742 | 0.9577 | 0.5167 | 0.1850 | 0.2983 | 0.2043 |

## Holm family (pre-fixed: 8 pairs sizeN vs size2000 × 8 metrics = 64 tests, alpha 0.05)

p fed to Holm: exact McNemar (break EM, hop-count EM, under, over), paired t (SARI, GED, chain
validity, step F1). Differences are sizeN minus size2000.

- **Uncorrected (evaluator verdict): 11 / 64 significant.** SARI at 3k (+0.0109, p=0.0144), 4k
  (+0.0149, p=0.0040), 5k (+0.0111, p=0.0353), 10k (+0.0117, p=0.0343); step F1 at 3k (+0.0130,
  p=0.0440), 4k (+0.0147, p=0.0427), 7k (+0.0176, p=0.0264), 8k (+0.0178, p=0.0221), 9k (+0.0181,
  p=0.0238), 10k (+0.0220, p=0.0070); over-decomposition at 4k (+0.0467, p=0.0347). The other 53
  are not significant.
- **Holm-corrected: 0 / 64 significant.** Smallest Holm-adjusted p: 0.2585 (4k SARI), then 0.4410
  (10k step F1). Under the family correction no size is distinguishable from 2k on any of the 8
  metrics.
- **Bootstrap CI vs paired t:** 0 disagreements across the 32 continuous cells.

Per-cell values (difference, p, Holm-adjusted p, discordant counts, bootstrap CI) are in
`metrics.json` `holm_family.pairs`.

## Reproducibility pair: size2000 re-run vs exp-020 (outside the family, uncorrected)

600/600 `decomposition_raw` and `decomposition` strings are identical to exp-020's by query_id.
All 8 family metrics are identical (difference 0.0); 0 tests significant in any section of the
compare. The paired t rows are degenerate (all differences zero, p = null). The 2k re-run
reproduces exp-020 exactly at the generation level.

**Caveats.** One model (Mistral-7B 4-bit), one seed, one nested pool chain. Non-significance is
"not distinguishable", not "equal" (ADR 0036 item 4). Old-parser numbers (ADR 0039).
