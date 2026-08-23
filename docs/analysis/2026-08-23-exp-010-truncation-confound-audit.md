# exp-010: is the hop-count gap a 128-token truncation artifact? — audit

- **Date:** 2026-08-23
- **Question (issue #14):** exp-010's only significant axis was `hop_count_exact_match`
  (imbalanced 0.5133 and clustered 0.5133 both beating balanced 0.4667). The sweep ran through
  `configs/decomposer.json`, whose generation cap is the model default **128** new tokens
  (exp-004/exp-008 ran `configs/decomposer_musique.json` at 1024), 4-bit quantized. Hop-count EM
  counts step lines; a truncated generation has fewer step lines than the model intended.
  So: is the gap a truncation artifact rather than a pool-construction effect?
- **Verdict: the truncation confound is DISMISSED.** Both gaps survive on cap-free rows, one
  strengthens, and the mechanism runs the *opposite* way from the hypothesis: rows that hit the
  cap **over**-decompose, they do not under-decompose.
- **Method:** read-only over committed metrics + on-box `runs/pool_sweep/**`. No GPU, no
  generation, `runs/run.lock` not taken, no pipeline code changed. Reproducible by
  `python docs/analysis/2026-08-23-exp-010-truncation-confound-audit.py`; machine-readable output
  in `docs/analysis/2026-08-23-exp-010-truncation-confound-audit.json`.

## 1. Truncation IS reliably detectable per row

Signal used: **`hit_max_new_tokens`**, written per row by the decomposer into
`runs/pool_sweep/decomposer/<cell>/results.json`, alongside `completion_tokens`, `step_lines`
and `stopped_at_step_line_cap`. Its definition is
`components/decomposer/run_decomposer.py:1697` — `completion_tokens >= max_new_tokens`, where
`completion_tokens` is the tokenizer's own count of the generated tensor
(`run_decomposer.py:1685`), not an estimate.

Why this is the right signal, verified rather than assumed, across all 18 cells:

- `generation.max_new_tokens = 128` and `quantization = 4bit` in every cell's own config snapshot
  (`runs/pool_sweep/decomposer/<cell>/*/config.json`) — the 128 cap is confirmed from the
  artifacts, not from the summary prose.
- The flag agrees with `completion_tokens >= 128` on **13,500/13,500 rows** (0 mismatches), and
  `max(completion_tokens) = 128` exactly in every cell — generation was budget-stopped, never
  over-run.
- `stopped_at_step_line_cap` fired on **0 rows in all 18 cells** (this sweep sets no
  `max_step_lines`), so there is exactly one cut-off mechanism to reason about, not two.
- Residual false-positive: a generation whose EOS would have landed on token 128 exactly is
  indistinguishable from a budget stop. This is the only ambiguity and it is negligible relative
  to the 44–73 cap-hits per cell.

The headline-cell counts reproduce `experiments/exp-010/notes.md` exactly: **50 / 48 / 73** of 750
for imbalanced / balanced / clustered at `biencoder_plus_ce` + `typed`.

## 2. Cap-hit rate does not track hop-count EM across cells

Per cell (n=750 each; `cap` = share of rows with `hit_max_new_tokens`; `hopEM` from
`runs/pool_sweep/eval/<cell>/eval_per_item.json`):

| balance | variant | mode | n_hit | cap | hopEM | hopEM \| hit | hopEM \| no-hit |
|---|---|---|---|---|---|---|---|
| imbalanced | biencoder_only | raw | 44 | 0.0587 | 0.4667 | 0.3409 | 0.4745 |
| imbalanced | biencoder_only | typed | 52 | 0.0693 | 0.4707 | 0.1923 | 0.4914 |
| imbalanced | biencoder_only | uniform | 55 | 0.0733 | 0.4640 | 0.2909 | 0.4777 |
| imbalanced | biencoder_plus_ce | raw | 44 | 0.0587 | 0.4840 | 0.2045 | 0.5014 |
| **imbalanced** | **biencoder_plus_ce** | **typed** | **50** | **0.0667** | **0.5133** | **0.2800** | **0.5300** |
| imbalanced | biencoder_plus_ce | uniform | 55 | 0.0733 | 0.4880 | 0.2727 | 0.5050 |
| balanced | biencoder_only | raw | 50 | 0.0667 | 0.4493 | 0.4000 | 0.4529 |
| balanced | biencoder_only | typed | 46 | 0.0613 | 0.4293 | 0.3043 | 0.4375 |
| balanced | biencoder_only | uniform | 64 | 0.0853 | 0.4160 | 0.2656 | 0.4300 |
| balanced | biencoder_plus_ce | raw | 50 | 0.0667 | 0.4480 | 0.3800 | 0.4529 |
| **balanced** | **biencoder_plus_ce** | **typed** | **48** | **0.0640** | **0.4667** | **0.3750** | **0.4729** |
| balanced | biencoder_plus_ce | uniform | 65 | 0.0867 | 0.4640 | 0.3077 | 0.4788 |
| clustered | biencoder_only | raw | 47 | 0.0627 | 0.4747 | 0.2553 | 0.4893 |
| clustered | biencoder_only | typed | 56 | 0.0747 | 0.4880 | 0.3214 | 0.5014 |
| clustered | biencoder_only | uniform | 59 | 0.0787 | 0.4680 | 0.2542 | 0.4863 |
| clustered | biencoder_plus_ce | raw | 52 | 0.0693 | 0.5040 | 0.3077 | 0.5186 |
| **clustered** | **biencoder_plus_ce** | **typed** | **73** | **0.0973** | **0.5133** | **0.3425** | **0.5318** |
| clustered | biencoder_plus_ce | uniform | 44 | 0.0587 | 0.4773 | 0.3182 | 0.4873 |

Two facts, in tension with the confound story:

- **Within every cell, cap-hit rows score far worse on hop-count EM** (0.19–0.40) than cap-free
  rows (0.43–0.53). Truncation is not innocuous at the row level.
- **Across cells, cap-hit rate carries no signal about hop-count EM**: Pearson r = 0.0965
  (p = 0.7033), Spearman ρ = 0.0624 (p = 0.8058), n = 18 cells.

The headline three make the point bluntly: **clustered has the highest cap-hit rate of all 18
cells (73, 9.73%) and still ties imbalanced (50, 6.67%) at the top of hop-count EM (0.5133)**,
while balanced — the loser — has the *lowest* cap-hit rate of the three (48, 6.40%). If truncation
drove the axis, the ordering would be the reverse of what was measured. (The 3-point Pearson
r = 0.561 is *positive* — more truncation associated with *higher* hop-EM — and with n = 3 it is
descriptive only, not a test.)

## 3. The decisive test: restricted to rows that never hit the cap

ADR 0009 protocol, seed 42, α = 0.05, `configs/musique_eval.json` parameters (10,000 bootstrap
resamples, chunk 1000), functions imported unmodified from
`scripts/musique_decompositions_evaluator.py` (`_mcnemar`, `_paired_t_test_row`). Restriction is
**union-drop**: a row is excluded if *either* arm truncated it, which is the only way to keep the
comparison paired and id-aligned (CLAUDE.md's same-evaluation-set rule). ADR 0009 assigns
`hop_count_exact_match` to **exact McNemar**; the bootstrap CI on the hop-EM rate is reported here
as a supplement beyond protocol, not as the protocol's own leg.

| comparison | n | A | B | diff | b / c | McNemar p | boot 95% CI | paired-t p |
|---|---|---|---|---|---|---|---|---|
| imbalanced vs balanced — full | 750 | 0.5133 | 0.4667 | +0.0467 | 160 / 125 | **0.0438** | [+0.0027, +0.0907] | 0.0381 |
| **imbalanced vs balanced — cap-free** | **657** | 0.5312 | 0.4825 | **+0.0487** | 140 / 108 | **0.0488** | [+0.0000, +0.0944] | 0.0421 |
| clustered vs balanced — full | 750 | 0.5133 | 0.4667 | +0.0467 | 152 / 117 | **0.0380** | [+0.0040, +0.0893] | 0.0328 |
| **clustered vs balanced — cap-free** | **640** | 0.5422 | 0.4859 | **+0.0563** | 134 / 98 | **0.0214** | [+0.0094, +0.1016] | 0.0180 |
| imbalanced vs clustered — full | 750 | 0.5133 | 0.5133 | +0.0000 | 129 / 129 | 1.0000 | [−0.0427, +0.0413] | 1.0000 |
| imbalanced vs clustered — cap-free | 637 | 0.5306 | 0.5338 | −0.0031 | 107 / 109 | 0.9458 | [−0.0471, +0.0424] | 0.8919 |

(Rows dropped: imbalanced-vs-balanced 93 of 750 — 50 ∪ 48, intersection 5; clustered-vs-balanced
110 — 73 ∪ 48, intersection 11; imbalanced-vs-clustered 113 — 50 ∪ 73, intersection 10.)

**Both gaps survive on the protocol's own test, and both grow.** clustered vs balanced strengthens
outright (p 0.0380 → 0.0214, diff +0.0467 → +0.0563). imbalanced vs balanced survives on McNemar
(p = 0.0488) and the paired t (p = 0.0421) with a *larger* effect (+0.0487); its supplementary
bootstrap CI lower bound lands **exactly on 0.000000**, so by the strict "CI excludes zero" rule it
would fail — by 0.36 pp of bootstrap mass. Characterised precisely: 2.15% of the 10,000 resampled
differences are < 0 and 0.36% are exactly 0, so the 2.5th percentile falls on the atom at zero
(the 2.6th percentile is +0.0015). That is discreteness at the boundary of a binary-mean statistic
at n = 657, not a sign change.

Restricting to the **cap-hit rows only** shows no difference anywhere (imb vs bal n = 93,
p = 0.7428; clu vs bal n = 110, p = 1.0000; imb vs clu n = 113, p = 0.8776) — as expected at those
n, and reported for completeness rather than as evidence.

**Power check (this is the "did the effect disappear or did the test lose power" control).** 1000
random subsets of the same size as each cap-free set, drawn without replacement from the full 750
(seed 42): at n = 657, McNemar reaches p < 0.05 in only **395/1000 draws (39.5%)**, median
p = 0.0611; at n = 640, **431/1000 (43.1%)**, median p = 0.0567. So a ~12–15% random thinning of
this eval set loses significance more often than it keeps it. The cap-free restriction did the
opposite — it *lowered* both p-values relative to the full set. The effect is not a power
survivor; it is concentrated in exactly the rows the confound hypothesis says are clean.

## 4. Why the confound premise itself is wrong

The hypothesis assumed a truncated decomposition has *fewer* step lines than intended. Measured on
the headline cells (`predicted_hop_count` vs `gold_hop_count` in `eval_per_item.json`, split by
`hit_max_new_tokens`):

| cell | subset | n | mean pred steps | mean gold steps | mean signed error | over-count rate | under-count rate |
|---|---|---|---|---|---|---|---|
| imbalanced | cap-hit | 50 | 5.24 | 3.04 | +2.20 | 0.660 | 0.060 |
| imbalanced | cap-free | 700 | 3.31 | 3.00 | +0.32 | 0.296 | 0.174 |
| balanced | cap-hit | 48 | 5.31 | 3.06 | +2.25 | 0.604 | 0.021 |
| balanced | cap-free | 702 | 3.56 | 3.00 | +0.56 | 0.397 | 0.130 |
| clustered | cap-hit | 73 | 5.08 | 3.14 | +1.95 | 0.630 | 0.027 |
| clustered | cap-free | 677 | 3.32 | 2.99 | +0.34 | 0.309 | 0.160 |

A row that hits the 128-token cap is a **runaway**, not a stub: it already emits ~5 step lines
against a gold of ~3, and only 2–6% of cap-hit rows under-count. The cap is what *stops* the
runaway. Raising it to 1024 would let those rows over-decompose further, which — if it moved
hop-count EM at all — would move it *down* on precisely those rows, not up.

This also names the real mechanism behind the axis, on cap-free rows where truncation cannot be
implicated: **balanced over-decomposes more** (over-count rate 0.397) than imbalanced (0.296) or
clustered (0.309). The exp-010 gap lives where the cap never bound.

## 5. Cap-hits by hop depth: a gradient in the questions, not a link to pools

Cap-hit counts per gold hop depth, headline cells (denominators are 250/250/250 in **every** cell —
the eval set is the same pinned 750-query dev sample regardless of pool strategy):

| cell | hop 2 | hop 3 | hop 4 | χ² (2 df) | p |
|---|---|---|---|---|---|
| imbalanced | 13/250 | 22/250 | 15/250 | 2.871 | 0.2379 |
| balanced | 14/250 | 17/250 | 17/250 | 0.401 | 0.8185 |
| clustered | 19/250 | 25/250 | 29/250 | 2.307 | 0.3156 |
| pooled (3 headline cells) | 46/750 | 64/750 | 61/750 | 3.532 | 0.1711 |

**No single cell shows a significant depth dependence.** Descriptively, pooling all 18 cells gives
rates 5.73% / 6.64% / 8.82% for hop 2/3/4 (258/299/397 of 4500 each) and 17 of 18 cells have a
higher cap-hit rate at hop 4 than at hop 2 — a consistent gradient. Its χ² (34.5, p = 3.19e-08) is
reported as **descriptive only and not a valid test**: the 18 cells score the same 750 questions,
so the counts are heavily non-independent and that p-value is anti-conservative.

Crucially, this does **not** entangle pools with truncation the way the brief's mechanism
supposed. The pools differ in hop composition (imbalanced 1461/413/126, balanced 667/667/666,
clustered 1396/495/109, per `experiments/exp-010/metrics.json`), but that is the composition of the
**few-shot retrieval pool**, not of the evaluated set — every cell is scored on the identical
250/250/250 split. The depth mix being measured is held fixed by construction, so a depth-linked
truncation rate cannot shift the hop-count comparison between strategies. Deeper questions do
truncate somewhat more, and that is a property of MuSiQue questions worth knowing; it is not a path
from pool construction to the exp-010 result.

## 6. What this does and does not settle

- **Confound: dismissed.** Three independent lines agree — the cap-free paired tests survive and
  strengthen (§3), the mechanism runs the wrong direction (§4), and the highest-truncation cell is
  tied for *best*, not worst (§2).
- **Effect on ADR 0028 item 1: strengthened, not reopened.** That decision excluded `balanced` on
  the hop-count axis and treated imbalanced/clustered as measured-indifferent. Both readings hold
  on cap-free rows: balanced still loses (p = 0.0488 / 0.0214, n = 657 / 640) and
  imbalanced-vs-clustered is still a clean null (diff −0.0031, p = 0.9458, n = 637). This note
  takes no position on which pool is better and does not reopen the choice.
- **Suggestive, not established:** the depth gradient in cap-hits (§5) — consistent across cells,
  significant only under a pooling that violates independence.
- **Unmeasured:** what these cells would score at a 1024-token cap. Nothing here predicts that
  number; §4 argues only about the *direction* a change would take on the ~6–10% of rows affected.
- **Also unmeasured:** any effect of the 128 cap on step-level metrics (`step_f1`,
  `ordered_step_accuracy`, ROUGE, EM). exp-010 found those flat across strategies; this audit did
  not re-test them under restriction.

## 7. If a 1024-token re-run is wanted, what it costs

Not run — that is Jahid's call. Measured from exp-010's own per-row `latency_seconds` and
`completion_tokens` (n = 13,500 rows): OLS gives **0.04619 s per completion token** (intercept
0.119 s). Total measured generation time across the 18 decompose cells was 29,021 s (**8.06 h**),
1505–1781 s per cell — consistent with the ~1515–1792 s/cell recorded in
`experiments/exp-010/notes.md`.

Cost of a re-run at 1024 = that floor, plus decoding for the 954 currently-capped rows (across all
18 cells) that would continue past 128:

| extra tokens per capped row | added time | full 18-cell total |
|---|---|---|
| +128 | +1.57 h | ~9.6 h |
| +256 | +3.13 h | ~11.2 h |
| +512 | +6.27 h | ~14.3 h |
| +896 (run to 1024) | +10.97 h | ~19.0 h |

A headline-only re-run (the 3 `biencoder_plus_ce` + `typed` cells, 171 capped rows) is roughly
**1.3 h floor + 0.3–2.0 h** ≈ 1.5–3.3 h. Both estimates assume unchanged throughput on an
uncontended GPU and exclude the 5–9 s/cell eval stage.

## Artifacts read

- `experiments/exp-010/metrics.json`, `experiments/exp-010/notes.md`, `experiments/exp-010/config.json`
- `runs/pool_sweep/decomposer/<18 cells>/results.json` and `.../<timestamp>/config.json`
- `runs/pool_sweep/eval/<18 cells>/eval_per_item.json`
- `runs/exp-010/compare/{imbalanced_vs_balanced,balanced_vs_clustered,imbalanced_vs_clustered}/compare_config.json`
- `components/decomposer/run_decomposer.py` (flag definition), `scripts/musique_decompositions_evaluator.py` (test functions), `configs/musique_eval.json` (test parameters)
