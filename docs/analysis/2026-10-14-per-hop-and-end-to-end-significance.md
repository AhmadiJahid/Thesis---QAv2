# Per-hop and end-to-end significance for slides A7, B1, D2, D3, E1

**Date:** 2026-10-14 · **Log:** `exp-029` (re-analysis, not a run) · **Refs:** #50, #16 · **ADR:** 0035 item 5
**Numbers:** `experiments/exp-029/metrics.json` (every row below, plus the full battery for each pair and stratum).
**Inputs and hashes:** `experiments/exp-029/config.json`. **Reproduce:** `.venv/bin/python -I experiments/exp-029/reanalysis.py` (about 6 s on CPU).

## Summary in plain English

1. **A7, hop-matched retrieval.** The hop-count gain is real only at 2 hops (0.710 → 0.845). At 3 and 4 hops it cannot be told apart from chance. At 4 hops, hop-matching **does lower step F1** (0.116 → 0.071), and that drop survives Holm. It is the only quality loss that does.
2. **B1, oracle hop count.** For Mistral the gain is a 2-hop effect (hop EM, GED), and nothing separates at 4 hops. For Qwen3.5-9B, hop-count EM rises at every depth. Its content gains hold overall (step F1, ordered accuracy) but are weak per hop. Its overall exact-match gain (p = 0.039) does not survive Holm.
3. **D2, fine-tune vs prompting by depth.** All 4 prompting models trail both fine-tunes on step F1 at every depth (24 of 24 tests survive Holm). At 4 hops the fine-tunes are the ones that stop short: they under-decompose more often than Mistral-7B, Qwen2.5-14B and Qwen3.5-9B (6 of 8 tests survive Holm). The exception is Qwen2.5-7B, which does not differ.
4. **D3, generalisation.** Every number on the slide is a tested claim: 7 of 8 tests survive Holm. The one that does not is chain validity against the full fine-tune, and the slide does not make that claim.
5. **E1, end to end.** Gold plans beat prompted plans and asking the question whole, overall and at 2 hops. The fine-tune beats asking whole on answer EM. **The fine-tune does not significantly beat prompted plans** (EM p = 0.12), and prompted plans do not beat asking whole. At 4 hops none of the four sources separates. The monotone ordering on the slide is therefore only partly tested.

## How to read the tables

- Differences are **a − b**. Each pair is id-aligned on the ADR 0007 pinned 600. The script asserts that both sides have the same ids, the same gold hop and the same question text. Each hop stratum has n = 200, so every stratum clears `min_items` 30.
- **Test:** for binary metrics, the exact McNemar test, with the discordant counts b/c (b = only a correct, or only a erroneous on the under-decomposition row). For continuous metrics, a paired percentile bootstrap: 10,000 resamples, seed 42, 95 % CI. The functions are the evaluator's own (`_paired_bootstrap`, `_mcnemar`, `_paired_t_test_row`), imported unchanged. The "sig" column is the evaluator's **uncorrected** verdict.
- **Holm** is a separate column, applied within each slide's family. The family size is given in each heading. The p-value fed to Holm is the McNemar p for binary metrics and the paired-t p for continuous ones, because the percentile bootstrap gives a CI, not a p. This is the convention exp-018 used. One cell has the two bases disagreeing: B1 Qwen ordered accuracy at 4 hops (bootstrap significant, t p = 0.054).
- The legacy `composite_score` is excluded from everything (ADR 0035 item 4).
- `ged`, `under_decomposition` and `over_decomposition` are lower-is-better.

## A7 — hop-matched retrieval (exp-009: a = mixed, b = oracle hop-matched). Family: 4 metrics × 4 strata = 16 tests

| stratum | metric | mixed | hop-matched | Δ | CI or McNemar b/c, p | sig | Holm adj p |
|---|---|---|---|---|---|---|---|
| overall | step F1 | 0.204 | 0.193 | +0.010 | [−0.006, +0.027] | – | 1 |
| overall | hop-count EM | 0.515 | 0.600 | −0.085 | 82/133, p 0.0006 | ✓ | **0.009** |
| 2 | hop-count EM | 0.710 | 0.845 | −0.135 | 9/36, p 6.6e-5 | ✓ | **0.001** |
| 3 | hop-count EM | 0.525 | 0.580 | −0.055 | 31/42, p 0.24 | – | 1 |
| 4 | hop-count EM | 0.310 | 0.375 | −0.065 | 42/55, p 0.22 | – | 1 |
| 2 | step F1 | 0.327 | 0.355 | −0.028 | [−0.061, +0.004] | – | 1 |
| 3 | step F1 | 0.168 | 0.153 | +0.015 | [−0.010, +0.040] | – | 1 |
| 4 | step F1 | 0.116 | 0.071 | +0.045 | [+0.017, +0.075] | ✓ | **0.038** |
| 4 | ordered acc. | 0.085 | 0.060 | +0.025 | [−0.001, +0.053] | – | 0.93 |
| 4 | exact match | 0.025 | 0.000 | +0.025 | 5/0, p 0.062 | – | 0.81 |

None of the other A7 cells is significant (2- and 3-hop ordered accuracy and exact match; overall ordered accuracy and exact match). **May claim:** the count gain is real overall and at 2 hops. At 4 hops, hop-matching lowers step F1. **May not claim:** a count gain at 3 or 4 hops. The 4-hop ordered-accuracy drop. Any gain in content at 2 hops: the visible +0.028 is not significant. The slide-27 gold on hop EM at 3 and 4 hops, and the red on 4-hop ordered accuracy, still assert untested verdicts.

## B1 — oracle hop count (a = unguided, b = oracle-guided; Mistral = exp-004 fp16, Qwen3.5-9B = exp-005 4-bit). Family: 9 metrics × 2 models × 4 strata = 72 tests

The table shows only the cells that are significant in at least one stratum. Every other cell (SARI, chain validity, Break EM, and the rest) is n.s. after Holm everywhere.

| model | metric | overall Δ (Holm) | 2-hop Δ (Holm) | 3-hop Δ (Holm) | 4-hop Δ (Holm) |
|---|---|---|---|---|---|
| Mistral | hop-count EM | −0.082 (**0.014**) | −0.140 (**0.001**) | −0.050 (1) | −0.055 (1) |
| Mistral | GED ↓ | +0.030 (**0.001**) | +0.047 (**0.005**) | +0.026 ✓unc (1) | +0.017 (1) |
| Mistral | ROUGE-L | −0.008 ✓unc (1) | −0.024 ✓unc (0.15) | −0.006 (1) | +0.005 (1) |
| Qwen3.5-9B | hop-count EM | −0.353 (**3.6e-46**) | −0.260 (**3e-14**) | −0.365 (**1e-13**) | −0.435 (**4e-18**) |
| Qwen3.5-9B | GED ↓ | +0.067 (**2e-13**) | +0.089 (**2e-8**) | +0.081 (**5e-6**) | +0.031 ✓unc (1) |
| Qwen3.5-9B | ordered acc. | −0.029 (**9e-6**) | −0.043 (**0.023**) | −0.031 (**0.029**) | −0.014 ✓unc (1) |
| Qwen3.5-9B | step F1 | −0.024 (**0.003**) | −0.035 ✓unc (0.17) | −0.024 ✓unc (0.95) | −0.012 (1) |
| Qwen3.5-9B | ROUGE-L | −0.012 ✓unc (0.47) | −0.025 (**0.049**) | −0.010 (1) | −0.001 (1) |
| Qwen3.5-9B | exact match | −0.012, p 0.039 ✓unc (1) | −0.030, 0/6 ✓unc (1) | −0.005 (1) | 0.000 (1) |
| Qwen3.5-9B | SARI | −0.005 (1) | −0.013 ✓unc (1) | −0.004 (1) | 0.000 (1) |

Bold means the test survives Holm. "✓unc" means the evaluator's uncorrected test is significant but Holm is not. All Δ are unguided − oracle, so negative means the oracle is higher, and on GED positive means the oracle is better. The overall rows reproduce exp-014's pairs exactly; for example, Mistral SARI gives CI [−0.0116, +0.0036].

**May claim:** for Mistral, oracle guidance improves hop count and GED overall, and the gain is concentrated at 2 hops. For Qwen3.5-9B it improves hop count at every depth, GED at 2 and 3 hops, ordered accuracy overall and at 2 and 3 hops, and step F1 overall. **May not claim:** a Mistral effect at 3 or 4 hops. The Mistral ROUGE-L gain (it is uncorrected only, and the CI upper bound is −0.0004). A Qwen exact-match gain, which does not survive Holm. "Six of nine move for Qwen" holds uncorrected; under Holm it is **4 of 9** overall (hop EM, GED, ordered accuracy, step F1). The precision and retrieval-artifact caveats from the plan audit still apply to any reading across the two models. This note tests within each model only.

## D2 — fine-tune vs prompting by depth (a = prompting 4-bit, b = fine-tune fp16). Family: 2 metrics × 8 pairs × 3 hops = 48 tests

**Step F1.** Every prompting arm is below both fine-tunes at every depth: **24/24 significant, 24/24 survive Holm**. The gaps run from −0.092 (Qwen3.5-9B vs pool-only FT, 2 hops) to −0.187 (Mistral vs full FT, 2 hops).

**Under-decomposition, lower is better.** Δ is prompting − FT. Holm-adjusted p is in brackets.

| prompting arm | vs FT | 2-hop | 3-hop | 4-hop |
|---|---|---|---|---|
| Mistral-7B (exp-020) | full (exp-008) | 0.000 vs 0.005 (1) | 0.220 vs 0.215 (1) | 0.285 vs 0.550 (**1.5e-6**) |
| Mistral-7B | pool-only (exp-026) | 0.000 vs 0.015 (1) | 0.220 vs 0.295 (0.55) | 0.285 vs 0.465 (**0.002**) |
| Qwen2.5-7B (exp-021) | full | 0.105 vs 0.005 (**2e-4**) | 0.320 vs 0.215 ✓unc (0.11) | 0.455 vs 0.550 (0.55) |
| Qwen2.5-7B | pool-only | 0.105 vs 0.015 (**0.002**) | 0.320 vs 0.295 (1) | 0.455 vs 0.465 (1) |
| Qwen2.5-14B (exp-022) | full | 0.010 vs 0.005 (1) | 0.075 vs 0.215 (**8e-5**) | 0.220 vs 0.550 (**2e-11**) |
| Qwen2.5-14B | pool-only | 0.010 vs 0.015 (1) | 0.075 vs 0.295 (**1e-8**) | 0.220 vs 0.465 (**1e-6**) |
| Qwen3.5-9B (exp-027) | full | 0.025 vs 0.005 (1) | 0.130 vs 0.215 ✓unc (0.11) | 0.250 vs 0.550 (**1e-10**) |
| Qwen3.5-9B | pool-only | 0.025 vs 0.015 (1) | 0.130 vs 0.295 (**4e-5**) | 0.250 vs 0.465 (**6e-6**) |

**May claim:** "the fine-tunes lead on step F1 at every depth" is tested against all four prompting models. "The fine-tunes stop short at 4 hops" is tested against Mistral-7B, Qwen2.5-14B and Qwen3.5-9B. **May not claim:** that they stop short against Qwen2.5-7B at 4 hops: that model under-decomposes as often (0.455 vs 0.550 / 0.465, n.s.). At 2 hops it is Qwen2.5-7B that runs short, not the fine-tunes. The precision confound still binds and must stay on the slide: the prompting arms are 4-bit and the fine-tune evaluations are fp16.

## D3 — generalisation to unseen 4-hop (a = exp-013 2/3-hop fine-tune; hop = 4 rows only, n = 200). Family: 4 metrics × 2 baselines = 8 tests

| metric | gen-FT | vs exp-004 unguided: b, Δ, test (Holm) | vs exp-008 full FT: b, Δ, test (Holm) |
|---|---|---|---|
| chain validity | 1.000 | 0.957, +0.043, [+0.022, +0.067] (**8e-4**) | 0.994, +0.006, [+0.000, +0.013] n.s. (0.09) |
| under-decomp. ↓ | 0.845 | 0.400, +0.445, 89/0 (**3e-26**) | 0.550, +0.295, 72/13 (**3e-10**) |
| hop-count EM | 0.145 | 0.300, −0.155, 13/44 (**2e-4**) | 0.410, −0.265, 12/65 (**4e-9**) |
| step F1 | 0.198 | 0.117, +0.081, [+0.054, +0.108] (**2e-7**) | 0.249, −0.051, [−0.078, −0.024] (**8e-4**) |

These reproduce exp-013's comparisons A and B exactly (hop EM p = 4.7e-5 and 5.9e-10). **May claim:** every number on the slide (200/200 valid, 169/200 too short, 14.5 % vs 30.0 %, 0.198 vs 0.117). It may also claim the omitted vs-full-FT row: 14.5 % vs 41.0 %, and step F1 0.198 vs 0.249. **May not claim** that its chains are more valid than the full fine-tune's.

## E1 — end to end (exp-017, reader Mistral-7B; a = the higher-ranked source). Family: 2 metrics × 6 pairs × 4 strata = 48 tests

Each cell shows answer EM Δ with its McNemar p, then answer F1 Δ with its bootstrap CI. Holm-adjusted p is in brackets. **Bold** means the test survives Holm.

| pair | overall | 2-hop | 3-hop | 4-hop |
|---|---|---|---|---|
| gold − FT | EM +0.038, p .023 (0.61); F1 +0.045 [+.013,+.077] (0.22) | EM +0.060, p .012 (0.35); F1 +0.046 ✓unc | EM +0.065 n.s.; F1 +0.081 [+.020,+.144] (0.35) | EM −0.010; F1 +0.007, n.s. |
| gold − prompted | **EM +0.067 (0.016)**; **F1 +0.078 (0.002)** | **EM +0.110 (0.017)**; **F1 +0.114 (0.009)** | EM +0.105, p .0025 (0.08); **F1 +0.145 (5e-4)** | EM −0.015; F1 −0.026, n.s. |
| gold − none | **EM +0.100 (6e-5)**; **F1 +0.101 (1e-4)** | **EM +0.185**; **F1 +0.185** | **EM +0.150**; **F1 +0.180** | EM −0.035; F1 −0.063, n.s. |
| FT − prompted | EM +0.028, p 0.12; F1 +0.033 [−.003,+.069], n.s. | EM +0.050 n.s.; F1 +0.069 [+.009,+.129] ✓unc (0.68) | EM +0.040; F1 +0.064, n.s. | EM −0.005; F1 −0.033, n.s. |
| FT − none | **EM +0.062 (0.042)**; F1 +0.056 ✓unc (0.17) | **EM +0.125 (0.017)**; **F1 +0.139 (0.009)** | EM +0.085 ✓unc (0.45); F1 +0.099 ✓unc (0.16) | EM −0.025 n.s.; F1 −0.070 [−.128,−.011] ✓unc **favours none** (0.56) |
| prompted − none | EM +0.033, p 0.09; F1 +0.023, n.s. | EM +0.075 (p .067); F1 +0.070, n.s. | n.s. | n.s. |

**May claim:** gold plans give better answers than prompted plans and than asking whole, overall and at 2 hops (and at 3 hops on F1). The fine-tune's plans beat asking whole at 2 hops and on overall EM. At 4 hops no source is distinguishable from any other: 0 of 12 tests survive Holm, and 1 is uncorrected-significant. **May not claim:** "better plans give better answers" as a monotone chain. The fine-tune vs prompted step was tested and does not separate: n.s. overall and at every hop after Holm. The prompted vs none step is n.s. everywhere. "At 4 they converge" may be said only as "are not separable at 4 hops", not as "are equal". The deck's red on the gold 4-hop 13.5 % has no test behind it: gold vs none at 4 hops is p = 0.35.

## Missing or skipped

- **Nothing was skipped for a missing file.** Every per-item file these slides need exists under `runs/`. A7, B1 and the exp-008 arm read the exp-016 re-scores, because those carry the suite columns. The script asserts that those re-scores are score-identical to the exp-011 files the original comparisons used, column by column. The exp-013 hop-4 baselines and exp-026's arm copy were checked the same way.
- **D2 overall** is not re-tested here, because it is D1's job. D1's tests are against exp-008 only. Overall prompting vs the pool-only FT stays as committed in exp-026 (vs exp-015 and exp-020 only).
- **Not done:** pairing across eval sets. All pairs here are on the pinned 600.
- **Suggestive, not tested:** that the A7 4-hop step-F1 drop and the D2 4-hop under-decomposition share a cause (exemplars or training that pull plans short at depth). No intervention separates the two.

## Ranked recommendations for the deck (Jahid's call)

1. **E1:** reword the headline to what is tested. Gold > prompted and gold > none, overall and at 2 hops. FT > none on EM. FT vs prompted is not separable. Nothing separates at 4 hops. Drop the red on 13.5 %. This is the largest gap between what the slide says and what was tested.
2. **A7:** give the slide one tested per-hop accent: the 2-hop hop-EM gain. Mark the 4-hop step-F1 drop as `bad` (Holm-surviving). Remove the gold at 3 and 4 hops and the red on 4-hop ordered accuracy.
3. **B1:** change "six of nine" for Qwen to "four of nine under Holm (six uncorrected)", and state that Mistral's gain is a 2-hop effect.
4. **D2:** add a per-hop accent. Step F1 is everywhere. Under-decomposition is at 4 hops except against Qwen2.5-7B.
5. **D3:** add the vs-full-FT row. Every claim on the slide already holds.
