# exp-030 — run note (issue #56, ADR 0038)

Five decomposer arms, each run once over `musique_dev_full` (all 2,417 answerable MuSiQue dev
questions, 1,252 / 760 / 405 at 2 / 3 / 4 hops), scored by the evaluator with
`--eval-set musique_dev_full`, then K = 50 stratified 200-per-hop draws scored offline
(seeds 42..91, the same draws for every arm) by `scripts/resample_eval_sets.py` in strict mode.
Measurement only, no direction asserted. Commands exactly as in `README-commands.md`; chain script
`tools/chain.sh`, startup checker `tools/check_startup.py`, determinism check `tools/determinism.py`
(runner-side helpers, not pipeline code).

## Provenance

- Launched at `4136473` (= `c96191f` + this lane's log row). Another session committed to `main` while
  the chain ran (`3bd3595`..`5226b84`: pool-sweep scripts, ADR 0039, an analysis note, the exp-031 log row),
  so the per-arm snapshots record `bafe44b` (Qwen2.5-7B) and `5226b84` (the other four arms + resampler).
  `git diff c96191f 5226b84` is **empty** for `components/`, `src/`, `configs/decomposer_musique.json`,
  `configs/musique_eval.json`, `configs/eval_sets.json`, `configs/paths.json`,
  `scripts/musique_decompositions_evaluator.py`, `scripts/resample_eval_sets.py` and `requirements.txt`.
  The "dirty tree" flag in every snapshot means Jahid's uncommitted `Slides/` files (7 files), not run code.
- Retrieval artifact `runs/exp-030/retrieval/dev_full_imbalanced_rerank_top5.jsonl`, sha256
  `98e4f7201b0b11cabfd9d1df9ee154464ebc230b7296c27be98ae02cd1efa114` (re-hashed before launch; asserted in every arm's startup dump).
- Every arm passed its startup check (`arms/<arm>/startup_check.txt`): `musique_dev_full` matched,
  2,417 rows (1,252/760/405), input sha256, quantization, `few_shot_ordering` as_is, parameter count
  equal to the arm's committed pinned run. Every arm evaluated 2,417 rows, 0 missing gold.

## Wall-clock (from `chain_status.txt` and each arm's `decomposer_metrics.json`)

| arm | pinned run | wall s (process) | generation s | peak GPU memory.used, 30 s samples (whole GPU, MiB) | eval wall s |
|---|---|---|---|---|---|
| Qwen2.5-7B 4-bit | exp-021 | 3031 | 2947.2 | 6360 | 7 |
| Pool-FT Mistral fp16 + adapter, no few-shot | exp-026 | 3219 | 3204.9 | 14710 | 6 |
| Qwen2.5-14B 4-bit | exp-022 | 5982 | 5864.8 | 10567 | 8 |
| Qwen3.5-9B 4-bit | exp-027 | 5710 | 5611.9 | 8137 | 7 |
| Mistral-7B 4-bit | exp-020 | 10063 | 10049.8 | 4833 | 12 |

Chain: 2026-10-14 04:57:49 → 12:45:14 (+03:00), 28,045 s ≈ 7 h 47 min.

## Determinism check on the pinned 600 (`determinism_check.json`)

New full-dev `decomposition` vs the arm's committed pinned `results.json`, exact string match.

| arm | match /600 | mismatching ids | identical-prompt rows |
|---|---|---|---|
| Qwen2.5-7B | 596 | 2hop__226204_69048, 2hop__115515_779396, 2hop__818692_113442, 2hop__765180_123956 | 589/589 |
| Pool-FT (no few-shot) | 600 | — | 589/589 |
| Qwen2.5-14B | 597 | 2hop__226204_69048, 2hop__115515_779396, 2hop__323534_810083 | 589/589 |
| Qwen3.5-9B | 600 | — | 589/589 |
| Mistral-7B | 598 | 2hop__115515_779396, 2hop__168816_144857 | 589/589 |

**Correction to the brief's expectation (≥597 for few-shot arms).** The prompt shows all k = 5
exemplars (`configs/decomposer_musique.json` `retrieval.k: 5`), not 3, so **all 11** tie rows of the
equivalence check have a different prompt (7 reorder tied exemplars, 4 change the top-5 set), not 3.
The correct floor for a few-shot arm is therefore 589/600. Every mismatch above is one of those
11 ids, and every arm matches **589/589** on the rows whose prompts are identical. Qwen2.5-7B's 596
is below the brief's 597; it is explained by the k = 5 prompt and is not a nondeterminism signal
(e.g. `2hop__818692_113442`: its two tied rank-4/5 exemplars swap order; the output changed
"What date did #1 die?" → "When did #1 die?"). The no-few-shot arm is 600/600 as expected.
Consistently, `committed_pinned` in `resample_metrics.json` shows a difference of exactly 0 on every
term for the two 600/600 arms, and |difference| ≤ 0.0017 overall for the others.

## Headline: overall report card across 50 stratified 200/hop draws (`resample_summary.csv`, scope overall)

SD is the raw across-draw SD; the FPC-corrected SD removes the finite-population shrinkage
(sqrt(1 − n/N) ≈ 0.92 / 0.86 / 0.71 at 2 / 3 / 4 hops). The pinned value is the pinned-600 subset
recomputed from this run, with its percentile within the 50 draws. Per-hop rows are in the CSV and `fig2`.

| arm | term | dir | mean ± SD | SD (FPC-corrected) | 95% band (p2.5–p97.5) | pinned-600 subset (recomputed) | pinned pct |
|---|---|---|---|---|---|---|---|
| Qwen2.5-7B | `break_exact_match_rate` | ⇑ | 0.0398 ± 0.0072 | 0.0085 | 0.0270–0.0530 | 0.0533 | 97 |
| Qwen2.5-7B | `sari_macro` | ⇑ | 0.5749 ± 0.0063 | 0.0074 | 0.5610–0.5865 | 0.5717 | 32 |
| Qwen2.5-7B | `ged_macro` | ⇓ | 0.4711 ± 0.0083 | 0.0097 | 0.4560–0.4841 | 0.4702 | 40 |
| Qwen2.5-7B | `chain_validity_macro` | ⇑ | 0.9166 ± 0.0088 | 0.0111 | 0.9017–0.9367 | 0.9210 | 72 |
| Qwen2.5-7B | `hop_count_exact_match_rate` | ⇑ | 0.5445 ± 0.0156 | 0.0197 | 0.5154–0.5757 | 0.5517 | 70 |
| Qwen2.5-7B | `under_decomposition_rate` | ⇓ | 0.2972 ± 0.0142 | 0.0177 | 0.2720–0.3213 | 0.2933 | 32 |
| Qwen2.5-7B | `over_decomposition_rate` | ⇓ | 0.1583 ± 0.0116 | 0.0139 | 0.1370–0.1768 | 0.1550 | 40 |
| Pool-FT (Mistral) | `break_exact_match_rate` | ⇑ | 0.0940 ± 0.0107 | 0.0117 | 0.0757–0.1117 | 0.0967 | 60 |
| Pool-FT (Mistral) | `sari_macro` | ⇑ | 0.6818 ± 0.0066 | 0.0073 | 0.6712–0.6940 | 0.6804 | 40 |
| Pool-FT (Mistral) | `ged_macro` | ⇓ | 0.3098 ± 0.0070 | 0.0082 | 0.2955–0.3228 | 0.3191 | 96 |
| Pool-FT (Mistral) | `chain_validity_macro` | ⇑ | 0.9728 ± 0.0047 | 0.0050 | 0.9645–0.9802 | 0.9750 | 62 |
| Pool-FT (Mistral) | `hop_count_exact_match_rate` | ⇑ | 0.7182 ± 0.0143 | 0.0186 | 0.6954–0.7492 | 0.7033 | 12 |
| Pool-FT (Mistral) | `under_decomposition_rate` | ⇓ | 0.2469 ± 0.0128 | 0.0173 | 0.2254–0.2676 | 0.2583 | 77 |
| Pool-FT (Mistral) | `over_decomposition_rate` | ⇓ | 0.0349 ± 0.0072 | 0.0078 | 0.0233–0.0467 | 0.0383 | 70 |
| Qwen2.5-14B | `break_exact_match_rate` | ⇑ | 0.0281 ± 0.0064 | 0.0073 | 0.0170–0.0413 | 0.0350 | 85 |
| Qwen2.5-14B | `sari_macro` | ⇑ | 0.5761 ± 0.0062 | 0.0070 | 0.5667–0.5867 | 0.5802 | 72 |
| Qwen2.5-14B | `ged_macro` | ⇓ | 0.4705 ± 0.0072 | 0.0091 | 0.4592–0.4849 | 0.4596 | 4 |
| Qwen2.5-14B | `chain_validity_macro` | ⇑ | 0.9763 ± 0.0048 | 0.0056 | 0.9676–0.9849 | 0.9838 | 96 |
| Qwen2.5-14B | `hop_count_exact_match_rate` | ⇑ | 0.5857 ± 0.0154 | 0.0190 | 0.5633–0.6189 | 0.6167 | 96 |
| Qwen2.5-14B | `under_decomposition_rate` | ⇓ | 0.1147 ± 0.0097 | 0.0124 | 0.0950–0.1326 | 0.1017 | 9 |
| Qwen2.5-14B | `over_decomposition_rate` | ⇓ | 0.2996 ± 0.0150 | 0.0178 | 0.2724–0.3263 | 0.2817 | 12 |
| Qwen3.5-9B | `break_exact_match_rate` | ⇑ | 0.0419 ± 0.0068 | 0.0080 | 0.0304–0.0530 | 0.0550 | 98 |
| Qwen3.5-9B | `sari_macro` | ⇑ | 0.5990 ± 0.0051 | 0.0069 | 0.5907–0.6086 | 0.5984 | 50 |
| Qwen3.5-9B | `ged_macro` | ⇓ | 0.4863 ± 0.0085 | 0.0101 | 0.4713–0.5014 | 0.4838 | 40 |
| Qwen3.5-9B | `chain_validity_macro` | ⇑ | 0.8498 ± 0.0108 | 0.0140 | 0.8314–0.8658 | 0.8471 | 36 |
| Qwen3.5-9B | `hop_count_exact_match_rate` | ⇑ | 0.5679 ± 0.0167 | 0.0183 | 0.5391–0.6016 | 0.5700 | 56 |
| Qwen3.5-9B | `under_decomposition_rate` | ⇓ | 0.1066 ± 0.0095 | 0.0114 | 0.0917–0.1255 | 0.1350 | 100 |
| Qwen3.5-9B | `over_decomposition_rate` | ⇓ | 0.3255 ± 0.0140 | 0.0173 | 0.3017–0.3500 | 0.2950 | 2 |
| Mistral-7B | `break_exact_match_rate` | ⇑ | 0.0323 ± 0.0069 | 0.0074 | 0.0233–0.0430 | 0.0400 | 87 |
| Mistral-7B | `sari_macro` | ⇑ | 0.5795 ± 0.0057 | 0.0069 | 0.5688–0.5891 | 0.5772 | 34 |
| Mistral-7B | `ged_macro` | ⇓ | 0.4860 ± 0.0085 | 0.0095 | 0.4700–0.5006 | 0.4842 | 36 |
| Mistral-7B | `chain_validity_macro` | ⇑ | 0.9632 ± 0.0052 | 0.0067 | 0.9518–0.9726 | 0.9626 | 36 |
| Mistral-7B | `hop_count_exact_match_rate` | ⇑ | 0.5036 ± 0.0162 | 0.0192 | 0.4737–0.5296 | 0.5133 | 72 |
| Mistral-7B | `under_decomposition_rate` | ⇓ | 0.1606 ± 0.0131 | 0.0144 | 0.1374–0.1833 | 0.1683 | 69 |
| Mistral-7B | `over_decomposition_rate` | ⇓ | 0.3357 ± 0.0179 | 0.0189 | 0.3028–0.3659 | 0.3183 | 18 |

## Full population (all 2,417, hop-imbalanced; from `arms/<arm>/eval_metrics.json`)

Not comparable to the stratified draws or the pinned 600 (different hop mix: 52 / 31 / 17 %).

| arm | break EM | SARI | GED ⇓ | chain validity | hop-count EM | under ⇓ | over ⇓ |
|---|---|---|---|---|---|---|---|
| Qwen2.5-7B | 0.0567 | 0.5865 | 0.4340 | 0.9029 | 0.6276 | 0.2329 | 0.1394 |
| Pool-FT (Mistral) | 0.1270 | 0.6966 | 0.2775 | 0.9771 | 0.7948 | 0.1713 | 0.0339 |
| Qwen2.5-14B | 0.0451 | 0.5862 | 0.4324 | 0.9735 | 0.6376 | 0.0765 | 0.2859 |
| Qwen3.5-9B | 0.0579 | 0.6092 | 0.4420 | 0.8765 | 0.6243 | 0.0749 | 0.3008 |
| Mistral-7B | 0.0447 | 0.5873 | 0.4488 | 0.9683 | 0.5722 | 0.1105 | 0.3173 |

## Figures

`fig1_report_card_spread.{png,pdf}`, `fig2_per_hop_spread.{png,pdf}`, `fig3_rank_stability.{png,pdf}`.
Rank stability across draws: `resample_metrics.json` → `rank_stability_pct`.

## Caveats

- Gold is joined by question text (issue #57): 6 duplicate-text pairs in dev, 5 with differing gold;
  one in the pinned 600 (`3hop2__127483_125104_10557`). Applies to every number here and to every committed pinned run.
- Draw overlap (mean shared fraction): 2-hop 0.16, 3-hop 0.26, 4-hop 0.49. Overlap correlates the 50 draws; it does not shrink the band.
- The spread is question-sampling variation only (greedy decoding).
- Heavy outputs (predictions, per-item files, `resample_per_draw.csv`) stay in `runs/exp-030/`.
