# Sampling variance of the decomposition report card (issue #56)

- Arms: `Qwen2.5-7B`, `Pool-FT (Mistral)`, `Qwen2.5-14B`, `Qwen3.5-9B`, `Mistral-7B` (population 2417 items, per hop {'2': 1252, '3': 760, '4': 405}; identical item set in every arm)
- Draws: K=50, 200 per hop (600 per draw), stratified, without replacement within a draw, seeds 42..91, the same draws for every arm
- Pinned 600 in the population: True (the diamond is the pinned 600 subset recomputed from this run)
- Population verified as a named eval set: True (`musique_dev_full`)
- **Caveat.** Each draw takes 200 questions per hop WITHOUT replacement from finite hop strata (N = {'2': 1252, '3': 760, '4': 405}), so the across-draw spread is smaller than the spread for independent samples from an unlimited population by the finite-population correction: the SD is scaled by sqrt(1 - n/N) = 0.92 at 2 hops, 0.86 at 3 hops, 0.71 at 4 hops. resample_metrics.json and resample_summary.csv report an FPC-corrected SD (sd_fpc_corrected) beside the raw SD; the figures plot the raw spread. Overlap between draws (measured mean shared fraction per hop: 2-hop 0.16, 3-hop 0.26, 4-hop 0.49) does not shrink the band: it only correlates the K draws, which limits how precisely K=50 draws estimate the band (SD, percentiles). The spread measures question-sampling variation only: decoding is greedy, so there is no decoding variation in it.

| arm | metric | dir | mean | SD | SD (FPC-corrected) | min | max | p2.5 | p97.5 | pinned subset (recomputed) | pinned pct |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-7B | break_exact_match_rate | ⇑ | 0.0398 | 0.0072 | 0.0085 | 0.0233 | 0.0633 | 0.0270 | 0.0530 | 0.0533 | 97 |
| Qwen2.5-7B | sari_macro | ⇑ | 0.5749 | 0.0063 | 0.0074 | 0.5604 | 0.5882 | 0.5610 | 0.5865 | 0.5717 | 32 |
| Qwen2.5-7B | ged_macro | ⇓ | 0.4711 | 0.0083 | 0.0097 | 0.4432 | 0.4857 | 0.4560 | 0.4841 | 0.4702 | 40 |
| Qwen2.5-7B | chain_validity_macro | ⇑ | 0.9166 | 0.0088 | 0.0111 | 0.8958 | 0.9426 | 0.9017 | 0.9367 | 0.9210 | 72 |
| Qwen2.5-7B | hop_count_exact_match_rate | ⇑ | 0.5445 | 0.0156 | 0.0197 | 0.5150 | 0.5817 | 0.5154 | 0.5757 | 0.5517 | 70 |
| Qwen2.5-7B | under_decomposition_rate | ⇓ | 0.2972 | 0.0142 | 0.0177 | 0.2633 | 0.3250 | 0.2720 | 0.3213 | 0.2933 | 32 |
| Qwen2.5-7B | over_decomposition_rate | ⇓ | 0.1583 | 0.0116 | 0.0139 | 0.1333 | 0.1850 | 0.1370 | 0.1768 | 0.1550 | 40 |
| Pool-FT (Mistral) | break_exact_match_rate | ⇑ | 0.0940 | 0.0107 | 0.0117 | 0.0717 | 0.1117 | 0.0757 | 0.1117 | 0.0967 | 60 |
| Pool-FT (Mistral) | sari_macro | ⇑ | 0.6818 | 0.0066 | 0.0073 | 0.6678 | 0.6975 | 0.6712 | 0.6940 | 0.6804 | 40 |
| Pool-FT (Mistral) | ged_macro | ⇓ | 0.3098 | 0.0070 | 0.0082 | 0.2847 | 0.3259 | 0.2955 | 0.3228 | 0.3191 | 96 |
| Pool-FT (Mistral) | chain_validity_macro | ⇑ | 0.9728 | 0.0047 | 0.0050 | 0.9644 | 0.9816 | 0.9645 | 0.9802 | 0.9750 | 62 |
| Pool-FT (Mistral) | hop_count_exact_match_rate | ⇑ | 0.7182 | 0.0143 | 0.0186 | 0.6883 | 0.7517 | 0.6954 | 0.7492 | 0.7033 | 12 |
| Pool-FT (Mistral) | under_decomposition_rate | ⇓ | 0.2469 | 0.0128 | 0.0173 | 0.2217 | 0.2750 | 0.2254 | 0.2676 | 0.2583 | 77 |
| Pool-FT (Mistral) | over_decomposition_rate | ⇓ | 0.0349 | 0.0072 | 0.0078 | 0.0217 | 0.0517 | 0.0233 | 0.0467 | 0.0383 | 70 |
| Qwen2.5-14B | break_exact_match_rate | ⇑ | 0.0281 | 0.0064 | 0.0073 | 0.0167 | 0.0467 | 0.0170 | 0.0413 | 0.0350 | 85 |
| Qwen2.5-14B | sari_macro | ⇑ | 0.5761 | 0.0062 | 0.0070 | 0.5663 | 0.5879 | 0.5667 | 0.5867 | 0.5802 | 72 |
| Qwen2.5-14B | ged_macro | ⇓ | 0.4705 | 0.0072 | 0.0091 | 0.4587 | 0.4886 | 0.4592 | 0.4849 | 0.4596 | 4 |
| Qwen2.5-14B | chain_validity_macro | ⇑ | 0.9763 | 0.0048 | 0.0056 | 0.9658 | 0.9875 | 0.9676 | 0.9849 | 0.9838 | 96 |
| Qwen2.5-14B | hop_count_exact_match_rate | ⇑ | 0.5857 | 0.0154 | 0.0190 | 0.5567 | 0.6200 | 0.5633 | 0.6189 | 0.6167 | 96 |
| Qwen2.5-14B | under_decomposition_rate | ⇓ | 0.1147 | 0.0097 | 0.0124 | 0.0933 | 0.1333 | 0.0950 | 0.1326 | 0.1017 | 9 |
| Qwen2.5-14B | over_decomposition_rate | ⇓ | 0.2996 | 0.0150 | 0.0178 | 0.2617 | 0.3350 | 0.2724 | 0.3263 | 0.2817 | 12 |
| Qwen3.5-9B | break_exact_match_rate | ⇑ | 0.0419 | 0.0068 | 0.0080 | 0.0300 | 0.0600 | 0.0304 | 0.0530 | 0.0550 | 98 |
| Qwen3.5-9B | sari_macro | ⇑ | 0.5990 | 0.0051 | 0.0069 | 0.5861 | 0.6091 | 0.5907 | 0.6086 | 0.5984 | 50 |
| Qwen3.5-9B | ged_macro | ⇓ | 0.4863 | 0.0085 | 0.0101 | 0.4700 | 0.5026 | 0.4713 | 0.5014 | 0.4838 | 40 |
| Qwen3.5-9B | chain_validity_macro | ⇑ | 0.8498 | 0.0108 | 0.0140 | 0.8244 | 0.8737 | 0.8314 | 0.8658 | 0.8471 | 36 |
| Qwen3.5-9B | hop_count_exact_match_rate | ⇑ | 0.5679 | 0.0167 | 0.0183 | 0.5333 | 0.6117 | 0.5391 | 0.6016 | 0.5700 | 56 |
| Qwen3.5-9B | under_decomposition_rate | ⇓ | 0.1066 | 0.0095 | 0.0114 | 0.0833 | 0.1300 | 0.0917 | 0.1255 | 0.1350 | 100 |
| Qwen3.5-9B | over_decomposition_rate | ⇓ | 0.3255 | 0.0140 | 0.0173 | 0.2900 | 0.3500 | 0.3017 | 0.3500 | 0.2950 | 2 |
| Mistral-7B | break_exact_match_rate | ⇑ | 0.0323 | 0.0069 | 0.0074 | 0.0167 | 0.0517 | 0.0233 | 0.0430 | 0.0400 | 87 |
| Mistral-7B | sari_macro | ⇑ | 0.5795 | 0.0057 | 0.0069 | 0.5685 | 0.5900 | 0.5688 | 0.5891 | 0.5772 | 34 |
| Mistral-7B | ged_macro | ⇓ | 0.4860 | 0.0085 | 0.0095 | 0.4650 | 0.5014 | 0.4700 | 0.5006 | 0.4842 | 36 |
| Mistral-7B | chain_validity_macro | ⇑ | 0.9632 | 0.0052 | 0.0067 | 0.9497 | 0.9729 | 0.9518 | 0.9726 | 0.9626 | 36 |
| Mistral-7B | hop_count_exact_match_rate | ⇑ | 0.5036 | 0.0162 | 0.0192 | 0.4700 | 0.5383 | 0.4737 | 0.5296 | 0.5133 | 72 |
| Mistral-7B | under_decomposition_rate | ⇓ | 0.1606 | 0.0131 | 0.0144 | 0.1333 | 0.1867 | 0.1374 | 0.1833 | 0.1683 | 69 |
| Mistral-7B | over_decomposition_rate | ⇓ | 0.3357 | 0.0179 | 0.0189 | 0.2967 | 0.3750 | 0.3028 | 0.3659 | 0.3183 | 18 |

Figures: `runs/exp-030/resample/fig1_report_card_spread.png`, `runs/exp-030/resample/fig1_report_card_spread.pdf`, `runs/exp-030/resample/fig2_per_hop_spread.png`, `runs/exp-030/resample/fig2_per_hop_spread.pdf`, `runs/exp-030/resample/fig3_rank_stability.png`, `runs/exp-030/resample/fig3_rank_stability.pdf`

- Code: commit `5226b8423d31b9cde2747ac8d81590309d5487ff` on `main` — **working tree dirty** (7 file(s)); this run did not come from committed code
- Config snapshot: `runs/exp-030/resample/resample_config.json`
- Metrics: `runs/exp-030/resample/resample_metrics.json`
