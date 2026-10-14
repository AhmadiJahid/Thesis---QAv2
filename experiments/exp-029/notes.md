# exp-029: per-hop and end-to-end significance (re-analysis, not a run)

ADR 0035 item 5. Refs #50, #16. No GPU, no generation, no pipeline code changed. `runs/run.lock` was never taken.

- **Script:** `experiments/exp-029/reanalysis.py`. It imports `scripts/musique_decompositions_evaluator.py` unchanged and calls its `_paired_bootstrap` (which uses `_statistics_for`), `_mcnemar`, `_paired_t_test_row` and `_significance_floor` on hop-stratified, id-aligned rows. It adds one piece of arithmetic, Holm-Bonferroni over the emitted p-values.
- **Settings:** `configs/musique_eval.json`: seed 42, 10,000 resamples, alpha 0.05, min_items 30. The bootstrap RNG is re-seeded for every pair × stratum, the same as one `--compare` call on that stratum's rows.
- **Inputs:** 17 per-item files plus 10 cross-check files. Paths and sha256 are in `config.json`.
- **Provenance checks:** 10 of 10 passed (asserted in the script, recorded in `metrics.json.provenance_cross_checks`):
  - The exp-016 re-scores are score-identical to the exp-011 files.
  - The exp-026 eval file matches its arm copy.
  - The exp-016 hop-4 slices match exp-013's baselines.
- **Reproduction check:** these overall numbers match the committed ones to every printed digit:
  - exp-014 Mistral SARI CI [−0.0116, +0.0036].
  - exp-009 hop-EM McNemar p 0.00062.
  - exp-013 comparisons A and B: hop-EM p 4.7e-5 and 5.9e-10, step-F1 CI [0.0538, 0.1083].
- **Holm families:** one per slide. Each family is the slide's metrics × pairs × strata, where the strata are overall plus 2/3/4 hops, except D2 (per hop only) and D3 (4 hops only).

| family | tests | uncorrected significant | Holm survivors |
|---|---|---|---|
| A7 | 16 | 3 | 3 |
| B1 | 72 | 27 | 16 |
| D2 | 48 | 37 | 35 |
| D3 | 8 | 7 | 7 |
| E1 | 48 | 25 | 14 |

- **Holm p basis:** exact McNemar p for binary metrics, paired-t p for continuous ones, because the bootstrap gives a CI and no p. The two bases disagree in one family cell: B1 Qwen ordered accuracy at 4 hops.
- **Reading:** `docs/analysis/2026-10-14-per-hop-and-end-to-end-significance.md`.
- **Id collision, flagged:** `runs/exp-029/` already holds artifacts written on 2026-10-14 by the issue #56 sampling-variance lane (`feature/56-sampling-variance`): `retrieval/`, `resample_smoke_pinned600/` and `smoke_*`. That lane had not logged an exp id. This entry writes nothing under `runs/exp-029/`. The #56 lane needs its own id.
