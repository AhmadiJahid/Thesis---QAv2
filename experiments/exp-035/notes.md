# exp-035: exp-030 re-scored with Han & Gardent's Break scorer path (CPU)

ADR 0041 (memo Part C, table C4 row 0, C3 item 6). This is a re-analysis: no inference and no GPU.
Measurement only; it asserts no direction.

## Provenance

- **Code:** `33ca4b8` on `feature/35-break-rescore`, clean tree (`git.dirty = false` in `metrics.json`).
  The scorer is `scripts/break_hg_rescore.py` and the config is `configs/break_hg_rescore_exp035.json`
  (copied here as `config.json`). Seed 42; nothing in the scorer is random.
- **Host:** `contabo` (Jahid's server). The run used a fresh clone at 33ca4b8 and a Python 3.12.3 venv
  built from the laptop `.venv-cpu`'s pip freeze: spaCy 3.8.7, en_core_web_sm 3.8.0, networkx 3.6.1,
  numpy 2.4.1. These are recorded under `environment` in `metrics.json`.
- **Run:** `--workers 6` (the config default). Start 2026-11-27T11:35:27+01:00, end 13:53:34, exit 0,
  wall clock 8282.6 s.
- **Inputs:** the five `runs/exp-030/<arm>/eval/eval_per_item.json` files, copied to the server by tar
  over ssh. Their md5 matches the laptop copies, and their sha256 is in `metrics.json` → `inputs`.
  Their stored overall EM / SARI / GED equal `experiments/exp-030/arms/<arm>/eval_metrics.json`
  exactly on all 5 arms (checked when this note was written).
- **Outputs:** `runs/exp-035/full/` (gitignored), and `metrics.json` md5
  `2734ea789f716c17172b8ee770e5fcf1` on both hosts. `run_notes_autogen.md` is the note the script wrote.
- **Two incomplete laptop attempts came first** (recorded in the log). (1) The main checkout's
  `runs/` was emptied mid-run by another session. (2) Claude Code stopped the run under system
  memory pressure before it wrote any output. Neither produced a number.
- No lock and no tmux: this was a CPU-only re-score (exp-032/034 precedent), and `docs/compute.md`
  describes the old server.

## Method

- **`hg_fork` (primary):** Han & Gardent's fork code path, read at
  hankelvin/complex_question_decomposition@9c9a669:
  `clean_lingering_operators_cot` → `format_qdmr` → `to_string` → strip trailing `?`+whitespace
  from the joined string. EM is lowercased. SARI is the official one. GED is the official one,
  with a spaCy-lemma node cost, the optimizer run to exhaustion, and items over 180 s dropped
  from the mean.
- **`hg_paper_literal`:** the paper's footnote 11 read literally. Every sub-question and the
  question are lowercased and `?`-stripped first. This is a sensitivity check, not a headline.
- **Repo port:** exp-030's stored evaluator values, shown for comparison.
- **Paired family:** fixed in the log row before any number was computed. It compares pool_ft
  with each of the 4 prompting arms × {EM exact McNemar, SARI paired t, GED paired t} ×
  {overall, hop 2, 3, 4} = 48 tests, with Holm over all 48 (ADR 0036). Only the `hg_fork`
  variant is tested.

**Parity check against the third-party fork: NOT run.** `experiments/exp-035/parity_check.py`
executes Han & Gardent's own code. The permission system denied running it, and the lead decided
it stays skipped. The port is checked only by the hand-computed unit tests in
`tests/test_break_hg_rescore.py`. It has not been executed side by side with the fork.

## Results (every number printed by `summarize.py metrics.json config.json`)

## Overall, per variant (EM / SARI / GED; GED dropped)

| arm | hg_fork EM | SARI | GED | dropped | paper-literal EM | SARI | GED | repo-port EM | SARI | GED |
|---|---|---|---|---|---|---|---|---|---|---|
| pool_ft | 0.1345 | 0.7035 | 0.2660 | 0 | 0.1398 | 0.7240 | 0.2629 | 0.1270 | 0.6966 | 0.2775 |
| qwen2_5_7b | 0.0670 | 0.5966 | 0.4253 | 0 | 0.0691 | 0.6178 | 0.4164 | 0.0567 | 0.5865 | 0.4340 |
| qwen2_5_14b | 0.0542 | 0.5984 | 0.4174 | 0 | 0.0621 | 0.6364 | 0.4084 | 0.0451 | 0.5862 | 0.4324 |
| qwen3_5_9b | 0.0707 | 0.6213 | 0.4315 | 1 | 0.0749 | 0.6646 | 0.4202 | 0.0579 | 0.6092 | 0.4420 |
| mistral_7b | 0.0554 | 0.5982 | 0.4382 | 0 | 0.0592 | 0.6296 | 0.4300 | 0.0447 | 0.5873 | 0.4488 |

## hg_fork per gold hop (EM / SARI / GED)

| arm | hop | n | EM | SARI | GED | GED dropped |
|---|---|---|---|---|---|---|
| pool_ft | 2 | 1252 | 0.2149 | 0.7396 | 0.1924 | 0 |
| pool_ft | 3 | 760 | 0.0645 | 0.6748 | 0.3261 | 0 |
| pool_ft | 4 | 405 | 0.0173 | 0.6457 | 0.3812 | 0 |
| qwen2_5_7b | 2 | 1252 | 0.1102 | 0.6302 | 0.3456 | 0 |
| qwen2_5_7b | 3 | 760 | 0.0250 | 0.5628 | 0.4877 | 0 |
| qwen2_5_7b | 4 | 405 | 0.0123 | 0.5561 | 0.5546 | 0 |
| qwen2_5_14b | 2 | 1252 | 0.0974 | 0.6280 | 0.3368 | 0 |
| qwen2_5_14b | 3 | 760 | 0.0118 | 0.5632 | 0.4840 | 0 |
| qwen2_5_14b | 4 | 405 | 0.0000 | 0.5730 | 0.5416 | 0 |
| qwen3_5_9b | 2 | 1252 | 0.1150 | 0.6466 | 0.3377 | 0 |
| qwen3_5_9b | 3 | 760 | 0.0263 | 0.6016 | 0.5073 | 0 |
| qwen3_5_9b | 4 | 405 | 0.0173 | 0.5799 | 0.5793 | 1 |
| mistral_7b | 2 | 1252 | 0.0903 | 0.6229 | 0.3559 | 0 |
| mistral_7b | 3 | 760 | 0.0224 | 0.5707 | 0.5076 | 0 |
| mistral_7b | 4 | 405 | 0.0099 | 0.5733 | 0.5626 | 0 |

## Stored (exp-030) under / over-decomposition per gold hop (descriptive)

| arm | under 2 | under 3 | under 4 | over 2 | over 3 | over 4 |
|---|---|---|---|---|---|---|
| pool_ft | 0.0136 | 0.2829 | 0.4494 | 0.0304 | 0.0368 | 0.0395 |
| qwen2_5_7b | 0.1078 | 0.3145 | 0.4667 | 0.1054 | 0.1658 | 0.1951 |
| qwen2_5_14b | 0.0136 | 0.0987 | 0.2296 | 0.2444 | 0.3434 | 0.3062 |
| qwen3_5_9b | 0.0152 | 0.1053 | 0.2025 | 0.2388 | 0.3737 | 0.3556 |
| mistral_7b | 0.0072 | 0.1882 | 0.2840 | 0.2875 | 0.3316 | 0.3827 |

## Paired family (hg_fork; 48 tests, 48 with p; Holm)

| A vs B | stratum | metric | A | B | diff | p | p_holm | sig (uncorr) | sig (Holm) |
|---|---|---|---|---|---|---|---|---|---|
| pool_ft vs qwen2_5_7b | overall | break_exact_match | 0.1345 | 0.0670 | 0.0674 | 8.31e-29 | 1.33e-27 | True | True |
| pool_ft vs qwen2_5_7b | overall | sari | 0.7035 | 0.5966 | 0.1069 | 8.55e-155 | 3.76e-153 | True | True |
| pool_ft vs qwen2_5_7b | overall | ged | 0.2660 | 0.4253 | -0.1593 | 7.5e-186 | 3.53e-184 | True | True |
| pool_ft vs qwen2_5_7b | 2 | break_exact_match | 0.2149 | 0.1102 | 0.1046 | 2.43e-24 | 3.16e-23 | True | True |
| pool_ft vs qwen2_5_7b | 2 | sari | 0.7396 | 0.6302 | 0.1094 | 6.92e-73 | 2.42e-71 | True | True |
| pool_ft vs qwen2_5_7b | 2 | ged | 0.1924 | 0.3456 | -0.1532 | 2.11e-89 | 8.24e-88 | True | True |
| pool_ft vs qwen2_5_7b | 3 | break_exact_match | 0.0645 | 0.0250 | 0.0395 | 2.83e-06 | 1.7e-05 | True | True |
| pool_ft vs qwen2_5_7b | 3 | sari | 0.6748 | 0.5628 | 0.1120 | 1.45e-58 | 4.35e-57 | True | True |
| pool_ft vs qwen2_5_7b | 3 | ged | 0.3261 | 0.4877 | -0.1617 | 9.99e-59 | 3.1e-57 | True | True |
| pool_ft vs qwen2_5_7b | 4 | break_exact_match | 0.0173 | 0.0123 | 0.0049 | 0.754 | 1 | False | False |
| pool_ft vs qwen2_5_7b | 4 | sari | 0.6457 | 0.5561 | 0.0895 | 1.7e-31 | 3.22e-30 | True | True |
| pool_ft vs qwen2_5_7b | 4 | ged | 0.3812 | 0.5546 | -0.1735 | 3.66e-43 | 9.16e-42 | True | True |
| pool_ft vs qwen2_5_14b | overall | break_exact_match | 0.1345 | 0.0542 | 0.0803 | 1.91e-36 | 4.2e-35 | True | True |
| pool_ft vs qwen2_5_14b | overall | sari | 0.7035 | 0.5984 | 0.1051 | 2.18e-148 | 9.17e-147 | True | True |
| pool_ft vs qwen2_5_14b | overall | ged | 0.2660 | 0.4174 | -0.1514 | 1.03e-168 | 4.73e-167 | True | True |
| pool_ft vs qwen2_5_14b | 2 | break_exact_match | 0.2149 | 0.0974 | 0.1174 | 5.61e-26 | 8.42e-25 | True | True |
| pool_ft vs qwen2_5_14b | 2 | sari | 0.7396 | 0.6280 | 0.1116 | 1.09e-70 | 3.7e-69 | True | True |
| pool_ft vs qwen2_5_14b | 2 | ged | 0.1924 | 0.3368 | -0.1444 | 2.54e-83 | 9.66e-82 | True | True |
| pool_ft vs qwen2_5_14b | 3 | break_exact_match | 0.0645 | 0.0118 | 0.0526 | 1.13e-10 | 9.01e-10 | True | True |
| pool_ft vs qwen2_5_14b | 3 | sari | 0.6748 | 0.5632 | 0.1116 | 1.09e-64 | 3.5e-63 | True | True |
| pool_ft vs qwen2_5_14b | 3 | ged | 0.3261 | 0.4840 | -0.1579 | 4.54e-58 | 1.32e-56 | True | True |
| pool_ft vs qwen2_5_14b | 4 | break_exact_match | 0.0173 | 0.0000 | 0.0173 | 0.0156 | 0.0625 | True | False |
| pool_ft vs qwen2_5_14b | 4 | sari | 0.6457 | 0.5730 | 0.0727 | 5.74e-23 | 6.89e-22 | True | True |
| pool_ft vs qwen2_5_14b | 4 | ged | 0.3812 | 0.5416 | -0.1605 | 5.2e-31 | 9.35e-30 | True | True |
| pool_ft vs qwen3_5_9b | overall | break_exact_match | 0.1345 | 0.0707 | 0.0637 | 1.63e-24 | 2.29e-23 | True | True |
| pool_ft vs qwen3_5_9b | overall | sari | 0.7035 | 0.6213 | 0.0822 | 1.16e-96 | 4.62e-95 | True | True |
| pool_ft vs qwen3_5_9b | overall | ged | 0.2660 | 0.4315 | -0.1655 | 8.94e-167 | 4.02e-165 | True | True |
| pool_ft vs qwen3_5_9b | 2 | break_exact_match | 0.2149 | 0.1150 | 0.0998 | 9.22e-21 | 9.22e-20 | True | True |
| pool_ft vs qwen3_5_9b | 2 | sari | 0.7396 | 0.6466 | 0.0930 | 1.6e-51 | 4.17e-50 | True | True |
| pool_ft vs qwen3_5_9b | 2 | ged | 0.1924 | 0.3377 | -0.1453 | 4e-77 | 1.48e-75 | True | True |
| pool_ft vs qwen3_5_9b | 3 | break_exact_match | 0.0645 | 0.0263 | 0.0382 | 4.87e-06 | 2.44e-05 | True | True |
| pool_ft vs qwen3_5_9b | 3 | sari | 0.6748 | 0.6016 | 0.0732 | 1.15e-31 | 2.29e-30 | True | True |
| pool_ft vs qwen3_5_9b | 3 | ged | 0.3261 | 0.5073 | -0.1813 | 6.19e-56 | 1.67e-54 | True | True |
| pool_ft vs qwen3_5_9b | 4 | break_exact_match | 0.0173 | 0.0173 | 0.0000 | 1 | 1 | False | False |
| pool_ft vs qwen3_5_9b | 4 | sari | 0.6457 | 0.5799 | 0.0658 | 1.04e-19 | 9.4e-19 | True | True |
| pool_ft vs qwen3_5_9b | 4 | ged | 0.3809 | 0.5793 | -0.1985 | 2.84e-38 | 6.53e-37 | True | True |
| pool_ft vs mistral_7b | overall | break_exact_match | 0.1345 | 0.0554 | 0.0790 | 2.17e-35 | 4.55e-34 | True | True |
| pool_ft vs mistral_7b | overall | sari | 0.7035 | 0.5982 | 0.1053 | 1.78e-148 | 7.66e-147 | True | True |
| pool_ft vs mistral_7b | overall | ged | 0.2660 | 0.4382 | -0.1722 | 1.93e-203 | 9.26e-202 | True | True |
| pool_ft vs mistral_7b | 2 | break_exact_match | 0.2149 | 0.0903 | 0.1246 | 1.14e-29 | 1.94e-28 | True | True |
| pool_ft vs mistral_7b | 2 | sari | 0.7396 | 0.6229 | 0.1167 | 5.65e-76 | 2.03e-74 | True | True |
| pool_ft vs mistral_7b | 2 | ged | 0.1924 | 0.3559 | -0.1635 | 2.39e-98 | 9.79e-97 | True | True |
| pool_ft vs mistral_7b | 3 | break_exact_match | 0.0645 | 0.0224 | 0.0421 | 4.43e-07 | 3.1e-06 | True | True |
| pool_ft vs mistral_7b | 3 | sari | 0.6748 | 0.5707 | 0.1041 | 6.61e-58 | 1.85e-56 | True | True |
| pool_ft vs mistral_7b | 3 | ged | 0.3261 | 0.5076 | -0.1815 | 4.47e-68 | 1.47e-66 | True | True |
| pool_ft vs mistral_7b | 4 | break_exact_match | 0.0173 | 0.0099 | 0.0074 | 0.508 | 1 | False | False |
| pool_ft vs mistral_7b | 4 | sari | 0.6457 | 0.5733 | 0.0723 | 8.15e-23 | 8.97e-22 | True | True |
| pool_ft vs mistral_7b | 4 | ged | 0.3812 | 0.5626 | -0.1815 | 3.26e-41 | 7.83e-40 | True | True |

significant uncorrected: 45/48; Holm survivors: 44/48
wall_seconds_total: 8282.6
environment: {"python": "3.12.3", "spacy": "3.8.7", "spacy_model": "en_core_web_sm", "spacy_model_version": "3.8.0", "networkx": "3.6.1", "numpy": "2.4.1"}

## Reading (descriptive, no direction asserted beyond the pre-registered family)

- **Variant effects:** the fork path moves every arm's EM, SARI and GED relative to the repo port
  (table 1). The literal reading moves SARI further. The size of these differences is the metric
  comparability gap the memo warned about, measured here.
- **The family:** 45/48 tests are significant uncorrected and 44/48 survive Holm. All four Holm
  non-survivors are 4-hop Break EM cells: vs qwen2_5_7b (p 0.754), qwen2_5_14b (p 0.0156,
  Holm 0.0625), qwen3_5_9b (p 1) and mistral_7b (p 0.508). At 4 hops every arm's EM is at most
  0.0173.
- **Under-decomposition by gold hop** comes from exp-030's stored values and is descriptive only;
  it is not tested here. See table 3.
- **One GED item was dropped** (qwen3_5_9b, 4-hop; >180 s). Its pair is excluded from that one
  GED t-test (n = 404). This is why the pool_ft value in that row reads 0.3809 against 0.3812
  elsewhere.

## Caveats carried

- **Gold is not Han & Gardent's gold.** They rewrote MuSiQue's `entity >> relation` sub-questions
  into natural language with their own templates; this run uses MuSiQue's stored gold. The scorer
  path matches theirs; the gold does not (ADR 0041 item 6).
- **spaCy 3.8.7 replaces their pinned 2.1.9,** so lemmas differ (for example `-PRON-`).
- **Issue #57:** gold is joined to predictions by question text. 6 dev question texts appear
  twice, 5 of them with different gold. Carried unchanged from exp-030.
- **Mistral-7B's exp-030 predictions were never re-parsed** with the PR #61 scaffold parser
  (95/2,417 rows contain a `Task:`/`Answer:` line). They are scored as stored.
- **A known defect in the run's `metrics.json` layout.** On the 32 paired-t rows of
  `paired_family.tests`, `system_a`/`system_b` hold the two arms' means, not arm names: the test
  row's numeric keys overwrote the names. No number is affected. `summarize.py` recovers the arm
  names from the family's fixed order and checks them against the 16 McNemar rows, which kept
  their names. The scorer now writes the names to `arm_a`/`arm_b` (fixed after the run, test
  added).
