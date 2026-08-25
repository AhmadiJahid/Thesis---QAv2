# exp-017 — S4 end-to-end MuSiQue answering (Issue #16 / issue #46 S4)

Four cells of `components/answerer/run_answerer.py` (reader `mistral_7b_instruct` /
`mistralai/Mistral-7B-Instruct-v0.3`, `context.policy=all_paragraphs`,
`max_new_tokens=64`, seed 42), all on the ADR 0007 pinned 600 (200/hop), so the four
numbers below are directly comparable to each other:

| cell | decomposition source | answer_em | answer_f1 | hop=2 EM/F1 | hop=3 EM/F1 | hop=4 EM/F1 |
|---|---|---|---|---|---|---|
| oracle | gold sub-questions (text only, ADR 0019) | 0.2800 | 0.3730 | 0.400 / 0.4909 | 0.305 / 0.4230 | 0.135 / 0.2053 |
| full_train | exp-008 fine-tuned decomposer | 0.2417 | 0.3283 | 0.340 / 0.4452 | 0.240 / 0.3417 | 0.145 / 0.1982 |
| unguided | exp-004 prompting baseline | 0.2133 | 0.2952 | 0.290 / 0.3764 | 0.200 / 0.2778 | 0.150 / 0.2313 |
| no_decomposition_control (lead-added) | question echoed as a single step | 0.1800 | 0.2723 | 0.215 / 0.3060 | 0.155 / 0.2426 | 0.170 / 0.2682 |

All four cells: 600/600 items scored, 200/200/200 per hop, `evaluation_set.pinned=true`,
`ids_missing_count=0`, `ids_unexpected_count=0` — the same pinned set for every row of
this table, so the comparison is on equal footing (CLAUDE.md evidence discipline).

## Reading

Answer EM (and F1, same order) is monotone across the four cells:

    no_decomposition_control (0.1800) < unguided (0.2133) < full_train (0.2417) < oracle (0.2800)

This is the first measurement in this repo that ties decomposition quality to final
answer correctness rather than to a decomposition-only metric: the ordering matches the
decomposition-quality ordering already established in exp-004/008/011/014/016 (oracle >
fine-tuned `full_train` > prompted `unguided`), and adding a sub-question step at all
(even the fine-tuned decomposer's imperfect one) beats asking the reader the raw
multi-hop question directly. No significance test was run in this log entry (no
`--compare`); the four cells are a descriptive coverage measurement, not a paired
significance claim.

## Caveat on cell 4

`no_decomposition_control` is a **lead-added** cell, not one of ADR 0019's three
conditions. It is a legitimate no-decomposition baseline (same full-paragraph context as
the other three cells — explicitly not a no-context/closed-book baseline), built by
echoing the original multi-hop question as the sole decomposition step, the same
construction as the decomposition-quality junk battery's `J2_question_echo` row
(`scripts/decomposition_junk_battery.py`). Its inclusion as a **standing** comparison arm
(as opposed to this one-off measurement) is still pending sign-off from Jahid and his
supervisor.

## Run mechanics

- Launched 2026-08-23 via `runs/exp-017/run_all.sh` (all 4 cells sequential, one
  `runs/run.lock` held for the whole sequence, released once by the script's `EXIT` trap).
- `runs/exp-017/run.log` ends: `all cells attempted, overall_rc=0` at
  `2026-08-23T11:33:17+03:00`.
- `runs/run.lock` confirmed absent after completion (released cleanly; no stale-lock
  cleanup needed).
- All four `answer_metrics.json` files were read directly and cross-checked against this
  note's table and against `experiments/exp-017/metrics.json` before this row was marked
  COMPLETE in `experiments/log.md`.
- Code: commit `24c11dc` on `main` (clean tree at generation time).
- Per-item prediction dumps and prompt logs are NOT copied here; they remain under
  `runs/exp-017/<cell>/<run_id>/` (gitignored `runs/`).

See `experiments/exp-017/config.json` for the exact CLI invocation per cell and
`experiments/exp-017/metrics.json` for the full per-cell metrics (including the
sub-question / step-failure counts summarized above).
