# exp-016 — twelve-arm `decomposition_report_card` coverage re-score (Refs #40)

## What this run is

PR #48 (merged `e3725c7`) made the six-term unblended `decomposition_report_card` the reported primary (`break_exact_match_rate`, `sari_macro`, `ged_macro` ⤓, `chain_validity_macro`, `hop_count_exact_match_rate`, the under/over-decomposition rate pair ⤓, kept split) and froze `composite_score` byte-identical as legacy (ADR 0029). Its Gate-1 review closed with one open nit: the nine arms' report-card values were measured once, for ADR 0029's junk-battery table (at commit `87e3765`), but that is prose in an ADR, not a log entry — per CLAUDE.md, no metric claim exists without a log entry to cite. exp-012's `router_guided` arm and its `unguided`/`oracle_guided` re-scores were also scored (at `c69ef5f`) before the report card existed. This run closes both gaps: re-score all 12 already-generated arms' committed `results.json` files at current `main` (`e3725c7`) so `decomposition_report_card` exists, and is logged, for every one of them.

**No new generation, no new comparison claims.** Every arm's predictions were already produced by an earlier logged run; this is a read-only re-score against gold, the same discipline exp-011 established for the first nine arms.

## CPU-only, no GPU, no lock

This run took **no GPU and no `runs/run.lock`**. It is a re-score of already-computed prediction files against gold — the same class of work exp-006/exp-007/exp-011/exp-014 already established needs no GPU (exp-014 is the precedent for this exact wording). `runs/run.lock` was held by `exp-013` (`generalisation_2_3hop`, Jahid's stated top-priority training run) throughout, with `exp-015`'s waiter armed behind it; neither was touched, neither was waited on, and this lane did not contend with either.

## Coverage

12 of 12 arms re-scored. None excluded — every arm's parent run's own `metrics.json.evaluation_set` was checked before scoring and all 12 show `pinned: true`, `ids_missing_count: 0`, `ids_unexpected_count: 0`, `rows_loaded_per_hop: {2: 200, 3: 200, 4: 200}`.

## The report card, per arm

All values macro means over n=600 (per-hop breakdown not repeated here; it is unchanged from each arm's original row and lives in `per_gold_hop_metrics` in each arm's `runs/exp-016/<arm>/eval_metrics.json`).

| arm | break_EM ⇑ | SARI ⇑ | GED ⤓ | ged_fallback_counts | chain_validity ⇑ | hop_EM ⇑ | under ⤓ | over ⤓ | decomp_mean (contingency, not suite) | composite_score (legacy, NOT headlined) |
|---|---|---|---|---|---|---|---|---|---|---|
| exp004_unguided | 0.0533 | 0.5850 | 0.4715 | {"node_cap": 1} | 0.9686 | 0.5083 | 0.2017 | 0.2900 | 0.5288 | 0.2098 (legacy) |
| exp004_oracle_guided | 0.0500 | 0.5890 | 0.4414 | {} | 0.9712 | 0.5900 | 0.0617 | 0.3483 | 0.5517 | 0.4197 (legacy) |
| exp004_unguided_capped | 0.0533 | 0.5849 | 0.4708 | {} | 0.9686 | 0.5083 | 0.2017 | 0.2900 | 0.5289 | 0.2128 (legacy) |
| exp005_unguided | 0.0517 | 0.5979 | 0.4910 | {} | 0.8605 | 0.5200 | 0.1150 | 0.3650 | 0.5085 | 0.4212 (legacy) |
| exp005_oracle_guided | 0.0600 | 0.6033 | 0.4241 | {"node_cap": 6} | 0.8697 | 0.8733 | 0.0333 | 0.0933 | 0.5965 | 0.4429 (legacy) |
| exp005_unguided_capped | 0.0517 | 0.5978 | 0.4909 | {} | 0.8605 | 0.5200 | 0.1150 | 0.3650 | 0.5085 | 0.4215 (legacy) |
| exp008_full_train | 0.1017 | 0.6823 | 0.3094 | {} | 0.9942 | 0.7100 | 0.2567 | 0.0333 | 0.6357 | 0.5236 (legacy) |
| exp009_mixed | 0.0533 | 0.5851 | 0.4713 | {"node_cap": 1} | 0.9686 | 0.5150 | 0.2000 | 0.2850 | 0.5301 | 0.2099 (legacy) |
| exp009_oracle_hop_matched | 0.0483 | 0.5879 | 0.4540 | {"node_cap": 1} | 0.9686 | 0.6000 | 0.1300 | 0.2700 | 0.5503 | 0.2633 (legacy) |
| exp012_router_guided | 0.0533 | 0.5851 | 0.4721 | {} | 0.9625 | 0.4917 | 0.2917 | 0.2167 | 0.5241 | 0.2150 (legacy) |
| exp012_unguided_rescored | 0.0533 | 0.5850 | 0.4715 | {"node_cap": 1} | 0.9686 | 0.5083 | 0.2017 | 0.2900 | 0.5288 | 0.2098 (legacy) |
| exp012_oracle_guided_rescored | 0.0500 | 0.5890 | 0.4414 | {} | 0.9712 | 0.5900 | 0.0617 | 0.3483 | 0.5517 | 0.4197 (legacy) |

**GED fallback counts, printed beside `ged_macro` per `docs/METRICS.md`**: the node-cap substituted bound is not tight (measured up to +0.0588 above the optimizer's value at the cap boundary), so an arm with more fallback rows carries a small systematic worse-`ged_macro` bias. `exp005_oracle_guided` has the largest count in this table (6/600, `node_cap`); `exp004_unguided`, `exp009_mixed`, `exp009_oracle_hop_matched` and `exp012_unguided_rescored` each have 1/600; every other arm has 0. All fallbacks are `node_cap` (deterministic, machine-independent) — none is `time_budget` or `no_optimizer_result`.

`composite_score` is recorded per schema (with `composite_score_status: "legacy"` stamped on every arm) but is **not headlined**: it carries the issue #40 `reference_validity_micro` reference-syntax defect (`_REF_RX` matches bracketed `[#k]`, MuSiQue's gold writes bare `#k`), so it has never measured reference validity on this data. `chain_validity_macro` (term 4 of the report card, above) is its intended replacement.

## exp-012's reused predictions

`exp012_unguided_rescored` and `exp012_oracle_guided_rescored` score the **exact same** `results.json` files as `exp004_unguided` and `exp004_oracle_guided` — exp-012's own log row states it reuses exp-004's already-committed `unguided`/`oracle_guided` predictions rather than generating new ones. Their report-card rows above are therefore, as expected, byte-identical to `exp004_unguided`/`exp004_oracle_guided`'s rows. This is coverage bookkeeping so exp-012's own per-item files (scored at `c69ef5f`, before the suite existed) carry the suite too — it is not a new measurement and not reported as one.

## Additive-only, proven empirically

For every one of the 12 arms, every pre-existing metric this run's prior committed file already carried (`step_f1_macro`, `ordered_step_accuracy_macro`, `hop_count_exact_match_rate`, `composite_score`, `exact_match_rate`, `rouge_l_f1_macro`, and — where the prior file already carried them — `break_exact_match_rate`, `sari_macro`, `ged_macro`, `chain_validity_macro`, `over_decomposition_rate`, `under_decomposition_rate`) reproduces **exactly** (byte-for-byte on the JSON values). Compared against `runs/exp-011/<arm>/eval_metrics.json` for the 9 exp-004/005/008/009 arms and against exp-012's own committed `runs/exp-012/eval/{router_guided,unguided_rescored,oracle_guided_rescored}/eval_metrics.json` for the 3 exp-012 arms. **No value failed to reproduce.** The only keys that changed from `None`/absent to a value are the ones this run's PR (#48) added: `decomposition_report_card`, `composite_score_status`, and (for the 3 exp-012 arms only, which predate PR #48) `decomp_mean_macro` — the 9 exp-011 arms already had `decomp_mean_macro` since exp-011 postdates PR #44. Full per-arm reproduction detail (keys checked, diffs found — none) is in `experiments/exp-016/metrics.json` under each arm's `reproduction_check`.

## Descriptive reading (no significance testing here)

This row's job is suite coverage across arms, not a new comparison — exp-011 and exp-014 already hold the paired significance tests for these arms on this eval set, and exp-012 holds them for `router_guided`. Any ordering noted here is **descriptive and untested**: `exp008_full_train` (LoRA fine-tuned) leads every term of the report card by a wide margin (break_EM 0.1017, SARI 0.6823, GED 0.3094, chain_validity 0.9942, hop_EM 0.7100), consistent with exp-011/exp-014's already-logged significant advantage. `chain_validity_macro` separates by base model rather than by prompting condition, as exp-011 already noted: all three exp-005 (Qwen3.5-9B) arms sit at 0.8605–0.8697, below every Mistral-7B arm (0.9625–0.9942). `exp012_router_guided` sits close to `exp004_unguided`/`exp012_unguided_rescored` on every term except `hop_count_exact_match` (0.4917, the lowest in the table) and the under/over split (0.2917/0.2167, the most under-decomposing arm in the table) — consistent with exp-012's already-logged finding that the prompted router (barely above chance) does not move decomposition quality.

## Provenance

Evaluator commit `e3725c778b5c9869ac3f0b53a9430a40b675dba8` (clean tree throughout), `configs/musique_eval.json` unedited, seed 42. Predictions sha256 for each arm and the prior file each was compared against are in `experiments/exp-016/metrics.json`. Original `experiments/exp-004/`, `exp-005/`, `exp-008/`, `exp-009/`, `exp-011/`, `exp-012/` directories were not modified — this row's artifacts land only under `experiments/exp-016/` (committed) and `runs/exp-016/` (gitignored: `eval_metrics.json`/`eval_per_item.json`/`eval_notes.md`/`eval_config.json` per arm).

