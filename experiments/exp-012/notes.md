# exp-012 — few-shot-prompted router + router_guided vs unguided/oracle_guided (Refs #27, #46 S1)

## What this run is

Issue #27 / issue #46 S1 / ADR 0024: a few-shot-prompted router (mistral_7b_instruct,
retrieved exemplars from the same artifact the decomposer reads) plus the `router_guided`
decomposer arm it feeds, compared against exp-004's already-committed `unguided` and
`oracle_guided` arms on the ADR 0007 pinned 600. Driven end to end by the committed
`runs/exp-012/run_all.sh`; all 7 stages ran, all rc=0 (verified independently off
`runs/exp-012/run_all.rc` and by grepping every per-stage `rc=` line in `run_all.log`,
per this row's brief — not just trusting the orchestrator's own "done, overall rc=0" line).
No pipeline code or config was edited to produce this run. **No keep/drop recommendation
is made here** — that decision belongs to Jahid and his supervisor (ADR 0028 item 2,
issue #46 section 4.2); this note states what was measured.

## Router accuracy

Overall accuracy **0.3833** (230/600) against a **3-class chance baseline of ~0.333**
(200 rows each of hop 2/3/4) — only ~0.05 above chance. Per hop (within-gold-class
recall, not precision, per ADR 0024 item 1a): hop=2 **0.875** (175/200), hop=3 **0.175**
(35/200), hop=4 **0.100** (20/200).

That hop=2 number is substantially inflated by defaulting: **394/600** router responses
were unparsed and scored at `parsing.default_hop` (=2), per gold hop 123/133/138
(2/3/4). Of hop=2's 175 "correct" rows, 123 are unparsed-defaulted — free credit from a
response the model never answered — leaving only 52 correct out of the 77 hop=2 rows
the model actually answered (52/77 ≈ 0.675 among parsed responses; simple arithmetic on
the two counts in `experiments/exp-012/metrics.json`, not a separately computed metric).
hop=3/hop=4's near-floor accuracy is consistent with `default_hop=2` giving them no free
credit at all. **Any router-accuracy claim from this run has to carry this split**
(ADR 0024's own requirement) — the headline 0.3833 is barely informative on its own.

## Headline: unguided vs router_guided (n=600, `runs/exp-012/compare_unguided_vs_router_guided/`)

**Every one of the 7 bootstrap metrics and 3 McNemar metrics is non-significant.**

| metric | unguided (a) | router_guided (b) | diff (a-b) | CI / p |
|---|---|---|---|---|
| rouge_l_f1 | 0.5429 | 0.5484 | -0.0055 | CI [-0.0144, +0.0032] |
| step_f1 | 0.2039 | 0.2035 | +0.0004 | CI [-0.0103, +0.0113] |
| ordered_step_accuracy | 0.1804 | 0.1853 | -0.0049 | CI [-0.0160, +0.0060] |
| sari | 0.5850 | 0.5851 | -0.0001 | CI [-0.0077, +0.0075] |
| ged (lower better) | 0.4715 | 0.4721 | -0.0006 | CI [-0.0163, +0.0152] |
| chain_validity | 0.9686 | 0.9625 | +0.0061 | CI [-0.0094, +0.0219] |
| composite_score (not headlined) | 0.2098 | 0.2150 | -0.0051 | CI [-0.2108, +0.0033] |
| exact_match | 0.0617 | 0.0583 | +0.0033 | McNemar p=0.7744 (b=7,c=5) |
| hop_count_exact_match | 0.5083 | 0.4917 | +0.0167 | McNemar p=0.500 (b=94,c=84) |
| break_exact_match | 0.0533 | 0.0533 | 0.0000 | McNemar p=1 (b=5,c=5) |

A router barely above chance produced a with-router arm that is indistinguishable from
no hop guidance at all, on every metric measured — including hop-count correctness
itself.

## Secondary: oracle_guided vs router_guided (n=600, `runs/exp-012/compare_oracle_guided_vs_router_guided/`)

`hop_count_exact_match` (0.5900 vs 0.4917, McNemar p=9.69e-06, b=117,c=58) and `ged`
(0.4414 vs 0.4721, CI [-0.0440,-0.0177], lower-is-better) are **significant, favouring
oracle**. `step_f1` (0.2061 vs 0.2035), `ordered_step_accuracy` (0.1852 vs 0.1853),
`sari` (0.5890 vs 0.5851) and `chain_validity` (0.9712 vs 0.9625) are all
**non-significant**. `exact_match` (p=1) and `break_exact_match` (p=0.6875) are also
non-significant.

**Even a PERFECT oracle's gain over an imperfect router is confined to hop-count
correctness and graph-edit distance — it does not move step content quality** (step_f1,
ordered_step_accuracy, sari, chain_validity all flat).

## composite_score — recorded, not headlined

Per issue #40 (the `reference_validity` regex/bare-vs-bracketed-`#k` defect in its
0.2-weight term, predating this run), `composite_score` is recorded in both compare
outputs but is **not** part of the headline, same convention as exp-010/exp-011. On the
oracle-vs-router pair its CI is **[+0.0001, +0.2103]** — it technically excludes zero
(so the bootstrap rule marks it `significant: true`), but the lower bound sits four
orders of magnitude from the interval's own width. That is a concrete instance of the
bimodal/asymmetric composite shape issue #40 documents, not a metric this row leans on.

## Drift-check argument for the unguided-vs-router_guided comparison

`unguided` is exp-004's already-committed arm — generated at commit `6fc4bba`
(confirmed in `experiments/exp-004/config.json`) — re-scored here only at the evaluator's
current commit; `router_guided` was generated fresh in this run. Per issue #27 comment
`5383430711` (the lead's drift check, cited per this row's brief) this is legitimate,
not a same-eval-set violation, because:

- `components/decomposer/run_decomposer.py`'s diff across that span (+283/-9) is
  entirely the `router_guided` addition (`hop_source` key, `resolve_hop_source`,
  `load_predicted_hops`); `resolve_hop_source` short-circuits on `if not guided:`, so an
  unguided run never touches the new code.
- `configs/decomposer_musique.json`'s diff (+14/-1) is additive only — the `unguided`
  condition block and `generation_overrides.max_new_tokens` (still 1024) are untouched.
- The `mistral_7b_instruct` prompt files have an **empty diff**; the unguided prompt
  still hashes to `e9fef279…`.
- The retrieval artifact is **byte-identical**: sha256 `e5c418a9b25f5ef290a78dd3a99642e4b0c5d7b6fbe6b5e2785a1424239ae70d`
  in both `experiments/exp-004/config.json` and this run's router/decomposer configs.
- The eval set loaded 600/600 with `ids_missing_count 0`, `ids_unexpected_count 0` on
  every stage of this run.

This is **an argument from targeted diffs, not proof by re-generation** — proving it
empirically would cost ~1130s of GPU not spent here (per the cited comment).

## Commit provenance — a discrepancy from the cited comment, resolved

Issue #27 comment `5383430711` names the router_guided generation commit as `c1369a9`.
Every on-box artifact of **this** run (router `metrics.json`, decomposer `config.json`,
both `eval_metrics.json`, both `compare_metrics.json`) instead stamps
`c69ef5f34db07483cf486cf33baca2a71faaf72d`. Checked directly: `run_all.sh` logs the
**launch** commit at start (`7314004`, the exp-012 pre-run log entry itself); each
stage's own JSON instead calls `git rev-parse HEAD` at write time, and the run's ~55 min
wall clock (03:38–04:33) overlapped several unrelated commits landing on `main` from a
concurrent lane (exp-013's pre-run entry, exp-014's S2 CPU-half work, ADR/analysis docs —
`7067e55, a6353d9, 6679fef, febe394, a6ab324, 222862f, 174920c, c69ef5f`, including the
`c1369a9` the comment names as an intermediate point). `git diff --stat 7314004 c69ef5f
-- components/ configs/ scripts/ src/` and `git diff --stat c1369a9 c69ef5f -- components/
configs/ scripts/ src/` are **both empty** — no pipeline code changed in that window, so
the drift-check's conclusion is unaffected either way. This row records `c69ef5f`, the
commit the artifacts actually stamp, not `c1369a9`.

## GED fallback counts

Per `docs/METRICS.md`, the `node_cap` substituted bound is **not tight** (measured up to
+0.0588 above the optimizer's own value at the cap boundary, ADR 0026's table).

- `router_guided`: `{}` — 0/600 fallback rows.
- `unguided_rescored`: `{"node_cap": 1}` — 1/600, at gold hop=4.
- `oracle_guided_rescored`: `{}` — 0/600 fallback rows.

At 1/600 the effect on `unguided_rescored`'s `ged_macro` is negligible, but it is a
small systematic upward (worse) bias relative to the other two arms, which have none.

## Provenance / reproducibility

- Run script: `runs/exp-012/run_all.sh` (committed, unedited during execution).
- Launch commit `7314004d6921e993c4c9bfa3d3827003242630c4`; artifacts stamp
  `c69ef5f34db07483cf486cf33baca2a71faaf72d` (see commit-provenance note above); both
  spans are pipeline-code-identical.
- Seed 42 throughout. GPU used for the router and decomposer stages; `runs/run.lock` was
  held by this run's own execution and released before exp-013 acquired it — this
  experiment-runner session read only committed/on-box artifacts, took no lock itself,
  and did not disturb exp-013's current lock (`exp-013 2026-08-23T04:34:27+0300` at the
  time this note was written).
- `runs/exp-012/{router,decomposer,eval,compare_*}` hold the full on-box trail
  (gitignored). `experiments/exp-012/{config.json,metrics.json,notes.md}` are the
  committed summary, created from those run outputs by this experiment-runner session
  (they did not exist before this commit).
- Verified independently off artifacts, not the orchestrator trail: `run_all.rc` = 0;
  every one of the 7 `rc=` lines in `run_all.log` reads `rc=0`; every eval's
  `total_evaluated` = 600 with `missing_gold_count` = 0; both compares' `num_aligned_items`
  = 600; the router's `total_questions` = 600 matches `evaluation_set.rows_loaded_total`
  with `ids_missing_count` 0 / `ids_unexpected_count` 0.
