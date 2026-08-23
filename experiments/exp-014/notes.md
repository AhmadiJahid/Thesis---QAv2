# exp-014 — S2 CPU half: Break-faithful significance + GPU-cell retrieval artifact (Refs #46)

## What this run is

Two deliverables from `docs/analysis/2026-08-23-s2-feasibility-prompting-vs-finetuned.md`
(commit `6679fef`), both CPU-only, no GPU touched, `runs/run.lock` never taken
(held by `exp-012`, the router evaluation, throughout this run's entire span —
verified before, during and after).

### Deliverable 1 — significance on the four Break-faithful columns

exp-011 re-scored nine pinned-600 arms under PR #44's additive columns
(`break_exact_match`, `sari`, `ged` — lower is better, `chain_validity`) but
ran no `--compare` ("a coverage run, not a comparison" — exp-011's own log
row). This run closes that gap by running `--compare` on exp-011's
already-committed `runs/exp-011/<arm>/eval_per_item.json` files — no new
generation, no re-scoring, no pipeline code or config touched.

**Headline pair: `exp004_unguided` vs `exp008_full_train`** (n=600, same
pinned eval set both sides). The fine-tuned arm (`full_train`) beats the
deployable prompting baseline (`unguided`, Mistral-7B) significantly on
**every** metric reported, both new and pre-existing:

| metric | unguided (a) | full_train (b) | a − b | 95% CI / p | significant |
|---|---|---|---|---|---|
| break_exact_match | 0.0533 | 0.1017 | −0.0483 | McNemar p=8.17e-05 (b=12,c=41) | yes, favours full_train |
| sari | 0.5850 | 0.6823 | −0.0973 | [−0.1114, −0.0832] | yes, favours full_train |
| ged (lower better) | 0.4715 | 0.3094 | +0.1621 | [+0.1422, +0.1821] | yes, favours full_train |
| chain_validity | 0.9686 | 0.9942 | −0.0256 | [−0.0387, −0.0133] | yes, favours full_train |
| step_f1 | 0.2039 | 0.3411 | −0.1372 | [−0.1601, −0.1150] | yes, favours full_train |
| ordered_step_accuracy | 0.1804 | 0.3237 | −0.1433 | [−0.1666, −0.1205] | yes, favours full_train |
| hop_count_exact_match | 0.5083 | 0.7100 | −0.2017 | McNemar p=1.12e-15 (b=57,c=178) | yes, favours full_train |
| composite_score (not headlined, issue #40) | 0.2098 | 0.5236 | −0.3137 | [−0.3307, −0.1055] | yes, favours full_train |

Paired t-tests (ADR 0017 item 4) agree in sign and significance on every row
above (e.g. GED t=+15.87, dof=599, p=1.29e-47). **GED fallback counts**
(node-cap, docs/METRICS.md's substituted-bound bias up to +0.0588): unguided
1/600, full_train 0/600 — negligible on both sides, so the fallback bias does
not meaningfully touch this comparison's direction.

**Seven further pairs**, all same-eval-set (pinned 600), all cheap CPU reads
of exp-011's existing per-item files:

- `exp004_unguided` vs `exp004_oracle_guided`: on the four new columns, only
  `ged` is significant (0.4715 vs 0.4414, CI [+0.0167,+0.0437], favours
  oracle); `break_exact_match`/`sari`/`chain_validity` all straddle zero — the
  same shape as exp-004's original finding (oracle moves hop-count-EM
  significantly, not step quality). GED fallback: unguided 1/600, oracle 0/600.
- `exp004_unguided` vs `exp004_unguided_capped`: near-identical on every new
  column (step-line cap fired on only 12/600 rows originally); `ged` bootstrap
  CI [+0.0002,+0.0014] is technically significant but the effect size is
  0.0007 — noise-scale, not a finding. GED fallback: 1/600 vs 0/600.
- `exp004_oracle_guided` vs `exp004_unguided_capped`: mirrors the first pair
  (oracle ahead on `ged`, CI [−0.0429,−0.0160]; others not significant). GED
  fallback: 0/600 vs 0/600.
- `exp005_unguided` vs `exp005_oracle_guided` (Qwen3.5-9B): **broader
  significance than Mistral** — `ged` significant (0.4910 vs 0.4241, CI
  [+0.0510,+0.0832], favours oracle) and `break_exact_match` significant by
  paired t-test (p=0.0252) but McNemar flags it `underpowered` (b=0,c=5,
  min-attainable-p=0.0625 > alpha) — reported both ways rather than picked.
  `sari`/`chain_validity` not significant. GED fallback: unguided 0/600,
  oracle_guided **6/600** (the largest fallback count of any arm in this
  study) — flagged per the brief: an arm with more fallback items carries a
  small systematic worse-`ged_macro` bias (up to +0.0588 per item,
  docs/METRICS.md), so oracle's true GED advantage may be marginally
  understated by its own fallback rows, not overstated — the direction of the
  finding is not put in question by this bias, only its exact magnitude.
- `exp005_unguided` vs `exp005_unguided_capped`: identical to third-decimal on
  every new column (cap fired on only 4/600 rows). GED fallback: 0/600 both.
- `exp005_oracle_guided` vs `exp005_unguided_capped`: mirrors the Qwen
  unguided-vs-oracle pair (ged significant favouring oracle, others not).
  GED fallback: 6/600 vs 0/600 — same caveat as above.
- `exp009_mixed` vs `exp009_oracle_hop_matched`: no new column reaches
  significance (`ged` CI [−0.0003,+0.0351] just crosses zero; McNemar/t-test on
  `hop_count_exact_match` remain the only significant axis, reproducing
  exp-009's original finding on the old metrics). GED fallback: 1/600 vs 1/600.

**Reading**: the four new Break-faithful columns do not change the
qualitative picture already on record — they add detail, not a reversal.
`ged` is consistently the most sensitive of the four new columns to the same
axes (oracle-vs-unguided hop guidance, fine-tuning vs prompting) that the
pre-existing metrics already flagged; `break_exact_match`/`sari`/
`chain_validity` mostly track `step_f1`/`ordered_step_accuracy` in direction
but reach significance less often at n=600 on the smaller within-triple
diffs. `composite_score` is recorded in every pair (present in each
`bootstrap` block) but not headlined for the reason above.

Full per-pair batteries (7 bootstrap rows + 3 McNemar rows + 9 paired
t-test rows each, per ADR 0009/0011): `runs/exp-014/compare/<a>_vs_<b>/
{compare_config.json,compare_metrics.json,compare_notes.md}` (gitignored,
on-box). Compact extract of the four new columns + composite +
step_f1/ordered_step_accuracy/rouge_l_f1/exact_match/hop_count_exact_match
for all 8 pairs: `experiments/exp-014/metrics.json` → `deliverable_1_compare.pairs`.

### Deliverable 2 — retrieval artifact for the S2 GPU cell

Built the one artifact the GPU half of S2 needs: a top-5 CE-reranked
retrieval file for the ADR 0007 pinned 600 over the adopted
`size2000_imbalanced` pool (ADR 0028 item 1), by exp-007's exact recipe.
**Not a logged experiment** (ADR 0027 point 5 — building an artifact scores
no system); provenance recorded here and in `experiments/exp-014/config.json`
/ `metrics.json` so the GPU run can cite it without re-deriving anything.

- **Pool** (not rebuilt): `runs/pool_sweep/pools/size2000_imbalanced_trial0_poolseed42/pool.jsonl`,
  2,000 rows, sha256 `cc3646625d759a1b58179b18c25f83d255db24f05acc08b666985af2e58d6646`
  — re-verified this session, matches ADR 0028's cited hash exactly.
- **Self-exclusion integrity re-verified this session**: pool ids (2,000) ∩
  pinned-600 ids (600) = **0**.
- **Step 1** (`check_question_similarity.py`, `--mode typed --top-k 20 --n 200
  --device cpu --seed 42 --no-hop-match`, embed model `intfloat/e5-small-v2`):
  600/600 rows written, hop split 200/200/200,
  `runs/exp-014/retrieval/s2_pinned600_top20.jsonl`, sha256
  `493360b4d4e43e0badbdce1162834979dfe80964a0dcc1f0ff71797df4b5f4c4`. The dev
  sample lacks `question_masked_typed` fields (exp-006's flag, carried
  forward), so queries were NER-masked on the fly (`Jean-Baptiste/roberta-large-ner-english`)
  per the script's documented fallback — same behaviour exp-010 relied on.
  Wall clock ~62.5s real (CPU-parallel, 26m37s user).
- **Step 2** (`rerank_similarity_results.py`, `--rerank-k 5 --device cpu
  --seed 42`, cross-encoder `cross-encoder/ms-marco-MiniLM-L-12-v2`): 600/600
  rows, hop split 200/200/200, top-5 per row,
  **`runs/exp-014/retrieval/s2_pinned600_imbalanced_rerank_top5.jsonl`**, sha256
  **`3f95c7dc18c14fc6bea3473f1be5c0d7fb21b7b2f2b2a1175dbfaebb5092fdb0`**. Wall
  clock ~51.5s real (22m56s user).
- Both stages ran `--device cpu` explicitly (deviating from
  `configs/similarity.json`'s own `cuda` default, same deliberate deviation
  exp-006/exp-007 made and justified: CPU is the only verified-reproducible
  path for this kind of precompute, ADR 0021 item 7, and the pool was not
  rebuilt so that caveat is not re-incurred here either way).
- `runs/run.lock` was `exp-012` before, during and after both stages —
  confirmed via `cat runs/run.lock` and `nvidia-smi --query-compute-apps`
  immediately before step 1 and immediately after step 2: lock unchanged,
  the only live GPU process throughout was exp-012's pid `1301582` (14.3 GiB).

**What the S2 GPU cell needs, spelled out**: run
`components/decomposer/run_decomposer.py --model mistral_7b_instruct --config
decomposer_musique.json --condition unguided --seed 42 --retrieval-input
runs/exp-014/retrieval/s2_pinned600_imbalanced_rerank_top5.jsonl
--output-root runs/<next-exp-id>` (exp-009's exact shape), evaluate with
`scripts/musique_decompositions_evaluator.py`, then `--compare` against
`runs/exp-011/exp008_full_train/eval_per_item.json` as the fine-tuned side —
exp-008 needs no re-run (`no_few_shot: true`, so pool choice cannot touch it,
per the feasibility note). Everything the feasibility note's §3 checklist
freezes (quantization `none`, `max_new_tokens 1024`, greedy/temp 0,
`prompt_unguided.md` sha `e9fef279e7c2…`, `few_shot.k 3`, `retrieval.mode
typed`, `retrieval.k 5`, self-exclusion on, seed 42) is unchanged by this
artifact build — only the pool behind the retrieval moves.

## What this run explicitly does NOT do

- Does not attempt ADR 0023 item 2.3's actual "best-pool vs fine-tuned"
  comparison — that needs the GPU decomposer cell (deliverable 2's
  consumer), which is out of scope for this CPU-only lane per the brief.
- Does not touch `runs/run.lock`, does not contend with `exp-012` (router,
  GPU) or the `exp-013-waiter` (generalisation fine-tuning, queued behind it).
- Does not modify `experiments/exp-004/005/008/009/010/011/` — read-only.
- Does not change `configs/musique_eval.json` or any pipeline code.

## Why no GPU / no lock, restated

Both deliverables are read-only comparisons over already-computed per-item
files (deliverable 1) and a bi-encoder + cross-encoder retrieval precompute
(deliverable 2) — the same class of CPU work exp-006/exp-007/exp-011 already
established does not need a GPU. `nvidia-smi --query-compute-apps` showed
exactly one live compute process throughout this run's entire span
(`exp-012`'s pid, 14.3 GiB), unchanged in memory footprint before and after
every step here.

## Provenance / reproducibility

- Evaluator: `scripts/musique_decompositions_evaluator.py` at `main` commit
  `7067e55` (clean tree at launch), `configs/musique_eval.json` unedited.
- Deliverable 1 inputs: `runs/exp-011/<arm>/eval_per_item.json` (all 9 arms),
  unchanged by this run, `config_weights_match_per_item_files: true` in every
  pair's `compare_metrics.json`.
- Deliverable 2 inputs: pool file above (sha256 verified); query files
  `/cta/users/fyilmaz/thesis-qav2-data/musique/dev_data/musique_ans_v1.0_dev_sample_{2,3,4}_hop_200.jsonl`
  (the ADR 0007 pinned 600).
- Seed 42 throughout (bootstrap resampling and the retrieval scripts' own
  seeded paths).
- `experiments/exp-014/{config.json,metrics.json,notes.md}` are the committed
  summary; heavy trails (`runs/exp-014/compare/**`, `runs/exp-014/retrieval/**`)
  stay on the box (gitignored).
- `runs/run.lock` was never acquired by this run — it took no lock and
  released none.
