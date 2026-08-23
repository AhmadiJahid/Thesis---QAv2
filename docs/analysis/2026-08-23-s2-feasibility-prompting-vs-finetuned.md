# S2 feasibility: is "dynamic prompting at its best pool vs the fine-tuned decomposer" answerable from committed artifacts?

- **Date**: 2026-08-23
- **Scope**: ADR [0023](../adr/0023-jahid-2026-08-22-direction-metric-pipeline-completion-generalisation.md) item 2.3, sequenced as S2 in issue #46. Read-only inspection of committed artifacts, committed configs and on-box evaluator outputs, begun at `main` `0ed7fd1` and written at `c1369a9` (ADR [0028](../adr/0028-jahid-2026-08-23-delegations-pool-choice-router-call-composite-authorized.md) landed mid-analysis and settles the pool choice — reflected in §3 and §5 below). No model run, no pipeline code touched, `runs/run.lock` not taken (S1/issue #27 owns the GPU).
- **What this is not**: not the comparison, not a claim that either method is better, not a research recommendation. Feasibility only.

## Verdict — (b), not available

**The ADR 0023 item 2.3 comparison as written is NOT answerable from committed artifacts.** No prompting arm exists on the ADR 0007 pinned 600 over *any* size-2000 pool of *any* construction. Every prompting arm on the pinned 600 (exp-004, exp-005, exp-009) ran over the **v1 9,156-row pool** (ADR 0014), and every arm over a size-2000 pool (exp-010's 18 cells) ran on a **different eval set** (pool_sweep's 750-query dev sample) at a **different generation configuration**. Per CLAUDE.md a comparison across differing eval sets is not a comparison, so the gap cannot be closed by re-reading numbers.

**A narrower comparison IS already available** and is worth stating separately, because it may be what Jahid actually wants: prompting **at exp-004's v1-pool configuration** vs the fine-tuned arm, on the pinned 600, is already measured and already paired-tested — `experiments/exp-008/metrics.json` → `arm_comparison` (n=600 aligned, `--eval-arm full_train`, baseline `unguided`, all four arms asserted at 200/200/200 per hop). What that comparison does *not* have is (i) the "best pool" of ADR 0023 item 2.1, and (ii) significance on the four Break-faithful columns, which exp-011 produced descriptively with no `--compare` (`experiments/exp-011/notes.md`: "This is a coverage run, not a comparison"). (ii) is cheap CPU and needs no new generation (see §4).

## 1. Eval-set identity — confirmed per experiment, from committed configs

| experiment | eval set | rows | evidence |
|---|---|---|---|
| exp-004 (Mistral, 3 prompting arms) | ADR 0007 pinned 600, 200/hop | 600, 200/200/200, `ids_missing_count 0`, `ids_unexpected_count 0` | `experiments/exp-004/config.json` → `decomposer_run_configs.<cond>.evaluation_set` (`pinned: true`, `pinned_id_count 600`) |
| exp-005 (Qwen3.5-9B, 3 prompting arms) | same pinned 600 | 600, 200/200/200 | `experiments/exp-005/config.json` → same key; `rows_source` is the same pinned-600 retrieval file |
| exp-008 (fine-tuned `full_train`) | same pinned 600 | 600, 200/200/200 | `experiments/exp-008/config.json` → `decomposer_run_config.evaluation_set` |
| exp-009 (mixed / oracle_hop_matched) | same pinned 600 | 600 each; hop split 200/200/200 verified on the final rerank files | `experiments/exp-007/config.json` (`eval_set`, `query_glob`), `experiments/exp-009/config.json`; `experiments/log.md` exp-007 row |
| **exp-010 (18 pool-sweep cells)** | **pool_sweep's own dev sample — a different set** | **750, 250/hop** | `experiments/exp-010/config.json` → `orchestrator_invocation.grid` ("dev_seed 42, dev_per_hop 250, 750 queries"); `configs/pool_sweep.json` (`dev_seed 42`, `dev_per_hop 250`); `experiments/log.md` exp-010 row states it verbatim: "distinct from the ADR 0007 pinned 600 used by exp-002..005/008/009" |

So the belief in the brief is confirmed exactly: exp-004/005/008/009 on the pinned 600; exp-010 elsewhere.

**Subsetting exp-010 down to the pinned 600 is not possible.** Measured (this analysis, read-only): the id set of `runs/pool_sweep/eval/size2000_imbalanced_trial0__biencoder_plus_ce__typed/eval_per_item.json` (750 ids) intersects the pinned-600 id set (`runs/exp-011/exp008_full_train/eval_per_item.json`, 600 ids) in **405 ids**. 195 of the pinned 600 were never decomposed in exp-010, so no aligned 600-item pairing exists to recover.

## 2. Pool identity — the crux: the prompting arms and exp-010 do not share a pool

| arm(s) | pool | rows | hop split (2/3/4) | candidate universe | provenance |
|---|---|---|---|---|---|
| exp-004, exp-005 (all 6 arms) | v1 pool via the v1 retrieval artifact `sim_dev_sample600_top20_rerankTop5.jsonl`, `input_sha256 e5c418a9b25f…` | 9,156 | 3,594 / 4,387 / 1,175 | `musique_ans_v1.0_train_0_questions_all.jsonl` (2-hop from train split 0 only, 3/4-hop from splits 0–3), pool file sha256 `212c2763…` | `experiments/exp-004/config.json` / `exp-005/config.json` → `retrieval`; ADR 0014 |
| exp-009 (both arms) | same v1 pool, re-retrieved (mixed / oracle-hop-matched) | 9,156 | 3,594 / 4,387 / 1,175 | same file, sha256 `212c2763…` re-verified at run time | `experiments/exp-007/config.json` (`pool_file`, `pool_file_sha256`, `pool_rows`) |
| exp-008 (fine-tuned) | **no pool** — `no_few_shot: true`; the retrieval file only supplies the query set | — | — | — | `experiments/exp-008/config.json` → `decomposer_run_config.no_few_shot`; `experiments/log.md` exp-008 row ("retrieval file supplies the query set even though `--no-few-shot` empties the example block") |
| exp-010 `size2000_imbalanced` | freshly constructed size-2000 pool, `pool_seed 42` | 2,000 | 1,461 / 413 / 126 | `musique_pool_enriched` (`…train_all_questions_all_expanded_enriched.jsonl`, 19,938 candidate rows) | `experiments/exp-006/config.json` (commands, knobs), `experiments/exp-006/notes.md` (pool table, k-means over 19,938 rows), `experiments/exp-010/config.json` → `pool_composition_recap_from_exp_006` |
| exp-010 `size2000_clustered` | k-means (k=2000, `nearest_to_centroid`) over the same candidates | 2,000 | 1,396 / 495 / 109 | same 19,938 | same, plus ADR 0021 |

**Concretely how they differ** — four axes at once, not one:

1. **Size**: 9,156 vs 2,000. ADR 0014 records this as a deliberate deviation from ADR 0006 §4's supervisor-pinned 2,000, "pending supervisor confirmation".
2. **Construction**: the v1 pool is not one of the sweep's three strategies at all — it is a fixed v1 file with its own hop mix (3,594/4,387/1,175, i.e. 3-hop-heavy), whereas `imbalanced` (1,461/413/126) and `clustered` (1,396/495/109) are 2-hop-heavy draws from a different candidate file.
3. **Candidate universe**: 9,156 rows from train split 0 (+ splits 0–3 for 3/4-hop) vs 19,938 rows of `musique_pool_enriched`. Different files, not a subset relation that anything committed asserts.
4. **Retrieval artifact**: exp-004/005 read one pinned pre-built file keyed to the pinned 600; exp-010's cells read per-cell artifacts keyed to the **750-query dev sample** (`runs/pool_sweep/rerank/size2000_<balance>_trial0/top5_ce.jsonl`). There is no size-2000 retrieval artifact for the pinned 600 anywhere on the box.

**Would the difference invalidate a prompting-vs-fine-tuned comparison?** Yes, if the exp-010 cells were used as the prompting side, and for two reasons that stack on top of the pool itself — the exp-010 cells also differ from exp-004/exp-008 in decoding and precision:

- **Token cap**: exp-010 ran through `configs/pool_sweep.json` → `configs.decomposer: decomposer.json`, which carries **no `generation_overrides`**, so the cap was the model default `max_new_tokens: 128` (`components/decomposer/models/mistral_7b_instruct/config.json`), and 50/48/73 of 750 rows hit it (`experiments/exp-010/notes.md`). exp-004 and exp-008 ran `decomposer_musique.json` with `generation_overrides.max_new_tokens: 1024` (exp-008: 0 rows at cap).
- **Quantization**: exp-010 `decomposer_quantization: 4bit` (`configs/pool_sweep.json`); exp-004 and exp-008 `quantization: "none"` (both configs' `decomposer_run_config`).

So exp-010's cells answer "which construction, holding everything else fixed within the sweep", on their own eval set. They are not a drop-in prompting side for exp-008. **No number is compared between them here, and none should be.**

## 3. The single minimal run that would make S2 answerable

One prompting cell, plus one CPU precompute step. **The fine-tuned side does not need re-running** — exp-008 used no few-shot pool at all, so the pool decision cannot touch it, and its predictions stay valid.

**Step 1 (CPU).** Build a top-5 retrieval artifact for the **pinned 600** over the adopted size-2000 pool, by exp-007's exact recipe: `MusiQue/scripts/check_question_similarity.py --pool-file runs/pool_sweep/pools/size2000_<balance>_trial0_poolseed42/pool.jsonl --query-glob musique/dev_data/musique_ans_v1.0_dev_sample_*_hop_200.jsonl --n 200 --mode typed --top-k 20 --seed 42` → `rerank_similarity_results.py --rerank-k 5`. **Which pool**: ADR 0028 item 1 settles this — the lead's choice is `size2000_imbalanced` (balanced excluded, clustered the fallback), so `<balance>` is `imbalanced`. The pools already exist on the box, so nothing is rebuilt and ADR 0021 item 7's GPU-embedding-determinism caveat is not re-incurred: `runs/pool_sweep/pools/size2000_imbalanced_trial0_poolseed42/pool.jsonl` (2,000 rows, sha256 `cc3646625d759a1b…`) and `…clustered…/pool.jsonl` (2,000 rows, sha256 `64f91f4768e3f597…`), both verified present and read this session. Rows carry `question_masked_typed`/`question_masked_uniform`, so `typed` mode reads stored fields.

  Two preconditions measured here, both clean: **pool/eval overlap is 0** for all three size-2000 pools against the pinned 600 ids (0/2000 in each of imbalanced, clustered, balanced), and the runner's `few_shot_self_exclusion` is on regardless (`experiments/exp-004/config.json`).

  Cost: unmeasured for this exact shape. The nearest measured figure is exp-006's ~13.7 min for **three** strategies × 750 queries *including* pool construction (`experiments/exp-006/notes.md`); one strategy × 600 queries with the pool already built is a fraction of that. CPU only.

**Step 2 (GPU, one cell).** exp-009's shape exactly — that run is the existence proof that `--retrieval-input` composes with the pinned-600 assertion:

```
components/decomposer/run_decomposer.py --model mistral_7b_instruct \
  --config decomposer_musique.json --condition unguided --seed 42 \
  --retrieval-input <the step-1 artifact> --output-root runs/<exp-id>
```

then `scripts/musique_decompositions_evaluator.py` on the result, then `--compare` against exp-008's per-item file (or `compare_decomposer_arms.py --eval-arm full_train` with the new arm swapped in as the prompting baseline).

  **Cost**: one cell. By exp-010's unit, **~1500–1800 s** (`experiments/exp-010/notes.md`, 750 rows/cell at 4-bit, cap 128). exp-004's own measured mean latency at *this* configuration is 1.879 s/query over 600 rows (`experiments/exp-004/notes.md`), i.e. ≈1130 s of generation; exp-008's comparable 600-row run took ~832 s. Call it well under an hour of GPU including eval, versus exp-010's 8h10m for 18 cells. It queues behind S1's lock.

**What it must NOT change, or it stops being comparable to exp-008** — every one of these is held at exp-004/exp-008's committed value, and only the pool behind the retrieval artifact moves:

- eval set: the pinned 600 ids, 200/hop (`decomposer_musique.json`'s `eval_rows_per_hop: 200` asserts it; do not pass any unpinned override)
- gold + evaluator: `musique_ans_v1.0_dev_clean.jsonl` via `gold_key musique_dev_gold_clean`, `configs/musique_eval.json` byte-identical (including `composite_score_weights` and the `break_metrics.ged` block)
- model: `mistral_7b_instruct` / `mistralai/Mistral-7B-Instruct-v0.3`, **`quantization: "none"`** (not the sweep's 4-bit)
- decoding: `generation_overrides.max_new_tokens: 1024`, `do_sample false`, `temperature 0.0`, `top_p 1.0`
- prompt + condition: `unguided`, `prompt_unguided.md`, `prompt_sha256 e9fef279e7c2…`, `prompt_style plain`
- few-shot / retrieval knobs: `few_shot.k 3`, `retrieval.mode typed`, `retrieval.k 5`, self-exclusion on
- seed 42 throughout, generation and evaluator

## 4. The Break-faithful columns are available for any arm on the pinned 600

exp-011 re-scored the nine older arms under PR #44's additive columns. Confirmed by reading the files this session: `runs/exp-011/<arm>/eval_per_item.json` for all nine arms (`exp004_unguided`, `exp004_oracle_guided`, `exp004_unguided_capped`, `exp005_unguided`, `exp005_oracle_guided`, `exp005_unguided_capped`, `exp008_full_train`, `exp009_mixed`, `exp009_oracle_hop_matched`; arm→predictions map in `experiments/exp-011/config.json`). Each carries 600 items with per-item `break_exact_match`, `sari`, `ged`, `ged_fallback`, `ged_fallback_seconds`, `chain_validity` alongside the pre-existing `step_f1` / `ordered_step_accuracy` / `rouge_l_f1` / `exact_match` / `hop_count_exact_match`. exp-010's 18 cells carry the same columns natively (`runs/pool_sweep/eval/<cell>/eval_per_item.json`, 750 items, columns verified present on the headline cell) — but on the other eval set.

**All nine pinned-600 per-item files are id-aligned in identical order** (verified: same id set and same order as `exp008_full_train` for all nine). So a paired `--compare` on the four new columns between `exp004_unguided` (or any other arm) and `exp008_full_train` is mechanically ready today, CPU-only, no GPU, no new generation. That is a **logged run for an experiment-runner**, not analysis output — it needs its own `experiments/log.md` row before any significance claim is made from it. Note `docs/METRICS.md` (`ged_fallback_counts` row): `ged_macro` is lower-is-better, and the substituted node-cap bound is not tight -- measured at up to +0.0588 above the optimizer's value (ADR 0026's table, `scripts/ged_cost_benchmark.py`) -- so an arm with more capped items carries a small systematic worse-`ged_macro` bias (fallback counts are 0–6/600 across these arms per `experiments/exp-011/notes.md`, i.e. small here).

## 5. One domain fact settled mid-analysis, one still missing

1. **Which pool: settled, and not by the data.** exp-010 found `imbalanced` and `clustered` **exactly tied** at `hop_count_exact_match` 0.5133, both significantly above `balanced` 0.4667, with step-level metrics flat across all three (`experiments/log.md` exp-010 row; `experiments/exp-010/notes.md`) — so the numbers do not pick a winner between the top two. ADR 0028 item 1 (Jahid delegated the choice; the lead picked) resolves it as **`size2000_imbalanced`**, on stated grounds of status quo and describability, with balanced excluded and clustered the fallback. It is a configuration choice on measured indifference, not a finding about pools, and it does not change the cost of the run in §3.
2. **What "best configuration" means beyond the pool is unstated.** ADR 0023 item 2.3 names only the pool. Whether "best configuration" also ranges over the retriever variant (`biencoder_only` vs `biencoder_plus_ce`), the retrieval mode (`raw`/`typed`/`uniform` — ADR 0018 leaves the masking default reopened), the condition (`unguided`, the deployable one, vs `oracle_guided`, which needs a gold hop count at inference), or the base model (Mistral-7B vs Qwen3.5-9B, exp-004 vs exp-005) is **not stated anywhere committed**, and is not derived here. ADR 0028 delegates the *pool* choice and the router call, and authorizes the composite work; it does not widen or narrow "best configuration" beyond the pool. If it ranges over more than the pool, the minimal run above multiplies by the number of cells chosen. That scoping is Jahid's call.

## What was run for this note

Read-only, CPU, no lock: the committed `config.json`/`notes.md`/`metrics.json` under `experiments/exp-001,004,005,006,007,008,009,010,011`; `experiments/log.md` rows for exp-004/005/007/008/010/011; `configs/pool_sweep.json`, `configs/decomposer.json`, `configs/decomposer_musique.json`, `configs/paths.json`, `components/decomposer/models/mistral_7b_instruct/config.json`; ADRs 0014, 0021, 0023, 0028. Three things were *measured* rather than read: the 405/600 id intersection between exp-010's headline cell and the pinned 600; the 0/2000 pool-vs-eval id overlap for all three size-2000 pools; and the id-set/order alignment plus new-column presence across all nine `runs/exp-011/<arm>/eval_per_item.json` files (600 items each). No other number in this note is new — each is cited to the artifact that produced it.
