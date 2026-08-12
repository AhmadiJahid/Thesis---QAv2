# Prior work — v1 results (Thesis---QA)

**Every number in this file is a v1 result**, measured in the predecessor repo (`Thesis---QA`, locally at `/cta/users/fyilmaz/Thesis---QA`) before this repo and its rules existed. **These are not v2 log entries.** No v2 claim may cite them as v2 measurements. Any baseline used for a v2 comparison must first be re-run in v2 — which baselines, and when, is Jahid's call, deferred until compute is settled (issue #2; see [ADR 0001](adr/0001-v1-to-v2-migration-scope-and-method.md)).

Primary source: `Thesis---QA/handoff/results_analysis/FINDINGS.md` (an analysis handoff over artifacts in that folder and the original `runs/` / `reports/`). Section references below are to that document. Figures are copied as FINDINGS states them, with two normalizations applied here: the dataset spelling is unified to "MuSiQue" (FINDINGS writes "MusiQue"), and where a value comes from a run file with more digits it is quoted to four decimal places, with the full value given alongside. Where FINDINGS defers to a run file, the run file is cited explicitly.

---

## 1. Router — hop-count classification, MetaQA (FINDINGS §2)

Few-shot model comparison on 300 questions, from `Thesis---QA/handoff/results_analysis/router/report.html`:

| Run | Model | Overall | 1-hop | 2-hop | 3-hop |
|-----|--------|---------|-------|-------|-------|
| 20260123_054856 | Qwen2.5-3B-Instruct | 90.33% | 94% | 95% | 82% |
| 20260123_060211 | Qwen2.5-3B-Instruct | 90.00% | 98% | 92% | 80% |
| 20260123_012144 | Qwen2.5-3B-Instruct | 89.33% | 100% | 77% | 91% |
| 20260123_014642 | Qwen2.5-7B-Instruct | 87.00% | 100% | 76% | 85% |
| 20260123_002620 | Qwen2.5-1.5B-Instruct | 81.33% | 100% | 70% | 74% |
| 20260123_042657 | Mistral-7B-Instruct-v0.3 | 76.67% | 97% | 34% | 99% |
| 20260123_011752 | Qwen2.5-0.5B-Instruct | 69.00% | 96% | 31% | 80% |
| 20260123_051315 | Phi-4-mini-instruct | 68.33% | 99% | 89% | 17% |

FINDINGS' reading: Qwen2.5-3B is the sweet spot in this table (high overall and strong 2-hop when the prompt/run is good); scaling to 7B does not clearly beat 3B; failure modes differ by family — Mistral and 0.5B collapse on 2-hop, Phi-4 collapses on 3-hop; 2-hop is consistently the hardest class. An earlier smaller sample (90 questions, Qwen2.5-1.5B, from project notes) scored 77.78% overall with the same qualitative story.

## 2. Similarity-based hop routing and entity masking (FINDINGS §3)

Source: `Thesis---QA/handoff/results_analysis/similarity_router/similarity_router_3way_20260311_064959.txt` (+ `.json`). Setup: 300 refined MetaQA questions, embedding retrieval, top-3 majority vote for hop.

| Mode | Query | Pool | Majority-vote accuracy | ≥1 correct hop in top-3 |
|------|-------|------|------------------------|-------------------------|
| A | raw | raw | 182/300 (60.7%) | 85.3% |
| B | raw | masked | 141/300 (47.0%) | 91.7% |
| C | masked | masked | 217/300 (72.3%) | 97.3% |

Per-hop, Mode C: 1-hop 79%, 2-hop 67%, 3-hop 71%. Average top-1 similarity is highest in C (0.906 vs ~0.81). FINDINGS' reading: matching on structure (both sides masked) is clearly best for hop prediction via neighbors; mismatched masking (Mode B) is worst for majority vote — don't mix a masked pool with raw queries.

## 3. MuSiQue decomposition eval, unguided (FINDINGS §4)

Source: `Thesis---QA/handoff/results_analysis/musique_decomposition_eval/eval_*_{metrics,notes,config}` files. Gold: MuSiQue clean dev, n = 600 (200 each of 2/3/4 hop). Unguided: hop count not forced in the prompt. Variant = retrieval text for few-shot selection. Provenance: these runs take their few-shots from the retrieval file named in `Thesis---QA/runs/decomposer_raw_unguided/*/config.json` (`retrieval_input: MusiQue/Data/sample_extracts/sim_dev_sample600_top20_rerankTop5.jsonl`). That file does not name its source split, but the artifact itself shows the runs are leak-free: across the 600 eval queries it holds 3,072 distinct retrieved `pool_id`s with **0 queries retrieving themselves and 0 retrieving any eval-set question** (`pool_id` ∩ `query_id` = ∅; verified in PR #5 review, 2026-08-12). The train-pool/dev-eval split is explicitly configured for the pool sweep in §4 (`Thesis---QA/configs/pool_sweep.json:16,26` — `dev_dir` points at `MusiQue/Data/dev_data`, `input_pool_jsonl` at a `_train_` file).

| Variant | Exact match | Step F1 | Ordered step acc | ROUGE-L F1 | Hop-count EM | Composite |
|---------|-------------|---------|------------------|------------|--------------|-----------|
| typed | 0.0583 | 0.2006 | 0.1794 | 0.5442 | 0.5217 | 0.3606 |
| uniform (`masked` preds path) | 0.0550 | 0.1896 | 0.1662 | 0.5431 | 0.5050 | 0.3554* |
| raw | 0.0367 | 0.1775 | 0.1581 | 0.5315 | 0.4850 | 0.1888 |

\* FINDINGS' table says "(see metrics file)" for the uniform composite; the run file `eval_uniform_unguided_metrics.json` in the same handoff folder records `composite_score` 0.3553568422318423, shown above to four decimal places to match the adjacent cells.

**Metric definitions** (from `Thesis---QA/scripts/musique_decompositions_evaluator.py`):

- **Composite** = 0.4·step_F1 + 0.3·ordered_step_accuracy + 0.2·reference_validity_micro + 0.1·max(0, 1 − mean_step_count_abs_error/3) (lines 411–416). Note the composite's reference term is the **micro** aggregate, not the macro reported in the table's source patterns below.
- Steps are normalized before any comparison: strip, lowercase, punctuation removed except `#`, whitespace collapsed (lines 48–62). **Exact match** requires the same number of steps with each normalized step equal in order (lines 194–199); **step F1** compares the normalized steps as sets (lines 170–171).

Shared patterns per FINDINGS: exact match is low (gold plans are hard to reproduce verbatim); ROUGE-L ~0.53–0.54, so lexical/partial overlap is much better than exact plan match; `[#k]` reference validity (macro) ≈ 0.997+, which FINDINGS reads as the reference wiring being usually syntactically fine; quality degrades with gold hop (2-hop exact ≈ 0.10–0.13, 4-hop exact ≈ 0.005–0.025); models often mis-estimate step count (raw hop EM 0.485; predicted hops include 5–15 for some items).

**Caveat on reference validity (from the evaluator and run files, not in FINDINGS).** The macro metric scores an item 1.0 when it emits **no** `[#k]` references at all, so the near-perfect macro reflects mostly reference-free predictions, not correct references. The micro values in the run metrics files are 0.75 (typed), 0.7778 (uniform, full value 0.7777777777777778), and 0.00 (raw) — exact counts 6/8 (typed), 7/9 (uniform), 0/4 (raw) total `[#k]` references across 600 predictions per run, from `runs/musique_decomposition_eval/eval_*_per_item.json` (per-item files live in v1's `runs/`, not in the committed handoff folder). Consequently the typed-vs-raw composite gap (0.3606 vs 0.1888) is dominated by the 0.2 × reference_validity_micro term flipping 0.75 → 0.00, while step F1 differs by only 0.023.

FINDINGS' takeaways for next work: prefer typed (or at least masked) few-shot retrieval over raw; unguided decomposition needs stronger length/hop control; optimize for step F1 / ordered accuracy / hop EM, not only exact match.

## 4. Pool sweep — pool size × retrieval × mask mode (FINDINGS §5)

Source: `Thesis---QA/handoff/results_analysis/pool_sweep_summary/all_runs.csv` + plots. 33 cells (sizes 1000/2000/4000/8000; balanced/imbalanced; biencoder_only vs biencoder_plus_ce; modes raw/typed/uniform); eval size typically 750 questions per cell. Same pool/eval provenance as §3: pool from MuSiQue train, eval from dev (`Thesis---QA/configs/pool_sweep.json:16,26`), which keeps the retrieval numbers leak-free.

Best cells by metric:

| Objective | Best run_key (approx) | Value |
|-----------|----------------------|-------|
| Composite | `size4000_imbalanced_trial0__biencoder_only__typed` | 0.3916 |
| Exact match | `size4000_imbalanced_trial0__biencoder_plus_ce__typed` | 0.0533 |
| Step F1 | `size8000_imbalanced_trial0__biencoder_plus_ce__typed` | 0.2113 |
| Hop-count EM | `size1000_imbalanced_trial0__biencoder_plus_ce__uniform` | 0.5333 |

Aggregate mean composite_score: mode typed 0.267, uniform 0.259, raw 0.243; biencoder_only 0.257, biencoder_plus_ce 0.256; size 4000 is highest among sizes at 0.311 (2000: 0.271, 1000: 0.232, 8000: 0.230).

FINDINGS' conclusions: typed masking is the best average retrieval mode in this sweep; larger pool is not always better (8000 drops vs 4000); cross-encoder rerank is not a free win on mean composite (it helps some metrics, especially hop-count EM / step F1 at larger pools — consult the delta plots per metric); exact match stays ~3–5% across the board — pool size alone does not solve gold-plan matching.

## 5. MetaQA KG compile/execute charts (FINDINGS §6)

`Thesis---QA/handoff/results_analysis/metaqa_kg_decomposition/` holds charts on whether decompositions execute on the MetaQA KG, split into compile fail / exec fail / success (taxonomy in §7 below). FINDINGS marks the chart counts as run-specific and says to read the PNGs rather than quote totals — so no numbers are carried here.

## 6. What v1 left open (FINDINGS §7–8)

Explicitly not concluded in v1:

- No single published best end-to-end KG-QA accuracy for the full Router→Decomposer→Jury pipeline (Jury still thin).
- Guided vs unguided MuSiQue comparison beyond the handoff artifacts should be checked against other `runs/` folders if present.
- Gemini / other LLM-as-judge evals under `runs/gemini_eval/` were not copied into the handoff (and see §8 below — that method is excluded in v2).

Open questions FINDINGS poses (questions, not decisions):

1. Should the router default to Qwen2.5-3B few-shot + masked similarity few-shots? (Flag for v2: Qwen2.5-3B exceeds v2's ~600M router cap and is excluded as-is; a cap change is a supervisor decision, per `CLAUDE.md` standing constraints.)
2. For MuSiQue, should production prompts be guided (force hop count), given low hop EM when unguided?
3. Is typed retrieval + pool size ~2000–4000 the default for decomposer few-shots?
4. Which metric is thesis-primary: exact match, step F1, hop EM, or MetaQA answer accuracy after compile/execute?

## 7. Pitfalls carried forward

**Re-masking corruption** (`Thesis---QA/docs/MASKING.md`; [ADR 0003](adr/0003-mask-queries-only-never-re-mask-the-few-shot-pool.md)). Dynamically applying the entity masker to the few-shot pool can corrupt pool items through KB entity overlap — v1's documented example: `"who stars in Baby Face"` → `"who [MOVIE] in [MOVIE]"`, because "Stars" is itself a KB movie title. The rule: use the pre-masked pool file, mask queries only, never re-mask the pool at runtime.

**compile_fail vs exec_fail taxonomy** (`Thesis---QA/docs/DECOMPOSITION_ERRORS.md`). When a decomposition is run against the MetaQA KG, failures split into two stages. **Compile fail** = the sub-question text could not be turned into a known KG op — a template-coverage or relation-inference problem (reasons: `missing_decomposition`, `unsupported_template`, `cannot_infer_relation`, `compile_error_other`). **Exec fail** = the op was valid but could not run on the KG — an entity-name or step-reference problem (reasons: `entity_not_in_kb`, `bad_reference_or_plan`, `exec_error_other`). The split tells you whether to fix templates/relations (compile) or entity names/`[#k]` plans (exec).

**Unseeded v1 subset sampling** ([ADR 0005](adr/0005-seed-before-sampling-v1-sampled-results-not-reproducible.md)). v1's router sampled its evaluation questions *before* seeding, so any v1 router result produced with `--sample_size` used an unreproducible draw. v2 seeds before sampling; v1 sampled-subset numbers are approximate and cannot be exactly re-run.

**Label integrity: `qwen3_1_7b`** (found during the PR #7 port review). v1's `components/router/qwen2_5_1_5b` config carried `model_name: "qwen3_1_7b"` while its `model_id` was `Qwen/Qwen2.5-1.5B-Instruct` (recorded in v2's `components/router/models/qwen2_5_1_5b/config.json` notes). Any v1 result labelled `qwen3_1_7b` may actually be Qwen2.5-1.5B — treat that label as suspect when reading v1 run outputs.

## 8. Excluded method: closed-model (Gemini) judging

v1 also contains evaluation runs that scored decompositions by asking Gemini's API to rate them — `Thesis---QA/scripts/evaluate_decompositions_gemini.py` and run outputs under `Thesis---QA/runs/gemini_eval/`. **This method was rejected by Jahid's supervisor as not scientifically defensible and is excluded from all scientific claims in v2** (standing constraint in [`CLAUDE.md`](../CLAUDE.md): no closed commercial model may score, rate, or judge decomposition quality). It is named here so it cannot re-enter silently: none of those runs are citable, in v1 form or re-run form, and no v2 evaluation may reintroduce closed-model judging.
