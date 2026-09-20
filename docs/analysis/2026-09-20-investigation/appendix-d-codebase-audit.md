<!-- Lane ledger produced by a read-only investigator agent on 2026-09-20 for the lead's state-of-the-work investigation (`docs/analysis/2026-09-20-state-of-the-work-investigation.md`). Every claim cites a file, commit, issue or log row. Numbers are quoted from committed artifacts, never re-computed. -->

# Codebase health audit — Thesis---QAv2

Read-only. Repo at `main` `f1bbdad`, 2026-09-20. Facts below were verified by running or reading;
inferences are marked `[inference]`.

---

## 1. Architecture map

**`src/` — shared library (2,835 LOC, 13 modules), no config of its own; every module is imported, none dead.**

| file | one line | config |
|---|---|---|
| `src/run_config.py` | config loader; `require()` raises, no silent defaults; repo-bound path resolution (`resolve_config_path`, :30-66) | reads `configs/paths.json` |
| `src/run_artifacts.py` | writes the three per-run artifacts (config.json / metrics.json / notes.md) + git provenance | — |
| `src/seeding.py` | `set_global_seed` for python/numpy/torch(+cuda) | — |
| `src/model_size.py` | parameter-ceiling load + hard assert at model load | `configs/model_limits.json` |
| `src/step_lines.py` | the single definition of "a step line" + reference regexes | — |
| `src/entity_masking.py` | Aho-Corasick masking of movie/person entities | `configs/masking.json`, `ner_masking_rules.json` |
| `src/pool_embeddings.py` | embedding similarity for few-shot pool selection | `configs/similarity.json` |
| `src/hop_matching.py` | hop-matched candidate filter, binds before top-k | `configs/similarity.json` |
| `src/finetune_data.py` | LoRA training-data + prompt construction | `configs/finetune_decomposer.json` |
| `src/answer_metrics.py` | MuSiQue official answer EM / F1 | — |
| `src/step_failures.py` | step-level failure taxonomy for executed decompositions | — |
| `src/anchor_set.py` | reads the frozen 45-id anchor list (ADR 0033) | `configs/anchor_set_45.json` |

**Data flow (MuSiQue), verified from configs + runner code:**

```
raw MuSiQue jsonl (data_root, outside the tree)
  → MusiQue/scripts/split_musique_train_stratified.py  (stratified chunks)
  → clean_musique_train_chunks.py / extract_musique_clean_questions.py
  → ner_mask_musique_question_chunks.py                (Jean-Baptiste/roberta-large-ner-english)
  → combine_train_split_masked_questions.py  →  enrich_pool_decompositions.py   [POOL]
  → sample_pool.py (seeded; k-means or imbalanced strategies)                   [POOL ARM]
  → check_question_similarity.py (e5-small-v2 bi-encoder top-20, hop_match filter)
  → rerank_similarity_results.py (ms-marco-MiniLM-L-12-v2 → top-5)              [RETRIEVAL ARTIFACT *.jsonl]
  → components/decomposer/run_decomposer.py --retrieval-input <artifact>
       few-shot block assembled (k from retrieval file; ordering from few_shot.ordering)
       → generation → runs/<exp>/decomposer/<run-id>/results.json + prompts_log/
  → scripts/musique_decompositions_evaluator.py  → eval_metrics.json / eval_per_item.json
  → experiments/<exp-id>/{config.json,metrics.json,notes.md}  (committed)
```

**Router.** `components/router/run_router.py` (1,428 LOC) predicts a hop count and writes
`predictions.jsonl` keyed by `query_id`. It is wired end-to-end by **file handoff, not by import**:
`run_decomposer.py --condition router_guided --hop-predictions <file>` (`configs/decomposer_musique.json`
`conditions.router_guided.hop_source = "predictions"`), and `scripts/smoke_test.py:505-528, 658-668`
exercises router → decomposer on fixtures. Real use: exp-012 only. Per ADR 0030 the router is off
the critical path; issue #47 says its predictions are ~66% fallback.

**Answerer.** `components/answerer/run_answerer.py` (1,376 LOC) reads a decomposer `results.json` via
`--predictions` and scores answer EM/F1. Wired and exercised for real in exp-017 (4 cells, 600 items each).
Smoke-covered at `scripts/smoke_test.py:685-717`.

Both are genuinely wired; the pipeline is a chain of committed JSON artifacts, not one process.
The **jury** named in `README.md` does not exist in code (confirmed: no file, no reference).

---

## 2. Test suite

**616 tests collected. Actual run:**

```
611 passed, 5 skipped, 3 warnings in 252.60s (0:04:12)
```

Command: `/usr/bin/python3 -m pytest -q -p no:cacheprovider tests/` — exit 0, wall 253.25 s.

**The repo venv cannot run the suite.** `.venv/bin/python -m pytest` → `No module named pytest`;
`pytest` is **not in `requirements.txt`** and not installed in `.venv`. The suite is actually run with
system `/usr/bin/python3` (3.10.12) + `~/.local/bin/pytest` 8.3.5 (confirmed by
`tests/__pycache__/*.cpython-310-pytest-8.3.5.pyc`). `pytest-timeout` is not installed either, so the
brief's `--timeout=600` cannot be passed.

**All 5 skips are artifacts of that split environment, not of missing hardware:**
- `tests/test_finetune_lora.py:824,837,844` — "peft/trl not installed". `.venv` **has** peft 0.20.0 + trl 1.10.0; system python3.10 does not. These are the LoRA fine-tuning guards — the arm CLAUDE.md names as the comparison against prompting.
- `tests/test_pool_clustering.py:455,493` — "intfloat/e5-small-v2 is not in the local cache (offline probe failed)". The model **is** cached (128 MB, §7); `sentence_transformers` is absent from system python3.10, present in `.venv`.

**Warning worth naming:** `tests/test_decomposer_conditions.py:125 def test_musique_config() -> dict`
returns a dict — `PytestReturnNotNoneWarning`, "will be an error in a future version of pytest".
14 such `return`s in that file. `tests/test_suite_collectability.py` guards the `def test_*(arg)`
shape (ADR 0016) but not this one.

**Areas with no test:** `src/run_artifacts.py` (git provenance, the #54 defect site — no test at all),
`src/seeding.py`, `src/run_config.py` (exercised transitively only), `scripts/analyze_runs.py`,
`scripts/kg.py`, `scripts/evaluate_decompositions.py`, `scripts/compare_answer_accuracy.py`,
`scripts/check_pool_coverage.py`, `scripts/refine_and_sample_pool.py`, `scripts/plot_pool_sweep.py`,
`scripts/compare_decomposer_arms.py`, `scripts/run_qwen_smoke.py`, the two `scripts/test_similarity_*.py`
probes, and `scripts/smoke_test.py` itself.

---

## 3. Config inventory

**`configs/model_limits.json`** (2,908 B, last touched 2026-09-20):
- `router_max_params` 8e9
- `default_max_params` 8e9 (inherited by retrieval, ner, reranker per `src/model_size.py:28-45`)
- `decomposer_max_params` **1.5e10** — the one key above the standing 8B, marked pending supervisor confirmation
- `answerer_max_params` 8e9

Matches CLAUDE.md exactly. `load_limits()` (`src/model_size.py:48-61`) requires every distinct key at
startup, dry runs included.

**`configs/decomposer_musique.json`** (18,597 B):
- `retrieval.k` = **5**; `retrieval.mode` "typed"; `retrieval.input_key` `musique_few_shot_top5_pinned600`; `require_input` true
- **`few_shot.ordering` = "as_is"` — the ordering key IS present** (`:81-83`), with `reversed` supported and `complexity_asc` reserved-and-refused
- **no `quantization` key at this level** — quantization lives in each model folder's `config.json` `loader.quantization` and is overridden per-run by `--quantization` (`run_decomposer.py:1917, 2247`)
- there is **no `few_shot.k`** here; `k=3` comes from the model folder (this is the #53 defect, §4)
- `generation_overrides.max_new_tokens` 1024; `seed` 42; `eval_rows_per_hop` 200; `logging.anchor_ids_file` `configs/anchor_set_45.json`

**`configs/finetune_decomposer.json`** arms: `pool_2000`, `pool_2000_imbalanced`, `full_train`,
`generalisation_2_3hop`. `model_folder` mistral_7b_instruct, `quantization` 4bit, `default_arm` pool_2000, seed 42.

**`configs/paths.json`** `data_root` = `/cta/users/fyilmaz/thesis-qav2-data` (a real directory, **not**
a symlink at the top level despite the note; the per-dataset entries inside are the symlink layer).
Resolution checked on this box: **17 of 18 `datasets.*` keys resolve**, plus both templated keys for
hops 1-4. The single miss is `musique_chunks_masked_dir` → `musique/chunks_only_question_masked`,
**MISSING** — exactly as the config's own `_note` documents (superseded by `_fixed`, kept for provenance).
All 5 `repo.*` keys resolve.

Every one of the 30 configs has at least one Python reader — **no orphan configs**.

---

## 4. Known defect sites

| issue | site | fixed on main? |
|---|---|---|
| **#40** `_REF_RX` / `reference_validity` | `scripts/musique_decompositions_evaluator.py:102` (`_REF_RX = re.compile(r"\[#(\d+)\]")`), consumed at `:616-628` `_reference_validity`. | **Deliberately frozen, not repaired.** The file's own header (`:33`) says `composite_score` and its `_REF_RX` are "frozen as legacy, not repaired". A replacement landed: `_BARE_REF_RX` at `:705` and `_chain_validity` at `:630-657` (commits `a99b573`, `e3725c7`). The hardened, shared grammar is `src/step_lines.py:59-64`. So: new metric present, legacy term untouched — matching the issue's "no code change until the humans decide". |
| **#47** router output parsing | `components/router/run_router.py:418-470` `parse_hop_response`; fallback chain answer_regex → digit-class → hop_words/number_words → `parsing.default_hop`, returning `(default_hop, False)` at `:470`. Config validation at `:383-412`. | **No.** `git log --since=2026-08-26 -- components/router/run_router.py` is empty; last touch was `fe4b32f`. The instrumentation exists (`parse_fallback` recorded per row, `:298`) — that is what measured the defect; the parser itself is unchanged. |
| **#53** `few_shot.k` in run note | `components/decomposer/run_decomposer.py:3184` renders `f"- Few-shot: enabled={few_shot_enabled} k={few_shot_k} …"`; `few_shot_k` is bound at `:2517` from the **model folder's** `few_shot.k` (=3 in every folder). The reranked path supplies 5. | **No.** Line still reads the model-config k. |
| **#54** `git.commit` at write time | `src/run_artifacts.py:108` — `provenance = git_provenance()` is called inside `write_run_artifacts`, i.e. after the run. `git_provenance()` itself at `:52-81`. | **No.** `git log -- src/run_artifacts.py` shows one commit, `655b935` (the original port). All 4 runners (`run_decomposer.py:3142`, `run_router.py:1220,1387`, `run_answerer.py:1297`, `train_lora.py:636`) inherit it. |
| **#55** script vs runtime ceiling | Script side: `scripts/print_model_params.py:107` `SAFETENSORS_INDEX`, `:124-127` sums every tensor in `weight_map`, `:147` totals shards. Runtime side: `src/model_size.py:64-72` `count_parameters` → `model.num_parameters()`, asserted at `:82-110`. | **No.** Last touch `feef949`, which registered the 7B/14B folders; no both-numbers reporting was added. |

---

## 5. Hygiene

**Dead code:** none found. All 13 `src/` modules are imported elsewhere; every `scripts/` and
`MusiQue/scripts/` file is referenced from a config, a test, a README, or the smoke test.

**Dormant, not dead:** ~8 scripts unchanged since the 2026-08-12 port serve the retired MetaQA lane
(`scripts/kg.py`, `evaluate_decompositions.py`, `analyze_runs.py`, `refine_and_sample_pool.py`,
`check_pool_coverage.py`, `test_similarity_few_shot.py`, `test_similarity_router.py`,
`plot_pool_sweep.py`). README says the dataset moved to MuSiQue. They still import the shared helpers,
so they are a maintenance surface that no current experiment exercises. `[inference]` on "no current
experiment": every log row since exp-004 is MuSiQue.

**Duplicated logic — three reference regexes, deliberately:** `src/step_lines.py:59` `_REF_RX`
(bracketed + malformed + tightened bare), `scripts/musique_decompositions_evaluator.py:102` `_REF_RX`
(bracketed only, frozen) and `:705` `_BARE_REF_RX` (bare, loose), `src/finetune_data.py:41`
`_BARE_REF_RX`. Each is documented with its reason; `tests/test_decomposer_conditions.py:622` asserts
the evaluator's `_split_decomposition_text` **is** `split_step_lines` (identity), so the splitter at
least cannot drift. The regexes can.

**Hard-coded paths:** exactly one, and it is a docstring, not code —
`src/answer_metrics.py:49` names `/cta/users/fyilmaz/thesis-qav2-data/reference/musique_official/`.
No hard-coded absolute path in any executable line of `src/`, `scripts/`, `components/`, `MusiQue/`.

**Magic constants:** `require()` has no default form and there is no `.get(key, 42)` in the pipeline —
spot-checked; the `.get(…, "")` hits in `scripts/run_qwen_smoke.py`, `render_anchor_set.py` and
`analyze_runs.py` are display fallbacks over run outputs, not experiment parameters. One exception
worth naming: `scripts/analyze_runs.py:143` defaults `overall_accuracy` to `0` — a missing metric
reads as zero rather than as missing.

**Data under git:** clean. Total tracked = **6.25 MB**. Largest tracked files are analysis JSONs
(931 KB, 385 KB), `experiments/log.md` (242 KB) and metrics JSONs. Checked for dataset content:
`experiments/exp-016/exp-026/exp-027/metrics.json` contain **zero** occurrences of
`"question"`/`"decomposition"`/`"paragraph"`/`"answer"` as keys; the two big analysis JSONs
self-declare "statistics only — no dataset content"; `configs/anchor_set_45.json` is ids only.
- **`MusiQue/`** holds only 16 prep scripts (`git ls-files MusiQue` = 16 `.py`). No data. Correctly tracked.
- **`few_shot/`** holds 3 JSONs, 36 KB total, tracked: `few_shot_decompositions.json`,
  `few_shot_router.json`, `few_shot_router_masked.json`. These **are** copied dataset rows (MetaQA
  questions + hand decompositions) — but they are prompt assets referenced by `paths.json`'s `repo.*`
  block, small, and reviewable. Correct to track; worth knowing they are the one place dataset text
  lives in git.

**`.gitignore` coverage (4,408 B, well-commented).** Blocks `*.jsonl`, `*.tsv`, `*.csv`, `*.txt`,
weights, `runs/`, `.worktrees/`, caches; negates `configs/**`, `docs/**`, `experiments/**/*.{json,md,yaml,txt}`,
`components/**/models/**/*.{md,json}`, `tests/fixtures/**`. Anchoring of `/models/` and the
extension-scoped negations are correct (a stray `.safetensors` in a model folder still ignored).
**Gap: plain `.json` is never ignored anywhere.** A bulk corpus dumped as `.json` would be silently
committable; only the behavioral rule in CLAUDE.md stops it. Working tree is clean apart from three
untracked PDFs in `Slides/`.

**`.claude/` drift vs. recorded decisions:**
- `.claude/agents/experiment-runner.md` frontmatter is `model: sonnet`; CLAUDE.md still says
  "(Sonnet)". Jahid's 2026-09-20 instruction (per session memory) is opus for every spawn including
  experiment-runner. The other three agents are `model: opus`.
- `.claude/settings.json` sets `"model": "fable"` for the session — consistent with "you lead on Fable 5".
- `.claude/hooks/tracking-gate.sh` (88 lines) implements the `gh pr create` issue-reference gate:
  fail-open by construction, denies the first violating call with a `permissionDecision: deny`
  (`:63`), lets a conscious retry through. Matches CLAUDE.md.

---

## 6. Reproducibility mechanics

**Commit + seed + config snapshot.** `src/run_artifacts.py:88-147` `write_run_artifacts` writes
`config.json`, `metrics.json` and `notes.md` into the run dir, stamping `git_provenance()`
(`commit`, `branch`, `dirty`, `dirty_file_count`, `dirty_files[:20]`) into **all three**, and prints a
loud warning when the tree is dirty (`:135-140`). Seeding: `set_global_seed` is called by every runner
and every sampling script — verified in `run_decomposer.py:2304`, `run_router.py:79/615`,
`run_answerer.py:686`, `train_lora.py` (via `SFTConfig(seed=…, data_seed=…)` at `:181-182`),
`musique_decompositions_evaluator.py:86`, `build_anchor_set.py:54`, `ged_cost_benchmark.py:457`,
and all six `MusiQue/scripts/` samplers. All 20 configs carrying a `seed` hold **42**.
Configs are repo-bound at load (`run_config.py:30-66`): a config outside the repo, or a same-named file
in the CWD, is refused — a real guard against Gate 2 drift.

**Gaps between CLAUDE.md and the code:**

1. **The run lock is prose only.** `docs/compute.md:33` specifies `runs/run.lock` with acquire/release
   and a 24 h stale rule. **No Python code reads or writes it** — the only three matches in `*.py` are
   docstrings (`run_answerer.py:48`, `train_lora.py:27`, `decomposition_junk_battery.py:29`). It is
   held by the `experiment-runner` agent's discipline and by ad-hoc `run_all.sh` traps that live under
   the gitignored `runs/`. Two concurrent runs would not collide on anything in code. (Lock absent now.)

2. **Experiment ids are assigned by hand.** `experiments/log.md` says "one row per experiment, ids
   `exp-001` onward"; nothing in code allocates, validates or de-duplicates an id. The id enters a run
   only as a path component the human types into `--output-root runs/exp-0NN/…`. A typo silently
   mis-files a run.

3. **Issue #54 is the provenance gap in this section**: the recorded `git.commit` is the commit at
   artifact-write time, so on a long run it names code that landed mid-run. The documented workaround
   is the experiment-runner hand-recording the launch SHA in the pre-run log row (exp-021 onward).

4. Launch and verification scripts (`runs/check_ordering_invariant.py`, `check_regen_parity*.py`,
   `run_all.sh`) sit under the gitignored `runs/` and are therefore **not** committed code — anything
   they assert is not reproducible from the repo. `[inference]` from their location; I did not read them.

---

## 7. Model-loading path

**Local HF cache:** `~/.cache/huggingface/hub`, 44 model repos + 1 dataset. No `HF_HOME` or
`TRANSFORMERS_CACHE` is set by any code path (only `tests/test_pool_clustering.py:404,432` sets
`HF_HUB_OFFLINE=1` for its offline probe), so the default cache is what the runners use.

Relevant to this pipeline (`du -sh`):

| model | size | role |
|---|---|---|
| `Qwen/Qwen2.5-14B-Instruct` | **28 G** | decomposer (14.77B, needs the 1.5e10 key) |
| `Qwen/Qwen3.5-9B` | **18 G** | decomposer (the #55 model) |
| `Qwen/Qwen3-8B` | 16 G | router |
| `Qwen/Qwen2.5-7B-Instruct` | 15 G | decomposer (7.62B) |
| `mistralai/Mistral-7B-Instruct-v0.3` | 14 G | decomposer + answerer reader |
| `Qwen/Qwen2.5-3B-Instruct` | 5.8 G | decomposer / router |
| `microsoft/Phi-4-mini` / `Phi-3.5-mini` / `Phi-3-mini-4k` | 7.2 G each | router |
| `Qwen3-4B-Instruct-2507` / `Qwen3-1.7B` / `Qwen2.5-1.5B` / `Qwen3-0.6B` / `Qwen2.5-0.5B` / `Qwen2-0.5B` | 7.5 / 3.8 / 2.9 / 1.5 / 0.95 / 0.95 G | router registry |
| `Jean-Baptiste/roberta-large-ner-english` | 1.4 G | NER masking |
| `intfloat/e5-small-v2` | 128 M | bi-encoder retrieval |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | 128 M | reranker |
| `sentence-transformers/all-MiniLM-L6-v2` | 88 M | alt embedder |

Every model the decomposer and router registries name is cached. The rest (NLLB, mBART, GLiNER,
NuExtract, WizardLM, mT5, Turkish BERT, …) are other projects' — roughly 80 GB of the cache is not
this thesis's.

**Quantization.** Supported at 4bit (NF4, double-quant, configurable compute dtype) and 8bit via
`transformers.BitsAndBytesConfig`, in both inference and training:
`run_decomposer.py:1114-1156` (`load_model`, refuses quantization without CUDA at `:1124`, rejects an
unknown value at `:1145`) and `train_lora.py:103-152, 207-211` (QLoRA). CLI `--quantization
{none,4bit,8bit}` at `run_decomposer.py:1917` overrides the folder default; the resolved value is
recorded into the config snapshot and metrics (`:2462, 2482`). `bitsandbytes==0.49.2` is pinned and
installed in `.venv`. The router runner has **no** quantization support (no `BitsAndBytesConfig`
reference in `run_router.py`).

**How the ceiling is asserted at runtime.** `src/model_size.py`:
- `load_limits()` (`:48-61`) is called by every runner including dry runs and requires every distinct
  key in `_LIMIT_KEY` — a missing key fails at startup, not mid-run (ADR 0016).
- `ceiling_for(component, limits)` (`:74-80`) raises `KeyError` on an unregistered component, so a new
  role cannot quietly inherit an exception.
- `assert_within_ceiling(...)` (`:82-110`) runs **after weights load**, prints
  `[model_size] <component> model=<id> parameters=N ceiling=M (x% of ceiling)`, and raises `SystemExit`
  above the ceiling. With a LoRA adapter attached, the count asserted is base + adapter
  (`run_decomposer.py` docstring, `:47-49`).
- A run that loads no model records `unasserted_note(...)` (`:113-123`) with
  `ceiling_asserted: false` and "parameter count is unmeasured" — no zero, no guess.
- **The smoke test never exercises it**: `scripts/smoke_test.py:20-22` states the router and decomposer
  runners are `--dry-run`, "so the parameter-count assertion in `src/model_size.py` is *not* exercised
  by this test." The only coverage is `tests/test_model_params.py` (199 lines) against fabricated
  safetensors headers.
- Issue #55 lives exactly here: the offline tool (`print_model_params.py:124-147`, sums every tensor in
  `weight_map`) and the runtime assert (`model_size.py:64-72`, `model.num_parameters()`) count
  different things — 9,653,104,368 vs 8,953,803,264 on Qwen3.5-9B.
