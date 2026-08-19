# Synthetic smoke-test fixtures

Everything under this directory is **hand-written and fabricated**. No row is copied from
MuSiQue or MetaQA: the movies, people, questions, decompositions and ids are invented, and
the numbers are chosen to be small enough to eyeball. They exist so the ported pipeline has
an executable end-to-end check *before* real data and compute are resolved (issue #2).

They are **not** an evaluation set. Any metric produced from them says nothing about the
research question — it only says the code path runs and writes its artifacts.

## Layout

```
data_root/                     # stands in for the real data_root (configs/smoke_paths.json)
  metaqa/
    kb.txt                     # 17 fabricated triples, MetaQA's subj|rel|obj format
    refined_{1,2,3}hop.txt     # questions, one per line
    answers_{1,2,3}hop.txt     # answers aligned line-by-line with the questions above
    pool/qa_train_{1,2,3}_hop.txt   # raw "question<TAB>answer" pool rows with [brackets]
  musique/
    musique_ans_v1.0_train.jsonl        # 8 rows: 4x 2hop, 4x 3hop1 (4 per stratum is the
                                        # minimum StratifiedKFold(n_splits=4) accepts)
    musique_ans_v1.0_dev.jsonl          # gold decompositions keyed by id
    dev_data/
      musique_ans_v1.0_dev_clean.jsonl  # gold for the decomposition evaluator
      musique_ans_v1.0_dev_questions_*.jsonl   # per-hop dev question files
      musique_ans_v1.0_dev_sample_{2,3,4}_hop_200.jsonl  # stands in for the pinned ADR 0007
                                        # evaluation set: real names, 3 fabricated rows per
                                        # hop, NOT 200. The ids match the gold rows above and
                                        # retrieval/top5_musique_conditions.jsonl, so both the
                                        # train/eval overlap assertion in train_lora.py and
                                        # the decomposer's id-identity check have something
                                        # real to check against.
    chunks_only_question_masked_fixed/roberta_large_ner_english/
      ..._questions_2_hop.jsonl, ..._questions_3_hop_1.jsonl   # chunks to combine
      ..._all_questions_all_expanded_enriched.jsonl            # the pool to sample from
    chunks_only_question_masked/{bert_large_NER,roberta_large_ner_english}/
      ..._train_0_questions_2_hop.jsonl   # two NER variants of the same rows
predictions/                   # decomposer-shaped prediction files
decomposer_arms/               # two fabricated "arms" (metrics.json only) for
                               # scripts/compare_decomposer_arms.py; the smoke test copies
                               # the same predictions file into both, so every comparison
                               # difference is exactly 0 by construction. The prompting arm
                               # carries a cost block with its 'definitions' (so the note's
                               # cost-column definitions are exercised); the finetuned arm
                               # deliberately has none, to exercise the cost-unmeasured
                               # branch.
adapter/                       # a fabricated LoRA adapter directory with NO weights:
                               # adapter_config.json (base_model_name_or_path) and
                               # training_provenance.json (the prompt it was "trained" on),
                               # which is what the smoke test's --adapter dry run is checked
                               # against by run_decomposer's base-model and prompt-parity
                               # guards
retrieval/
  top20.jsonl                  # a similarity top-k file (also used as decomposer input)
  top5_musique_conditions.jsonl # retrieval input for the three MuSiQue conditions: one row
                               # per dev_sample fixture question (9), typed_top_k of 6 so
                               # k=5 still holds after self-exclusion. Two rows exercise it:
                               # the first lists the query itself (same id), and
                               # 4hop1__d003_c's candidate 4hop1__t010_j repeats the query's
                               # question text under a different id.
pool_sweep_summary/            # a sweep summary CSV for the plotting script
router_runs/                   # two completed router runs for the analysis script
decomposer_run/                # a run directory with analysis/ dumps
```

## Running the smoke test

```bash
python scripts/smoke_test.py            # every stage that runs without model downloads
python scripts/smoke_test.py --list     # show the stages
```

The runner sets `QAV2_PATHS_CONFIG=configs/smoke_paths.json`, so every script reads its
normal committed config but resolves data paths into this directory. Output goes to
`runs/smoke/` (gitignored).

Stages that need a model download (NER masking, bi-encoder similarity, cross-encoder
rerank, the similarity probes) and the two model-loading runners are **not** executed here;
the runners are exercised with `--dry-run`, which assembles prompts and writes artifacts
without loading weights.
