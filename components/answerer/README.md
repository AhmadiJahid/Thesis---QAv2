# Answerer

The **end-to-end** half of MuSiQue evaluation (ADR [0006](../../docs/adr/0006-drop-the-jury-fix-dataset-roles-and-the-few-shot-method.md),
issue #16): it executes a decomposition and scores the answer it produces, so a decomposition
is judged by where it leads and not only by how it looks. The decomposition-quality half is
`scripts/musique_decompositions_evaluator.py`.

```bash
# whole loop on a tiny sample, no weights (this is what the smoke test runs)
python components/answerer/run_answerer.py --predictions <run>/results.json \
    --dry-run --dry-run-limit 5 --allow-unpinned-eval-set

# score one decomposition arm end to end
python components/answerer/run_answerer.py --predictions runs/<arm>/results.json

# the oracle-decomposition ceiling: gold plans, same reader, same evaluation set
python components/answerer/run_answerer.py --gold-decompositions
```

The three conventions it implements are Jahid's, recorded in ADR
[0019](../../docs/adr/0019-musique-answering-backend-conventions.md):

1. the **reader** is a model folder from the decomposer's registry
   (`components/decomposer/models/`), default `mistral_7b_instruct` — no new model family;
2. the **context** for every sub-question is the MuSiQue item's full paragraph list;
3. **scoring** is MuSiQue's official answer EM / answer F1 against the gold answer plus its
   aliases (`src/answer_metrics.py`).

Everything else lives in `configs/answer_musique.json`.

## Files here

- `run_answerer.py` — the runner. Loads its model through `run_decomposer.load_model` and
  asserts the parameter ceiling (`src/model_size.py`, component `answerer`) against
  `answerer_max_params` = **8e9**: the reader keeps the standing ~8B ceiling and does not
  inherit the decomposer's ADR 0015 raise, so a 9B *reader* is refused at load time until
  Jahid records an extension (ADR 0019).
- `prompts/reader.md` — the reader prompt, one file for the whole registry. A
  `chat_template` model folder gets the halves either side of `<<<USER>>>` as system/user
  messages; a `plain` folder gets them concatenated. `{context}` and `{question}` are
  required **in the half that gets filled** — for a chat folder, the user half: a
  placeholder above the marker is passed through as literal text, so putting `{context}`
  there would run the reader closed-book, and the runner refuses that.

There are no per-model folders here: a reader prompt is not model-specific the way a
decomposition prompt is (those are byte-identical to v1 and must stay so). If a model ever
needs its own reader prompt, `reader_prompt_file` in the config is the place it goes.

## What lands where

Per run: a config snapshot, a metrics JSON and a run note (`answer_config.json`,
`answer_metrics.json`, `answer_notes.md`) plus `answer_per_item.json` and a
`prompts_log/` sample, all under the gitignored runs root. The metrics JSON carries
aggregates and counts only; the per-item file carries questions and answers, which is why it
stays out of git (CLAUDE.md: data never enters git).
