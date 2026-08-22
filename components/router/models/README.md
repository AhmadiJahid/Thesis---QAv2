# Router Models

One folder per model. Each contains:

- `prompt.md` — the hop-count prompt template (few-shot). **Byte-identical to v1**: these
  are experimental assets, and editing the text silently breaks comparability with v1 results.
- `config.json` — `model_id`, `generation` (max_new_tokens and decoding), `loader`
  (tokenizer/dtype/device-map flags) and `parsing` (how the hop count is read out of the
  response). The `parsing` and `generation` fields are where v1's per-copy code differences
  now live; the `notes` field in each config records which ones.

Shared in this directory (not per-model, and not v1 assets):

- `prompt_zero_shot.md` — zero-shot prompt, selected with `--prompt-file prompt_zero_shot.md`.
- `prompt_few_shot_musique.md` — the retrieved-few-shot prompt of issue #27: an instruction, a
  `{few_shot_examples}` block the runner fills with the query's retrieved exemplars (each
  labelled with its own gold hop depth), then the query. Named by
  `configs/router_musique.json`, so that config cannot be run with a model folder's MetaQA
  prompt by accident.

Folder naming is lowercase with underscores.
