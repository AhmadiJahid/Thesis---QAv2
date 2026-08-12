# Decomposer Models

One folder per model. Each contains:

- `prompt.md` — the decomposition prompt. **Byte-identical to v1.**
- `prompt_unguided.md` — used when the run does not put the hop count in the prompt. Only the
  models that had one in v1 have one here; for the others an unguided run substitutes
  `unguided_hop_placeholder` from `configs/decomposer.json` for `{hop_count}`, as v1 did.
- `config.json` — `prompt_style`, `generation`, `loader` (including default `quantization`),
  `few_shot`, `post_process` and `logging`. The `notes` field records which of these fields
  carry a behaviour that was hard-coded in v1's copy of `decomposer.py` for that model.

Folder naming is lowercase with underscores.
