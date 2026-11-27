# 0044 — exp-036: "same recipe" on Qwen3.5-9B, the 4-bit primary pair, and Colab as a run venue

- **Status:** accepted (lead, 2026-11-27, after the code review of `feature/50-qwen-pool-finetune`
  at 823614e; Jahid's to reverse). Jahid decided to run exp-036 and to run it on Colab, driving it
  himself; the items under "Decided by the lead" are the lead's; the items under "Implementer
  defaults" are the implementer's.
- **Context:** exp-036 replicates exp-026 (pool-only QLoRA fine-tune of Mistral-7B-Instruct-v0.3)
  with one change, the base model, to ask whether the fine-tuning gain holds for a second family
  (issue #49: every fine-tune claim so far is Mistral-only). Qwen3.5-9B is a hybrid architecture
  (transformers 5.5.4 `models/qwen3_5/modeling_qwen3_5.py`, HF config at revision
  `c202236235762e1c871ad0ccb60c8ee5ba337b9a`): 32 decoder layers, 8 full attention
  (`q_proj`/`k_proj`/`v_proj`/`o_proj`) and 24 Gated DeltaNet linear attention (`in_proj_qkv`,
  `in_proj_z`, `in_proj_b`, `in_proj_a`, `out_proj`), MLP (`gate_proj`/`up_proj`/`down_proj`) in all
  32. The only Qwen3.5-9B prompting arm on this pool, exp-027, ran at 4-bit; exp-026 evaluated at
  fp16. The run box (docs/compute.md, LittleGazor) is no longer used; the GPU work runs on Colab.

## Decided by the lead

1. **"Same recipe" means the same `target_modules` names, not the same coverage.** exp-036 keeps
   `configs/finetune_decomposer.json` `lora.target_modules` byte-identical to exp-026's. On Qwen3.5-9B
   that adapts attention in 8 of 32 layers and the MLP in all 32 (expected 29,097,984 trainable
   parameters, derived from the HF config and to be measured by the run, vs exp-026's measured
   41,943,040). Adding the Gated DeltaNet projections would be a second change to the recipe. Every
   statement comparing exp-036 with exp-026 carries this: the base model moves and, with it, which
   share of the network the adapter touches.
2. **The 4-bit evaluation is kept, made mandatory, and is the primary.** The primary comparison for
   "does the fine-tuning gain hold for a second family" is pair (b): the exp-036 adapter on a 4-bit
   base vs exp-027 (Qwen3.5-9B prompting, 4-bit), same 600, so only adapter + prompt moves. The fp16
   evaluation replicates exp-026's protocol and serves pair (a) (vs exp-026); it is secondary.
   **Precision asymmetry, stated wherever the two families' gains are set side by side:** the
   Mistral reference gain (exp-026 vs exp-015) is fp16 vs fp16, the Qwen gain is 4-bit vs 4-bit.
3. **The cross-family claim needs its own test.** A paired bootstrap (with the paired-t p for Holm)
   on the per-item difference of differences — (exp-036 4-bit − exp-027) − (exp-026 − exp-015) over
   the shared pinned 600 — is pre-registered as its own Holm family (F3, 7 report-card terms).
   Without it, a cross-family statement is qualitative sign agreement of two gains and is worded so.
   Families (ADR 0036): F1 pair (b) overall (7), F2 pair (b) per hop (21), F3 the difference of
   differences (7), F4 pair (a) overall + per hop (28); context pairs are in no family.
4. **Colab is a run venue, without the run lock.** A Colab runtime is a single-user VM, so the
   `runs/run.lock` convention of docs/compute.md (which exists because LittleGazor's GPU was shared)
   does not apply there. What replaces it for integrity: the notebook pins one commit for the whole
   experiment (first run records it on Drive; training and both evaluation snapshots must carry it),
   builds the exp-026 environment exactly (Python 3.13.9 via uv, the `requirements.txt` pins, torch
   2.10.0+cu128) and refuses on drift, and writes every output to Drive as it is produced.
   docs/compute.md gets a short Colab section pointing here.
5. **An interrupted training restarts from step 0** (same seed), rather than adding resume logic to
   `train_lora.py`; the reviewer agreed. Finished-epoch checkpoints stay on Drive as a record.

## Implementer defaults

- The fp16 output-health gate (code-review item I-4), applied to both evaluations: at most **1**
  empty decomposition — exp-027's recorded `rows_with_empty_decomposition` for the same base model
  on the same 600 (runs/exp-027/decomposer/20260920_102503/metrics.json); not all rows empty; **0**
  rows whose raw output contains a run of `!` (token id 0 of the Qwen3.5 tokenizer, what greedy
  argmax returns over all-NaN logits). The run length of 8 is **arbitrary**; 0 such rows were found
  in exp-026's and exp-027's results.json. A failed fp16 gate stops that evaluation, is recorded,
  and leaves the primary (4-bit) result standing; `compare.py --fp16-failed-gate` then drops pair
  (a), the context pairs and F4, never F1-F3.
- flash-linear-attention / causal-conv1d are not installed, because exp-027 ran Qwen3.5 on the
  torch fallback.
- Comparator per-item files are exp-032's re-parsed re-evaluations (ADR 0039 cross-parser rule).

## Consequences

- exp-036 vs exp-026 is never reported as a one-factor family effect (family, generation, size and
  adapter coverage move together).
- A reader comparing the two families' gains is pointed to F3 and to the precision asymmetry.
- Any further Colab run follows docs/compute.md's Colab section rather than the lock.
