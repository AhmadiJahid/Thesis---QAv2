# 0042. Weight-Space Repair of Under-Decomposition at Depth (exp-038): Method, Leakage Guard and Controls

- **Status**: Accepted. The track and Claude's role in choosing the edit: Jahid's decisions (2026-11-27). The method, guard and controls below: the implementer's design under the lead's brief, amended after code review of `0a47207`. The track is timeboxed to GPU runs by ~5 December.
- **Date**: 2026-11-27
- **Scope**: a **separate track outside the few-shot pool study**. It changes no pool-study arm, no pool, no retrieval artifact, and no reported pool-study number. Its results are reported as their own section, never folded into the pool-study tables.

## Context

The decomposer fine-tunes stop planning too early at depth. On the ADR 0007 pinned 600:

- **exp-026**, the pool-only LoRA fine-tune (`pool_2000_imbalanced`, Mistral-7B-Instruct-v0.3). At 4 hops: under-decomposition 0.465, over-decomposition 0.035, hop-count EM 0.500. At 3 hops: under 0.295, over 0.060 (`experiments/exp-026/notes.md`, per-hop table).
- **exp-013**, the 2/3-hop-only fine-tune. At 4 hops: under-decomposition 0.845 (169/200), hop-count EM 0.145 (29/200) (`experiments/exp-013/notes.md`).
- **The prompted Mistral-7B arm** mixes the two errors at 4 hops: under 0.285, over 0.415 on the pinned subset (`experiments/exp-030/resample_summary.csv`, `pinned_subset_recomputed_value`).

"Weight-space programming" edits a model's weights directly, without gradients, after inspecting its activations. Two November 2026 papers were read:

- **arXiv 2611.09382** (Hu et al., *Exploring Weight-Space Programming*): an LLM agent locates directions by contrastive difference-of-means and edits `down_proj` / `o_proj` with rank-one and projection edits. It holds out a test split the agent never sees, runs a random equal-norm orthogonal direction and a dose sweep, and reports that discovery-set gains sometimes shrink on fresh items.
- **arXiv 2611.20953** (Kundu, *Same Fix, Different Weights*) replicates this with Claude on Qwen3.5-0.8B: MATH-500 6.7% → 51.4–58.1%. The final edits were output-side: final-norm gain ×0.6–0.68, and scaling of the "Wait" and `</think>` token rows. No internal direction was removed. A norm-only control, which acts as a temperature change under that paper's sampling decoder, explains 28–35% of the MATH-500 gain.

Directional ablation by weight orthogonalisation follows Arditi et al. 2024 (arXiv 2406.11717).

The obvious cheap "fix" for stopping too early is a length bias: make EOS less likely. A repair that is only that is not worth a thesis section, so the controls have to separate a direction edit from a length bias.

## Decision

1. **Target: the exp-026 adapter on its base, merged in fp16** (`configs/exp038_weight_space.json`).
   - Its error is one-sided: it under-decomposes at depth and barely over-decomposes.
   - The prompted arm's error is two-sided, so "plans got longer" would be confounded with "over-decomposition got worse".
   - exp-013's stronger defect comes from leaving 4-hop rows out of training, a data gap rather than a model behaviour.
   - **Disclosure (from code review):** the target *model* was picked by reading its pinned-600 per-hop numbers above. The *edit*, meaning direction, layers, α, the length control and the final choice, is never tuned on the pinned 600 (item 5).
2. **The defect is operationalised at the decision token.**
   - A step boundary is the generated token that ends a step line, where the next token is a newline (continue) or EOS (stop).
   - With gold hop count *k*: a stop at boundary *j < k* is `premature_stop`; a newline there is `continue_needed`; a stop at *j = k* is `correct_stop` (`src/weight_space.py`).
3. **Locate.**
   - Greedy-generate on the locate split. Capture the residual stream after every decoder layer at every boundary.
   - Compute two per-layer difference-of-means directions. Each is **matched on the boundary index *j***, so that position in the plan cannot pose as the stop signal:
     - `stop_vs_continue` = premature_stop − continue_needed
     - `premature_vs_correct_stop` = premature_stop − correct_stop. This compares, for example, a 3-hop item stopping after step 2 with a 2-hop item stopping after step 2.
   - Rank layers by AUROC of the projection on val_select, within layers 2–28. The top 3 per direction go into the sweep.
4. **Edit.**
   - (a) Diagnostic: activation steering at one layer, by hook.
   - (b) The weight edit proper: orthogonalise every writer to the residual stream against the unit direction, with α ∈ {0.5, 1}:
     - `W ← W − α r rᵀ W` for `o_proj` and `down_proj` in all layers;
     - `E ← E − α (E r) rᵀ` for the embedding.
   - lm_head and the embedding are asserted untied before any edit.
   - The hook-equivalent ablation is compared with the weight edit on 8 locate prompts. The run stops unless every argmax agrees and the largest logit difference is ≤ 0.5. That allowance is unmeasured and in the config; relaxing it is a `choices.md` entry.
   - The edit ships as a spec JSON. `run_decomposer.py --weight-edit <spec>` is refused unless quantization is `none`. It merges the adapter, applies the spec, and records the validated spec, the spec file's sha256, the direction file's sha256 and the per-tensor change (Frobenius norm of the change and its ratio to the weight's norm). Without the flag, no `weight_edit` key is written anywhere.
5. **Leakage guard and freeze.**
   - Locating, layer choice, the sweep, the length match and the choice use MuSiQue **train** rows only.
   - The adapter's 2,000 training rows are excluded.
   - Every id and every normalised question text of the whole MuSiQue dev file, which contains the pinned 600, is excluded and asserted disjoint.
   - Three seeded splits: locate 400/hop; val_select 100/hop, for choosing; val_confirm 100/hop, run once and never used for choosing. Their id hashes are pinned in the notebook.
   - The pinned 600 are generated only after `experiments/exp-038/choice.json` is committed, in **exactly one commit**: a second commit makes the freeze refuse.
   - `specs_frozen.json` records every spec and direction file's sha256 and is never rewritten. `make-specs` refuses once it exists. Each confirm item and each pinned-600 arm is checked against it, and its output directory is keyed to its spec's sha256.
   - Sweep done-files are keyed to each candidate's content hash. A revised spec or candidate therefore re-runs; it can never reuse an earlier output.
6. **Claude's role, explicit and logged.**
   - The notebook stops twice: after the directions, and after the val_select sweep.
   - At each stop Claude, in Jahid's terminal, reads the saved statistics. It may change the sweep (`candidates_override.json`). It chooses the edit and the length-matched control (`choice.json`), with its reasoning in `experiments/exp-038/choices.md`, before any pinned-600 generation.
   - The pre-registered default rule (`select_candidate`) picks the highest val_select hop-count EM among orthogonalisation edits that raise over-decomposition by ≤ 0.05 and lower 2-hop hop-count EM by ≤ 0.05. Claude may depart from it only in writing.
7. **Controls, all on the pinned 600 beside the edit.**
   - **Length-matched EOS-bias control.** An *additive* bias on the EOS logit, written as an lm_head bias vector that is zero elsewhere.
     - Additive because a row scale (the 2611.20953 style) multiplies a logit whose sign varies with context, so it can raise or lower stopping (code review).
     - The bias is chosen on val_select as the one whose mean predicted step count is closest to the chosen edit's.
     - If the gap exceeds `length_match.tolerance_mean_steps` (0.05), the bracket is refined for up to 3 rounds.
     - The achieved gap is recorded on val_select (`proposed_choice.json`, `spec_eos_control.json`) and on the pinned 600, overall and per hop (`analysis_metrics.json`). Per-hop step-count histograms are reported for every arm.
   - **Random direction.** A seeded unit vector orthogonal to the chosen direction, with the same layers, matrices and α. The size of its weight change relative to the edit's is reported.
   - **Norm-only sanity check.** Final-norm gain ×0.6, as in 2611.20953. Under greedy decoding a uniform logit scale cannot change an argmax, so this arm should reproduce the baseline up to fp16 rounding. It can rule nothing in or out about the edit; it shows the decoder is what we think it is.
   - **Over- and under-decomposition** are reported overall and per hop for every arm.
8. **Evaluation and the headline rule.**
   - All five arms are scored with the full report card (ADR 0029), overall and per hop: baseline (merged, unedited, generated in the same notebook), edit, random, EOS control and norm-only.
   - The baseline of every paired test is the in-notebook baseline. exp-026's stored 3090 file is used only for a reproduction count, because merged fp16 on a different GPU can flip greedy tokens.
   - Paired tests use the evaluator's own bootstrap, McNemar and paired-t functions, unchanged, as exp-029 does (ADR 0009).
   - Holm (ADR 0036) is applied **within each pair's family** of 7 report-card terms × 4 strata = 28 tests (`experiments/exp-038/analysis.py`).
   - **Headline claim.** "The edit repairs 4-hop under-decomposition beyond a length bias" is made only if hop-4 `under_decomposition` is lower for the edit and Holm-significant in **both** family P (edit vs baseline) **and** family L (edit vs the length-matched EOS-bias control).
     - The two families are not pooled; the claim is the conjunction.
     - P alone supports at most "the edit changes depth behaviour", which a length bias may also do.
     - Every other outcome is reported as what it is.

## Consequences

- The thesis can make one bounded claim, by the rule in item 8. A null or negative result is reportable.
- `run_decomposer.py` gains one default-off flag, `--weight-edit`, so the five arms run through the exact evaluation path of exp-026. Without the flag, the snapshot and metrics are unchanged. A test asserts the key sets differ only by the two `weight_edit` keys.
- Weights are never committed. The direction `.npy` files and activations stay on Drive (`*.npy` is gitignored). Their sha256 values are in the committed spec files.
- The repo is public, so the notebook clones without a token. The only secret, the HF token, lives in the environment, and the notebook's shell helper refuses any command containing it.
- One merged-model load per GPU stage. The Colab plan is in `experiments/exp-038/README.md`: A100 recommended, L4 workable.
- **Risk: noisy directions.** With few `premature_stop` boundaries at a given *j*, a direction is noisy. `min_per_cell` (5) drops thin cells, and the counts are in `directions_table.md`.
- **Risk: plans that run on.** Removing a stop direction everywhere can make plans run on. The guard is the over-decomposition limit in the default rule plus over-decomposition reported per hop.

## Decided by Jahid

- **Claude chooses the edit (2026-11-27).** CLAUDE.md forbids a closed commercial model from scoring, rating or judging decomposition quality. Here Claude chooses an edit from automatic step counts on train rows. It never scores a decomposition, and every pinned-600 number comes from the repo evaluator. Jahid approved this role on 2026-11-27 and does not consider it a closed-model judge. The thesis text says who chose what: `choices.md` and `choice.json` are the record.
- **Ids.** 0041 = exp-035, 0042 = this record, 0043 = count-first, 0044 = exp-036. The lead reserves exp-038 and this ADR number on `main` before merge (ADR 0040 item 2).

## Alternatives considered

- **Teacher-forced boundaries** (gold steps fed in, P(EOS) read at each gold boundary). Cheaper and gives more decisions, but it measures behaviour on contexts the model never produces. The defect is a free-generation one. Not taken.
- **Output-side edit only** (the 2611.20953 route). Kept as a *control*, because here it is exactly the trivial length bias the experiment must beat.
- **Multiplicative EOS-row scaling as that control.** Used in the first draft, replaced after review: it is sign-dependent, so it does not reliably lengthen plans.
- **Locating on the pinned 600 or anywhere in dev.** Refused: that is the test set.
- **Editing only the LoRA factors.** The unmerged base would still write the direction. Merging first is simpler and exact.
- **4-bit base.** It would move precision away from exp-026's fp16 evaluation (ADR 0034 decision 3), and editing quantised weights is not exact. `--weight-edit` refuses it.
