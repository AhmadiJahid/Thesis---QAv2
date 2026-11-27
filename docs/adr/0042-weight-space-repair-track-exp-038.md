# 0042. Weight-Space Repair of Under-Decomposition at Depth (exp-038): Method, Leakage Guard and Controls

- **Status**: Accepted. The track: Jahid's decision (relayed by the lead, 2026-11-27), timeboxed to GPU runs by ~5 December. The method, guard and controls below: the implementer's design under that brief, open to Jahid's amendment before the run.
- **Date**: 2026-11-27
- **Scope**: a **separate track outside the few-shot pool study**. It changes no pool-study arm, no pool, no retrieval artifact, and no reported pool-study number. Its results are reported as their own section, never folded into the pool-study tables.

## Context

The decomposer fine-tunes stop planning too early at depth. On the ADR 0007 pinned 600:

- exp-026, the pool-only LoRA fine-tune (`pool_2000_imbalanced`, Mistral-7B-Instruct-v0.3): under-decomposition 0.465 and over-decomposition 0.035 at 4 hops, 0.295 / 0.060 at 3 hops, hop-count EM 0.500 at 4 hops (`experiments/exp-026/notes.md`, per-hop table).
- exp-013, the 2/3-hop-only fine-tune: under-decomposition 0.845 (169/200) and hop-count EM 0.145 (29/200) at 4 hops (`experiments/exp-013/notes.md`).
- The prompted Mistral-7B arm mixes the two errors at 4 hops: under 0.285, over 0.415 on the pinned subset (`experiments/exp-030/resample_summary.csv`, `pinned_subset_recomputed_value`).

"Weight-space programming" edits a model's weights directly, without gradients, after inspecting its activations. Two November 2026 papers were read for this ADR:

- arXiv 2611.09382 (Hu et al., *Exploring Weight-Space Programming*) has an LLM agent locate directions by contrastive difference-of-means and edit `down_proj` / `o_proj` with rank-one and projection edits. It holds out a test split the agent never sees, runs a random equal-norm orthogonal direction and a dose sweep, and reports that discovery-set gains sometimes shrink on fresh items.
- arXiv 2611.20953 (Kundu, *Same Fix, Different Weights*) replicates this with Claude on Qwen3.5-0.8B: MATH-500 6.7% → 51.4–58.1%. Its final edits are output-side (final-norm gain ×0.6–0.68, scaling of token rows such as "Wait" and `</think>`), not removal of an internal direction. Its norm-only control, a temperature change under its sampling decoder, explains 28–35% of the MATH-500 gain.
- Directional ablation by weight orthogonalisation follows Arditi et al. 2024 (arXiv 2406.11717).

The obvious cheap "fix" for stopping too early is a length bias: make EOS less likely. A repair that is only that is not worth a thesis section. So the controls have to separate a direction edit from a length bias.

## Decision

1. **Target**: the exp-026 adapter on its base, merged in fp16 (`configs/exp038_weight_space.json`). It is the pool-study fine-tune whose error is one-sided: it under-decomposes at depth and barely over-decomposes. The prompted arm's error is two-sided, so "plans got longer" would be confounded with "over-decomposition got worse". exp-013's stronger defect comes from leaving 4-hop rows out of training, a data gap rather than a model behaviour.
2. **Defect operationalised at the decision token.** A step boundary is the generated token ending a step line, where the next token is a newline (continue) or EOS (stop). With gold hop count *k*, a stop at boundary *j < k* is `premature_stop`, a newline there is `continue_needed`, and a stop at *j = k* is `correct_stop` (`src/weight_space.py`).
3. **Locate.** Greedy-generate on the locate split, then capture the residual stream after every decoder layer at every boundary. Compute two per-layer difference-of-means directions, each **matched on the boundary index *j*** so that position in the plan cannot pose as the stop signal:
   - `stop_vs_continue` = premature_stop − continue_needed
   - `premature_vs_correct_stop` = premature_stop − correct_stop. This compares, for example, a 3-hop item stopping after step 2 with a 2-hop item stopping after step 2.
   Layers are ranked by AUROC of the projection on val_select, within layers 2–28. The top 3 per direction go into the sweep.
4. **Edit.** (a) Diagnostic: activation steering at one layer, by hook. (b) The weight edit proper: orthogonalise every writer to the residual stream against the unit direction, `W ← W − α r rᵀ W` for `o_proj` and `down_proj` in all layers and `E ← E − α (E r) rᵀ` for the embedding, with α ∈ {0.5, 1}. The hook-equivalent ablation is checked against the weight edit on 8 locate prompts, max logit difference recorded. The edit ships as a spec JSON. `run_decomposer.py --weight-edit <spec>` merges the adapter, applies the spec and records spec, spec sha256 and direction sha256 in the run's snapshot and metrics.
5. **Leakage guard.** Locating, layer choice, the sweep and the choice use MuSiQue **train** rows only. The adapter's 2,000 training rows are excluded. Every id and every normalised question text of the whole MuSiQue dev file, which contains the pinned 600, is excluded and asserted disjoint. There are three seeded splits: locate 400/hop, val_select 100/hop for choosing, val_confirm 100/hop run once and never used for choosing. Their id hashes are pinned in the notebook. The pinned 600 are generated only after `experiments/exp-038/choice.json` is **committed**. The spec files' hashes are frozen and re-checked before every pinned-600 arm.
6. **Claude's role, explicit and logged.** The notebook stops twice: after the directions, and after the val_select sweep. At each stop Claude, in Jahid's terminal, reads the saved statistics. Claude may change the sweep (`candidates_override.json`) and chooses the edit and the length-matched control (`choice.json`), with its reasoning in `experiments/exp-038/choices.md`, before any pinned-600 generation. The pre-registered default rule (`select_candidate`) picks the highest val_select hop-count EM among orthogonalisation edits that raise over-decomposition by ≤ 0.05 and lower 2-hop hop-count EM by ≤ 0.05. Claude may depart from the rule only in writing.
7. **Controls, all on the pinned 600 beside the edit**:
   - **norm-only**: final-norm gain ×0.6, as in 2611.20953. Under greedy decoding a uniform logit scale cannot change the argmax, so this is expected to change nothing beyond fp16 rounding. It is run to show that, not assumed.
   - **random direction**: a seeded unit vector orthogonal to the chosen direction, same layers, matrices and α.
   - **length-matched EOS-row control**: lm_head's EOS row scaled. 2611.20953 made its edits as token-row scaling. The scale is the one whose val_select mean step count is closest to the edit's. "The edit only lengthens plans" is thus tested at equal length.
   - over- **and** under-decomposition are reported overall and per hop for every arm.
8. **Evaluation.** All five arms are scored with the full report card (ADR 0029), overall and per hop: baseline (merged, unedited, generated in the same notebook), edit, random, EOS control and norm-only. The baseline of every paired test is the in-notebook baseline. exp-026's stored 3090 file is used only for a reproduction count, because merged fp16 on a different GPU can flip greedy tokens. Paired tests use the evaluator's own bootstrap, McNemar and paired-t functions, unchanged, as exp-029 does (ADR 0009). Holm (ADR 0036) is applied within each pair's family of 7 report-card terms × 4 strata (`experiments/exp-038/analysis.py`). The primary pair is edit vs baseline. The length-bias test is edit vs EOS control.

## Consequences

- The thesis can make one bounded claim: whether a weight edit found on train rows changes depth behaviour on the pinned 600 beyond what a length bias, a random direction and a temperature change do. Possible results include a negative or null one, and that is reportable.
- `run_decomposer.py` gains one default-off flag, `--weight-edit`, so the five arms run through the exact evaluation path of exp-026.
- Weights are never committed. The direction `.npy` files and activations stay on Drive (`*.npy` is gitignored). Their sha256 values are in the committed spec files.
- One merged-model load per stage. The Colab plan, A100 recommended and L4 workable, is in `experiments/exp-038/README.md`.
- Risk: with few `premature_stop` boundaries at a given *j*, a direction is noisy. `min_per_cell` (5) drops thin cells and the counts are in `directions_table.md`.
- Risk: removing a stop direction everywhere can make plans run on. The over-decomposition limit in the rule and the per-hop over-decomposition reporting are the guard.

## Open for Jahid

- **The closed-model constraint.** CLAUDE.md forbids a closed commercial model from scoring, rating or judging decomposition quality. In this design Claude chooses an edit from automatic step counts on train rows. It never scores a decomposition, and every pinned-600 number comes from the repo evaluator. Our reading is that this is outside the constraint, much like a researcher choosing a hyperparameter. Whether the supervisor reads it the same way is his call, and the thesis text should say who chose what.
- **Number reservation.** ADR 0040 item 2 asks for this ADR number and exp-038 to be reserved on `main` before use. This lane was told not to push, so it has not done that. Reserve both, or renumber, before merging.

## Alternatives considered

- **Teacher-forced boundaries** (gold steps fed in, read P(EOS) at each gold boundary). Cheaper and gives more decisions, but it measures behaviour on contexts the model never produces. The defect is a free-generation one. Not taken.
- **Output-side edit only** (the 2611.20953 route: EOS row / final norm). Kept as a *control*, because here it is exactly the trivial length bias the experiment must beat.
- **Locating on the pinned 600 or anywhere in dev.** Refused: that is the test set.
- **Editing only the LoRA factors.** The unmerged base would still write the direction. Merging first is simpler and exact.
- **4-bit base.** It would move precision away from exp-026's fp16 evaluation (ADR 0034 decision 3), and editing quantised weights is not exact.
