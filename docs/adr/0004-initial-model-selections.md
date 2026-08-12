# 0004. Initial Model Selections for Pipeline Components

- **Status**: Accepted
- **Date**: captured retroactively 2026-08-12

## Context

Recorded in v1 at `Thesis---QA/docs/MODEL_SELECTION.md` (whole document, "researcher-mode analysis, Dec 2025") and cross-referenced from `Thesis---QA/docs/DECISIONS.md` (2025-12-15 entry, "Model Selection" section). The 3-stage architecture ([0002](./0002-three-stage-router-decomposer-jury-architecture.md)) needed concrete open-weight models per component, small enough to run on Kaggle/Colab GPUs.

## Decision

Initial model selections, as v1 records them:

- **Router**: `Qwen/Qwen2.5-0.5B-Instruct` (494M, Apache 2.0) — fast classification baseline, strong instruction-following.
- **Decomposer**: `Qwen/Qwen2.5-7B-Instruct` (7.6B, Apache 2.0) — strong instruction-following and structured JSON generation; 4-bit quantization expected on Kaggle GPU.
- **Jury**: the same model as the decomposer, to reduce infrastructure complexity. (The jury stage was never implemented or used — see the confirmed idea paragraph in v2's [`README.md`](../../README.md) and [0002](./0002-three-stage-router-decomposer-jury-architecture.md).)

## Consequences

- All selected models support instruction-following and structured generation; 4-bit/8-bit quantization may be needed for the 7B–8B models on Kaggle GPU (per v1's implementation notes).
- These were **initial** selections. v1's later router measurements (summarized in [`docs/prior-work.md`](../prior-work.md), from `Thesis---QA/handoff/results_analysis/FINDINGS.md`) show other models performing differently — e.g. Qwen2.5-3B-Instruct leading the few-shot router table — but v1 records no decision changing the defaults; whether v2 changes them is an open call for Jahid.
- The primary selections sit inside v2's standing size caps (~8B overall, ~600M router). However, against the ~600M router cap, v1's router alternatives Qwen2.5-1.5B-Instruct (1.5B) and DeepSeek-R1-Distill-Qwen-1.5B (1.8B) — and the table-leading Qwen2.5-3B-Instruct from v1's later measurements — all exceed the cap and are excluded in v2 as-is; changing the cap is a supervisor decision, per `CLAUDE.md` standing constraints.

## Alternatives considered

Explicitly listed in v1's `MODEL_SELECTION.md`:

- **Router alternatives**: `Qwen/Qwen2.5-1.5B-Instruct` (1.5B, Apache 2.0) — better accuracy if 0.5B insufficient; `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` (1.8B, MIT) — reasoning-focused distillation.
- **Decomposer alternative**: `EssentialAI/rnj-1-instruct` (8.3B, Apache 2.0) — recent, possibly better structured output.

v1 kept these as fallbacks rather than primaries; it records no further reasoning for the ranking beyond the fit notes above.
