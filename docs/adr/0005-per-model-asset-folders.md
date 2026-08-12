# 0005. Per-Model Asset Folders with Standard Prompt and Config Files

- **Status**: Accepted
- **Date**: captured retroactively 2026-08-12

## Context

Recorded in v1 at `Thesis---QA/docs/DECISIONS.md`, entry "2026-01-21 — Component model folders and prompt templates". Experiments were expanding to multiple small models per component, and per-model prompts, decoding configs, and outputs needed a consistent home so later analysis could tell runs apart, across Kaggle and Colab.

## Decision

As v1 records it:

- Store per-model assets under `components/<component>/models/<model_name>/`.
- Standardize a `prompt.md` and a `config.json` per model, keeping the prompt strategy and decoding settings explicit and committed.

## Consequences

- Prompt and decoding choices are explicit files, reproducible across environments — at the cost of extra folder overhead per model (the tradeoff v1 accepted).
- The lasting principle for v2 is that **each model's prompt and decoding config are explicit committed artifacts**, which v2's reproducibility rules (config snapshot per experiment) already demand; the exact `components/.../models/...` path is a v1 layout that the port (issue #3) may adapt to v2's repo structure.

## Alternatives considered

The v1 record names no rejected alternatives; the recorded tradeoff was folder overhead per model, accepted for the sake of organized, reproducible per-model experiments.
