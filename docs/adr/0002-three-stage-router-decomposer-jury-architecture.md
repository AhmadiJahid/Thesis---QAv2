# 0002. Three-Stage Router–Decomposer–Jury Architecture

- **Status**: Accepted
- **Date**: captured retroactively 2026-08-12

## Context

Recorded in v1 at `Thesis---QA/docs/DECISIONS.md`, entry "2025-12-15 — Multi-hop Question Decomposition Pipeline Architecture". The pipeline decomposes multi-hop questions (1-hop, 2-hop, 3-hop, initially from MetaQA), and an architecture was needed for its components.

## Decision

We build a 3-stage pipeline (as v1 records it):

- **Router** outputs a hop count (1/2/3), using a small model (0.5B–1.5B). The hop count from the router provides specific guidance for the decomposer.
- **Decomposer** outputs a JSON list of sub-questions — no reasoning in the output — using a medium model (7B–8B). The JSON format enables structured processing and validation.
- **Jury** validates that (1) sub-questions are in correct order, (2) they compose to the original question, (3) sub-questions make sense. Output is pass/fail. It uses the same model as the decomposer, to reduce infrastructure complexity.
- **Evaluation** was manual review at the time of the decision, to allow iterative refinement before automated metrics.

## Consequences

As recorded in v1: the decomposer output needs JSON schema validation; the jury needs clear pass/fail criteria; automated evaluation metrics are follow-up work. No reasoning in decomposer output simplifies the pipeline but may reduce interpretability; jury pass/fail is binary but sufficient for initial validation; manual evaluation is time-consuming but was necessary without ground truth. In v2, the router-guides-decomposer structure is the shape the ported pipeline (issue #3) preserves. Note: the jury stage exists in the design but has never been implemented or used (per the confirmed idea paragraph in v2's [`README.md`](../../README.md)).

## Alternatives considered

The v1 record names no rejected alternatives; it records the tradeoffs accepted instead (listed under Consequences above). Model choices attached to this decision are recorded separately in [0004](./0004-initial-model-selections.md).
