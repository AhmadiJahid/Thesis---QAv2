# 0006. Drop the Jury; Fix Dataset Roles and the Few-Shot Method

- **Status**: Accepted
- **Date**: 2026-08-12

Supersedes [0002](./0002-three-stage-router-decomposer-jury-architecture.md).

## Context

Jahid met his supervisor on 2026-08-12. The v1 architecture recorded in ADR [0002](./0002-three-stage-router-decomposer-jury-architecture.md) was a three-stage router → decomposer → jury pipeline, and several things v1 treated as live variables (masking scheme, retrieval scheme, pool size) had never been settled as decisions. The meeting settled them. This record states what Jahid and his supervisor decided; it adds nothing to it.

## Decision

**1. The jury stage is dropped.** The supervisor's reasoning: an LLM committee voting on decomposition quality is an engineering pattern, not a research contribution — and `CLAUDE.md` already forbids a closed commercial model scoring decomposition quality. The jury moves to Future Work.

**2. Dataset roles.**

- **MuSiQue** carries **decomposition-quality evaluation** and **end-to-end evaluation**, because it ships gold decompositions.
- **MetaQA** carries **end-to-end evaluation only**, through the supervisor's GRAG system, because it has no gold decompositions.

**3. The few-shot method is fixed:** bi-encoder retrieves **top-20**, cross-encoder re-ranks to **top-5**, with **typed entity masking**. Raw masking, uniform masking, and bi-encoder-only retrieval become **ablations**, not live variables.

**4. Pool size is fixed at 2000.**

**5. Contribution framing:** the contribution is **a strategy for constructing the few-shot pool and retrieving examples for hop-aware decomposition, validated empirically** — not "we decompose multi-hop questions".

**6. OPEN — whether the pipeline keeps the router.** To be decided by a guided-versus-unguided experiment. This ADR records it as open and does not resolve it.

## Consequences

- ADR 0002's third stage is no longer part of the pipeline; its router-guides-decomposer structure survives only insofar as item 6 resolves.
- Masking scheme, retrieval scheme, and pool size stop being swept as live variables and become fixed defaults with named ablations against them.
- MetaQA end-to-end evaluation depends on the supervisor's GRAG system, which is an external dependency.
- The router's fate — and therefore the ~600M router cap and the router code paths — stays unresolved until the guided-versus-unguided experiment runs.

## Alternatives considered

Recorded as decided by Jahid and his supervisor at the 2026-08-12 meeting; the meeting's reasoning for dropping the jury is stated under Decision item 1. No further alternatives are recorded, and an agent must not invent any.
