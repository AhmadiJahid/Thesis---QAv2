# 0003. Mask Queries Only, Never Re-Mask the Few-Shot Pool

- **Status**: Accepted
- **Date**: captured retroactively 2026-08-12

## Context

Recorded in v1 at `Thesis---QA/docs/MASKING.md` (sections "Purpose" and "Integration"). Entity masking replaces movie titles and person names in questions with `[MOVIE]` / `[PERSON]` placeholders so that similarity retrieval (e.g. router few-shot matching) matches on question **structure** rather than being inflated by entity overlap. The masker is an Aho-Corasick automaton built from the MetaQA KB with longest-match and word-boundary handling. Applying that masker to the few-shot pool at runtime can corrupt pool items through entity overlap: v1 documents the example `"who stars in Baby Face"` → `"who [MOVIE] in [MOVIE]"`, because "Stars" is itself a KB movie title.

## Decision

We use the **pre-masked pool file** when available (in v1, `Pool/few_shot_router_masked.json`) and apply the mask function **only to queries**. We never dynamically re-mask the few-shot pool at runtime.

## Consequences

- The masked pool is a maintained artifact: regenerating it is a deliberate, checkable step, not a runtime side effect.
- Query-side masking stays cheap and safe (a query corrupted by masking affects one lookup, not the shared pool).
- Any v2 port of the masking/similarity tooling (issue #3) must preserve this split: pool pre-masked, queries masked at lookup time.

## Alternatives considered

- **Dynamically masking the pool at runtime** — rejected in v1 because KB entity overlap can corrupt pool items (the "Baby Face"/"Stars" example above), silently degrading the retrieval pool.
