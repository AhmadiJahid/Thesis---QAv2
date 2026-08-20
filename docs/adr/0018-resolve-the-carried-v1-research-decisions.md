# 0018. Resolve the Carried v1 Research Decisions

- **Status**: Accepted (Jahid, 2026-08-20, in session; item 2 pending supervisor confirmation)
- **Date**: 2026-08-20

## Context

Issue #6 carried seven research decisions implied by v1 but never explicitly decided or
recorded. Jahid resolved the resolvable ones in session on 2026-08-20; this is the single
retroactive-free ADR that issue called for.

## Decision

1. **MetaQA + MuSiQue are the thesis datasets** — confirmed as a research decision, not
   merely migration scope (upgrades ADR 0001's framing; roles per ADR 0006).
2. **e5-small is the working default embedding model** for similarity retrieval, pending
   supervisor confirmation. Every retrieval artifact so far (incl. ADR 0014's) was built
   with it; swapping later is an ADR + artifact rebuild.
3. **The router-default question folds into #23.** The old classifier framing
   (Qwen2.5-3B few-shot + masked-similarity) is moot under ADR 0010's regressor reframing;
   model choice is part of the #23 design, sequenced after #12's numbers.
4. **Overtaken items, closed**: guided vs unguided prompting is being *measured* by #12
   (exp-004/005) rather than decided a priori; the MuSiQue eval-set construction was fixed
   by ADR 0007.

**Remaining open on #6** (not decided here): the decomposer retrieval masking default —
Jahid marked typed masking **not settled**, to be raised with the supervisor — and the
thesis-primary metric, explicitly deferred to a supervisor meeting (fed by #29).

## Consequences

#6 narrows to two supervisor-facing items. The dataset pair and embedding model are now
citable decisions for the November write-up. If the supervisor rejects e5-small, retrieval
artifacts rebuild under a new ADR.

## Alternatives considered

- Deciding the primary metric now — declined; it is explicitly on the supervisor's agenda
  ([31:59] in the 2026-08-12 transcript).
- Recording typed masking as the default — declined by Jahid; the masking strategy is
  genuinely still open with the supervisor (ADR 0003 continues to govern *how* masking is
  applied wherever it is used).
