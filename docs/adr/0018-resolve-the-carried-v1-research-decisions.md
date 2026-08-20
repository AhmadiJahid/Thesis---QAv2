# 0018. Resolve the Carried v1 Research Decisions

- **Status**: Accepted (Jahid, 2026-08-20, in session; item 2 pending supervisor confirmation)
- **Date**: 2026-08-20

Amends [0001](./0001-v1-to-v2-migration-scope-and-method.md) (item 1 below upgrades its dataset framing) and [0006](./0006-drop-the-jury-fix-dataset-roles-and-the-few-shot-method.md) (the typed-masking clause of its §3 is reopened below).

## Context

Issue #6 carried seven research decisions implied by v1 but never explicitly decided or
recorded. Jahid resolved the resolvable ones in session on 2026-08-20; this is the single
retroactive-free ADR that issue called for.

## Decision

1. **MetaQA + MuSiQue are the thesis datasets** — confirmed as a research decision, not
   merely migration scope (upgrades ADR 0001's framing; roles per ADR 0006).
2. **`intfloat/e5-small-v2` is the working default embedding model** (the repo alias
   `e5-small`, `configs/similarity.json`) for bi-encoder similarity retrieval, pending
   supervisor confirmation. Every retrieval artifact so far (incl. ADR 0014's) was built
   with it; swapping later is an ADR + artifact rebuild. This decides the **bi-encoder slot
   only**: the cross-encoder re-ranker slot (currently
   `cross-encoder/ms-marco-MiniLM-L-12-v2` in configs) remains a working value with no
   recorded decision behind it.
3. **The router-default question folds into #23.** The old classifier framing
   (Qwen2.5-3B few-shot + masked-similarity) is moot under ADR 0010's regressor reframing;
   model choice is part of the #23 design, sequenced after #12's numbers.
4. **Overtaken items, closed**: guided vs unguided prompting is being *measured* by #12
   (exp-004/005) rather than decided a priori; the MuSiQue eval-set construction was fixed
   by ADR 0007.

**Remaining open on #6** (not decided here): the decomposer retrieval masking default —
Jahid marked typed masking **not settled**, to be raised with the supervisor *(pool size
stays fixed at 2000 by ADR 0006 §4, with ADR 0014's recorded 9,156-pool deviation pending
confirmation)* — the thesis-primary metric, explicitly deferred to a supervisor meeting
(fed by #29), and the cross-encoder re-ranker slot noted in item 2.

**Deviation recorded, not hidden:** the 2026-08-12 cross-check records typed masking as
supervisor-confirmed ([1:24:27]); marking it "not settled" here is Jahid's 2026-08-20 call
and a departure from that record. If the supervisor reasserts typed masking as fixed, that
supersedes this reopening.

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
