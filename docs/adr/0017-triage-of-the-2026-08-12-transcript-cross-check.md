# 0017. Triage of the 2026-08-12 Transcript Cross-Check

- **Status**: Accepted (Jahid, 2026-08-20, in session)
- **Date**: 2026-08-20

Amends [0006](./0006-drop-the-jury-fix-dataset-roles-and-the-few-shot-method.md) (item 1 below softens its §3 "fixed" k values) and [0009](./0009-paired-bootstrap-and-mcnemar-as-the-significance-protocol.md) (item 4 below).

## Context

`docs/meetings/2026-08-12-supervisor-meeting-transcript-crosscheck.md` (PR #17) found a
handful of glosses and unrecorded items when the Fathom transcript was checked against ADRs
0006/0007/0008 and the post-meeting plan. Issue #19 held them for Jahid's eye. Jahid triaged
every bullet in session on 2026-08-20; this ADR records the decisions.

## Decision

1. **Retrieval top-20 / rerank top-5 are demoted to ablation status.** They are v1 practice,
   not a fixed method: runs continue to use them as the working values, but the exact k
   values are a knob to be ablated, and no claim may present them as method constants fixed
   by the supervisor.
2. **The reported length-cap default is 8 lines.** The supervisor's 6 was an illustrative
   example, not a decision; exp-002..005 configs already run with 8, and both values remain
   runnable via config.
3. **Fine-tuning sequencing is settled by events.** ADR 0010 prioritized fine-tuning at the
   supervisor's ask and exp-001 has trained; the transcript ambiguity is overtaken.
4. **A paired t-test is added alongside bootstrap CIs + McNemar** (issue #30). The
   supervisor's literal ask ("a t-test, a statistical test") gets covered rather than
   silently substituted; ADR 0009's protocol remains the headline. This amends ADR 0009.
5. **The supervisor's error-asymmetry judgment** (over-decomposition tolerable,
   under-decomposition not) **is recorded in `docs/METRICS.md`** next to the directional
   step-count metrics (this PR).
6. **The unrecorded meeting ideas all get durable homes**: few-shot-prompted router +
   with/without-router evaluation → issue #27; MetaQA+MuSiQue training-data mix → issue #28;
   composite-score literature/bias check → issue #29. (Router-as-regression had already
   landed as ADR 0010 / #23.)
7. **Scope notes, recorded here**: **RL is out of scope for the master's**; **failure cases
   are kept and reported in ablations**, not discarded.
8. **Still open on #19**: the pipeline figure the supervisor screenshotted (added when he
   sends it), and the GRAG handover, which Jahid chases with the supervisor himself
   (blocks the MetaQA half of #16).

## Consequences

#19 narrows to the two waiting-on-supervisor bullets. Comparison artifacts gain a t-test
(#30, Gate 1). Any future write-up describing top-20/top-5 must present them as ablatable
working values. Three new issues (#27–#29) enter the backlog. Note the k-ablation is not a
config flip on the #12 path: that path reads a pinned, content-addressed artifact built at
top-20/top-5 (ADR 0014), so ablating k means rebuilding the retrieval artifact.

## Alternatives considered

- Keeping top-20/top-5 as the fixed method — rejected by Jahid; the k values should be
  defensible by measurement, not inheritance.
- Treating "a statistical test" as generic license for the bootstrap+McNemar substitution —
  rejected; covering the literal ask is cheap and removes a reviewable discrepancy.
- Switching the length cap to 6 — rejected; it would fork comparability with the in-flight
  exp-004/005 for no measured reason.
