# 0001. V1 to V2 Migration Scope and Method

- **Status**: Accepted
- **Date**: 2026-08-12

## Context

The predecessor repo (`/cta/users/fyilmaz/Thesis---QA`, "v1") holds working tooling for two datasets (MetaQA and MuSiQue), a body of measured results (`handoff/results_analysis/FINDINGS.md`), and documented decisions (`docs/DECISIONS.md`, `docs/MASKING.md`, `docs/MODEL_SELECTION.md`). This repo ("v2") starts fresh under a stricter operating contract — committed configs, fixed seeds, an append-only experiment log, review gates. The question was what carries over, and how: v1's code does not meet v2's reproducibility rules as-is, and v1's numbers were not produced under v2's logging discipline. Compute for v2 is not yet settled (issue #2).

## Decision

Decided by Jahid, in session, 2026-08-12:

1. **Both datasets migrate.** MuSiQue and MetaQA tooling both come over from v1 — not MuSiQue alone.
2. **Code is ported selectively and adapted**, piece by piece, to v2's reproducibility rules (config snapshots, fixed seeds, metrics JSON, log entries) — neither copied wholesale nor rewritten from scratch. This is issue #3.
3. **v1 knowledge is carried as docs now** — retroactive ADRs plus a prior-work note (issue #4). Re-running v1 baselines under v2 is **deferred** until compute is settled (issue #2), and which baselines to re-run is Jahid's decision at that point.

## Consequences

- v2 gets the decision trail and results context immediately, without waiting on compute.
- v1 numbers are citable only as prior work (`docs/prior-work.md`); no v2 claim may present them as v2 measurements, and any v2 comparison baseline must first be re-run in v2.
- The port (issue #3) must adapt each piece to v2's rules as it lands, which is slower than a bulk copy but keeps `main` compliant with the operating contract.
- A later decision point remains open: which v1 baselines to re-run once compute settles.

## Alternatives considered

- **MuSiQue-only scope** — rejected: MetaQA tooling (KG execution, masking, router experiments) carries results and machinery the thesis draws on.
- **Wholesale copy of the v1 repo** — rejected: imports code and artifacts that do not meet v2's reproducibility rules, and buries the useful pieces in noise.
- **Full rewrite** — rejected: discards working, already-debugged tooling against a fixed calendar (experiments end October 2026).
- **Re-run everything before citing anything** — rejected: blocks all documentation work on unsettled compute; carrying knowledge as clearly-labeled prior work is available now.
