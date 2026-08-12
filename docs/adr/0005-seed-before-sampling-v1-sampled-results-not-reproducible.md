# 0005. Seed Before Sampling; v1 Sampled-Subset Results Are Not Reproducible

- **Status**: Accepted
- **Date**: 2026-08-12

## Context

During the v1→v2 port (PR #7, issue #3), the ml-engineer found and the independent review confirmed a v1 reproducibility bug: v1's `router.py` called `random.sample` on its evaluation questions (v1 lines 180–183) **before** calling `set_seed` (v1 line 217). Any v1 router result produced with `--sample_size` therefore used an unseeded draw — the sampled subset was never reproducible, in v1 or anywhere else. v2's operating contract (CLAUDE.md, Reproducibility) requires fixed seeds threaded through all randomness, and ADR [0001](./0001-v1-to-v2-migration-scope-and-method.md) governs how v1 results may be compared against.

## Decision

v2 seeds all randomness before any sampling: every entry point calls `set_global_seed` at run start, before evaluation questions are drawn (`components/router/run_router.py` seeds at run start and samples after). We do not replicate v1's ordering for parity.

## Consequences

- v1 router results produced with `--sample_size` cannot be exactly reproduced by v2 (nor by v1 itself — the draws were unseeded by construction). Comparisons against those numbers are approximate at the subset level; full-set v1 results are unaffected by this bug.
- Any v2-vs-v1 comparison on a sampled subset must re-run the baseline in v2 (which ADR 0001 and `docs/prior-work.md` already require for all comparison claims).
- `docs/prior-work.md` §7 carries the pointer so a reader of v1 numbers finds this.

## Alternatives considered

- **Replicate v1's sample-then-seed ordering for parity.** Rejected: it reproduces nothing (the draw was unseeded, so there is no fixed subset to be faithful to) and violates v2's seeding rule.
