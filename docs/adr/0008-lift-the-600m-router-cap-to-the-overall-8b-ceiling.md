# 0008 — Lift the 600M router cap to the overall 8B ceiling

- **Status:** accepted
- **Date:** 2026-08-17
- **Decider:** Jahid Ahmadi (human decision, stated in-session)

## Context

`CLAUDE.md`'s standing constraints recorded a router-specific parameter cap of
~600M, alongside the ~8B overall cap. Provenance of the 600M figure: it entered
the repo in commit `1e3957a` ("chore: Codery system render for the thesis
work", the 2026-08-11 setup session with Tural), where it was attributed to
Jahid's supervisor and professor. **No independent record of the supervisor
stating a 600M router cap exists** — every later occurrence
(`configs/model_limits.json`, `src/model_size.py`, ADR 0004, `docs/prior-work.md`)
copies that render.

The cap conflicted with the evidence (issue #10): the best v1 router result was
Qwen2.5-3B-Instruct (~5× the ceiling), and nine of the ten router model folders
ported into v2 would be refused by the load-time assertion — the router numbers
in Jahid's slides could not be reproduced in v2 as configured.

## Decision

Asked where the 600M figure came from, Jahid stated on 2026-08-17 that he is
not bound by a 600M router cap and instructed that it be lifted. The router is
now bound only by the overall ~8B ceiling: `router_max_params` in
`configs/model_limits.json` is set to `8000000000`. The key is kept (rather
than deleted) so `src/model_size.py` still asserts a ceiling at every router
load.

This is issue #10's option (b) — raise the cap with an ADR recording why —
chosen by Jahid directly, which also answers that issue.

## Consequences

- The v1 router models (Qwen2.5-1.5B/3B/7B, Phi variants, Mistral-7B) are
  loadable in v2 again; the v1 router results are reproducible as configured.
- Whether the pipeline keeps a router at all remains **OPEN** per ADR 0006,
  decided by the guided-versus-unguided experiment (issue #12). This ADR only
  removes the size conflict; it does not decide the router's fate.
- The ~8B overall cap stands untouched.

## Alternatives considered

- **Re-run the router at 0.5B** (issue #10 option a): rejected by the decision
  above — it would trade the slide results for a cap with no substantiated
  origin.
- **Drop the router now** (option c): premature; that is what issue #12's
  experiment decides.
- **Caveat:** if the supervisor did in fact set a router cap and reasserts it,
  his decision supersedes this record.
