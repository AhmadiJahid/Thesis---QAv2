# 0027. A Number That Justifies a Config Constant Is a Committed Script, Not Prose

- **Status**: Accepted (implementer convention, pending Jahid) — specifically, **point 5 is the clause that needs Jahid's nod**: it narrows CLAUDE.md's "no experiment exists unless it is in the log" by carving out tools that score no system. Points 1–4 are implementer record-keeping and stand on their own.
- **Date**: 2026-08-23

## Context

`break_metrics.ged.max_nodes_for_optimizer` in `configs/musique_eval.json` is 16 because of a
timing measurement. For two revisions that measurement lived only as prose in
[ADR 0026](./0026-break-faithful-metrics-the-implementers-conventions.md): the prediction
"shapes" it timed were *named* ("repeated step text", "chain-shaped") but never defined, so
nobody could rebuild them. The PR #44 review could not reproduce one headline row; the PR #45
review then falsified the causal sentence that the (still shape-underspecified) table was used
to support. Both failures had the same cause — a number and a story about it, with no
executable definition behind either.

Committing the benchmark fixed both in one move, and it changed what could be said: the
regenerated table withdrew a claim rather than decorating it. That is the property worth
keeping, and it generalises past GED — the same shape of gap exists wherever a constant in
`configs/` is defended by a remembered measurement (a batch size, a retrieval `top_k`, a
timeout, a cluster count).

Nothing here is a research decision. This is an implementer convention about *records*, which
is why it is short.

## Decision

1. **A number that justifies a config constant ships as a script that prints it.** The script
   lives beside the pipeline (`scripts/`), reads its parameters from a committed config under
   `configs/`, defines its inputs *in code* (the exact strings, the exact construction — not a
   description of them), and prints the table in the form the record carries.
2. **The record carries the SHA the measurement was taken at**, plus enough machine context to
   explain a wall-clock number (host, CPU, the library version that does the work). One
   canonical copy: other files point at it and do not restate the numbers.
3. **What the script prints is what the record shows** — legends and markers included. A marker
   dropped on the way into prose makes a future cell uninterpretable.
4. **The prose says only what the table settles.** Where a causal reading is offered, the shape
   set has to be able to *separate* the factors it names; a factor that cannot be isolated is
   named as not isolated. When a measurement contradicts an earlier reading, the earlier reading
   is withdrawn in place, with what did and did not reproduce stated.
5. **A tool that scores no system is not an experiment.** It takes no `experiments/log.md`
   entry, no `experiments/<exp-id>/` trail and no GPU: the log is for runs that produce a metric
   for a system, and diluting it with tooling costs the log its usefulness. The tool is instead
   covered by tests that pin its *construction* (its numbers are machine-dependent and cannot
   be asserted), and it carries a tiny mode — the GED benchmark's `--max-node-count 8` — so the
   full path is smoke-testable before it earns real time.

## Consequences

- Justifying a constant now costs a script plus a test rather than a paragraph, which is the
  intended friction: a constant nobody can re-measure is a constant nobody can defend in
  November.
- The November write-up can regenerate its cost/parameter tables instead of trusting a quoted
  number, and the ADR trail says which numbers reproduced.
- Re-measurement can *lower* a claimed figure (it did: ~1.8 s → ~1.0 s at the cap) or delete a
  row. That is the convention working, not a regression.
- Timings stay machine-dependent. The convention makes them re-derivable, not portable, so a
  constant is set with an order of magnitude of headroom rather than at the edge of a timing.
- Existing constants defended only by prose are **not** retro-fitted by this record. It binds
  the next one, and an old one when it is next touched.

## Alternatives considered

- **Leave the measurement in prose but spell each input out.** Cheaper, and the PR #44 reviewer
  offered it as an option. Rejected: prose that fully specifies a construction is longer than
  the code that performs it, cannot be executed, and drifts from the implementation it claims
  to measure — which is exactly what happened twice here.
- **Make it an experiment with an `experiments/log.md` entry.** Rejected: it scores no system on
  no eval set, so the log's schema does not fit it, and the log is the one record that must stay
  scannable.
- **Commit the printed output as an artifact under `experiments/`.** Rejected for the same
  reason, and because a stale committed artifact is worse than a script anyone can re-run; the
  benchmark writes its raw numbers only when asked (`--json`), into an ignored path.
