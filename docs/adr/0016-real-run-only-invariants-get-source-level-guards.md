# 0016. Real-Run-Only Invariants Get Source-Level Guards

- **Status**: Accepted
- **Date**: 2026-08-19

## Context

On 2026-08-19 all six exp-002/exp-003 generation runs (issue #12) crashed at their first
logged query with `KeyError: 'raw'`: the per-query dump consumed `gen["raw"]` while
`generate()` returns the raw text under `"text"`. The defect survived **261 harness checks,
33 smoke stages, and six dry-run preflights**, because every one of them runs `--dry-run` —
and a dry run sets `gen = None`, which short-circuits every `gen[...]` access. The entire
class of defect (a consumer asking the generation result for a key it does not have) is
structurally unreachable by any test that does not perform real generation, and real
generation needs a loaded model, which the test suites deliberately never do.

The failed runs cost the full launch (six arms recorded as `rc=1` in
`runs/exp-002-003-combined.log`) and a relaunch. The same blind spot applies to any future
field added to the generation result or its consumers, including the cost fields.

## Decision

An invariant that **only a real (non-dry) run can violate** gets a **source-level guard**: a
test that parses the source (AST) and asserts the invariant statically, so it fails with no
model loaded. Two rules bind such guards:

1. **The guard ships with negative controls** that re-break the source in memory and assert
   the guard catches exactly the expected violation — and the controls assert their
   replace-target actually matched, so a moved line fails loudly instead of passing
   vacuously.
2. **Dynamic consumption gets made visible.** Where a consumer reads keys through a variable
   (the `cost = {k: gen[k] for k in cost}` copy), the guard checks the literal that seeds it
   rather than ignoring the site.

First instances: `tests/test_generation_contract.py` (every `gen[...]` key consumed in
`run_decomposer.py`, plus the seeded cost keys, must be in every literal return dict of
`generate()`; PR #26) and `tests/test_suite_collectability.py` (no module-level
`def test_*(arg)` in a script-style suite, the pytest-collection regression; PR #25).

## Consequences

- The next key-contract mismatch is caught at commit time, not at launch time on the GPU.
- The limit is stated honestly: a static guard checks the source shape it was written for.
  If the guarded code is restructured (e.g. the cost copy iterates a different container),
  the guard must move with it — the negative controls are what make that move loud.
- This does not substitute for a real-generation smoke stage; it narrows the gap dry runs
  leave. Whether a tiny real-generation stage is worth its cost on the shared GPU remains
  open.

## Alternatives considered

- **A real-generation smoke stage** (tiny model, few rows). Not chosen as the immediate fix:
  it needs GPU or slow CPU time on a box whose GPU is contended and mid-experiment, and it
  would still only cover the paths that stage happens to execute.
- **Monkeypatched end-to-end test** (stub `load_model`/`generate`, run `main()` for real).
  More faithful but far heavier to build and maintain than the AST guard; may still be worth
  adding later — this ADR does not preclude it.
- **Leaving it to review.** This class of error was caught by review process twice in one
  day (the pytest-collection shape) and by a crashed launch once (the key mismatch); a
  mechanical check is strictly cheaper.
