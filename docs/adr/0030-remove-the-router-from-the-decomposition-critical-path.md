# 0030. Remove the Router from the Decomposition Pipeline's Critical Path

- **Status**: Accepted (lead's call, on the delegation in [ADR 0028](./0028-jahid-2026-08-23-delegations-pool-choice-router-call-composite-authorized.md) item 2 — Jahid, 2026-08-23: "you decide if it stays or not"). *(supervisor)* may overturn from this record.
- **Date**: 2026-08-23

Amends [0028](./0028-jahid-2026-08-23-delegations-pool-choice-router-call-composite-authorized.md) (discharges its item 2) and [0010](./0010-keep-the-router-as-a-hop-count-regressor-prioritize-fine-tuning.md) (which kept the router as a hop-count regressor). Bears on [0002](./0002-three-stage-router-decomposer-jury-architecture.md)'s three-stage architecture. Evidence: **exp-012** (`experiments/log.md`, artifacts `experiments/exp-012/`, commit `064665e`) and **exp-014** (commit `222862f`). Issues [#27](https://github.com/AhmadiJahid/Thesis---QAv2/issues/27), [#23](https://github.com/AhmadiJahid/Thesis---QAv2/issues/23), [#46](https://github.com/AhmadiJahid/Thesis---QAv2/issues/46).

*(Numbering note: 0029 is reserved for the decomposition-metric-suite ADR being written on `feature/40-decomposition-metric-suite`, which was in flight when this record was committed to `main`. The gap closes when that PR merges.)*

## Context

Issue #27 asked whether the router earns its place. exp-012 ran the instrument on the ADR 0007 pinned 600: a few-shot-prompted router (Mistral-7B), a `router_guided` decomposer arm consuming its predictions, and paired comparisons against both the without-router (`unguided`) and perfect-information (`oracle_guided`) arms, all re-scored at one evaluator commit.

## What was measured

**Router accuracy: 0.3833 overall (230/600)** against a ~0.333 three-class chance baseline; per hop 0.875 / 0.175 / 0.100 at 2 / 3 / 4.

**With-router vs without-router (n=600): non-significant on every metric.** All seven bootstrap intervals cross zero and all three McNemar tests are null, including `hop_count_exact_match` — the metric the router exists to serve — at 0.5083 unguided vs 0.4917 router-guided (p=0.500).

**Perfect hop information vs the real router (n=600):** `hop_count_exact_match` significant favouring the oracle (0.5900 vs 0.4917, McNemar p=9.69e-06) and `ged` significant favouring the oracle (0.4414 vs 0.4721, CI [-0.0440, -0.0177]); `step_f1`, `ordered_step_accuracy`, `sari` and `chain_validity` all non-significant.

## The null is confounded, and the decision does not rest on it

**394 of 600 router responses failed to parse and were scored at `default_hop=2`.** Hop-2's apparent 0.875 is therefore mostly free credit: 123 of its 175 "correct" rows were defaulted rather than parsed, and only 52 of 77 *parsed* hop-2 responses were correct. On parsed responses alone the router is right about 107/206 ≈ 52% of the time — better than chance, on barely a third of the eval set.

This matters for what may be concluded. The with/without null describes a component that emitted "2" for two thirds of its inputs, **not** a working router, so it is weak evidence about a repaired one. A router that parsed reliably at ~52% accuracy might well move `hop_count_exact_match` where this one did not. **The null is therefore recorded and not relied upon.**

## Decision

**The router leaves the decomposition pipeline's critical path.** The justification is the *ceiling*, which is implementation-independent:

1. **Perfect hop information is worth little.** An oracle — the upper bound on any router, at any accuracy, ever — buys **+0.082 `hop_count_exact_match`** (0.5900 vs 0.5083) and a GED improvement, and moves **nothing** in step content: `step_f1` 0.2061 vs 0.2039, with `ordered_step_accuracy`, `sari` and `chain_validity` all non-significant. Whatever a better router recovers, it recovers a fraction of that.
2. **Fine-tuning dominates the router's own metric.** exp-014, same pinned 600, n=600: fine-tuning beats prompting on **every** axis — `step_f1` 0.2039→0.3411, `ordered_step_accuracy` 0.1804→0.3237, `hop_count_exact_match` 0.5083→0.7100 (p=1.12e-15), plus all four Break-faithful columns. The router's entire value proposition is a metric fine-tuning already improves by 0.202 — two and a half times the oracle's whole ceiling.
3. So the best case for a repaired router is a fraction of +0.082 on one axis, bought with an extra model, an extra failure mode and extra inference cost, inside a pipeline where a different intervention moves everything at once.

## Scope — what this does not say

- It does **not** say hop-count routing is scientifically uninteresting, nor that issue #23's regressor design is refuted. That design is untested here; it is simply bounded by the same ceiling, which is what makes it a poor use of the remaining calendar.
- It does **not** retire the router code, the `router_guided` arm, or ADR 0024's instrumentation. They stay, committed and working: exp-012 is a reportable negative result and must remain reproducible, and issue #15's router-hop-matched retrieval regime still reads router predictions.
- It does **not** touch the oracle-guided arm, which remains the upper-bound reference in the guided-vs-unguided comparison (issue #12).
- The 34% parse rate is a **defect**, recorded separately (issue #47). Fixing it is the precondition for any future router work; this ADR does not schedule that work.

## Consequences

- Issue #27 is answered: on this evidence the router does not earn its place, and the thesis reports it as a measured negative result with the parse-rate caveat stated — not as "routers do not work".
- Issue #46's sequence drops its router step; effort moves to the fine-tuning and generalisation arms (exp-013, exp-015), which is where the measured gains are.
- ADR 0010's "keep the router, as a hop-count regressor" is superseded for the critical path. ADR 0002's three-stage architecture becomes two-stage in practice, the jury having already gone in ADR 0006.
- If the supervisor reasserts the router, the first task is issue #47 (parse rate), then a re-run of exp-012's comparison — because the null on record cannot carry a "router does not help" claim on its own.
