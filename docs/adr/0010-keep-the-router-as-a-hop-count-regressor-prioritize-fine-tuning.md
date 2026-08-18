# 0010. Keep the Router, as a Hop-Count Regressor; Prioritize Fine-Tuning

- **Status**: Accepted
- **Date**: 2026-08-18

Amends [0006](./0006-drop-the-jury-fix-dataset-roles-and-the-few-shot-method.md) (resolves its open item 6).

## Context

ADR 0006 item 6 left open whether the pipeline keeps the router, to be decided by a guided-versus-unguided experiment. On 2026-08-18 Jahid relayed two decisions from his supervisor. This record states them; it adds nothing to them.

## Decision

**1. The router stays in the pipeline.** ADR 0006's open item 6 is resolved: the router is kept and gets built. The guided-versus-unguided experiment (issue #12) still runs, but its role changes from *deciding whether the router exists* to *informing its design* — how much a correct hop count actually helps the decomposer, and whether runaway generation length rather than hop ignorance is the dominant failure.

**2. The router is reframed as a regressor, not a classifier.** In Jahid's words: v1 asked the router to tell us how many hops are in the question — "is it 1, 2, 3" — which is classifying over fixed classes. The supervisor wants to see what happens if the model is instead asked to *count* the number of hops, as regression. Design specifics — target encoding, loss, rounding a continuous prediction back to an integer hop count, and the evaluation metric for the regressor — were **not** specified in this decision. They are open engineering questions to bring back to the supervisor, not details for an agent to invent.

**3. Fine-tuning is a focus area.** The LoRA fine-tuning comparison arm for the decomposer (issue #13) is prioritized to start immediately, in parallel with the guided-versus-unguided work, rather than queuing behind it. (Fine-tuning as a comparison arm was already allowed and expected per `CLAUDE.md`; what changes is priority.)

## Consequences

- Issue #12 is reframed (title and body updated in the same pass as this ADR): it informs router design instead of deciding keep/drop.
- A router-regressor work item exists (issue filed alongside this ADR) and inherits the 8B overall ceiling of ADR [0008](./0008-lift-the-600m-router-cap-to-the-overall-8b-ceiling.md); there is no router-specific parameter cap.
- Hop-matched retrieval (issue #15) regime 3 (`router_hop_matched`) becomes a real pipeline configuration rather than a contingent one.
- The regressor's open design questions must be answered by Jahid and his supervisor before the regressor's evaluation protocol is fixed.
- The GPU run queue now has two prioritized consumers (#12's three conditions and #13's multi-day training); the run lock in `docs/compute.md` serializes them.

## Alternatives considered

Recorded as decided by Jahid's supervisor and relayed by Jahid on 2026-08-18. No alternatives are recorded, and an agent must not invent any.
