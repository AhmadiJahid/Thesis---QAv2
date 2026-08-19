# 0014. The Guided-vs-Unguided Experiment Runs on the v1 Pool and Retrieval Artifact

- **Status**: Accepted (Jahid, 2026-08-19, in-session) — **pending supervisor confirmation**, because it deviates from a pool-size decision recorded to the supervisor (see below)
- **Date**: 2026-08-19

## Context

The guided-vs-unguided experiment (issue #12) needs a few-shot retrieval artifact. The only existing one, v1's `sim_dev_sample600_top20_rerankTop5.jsonl`, was provenance-traced on 2026-08-19 (recorded on issue #12): produced 2026-04-13 by the pinned method (bi-encoder `intfloat/e5-small-v2` top-20 → cross-encoder `cross-encoder/ms-marco-MiniLM-L-12-v2` top-5, typed masking available per mode, seed 42), query ids exactly the ADR 0007 pinned 600, zero pool/eval overlap — but over a **9,156-example pool** (sha256 `212c27634291b27ef55cebdb5feaba08a32a59f21976dd72a96a8f9f1ae66b2a`; hop mix 2-hop 3,594 / 3-hop 4,387 / 4-hop 1,175; 2-hop from train split 0 only, 3/4-hop from splits 0–3), not the 2,000 fixed by the supervisor on 2026-08-12 (ADR 0006 §4). The 2,000 pin postdates the artifact by four months, and no v2 "best 2,000 pool" exists yet (the #14 sweep has not run).

## Decision

Jahid decided in-session on 2026-08-19: **the pool is settled to what it was before — issue #12 runs on the v1 9,156-example pool via the existing v1 retrieval artifact, unchanged, to keep the experiments consistent** with the v1 work.

Jahid's clarification, same session: the 2,000-example figure originated from the ablation studies that were done, but the pool itself was never built properly — so proper pool construction is **deferred, to be worked on later** (issue #14 remains the open work item), and it does not block #12.

## Consequences

- `configs/decomposer_musique.json` gets `retrieval.input` set to the v1 artifact (content-addressed: the runner records its sha256, `e5c418a9b25f…`), unblocking the #12 run.
- **Deviation recorded, not hidden:** this contradicts ADR 0006 §4 (pool size 2,000, supervisor decision). It stands as Jahid's call pending supervisor confirmation; if the supervisor reasserts the 2,000 pin, that supersedes this ADR and #12 re-runs on a conforming artifact.
- Scope is #12's retrieval only. This does **not** resolve the "best pool" identity for the `pool_2000` fine-tuning arm (issue #13), and does not close the #14 pool-construction sweep — pool strategies remain a live research question.
- Thesis reporting must state the pool as 9,156 (v1), not 2,000, wherever #12's results appear.

## Alternatives considered (recorded from the 2026-08-19 issue #12 options, decided by Jahid)

- Run the #14 sweep first and pick a 2,000 pool with the supervisor — not taken (costs an experiment cycle before #12).
- Build a single 2,000 pool now by one strategy — not taken (the strategy choice would itself need the supervisor).
