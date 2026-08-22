# 0023 — Jahid's 2026-08-22 direction: composite replacement, pipeline completion order, generalisation framing

- **Status:** accepted (Jahid, 2026-08-22, in session). Items marked *(supervisor)* still need his supervisor's confirmation before they harden into the thesis.
- **Context:** exp-004/005 (issue #12), exp-008 (issue #13), exp-009 (issue #15) all have committed metrics; the same-day error analysis (`docs/analysis/2026-08-22-finetuned-vs-prompting-error-analysis.md`, issue #40) showed the composite score's reference-validity term is decided by 2/600 items via a regex/data syntax mismatch.

## Decisions (Jahid's words, at his altitude — not extended here)

1. **Work on the composite score starts now: "bring something better from the literature."** A literature-grounded replacement (or repair) for the decomposition-quality composite is authorized work, not just an open question. The standing constraint is unchanged: no closed commercial model may score, rate, or judge decomposition quality. Whether the chosen metric becomes the thesis's primary metric is *(supervisor)* — issue #6 item 5.
2. **The most important goal is finishing the decomposition pipeline**, in this order:
   1. **Better pool first** — the pool-construction sweep's GPU stage (issue #14, running as exp-010 at the time of this ADR) produces the evidence; the best-performing construction is adopted for downstream runs *(supervisor confirmation pending, recorded when the numbers exist)*.
   2. **Then the router decision** — use the router or omit it and decompose anyway. Issue #27's with/without-router evaluation is the instrument. The decision itself remains Jahid + supervisor; the system produces the numbers.
   3. **Then compare decompositions with the trained Mistral** — the dynamic-prompting method at its best configuration (best pool) against the fine-tuned decomposer (exp-001 adapter, evaluated as exp-008), same pinned eval set.
3. **Generalisation framing:** compare the MuSiQue-trained model and the dynamic-prompting system beyond the data the model was trained for. Jahid's stated hypothesis: dynamic prompting may generalise where the fine-tuned model "should be trained on the other dataset to do better" — i.e. fine-tuning is dataset-bound, prompting is not. This is a hypothesis to measure, not a conclusion. Instruments available: the already-designed `generalisation_2_3hop` fine-tuning arm (train 2/3-hop, eval 4-hop, within-MuSiQue), and cross-dataset end-to-end evaluation on MetaQA (issue #16; the MetaQA half is blocked on the supervisor's GRAG system, an external dependency Jahid chases).
4. **Documented issues and results discussion continue later** — the tracking surfaces stay authoritative in the meantime.

## Consequences

- An analyst lane surveys the literature for decomposition-quality metrics and returns a shortlist with trade-offs; the implementation (Gate 1 reviewed) follows the shortlist, and existing per-item outputs are re-scored (cheap, CPU) rather than re-run.
- A router-readiness lane makes the router emit query-id-keyed predictions (ADR 0022 recorded that today's router writes none), enabling issue #27's with/without-router comparison and issue #15's third regime (router-hop-matched retrieval; retrieval already accepts a `predictions` hop source since PR #39).
- Issue #16's MuSiQue answering backend proceeds; the MetaQA/GRAG dependency stays flagged, not stubbed.
