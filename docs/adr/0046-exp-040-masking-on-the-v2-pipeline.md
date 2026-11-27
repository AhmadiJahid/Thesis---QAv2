# 0046 — exp-040: query masking re-run on the v2 pipeline, the rebuilt artifact, and the typed comparator

- **Status:** accepted. The lead decided on 2026-11-27, under Jahid's grant of full authority over
  technical choices. The items are the implementer's conventions under that brief, and Jahid may
  reverse them.
- **Date:** 2026-11-27.
- **Context:** The thesis states that typed masking beats uniform masking, which beats raw. That
  ordering comes from v1 (ADR 0001 prior work) and was never re-run in v2. None of its 15 tests
  survives Holm (`docs/analysis/2026-08-20-v1-masking-and-retrieval-significance.json`;
  writing audit, item M1 and section (d), rank 1). Every v2 arm since exp-014 retrieves exemplars
  in typed mode (ADR 0006 item 4). Whether masking is the right default was carried as an open
  supervisor question (ADR 0018). exp-040 measures the three modes under the current fixed method.

## Decision

1. **One factor moves: `--retrieval-mode` (raw / uniform / typed).** Everything else is exp-020's
   command: Mistral-7B 4-bit unguided, the size2000_imbalanced pool, e5-small-v2 top-20 followed by
   the cross-encoder top-5, the pinned 600, greedy decoding with seed 42. Masking only selects which
   exemplars are retrieved. The prompt shows raw text in every arm. No pipeline code changes:
   `--retrieval-mode` and the `<mode>_top_k` artifact schema already exist.
2. **The raw and uniform lists require a rebuilt artifact.** exp-014 built with `--mode typed`, so
   its file holds only `typed_top_k`. The notebook runs exp-014's own commands with `--mode all`
   and `--no-cache` on CPU. The cache is skipped so the build depends only on the pinned pool and
   not on a cache keyed by another machine's path. The artifact's sha256 is pinned on Drive at
   the first build, and every later step refuses any other file.
3. **The typed arm is a same-machine re-run of exp-020** (ADR 0043 item 7 convention), and the
   stored exp-020 serves as the hardware control. The rule below is declared before the run:
   - **Rebuild reproduces exp-014's typed exemplars for 600/600 queries** (ordered pool ids): the
     re-run is the typed arm. Its prompts are byte-identical to typed mode on the rebuilt
     artifact.
   - **Otherwise:** `typed_rebuilt` runs, and it becomes the typed side of the primary pairs, so
     all three arms share one build.
4. **Primary contrasts and Holm families, fixed before any run:** typed − uniform and typed − raw,
   scored on the 7 report-card terms.
   - F1 is the overall family (14 tests).
   - F2 is the per-hop family (42 tests).
   - uniform − raw and the hardware control are reported uncorrected and belong to no family.
   - No direction is pre-declared. A claim that masking helps is made only for cells that survive
     Holm.
5. **Canary and output health.** Before each full decode, a 20-question canary runs. The same
   gate then checks the full 600. Its thresholds (≤ 1% empty, ≤ 5% at `max_new_tokens`) are
   arbitrary, set to catch a broken run, and stated as arbitrary in `config.json`.

## Consequences

- If F1/F2 support it, the thesis can state the masking result as a v2 measurement. If not, it
  says the v1 ordering is not reproduced or not distinguishable (ADR 0036 item 4). The v1 numbers
  remain prior work either way.
- The rebuilt artifact is a new input. Its sha256 is recorded in `compare_metrics.json` and in
  every arm's run snapshot.

## Alternatives considered

- **Reuse the stored exp-020 as the typed arm.** Rejected: GPU and library differences could
  then pass for a masking effect (exp-037's finding, ADR 0043 item 7).
- **Run the typed arm on the rebuilt artifact unconditionally.** Rejected as the default because
  it costs one more decode. When the rebuild reproduces exp-014's typed exemplars, the two typed
  runs see identical prompts. The rule in item 3 covers the case where it does not.
- **Rebuild on GPU for speed.** Rejected: exp-014 ran on CPU, and a different device could reorder
  near-ties, which would undermine the identity check.
