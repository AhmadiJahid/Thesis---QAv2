# 0035 — Results-section decisions, the decomposer size exception confirmed, and the v2 pool-size sweep

- **Status:** accepted (Jahid, 2026-10-14, in session; operational details decided by the lead under
  his delegation — *"I am leaving you should work on your own"* — and his to reverse).
- **Context:** the results-section redesign plan
  ([`docs/plan/2026-10-14-results-section-redesign.md`](../plan/2026-10-14-results-section-redesign.md))
  ended with five open questions; Jahid answered them the same day, and added a sixth fact.

## Decisions (Jahid's words in italics)

1. **Separate slides stay separate.** Pool construction, reranking and hop-matched retrieval keep
   their own slides — *"lets keep it not move into one slide"*.
2. **750-question slides stay in the section behind a badge** — Jahid deferred to the plan's
   recommendation (*"do as recommended"*). They are never charted against pinned-600 numbers as if
   comparable.
3. **The v1 prior-work slides (masking, pool size) are dropped** — *"lets drop it"*. Pool size is
   instead measured in v2 (item 7).
4. **The legacy composite leaves every "k of n" count** — *"done remove composite"*.
5. **A CPU-only re-analysis adds significance tests** for the per-hop and end-to-end numbers
   (*"Yes"*), logged as its own experiment before any slide claims from it.
6. **The decomposer's 15B exception is confirmed by the supervisor** — *"supervisor approved it"*.
   This closes the "pending supervisor confirmation" status that ADR 0031 item 4 and ADR 0032 item
   2 left open. `decomposer_max_params` stays 1.5e10; every other role stays at 8B.
7. **Pool size is swept in v2, extended to 10k** — *"slowly increasing the number of elements in the
   pool like 8k to 9k 10k"*. This widens issue #51 (2k–8k at 1k steps) to **2k–10k at 1k steps**.

## Shape of the sweep (lead, under delegation)

- **One factor moves:** pool size. Everything else is exp-020's: Mistral-7B-Instruct-v0.3, 4-bit,
  `unguided`, pinned 600 (ADR 0007), `imbalanced` construction (ADR 0031 item 1), e5-small-v2 top-20
  → cross-encoder rerank → top-5, seed 42. Reason: exp-020 is the cheapest #50 arm and is the 2k
  point already, if — and only if — the rebuilt 2k pool and retrieval artifact are byte-identical
  to exp-020's (sha256 checked); otherwise 2k is re-run too.
- **Imbalanced only.** Re-crossing construction strategies would multiply cells; exp-019 found no
  defect behind the "matching numbers" (issue #52) and a null construction effect, so #51's
  blocking question is settled.
- **Every pool is checked for zero overlap with the pinned 600**, and how the larger pools relate to
  the smaller ones (nested, or independent draws at the same seed) is recorded in the run note, since
  it decides whether size is confounded with a different draw.
- **Tests:** each size against 2k with the evaluator's battery (ADR 0009); no correction unless a
  slide states Holm.

## Amendment, 2026-10-14: one nested chain for the sweep's pools (lead, under delegation)

- **Finding (ml-engineer, measured).** `sample_pool.py --balance imbalanced` is
  `random.Random(42).sample(rows, size)` over the 19,938 enriched train rows. Under Python 3.13,
  `random.sample` switches algorithm at size ≥ 5,462 of 19,938: set-based below that size,
  pool-swap from it upward. Each method is prefix-consistent, so the builder gives two nested
  chains, 2k ⊂ … ⊂ 5k and 6k ⊂ … ⊂ 10k. 5k and 6k share only 3,553 ids, so the 5k→6k step
  would have changed the draw as well as the size.
- **Decision.** Use one nested chain, anchored on the adopted 2k pool (`cc364662…`, ADR 0028/0031).
  2k–5k are kept as `sample_pool.py` builds them, each an exact prefix of the next. 6k–10k are
  the 5k pool plus a prefix of the remaining 14,938 rows, sorted by id and shuffled once with
  seed 42 (`scripts/extend_pool_nested.py`). There is no hop stratification, so proportions are
  natural. Size is then the only moving factor, and the 2k point is the pool every #50 arm used.
- **2k gate.** The rebuilt 2k pool is byte-identical. The retrieval artifacts are not: the
  pinned-600 query files were rewritten on 2026-09-14 (same ids, different order), so their
  sha256 differs from exp-014's by row order only. Per the rule above, 2k is re-run, and the
  2k-re-run vs exp-020 comparison serves as the reproducibility check. Hashes and commands:
  `configs/pool_size_sweep_v2.json`.
