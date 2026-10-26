# 0032 — Jahid's 2026-09-20 go-ahead for the issue #50 block, and the defaults taken for what he left open

- **Status:** accepted (Jahid, 2026-09-20, in session).
- **Context:** [`docs/plan/2026-09-20-next-step.md`](../plan/2026-09-20-next-step.md) §4 listed six
  decisions gating the block Ahmmad assigned on 2026-08-27 (issue #50). Jahid answered with
  *"I agree with the most of the stuff. The reversed prompt is gonna run as well right? If yes,
  Start building and testing… The limit threshold for the parameter size is also increased."*
  This ADR records what that settles and which items are defaulted by the lead under the
  standing delegation pattern (ADR 0028, ADR 0031 item 2).

## Settled by Jahid

1. **Go.** The 14B download, the three build lanes (model folders + ceiling, anchor set, ordering
   knob) and the GPU lane start 2026-09-20. The GPU was verified idle by Jahid and by `nvidia-smi`.
2. **The size ceiling raise to 1.5e10 is confirmed by Jahid**, not only pending — *"The limit
   threshold for the parameter size is also increased."* ADR 0031 item 4's "pending supervisor
   confirmation" flag stays on the record as history; the constraint in `CLAUDE.md` still names
   the supervisor as the authority who can reassert a lower ceiling.
3. **The reversed few-shot ordering runs** in this block (A5, G3). Jahid asked for it explicitly.
4. **Agents run on Opus only** for this block — an operational constraint from Jahid on cost.
   The `experiment-runner` role, defined on Sonnet, is spawned with an Opus override.
   *(Amended 2026-10-26 by [ADR 0040](./0040-harness-conventions-opus-only-agents-and-id-reservation.md) item 1: the rule is standing, not scoped to this block; `experiment-runner` is now defined on Opus.)*

## Defaulted by the lead (Jahid did not answer; each is reversible)

5. **"Complexity" for exemplar ordering stays undefined.** The knob ships `as_is` and `reversed`
   only. `complexity_asc` is not implemented until Jahid or Ahmmad names the key — hop count,
   question length or gold-chain length were the candidates. Picking one silently would be
   inventing methodology.
6. **Qwen3.5-9B is not in this block.** It is the confounded fourth point (ADR 0031 item 5); the
   default was stated as "not in this block" and not contradicted.
7. **The stop condition is adopted:** if the three G1 arms come back near-identical on the
   report card, G3 does not launch until the lead has reported that to Jahid.
8. **G2's precision follows exp-008** (`quantization: none`) so the pool-only vs full-train
   contrast moves one factor. The G1 arms run 4-bit (ADR 0031 amendment). The two are on
   different precision axes and every write-up says so.

## Consequences

- Next free experiment id is exp-020 (G1a, Mistral 4-bit). Runs proceed one at a time under
  `runs/run.lock`; builders may share the GPU only for ≤5-row smoke tests and never take the lock.
- `configs/model_limits.json` carries the raise as `decomposer_max_params: 1.5e10` while
  `default_max_params` returns to 8e9 (ADR 0031 amendment of 2026-09-20), with this ADR cited in
  its note.
- Commits stay local: this box's `gh` credential cannot push, so review and merge happen in the
  local checkout and Jahid pushes.

## Amendment, 2026-09-20 (Gate 1 review of the ordering knob, Minor 4) — two mechanisms behind item 5

- **Reserved-but-refused option names.** An option the supervisor named but nobody has defined
  ships as a *reserved* value that fails loudly at load time with a pointer to the ADR that left
  it undefined — here `few_shot.ordering: complexity_asc` refuses with "undefined pending Jahid /
  the supervisor (ADR 0032 item 5)". The name is in the config's legal-values message so the
  reader knows it exists; no code path implements it. This is the standing pattern for "named,
  not defined".
- **Presentation ordering is applied at exactly one funnel and never at selection.**
  `format_few_shot_examples()` is the single point where exemplar order is changed;
  `examples_from_reranked_row()` and the bi-encoder/random selectors return their top-k in
  retrieval-rank order, untouched. Applying the ordering at selection too would cancel on the
  reranked path and not on the others. The A5 contrast (as_is vs reversed) rests on this
  invariant: the two arms show the same k exemplars and differ only in order. A regression test
  pins it. Any future edit that reorders at selection breaks A5 and needs a new ADR.

## Amendment, 2026-09-20 (Jahid) — item 6 reversed: Qwen3.5-9B runs

Jahid: *"I need you to run Qwen 9B parameter one. Why haven't you run that one?"* The lead's
default in item 6 (not in this block) is overturned. Ahmmad named "QN 9 billion" as the third
model on 2026-08-27; the confound with the Qwen2.5 size rung (different generation and different
size at once) stands as a caveat on how it is compared, not as a reason to omit it. It runs as
exp-027 (`as_is`) and exp-028 (`reversed`) on the block's exact recipe — unguided, 4-bit, pinned
600, the exp-014 retrieval artifact — so it sits on the same axis as exp-020/021/022. exp-005
(Qwen3.5-9B on the v1 9,156 pool) is not on that axis and is not its comparator.
