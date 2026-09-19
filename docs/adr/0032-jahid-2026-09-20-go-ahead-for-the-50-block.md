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
- `configs/model_limits.json` `default_max_params` goes to 1.5e10 on the model-folder branch,
  with this ADR cited in its note.
- Commits stay local: this box's `gh` credential cannot push, so review and merge happen in the
  local checkout and Jahid pushes.
