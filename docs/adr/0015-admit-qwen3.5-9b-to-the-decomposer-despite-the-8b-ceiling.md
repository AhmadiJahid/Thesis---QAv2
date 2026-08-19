# 0015. Admit Qwen3.5-9B to the Decomposer Despite the ~8B Ceiling

- **Status**: Accepted (Jahid, 2026-08-19, in session) — **pending supervisor confirmation**, because the ~8B ceiling is recorded in CLAUDE.md as the supervisor's decision
- **Date**: 2026-08-19

Related: [0008](./0008-lift-the-600m-router-cap-to-the-overall-8b-ceiling.md) (the router-cap precedent for a Jahid-stated cap change), [0014](./0014-guided-vs-unguided-runs-on-the-v1-pool-and-retrieval-artifact.md) (the other pending-confirmation deviation on issue #12), 0013 on the `feature/12-guided-vs-unguided` branch (the #12 condition conventions).

## Context

The standing constraint in CLAUDE.md caps model size at roughly **8B parameters overall**, asserted at every model load by `src/model_size.py` against `configs/model_limits.json`. As configured, issue #12 (guided vs unguided) is runnable on exactly one model folder: `mistral_7b_instruct`. The `qwen3_5_9b` folder ships a conforming unguided prompt and passed the same prompt-parity guards, but at 9B it is refused by the ceiling; `qwen2_5_3b` and `phi_4_mini_instruct` are within the ceiling but ship no unguided prompt. The single-model scope was surfaced to Jahid as a decision on issue #12 (2026-08-19 comment, item 2).

## Decision

Asked whether #12 runs on Mistral alone or a second model should be prepared, **Jahid answered (2026-08-19, in session): "You can use 9b parameter as well."**

Recorded as:

1. **`qwen3_5_9b` (Qwen/Qwen3.5-9B) is admitted as a decomposer model**, and issue #12 runs its three conditions on **both** `mistral_7b_instruct` and `qwen3_5_9b`, same eval set, same retrieval artifact, same decoding overrides.
2. **`default_max_params` in `configs/model_limits.json` rises from 8,000,000,000 to 10,000,000,000** so the load-time assertion admits the 9B model while still refusing anything materially larger. The **router cap stays at 8,000,000,000** (`router_max_params` unchanged): Jahid's answer was about the decomposer for #12, and nothing wider is inferred from it.
3. The ceiling change is a config edit on the `feature/12-guided-vs-unguided` branch and goes through that PR's review (Gate 1), not directly to main.

## Consequences

- Thesis reporting must state the exception wherever Qwen3.5-9B results appear: the headline constraint elsewhere in the work remains ~8B, and one model in this comparison is above it.
- **If the supervisor reasserts the 8B ceiling, that supersedes this ADR**: the ceiling reverts, the Qwen arms of #12 become out-of-scope results (recorded but not reported), and the #12 comparison rests on Mistral-7B alone — which remains a complete experiment on its own.
- The exp-001 LoRA arm and everything else already built are unaffected; they were sized under the 8B ceiling and stay there.
