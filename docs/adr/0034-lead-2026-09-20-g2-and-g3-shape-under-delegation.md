# 0034 — The shape of G2 (pool-only fine-tune) and G3 (reversed ordering), decided by the lead under delegation

- **Status:** accepted (lead, 2026-09-20). Jahid's words that morning: *"continue all the experiments.
  You are on your own."* Operational calls below are made on the evidence and recorded here so the
  November write-up can trace them; any of them is Jahid's to reverse.
- **Context:** [ADR 0032](0032-jahid-2026-09-20-go-ahead-for-the-50-block.md) items 3 and 8 fix
  that G3 runs and that G2 follows exp-008's precision. Two things were left open: *which model(s)*
  the reversed ordering runs on, and *how* the pool-only arm pins the pool.

## Decisions

**1. G3 runs the `reversed` ordering on all three A1 arms** — Mistral-7B, Qwen2.5-7B and
Qwen2.5-14B, each 4-bit, unguided, pinned 600, same retrieval artifact — so each G3 arm is the
exact twin of its A1 arm with only exemplar order moved (ADR 0032 amendment: ordering is applied at
one funnel and selection is untouched, so the k exemplars are identical). Reason: the supervisor's
instruction was *"try all those things we mentioned"* and put a magnitude on it (*"up to 10%"*);
exp-021 showed the two 7B families fail in opposite directions on length, so an ordering effect may
differ by family, and one model would not show that. Cost is roughly 14 + 42 + (14B time) minutes
of GPU, measured from exp-020/021. `complexity_asc` still does not run (ADR 0032 item 5).

**2. G2's pool is pinned by a new fine-tune arm, `pool_2000_imbalanced`, not by editing
`pool_2000`.** The existing `pool_2000` arm draws a seeded 2,000 from the enriched train pool; it
stays as written so its definition does not change under anyone. The new arm sets
`train_source_path` to the adopted pool artifact
`runs/pool_sweep/pools/size2000_imbalanced_trial0_poolseed42/pool.jsonl` (ADR 0031 item 1) with no
cap and no hop filter, so the adapter is trained on exactly the 2,000 questions the prompting arms
retrieve exemplars from. The training record must carry the source file's sha256 so a later reader
can check the adapter was trained on that pool and not a regenerated one.

**3. G2 precision and comparator follow ADR 0032 item 8, stated precisely:** exp-001 trained the
`full_train` adapter as **QLoRA on a 4-bit base** (`experiments/exp-001/config.json`,
`quantization: "4bit"`), and exp-008 evaluated that adapter on the **fp16 base**
(`quantization: "none"`). G2 does the same: train 4-bit QLoRA, evaluate with `--no-few-shot` at
`quantization: none` on the pinned 600 exactly as exp-008 was, compared against exp-008 `full_train` (training data is the one moving
factor) and, as context only, against the 4-bit prompting arms — with the precision difference
stated every time.

**4. Stop condition carried forward** (ADR 0032 item 7): if the three A1 arms are near-identical,
G3 waits for Jahid. exp-021 vs exp-020 already shows significant differences (chain validity,
exact match, both length rates), so as of exp-021 the condition has not fired.

## Consequences

- One small config branch (the new arm, plus sha256 in the training record if the loader does not
  already write it), Gate 1 delta review, then G2 as one experiment id (train + eval + compares).
- G3 is three experiment ids, one per model, each comparing against its own A1 twin.
- Total remaining GPU after exp-022: roughly 2.5–3.5 hours, extrapolated from exp-001/020/021.

## Amendment, 2026-09-20 (Gate 1 review of the pool-arm branch) — three things the G2 write-up must carry

- **The exp-008 comparison moves training-set size and hop mix, not only identity.** exp-001
  trained `full_train` on 19,938 examples (2,494 optimizer steps, 15,497 s). `pool_2000_imbalanced`
  trains on 2,000 examples at the same 2 epochs — roughly 250 steps, about a tenth of the compute —
  with the pool's own hop mix (1461/413/126, 73% two-hop) instead of the split's. So "training data
  is the one moving factor" means identity, volume and hop mix together. Every G2-vs-exp-008 claim
  states 2,000 vs 19,938 and ~250 vs 2,494 steps beside the delta. The seeded hop-stratified
  `pool_2000` arm, the only clean same-size-different-draw control, has never been trained
  (issue #13) and is not in this block.
- **G2 passes `--retrieval-input runs/exp-014/retrieval/s2_pinned600_imbalanced_rerank_top5.jsonl`
  at evaluation.** exp-008 took the config default (the v1 artifact). The reviewer diffed the two:
  identical 600 query ids, identical order, identical question text, and with `--no-few-shot` the
  exemplar block is empty in both, so the prompts are byte-identical and nothing measurable moves.
  Passing the flag makes G2's recorded provenance name this block's pool like every other #50 arm
  and removes a silent dependency on `retrieval.input_key`.
- **A relative `train_source_path` resolves against the paths config's `runs_root`**, because it
  names a produced artifact, mirroring `--retrieval-input`. This supersedes ADR 0012's line that a
  pool is pinned by editing `arms.pool_2000.train_source_path` "with no code change": the pin is a
  separate arm, and it took a resolution-root change. ADR 0012 is amended by reference here.
- **G2 launch preconditions** (the block's practice, issue #50): at train time the record's
  `training_source_sha256` equals `cc3646625d759a1b58179b18c25f83d255db24f05acc08b666985af2e58d6646`
  and `quantization` is `4bit`; at eval time `quantization` is `none`, adapter parity is
  `checked: true` with 0 mismatches, and `evaluation_set.pinned` is true over 600. A mismatch
  aborts before tokens are spent.
