# 0031 — Jahid's 2026-09-14 gate decisions for the 2026-08-27 supervisor assignment

- **Status:** accepted (Jahid, 2026-09-14, in session). Item 4 is *pending supervisor confirmation*; item 5 is the lead's reading of items 1–4 and is flagged for Jahid's veto.
- **Context:** [`docs/plan/2026-09-14-post-supervisor-meeting-phase.md`](../plan/2026-09-14-post-supervisor-meeting-phase.md) §7 listed seven decisions that gate the experiment block Ahmmad assigned on 2026-08-27 (issue #50). Jahid answered them in session. This ADR records the answers; it adds nothing to them.

## Decisions (Jahid's, at his altitude)

**1. The pool is the imbalanced 2,000.** Plan §7.1. The artifact is
`runs/pool_sweep/pools/size2000_imbalanced_trial0_poolseed42`. This is also the status quo
issue #46 §4.1 had been proceeding on, so nothing downstream is invalidated by the choice.
It is now an adoption rather than a default, for this block.

**2. The evaluation set is the ADR 0007 pinned 600 — decided by the lead under delegation.**
Plan §7.2. Jahid's words: *"evaluation set I mean you need to do recommended ones either bigger
one 750 if you wanted or 600"* — i.e. he handed the call back with both candidates named. The
lead's reading, on the evidence already in the log: the pinned 600 (200/hop) is the set
exp-004, exp-005, exp-008, exp-011 and exp-015 were all evaluated on, and every Mistral number
the new models are being compared against sits on it. The pool_sweep dev sample of 750
intersects it at only **405/600** (measured, exp-018), so the two are not interchangeable and
choosing 750 would put the new arms on an axis where no existing baseline lives. Continuity
wins. Recorded as a delegation, per the standing pattern that an operational call handed back
is decided on the evidence and written down rather than bounced again.

**3. All arms run unguided.** Plan §7.4. Jahid's words: *"we do not — do all of them unguided.
Like, we did not do anything guided here."* So the model contrast is measured on the method as
it actually stands, with no hop-count guidance of any kind. This constrains which models can
participate: a decomposer model must ship `prompt_unguided.md` to run this condition
(`configs/decomposer_musique.json:85`), which today only `mistral_7b_instruct` and
`qwen3_5_9b` do.

**4. The model-size ceiling is raised again, to admit a ~14B decomposer.**
`configs/model_limits.json` `default_max_params` goes from **1e10 to 1.5e10**. Jahid's words:
*"about 10 billion let's raise the cap"*. Supporting evidence, which did not exist when ADR 0015
was written: the supervisor himself named this range in the 2026-08-27 meeting — *"12, 14, check
what is above 7 billion"*
([1:02:47](https://fathom.video/calls/801583891?timestamp=3767)) — so unlike ADR 0008 and ADR
0015 this raise has a recorded supervisor utterance behind it. It is still marked **pending
supervisor confirmation**, because a verbal suggestion to try a larger model in one experiment
is not the same act as resetting the thesis's standing size constraint, and `CLAUDE.md` reserves
that constraint to the supervisor. **If the supervisor reasserts 8B, or 10B, that supersedes
this.** `router_max_params` (8e9) and `answerer_max_params` (8e9) are untouched.

**5. The three arms, as the lead reads items 1–4 — Jahid to confirm or veto.**
Ahmmad's stated purpose was to separate two factors: *"if we fix the size of the model, but we
try different model from Mistral to QN. Does the result will change significantly or not?
Within the same model, if we change the size, if the model will change or not"*
([1:03:13](https://fathom.video/calls/801583891?timestamp=3793)). That purpose is only served if
each contrast moves one factor:

| contrast | arms | factor that moves |
|---|---|---|
| family, size held at 7B | Mistral-7B-Instruct-v0.3 vs **Qwen2.5-7B-Instruct** | model family |
| size, family held at Qwen2.5 | Qwen2.5-7B-Instruct vs **Qwen2.5-14B-Instruct** | parameter count |

**Amendment, 2026-09-14 (Gate 1 finding I-2): weight precision is a third factor, and the
table above is incomplete without it.** Qwen2.5-14B cannot run unquantized on this box — its
bf16 weights are 29.54 GB against the 3090's 24 GiB — so the 14B arm is necessarily 4-bit. If
the 7B arm runs unquantized (mirroring Mistral in exp-004), then the size contrast moves
parameter count **and** precision together, which is the exact confound the supervisor's "one
change, and test" instruction exists to prevent, reintroduced on the axis it was meant to
protect.

**The lead's resolution, under the same delegation as item 2, pending Jahid's veto: all three
arms run at 4-bit.** The premise that makes this free is that *no arm inherits exp-004's
precision* — every arm in this block is a fresh run on the size-2000 imbalanced pool, which is
not the pool exp-004 used, so the Mistral arm is being re-run regardless. Pinning all three to
4-bit therefore holds precision constant at no extra cost, and both contrasts in the table above
move exactly one factor as stated. What is given up is direct comparability to exp-004's
unquantized Mistral numbers — not what this block is for, and Mistral has already run 4-bit in
exp-010. The pinning happens in the **run config**, not in the model folders' defaults, so the
folders keep their per-model defaults and each run records its resolved `quantization` in
metrics.

**Qwen3.5-9B does not fit either contrast.** Against Qwen2.5-7B it moves family *and* size at
once (Qwen3.5 is a different generation from Qwen2.5), which is precisely the confound
Ahmmad's "one change, and test" instruction exists to prevent. It is not dropped — exp-005
already carries a Qwen3.5-9B result and it is the model that contradicts ADR 0030's premise
(issue #49) — but it is a fourth, confounded point rather than the size rung, and this block
does not need it. Cost of including it anyway is roughly +25 minutes of GPU.

**6. No experiment data goes to GitHub.** Plan §7.6. Jahid's words: *"Let's not share data with
GitHub right now. I'm going to share it with them."* The standing rule in `CLAUDE.md` is
unchanged, the supervisor's request to see the data
([45:19](https://fathom.video/calls/801583891?timestamp=2719)) is satisfied by Jahid sharing it
directly outside the repo, and no agent arranges that transfer.

**7. Slides come before any GPU work**, and ship as two artifacts from one source of truth: an
HTML deck (previewable, publishable as a link) and a self-contained Beamer `.tex` Jahid pastes
into Overleaf. Jahid accepted that the deck will be rebuilt more than once — *"it's fine to
build a couple times"* — so the first pass covers the full deck rather than the front half.

**8. The finer pool-size sweep is deferred, not dropped.** Jahid's words: *"Working on the pool
will leave to the next step, but record on GitHub issues… instead of 2, 4, 8 let's do 2, 3, 4,
5, 6, 7, 8 increase by a K."* So the sweep becomes **1,000-step granularity from 2,000 to
8,000** — seven sizes — scheduled after this block. Note for the record: the supervisor raised
this himself and then downgraded it to *"not a high priority, it's an idea"*
([28:56](https://fathom.video/calls/801583891?timestamp=1736)); Jahid is choosing to keep it,
later. Tracked as issue #51.

## Consequences

- Plan §7.1, §7.2, §7.4, §7.6 and §7.7 are closed; §7.3 is closed by items 4–5; §7.5 (what
  "least complex" means for exemplar ordering) remains open and still gates A5.
- `configs/model_limits.json` changes, and `src/model_size.py`'s ceiling assertion will admit
  14B decomposers. That is a shared-config change and goes through Gate 1 review with the
  model-folder work, not as a silent edit.
- A Qwen2.5-7B-Instruct **and** a Qwen2.5-14B-Instruct decomposer folder are now both needed,
  each with a parity-guarded `prompt_unguided.md` (item 3). Qwen2.5-7B-Instruct is already in
  the local HF cache; Qwen2.5-14B-Instruct is not and needs ~28 GB of download against 178 GB
  free.
- Issue #50 is updated to name the arms and the raised ceiling.
- Nothing here unblocks a GPU run: the shared RTX 3090 is still held by another user's process
  (plan §1), and every arm above waits on it.
