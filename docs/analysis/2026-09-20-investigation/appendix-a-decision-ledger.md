<!-- Lane ledger produced by a read-only investigator agent on 2026-09-20 for the lead's state-of-the-work investigation (`docs/analysis/2026-09-20-state-of-the-work-investigation.md`). Every claim cites a file, commit, issue or log row. Numbers are quoted from committed artifacts, never re-computed. -->

# Decision ledger — ADRs 0001–0034, the meeting cross-check, and the three plan docs

Read-only investigation, 2026-09-20. Paths are relative to `/cta/users/fyilmaz/Thesis---QAv2/`. Sources: `docs/adr/0001…0034` + `docs/adr/README.md`, `docs/meetings/2026-08-12-supervisor-meeting-transcript-crosscheck.md`, `docs/plan/2026-09-14-post-supervisor-meeting-phase.md`, `docs/plan/2026-09-20-next-step.md`, `Plan after the meeting.md`.

## 1. One line per ADR

| # | Date | Decider | Decision (≤25 words) | Status |
|---|---|---|---|---|
| 0001 | 2026-08-12 | Jahid | Both MetaQA and MuSiQue migrate from v1; code ported selectively; v1 knowledge carried as docs, baseline re-runs deferred. | active; amended by 0018 |
| 0002 | retro 2026-08-12 | v1 record | Three-stage router → decomposer → jury pipeline. | superseded by 0006 |
| 0003 | retro 2026-08-12 | v1 record | Mask queries only; use the pre-masked few-shot pool; never re-mask the pool at runtime. | active |
| 0004 | retro 2026-08-12 | v1 record | Initial models: router Qwen2.5-0.5B, decomposer Qwen2.5-7B, jury = decomposer. | active (initial selections; later changed in practice) |
| 0005 | 2026-08-12 | implementer | Seed before sampling; v1 `--sample_size` results are not reproducible. | active; amended by 0020 |
| 0006 | 2026-08-12 | supervisor (Ahmmad) | Jury dropped; dataset roles fixed; few-shot method bi-encoder→cross-encoder + typed masking; pool 2000; contribution framing; router OPEN. | active; §3 amended by 0017 & 0018; item 6 resolved by 0010, then 0030 |
| 0007 | 2026-08-17 | Jahid | MuSiQue eval set = v1's identical 600 ids, 200 per hop (2/3/4), from dev. | active |
| 0008 | 2026-08-17 | Jahid | 600M router cap lifted; router bound only by the overall ~8B ceiling. | active |
| 0009 | 2026-08-17 | implementer | Paired bootstrap 95% CIs + exact McNemar as the house significance protocol; six tests, no correction. | accepted, **pending supervisor confirmation**; amended by 0011, 0017 |
| 0010 | 2026-08-18 | supervisor via Jahid | Router stays, reframed as a hop-count regressor; fine-tuning prioritised. | superseded for the critical path by 0030 |
| 0011 | 2026-08-18 | implementer | Comparison conventions: n≥30 `underpowered` floor, versioned per-item schema, weight-mismatch refusal. | accepted, provisional items **pending Jahid**; amended by 0020 |
| 0012 | 2026-08-19 | Jahid + implementer | LoRA fine-tuning arm conventions: three data arms, zero-overlap assertion, cost reporting, zero-shot adapter eval. | active; items 7/8 settled 2026-08-19; pool identity deferred; amended by 0034 |
| 0013 | 2026-08-19 | implementer (Jahid's design) | Guided/unguided/unguided_capped condition conventions; derived unguided prompts; exemplar hop lines state own gold hops. | active; its three open questions settled by Jahid 2026-08-19 |
| 0014 | 2026-08-19 | Jahid | Issue #12 runs on the v1 9,156-example pool and its retrieval artifact, not the 2,000 pool. | accepted, **pending supervisor confirmation** (deviates from 0006 §4) |
| 0015 | 2026-08-19 | Jahid | Admit Qwen3.5-9B as decomposer; `default_max_params` 8e9 → 1e10; router cap unchanged. | overtaken by 0031 item 4 + its 2026-09-20 amendment (no in-place note on 0015 — see §4) |
| 0016 | 2026-08-19 | implementer | Invariants only a real run can break get source-level (AST) guards with negative controls. | active |
| 0017 | 2026-08-20 | Jahid | Triage of the transcript cross-check: k values demoted to ablations, cap 8, t-test added, RL out of scope. | active; amends 0006, 0009 |
| 0018 | 2026-08-20 | Jahid | MetaQA+MuSiQue confirmed as thesis datasets; `intfloat/e5-small-v2` the bi-encoder default; typed masking reopened. | active; item 2 **pending supervisor confirmation** |
| 0019 | 2026-08-20 | Jahid | MuSiQue answering backend: reader from the decomposer registry, all-paragraphs context, official MuSiQue EM/F1. | accepted, **pending supervisor confirmation** |
| 0020 | 2026-08-20 | implementer | v1 per-item artifacts may be re-analysed under v2 statistics under six conditions. | active; amends 0005, 0011 |
| 0021 | 2026-08-20 | implementer | Clustering pool construction = third value of the existing `balance` axis; k-means, nearest-to-centroid. | accepted, design **pending Jahid** |
| 0022 | 2026-08-21 | implementer | Hop-matched retrieval is an opt-in candidate filter before top-k; no fallback to mixed, ever. | accepted, design **pending Jahid** |
| 0023 | 2026-08-22 | Jahid | Composite replacement starts now; pipeline completion order pool→router→fine-tune; generalisation framing. | active; items marked *(supervisor)* pending |
| 0024 | 2026-08-22 | implementer | Router emits query-id-keyed `predictions.jsonl`; retrieved-few-shot router; decomposer gains `hop_source`. | accepted, design **pending Jahid** |
| 0025 | 2026-08-22 | implementer | End-to-end backends: two MetaQA denominators, unavailable-not-zero taxonomy, no path is labelled GRAG. | accepted, design **pending Jahid** |
| 0026 | 2026-08-22 | implementer | Break-faithful EM/SARI/GED + chain_validity ported additively, with three named deviations. | active; amended by 0029 |
| 0027 | 2026-08-23 | implementer | A number justifying a config constant ships as a committed script; a tool that scores no system is not an experiment. | accepted; **point 5 pending Jahid** |
| 0028 | 2026-08-23 | Jahid | Pool choice and the router keep/drop call delegated to the lead; composite replacement authorised to implement. | active; item 2 discharged by 0030 |
| 0029 | 2026-08-23 | implementer | Reported primary = unblended six-term `decomposition_report_card`; `composite_score` frozen as legacy. | accepted, **pending Jahid and his supervisor**; amends 0026 |
| 0030 | 2026-08-23 | lead (under 0028 delegation) | Router leaves the decomposition critical path, on the oracle ceiling (+0.082) vs fine-tuning (+0.202). | active; *(supervisor may overturn)*; amends 0028, 0010 |
| 0031 | 2026-09-14 | Jahid (+lead items 2, 5) | Gate decisions for the #50 block: imbalanced 2,000 pool, pinned 600, all unguided, ceiling raised, no data to GitHub, slides first, sweep deferred. | active; item 4 **pending supervisor**; amended 2026-09-20 |
| 0032 | 2026-09-20 | Jahid (+lead defaults) | Go-ahead for the #50 block; 1.5e10 confirmed by Jahid; reversed ordering runs; Opus-only agents. | active; item 6 reversed by its own 2026-09-20 amendment |
| 0033 | 2026-09-20 | lead (delegation) | Anchor set = frozen 45-id list in `configs/anchor_set_45.json`; inspection by post-filtering finished evaluations. | active; amended same day (anchor prompts logged by id) |
| 0034 | 2026-09-20 | lead (delegation) | G3 reversed ordering on all three A1 arms; G2 pins the pool via a new `pool_2000_imbalanced` arm. | active; amended same day |

## 2. Standing constraints as they now stand

- **Model size.** `default_max_params` = **8e9** (the supervisor's floor, restored); **`decomposer_max_params` = 1.5e10**; `router_max_params` = 8e9 (0008); `answerer_max_params` = 8e9 (0019 decision 1); retrieval/NER/reranker inherit 8e9. Set by 0008 → 0015 (1e10) → 0031 item 4 and its **2026-09-20 amendment**, confirmed by Jahid in 0032 item 2, still marked pending the supervisor. Convention: *a role granted an exception gets its own key* (0031 amendment).
- **No closed commercial model judges decomposition quality.** 0006 §1; re-stated in 0012 item 2, 0019, 0023 item 1, 0028 item 3. Corroborated in the transcript ([49:29], [1:10:53], cross-check §"GRAG, fine-tuning, timeline").
- **Dataset roles.** MuSiQue = decomposition-quality **and** end-to-end; MetaQA = end-to-end only via GRAG (0006 §2, upgraded to a research decision by 0018 item 1). GRAG is unavailable and no path may be labelled GRAG (0025 §3).
- **Few-shot method.** Bi-encoder top-20 → cross-encoder top-5 with typed masking (0006 §3); the **k values 20/5 are ablatable working values, not method constants** (0017 item 1); **typed masking is reopened as not settled** (0018). Bi-encoder = `intfloat/e5-small-v2` (0018 item 2, pending supervisor); the cross-encoder slot has no recorded decision.
- **Pool.** Size fixed at 2000 by the supervisor (0006 §4); issue #12 ran on the v1 9,156 pool as a recorded deviation (0014); the adopted 2,000 pool for the #50 block is `runs/pool_sweep/pools/size2000_imbalanced_trial0_poolseed42` (0031 item 1, on 0028 item 1's delegation).
- **Evaluation set.** The ADR 0007 pinned 600 (200/hop), re-affirmed for the #50 block by 0031 item 2.
- **Significance protocol.** Paired percentile bootstrap 95% CIs + exact McNemar, six tests, no correction (0009); **plus a paired t-test alongside** (0017 item 4); `underpowered` flag below n=30 (0011 item 1); comparisons across different id sets are refused (0009, 0011).
- **Metric suite.** Reported primary = the six-term `decomposition_report_card`; `ged_macro` where one number is required, with three mandatory caveats; `composite_score` frozen `legacy` (0029). The **thesis**-primary metric is still undecided (issue #6 item 5).
- **Fine-tuning** is allowed and expected as a comparison arm (CLAUDE.md; ordered, not merely allowed, per the cross-check).

## 3. Everything marked OPEN / PENDING / DEFERRED / "Jahid's call"

| Item | Source | Resolved later? |
|---|---|---|
| Which v1 baselines to re-run once compute settles | 0001 | no |
| Whether v2 changes the initial model defaults | 0004 | overtaken in practice by 0013/0015/0031, never recorded |
| Router keep/drop | 0006 item 6 | yes — 0010 (keep), then 0030 (off critical path) |
| t-test substitution unconfirmed by supervisor | 0009 "Pending" | partly — 0017 item 4 adds a t-test; supervisor sign-off still absent |
| Significance-claim floor of 30; weight-mismatch rule 3; whether the comparison artifact gets a schema version | 0011 items 1, 3 and the 2026-08-20 note | no — all still Jahid's |
| Prompt parity between arms (item 7); `target_reference_style` (item 8) | 0012 | yes — both settled by Jahid 2026-08-19 |
| Which 2,000 pool the `pool_2000` fine-tune arm uses | 0012 | partly — 0034 item 2 pins a **new** arm `pool_2000_imbalanced`; `pool_2000` stays untrained |
| Step-line cap 8 vs the supervisor's example of 6 | 0013 item 4 | yes — 0017 item 2 (8) |
| `max_new_tokens` 256; single-model scope | 0013 open questions | yes — 1024; Qwen3.5-9B admitted (0015) |
| Pool deviation to 9,156 pending supervisor | 0014 | no |
| Qwen3.5-9B admission pending supervisor | 0015 | no (re-confirmed by Jahid 2026-08-22; folded into the decomposer key 2026-09-20) |
| A tiny real-generation smoke stage | 0016 | no |
| Pipeline figure the supervisor screenshotted; GRAG handover | 0017 item 8 | no |
| Masking default; thesis-primary metric; cross-encoder re-ranker slot | 0018 "Remaining open on #6" | no — all three still supervisor-facing |
| Embedding model pending supervisor | 0018 item 2 | no |
| Whether a defaulted router hop should be excluded or scored as a refusal | 0024 item 1a | no |
| Headline MetaQA denominator; MetaQA adopting SQuAD normalisation; `wrong_intermediate_answer` from gold sub-answers | 0025 §1–§2 | no — "three decisions queued for Jahid" |
| Whether `_REF_RX`'s bracketed syntax was a defect or intended | 0026 item 2 | no |
| `num_trials` > 1 for the pool sweep (confound 1) | 0021 | no |
| Which pool; router call; composite; priority | issue #46 §4 → 0028 | yes — 1 and 2 delegated to the lead; 3 authorised; 4 set |
| Phase plan §7.5 / §7.8 — what "least complex" means for exemplar ordering | phase plan §7, item 8 "STILL OPEN" | no — 0032 item 5 keeps `complexity_asc` **reserved and refused at load** |
| Whether to keep working on the pool at all | phase plan §8 ("the supervisor never answered") | no |
| Ahmmad's confirmation of the 1.5e10 ceiling | next-step §4.6 | no — Jahid confirmed, supervisor has not |
| Qwen3.5-9B as a fourth point | next-step §4.4 → 0032 item 6 | yes — reversed by Jahid the same day; runs as exp-027/028 |
| Stop condition on G1 | phase plan §6 ("flagged, not adopted") | yes — adopted by the lead, 0032 item 7 |
| Finer pool-size sweep (2k→8k by 1k) | 0031 item 8 | deferred to issue #51, after the block |

## 4. Contradictions and drift between documents

1. **The ADR index is stale.** `docs/adr/README.md`'s table ends at 0030, though 0031–0034 exist — against its own rule: *"every new record is added to the index table above in the same commit."*
2. **0015 has no forward amendment note.** README requires *"Amendments are marked on both ends."* 0031 item 4 and its amendment fold 0015's 1e10 into `decomposer_max_params: 1.5e10` with `default_max_params` back to 8e9, but `0015-...md` still reads *"`default_max_params` … rises from 8,000,000,000 to 10,000,000,000"* with no pointer forward.
3. **0006 §4 likewise.** *"Pool size is fixed at 2000."* carries no in-place note pointing at 0014, which runs on 9,156 — the amendment is marked on 0014's end only.
4. **Both plans still describe the ceiling the way 0031's amendment retracted.** Phase plan §3 P1: *"raise `default_max_params` to 1.5e10"*; next-step §2 B1: *"raise `default_max_params` to 1.5e10 (ADR 0031 item 4)"*. The branch that landed put the raise on `decomposer_max_params` and returned the default to 8e9 (0031, "Amendment, 2026-09-20 … I-1").
5. **Anchor-set build: plan vs ADR.** Phase plan §2 A4 and §3 P0 specify *"a committed id list plus an id-filter path into the decomposer and evaluator"*, and next-step B2 repeats *"an `--ids` filter path through `run_decomposer.py` and the evaluator"*. ADR 0033 item 3 rejects exactly that: *"Inspection is post-filtering, not an id-filter into the run path… The id-filter path from the older branch (`scripts/anchor_inspect.py`) is not adopted."* 0033 says it *"Supersedes the phase plan §2 A4 build sketch"*; the plan files were not annotated.
6. **Pool pinning "no code change" retracted.** 0012 said `arms.pool_2000.train_source_path` pins a pool *"with no code change"*; 0034's amendment: *"the pin is a separate arm, and it took a resolution-root change. ADR 0012 is amended by reference here."* 0012 now carries the matching note.
7. **Qwen3.5-9B in/out, same day.** 0032 item 6: *"Qwen3.5-9B is not in this block."* Its own amendment: *"Jahid: 'I need you to run Qwen 9B parameter one.' … The lead's default in item 6 … is overturned."*
8. **Stop condition status differs by document.** Phase plan §6: *"flagged here for Jahid's agreement, not adopted."* 0032 item 7: *"The stop condition is adopted."* (a lead default under delegation, Jahid not having answered).
9. **The transcript does not support two things ADR 0006 states as fixed.** Cross-check: *"the numbers 20 and 5 were not fixed in the meeting"* — repaired by 0017 item 1; and typed masking was supervisor-confirmed at [1:24:27] while 0018 records Jahid marking it *"not settled"* — a departure the ADR itself flags: *"If the supervisor reasserts typed masking as fixed, that supersedes this reopening."*
10. **A sanctioned fallback recorded nowhere.** Cross-check: *"ADR omits a sanctioned fallback: [1:15:24] 'If you want to shrink… to one dataset, continue with the music'"* — i.e. MuSiQue-only. No ADR carries it.
11. **Pool-size evidence is internally inconsistent in the source.** Cross-check: *"the transcript is internally inconsistent on the evidence — [26:16]/[47:17]/[1:29:25] describe 4000 as (co-)best"* against the 2000 pin.
12. **Contribution framing vs the supervisor's later caution.** 0006 §5 states the contribution; phase plan §5 records him on 2026-08-27: *"we cannot say there is a contribution up to this point because we didn't do testing. You may say this is our objective."* Not reconciled in any ADR.
13. **`Plan after the meeting.md` is a pre-execution prompt script, not a current record** — prompt 4 fixes *"pool 2000"* (overridden by 0014) and prompt 7 assumes *"the 8B ceiling"*. It survives only as provenance for 0012/0013's "Jahid's instruction, prompt N" citations.

## 5. What the supervisor assigned on 2026-08-27 (per `docs/plan/2026-09-14-post-supervisor-meeting-phase.md` §2)

Scope limit, verbatim: *"Only those. Don't do anything further."* ([1:05:22]).

- **A1** — three decomposer models on one fixed 2,000 pool, one factor changed: *"Mistral 7 billion, QN 7 billion, QN 9 billion"* ([1:03:04]); purpose is to separate family from size.
- **A2** — a Mistral fine-tune trained **on the 2,000 pool only**, not the full split ([1:03:32]).
- **A3** — Jahid personally understands every metric in use: *"this is for you"* ([1:04:00], [20:53]).
- **A4** — a fixed anchor set, *"15 from each hop. So in total 45 samples… fixed over all the experiments"*, hand-inspected every run ([1:04:00]–[1:05:11]).
- **A5** — few-shot exemplar ordering: reversed, and least-complex → most-complex; *"the order is matter. It can change up to 10%"* ([1:07:43]).
- **A6** — report results per hop count as well as overall ([42:29], [1:09:33]).
- **A7** — eye-to-eye qualitative comparison, Qwen vs Mistral, on the same items ([42:29]).
- **A8** — slides, in his ordering, with ~18 named fixes ([09:40], [1:05:38]).
- **A9** — show him the experiment data ([45:19]).

Also given as method: *"You do change, one change, and test"* ([1:07:33]).

**Explicitly parked** (phase plan §4): the finer pool-size sweep (*"not a high priority, it's an idea"*); repeating experiments for variance; cutting the four statistical tests to one (*"you don't need to pick now"*); DBSCAN (*"I will also maybe take a quick look"*); training on MetaQA and testing on MuSiQue; **the router — no instruction of any kind in 71 minutes**; a bigger model beyond the three (*"You may try. You may try."* — permission, not assignment).

**Rejected / criticised** (§5): the composite score (*"Does that mean you will stop using the composite score?" / "Yes."*); claiming a contribution before testing; evaluation and datasets inside the methodology section; the deck reading as AI-generated. **Unchased concern** (§6): *"There's something wrong in the experiment… you try clustering with and without clustering and the result is matching exactly"* ([1:09:02]) — investigated as issue #52 / exp-019 and resolved as a real null (next-step §1).

## 6. The thesis contribution framing, verbatim

From `docs/adr/0006-drop-the-jury-fix-dataset-roles-and-the-few-shot-method.md`, Decision item 5:

> **5. Contribution framing:** the contribution is **a strategy for constructing the few-shot pool and retrieving examples for hop-aware decomposition, validated empirically** — not "we decompose multi-hop questions".

No later ADR amends item 5. The cross-check qualifies its wording: *"Confirmed ([1:11:38] 'You are proposing a strategy, how you choose your few shots… with empirical study'); \"hop-aware\" wording is a gloss."*
