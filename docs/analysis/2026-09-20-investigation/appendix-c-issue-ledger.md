<!-- Lane ledger produced by a read-only investigator agent on 2026-09-20 for the lead's state-of-the-work investigation. Every claim cites a file, commit, issue or log row; two claims were corrected by the lead and are marked as such. Numbers here are quoted from committed artifacts, never re-computed. -->

# Issue-tracker ledger — Thesis-QAv2 (read-only investigation, 2026-09-20)

Source: `gh issue list --state all --limit 100` + `gh issue view N --comments` for all 32 issues.
Repo `main` at `f1bbdad`, 53 commits ahead of `origin/main` (nothing pushable from this box).

---

## 1. OPEN ISSUES (21)

### #6 — Open research decisions carried from v1 — need Jahid + supervisor (opened 2026-08-12)
**Asks:** seven v1-implied decisions get explicitly decided and recorded as ADRs.
**State (latest comment 2026-08-23):** partially done — 4 of 7 discharged. Items 1/3/4-poolsize/6 closed by ADR 0006, ADR 0007, ADR 0018; item 2 folded into #23 (ADR 0010); item 7 split — bi-encoder `intfloat/e5-small-v2` recorded "pending supervisor confirmation", **cross-encoder re-ranker slot has no recorded decision** (correction comment 2026-08-20T05:23:48Z).
**Still live, blocked on:** supervisor, via Jahid. (a) **item 4** retrieval *masking* default (typed) — "not settled — typed masking stays open with the supervisor"; (b) **item 5** thesis-primary metric — deferred in the meeting: *"For the other metric, we will leave it for the next meeting."* [31:59]; (c) cross-encoder model; (d) new item added 2026-08-23: **multiple-comparisons correction** — "Each `--compare` now reports **25 statistical tests** … and **no correction for multiplicity is applied to any of them**."
**Refs:** ADR 0006/0007/0009/0010/0017/0018/0029; #10, #11, #23, #29, #30, #40. **Live?** Yes — ADR 0029's suite was *implemented, not adopted*.

### #12 — Guided vs unguided decomposition on MuSiQue (opened 2026-08-17)
**Asks:** three conditions (unguided / oracle_guided / unguided_capped) on the ADR 0007 pinned 600, both logged.
**State:** **done but open.** exp-004 (Mistral-7B) + exp-005 (Qwen3.5-9B) complete; re-scored at exp-011, paired-tested at exp-014; #46 §1 lists it "complete". Last comment 2026-08-19T10:02:48Z is pre-run, so the issue text never records the result. **Blocked on:** nothing — effectively closed by exp-004/005 + exp-014. **Refs:** ADR 0006 item 6, 0007, 0010, 0014, 0015; PR #21.

### #13 — LoRA fine-tuning baseline for the decomposer (opened 2026-08-17)
**Asks:** a config-driven, seeded LoRA arm whose metrics sit beside the prompting arm on the same eval set.
**State:** **done but open, one arm outstanding.** PR #24 merged; exp-001 trained (loss 0.140, 2 epochs); exp-008 `full_train` evaluated; exp-026 = pool-only QLoRA (#50 A2). The stratified `pool_2000` control is *"Not run"* (#50, 2026-09-20). **Blocked on:** nothing hard; unscheduled. **Refs:** ADR 0012 (items 7/8 confirmed 2026-08-19), 0014, 0034; exp-001/008/013/026.

### #14 — Pool construction: add a clustering strategy to the sweep (opened 2026-08-17)
**Asks:** clustering strategy inside the existing sweep, metrics beside the other strategies.
**State:** **done but open.** exp-006 (CPU) + exp-010 (18 GPU cells) complete; truncation confound audited and **DISMISSED** (2026-08-23T01:03:02Z, commit `174920c`); pool adopted `size2000_imbalanced` (ADR 0028 item 1). **Live?** "Done when" is met; what keeps it relevant is **#52** — a plumbing defect there *"would invalidate the pool-construction conclusion and #14 with it."* **Refs:** ADR 0006/0021/0028; PR #34.

### #15 — Hop-matched retrieval: mixed vs oracle-hop-matched vs router-hop-matched (opened 2026-08-17)
**Asks:** three retrieval conditions on the same eval set.
**State:** **2 of 3 conditions.** exp-009 posted 2026-08-27T14:44:51Z: hop-count EM 0.5150 → 0.6000 (McNemar p=0.00062); step F1, ordered step acc., ROUGE-L, EM all non-significant.
**Blocked on:** (1) `router-hop-matched` **never run** — *"Precondition: issue #47"* (running it today measures the 34% parse defect); (2) **size-matched control never built** — *"Without it, exp-009 supports 'consistent with hop matching helping', not 'caused by hop depth'."* That is a lead/agent CPU-only job over an existing pool; #47 is the gating defect.
**Refs:** ADR 0006, ADR 0022, ADR 0024, ADR 0030; exp-007, exp-009, exp-014; PR #39; #23, #27, #40, #47.
**Live?** Yes — genuinely incomplete.

### #16 — End-to-end evaluation: MuSiQue answering backend + MetaQA compile-execute wrapper (opened 2026-08-17)
**Asks:** decompositions scored on whether they lead to the right answer.
**State:** **partially done.** PRs #32/#42 merged (MuSiQue taxonomy + reader; exp-017 is the reader row). This is "S4", the **only step of #46's sequence still open**. No comments on the issue itself.
**Blocked on:** **Jahid** — the MetaQA/GRAG half is an external dependency (GRAG is the supervisor's system; 7 questions listed in PR #42), *"flagged and never stubbed, per ADR 0023 consequence 3."*
**Refs:** ADR 0006, ADR 0019, ADR 0023; exp-017; #19, #41 (instrument 2), #46 S4.
**Live?** Yes.

### #19 — Triage the 2026-08-12 transcript cross-check (opened 2026-08-17)
**Asks:** Jahid confirms/dispatches ~8 bullets (top-20/top-5 status, 6-vs-8-line cap, fine-tuning sequencing, unrecorded ideas, the stats substitution, error-asymmetry wording, GRAG handover, the missing pipeline figure).
**State:** **partially done.** Triaged 2026-08-20 — that triage spawned #23, #27, #28, #29, #30 and ADRs 0017/0018 (PR #31). No comment since.
**Blocked on:** **Jahid** for the residue (length-cap reported default, top-k demotion to ablation, fine-tuning sequencing, the supervisor's pipeline figure "add when he sends it").
**Live?** Partly — most bullets are dispatched into their own issues; the record-quality residue is unrecorded as closed.

### #23 — Router v2: predict hop count as a regressor (opened 2026-08-18)
**Asks:** a regressor router evaluated against the v1 classifier framing, per-hop-depth error reported.
**State:** **not started.** No comments at all.
**Blocked on:** **Jahid + supervisor** — target encoding, loss, rounding rule, headline metric (MAE vs exact-hop vs both) are explicitly "not an agent's call". Also sequenced after #12 (met) and now bounded by ADR 0030.
**Live?** Formally yes, practically parked: ADR 0030 removed the router from the critical path and states it "does not refute issue #23's regressor design (untested — merely bounded by the same ceiling)". #46 lists it "open and unscheduled"; #50's supervisor assignment issued **no router instruction at all**.
**Refs:** ADR 0008, ADR 0010, ADR 0030; #12, #15, #27, #47.

### #27 — Few-shot-prompted router + with/without-router MuSiQue evaluation (opened 2026-08-20)
**Asks:** measured answer to "does the router earn its place".
**State:** **done-but-open.** Answered 2026-08-23T01:45:16Z by **ADR 0030** on exp-012 + exp-014 evidence. Router accuracy 0.3833 vs ~0.333 chance; with/without non-significant on every metric at n=600 — but that null is **explicitly not the basis**; the oracle-ceiling argument is (+0.082 hop-EM vs fine-tuning's +0.202, p=1.12e-15).
**Live?** Effectively closed by ADR 0030 (commit `cff8a30`). What survives is the defect in **#47**.

### #28 — MetaQA+MuSiQue training-data mix: decision needed (opened 2026-08-20)
**Asks:** whether fine-tuning data mixes MetaQA with MuSiQue or stays MuSiQue-only.
**State:** **not started** — no comments. Body: *"This is a research-direction call — **Jahid + supervisor decide**; this issue is the durable home so it doesn't get lost."* Current state: exp-001 trained on full MuSiQue train only (ADR 0012).
**Blocked on:** Jahid + supervisor. #46 §4.9 lists it "open and unscheduled". **Live.**

### #40 — Composite score's reference_validity term is decided by 2/600 items (opened 2026-08-22)
**Asks:** decide whether `_REF_RX = r"\[#(\d+)\]"` is a defect or an intended v1 definition, and whether composite is dropped or repaired.
**State:** **done-but-open, with a reversal on record.** Open question 1 answered *"defect in effect"* (2026-08-23T00:54:35Z) with the recommendation to **freeze rather than repair**, reversing the earlier same-day plan to ship a repaired baseline. Shipped: the unblended six-term suite (ADR 0029, PR #48); `composite_score`/`_REF_RX` frozen byte-identical and marked **legacy**; exp-016 logged all 12 pinned-600 arms.
**Blocked on:** **supervisor** for adoption only (issue #6 item 5). The optional diagnostic repair is **Jahid's discretionary call** — *"Say the word."*
**Live?** Mostly discharged by ADR 0029 / PR #48 / exp-011 / exp-016; the adoption question belongs to #6 item 5.

### #41 — Generalisation: MuSiQue-trained decomposer vs dynamic prompting (opened 2026-08-22)
**Asks:** two instruments — (1) `generalisation_2_3hop` within-MuSiQue, (2) cross-dataset MetaQA end-to-end.
**State:** **partially done.** Instrument 1 complete (exp-013, comment 2026-08-23T07:01:14Z): on the unseen 4-hop depth, hop-count EM 0.145, under-decomposition 0.845, exact structural agreement 0/200, chain_validity 1.000; vs exp-008 all-hop, hop-EM −0.2650 (p=5.92e-10). Reading: *"the fine-tuned win is coverage-bound, not dataset-bound-in-general."*
**Blocked on:** instrument 2 = **#16's GRAG half = Jahid's external dependency.** **Live.**

### #46 — Execution sequence to the end-of-October ceiling (opened 2026-08-23) — see §3 below.

### #47 — Few-shot router parses only 34% of its own outputs (opened 2026-08-23) — see §4 below.

### #49 — Model coverage: ADR 0030's premise is Mistral-only (opened 2026-08-27) — see §4 below.

### #50 — Supervisor assignment 2026-08-27 (opened 2026-09-14) — see §3 below.

### #51 — Pool-size sweep at 1k granularity: 2k…8k (opened 2026-09-14)
**Asks:** replace `imbalanced_sizes: [1000,2000,4000,8000]` with seven 1k-step sizes (2k…8k).
**State:** **not started, deliberately.** Jahid 2026-09-14 (ADR 0031 item 8); supervisor downgraded it himself — *"not a high priority, it's an idea"* [28:56]. ~3–3.5 GPU-hours.
**Blocked on:** **#50** (must not start while the three-model block is outstanding — now complete, so this gate has lifted) and **#52** — *"If that investigation finds a plumbing defect, this sweep would re-measure the same defect seven times. Settle that first."* Two open scoping questions are **Jahid's**: imbalanced-only vs re-crossing constructions.
**Live?** Yes, and now nearest-to-runnable of the deferred items.

### #52 — Why do the imbalanced and clustered pool arms produce matching numbers? (opened 2026-09-14)
**Asks:** establish whether the two arms' per-item predictions are byte-identical or merely average alike, and which of (a) real null / (b) plumbing defect / (c) degenerate clustering holds, with counts. Output to `docs/analysis/2026-09-14-pool-arm-identity-investigation.md`.
**State (corrected by the lead, 2026-09-20):** **done but open.** The issue carries no comment, but the work is on `main`: `docs/analysis/2026-09-14-pool-arm-identity-investigation.md` + `.json` (commit `2354c82`) and log row **exp-019** (2026-09-14). Verdict there: **no defect** — pools share 201/2000 ids, 0/750 identical top-5 exemplar lists, 0/150 identical prompts, 688–700 of 750 generations differ per cell; the exact 0.5133 tie is a 129/129 McNemar split in one cell of six. What is missing is a closing comment on the issue, not the investigation.
**Blocked on:** nothing — needs the closing comment and Jahid's word on whether it closes.
**Gates lifted:** #51 is no longer gated on a defect hunt; #14's null reading stands.
**Refs:** ADR 0021, ADR 0028; exp-010, exp-018; #14, #46 §4.1, #50, #51.

### #53 — Run note reports model-config `few_shot.k` (3) on reranked runs that showed 5 exemplars (opened 2026-09-19)
**State:** **not started.** Found by the Gate 1 review of `feature/50-fewshot-ordering`; pre-existing defect at `run_decomposer.py` ~:3039-3041. Shared code → branch + Gate 1. Cosmetic-but-record-corrupting: *"every reranked run's notes.md states the wrong k."* Corroborated by #50's block comment, which lists the same defect in the exp-021/022 log-row prose.
**Blocked on:** nothing; agent work. **Live.** Refs #50.

### #54 — Run snapshot records `git.commit` at artifact-write time, not at launch (opened 2026-09-20)
**State:** **not started; workaround in place.** exp-020 stamps `cc13bee`, 40 min after launch; code executed was `990be38`. Workaround: runner records the launch SHA by hand in the pre-run row from exp-021 on.
**Blocked on:** nothing; agent work, branch + Gate 1. **Live.** Refs #50.

### #55 — `print_model_params.py` and the runtime ceiling assertion disagree by 0.7B on Qwen3.5-9B (opened 2026-09-20)
**State:** **not started.** Script reports 9,653,104,368 (all checkpoint tensors); runtime `model.num_parameters()` reports 8,953,803,264 — a 15-tensor `mtp` head, difference 699,301,104. Both under the 1.5e10 decomposer ceiling, so no decision turns on it today; the exp-027 launch gate killed one attempt at 115 s.
**Blocked on:** nothing; agent work, branch + Gate 1. **Live.** Refs #50.

---

## 2. CLOSED ISSUES (11) — one line each

- **#1** Confirm/replace the README idea paragraph — closed 2026-08-12 once Jahid gave his word on the transcribed paragraph.
- **#2** Settle the compute surface and fill `docs/compute.md` — closed 2026-08-17 when the `_TBD_` fields (surface, access, rebuild command, run-lock path, metrics path) were filled.
- **#3** Port v1 pipeline code into v2 — closed 2026-08-12 by PR #7 (router, decomposer, evaluation, prep tooling), with the Gemini judge explicitly excluded.
- **#4** Carry v1 decisions into v2 docs — closed 2026-08-12 by PR #5 (retroactive ADRs + `docs/prior-work.md`).
- **#9** Punctuation-only-variance smoke fixture — closed 2026-08-18 by PR #22 / commit `6bcca64` ("Closes #9").
- **#10** Router parameter cap conflicts with the best router result — closed 2026-08-17 by Jahid lifting the ~600M `router_max_params` (ADR 0008); `router_max_params` now 8e9.
- **#11** Directional step-count error + paired significance — closed 2026-08-17 by PR #17 (ADR 0009 protocol: paired bootstrap CIs + exact McNemar).
- **#18** `cta` pool 100% full, 2.2T in ZFS snapshots — closed 2026-08-18 when the admin freed the pool; interim work ran from a `/tmp` clone.
- **#20** Six minor findings from PR #17's review — closed 2026-08-18 by PR #22 / `6bcca64` ("Closes #20").
- **#29** Composite-score literature/bias check — closed 2026-08-20 by PR #38 (the analysis note that later argued against repairing the composite, feeding #40).
- **#30** Paired t-test alongside bootstrap CIs + McNemar — closed 2026-08-20 by PR #36 / `b0a9ce8` ("Closes #30").

---

## 3. #46 AND #50, STRUCTURE REPRODUCED

### Issue #46 — Execution sequence to the end-of-October 2026 ceiling
Body sections: §1 state of play (verified 2026-08-23, `main` at `0ed7fd1`), §2 the sequence, §3 record-quality items, §4 decisions that gate the sequence.

**§2 — the sequence (order is ADR 0023 item 2, not chosen here)**

| step | item | latest recorded state (comment 2026-08-23, "Sequence state") |
|---|---|---|
| **S1** | Router: does it earn its place? (#27) | **complete** — exp-012 logged; accuracy 0.3833 vs ~0.333 chance; with/without non-significant at n=600; keep/drop recorded as **ADR 0030** (router leaves the critical path) on the *oracle ceiling*, not on the null; #47 opened. |
| **S2** | Best-config prompting vs the fine-tuned decomposer | **complete** — preflight verdict "(b), not answerable from committed artifacts"; exp-014 (CPU half) then exp-015 (GPU cell). All 10 statistics favour fine-tuning, all significant. Opposite failure directions: prompting under 0.2317/over 0.2283; fine-tuned under 0.2567/over 0.0333. |
| **S3** | Generalisation within MuSiQue (#41 instrument 1) | **complete** — exp-013. Hop-EM −0.2650 (p=5.92e-10), under-decomposition +0.2950 (p=4.63e-11) vs all-hop; **chain_validity not significant**. |
| **S4** | End-to-end evaluation (#16) | **only step still open.** MuSiQue half proceeds on PR #42; MetaQA/GRAG half blocked on Jahid's external dependency (7 questions in PR #42). Also #41 instrument 2. |

**§4 — decisions that gate the sequence, item by item, latest state**

| # | item | state |
|---|---|---|
| 1 | Which pool is adopted | **discharged** — delegated to the lead (ADR 0028 item 1, Jahid: *"pick any of the 3"*); chosen `size2000_imbalanced`; truncation confound audited and dismissed. |
| 2 | Router keep or drop | **discharged** — delegated to the lead (ADR 0028 item 2, *"you decide if it stays or not"*); **ADR 0030**. |
| 3 | The composite score | **implemented, not adopted** — authorized (ADR 0028 item 3); six-term unblended suite shipped (ADR 0029, PR #48); legacy composite frozen; exp-016 logged all 12 arms' values. |
| 4 | Metric adoption for the Break-faithful family | **open — supervisor** (= #6 item 5). |
| 5 | Qwen 10B ceiling | superseded — the raise now lives in `decomposer_max_params` at 1.5e10 (ADR 0031 item 4 / ADR 0032 item 2), **still pending supervisor confirmation**. |
| 6 | ADR 0027 point 5 (tools carved out of the log rule) | **open — needs Jahid's nod.** |
| 7 | ADRs 0024/0025/0026 implementer designs | **open — supervisor confirmation pending**; the 2026-08-23 comment extends the list to 0015, 0024, 0025, 0026, 0029, 0030. |
| 8 | Model swap (Granite 4.1 8B, survey `f920fb8`) | **open — awaiting Jahid's go.** No later mention; not in #50's scope. |
| 9 | Training-data mix (#28) and router-as-regressor (#23) | **open and unscheduled.** |
| + | Multiple-comparisons correction (added in the last comment) | **open — 25 uncorrected tests per comparison (#6).** |

### Issue #50 — Supervisor assignment (2026-08-27)
Closed by Ahmmad with *"Only those. Don't do anything further."*

| item | assignment | latest recorded state |
|---|---|---|
| **A1** | Decomposition on one fixed 2,000 pool with three decomposer models (Mistral-7B, Qwen2.5-7B, a Qwen above 7B) | **done.** exp-020 (Mistral 4-bit, 3/13 significant vs exp-015 fp16), exp-021 (Qwen2.5-7B, 6/13 vs exp-020), exp-022 (Qwen2.5-14B, 7/13 vs exp-021). Arms were changed by the lead and **flagged for Jahid's veto**: Mistral-7B vs Qwen2.5-7B holds size, Qwen2.5-7B vs Qwen2.5-14B holds family; *"Qwen3.5-9B fits neither."* exp-027 later added Qwen3.5-9B anyway (commit `4e87cf4`, "best-on-overlap, worst-on-chaining"). |
| **A2** | Fine-tune Mistral on the 2,000 pool only | **done.** exp-026 (2,000 ex., 250 steps) vs exp-008 full_train (19,938 ex., 2,494 steps): 4/13 significant, all ≤2.7 pp favouring full_train; vs exp-015 prompting, 11/13 favouring the fine-tune. ADR 0034. |
| **A3** | Jahid personally understands every metric | **Jahid's own item** — no recorded state. |
| **A4** | Fixed 45-item anchor set (15/hop), hand-inspected every experiment from now on | **built, not hand-inspected.** ADR 0033, `configs/anchor_set_45.json`, merged. Anchor sheet for all 8 arms at `runs/anchor/2026-09-20-all-arms.md` (on the box, not committed — question text). *"**Not yet hand-inspected (A4/A7 — Jahid).**"* A qualitative analysis note did land (`docs/analysis/2026-09-20-anchor-set-qualitative-analysis.md`, plus a seed-43 replication). |
| **A5** | Few-shot ordering: current vs reversed vs least-complex-first | **2 of 3.** Knob merged (ADR 0032 amendment). exp-023 Mistral reversed (0/13), exp-024 Qwen2.5-7B reversed (1/13, step_f1 +0.015), exp-025 Qwen2.5-14B reversed (0/13); exp-028 Qwen3.5-9B reversed — the **only G3 arm where any report-card term moves** (commit `5284ab3`). **`complexity_asc` not run — undefined (ADR 0032 item 5).** |
| **A6** | Report per hop count (2/3/4) as well as overall | evaluator already emits `per_gold_hop_metrics`; **per-hop significance does not exist** — no later comment says it was added. |
| **A7** | Eye-to-eye Qwen vs Mistral on bad examples over the anchor set | **Jahid's, not done** — same line as A4. |
| **A8** | Slide restructure + ~18 named fixes, deck then goes to him | decided "before any GPU work" (ADR 0031); `Slides/` holds three untracked PDFs; no comment records handover. |

**§ Decisions needed before any run (7)** — all answered by Jahid 2026-09-14, ADR 0031 (comment 2026-09-14T08:05:07Z): pool = imbalanced 2,000; eval set = ADR 0007 pinned 600; condition = **all arms unguided**; ceiling 1e10 → 1.5e10 *pending supervisor confirmation*; data to GitHub = **No**; slides first; finer sweep deferred to #51. **§7.5 left open:** what "least complex to most complex" means (A5) — *"Ahmmad did not say, and picking one silently would be inventing methodology."*

**Known record defect declared in the block comment:** exp-021/022 log-row prose says "k=3 exemplars"; config and logged prompts show k=5 — corrected in the comment and report §2.2, not in the append-only rows (this is #53's defect).

**Note:** #50's latest comment is 2026-09-20T03:47:12Z and predates exp-027, exp-028 and the block-report addendum (`f1bbdad`), so the issue understates completed work by two runs.

---

## 4. #47 AND #49 — FULL STATE

### #47 — Few-shot router parses only 34% of its own outputs: 394/600 default to hop=2
**Measured (pinned 600, exp-012):** 394/600 responses failed to parse and were scored at `default_hop=2`; headline accuracy 0.3833 (230/600) is substantially that fallback landing on a set where a third of items are genuinely 2-hop; hop-2's 0.875 is *"mostly free credit"* — **123 of its 175 correct rows were defaulted, not parsed**; on parsed responses alone the router is right **≈107/206 ≈ 52%**.
**Status in body:** *"Not scheduled."* Per ADR 0030 it is **the precondition** for any future router work: fix first, then re-run exp-012's comparison — *"the existing null cannot be reused."*
**Comment 2026-08-27T14:30:11Z — four implementation gaps "not written down anywhere":**
1. **Raw responses were never persisted.** `detailed_results.json` carries six fields; *"The model's response text is not among them."* Consequence: *"it is not an analysis task over existing data, it is a re-generation"* — needs the GPU plus a response-logging field in `write_detailed_results`.
2. **exp-012 ran v1's weakest router model** (`mistral_7b_instruct`, 76.67% in v1 with a 2-hop collapse at 34%), while **Qwen2.5-3B-Instruct** — v1's best at 90.33% — has been admissible since ADR 0008 lifted the 600M cap and has **never been run as a MuSiQue router**.
3. **Three untested mechanical suspects:** the digit class `[234]` makes a "1" *unparseable rather than wrong*; `response_truncate_at: ["\n"]` discards anything after a preamble line; `max_new_tokens: 128` was inherited from a thinking-model justification that does not apply to Mistral.
4. **The MetaQA 90% and the MuSiQue 38% are not comparable** — dataset, hop range, prompt and n all differ — *"and nothing says so"*; both appear in the slide deck on different slides with no note.
**Verdict:** ADR 0030 closed the router as a *decision*; #47 is open as a *defect*, unscheduled, and it also blocks #15's `router-hop-matched` arm and any #23 work.

### #49 — ADR 0030's oracle-ceiling premise is Mistral-only; exp-005 contradicts it
**The contradiction (body, from `experiments/exp-014/metrics.json`):** ADR 0030's first argument — an oracle *"moves nothing in step content"* — is exp-004's Mistral numbers (+0.082 hop-EM; step_f1 0.2061 vs 0.2039). exp-005 (Qwen3.5-9B), same conditions, same pinned 600, same seed, same paired protocol, gives **+0.353 hop-EM (0.5200 → 0.8733, p=5.0e-48) plus significance on step F1, ordered step accuracy, ROUGE-L and EM.** *"The sentence 'an oracle moves nothing in step content' is true of the model it was measured on and false of the other model in the same experiment."* ADR 0030 never mentions exp-005 or Qwen (verified: no occurrence of "Qwen" or "exp-005" in the record).
**Coverage (body §2):** every logged MuSiQue arm except exp-005 was Mistral-7B — exp-004, exp-008, exp-009, exp-010, exp-012, exp-013, exp-015, exp-017. *"exp-005 is the only Qwen3.5-9B row, and nothing was ever run on it again."*
**Deck defect (§3):** `Slides/deck_corrections.md` said *"Only GED significant"* for the Mistral oracle pair, dropping `hop_count_exact_match` (p=2.41e-04); the error reached **slide 25** of `Artifact slides review.pdf`. Corrected in the corrections file 2026-08-27; **the PDF still carries the old line and needs a regenerate.**
**Latest state (comment 2026-09-20T11:26:39Z):** *"model coverage on the pinned 600 is no longer Mistral-only for the unguided prompting method — exp-020 (Mistral-7B), exp-021 (Qwen2.5-7B), exp-022 (Qwen2.5-14B) and exp-027 (Qwen3.5-9B) all ran the same recipe at 4-bit. … What this does NOT yet cover: ADR 0030's oracle-ceiling premise (guided arms) — this block ran unguided only per ADR 0031 item 3, so the specific contradiction exp-005 raised (Qwen3.5-9B under oracle guidance) is still unmeasured on this axis. Leaving open for that."*
**Blocked on / who decides:** the issue's own ordering puts **Jahid + supervisor** first — *"Decide … whether the thesis claims are model-general or Mistral-scoped — this is the question the two items above actually pose, and it gates whether anything below is worth GPU time."* Step 2 (cheapest informative run = the Qwen3.5-9B fine-tune arm) and step 3 (regenerate slide 25) follow.
**Live?** Yes. #50's block partly discharged §2 (coverage on the unguided axis) but not §1 (the guided/oracle contradiction) or §3 (the slide regenerate).

---

## 5. QUESTIONS EXPLICITLY WAITING ON JAHID OR THE SUPERVISOR (quoted)

**Supervisor**
1. #6 (2026-08-17 / re-stated 2026-08-20): item 5, thesis-primary metric — *"For the other metric, we will leave it for the next meeting."* [31:59, 2026-08-12 meeting]. Still *"stays deferred to a supervisor meeting."*
2. #6 (2026-08-20): item 4 retrieval masking — *"Retrieval masking default → **not settled** — typed masking stays open with the supervisor."*
3. #6 (2026-08-20T05:23:48Z): *"The **cross-encoder re-ranker slot** (`cross-encoder/ms-marco-MiniLM-L-12-v2` in configs) has no recorded decision behind it and is a **third open item** here."*
4. #6 (2026-08-23): multiplicity — *"**The decision** is whether the thesis (a) reports uncorrected tests with the multiplicity disclosed … (b) applies a correction … or (c) declares one pre-registered primary metric per comparison."*
5. #46 §4.7 / 2026-08-23 comment: *"supervisor sign-off on ADRs 0015, 0024, 0025, 0026, 0029, 0030."*
6. #50 (2026-09-14) / `configs/model_limits.json`: the 1e10 → 1.5e10 decomposer raise is *"Pending supervisor confirmation."*
7. #40 (2026-08-22, 2026-08-23, repeated): *"whether the suite becomes the thesis's **primary** metric stays with the supervisor (issue #6 item 5)."*

**Jahid (alone, or Jahid + supervisor)**
8. #16 (body, 2026-08-17): *"GRAG is the supervisor's system — access, interface and how sub-questions are handed to it are unknown here and need Jahid to obtain them; that blocks the MetaQA half."* Re-stated in #46 S4 and #41 instrument 2.
9. #19 (body, 2026-08-17): *"top-20 / top-5 were not fixed in the meeting … Confirm they stand as the fixed method, or demote the exact k values to ablation status."* Plus the length-cap default (6 vs 8 lines) and fine-tuning sequencing — *"Done when Jahid has confirmed/dispatched each bullet."*
10. #19 (body): *"The pipeline figure the supervisor screenshotted ([1:33:00]) is nowhere in the repo — add when he sends it."*
11. #23 (body, 2026-08-18): *"**Open design questions — Jahid + supervisor, not an agent's call:** target encoding (integer count vs continuous), loss, how a continuous prediction rounds back to an integer hop for routing, and the headline metric (MAE vs exact-hop accuracy vs both)."*
12. #28 (body, 2026-08-20): *"This is a research-direction call — **Jahid + supervisor decide**."*
13. #40 (2026-08-23T00:54:35Z): *"**Jahid's call, not blocking anything:** if you want the repair anyway — as a diagnostic showing what the defect was worth … Say the word."*
14. #46 §4.6 (2026-08-23): *"**ADR 0027 point 5** — carves tools out of 'no experiment exists unless it is in the log'; needs Jahid's nod."*
15. #46 §4.8: *"**Model swap** (Granite 4.1 8B via the exp-004 protocol, survey `f920fb8`) — awaiting Jahid's go."*
16. #49 (2026-08-27, suggested order item 1): *"Decide (Jahid + supervisor) whether the thesis claims are model-general or Mistral-scoped — this is the question the two items above actually pose, and it gates whether anything below is worth GPU time."*
17. #50 (2026-09-14, §7.5 still open): *"what 'least complex to most complex' means when ordering exemplars (A5). Hop count is the obvious candidate; question length and gold-chain length are others. Ahmmad did not say, and picking one silently would be inventing methodology."*
18. #50 (2026-09-14): the A1 arm substitution is *"**Arms changed — flagged for Jahid's veto**"* — no veto or confirmation is recorded.
19. #50 (2026-09-20): *"**Not yet hand-inspected (A4/A7 — Jahid).**"*
20. #51 (2026-09-14, "Open before it runs"): *"Whether the sweep re-runs at the adopted **imbalanced** construction only … or re-crosses construction strategies."*
21. #52 (2026-09-14): *"Whether a log row is owed follows the exp-018 precedent and is Jahid's call."*
22. #15 (2026-08-20): router-hop-matched predictions source — *"Where predictions come from … is a Jahid + supervisor call, tied to #23/#27."* Plus the size-matched control, *"flagging it as a design option for Jahid rather than deciding it."*

---

## 6. CROSS-CHECK: OPEN ISSUES vs UNMERGED LOCAL BRANCHES

`main` = `f1bbdad`, 53 commits ahead of `origin/main`. `git log main..<branch> --oneline` and a branch-only-file diff:

| branch | `main..branch` commits | branch-only files | verdict |
|---|---|---|---|
| `feature/12-guided-vs-unguided-alt` | `c1b4aed` feat: three MuSiQue decomposition conditions in one decomposer config | `src/step_cap.py`, `tests/test_step_cap.py`, `tests/test_musique_conditions.py`, one fixture | **Real unmerged content** *(corrected by the lead, 2026-09-20: the two pieces the branch was held for were re-implemented in PR #21's squash `1ccea11` — `src/step_lines.py` and `tests/test_decomposer_conditions.py:610` — so this is a superseded duplicate, not unmerged work; see appendix E §1)*. #12's 2026-08-18T15:51:05Z comment: *"#12 turned out to be implemented twice … **PR #21 is canonical**. The alt branch is held as a cherry-pick source for two pieces PR #21 lacks … The alt branch gets deleted once that's settled."* Never deleted; the reviewer's verdict is not recorded on the issue. |
| `feature/13-lora-finetuning` | `a6892bb`, `eab9a3e`, `aa8b0b7` | **none** | Superseded — PR #24 merged (issue #13, 2026-08-18T22:21:16Z: *"PR #24 merged after Gate-1 review … at a6892bb"*). Stale branch pointer only. |
| `feature/40-literature-metrics` | `f96b3fe`, `b770c42`, `b89441f`, `610431e`, `c832721` | **none** | Superseded — landed as PR #44 (+ record fixes PR #45), per #46 §1. Stale pointer; note it is checked out in a scratchpad worktree. |
| `feature/50-anchor-set` | `db16e82` | `configs/anchor_inspect.json`, `configs/anchor_set_ids.json`, `scripts/anchor_inspect.py`, `docs/adr/0032-the-fixed-45-item-anchor-set-…md` | **Superseded by a different design**, not by a merge. `main` carries `e95e550` + ADR **0033** with `configs/anchor_set_45.json`, `scripts/build_anchor_set.py`, `scripts/render_anchor_set.py`, `src/anchor_set.py`. The branch's ADR 0032 draft number was reused on main for the ordering knob. Live in a worktree at `/cta/users/fyilmaz/worktrees/50-anchor-set`. |
| `feature/50-qwen-decomposer-models` | `9f0c2fd`, `b03f1b1` | **none** | Superseded — `main` has `feef949` (Qwen2.5-7B/14B folders, `decomposer_max_params` 1.5e10 via its own key) and `29c2ef1` (ADR 0031/0032 amendment). Stale pointer in a scratchpad worktree. |

**Net:** only **`feature/12-guided-vs-unguided-alt`** holds code that exists nowhere on `main` (the `step_cap` module and its two test files), and it is tied to open issue #12. **`feature/50-anchor-set`** holds an alternative anchor-inspection design (`anchor_inspect.py` + `anchor_set_ids.json`) that main's ADR 0033 implementation replaced; #50 A4 is satisfied by main, not by that branch. Four stale `worktree-agent-*` branches also sit at old commits (`0ab6abb`, `b6329ca`) and carry nothing not on main.

Everything else referenced by open issues is on `main` — including all of #50's block (exp-020…exp-028) — but **local only**: `main` is 53 commits ahead of `origin/main` and this box cannot push, so an issue reader on GitHub sees none of it.
