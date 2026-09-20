# Road plan to the defence — a proposal, 2026-09-20

- **Written:** 2026-09-20 by the lead session, at Jahid's request ("After the investigation, you
  should come up with a road plan that we could apply").
- **Built on:** the investigation record
  [`docs/analysis/2026-09-20-state-of-the-work-investigation.md`](../analysis/2026-09-20-state-of-the-work-investigation.md)
  and its five appendices; the 2026-08-27 assignment (issue #50, phase plan
  [`2026-09-14-post-supervisor-meeting-phase.md`](2026-09-14-post-supervisor-meeting-phase.md));
  the block report (`docs/analysis/2026-09-20-issue-50-block-report.md`).
- **Status: a proposal.** It sequences work the record already names against the fixed calendar.
  Every GPU item beyond §2 is a *candidate for Ahmmad to choose from*, not a plan the system has
  adopted — his last instruction was "Only those. Don't do anything further", and that instruction
  is discharged on the machine side only — A3, A4/A7 and A9 are Jahid's and still outstanding (§2.1). Where a call is Jahid's or Ahmmad's, §6 says so and gives the default the
  lead will apply if it goes unanswered, per the standing delegation pattern (ADR 0028, 0032).
  Nothing here changes a standing constraint, a gate, or the research direction.

---

## 1. The calendar, as arithmetic

| date | what | days from today |
|---|---|---|
| 2026-09-20 | today; #50 block complete on the machine side | 0 |
| 2026-10-31 | **experiments end** (CLAUDE.md) | 41 |
| 2026-11-01 → 11-30 | thesis writing | 42 → 71 |
| ~2026-12-20 | defence | ~91 |

Forty-one days is **two supervisor-meeting cycles at most**. The 27 August block took 24 days from
assignment to results, 21 of them waiting (occupied GPU, then slides) and about 3 hours 24 minutes
of GPU time for the seven runs exp-020 to exp-026 (block report §7), plus 53 minutes for exp-027/028 (§9.1). The binding resource is not compute; it is the round-trip to Ahmmad and
Jahid's own reading time. The plan is therefore phased around **meetings**, not around runs.

## 2. Phase 0 — close the block and hold the meeting (now → ~2026-09-27)

Nothing in this phase needs a decision from anyone but Jahid, and nothing in it is new research.

### 2.1 Jahid's own items (cannot be delegated)

| item | what "done" looks like | source |
|---|---|---|
| **Push `main`** | 53 local commits reach `origin/main`; verified with `git status -sb` showing no "ahead" | investigation §5 — the single largest risk |
| **A4 / A7 — read the anchor sheet** | `runs/anchor/2026-09-20-all-arms-plus-9b.md`: 45 items × 9 arms. Notes on what the failures look like, per model, in his own words | #50 (2026-09-20): "Not yet hand-inspected" |
| **A3 — the metrics** | able to say in the meeting what a SARI move without a GED move means | phase plan §2 A3: "this is for you" |
| **A9 — data to Ahmmad** | the block's `runs/` outputs shared outside the repo (ADR 0031 item 6) | #50 |
| **Book the meeting** | "Whenever you finish them, let's arrange a quick meeting" | 2026-08-27 [1:05:22] |

### 2.2 Record hygiene — CPU-only agent lanes, one fresh spawn each, all Gate 1

Small, unglamorous, and the write-up depends on every one of them. Batched so the machine idles
while Jahid reads.

| lane | work | why now |
|---|---|---|
| **H1** `ml-engineer` | one branch fixing #53 (run note prints the retrieval-file k), #54 (git provenance captured at launch and carried to write time), #55 (report both parameter counts, assert on the runtime one) | three known record defects; every future run note inherits them |
| **H2** `ml-engineer` — *conditional on D5* | per-hop paired significance in `compare_decomposer_arms.py` (bucket the aligned per-item file by gold hop; same tests; `underpowered` flag at n<30 already exists). It roughly triples the statistic count per comparison, so it inherits whatever multiplicity policy D5 sets; until then any per-hop test is reported with that disclosure | A6 asked for per-hop results; today they are descriptive only |
| **H3** `ml-engineer` | add `pytest` (and `pytest-timeout`) to `requirements.txt` so the suite runs from `.venv`; the 5 skips should become passes; fix the 14 value-returning tests | investigation §5 — the fine-tuning tests are silently skipped today |
| **H4** lead, inline | ADR index rows for 0031–0034; forward notes on ADR 0015 and 0006 §4; annotate both plan docs where ADR 0031-amendment and ADR 0033 reversed them; `experiment-runner.md` → `model: opus` | drift the reviewer will otherwise trip on |
| **H5** lead, inline | tracker: closing comments on #12, #14, #27, #40, #52 with the citing commits/exp ids; #50 updated with exp-027/028; #46 marked superseded by this plan | so GitHub agrees with `main` once pushed |
| **H6** slide lane | regenerate the deck: slide 25 correction (#49 §3), the block's results as two slides — overall, then per hop (A6) — from the master table (block report §4.14 + §9.1) | the deck goes to Ahmmad before the meeting |

The run lock and hand-typed experiment ids (investigation §5) are **deliberately not on this
list** — a code-enforced lock and an id allocator are the kind of tooling CLAUDE.md warns against
adding to a system this small, with six weeks left and one runner. Named so the omission is a
choice.

### 2.3 The meeting agenda — decisions only he can make

Bring the block report §0 table, the two per-hop slides, Jahid's anchor notes, and this list:

1. **The 1.5e10 decomposer ceiling.** Confirm, or exp-022/025 are withdrawn (block report §6.9).
2. **Are the thesis claims model-general or Mistral-scoped?** (#49). Decides whether Phase 1
   candidates C1 and C3 are worth GPU time.
3. **The contribution statement.** ADR 0006 item 5 says "a strategy for constructing the few-shot
   pool and retrieving examples"; the evidence says pool construction, hop-matching and ordering
   barely move quality while a 250-step fine-tune on the same pool wins everywhere (investigation
   §4 tension 1). Does the framing stay, or does the pool become the *training set* of the story?
4. **What "least complex → most complex" means** for exemplar ordering, or drop the third arm.
5. **One test or several.** He wants one eventually; 13 uncorrected statistics per comparison
   today (#6). Options are recorded on #6: disclose, correct, or pre-register one primary metric.
6. **The thesis-primary metric** (#6 item 5) and typed masking as the default (#6 item 4).
7. **MetaQA.** Zero v2 experiments; GRAG access still unknown (#16). Keep, or take his own
   2026-08-12 fallback — transcript [1:15:24], recorded verbatim in the cross-check as "If you want to
   shrink… to one dataset, continue with the music" (a transcription of "MuSiQue") — and move MetaQA to
   future work.
8. **Which Phase 1 candidates**, from §3, and whether repeats for variance are wanted in Phase 2.
9. The pool question he never answered: "Should I still be working on the pool itself?"

## 3. Phase 1 — the post-meeting block (meeting → ~2026-10-18)

Candidates only. Each is one factor moved on the block recipe (unguided, 4-bit, pinned 600,
`size2000_imbalanced`, k=5 reranked, seed 42) unless stated, so it lands on the axis where every
existing baseline lives. Every estimate is extrapolated from a committed log row and labelled with
it; none is a measurement of these runs. Log row before launch; three artifacts after.

| id | candidate | answers | needs | est. GPU (basis) | discharges |
|---|---|---|---|---|---|
| **C1** | oracle-guided arm on Qwen2.5-7B and Qwen2.5-14B | is "oracle moves count, not content" Mistral-only? | `prompt_guided.md` parity-guarded for both Qwen folders (small build) | ~15 + ~30 min (exp-021/022 wall clocks) | #49 §1; tests ADR 0030's premise |
| **C2** | the `pool_2000` seeded hop-stratified control fine-tune | does exp-026's result come from *this* pool or from any 2,000? | config only — the arm exists (#13) | ~26 min train + ~14 min eval (exp-026) | #13; separates exp-026's three factors |
| **C3** | Qwen2.5-7B QLoRA on the imbalanced pool | is fine-tuning's win model-general? | `train_lora.py` against a Qwen folder — check the loader and chat template; `ml-engineer` | ~30 min train + ~14 min eval (exp-026, exp-021) | #49 step 2 — "the cheapest informative run" |
| **C4** | end-to-end answer accuracy for exp-026 and the best prompting arm(s) via `run_answerer.py` | does the pool-only fine-tune's decomposition win reach the *answer*? | nothing new; predictions files exist | **unmeasured** — exp-017's per-cell wall clock is not in its committed artifacts | #16 MuSiQue half; extends exp-017's ordering beyond Mistral prompting |
| **C5** | `complexity_asc` ordering on the four models | A5's third arm | the definition (§6 D4); then the reserved value is unblocked | ~42 + 14 + 28 + 28 min ≈ 1 h 52 (exp-020/021/022/027 wall clocks) | A5 complete |
| C6 | Mistral rendered through `chat_template` (or Qwen through `plain`) | how much of the family effect is the template? | a prompt-style override per run | ~42 min on Mistral (exp-020), ~14 on Qwen2.5-7B (exp-021) | block report §6.5 |
| C7 | pool-size sweep 2k…8k at 1k steps | is 2,000 the right size? | config change (#51) | ~3.5 h (exp-010 per-cell) | #51 — supervisor: "not a high priority" |
| C8 | router parse fix + re-run of exp-012 | does a working router earn its place? | GPU re-generation (responses were never persisted, #47) | ~45 min + build | #47, #15 third arm, #23 — **no supervisor instruction exists** |

**The lead's recommended default set, if Ahmmad delegates the choice:** C1, C2, C3, C4 — the cited
components sum to about 129 minutes (roughly 2.2 GPU-hours) plus C4's unmeasured cost. Reasoning: each closes an *open question the record
itself raised* (#49, #13, #16) rather than opening a new axis; C1 and C3 settle whether the
thesis may speak of "models" or only of Mistral, which changes every sentence in the results
chapter; C2 is the missing control for the block's strongest finding; C4 gives the thesis an
answer-side number for its best system. C5 waits on a definition. C6 is a footnote. C7 and C8 are
not recommended: C7 re-measures a null at seven sizes, and C8 reopens a component the supervisor
has not asked about and ADR 0030 retired.

Sequence inside the phase: builds first (C1's prompts, C3's loader check), one run at a time
under `runs/run.lock`, `experiment-runner` owns launches, one `code-reviewer` delta pass per
build branch, a 5-row smoke before any decomposer-path change merges. ADR 0032 item 7 was block-specific — "if the three G1 arms come back
near-identical on the report card, G3 does not launch until the lead has reported that to Jahid" —
and does not carry over by itself; whether a like gate applies to this block is a §6 question (D12).

## 4. Phase 2 — freeze (~2026-10-18 → 2026-10-31)

- **A last-new-arm date, proposed for the meeting: 2026-10-24** (nobody has set one; CLAUDE.md fixes only
  31 October). The intent is that the last week goes to repeats and re-scores, not new questions.
- **Variance, if he asks for it:** greedy decoding at a fixed seed reproduces itself, so the only
  meaningful repeats are *training-seed* repeats of the headline fine-tune (exp-026 at two more
  seeds, ~40 min each) and, if the pool story matters, *pool-draw* repeats (`num_trials` > 1).
- **One master table.** Every arm ever placed on the pinned 600, re-scored under the current
  evaluator into a single committed JSON + Markdown — the block report §4.14 table extended. This
  is the results chapter's spine; it costs CPU only (exp-011/016 precedent).
- **Final anchor pass.** The 45 anchors rendered for every arm in the master table, read by Jahid,
  notes committed as an analysis file (question text stays on the box).
- **ADR closure.** Each pending "supervisor confirmation" flag either confirmed at the Phase 1
  meeting and marked, or carried into the thesis as an explicit limitation.
- **Data hand-off** to Ahmmad, outside the repo, of everything he will be asked to defend.

## 5. November — writing

Not the system's to structure; two things the record supports saying:

- **The material exists and is indexed.** ADRs 0001–0034 are the "why"; `experiments/log.md`
  and the master table are the "what"; `docs/METRICS.md` and the 2026-08-23 specification are the
  metric definitions; the block report §4 is already a plain-English metrics chapter draft; the
  investigation appendices B §3 and §4 are the legitimate-comparisons list and the established
  findings, with citations. Appendix B §5 is the limitations section's skeleton.
- **Ahmmad gave an ordering for the deck** — Motivation, Definition, Examples, Methodology,
  Research Questions, Experimental Setup (metrics, datasets, baselines), Results overall then per
  hop — and said results and evaluation belong in setup, not methodology. Using the same ordering
  for the thesis is the obvious default; whether it is the right one is his call, and worth one
  question at the Phase 1 meeting.
- **Agents draft from artifacts, Jahid writes.** Drafting a chapter from the ADR trail and the log
  is production and can be delegated; the argument is his. Any number in a draft cites a log row
  or the master table, or it is deleted.

## 6. Decisions this plan waits on — owner and default

| # | decision | owner | default if unanswered |
|---|---|---|---|
| D1 | 1.5e10 decomposer ceiling | Ahmmad | stays raised, flagged pending; exp-022/025 reported with the caveat |
| D2 | claims model-general vs Mistral-scoped | Ahmmad + Jahid | no default — the question is stated in the thesis as open, not resolved by an agent |
| D3 | contribution statement kept or re-framed | Ahmmad + Jahid | ADR 0006 item 5 stands; the tension is stated in the thesis, not resolved by an agent |
| D4 | definition of "complexity" for ordering | Ahmmad (Jahid may propose) | C5 does not run |
| D5 | multiplicity policy (one test, correction, or disclosure) | Ahmmad | disclose, uncorrected, as today (ADR 0009) |
| D6 | thesis-primary metric; typed masking default | Ahmmad | report card as primary (ADR 0029); typed as today |
| D7 | MetaQA kept or moved to future work | Ahmmad + Jahid | no MetaQA run is scheduled; #16's MetaQA half stays blocked and is stated as such |
| D8 | Phase 1 candidate set | Ahmmad | none run until he speaks; if he delegates, C1–C4 |
| D9 | repeats for variance in Phase 2 | Ahmmad | none |
| D10 | close #12, #14, #27, #40, #52 as done | Jahid | lead posts the closing comments; Jahid closes |
| D11 | delete the five superseded `feature/*` branches and prune the `/tmp` worktrees | Jahid | left as they are; nothing is lost either way |
| D12 | whether a near-identical-numbers stop gate (ADR 0032 item 7's shape) applies to Phase 1 | Jahid | none; the lead reports and waits, as item 7 said |

## 7. Risks, in the order they would hurt

1. **The unpushed 53 commits.** One filesystem incident and the block is gone. Push first.
2. **D1 goes against the raise.** Two arms and the size contrast disappear; the family contrast
   (exp-020 vs exp-021) survives.
3. **The meeting slips.** Every GPU week lost after the meeting is unrecoverable; Phase 0 is sized
   so the machine never waits on an agent, only on the two humans.
4. **Scope creep against "only those".** The candidate list is long; the recommended set is
   four. Anything not chosen at the meeting does not run.
5. **The anchor sheet stays unread.** A4/A7 were assigned as *Jahid's* habit; a thesis defended
   without his own qualitative reading of the failures is the outcome the supervisor was
   guarding against.
6. **Shared GPU.** Another user's process held 21 GiB for six days in September. Launches go
   through the detached waiter; nothing of anyone else's is ever killed.

## 8. Not in this plan

The jury (never built; future work per ADR 0006). Any new metric. Any change to the evaluation
set or the pool. Any closed-model judging. A code-enforced run lock or id allocator (§2.2). A
bigger model beyond the four already run — his "You may try" was permission, not assignment.

Refs #50, #49, #46, #16, #13, #6, #51, #47, #53, #54, #55; ADR 0006, 0028, 0030, 0031, 0032
