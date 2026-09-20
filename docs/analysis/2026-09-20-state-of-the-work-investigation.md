# State of the work — investigation record, 2026-09-20

- **Written:** 2026-09-20 by the lead session, at Jahid's request ("do a thorough investigation and
  research over all of the codebase and decisions so far. Record your investigation").
- **Method:** five read-only investigator lanes, run in parallel, each confined to one surface and
  each required to cite a file, commit, issue number or log row for every claim. Their ledgers are
  committed unedited as appendices under [`2026-09-20-investigation/`](2026-09-20-investigation/),
  except for two corrections the lead verified and marked in place (§7). This document is the
  synthesis; the appendices are the evidence.
- **Repo state:** `main` at `f1bbdad` (2026-09-20 14:26), **53 commits ahead of `origin/main`** and
  unpushable from this box; working tree clean except three untracked PDFs in `Slides/`;
  `runs/run.lock` absent; RTX 3090 at 10 MiB used; `/cta/users` 339 GB free; every model the
  decomposer and router registries name is in the local HF cache, including Qwen2.5-14B-Instruct.
- **Companion:** the road-plan proposal written from this record is
  [`docs/plan/2026-09-20-road-plan-proposal.md`](../plan/2026-09-20-road-plan-proposal.md).

Nothing here is a decision. Where a fact is unmeasured it is called unmeasured.

---

## 1. Where the work stands, in one paragraph

Six weeks of the experiment window remain (ceiling 31 October 2026). The supervisor's 2026-08-27
assignment (issue #50, items A1–A9) is **complete on the machine side**: nine runs (exp-020 to
exp-028) landed on 2026-09-20 alone, all on the ADR 0007 pinned 600, all logged before launch,
all with committed config snapshot, metrics JSON and run note, and a plain-English report exists
(`2026-09-20-issue-50-block-report.md` with its §9 addendum). What remains of the assignment is
Jahid's own: understanding every metric (A3), hand-inspecting the 45 anchors across the arms
(A4/A7 — the sheet exists at `runs/anchor/2026-09-20-all-arms-plus-9b.md`, nobody has read it),
sharing the data with Ahmmad outside the repo (A9), and the meeting he asked for ("Whenever you
finish them, let's arrange a quick meeting"). Fathom records no supervisor meeting after 2026-08-27;
unrecorded contact is not excluded, so this document is written against the 27 August assignment. The one open line item in his list is the third ordering,
`complexity_asc`, which the code refuses at load because "complexity" was never defined.

## 2. What the record establishes

Each line is ran-X-observed-Y on one stated evaluation set. Appendix B §1 and §4 carry the full
numbers; the exp ids are the citations.

1. **Fine-tuning beats prompting on decomposition quality, on the same 600, for every prompting
   arm and every model tried.** exp-008 (`full_train` LoRA) wins every statistic against exp-004
   (exp-014); exp-015, exp-020, exp-021 and exp-027 each lose 10 of 10 headline statistics to it,
   exp-022 loses 9 of 10. Gaps: step-F1 +0.13 to +0.16, hop-count EM +0.09 to +0.20, GED −0.15 to
   −0.18. It is also cheaper to serve — about 32% of the prompting arm's tokens per query.
2. **A fine-tune on only the 2,000-question pool reaches the full-split fine-tune.** exp-026 vs
   exp-008: tied on 9 of 13 statistics; the four that differ favour the full split by ≤2.7 pp and
   only one is a report-card term. Training cost about 10% of the full-split adapter's (1,568.8 s vs exp-001's 15,497 s).
3. **Model family and size move plan *length* and *chaining*, not step wording.** exp-021 vs
   exp-020 (family, 7B): 6 of 13 significant, none a wording metric. exp-022 vs exp-021 (size,
   Qwen2.5): 7 of 13, again no wording metric; the 14B halves under-decomposition
   (0.2933 → 0.1017) and lifts chain validity (0.9210 → 0.9838). Mistral over-decomposes (signed
   error +0.585), Qwen2.5-7B under-decomposes (−0.150).
4. **Exemplar order does not move the report card on three of four models** — 0/13, 1/13, 0/13
   (exp-023/024/025) and 3/13 on Qwen3.5-9B (exp-028: chain validity +3.5 pp, GED −1.9 pp), 4 of 52
   statistics in total against the supervisor's "up to 10%". The null is **cancellation, not
   insensitivity**: reversal rewrites roughly 82% of plans (491–496 of 600) with per-item swings up
   to ±0.5 that average out (anchor analyses, block report §9.2).
5. **4-bit costs Mistral nothing on the quality terms** (exp-020 vs exp-015: 10 of 13 tied; only the
   two length rates and the legacy composite move) but makes it 2.3× slower. Measured for Mistral
   only.
6. **An oracle hop count moves step count, not step content — on Mistral.** exp-004: hop-count EM
   +0.08 (p=2.4e-4), step-F1 and ordered accuracy straddle zero. The same shape appears for
   hop-matched retrieval (exp-009) and pool construction (exp-010). **On Qwen3.5-9B the oracle
   moves everything** (exp-005: hop-count EM 0.52 → 0.87, plus step-F1, ordered accuracy, ROUGE-L,
   EM). This is the model-generality problem issue #49 names, and ADR 0030 was written on the
   Mistral half only.
7. **The prompted router does not earn its place as built.** exp-012: accuracy 0.3833 against
   0.333 chance, with 394 of 600 responses unparsed and defaulted to hop 2; unguided vs
   router-guided 0 of 10 statistics significant.
8. **Pool construction is a measured null with a verified mechanism.** exp-010 shows no
   step-quality separation between imbalanced, balanced and clustered pools; exp-019 proved the
   arms were genuinely different (201/2000 shared ids, 0/750 identical exemplar lists, 688–700 of
   750 generations differ). The supervisor's "matching numbers" suspicion has a citable answer.
9. **Cross-encoder re-ranking does move quality** (exp-018: 9/9 pairs positive on ROUGE-L and
   GED, some surviving Holm) — the positive control that says the harness discriminates.
10. **Better decompositions give better answers, in ordering** (exp-017, same Mistral reader and
    context): EM 0.18 no-decomposition < 0.21 prompting < 0.24 fine-tuned < 0.28 gold. Descriptive
    only; no paired test across the four cells.
11. **A 2/3-hop-only fine-tune does not transfer to 4 hops on step count** (exp-013: hop-count EM
    0.145 vs 0.300 prompting and 0.410 all-hop fine-tune at hop 4), while chain well-formedness
    survives (200/200).

## 3. What is unmeasured that the thesis idea implies

The README's pipeline is router → decomposer → jury, on MetaQA then MuSiQue, with MuSiQue carrying
both decomposition quality and end-to-end evaluation (ADR 0006). Against that:

- **End-to-end answer accuracy exists for four cells only** (exp-017), all Mistral-reader, sourced
  from exp-004 and exp-008. No answer-side number exists for any Qwen arm, for the pool-only
  fine-tune (exp-026), for the 2/3-hop fine-tune, or for the router.
- **MetaQA has zero v2 experiments.** All 28 log rows are MuSiQue. The GRAG half of issue #16 is
  an external dependency only Jahid can obtain; the transcript cross-check records a sanctioned
  fallback the supervisor voiced on 2026-08-12 — [1:15:24], verbatim "If you want to shrink… to one
  dataset, continue with the music" (a transcription of "MuSiQue") — that no ADR carries.
- **The jury** was never implemented (README; confirmed: no code) — and is not a gap: ADR 0006 item 1
  dropped it to future work.
- **No Qwen model has ever been fine-tuned**; every LoRA arm is Mistral. No guided/oracle arm
  exists on the block pool for any Qwen model, so #49's specific contradiction is still open.
- **Every arm ran once.** Decoding is greedy at seed 42, so a re-run reproduces the numbers;
  that is reproducibility, not variance. Training-seed variance for the fine-tunes and pool-draw
  variance (`num_trials: 1`, ADR 0021) are both unmeasured. The supervisor parked repeats "for
  now… later, when we fix what are the main experiments".
- **Per-hop significance does not exist.** Every per-hop table is descriptive; all tests run on the
  pooled 600. A6 asked for per-hop results.
- **The `pool_2000` seeded hop-stratified control** was never trained (issue #13), so exp-026 vs
  exp-008 moves three factors (training identity, volume, hop mix) at once.
- **The third ordering, `complexity_asc`**, is reserved and refused at load (ADR 0032 item 5).
- **The prompt-template confound.** Mistral renders `prompt_style: plain`, the Qwen folders
  `chat_template`; how much of finding 3 is template rather than model is unmeasured.
- **4-bit vs fp16** is measured for Mistral only; the 14B cannot run uncompressed on this GPU.
- **exp-015's and exp-008's peak VRAM** were never sampled, so no memory claim exists for 4-bit.
- **The pool-size sweep at 1k granularity** (issue #51) is deferred; the supervisor rated it "not a
  high priority".

## 4. The decision state

**Standing constraints, as they stand today** (Appendix A §2): `default_max_params` 8e9 for every
role, with one exception key `decomposer_max_params: 1.5e10` (ADR 0031 item 4 + amendment; Jahid
confirmed in ADR 0032; **supervisor confirmation still pending** — exp-022 and exp-025 exist only
under it); no closed commercial model judges decomposition quality; dataset roles per ADR 0006/0018;
few-shot method bi-encoder top-20 → cross-encoder top-5 with the k values demoted to ablatable
(ADR 0017) and typed masking reopened (ADR 0018); pool 2,000, adopted as `size2000_imbalanced`;
eval set the ADR 0007 pinned 600; significance = paired bootstrap + McNemar + t-test, **no
multiple-comparison correction across the 13 statistics**; reported primary = the six-term
`decomposition_report_card`, composite frozen as legacy.

**Awaiting the supervisor** (Appendix C §5, Appendix A §3): the thesis-primary metric (issue #6
item 5); typed masking as the default; the cross-encoder slot; the multiplicity policy; sign-off on
ADRs 0009, 0014, 0015, 0018 item 2, 0019, 0023, 0024–0027, 0029, 0030; the 1.5e10 ceiling; whether
thesis claims are model-general or Mistral-scoped (#49); what "least complex to most complex"
means for exemplar ordering (A5).

**Awaiting Jahid alone:** the A4/A7 hand inspection; A3; A9 (data to Ahmmad outside the repo);
the GRAG access question (#16); the residue of #19; ADR 0027 point 5; the Granite model-swap
offer (#46 §4.8, never answered); #51's scoping; whether #52 closes; whether the pool is still
worth working on at all — he asked this on 2026-08-27 ("Should I still be working on the pool
itself?") and the supervisor never answered.

**Tensions the record itself raises**, for the meeting rather than for an agent to resolve:

1. **The contribution framing versus the evidence.** ADR 0006 item 5 states the contribution as
   "a strategy for constructing the few-shot pool and retrieving examples for hop-aware
   decomposition, validated empirically". The evidence since says: pool construction null
   (exp-010/019), hop-matched retrieval moves count only (exp-009), ordering null on the card
   (G3), re-ranking moves ROUGE-L/GED (exp-018) — while a 250-step fine-tune on the same 2,000
   pool beats every prompting arm on every model (exp-026, finding 1–2). The supervisor also said
   on 2026-08-27 "we cannot say there is a contribution up to this point because we didn't do
   testing. You may say this is our objective." No ADR reconciles the two. Whether the thesis's
   claim becomes "the pool is a training set, not a prompt library" is his and Jahid's call.
2. **Two arms rest on a pending number.** If the supervisor reasserts 8B or 10B, exp-022 and
   exp-025 are withdrawn and the size contrast has no answer (block report §6.9).
3. **Model generality.** ADR 0030 removed the router on a Mistral-only premise that exp-005
   contradicts. The block widened unguided coverage to four models but ran no guided arm, so the
   contradiction is unmeasured on the block axis (#49, comment 2026-09-20).
4. **"Only those. Don't do anything further."** The assignment is finished and no further
   instruction exists. Any GPU work beyond it is outside the supervisor's stated scope until he
   speaks; the six remaining weeks are therefore gated on the meeting, not on the machine.

## 5. Record and tooling health

Verified by the code and branch lanes (Appendices D, E). Facts, then what each implies.

**Nothing is stranded.** All five `feature/*` branches are either already merged (13, 40),
superseded by a re-implementation on `main` that cites them (50-anchor-set → ADR 0033,
50-qwen-decomposer-models → `feef949`), or a parked duplicate the issue itself calls non-canonical
(12-alt). The four `worktree-agent-*` refs are merged ancestors. Four of six worktrees live under
`/tmp` and vanish on reboot; all are clean and every commit is reachable from a ref. The three
untracked PDFs in `Slides/` are build outputs; the deck sources (`thesis-deck-v2.tex/.html`) are
gitignored and newer than any PDF.

**The single largest risk to the thesis is the unpushed history.** 53 commits, including every
#50 run and every ADR since 0030, exist only in one `.git` on a shared box whose home filesystem
filled to 100% once already (issue #18). Only Jahid can push.

**Test suite:** 616 tests collected; the code lane's run on 2026-09-20 gave **611 passed, 5 skipped**
in 4 min 13 s (Appendix D §2; the reviewer re-verified collection only) — but run with system
Python 3.10 and a user-local pytest, because `pytest` is in neither `requirements.txt` nor the
repo venv. All five skips are environment artifacts of that split (peft/trl and
sentence-transformers are in the venv, not in system Python); three of them are the LoRA
fine-tuning guards. Fourteen test functions return a value, which a future pytest will treat as an
error.

**Code paths are healthy.** No dead modules; every config has a reader; no hard-coded absolute
path in executable code; no dataset text in git except the 36 KB MetaQA few-shot prompt assets
under `few_shot/`, which are intentional and referenced from `paths.json`. The router and answerer
are wired by file hand-off and exercised for real (exp-012, exp-017). Roughly eight scripts serve
the retired MetaQA lane and are dormant, not dead.

**Gaps between what CLAUDE.md and `docs/compute.md` require and what the code does:**

| gap | evidence | consequence |
|---|---|---|
| The run lock is prose only | no `.py` reads or writes `runs/run.lock`; three docstring mentions | two concurrent runs collide on nothing in code |
| Experiment ids are hand-typed | nothing allocates or validates `exp-NNN`; the id is a path the human types | a typo mis-files a run silently |
| `git.commit` stamped at write time, not launch | `src/run_artifacts.py:108`; issue #54 | exp-020's snapshot names a commit 40 min after launch; hand-recorded launch SHA is the workaround |
| Run note prints the model-folder `few_shot.k` (3) on reranked runs that showed 5 | `run_decomposer.py:3184`, `:2517`; issue #53 | every reranked run's `notes.md` and the exp-021/022 row prose state the wrong k |
| Offline parameter count vs runtime assertion differ by 0.7B on Qwen3.5-9B | `print_model_params.py:124-147` vs `src/model_size.py:64-72`; issue #55 | no decision turns on it under 1.5e10; would under 8e9 |
| The smoke test never exercises the ceiling assertion | `scripts/smoke_test.py:20-22` (dry runs) | the guard is covered only by fabricated-header unit tests |
| `.gitignore` never ignores plain `.json` | Appendix D §5 | a bulk corpus dumped as `.json` is committable; only the behavioural rule stops it |
| Launch/verification scripts live under gitignored `runs/` | `runs/check_ordering_invariant.py`, `check_regen_parity*.py`, waiters | what they asserted is not reproducible from the repo |

**Record drift** (Appendix A §4): the ADR index stops at 0030 though 0031–0034 exist, against
its own "same commit" rule; ADR 0015 carries no forward note to the 0031 amendment that folded
its 1e10 into the decomposer key; ADR 0006 §4 carries no note to 0014's 9,156-pool deviation;
both plan documents still describe the ceiling raise as `default_max_params → 1.5e10` and the
anchor set as an id-filter into the run path, both of which the landed code and ADRs 0031/0033
reversed; the stop condition is "flagged, not adopted" in the phase plan and "adopted" in
ADR 0032. `.claude/agents/experiment-runner.md` still says `model: sonnet` against Jahid's
2026-09-20 Opus-only instruction.

**Tracker lag** (Appendix C): #12, #14, #27, #40, #41-instrument-1 and #52 are done but open;
#50's last comment predates exp-027/028; #49's 2026-09-20 comment is the only place the Qwen3.5-9B
result touches GitHub. Three small defects from the block (#53, #54, #55) are unassigned, each a
branch + Gate 1 job. Slide 25 of the circulated PDF still carries the "only GED significant"
error that `deck_corrections.md` fixed on 2026-08-27.

## 6. Comparisons that are and are not legitimate

Appendix B §3 lists both. The rules that matter most for the write-up:

- Two evaluation sets exist — the pinned 600 and the pool-sweep's own 750 — intersecting at
  405/600. exp-006/010/018/019 are on the 750; **nothing from them goes beside a 600 number**.
- Two few-shot pools exist — v1's 9,156 and `size2000_imbalanced`. exp-004/005/008/009/012 are on
  the first, exp-014/015/020–028 on the second. exp-005 is not a comparator for exp-027 despite
  being the same model.
- The precision axis moves silently across blocks: exp-004/005/008/015 fp16, exp-010 and
  exp-020–028 4-bit, exp-026 trained 4-bit and evaluated fp16. Only exp-015 ↔ exp-020 isolates it.
- `--compare` against exp-008 via exp-011's per-item files omits over/under-decomposition and
  `decomp_mean`; those are reported descriptively there.
- Nothing is ranked on `composite_score`; nothing arm-vs-arm is read off a 45-item sheet.

## 7. Corrections to the lane ledgers

Two claims in the appendices were checked against the repo and corrected in place, marked
"corrected by the lead":

1. Appendix C said issue #52 was unstarted and its analysis note absent from `main`. Both are
   false: `docs/analysis/2026-09-14-pool-arm-identity-investigation.md` is on `main`
   (commit `2354c82`) and log row exp-019 records the result. #52 is done-but-open.
2. Appendix C's cross-check calls `feature/12-guided-vs-unguided-alt` "real unmerged content".
   Appendix E shows the two pieces it was held for were re-implemented in PR #21's squash
   (`1ccea11`: `src/step_lines.py`, `tests/test_decomposer_conditions.py:610`). The branch holds
   code absent from `main` only in the sense that a superseded duplicate is absent. Appendix E's
   reading stands.

## 8. Sources

The five appendices, and behind them: `docs/adr/0001–0034`, `docs/meetings/`, `docs/plan/`,
`Plan after the meeting.md`, `experiments/log.md` (28 rows), `experiments/exp-*/`,
`docs/METRICS.md`, all 24 files in `docs/analysis/`, `gh issue view` for all 32 issues,
`git log`/`git merge-tree` on every local ref, the test run of 2026-09-20, `nvidia-smi`, `df`,
and the Fathom meeting list from 2026-08-27 onward.

Refs #50, #49, #46, #6, #16, #13, #47, #52, #53, #54, #55
