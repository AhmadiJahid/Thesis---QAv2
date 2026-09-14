# Phase plan — the 2026-08-27 supervisor assignment

- **Written:** 2026-09-14, by the lead session, at Jahid's request ("check the latest meeting I
  had with Ahmmad. Based on that meeting, lets plan the next steps and create a new md for this
  phase of the plan").
- **Source meeting:** 2026-08-27, Jahid + Ahmmad, 71 minutes —
  [recording](https://fathom.video/calls/801583891). Timestamps below are deep links into it.
- **Repo state it was written against:** `main` at `0995c64`, working tree clean apart from the
  three untracked PDFs in `Slides/`.

## What this document is, and is not

It is a **record of what Jahid's supervisor assigned on 2026-08-27**, plus a sequencing of that
assignment against what the code can do today and what the machine can run today. Every work
item traces to a quoted instruction. It follows the precedent of issue #46: it invents no
research direction, no methodology and no success criteria, and per `CLAUDE.md` it is not a PRD.

Where the meeting left something open, this document says so and routes it to §7 or §8 rather
than resolving it. Where a number is quoted, it comes from `experiments/log.md` or a committed
artifact — **never from the meeting transcript**, whose numeric passages are badly garbled by
transcription.

A caution for anyone re-reading the transcript: Fathom's speaker labels in this recording are
unreliable and swap mid-segment. Attribution below is by content, and the items marked
*(garbled)* reproduce what the transcript literally says rather than a cleaned-up guess.

---

## 1. Where things actually stand

Two facts matter more than anything in the meeting, and the plan is built around them.

**The work has not started.** The newest commit on `main` is `0995c64`, dated **2026-08-27** —
the slide fact-check, made the same day as the meeting. Eighteen days have passed. The
supervisor asked to meet again "**maximum in any way we will meet next Tuesday**"
([1:09:55](https://fathom.video/calls/801583891?timestamp=4195)), which was 2026-09-01; there is
no recording of a follow-up and no commit implying one happened. Against the fixed calendar —
experiments end **31 October 2026** — about **6.5 weeks of experiment time remain**.

**The GPU is occupied by someone else.** `nvidia-smi` today shows another user's `ollama`
process (pid 3309317, user `nmugurt…`) holding **21,190 MiB of the 24,576 MiB** RTX 3090, and it
has held it for **6 days 11 hours**. That leaves ~3.3 GiB — not enough for a 4-bit 7B decoder.
Per the standing convention on this shared box, that process is never killed. Every GPU item in
§3 is therefore **blocked on capacity, not on readiness**, and should be launched through a
detached waiter that polls for free VRAM rather than by a foreground launch that fails.
`runs/run.lock` is free and `/cta/users` has 178 GB.

Everything in §2 that is CPU-only or authoring work can proceed today regardless.

---

## 2. The assignment, as given

The supervisor closed the assignment with an explicit scope limit:

> "And then we will discuss those results in the next meeting. **Only those. Don't do anything
> further.** Whenever you finish them, let's arrange a quick meeting."
> — [1:05:22](https://fathom.video/calls/801583891?timestamp=3922)

So the items below are the whole of it. §4 lists what he deliberately kept out.

### A1 — Three decomposer models, one fixed 2,000 pool, one factor changed

> "Let's pick one experiment only… the pool size, 2,000… one time on Mistral, 7 billion, one
> time on Quen, 7 billion, Quen 2.5, 7 billion… and do it on another Quen model, bigger size."
> — [1:01:38](https://fathom.video/calls/801583891?timestamp=3698)

> "So you have three experiments, Mistral 7 billion, QN 7 billion, QN 9 billion."
> — [1:03:04](https://fathom.video/calls/801583891?timestamp=3784)

His stated purpose: *"if we fix the size of the model, but we try different model from Mistral
to QN. Does the result will change significantly or not? Within the same model, if we change the
size, if the model will change or not"*
([1:03:13](https://fathom.video/calls/801583891?timestamp=3793)). This directly answers the
concern recorded in **issue #49** — that every MuSiQue arm but exp-005 is Mistral-only — and it
arose in the meeting for exactly that reason: Jahid said *"Quen was used only once"* and the
supervisor replied *"The result you are doing is mainly following Mistral?"*
([53:52](https://fathom.video/calls/801583891?timestamp=3232)).

**Readiness.** Only Mistral-7B-Instruct-v0.3 is ready. It and Qwen3.5-9B are the two registered
decomposer models (`components/decomposer/models/{mistral_7b_instruct,qwen3_5_9b}/config.json`)
and the only two shipping the `prompt_unguided.md` that the unguided condition requires. The two
Qwen arms now chosen — **Qwen2.5-7B-Instruct** and **Qwen2.5-14B-Instruct** (§7.6) — each need a
new decomposer model folder with a parity-guarded unguided prompt before A1 can run.
Qwen2.5-7B-Instruct is already in the local HF cache; **Qwen2.5-14B-Instruct is not** and needs
~28 GB of download against 178 GB free. The size ceiling rises to `1.5e10` to admit it
(`configs/model_limits.json`, ADR 0031 item 4, superseding ADR 0015's 1e10).

### A2 — A Mistral fine-tune trained on the 2,000 pool only

> "And then do a tuned version of Mistral, but trained on the pool only, not the full data, on
> the pool, same pool… So no full training, just 2,000. Just let's see the effect of the pool."
> — [1:03:32](https://fathom.video/calls/801583891?timestamp=3812)

**Readiness: config only, no new code.** `configs/finetune_decomposer.json` already declares a
`pool_2000` arm (`max_examples: 2000`, `stratify_cap_by_hop: true`) and it is the file's
`default_arm`. The adapter has **never been trained** — exp-008's log row records it as blocked
on the best-pool decision (issue #13). As written the arm draws 2,000 seeded rows from the
enriched train pool rather than from a pinned pool artifact; pinning the specific pool the
supervisor means requires setting `train_source_path` (§7.1).

### A3 — Understand every metric being used

> "I want you to understand every metric you are using. So that if we saw like one of the metric
> change but the another does not change, what does that mean?"
> — [1:04:00](https://fathom.video/calls/801583891?timestamp=3840)

and on SARI and GED specifically: *"It's good to know how they work. Even if you don't have a
formula, at least we need to understand how they work"*
([20:53](https://fathom.video/calls/801583891?timestamp=1253)).

**This is Jahid's own task, explicitly** — *"this is for you"*. It is not delegable to an agent
and it is not a documentation ticket; the deliverable is that he can answer, in the next
meeting, what a movement in one metric without a movement in another means.
`docs/METRICS.md` and `docs/analysis/2026-08-23-decomposition-quality-metric-specification.md`
are the existing material to read, not a substitute for the task.

### A4 — A fixed 45-item anchor set, inspected by hand every experiment

> "I want you to pick some anchor sample that you will check manually by yourself in every
> experiment… let's say for 15 from each hop. So in total 45 samples. Those samples you will
> look at them manually… and those will be fixed over all the experiments. We will do it from
> now on."
> — [1:04:00](https://fathom.video/calls/801583891?timestamp=3840) →
> [1:05:11](https://fathom.video/calls/801583891?timestamp=3911)

Jahid proposed the sampling rule and the supervisor confirmed it: *"I pick it randomly, like 15
random, but in all of them, same random, same 15."*

**Readiness: nothing like this exists.** The nearest artifacts are `configs/qwen_smoke.json`
(20/hop, sampled at run time, ids not committed) and the ADR 0007 pinned 600. Building it means
a committed id list plus an id-filter path into the decomposer and evaluator — the evaluator's
`--limit` is a head-slice, not an id filter. It is small, CPU-only work that unblocks the manual
inspection attached to every run below, so it should land **before** the first GPU arm, not after.

### A5 — Few-shot example ordering

> "Those top five, you can, you are now putting the most highest matching, in your matching
> score to the lowest. Am I right? You may try to reverse it. Also, may try it from the least
> complex to the highest complex… Try all those things we mentioned and let's meet."
> — [1:07:43](https://fathom.video/calls/801583891?timestamp=4063)

He put a magnitude on it: *"the order is matter. It can change up to 10%."*

**Readiness: new code.** Order is hard-coded. `examples_from_reranked_row()`
(`components/decomposer/run_decomposer.py:618`) slices `kept[:k]` in retrieval-rank order and
`format_few_shot_examples()` (`:727`) renders that list as-is. There is no ordering key in any
config. Three orderings are named: **as-is (highest similarity first)**, **reversed**, and
**least-complex → most-complex**. "Complexity" is not defined in the meeting — see §7.5.

### A6 — Report results per hop count, as well as overall

> "check the performance on two hub questions, three hub questions, four hub questions, and do
> it separately." — [42:29](https://fathom.video/calls/801583891?timestamp=2549)

> "you may report the result in two ways… a first slide over all results. The next slide is like
> a result on each hub… maybe it gets improved in two hub, but it decreases in three hub."
> — [1:09:33](https://fathom.video/calls/801583891?timestamp=4173)

**Readiness: mostly exists.** The evaluator already emits `per_gold_hop_metrics` and
`gold_hop_distribution` (`scripts/musique_decompositions_evaluator.py:3172-3207`), bucketed on
*gold* hop count. What does **not** exist is per-hop *significance* — `compare_decomposer_arms.py`
runs its tests on the pooled aligned set. exp-013 got a per-hop test by hand-filtering a
per-item file; if per-hop significance is wanted for this block, that is either a small code
addition or the same manual filtering repeated.

### A7 — Eye-to-eye qualitative comparison, Qwen vs Mistral

> "I would recommend you to make a comparison… eye-to-eye comparison. Pick some bad example with
> unguided and see how it improves with guided. And the same example, check them with Quen and
> Mistral. Maybe Mistral is bad in this capability."
> — [42:29](https://fathom.video/calls/801583891?timestamp=2549)

This is the qualitative half of A1 and it rides on the A4 anchor set — the same 45 items,
inspected across models.

### A8 — Slides

The supervisor gave a required ordering and about eighteen specific fixes, and asked for the
deck: *"share with me this slide. I want to take further look about it"*
([1:05:38](https://fathom.video/calls/801583891?timestamp=3938)). The full itemised list is in
§9 and in `Slides/deck_corrections.md`.

### A9 — Show him the experiment data

> "Is your data existing in your GitHub? Those experiment data?… I may take a look about it. I
> mean, to see it, how it's changing, because why it does not improving?"
> — [45:19](https://fathom.video/calls/801583891?timestamp=2719)

This runs into a standing repo rule and needs Jahid's call — §7.6.

---

## 3. Sequence

Ordering is forced by two things the supervisor said, not chosen here: *"You do change, one
change, and test. Don't do multiple changes in one experiment, always one by one"*
([1:07:33](https://fathom.video/calls/801583891?timestamp=4053)), and the scope limit at
[1:05:22]. The GPU constraint in §1 forces everything else.

**Now, no GPU needed, no decision needed:**

| # | Work | Owner | Blocks |
|---|---|---|---|
| P0 | Build the A4 anchor set: 45 ids (15/hop, seeded, committed) + an id-filter path through decomposer and evaluator | `ml-engineer` | every run's manual inspection |
| P1 | Add the Qwen2.5-7B-Instruct **and** Qwen2.5-14B-Instruct decomposer model folders, each with a parity-guarded `prompt_unguided.md`, and raise `default_max_params` to 1.5e10 | `ml-engineer` | A1 |
| P1b | Chase the supervisor's suspected bug: why the imbalanced and clustered arms produce matching numbers (issue #52) | `analyst` | #51, and the reading of exp-010 |
| P2 | Add the few-shot ordering knob (`as_is` / `reversed` / `complexity_asc`) | `ml-engineer` | A5 |
| P3 | A3: metric study — Jahid personally | Jahid | the next meeting |
| P4 | Slide restructure per §9 | slide lane | sending the deck to Ahmmad |

P0–P2 are shared-pipeline changes, so they go through a branch and a `code-reviewer` pass
(Gate 1). They are small and independent; they can run as parallel lanes.

**Then, when the GPU frees — one run at a time, holding `runs/run.lock`:**

| # | Run | Est. GPU | Basis for the estimate |
|---|---|---|---|
| G1 | A1 decomposition: 3 models × the fixed 2,000 pool, one condition, one mask mode | ~1.5–2 h | exp-010 measured 1,515–1,792 s per decompose cell |
| G2 | A2 fine-tune `pool_2000` + evaluate | ~30–45 min | exp-001 took 4 h 18 min for 19,938 examples; 2,000 is ~1/10 the steps |
| G3 | A5 ordering: 2 further orderings on the agreed model(s) | ~1–2 h | same per-cell basis as G1 |

**These are estimates extrapolated from committed log rows, not measurements of these runs.**
Total is roughly 3–5 GPU-hours — comfortably inside the ad-hoc 10-GPU-hour threshold cited in
the exp-017 row, and small against the 6.5 weeks left. The binding risk is the occupied GPU, not
the compute cost.

Each run gets its pre-run entry in `experiments/log.md` before it starts, per `CLAUDE.md`. The
next free id is **exp-019**. Each run's manual anchor-set inspection (A4/A7) happens as part of
the run's write-up, not as a separate later pass — that is the point of a fixed anchor set.

---

## 4. Explicitly out of scope for this phase

Recorded so they are not silently picked up, and so they are not silently lost.

- **Finer pool-size sweep (3k/5k/6k, or 9k–12k).** He raised it
  ([28:18](https://fathom.video/calls/801583891?timestamp=1698)) and then downgraded it himself:
  *"not a high priority, it's an idea"*.
- **Repeating experiments for variance.** *"for now, don't waste time on doing the experiment
  multiple time. Later, we will, when we fix what are the main experiments to be done"*
  ([35:27](https://fathom.video/calls/801583891?timestamp=2127)).
- **Cutting the four statistical tests down to one.** He wants one eventually — *"I don't know
  if we need to use multiple ones, we need to pick one"*
  ([23:00](https://fathom.video/calls/801583891?timestamp=1380)) — but then declined to decide
  now: *"you don't need to pick now as much as also they are not taking time"*
  ([24:10](https://fathom.video/calls/801583891?timestamp=1450)). Nothing changes this phase.
- **DBSCAN as a pool-construction strategy.** Raised as a concern about k-means seed sensitivity
  (*"it's very sensitive to the seed. The initial position of the centroid are very affecting
  that"*, [33:59](https://fathom.video/calls/801583891?timestamp=2039)) and left as *"I will
  also maybe take a quick look"*. The transcript says **"DBMS" (garbled)** throughout; from his
  description — *"It goes over the condensed area in your space to be representative"* — it is
  almost certainly DBSCAN, but he never said the word. sklearn 1.8.0 is already a pinned
  dependency and ships both DBSCAN and HDBSCAN, so this is cheap if it is ever promoted; it is
  not assigned now.
- **Training on MetaQA and testing on MuSiQue.** Jahid raised it; the supervisor noted the
  obstacle — MetaQA has no gold decompositions — and said *"it would be difficult to find it"*
  ([59:30](https://fathom.video/calls/801583891?timestamp=3570)). Not assigned.
- **The router.** The only router exchange in 71 minutes was Jahid reporting that the regressor
  defaults to 2 and the supervisor replying *"So it was just guessing"*
  ([57:39](https://fathom.video/calls/801583891?timestamp=3459)). **He issued no router
  instruction of any kind** — no fix, no re-run, no requirement. ADR 0030 (router off the
  decomposition critical path) stands; issues #47 and #23 stay open and unscheduled. Worth
  noting he was careful to distinguish the oracle arm from a router: *"And you used the correct
  hop count." / "Yes, yes, correct, not a router"*
  ([36:29](https://fathom.video/calls/801583891?timestamp=2189)).
- **A bigger model beyond the three.** *"You may try. You may try."*
  ([58:22](https://fathom.video/calls/801583891?timestamp=3502)) — permission, not an assignment.

---

## 5. What he rejected or criticised

- **The composite score is dropped.** Jahid brought the decision — he called the formula
  *"artificially made, basically"* and traced the reference-validity term to the
  `[#k]`-vs-`#k` syntax mismatch (issue #40) — and the supervisor ratified it: *"Does that mean
  you will stop using the composite score?" / "Yes." / "Okay. Sounds good"*
  ([21:31](https://fathom.video/calls/801583891?timestamp=1291)). His alternative, had it been
  kept, was *"we need to revise it carefully… we need to discuss how the weights"*. The repo
  already reflects this: `composite_score_status: "legacy"`, with
  `decomposition_report_card` as the reported primary (ADR 0029).
- **Claiming a contribution before testing.** *"we cannot say there is a contribution up to this
  point because we didn't do testing. You may say this is our objective"*
  ([08:00](https://fathom.video/calls/801583891?timestamp=480)).
- **Evaluation and datasets placed inside the methodology section** — both belong in
  experimental setup ([09:14](https://fathom.video/calls/801583891?timestamp=554),
  [11:48](https://fathom.video/calls/801583891?timestamp=708)).
- **The deck reading as AI-generated.** *"I'm assuming this presentation has been done by an AI
  tool. Am I right?… there is too much formatting issue… this is a clue that you use AI"*
  ([08:30](https://fathom.video/calls/801583891?timestamp=510)). On his own defence deck: *"I
  didn't do it by AI… I don't care about coloring or to be look fancy… the most important is to
  be clear"*.

---

## 6. A concern he raised that nobody has chased

> "There's something wrong in the experiment. The experiment itself. It can be a bug… sometimes
> I'm feeling you are doing many changes. At the end of the day, we cannot see an effect in the
> result. Even we cannot see a drop. We see like, for example, you try clustering with and
> without clustering and the result is matching exactly."
> — [1:09:02](https://fathom.video/calls/801583891?timestamp=4142)

He assigned no debugging task, so this is not in §3. But it is the kind of suspicion that is
cheap to answer and expensive to leave standing, and A1 partly answers it by construction: if
three different models on an identical pool produce visibly different numbers, the pipeline is
discriminating; if they produce near-identical numbers, his suspicion is the leading hypothesis
and the phase should stop and investigate rather than proceed to A5. **Recommend treating that
as the stop condition on G1** — flagged here for Jahid's agreement, not adopted.

He also asked for one specific verification, about the oracle arm:

> "It needs to be checked because other than that, there is multiple factors changing. And it's
> difficult to say if this is the effect of Oracle or not."
> — [47:11](https://fathom.video/calls/801583891?timestamp=2831)

---

## 7. Decisions that gate this phase — RESOLVED 2026-09-14

Jahid answered these in session on 2026-09-14. Recorded in
[ADR 0031](../adr/0031-jahid-2026-09-14-gate-decisions-for-the-supervisor-assignment.md);
summarised on issue #50. The original seven are kept below with their answers so the reasoning
that produced each question stays readable.

1. **Which 2,000 pool → the imbalanced one.** `runs/pool_sweep/pools/size2000_imbalanced_trial0_poolseed42`.
   Also the status quo issue #46 §4.1 was proceeding on, so nothing downstream is invalidated.
   **A2 still needs it pinned via `train_source_path`**, or the fine-tune trains on a different
   2,000 than the prompting arms use.
2. **Which evaluation set → the ADR 0007 pinned 600**, decided by the lead under delegation.
   Jahid named both candidates and handed the call back. The 750 sample intersects the pinned
   600 at 405/600 (exp-018), so choosing it would place the new arms on an axis where no
   existing baseline lives. Continuity wins.
3. **Which model above 7B → see 4 and 5.**
4. **The size ceiling is raised, 1e10 → 1.5e10**, to admit a ~14B decomposer. Pending supervisor
   confirmation, but with his own words behind it this time ("12, 14, check what is above 7
   billion"). `router_max_params` and `answerer_max_params` are untouched.
5. **Which prompting condition → all arms unguided.** No hop-count guidance of any kind. This
   constrains participation: a model must ship `prompt_unguided.md` to run the condition.
6. **The arms, as the lead reads 1–5 — Jahid to veto if wrong.** Mistral-7B-Instruct-v0.3 vs
   **Qwen2.5-7B-Instruct** moves family with size held; Qwen2.5-7B-Instruct vs
   **Qwen2.5-14B-Instruct** moves size with family held. **Qwen3.5-9B fits neither contrast** —
   against Qwen2.5-7B it moves generation and size at once, the precise confound "one change,
   and test" exists to prevent. It stays available as a fourth, confounded point (+~25 min GPU),
   not as the size rung.
7. **Experiment data does not go to GitHub.** Jahid shares it with Ahmmad directly, outside the
   repo. The `CLAUDE.md` rule is unchanged and no agent arranges the transfer.
8. **Ordering arms — STILL OPEN.** What "least complex to most complex" means for an exemplar
   list (A5) was never specified. Hop count is the obvious candidate; question length and
   gold-chain length are others. Still gates A5, and picking one silently would be inventing
   methodology.

Two further calls made the same day, outside the original seven:

- **Slides come before any GPU work**, shipping as two artifacts from one source — an HTML deck
  and a self-contained Beamer `.tex` for Overleaf. The first pass covers the full deck, not just
  the front half, because Jahid accepted it will be rebuilt more than once.
- **The finer pool-size sweep is kept but deferred** — 1,000-step granularity from 2,000 to
  8,000, tracked as issue #51, scheduled after this block. The supervisor had downgraded it to
  "not a high priority, it's an idea"; Jahid is choosing to keep it, later.
- **The suspected bug in §6 is now being chased** as issue #52 — CPU-only, running now precisely
  because it needs no GPU.

## 8. Items Jahid personally owes, from his own answers in the meeting

- **Ordered step accuracy, when the predicted hop count differs from gold.** Asked directly at
  [03:15](https://fathom.video/calls/801583891?timestamp=195); Jahid's answer was *"I don't have
  anything. I will check it as well."* This is a metric definition gap sitting under a metric
  already reported in the deck.
- **What the exact-match normalisation actually does.** *"I will be honest, I'm not that sure,
  but these two are definitely in there, I'm going to check it"*
  ([02:22](https://fathom.video/calls/801583891?timestamp=142)) — and the supervisor's response
  was that this means a dedicated slide. The implementation is in
  `scripts/musique_decompositions_evaluator.py`; the answer has to be read off the code, not
  recalled.
- **Whether to keep working on the pool at all.** Jahid asked this directly —
  [29:13](https://fathom.video/calls/801583891?timestamp=1753), *"Should I still be working on
  the pool itself? Because changing it did not produce that much of difference"* — and **the
  supervisor never answered**; the conversation diverted to k-means. It is worth re-asking,
  because A1 and A2 both hold the pool fixed, which implicitly answers "no, not for now".

---

## 9. Slides

Required ordering, verbatim ([09:40](https://fathom.video/calls/801583891?timestamp=580)):

> "you will start with a motivation definition… like examples, challenges, and then you will put
> your methodology. **Motivation, Definition, Examples, Methodology.** And then you will go to
> Experiment Setup. In the Experiment Setup, at that time, you will start discussing your
> metric, your datasets, if you want to compare with the baseline, and after that, you will go
> to Results."

Research questions move to **after** the methodology and immediately before experiments, because
*"now the research question will become relevant"* ([07:01](https://fathom.video/calls/801583891?timestamp=421)).

Itemised fixes he named: motivation slides plus a concrete example; a formal problem-definition
slide (domain D, multi-hop Q in, Q′ = q₁…qₙ out); an examples-and-challenges slide; the pipeline
slide split into an overview plus one slide per component; dedicated slides for type-masking and
for the normalisation used by exact match; a clear definition of ordered step accuracy; a
visualisation of the re-ranking funnel (pool → top 20 → top 5); datasets moved into experimental
setup with an explanation of why there are two; evaluation removed from methodology; dense
slides split; grammar and formatting cleaned before the professor sees it; and results presented
twice — overall, then per hop.

Two further constraints on the deck: he wants it sent to him for review, and he shared his own
defence deck as a structural template with *"please keep it between us"*
([1:10:15](https://fathom.video/calls/801583891?timestamp=4215)) — so that file is not to be
committed to this repo or circulated.

The existing `Slides/deck_corrections.md` already carries a slide-by-slide numeric fact-check
against the log, including four numeric defects found on 2026-08-27. That work is complete and
feeds this restructure; the restructure is what remains.

---

## 10. What this phase is measured by

The supervisor set exactly one completion condition, and it is not a metric:

> "Whenever you finish them, let's arrange a quick meeting."

The phase ends when A1, A2, A3, A4 and A5 have produced results, the anchor set has been
inspected by hand across them, the deck is restructured and sent, and the meeting is booked.
No performance threshold was set and none is invented here.
