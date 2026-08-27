# Deck corrections — `Thesis_QA_Slide_Deck.pdf` (34 slides), 2026-08-25

Slide-by-slide fact-check against `experiments/log.md` and run artifacts, prepared for
this evening's talk. Every number below carries its source. Each item includes a
**prompt block** you can paste into Claude Design (or any slide-generating session) to
produce the corrected or new slide.

**Shared style prompt** — prepend this to any slide prompt below so the output matches
the existing deck:

```text
Slide style: 16:9 (1440x810), white/very-light background, dark near-black body text.
A small uppercase blue eyebrow label top-left (e.g. "MUSIQUE DEV · N = 600"), then a
large plain-sans title in sentence case, then a short blue underline rule. Accent color
is the deck's medium blue (#1a56b8 range). Tables: thin light row rules, uppercase blue
column headers, bold for best-in-column values, red only for genuinely bad/collapsed
values. Dense but calm; no decoration, no icons, no gradients. Footnote line in grey at
the bottom citing the data source.
```

---

## 1. Fix before export (mechanical defects)

### 1.1 Slide 4 "Two datasets, two jobs" — six red `[count]` placeholders + overlapping callout

Verified values (sources below the tables):

| MuSiQue | Train | Dev avail. | Dev used |
|---|---|---|---|
| 2-hop | 14,376 | 1,252 | 200 (already on slide) |
| 3-hop | 4,387 | 760 | 200 |
| 4-hop | 1,175 | 405 | 200 |

- Train: counted per hop-prefix from `musique_ans_v1.0_train.jsonl` on 2026-08-25;
  sum 19,938 = official MuSiQue-Ans train size, so the count is complete.
- Dev available: `runs/pool_sweep/dev_sample/dev_sample_250per_hop_seed42_stats.json`.
- Dev used: the ADR 0007 pinned 600 (200/hop).

| MetaQA | KB questions (refined) | Eval used |
|---|---|---|
| 1-hop | 96,106 | 100 |
| 2-hop | 118,980 | 100 |
| 3-hop | 114,196 | 100 |

- KB questions: line counts of v1 `Pool/qa_train_{1,2,3}hop_refined.txt`.
- Eval used: 100/hop is what the deck's own metrics footnote and confusion matrix
  already assume.

Also: the orange callout's label "READ EVERY LATER RESULT WITH THIS IN MIND" overlaps
its body text — layout fix needed.

```text
Regenerate the "Two datasets, two jobs" slide. Eyebrow: "DATA". Title: "Two datasets,
two jobs". Two side-by-side cards.

Left card — MuSiQue (tag: "2, 3 AND 4 HOPS"): "Ships human-written gold decompositions,
so it carries all decomposition-quality evaluation: the plan itself can be scored."
Table with columns HOP / TRAIN / DEV AVAIL. / DEV USED, rows: 2-hop 14,376 / 1,252 /
200; 3-hop 4,387 / 760 / 200; 4-hop 1,175 / 405 / 200. Below: "Train is split four
ways, stratified on hop-prefix strata, seed 42; few-shot pools of 1000–8000 are sampled
from it." and in blue: "Masking is NER-based: with no entity inventory, masks are
predicted."

Right card — MetaQA (tag: "1, 2 AND 3 HOPS · 300-QUESTION EVAL SETS"): "A movie-domain
knowledge base of subject–relation–object triples. No gold decompositions, so it
carries end-to-end evaluation only: a plan is judged by whether it executes and returns
the right answer." Table with columns HOP / KB QUESTIONS / EVAL USED, rows: 1-hop
96,106 / 100; 2-hop 118,980 / 100; 3-hop 114,196 / 100. Below in blue: "The KB triples
make exact masking possible: every film and person name is a known string, matched by
lookup rather than predicted."

Bottom: a full-width orange warning callout with the label "READ EVERY LATER RESULT
WITH THIS IN MIND" on its own line (do NOT overlap it with the body text), body: "4-hop
is a small minority class in MuSiQue next to 2-hop and 3-hop. Retrieval pools
under-represent it unless balanced, and every 4-hop figure rests on the fewest
questions."
```

### 1.2 Slide 13 "Typed masking leads on every plan metric" — truncated header

The fourth column header renders as "ORDERED STEP ACC" with the word cut. Widen the
column or shorten the header to "ORDERED ACC.". (Content is otherwise correct; no
regeneration needed if you can edit the header in place.)

### 1.3 Slide 1 title slide — old repository name

Says `AhmadiJahid/Thesis---QA`; current work is in `Thesis---QAv2`. Update or drop the
line.

### 1.4 Appendix "Experimental infrastructure" — v1-era ops, wrong GPU

Slide says Kaggle notebooks + "24 GB RTX 2090", Jira project TQ, changelog, decisions
log. Per `docs/compute.md`: the GPU is an **RTX 3090 (24 GB)** on the local box,
**Kaggle and the cluster are not in use**, and tracking is GitHub Issues + the
append-only `experiments/log.md` + `docs/adr/`. ("RTX 2090" is not a real card — it
will be noticed.)

```text
Regenerate the "Experimental infrastructure" appendix slide. Eyebrow: "APPENDIX · HOW
EXPERIMENTS RUN". Title: "Experimental infrastructure". Four columns with thin blue
top rules:

AUTHOR — "Code in the repository; one branch per GitHub issue, small reviewed diffs."
EXECUTE — "A local Ubuntu box with an RTX 3090, 24 GB VRAM; long runs launched
detached under a run lock."
RECORD — "Every run writes a config snapshot, metrics.json and a short run note under
experiments/<exp-id>/; the append-only experiment log is the single record — no
experiment exists unless it is in the log."
ANALYSE — "Evaluator + report scripts sweep run folders into metrics tables; decisions
are recorded as ADRs."

Bottom grey line: "Work is tracked in GitHub Issues; every PR references its issue;
design choices land in docs/adr/."
```

---

## 2. Claims overtaken by evidence (deck was drafted before last week's runs)

### 2.1 Slide "What gets run, and what it settles" — four of six rows have already run

| Row on the slide | Status now | What it settled |
|---|---|---|
| Fine-tuned decomposer | **Done** — exp-008, exp-015 | Fine-tuned wins decisively: 10/10 paired statistics significant vs prompting at its best pool. No longer "the baseline to beat" — it is the current leader. |
| Guided vs unguided | **Done** — exp-012 | Prompted router does not measurably help (lowest hop-count EM 0.492, highest under-decomposition 0.292 in its table). This is the evidence behind the de-scope the deck already shows. |
| Same-hop few-shot test | **Done** — exp-009 | Oracle hop-matched pool beats mixed modestly on the report card (cite exp-009's log row); router-hop-matched variant remains open (issue #15). |
| End-to-end, MusiQue | **Done** — exp-017 (log row completed 2026-08-25, commit `edfa27e`) | Better plans produce better answers — see §3.1. |
| Pool strategy sweep (clustering) | Still open | Issue #14. |
| End-to-end, MetaQA (GRAG) | Still blocked | External dependency on the supervisor's GRAG system (issue #16). |

```text
Regenerate the experiment-plan slide as a results slide. Eyebrow: "EXPERIMENTS".
Title: "What was run, and what it settled". Three-column table EXPERIMENT / STATUS /
WHAT IT SETTLED with rows:

Same-hop few-shot test | Done (exp-009) | "Oracle hop-matched examples beat a mixed
pool modestly; the router-matched variant is still open."
Guided vs unguided | Done (exp-012) | "The prompted router does not measurably help:
lowest hop-count accuracy and highest under-decomposition of its table. This is the
evidence behind dropping the router."
Fine-tuned decomposer | Done (exp-008, exp-015) | "The fine-tune leads every
decomposition metric; 10 of 10 paired significance tests favour it over prompting at
its best pool."
End-to-end, MuSiQue | Done (exp-017) | "Answer accuracy rises monotonically with plan
quality; see the end-to-end slide."
Pool clustering strategy | Open | "Issue #14."
End-to-end, MetaQA | Blocked | "Depends on the supervisor's GRAG system (issue #16)."

Style the STATUS column with small chips: green "Done", grey "Open", orange "Blocked".
Bottom grey line: "Reporting rules unchanged: accuracy broken out by hop depth, and a
paired significance test on any comparison narrower than a point."
```

### 2.2 Limitations slide — three bullets now false or overstated

- **"No single end-to-end accuracy figure exists; no integrated run exists"** — now
  false: exp-017 finished 08-23, four cells, 600 questions each, rc=0. Replace with the
  figures (§3.1) or point to the new slide.
- **"No significance test has been run yet"** — true only for typed-vs-uniform
  masking; keep it scoped to that. Paired bootstrap + McNemar tests are logged for
  prompting-vs-fine-tuned (exp-011/014/015) and the hop-4 generalisation comparisons
  (exp-013).
- **"The retrieval premise was assumed, not tested"** — partially overtaken: exp-009
  measured the oracle hop-matched variant; the router-matched variant is open
  (issue #15). Soften to "measured in oracle form; router-matched variant open".

### 2.3 Slide 7 "Metrics used in this talk" — composite score status changed

The composite is no longer "to be checked": since 08-23 it is **frozen as legacy**
(issue #40 — its reference-validity term was decided by 2/600 items; PR #48). The
reported primary is the **six-term decomposition report card**: break EM, SARI, GED,
chain validity, hop-count EM, under-/over-decomposition (exp-016 re-scored all 12 arms
under it). Replace the composite row's caption with: *"Legacy ranking blend used by
the earlier pool sweep; frozen 2026-08-23 and superseded by the decomposition report
card as the reported primary."* Pool-sweep slides that show it should label it
"legacy ranking metric".

### 2.4 Pool building — the deck's open decision 2 has since been answered

The deck's ablation slides (33-cell sweep: peak at 4000, rerank doesn't move the
composite) need **one factual correction plus an ending**.

**Correction — "quality peaks at 4000" is not supportable.** The 2026-08-20 re-analysis
of that same v1 sweep (27 matched pairs, n = 750, Holm-corrected) finds **1000 measurably
worst** (1000 vs 8000: step F1 favours the larger pool 6/6, 4 surviving Holm) and **no
doubling above 1000 significant on any metric** (4000 vs 8000: 0 of 30 metric-cells
CI-significant, smallest Holm-adjusted p 0.2855). The "peak at 4000" reading is the legacy
composite's reference-validity term — **104.2 %** of the 4000→8000 drop, turning on **13 vs
37 `[#k]` references** — and it *reverses* on a reference-free metric (0.2312 → 0.2465,
monotone up from 1000 to 8000). Full numbers and a drop-in slide sentence:
[`docs/analysis/2026-08-27-pool-construction-slide-facts.md`](../docs/analysis/2026-08-27-pool-construction-slide-facts.md)
§1. Label the slide **v1 prior work** — there was no v2 pool-size rerun.

And the story now has an ending the deck lacks:

- **The pool choice was made.** Jahid delegated it (ADR 0028 item 1) and
  `size2000_imbalanced` was fixed as the operating pool for the prompting arm.
- **exp-014** built the reranked top-5 retrieval artifact over that pool (deliverable 2
  of its run) — the artifact exp-015 then consumed.
- **exp-015** evaluated prompting *at that best pool* and it still lost to the
  fine-tune 10/10 (see §3.2). So pool engineering was given its best case and measured.
- **exp-010 evaluated all three construction strategies** (2026-08-22, 18 cells =
  3 strategies x 2 retriever variants x 3 mask modes, n = 750). Step F1, ordered accuracy,
  ROUGE-L and EM are **non-significant in all three pairwise comparisons**; the only axis
  that separates them is hop-count EM, where imbalanced and clustered **tie exactly at
  0.5133** and both beat balanced at 0.4667 (McNemar p = 0.0438 / 0.0380). Clustering does
  not beat imbalanced.
- **Consequence for the "Three to agree on" slide:** open decision 2 ("contribution to
  defend or hyperparameter to fix and move past?") is effectively answered by the
  evidence — *hyperparameter, fixed and moved past*.

```text
Add one closing slide to the Ablations section. Eyebrow: "MUSIQUE · POOL SWEEP →
DECISION". Title: "The pool question, closed". Three-step horizontal flow with arrows:

Step 1 card — "33-cell sweep (v1)" — "Pool size, balance, retrieval mode, reranking.
Size 1000 is measurably worst; no doubling above 1000 is significant. Cross-encoder
rerank does not move the ranking metric (legacy composite)."
Step 2 card — "Construction measured" — "All three strategies evaluated at size 2000
(exp-010, n = 750): step F1, ordered accuracy, ROUGE-L and EM non-significant in every
pair. Only hop-count EM separates — imbalanced and clustered tie at 0.5133, balanced
trails at 0.4667."
Step 3 card — "Pool fixed" — "size-2000 imbalanced chosen as the operating pool on
measured indifference (delegated decision, ADR 0028); a reranked top-5 retrieval
artifact built over it (exp-014)."
Step 4 card, shaded light blue — "Best case, measured" — "Prompting evaluated at this
best pool still loses to the fine-tuned decomposer on 10 of 10 paired statistics
(exp-015)."

Bottom line in grey: "Conclusion carried into the thesis: pool construction is a
hyperparameter to fix, not a contribution to defend — measured, not assumed."
```

---

## 3. New results worth a slide tonight

### 3.1 End-to-end MuSiQue answering (exp-017) — the deck's own open question, answered

Same reader (Mistral-7B-Instruct), same full-paragraph context, same pinned 600
(200/hop) in all four cells; only the decomposition varies.

| Decomposition source | Answer EM | Answer F1 | EM by hop (2/3/4) |
|---|---|---|---|
| None (question asked whole; control) | 18.0% | 27.2% | 21.5 / 15.5 / 17.0 |
| Prompted, unguided (exp-004) | 21.3% | 29.5% | 29.0 / 20.0 / 15.0 |
| Fine-tuned full_train (exp-008) | 24.2% | 32.8% | 34.0 / 24.0 / 14.5 |
| Oracle gold decompositions (ceiling) | **28.0%** | **37.3%** | 40.0 / 30.5 / 13.5 |

Caveats to state on the slide: at **4-hop the ordering collapses** (all cells
13.5–17.0%, no-decomposition nominally highest) — decomposition currently pays off at
2–3 hops only; the no-decomposition control is a lead-added condition pending
Jahid's + supervisor's acceptance; no significance tests between these four cells yet.
Cite: exp-017 log row (completed, commit `edfa27e`) · `runs/exp-017/*/answer_metrics.json`.

```text
New slide. Eyebrow: "MUSIQUE DEV · PINNED 600, SAME READER AND CONTEXT IN ALL CELLS".
Title: "Better plans produce better answers". Table with columns DECOMPOSITION SOURCE /
ANSWER EM / ANSWER F1 / EM BY HOP (2 / 3 / 4):

None — question asked whole (control) | 18.0% | 27.2% | 21.5 / 15.5 / 17.0
Prompted, unguided | 21.3% | 29.5% | 29.0 / 20.0 / 15.0
Fine-tuned (full_train) | 24.2% | 32.8% | 34.0 / 24.0 / 14.5
Oracle gold decompositions (ceiling) | 28.0% bold | 37.3% bold | 40.0 / 30.5 / 13.5

Shade the oracle row light blue as the ceiling. Two short text columns below the
table: left — "Answer accuracy rises monotonically with decomposition quality:
+3.3 points from decomposing at all, +2.9 more from fine-tuning, and a ceiling 3.8
above that with gold plans." right — "At 4 hops the ordering collapses: all four cells
sit at 13.5–17.0% and the no-decomposition control is nominally highest — decomposition
pays off at 2–3 hops only. No significance tests between cells yet; the control cell is
a lead-added condition pending supervisor acceptance." Footnote: "Source: exp-017,
Mistral-7B reader, full-paragraph context, 200 questions per hop."
```

### 3.2 Prompting vs fine-tuning, settled on the same 600 (exp-015)

Prompting got its best shot (best pool from the sweep, reranked top-5) vs the exp-008
fine-tune: **10 of 10 paired statistics significant, every one favouring the
fine-tuned arm.** Step F1 0.189 vs 0.341; ROUGE-L 0.545 vs 0.665.
Cite: exp-015 log row · `experiments/exp-015/notes.md`.

```text
New slide. Eyebrow: "MUSIQUE DEV · PINNED 600 · PAIRED BOOTSTRAP + McNEMAR".
Title: "Fine-tuning beats prompting at its best". Left: a two-row comparison table
VARIANT / STEP F1 / ROUGE-L F1: "Prompting, best pool (reranked top-5)" 0.189 / 0.545;
"Fine-tuned LoRA (full_train)" 0.341 bold / 0.665 bold. Right: a large stat callout
"10 / 10" with caption "paired statistics significant — every one favours the
fine-tuned arm (7 bootstrap, 3 McNemar)". Footnote: "Source: exp-015; prompting arm
uses the sweep's best pool, so this is prompting at its strongest measured
configuration."
```

### 3.3 Generalisation: the fine-tune is dataset-bound in a specific, showable way (exp-013)

Same LoRA recipe trained on 2/3-hop only, 4-hop held out. On the unseen depth it still
emits well-formed chains (chain validity **200/200**) but systematically too short
(**169/200 under-decomposed**, hop-count EM 14.5%). Against the prompting baseline at
hop 4 it wins the content metrics (step F1 0.198 vs 0.117, significant) yet **loses
hop-count EM** (14.5% vs 30.0%, McNemar p≈5e-5). One-liner: *"the fine-tune transfers
form, not depth."* A 2/3-hop-trained vs all-hop-trained comparison is also in exp-013's
notes (comparison B) if pressed. Cite: exp-013 log row · `experiments/exp-013/notes.md`
· issue #41.

```text
New slide. Eyebrow: "MUSIQUE DEV · 4-HOP HELD OUT OF TRAINING (N = 200)".
Title: "The fine-tune transfers form, not depth". Three stat tiles across the top:
"200 / 200 — chains still well-formed on the unseen depth (chain validity 1.000)";
"169 / 200 — but systematically too short (under-decomposition 0.845)";
"14.5% — hop-count exact match on 4-hop, vs 30.0% for the prompting baseline".
Below, a small comparison table vs the prompting baseline at hop 4 only: STEP F1 0.198
vs 0.117 (significant, favours fine-tune); HOP-COUNT EM 14.5% vs 30.0% (McNemar
p≈5e-5, favours prompting). Bottom line: "Trained on 2- and 3-hop only, the model keeps
producing valid chains at 4 hops — just too few steps. Fine-tuning is dataset-bound in
depth, not in form." Footnote: "Source: exp-013 (LoRA trained on 2/3-hop, evaluated on
the pinned 600's 4-hop slice)."
```

### 3.4 Statistical testing — a slide of its own

The deck states the reporting rule ("a paired significance test on any comparison
narrower than a point") but shows no test results — and the limitations slide even
said none had been run. The logged record is now substantial and worth one slide:

**Protocol** (ADR 0009/0011, ADR 0017 item 4): every comparison is paired on the same
pinned 600 items (200/hop); continuous metrics get a paired bootstrap (10,000
resamples, 95% CI), binary metrics (exact-match-type) get McNemar, and a paired t-test
runs alongside as an agreement check.

**Logged results:**

| Comparison | Where | Outcome |
|---|---|---|
| Prompting (unguided) vs fine-tuned full_train | exp-014, n=600 | Every reported metric significant, all favouring the fine-tune: hop-count EM McNemar p=1.1e-15; GED paired t=+15.87 (p≈1e-47); step F1 CI [−0.160, −0.115]. |
| Prompting at its **best pool** vs fine-tuned | exp-015, n=600 | 10/10 statistics significant (7 bootstrap + 3 McNemar), every one favouring the fine-tune. |
| 2/3-hop-trained LoRA vs prompting, 4-hop only | exp-013, n=200 | Content metrics significant favouring the fine-tune (step F1 0.198 vs 0.117); hop-count EM flips: McNemar p=4.7e-5 favouring prompting. |
| Oracle-guided vs unguided (Mistral) | exp-014 | **Two** statistics significant, both favouring the oracle: hop-count EM 0.5083 → 0.5900 (McNemar p=2.41e-04) and GED 0.4715 → 0.4414 (CI [+0.0167, +0.0437]). The *content* metrics straddle zero (step F1, ordered step acc., SARI, chain validity) — so the hop count moves count accuracy, not step quality. |
| Oracle-guided vs unguided (**Qwen3.5-9B**) | exp-014 | A **different result on a different model**, same eval set and protocol: step F1 0.2131 → 0.2368, ordered step acc. 0.1911 → 0.2205, ROUGE-L, exact match, GED and hop-count EM 0.5200 → 0.8733 (McNemar p=5.0e-48) — **all significant** favouring the oracle. Only SARI and chain validity straddle zero; break-EM underpowered (p=0.0625, reported as such). |
| + 3 further arm pairs | exp-014 | Logged with the same protocol (capped variant = noise-scale in both models). |

> **Correction (2026-08-27).** The row above previously read *"Only GED significant"* for the
> Mistral oracle pair. That was scoped to the four Break-faithful columns PR #44 added and
> dropped `hop_count_exact_match`, which **is** significant (p=2.41e-04) and is the metric the
> guided-vs-unguided question exists to answer. Verified against
> `experiments/exp-014/metrics.json` → `deliverable_1_compare.pairs.exp004_unguided_vs_exp004_oracle_guided`.
> The error propagated onto slide 25 of `Artifact slides review.pdf`. The Qwen row was also
> compressed into a parenthetical ("broader significance") and is now stated, because it is the
> only cell in the study where hop guidance moves step *content* — see issue on model coverage.

**Discipline points worth saying out loud tonight:** underpowered McNemar cells are
reported as underpowered rather than cherry-picked; a technically-significant
noise-scale effect (GED delta 0.0007) is called noise, not a finding; still untested —
typed-vs-uniform masking and the four exp-017 end-to-end cells.

```text
New slide. Eyebrow: "STATISTICAL TESTING · PAIRED, SAME PINNED 600 BOTH SIDES".
Title: "Every claim narrower than a point gets a paired test". Top: one line
describing the protocol — "Paired bootstrap, 10,000 resamples with 95% CIs, for
continuous metrics; McNemar for exact-match metrics; paired t-test alongside as an
agreement check. Both sides of every comparison are the same 600 questions."

Table with columns COMPARISON / N / RESULT:
"Prompting vs fine-tuned (full_train)" | 600 | "All metrics significant, all favour
the fine-tune — hop-count EM McNemar p = 1.1e-15, GED paired t = +15.9."
"Prompting at its best pool vs fine-tuned" | 600 | "10 / 10 significant (7 bootstrap,
3 McNemar), every one favours the fine-tune."
"2/3-hop-trained LoRA vs prompting, on held-out 4-hop" | 200 | "Content metrics favour
the fine-tune; hop-count EM flips to the prompting side (McNemar p = 4.7e-5)."
"Oracle hop count vs unguided, Mistral-7B" | 600 | "Hop-count EM and GED significant,
content metrics flat — the hop count moves count accuracy, not step quality."
"Oracle hop count vs unguided, Qwen3.5-9B" | 600 | "Every content metric significant
too, and hop-count EM 0.52 to 0.87 — the ceiling is model-dependent."

Bottom grey line, three clauses: "Underpowered McNemar cells are reported as
underpowered, and noise-scale significant effects are named as noise. Every other
comparison on this slide is Mistral-7B only. Not yet tested: typed-vs-uniform masking;
the four end-to-end answering cells."
Sources footnote: "exp-013, exp-014, exp-015 log rows."
```

### 3.5 "Three to agree on" — decisions 1 and 2 largely settled

Decision 1 (thesis-primary metric) has an answer on record: the decomposition report
card is the reported primary (PR #48), with end-to-end answer accuracy as the
downstream complement. Decision 2 (pool construction) is answered by the evidence in
§2.4: hyperparameter, fixed at size-2000 imbalanced and moved past. Present both as
proposals to confirm, not open decisions — only decision 3 (whether both datasets stay
in scope) remains genuinely open, pending the supervisor's GRAG system for the MetaQA
half.

---

## Not re-verified today

The MetaQA router tables (90.33% etc.), similarity-routing modes A/B/C
(60.7 / 47.0 / 72.3), and the KG execution counts (177 / 114 / 9) trace to v1-era
artifacts the deck already cites (`results_analysis/router/report.html` and
neighbours). Internally consistent with the deck's own footnotes, but not
independently re-scored today. All MuSiQue-side numbers above were checked against the
v2 log and run artifacts on 2026-08-25.
