# Anchor-set qualitative analysis — 45 items × 8 arms

Supervisor item **A7** ("eye-to-eye comparison… pick some bad example with unguided and see how it
improves… check them with Quen and Mistral. Maybe Mistral is bad in this capability",
phase plan §2 A7). This is a **first pass so Jahid can read the sheet himself with a map**. It is
not a substitute for his own reading; §7 says where to start.

- **Data:** the 45 frozen anchor ids (`configs/anchor_set_45.json`, ADR 0033), 15 per gold hop
  depth, over 8 arms. The rendered sheet is `runs/anchor/2026-09-20-all-arms.md` (uncommitted,
  gitignored — the data stays there).
- **Every count here is recomputed from the committed per-item files and the runs' `results.json`,
  not read off the sheet.** Sources are listed in the companion
  [`2026-09-20-anchor-set-qualitative-analysis.json`](2026-09-20-anchor-set-qualitative-analysis.json)
  under `sources`, one `per_item` + `results` path per arm.
- **Nothing was re-run and no model was run.** No GPU. Read-only over finished artefacts.
- The note carries **8 short excerpts**, each marked as an excerpt, to make a pattern concrete.
  Everything else is counts, categories and item ids.

Arms, in the column order used throughout: **exp-020** Mistral-7B 4-bit as_is · **exp-023** its
reversed twin · **exp-021** Qwen2.5-7B 4-bit as_is · **exp-024** its reversed twin · **exp-022**
Qwen2.5-14B 4-bit as_is · **exp-025** its reversed twin · **exp-008** full_train fine-tune
(19,938 ex.) · **exp-026** pool-only fine-tune (2,000 ex.).

---

## 0. The seven things this pass found

Counts first. Each is expanded below with its section.

1. **29% of the anchor set is a metric problem, not a model problem.** On **13 of 45** items, all
   eight arms score `step_f1 = 0.0` — including items where all eight produce the *same*, and
   arguably correct, plan. Replacing exact step-string matching with token-overlap matching lifts
   mean step-F1 from **0.155–0.305** to **0.50–0.68** on the same 45 (§1.5).
2. **A sub-question is computed and then never used — on up to 20 of 45 items — and
   `chain_validity` scores every one of them 1.0.** Across the 8 arms, **86** item×arm pairs carry
   a dangling step; **85 of the 86** have `chain_validity = 1.0`. All 45 golds are fully chained
   (0 dangling). This failure mode is invisible to every metric on the report card (§1.3).
3. **Reversing the few-shot order changes the text on 37, 37 and 41 of 45 items** (Mistral,
   Qwen-7B, Qwen-14B) — **82% of the full 600** — and the per-item score moves up and down in
   almost equal numbers. The 0-of-13 null in the block report is **cancellation, not
   insensitivity** (§3).
4. **Mistral's format failures are real, Mistral-only, and small in magnitude.** Prompt scaffold
   lines (`Task:`, `Answer:`) are parsed as steps in **24/600** exp-020 and **31/600** exp-023 rows
   and **0/600** in every other arm — but removing them moves over-decomposition only
   0.3200 → 0.3050 (§1.2).
5. **Qwen2.5-14B has its own signature over-decomposition move: the yes/no verification step**
   ("Was Sarah married to #1?"). **35/600** exp-022 and **47/600** exp-025 items, against ≤7/600
   in every other arm and **0** in gold (§5).
6. **Hallucinated entities are essentially not a failure mode here.** Across 360 item×arm pairs,
   **9** carry an entity that is in neither the question nor the few-shot block, and exactly
   **1** of those is an entity copied from the prompt's own examples (§1.4).
7. **Ahmmad's hypothesis, answered with counts: partly.** On the 12 items where exp-020 is
   worst, Qwen-7B is better on 5, same on 6, worse on 1; Qwen-14B better on 6, same on 5, worse
   on 1. Over all 45, Qwen-7B is better on 9, same on 20, **worse on 16** — no net advantage. The
   capability Mistral is specifically bad at is **output-format discipline**, not decomposition
   content (§2).

**One caveat that binds everything below.** The 45 are **not** a scale model of the 600. The 15
three-hop anchors are markedly harder than the other 185 three-hop items, and the 15 four-hop
anchors are easier than the other 185 — **consistently in the same direction for all eight arms**,
so this is a property of the draw, not of any model. Example (exp-026): step_f1 0.209 on the 15
three-hop anchors vs 0.296 on the other 185; 0.271 on the 15 four-hop anchors vs 0.212 on the
other 185. Full table in the JSON under `anchor_vs_rest_of_600_by_hop`. **Read the anchors for
mechanism; read the 600 for numbers.**

---

## 1. The error taxonomy, built from what is in the sheet

Categories were not taken from a list — they are the failure shapes that actually occur in these
360 item×arm pairs. Each detector is defined in the JSON under `category_definitions`; each was
hand-checked on the cases it fires on. An item×arm pair can carry several.

### 1.1 Counts, all 45 items

| category | 020 | 023 | 021 | 024 | 022 | 025 | 008 | 026 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| right number of steps | 27 | 26 | 25 | 23 | 28 | 30 | 32 | 25 |
| too many steps | 12 | 15 | 6 | 8 | 14 | 10 | 4 | 4 |
| too few steps | 6 | 4 | 14 | 14 | 3 | 5 | 9 | 16 |
| runaway (≥10 steps) | 0 | **1** | 0 | 0 | 0 | 0 | 0 | 0 |
| prompt-scaffold line as a step | **1** | **4** | 0 | 0 | 0 | 0 | 0 | 0 |
| template loop (≥3 near-identical steps) | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 |
| duplicate step | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 |
| **orphan step (computed, never used)** | **14** | **15** | **8** | **8** | **20** | **13** | **5** | **3** |
| out-of-range / self `#k` | 1 | 0 | 2 | 2 | 0 | 0 | 0 | 3 |
| no `#k` at all though gold chains | 0 | 2 | 3 | 2 | 0 | 2 | 0 | 1 |
| `[#k]` bracket syntax | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| a step restates the whole question | 4 | 2 | 3 | 4 | 1 | 3 | 0 | 3 |
| entity copied from the few-shot block | 0 | 0 | 0 | 0 | **1** | 0 | 0 | 0 |
| entity in neither question nor examples | 0 | 1 | 1 | 1 | 0 | 1 | 2 | 3 |
| whole plan verbatim = gold | 1 | 3 | 5 | 3 | 2 | 2 | 4 | 6 |
| right length, `step_f1`=0, ROUGE-L ≥0.5 | 6 | 4 | 3 | 5 | 4 | 3 | 7 | 5 |

Per-hop breakdowns of every row (15 items per hop) are in the JSON under
`counts_per_arm_per_hop_of_15`. The ones that move are discussed in §6.

**Categories that do not appear at all, stated so they are not assumed:** `<think>`-style
reasoning leakage — **0/360**; answering instead of decomposing (a declarative answer emitted in
place of a plan) — **0/360**; wrong decomposition *order* with otherwise-correct steps (same step
set, permuted) — **0/360**. Ordering errors show up only as content errors, never as pure
permutations.

### 1.2 The format failures are Mistral's alone, and the parser helps them along

Both Mistral arms sometimes keep generating past their answer and start the *next* few-shot block.
Six exp-020 anchor generations and four exp-023 ones contain a second `Question:` / `Decomposition:`
pair. The step parser drops `Question:`-led continuations but **not** `Task:` or `Answer:`, so
those two land in `pred_steps` and are scored as steps.

> **Excerpt 1** — `4hop1__788226_32392_823060_610794`, exp-020, predicted steps 3 and 4:
> `#2 >> county of` / `Task:`

Measured over the full 600: **24** exp-020 rows and **31** exp-023 rows have a scaffold line
counted as a step; **0** in exp-021, exp-022, exp-024, exp-025, exp-008, exp-026. Both Mistral arms
also hit the generation cap — **26/600** (exp-020) and **22/600** (exp-023) rows at
`hit_max_new_tokens`, against **0/600** in all six other arms.

**The honest size of it:** dropping scaffold lines moves exp-020's over-decomposition rate from
**0.3200 to 0.3050** and exp-023's from **0.2950 to 0.2817**. It is a real defect worth fixing, and
it explains ~1.5 points of a ~15-point gap. It is not the story.

**Is the 126-step plan an anchor? No.** exp-020's longest plan on the 600 is 126 steps at
`2hop__207476_625987`, which is *not* in the 45. But the same failure mode **is** in the anchor set
in the reversed twin: exp-023 on `3hop1__857_846_7870` emitted **79 steps**, hit the 1024-token cap,
and degenerated into a counter loop.

> **Excerpt 2** — `3hop1__857_846_7870`, exp-023, steps 2–5 of 79:
> `In #1, where are there lots of bars?` / `In #2, where are there lots of bars?` /
> `In #3, where are there lots of bars?` / `In #4, where are there lots of bars?`
> (exp-020 answers the same item in 3 steps; all four Qwen arms in 2–3.)

### 1.3 The biggest finding: the orphan step, and the metric that cannot see it

A **fully chained** plan is one where every non-final step is consumed by some later step. **All
45 gold decompositions are fully chained** — 45/45 emit at least one `#k`, and **0/45** leave a
step dangling. The predictions do not behave that way:

| arm | 020 | 023 | 021 | 024 | 022 | 025 | 008 | 026 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| items with ≥1 dangling step | 14 | 15 | 8 | 8 | **20** | 13 | 5 | 3 |
| …of which `chain_validity` = 1.0 | 14 | 15 | 8 | 7 | 20 | 13 | 5 | 3 |

**85 of the 86** dangling-step cases score a perfect 1.0 on `chain_validity`. That is by
construction: `chain_validity` (METRICS.md §2.1) checks that each `#k` points *backwards* to an
existing step. It never checks that every step is *used*. A plan that computes a fact and then
throws it away is, to the report card, perfectly chained.

This is not a length artefact. Splitting the dangling cases by whether the plan had the right
number of steps: exp-022 **9 of its 20** dangling-step items have exactly the gold step count;
exp-020 **7 of 14**; exp-025 **7 of 13**. The most common concrete shape is an **off-by-one stale
referent** — the final step refers to `#1` where `#2` was meant.

> **Excerpt 3** — `3hop1__30240_740597_41132` (3 hops), exp-022 and exp-025, step 3:
> `How many times did plague occur in #1?` — where `#1` is the *person* and `#2` is the birthplace
> the step is asking about. `chain_validity` for both: **1.0**.

The fine-tuned arms are far cleaner here (5 and 3 of 45) — which is a finding in its own right,
since the fine-tunes were never optimised for this property.

### 1.4 Hallucinated entities: almost absent

Across 360 item×arm pairs, only **9** contain a capitalised entity that appears neither in the
question nor (for the six prompting arms) in that item's own few-shot block: `Islam` (exp-021,
exp-024), `Pakistan` (exp-025), `EMI Group` (exp-008), `California` (exp-026), `Romance`
(exp-023), `John` (exp-008), `American` (exp-026), `Genius of Victory` (exp-022). Hand-checked,
about **7** are genuine *pre-solved hops* — the model answers a hop inside the sub-question instead
of leaving a slot for it. Two (`John`, `American`) are elaborations, not inventions.

Exactly **one** of the nine is entity bleed from the prompt's own examples, and it is instructive:

> **Excerpt 4** — `2hop__115515_779396`, question *"Where was the creator of the Lap Engine
> educated?"*, exp-022 step 1: `The Genius of Victory >> creator` — *The Genius of Victory* is the
> subject of the fifth few-shot example in that item's prompt, not of the question. (exp-008
> answers this item exactly: `Who developed Lap Engine?` / `#1 >> educated at`.)

**Read this as a negative result:** the errors in this block are *structural* (length, chaining,
granularity) and *lexical* (wording), not factual. Nothing suggests a hallucination problem.

### 1.5 Most of the step-F1 shortfall is wording, not wrong sub-questions

`step_f1` (METRICS.md §2) matches steps as **exact normalised strings**. Re-scoring the same 45
predictions against the same golds with a step matched when token-overlap F1 clears a threshold —
**an unregistered scratch diagnostic, not a metric in `METRICS.md`; two thresholds are shown so the
reader can see how much it depends on one** — gives:

| arm | evaluator `step_f1` | soft @ ≥0.60 | soft @ ≥0.50 |
|---|---:|---:|---:|
| exp-020 | 0.1552 | 0.5721 | 0.7423 |
| exp-023 | 0.1921 | 0.5109 | 0.6837 |
| exp-021 | 0.1942 | 0.4951 | 0.6756 |
| exp-024 | 0.1746 | 0.5527 | 0.6635 |
| exp-022 | 0.1931 | 0.5892 | 0.6753 |
| exp-025 | 0.1928 | 0.5753 | 0.6913 |
| exp-008 | 0.3004 | 0.6756 | 0.8146 |
| exp-026 | 0.3045 | 0.6537 | 0.7649 |

Two things follow. **(a)** The fine-tune advantage survives relaxation but shrinks: +0.11 exact,
+0.08 to +0.18 at 0.60. Fine-tuning is doing more than learning gold phrasing, but a large part of
what it does is learn gold phrasing. **(b)** The *ranking among prompting arms is not stable under
relaxation* — exp-020 is last on exact `step_f1` and first among prompting arms at the 0.50
threshold. Any claim of the form "model X writes better steps than model Y" that rests on exact
`step_f1` alone is fragile.

The 13 items where **all eight** arms score `step_f1 = 0.0` are the sharpest version of this:

> **Excerpt 5** — `2hop__152027_141308`, *"What is the parent company of the record label releasing
> Cold Turkey?"*. Gold: `What was the record label of Cold Turkey?` / `What company is #1 part
> of?`. **Seven of the eight arms** emit, character for character, `Cold Turkey >> record label` /
> `#1 >> parent company` (exp-008 differs only in `#1 >> owned by`). `step_f1 = 0.0` for all eight;
> soft@0.60 = 0.94; ROUGE-L F1 averages 0.261 across the arms.

Note that ROUGE-L is *also* low here (0.261) — this item is a case where the `>>` relational style
and the natural-question style are both licensed by MuSiQue's gold but share almost no tokens. The
full list of 13 is in the JSON (`items_all_8_arms_step_f1_zero`).

At the other end: exactly **one** of the 45 is answered exactly by all eight arms
(`2hop__89403_602275`), and only **8 of 45** are answered exactly by at least one arm.

---

## 2. Ahmmad's question: is Mistral bad at a specific capability?

**Definition of "bad".** The 12 anchor items with the lowest exp-020 per-item `decomp_mean`
(the blended per-item term; exp-008's stored file predates it, so this comparison is between the
three prompting families only, which is exactly what he asked for). Verdict threshold: a
`decomp_mean` gap of more than 0.05 counts as better/worse, otherwise same.

**On those 12 items:**

| | better than Mistral | same | worse |
|---|---:|---:|---:|
| Qwen2.5-7B (exp-021) | **5** | 6 | 1 |
| Qwen2.5-14B (exp-022) | **6** | 5 | 1 |

**Over all 45 items** — the control that stops this from being regression to the mean:

| | better | same | worse |
|---|---:|---:|---:|
| Qwen2.5-7B vs Mistral-7B | 9 | 20 | **16** |
| Qwen2.5-14B vs Mistral-7B | 11 | 23 | 11 |
| Qwen2.5-14B vs Qwen2.5-7B | **14** | 20 | 11 |

**The answer, in one line.** Swapping to Qwen rescues about half of Mistral's worst items, but
loses as many elsewhere — over the whole 45 there is no net advantage for Qwen-7B, and Qwen-14B is
a wash against Mistral while clearly ahead of Qwen-7B. So **"Mistral is bad at this" is not
supported as a content claim.** What *is* supported, and is a clean capability difference, is
**output-format discipline**: scaffold-line leakage 1 and 4 of 45 (exp-020, exp-023) vs **0 of 45**
in all four Qwen arms; runaway generations 1 of 45 vs 0; generation-cap hits 3 and 3 of 45 vs 0;
and on the full 600, 24 and 31 scaffold rows and 26 and 22 cap hits vs 0 everywhere else.

Three items that show what "better" and "worse" look like:

> **Excerpt 6 (Qwen better)** — `2hop__776637_87295`, *"Who does the cast member from Julie play in
> Princess Diaries?"*. Gold: `Julie >> cast member` / `who does #1 play in princess diaries`.
> **exp-021 and exp-026 reproduce it exactly.** exp-020 and exp-023 invent a preamble hop:
> `What movie is Princess Diaries?` / `Who is the cast member from #1?` / `Who does #2 play in #1?`
> — a lookup for an entity the question already names.

> **Contrast (Qwen worse), scores only** — `4hop2__103790_14670_8987_8974` (the "Country A / Country B" item).
> exp-020 `decomp_mean` 0.319, exp-021 **0.115** — Qwen-7B writes 7 steps for a 4-step gold, emits
> a forward reference, and scores `chain_validity` 0.0.

> **Contrast (size helps), scores only** — `3hop1__337705_132457_47686`. exp-020 and exp-021 both collapse the
> band-lookup and the city-lookup into one step (2 steps for a 3-step gold); exp-022 and exp-025
> keep them apart and score `step_f1` 0.33 where the 7B scores 0.00.

---

## 3. Ordering: the per-item view of the 0-of-13 null

This is the section that most changes how §5.4 of the block report should be read.

| pair | model | anchor items where `pred_steps` changed | identical | of the changed: `decomp_mean` up / down / flat | mean Δ over 45 |
|---|---|---:|---:|---|---:|
| exp-020 → exp-023 | Mistral-7B | **37 / 45** | 8 | 10 / 13 / 14 | −0.0132 |
| exp-021 → exp-024 | Qwen2.5-7B | **37 / 45** | 8 | 12 / 14 / 11 | −0.0177 |
| exp-022 → exp-025 | Qwen2.5-14B | **41 / 45** | 4 | 14 / 13 / 14 | −0.0017 |

Checked on the full 600, so this is not an anchor-set quirk:

| pair | changed / 600 | up | down | flat |
|---|---:|---:|---:|---:|
| Mistral-7B | **491 (81.8%)** | 177 | 166 | 148 |
| Qwen2.5-7B | **496 (82.7%)** | 185 | 188 | 123 |
| Qwen2.5-14B | **492 (82.0%)** | 159 | 169 | 164 |

Per-item moves are large in both directions — the biggest single anchor swings are
`4hop1__767417_624859_355213_203322` **+0.593** (Mistral gains a full exact match when the examples
are reversed) and `3hop1__107195_160713_77246` **−0.511** (Mistral loses chaining entirely).

**What this means, said carefully.** The block report's "reversing the example order changes
**nothing**" is correct *about the aggregate*, and the 0-of-13 significance result stands. But the
mechanism is **not** "the model ignores the order". The model is highly sensitive to it — it
rewrites four plans in five — and the rewrites are **as likely to help as to hurt**, so they cancel
in the mean. That is a different and more useful claim for the thesis, and it also means the
per-item variance induced by an irrelevant prompt permutation is **comparable in size to the
effects the block is trying to measure**. Whether that undermines single-run arm comparisons is a
question this pass cannot answer — nothing in this block re-ran an arm under a different seed
(block report §6).

---

## 4. Fine-tune vs prompting, item by item

**The two fine-tunes are not the same model with the same answers.** exp-008 and exp-026 agree
byte-for-byte on only **4 of 45** items (7 of 45 after case/punctuation normalisation). On the 45,
their per-item wins are nearly even — `step_f1` 9–9, `ordered_step_accuracy` 8–6, ROUGE-L 21–19,
SARI 23–18. The 45 have **no power** to resolve the four statistics that separated them on the 600
(block report §5.5a); do not try to read that comparison off the sheet.

**How much gold wording the fine-tunes actually reproduce:**

| arm | whole plan verbatim | ≥1 gold step verbatim | gold steps reproduced verbatim | ROUGE-L F1 ≥ 0.80 |
|---|---:|---:|---:|---:|
| exp-020 | 1/45 | 16/45 | 20/135 (14.8%) | 5/45 |
| exp-023 | 3/45 | 17/45 | 24/135 (17.8%) | 5/45 |
| exp-021 | 5/45 | 15/45 | 24/135 (17.8%) | 5/45 |
| exp-024 | 3/45 | 15/45 | 23/135 (17.0%) | 3/45 |
| exp-022 | 2/45 | 19/45 | 25/135 (18.5%) | 5/45 |
| exp-025 | 2/45 | 18/45 | 25/135 (18.5%) | 5/45 |
| **exp-008** | 4/45 | **29/45** | **38/135 (28.1%)** | 10/45 |
| **exp-026** | **6/45** | 26/45 | **37/135 (27.4%)** | **12/45** |

So the fine-tunes reproduce gold *step strings* about **1.6×** as often as prompting — and still
paraphrase roughly **72%** of gold steps. "The fine-tune memorised the gold phrasing" is too
strong; "the fine-tune learned the *house style* (`X >> relation`, lowercase, MuSiQue's relation
vocabulary) and applies it to new content" fits the evidence better. That is also why its whole-plan
verbatim rate stays low (4 and 6 of 45) while its per-step verbatim rate nearly doubles.

**Where prompting beats the fine-tune.** Rare, and concentrated:

- on `step_f1`, exp-020 beats exp-008 on **3/45**, exp-021 on **5/45**, exp-022 on **3/45**;
- on ROUGE-L, 11, 9 and 7 of 45 respectively; on SARI, 10, 10 and 8 of 45.
- **`4hop1__199881_765799_282674_759393` is the one item where all three prompting arms beat both
  fine-tunes** (step_f1 0.75 vs 0.29): the prompting arms lay out the full four-link chain, while
  both fine-tunes collapse it to three by dropping gold step 2 (`#1 >> capital of`).

That last pattern is the fine-tunes' characteristic weakness and it is the subject of §5.

---

## 5. Length-error mechanics: what the extra and the missing steps actually are

**Missing steps (under-decomposition).** Aligning predicted steps to gold steps by token overlap
and reading off which gold step has no partner: the missing hop is **predominantly interior**, not
the first and not the last. On under-decomposed **4-hop** items, the share of missing gold steps
that are gold step 2 or 3 is exp-026 **10/14**, exp-008 **7/9**, exp-021 **8/13**, exp-024 **6/11**.
The models get the entity lookup right, get the final composition right, and skip a link in the
middle.

> **Excerpt 7** — `4hop1__767417_624859_355213_203322`. Gold: `Change of Heart >> performer` /
> `#1 >> place of birth` / `#2 >> located in the administrative territorial entity` /
> `#3 >> shares border with`. **exp-026 and exp-024** drop step 3 and jump from birthplace straight
> to `#2 >> shares border with`. **exp-008, exp-021, exp-023 and exp-025 get this item exactly
> right** — one of only 8 items any arm solves exactly.

I looked for a single mechanism and did **not** find a dominant one. The "skipped link is always
a granularity lift (city → county, town → state)" hypothesis is only **24%** of exp-026's missing
gold steps and **40%** of exp-008's — **suggestive, not established**. What *is* established is the
positional pattern above and one concrete item: on `4hop1__342858_131850_33952_34109`, **all eight
arms** drop gold step 1 (`Yuma >> located in the administrative territorial entity`); every model
treats "Yuma" and "Yuma County" as the same thing. The JSON lists all ten gold steps missed by ≥6
of the 8 arms.

**Extra steps (over-decomposition).** Classifying exp-020's 14 unmatched extra steps and exp-022's
extras by hand, they are of five kinds, and the mix differs by model:

- **Spurious preamble** — a lookup for an entity the question already names
  (*"What movie is Princess Diaries?"*, *"What is the title of the film?"*, *"Who is Jean
  Mathonet?"*). Present in Mistral and Qwen-14B.
- **Granularity split** — one gold hop becomes a relation hop plus a type-narrowing hop
  (*"#1 >> place of birth"* then *"What city is #2?"*). The mirror image of the missing interior
  step in §5's first half, and the most common extra in both Mistral and Qwen-14B.
- **Conjunct split** — one gold step over two entities becomes one step per entity
  (*"When did #1 open US assembly plants?"* / *"When did #2 open US assembly plants?"*).
- **Yes/no verification** — **Qwen-14B's signature, and it belongs to no other arm.**
- **Scaffold leak** — Mistral only (§1.2).

The verification step is measured, not impressionistic:

| items containing a yes/no step | 020 | 023 | 021 | 024 | 022 | 025 | 008 | 026 | gold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| anchor 45 | 0 | 0 | 0 | 0 | 3 | 5 | 0 | 0 | **0** |
| full 600 | 7 | 2 | 5 | 6 | **35** | **47** | 1 | 2 | — |

> **Excerpt 8** — `3hop2__132957_379231_40768`, exp-022 step 3: `Is Nissan the same company as both
> #1 and #2?` — a check, not a retrieval, and unanswerable by the retriever. gold has 0 such steps
> in all 45.

**Do the extra and the missing steps correspond to the same gold hops?** Partly — both cluster on
the *interior* of the chain (exp-022: 33 of its 43 unmatched predicted steps are in interior
positions; 12 of its 26 unmatched gold steps are interior, 9 final). The shared mechanism is that
models disagree with MuSiQue about **where one hop ends and the next begins**, in both directions.
Nothing here isolates a specific gold relation type as the culprit.

---

## 6. Per-hop patterns, 15 items each

Mean predicted length and mean signed step-count error on the anchors:

| arm | 2-hop n̄ / signed | 3-hop n̄ / signed | 4-hop n̄ / signed |
|---|---|---|---|
| exp-020 | 2.33 / **+0.33** | 2.93 / −0.07 | 4.20 / +0.20 |
| exp-023 | 2.20 / +0.20 | 8.13 / **+5.13** (the 79-step item) | 4.87 / +0.87 |
| exp-021 | 2.00 / 0.00 | 3.07 / +0.07 | 3.87 / −0.13 |
| exp-024 | 1.93 / −0.07 | 2.93 / −0.07 | 4.20 / +0.20 |
| exp-022 | 2.40 / +0.40 | 3.47 / +0.47 | 4.27 / +0.27 |
| exp-025 | 2.27 / +0.27 | 3.00 / 0.00 | 4.20 / +0.20 |
| exp-008 | 2.00 / 0.00 | 3.07 / +0.07 | 3.60 / **−0.40** |
| exp-026 | 2.00 / 0.00 | 2.93 / −0.07 | 3.20 / **−0.80** |

What changes with depth, per arm:

- **Mistral (020/023):** over-decomposes at every depth, and the *format* failures concentrate at
  4 hops — 1 of 1 scaffold leaks for exp-020 and 3 of 4 for exp-023 are 4-hop items. The longer the
  question, the more likely it is to run past its answer.
- **Qwen-7B (021/024):** the only arms that under-decompose *at two hops* (1 and 2 of 15), and the
  deficit grows sharply — 8 and 6 of 15 at four hops. Its reference failures are a **two-hop**
  problem on the anchors (`no #k at all`: 1 and 2 of 15 at two hops), matching the block report's
  600-item finding.
- **Qwen-14B (022/025):** the *only* arms with zero under-decomposition at two hops, and the
  highest over-decomposition at two hops (5 and 4 of 15). Its dangling-step count peaks at 3–4
  hops (7 and 10 of 15 for exp-022). Perfect `chain_validity` at every depth on the anchors.
- **Fine-tunes (008/026):** flawless on length at two hops (15/15 exact for both), and the
  shortfall is entirely at four hops (8 and 11 of 15 too few). exp-026 has **zero** dangling steps
  at four hops — when it shortens a plan it shortens it coherently.

**Do not read the anchor per-hop step_f1 as the per-hop story.** As flagged in §0, the three-hop
anchors are anomalously hard for every arm; on the anchors 3-hop step_f1 sits *below* 4-hop for all
eight arms, which reverses on the other 185 items per hop. The 600-item per-hop numbers in block
report §5.7 are the ones to quote.

---

## 7. What to look at first — 9 items

Ordered so the contrasts get sharper as you go. Each is a section of
`runs/anchor/2026-09-20-all-arms.md`; search the id.

| # | item id | hop | why it is instructive |
|---|---|---|---|
| 1 | `2hop__152027_141308` | 2 | Seven of eight arms write the *same* plan; all eight score `step_f1` 0.0. The clearest case that the metric, not the model, is failing. |
| 2 | `2hop__89403_602275` | 2 | The only item all eight arms get exactly right. Useful as the shape of "solved". |
| 3 | `2hop__776637_87295` | 2 | The cleanest Mistral-vs-Qwen contrast in the set: Qwen-7B and the pool fine-tune nail the gold; both Mistral arms insert a spurious preamble hop; Qwen-7B *reversed* breaks its reference. |
| 4 | `2hop__115515_779396` | 2 | The single case of an entity copied from the prompt's own few-shot block (exp-022) — next to exp-008 answering it exactly. |
| 5 | `3hop1__30240_740597_41132` | 3 | An off-by-one stale referent in exp-022 and exp-025 that `chain_validity` scores 1.0. Read this before trusting the chaining column. |
| 6 | `3hop1__857_846_7870` | 3 | exp-023's 79-step runaway loop, truncated at the token cap, beside exp-020's clean 3 steps on the same item. Mistral's catastrophic mode, live in the anchor set. |
| 7 | `3hop1__140786_2053_5289` | 3 | Widest over-decomposition spread in the set: exp-025 writes 2 steps, exp-022 writes 8, for a 3-step gold. Every arm scores `step_f1` 0.0. |
| 8 | `4hop1__342858_131850_33952_34109` | 4 | All eight arms drop the same gold step (`Yuma >> located in the administrative territorial entity`). A question about whether the gold hop is real. |
| 9 | `4hop1__199881_765799_282674_759393` | 4 | The one item where all three prompting arms beat both fine-tunes — the fine-tunes collapse a four-link chain to three. |

---

## 8. Ranked hypotheses

Ranked by expected impact on the report card. Each names the experiment that would test it. **These
are hypotheses, not recommendations about direction** — choosing among them is Jahid's call with his
supervisor.

**H1 — A large share of the prompting-vs-gold gap is style, not structure; a step-level soft-match
term would show it.** §1.5 measures exact `step_f1` 0.155–0.305 lifting to 0.50–0.68 under
token-overlap matching on the same predictions, and §1.1 finds 13 of 45 items scoring zero for all
eight arms, some of them with near-unanimous and plausible plans. *Test:* re-score the **existing**
per-item files of all eight arms with a soft step-match term added to the evaluator (no inference,
no GPU) and report it beside the exact term, as `chain_validity` was added beside
`reference_validity`. *Expected direction:* every arm rises; the prompting-vs-fine-tune gap narrows
but does not close; the ranking among prompting arms may reorder (it does in this scratch
calculation), which is itself the finding. *Risk:* a new metric invites the "you moved the
goalposts" objection — which is why it must be additive and reported next to the exact term, never
instead of it.

**H2 — `chain_validity` is blind to the dominant chaining defect, and a "plan connectedness" term
would separate the arms.** §1.3: 86 dangling-step cases across the eight arms, 85 of them scoring
`chain_validity` 1.0, against 0 in all 45 golds; the spread across arms is wide (3 for exp-026, 20
for exp-022) and is *not* explained by length. *Test:* add a per-item `plan_connectedness` term —
fraction of non-final steps consumed by a later step — and re-score the existing per-item files
(CPU only), then run the standard paired comparison. *Expected direction:* the fine-tuned arms gain
a clear, currently invisible advantage; Qwen-14B loses part of the structural credit it gets today.
*Why it ranks second:* it is as cheap as H1 and it measures something no current term measures, but
it affects fewer items.

**H3 — Mistral's over-decomposition is partly a post-processing defect, and fixing the step parser
moves its numbers.** §1.2: 24/600 (exp-020) and 31/600 (exp-023) rows contain a bare `Task:` or
`Answer:` scaffold line counted as a decomposition step, 0/600 in all six other arms. *Test:* add
`Task:` / `Answer:` to the parser's existing `Question:` / `Decomposition:` stop set and re-parse
the **stored** `decomposition_raw` (no inference), then re-evaluate. *Expected direction:*
exp-020's over-decomposition falls 0.3200 → 0.3050, exp-023's 0.2950 → 0.2817, and the
verbatim/EM terms rise slightly; **this is already measured here and is small.** It is ranked third
because the fix is nearly free and removes a confound from the family comparison, not because the
effect is large.

**H4 — Prompt-permutation variance is of the same order as the model effects being reported, so
single-run arm comparisons may be under-powered.** §3: reversing an irrelevant example ordering
rewrites 82% of predictions on the full 600 with per-item `decomp_mean` swings up to ±0.59, netting
to approximately zero. *Test:* re-run **one** arm (cheapest is exp-021, 813 s) at two or three
further example permutations — e.g. a random shuffle under a different seed — and report the
spread of each report-card term across permutations as an empirical noise floor, against which the
family (6/13) and size (7/13) contrasts can be read. *Expected direction:* if the per-term spread
across permutations is comparable to the 0.02–0.06 differences the block calls significant, several
of those results need a variance caveat; if it is an order of magnitude smaller, the block's
conclusions get stronger. **Either outcome is publishable and the experiment is under an hour of
GPU.** *Caveat:* this is compute and direction, so it is Jahid's and the supervisor's call.

**H5 — The fine-tunes' four-hop shortfall is a skipped interior link, and hop-count conditioning
would close part of it.** §5: on under-decomposed 4-hop items the missing gold step is interior in
10/14 (exp-026) and 7/9 (exp-008) cases, and §6 shows both fine-tunes are perfect on length at two
hops and short by 0.40–0.80 steps at four. *Test:* an arm that supplies the gold hop count in the
prompt to the **fine-tuned** decomposer — the `prompt_hop_count` / `hop_count_source` fields already
exist in `results.json`, so the plumbing is present. *Expected direction:*
`under_decomposition_rate` falls at four hops and `hop_count_exact_match_rate` rises; `step_f1` may
not move, because §5 shows the models also disagree about *where* the boundary is, not only how
many. *Ranked last* because it needs a real inference run and because the router work it touches is
an open defect (issue #47), and because the hypothesis about *which* link is skipped is only
suggestive (24–40% granularity lifts), not established.

---

## 9. What this note does not show

- **Nothing here is a significance test.** 45 items has no power; every count is descriptive. The
  significance claims live in the block report and the `compare_*` artefacts.
- **The soft step-F1 numbers in §1.5 are a scratch diagnostic**, not a registered metric, computed
  in a throwaway script with a hand-chosen threshold and a hand-chosen stopword list. They are
  reported at two thresholds for exactly that reason, and no arm ranking should be taken from them.
- **The category detectors are heuristics.** `orphan_step`, `question_restatement` and the entity
  detectors were hand-checked on the cases they fire on, but no manual sweep was done for cases they
  *miss*, so all category counts are lower bounds on the anchors.
- **The step-alignment used in §5 is token-overlap greedy matching at a 0.34 threshold**, not a
  semantic judgement. It is a way to point at which gold step went missing, not a claim that the
  model understood it.
- **No claim about hop 3 vs hop 4 difficulty should be taken from the anchors** (§0).
- **Nothing here is a comparison to exp-005 or exp-015**; those arms are not on this sheet.
