# A second random 45 (seed 43) — replication of the anchor-set analysis

Jahid, 2026-09-20: *"Pick another random 45 and analyze."* The purpose is to test whether the
patterns in [`2026-09-20-anchor-set-qualitative-analysis.md`](2026-09-20-anchor-set-qualitative-analysis.md)
hold on a fresh sample, and in particular whether that note's sampling caveat — the frozen 3-hop
anchors are harder and the 4-hop ones easier than the rest of the 600 — is specific to the frozen
draw.

- **The frozen 45 are untouched.** `configs/anchor_set_45.json` remains the official anchor set
  (ADR 0033) and `scripts/build_anchor_set.py --verify` re-derives it exactly, re-run today. The
  draw below is a **validation sample**, not a new anchor set, and its id file lives under `runs/`.
- **Draw.** `scripts/build_anchor_set.py --seed 43 --n-per-hop 15 --out runs/anchor/validation_45_seed43.json --write`,
  over the same ADR 0007 pinned 600 (three files, sha256s recorded in the companion JSON).
  Counts `{"2": 15, "3": 15, "4": 15, "total": 45}`. **Overlap with the frozen 45: 0 items**
  (0 at every hop), so no with/without split is needed — the two samples are disjoint and together
  cover 90 of the 600.
- **Sheet.** `runs/anchor/2026-09-20-validation-45-seed43.md`, eight arms, 45/45 items and 45/45
  raw generations each, rendered by `scripts/render_anchor_set.py` from the same per-item files and
  the same prompt sources the frozen sheet used. **Prompts render as 0/45 on that sheet**: the
  `anchor_<id>.txt` prompt files were written for the *frozen* ids only, so the renderer finds none
  for these. The prompts were recovered for the analysis another way (§4).
- **exp-027 / exp-028 (Qwen3.5-9B) are not on this sheet.** `experiments/exp-028/metrics.json` does
  not exist; at render time `runs/exp-027/eval` had landed but `runs/exp-028` held decomposer output
  only, and both log rows still read PENDING. They are left out, as briefed.
- **Nothing was re-run and no model was run.** No GPU, no lock. Read-only over finished artefacts.
- **Six short marked excerpts.** Everything else is counts, categories and item ids.

Arms, in the column order used throughout: **exp-020** Mistral-7B 4-bit as_is · **exp-023** its
reversed twin · **exp-021** Qwen2.5-7B 4-bit as_is · **exp-024** its reversed twin · **exp-022**
Qwen2.5-14B 4-bit as_is · **exp-025** its reversed twin · **exp-008** full_train fine-tune ·
**exp-026** pool-only fine-tune.

### How the comparison was made honest

The first note's scratch scripts are gone, so every detector here was **rebuilt from its definition
in that note's JSON `category_definitions`, then run on BOTH id sets**. The frozen-45 pass is a
calibration, not a copy. It agrees with the published counts exactly on: all 14 categories × 8 arms;
the 13 all-zero ids; every ordering-twin figure at 45 and at 600; every dangling/`chain_validity`
count; `anchor_vs_rest_of_600_by_hop` to 4 dp; every full-600 check; both A7 tallies; the
exp-008/exp-026 agreement and win counts; and the single exemplar-bleed case.

Three detectors differ and the differences are stated wherever they are used:

- **soft step-F1** — my stopword list is not the first note's; on the frozen 45 my soft@0.60 runs
  **+0.009 to +0.067** above the published value. Both draws are reported under *my* implementation,
  so they are comparable to each other but not to the published table row by row.
- **step alignment** — 2–6 fewer unmatched steps per arm than the published block; the positional
  shares agree in shape.
- **verbatim recovery** — my counts reproduce the first note's **§4 table** exactly and differ from
  that note's own JSON block, which runs 1–6 higher per arm. **The first note's markdown and JSON
  disagree with each other on this one table**; the markdown is what it argues from, so that is what
  I replicate against.

---

## 0. The five things this pass found

1. **The frozen note's sampling caveat is real but its *direction* is specific to the frozen draw —
   and the seed-43 draw has a bias of its own, in the opposite place.** The frozen 45 are harder than
   the rest of the 600 at 3 hops (mean step_f1 −0.089, **0 of 8 arms above the rest**) and easier at
   4 hops (+0.052, **8 of 8**). The seed-43 45 are *easier* at 2 hops (+0.067, **7 of 8**) and at
   3 hops (+0.043, **7 of 8**), and representative at 4 hops (−0.003, 4 of 8). The frozen anomaly
   "3-hop scores below 4-hop for every arm" **reverses** on seed-43: 3-hop is above 4-hop for all
   eight arms, which is the ordering the other 555 items show. The generalisable finding is not the
   direction but the **magnitude**: a 15-item-per-hop draw shifts per-hop step_f1 by up to
   **0.158**, with the sign consistent across all eight arms because it is item difficulty, not
   model behaviour (§7).
2. **The metric finding replicates, smaller.** All eight arms score `step_f1 = 0.0` on **8 of 45**
   items here against 13 of 45 frozen. The mechanism is identical: on `2hop__81268_85407` all eight
   arms write essentially the same, plainly correct two-step plan and all eight score zero.
   Token-overlap re-scoring lifts mean step-F1 from **0.224–0.402** to **0.570–0.757** at ≥0.60 and
   **0.687–0.883** at ≥0.50 (§3).
3. **The orphan step and the metric that cannot see it replicate almost exactly.** **83** item×arm
   pairs carry a step that is computed and never used (86 frozen), **76 of the 83** score
   `chain_validity = 1.0`, and **all 45 golds are fully chained**, again with zero dangling steps.
   The arm ordering holds: the two Qwen2.5-14B arms are worst (17 and 19 of 45), the two fine-tunes
   cleanest (1 and 3) (§2).
4. **Ahmmad's A7 answer replicates.** On the 12 items where exp-020 is worst, Qwen-7B is better on
   5, same on 7, worse on **0**; Qwen-14B better on 7, same on 5, worse on 0. Over all 45, Qwen-7B is
   better on 12, same on 18, **worse on 15** — still no net advantage. Qwen-14B is now slightly ahead
   of Mistral overall (16/17/12) where the frozen draw made it a wash. The clean capability
   difference is again **output-format discipline** (§5).
5. **Two things do *not* replicate, and one of them is the first note's own warning firing.**
   exp-008 and exp-026 were near-even on the frozen 45 (step_f1 9–9, ROUGE 21–19); on seed-43
   exp-008 wins every one of the four (15–9, 23–15, 15–5, 23–15). That is exactly the "the 45 have
   no power to resolve these two" caveat the first note wrote, now demonstrated. And the claim that
   the fine-tune advantage *shrinks* under soft matching does not survive: under one implementation
   the gap is flat on the frozen draw (+0.119 exact → +0.118 soft@0.60) and **grows** on seed-43
   (+0.131 → +0.155) (§3, §6).

**Replication tally over 40 checked findings: 29 replicate, 7 partly, 4 not.** Full table in §8.

---

## 1. The taxonomy on the new 45

Same detectors, same definitions. An item×arm pair can carry several.

| category | 020 | 023 | 021 | 024 | 022 | 025 | 008 | 026 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| right number of steps | 24 | 26 | 21 | 19 | 24 | 24 | 33 | **34** |
| too many steps | 17 | 15 | 8 | 14 | 18 | 18 | 2 | 3 |
| too few steps | 4 | 4 | 16 | 12 | 3 | 3 | 10 | 8 |
| runaway (≥10 steps) | 0 | **1** | 0 | 0 | 0 | **1** | 0 | 0 |
| prompt-scaffold line as a step | **2** | **2** | 0 | 0 | 0 | 0 | 0 | 0 |
| template loop (≥3 near-identical steps) | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 |
| duplicate step | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **orphan step (computed, never used)** | 14 | 14 | 7 | 8 | 17 | **19** | **1** | 3 |
| out-of-range / self `#k` | 2 | 2 | 3 | 0 | 1 | 1 | 1 | 2 |
| no `#k` at all though gold chains | 2 | 1 | 4 | 1 | 0 | 0 | 0 | 0 |
| `[#k]` bracket syntax | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| a step restates the whole question | 1 | 1 | 2 | 1 | 4 | 1 | 2 | 5 |
| yes/no verification step | 2 | 1 | 1 | 0 | **4** | **5** | 1 | 1 |
| entity copied from the few-shot block | 0 | 0 | 0 | 0 | 2 | 1 | 0 | 0 |
| entity in neither question nor examples | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| whole plan verbatim = gold | 3 | 4 | 4 | 4 | 4 | 4 | **7** | 3 |
| right length, `step_f1`=0, ROUGE-L ≥0.5 | 2 | 5 | 4 | 4 | 3 | 4 | 2 | 5 |

Per-hop breakdowns are in the JSON under `seed43.counts_per_arm_per_hop`. Categories that again do
not appear at all: `<think>`-style reasoning leakage **0/360**; a declarative answer emitted instead
of a plan **0/360**; a pure permutation of otherwise-correct steps **0/360**.

Two shifts worth naming. **Over-decomposition is more common on this draw for six of the eight
arms** (exp-025 goes 10 → 18 of 45, exp-024 8 → 14) and the fine-tunes are correspondingly *more*
often exactly the right length (exp-026 25 → 34 of 45). And **exp-026's under-decomposition halves**
(16 → 8 of 45), which is the single largest per-arm move between the draws.

### 1.2 Format failures: still Mistral's, with one exception

Scaffold lines counted as steps: **2 of 45** (exp-020) and **2 of 45** (exp-023), **0 of 45** in all
six other arms. Generation-cap hits: **2** and **3** of 45, **0** everywhere else. Both replicate the
frozen pattern.

> **Excerpt 1** — `4hop3__129318_29339_508306_70744`, exp-023, last of 6 predicted steps: `Task:`
> — the parser drops `Question:`-led continuations but not `Task:`, so it is scored as a step.

The full-600 numbers were **recomputed with these detectors and come out identical to the first
note's**: 24 (exp-020) and 31 (exp-023) scaffold rows, 0 in every other arm; 26 and 22 cap hits,
0 elsewhere; dropping scaffold lines moves over-decomposition 192 → 183 and 177 → 169 items. That
half of the picture is a property of the runs, not of either draw.

**What does not replicate: runaway generation is not Mistral-exclusive on this draw.** exp-023's
longest plan here is 12 steps (`4hop1__152146_5274_458768_33637`, where exp-020 writes 5), but
**exp-025 also produces a 10-step plan** for a 4-step gold on `4hop3__129318_29339_508306_70744`.
It is a milder failure than the frozen draw's 79-step Mistral loop, and it is one item — but the
categorical claim "runaway is Mistral only" is not supported by both draws together.

---

## 2. The orphan step, again — and `chain_validity` still cannot see it

| arm | 020 | 023 | 021 | 024 | 022 | 025 | 008 | 026 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| items with ≥1 dangling step | 14 | 14 | 7 | 8 | 17 | **19** | **1** | 3 |
| …of which `chain_validity` = 1.0 | 12 | 12 | 6 | 8 | 17 | 18 | 1 | 2 |

**76 of 83** dangling cases score a perfect 1.0 (frozen: 85 of 86). **All 45 golds emit at least one
`#k` and none leaves a step dangling** — 0/45, identical to the frozen draw. The arm ordering is
preserved: Qwen2.5-14B worst, fine-tunes cleanest, exp-008 down to a single item.

The concrete shape replicates to the level of the individual template. The frozen note's Excerpt 3
was an off-by-one stale referent on `3hop1__30240_740597_41132`. The seed-43 draw contains a
different MuSiQue item built on the same final hop:

> **Excerpt 2** — `3hop1__208108_547811_41132`, gold step 3 `How many times did plague occur in #2 ?`.
> **exp-020, exp-022 and exp-025 all write** `How many times did plague occur in #1 ?` — where `#1`
> is the composer and `#2` is the birthplace the step is asking about. All three have exactly the
> gold step count and all three score `chain_validity` **1.0**.

**The "not a length artefact" claim weakens.** On the frozen draw, 9 of exp-022's 20 dangling items
had exactly the gold step count (45%); here it is 4 of 17 (24%), exp-025 5 of 19 (26%), exp-020 5 of
14 (36%). Dangling steps are still not *only* a length artefact — a quarter to a third of them sit
in correctly-sized plans — but the association with over-decomposition is stronger on this draw.

---

## 3. Wording, not wrong sub-questions — replicates

| arm | evaluator `step_f1` | soft @ ≥0.60 | soft @ ≥0.50 |
|---|---:|---:|---:|
| exp-020 | 0.2357 | 0.5699 | 0.7304 |
| exp-023 | 0.2340 | 0.5902 | 0.7356 |
| exp-021 | 0.2272 | 0.5728 | 0.6867 |
| exp-024 | 0.2237 | 0.5983 | 0.7328 |
| exp-022 | 0.2258 | 0.6070 | 0.7380 |
| exp-025 | 0.2287 | 0.6107 | 0.7600 |
| exp-008 | **0.4019** | **0.7364** | 0.8410 |
| exp-026 | 0.3190 | 0.7573 | **0.8830** |

(Unregistered scratch diagnostic, two thresholds, hand-chosen stopword list — see the calibration
note above. No arm ranking should be taken from it.)

**(a)** The lift replicates: every arm roughly triples its exact step-F1 under a token-overlap match.
**(b)** The prompting-arm ranking is again unstable — exp-020 is *first* among prompting arms on
exact step_f1 here and *fifth* at the 0.50 threshold, the mirror image of the frozen draw, where it
was last on exact and first at 0.50. Two draws, two different reorderings: **the instability is the
finding, and it is now observed twice.** **(c)** The fine-tune advantage survives relaxation, but it
does **not** shrink: under one implementation the mean prompting-to-fine-tune gap is +0.119 exact /
+0.118 at 0.60 on the frozen 45, and +0.131 / +0.155 on seed-43.

**8 of 45** items score zero for all eight arms (13 of 45 frozen). On those 8, the eight arms write
between 6 and 8 distinct normalised plans, so they are mostly not unanimous — except the clearest one:

> **Excerpt 3** — `2hop__81268_85407`, *"What was the 2018 population of the Italian city that's
> underwater?"*. Gold: `what is the city in italy that is under water` / `what was the population of
> the city of #1`. Seven of eight arms write a variant of `Which Italian city is underwater?` /
> `What was the population of #1 in 2018?`. **All eight score `step_f1` 0.0**; mean ROUGE-L across
> the arms is 0.535.

At the other end: **no item is answered exactly by all eight arms** (the frozen draw had one), and
**9 of 45** are answered exactly by at least one arm (frozen: 8).

---

## 4. Hallucinated entities: still essentially absent

**Prompt recovery for this draw.** The `anchor_<id>.txt` files exist only for the frozen ids, so the
seed-43 items were joined to the every-Nth **positional** prompt logs by question text.
exp-020/021/022/008 have all 600 positional prompts (the committed CPU dry-run regenerations), giving
45/45 coverage. exp-023/024/025 were given their as_is twin's few-shot block, which is legitimate
because the exemplar **set** is identical and only the order differs — verified `same=45 diff=0` on
the frozen 45, where both sides have a prompt. exp-008 and exp-026 log `Few-shot source: none (k=0)`
on **every** prompt file (645/645 and 165/165), so their few-shot block is empty by construction.

Across 360 item×arm pairs: **3** carry an entity present in the item's own few-shot block but not in
the question, and **1** carries an entity in neither. Hand-checked, **none of the four is an
invention**:

- `2hop__107238_64918`, exp-022 *United States* and exp-025 *US* — an elaboration ("the president of
  the United States") on a question about a US congressional declaration of war.
- `4hop3__129318_29339_508306_70744`, exp-022 *South Carolina* — a **pre-solved hop**: Fort Hill's
  state, written into a later sub-question instead of being left as a slot. This is the same shape
  the first note found in about 7 of its 9 cases.
- `2hop__81268_85407`, exp-026 *Italy* — a detector false positive: a morphological variant of
  "Italian" in the question.

Read as a negative result, exactly as the first note did: the errors in this block are structural and
lexical, not factual. Two disjoint 45-item draws, 720 item×arm pairs, **zero** hallucinated facts.

---

## 5. Ahmmad's A7 question on a second sample

Same definition: the 12 items with the lowest exp-020 `decomp_mean`; a gap of more than 0.05 counts
as better/worse.

| on Mistral's worst 12 | better | same | worse |
|---|---:|---:|---:|
| Qwen2.5-7B (exp-021) | **5** | 7 | 0 |
| Qwen2.5-14B (exp-022) | **7** | 5 | 0 |

| over all 45 | better | same | worse |
|---|---:|---:|---:|
| Qwen2.5-7B vs Mistral-7B | 12 | 18 | **15** |
| Qwen2.5-14B vs Mistral-7B | **16** | 17 | 12 |
| Qwen2.5-14B vs Qwen2.5-7B | **19** | 16 | 10 |

The frozen conclusion holds: **swapping to Qwen rescues about half of Mistral's worst items and
loses roughly as many elsewhere.** Qwen-7B has no net advantage over the whole 45 on either draw.
Qwen-14B is ahead of Qwen-7B on both draws, and moves from a wash against Mistral (11/23/11) to a
small net advantage (16/17/12) — a difference of four items, well inside what a 45-item draw moves.

What is clean on both draws is **format discipline**: scaffold leakage 2 and 2 of 45 in the Mistral
arms against **0 of 45** in all four Qwen arms, cap hits 2 and 3 against 0, and on the full 600, 24
and 31 scaffold rows and 26 and 22 cap hits against 0 everywhere else. **"Mistral is bad at
decomposition content" is not supported by either draw; "Mistral is bad at output-format discipline"
is supported by both.**

Qwen2.5-14B's yes/no verification step also survives as its signature, with a narrower margin: **4**
(exp-022) and **5** (exp-025) of 45 against 0–2 in every other arm, and **0 in gold**. The full-600
counts, recomputed and identical to the first note, are the decisive ones: **35** and **47** against
≤7 everywhere else.

> **Excerpt 4** — `4hop3__129318_29339_508306_70744`, exp-022, step 7 of 7: `Is the number of
> championships of the University of #1 greater than the University of #4?` — a check, not a
> retrieval, and unanswerable by the retriever.

---

## 6. Ordering, and fine-tune vs prompting

**Ordering.** The 0-of-13 null is again **cancellation, not insensitivity**.

| pair | model | `pred_steps` changed | identical | of the changed: up / down / flat | mean Δ over 45 |
|---|---|---:|---:|---|---:|
| exp-020 → exp-023 | Mistral-7B | **34 / 45** | 11 | 15 / 13 / 6 | **+0.0142** |
| exp-021 → exp-024 | Qwen2.5-7B | **39 / 45** | 6 | 18 / 14 / 7 | **+0.0069** |
| exp-022 → exp-025 | Qwen2.5-14B | **36 / 45** | 9 | 12 / 12 / 12 | **+0.0016** |

Three quarters to nine tenths of plans are rewritten by an irrelevant permutation, the per-item moves
split almost evenly, and the mean is near zero. Note that **the sign of the mean Δ flips between the
two draws** — frozen −0.0132 / −0.0177 / −0.0017, seed-43 +0.0142 / +0.0069 / +0.0016. The aggregate
ordering effect on 45 items is indistinguishable from zero in both magnitude and sign. Largest single
swings here: `2hop__73615_16848` **−0.628** (Qwen-7B), `2hop__481809_184805` −0.569,
`3hop1__662413_42197_18397` **+0.515** (Mistral) — the same ±0.5-and-up range as the frozen draw.

**Fine-tune vs prompting.** exp-008 and exp-026 agree byte-for-byte on **6 of 45** items (7 after
normalisation) — two different models, as on the frozen draw. But their per-item wins are **not**
even here: exp-008 takes step_f1 15–9, ROUGE-L 23–15, ordered-step-accuracy 15–5, SARI 23–15, where
the frozen draw had 9–9, 21–19, 8–6, 23–18. **This is the first note's own caveat demonstrated:** the
45 cannot resolve these two arms, and two disjoint draws give two different answers. The block
report's 600-item paired tests are the ones to quote.

How much gold wording the fine-tunes reproduce (same detector, same normalisation):

| arm | whole plan verbatim | ≥1 gold step verbatim | gold steps reproduced verbatim | ROUGE-L F1 ≥ 0.80 |
|---|---:|---:|---:|---:|
| exp-020 | 3/45 | 20/45 | 28/135 (20.7%) | 6/45 |
| exp-023 | 4/45 | 21/45 | 29/135 (21.5%) | 7/45 |
| exp-021 | 4/45 | 19/45 | 27/135 (20.0%) | 8/45 |
| exp-024 | 4/45 | 19/45 | 26/135 (19.3%) | 4/45 |
| exp-022 | 4/45 | 20/45 | 27/135 (20.0%) | 8/45 |
| exp-025 | 4/45 | 20/45 | 27/135 (20.0%) | 9/45 |
| **exp-008** | **7/45** | **33/45** | **48/135 (35.6%)** | **14/45** |
| **exp-026** | 3/45 | 28/45 | **38/135 (28.1%)** | 9/45 |

The "house style, not memorisation" reading holds: the fine-tunes reproduce gold step strings
**1.4–1.8×** as often as prompting and still paraphrase 64–72% of gold steps, while whole-plan
verbatim stays at 3–7 of 45.

> **Excerpt 5** — `2hop__107238_64918`, *"Who was president when congress declared war on the country
> where Glumpers was filmed?"*. **exp-008 reproduces the gold plan exactly**
> (`Which was the country for Glumpers?` / `who was president when congress declared war on #1`);
> all six prompting arms write three or four steps for the two-step gold — four of them via an
> inserted *"When did congress declare war on #1?"* hop — and all six score `step_f1` 0.0.

Prompting beats a fine-tune rarely and unevenly, as before: on step_f1, exp-020/021/022 beat exp-008
on 6, 8 and 3 of 45 and exp-026 on 3, 5 and 3 of 45. **No item here has all three prompting arms
beating both fine-tunes** (the frozen draw had exactly one).

**Length mechanics replicate.** On under-decomposed 4-hop items the missing gold step is
predominantly **interior** (gold step 2 or 3): exp-026 6/7, exp-008 9/13, exp-021 6/10, exp-024 5/10
— frozen was 10/14, 7/9, 8/13, 6/11. And gold steps that **every** arm misses exist again: 12 gold
steps are missed by ≥6 of the 8 arms (frozen: 10), four of them by all eight, listed in the JSON.

**Per-hop, 15 items each.** Qwen-7B is again the **only** family that under-decomposes at two hops
(exp-021 2/15, exp-024 1/15; every other arm 0/15). Both fine-tunes are again near-perfect on length
at two hops (exp-026 15/15 exact, exp-008 14/15) with the shortfall at four hops (exp-008 10/15 too
few, exp-026 5/15). Qwen-14B again has **zero** under-decomposition at 2 and 3 hops. What does not
replicate: Qwen-14B was the highest over-decomposer at two hops on the frozen draw (5 and 4 of 15);
here it is the lowest of the prompting arms (2 and 3 of 15) and Mistral as_is is highest (5 of 15).

---

## 7. The sampling caveat — the check this pass was drawn for

Per-hop mean `step_f1` of the 15 drawn items minus the other 185 items at that hop, per arm:

| draw | hop | mean Δ (anchor − rest) | range over the 8 arms | arms where the draw is easier |
|---|---|---:|---|---:|
| frozen 45 | 2 | −0.0140 | −0.081 … +0.045 | 3 / 8 |
| frozen 45 | 3 | **−0.0892** | −0.128 … −0.061 | **0 / 8** |
| frozen 45 | 4 | **+0.0521** | +0.010 … +0.089 | **8 / 8** |
| seed-43 45 | 2 | **+0.0673** | −0.018 … +0.158 | **7 / 8** |
| seed-43 45 | 3 | **+0.0426** | −0.007 … +0.091 | **7 / 8** |
| seed-43 45 | 4 | −0.0033 | −0.044 … +0.052 | 4 / 8 |

Pooled over all 45 against the other 555, the frozen draw is *harder* than the remainder for 7 of 8
arms and the seed-43 draw is *easier* for 8 of 8 (e.g. exp-008: 0.300 vs 0.344 frozen; 0.402 vs 0.336
seed-43).

**Three conclusions, stated at the strength the data supports.**

1. **The frozen note's specific caveat is specific to the frozen draw.** Its 3-hop deficit does not
   appear here — seed-43's 3-hop items are *above* the rest for 7 of 8 arms — and its 4-hop surplus
   does not either. The frozen sheet's anomaly that 3-hop step_f1 sits below 4-hop for every arm
   **reverses on this draw for every arm**, back to the ordering the other 555 items show. The first
   note called this "a property of the draw, not of any model"; that is now confirmed against a
   second draw rather than asserted.
2. **But "the frozen 45 are unrepresentative" is the wrong lesson. Every 45 is.** Both draws carry a
   per-hop bias whose sign is consistent across all eight arms — that consistency is the signature of
   item difficulty rather than model behaviour, and it appears in both. The magnitude is the number
   to carry: across 24 arm×hop cells per draw, |Δ| reaches **0.128** (frozen) and **0.158**
   (seed-43), with SDs of 0.064 and 0.047.
3. **This does not weaken the block report's significance results, and saying so matters.** Those are
   **paired** tests over the same 600 items for both arms, so item difficulty cancels. What this
   bounds is any number *read off a 45-item subset* — including this note's and the first note's. The
   operating rule the first note gave stands and is now measured: **read the anchors for mechanism,
   read the 600 for numbers.**

---

## 8. Does it replicate? Finding by finding

| # | finding in the frozen-45 note | value there | value on seed-43 | replicates? |
|---|---|---|---|---|
| 1 | all eight arms score `step_f1` 0 | 13 / 45 | 8 / 45 | partly |
| 2 | a near-unanimous, plausible plan scoring 0 exists | `2hop__152027_141308` | `2hop__81268_85407` | yes |
| 3 | soft matching lifts every arm | 0.155–0.305 → 0.50–0.68 | 0.224–0.402 → 0.570–0.757 (@0.60) | yes |
| 4 | the fine-tune advantage survives relaxation | +0.11 exact | +0.131 exact | yes |
| 5 | …and shrinks under relaxation | "+0.08–0.18 @0.60" | +0.155 @0.60, i.e. it grows | **no** |
| 6 | prompting-arm ranking unstable under relaxation | exp-020 last exact, 1st @0.50 | exp-020 1st exact, 5th @0.50 | yes |
| 7 | one item solved exactly by all eight arms | 1 / 45 | 0 / 45 | partly |
| 8 | items solved exactly by ≥1 arm | 8 / 45 | 9 / 45 | yes |
| 9 | scaffold-line leakage is Mistral-only | 1 and 4 of 45; 0 elsewhere | 2 and 2 of 45; 0 elsewhere | yes |
| 10 | generation-cap hits are Mistral-only | 3 and 3 of 45; 0 elsewhere | 2 and 3 of 45; 0 elsewhere | yes |
| 11 | runaway generation is Mistral-only | 1 of 45 (exp-023, 79 steps) | exp-023 12 steps **and exp-025 10 steps** | partly |
| 12 | the full-600 format counts | 24 / 31 scaffold, 26 / 22 cap | recomputed: identical | yes |
| 13 | predictions dangle, gold never does | 86 pairs; gold 0/45 | 83 pairs; gold 0/45 | yes |
| 14 | `chain_validity` scores dangling plans 1.0 | 85 of 86 | 76 of 83 | yes |
| 15 | Qwen-14B worst, fine-tunes cleanest | 20 vs 5 and 3 | 17 and 19 vs 1 and 3 | yes |
| 16 | dangling is not a length artefact | exp-022 9 of 20 exact-length | exp-022 4 of 17 exact-length | partly |
| 17 | hallucinated entities are near-absent | 9 of 360, ~7 pre-solved hops | 1 of 360 novel, 0 inventions | yes |
| 18 | exemplar bleed is a single case | 1 of 360 | 3 of 360, 2 of them benign elaboration | yes |
| 19 | Qwen rescues about half of Mistral's worst 12 | 5 and 6 better of 12 | 5 and 7 better of 12 | yes |
| 20 | Qwen-7B has no net advantage over all 45 | 9 / 20 / 16 | 12 / 18 / 15 | yes |
| 21 | Qwen-14B is a wash against Mistral over 45 | 11 / 23 / 11 | 16 / 17 / 12 | partly |
| 22 | Qwen-14B clearly ahead of Qwen-7B | 14 / 20 / 11 | 19 / 16 / 10 | yes |
| 23 | the clean Mistral deficit is format, not content | see §2 there | see §5 here | yes |
| 24 | Qwen-14B's yes/no verification step is its signature | 3 and 5 of 45; 35 / 47 of 600 | 4 and 5 of 45; 600 identical | yes |
| 25 | reversing the example order rewrites most plans | 37 / 37 / 41 of 45 | 34 / 39 / 36 of 45 | yes |
| 26 | the rewrites cancel in the mean | Δ −0.013 / −0.018 / −0.002 | Δ +0.014 / +0.007 / +0.002 | yes |
| 27 | per-item ordering swings reach ±0.5 | +0.593 / −0.511 | −0.628 / +0.515 | yes |
| 28 | the two fine-tunes are different models | 4 / 45 byte-identical | 6 / 45 byte-identical | yes |
| 29 | their per-item wins are near even on 45 | 9–9, 21–19, 8–6, 23–18 | 15–9, 23–15, 15–5, 23–15 | **no** |
| 30 | fine-tunes reproduce gold step strings ≈1.6× | 28.1% / 27.4% vs 14.8–18.5% | 35.6% / 28.1% vs 19.3–21.5% | yes |
| 31 | one item where all 3 prompting arms beat both FTs | 1 / 45 | 0 / 45 | partly |
| 32 | prompting beats a fine-tune on a handful of items | 3 / 5 / 3 of 45 vs exp-008 | 6 / 8 / 3 of 45 vs exp-008 | yes |
| 33 | the missing 4-hop step is interior | 10/14, 7/9, 8/13, 6/11 | 6/7, 9/13, 6/10, 5/10 | yes |
| 34 | some gold steps are missed by ≥6 of 8 arms | 10 gold steps | 12 gold steps | yes |
| 35 | Qwen-7B is the only arm short at two hops | 1 and 2 of 15 | 2 and 1 of 15; others 0 | yes |
| 36 | fine-tunes are perfect on length at two hops | 15/15 and 15/15 | 15/15 and 14/15 | yes |
| 37 | the fine-tune shortfall is entirely at four hops | 8 and 11 of 15 | 10 and 5 of 15 | partly |
| 38 | Qwen-14B over-decomposes most at two hops | 5 and 4 of 15 | 2 and 3 of 15 (Mistral highest) | **no** |
| 39 | the anchors are harder at 3 hops, easier at 4 | −0.089 / +0.052, 8 of 8 arms | +0.043 / −0.003 | **no** (draw-specific) |
| 40 | …but a 45-item draw carries a per-hop bias consistent across all arms | implied | measured: up to 0.158, 7–8 of 8 arms | yes |

**Tally: 29 yes · 7 partly · 4 no.** The four that do not: #5, #29, #38, #39. #40 is the
generalised form of #39 and does replicate.

---

## 9. What the two draws say that one did not

- **The taxonomy is stable; the numbers on any 45 are not.** Every *mechanism* the first note named
  appears again in a disjoint sample — orphan steps invisible to `chain_validity`, exact-string
  step-F1 punishing correct plans, Mistral's scaffold leak, Qwen-14B's yes/no step, the interior
  skipped hop, near-total absence of hallucination. Every *arm-vs-arm ordering* read off 45 items
  moved: exp-008 vs exp-026 flipped from even to one-sided, Qwen-14B vs Mistral from wash to small
  advantage, exp-020's position among prompting arms on soft step-F1 from first to fifth. The two
  draws together license a sharp rule for the thesis: **the anchor set is evidence about failure
  modes and is not evidence about which arm is better.**
- **The "cancellation, not insensitivity" reading of the ordering null is now twice-observed, and
  stronger than before** — because the sign of the aggregate flipped between draws while the rewrite
  rate stayed at 76–87%. An effect whose aggregate sign is set by which 45 items you look at is an
  effect that is zero in the mean and large per item.
- **Zero hallucinated facts in 720 item×arm pairs.** The first note called this a negative result on
  360 pairs; it is now a negative result on twice as many, over disjoint items. This is worth a line
  in the thesis, since "LLM decomposers hallucinate entities" is the objection a reader will expect.
- **One thing the first note could not see: the metric problem is not a constant.** 13 of 45 versus
  8 of 45 all-zero items means the share of the report card that is a string-matching artefact is
  itself sample-dependent, somewhere in the high teens to high twenties percent. That makes the case
  for a soft term stronger, not weaker — the current number is unstable *and* known to understate
  correct plans.
- **The sampling caveat should be restated in the general form.** "The frozen 3-hop anchors are hard"
  was a true statement about a specific draw. The durable statement is the one in §7.3.

---

## 10. Should any hypothesis in the first note's §8 be re-ranked?

**One re-rank: H4 moves above H3, to third.** Everything else keeps its position. These remain
hypotheses; **choosing among them is Jahid's call with his supervisor.**

| rank | was | hypothesis | what the second draw did to it |
|---|---|---|---|
| 1 | H1 | soft step-match term reported beside the exact one | **Strengthened.** The lift replicates on disjoint items; the prompting-arm reordering under relaxation replicates *and reverses direction*, which is the sharpest possible demonstration that exact step-F1 ranking is fragile. Still CPU-only over existing per-item files. |
| 2 | H2 | a `plan_connectedness` term | **Replicated, unchanged in size.** 83 pairs vs 86, 76 of them at `chain_validity` 1.0, 0 in gold on both draws, arm spread 1–19. The one qualification: more of the dangling cases sit in over-long plans here (§2), so part of what the term would measure is already visible in `over_decomposition`. Still CPU-only. |
| **3** | **H4** | measure the prompt-permutation / sampling noise floor | **Promoted above H3.** Two new pieces of evidence: the aggregate ordering Δ **changes sign** between draws while 76–87% of plans are rewritten, and a 15-per-hop draw shifts per-hop step_f1 by up to 0.158 with the sign consistent across all eight arms. Note the honest scope: neither number undercuts the block's **paired** 600-item tests, in which item difficulty cancels. What they bound is every claim read off a 45-item sheet. Costs GPU; a direction call. |
| 4 | H3 | add `Task:`/`Answer:` to the parser's stop set | **Unchanged and confirmed small.** The full-600 numbers recomputed identically (24 and 31 rows; over-decomposition 192 → 183 and 177 → 169 items). Demoted only because H4 gained, not because H3 lost. One new caveat: a 10-step Qwen-14B plan on this draw shows runaway is not exclusively a Mistral parser problem. |
| 5 | H5 | supply the gold hop count to the fine-tuned decomposer | **Premise splits between the two fine-tunes.** exp-008's 4-hop shortfall got worse (8 → 10 of 15 too few) and exp-026's got much better (11 → 5 of 15), while the interior-skip pattern held for both. So "the fine-tunes are short at four hops" is solid for exp-008 and shaky for exp-026 on 45 items. Still last: it needs a real inference run and touches the open router defect (issue #47). |

No new hypothesis of comparable rank came out of this draw. The one non-experimental action it
suggests is documentation: whenever an anchor-set number is quoted, quote the 600-item number beside
it.

---

## 11. Where to start reading the sheet — 6 items

Each is a section of `runs/anchor/2026-09-20-validation-45-seed43.md`; search the id.

| # | item id | hop | why |
|---|---|---|---|
| 1 | `2hop__81268_85407` | 2 | Seven of eight arms write the same sensible plan; all eight score `step_f1` 0.0. The metric, not the model. |
| 2 | `3hop1__208108_547811_41132` | 3 | `#1` where `#2` was meant, in three arms, all at `chain_validity` 1.0 — the frozen sheet's Excerpt 3 on a different item. |
| 3 | `2hop__107238_64918` | 2 | exp-008 reproduces gold exactly; all six prompting arms add a third hop and score zero. The fine-tune house style in one screen. |
| 4 | `4hop3__129318_29339_508306_70744` | 4 | Three failure modes on one item: exp-023's `Task:` leak, exp-025's 10-step runaway, exp-022's yes/no verification step. |
| 5 | `4hop1__152146_5274_458768_33637` | 4 | exp-023 writes 12 steps where exp-020 writes 5 — the ordering permutation turning a plan into a chain of speculation. |
| 6 | `2hop__73615_16848` | 2 | The largest ordering swing in the draw: Qwen-7B loses 0.628 of `decomp_mean` to a reversed exemplar order. |

---

## 12. What this note does not show

- **No significance test.** 45 items has no power; every count is descriptive. The significance
  claims live in the block report and the `compare_*` artefacts.
- **The soft step-F1 numbers are a scratch diagnostic**, not a registered metric, with a hand-chosen
  threshold and stopword list, and they are **not** the first note's implementation — see the
  calibration section. Two thresholds are shown for that reason.
- **The category detectors are heuristics**, hand-checked on the cases they fire on and not swept for
  cases they miss, so every count is a lower bound.
- **Step alignment is greedy token-overlap at 0.34** — a pointer to which gold step went missing, not
  a semantic judgement.
- **The seed-43 prompts are recovered by question-text join to positional logs**, and for
  exp-023/024/025 the few-shot block comes from their as_is twin (same exemplar set, verified). That
  is sound for the entity detectors, which depend only on the set of strings in the block, and it
  would not be sound for anything order-sensitive.
- **Nothing here compares to exp-005, exp-015, exp-027 or exp-028**; those arms are not on this sheet.
- **The seed-43 45 are not a new anchor set.** ADR 0033 stands; `configs/anchor_set_45.json` is
  unchanged and re-verified.
