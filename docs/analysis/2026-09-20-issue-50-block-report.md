# The issue #50 block, explained in plain English — exp-020 to exp-026

**Date:** 2026-09-20. **Written for:** Jahid, to read with Ahmmad, and to reuse in the thesis.
**Covers:** seven new runs (exp-020, 021, 022, 023, 024, 025, 026) and the two older runs they
are measured against (exp-008, exp-015).

Every number in this report was copied from a file that already exists. Nothing here was
re-computed, re-run, or estimated. The list of source files is at the very end (§8). Where a
number has never been measured, this report says so instead of guessing.

---

## 0. The five-minute summary

This is the part to read aloud in the meeting.

Ahmmad asked four questions on 2026-08-27. All four now have measured answers on the **same
600 questions**. Here they are in one line each.

| # | His question | The measured answer |
|---|---|---|
| 1 | Does swapping the model **family** (Mistral → Qwen) change the result? | **Yes, but only on plan length and chaining.** 6 of 13 statistics moved. No wording metric moved. |
| 2 | Does doubling the **size** inside Qwen2.5 (7B → 14B) change the result? | **Yes, and in the useful direction.** 7 of 13 moved. Again no wording metric moved. |
| 3 | Does fine-tuning on the **2,000-question pool only** work? | **Yes — it lands right next to the full fine-tune** at about a tenth of the training compute. |
| 4 | Does **reversing the few-shot example order** change anything? | **No.** Across three models, 1 of 39 statistics moved, and it was not a report-card metric. |

The one sentence that ties it together: **changing the prompting model changes *how long* the
plans are and *whether the steps refer to each other*, but it does not change *the words in
the steps*; the thing that changes the words is fine-tuning, and a tiny fine-tune is nearly as
good as a big one.**

Three headline facts with their evidence:

1. **The models fail in opposite directions.** Mistral-7B writes **too many** steps
   (over-decomposition 0.3200, i.e. 192 of 600 questions). Qwen2.5-7B writes **too few**
   (under-decomposition 0.2933, 176 of 600). Qwen2.5-14B fixes the "too few" problem almost
   completely (under 0.1017, 61 of 600) but keeps a "too many" problem (over 0.2817).
   Ahmmad's own rule (ADR 0017) is that writing too many steps is tolerable and writing too few
   is not — so the 14B makes the trade he prefers.
2. **Example order does not matter here.** Ahmmad said "the order is matter. It can change up
   to 10%". On these 600 questions, with these five retrieved examples, the biggest movement
   anywhere was **2.5 percentage points**, and it was not statistically real. This is a
   measurement on our setup, not a claim that he is wrong in general.
3. **Fine-tuning still beats prompting by a wide margin, and cheap fine-tuning is enough.**
   The pool-only fine-tune (2,000 training examples, ~250 training steps, 26 minutes) is
   statistically tied with the full fine-tune (19,938 examples, 2,494 steps, 4 hours 18
   minutes) on 9 of 13 statistics, and it beats prompting on 11 of 13.

Total GPU cost of the whole block: **about 3 hours 24 minutes** on one RTX 3090 (§7).

What is *not* in this block: the third ordering Ahmmad named ("least complex to most complex")
never ran, because "complexity" was never defined; a clean control for the pool fine-tune was
never trained; and each arm ran only once, so we have no measure of run-to-run variation. §6
lists these honestly.

---

## 1. Glossary — the words used in the arm names

| word | what it means |
|---|---|
| **arm** | One experimental setting. "Mistral at 4-bit with examples in the normal order" is one arm. |
| **as_is** | The five retrieved examples are shown to the model **highest-similarity first** — the normal order, the one every run before this block used. |
| **reversed** | The **same five** examples, shown **lowest-similarity first**. Only the order changes; the examples themselves are identical. |
| **unguided** | The model is **not told** how many steps to produce. It decides. Every arm in this block is unguided (ADR 0031 item 3, Jahid's decision). |
| **full_train** | The LoRA adapter trained on the **whole** MuSiQue training split — 19,938 examples, 2,494 optimizer steps. Trained in exp-001, evaluated in exp-008. |
| **pool_2000_imbalanced** | The LoRA adapter trained on **only the 2,000-question pool** — the same 2,000 questions the prompting arms pull their examples from. 250 optimizer steps. Trained and evaluated in exp-026. |
| **4-bit** | The model's weights are squeezed from 16 bits per number down to 4 bits, so the model fits in less GPU memory. It is a compression of the model, not a different model. |
| **fp16** | The uncompressed version — 16 bits per number. Uses more memory, no compression loss. |
| **QLoRA** | A way of fine-tuning where the big model stays frozen and 4-bit compressed, and only a small set of extra weights (the "adapter", here 41.9 million of 7.29 billion, 0.58%) is trained. Cheap to train and cheap to store. |
| **LoRA adapter** | That small trained add-on. You load the base model plus the adapter. |

**Where the hand-inspection sheet lives.** Ahmmad asked for 45 fixed examples (15 per hop) to be
read by hand after every experiment. The sheet is
`runs/anchor/2026-09-20-all-arms.md`. It holds **8 arms** (exp-020, 021, 022, 023, 024, 025,
exp-008 full_train, exp-026 pool-only), with **45 of 45** anchor items found for every arm, each
with the prompt, the raw model output, and the per-item scores side by side. It is **not
committed to git**, because it contains question text and the repo rule is that dataset content
never enters git (ADR 0033 decision 4; CLAUDE.md reproducibility rules). It sits on the compute
box only. **It has not yet been read by hand** — that reading is Jahid's task and it has not
happened, so nothing in this report is based on it.

---

## 2. What was asked, and what was run

### 2.1 The four questions, in Ahmmad's own words

Quoted from `docs/plan/2026-09-14-post-supervisor-meeting-phase.md` §2, which quotes the
2026-08-27 meeting transcript.

**Question 1 and 2 — family and size (item A1).**

> "Let's pick one experiment only… the pool size, 2,000… one time on Mistral, 7 billion, one
> time on Quen, 7 billion, Quen 2.5, 7 billion… and do it on another Quen model, bigger size."

> "if we fix the size of the model, but we try different model from Mistral to QN. Does the
> result will change significantly or not? Within the same model, if we change the size, if the
> model will change or not"

> "12, 14, check what is above 7 billion"

**Question 3 — the pool-only fine-tune (item A2).**

> "And then do a tuned version of Mistral, but trained on the pool only, not the full data, on
> the pool, same pool… So no full training, just 2,000. Just let's see the effect of the pool."

**Question 4 — example ordering (item A5).**

> "Those top five, you can, you are now putting the most highest matching, in your matching
> score to the lowest. Am I right? You may try to reverse it. Also, may try it from the least
> complex to the highest complex… Try all those things we mentioned and let's meet."

> "the order is matter. It can change up to 10%."

He also gave the rule that shapes every run below:

> "You do change, one change, and test. Don't do multiple changes in one experiment, always one
> by one"

That rule is why there are seven runs and not two. Each run differs from exactly one other run
by exactly one thing.

### 2.2 What was run

| exp | model | precision | example order | few-shot examples? | training data | GPU time |
|---|---|---|---|---|---|---|
| exp-015 *(older baseline)* | Mistral-7B-Instruct-v0.3 | fp16 | as_is | yes, 5 | none (prompting) | 18.1 min of generation |
| exp-020 | Mistral-7B-Instruct-v0.3 | 4-bit | as_is | yes, 5 | none (prompting) | 42.3 min total |
| exp-021 | Qwen2.5-7B-Instruct | 4-bit | as_is | yes, 5 | none (prompting) | 13.9 min total |
| exp-022 | Qwen2.5-14B-Instruct | 4-bit | as_is | yes, 5 | none (prompting) | 27.9 min total |
| exp-023 | Mistral-7B-Instruct-v0.3 | 4-bit | **reversed** | yes, 5 | none (prompting) | 38.7 min total |
| exp-024 | Qwen2.5-7B-Instruct | 4-bit | **reversed** | yes, 5 | none (prompting) | 14.2 min total |
| exp-025 | Qwen2.5-14B-Instruct | 4-bit | **reversed** | yes, 5 | none (prompting) | 27.1 min total |
| exp-026 | Mistral-7B-Instruct-v0.3 + new LoRA adapter | 4-bit to train, fp16 to evaluate | n/a | **no** | **pool_2000_imbalanced**, 2,000 examples | 26.1 min train + 14.2 min evaluate |
| exp-008 *(older baseline)* | Mistral-7B-Instruct-v0.3 + full_train adapter | fp16 | n/a | **no** | **full_train**, 19,938 examples | 13.9 min of generation (training was exp-001, 258.3 min) |

Notes on this table, each sourced:

- "5 examples" is `retrieval.k: 5` in `configs/decomposer_musique.json`, and it was confirmed by
  counting a logged prompt (`runs/exp-023/decomposer/20260920_011726/prompts_log/prompt_idx0005_hop2.txt`
  shows five example questions plus the target question). **Small inconsistency worth knowing:**
  the exp-021 and exp-022 log rows say "k=3 exemplars" in their prose. The committed config and
  the logged prompts both say five. The prose is wrong; the runs are fine, because all arms used
  the identical retrieval artifact.
- The two fine-tuned arms use **no** examples in the prompt at all (`--no-few-shot`). That is why
  their prompts are much shorter (122.5 tokens vs 395.2 for prompting).
- All six prompting arms in this block are pinned to 4-bit. The reason (ADR 0031's 2026-09-14
  amendment) is that Qwen2.5-14B's uncompressed weights are 29.54 GB and the GPU has 24 GiB, so
  the 14B **has** to be 4-bit. If the 7B arms had run uncompressed, the size comparison would
  have moved size *and* precision together — the exact double-change Ahmmad forbade.
- exp-026 trains at 4-bit and evaluates at fp16, because that is exactly what exp-001/exp-008 did
  for `full_train` (ADR 0034 decision 3). Copying that keeps the comparison honest.

---

## 3. The evaluation set, and how a comparison is decided

### 3.1 The 600 questions

Every single number in this report is measured on the **same 600 MuSiQue questions**: 200
two-hop, 200 three-hop, 200 four-hop. This set is fixed and written down (ADR 0007, "the pinned
600"), and it was chosen for this block by the lead under Jahid's delegation (ADR 0031 item 2)
because every existing Mistral baseline already sits on it.

"Hop" means how many reasoning steps the gold answer needs. A two-hop question needs two
sub-questions; a four-hop question needs four.

Every run in this block was checked at launch: 600 rows, 200 per hop, 0 missing, 0 unexpected.
Because the set never changes, two runs can be compared **question by question** — the same
question, scored twice.

### 3.2 What a "report card" is

Our evaluator does **not** produce one overall score. It produces a **report card**: seven
numbers reported together, each measuring a different thing, each with its own good direction.
This was a deliberate decision (`docs/METRICS.md` §2.3), taken because the decomposition
research literature does not blend metrics either — Break reports four metrics side by side,
EntailmentBank four, Spider three.

The rule attached to it: **quoting fewer than all seven is not reporting the metric.**

The seven report-card terms:

| # | term | good direction |
|---|---|---|
| 1 | `break_exact_match_rate` | higher |
| 2 | `sari_macro` | higher |
| 3 | `ged_macro` | **lower** |
| 4 | `chain_validity_macro` | higher |
| 5 | `hop_count_exact_match_rate` | higher |
| 6 | `under_decomposition_rate` | **lower** |
| 7 | `over_decomposition_rate` | **lower** |

Six more statistics are reported beside the card, because they are the ones earlier experiments
used and the thesis will want continuity:

| # | term | good direction |
|---|---|---|
| 8 | `exact_match_rate` | higher |
| 9 | `rouge_l_f1_macro` | higher |
| 10 | `step_f1_macro` | higher |
| 11 | `ordered_step_accuracy_macro` | higher |
| 12 | `decomp_mean_macro` | higher |
| 13 | `composite_score` (legacy) | higher |

**Seven plus six is thirteen.** Whenever this report says "6 of 13 were significant", it means
six of those thirteen.

"**Macro**" appears in several names. It means: score each question on its own, then take the
plain average over the 600 questions. Every question counts the same amount. The opposite
("micro") would pool all the counts together first, which lets a model with more output
dominate the number; the report card forbids that (`docs/METRICS.md` §2.3 rule 4).

### 3.3 How significance is decided, in one paragraph

When two arms differ by, say, 2 points, we have to ask: is that a real difference, or would we
get a 2-point wobble just by luck if we had picked a different 600 questions? That is what
**"significant"** means here: **the difference is bigger than the noise you would get from
picking a different 600.** Two tests decide it, both run on the *same* questions scored by both
arms (`docs/METRICS.md` §5). The **bootstrap** re-draws the 600 questions at random with
replacement 10,000 times, re-scores both arms on each fake sample, and looks at the spread of
the difference; if that spread never crosses zero, the difference is significant. **McNemar's
test** is for the yes/no metrics: it counts only the questions where the two arms disagree — "a
got it right, b got it wrong" versus the reverse — and asks whether that split is too lopsided
to be a coin flip. Eight statistics get the bootstrap and five get McNemar, which is where 13
comes from. A paired t-test is also reported for twelve of them, but it is not the headline
(ADR 0009), because these 0/1-heavy scores are not normally distributed. Nothing is corrected
for testing thirteen things at once, so a single barely-significant result out of thirteen
should be read as weak. And "not significant" is **not** the same as "equal" — it means we could
not tell them apart at this sample size; this report says **"tied within noise"** for that case.

---

## 4. The metrics, one at a time

Each one below: what it measures, which direction is good, and one concrete example of what
moves it. All definitions are from `docs/METRICS.md` and
`docs/analysis/2026-08-23-decomposition-quality-metric-specification.md`.

**A note that applies to all of them: no model judges anything.** Every metric here is
string-level — it compares text to text. Two decompositions that mean the same thing but use
different words score as a mismatch. This is on purpose: Ahmmad ruled that no closed commercial
model may judge decomposition quality.

### 4.1 `break_exact_match_rate` — "is the whole plan character-perfect?"

**What it measures.** All the predicted steps are glued into one string, all the gold steps are
glued into one string, and the two are compared **exactly** (only lower-cased). 1 if identical,
0 otherwise. The reported number is the fraction of the 600 that scored 1.

**Good direction:** higher. **Range:** 0 to 1.

**What moves it.** One extra comma anywhere in the plan turns a 1 into a 0. This is the
strictest metric in the set. A model that gets the meaning right but writes "Who created the Lap
Engine?" where the gold says "Who developed Lap Engine?" scores 0.

**Why it exists.** It comes from the official Break leaderboard evaluator, so it is a metric the
field recognises, not one we invented.

### 4.2 `sari_macro` — "did it edit the question into steps the right way?"

**What it measures.** SARI compares three things: the original question (the source), the model's
decomposition, and the gold decomposition. It rewards words that were correctly **kept** from the
question, words correctly **added**, and words correctly **deleted**, averaged over 1-, 2-, 3-
and 4-word sequences.

**Good direction:** higher. **Range:** 0 to 1.

**What moves it.** Copying the question verbatim as a single step scores badly on "add". Writing
steps that reuse the question's key entities and add the right relation words scores well.

**Sharp edge, important for the thesis.** SARI's floor on this data is **not** zero. Every
decomposition shares the same separator and boilerplate, so an **empty** prediction measured
0.29 on this same 600. So **an absolute SARI number is not interpretable here; only differences
between arms on the same data are.** Our arms sit between 0.5720 and 0.6823, and the gap between
the empty baseline and a real arm is smaller than it looks.

### 4.3 `ged_macro` — "how far is the plan's shape from the gold's shape?"

**What it measures.** Graph edit distance. Each decomposition is turned into a graph: one node
per step (labelled with its text), one edge wherever a step refers back to another step (`#2`).
GED counts how much editing it takes to turn the predicted graph into the gold graph —
substituting node labels, inserting and deleting nodes and edges — then normalises by graph
size. It is the one metric that combines wording, chaining and step count in a single number.

**Good direction: LOWER.** This is the only report-card term where a bigger number is worse.
**Range:** 0 upward; it can exceed 1.

**What moves it.** A plan with four steps where the gold has two pays for two node insertions.
A plan whose step text shares no words with the gold pays a high substitution cost on every node.
A `#7` reference in a three-step plan creates a dangling node and pays for it.

**Three caveats that travel with it.** (a) Our implementation does not use the spaCy lemmatizer
the official Break code uses, so **our absolute GED values are not comparable to published Break
GED numbers** — within this evaluation everything is scored identically, so comparisons here are
valid. (b) GED is order-light, and on a two-step plan it is order-**blind**: a reversed two-step
plan scores GED 0.0. That is why exact-match and the length metrics must stay beside it. (c) For
very large graphs the optimizer is skipped and a bound is used instead; the count of such items
is reported as `ged_fallback_counts`. In this block that count is `{}` (empty, meaning no
fallbacks, so nothing is machine-dependent) for every arm **except** exp-020 (`{node_cap: 2}`)
and exp-023 (`{node_cap: 3}`) — the two Mistral 4-bit arms, whose runaway long plans hit the cap.
A fallback item is slightly penalised, so those two arms carry a very small bias toward a worse
GED.

### 4.4 `chain_validity_macro` — "do the steps actually refer back correctly?"

**What it measures.** MuSiQue's gold steps chain: step 2 says "#1 >> educated at", meaning "the
answer to step 1". This metric checks the predicted references. Per question: if the gold needs
no reference, score 1.0. If the gold chains but the prediction emits **no reference at all**,
score **0.0** — silence gets no free credit. Otherwise, score = valid references / total
references, where valid means `#k` points to an **earlier** step.

**Good direction:** higher. **Range:** 0 to 1.

**What moves it.** A three-step plan whose steps are three independent questions with no `#1` or
`#2` anywhere scores 0.0. A plan where step 3 says `#4` (pointing forward) scores partial credit.

**Why it exists, and why it is "house-built".** The older metric `reference_validity` gave 1.0
to a plan that emitted no references at all — a model that never chains looked perfect. That
free credit was measured to **flip a model ranking** once (76 of 600 items in one arm paid 1.0
for not chaining). `chain_validity` is the repair. It is **not** a published metric and the
report card labels it as ours.

**Bookkeeping fact:** `chain_validity_gold_unchained_items` is **0** on all 600 — no item gets
the free-1.0 branch. Checked for every arm in this block.

### 4.5 `hop_count_exact_match_rate` — "the right number of steps?"

**What it measures.** Did the model produce exactly as many steps as the gold has hops? 1 or 0
per question, averaged.

**Good direction:** higher. **Range:** 0 to 1.

**What moves it.** A four-hop question answered with a three-step plan scores 0, even if all
three steps are perfect. A four-hop question answered with four steps scores 1, even if all four
steps are nonsense. It measures **granularity only**, nothing about the words.

### 4.6 and 4.7 `under_decomposition_rate` and `over_decomposition_rate` — the two ways to get the length wrong

**What they measure.** `under` = fraction of questions where the model wrote **fewer** steps than
the gold. `over` = fraction where it wrote **more**.

**Good direction: LOWER, for both.** **Range:** 0 to 1.

**Why they are two numbers and never one.** Ahmmad's own judgment, recorded in ADR 0017 from the
2026-08-12 meeting: *"a three-hop question, it's fine to make it a four-hop, but it's not fine to
make it a two-hop."* **Over-decomposition is tolerable; under-decomposition is not.** Adding them
together would erase exactly that asymmetry, so the report card forbids summing them. The
asymmetry is in how heavily **you** weigh them, not in their sign — both are lower-is-better.

**What moves them.** A two-hop question answered with three steps adds to `over`. The same
question answered in one step adds to `under`. A model that answers everything in two steps will
have low `over` and high `under` on the four-hop third of the set.

**Two companion numbers appear in the tables below, and they are not report-card terms.**
`mean_signed_step_count_error` is the average of (predicted steps − gold steps). Its **sign**
tells you which way a model leans: positive = writes too many on average, negative = writes too
few. Over- and under-errors **cancel** inside it, which is exactly why the two rates are reported
beside it. `step_count_mae` is the same quantity with the sign removed — the average size of the
length error, in steps.

### 4.8 `exact_match_rate` — "is every step right, in the right place?"

**What it measures.** 1 if the predicted step list and gold step list have the same length and
matching normalised text at every position, else 0.

**Good direction:** higher.

**What moves it.** It is slightly **more forgiving** than `break_exact_match` (§4.1), because it
strips punctuation before comparing. A plan differing from gold only by a question mark scores
1.0 here and 0.0 there. Both are correct; they measure different strictness levels.

### 4.9 `rouge_l_f1_macro` — "how much of the gold wording did it recover, in order?"

**What it measures.** All steps are flattened into one text. ROUGE-L finds the **longest common
subsequence** of words between prediction and gold — words that appear in both, in the same
relative order, not necessarily adjacent. Precision = LCS / predicted words, recall = LCS / gold
words, F1 = their harmonic mean.

**Good direction:** higher. **Range:** 0 to 1.

**What moves it.** Reusing the gold's phrasing raises it. Writing a correct plan in completely
different words lowers it. Writing a very long plan lowers **precision** (the denominator grows)
even when recall is fine.

**Why it is here:** it is the metric Ahmmad asked about directly. Note it is our own
implementation with our own tokenizer, so it is **not** numerically comparable to a
`rouge-score` package number without checking tokenizers.

### 4.10 `step_f1_macro` — "which steps overlap, ignoring order?"

**What it measures.** Treat the predicted steps as a **set** and the gold steps as a set.
Precision = how many predicted steps appear in gold, divided by how many were predicted. Recall =
same intersection divided by the number of gold steps. F1 is their harmonic mean, computed **per
question** and then averaged. Duplicate steps collapse in a set.

**Good direction:** higher. **Range:** 0 to 1.

**What moves it.** Getting one of two gold steps exactly right, in a two-step plan, gives
precision 0.5, recall 0.5, F1 0.5. Order does not matter. Matching is on normalised text, so
close-but-not-identical wording counts as a miss.

**Note on normalisation:** `[#2]` and `#2` compare **equal** (the `#` survives, the brackets do
not), but punctuation is deleted rather than replaced by a space, so `well-known` becomes
`wellknown` and does **not** match `well known`.

### 4.11 `ordered_step_accuracy_macro` — "step F1, but position matters"

**What it measures.** Count the positions where predicted step *i* equals gold step *i*, then
divide by `max(len(pred), len(gold))`.

**Good direction:** higher. **Range:** 0 to 1.

**What moves it.** A plan with the right two steps in the wrong order scores 0. A plan with the
right steps but an extra step appended is penalised through the denominator. It is always ≤
`step_f1` for the same item in practice, and the gap between the two tells you how much of the
overlap is in the wrong place.

### 4.12 `decomp_mean_macro` — one blended number, reported beside the card and never as it

**What it measures.** Per question, the unweighted average of five values in 0–1:
`break_exact_match`, `sari`, `1 − min(clamp, ged)`, `chain_validity`, `hop_count_exact_match`.

**Good direction:** higher.

**Why it is not the headline.** The GED clamp is "a judgment call with no source" — GED is
unbounded above, so a blend has to cut it off somewhere, and nothing justifies where. Equal
weights are equally unjustified: there is no unit conversion between "verbatim identity" and
"graph distance". It exists only for the cases where a single number is structurally required.
**Adopting any single number as the thesis's primary metric is Jahid's decision with Ahmmad, not
a settled one** (issue #6 item 5).

### 4.13 `composite_score` — LEGACY, frozen, do not lead with it

**What it measures.** `0.4·step_f1 + 0.3·ordered_step_accuracy + 0.2·reference_validity_micro +
0.1·(1 − step_count_MAE/3)`, computed from the **aggregate** values.

**Good direction:** higher.

**Why it is marked legacy and frozen.** Its 0.2-weight reference term looks for references
written as `[#k]` with brackets. MuSiQue's gold writes bare `#k` — **1,200 bare references and 0
bracketed across the pinned 600** — so that term has never measured what its name says on this
data. It is kept unchanged so that nine older committed arms and a 33-cell sweep stay quotable,
**not** because it is correct. The weights were also flagged by Ahmmad himself as handmade and
possibly biased.

**This matters for reading the tables below.** The composite has a step-count-error term that is
direction-blind and scale-sensitive, so it swings hard whenever a model's plan lengths change —
which is exactly what changes most in this block. When you see the composite move by 0.12 or
0.22 while every real quality metric is tied, **that is the composite reacting to plan length,
not a quality finding.** It is an argument for retiring it (issue #40), and it should not be
quoted to Ahmmad as a result.

### 4.14 One master table of all nine arms

All numbers on the same 600 questions. Arrows show the good direction.

**The seven report-card terms:**

| arm | break_EM ⇑ | SARI ⇑ | GED ⇓ | chain ⇑ | hop_EM ⇑ | under ⇓ | over ⇓ |
|---|---|---|---|---|---|---|---|
| exp-015 Mistral-7B fp16, as_is | 0.0350 | 0.5785 | 0.4714 | 0.9694 | 0.5400 | 0.2317 | 0.2283 |
| exp-020 Mistral-7B 4-bit, as_is | 0.0400 | 0.5766 | 0.4846 | 0.9626 | 0.5117 | 0.1683 | 0.3200 |
| exp-021 Qwen2.5-7B 4-bit, as_is | 0.0550 | 0.5720 | 0.4694 | 0.9210 | 0.5533 | 0.2933 | 0.1533 |
| exp-022 Qwen2.5-14B 4-bit, as_is | 0.0333 | 0.5795 | 0.4605 | 0.9838 | 0.6167 | 0.1017 | 0.2817 |
| exp-023 Mistral-7B 4-bit, reversed | 0.0483 | 0.5831 | 0.4813 | 0.9629 | 0.5267 | 0.1783 | 0.2950 |
| exp-024 Qwen2.5-7B 4-bit, reversed | 0.0583 | 0.5771 | 0.4728 | 0.9282 | 0.5400 | 0.2900 | 0.1700 |
| exp-025 Qwen2.5-14B 4-bit, reversed | 0.0333 | 0.5832 | 0.4670 | 0.9767 | 0.5967 | 0.1150 | 0.2883 |
| **exp-008 full_train fine-tune** | **0.1017** | **0.6823** | **0.3094** | **0.9942** | **0.7100** | 0.2567 | **0.0333** |
| **exp-026 pool-only fine-tune** | 0.0967 | 0.6804 | 0.3191 | 0.9750 | 0.7033 | 0.2583 | 0.0383 |

**The six statistics reported beside the card:**

| arm | exact_match ⇑ | ROUGE-L F1 ⇑ | step_f1 ⇑ | ordered_acc ⇑ | decomp_mean ⇑ | composite (legacy) ⇑ |
|---|---|---|---|---|---|---|
| exp-015 | 0.0450 | 0.5447 | 0.1894 | 0.1662 | 0.5304 | 0.3132 |
| exp-020 | 0.0450 | 0.5385 | 0.1823 | 0.1632 | 0.5213 | 0.1898 |
| exp-021 | 0.0633 | 0.5412 | 0.1929 | 0.1771 | 0.5264 | 0.4111 |
| exp-022 | 0.0450 | 0.5453 | 0.1969 | 0.1778 | 0.5507 | 0.4150 |
| exp-023 | 0.0550 | 0.5413 | 0.1958 | 0.1708 | 0.5280 | 0.1920 |
| exp-024 | 0.0700 | 0.5431 | 0.2081 | 0.1905 | 0.5262 | 0.4204 |
| exp-025 | 0.0417 | 0.5473 | 0.1973 | 0.1744 | 0.5446 | 0.4137 |
| **exp-008** | **0.1083** | **0.6647** | **0.3411** | **0.3237** | **0.6357** | **0.5236** |
| **exp-026** | **0.1083** | 0.6607 | 0.3185 | 0.2969 | 0.6272 | 0.5059 |

**The length picture in two numbers:**

| arm | mean signed step-count error (sign = which way it leans) | step_count_MAE (size of the error) |
|---|---|---|
| exp-015 | +0.145 | 0.6717 |
| exp-020 | +0.585 | 0.9617 |
| exp-021 | **−0.150** | 0.5767 |
| exp-022 | +0.285 | **0.5117** |
| exp-023 | +0.7133 | 1.1267 |
| exp-024 | **−0.1267** | 0.6000 |
| exp-025 | +0.280 | 0.5267 |
| exp-008 | **−0.2333** | **0.300** |
| exp-026 | **−0.235** | 0.3183 |

Read the sign column: **every prompting arm except the two Qwen-7B arms writes too many steps on
average; both fine-tuned arms write too few.** That single fact explains most of what follows.

---

## 5. Results, one question at a time

### 5.1 Does 4-bit compression cost Mistral anything? (exp-020 vs exp-015)

**Same 600 questions. Two arms: Mistral-7B at 4-bit (exp-020) vs the identical Mistral-7B at
fp16 (exp-015).** Everything else identical — same examples, same prompt, same seed, same
decoding. This comparison came free: exp-020 had to run anyway so that all three model arms sat
on the same 4-bit axis, and exp-015 already existed.

**Result: 3 of 13 statistics significant. All five report-card *quality* terms are tied within
noise.**

| statistic | exp-020 (4-bit) | exp-015 (fp16) | difference | verdict |
|---|---|---|---|---|
| break_EM ⇑ | 0.0400 | 0.0350 | +0.0050 | tied within noise (McNemar p=0.375) |
| SARI ⇑ | 0.5766 | 0.5785 | −0.0019 | tied within noise |
| GED ⇓ | 0.4846 | 0.4714 | +0.0132 | tied within noise (CI touches zero) |
| chain_validity ⇑ | 0.9626 | 0.9694 | −0.0067 | tied within noise |
| hop_EM ⇑ | 0.5117 | 0.5400 | −0.0283 | tied within noise (p=0.149) |
| **under ⇓** | **0.1683** | **0.2317** | **−0.0633** | **significant, favours 4-bit** (p=1.21e-06) |
| **over ⇓** | **0.3200** | **0.2283** | **+0.0917** | **significant, favours fp16** (p=1.96e-09) |
| exact_match ⇑ | 0.0450 | 0.0450 | 0.0000 | tied (p=1.0) |
| ROUGE-L ⇑ | 0.5385 | 0.5447 | −0.0062 | tied within noise |
| step_f1 ⇑ | 0.1823 | 0.1894 | −0.0070 | tied within noise |
| ordered_acc ⇑ | 0.1632 | 0.1662 | −0.0030 | tied within noise |
| decomp_mean ⇑ | 0.5213 | 0.5304 | −0.0091 | tied within noise |
| **composite (legacy)** | 0.1898 | 0.3132 | −0.1234 | significant, favours fp16 |

**In plain words.** Compressing Mistral to 4-bit does not make its decompositions worse in
quality — every wording, graph and step-overlap metric is tied within noise. What it does is
change the **shape** of its length errors: the 4-bit version writes too many steps more often
(+9.2 points) and too few steps less often (−6.3 points). Under Ahmmad's asymmetry rule, those
two significant moves point in **opposite normative directions**, so this comparison does not
declare a winner. The tail is visible directly: exp-020 emitted single plans of 13, 14, 15, 19
and **126** steps; exp-015's longest was 12. The big legacy-composite drop is that step-count
term reacting to the 126-step plan and its friends — it is not a quality finding.

**Cost side:** 4-bit was **2.3× slower** to generate (2,469.5 s vs 1,086.2 s for the same 600),
in exchange for a 4,818 MiB footprint. exp-015's memory was never measured, so no memory saving
is claimed.

### 5.2 Does the model **family** matter, at the same size? (exp-021 vs exp-020)

**Same 600 questions. Two arms: Qwen2.5-7B-Instruct (exp-021) vs Mistral-7B-Instruct-v0.3
(exp-020).** Both 4-bit, both unguided, both with the same five examples in the same order, same
seed, same decoding. **The only thing that moves is the model family.** This is Ahmmad's question
1, asked with one factor moving.

**Result: 6 of 13 significant, in both directions. No wording metric moved.**

| statistic | exp-021 Qwen-7B | exp-020 Mistral-7B | difference | verdict |
|---|---|---|---|---|
| break_EM ⇑ | 0.0550 | 0.0400 | +0.0150 | **significant, favours Qwen** (p=0.0352) |
| SARI ⇑ | 0.5720 | 0.5766 | −0.0046 | tied within noise |
| GED ⇓ | 0.4694 | 0.4846 | −0.0152 | tied within noise |
| chain_validity ⇑ | 0.9210 | 0.9626 | −0.0416 | **significant, favours Mistral** |
| hop_EM ⇑ | 0.5533 | 0.5117 | +0.0417 | tied within noise (p=0.108) |
| under ⇓ | 0.2933 | 0.1683 | +0.1250 | **significant, favours Mistral** (p=2.85e-09) |
| over ⇓ | 0.1533 | 0.3200 | −0.1667 | **significant, favours Qwen** (p=1.25e-15) |
| exact_match ⇑ | 0.0633 | 0.0450 | +0.0183 | **significant, favours Qwen** (p=0.0192) |
| ROUGE-L ⇑ | 0.5412 | 0.5385 | +0.0027 | tied within noise |
| step_f1 ⇑ | 0.1929 | 0.1823 | +0.0106 | tied within noise |
| ordered_acc ⇑ | 0.1771 | 0.1632 | +0.0139 | tied within noise |
| decomp_mean ⇑ | 0.5264 | 0.5213 | +0.0051 | tied within noise |
| composite (legacy) | 0.4111 | 0.1898 | +0.2212 | significant, favours Qwen |

**In plain words.** The answer to Ahmmad's question is **yes, the family matters — but not in
the way "better model" would suggest.** Every metric about *wording* (SARI, GED, ROUGE-L,
step_f1, ordered accuracy) is tied within noise. Net granularity accuracy (hop_EM) is also tied.
What moves is a **sign flip in the length error**: Mistral writes too many steps (mean signed
+0.585), Qwen writes too few (−0.150), and **each model is significantly better than the other
on the error it does not make.** Qwen is also cleaner in one practical way — no runaway plans
(longest plan 8 steps), 0 rows cut off at the token limit versus Mistral's 26, and no GED
fallbacks. Where Qwen is worse is **chaining**, and it is concentrated at two hops.

**Per-hop detail** (Ahmmad asked for per-hop reporting, item A6):

| gold hops | Qwen-7B under / over | Mistral-7B under / over | Qwen chain | Mistral chain |
|---|---|---|---|---|
| 2 | 0.105 / 0.085 | 0.000 / 0.230 | 0.8775 | 0.9869 |
| 3 | 0.320 / 0.155 | 0.220 / 0.315 | 0.9317 | 0.9582 |
| 4 | 0.455 / 0.220 | 0.285 / 0.415 | 0.9538 | 0.9428 |

The flip is monotone across hop depth. Qwen's chaining problem is a **two-hop** problem: at four
hops it is actually slightly ahead. A plausible mechanism is that shorter plans contain fewer
`#k` references to get right — **this is untested.**

**Cost side:** Qwen2.5-7B produced the same 600 decompositions **3.0× faster** (812.9 s vs
2,469.5 s) for 1.26× the memory (6,082 vs 4,818 MiB).

### 5.3 Does **size** matter, inside the same family? (exp-022 vs exp-021)

**Same 600 questions. Two arms: Qwen2.5-14B-Instruct (exp-022) vs Qwen2.5-7B-Instruct
(exp-021).** Both 4-bit, both unguided, same examples, same order, same seed. **The only thing
that moves is parameter count: 7.62 billion → 14.77 billion.** This is Ahmmad's question 2.

**Result: 7 of 13 significant, in both directions. Again no wording metric moved.**

| statistic | exp-022 Qwen-14B | exp-021 Qwen-7B | difference | verdict |
|---|---|---|---|---|
| break_EM ⇑ | 0.0333 | 0.0550 | −0.0217 | **significant, favours 7B** (p=0.0072) |
| SARI ⇑ | 0.5795 | 0.5720 | +0.0075 | tied within noise |
| GED ⇓ | 0.4605 | 0.4694 | −0.0089 | tied within noise |
| chain_validity ⇑ | 0.9838 | 0.9210 | +0.0628 | **significant, favours 14B** |
| hop_EM ⇑ | 0.6167 | 0.5533 | +0.0633 | **significant, favours 14B** (p=0.0120) |
| under ⇓ | 0.1017 | 0.2933 | −0.1917 | **significant, favours 14B** (p=7.32e-23) |
| over ⇓ | 0.2817 | 0.1533 | +0.1283 | **significant, favours 7B** (p=1.87e-10) |
| exact_match ⇑ | 0.0450 | 0.0633 | −0.0183 | **significant, favours 7B** (p=0.0433) |
| ROUGE-L ⇑ | 0.5453 | 0.5412 | +0.0041 | tied within noise |
| step_f1 ⇑ | 0.1969 | 0.1929 | +0.0039 | tied within noise |
| ordered_acc ⇑ | 0.1778 | 0.1771 | +0.0006 | tied within noise |
| decomp_mean ⇑ | 0.5507 | 0.5264 | +0.0242 | **significant, favours 14B** |
| composite (legacy) | 0.4150 | 0.4111 | +0.0039 | tied within noise |

**In plain words.** **What size buys is granularity and chaining, not wording.** Doubling the
parameters nearly eliminates the "too few steps" failure — under-decomposition falls from 0.2933
to 0.1017, which at p=7.32e-23 is **the largest single effect anywhere in this block**. It makes
the chaining near-perfect and, importantly, *uniform* across hop depth (0.9800 / 0.9838 / 0.9875
at 2 / 3 / 4 hops, versus the 7B's collapse to 0.8775 at two hops). It gets the step count right
more often (hop_EM 0.6167, the best of any prompting arm) and has the smallest length error of
the block (step_count_MAE 0.5117).

What it gives back is over-decomposition (0.2817 vs 0.1533) and the two verbatim-identity
metrics (break_EM, exact_match) — a longer and more elaborated plan is simply less likely to be
character-identical to a short gold plan. Under Ahmmad's own asymmetry rule (over is tolerable,
under is not), **the 14B makes the trade he prefers.** That reading is his and Jahid's to
confirm; this report does not make the call.

**Per-hop detail:**

| gold hops | 14B under / over | 7B under / over | 14B chain | 14B hop_EM |
|---|---|---|---|---|
| 2 | 0.010 / 0.200 | 0.105 / 0.085 | 0.9800 | 0.790 |
| 3 | 0.075 / 0.345 | 0.320 / 0.155 | 0.9838 | 0.580 |
| 4 | 0.220 / 0.300 | 0.455 / 0.220 | 0.9875 | 0.480 |

**The confounded pair, reported as confounded: exp-022 vs exp-020 (14B Qwen vs 7B Mistral).**
This moves **both** family and size, so it is **not** one of Ahmmad's contrasts and cannot answer
either of them. It is here because a reader will ask. Result: **6 of 13 significant, all six
favouring the 14B** — hop_EM +0.1050 (p=1.72e-05), under −0.0667 (p=9.31e-05), GED −0.0241,
chain_validity +0.0211, decomp_mean +0.0293, legacy composite +0.2252. Tied within noise: ROUGE-L
+0.0068, step_f1 +0.0145, ordered_acc +0.0146, SARI +0.0029, exact_match 0.0000 (p=1.0), break_EM
−0.0067 (p=0.503), over −0.0383 (p=0.094). Read it as "the biggest Qwen beats the smallest
Mistral on structure", and nothing more precise than that.

**Cost side:** the 14B took 2.03× the generation time of the 7B (1,648.0 s vs 812.9 s) and 1.71×
the memory (10,416 vs 6,082 MiB). It is still **1.50× faster** than 4-bit Mistral-7B.

**One flag on validity.** The 14B has 14,770,033,664 parameters, which is **98.5%** of the
recorded ceiling of 1.5e10. That ceiling was raised by Jahid (ADR 0031 item 4, confirmed in ADR
0032 item 2) and is marked **pending supervisor confirmation**. If Ahmmad reasserts 8B or 10B,
exp-022 and exp-025 are withdrawn. exp-026 is not affected (7.29B).

### 5.4 Does the **order of the examples** matter? (exp-023 / 024 / 025 vs their twins)

**Same 600 questions each time. Three pairs, each pair identical in every way except that the
five retrieved examples are shown lowest-similarity-first instead of highest-similarity-first.**
This is Ahmmad's question 4, and he put a number on it: "up to 10%".

That the examples really were the same five was **verified on the artifacts, not argued from the
code**: for each pair, 120 shared logged prompts were checked, and all 120 had an **identical
example set** and were **exactly reversed**. As a control, two same-order arms (exp-021 vs
exp-022) gave 120/120 identical sets and **0/120** reversed, which is the right answer for two
arms that are not reversals of each other.

**Result across the three pairs: 1 of 39 statistics significant, and 0 of 21 report-card terms.**

| pair | model | significant / 13 | report-card terms significant / 7 | biggest movement anywhere |
|---|---|---|---|---|
| exp-023 vs exp-020 | Mistral-7B | **0** | 0 | over-decomposition −0.0250 (p=0.228) |
| exp-024 vs exp-021 | Qwen2.5-7B | **1** (step_f1, not a report-card term) | 0 | over-decomposition +0.0167 (p=0.326) |
| exp-025 vs exp-022 | Qwen2.5-14B | **0** | 0 | hop_EM −0.0200 (p=0.349) |

**Pair 1 — Mistral-7B, exp-023 reversed vs exp-020 as_is:**

| statistic | reversed | as_is | difference | verdict |
|---|---|---|---|---|
| break_EM ⇑ | 0.0483 | 0.0400 | +0.0083 | tied within noise |
| SARI ⇑ | 0.5831 | 0.5766 | +0.0065 | tied within noise |
| GED ⇓ | 0.4813 | 0.4846 | −0.0034 | tied within noise |
| chain_validity ⇑ | 0.9629 | 0.9626 | +0.0003 | tied within noise |
| hop_EM ⇑ | 0.5267 | 0.5117 | +0.0150 | tied within noise |
| under ⇓ | 0.1783 | 0.1683 | +0.0100 | tied within noise |
| over ⇓ | 0.2950 | 0.3200 | −0.0250 | tied within noise |
| exact_match ⇑ | 0.0550 | 0.0450 | +0.0100 | tied within noise |
| ROUGE-L / step_f1 / ordered_acc / decomp_mean / composite | 0.5413 / 0.1958 / 0.1708 / 0.5280 / 0.1920 | 0.5385 / 0.1823 / 0.1632 / 0.5213 / 0.1898 | +0.0028 / +0.0134 / +0.0076 / +0.0067 / +0.0022 | all tied within noise |

**Pair 2 — Qwen2.5-7B, exp-024 reversed vs exp-021 as_is:**

| statistic | reversed | as_is | difference | verdict |
|---|---|---|---|---|
| break_EM ⇑ | 0.0583 | 0.0550 | +0.0033 | tied within noise |
| SARI ⇑ | 0.5771 | 0.5720 | +0.0051 | tied within noise |
| GED ⇓ | 0.4728 | 0.4694 | +0.0034 | tied within noise |
| chain_validity ⇑ | 0.9282 | 0.9210 | +0.0072 | tied within noise |
| hop_EM ⇑ | 0.5400 | 0.5533 | −0.0133 | tied within noise |
| under ⇓ | 0.2900 | 0.2933 | −0.0033 | tied within noise |
| over ⇓ | 0.1700 | 0.1533 | +0.0167 | tied within noise |
| **step_f1 ⇑** | **0.2081** | **0.1929** | **+0.0152** | **significant, favours reversed** (CI +0.0013 to +0.0292) |
| exact_match / ROUGE-L / ordered_acc / decomp_mean / composite | 0.0700 / 0.5431 / 0.1905 / 0.5262 / 0.4204 | 0.0633 / 0.5412 / 0.1771 / 0.5264 / 0.4111 | +0.0067 / +0.0019 / +0.0133 / −0.0003 / +0.0093 | all tied within noise |

**Pair 3 — Qwen2.5-14B, exp-025 reversed vs exp-022 as_is:**

| statistic | reversed | as_is | difference | verdict |
|---|---|---|---|---|
| break_EM ⇑ | 0.0333 | 0.0333 | 0.0000 | tied (p=1.0) |
| SARI ⇑ | 0.5832 | 0.5795 | +0.0038 | tied within noise |
| GED ⇓ | 0.4670 | 0.4605 | +0.0065 | tied within noise |
| chain_validity ⇑ | 0.9767 | 0.9838 | −0.0071 | tied within noise |
| hop_EM ⇑ | 0.5967 | 0.6167 | −0.0200 | tied within noise |
| under ⇓ | 0.1150 | 0.1017 | +0.0133 | tied within noise |
| over ⇓ | 0.2883 | 0.2817 | +0.0067 | tied within noise |
| exact_match / ROUGE-L / step_f1 / ordered_acc / decomp_mean / composite | 0.0417 / 0.5473 / 0.1973 / 0.1744 / 0.5446 / 0.4137 | 0.0450 / 0.5453 / 0.1969 / 0.1778 / 0.5507 / 0.4150 | −0.0033 / +0.0020 / +0.0005 / −0.0033 / −0.0060 / −0.0013 | all tied within noise |

**In plain words.** Against "the order is matter, it can change up to 10%", the measured effect
on this evaluation set is **an order of magnitude below 10% and statistically indistinguishable
from zero**. Across two model families and a 2× size range, reversing the example order changes
**nothing** on the seven-term report card, in every arm. The single significant statistic in 39
is Qwen-7B's step_f1 at +0.0152, whose confidence interval starts at **+0.0013** — the weakest
form evidence can take and still be called significant — with no correction applied for testing
thirteen things at once. It is recorded as what it is: a small positive step-overlap difference
on one model, not a report-card effect.

**Is the harness simply blind?** No, and this matters for the meeting. The **same** evaluator on
the **same** 600 questions found 6 of 13 significant for the family contrast, 7 of 13 for the
size contrast, and 9 or 10 of 10 against the fine-tuned arm. It discriminates when a real
difference exists. A null from this harness is a measurement, not a failure to look.

**Scope of the claim, stated so it is not over-read.** This holds for: unguided prompting, 4-bit,
five reranked examples from the size-2000 imbalanced pool, the pinned 600, greedy decoding, seed
42. It says nothing about the third ordering Ahmmad named — least-complex-first — which **never
ran** (§6).

**One observation offered as an observation only.** exp-023 generated 6.7% faster than exp-020
(2,304.0 s vs 2,469.5 s) at identical memory. That is a single unreplicated timing pair on a
shared machine. **No ordering effect on speed is claimed.**

### 5.5 Does fine-tuning on the pool alone work? (exp-026)

This is Ahmmad's question 3: *"just let's see the effect of the pool."* A LoRA adapter was
trained on **only the 2,000 questions in the pool** — the same 2,000 the prompting arms retrieve
their examples from — and then evaluated exactly the way exp-008 was.

**Training facts:** 250 optimizer steps (2 epochs × 2,000 examples), 26 minutes 9 seconds, loss
falling from 1.62 at step 10 to 0.1361 at step 250, 41,943,040 trainable parameters out of
7,289,966,592 (0.58%). The 2,000 training questions were checked to have **zero overlap** with
the 600 evaluation questions. The pool's hop mix is **1461 / 413 / 126** — 73% two-hop — which is
**not** the balanced mix of the evaluation set.

**Framing that binds every comparison below, and must be said out loud every time** (ADR 0034
amendment): comparing exp-026 to exp-008 moves training-set **identity, volume AND hop mix
together** — **2,000 vs 19,938 examples** and **~250 vs 2,494 optimizer steps**. It is not a
clean one-factor comparison, and no one should present it as one.

#### 5.5a The headline: pool-only vs full-split fine-tune (exp-026 vs exp-008)

**Same 600 questions. 2,000 training examples / 250 steps vs 19,938 training examples / 2,494
steps.**

**Result: 4 of 13 significant, all four favouring the full fine-tune, all four ≤ 2.7 percentage
points. Of the seven report-card terms, exactly one is significant.**

| statistic | exp-026 pool-only | exp-008 full_train | difference | verdict |
|---|---|---|---|---|
| break_EM ⇑ | 0.0967 | 0.1017 | −0.0050 | tied within noise (p=0.766) |
| SARI ⇑ | 0.6804 | 0.6823 | −0.0019 | tied within noise |
| GED ⇓ | 0.3191 | 0.3094 | +0.0097 | tied within noise |
| **chain_validity ⇑** | 0.9750 | 0.9942 | −0.0192 | **significant, favours full_train** (CI −0.0306 to −0.0086) |
| hop_EM ⇑ | 0.7033 | 0.7100 | −0.0067 | tied within noise (p=0.789) |
| under ⇓ | 0.2583 | 0.2567 | +0.0017 | tied within noise (p=1.0) |
| over ⇓ | 0.0383 | 0.0333 | +0.0050 | tied within noise (p=0.728) |
| exact_match ⇑ | 0.1083 | 0.1083 | **0.0000** | tied exactly (p=1.0; 25 items right only in each) |
| ROUGE-L ⇑ | 0.6607 | 0.6647 | −0.0039 | tied within noise |
| **step_f1 ⇑** | 0.3185 | 0.3411 | −0.0226 | **significant, favours full_train** (CI −0.0421 to −0.0027) |
| **ordered_acc ⇑** | 0.2969 | 0.3237 | −0.0268 | **significant, favours full_train** (CI −0.0464 to −0.0067) |
| decomp_mean ⇑ | 0.6272 | 0.6357 | −0.0085 | tied within noise |
| **composite (legacy)** | 0.5059 | 0.5236 | −0.0177 | significant, favours full_train |

**In plain words.** Training on **only** the 2,000-question pool gets essentially all the way to
full-split fine-tuning at about **10% of the training compute** (1,568.8 s vs exp-001's 15,497 s).
Nine of thirteen statistics are indistinguishable from `full_train`. The four that differ all
favour `full_train`, but by at most 2.7 points, and only one of them (chain_validity) is a
report-card term. **It does not land midway between prompting and full training; it lands next to
full training.**

The honest caveat, again: this is 2,000 vs 19,938 examples and 250 vs 2,494 steps, with a
different hop mix. **We cannot say whether the near-tie is because the pool is well chosen, or
because 2,000 examples is simply enough, or because 73%-two-hop training happens to suit this
evaluation set.** The control that would separate those — a same-size but hop-stratified draw
(the `pool_2000` arm) — has **never been trained** (§6).

#### 5.5b Against the precision-matched prompting arm (exp-026 vs exp-015)

**Same 600 questions. Both Mistral-7B at fp16. Both over the same 2,000-question pool — one
*trained on* it, one *retrieving examples from* it.** This is the cleanest prompting-vs-tuning
pair in the block, because the precision matches.

**Result: 11 of 13 significant, all eleven favouring the fine-tune. Six of the seven
report-card terms favour the fine-tune significantly.**

| statistic | exp-026 fine-tune | exp-015 prompting | difference | verdict |
|---|---|---|---|---|
| break_EM ⇑ | 0.0967 | 0.0350 | +0.0617 | **significant, favours fine-tune** (p=1.21e-07) |
| SARI ⇑ | 0.6804 | 0.5785 | +0.1019 | **significant, favours fine-tune** |
| GED ⇓ | 0.3191 | 0.4714 | −0.1523 | **significant, favours fine-tune** |
| chain_validity ⇑ | 0.9750 | 0.9694 | +0.0056 | tied within noise |
| hop_EM ⇑ | 0.7033 | 0.5400 | +0.1633 | **significant, favours fine-tune** (p=7.11e-12) |
| under ⇓ | 0.2583 | 0.2317 | +0.0267 | tied within noise (p=0.205; slightly worse for the fine-tune) |
| over ⇓ | 0.0383 | 0.2283 | −0.1900 | **significant, favours fine-tune** (p=1.0e-26) |
| exact_match ⇑ | 0.1083 | 0.0450 | +0.0633 | **significant, favours fine-tune** (p=4.5e-07) |
| ROUGE-L ⇑ | 0.6607 | 0.5447 | +0.1161 | **significant, favours fine-tune** |
| step_f1 ⇑ | 0.3185 | 0.1894 | +0.1291 | **significant, favours fine-tune** |
| ordered_acc ⇑ | 0.2969 | 0.1662 | +0.1307 | **significant, favours fine-tune** |
| decomp_mean ⇑ | 0.6272 | 0.5304 | +0.0969 | **significant, favours fine-tune** |
| composite (legacy) | 0.5059 | 0.3132 | +0.1926 | significant, favours fine-tune |

**In plain words.** A 26-minute fine-tune on 2,000 questions beats best-pool few-shot prompting
on the same pool by **6 to 19 points**, on the same 600 questions, on almost everything. The two
exceptions are instructive: **chaining is tied** (prompting already chains well, 0.9694) and
**under-decomposition is tied and if anything slightly worse for the fine-tune**. So the
fine-tune's gain is *not* "it fixes short plans" — it is "it writes the right words, the right
number of steps, and almost never rambles".

#### 5.5c Against the 4-bit prompting arm (exp-026 vs exp-020) — context only, precision differs

**This pair moves precision as well as method** (exp-026 generates at fp16, exp-020 at 4-bit), so
it is context, not a clean comparison. **12 of 13 significant.** Eleven favour the fine-tune,
including over-decomposition −0.2817 (p=1.2e-42), hop_EM +0.1917, GED −0.1655, step_f1 +0.1362,
ROUGE-L +0.1222. The one favouring exp-020 is **under-decomposition, −0.0900** (p=8.0e-06) —
which the prompting arm "buys" by over-decomposing far more (0.3200 vs 0.0383). Chain validity is
tied (+0.0124).

### 5.6 The whole fine-tuned-versus-prompting picture

Four prompting arms were each compared against the full fine-tune on the same 600.

| prompting arm | vs exp-008 full_train | result |
|---|---|---|
| exp-015 Mistral-7B fp16 | 10 statistics compared | **10 of 10 significant, all favouring the fine-tune** |
| exp-020 Mistral-7B 4-bit | 10 statistics compared | **10 of 10 significant, all favouring the fine-tune** |
| exp-021 Qwen2.5-7B 4-bit | 10 statistics compared | **10 of 10 significant, all favouring the fine-tune** |
| exp-022 Qwen2.5-14B 4-bit | 10 statistics compared | **9 of 10 significant**; chain_validity falls out (−0.0104, CI −0.0219 to +0.0006) |

Why ten and not thirteen: exp-008's stored per-item file was written before three columns
(`over_decomposition`, `under_decomposition`, `decomp_mean`) existed, so those three are flagged
**NOT COMPARED** rather than silently dropped. exp-026's headline comparison has a re-scored
version that carries all thirteen, and the two versions agree on every statistic they share.

The size of the gap, from the four arms above: ROUGE-L −0.119 to −0.126, step_f1 −0.144 to
−0.159, ordered accuracy −0.146 to −0.161, SARI −0.103 to −0.110, GED +0.151 to +0.175 (worse for
prompting). Those are large and consistent.

**In plain words.** Fine-tuning beats prompting on this task, on this evaluation set, and it is
**not a Mistral quirk** — changing the prompting family does not close the gap, and neither does
doubling the prompting model's size. The only thing a bigger prompting model buys against the
fine-tune is a **tie on chaining** (the 14B, −0.0104, not significant). Every wording and
step-overlap term still favours the fine-tune by 0.10 to 0.15.

And the fine-tune is also **cheaper to run**: 152.3 tokens and 1.404 s per query, versus 471.6
tokens and 1.810 s for prompting — **about 32% of the tokens**, because the fine-tuned model
needs no examples in its prompt.

### 5.7 The length-error pattern, in one place

This is the single clearest story in the block and it is worth its own slide.

| arm | leans | under ⇓ | over ⇓ | one-line description |
|---|---|---|---|---|
| exp-020 Mistral-7B 4-bit | too many (+0.585) | 0.1683 | 0.3200 | **over-decomposes**; emitted plans of 13, 14, 15, 19 and 126 steps |
| exp-023 Mistral-7B reversed | too many (+0.713) | 0.1783 | 0.2950 | same behaviour, order made no difference |
| exp-021 Qwen2.5-7B | too few (−0.150) | 0.2933 | 0.1533 | **under-decomposes**; longest plan 8 steps, no runaways |
| exp-024 Qwen2.5-7B reversed | too few (−0.127) | 0.2900 | 0.1700 | same behaviour, order made no difference |
| exp-022 Qwen2.5-14B | too many (+0.285) | **0.1017** | 0.2817 | **fixes "too few", does not fix "too many"** |
| exp-025 Qwen2.5-14B reversed | too many (+0.280) | 0.1150 | 0.2883 | same behaviour, order made no difference |
| exp-008 full_train fine-tune | too few (−0.233) | 0.2567 | **0.0333** | **almost never over-decomposes** |
| exp-026 pool-only fine-tune | too few (−0.235) | 0.2583 | **0.0383** | same — almost never over-decomposes |

Four sentences on what this means. **Mistral's failure mode is rambling** — it keeps adding steps,
sometimes catastrophically. **Qwen-7B's failure mode is the mirror image** — it stops too early,
and it does this more the deeper the question. **Doubling Qwen's size fixes the stopping-too-early
problem almost completely but leaves a rambling problem behind**, which is the trade Ahmmad's
asymmetry rule says is worth making. **Both fine-tuned arms behave differently from all of them**:
they essentially never ramble (3-4% over-decomposition against prompting's 15-32%), but they still
stop too early on deep questions — exp-026's under-decomposition is 0.015 at two hops, 0.295 at
three hops, and **0.465 at four hops**; exp-008's is 0.005 / 0.215 / 0.550. That deep-question
shortfall is the clearest remaining weakness of the fine-tuned arm, and it is visible in the
per-hop step_f1 too (exp-026: 0.4495 at two hops, 0.2899 at three, 0.2162 at four).

---

## 6. What this block does **not** show

Stated plainly, because an unmeasured thing presented as measured is the worst kind of error in
a thesis.

1. **The third ordering was never run.** Ahmmad named three: as-is, reversed, and
   least-complex-to-most-complex. Only the first two exist. `complexity_asc` is a **reserved
   value that the code refuses at load time**, with an error pointing at ADR 0032 item 5. The
   reason is that "complexity" was never defined in the meeting — it could mean gold hop count,
   question length, or gold chain length — and picking one silently would be inventing
   methodology. **It ships the moment Ahmmad or Jahid names the key.**
2. **The pool fine-tune's three factors are not separated.** exp-026 vs exp-008 moves training
   identity, training volume and training hop mix all at once. The one control that would
   separate them — `pool_2000`, a seeded hop-stratified draw of the same size — **has never been
   trained** (issue #13). So "the pool is a good pool" and "2,000 examples is enough" and
   "two-hop-heavy training suits this set" are all still on the table, and this block cannot
   choose between them.
3. **Every arm ran once. There is no variance estimate.** Decoding was greedy and the seed was
   fixed at 42 throughout, so a re-run of the same command should reproduce the same numbers —
   but that is reproducibility, not variance. We do **not** know how much these numbers would
   move under a different seed, and no arm was repeated to find out. Every "tied within noise"
   statement above is about sampling noise over the 600 questions only.
4. **4-bit versus fp16 was measured for Mistral only.** exp-020 vs exp-015 is a Mistral-only
   reading. Whether 4-bit costs Qwen2.5-7B or Qwen2.5-14B anything is **unmeasured** — the 14B
   cannot run uncompressed on this GPU at all, and the 7B was never run uncompressed.
5. **The two families do not render their prompts the same way.** The Mistral folder uses
   `prompt_style: plain`; the Qwen folders use `prompt_style: chat_template`, meaning the prompt
   is wrapped in the model's own chat formatting. Both arms see the same five examples and the
   same instruction text, and both were verified 120/120 against their own logged prompts — but
   the literal strings the two models receive are **not byte-identical**, and they cannot be,
   because each model has its own chat template. **How much of the family difference in §5.2 is
   the template rather than the model is unmeasured.**
6. **exp-015's memory use was never measured**, so no memory saving is claimed for 4-bit
   anywhere in this report.
7. **The anchor sheet exists but has not been read.** `runs/anchor/2026-09-20-all-arms.md` holds
   all 8 arms at 45/45 coverage with prompts, raw generations and per-item scores side by side.
   **No hand inspection has been done.** Nothing in this report rests on it. That reading is item
   A4/A7 and it is Jahid's own task.
8. **Per-hop numbers are descriptive, not significance-tested.** The evaluator reports metrics
   per gold hop depth, and those tables appear above. But the significance tests all run on the
   pooled 600. **There is no per-hop significance test in this block.** So "Qwen's chaining
   problem is a two-hop problem" is a description of the numbers, not a tested claim.
9. **Two arms depend on a pending decision.** exp-022 and exp-025 load only because the parameter
   ceiling was raised to 1.5e10. That raise is Jahid's, recorded as **pending supervisor
   confirmation**. If Ahmmad reasserts 8B or 10B, both arms are withdrawn and the size question
   has no answer.
10. **A mechanism named but not tested.** The suggestion that Qwen-7B chains worse because
    shorter plans contain fewer `#k` references to get right is **plausible and untested**.
11. **Absolute SARI and GED values are not externally comparable.** SARI's floor on this data is
    ~0.29, not 0; GED is computed without the spaCy lemmatizer the official Break code uses.
    Differences between our arms are valid; the raw levels should not be put next to published
    numbers.

---

## 7. Cost

All on one NVIDIA RTX 3090 (24 GiB), shared with other users throughout; nothing of anyone
else's was ever killed. "Peak VRAM" is a sampled per-process maximum from `nvidia-smi`, filtered
to the run's own process id — not `torch.cuda.max_memory_allocated`, except where stated.

| exp | what ran | wall clock | generation time (600 queries) | mean latency/query | peak VRAM |
|---|---|---|---|---|---|
| exp-015 | Mistral-7B fp16 prompting | — | 1,086.2 s | 1.810 s | **never measured** |
| exp-020 | Mistral-7B 4-bit prompting | 2,537 s (42 m 17 s) | 2,469.5 s | 4.116 s | 4,818 MiB |
| exp-021 | Qwen2.5-7B 4-bit prompting | 833 s (13 m 53 s) | 812.9 s | 1.355 s | 6,082 MiB |
| exp-022 | Qwen2.5-14B 4-bit prompting | 1,674 s (27 m 54 s) | 1,648.0 s | 2.747 s | 10,416 MiB |
| exp-023 | Mistral-7B 4-bit reversed | 2,324 s (38 m 44 s) | 2,304.0 s | 3.840 s | 4,818 MiB |
| exp-024 | Qwen2.5-7B 4-bit reversed | 851 s (14 m 11 s) | 832.3 s | 1.387 s | 6,082 MiB |
| exp-025 | Qwen2.5-14B 4-bit reversed | 1,624 s (27 m 4 s) | 1,602.2 s | 2.670 s | 10,416 MiB |
| exp-026 **train** | QLoRA on 2,000 examples, 250 steps | 1,568.8 s (26 m 9 s) | n/a | 6.3 s/step | 5.035 GiB (`max_memory_allocated`); 5,648 MiB by `nvidia-smi` |
| exp-026 **evaluate** | fp16 base + new adapter | 854 s (14 m 14 s) | 842.3 s | 1.404 s | 14,438 MiB |
| exp-008 | fp16 base + full_train adapter | — | 832.5 s | 1.387 s | not recorded |
| exp-001 *(for scale)* | QLoRA on 19,938 examples, 2,494 steps | 15,497 s (4 h 18 m) | n/a | 6.2 s/step | not quoted here |

**Total GPU time for the seven new runs: 12,265.8 seconds = 204.4 minutes = about 3 hours 24
minutes.** (Sum of the wall-clock column for exp-020 through exp-026, including both exp-026
steps. This is the one arithmetic operation in this report.)

**Per-query serving cost** (this is the number that matters if the system is ever deployed):

| arm | prompt tokens | completion tokens | total tokens | latency |
|---|---|---|---|---|
| exp-015 prompting (fp16) | 395.2 | 76.4 | 471.6 | 1.810 s |
| exp-008 full_train fine-tune | 122.5 | 29.4 | 151.9 | 1.387 s |
| exp-026 pool-only fine-tune | 122.5 | 29.8 | 152.3 | 1.404 s |

Both fine-tuned arms cost **about 32% of the prompting arm's tokens**, because they need no
examples in the prompt. On this task, fine-tuning is both better and cheaper to serve.

**Three cost readings worth quoting:**

- 4-bit made Mistral **2.3× slower** to generate than fp16 (2,469.5 s vs 1,086.2 s).
- Qwen2.5-7B is **3.0× faster** than Mistral-7B at the same precision. Qwen2.5-14B is still
  **1.5× faster** than Mistral-7B.
- The pool-only fine-tune used **about 10%** of the full fine-tune's training compute
  (1,568.8 s vs 15,497 s) for a result that is statistically tied on 9 of 13 statistics.

---

## 8. Sources

Every number above comes from one of these. Nothing else was consulted.

**Committed (in git):**

- `experiments/log.md` — the rows for exp-008, exp-015, exp-020, exp-021, exp-022, exp-023,
  exp-024, exp-025, exp-026.
- `experiments/exp-015/metrics.json`, `experiments/exp-020/metrics.json`,
  `experiments/exp-021/metrics.json`, `experiments/exp-022/metrics.json`,
  `experiments/exp-023/metrics.json`, `experiments/exp-024/metrics.json`,
  `experiments/exp-025/metrics.json`, `experiments/exp-026/metrics.json`, and each run's
  `notes.md` and `config.json`.
- `docs/METRICS.md` §1–§5 — every metric definition in §4 of this report.
- `docs/analysis/2026-08-23-decomposition-quality-metric-specification.md` — the report-card
  specification.
- `docs/plan/2026-09-14-post-supervisor-meeting-phase.md` §2 — Ahmmad's quoted questions
  (A1, A2, A3, A5, A6).
- `docs/adr/0031-…`, `0032-…`, `0033-…`, `0034-…` — why each arm exists, the precision pinning,
  the ceiling raise, the reserved `complexity_asc`, the anchor-set convention, and the G2/G3
  shape.
- `configs/decomposer_musique.json` — `retrieval.k: 5`, `few_shot.ordering`, decoding settings.

**On the compute box only (under the gitignored `runs/` tree — heavy outputs and data never
enter git):**

- `runs/exp-020/compare_vs_exp015_unguided_fp16/compare_metrics.json`
- `runs/exp-021/compare_vs_exp020_mistral_4bit/compare_metrics.json`
- `runs/exp-022/compare_vs_exp021_qwen7b_4bit/compare_metrics.json`,
  `runs/exp-022/compare_vs_exp020_mistral_4bit/compare_metrics.json`
- `runs/exp-023/compare_vs_exp020_as_is/compare_metrics.json`
- `runs/exp-024/compare_vs_exp021_as_is/compare_metrics.json`
- `runs/exp-025/compare_vs_exp022_as_is/compare_metrics.json`
- `runs/exp-026/compare_vs_exp008_full_train_rescored/compare_metrics.json`,
  `runs/exp-026/compare_vs_exp015_prompting_fp16/compare_metrics.json`,
  `runs/exp-026/compare_vs_exp020_prompting_4bit/compare_metrics.json`
- `runs/exp-026/arm_comparison/arms_metrics.json` and the three re-scored
  `eval_metrics.json` files under it — **the source of exp-008's and exp-015's full re-scored
  report cards, and of exp-015's `mean_signed_step_count_error` (+0.145) and `step_count_mae`
  (0.6717)**, which are not in any committed snapshot.
- `runs/exp-0NN/eval/eval_metrics.json` — the per-hop tables.
- `runs/anchor/2026-09-20-all-arms.md` — the 8-arm, 45-item hand-inspection sheet (not read yet).

**Two numbers in this report are not directly copied from a file:** the total GPU time (3 h 24 m,
a sum of the wall-clock column) and the token-share figure (32%, a division of two token counts).
Both are arithmetic over the committed numbers immediately above them.

## 9. Addendum, later on 2026-09-20 — Qwen3.5-9B, and two corrections from the hand-inspection notes

Written by the lead after the report above was committed. Sources: the exp-027 and exp-028 rows in
`experiments/log.md`, `experiments/exp-027|028/metrics.json`, `runs/exp-027/compare_*/`,
`runs/exp-028/compare_vs_exp027*/`, and the two analysis notes
`2026-09-20-anchor-set-qualitative-analysis.md` and `2026-09-20-validation-45-seed43-analysis.md`.

### 9.1 Qwen3.5-9B joins the axis (exp-027 as_is, exp-028 reversed)

Jahid asked for Ahmmad's "QN 9 billion" to be run. Same recipe as every other arm: unguided, 4-bit,
the pinned 600, the same five retrieved examples. **One caveat travels with every comparison
below:** against the two Qwen2.5 arms, Qwen3.5-9B changes the model *generation* and the *size* at
the same time, so nothing here says which of the two did the work (ADR 0031 item 5).

| metric | Mistral-7B (020) | Qwen2.5-7B (021) | Qwen2.5-14B (022) | **Qwen3.5-9B (027)** | good direction |
|---|---|---|---|---|---|
| step F1 | 0.1823 | 0.1929 | 0.1969 | **0.2135** | high |
| ROUGE-L F1 | 0.5385 | 0.5412 | 0.5453 | **0.5625** | high |
| SARI | 0.5766 | 0.5720 | 0.5795 | **0.5984** | high |
| GED | 0.4846 | 0.4694 | 0.4605 | 0.4838 | low |
| chain validity | 0.9626 | 0.9210 | 0.9838 | **0.8471** | high |
| hop-count exact match | 0.5117 | 0.5533 | 0.6167 | 0.5700 | high |
| Break exact match | 0.0400 | 0.0550 | 0.0333 | 0.0550 | high |
| under-decomposition | 0.1683 | 0.2933 | 0.1017 | 0.1350 | low |
| over-decomposition | 0.3200 | 0.1533 | 0.2817 | 0.2950 | low |

What the tests say (same 600 in every pair):

- **vs Qwen2.5-7B: 7 of 13 significant.** Qwen3.5-9B is better on the wording metrics (SARI,
  ROUGE-L, step F1, ordered accuracy) and writes far fewer too-short plans. Qwen2.5-7B is better on
  chaining and writes fewer too-long plans. *Confounded: generation and size both move.*
- **vs Qwen2.5-14B: 10 of 13 significant.** Qwen3.5-9B is better on the wording metrics and on
  the two verbatim-match metrics. The 14B is much better on chaining (0.9838 vs 0.8471) and on GED.
  *Confounded.*
- **vs Mistral-7B: 8 of 13.** Qwen3.5-9B is better on seven, Mistral on one (chaining).
- **vs the full fine-tune exp-008: 10 of 10, all favouring the fine-tune.** The fourth prompting arm
  in a row to lose on every statistic.

In one sentence: **Qwen3.5-9B is the best of the four prompting models at writing the step text
and the worst at making the steps refer to each other.** It also produced the block's only empty
output (one row, where the model wrote "Question:" and the parser cut everything) and two runaway
plans of 141 and 91 steps. Its weights hold no "thinking" tags, so that setting never fired.

**Reversed order on this model (exp-028 vs exp-027): 3 of 13 significant, all favouring reversed**
— chain validity up 3.5 points, GED down 1.9 points, and the legacy composite. This is the first
model in the block where reversing the example order moved a report-card term. It is still about
a third of Ahmmad's "up to 10%", and no multiple-comparison correction was applied. G3 now reads
0/13, 1/13, 0/13, 3/13 across four models: 4 of 52 statistics.

Cost: 28 minutes for as_is, 25 for reversed, peak VRAM about 8.0 GiB.

### 9.2 Two corrections to the sections above

1. **§5.4 (ordering) is right about the averages and incomplete about the mechanism.** The
   hand-inspection notes show the models rewrite roughly 82% of their plans when the example
   order is reversed (37, 37 and 41 of the 45 anchors; 491, 496 and 492 of the 600). The per-item
   scores swing up and down by amounts as large as the family and size effects, and the swings
   cancel. "Order does not matter" is true of the report card and false of any single question.
2. **§5.1 mentions Mistral's 126-step plan; that item is not an anchor.** But the same failure is
   in the anchor set: the reversed Mistral run (exp-023) produced a 79-step plan on one three-hop
   anchor that the as-is run answered in three steps. And the second random 45 (seed 43) showed a
   ten-step plan for a four-step gold from the reversed 14B run, so runaways are not Mistral-only.

### 9.3 What the two 45-item draws add

Any random 45 is biased by item difficulty, in the same direction for every model: the frozen 45
are harder than the other 555 for seven of eight arms, the seed-43 draw easier for eight of eight.
That does not weaken the 600-item paired tests, where the same items sit on both sides. It bounds
what a 45-item sheet can show: it can show *how* a model fails, never *how much better* one model
is than another. Read per-hop conclusions off the 600, not the sheet.
