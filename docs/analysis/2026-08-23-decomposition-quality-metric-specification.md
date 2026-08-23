# Specification: what to score a decomposition with, and how to combine it (or not)

**Date:** 2026-08-23 · **Author:** analyst agent session · **Base commit:** `a6353d9` ·
**Authorized by:** [ADR 0023](../adr/0023-jahid-2026-08-22-direction-metric-pipeline-completion-generalisation.md)
item 1 ("bring something better from the literature"), re-confirmed by Jahid in session 2026-08-23 ·
**Feeds:** [issue #40](https://github.com/AhmadiJahid/Thesis---QAv2/issues/40),
[issue #6](https://github.com/AhmadiJahid/Thesis---QAv2/issues/6) item 5 ·
**Runs:** none — no model was run, no GPU touched, `runs/run.lock` not taken, no training or inference.

This is a **specification for an implementer**, not a measurement and not an adoption. Whether any
metric here becomes the *thesis-primary* metric is issue #6 item 5 — Jahid's with his supervisor.
The wording throughout is "specify", never "adopt".

---

## 0. Read this before quoting anything below

- **No new numbers were produced in this session.** Every quantitative statement is cited to a
  committed source: [issue #40](https://github.com/AhmadiJahid/Thesis---QAv2/issues/40),
  [`2026-08-22-finetuned-vs-prompting-error-analysis.md`](2026-08-22-finetuned-vs-prompting-error-analysis.md),
  [`2026-08-22-metric-candidates.md`](2026-08-22-metric-candidates.md),
  [`composite-score-literature-check.md`](composite-score-literature-check.md),
  the exp-011 row of [`experiments/log.md`](../../experiments/log.md), or a **published paper's own
  table**. Nothing here is a re-measurement, and there is therefore **no JSON companion** — the
  convention of [ADR 0011](../adr/0011-comparison-artifact-conventions-and-the-significance-claim-floor.md)
  (as amended by [ADR 0020](../adr/0020-prior-work-re-analysis-convention.md)) attaches a companion to
  a note that runs a statistics battery, and this note runs none.
- **What *is* new here:** eight primary sources fetched and read **this session** that the two earlier
  notes did not cover (§1, §6) — Break's own metric table and its *Copy* baseline read from the paper
  rather than from the evaluator, EntailmentBank, Spider, Smatch, GLUE, ONUS, SubQuestRater, and the
  RAGAS source history — and the specification in §2, which no earlier note contains.
- **Binding constraint, restated:** no closed commercial model may score, rate or judge decomposition
  quality (CLAUDE.md standing constraint). **Every term specified in §2 is string-level or
  graph-level; no model of any kind is in the scoring loop.** Where the literature's own metric uses a
  model (EntailmentBank scores its *Intermediates* dimension with BLEURT; SubQuestRater scores with
  GPT-4V; ONUS uses GPT-2 and a BERT classifier), that part is **deliberately not carried over**, and
  §4 says so at the point of the choice. No open-weight model term is proposed either; if one is ever
  wanted it stays where the earlier survey left it — a supervisor question
  ([`2026-08-22-metric-candidates.md`](2026-08-22-metric-candidates.md) §6 item 2).
- **Scope.** This is the decomposition-*quality* half only. Answer EM/F1 from the answering backend
  (`src/answer_metrics.py`, [ADR 0019](../adr/0019-musique-answering-backend-conventions.md)) is the
  extrinsic half and is orthogonal to everything specified here; ReCEval's argument for reporting both
  rather than one (carried, `composite-score-literature-check.md` §3.4) still stands.

---

## 1. What the literature actually uses — and whether it composites

### 1.1 The direct answer to the question asked

**The literature on decomposition quality does not composite. It reports a suite.** That is the
honest finding, it is not a failure to deliver, and it is the reason §2's primary recommendation is a
suite rather than a formula.

Fetched and read this session, in the decomposition line itself:

- **Break / QDMR — Wolfson, Geva, Gupta, Gardner, Goldberg, Deutch, Berant, *Break It Down: A
  Question Understanding Benchmark*, TACL 2020** (fetched `https://ar5iv.labs.arxiv.org/html/2001.11770`).
  §7.1 *Evaluation Metrics* defines **four** metrics and reports them **side by side in one table,
  never blended** (its Table 9). Verbatim structure: *"Sequence-based scores, where higher values are
  better, are denoted by ⇑. Graph-based scores, where lower values are better, are denoted by ⇓."*
  - **Exact Match ⇑** — *"Measures exact match between s and ŝ, either 0 or 1."*
  - **SARI ⇑** (Xu et al. 2016) — *"we consider the sets of added, deleted, and kept n-grams when
    mapping the question x to s … using the standard of up to 4-grams, then average (a) the F1 for
    added n-grams …, (b) the F1 for kept n-grams, and (c) the precision for the deleted n-grams."*
  - **GED ⇓** — *"GED computes the minimal-cost graph edit path required for transitioning from G_s to
    G_ŝ (and vice versa), normalized by max(|G_s|,|G_ŝ|). Operation costs are 1 for insertion and
    deletion of nodes and edges. The substitution cost of two nodes u,v is set to be 1 − Align(u,v),
    where Align(u,v) is the ratio of aligned tokens between these steps."*
  - **GED+ ⇓** — GED plus node merge/split, *"computed only for graphs with up to 5 nodes, covering
    75.2% of the examples in the development set"* (their footnote 8; the Table 9 caption gives 66.1%
    of QDMR and 97.6% of the high-level data).
  This is the metric family PR #44 already ports (three of the four; `norm_EM` and `GED+` are not
  ported — [ADR 0026](../adr/0026-break-faithful-metrics-the-implementers-conventions.md) item 1 and
  §5 of this note).
- **The same paper reports a trivial baseline in the same table, which is directly load-bearing for
  §3.** Break's **`Copy`** system *"copies the input question x, without introducing any
  modifications"* — the published equivalent of this repo's "echo the question" junk system. Its
  published QDMR-dev row: **EM 0.001, SARI 0.431, GED 0.937** (Table 9), against the best model's
  0.154 / 0.748 / 0.318. So **the field itself puts a question-echo baseline in the reported table**,
  and in the published numbers it is EM and GED that separate it — **SARI gives it 0.431**, i.e. more
  than half of the best system's 0.748. §3 turns that into a test requirement.
- **Question decomposition with dependency graphs (Hasson & Berant, \*SEM 2021)** — carried, not
  re-fetched: the *official* QDMR metric is a single **canonicalized EM** (NormEM), and LF-EM was
  validated against human judgment on 50 examples (`composite-score-literature-check.md` §3.2). Note
  the shape of that precedent: where the field does use one number, it is **one member of the family
  made stricter**, not a blend of the family.
- **Visual Question Decomposition on MLLMs (Zhang, Liu, Han, Chen, He et al., EMNLP 2024)** — fetched
  `https://ar5iv.labs.arxiv.org/html/2409.19339`. The newest decomposition-quality framework found in
  this pass. It proposes **SubQuestRater** with *"three critical criteria … 1) Non-Repetition, 2)
  Relevance, and 3) Groundedness"*, and its Table 1 reports **three separate rows per model — no
  composite**. Its scorer is a closed commercial model (*"we choose GPT-4V as the scoring model"*),
  which is exactly the design the supervisor excluded; carried here as evidence about the field's
  direction, not as a candidate.
- **Unsupervised Question Decomposition for QA (Perez, Lewis, Yih, Cho, Kiela, EMNLP 2020)** — fetched
  `https://ar5iv.labs.arxiv.org/html/2002.09758`. Its §5.1 *"Intrinsic Evaluation of Decompositions"*
  is a **four-column panel, unblended** (its Table 4): GPT-2 negative log-likelihood (fluency, lower
  better), % classified well-formed by a BERT classifier, edit distance to the multi-hop question, and
  length ratio. Two of the four are model-based, and it is **reference-free** — it never compares to a
  gold decomposition. Relevant as a fourth independent instance of "panel, not blend"; not a candidate
  here (model-based, and reference-free metrics answer a different question).

Adjacent lines of work where a *structured* prediction is scored against a *structured* gold — the
closest formal analogue to scoring a decomposition, and the source of the only combination forms that
do have precedent:

- **EntailmentBank — Dalvi, Jansen, Tafjord, Xie, Smith, Pipatanangkura, Clark, *Explaining Answers
  with Entailment Trees*, EMNLP 2021** (fetched `https://ar5iv.labs.arxiv.org/html/2104.08661`). The
  single most transferable design in this survey. Its §6.1: predicted tree nodes are **first aligned**
  to gold nodes (*"using the sent* labels and Jaccard similarity for intermediate nodes … instead of
  doing exact match against gold tree, we account for semantic-preserving variants"*), then scored on
  **four dimensions × two aggregations**: Leaves (F1, AllCorrect), Steps (F1, AllCorrect),
  Intermediates (F1, AllCorrect), and **Overall Proof (AllCorrect)** — verbatim: *"The overall
  'AllCorrect' score for a generated proof is 1 only if all of the leaves, steps, and intermediates
  are all correct … This is a strict metric: any error in the generated tree will result in a score of
  0."* Their Table 4 reports all seven columns; the single number they select models on is the strict
  **conjunction**, not a weighted sum (Appendix C: *"picked the model that gives best Overall
  AllCorrect score on the Dev set"*). Two further honest details: the *Intermediates* dimension is
  scored with **BLEURT** (a model — not carried here), and the authors state their own metric
  **underestimates** validity because *"even with the same input sentences, the tree can be structured
  in several valid ways"* (≈25% of the imperfect trees they inspected).
- **Spider — Yu, Zhang, Yang, Yasunaga, Wang, Li et al., EMNLP 2018** (fetched
  `https://ar5iv.labs.arxiv.org/html/1809.08887`). §6: *"Our evaluation metrics include Component
  Matching, Exact Matching, and Execution Accuracy"* — component-wise **F1 on exact set matching**
  reported per component (*"To report a model's overall performance on each component, we compute F1
  score on exact set matching"*), and the single number is again a **conjunction**: *"The predicted
  query is correct only if all of the components are correct."* Components are compared **as sets**, so
  benign ordering inside a clause is neutralised by the definition rather than by a weight.
- **Smatch — Cai & Knight, ACL 2013, pp. 748–752** (paper fetched from
  `https://aclanthology.org/P13-2131.pdf`; official implementation README fetched from
  `snowblink14/smatch`). The field standard for scoring one graph against another with **one** number,
  and it is a plain F1, not a blend: *"we define the smatch score (for semantic match) as the maximum
  f-score obtainable via a one-to-one matching of variables between the two AMRs"*; the official
  README states it exactly — `P = M/T`, `R = M/G`, `F = 2PR/(P+R)` over matching triples under the
  best node mapping. Finding NP-complete in general, solved by ILP or hill-climbing. Worth knowing as
  the alternative *shape* to GED for graph comparison (overlap-F1 instead of edit distance); **not**
  specified in §2, because adapting it to free-text step labels requires choosing a label-match
  criterion, which is precisely the choice GED's `1 − Align(u,v)` already makes and which Break has
  already made for this exact data type.
- **HotpotQA's joint EM/F1** (carried): per-example **product** of answer and support scores. The one
  published "combine two components" metric in multi-hop QA, and it is multiplicative and per-example
  with no free credit (`composite-score-literature-check.md` §3.1).

### 1.2 Where compositing *does* exist, it is benchmark-level and it takes exactly four forms

None of these is a decomposition metric; they are the industry/benchmark precedent for the *act* of
combining, and they are the only defensible templates available:

| form | precedent (verified) | definition | property that matters here |
|---|---|---|---|
| **unweighted mean of components** | **GLUE** (Wang et al., ICLR 2019; fetched `https://ar5iv.labs.arxiv.org/html/1804.07461` §3.4): *"The benchmark site shows per-task scores and a macro-average of those scores to determine a system's position on the leaderboard. For tasks with multiple metrics (e.g., accuracy and F1), we use an unweighted average of the metrics as the score for the task"* | arithmetic mean, equal weights, components each reported | **equal weights are the only weighting with a precedent**; components must be on a common scale and same direction |
| **unweighted harmonic mean of components** | **RAGAS** (industry RAG-eval framework). Read off the source: `src/ragas/evaluation.py` at tag `v0.0.16` computes `self["ragas_score"] = len(values) / reciprocal_sum` over the component means — an unweighted harmonic mean — and it is **gone by `v0.1.0` and absent from `main` today** | harmonic mean of column means | penalises a weak component (unlike an additive blend), and **the best-known industry composite of eval sub-metrics was retired** |
| **strict per-example conjunction** | EntailmentBank *Overall AllCorrect*; Spider *Exact Matching* (§1.1) | 1 iff every dimension is perfect | no free credit is structurally impossible; per-example, so McNemar applies |
| **rank aggregation / win rate** | **HELM** (Liang et al., 2022; fetched `https://ar5iv.labs.arxiv.org/html/2211.09110`, Figure 26): *"We report the fraction of head-to-head comparisons between the given model and all other models, across all scenarios, where the given model is higher along the metric … If a model was the highest for the given metric for every scenario, it would receive a score of 1.0"* | fraction of pairwise wins | **needs no weights and no unit conversion**; note HELM computes it **per metric across scenarios**, not across metrics |
| *(the one weighted-average precedent)* | **Dynascore** (carried, `composite-score-literature-check.md` §3.5) | weighted mean **after** converting every metric into common units via an estimated average marginal rate of substitution, weights user-adjustable | a weighted blend is defensible **only** with a unit conversion; nothing in this repo estimates one |

And the negative precedent, carried: **The Benchmark Lottery** (Dehghani et al., 2021) §3.2 — rankings
change with the aggregation choice; averaging up-weights components with more headroom. Measured on
this repo's own metric: the v1 "typed beats uniform" ordering fails in ~27–28% of weightings on the
4-simplex (`composite-score-literature-check.md` §4.8).

### 1.3 What this pass did *not* establish

- **No published intrinsic decomposition-quality metric on MuSiQue specifically was found**, in this
  pass or the previous one. The searches here were arXiv **metadata** queries (title/abstract), which
  is weak evidence: a paper that uses SARI/GED in its results table without naming the metric in its
  abstract is invisible to them. Treat as **unestablished**, not as "no such work exists".
- **The Break leaderboard page itself is still unverified.** `leaderboard.allenai.org/break` did not
  resolve from the compute box this session (curl returned HTTP `000`). The four-metric set is
  verified from the *paper* (§1.1) and from the official evaluator (previous note §1.2), which is
  stronger evidence than the page; the page remains an open gap.
- **No decomposition paper found reports a weighted linear blend of sub-metrics.** Two independent
  passes have now looked (PR #38's, the 2026-08-22 survey's) and this one adds five more works. That
  is convergent, but it is still absence-of-evidence: state it as "no precedent found", not "none
  exists".

---

## 2. The recommended primary specification

**Recommendation, in one line: make the primary an unblended suite of six per-item terms, and take the
single ordering number from inside that suite rather than from a formula over it.** Everything in
tier P1 is already computed at `a6353d9`; the implementation work is registration, guards, reporting
and tests, not new metric mathematics.

### 2.1 P1 — the primary: the suite (no composite)

Name it `decomposition_report_card` in the run note; in the metrics JSON these are the existing keys.
All six are **macro means over every evaluated item** — there is no pooled ratio anywhere in it, which
is the structural fix for the issue #40 defect class (§3).

| # | term (metrics-JSON key) | per-item column | dir | range | what it prices | provenance |
|---|---|---|---|---|---|---|
| 1 | `break_exact_match_rate` | `break_exact_match` | ↑ | {0,1} | whole-plan verbatim identity of the `@@SEP@@`-joined string, lowercased | **literature**: Break official `get_exact_match` |
| 2 | `sari_macro` | `sari` | ↑ | [0,1] | n-gram *edit* quality vs the question: kept-F1, added-F1, deleted-precision, n = 1…4 | **literature**: Break/Xu et al. |
| 3 | `ged_macro` | `ged` | **↓** | [0,∞) | one distance over wording (node substitution), chaining (edges) and step count (node ins/del) | **literature**: Break official GED |
| 4 | `chain_validity_macro` | `chain_validity` | ↑ | [0,1] | did the plan chain where the gold requires chaining, and do its `#k` resolve backwards | **house repair**, not literature (PR #44 / ADR 0026) |
| 5 | `hop_count_exact_match_rate` | `hop_count_exact_match` | ↑ | {0,1} | granularity: right number of steps | **house** |
| 6 | `under_decomposition_rate` **and** `over_decomposition_rate`, reported as a **pair, never summed** | `step_count_signed_error` (add 0/1 columns, §2.5) | under **↓**, over ↓ but tolerated | [0,1] | the direction of a length error, which the supervisor judged asymmetric | **house**, required by [ADR 0017](../adr/0017-triage-of-the-2026-08-12-transcript-cross-check.md) item on [33:23] |

Reporting rules that are part of the specification, not decoration:

1. **All six are always reported together**, with a `direction` column, exactly as Break prints ⇑/⇓
   (§1.1). A note or table that quotes fewer than six is not reporting this metric.
2. **Every comparison carries the ADR 0009 battery per term** — paired bootstrap CI, paired t-test,
   and McNemar for the binary terms (1, 5, 6). Every term above has a per-item value, so the composite's
   structural weakness (1 of 3 tests — `composite-score-literature-check.md` §4.9) disappears.
3. **Per gold hop depth**, as `per_gold_hop_metrics` already does.
4. **A junk-baseline block is reported beside the arms** (§3.3). Break prints `Copy` in its own results
   table (§1.1); this is the same practice, and it is what makes the issue #40 pathology visible at a
   glance instead of two analysis notes later.
5. **Composite terms with model-dependent denominators are forbidden in the headline** (§3.1). This is
   the general rule; `reference_validity_micro` is the instance that motivated it.

*Why a suite and not a formula:* it is what every decomposition paper in §1.1 does; it removes every
pathology of §3 that arises from *combining*; and it costs zero new metric code. What it costs is
rhetorical — six numbers read weaker in an abstract than one — and a multi-cell sweep needs a stated
ordering rule, which is P2.

### 2.2 P2 — the single ordering number, when one is required: `ged_macro`

For the places that structurally need one number (ranking the 33-cell pool sweep, model selection, an
abstract sentence), **use one member of the suite rather than a function of it** — the Break/QDMR
precedent for a single number (NormEM) and the EntailmentBank precedent (select on one strict column)
both have that shape. The member to use is **`ged_macro`**, on four properties, three of them measured:

- it is a **published metric of the field's flagship decomposition benchmark**, with the definition
  quoted verbatim in §1.1;
- it is the only single candidate **substantially correlated with every house metric** (measured
  Spearman ρ −0.669 `step_f1`, −0.689 `ordered`, −0.784 `rouge_l_f1`, −0.620 `hop_count_EM`, −0.430
  `exact_match` over 5400 pooled per-item values — `2026-08-22-metric-candidates.md` §3.6), i.e. it
  behaves as a summary of the panel;
- it **ranks every junk system below every real arm** on this eval set (junk floor 0.9011 vs real-arm
  worst 0.4910 — ibid. §3.1), which the house composite does not;
- it is **per-item**, so it takes the whole ADR 0009 battery, and its published interval on the
  exp-008 comparison is 0.040 wide and symmetric against the composite's 0.225 wide and asymmetric
  (ibid. §3.3).

**Three caveats that must travel with it, every time:** (a) **lower is better** — the repo already
carries this in data (`LOWER_IS_BETTER_STATISTICS`), keep it there; (b) it is **order-light**, and on a
2-step plan order-blind (a reversed gold scores GED 0.2875, better than every real arm — ibid. §3.7),
which is exactly why terms 1, 5 and 6 of the suite must stay reported beside it; (c) **absolute values
are not comparable to published Break GED**, because this port has no spaCy lemmatizer in the
substitution cost (ADR 0026 item 3).

### 2.3 P3 — if a blended single number is required anyway: `decomp_mean`, unweighted, per item

Specify this **only** for the case where Jahid's supervisor requires one blended number. It is
deliberately the GLUE form (§1.2): unweighted, components individually reported, common scale, common
direction — and, unlike today's composite, **computed per item** so it has a per-item value and takes
the full battery.

Per item *i*, every term in [0,1] and higher-is-better:

```
t1(i) = break_exact_match(i)                     # 0/1
t2(i) = sari(i)
t3(i) = 1 - min(1.0, ged(i))                     # direction flip + clamp (judgment call, §4)
t4(i) = chain_validity(i)
t5(i) = hop_count_exact_match(i)                 # 0/1

decomp_mean(i)     = (t1 + t2 + t3 + t4 + t5) / 5
decomp_mean_macro  = mean over evaluated items of decomp_mean(i)
```

Weights: **all equal, 1/5, and there is no weighted variant.** Stated plainly: equal weighting is the
only weighting this survey found a precedent for (GLUE's unweighted average, RAGAS's unweighted
harmonic mean); **any unequal weighting would be arbitrary** unless it comes with a Dynascore-style
unit conversion, and nothing in this repo estimates a marginal rate of substitution between GED and
SARI. If the supervisor wants unequal weights, the honest form is Dynaboard's: publish the number
**under a weight sweep** so the weight-dependence of a conclusion is visible
(`composite-score-literature-check.md` §3.5, §4.8).

**Known hazard of P3, stated because it is the same failure family as issue #40:** SARI's floor on this
data is **not** 0 — an empty prediction measured **0.2911** and fixed-junk systems 0.3229 / 0.3361
(`2026-08-22-metric-candidates.md` §3.1), and Break's own published table gives its `Copy` baseline
SARI 0.431 (§1.1). So **any additive blend containing SARI hands junk a nonzero floor**, and the size
of that floor is what decided the composite's junk-outranks-real pathology. In a suite this is
harmless (SARI is read beside GED and EM, which both separate junk); inside a sum it is the defect.
**P3 must therefore not be quoted at all until the §3.3 junk battery has been run on it**, and if a
junk system outranks a real arm, P3 is disqualified and P1/P2 stand.

**A harmonic-mean variant (the RAGAS form) is not recommended per item** and the reason is arithmetic:
`break_exact_match` is 0 on roughly 90–95% of items in every committed arm (0.0483–0.1017 —
exp-011 row), and a harmonic mean containing a 0 is 0, so a per-item harmonic mean would be
identically 0 for almost every item. If a harmonic mean is wanted, it must be taken over the
**aggregate** column means, which is what RAGAS did — and that reintroduces the aggregate-of-aggregates
problem that costs the composite two of its three tests. Recorded so the option is not re-derived.

### 2.4 P4 — ranking many cells across the whole suite with no weights: `suite_win_rate`

For the 33-cell sweep, where the reason a single number is wanted is *ordering* rather than
*measurement*, there is a weight-free option from HELM (§1.2). Specified exactly:

```
Inputs: a set A of arms scored on the SAME eval set; the suite's K headline terms
        (K = 5: break_exact_match_rate, sari_macro, ged_macro, chain_validity_macro,
         hop_count_exact_match_rate), each with its direction.

For metric m and ordered pair (a, b), a != b:
    w(a, b, m) = 1.0  if a is strictly better than b on m (direction applied)
               = 0.5  if the two values are exactly equal
               = 0.0  otherwise

suite_win_rate(a) = ( 1 / (K * (|A| - 1)) ) * sum_m sum_{b != a} w(a, b, m)
```

Properties, all of them consequences of the definition: no weights, no unit conversion, no
normalization, direction handled per metric, GED's unboundedness irrelevant, 0.5 = average of the
comparison set. **It is relative to `A`** — adding or removing an arm can change it — which is HELM's
own property and must be printed next to it, together with `K`, `|A|` and the member list. It has **no
per-item value**, so it is a **descriptive ordering device only**: significance claims stay with the
suite's own paired tests. Its one honest weakness: equal implicit weight per metric means correlated
members double-count, and the correlations are measured (`step_f1` vs `ordered_step_accuracy` ρ 0.947;
SARI vs `rouge_l_f1` 0.756; `chain_validity` vs everything 0.18–0.25 — ibid. §3.6). The K = 5 list
above deliberately takes one member per axis and excludes the ρ = 0.947 pair from the aggregate.

### 2.5 What the implementer has to build (P1 is mostly already built)

1. **New per-item 0/1 columns `over_decomposition` and `under_decomposition`**, derived from the
   existing `step_count_signed_error`, plus their registration in `MCNEMAR_STATISTICS` and
   `T_TEST_STATISTICS`, and `under_decomposition` in `LOWER_IS_BETTER_STATISTICS`. Today the two rates
   are aggregate-only, so the ADR 0017 asymmetry cannot be significance-tested. This is the only
   genuinely missing *measurement* in P1.
2. **A guard on `chain_validity`'s free-credit branch:** report
   `chain_validity_gold_unchained_items` (count of items where the gold emits no `#k`, which is the
   branch that returns 1.0 for free). On the ADR 0007 pinned 600 this count is **0** by construction —
   every gold decomposition is a tree with exactly n−1 references, measured
   (`2026-08-22-metric-candidates.md` §3.7) — but MetaQA or any future gold could reintroduce free
   credit silently, which is exactly how issue #40 happened. A count in the JSON makes it visible.
3. **Denominator transparency on every headline term:** each term reports the number of items it was
   averaged over (`n_items`), and no headline term may be a ratio pooled across items (§3.1).
4. **P2 needs nothing** — `ged_macro` exists, is registered, is bootstrapped, and carries its
   direction in data.
5. **P3, if built:** one new per-item column `decomp_mean` (formula in §2.3) plus its macro aggregate,
   registered in `BOOTSTRAP_STATISTICS` and `T_TEST_STATISTICS`. Its component list and the GED clamp
   go in `configs/musique_eval.json` so a run snapshot records them, exactly as the current weights do.
6. **P4, if built:** a small reporting-side function over already-written metrics JSONs (it is
   cross-arm, so it does not belong inside a single scoring run); `scripts/pool_sweep_orchestrator.py`
   is the natural host.
7. **Data required, all of it already on disk:** `pred_steps`, `gold_steps` (terms 1, 3, 4, 5, 6),
   the `question` (term 2, SARI's source), the gold `hop_count` (term 5), and references read as bare
   `#k` (terms 3, 4). Re-scoring all nine committed arms is therefore CPU-only from existing per-item
   files — the exp-011 precedent, which took no GPU.
8. **Verification scope:** any metric definition is shared pipeline code → Gate 1 review plus a smoke
   stage, and hand-computed golden values in `tests/test_decomposition_metrics.py` for anything new
   (the existing pattern). §3.3's junk battery belongs in that test file, not in an analysis note.

---

## 3. Defect avoidance — how this specification fails differently

The three measured defects, and the structural reason each cannot recur under §2.

### 3.1 Empty and degenerate denominators

*What happened:* `reference_validity_micro` is a ratio pooled over items whose **denominator is
produced by the model**. On five of seven arms the denominator was empty and the term scored 1.0 by
rule; on the other two, **2 of 600 items owned the whole denominator** and it scored 0.0 (issue #40;
`2026-08-22-finetuned-vs-prompting-error-analysis.md` §1). One invalid reference on 1 of 600 items
moves the composite by exactly −0.2000, equivalent to 300 items each losing a perfect step F1
(ibid. §1.1).

*The rule this specification adopts, which is general and not about references:* **no headline term may
have a model-dependent denominator.** Every term in §2.1 is a per-item value in a bounded range,
averaged over a denominator fixed by the **evaluation set** (600), not by what the model emitted. An
item that emits nothing scores that item badly; it cannot vanish from the denominator and it cannot
score 1.0 for abstaining.

*The one residual conditional-credit branch is `chain_validity`'s* "gold emits no `#k` → 1.0", and it
is (a) conditioned on the **gold**, never on the prediction, so no model can trigger it, and (b)
**dead on this eval set** — 0 of 600 gold items are unchained (§2.5 item 2). The required
`chain_validity_gold_unchained_items` counter is what keeps it visible on a future dataset.

### 3.2 Terms decided by a handful of items

*What happened:* a term carrying 20% of the primary metric was determined by **2 references over 2 of
600 items** in exp-004 `unguided`/`unguided_capped` (issue #40) and, in v1, by 8, 9 and 4 references
over 3, 4 and 1 items respectively ([`composite-score-literature-check.md`](composite-score-literature-check.md)
§4.1) — while the other 596–599 items contributed nothing to it.

*Why it cannot recur:* every §2.1 term is defined on **every** item — EM and hop-count EM are 0/1 for
each of the 600, SARI and GED are defined for every pair of non-empty strings/graphs (GED's normalizer
is `max(|G_pred|, |G_gold|)`, and the gold graph has ≥ 2 nodes on this data, so the denominator is
never 0 even for an empty prediction), and `chain_validity` is defined per item with the gold-side
condition above. **A term whose per-item denominator is 1 cannot be decided by a subset of items.**
Add the `n_items` requirement of §2.5 item 3 and the property is auditable rather than argued.

*The corollary for significance:* because every term is per-item, "decided by a handful of items"
becomes a statement the bootstrap can *see* — the exp-008 composite's bimodal, asymmetric,
0.225-wide interval (`experiments/exp-008/metrics.json`, decomposed in
`2026-08-22-finetuned-vs-prompting-error-analysis.md` §0) was the interval trying to tell us this.

### 3.3 Rankability of junk — the battery the implementation must pass

*What happened:* under the house composite, three fixed junk steps score **0.2778** and a
question-echo **0.2333**, against exp-004 `unguided`'s **0.2098** — junk outranks the deployable
baseline on its own eval set (issue #40; `2026-08-22-finetuned-vs-prompting-error-analysis.md` §1.2).

*The requirement:* the implementation ships a **junk battery** in
`tests/test_decomposition_metrics.py`, run over the pinned gold, with these systems built from the
gold column (the same six the earlier notes used, plus Break's own `Copy` in spirit — the question-echo
row *is* `Copy`):

| # | system | construction |
|---|---|---|
| J1 | empty prediction | zero steps |
| J2 | question echo (= Break's `Copy`) | one step: the question verbatim |
| J3 | one fixed junk step | one constant string, same for every item |
| J4 | three fixed junk steps | three constant strings, same for every item |
| J5 | gold, order reversed | gold steps in reverse order (content-perfect, maximally mis-ordered) |
| J6 | gold itself | the gold decomposition verbatim |

**Assertions (these are the acceptance criteria, not observations):**

- **A1 — anchors.** J6 scores the extremum on every term: `break_exact_match` 1.0, `sari` 1.0,
  `ged` 0.0, `chain_validity` 1.0, `hop_count_exact_match` 1.0, over/under rates 0.0. (Measured to
  hold for the four ported terms in `2026-08-22-metric-candidates.md` §3.1; assert, don't assume.)
- **A2 — junk is beaten.** For J1–J4, **every** real arm must be strictly better on **`ged_macro`,
  `chain_validity_macro` and `break_exact_match_rate`**. This is the assertion the composite fails and
  the one that matters. Reference values measured at commit `efb8530` on the pinned 600 (ibid. §3.1),
  for the *ordering*, not the digits: GED junk floor **0.9011** vs real-arm worst **0.4910**;
  chain-validity junk **0.0000** vs real-arm worst **0.8605**.
- **A3 — SARI is explicitly exempted from A2, and that is the point.** SARI's junk values are
  0.2911–0.4676 against a real-arm worst of 0.5849 (ibid.), and Break's own published `Copy` row
  scores SARI **0.431** (§1.1). So SARI *does* separate junk on this data, but by a small margin and
  from a high floor. The test asserts the ordering **and** records the margin, so a future arm that
  falls below the junk floor is caught rather than reported.
- **A4 — no additive blend may rank junk above a real arm.** If P3 is implemented,
  `decomp_mean(J1..J4) < decomp_mean(every real arm)` is a **hard gate**; failing it disqualifies P3
  rather than prompting a weight tweak. This is the direct lesson of issue #40: the composite's
  weights were never the disease, the free-credit term was, and a weight tweak would have hidden it.
- **A5 — order sensitivity survives.** J5 must be caught: `break_exact_match` 0.0 and
  `ordered_step_accuracy` far below every real arm. GED is **not** allowed to carry this assertion —
  measured, J5 scores GED 0.2875, *better* than every real arm (ibid. §3.7). This is why term 1 stays
  in the suite and why P2's caveat (b) is not optional.
- **A6 — the composite regression test stays.** Whatever is added, the existing golden values for
  `composite_score` and every pre-existing metric must not move (the exp-011 precedent verified
  additivity empirically for PR #44; hold the same bar).

---

## 4. Every choice, labelled: literature or judgment call

| choice | label | citation / alternatives |
|---|---|---|
| Report a suite, not a composite | **from the literature** | Break §7.1 four metrics side by side; EntailmentBank 4 dimensions; Spider 3 metrics; ROSCOE suite; HELM multi-metric; ONUS 4-column panel; SubQuestRater 3 criteria (§1.1) |
| Terms 1–3 (Break EM, SARI, GED) | **from the literature** | Break/QDMR, definitions quoted verbatim §1.1; already ported per ADR 0026 |
| Term 4, `chain_validity` (no free credit for silence) | **judgment call — house repair, not a published metric** | PR #38 §5 option 3(a) made concrete; the literature-grounded alternative for the same axis is Break's `structural_match`, which bundles step count with structure and gives three junk steps 0.5558 (`2026-08-22-metric-candidates.md` §3.7b). Kept because it is measured near-independent of everything else (ρ 0.18–0.25) and because the free-credit convention **flipped a model ranking** (ibid. §3.2) |
| Terms 5–6 (hop-count EM; over/under split) | **judgment call, following a stated supervisor judgment** | ADR 0017 / [33:23] "over-decomposition is tolerable, under-decomposition is not". **No metric in this survey expresses that asymmetry**, so it is carried by reporting the two rates separately rather than inside a term. The alternative — an asymmetric penalty inside the headline — would be an invented metric with no precedent found, and is not proposed |
| The six-term membership (which axes, and only these) | **judgment call** | driven by measured redundancy (ρ table, ibid. §3.6): one wording axis, one graph axis, one strict axis, one chaining axis, one granularity axis. Alternatives: add `structural_match` (redundant with GED + count), add `rouge_l_f1` (ρ 0.756 with SARI), add `step_f1`/`ordered` (ρ 0.947 with each other) |
| P2 = `ged_macro` as the single ordering number | **judgment call, on a literature metric** | the *metric* is Break's; *choosing it as the ordering number* is this note's judgment, argued on three measured properties (§2.2). Alternatives: `break_exact_match_rate` (strictest and most literature-canonical, but floors at 0.048–0.102 across committed arms — exp-011 row — so it cannot order a 33-cell sweep); a canonicalized EM in NormEM/LF-EM form (best literature backing for a single number, but needs a new MuSiQue canonicalizer whose validity is argued from scratch — ADR 0026 item 1) |
| P3 exists at all | **judgment call, contingent** | offered only against a supervisor requirement for one blended number; the recommendation is P1 + P2 |
| P3's equal weights (1/5 each) | **from the literature for the *form*, arbitrary for the *content*** | GLUE's unweighted average and RAGAS's unweighted harmonic mean are the precedents for *equal* weighting (§1.2). **Any unequal weighting is arbitrary** absent a Dynascore-style unit conversion, which this repo cannot estimate. The current 0.4/0.3/0.2/0.1 has no recorded derivation in either repo (`composite-score-literature-check.md` §1) |
| P3's GED normalization `1 − min(1, ged)` | **judgment call** | GED is unbounded above and can exceed 1.0 (measured 1.0275 for three junk steps, ibid. §3.1), so an additive blend needs a clamp. Alternatives: `1/(1+ged)` (smooth, no clamp, changes the implied exchange rate); `exp(−ged)`; leave GED out of the blend entirely and let P2 carry it. **None is from a source** |
| P3 uses `hop_count_exact_match` as its length term rather than `max(0, 1 − MAE/3)` | **judgment call** | the current term is built from an aggregate MAE, is direction-blind, and is nearly a constant offset at the observed operating point (spread 0.0062 across three v1 systems whose composites span 0.17 — ibid. §4.7). A 0/1 per-item term is testable and per-item; it is also cruder |
| P4 = win rate across metrics | **judgment call adapting a literature form** | HELM's win rate is computed **per metric across scenarios**; applying it **across metrics** is this note's adaptation and must be labelled as such wherever it is printed. Alternatives from Benchmark Lottery §3.2: geometric mean, macro-grouping, rank aggregation |
| No model-based term anywhere | **from a standing constraint, plus a judgment call** | the constraint excludes closed commercial scorers; extending the *reasoning* to open-weight scorers is the supervisor's call (`2026-08-22-metric-candidates.md` §6 item 2). This spec stays entirely string/graph-level so that it is admissible under either answer. Note what is given up: EntailmentBank's BLEURT-scored Intermediates dimension, and every near-miss paraphrase (measured: 128/600 exp-008 items one step short of EM, median character similarity 0.537 — ibid. §4 item 7) |
| Junk battery as an acceptance test | **from the literature (practice), formalised here** | Break prints a `Copy` baseline in its own results table (§1.1). Turning that into a **pass/fail test** is this note's judgment call, and it is the single cheapest guard against a repeat of issue #40 |

---

## 5. Recommendation on `_REF_RX` (issue #40 open question 1)

**Verdict, offered as a recommendation with its evidence, not as a settled fact: treat
`_REF_RX = r"\[#(\d+)\]"` as a defect *in effect*, and do not repair it in place.**

The evidence, in order of weight:

1. **The field's own syntax is bare `#k`.** Break's official `format_qdmr` rewrites `#(\d+)` →
   `@@\1@@`, and its graph builder reads `@@(\d+)@@` (verified from the official evaluator, previous
   note §1.2). Break's own paper prints decompositions with bare `#1`/`#2` references (§1.1's Figure 7
   example). Nothing in the field writes `[#k]`.
2. **This eval set's gold contains zero bracketed references** — 1200 bare `#k` across 600 items, 0
   bracketed (`2026-08-22-finetuned-vs-prompting-error-analysis.md` §1, companion `finding_2b`). So
   the term named "reference validity" has never measured reference validity on this data.
3. **It was already near-vacuous in v1**, i.e. the regex mismatch is not new to v2: items emitting any
   `[#k]` were 3/600, 4/600 and 1/600 across v1's three runs, total references 8, 9 and 4
   ([`composite-score-literature-check.md`](composite-score-literature-check.md) §4.1).
   Whatever the original intent, the term has never had a working denominator in either repo.
4. **Its measured effect is to move the metric without measuring anything**: it explains 95.3% of the
   Mistral unguided→oracle_guided composite gap and 93% of the apparent Qwen-over-Mistral gap
   (issue #40), and it is the mechanism behind exp-008's bimodal composite interval.

**On intent, the honest answer is that this note cannot settle it.** No document in either repo records
a decision to use brackets; `docs/prior-work.md` records the free-credit edge but not the syntax. Intent
is Jahid's to state. **The recommendation does not depend on it**: even a `_REF_RX` corrected to bare
`#k` would still be a **micro-pooled rate with a model-dependent denominator** inside an
aggregate-of-aggregates composite — §3.1's defect class, not just its instance. Fixing the regex would
convert a term that measures nothing into a term that measures the wrong shape.

**Recommended handling, in preference order:**

1. **Freeze, don't fix.** Leave `_REF_RX` and `composite_score` byte-identical, so every committed
   number (nine arms, the 33-cell pool sweep, both v1 re-analysis notes) stays valid and quotable; mark
   the composite **legacy/deprecated** in the metrics JSON (e.g. a `legacy: true` flag beside it, or a
   `composite_score_status` string) and in `docs/METRICS.md` §4; headline the §2.1 suite instead; and
   record in an ADR that the bracketed regex is **retained for reproducibility of published numbers,
   not because it is correct**. This is the only option that costs no re-scoring and loses no history.
2. **If Jahid wants the composite repaired in place instead:** the minimum defensible repair is to
   replace `reference_validity_micro` with `chain_validity_macro` (per-item, bare `#k`, no free credit)
   at the same 0.2 weight, then re-score all nine arms plus the sweep, and state a cut-off date after
   which the old and new composites are **not comparable** (they are different quantities; ADR 0011
   convention 3 already refuses to compare across weights, and this is a stronger break than a weight
   change). Expected direction of the change, measured: the Qwen/Mistral chaining ranking **reverses**
   under the no-free-credit convention (0.8605 vs 0.9686 against 0.9817 vs 0.9690 —
   `2026-08-22-metric-candidates.md` §3.2), so this repair moves a published reading and must be
   announced, not slipped in.
3. **Not recommended: silently correcting the regex.** It changes the value of a metric under an
   unchanged name, moves seven committed arms plus 33 sweep cells, and leaves the pooled-denominator
   defect intact.

---

## 6. Citations

**Fetched and read this session (new to this note).** Titles, authors and venues below were read off
the fetched page or the arXiv Atom API in this session; where a definition is quoted, the quotation is
verbatim from the fetched text.

| # | Work | What was taken from it | Fetched |
|---|---|---|---|
| 1 | Wolfson, Geva, Gupta, Gardner, Goldberg, Deutch, Berant — *Break It Down: A Question Understanding Benchmark*, TACL 2020 | §7.1 metric definitions (EM, SARI, GED, GED+) verbatim; the ⇑/⇓ direction convention; Table 9 including the `Copy` baseline row (EM 0.001 / SARI 0.431 / GED 0.937) | `https://ar5iv.labs.arxiv.org/html/2001.11770` |
| 2 | Dalvi, Jansen, Tafjord, Xie, Smith, Pipatanangkura, Clark — *Explaining Answers with Entailment Trees*, EMNLP 2021 | §6.1: align-then-score; four dimensions × (F1, AllCorrect); *Overall Proof (AllCorrect)* as a strict conjunction and as the model-selection metric; BLEURT-0.28 threshold; the authors' own "metric underestimates validity" caveat | `https://ar5iv.labs.arxiv.org/html/2104.08661` |
| 3 | Yu, Zhang, Yang, Yasunaga, Wang et al. — *Spider*, EMNLP 2018 | §6: Component Matching (F1 on exact **set** matching), Exact Matching as a conjunction over components, Execution Accuracy | `https://ar5iv.labs.arxiv.org/html/1809.08887` |
| 4 | Cai & Knight — *Smatch: an Evaluation Metric for Semantic Feature Structures*, ACL 2013, pp. 748–752 | the definition (max F1 over triples under a one-to-one variable mapping; NP-complete; ILP / hill-climbing), and the official implementation's `P = M/T`, `R = M/G`, `F = 2PR/(P+R)` | `https://aclanthology.org/P13-2131/` · `https://aclanthology.org/P13-2131.pdf` · `https://raw.githubusercontent.com/snowblink14/smatch/master/README.md` |
| 5 | Wang, Singh, Michael, Hill, Levy, Bowman — *GLUE*, ICLR 2019 | §3.4 verbatim: per-task scores plus a macro-average; **unweighted** average across metrics within a task | `https://ar5iv.labs.arxiv.org/html/1804.07461` |
| 6 | Liang, Bommasani, Lee, Tsipras, Soylu et al. — *Holistic Evaluation of Language Models*, 2022 | Figure 26 verbatim: head-to-head win rate as a fraction of pairwise comparisons **per metric across scenarios** | `https://ar5iv.labs.arxiv.org/html/2211.09110` |
| 7 | Perez, Lewis, Yih, Cho, Kiela — *Unsupervised Question Decomposition for Question Answering*, EMNLP 2020 | §5.1 Table 4: the four-column reference-free intrinsic panel (GPT-2 NLL, % well-formed, edit distance, length ratio), unblended | `https://ar5iv.labs.arxiv.org/html/2002.09758` |
| 8 | Zhang, Liu, Han, Chen, He et al. — *Visual Question Decomposition on Multimodal Large Language Models*, EMNLP 2024 | SubQuestRater's three criteria (Non-Repetition, Relevance, Groundedness), reported as three separate rows; GPT-4V as the scoring model | `https://ar5iv.labs.arxiv.org/html/2409.19339` |
| 9 | RAGAS (`explodinggradients/ragas`), industry RAG-evaluation framework | `src/ragas/evaluation.py` at `v0.0.16`/`v0.0.19`: `ragas_score` = unweighted harmonic mean of the component means (`len(values) / reciprocal_sum`); **absent** from `v0.1.0` and from `main` today | `https://raw.githubusercontent.com/explodinggradients/ragas/{v0.0.16,v0.0.19,v0.1.0,main}/src/ragas/evaluation.py` |

**Carried from the two earlier notes, not re-fetched here** (each records its own fetched URL and
verbatim quotation): MuSiQue (Trivedi et al., TACL 2022), HotpotQA (Yang et al., 2018), StrategyQA
(Geva et al., 2021), Hasson & Berant (\*SEM 2021), SARI (Xu et al., TACL 2016), ROSCOE (Golovneva et
al., ICLR 2023), ReCEval (Prasad et al., 2023), The Benchmark Lottery (Dehghani et al., 2021),
Dynaboard/Dynascore (Ma et al., 2021), Opitz & Burst (2019), Successive Prompting (Dua et al., 2022),
DecompRC (Min et al., 2019), Least-to-Most, Decomposed Prompting, Self-Ask, IRCoT, BERTScore, and the
official code of `allenai/break-evaluator`, `StonyBrookNLP/musique`, ROSCOE and `bert_score`.

**Recalled but not verified: none.** Two gaps, named: the Break **leaderboard page** did not resolve
from the compute box this session (HTTP `000`), so the four-metric set rests on the paper and the
official evaluator; and the arXiv searches run here were **metadata-only**, so "no intrinsic
decomposition metric reported on MuSiQue" remains **unestablished** rather than established.

---

## 7. What this note does not decide

1. **Whether any of P1–P4 becomes the thesis-primary metric** — issue #6 item 5, Jahid's with his
   supervisor. §2 specifies; it does not adopt.
2. **What happens to the published composites** — freeze-and-deprecate (§5 option 1) versus
   repair-and-re-score (§5 option 2). This decides whether the 33-cell sweep stays quotable.
3. **Whether the no-closed-model constraint extends to open-weight scorers.** This spec is built so the
   answer does not matter; if the answer is "it does not extend", the near-miss paraphrase axis
   (§4, last row) becomes addressable and this spec should be revisited.
4. **Whether a ~50-item human validation is worth Jahid's time.** Two independent precedents at n ≈ 50
   (Hasson & Berant; Successive Prompting) plus ROSCOE's Somers' D meta-evaluation, all carried. It
   remains the only action that would turn "is the metric biased?" from an argument into a measurement,
   and no metric in §2 has been validated against human judgment — **unmeasured**, in v1 and v2.
5. **Whether the ADR 0017 over/under asymmetry must live inside the headline number.** No metric in
   this survey expresses it; §2.1 term 6 carries it beside the headline instead, and inventing an
   asymmetric penalty is not this lane's to do.
