# What published multi-hop decomposition work actually scores, and what this repo could score instead

Literature survey plus a re-scoring diagnostic for the decomposition-quality metric, commissioned by
[ADR 0023](../adr/0023-jahid-2026-08-22-direction-metric-pipeline-completion-generalisation.md) item 1
("bring something better from the literature"). Context: the house composite's 0.2-weight
`reference_validity_micro` term is decided by 2 of 600 items ([issue #40](https://github.com/AhmadiJahid/Thesis---QAv2/issues/40)),
and PR #38's bias check already found it is the least testable metric this pipeline reports
([`composite-score-literature-check.md`](composite-score-literature-check.md) §4.9).

Statistics companion beside this note:
[`2026-08-22-metric-candidates.json`](2026-08-22-metric-candidates.json).

**This note surveys, measures and ranks. It does not choose.** The primary-metric decision is
[issue #6](https://github.com/AhmadiJahid/Thesis---QAv2/issues/6) item 5 — Jahid's with his supervisor.
§6 is the explicit list of what is theirs to settle.

---

## 0. Read this before quoting any number

- **Nothing in this note is a house metric.** Every candidate score below was computed *in this note*,
  from the per-item outputs of committed runs, as analysis. None appears in any `eval_metrics.json`,
  none is in `configs/musique_eval.json`, and **no experiment-log entry or claim rests on any of
  them.** Where a candidate disagrees with a published number, the published number stands as written
  and the disagreement is the finding.
- **Base commit** `efb8530`. **Eval set:** the ADR 0007 pinned 600 MuSiQue dev questions, 200 per gold
  hop depth — the *same* set for all nine arms, so every comparison below is same-eval-set.
- **Inputs:** the nine evaluator per-item files under `runs/` for exp-004 (3 conditions), exp-005 (3),
  exp-008 (1) and exp-009 (2). `runs/` is git-ignored, so the companion pins each input by
  sha256/size/mtime (`provenance.inputs_pinned_by_content`). Verified before scoring: all nine carry
  **identical `item_id` sets (600)**, identical `question` and identical `gold_steps` per item.
- **The statistics harness is validated, not asserted.** The paired bootstrap (seed 42, 10000
  iterations, chunk 1000, items in sorted `item_id` order), the paired t-test and the exact McNemar
  used for every candidate below reproduce the committed
  `experiments/exp-008/metrics.json` values **bit-identically** on the three metrics that exist in
  both: `step_f1` +0.137239501239501 CI [0.115019706219706, 0.160082440476191],
  `ordered_step_accuracy` +0.143309523809524 CI [0.120483730158730, 0.166559821428571],
  `rouge_l_f1` +0.121808572549593 CI [0.106509791775195, 0.136882915563767]; McNemar
  `exact_match` b=41 c=13 p=1.7513e-4 and `hop_count_exact_match` b=178 c=57 p=1.1151e-15
  (companion `harness_validation`). So the candidate intervals below are produced by a harness that
  reproduces this repo's own published intervals exactly.
- **Every literature claim here was fetched this session** — and unlike PR #38, which verified
  *papers*, this note verified *official metric code* line by line where it exists (§1, §7). Anything
  not verified is marked as such.
- **Binding constraint, unchanged:** no closed commercial model may score, rate or judge decomposition
  quality (CLAUDE.md standing constraint). One shortlist row is an open-weight model-based scorer; it
  is flagged as a **supervisor question**, not assumed admissible.

---

## 1. What the field reports — verified against official code, not recalled

PR #38 §3 established the pattern from the *papers*. Five things this pass adds by reading the
**official evaluation code**:

**1. MuSiQue's own repository contains no decomposition-quality metric, and its own decomposer is
selected on BLEU.** `evaluate_v1.0.py` in `StonyBrookNLP/musique` computes exactly `answer_f1`,
`answer_em`, `support_f1`, `group_answer_sufficiency_f1`, `group_support_sufficiency_f1` — nothing
about decompositions. The repo *does* ship a decomposer (`question_translator` reader + a
`facebook/bart-large` model, `experiment_configs/execution_model_decomposer_for_musique_ans_and_full.jsonnet`),
and its model-selection metric is a single line: `"validation_metric": "+BLEU"`. That is the only
decomposition-quality signal anywhere in the official MuSiQue codebase, and it is a *training*
validation metric, not a reported result.

**2. The Break leaderboard's official metric set is exactly four scores, all per-example.**
`allenai/break-evaluator` (`scripts/evaluate_predictions.py`) computes `exact_match`, `sari`, `ged`,
`normalized_exact_match`, renames them `EM / SARI / GED / norm_EM` for the leaderboard, and averages
per-example lists with `np.mean`. Two details matter for this repo:

- `format_qdmr` rewrites **bare `#k`** into `@@k@@`. The reference syntax in the field's flagship
  decomposition benchmark is bare `#k` — the same syntax MuSiQue's gold uses and the syntax this
  repo's `_REF_RX = r"\[#(\d+)\]"` does *not* match. That is external evidence on issue #40's open
  question 1, though whether v1 *intended* brackets is still only Jahid's to say.
- `get_exact_match` is `d.lower() == g.lower()` over the whole `@@SEP@@`-joined string — no punctuation
  stripping, unlike this repo's `_normalize_step`.

**3. `normalized_exact_match` is not portable to MuSiQue.** Its normalizer
(`evaluation/normal_form/normalized_graph_matcher.py`) is a stack of ~14 QDMR-*operation-specific*
rewrite rules (`FilterAdjectiveDecomposeRule`, `AggregateDecomposeRule`, `WrapperDecomposeRule`, …)
built on spaCy `en_core_web_sm` parses plus a 16-way operation classifier
(`scripts/qdmr_to_program.py`: FIND, SELECT, FILTER, PROJECT, AGGREGATE, GROUP, SUPERLATIVE,
COMPARATIVE, UNION, INTERSECTION, DISCARD, SORT, BOOLEAN, ARITHMETIC, COMPARISON, NONE). MuSiQue's
sub-questions are free-form natural language and `X >> relation` templates, not QDMR operations.
Adopting NormEM here would mean **writing a new MuSiQue canonicalizer**, not porting Break's — and its
validity would then have to be argued from scratch. PR #38 §5 option 4 costed this as "real
implementation work"; this is the specific reason.

**4. Prompted-decomposition work still scores decompositions by hand, on ~50 examples.** Successive
Prompting (Dua et al., 2022; fetched ar5iv 2212.04092) reports DROP F1 and, verbatim: *"To evaluate
the correctness of decomposed QA pairs, we manually analyze a subset of predictions on the dev set …
by randomly sampling 50 correct predictions"*. That is the second independent occurrence of **n≈50
human validation** in this literature (Hasson & Berant validated NormEM/LF-EM on 50 dev examples,
PR #38 §3.2). It converges with PR #38 §5 option 6.

**5. ROSCOE is entirely model-based.** Read off `projects/roscoe/score.py` in
`facebookresearch/ParlAI`: the embedding family needs `all-mpnet-base-v2` or
`princeton-nlp/sup-simcse-roberta-large` / `facebook/roscoe-512-roberta-base`; ROSCOE-LI needs
`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`; ROSCOE-LC needs a language model for
perplexity plus `cointegrated/roberta-large-cola-krishna2020` for grammar. All open weights, all far
under 8B — so ROSCOE is not excluded by the letter of the standing constraint, but it is a *model*
judging decomposition quality, and it also assumes a source context per step that this
decomposition-only pipeline does not feed it.

---

## 2. The shortlist

Judged against this repo's reality: does it give a **per-item value** (required for the ADR 0009
bootstrap/McNemar/t-test battery), is it **model-free**, does it run **offline and reproducibly**, and
can it **re-score the existing per-item files** (steps as strings with bare `#k`) so all nine
committed arms are re-scored rather than re-run?

| candidate | source (verified) | what it measures | per-item? | model-free? | verified against official code? | re-scores existing files? |
|---|---|---|---|---|---|---|
| **GED** (normalized graph edit distance) | Break official `evaluation/graph_matcher.py` + `decomposition.py`; metric named in Wolfson et al. 2020 | one distance over the whole plan: step wording (node substitution), chaining (edges from `#k`), and step count (node insert/delete) | **yes** | **yes** (with one deviation, §3.7) | **yes** — code fetched and reimplemented; official node cost lemmatizes with spaCy, this note did not | **yes** |
| **SARI** | Break official `evaluation/sari_hook.py` (T2T port of Xu et al., TACL 2016) | n-gram *edit* quality against the question: what was kept, added, deleted (n=1..4, keep-F1 + add-F1 + delete-precision)/3 | **yes** | **yes** | **yes** — reimplemented line by line | **yes** (needs the `question`, which the files carry) |
| **`structural_match`** | Break official `evaluation/sequence_matcher.py` `clean_structural` | alignment of the reference/`@@SEP@@` token skeleton only — chaining shape and step count, no wording | **yes** | **yes** | **yes** | **yes** |
| **Break EM** | Break official `get_exact_match` | whole-plan verbatim equality (lowercased, joined) | **yes** (binary → McNemar) | **yes** | **yes** | **yes** |
| **Per-item chaining validity, no free credit** | *not* from the literature — PR #38 §5 option 3(a) made concrete | did the prediction chain where the gold requires chaining, and are the references resolvable | **yes** | **yes** | n/a (house repair, not a published metric) | **yes** |
| **Canonicalized EM (NormEM / LF-EM style)** | Wolfson et al. 2020; Hasson & Berant 2021 (official metric of QDMR) | verbatim equality after a canonical form that neutralises benign reordering and wording | yes (binary) | depends on the canonicalizer (Break's uses spaCy) | **partially** — Break's normalizer read and found QDMR-specific (§1.3) | only after a new MuSiQue canonicalizer exists |
| **Answer EM / F1 (extrinsic)** | MuSiQue official `evaluate_v1.0.py` | whether the decomposition, executed, produces the right answer | yes | yes (string metrics) | **yes** | **no** — needs an answering run per condition (GPU) |
| **Open-weight embedding scorer** (BERTScore-style) | Zhang et al. 2020; official `bert_score` defaults to `roberta-large` (355M), layer 17 | soft token-level similarity, sees paraphrase where string metrics see a mismatch | yes | **no** | defaults read off official `bert_score/utils.py` | yes, but see below |
| **ROSCOE suite** | Golovneva et al. 2023; official `projects/roscoe/score.py` | reasoning-chain quality (semantic alignment, NLI-based, perplexity, grammar) | yes | **no** (4 model families, §1.5) | **yes** | partly — several scores need per-step source context this pipeline does not produce |
| **BLEU** | MuSiQue official decomposer config (`+BLEU`) | n-gram precision with brevity penalty | corpus-level BLEU: **no**; sentence-BLEU: yes but noisy | yes | **yes** (it is the config line) | yes |

**Flagged, not assumed:** the last three rows are model-based scorers. Open weights and ≤8B do not by
themselves make them admissible — "no closed commercial model may score decomposition quality" is a
supervisor decision whose *reasoning* (a model judging quality is not scientifically defensible) may
or may not extend to open models. **That extension is the supervisor's call, not this note's** (§6).

---

## 3. What each candidate says about this repo's own runs

All figures: n = 600, the pinned set, computed in this note (companion `per_system_macro`,
`comparisons`, `per_gold_hop`).

### 3.1 The nine arms and the null systems, side by side

GED is a **distance** — lower is better. Everything else: higher is better.

| system | GED ↓ | SARI ↑ | `structural` ↑ | chain-validity ↑ | Break EM ↑ | house composite ↑ |
|---|---|---|---|---|---|---|
| exp-008 `full_train` | **0.3094** | **0.6823** | **0.8822** | **0.9942** | **0.1017** | 0.5236 |
| exp-005 `oracle_guided` | 0.4236 | 0.6033 | 0.7778 | 0.8697 | 0.0600 | 0.4429 |
| exp-004 `oracle_guided` | 0.4414 | 0.5890 | 0.7808 | 0.9712 | 0.0500 | 0.4197 |
| exp-009 `oracle_hop_matched` | 0.4540 | 0.5879 | 0.7565 | 0.9686 | 0.0483 | 0.2633 |
| exp-009 `mixed` | 0.4712 | 0.5851 | 0.7421 | 0.9686 | 0.0533 | 0.2099 |
| exp-004 `unguided` | 0.4714 | 0.5850 | 0.7418 | 0.9686 | 0.0533 | 0.2098 |
| exp-004 `unguided_capped` | 0.4708 | 0.5849 | 0.7431 | 0.9686 | 0.0533 | 0.2128 |
| exp-005 `unguided` | 0.4910 | 0.5979 | 0.7246 | 0.8605 | 0.0517 | 0.4212 |
| exp-005 `unguided_capped` | 0.4909 | 0.5978 | 0.7251 | 0.8605 | 0.0517 | 0.4215 |
| — *empty prediction* | 1.0000 | 0.2911 | 0.0000 | 0.0000 | 0.0000 | *0.2000* |
| — *echo the question as one step* | 0.9011 | 0.4676 | 0.0000 | 0.0000 | 0.0000 | *0.2333* |
| — *one fixed junk step* | 0.9461 | 0.3229 | 0.0000 | 0.0000 | 0.0000 | *0.2333* |
| — *three fixed junk steps* | 1.0275 | 0.3361 | 0.5558 | 0.0000 | 0.0000 | *0.2778* |
| — *gold, order reversed* | 0.2875 | 0.9414 | 0.3446 | 0.2328 | 0.0000 | *0.7333* |
| — *gold itself* | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | — |

House-composite values for the seven arms are the committed ones (exp-008 note §1, exp-009 log row);
the italicised null-system composites are quoted from
[`2026-08-22-finetuned-vs-prompting-error-analysis.md`](2026-08-22-finetuned-vs-prompting-error-analysis.md)
§1.2 and were not recomputed here. The house composite of the gold itself is left blank because no
committed source states it and this note did not measure it.

**The discriminator: every model-free candidate ranks all four junk systems below every real arm; the
house composite does not.** Junk floors: GED ≥ 0.9011 against a real-arm worst of 0.4910; SARI
≤ 0.4676 against a real-arm worst of 0.5849; `structural_match` ≤ 0.5558 against 0.7246;
chain-validity 0.0000 against 0.8605. Under the house composite, three fixed junk steps score 0.2778
and exp-004 `unguided` scores 0.2098 — the failure PR #38 §4.3 predicted and issue #40 recorded.

### 3.2 Where the free credit for silence flips a ranking

Measured (companion `reference_emission`): the fine-tuned arm emits **1028** bare `#k` references with
**2 of 600** items chaining not at all; exp-005 `unguided` emits **1147** with **76 of 600** items
emitting no reference where the gold requires one. Under the house convention (an item with no
references scores 1.0) those 76 items are free 1.0s and the pooled micro rate puts **Qwen ahead**
(0.9817 vs Mistral's 0.9690). Under the no-free-credit per-item version the ranking **reverses**
(0.8605 vs 0.9686). Same predictions, same eval set, opposite ordering — the convention, not the
systems, decides it.

*Reconciliation with the prior note, stated because the numbers differ:* this note counts 1354 bare
refs for exp-004 `unguided` where
[`…error-analysis.json`](2026-08-22-finetuned-vs-prompting-error-analysis.json) `finding_9` counts
1352. The gap is **exactly** the 2 bracketed `[#1]` references of issue #40 (items
`3hop1__86965_488922_4093`, `4hop1__161765_53706_795904_580996`), which this note's `#(\d+)` regex
counts and that note's counter excluded. exp-008 agrees to the digit (1028 refs, 0.9961).

### 3.3 exp-008 (fine-tuning): every candidate agrees, and all of them are tighter than the composite

exp-008 `full_train` vs exp-004 `unguided`, same 600 items, paired bootstrap 10000 / seed 42 + t-test:

| metric | A | B | difference | 95% CI | verdict | t p |
|---|---|---|---|---|---|---|
| GED ↓ | 0.3094 | 0.4714 | **−0.1620** | [−0.1821, −0.1421] | **significant** | 1.2e-47 |
| SARI | 0.6823 | 0.5850 | **+0.0973** | [+0.0832, +0.1114] | **significant** | 6.9e-37 |
| `structural_match` | 0.8822 | 0.7418 | **+0.1403** | [+0.1210, +0.1601] | **significant** | 7.5e-38 |
| chain-validity | 0.9942 | 0.9686 | **+0.0256** | [+0.0133, +0.0387] | **significant** | 6.4e-05 |
| Break EM | 0.1017 | 0.0533 | +0.0483 | McNemar b=41 c=12 | **significant** p=8.2e-05 | — |
| *house* `step_f1` | 0.3411 | 0.2039 | +0.1372 | [+0.1150, +0.1601] | significant | 4.0e-29 |
| *house* composite | 0.5236 | 0.2098 | +0.3137 | [+0.1055, +0.3307] | significant, **bimodal/asymmetric** | no t-test possible |

**The exp-008 headline survives every candidate metric, in the same direction, at significance.** What
changes is only how defensible the interval is: the composite's CI is 0.225 wide and asymmetric
(issue #40 traced that to the reference term), while GED's is 0.040 wide and symmetric, SARI's 0.028.
Per gold hop, GED has the fine-tuned arm ahead at **every depth** — 0.1935 / 0.3352 / 0.3995 at hop
2/3/4 versus 0.3514 / 0.4900 / 0.5729 (companion `per_gold_hop`), which agrees with §4 of the exp-008
error-analysis note rather than adding a new claim.

### 3.4 exp-009 (hop-matched retrieval): no candidate rescues the null

exp-009 `oracle_hop_matched` vs `mixed`, same 600 items:

| metric | A | B | difference | 95% CI | verdict |
|---|---|---|---|---|---|
| GED ↓ | 0.4540 | 0.4712 | −0.0172 | [−0.0351, **+0.0003**] | not significant (t p=0.0555) |
| SARI | 0.5879 | 0.5851 | +0.0028 | [−0.0077, +0.0136] | not significant (t p=0.605) |
| `structural_match` | 0.7565 | 0.7421 | +0.0143 | [−0.0041, +0.0328] | not significant (t p=0.126) |
| chain-validity | 0.9686 | 0.9686 | +0.0000 | [−0.0143, +0.0144] | not significant (t p=0.999) |
| Break EM | 0.0483 | 0.0533 | −0.0050 | McNemar b=8 c=11 | not significant (p=0.648) |
| *house, from the log row* | | | | | `step_f1` ns, `ordered` ns, `hop_count_EM` **significant** |

**The exp-009 step-quality null holds under every candidate.** Nothing here would have changed that
run's reading; `hop_count_exact_match` remains the only metric that fires. GED comes closest — its
interval's upper end is +0.0003, i.e. it *just* fails — which is suggestive of a small real effect
that n=600 cannot resolve, not evidence of one. The MDE at this n for GED is 0.0252 against an
observed 0.0172 (companion `comparisons`), so this is a power statement, not an equality claim
(`docs/METRICS.md` §5.1).

### 3.5 The one place a metric swap would change a published reading — flagged, not acted on

exp-004 `oracle_guided` vs `unguided` (the Mistral guidance comparison, exp-004's log row reads *"helps
get the step count right … without a measured step-quality gain"*):

| metric | difference | 95% CI | verdict |
|---|---|---|---|
| *house* `step_f1` | +0.0022 | [−0.0085, +0.0129] | not significant |
| *house* `ordered_step_accuracy` | +0.0048 | [−0.0061, +0.0155] | not significant |
| **GED ↓** | **−0.0300** | **[−0.0436, −0.0166]** | **significant** (t p=1.5e-05) |
| **`structural_match`** | **+0.0390** | **[+0.0241, +0.0538]** | **significant** (t p=3.2e-07) |
| SARI | +0.0040 | [−0.0036, +0.0116] | not significant |
| chain-validity | +0.0026 | [−0.0112, +0.0170] | not significant |

Under GED or `structural_match`, oracle guidance on Mistral *is* a measured quality gain; under the
house step metrics it is not. The mechanism is visible in the table: both GED and `structural_match`
price step **count** and chaining shape, which guidance demonstrably improves (`hop_count_EM` 0.5083 →
0.5900, McNemar p=2.4e-4), whereas `step_f1`/`ordered_step_accuracy` do not price count except through
`ordered`'s denominator. **This is the strongest argument in the note for taking the metric decision
deliberately: it changes a conclusion already written into the log.** The log row stands as written;
this is a diagnostic.

### 3.6 What is redundant, at item level

Spearman ρ over per-item values pooled across the nine real arms (n = 5400, companion
`spearman_per_item_pooled_over_nine_real_arms`):

- `step_f1` vs `ordered_step_accuracy`: **ρ = 0.947**. At item level these are very nearly the same
  measurement — which is why giving them 0.4 and 0.3 in a blend spends 70% of the weight on one signal.
- GED vs the house panel: −0.669 (`step_f1`), −0.689 (`ordered`), −0.784 (`rouge_l_f1`), −0.620
  (`hop_count_EM`), −0.430 (`exact_match`). **GED is the only single candidate substantially
  correlated with every house metric** — consistent with it being a summary of the panel rather than a
  new axis.
- chain-validity vs everything: 0.18–0.25. It is **nearly independent** of the rest; it measures
  something no other metric here sees.
- SARI vs `rouge_l_f1`: 0.756 — largely the same surface-overlap axis this repo already reports.

### 3.7 Order-blindness, cost, and the failure modes each candidate carries

**Order.** On the *content-perfect, order-reversed* probe (gold steps reversed — a degenerate case, not
a junk null): `step_f1` **1.0000** (set-based, wholly blind), SARI 0.9414 (n-gram bag, nearly blind),
house composite 0.7333, **GED 0.2875 — better than every real arm's 0.31–0.49**,
`structural_match` 0.3446, chain-validity 0.2328, `ordered_step_accuracy` 0.1111, EM 0.0000. So GED
does penalise reversal but still prices a maximally mis-ordered plan ahead of every real system, and
SARI barely sees it at all. **Whatever is chosen, an order-sensitive term has to survive in the
reported set.**

**Gold structure, which bounds what canonicalization can buy.** Measured on this gold: every
decomposition is a tree with exactly n−1 references (shapes 2/1, 3/2, 4/3 at 200 items each);
**460 of 600 are strict linear chains** (step i references only step i−1) and **140 are not** (146/54
at hop 3, 114/86 at hop 4). Order canonicalization of the LF-EM kind therefore has something to do on
140 items and nothing to do on the other 460, where the reference chain already fixes a unique order.

**Cost.** The cheap pass — SARI + `structural_match` + Break EM + chain-validity over 15 systems ×
600 items — took **3.7 s** on one CPU core with no model. The GED pass took **261 s** for the nine real
arms, but with a vicious tail: two arms cost ~124 s each and the rest ~2 s, because exp-004 `unguided`
and exp-009 `mixed` both contain the same 39-step runaway item whose optimizer takes ~115 s alone
(exp-009 `oracle_hop_matched` has a 51-step item). Break's own implementation carries
`@exit_after(180)` and **drops** examples that time out; this note used a 20 s post-yield budget and
recorded **0 truncations across all 15 systems**, so no value below is a truncated approximation. An
implementation has to state a budget policy, and dropping items would break the paired battery.

**Named deviations from official code, so nothing here is mistaken for a leaderboard number.**
(a) GED's official node substitution cost lemmatizes with spaCy `en_core_web_sm`; spaCy is not in this
repo's `.venv`, so this note used lowercased whitespace tokens. Absolute GED values are therefore
**not comparable to Break leaderboard GED**; within this note every system is scored identically, so
the comparisons are valid. (b) Break's `clean_structural` keeps *every* token starting with `@@`,
which on its own `to_string()` output includes the `@@SEP@@` separators — so the official
`structural_match` is part chaining-shape and part step-count. That is why three junk steps score
0.5558 on it. (c) SARI here scores the `@@SEP@@`-joined string against the question, exactly as Break
does; the shared boilerplate (`@@SEP@@`, the `>>` template) inflates its keep/add terms, which is why
its floor on this data is 0.29 rather than 0. **Absolute SARI levels on this data are not
interpretable; differences on the same data are.**

---

## 4. Ranked recommendation

Ranked by **defensibility gained per unit of work**, with the expected direction and the reason. **A
list, not a choice.**

1. **Adopt Break's `GED` + `SARI` + `EM` triple as the intrinsic set, reported side by side, and
   re-score all nine committed arms.** *Direction:* the largest single gain available. *Why:* it is the
   **official metric set of the field's flagship decomposition benchmark**, verified from its
   leaderboard evaluator, not from a paper's prose (§1.2); every one of the three is **per-item**, so
   the full ADR 0009 battery applies (the composite takes only 1 of the protocol's 3 tests —
   PR #38 §4.9); all three are **model-free and offline**; all three rank every junk system below every
   real arm (§3.1); and re-scoring is CPU-only from files already on disk (§3.7). *What it costs:* GED
   needs a per-item time budget policy and a documented substitution cost (§3.7), and it invalidates
   comparability with every v1/v2 composite — the pool sweep included. *Risk stated plainly:* GED is
   order-light (§3.7) and, without spaCy, not numerically comparable to Break's own published GEDs.
2. **Keep one order-sensitive and one directional-length metric in the reported set regardless of
   which candidate leads.** *Direction:* prevents a known blind spot from becoming the headline.
   *Why:* measured — a reversed gold scores `step_f1` 1.0000, SARI 0.9414, GED 0.2875 (better than any
   real arm), and only `ordered_step_accuracy`/EM/`structural_match` see it (§3.7); and ADR 0017's
   supervisor asymmetry (over-decomposition tolerable, under-decomposition not) is expressible by
   *none* of the candidates — GED and `structural_match` price count symmetrically, exactly the defect
   PR #38 §4.6 found in the composite's length term. *Cost:* zero — those metrics are already computed.
3. **Repair the chaining term rather than deleting it: per-item, bare `#k`, no free credit for
   silence.** *Direction:* turns the term that is currently decided by 2 items into a real
   measurement that is **nearly independent of everything else reported** (ρ 0.18–0.25, §3.6) and gives
   it a per-item value. *Why:* §3.2 — the free-credit convention flips the Mistral/Qwen chaining
   ranking, and 76 of 600 Qwen items are currently paid 1.0 for chaining not at all. *Cost:* a metric
   definition change to shared pipeline code (Gate 1) and every published composite moves. *Not from
   the literature:* this is PR #38 §5 option 3(a) made concrete; the literature-grounded alternative
   for the same axis is Break's `structural_match`, which bundles count with structure (§3.7).
4. **Report answer EM / F1 from the answering backend beside the intrinsic set.** *Direction:* the
   only option that makes a number **comparable to published work** — it is MuSiQue's own official
   metric (§1.1) and what every prompted-decomposition paper reports. *Why:* PR #38 §5 option 1, and
   ReCEval's argument for keeping an intrinsic metric alongside it rather than instead of it. *Cost:*
   one GPU answering run per condition, so it cannot re-score existing outputs — the one candidate
   here that is a *run*, not a re-score. Ranked below 1–3 for that reason alone, not on merit.
5. **Drop the blended composite as a headline; if a single number is wanted, use one measured
   per-item metric rather than a hand-weighted sum.** *Direction:* removes every §3.1/§3.2 pathology
   that comes from *combining*. *Why:* in this pass, as in PR #38's, **no published work uses a
   hand-weighted linear blend of decomposition sub-metrics as its primary score**, and the two house
   metrics carrying 70% of the weight are ρ=0.947 duplicates (§3.6). *Cost:* rhetorical — five numbers
   read weaker in an abstract than one, and a 33-cell sweep needs a stated objective.
6. **Validate whatever is chosen against Jahid's own judgment on ~50 items.** *Direction:* the
   cheapest action that answers the supervisor's actual question ("maybe it's biased"), because it
   converts an argument about construction into a measurement. *Why:* now **two** independent
   precedents at n≈50 — Hasson & Berant's NormEM/LF-EM validation and Successive Prompting's manual
   analysis (§1.4) — plus ROSCOE's Somers' D meta-evaluation. *Cost:* human annotation time. *Note:*
   the annotation must be human; a closed commercial model doing it is the excluded method.
7. **Open-weight model-based scoring (BERTScore-style, or ROSCOE) — hold pending the supervisor.**
   *Direction:* would see the failure mode §6 of the exp-008 note measured most of (near-misses that
   are surface form: 128/600 items one step short of EM, median character similarity 0.537). *Why it
   is ranked last:* not on capability but on **admissibility** — it is a model judging decomposition
   quality, and whether the supervisor's rejection of that extends past closed commercial models is
   his call (§6). Also: `roberta-large`-based BERTScore would add a model dependency to an evaluation
   path that is currently entirely string-level, which `docs/METRICS.md` states as a property.

**Not recommended, and stated plainly:** keeping the composite as the thesis-primary metric.
§3.1 is the reason in one line — three fixed junk steps (0.2778) outrank exp-004's deployable baseline
(0.2098) on its own eval set, and no candidate in §2 has that property.

---

## 5. What this would require in the evaluator — implementation altitude, not code

Stated so the implementation brief can be written by someone else. **No code is proposed here and none
was written; `scripts/`, `configs/` and `experiments/` were not touched by this note.**

1. **New per-item columns, not a new script.** Every §2 candidate that re-scores is a function of
   `pred_steps`, `gold_steps` and `question` — all three already in `<prefix>_per_item.json`. The
   natural shape is new per-item fields plus their macro aggregates in the same `_aggregate` pass, so
   `--compare` picks them up for free.
2. **`BOOTSTRAP_STATISTICS` / `T_TEST_STATISTICS` / `MCNEMAR_STATISTICS` need the new names added.**
   Today those tuples are closed and `_statistic_arrays` hard-codes its columns. Any candidate that is
   not registered gets no interval, which is how the composite ended up with 1 of 3 tests.
3. **A re-score path over existing per-item files.** The cheapest route to all nine arms is to score
   from the stored `pred_steps`/`gold_steps` rather than re-reading `results.json`, which also removes
   the question-text join. This is the difference between a CPU minute and nine GPU runs.
4. **GED needs three policy decisions before it is implementable:** the node substitution cost (with
   or without a lemmatizer, i.e. with or without a spaCy dependency); the per-item time budget and
   what happens on exhaustion (Break *drops* the item — which would break the paired battery, so a
   fallback that preserves pairing is needed); and the direction convention, since it is a distance and
   every other metric in the report is a score. `networkx` 3.6.1 is already in `.venv`; spaCy is not.
5. **`_REF_RX`'s syntax is a prerequisite for item 3, not a side issue.** A per-item chaining metric
   has to know whether `[#k]` or `#k` is the reference syntax. Issue #40 open question 1.
6. **Directionality has no home in any candidate.** ADR 0017's over/under asymmetry is expressible
   only in the directional family this repo already reports separately; if it must be inside the
   headline metric, that is a new definition with no precedent found in this survey — and inventing
   one is exactly what "no PRD" forbids an agent to settle.
7. **Every published composite becomes non-comparable** the moment a term changes. The pool sweep,
   both v1 re-analysis notes and the exp-004/005/008/009 rows all quote `composite_score`. A migration
   convention (re-score and report both, or retire the metric with a stated cut-off) is a decision, not
   an implementation detail.
8. **Verification scope, per CLAUDE.md cost discipline:** a metric definition is shared pipeline code
   → Gate 1 review plus a smoke-test stage; the hand-computed golden values in
   `tests/test_decomposition_metrics.py` are the existing pattern for pinning a new metric's arithmetic.

---

## 6. Decisions left to Jahid and his supervisor

1. **Is `_REF_RX`'s bracketed `[#k]` a defect or an intended v1-carried definition?** (Issue #40 open
   question 1.) Break's official evaluator uses bare `#k`; that is evidence, not a decision.
2. **Does the "no closed commercial model may judge decomposition quality" constraint extend to
   open-weight scorers** (BERTScore with `roberta-large`, ROSCOE's DeBERTa/SimCSE stack)? Everything
   in §2's last three rows depends on this answer, and this note does not assume it either way.
3. **Which candidate leads, and does anything remain a single headline number** — issue #6 item 5,
   already deferred. §4 ranks; it does not choose.
4. **What happens to the published composites** if a term changes: re-score and report both, or retire
   the metric from a stated date. This decides whether the pool sweep's 33 cells stay quotable.
5. **Is a GPU answering run per condition worth spending** for the extrinsic metric (§4 item 4), given
   the end-of-October ceiling and the contention recorded in `docs/compute.md`.
6. **Is the ~50-item human validation (§4 item 6) worth Jahid's own time**, and if so on which
   comparison.
7. **Does the ADR 0017 over/under asymmetry have to live inside the headline metric**, or is reporting
   the directional family beside it sufficient? No candidate surveyed expresses it.

---

## 7. Citations — verified this session

**Official metric code, fetched and read (new to this note).**

| # | Artifact | URL |
|---|---|---|
| 1 | Break official evaluator — driver, metric set, `format_qdmr` (`#k` → `@@k@@`), `get_exact_match` | `https://raw.githubusercontent.com/allenai/break-evaluator/master/scripts/evaluate_predictions.py` |
| 2 | Break official evaluator — README, leaderboard metric list, NormEM description | `https://raw.githubusercontent.com/allenai/break-evaluator/master/README.md` |
| 3 | Break official SARI (T2T port; `max_gram_size=4`, deletion β=0) | `.../break-evaluator/master/evaluation/sari_hook.py` |
| 4 | Break official GED — `normalized_graph_edit_distance`, node substitution cost, `@exit_after(180)` | `.../break-evaluator/master/evaluation/graph_matcher.py` |
| 5 | Break official graph construction from `@@k@@` references | `.../break-evaluator/master/evaluation/decomposition.py` |
| 6 | Break official `SequenceMatchScorer` — spaCy lemmatization, `clean_structural` | `.../break-evaluator/master/evaluation/sequence_matcher.py` |
| 7 | Break official NormEM normalizer — the ~14 QDMR-specific rules, spaCy dependency | `.../break-evaluator/master/evaluation/normal_form/normalized_graph_matcher.py` |
| 8 | Break official QDMR operation vocabulary (16 operations) | `.../break-evaluator/master/scripts/qdmr_to_program.py` |
| 9 | Break evaluator requirements (spaCy 2.1.9, networkx 2.4, `edit-distance`) | `.../break-evaluator/master/requirements.txt` |
| 10 | `edit_distance.SequenceMatcher.ratio()` = `2*matches/(len(a)+len(b))`, and the DP that counts matches | `https://raw.githubusercontent.com/belambert/edit-distance/main/edit_distance/edit_distance.py` |
| 11 | MuSiQue official evaluator — `answer_em/f1`, `support_f1`, group sufficiency; **no decomposition metric** | `https://raw.githubusercontent.com/StonyBrookNLP/musique/main/evaluate_v1.0.py` |
| 12 | MuSiQue official decomposer config — `"validation_metric": "+BLEU"`, `facebook/bart-large` | `.../musique/main/experiment_configs/execution_model_decomposer_for_musique_ans_and_full.jsonnet` |
| 13 | ROSCOE official scorer — the four model families and their checkpoints | `https://raw.githubusercontent.com/facebookresearch/ParlAI/main/projects/roscoe/score.py` |
| 14 | `bert_score` official defaults — English → `roberta-large`, layer 17 | `https://raw.githubusercontent.com/Tiiiger/bert_score/master/bert_score/utils.py` |

**Papers fetched this session (new to this note).** Successive Prompting (Dua et al., 2022) —
`https://arxiv.org/abs/2212.04092` and `https://ar5iv.labs.arxiv.org/html/2212.04092`, full text read
for its evaluation paragraph; DecompRC (Min et al., 2019) — `https://arxiv.org/abs/1906.02916`,
abstract only, which states the extrinsic evaluation ("sub-questions that are as effective as
human-authored sub-questions") and **not** an intrinsic metric; BERTScore (Zhang et al., 2020) —
`https://arxiv.org/abs/1904.09675`, abstract, via the arXiv Atom API.

**Carried from PR #38's verified set, not re-fetched here** (that note records the URLs and the
verbatim quotations): MuSiQue, HotpotQA, StrategyQA, Break/QDMR, Hasson & Berant, SARI (Xu et al.),
ROSCOE, ReCEval, HELM, The Benchmark Lottery, Dynaboard, Least-to-Most, Decomposed Prompting,
Self-Ask, IRCoT, Opitz & Burst.

**Recalled but not verified: none.** Two gaps named so a later pass can close them: the Break
*leaderboard page's* own current metric list (this note read the evaluator that feeds it, which is
stronger, but not the page); and whether any published work reports an intrinsic decomposition metric
**on MuSiQue specifically** — nothing found in this pass, and absence from a keyword-level search is
weak evidence, so treat it as **unestablished** rather than as "no such work exists".

---

## 8. Reproducing §3

The harness is session-local and uncommitted, so — following the convention of the three earlier notes
in this directory (ADR 0020) — the verifiable artifact is
[`2026-08-22-metric-candidates.json`](2026-08-22-metric-candidates.json). It records the base commit,
the sha256/size/mtime of all nine per-item inputs, the alignment checks, each candidate's definition
*and its named deviation from the official code*, every per-system value, every comparison with its
interval and p-value, the item-level correlation matrix, the per-hop breakdown, and the cost figures.
The harness's own validation is §0's bit-identical reproduction of exp-008's committed intervals and
McNemar counts. Deriving §3 from committed code would require the evaluator to compute metrics it does
not have and to accept synthetic predictions built from gold — both pipeline changes, and therefore not
this lane's to make.
