# 0029. The Decomposition-Quality Suite: Six Unblended Terms as the Reported Primary, and the Composite Frozen as Legacy

- **Status**: Accepted (implementer registration, pending Jahid and his supervisor)
- **Date**: 2026-08-23
- **Amends** [0026](./0026-break-faithful-metrics-the-implementers-conventions.md) — the four
  columns that ADR landed as *additive* are now four of the six terms of the **reported
  primary**; nothing about how they are computed changes.

## Context

[ADR 0023](./0023-jahid-2026-08-22-direction-metric-pipeline-completion-generalisation.md)
item 1 ("bring something better from the literature") and
[ADR 0028](./0028-jahid-2026-08-23-delegations-pool-choice-router-call-composite-authorized.md)
item 3 authorized replacing or repairing the decomposition-quality composite; Jahid
re-confirmed it in session on 2026-08-23. The defect is
[issue #40](https://github.com/AhmadiJahid/Thesis---QAv2/issues/40): the composite's
0.2-weight `reference_validity_micro` term is a **micro-pooled rate whose denominator is
produced by the model**, and its `_REF_RX` matches only bracketed `[#k]` while MuSiQue's gold
writes bare `#k` — so on five of seven committed arms the denominator was empty and the term
scored 1.0 by rule, and on the other two, 2 of 600 items owned the whole denominator. Under
that composite, three fixed junk steps scored 0.2778 and a question echo 0.2333 against
exp-004 `unguided`'s 0.2098: **junk outranked the deployable baseline on its own evaluation
set**, and it took two analysis notes to notice.

The analyst lane then surveyed the literature and produced the specification this record
implements:
[`docs/analysis/2026-08-23-decomposition-quality-metric-specification.md`](../analysis/2026-08-23-decomposition-quality-metric-specification.md).
Its headline finding is a negative one and it is the load-bearing fact here: **the
decomposition literature does not composite — it reports a suite.** Break reports four metrics
side by side (and prints a trivial `Copy` baseline in the same table), EntailmentBank four
dimensions × two aggregations, Spider three metrics, ONUS a four-column panel, SubQuestRater
three separate criteria. Where the field does use a single number it is **one member of the
family made stricter** (QDMR's NormEM) or **one strict column selected on** (EntailmentBank's
*Overall AllCorrect*) — never a blend over the family. No decomposition paper found in three
independent passes reports a weighted linear blend of sub-metrics.

This ADR records what the implementer built. It is **not** an adoption: whether any of it
becomes the *thesis*-primary metric is [issue #6](https://github.com/AhmadiJahid/Thesis---QAv2/issues/6)
item 5, Jahid's decision with his supervisor.

## Decision

**1. The reported primary is an unblended six-term suite, `decomposition_report_card`.** Six
terms, all macro means over every evaluated item, all reported together with their direction,
their range, their provenance and the `n` they were averaged over:

| # | metrics JSON key | per-item column | better | provenance |
|---|---|---|---|---|
| 1 | `break_exact_match_rate` | `break_exact_match` | ⇑ | literature (Break) |
| 2 | `sari_macro` | `sari` | ⇑ | literature (Break / Xu et al.) |
| 3 | `ged_macro` | `ged` | **⇓** | literature (Break) |
| 4 | `chain_validity_macro` | `chain_validity` | ⇑ | house repair (ADR 0026) |
| 5 | `hop_count_exact_match_rate` | `hop_count_exact_match` | ⇑ | house |
| 6a | `under_decomposition_rate` | `under_decomposition` | **⇓** | house (ADR 0017) |
| 6b | `over_decomposition_rate` | `over_decomposition` | **⇓** | house (ADR 0017) |

All six were already computed at the commit the specification was written against, so this is
**registration, direction metadata, guards and reporting — not new metric mathematics**, and
no committed number moves. A report quoting fewer than six is not reporting this metric; the
run note prints the whole card, because the note is what gets quoted into
`experiments/log.md`. Terms 6a and 6b are a **pair and are never summed** — ADR 0017 records
that over-decomposition is tolerable and under-decomposition is not, and one "wrong length"
figure erases exactly that.

**2. `ged_macro` is the single number where one is structurally required**, and it carries
three mandatory caveats every time it is printed. It is a *member* of the suite, not a
function over it — the shape both single-number precedents in the field have. The caveats,
emitted in the report card beside the value:

- **(a) lower is better** — it is a distance, where terms 1, 2, 4 and 5 are scores;
- **(b) it is order-light, and on a 2-step plan order-blind** — the reversed gold measured GED
  0.2875 on the pinned 600, *better than every real arm* — which is why terms 1, 5 and 6 must
  stay reported beside it;
- **(c) absolute values are not comparable to published Break GED** — this port has no spaCy
  lemmatizer in the node substitution cost (ADR 0026 item 3), and it scores against MuSiQue
  gold rather than QDMR annotations.

It was chosen on three measured properties (it is a published metric of the field's flagship
decomposition benchmark; it is the only single candidate substantially correlated with every
house metric; it ranks every junk system below every real arm) plus one structural one (it is
per-item, so it takes the whole ADR 0009 battery). `break_exact_match_rate` is the
literature-canonical alternative and is rejected for this job only because it floors at
0.048–0.102 across every committed arm and so cannot order a 33-cell sweep.

**3. `composite_score` and `_REF_RX` are FROZEN as legacy, byte-identical, and deliberately
not repaired.** They are still computed, still reported, and now stamped
`composite_score_status: "legacy"` in every metrics JSON, marked legacy in
`configs/musique_eval.json`, and marked legacy in `docs/METRICS.md` §4.

The reason to preserve, because it is the part most likely to be re-litigated: **even a
corrected regex leaves a micro-pooled rate with a model-dependent denominator inside an
aggregate-of-aggregates.** That is the defect *class*, not just its instance — fixing the
regex would convert a term that measures nothing into a term that measures the wrong shape,
while moving the value of a metric under an unchanged name. Freezing keeps every committed
number quotable: nine arms, exp-010's 33 sweep cells, both v1 re-analysis notes. **Frozen for
reproducibility, not because it is correct.** Retiring it outright is issue #6 item 5.

**4. `decomp_mean` is a blended contingency, gated by measurement, and it is not part of the
suite.** Per item, the unweighted mean of five [0,1] higher-is-better terms
(`break_exact_match`, `sari`, `1 − min(ged_clamp, ged)`, `chain_validity`,
`hop_count_exact_match`). Equal weights and **no weighted variant**: GLUE's unweighted average
is the only weighting form with a precedent, and an unequal weighting would need a
Dynascore-style unit conversion nothing in this repo estimates. Components and clamp live in
`configs/musique_eval.json` so a run snapshot records them.

It ships **only because its own gate passed empirically.** An additive blend containing SARI
hands junk a nonzero floor (SARI's floor on this data is 0.29 and Break's own published `Copy`
row scores 0.431), which is the same failure family as issue #40 — so a junk baseline
outranking a real arm would have **disqualified** the blend rather than prompting a weight
tweak. The composite's weights were never the disease; its free-credit term was, and a weight
tweak would have hidden that.

**5. The junk battery is a committed script, and its acceptance criteria are tests.**
`scripts/decomposition_junk_battery.py` + `configs/decomposition_junk_battery.json` build six
systems out of the gold column — J1 empty, J2 question echo (Break's own `Copy`), J3 one fixed
step, J4 three fixed steps, J5 gold reversed, J6 gold — score them with the evaluator's own
`score_item`, and print the report card for each beside every committed arm. Per ADR 0027 it
is a **tool, not an experiment**: it scores no system of the thesis, takes no
`experiments/log.md` entry, needs no GPU and takes no `runs/run.lock`, and it carries a tiny
mode (`--limit`, `--eval-set all-gold`, `--no-real-arms`) so the full path is smoke-testable.
A1–A6 are asserted over the fixture gold in
`tests/test_decomposition_metrics.py::TestJunkBattery`.

**6. Three guards ship with the suite**, each replacing an argument with a check:

- **Every headline term is asserted to be the macro mean of its per-item column** at the point
  of reporting (`_assert_suite_terms_are_macro_means`); a term that ever drifts aborts the run.
  `reference_validity_micro` and `composite_score` are named in the card as excluded, with the
  reason.
- **`chain_validity_gold_unchained_items`** counts the items whose *gold* emits no `#k` and
  which therefore score 1.0 on term 4 for free. It is 0 on the pinned 600 by construction, but
  a future gold could reintroduce free credit silently — which is exactly how issue #40
  happened.
- **Every term reports its `n_items`**, so "no term is decided by a handful of items" is
  auditable rather than argued.

## The measured junk battery

Printed by `scripts/decomposition_junk_battery.py` at commit `222862f` (working tree carrying
this change), seed 42, on the **ADR 0007 pinned 600** — all fifteen rows on the same 600
items, with each arm's per-item file checked to cover exactly that set. The nine real arms are
the ones exp-011 re-scored under the Break-faithful columns; nothing was re-scored here, and
`decomp_mean` and the two direction columns are derived from the columns already in those
files. Re-run the script rather than quoting this table from memory.

| system | break EM ⇑ | SARI ⇑ | GED ⇓ | chain ⇑ | hop EM ⇑ | under ⇓ | over ⇓ | decomp_mean ⇑ |
|---|---|---|---|---|---|---|---|---|
| J1 empty | 0.0000 | 0.2911 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0582 |
| J2 question echo (`Copy`) | 0.0000 | 0.4676 | 0.9011 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.1133 |
| J3 one fixed step | 0.0000 | 0.3141 | 0.9546 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0719 |
| J4 three fixed steps | 0.0000 | 0.3262 | 1.0421 | 0.0000 | 0.3333 | 0.3333 | 0.3333 | 0.1412 |
| J5 gold reversed | 0.0000 | 0.9414 | 0.2875 | 0.2328 | 1.0000 | 0.0000 | 0.0000 | 0.5774 |
| J6 gold | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 |
| exp004 unguided | 0.0533 | 0.5850 | 0.4715 | 0.9686 | 0.5083 | 0.2017 | 0.2900 | 0.5288 |
| exp004 oracle_guided | 0.0500 | 0.5890 | 0.4414 | 0.9712 | 0.5900 | 0.0617 | 0.3483 | 0.5517 |
| exp004 unguided_capped | 0.0533 | 0.5849 | 0.4708 | 0.9686 | 0.5083 | 0.2017 | 0.2900 | 0.5289 |
| exp005 unguided | 0.0517 | 0.5979 | 0.4910 | 0.8605 | 0.5200 | 0.1150 | 0.3650 | 0.5085 |
| exp005 oracle_guided | 0.0600 | 0.6033 | 0.4241 | 0.8697 | 0.8733 | 0.0333 | 0.0933 | 0.5965 |
| exp005 unguided_capped | 0.0517 | 0.5978 | 0.4909 | 0.8605 | 0.5200 | 0.1150 | 0.3650 | 0.5085 |
| exp008 full_train | 0.1017 | 0.6823 | 0.3094 | 0.9942 | 0.7100 | 0.2567 | 0.0333 | 0.6357 |
| exp009 mixed | 0.0533 | 0.5851 | 0.4713 | 0.9686 | 0.5150 | 0.2000 | 0.2850 | 0.5301 |
| exp009 oracle_hop_matched | 0.0483 | 0.5879 | 0.4540 | 0.9686 | 0.6000 | 0.1300 | 0.2700 | 0.5503 |

Verdicts as printed: **A1 PASS · A2 PASS · A3 ordering holds (junk max SARI 0.4676 on J2, real
min 0.5849 on exp004 `unguided_capped`, margin +0.1173), exempt from A2 by design · A4 PASS ·
A5 PASS.**

**The A4 gate, stated as the number it turns on:** the best junk baseline scores
`decomp_mean` **0.1412** (J4) against a real-arm worst of **0.5085** (exp005 `unguided` /
`unguided_capped`) — margin **+0.3673** — and every one of J1–J4 ranks below every one of the
nine arms. The blend is therefore **not disqualified** and ships as a contingency.

Two things this table is worth reading for beyond the gate. First, it **independently
reproduces** every junk figure the analyst's survey measured at commit `efb8530` from a
different implementation path — GED junk floor 0.9011, SARI empty 0.2911, reversed-gold GED
0.2875, chain-validity junk 0.0000, real-arm GED worst 0.4910, real-arm chain worst 0.8605,
SARI junk max 0.4676 versus real min 0.5849. (J3/J4's SARI differ from that note's 0.3229 /
0.3361 because the fixed step texts are this repo's, committed in the battery config; the
note's were never recorded.) Second, **J5 is the argument for the suite in one row**: a
content-perfect, maximally mis-ordered plan scores GED 0.2875 and SARI 0.9414 — better than
every real arm on both — and is caught only by `break_exact_match` (0.0000) and
`ordered_step_accuracy`. A single number cannot carry that; six can.

## Consequences

- **A run reports six numbers where it reported one.** That is the intended cost: six numbers
  read weaker in an abstract than one, and the ordering rule for a multi-cell sweep now has to
  be stated (`ged_macro`) rather than assumed.
- **Every headline term now takes the full ADR 0009 battery.** The 1-of-3-tests weakness is the
  legacy composite's alone. The comparison grew from 7 bootstrap / 3 McNemar / 9 t-tests to 8 /
  5 / 12 — the counts are in `tests_reported` and no correction for multiple comparisons is
  applied to any of them, exactly as before.
- **The ADR 0017 asymmetry became testable.** Until the two direction columns existed, the
  over/under rates were aggregate-only and no paired test could be run on either.
- **Old per-item files still compare.** The three new columns are computed on every run but are
  **not required** of a file being compared: nine committed arms and 33 sweep cells were scored
  before they existed, and requiring them would refuse those files instead of comparing what
  they carry. What is missing is named in `statistics_not_available_in_inputs`. Verified on a
  real 600-item paired comparison of two exp-011 arms: every statistic byte-identical, the
  three new columns reported as unavailable.
- **A re-score buys the new columns, and only those.** Any arm re-scored from its existing
  `results.json` gains `over_decomposition`, `under_decomposition`, `decomp_mean` and the
  report card; every pre-existing value, including `composite_score`, is unchanged. It is
  CPU-only, needs no GPU and no `run.lock` — the exp-011 precedent.
- **`_REF_RX` is now guarded by a test and a comment that both say why not to fix it.** A future
  contributor "correcting" it would silently invalidate 33 sweep cells and nine arms.
- **Nothing here is adopted.** The thesis-primary metric is issue #6 item 5. If the supervisor
  asks for one blended number, `decomp_mean` exists and has passed its gate; if he asks for one
  number full stop, it is `ged_macro` with its three caveats; the recommendation on the table is
  the suite.

## Alternatives considered

- **Repair the composite in place** (replace `reference_validity_micro` with
  `chain_validity_macro` at the same 0.2 weight and re-score everything). Rejected as the
  primary move: it keeps an aggregate-of-aggregates with no per-item value and one of three
  tests, it invalidates the comparability of every published composite, and it moves a
  published reading — the Qwen/Mistral chaining ranking *reverses* under the no-free-credit
  convention. It stays available as the specification's §5 option 2 if Jahid wants it.
- **Silently correct `_REF_RX`.** Rejected outright: it changes the value of a metric under an
  unchanged name and leaves the pooled-denominator defect intact.
- **A weighted blend with tuned weights.** Rejected: no decomposition paper found in three
  passes reports one, and measured on this repo's own metric the "typed beats uniform" ordering
  fails in ~27–28% of weightings on the 4-simplex. Equal weighting is the only weighting with a
  precedent, and `decomp_mean` uses it.
- **A per-item harmonic mean (the RAGAS form).** Rejected arithmetically: `break_exact_match` is
  0 on ~90–95% of items in every committed arm, and a harmonic mean containing a 0 is 0, so it
  would be identically 0 almost everywhere. Taken over aggregate column means instead, it
  reintroduces the aggregate-of-aggregates problem. (Recorded so it is not re-derived; note that
  the best-known industry composite of this shape was itself retired at RAGAS `v0.1.0`.)
- **`suite_win_rate` (the HELM-style, weight-free cross-arm ordering of the specification's
  §2.4).** Not built. It is cross-arm, so it belongs in a reporting script rather than in a
  scoring run, it has no per-item value and therefore supports no significance claim, and it
  applies HELM's *per metric across scenarios* form *across metrics*, which is the
  specification's own adaptation rather than a published one. Out of scope for this change; the
  specification's definition stands if a sweep ever needs it.
- **Registering `over_decomposition` as higher-is-better** (following §2.5 item 1 literally,
  which names only `under_decomposition` for the lower-is-better registry). Rejected: it would
  stamp every `over_decomposition` comparison row `higher_is_better` and make `favours` name the
  wrong system on it. §2.1 of the same specification prints the pair as "under ⇓, over ⇓ but
  tolerated", so both are registered as lower-is-better and the tolerance is carried by the two
  rates staying split — never by calling more over-decomposition better.
- **Making the suite membership a config knob.** Rejected: the membership is a metric
  definition, and a definition that can be edited per run is not a definition. It lives in
  `DECOMPOSITION_SUITE_TERMS` beside the code computing each term. Only the contingency blend's
  parameters — a genuine choice with no source — live in config.
- **Requiring the three new columns of every compared per-item file.** Rejected: it would refuse
  every committed artifact at `--compare` until a full re-score landed, including the in-flight
  exp-014 significance work, and gain nothing the `statistics_not_available_in_inputs` record
  does not already give.
