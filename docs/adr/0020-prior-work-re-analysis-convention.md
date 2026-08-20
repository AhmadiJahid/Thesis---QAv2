# 0020. Prior-Work Re-Analysis Convention

- **Status**: Accepted
- **Date**: 2026-08-20

Amends [0005](./0005-seed-before-sampling-v1-sampled-results-not-reproducible.md) (v1
results stay non-reproducible as *runs*; their surviving per-item artifacts may still be
re-analyzed) and [0011](./0011-comparison-artifact-conventions-and-the-significance-claim-floor.md)
(analysis notes reporting a statistics battery carry a machine-readable companion).

## Context

The first re-analysis of v1 artifacts under v2's statistical protocol
(`docs/analysis/2026-08-20-v1-masking-and-retrieval-significance.md`, PR #33, analyst lane
approved by Jahid 2026-08-20) established a method its Gate-1 review found sound but
unrecorded, and warned would read as settled in November unless written down. This is that
record. The method will recur (the note's own §7.6 anticipates follow-ups).

## Decision

v1 per-item artifacts **may** be re-analyzed with v2's committed statistical protocol
(ADR 0009 as amended), under all of the following:

1. **v1 stays read-only** — nothing in `/cta/users/fyilmaz/Thesis---QA` is modified.
2. **Inputs are pinned by content**: sha256 (+ mtime) of every input file is recorded in
   the note, since v1 artifacts carry no commit SHA.
3. **Alignment is stated**: how per-item rows of the compared runs were paired (key and
   order), because bootstrap CI digits are alignment-dependent in the 3rd–4th decimal.

   *Note 2026-08-20 (from `docs/analysis/2026-08-20-v1-pool-size-significance.md`, PR #35):*
   an id or question-text key is preferred, but where a v1 artifact has neither — no
   `item_id`, and question texts that are not unique — **positional alignment in v1 file
   order is the accepted fallback**, under both preconditions, both verified and reported:
   (a) the per-item sequences of *every* compared file are identical **including order**, and
   (b) for every matched row, the alignment field (normalized question) **and** the gold
   (`gold_steps`) are asserted equal between the two files. Without (a) and (b) positional
   alignment silently compares different items, so a note that uses it states that it did and
   states the check. (The pool-size note hit exactly this: one eval question appears twice
   with identical gold, so a text key is not unique across its 750 rows.)
4. **A machine-readable JSON companion** sits beside the note carrying every reported
   statistic and the inputs' hashes — statistics only, never dataset content.
5. **The no-SHA caveat leads the note**: results are citable *prior work*, never v2
   measurements; Gate 2 is not satisfied; no `experiments/log.md` entry exists because
   nothing was run.
6. **Findings are options, not decisions** — imperative recommendations are recast as
   conditionals; which baselines the thesis leans on stays Jahid's call with his
   supervisor.

## Consequences

v1's measured comparisons become statistically characterizable without pretending they are
v2 evidence. The one open gap is re-derivability from committed code: until a `--compare`
shim accepts v1-format per-item files (note §7.6(b), unscheduled), the JSON is the
verifiable artifact and independent recomputation is the check — as performed by the PR #33
review.

**Note added 2026-08-20 (PR #36, issue #30): the gap above is now partly closed, not
closed.** `scripts/musique_decompositions_evaluator.py --compare --v1-per-item` reads
v1-format per-item files, so **the ADR 0009 battery over a comparison's full item set is
re-derivable from committed code**: the four bootstrap statistics (`rouge_l_f1`, `step_f1`,
`ordered_step_accuracy`, `composite_score`) with their CIs, both exact McNemar tests, and
the five paired t-tests added by issue #30. Verified against the committed masking note's
JSON at seed 42 under its stated alignment: 43/43 values bit-identical on each of Task A
typed-vs-raw, Task A uniform-vs-raw and one Task B pair. (43 is the acceptance harness's
compared-field count per pair — the bootstrap, McNemar and t-test fields it checks; the
PR #36 review compared a broader per-pair field set, 56–57 fields including `dof` and the
`significant` flags, with the same all-equal result.)

**Five families of numbers in that note remain harness-only** — computed by a session-local
analyst harness, not reproducible by committed code, and still resting on the PR #33
review's independent recomputation:

1. the **bootstrap CIs for `exact_match` and `hop_count_exact_match`** (v2 bootstraps four
   statistics; those two get McNemar and a t-test instead, and widening
   `BOOTSTRAP_STATISTICS` would change the ADR 0009 protocol — Jahid's call, not the
   implementer's);
2. the **`composite_no_ref_renorm` diagnostic** (a reference-term-free composite constructed
   for that note, explicitly not a house metric);
3. the **`power` blocks** (sd of paired differences, minimum detectable effect, n needed);
4. the **`holm_bonferroni` adjusted p-values** (ADR 0009 reports uncorrected; correction
   stays post-hoc);
5. the **`per_gold_hop` strata** (the shim compares one item set per invocation; per-hop
   slices would need the caller to split the files).

**Condition 3 is enforced in code on that path, not merely reported.** The shim refuses to
compare unless the witness fields condition 3(b) names are present on every row of both files
— `question` **and** `gold_steps` under positional alignment, and `gold_steps` under
question-key alignment, where the question's equality holds by construction and so witnesses
nothing — and it refuses on any mismatch. The alignment used is recorded in the output, since
3's reason for existing (CI digits move with the row order) applies to a shim run exactly as
it does to a note.

The shim's other conventions — the explicit opt-in, the two alignments, the prior-work
provenance block — are documented in `docs/METRICS.md` §5, not here; this note records only
what changed about re-derivability.

**Note added 2026-08-20 (PR #37, issue #6): the verification-addendum format.** Re-checking a
claim that already cites one of these notes produces a recurring artifact, and this paragraph is
its shape, so the next one does not reinvent it. When the claim survives, its sentence is left
**untouched**; a dated note in the host file's own style (`**Label (date):** …`) goes beside it,
stating which quantities were re-derived from committed code and at which SHA, and labelling every
remaining quantity with the harness-only family above (1–5) that covers it — a "verified from
committed code" banner may not blanket numbers a harness produced. A verification that only
recomputes committed artifacts is not an experiment, so condition 5 applies unchanged: no
`experiments/log.md` entry, and the addendum is prior work, not v2 evidence. When the claim does
**not** survive, the sentence is corrected and the addendum says what changed — retiring a claim is
never a silent edit. First instance: `docs/prior-work.md` §4's cross-encoder exact-match claim,
which held at 0/15 under McNemar and the paired t-test from committed code at b0a9ce8.

**Note added 2026-08-20 (PR #38, issue #29): a sixth harness-only family — synthetic-prediction
probes.** The five families above are enumerated *for the masking note*; the enumeration is
per-note, and `docs/analysis/composite-score-literature-check.md` adds one the list does not
cover. Its §4 constructs **synthetic prediction sets from the `gold_steps` column** of a v1
per-item file and scores them against that same gold — an oracle, a reversed oracle, a
reference-stripped oracle, over- and under-decomposed oracles, an empty prediction, a fixed junk
step, an echo of the question, and the three real runs with every `[#k]` stripped or one invalid
`[#k]` injected. Two adjacent quantities in the same note share the family: the **weight-simplex
sweep** (the committed `_composite_score` evaluated at 286 and 1771 weightings that
`configs/musique_eval.json` does not hold) and the **term-contribution / headroom / exchange-rate
decompositions** of the composite.

6. the **synthetic-prediction probes** just described. They are harness-only for a structural
   reason, stated in that note's §7: the committed evaluator scores a *predictions file against
   gold*, so deriving them from committed code would require it to accept a synthetic predictions
   input constructed from the gold column — a shared-pipeline change, and therefore not an
   analyst lane's to make. `--compare --v1-per-item` does not help here, because these probes
   compare a *constructed* system against gold rather than two existing runs against each other.

What the probes do **not** rest on is faith in the harness: it imports the evaluator's own
`_step_prf`, `_ordered_step_accuracy`, `_reference_validity`, `_composite_score` and `_REF_RX`
rather than re-implementing any formula, and it is validated by reproducing v1's three published
`composite_score` values **bit-identically** (3/3, < 1e-9) before any probe is run — a
self-check any future note in this family should carry. Conditions 2, 4 and 5 apply unchanged:
the inputs are pinned by content in the note itself, the JSON companion carries every reported
number, and no `experiments/log.md` entry exists because nothing was run.

## Alternatives considered

- Forbidding any use of v1 numbers — loses real, per-item-verifiable evidence that directly
  informs open decisions (masking default, CE, primary metric).
- Re-running the comparisons in v2 — the right long-term answer where they matter, but GPU
  time is contended and several v1 questions (e.g. typed vs uniform) need larger n than a
  re-run would grant anyway.
