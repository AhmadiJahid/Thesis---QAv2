# 0026. Break-Faithful Metrics (EM / SARI / GED) and the Repaired Chain Validity: The Implementer's Conventions

- **Status**: Accepted (implementer design, pending Jahid)
- **Date**: 2026-08-22

## Context

[ADR 0023](./0023-jahid-2026-08-22-direction-metric-pipeline-completion-generalisation.md)
item 1 authorized work on the decomposition-quality metric ("bring something better from the
literature"), and [issue #40](https://github.com/AhmadiJahid/Thesis---QAv2/issues/40) recorded
the specific defect that prompted it: the composite's 0.2-weight `reference_validity_micro`
term is decided by 2 of 600 items, because `_REF_RX` matches only bracketed `[#k]` while the
eval set's gold uses bare `#k`.

The analyst's survey
([`docs/analysis/2026-08-22-metric-candidates.md`](../analysis/2026-08-22-metric-candidates.md))
verified the official metric code of the field's flagship decomposition benchmark, ranked the
candidates, and — deliberately — chose nothing. Its §5 named the implementation questions and
its §6 named the seven decisions that belong to Jahid and his supervisor.

**This record decides none of §6.** It records the *implementation* conventions of the
measurement, so that the November write-up does not mistake an implementer default for a
research choice, and so changing one is a visible edit rather than archaeology. Nothing here
is a measurement, and nothing here adopts a primary metric: **which metric leads, and what
happens to the published composites, remains issue #6 item 5.**

**What was actually implemented** (`scripts/musique_decompositions_evaluator.py`, additive):
four new per-item columns — `break_exact_match`, `sari`, `ged`, `chain_validity` — their macro
aggregates, their per-gold-hop breakdowns, and their registration in the ADR 0009 paired
battery. `composite_score` is untouched, and no metric that existed before this change moves
by a digit.

## Decision (the implementer's, in one line)

Port the official `allenai/break-evaluator` EM / SARI / GED semantics as **additive** per-item
columns, with three named deviations from the official code (no spaCy lemmatizer, no `norm_EM`,
no `format_qdmr` `return`-stripping) and a **pairing-preserving** GED budget policy that
reports a documented upper bound where Break drops the item.

## Implementation conventions (agent-authored)

1. **`norm_EM` is not ported.** Break's fourth leaderboard metric normalizes with ~14
   QDMR-*operation-specific* rewrite rules over spaCy parses plus a 16-way operation
   classifier. MuSiQue sub-questions are free-form natural language, so porting it means
   writing a **new** MuSiQue canonicalizer whose validity has to be argued from scratch
   (survey §1.3). Not a cheaper version of Break's metric — a different metric.

2. **References are read with `#(\d+)`, which is Break's own rule, and the old regex is left
   alone.** Break's `format_qdmr` rewrites `#(\d+)` → `@@\1@@` and its graph builder reads
   `@@(\d+)@@`; that regex matches the `#1` inside `[#1]` as well as a bare `#1`, so the new
   metrics see references in **both** syntaxes. `_REF_RX = r"\[#(\d+)\]"` and every metric
   built on it (`reference_validity_macro`, `reference_validity_micro`, and therefore
   `composite_score`) are **unchanged**. Issue #40's open question 1 — whether the bracketed
   syntax was a defect or an intended v1-carried definition — is *not* answered here; the new
   column sits beside the old one so both can be read.

3. **GED policy call 1 — the node substitution cost has no lemmatizer.** Break's cost is
   `1 - SequenceMatchScorer.get_match_score(label_a, label_b)`, whose "base" processing
   lemmatizes with spaCy `en_core_web_sm`. spaCy is not a dependency of this repo and its
   model requires a download, and the evaluator must stay runnable offline. The cost here is
   the same `edit_distance.SequenceMatcher.ratio()` = `2·matches / (len(a) + len(b))`, over
   **lowercased whitespace tokens** (the repo's existing `_tokenize`). The ratio's `matches`
   count is a faithful port of `belambert/edit-distance`'s DP, including its tie-break order
   (substitution, then insertion, then deletion) — reimplemented in ~20 lines rather than
   added as a dependency, because the number of matches on a *minimum-cost* path is not the
   LCS length and the difference is real (`AB` vs `BA`: 0 matches, LCS 1).
   **Consequence, stated plainly: absolute GED values here are NOT comparable to published
   Break leaderboard GED.** Within one evaluation every system is scored identically, so
   comparisons on this data are valid. This is the same deviation the survey note declared
   for its own §3 figures (§3.7a).

4. **GED policy call 2 — the budget preserves pairing; Break's does not.** Break wraps the
   optimizer in `@exit_after(180)` (a thread that raises `KeyboardInterrupt` in the main
   thread) and its driver turns a timeout into `None`, which `np.mean` then **drops**.
   Dropping is unavailable here: a dropped item has no pair, and every test in the ADR 0009
   battery (paired bootstrap, paired t-test, McNemar) rests on the pairing. So an item that
   cannot be optimized within its budget is **reported with a documented upper bound and
   flagged**, never dropped. Two guards, both in `configs/musique_eval.json` under
   `break_metrics.ged`:
   - **`max_nodes_for_optimizer` (default 16)** — the guard that bounds cost, and the **only**
     one that bounds it: it is checked before any search starts, and it is deterministic.
     Above it the optimizer is not called; the reported value is the cost of one concrete edit
     path (pair nodes in sorted id order, substitute each pair, delete/insert the surplus, keep
     the edges whose endpoints are both paired and whose image exists on the other side,
     delete/insert the rest), normalized the same way. A concrete edit path's cost is a valid
     **upper bound** on the edit distance, and it is computed in one pass, so it is the same
     number on every machine. The survey measured a single 39-step runaway prediction costing
     ~115 s of optimizer time by itself (§3.7).
   - **`per_item_time_budget_seconds` (default 20.0)** — a **backstop, not a timeout**. The
     deadline is only tested *between* the optimizer's successive approximations, so a single
     long-running approximation — including the first — cannot be interrupted; this is why the
     node cap and not the budget is what bounds cost. When it does fire the last approximation
     is kept (still a valid upper bound), and that value is **machine-dependent**: so the
     firing is recorded per item in `ged_fallback`, its elapsed seconds in
     `ged_fallback_seconds` (the budget can overshoot, and by how much is then visible),
     counted in `ged_fallback_counts`, and named in the run note. A run whose
     `ged_fallback_counts` is empty has no machine-dependent value in it at all.
   - **A third reason exists and is not a timeout:** `no_optimizer_result`, when networkx
     yields no approximation whatsoever for a pair of graphs. It was originally named
     `time_budget_no_yield`, which was wrong — it does not depend on the clock (PR #44 review,
     nit 5). The deterministic bound is reported for it too.
   - **Both knobs are validated at load** (`_ged_policy`), in the style of `gold_validation`:
     `max_nodes_for_optimizer` must be an integer ≥ 8 (below the capped decomposer arm's
     8-step budget the cap would silently change what `ged` measures on ordinary predictions)
     and the budget must be a finite positive number. Either failure aborts naming the config
     key.

   **The cap of 16 comes from measurement, and the measurement is now a committed script.**
   The first revision of this record set 30 on timings taken against the *2-hop* fixture gold,
   where everything is ≤ 0.06 s; the Gate-1 review measured the same implementation against the
   *4-hop* gold and found ~200× that, and the cap moved to 16. That second revision printed a
   table whose **shapes were named but not defined**, so the reviewer could not reproduce one of
   its headline rows (PR #44 review, residual finding 1). The table below is therefore not typed
   prose: it is the stdout of
   [`scripts/ged_cost_benchmark.py`](../../scripts/ged_cost_benchmark.py)
   (config [`configs/ged_cost_benchmark.json`](../../configs/ged_cost_benchmark.json)), which
   constructs every shape explicitly — exact step texts, exact reference pattern, pinned by
   `tests/test_ged_cost_benchmark.py` — and times the evaluator's own `_normalized_ged` with the
   node cap lifted, against the committed fixture gold.

   Measured at commit **`21a1203`**, on LittleGazor (AMD EPYC 7513, CPU only, networkx 3.6.1,
   Python 3.13.9), gold hop depth stated per column because the gold's size is part of the cost.
   The script's own legend travels with the rows (PR #45 review, nit 2), because a marked cell
   pasted without it is uninterpretable: **per-cell stop 300.0 s; `*` = the stop fired, so the
   time is a lower bound; `†` = networkx yielded no approximation, so the cell is not a timing.**
   No cell below carries a mark.

   | prediction shape | nodes | edges | vs 2-hop gold | vs 4-hop gold |
   |---|---|---|---|---|
   | `repeated_step_text` | 8 | 0 | 0.00 s | 0.00 s |
   | `repeated_step_text` | 12 | 0 | 0.00 s | 0.00 s |
   | `repeated_step_text` | 14 | 0 | 0.00 s | 0.00 s |
   | `repeated_step_text` | 16 | 0 | 0.00 s | 0.00 s |
   | `repeated_step_text` | 20 | 0 | 0.00 s | 0.00 s |
   | `repeated_step_text` | 30 | 0 | 0.00 s | 0.00 s |
   | `gold_step_texts_repeated` | 8 | 6 | 0.00 s | 0.03 s |
   | `gold_step_texts_repeated` | 12 | 9 | 0.01 s | 0.24 s |
   | `gold_step_texts_repeated` | 16 | 12 | 0.01 s | **0.97 s** |
   | `gold_step_texts_repeated` | 20 | 15 | 0.02 s | 2.87 s |
   | `gold_step_texts_repeated` | 30 | 22 | 0.04 s | 18.28 s |
   | `chain_shaped` | 12 | 11 | 0.00 s | 0.00 s |
   | `chain_shaped` | 39 | 38 | 0.01 s | 0.01 s |
   | `all_pairs_referencing` | 20 | 190 | 0.04 s | 0.01 s |
   | `all_pairs_referencing` | 30 | 435 | 0.19 s | 0.06 s |
   | `nonsense_text_repeated_no_reference` | 8 | 0 | 0.00 s | 0.00 s |
   | `nonsense_text_repeated_no_reference` | 16 | 0 | 0.00 s | 0.00 s |
   | `nonsense_text_repeated_no_reference` | 30 | 0 | 0.00 s | 0.00 s |
   | `nonsense_text_repeated_with_reference` | 8 | 8 | 0.00 s | 0.08 s |
   | `nonsense_text_repeated_with_reference` | 16 | 16 | 0.00 s | **1.01 s** |
   | `nonsense_text_repeated_with_reference` | 30 | 30 | 0.00 s | 10.10 s |
   | `gold_step_text_repeated_no_reference` | 8 | 0 | 0.00 s | 0.00 s |
   | `gold_step_text_repeated_no_reference` | 16 | 0 | 0.00 s | 0.00 s |
   | `gold_step_text_repeated_no_reference` | 30 | 0 | 0.00 s | 0.00 s |
   | `gold_step_text_repeated_with_reference` | 8 | 8 | 0.00 s | 0.08 s |
   | `gold_step_text_repeated_with_reference` | 16 | 16 | 0.00 s | **1.01 s** |
   | `gold_step_text_repeated_with_reference` | 30 | 30 | 0.00 s | 10.12 s |

   **Which of the previous revision's numbers survived re-measurement, and which did not.** The
   `gold_step_texts_repeated` row reproduced (0.03 / 0.24 / 0.97 / 2.87 / 18.28 s against
   0.04 / 0.24 / 0.98 / 3.05 / 20.05 s — the two largest cells differ by wall-clock load, not by
   arithmetic), and it is the row the Gate-1 reviewer independently reproduced (0.99 s @ 16,
   18.4 s @ 30). `all_pairs_referencing` reproduced exactly against the 4-hop gold (0.01 / 0.06 s)
   but its 30-node cell against the *2-hop* gold is 0.19 s, above the "≤ 0.06 s" the old table
   claimed for that column. The other two rows did **not** reproduce and their old numbers are
   deleted rather than kept: under an explicit construction, `repeated_step_text` (identical
   labels, no references) and `chain_shaped` cost ~0.00 s everywhere, against the 0.14–18.44 s
   and 0.07 / 1.27 s the old table showed. Nothing in this repo produced those numbers, so
   nothing here can defend them; what the shape-name prose meant is unrecoverable.

   **What drives the cost, read off the 2×2 the PR #45 review asked for.** The first attempt at
   this paragraph asserted that the expensive case is a prediction whose labels *near-tie against
   the gold's own labels* while carrying references. That was past what the table could carry —
   the only expensive shape in it was simultaneously gold-derived *and* edge-carrying — and the
   reviewer falsified the gold-similarity half with a two-line counter-construction. The last
   four rows above are the resulting factorial: {nonsense label, gold-derived label} ×
   {no reference, one reference}, with label ties held at maximum in all four, the nonsense pair
   sharing vocabulary and token count and differing in exactly one token. What they settle:

   - **Similarity to the gold's labels costs nothing.** Holding ties and references fixed, a
     label that shares **no word** with the gold and a label that **is** one of the gold's steps
     cost the same to a hundredth of a second: 0.08 / 1.01 / 10.10 s against 0.08 / 1.01 /
     10.12 s at 8 / 16 / 30 nodes. The earlier claim is **withdrawn**, not softened.
   - **Carrying references is necessary.** Toggle the reference off the same nonsense sentence
     and every size drops to 0.00 s. It is the reference, and nothing else about the text, that
     turns the cell on.
   - **Ties among the prediction's own steps are necessary too.** `chain_shaped` and
     `all_pairs_referencing` carry edges — up to 435 of them — but their labels are
     distinguishable, and they cost ≤ 0.19 s everywhere.
   - **Gold depth is the third factor.** Both referenced shapes cost 0.00 s against the 2-hop
     gold and ~10.1 s against the 4-hop gold at 30 nodes.
   - **Edge *count* is not a factor**, which is why an edge bound was considered and rejected:
     the densest graph measured (435 edges) is among the fastest, while a 30-edge star is the
     second most expensive shape in the table.

   So, stated as narrowly as the measurements allow: **cost rises with the prediction's step
   labels tying with *each other*, × the prediction carrying `#k` references at all, × the
   gold's depth — all three necessary in the measured set, and none of them the same thing as
   resembling the gold's text.** The mechanism behind the ~0.00 s cells is visible in the
   optimizer: with no edges and N identical labels networkx yields exactly one approximation and
   stops (checked directly: one yielded value at 16 nodes and at 30).

   The honest guard is therefore the node cap, set where the measured worst case is far inside
   the budget: **at 16 nodes the worst case measured is ~1.0 s against a 4-hop gold**, against
   ~18.6 s at 30 — an order of magnitude inside the 20 s budget, so the machine-dependent path
   is effectively unreachable under the cap. (The Gate-1 reviewer's own adversarial probing of
   five further shapes at 16 nodes found 1.11 s as its worst, the same order.) 16 is still twice
   the capped arm's 8-step budget and four times the deepest gold, so no plausible decomposition
   is near it.

   What the cap trades away is small and measured. The same script prints the reported bound
   beside the optimizer's own value just above the cap, against the 4-hop gold (same commit):

   | prediction shape | nodes | optimizer | reported bound | gap |
   |---|---|---|---|---|
   | `repeated_step_text` | 17 | 1.1450 | 1.1450 | +0.0000 |
   | `gold_step_texts_repeated` | 17 | 0.7586 | 0.7586 | +0.0000 |
   | `chain_shaped` | 17 | 0.8485 | 0.8485 | +0.0000 |
   | `chain_shaped` | 39 | 0.9351 | 0.9351 | +0.0000 |
   | `nonsense_text_repeated_with_reference` | 17 | 0.9622 | 1.0210 | +0.0588 |

   On four of these five shapes the bound is **tight** — equal to the optimizer's value to four
   decimals, including on the 39-step runaway. On the fifth, the star-shaped shape added after
   the PR #45 review, it is **0.0588 above** it: the bound is an upper bound, not an estimate,
   and the honest statement is second-decimal on a junk-by-construction item rather than "tight
   everywhere". (The revision before this one reported a 0.0071 gap on "the chain shape"; that
   shape is one of the two that did not reproduce, so its gap is not carried forward either.) A
   bound is never *below* the true distance, and `tests/test_ged_cost_benchmark.py` asserts that
   ordering across every shape at three sizes rather than trusting it. Neither guard fires on
   well-behaved predictions, and `ged_fallback_counts` says so per run.

   **What is and is not reproducible here.** The shapes, the graphs and the GED *values* are
   deterministic and reproduce on any machine; the *seconds* are wall clock on a shared box and
   will not. That asymmetry is the reason the cap is set an order of magnitude below the budget
   rather than at the edge of a timing.

   **Known blind spot, inherited and pinned by a test rather than hidden.** networkx prices a
   *self-loop* edge as substitutable with an ordinary edge, so a **2-step** decomposition
   predicted in reverse order — where step 1's `#1` becomes a self-loop — scores GED **0.0**,
   although the true minimum edit path costs 2 of a possible 3. A 3-step reversal is priced
   correctly (0.4 on the fixture). This is what the official implementation's own nx call does,
   so it is kept and documented rather than "fixed" (fixing it would be inventing a metric);
   `TestBreakMetrics` pins both numbers, and Break EM catches the reversal in both cases. It is
   a sharper instance of the order-lightness the survey already measured (§3.7).

5. **GED policy call 3 — direction is carried in the data, not in a convention.** GED is a
   **distance**: lower is better, and it can exceed 1.0. It is the only such metric this
   evaluator reports, so a bare `-0.16` on a `ged` row means the *opposite* of what it means
   on every other row. Therefore: `LOWER_IS_BETTER_STATISTICS` names it in source; every
   bootstrap / McNemar / t-test row in a comparison carries `direction`
   (`higher_is_better` / `lower_is_better`) and, when significant, `favours`
   (`system_a` / `system_b`) with the direction already applied; the run note's table has a
   `better` column and annotates each significant verdict with the system it favours. Argument
   order follows the official driver: **prediction first, gold second**, and the graph's edges
   run **from** the referencing step **to** the step it references (official `(i+1, ref)`).

6. **`chain_validity` is a house repair and is labelled as one.** Per item: 1.0 when the
   *gold* emits no reference, **0.0** when the gold chains and the prediction emits no
   reference at all, else valid / total bare `#k` references (valid = `1 <= k <` the step's own
   1-based index). The no-free-credit rule is the point: the survey measured the free-credit
   convention **flipping** the Mistral/Qwen chaining ranking, with 76 of 600 Qwen items paid
   1.0 for chaining not at all (§3.2). It is **not** from the literature (survey §2, §4 item
   3), it is reported beside the Break metrics rather than as one of them, and it does **not**
   enter `composite_score`.

7. **Two SARI caveats that travel with the number.** (a) The source is the *question* and the
   prediction/target are the `@@SEP@@`-joined decompositions, exactly as the official driver
   calls it — so the shared boilerplate (`@@SEP@@`, the gold's templates) inflates the keep and
   add terms: **absolute SARI levels on this data are not interpretable; differences on the
   same data are** (survey §3.7c). (b) The official hook supports several references and
   divides target counts by the number of non-empty ones; with exactly one reference — always
   the case here — the weighted counts equal the plain ones, so that layer is not reproduced.

8. **`format_qdmr`'s `return`-stripping and `;`-splitting are not applied.** The first
   (`re.sub(r'return', '', part)`) removes QDMR's keyword and would also eat the substring
   inside ordinary English words ("returned" → "ed"); the second splits QDMR's step separator,
   where this pipeline's steps are already split by `src/step_lines.py`. Whitespace collapsing
   and the reference rewrite **are** applied.

9. **The new metrics compute by default; there is no off switch.** They are additive columns,
   they cost ~milliseconds per item apart from GED's guarded tail, and a metric nobody computes
   is a metric nobody can check. Only GED's two cost guards are configurable.

10. **A v1 prior-work per-item file cannot carry these columns, and the comparison says so.**
    `--compare --v1-per-item` (ADR 0020) reads files written before these metrics existed. The
    new columns are **required** of a v2 per-item file (a stale one gets the existing "re-run
    the evaluation to regenerate them" refusal) and **optional** on the v1 path, where the
    battery covers the legacy statistics and the metrics JSON lists what it could not compute
    in `statistics_not_available_in_inputs`. Computing them from a v1 file's stored steps would
    be a *re-score of v1 output*, which is a different thing from comparing what v1 measured,
    and the survey's §5 item 3 re-score path is not implemented here.

## What was verified (and what was not)

Ran, in a throwaway worktree at this branch's tip, CPU only:

- `tests/test_decomposition_metrics.py` — the full module, including new hand-computed
  expected values for identical / reversed / over-long / empty / no-reference predictions and
  a `--compare` of a file with itself.
- `scripts/musique_decompositions_evaluator.py --predictions tests/fixtures/predictions/decomposer_results_musique.json`
  — the committed fixture, 4 evaluated rows, new columns present, existing golden values
  unchanged.
- `scripts/musique_decompositions_evaluator.py --compare` over two fixture-derived per-item
  files, and over one file with itself.
- `scripts/smoke_test.py` (`decomposition_metric_tests` + `musique_eval` stages).
- The GED cost table above, and the bound-versus-optimizer gap at the cap boundary — since the
  record-quality pass, by `scripts/ged_cost_benchmark.py` at the commit the table names, with
  `tests/test_ged_cost_benchmark.py` pinning each shape's construction and
  `--max-node-count 8` as its tiny smoke path.

After the Gate-1 review the following were added or corrected, and re-verified the same way:
the v2 per-item loader's finite-number gate now covers the issue #40 columns (a `null` there
used to die as a raw `TypeError` and a `NaN` to reach the run note); the cost record was
re-measured with the gold hop depth stated and the cap moved 30 → 16; the `time_budget`
wording no longer calls itself a hard timeout and its elapsed seconds are recorded;
`time_budget_no_yield` became `no_optimizer_result`; both knobs are validated at load; the
metric name is required for a t-test row; and the run note carries the "levels are not
comparable to published Break" caveat, because the run note is what gets quoted into
`experiments/log.md`.

The Gate-1 review also pinned six **record-quality** findings for the next evaluator-touching
pass, and this is what that pass did (all additive or prose; the evaluator was re-run on the
committed fixture at `8603e97` and at the branch, and **no shared key changed value** — 160
shared keys, 4 per-item rows × 31 columns identical, `lower_is_better_statistics` the only new
key):

- the cost table above is now generated by a committed micro-benchmark and carries the commit
  it was measured at, and the rows that did not reproduce were deleted rather than kept;
- the scoring run note's caveat says "four numbers" (the line above it carries four) and names
  the comparability gap that matters most for Break EM: these are scored against **MuSiQue gold
  decompositions, not Break's QDMR annotations**, so even an exact string match is measuring
  agreement with a different kind of target;
- `_GED_MIN_NODE_CAP`'s comment names `conditions.unguided_capped.stop_after_step_lines` in
  `configs/decomposer_musique.json` as the source of its 8;
- the gate-coverage test spells the four issue #40 columns literally, instead of asserting a
  set difference against the constant the v1 field list is derived from;
- `configs/musique_eval.json`'s `break_metrics._note` no longer carries a second prose copy of
  the cost table — **this record is the single canonical copy**, and the config points here;
- a scoring run's `eval_metrics.json` now carries `lower_is_better_statistics` (it already
  existed in a comparison's), so a machine reader can discover that `ged_macro` points the
  other way without parsing prose. The nine arms re-scored by exp-011 predate the key; their
  committed metrics files do not have it, which changes no value they carry.

The **PR #45** review of that pass then found that the paragraph replacing the cost story was
itself past its evidence, and the fix is above: four shapes were added to
`configs/ged_cost_benchmark.json`, the table was regenerated at `21a1203`, the gold-similarity
claim is **withdrawn** on measurement, the script's `*`/`†` legend now travels with the rows,
and the bound-versus-optimizer table gained the one shape where the bound is *not* tight
(+0.0588). The evaluator was not touched in that pass, so the additivity evidence above still
holds unchanged. The convention this whole episode establishes — a number that justifies a
config constant is a committed script that prints its own table plus the SHA it was measured
at, and a tool that scores no system takes no `experiments/log.md` entry — is recorded on its
own in [ADR 0027](./0027-a-number-that-justifies-a-constant-is-a-committed-script.md), because
the next person needing that rule will not look for it in a Break-metrics record.

**Not measured here, and therefore not claimed:** any number on the ADR 0007 pinned 600 or on
any committed run. Re-scoring the nine committed arms with these columns is a separate run and
needs an `experiments/log.md` entry; **no metric value for any real system appears in this
record.** The survey note's §3 figures were computed by a session-local harness, not by this
code, and this implementation deviates from that harness in one respect that can move a value:
the `max_nodes_for_optimizer` cap (item 4) applies where the harness used a 20 s post-yield
budget with 0 truncations.

## Consequences

- Five metrics now have a per-item value and therefore the full battery; `composite_score`
  still has one third of it. That asymmetry is the survey's §4 argument in mechanical form —
  and it is an argument, not a decision.
- The compared-statistics count per comparison grew. Nothing corrects for multiple
  comparisons, and `tests_reported` in every comparison's metrics JSON carries the exact
  counts, so no prose has to be trusted for them.
- `networkx` is now a **direct** dependency of the evaluation path (it arrived transitively via
  torch; a metric should not rest on a transitive pin) and is pinned in `requirements.txt` at
  the version read from the installed `.venv`.
- Every existing per-item file on disk (`runs/`, from exp-004 … exp-009) lacks the new columns,
  so `--compare` refuses it with the existing "re-run the evaluation" message. Re-scoring those
  runs is cheap (CPU, from the stored predictions) but it is a **run**, with a log entry — not
  part of this change.
- If Jahid or his supervisor answers any of the survey's §6 questions differently from an
  assumption above, the fix is an edit to this file plus the code it names. Nothing here is
  load-bearing for a claim, because nothing here has been claimed.
