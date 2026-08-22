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
   - **`max_nodes_for_optimizer` (default 30)** — the guard that bounds cost, and bounds it
     *deterministically*. Above it the optimizer is not called; the reported value is the cost
     of one concrete edit path (pair nodes in sorted id order, substitute each pair,
     delete/insert the surplus, keep the edges whose endpoints are both paired and whose image
     exists on the other side, delete/insert the rest), normalized the same way. A concrete
     edit path's cost is a valid **upper bound** on the edit distance, and it is computed in
     one pass, so it is the same number on every machine. The survey measured a single 39-step
     runaway prediction costing ~115 s of optimizer time by itself (§3.7); a wall-clock budget
     alone cannot bound that, because it cannot interrupt the optimizer's *first* yield.
   - **`per_item_time_budget_seconds` (default 20.0)** — a backstop for graphs under the cap,
     checked between the optimizer's successive approximations; the last approximation yielded
     is kept (still a valid upper bound). This is the **one machine-dependent path** in the
     metric, so every firing is recorded per item in `ged_fallback`, counted in
     `ged_fallback_counts`, and named in the run note. A run whose `ged_fallback_counts` is
     empty has no machine-dependent value in it at all.

   **The cap of 30 comes from measurement, not from taste.** MuSiQue gold is 2–4 steps and the
   capped decomposer arm emits at most 8, so no plausible decomposition is near it. Measured in
   this implementation against the fixture gold (synthetic predictions, one CPU core): a
   chain-shaped prediction cost 0.07 s at 12 steps and 1.27 s at 39; a 20-step prediction whose
   every step references every earlier one cost 0.01 s; a 30-step prediction with no references
   cost under 0.01 s. So the shapes that are cheap stay with the optimizer, and the survey's
   39- and 51-step runaways (the ones that cost ~115 s on real predictions) go to the
   deterministic bound. On the 39-step chain the bound was 0.9620 against the optimizer's
   0.9597, i.e. the trade is "a stated upper bound in microseconds, identical on every machine"
   for a third-decimal difference — on items that are junk by construction. Neither guard fires
   on well-behaved predictions, and `ged_fallback_counts` says so per run.

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
