# 0022. Hop-Matched Retrieval: The Implementer's Design

- **Status**: Accepted (design agent-authored, pending Jahid)
- **Date**: 2026-08-21

## Context

Issue #15 asks for three retrieval conditions on the MuSiQue evaluation set: **mixed** (no
hop constraint, today's behaviour), **oracle-hop-matched** (few-shot examples of the
query's *gold* hop depth) and **router-hop-matched** (the same, from a *predicted* hop
depth). "Hop-aware" retrieval sits in the contribution statement (ADR
[0006](./0006-drop-the-jury-fix-dataset-roles-and-the-few-shot-method.md)) as an
assumption, and the point of #15 is to turn it into a measured number.

**That the three conditions exist and get measured is Jahid's decision** (issue #15). How
the filter is implemented was not decided by anyone; this record states the implementation
so the November write-up does not mistake an implementer default for a research choice, and
so changing one is a visible edit rather than archaeology.

**Nothing in this record is a measurement.** No condition has been run on the evaluation
set; there is no metric here, and none may be claimed until an `experiments/log.md` entry
exists. What *has* been executed is listed under "What was verified" below.

## Decision (the implementer's, in one line)

Hop matching is an **opt-in candidate filter inside
`MusiQue/scripts/check_question_similarity.py`, applied before top-k selection**, with the
query's hop depth coming from a pluggable source (`gold` or a predictions file), and with
mixed being the literal absence of the filter. Knobs live in `hop_match` in
`configs/similarity.json`; no hop-matching constant lives in the source.

## Implementation conventions (agent-authored)

1. **The filter binds before top-k, in the bi-encoder stage — not after it.** With matching
   on, the top-20 for a query is computed over the pool rows in that query's hop bucket
   only, so `truncate_top20.py`, `rerank_similarity_results.py` and the decomposer's
   few-shot block inherit the restriction with **no changes at all** (verified end-to-end,
   below). The rejected alternative — retrieve mixed, then drop out-of-bucket neighbours —
   is a different method, not a cheaper version of this one: it returns fewer than k
   examples for some queries, and it makes the matched condition's example count a function
   of the mixed ranking. Filtering at candidate level keeps k constant across conditions, so
   mixed and matched differ in *which* examples are retrieved, not *how many*.

2. **Mixed is the absence of the filter, not the filter with everything allowed.**
   `hop_match.enabled` defaults to **false**; a disabled filter is `None`, and
   `_top_k_from_scores` then runs its original, untouched branch. This is deliberate: the
   guarantee wanted from the mixed arm is that it is the same code, not merely the same
   intent. The pool sweep passes no hop flag, so every existing sweep cell is unaffected.

3. **Nothing falls back to mixed, ever.** Each of these is a hard error with counts: a pool
   row whose id does not parse (it would be silently unreachable under matching, quietly
   shrinking the pool for this condition only); a query id that does not parse under
   `gold`; a query with no entry in the predictions file under `predictions`; a predictions
   file that is malformed, repeats a query id, or is empty; a `predictions` source with no
   file named. A per-query fallback would make "hop-matched" a blend of two conditions and
   the comparison meaningless.

4. **`min_candidates` defaults to `top_k`**: the query's bucket must be able to supply the
   full top-k, or the run refuses with the bucket sizes printed. The reason is
   comparability — a bucket that supplies 12 candidates where mixed supplies 20 would make
   the two conditions differ in candidate-set size *and* in hop composition. The knob
   accepts an integer instead, which deliberately admits a smaller in-bucket candidate set
   for someone who wants it; nothing in the repo sets it that way. **The default is a
   comparability argument, not a measured one** — no evidence says 20 in-bucket candidates
   beats 12. The *effective* floor is never below 1 whatever the config says: a query with
   an empty candidate set is not a hop-matched query but a broken one, so `min_candidates: 0`
   still refuses a hop the pool cannot serve (PR #39 review finding 1, which found that
   combination raising a bare `KeyError` instead).

5. **The hop source is pluggable, and the router's side of it is an interface, not a
   router.** `gold` parses the coarse hop depth from the query id (`2hop__…` / `3hop1__…` /
   `4hop2__…`), the same rule `sample_pool.py` and the decomposer use. `predictions` reads a
   **JSONL, one object per query**, with the id and hop field names taken from config
   (`predictions_id_field` / `predictions_hop_field`, defaulting to `query_id` /
   `predicted_hop`). A prediction *overrides* the gold hop even when it disagrees — that is
   the whole content of the router condition, and it is the behaviour under test.

   **Today's router output does not conform**, and this is flagged rather than
   accommodated: `components/router/run_router.py` writes `detailed_results.json` as a JSON
   *array* of `{question, expected, predicted, correct}`, keyed by question text with **no
   query id**. Whoever produces the router-hop-matched condition must emit the documented
   shape (or write a small adapter). Silently keying on question text was rejected: it makes
   an exact-string match load-bearing on the very field the masking modes rewrite.

6. **The parsing rule is duplicated, and stays duplicated for now.** This repo already had
   three copies of "hop from id" (`MusiQue/scripts/musique_ids.py`,
   `MusiQue/scripts/score_similarity_results.py`,
   `components/decomposer/run_decomposer.py`); `src/hop_matching.py` is the fourth.
   Consolidating them would mean editing `run_decomposer.py`, which was **frozen** for the
   queued exp-004/exp-005 launch at the time of writing. The four agree **on well-formed
   ids only**: `score_similarity_results.py` additionally requires the `__` separator
   (`^(\d+)hop(?:\d+)?__`), so it returns None for a malformed id like `2hopfoo` where the
   other three return 2. No such id exists in MuSiQue, so the copies agree on every id the
   pipeline has actually seen — but they are not the same rule, which is the drift risk, and
   consolidating them is the obvious cleanup once the freeze lifts.

7. **A matched row carries its own provenance**: when the filter is on, each output row gets
   `hop_match = {hop_source, query_hop, pool_candidates}`, so a later reader can tell which
   condition produced a file from the file itself. The field is **absent** when matching is
   off, which is what keeps the mixed **retrieval JSONL** byte-identical. Downstream stages
   pass it through untouched.

   To be precise about the scope of that guarantee: byte-identity holds for the retrieval
   output — the artifact the rest of the pipeline consumes. The **run trail does gain a
   key**: `similarity_metrics.json` and `similarity_config.json` now carry
   `hop_match: {"enabled": false}` (plus `hop_match_per_query_file: []`) on a mixed run. That
   is deliberate — a run record should state which condition produced it rather than leave a
   reader to infer "mixed" from silence — but it means a diff of two trails across this
   commit is not empty, and nothing downstream reads those keys.

8. **`--dry-run` is the preflight, and it is the only part a machine without weights can
   run.** It loads the pool and the queries, resolves and validates the whole hop side, and
   writes the standard trail (config snapshot + metrics + note, `dry_run: true`) **without
   loading the encoder** and without writing the output JSONL. It exists because every
   hop-side failure mode is cheap to hit and expensive to hit late; it is also how
   `scripts/smoke_test.py` exercises this feature (stage `similarity_hop_match_dryrun`).
   Caveat, in the shape ADR [0016](./0016-real-run-only-invariants-get-source-level-guards.md)
   asks to be explicit about: because no model is loaded, a dry run does **not** exercise the
   parameter-count assertion in `src/model_size.py`.

9. **The hop filters for every query file are built before the output file is opened.** Not
   a style choice: the output handle is opened once for all query files, so an infeasible
   bucket discovered in the third file would otherwise leave a truncated JSONL at a path the
   sweep's skip-if-exists logic treats as finished work. Validating up front also means an
   infeasible run costs no embedding time.

10. **No new sweep axis.** `scripts/pool_sweep_orchestrator.py` is untouched: it passes no
    hop flag and its cells stay mixed. Issue #15's eval set is ADR
    [0007](./0007-musique-evaluation-set-reuses-v1-600-questions-200-per-hop.md)'s pinned
    600 (200/hop), not the sweep's 750-query dev sample, so the three conditions are
    produced by driving `check_question_similarity.py` → `truncate_top20.py` /
    `rerank_similarity_results.py` directly over the pinned set — the same way exp-006 drove
    the chain. Adding a hop axis to the sweep would multiply every sweep cell for a question
    the sweep is not asking.

11. **The pool embedding cache is untouched and is shared across the three conditions.** The
    filter selects among vectors; it does not change them, so the cache key (pool content
    hash + model + mode) stays correct and the three conditions can reuse one embedding
    pass.

## What was verified (and what was not)

Ran on 2026-08-21, CPU only, on this branch (counts as of the PR #39 review fix pass):

- `tests/test_hop_matching.py` — **38 tests, all passing**: the three hop buckets, both hop
  sources, a prediction that disagrees with gold, and every failure mode in item 3.
- Full suite `python -m unittest discover -s tests` (**310 tests, OK**) and
  `scripts/smoke_test.py` (**37/37 stages**, including the new
  `similarity_hop_match_dryrun`).
- **Mixed-condition regression, byte level.** The same 9 queries × 45-row throwaway pool
  through the real bi-encoder (`intfloat/e5-small-v2`, CPU, top-k 20, mode `raw`) at
  `origin/main` and on this branch with the feature off (both by config default and by an
  explicit `--no-hop-match`): all three output files sha256
  `1062c5fc8046f364c9d2499b98ab2cdb0e8e23dec854c74060bc2fd66b0a33e6`. In the suite the same
  property is guarded without weights by re-implementing the pre-change `_top_k_from_scores`
  verbatim in the test file and comparing on a seeded score matrix that contains ties.
- **The conditions do what they say.** Same 9 queries, top-k 5: mixed put 14/45 neighbours in
  the query's gold bucket; oracle-hop-matched 45/45; router-hop-matched, driven by a
  predictions file where **every** prediction deliberately disagrees with gold, 45/45 in the
  *predicted* bucket. All three returned exactly k = 5 neighbours per query.
- **The chain inherits the filter unchanged**: `truncate_top20.py` and
  `rerank_similarity_results.py` run on the matched artifact with no code change, keeping
  45/45 neighbours in bucket and preserving the `hop_match` provenance field.

Not verified, and not claimed: **any effect on decomposition quality** (unmeasured — that is
the run issue #15 still needs), the feasibility of the real pool at the real top-k under the
`predictions` source (no predictions file exists yet), and GPU determinism of the embedding
step (this repo sets no torch determinism flags; same caveat as ADR 0021 item 7).

## Consequences

- The oracle-hop-matched condition is runnable now over the ADR 0014 artifact's pool. Its
  hop mix is 2-hop 3,594 / 3-hop 4,387 / 4-hop 1,175 (recorded in ADR 0014), so every bucket
  clears the default `min_candidates = top_k = 20` — checkable without a GPU by running the
  chain's `--dry-run` first, which is the recommended order.
- **The router-hop-matched condition is blocked on a predictions file** in the shape of item
  5. Until one exists, issue #15's "done when" can be met for two of its three conditions
  only. Whether to produce it from the existing router component, from a fine-tuned router
  (ADR 0010), or from a hop predictor yet to be trained is **not** an implementer decision.
- **A confound to state in the write-up, not to fix in code.** Hop matching necessarily
  shrinks the candidate set: the 4-hop bucket is 1,175 rows against the pool's 9,156, so a
  matched-versus-mixed difference mixes "examples of the same hop depth" with "examples drawn
  from a smaller, less diverse candidate set". The two cannot be separated by this
  manipulation — a size-matched control (mixed retrieval from a random 1,175-row subsample)
  would be needed, and nobody asked for one. Any gain is therefore "consistent with hop
  matching helping", not "caused by hop depth alone".
- Because the filter binds in the retrieval stage, each condition is its **own retrieval
  artifact**. Three conditions × the pinned 600 means three top-20 files and their
  truncate/rerank descendants; they are cheap (one shared pool embedding pass) but they are
  distinct files, and a comparison claim must confirm the three were built from the same pool
  and the same query id list.
- `configs/similarity.json` now requires a `hop_match` block: a config without it is refused
  loudly by `require()`, in this repo's no-silent-default style. `configs/similarity_probe.json`
  does **not** need one — the probes never call this script.

### Two conventions this establishes, reusable beyond hop matching

Recorded here at the PR #39 review's request (2026-08-21): both were written as facts about
this one script, and both are general rules the next change of the same shape should reuse.

- **A behaviour-preserving change proves it against a verbatim copy of the pre-change
  function.** When a feature is added *inside* an existing function and the old behaviour has
  to survive untouched under a flag, the guard is: copy the pre-change function body verbatim
  into the test file, marked as the frozen reference, and assert the live function equals it on
  input designed to expose ordering and tie-breaking (here a seeded score matrix rounded to two
  decimals, so ties actually occur). It is stronger than a golden output file — it re-derives
  the expectation, so it also covers inputs no fixture happens to contain — and it works with
  no model weights, which is what let this guard live in the ordinary suite. It is *weaker*
  than an end-to-end byte comparison against the previous commit, so do both when weights are
  available: the byte check catches everything the test's input space does not reach.
  First instance: `tests/test_hop_matching.py::TestMixedConditionRegressionGuard` against
  `_top_k_from_scores`.
- **Validate every input before opening the single output handle.** A stage that opens one
  output file and then loops over inputs must complete all validation in a pre-pass. Otherwise a
  refusal triggered by the *n*-th input leaves a truncated file at the output path — and in this
  repo that path is exactly what the sweep's skip-if-exists logic reads as finished work, so a
  loud failure becomes a silent wrong result on the next run. Two observations, unfixed here
  because they are outside issue #15's scope: `truncate_top20.py` streams input to output line
  by line (a malformed later line leaves a partial file), and `rerank_similarity_results.py`
  parses all input up front but writes incrementally (a cross-encoder failure mid-loop leaves a
  partial file). Neither is a new regression, and both would be a small, separate change.

## Alternatives considered

- **Post-filter a mixed top-20.** Rejected; see item 1. It is cheaper (one retrieval run
  serves all conditions) and that is its only advantage.
- **A separate `hop_matched_retrieval.py` stage.** Rejected for the reason ADR 0021 rejected
  a separate clustering stage: it would need its own output conventions and skip logic, and
  the repo would have two producers of top-k files.
- **A new axis in the pool sweep.** Rejected; see item 10.
- **Fine-hop matching (`3hop1` vs `3hop2`) instead of coarse (2/3/4).** Not implemented. The
  eval set's fine strata are uneven (ADR 0007 records 146/54 for 3-hop and 114/34/52 for
  4-hop), the pool's fine buckets would be thinner still, and issue #15 says "hop depth",
  which everything else in this repo reads as the coarse count. A fine-grained variant would
  be a new `hop_match` mode and a new feasibility check.
- **Keying the predictions file on question text** (what the current router output would
  allow). Rejected; see item 5.
- **Defaulting `min_candidates` to 1** (retrieve whatever the bucket has). Rejected as the
  default because it silently changes k per query, but it is reachable by config for someone
  who wants that behaviour deliberately.
