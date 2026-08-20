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

## Alternatives considered

- Forbidding any use of v1 numbers — loses real, per-item-verifiable evidence that directly
  informs open decisions (masking default, CE, primary metric).
- Re-running the comparisons in v2 — the right long-term answer where they matter, but GPU
  time is contended and several v1 questions (e.g. typed vs uniform) need larger n than a
  re-run would grant anyway.
