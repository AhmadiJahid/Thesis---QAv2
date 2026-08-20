# 0009 — Paired bootstrap + exact McNemar as the significance protocol (pending supervisor confirmation)

- **Status:** accepted, **pending supervisor confirmation** of the t-test substitution
- **Amended 2026-08-20** ([ADR 0017](./0017-triage-of-the-2026-08-12-transcript-cross-check.md), Jahid): a paired t-test is **added alongside** the bootstrap CIs + McNemar (issue #30), covering the supervisor's literal ask; this protocol remains the headline.
- **Date:** 2026-08-17
- **Context:** PR #17 (issue #11), independent review of 2026-08-17

## Context

The supervisor asked for statistical testing of metric differences in the
2026-08-12 meeting, naming "a t-test, a statistical test" ([31:33] in
`docs/meetings/2026-08-12-supervisor-meeting-transcript-crosscheck.md`).
Jahid's post-meeting plan specified paired bootstrap confidence intervals and
McNemar instead, and PR #17 implemented that. Every comparison claim in the
thesis will rest on this protocol, so the choice is recorded here rather than
living only in code and `docs/METRICS.md`.

## Decision

The `--compare` mode of `scripts/musique_decompositions_evaluator.py` is the
house significance protocol for decomposition-quality comparisons:

- **Paired percentile bootstrap 95% CIs** (one index matrix applied to both
  systems; 10,000 resamples; α = 0.05 from `configs/musique_eval.json`;
  seeded per ADR 0005) for ROUGE-L F1, step F1, ordered step accuracy and
  composite score. "Significant" = the CI excludes zero (a CI-based decision,
  not a p-value).
- **Exact two-sided McNemar** (binomial on discordant pairs, integer
  arithmetic; b+c=0 → p=1.0) for exact match and hop-count exact match;
  "significant" = p < α.
- **Six tests per comparison, no multiple-comparison correction** — reported
  as-is; a correction (e.g. Holm) can be layered on without re-running
  anything since per-test results are emitted.
- **Refuse rather than intersect**: comparisons across different id sets are
  refused loudly, naming offending ids. This operationalizes CLAUDE.md's
  same-evaluation-set rule and is the house rule for every future comparison
  tool.

## Why bootstrap + McNemar rather than the named t-test

The compared metrics are bounded, non-normal per-item scores (many exactly 0
or 1); a paired t-test's normality-of-differences assumption is doubtful at
n = 600 with such distributions, while the paired bootstrap makes no
distributional assumption and McNemar is the standard exact test for paired
binary outcomes. This is a defensible substitution of the *specific test
named*, not of the supervisor's intent (a statistical test of significance).

## Pending

The substitution has not been confirmed by the supervisor. It is flagged in
issue #19 (triage item: "stats substitution on record") and in
`docs/METRICS.md`. If he insists on a t-test, a paired t-test over per-item
differences can be added alongside without disturbing anything recorded here;
this ADR is then amended, not silently rewritten.

## Alternatives considered

- **Paired t-test** (as named in the meeting): assumption concerns above;
  remains a possible addition, not a replacement, pending the supervisor.
- **Unpaired tests**: wrong — the runs share the identical 600-question set
  (ADR 0007), and pairing is what makes n = 600 powerful.
- **Multiple-comparison correction by default**: deferred; with six tests the
  Bonferroni/Holm adjustment is trivial to apply post-hoc from the emitted
  per-test results, and the primary-metric question (issue #6 item 5) is
  itself still open.
