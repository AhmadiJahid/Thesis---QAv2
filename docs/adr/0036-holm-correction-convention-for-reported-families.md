# 0036 — How multiple-comparison correction is applied to reported families

- **Status:** accepted (lead, 2026-10-14, under Jahid's delegation that day; his to reverse).
  Records a convention already used twice (exp-018, exp-029) so the thesis can state and defend it.
- **Context:** the evaluator (`scripts/musique_decompositions_evaluator.py`, ADR 0009) applies **no**
  multiple-comparison correction: each comparison's "significant" flag is uncorrected
  (`multiple_comparison_correction: None`). Slides and analysis notes that read many tests at once
  need a correction, and ADR 0009 / 0020 / 0035 only say it is applied post hoc.

## Decision

1. **The evaluator's verdict stays uncorrected.** Holm–Bonferroni is reported **beside** it, as a
   separate, labelled column — never by overwriting it.
2. **A family is one claim surface:** all tests one slide (or one analysis table) reads, i.e. its
   metrics × its comparison pairs × its strata (overall and/or per hop), exactly as listed in the
   experiment's log row. The family size is stated wherever Holm is shown.
3. **The p-value fed to Holm** is the exact McNemar p for binary metrics and the paired-t p for
   continuous metrics, because the paired percentile bootstrap yields a CI but no p. Where the
   bootstrap CI and the t-test disagree on a cell, the disagreement is disclosed.
4. **Non-significance is worded as "not distinguishable" / "not significant"**, never as "equal" or
   "the same".
5. **Families are fixed before the p-values are seen:** for a re-analysis, the log row naming the
   families is committed before the results (exp-029 did not do this; its note says so).
