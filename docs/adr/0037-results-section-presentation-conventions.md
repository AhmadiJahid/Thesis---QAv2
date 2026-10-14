# 0037 — Presentation conventions for the deck's results section

- **Status:** accepted (lead, 2026-10-14, under Jahid's delegation that day; his to reverse).
- **Context:** the results-section rewrite (plan `docs/plan/2026-10-14-results-section-redesign.md`,
  ADR 0035) introduced conventions no ADR recorded, and its review found that "Holm only where a slide
  says so" conflicts with ADR 0036 item 2 (a family is every test one slide reads).

## Decisions

1. **Marks show the evaluator's uncorrected verdict.** A star is the per-comparison result of the
   evaluator's battery (bootstrap 95% CI excludes 0, or exact McNemar p < 0.05), as ADR 0036 item 1
   keeps it. Accent star = significantly better than the slide's named baseline; red star =
   significantly worse; no other colour carries a verdict. Bold is used only for best-in-column in a
   table, and only where a table has such a column.
2. **Words claim only what survives Holm over the slide's family** (ADR 0036 items 2–3). A title or
   callout that rests on a result that does not survive Holm must say "uncorrected" or be worded as
   a pattern ("2 of 28 uncorrected"), never as an effect. Every slide's source line names its family
   size when Holm is relevant. Non-significance is worded "not detectably" / "not separable", never
   as "no effect" or "equal" (ADR 0036 item 4).
3. **Sign counts are not significance.** "k of n pairs" counts the direction of differences; when
   shown, the number surviving Holm is shown beside it and the title uses the surviving count.
4. **Evaluation-set badge.** Slides on the 750-question pool_sweep sample carry a "750 QUESTIONS"
   badge and are never charted with pinned-600 numbers (ADR 0035 item 2).
5. **Entity colours.** Each decomposer arm keeps one colour on every chart (palette in the deck
   preamble, checked with the dataviz validator; identity is never colour-alone — labels or markers
   carry it). A colour denotes a model *and* its run conditions; where two runs of one model differ
   (e.g. exp-005 vs exp-027 retrieval), the slide's source line says so.
6. **Every plotted number is printed by `Slides/results_chart_data.py`** from committed
   `experiments/<exp-id>/` files, or, for re-analyses without a directory (exp-018, exp-019), cited
   to their log row.

## Amendment (same day, after the fix pass)

- **Item 1, per-depth slides:** on the slides built from exp-029 (hop-matched retrieval, hop guidance,
  fine-tune by depth, generalisation, end to end) a mark denotes a **Holm survivor** within that
  slide's exp-029 family, because those slides read dozens of per-depth tests at once. The results
  divider states this. Every other slide's star remains the uncorrected verdict.
- **Item 6, t-test p-values:** exp-020/021/022's committed `metrics.json` carry no `t_test` block, so
  the family-size slide's Holm column reads the paired-t p from the `runs/` compare file each
  committed `metrics.json` names under `source_files`, after asserting that file's bootstrap and
  McNemar values equal the committed ones. Those Holm values are reproducible only on the compute box.
