# exp-038 — choices log (ADR 0042 item 6)

Every choice Claude makes in this experiment goes here, **before** the stage it affects runs.
Each entry needs: the date, who chose (model id, in whose terminal), which files were read
(with their Drive path), what was chosen, and why. Where a choice departs from the
pre-registered default rule (`src/weight_space.py::select_candidate`, thresholds in
`configs/exp038_weight_space.json`), say so and say why. Never edit an entry after the stage it
governs has run; add a new one instead.

Claude's role in choosing was approved by Jahid on 2026-11-27 (ADR 0042, "Decided by Jahid").
`choice.json` is committed exactly once: a second commit makes `freeze` refuse.

## Pause 1 — after `directions` (sweep candidates)

_Not yet reached._

## Pause 2 — after the val_select sweep (the edit and the length-matched control)

_Not yet reached._
