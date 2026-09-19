# 0033 — The anchor set is a frozen 45-id list, inspected by post-filtering finished evaluations

- **Status:** accepted (lead, 2026-09-20, under the standing delegation for operational calls; Jahid may veto).
- **Context:** On 2026-08-27 the supervisor asked for a fixed set of anchor items — *"15 from each
  hop. So in total 45 samples… those will be fixed over all the experiments. We will do it from now
  on"* — inspected by hand on every run (phase plan §2 A4, A7). The 2026-09-14 phase plan sketched
  the build as an id list **plus an id-filter path into the decomposer and evaluator**. Lane B2
  built it differently, and the Gate 1 review (2026-09-20) flagged that the departure and the
  freezing convention were recorded nowhere but script docstrings.

## Decision

1. **The set is the 45 ids in `configs/anchor_set_45.json`**, 15 per hop, drawn once by
   `scripts/build_anchor_set.py` with `random.Random(42).sample` over the sorted ids of each of
   the three ADR 0007 pinned-600 files. The file holds ids and provenance only (seed, hops,
   per-hop count, source file names and sha256s, Python version) — never question text.
2. **It is frozen.** `build_anchor_set.py --verify` re-derives the draw and fails on any drift;
   that check runs in the test suite, so the set cannot move without a failing test. Redrawing
   requires `--write --force` and a new ADR saying why. Two independent implementations (the
   2026-09-14 branch `feature/50-anchor-set` and the 2026-09-20 branch) produced the identical 45.
3. **Inspection is post-filtering, not an id-filter into the run path.** Every arm in this block
   is already scored on the full pinned 600, of which the 45 are a subset, so the anchor sheet is
   produced by `scripts/render_anchor_set.py` from finished `eval_per_item.json` files, joined on
   the item id. The decomposer and evaluator are untouched; the pinned-set assertion stays
   intact; no extra inference is ever run for the anchors. The id-filter path from the older
   branch (`scripts/anchor_inspect.py`) is not adopted; it remains on that unmerged branch.
4. **The rendered sheet is data and stays under `runs/`** (gitignored). The renderer refuses an
   output path inside the repo unless git ignores it.

## Consequences

- Every run's write-up in the #50 block renders the 45 for the new arm beside the relevant
  baselines and Jahid reads them by hand (A4/A7). The looking is his; the tooling is one command.
- `runs/anchor/2026-09-20-baselines.md` is the first sheet: exp-015 unguided, exp-008 full_train,
  exp-005 unguided, 45/45 each. exp-008's per-item file predates three report-card columns
  (`over_decomposition`, `under_decomposition`, `decomp_mean`), which render as absent, not zero,
  until exp-008 is re-scored under the current evaluator.
- Supersedes the phase plan §2 A4 build sketch; does not change what the supervisor asked for.
