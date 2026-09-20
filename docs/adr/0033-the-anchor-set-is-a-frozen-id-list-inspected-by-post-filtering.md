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

## Amendment, 2026-09-20 — the prompts half: logged by id, recoverable by dry run

Jahid asked, of the sheet this ADR describes: *"will I be able to compare 45 examples with
their prompts in each run?"* The answer was no. The decomposer logged every Nth prompt **by
position** (`logging.prompt_log_every`, file name `prompt_idx{i:04d}_hop{h}.txt`), and
position is not a property of an item: a run over a different retrieval artifact orders the
600 rows differently, so the sample catches a different subset each time and the file name
carries no id to join on. Measured on exp-020: of its 120 logged prompts, 12 happen to be
anchor items, and those 12 are identifiable only by counting rows in `results.json`; the
other 33 anchors have no prompt at all.

1. **Anchor prompts are logged by id, additively.** Every row whose `query_id` is in the
   frozen 45 gets its full prompt written to the same `prompts_log/` as
   `anchor_{query_id}_hop{h}.txt`, whatever position it sits at. The every-Nth sample and
   its file names are untouched — existing parity checks read `prompt_idx*.txt` by name —
   and a row that is both is written to both names from one string, so the two can never
   drift. The id list comes from `logging.anchor_ids_file`, which the runner `require()`s
   rather than defaults (the `few_shot.ordering` precedent); `null` disables it, which is
   what `configs/decomposer.json` sets, since the 45 are MuSiQue ids and the MetaQA question
   format carries none. The snapshot content-addresses the list
   (`logging.anchor_ids_file_sha256`), as it already does the prompt and the retrieval
   artifact, so a run that inspected a different 45 is detectable from its snapshot alone.
   `metrics.json` records `anchor_prompts_logged` against `anchor_prompts_expected` and the
   run exits non-zero if they disagree — but **after** `results.json`, the snapshot, the
   metrics and the note are all on disk. A mismatch means the anchor *naming* drifted; it
   says nothing about the generations, and on a real run those are hours of GPU time that
   must not be discarded to report a logging defect. The refusal still stands: the anchor
   coverage of such a run may not be trusted and no sheet is rendered from it.
2. **The prompt text does not go into `results.json`.** That file keeps its schema and its
   size; `prompts_log/` is the prompt's home and `scripts/render_anchor_set.py` joins the
   two on `query_id`.
3. **Finished runs are recovered on CPU, not re-run.** `--dry-run --dry-run-limit 600`
   rebuilds a run's prompts from its committed config snapshot and the same retrieval
   artifact, with no GPU and no model weights, and writes the anchor files too. Verified
   2026-09-20 for three arms — exp-020, exp-015 and exp-008 `full_train` — at **120/120**
   logged prompt bodies byte-identical to the regeneration each (the regenerated header
   additionally carries the `Few-shot ordering:` line all three predate; the runs are older
   than ADR 0032's ordering knob). The renderer takes such a directory per arm via
   `--prompts-dir` and names it on the page in words — *written by this run* or
   *regenerated (CPU dry run, not written by this run)* — because a regenerated prompt must
   not be indistinguishable from a logged one. When the arm also names its run directory,
   the regeneration is **checked** against it: the two config snapshots must agree on
   `model`, `condition`, `prompt_sha256` and `retrieval.input_sha256`, and a mismatch, or a
   snapshot that cannot be found, is refused. Pointing at the wrong regeneration is the one
   failure this sheet cannot show, because the page would look correct. A regeneration with
   no run directory to check against is marked *unverified against a run*.
4. **An absent prompt renders a marker, never silence.** A collapsible section that opens
   onto nothing reads as "the prompt was empty", which is a different and worse claim than
   "this run did not log it".

Consequence: `runs/anchor/2026-09-20-exp020-vs-exp015-vs-exp008.md` is the first sheet with
prompts — three arms, 45/45 anchor coverage and 45/45 prompts and raw generations each.
Runs from here on carry their anchor prompts natively and need no recovery step.
