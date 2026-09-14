# 0032. The Fixed 45-Item Anchor Set: The Implementer's Conventions

- **Status**: Accepted (implementer conventions, pending Jahid). The *instruction* is the
  supervisor's and is not in question; every convention below is reversible on his or Jahid's
  word, and each one says what reversing it costs.
- **Date**: 2026-09-14

## Context

On 2026-08-27 Jahid's supervisor assigned a fixed anchor sample, recorded as item **A4** of
[`docs/plan/2026-09-14-post-supervisor-meeting-phase.md`](../plan/2026-09-14-post-supervisor-meeting-phase.md)
and scheduled there as **P0**:

> "I want you to pick some anchor sample that you will check manually by yourself in every
> experiment… let's say for 15 from each hop. So in total 45 samples. Those samples you will
> look at them manually… and those will be fixed over all the experiments. We will do it from
> now on."

Jahid proposed the sampling rule and the supervisor confirmed it: *"I pick it randomly, like
15 random, but in all of them, same random, same 15."*

So the research content is settled: **45 items, 15 per hop count (2/3/4), drawn once at
random, then frozen and inspected by hand in every experiment.** What was *not* settled was
everything an implementer has to choose to make that real — which population the draw comes
from, where the ids live, what shape they live in, and where the code that restricts a run or
a re-score to them sits. Nothing like it existed: the nearest artifacts were
`configs/qwen_smoke.json` (20 per hop, sampled at run time, ids never committed) and the
ADR [0007](./0007-musique-evaluation-set-reuses-v1-600-questions-200-per-hop.md) pinned 600.
The evaluator's `--limit` is a head-slice, not an id filter, so it could not do this.

This record is the conventions, not the instruction.

## Decision

**1. The draw source is the ADR 0007 pinned 600, not the wider dev set.** Every experiment in
this phase is evaluated on the pinned 600 (ADR 0031 item 2), so an anchor item drawn from it
is scored as part of the normal run: the manual inspection costs **no extra inference**, and
the same 45 rows exist in every arm's outputs, which is what makes the arms comparable
item-by-item by hand. Drawing from the dev set at large would have meant either extra
generation per run or an anchor item that no arm had scored.

**2. Seed 42**, the repo-wide convention, threaded through `src/seeding.py` like every other
sampler here.

**3. The draw is a committed script, not a pasted list** — ADR
[0027](./0027-a-number-that-justifies-a-constant-is-a-committed-script.md) applied to a
constant that happens to be 45 ids long. `scripts/build_anchor_set.py` re-derives them from
the committed config, the fixed seed and the pinned files, and its **default action is
`--verify`**: running it with no flags checks the committed list and can never overwrite it.
`--write` is explicit and refuses to clobber an existing list without `--force`, because
re-drawing changes what every experiment's manual inspection is looking at.

**4. Two files, two roles, and the filenames say which.**
`configs/anchor_set_draw.json` holds the draw's **parameters** and is the only one a human
edits; `configs/anchor_set_ids.json` is the **generated, frozen** output and is never
hand-edited. They are not two copies of the same thing: one is input, one is output, and
`--verify` fails if they disagree. The output file carries **ids and provenance only** — the
source file names, their id counts and a SHA-256 fingerprint of each hop's id population —
and no question text, no gold decompositions and no predictions, because data never enters
git (`CLAUDE.md`).

**5. Two details make the draw independent of the file it reads.** The population is
**sorted** before sampling, so a re-ordered or re-generated source file cannot silently move
which items are inspected; and the drawn ids are sorted in the committed file, which loses
nothing (a sample is a set) and makes the artifact diff-readable. One RNG walks the hops in
the committed order, following `scripts/run_qwen_smoke.py::_sample_per_hop`; the hop list is
therefore part of the draw, which is why it is committed.

**6. The id filter lives beside the evaluator, not inside it.** `scripts/anchor_inspect.py`
**selects**; it never scores. Two modes, one committed id list:

- `--per-item <eval_*_per_item.json>` — the preferred path. The anchors are already inside
  every pinned-600 run's per-item file, so the cheap move is to select them out of it: no
  GPU, no re-score, and the values in the dump are by construction the run's own. It writes
  a filtered per-item JSON plus a CSV with question, gold decomposition, predicted
  decomposition and the per-item metric columns side by side — the CSV convention
  `scripts/run_qwen_smoke.py` already established for eyeballing decompositions, not a second
  format.
- `--filter <predictions.json|rows.jsonl> --filter-out <path>` — restricts a predictions file
  (the evaluator then re-scores exactly those 45 with **no evaluator change**: the filtered
  file is passed to `--predictions`) or a retrieval JSONL (a generation run restricted to the
  anchors, via the decomposer's `--retrieval-input`; that run is not the pinned set, so it
  also needs `--allow-unpinned-eval-set` and it is not an experiment arm).

`scripts/musique_decompositions_evaluator.py` is **untouched**. An `--ids` flag inside it
would have been a second selection path through the scoring code for no gain, and the rule
for this lane was to add a selection path, not to change how anything is scored.

**7. A dump missing an anchor item is a refusal.** A run over the pinned 600 contains all 45;
fewer means the input is not what the reader thinks it is, and a silently short dump would be
read as "the model did this on the anchors". `--allow-missing` is the explicit, recorded
opt-out for a subset run, in the shape `--allow-unpinned-eval-set` already established.

**8. Rows come out in anchor order** — hops in the committed order, ids sorted within a hop —
so two arms' dumps sit line-by-line beside each other. That is the whole point of A7's
eye-to-eye comparison.

## Consequences

- The manual inspection attached to every run in this phase (A4, and A7's Qwen-vs-Mistral
  comparison) is now a CPU-only selection over artifacts the run already produced.
- **The 45 ids are a research-visible constant.** Changing the seed, the hop list, the
  per-hop count or the source invalidates the committed list and `--verify` says so — which
  is the intended friction, because the supervisor asked for the *same* 45 every time.
- The anchor set is **not** an evaluation set: it is 45 of the 600 items, and no aggregate
  computed over it is a comparison against anything scored on the 600. Nothing in the code
  reports an anchor-set mean, deliberately.
- If the pinned files ever disappear the list survives (it is committed) but stops being
  re-derivable — the same exposure ADR 0007 already records for the 600 itself. The recorded
  per-hop SHA-256 is what would detect a changed population.
- Re-drawing is a decision, not a command: `--write --force`, plus an updated record here.

## Alternatives considered

- **Draw from the MuSiQue dev set at large.** Rejected under decision 1: the anchors would
  not be scored by the arms' own runs, so inspecting them would cost extra inference per
  experiment and could not be compared against the numbers in the log.
- **Stratify the draw by fine-hop label (`3hop1`/`3hop2`, `4hop1`/`4hop2`/`4hop3`).**
  Rejected: the supervisor said hop *count*, and stratifying further would be methodology
  invented by an implementer. The observed fine-hop split of the draw is recorded in the PR
  as an observation, not as a design property.
- **Add `--ids` to the evaluator.** Rejected under decision 6.
- **Paste the 45 ids into a doc or an issue.** Rejected under ADR 0027: a list nobody can
  re-derive is a list nobody can defend in November.
- **Emit Markdown as well as CSV for the inspection.** Rejected as a second format for the
  same content; `run_qwen_smoke.py`'s CSV is the existing convention and spreadsheets read it.
