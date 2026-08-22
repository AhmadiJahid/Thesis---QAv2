# exp-007 — issue #15 / ADR 0022 — CPU retrieval-artifact precompute

Produces the retrieval-side artifacts for two of the three hop-matched-retrieval
conditions (`mixed`, `oracle` = oracle-hop-matched) over the ADR 0007 pinned 600
(200/hop). The third condition, router-hop-matched, is not attempted: today's router
writes no query id (ADR 0022 item 5 / Consequences), so it is not runnable, not a
stub, and not estimated.

## What ran

`MusiQue/scripts/check_question_similarity.py` -> `truncate_top20.py` ->
`rerank_similarity_results.py`, run directly via `runs/exp-007/run_cpu_stages.sh`
(committed script content mirrored into `experiments/exp-007/config.json`'s
`commands_run`), CPU only, seed 42, `--mode typed` (ADR 0006 item 4 default), top-k
20 -> rerank top-5. `mixed` uses `--no-hop-match` (the literal absence of the ADR
0022 filter, not "filter with everything allowed" — verified: this is byte-identical
code path per ADR 0022 item 2/PR #39). `oracle` uses `--hop-match --hop-source gold`.
Pool: the ADR 0014 v1 9,156-row pool, located and hash-verified
(`sha256 212c27634291b27ef55cebdb5feaba08a32a59f21976dd72a96a8f9f1ae66b2a`, exact
match against the value recorded at launch).

## What is verified here (re-checked independently by this completion pass, not just
copied from the per-stage trail)

- 600/600 rows at every stage, both conditions (`wc -l`, re-run).
- Hop split by `query_id` prefix on the final `*_rerank_top5.jsonl` files: 200/200/200
  for both `mixed` and `oracle` (re-derived from the files, not from the trail).
- `typed_top_k` carries exactly 5 examples per row on both final artifacts (spot
  checked row 0; per-stage `rerank_metrics.json` reports `reranked_per_mode.typed:
  600` for both, `raw`/`uniform` both 0 as expected for `--mode typed`).
- `oracle`'s per-file `hop_match` block (from `oracle/similarity/similarity_metrics.json`)
  shows `hop_source: gold`, `min_candidates_resolved: 20`, and per-hop pool bucket
  sizes 3594/4387/1175 for hop 2/3/4 respectively — every bucket clears
  `min_candidates = top_k = 20`, matching ADR 0022's Consequences section exactly.
- `mixed`'s per-file `hop_match` block shows `enabled: false` on every query file, per
  ADR 0022 item 2's guarantee.
- Final artifact sha256 recorded in `metrics.json`, both independently recomputed by
  this pass and matching the values `run_cpu_stages.sh`'s own tail printed at launch.

## What is NOT measured here (by design)

Decomposition quality for either condition — this run is retrieval-artifact
production only. That is exp-009 (GPU decomposer generation + eval), which reads
these two files directly via `--retrieval-input`.

## Housekeeping note

This entry (config/metrics/notes under `experiments/exp-007/`) was written during the
exp-009 preflight, after finding the log row's metrics field still read "PENDING — run
launching" from the 2026-08-21 01:01:05 commit (6fc4bba), even though the artifacts on
disk (`runs/exp-007/{mixed,oracle}/`) were already complete (all timestamps
2026-08-21 01:02–01:05, i.e. after that commit). No new run was performed to produce
this entry — every number above is read from files already on disk. The
`experiments/log.md` exp-007 row's own "metrics" cell is corrected in place in this
same pass, per CLAUDE.md's "written before, completed after" schema; this is a
completion of an existing row, not a new experiment.

Working tree carried one untracked non-pipeline file (`Plan after the meeting.md`)
during the original run and still does now — flagged by the run's own dirty-tree
warning at the time; does not affect any config or code path.
