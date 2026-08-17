# 0007. MuSiQue Evaluation Set Reuses v1's 600 Questions (200 per Hop Depth)

- **Status**: Accepted
- **Date**: 2026-08-17

## Context

Issue #6 item 6 carried the MuSiQue evaluation-set construction forward from v1 as an open decision. ADR [0006](./0006-drop-the-jury-fix-dataset-roles-and-the-few-shot-method.md) gives MuSiQue both decomposition-quality and end-to-end evaluation, so the set it is measured on has to be fixed before v2 experiments start.

## Decision

Jahid decided on 2026-08-17 that the **MuSiQue evaluation set for v2 experiments reuses v1's construction: 600 questions, 200 per hop depth (2, 3, 4 hops), drawn from the MuSiQue dev split.**

The concrete v1 artifact was located on the compute box, so v2 uses the **identical question ids** rather than re-drawing. The set is the union of these three files in the v1 repo:

- `/cta/users/fyilmaz/Thesis---QA/MusiQue/Data/dev_data/musique_ans_v1.0_dev_sample_2_hop_200.jsonl`
- `/cta/users/fyilmaz/Thesis---QA/MusiQue/Data/dev_data/musique_ans_v1.0_dev_sample_3_hop_200.jsonl`
- `/cta/users/fyilmaz/Thesis---QA/MusiQue/Data/dev_data/musique_ans_v1.0_dev_sample_4_hop_200.jsonl`

Verified on 2026-08-17: 200 rows each, 600 distinct ids in total, ids of the form `2hop__…`, `3hop1__…`, `3hop2__…`, `4hop1__…`, `4hop2__…`, `4hop3__…`; within the coarse buckets the fine-hop files contribute 146/54 (3-hop) and 114/34/52 (4-hop). The same 600 ids, **in the same order**, are the query ids of v1's downstream retrieval artifact `/cta/users/fyilmaz/Thesis---QA/MusiQue/Data/sample_extracts/sim_dev_sample600_top20.jsonl` — so this is the set v1's reported MuSiQue numbers were produced on.

## Consequences

- v2 MuSiQue results are directly comparable to v1's on question identity, not merely on construction rule.
- **The generating command and seed for that draw are not recorded in v1.** No stats JSON sits beside the three files and no v1 script references them by name (`MusiQue/scripts/sample_dev.py` samples per hop but writes a single combined file with a stats sidecar, which these files do not have). Reproducibility of this set therefore comes from **reusing the id list**, not from re-running a sampler. Compare ADR [0005](./0005-seed-before-sampling-v1-sampled-results-not-reproducible.md).
- The three files are **untracked in v1's git** (caught by its `.gitignore` `data/` rule) and live only on the compute box. Per `CLAUDE.md`, data never enters git, so they are not copied into this repo; v2 reads them from the paths above.
- If those paths ever disappear, the set is not re-derivable exactly — only the construction rule (200/hop from dev) survives.

## Alternatives considered

Re-drawing 200 per hop under a v2 seed. Rejected by this decision: the v1 artifact exists and reusing its ids keeps v2 comparable to v1's MuSiQue numbers.
