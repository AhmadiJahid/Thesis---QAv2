# exp-009 — issue #15 / ADR 0022 GPU follow-up — mixed vs oracle-hop-matched retrieval

GPU decomposer generation + evaluation for the two runnable hop-matched-retrieval
conditions from exp-007 (`mixed`, `oracle-hop-matched`), on Mistral-7B-Instruct-v0.3,
`unguided` condition, ADR 0007 pinned 600 (200/hop). Router-hop-matched is out of
scope this cycle — today's router writes no query id (ADR 0022 item 5/Consequences) —
not attempted, not stubbed, not estimated.

## What ran

Dry-run preflight (both conditions) before real launch: 600/600 pinned ids, 200/200/200
per hop, 0 missing/unexpected ids, retrieval input sha256 matched exp-007's artifacts
exactly. Real generation launched detached in tmux (`runs/exp-009/run_exp009.sh`),
GPU idle at acquisition (10 MiB used / 24242 MiB free, no compute processes),
`runs/run.lock` held for `exp-009` throughout. Both arms rc=0, 600/600 rows, 0 empty
decompositions. `mixed` 23/600 rows hit `max_new_tokens` (1024); `oracle` 32/600.

Evaluated both through `scripts/musique_decompositions_evaluator.py` (per-hop
`per_gold_hop_metrics` reported automatically), then `--compare` (paired bootstrap
95% CI x4, McNemar x2, paired t-test x5, per ADR 0009/0017).

## Headline numbers (composite is NOT the headline metric — see caveat below)

| metric | mixed | oracle_hop_matched | diff (mixed - oracle) | significant? |
|---|---|---|---|---|
| step_f1_macro | 0.2035 | 0.1931 | +0.0104 | no (bootstrap CI [-0.0064,+0.0273]; t p=0.226) |
| ordered_step_accuracy_macro | 0.1804 | 0.1799 | +0.0005 | no (CI [-0.0157,+0.0168]; t p=0.950) |
| hop_count_exact_match_rate | 0.5150 | 0.6000 | -0.0850 | **yes** (McNemar p=0.0006; t p=0.00048) |
| exact_match_rate | 0.0617 | 0.0567 | +0.0050 | no (McNemar p=0.664) |
| composite_score | 0.2099 | 0.2633 | -0.0534 | no by bootstrap (CI [-0.1974,+0.1856]), but see caveat |

Per-hop (step_f1 / ordered_step_acc / hop_count_EM), hop 2/3/4:
- mixed: 0.327/0.318/0.710, 0.168/0.138/0.525, 0.116/0.085/0.310
- oracle: 0.355/0.345/0.845, 0.153/0.135/0.580, 0.071/0.060/0.375

Oracle's hop_count_EM gain concentrates at hop 2 (+0.135) and holds at hop 3/4 (+0.055,
+0.065); its step_f1/ordered_step_accuracy are essentially tied with mixed at hop 2/3
and actually *behind* mixed at hop 4 (0.071 vs 0.116) — the smallest pool bucket
(1,175 rows for hop 4, ADR 0022's stated confound) is where hop-matching helps hop-count
correctness least on step quality.

## Reading

Same shape as exp-004's within-model finding on the (unrelated) guided/unguided axis:
an oracle-style signal — here, hop-matched few-shot examples rather than a stated hop
count — moves whether the model gets the step *count* right significantly, without a
measured step-*quality* gain (step_f1, ordered_step_accuracy both not significant by
bootstrap CI or paired t-test). No claim is made about *why* — ADR 0022's own
Consequences section already names the confound: hop-matching shrinks the candidate
pool (hop 4's bucket is 1,175 of the 9,156-row pool), so any gain is "consistent with
hop matching helping," not isolated from "examples drawn from a smaller, less diverse
set." No router-design or retrieval-design recommendation is made here; that decision
is Jahid's/the supervisor's.

## Composite-score caveat (issue #40)

`docs/analysis/2026-08-22-finetuned-vs-prompting-error-analysis.md` landed on `main`
(commit `ec2f844`) while this run's GPU generation was in flight — a docs-only commit,
verified to touch no pipeline code. It found the composite's 0.2-weight
`reference_validity` term is decided by a regex/data-syntax mismatch: the evaluator's
`_REF_RX` matches bracketed `[#k]` only, but this eval set's gold decompositions use
bare `#k` exclusively, so the term is 1.0 whenever a run's *own* predictions also emit
no bracketed refs (denominator zero) and 0.0 the moment a run emits even one. Here:
`mixed` scores `reference_validity_micro = 0.0`, `oracle` scores `0.278` — this is very
likely the same artifact (a difference in how many predictions in each arm happen to
emit a stray bracketed reference, not a difference in reference *validity*), and it is
the dominant term in composite's apparent +0.053 gap in oracle's favour. composite is
recorded in `metrics.json` per the log schema — the schema does not change — but it is
explicitly not headlined here; step_f1 / ordered_step_accuracy / hop_count_exact_match
are the metrics this note leads with, per issue #40's finding.

## What remains unmeasured

- **Router-hop-matched** (ADR 0022's third condition): blocked on the router writing
  query ids. The oracle-minus-router error-propagation number is unmeasurable this
  cycle and is not estimated anywhere in this entry.
- **Whether the effect (or its absence) survives fixing the reference_validity regex**:
  out of scope for this run; issue #40 raises it as a separate follow-up.

## Provenance

Code at commit `ec2f844` for the trail the runner itself wrote (git.commit reads HEAD
at write time); the actual generation ran from the process image loaded at `133c45f`,
before `ec2f844` (a docs-only commit, no pipeline code diff) landed mid-run — see
`experiments/exp-009/config.json`'s `note_on_commit_delta`. Retrieval inputs are
exp-007's committed artifacts (`experiments/exp-007/`), sha256-verified to match at
generation time. Committed artifacts: `experiments/exp-009/{config.json,metrics.json,
notes.md}`; heavy outputs stay on the box under `runs/exp-009/`.
