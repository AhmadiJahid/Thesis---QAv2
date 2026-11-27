# Claude Code — prompt sequence

Paste one at a time, in order. Wait for each to finish before the next.

**Before you start:** put the standing preamble below at the top of prompt 1, or
paste it once as its own message so it's in the session. It tells the agent where
your v1 data lives.

---

## Standing preamble

```
Standing context for everything in this session.

The v1 repository, Thesis---QA, is on this machine and holds all the data:
MetaQA, MuSiQue, the masked question chunks, the enriched pool, and the v1 run
outputs. Locate it yourself and read from it whenever you need data, prior
metrics, or an artefact that v2 does not have — you do not need to ask me for it.

Two rules about that folder. Never copy data files into the v2 repo; v2 keeps
data outside the working tree and configs/paths.json resolves it from data_root.
And v1 results are evidence of what v1 did, not of what v2 does — never present
a v1 number as a v2 result.
```

---

## Prompt 1 — Unblock the repo

```
Read CLAUDE.md, docs/compute.md and configs/paths.json.

Two things are unresolved and they block every experiment.

First, configs/paths.json has data_root set to the placeholder
"~/thesis-qav2-data", with a note saying the real location is unresolved.
The real data is in the v1 repository. Find it, verify that every path under
datasets. in configs/paths.json actually resolves against it — especially
musique_pool_enriched, musique_dev_gold_clean, musique_train and metaqa_kb —
and report exactly which files exist and which are missing, with sizes. Then
set data_root to the correct location. Do not hard-code paths in any script.

Second, every field in docs/compute.md is _TBD_, and CLAUDE.md says a run
against an unfilled field is blocked rather than improvised around. Fill in what
you can verify yourself from this machine: OS, CUDA and driver version, PyTorch
version, GPU and VRAM, env manager, the command that rebuilds the environment,
repo location, dataset location, free disk. Then ask me, one question at a time,
only for the fields you genuinely cannot determine — which surface runs
experiments, whether the Windows 3090 box and the Ubuntu box are the same
machine, SSH access, the run lock path, and the stale lock age.

Do not guess a field. Do not fill a field I have not answered.

Commit both as a docs and config change.
```

---

## Prompt 2 — Record the meeting decisions

```
Read CLAUDE.md and docs/adr/0002-three-stage-router-decomposer-jury-architecture.md.

My supervisor and I made architecture decisions on 2026-08-12. These are human
decisions. Record them exactly. Do not extend them, do not add success criteria,
do not write a plan.

Write a new ADR superseding 0002 with exactly this content:

- The jury stage is dropped. The supervisor's reasoning: an LLM committee voting
  on decomposition quality is an engineering pattern, not a research
  contribution, and CLAUDE.md already forbids a closed commercial model scoring
  decomposition quality. It moves to Future Work.
- Dataset roles. MuSiQue carries decomposition-quality evaluation and end-to-end
  evaluation, because it ships gold decompositions. MetaQA carries end-to-end
  evaluation only, through my supervisor's GRAG system, because it has no gold
  decompositions.
- The few-shot method is fixed: bi-encoder retrieves top-20, cross-encoder
  re-ranks to top-5, typed entity masking. Raw masking, uniform masking and
  bi-encoder-only retrieval become ablations, not live variables.
- Pool size is fixed at 2000.
- Contribution framing: a strategy for constructing the few-shot pool and
  retrieving examples for hop-aware decomposition, validated empirically. Not
  "we decompose multi-hop questions".
- Whether the pipeline keeps the router is OPEN, to be decided by a
  guided-versus-unguided experiment. Record it as open; do not resolve it.

Add a "Superseded by" line to the Status of 0002.

Then open a GitHub issue titled "Router parameter cap conflicts with the best
router result". Verify the facts by reading the files before filing:
configs/model_limits.json sets router_max_params to 600,000,000 and
src/model_size.py asserts it with a hard SystemExit at load, called from
components/router/run_router.py — but the best v1 router result was
Qwen2.5-3B-Instruct, five times that ceiling, and the model folders under
components/router/models/ include 1.5B, 3B, 7B, Phi and Mistral variants that
will all be refused. So the router numbers in my slides cannot be reproduced in
v2 as configured. Give three options without recommending one: re-run the router
at 0.5B, raise the cap with an ADR recording why, or drop the router.

Also open one issue each, a few lines at outcome altitude, for: metrics
significance testing; guided vs unguided on MuSiQue; LoRA fine-tuning baseline;
pool construction strategies; hop-matched retrieval; end-to-end evaluation.

Docs-only, so commit directly to main.
```

---

## Prompt 3 — Metrics

```
Read CLAUDE.md, the last 20 lines of experiments/log.md, then
scripts/musique_decompositions_evaluator.py, configs/musique_eval.json and
tests/fixtures/predictions/decomposer_results_musique.json.

The evaluator already computes exact match, ROUGE-L precision/recall/F1, step
precision/recall/F1, ordered step accuracy, reference validity, hop-count exact
match, per-hop breakdowns and a weighted composite score. Extend that script.
Do not create a parallel evaluator or a new metrics module.

1. Directional step-count error. My supervisor asked for over- and
under-decomposition to be separated, because a 3-hop question decomposed into 4
steps is a different failure from one decomposed into 14. Add a signed
step-count error per item, and to the aggregate add over_decomposition_rate,
under_decomposition_rate, step_count_mae and mean_signed_step_count_error. All
four must also appear in the per-hop-depth breakdown.

2. Paired significance testing. Add a --compare mode taking two per-item JSON
files. It aligns rows by question id and fails loudly naming offending ids if
the two sets differ — a comparison across different evaluation sets is not a
comparison. It reports paired bootstrap 95% confidence intervals for ROUGE-L F1,
step F1, ordered step accuracy and composite score, and McNemar for exact match
and hop-count exact match, annotating each difference as significant or not.
Bootstrap iterations and alpha come from configs/musique_eval.json with a _note
matching the existing style.

3. Tests with hand-computed expected values against the existing fixture:
prediction longer than gold, shorter than gold, empty prediction, a forward
reference (#3 used in step 2), a self-reference, and --compare on two identical
inputs reporting no significant difference anywhere. They must pass.

4. Write docs/METRICS.md from what the code actually does. It must state
explicitly that ROUGE-L is computed by joining the steps into one string, is
LCS-based, reports precision/recall/F1 and is macro-averaged across items — my
supervisor asked me this in the meeting and I could not answer it. It must list
the composite weights from configs/musique_eval.json. And it must flag that
"composite score" currently names two different things in my work — this
decomposition-quality composite, and a rank-weighted retrieval score — and
propose "retrieval rank score" for the second.

Branch, code-reviewer pass, Closes #N.
```

---

## Prompt 4 — Guided vs unguided (the experiment that decides the router)

```
Read CLAUDE.md, docs/compute.md, the last 20 lines of experiments/log.md, then
components/decomposer/run_decomposer.py and configs/decomposer.json.

configs/decomposer.json defaults to "guided": false, "hops": [1,2,3] and the
MetaQA questions template. Add a MuSiQue variant with hops [2,3,4] and the
MuSiQue template rather than mutating the MetaQA defaults — the MetaQA path
still has to work.

Build and run three conditions on the same MuSiQue evaluation set, everything
else held identical (same model, same decoding config, same seed, same
retrieval: bi-encoder top-20, cross-encoder to top-5, typed masking, pool 2000):

  unguided         no hop count in the prompt
  oracle_guided    gold hop count injected
  unguided_capped  no hop count, but generation stops after N step lines, N in
                   config, default 8, plus a max-token cap

The third exists because my supervisor confirmed a length cap is not cheating,
and runaway 14-step decompositions may be the real problem rather than hop
ignorance.

Write the experiments/log.md entry before launching, with this hypothesis
verbatim: "If the oracle hop count materially improves decomposition quality
over unguided, a router is justified. If unguided_capped closes most of that
gap, the problem was runaway generation length rather than hop ignorance."

Acquire the run lock, release it after. Fixed seed, committed code, committed
config.

Evaluate all three through scripts/musique_decompositions_evaluator.py, run
--compare pairwise, report per gold hop depth including the over- and
under-decomposition rates. Complete the log entry with the real metrics.

Report the three conditions side by side with evaluation-set sizes and
significance annotations, and state which reading the numbers support. Do not
recommend whether to keep the router — that is my decision with my supervisor.
```

---

## Prompt 5 — Pool construction strategies

```
Read CLAUDE.md, then scripts/pool_sweep_orchestrator.py, configs/pool_sweep.json,
scripts/refine_and_sample_pool.py, scripts/check_pool_coverage.py and
src/pool_embeddings.py.

The sweep already covers imbalanced sizes 1k/2k/4k/8k, balanced 1k/2k/3k,
raw/typed/uniform, and biencoder_only vs biencoder_plus_ce. Extend it, do not
replace it.

Add a "clustering" strategy alongside random and balanced: embed the masked
questions with the configured bi-encoder, k-means cluster, sample proportionally
from clusters within each hop stratum so the pool covers question shapes rather
than only hop counts. k in config. The rationale from my supervisor: the point
of the pool is not to match the query's hop count, it is to make sure every hop
depth is well represented so retrieval is not dominated by the most frequent
class.

For the headline run hold pool size at 2000 and retrieval at biencoder_plus_ce
with typed masking. Other sizes and modes stay available as ablations.

Log pool composition per strategy into the metrics JSON: hop distribution,
clusters covered, mean pairwise cosine similarity.

configs/pool_sweep.json caps balanced sizes at 3000 with the note that the 4-hop
train pool has only about 1.2k rows, marked unverified in v2. Count the real
per-hop rows from the actual data and record them. If the number differs from
1.2k, surface it loudly rather than absorbing it silently.

Log entry before the run. Evaluate every strategy through the evaluator, report
per gold hop depth, --compare pairwise. Per-hop reporting is not optional —
aggregate accuracy hides that 4-hop is a minority class, and the per-depth view
is what lets me argue a 4-hop drop is a representation problem rather than a
method failure.
```

---

## Prompt 6 — Hop-matched retrieval

```
Read CLAUDE.md, then MusiQue/scripts/rerank_similarity_results.py,
MusiQue/scripts/check_question_similarity.py and src/pool_embeddings.py.

I assumed that giving a 3-hop question only 3-hop few-shot examples would improve
decomposition, and I built pools, bi-encoders and cross-encoder re-ranking on top
of that assumption without ever measuring it. My supervisor named the habit
directly. Measure it.

Hold the best pool strategy and pool size 2000 fixed. Three regimes on the same
MuSiQue evaluation set:

  mixed               top-k by similarity from the whole pool, any hop depth
  oracle_hop_matched  top-k from the gold hop bucket only
  router_hop_matched  top-k from the router-predicted bucket

The router decision is still open, so put regime 3 behind a config flag and make
1 and 2 runnable with no router present.

Report oracle_hop_matched minus router_hop_matched as its own named headline
number: the router error-propagation cost. It matters because hop-matched
retrieval requires a router at inference, so one router mistake costs twice —
wrong hop count and wrong-bucket examples. Make it impossible to miss.

Log entry before the run, hypothesis stated. Evaluate through the evaluator, per
gold hop depth, --compare between regimes.
```

---

## Prompt 7 — Fine-tuning baseline

```
Read CLAUDE.md, docs/compute.md, configs/model_limits.json, src/model_size.py,
requirements.txt and components/decomposer/run_decomposer.py.

There is no fine-tuning code here and no peft or trl in requirements.txt.
CLAUDE.md records that fine-tuning is allowed and expected as a comparison arm
against prompting. Build it and run it.

LoRA fine-tuning of the decomposer on MuSiQue, two training-data arms in config:
the 2000-example best pool, and the full MuSiQue training split.

Hard requirements. Add peft and trl to requirements.txt, pinned consistently
with transformers==5.5.4, torch==2.10.0 and accelerate==1.12.0 — resolve it
properly, and if you cannot without installing, leave them unpinned with a
comment saying why, matching the file's convention of not inventing versions.
Assert zero overlap between training data and the evaluation set by question id,
failing loudly with the offending ids; this gets a test. Route the base model
through src/model_size.py so the 8B ceiling is asserted. Everything
configurable in a committed config. Adapters and checkpoints stay under the
gitignored runs/ directory; only config, metrics and the run note enter git.

Evaluate both arms on the same evaluation set through the evaluator and compare
against the prompting arms with --compare. Report tokens per query and
wall-clock latency per query alongside quality — that is what lets me argue
cost versus quality if fine-tuning wins on raw score.

Add a generalisation check: train on 2-hop and 3-hop only, evaluate on 4-hop, to
distinguish learning the task from memorising hop patterns.

This is a multi-day run by my own estimate. Launch it, write the log entry
first, and report progress rather than waiting for completion in one sitting.

If fine-tuning beats the prompting method, report that plainly. It is a finding,
not a failure, and I do not want it softened.
```

---

## Prompt 8 — End-to-end

```
Read CLAUDE.md, then scripts/kg.py, scripts/evaluate_decompositions.py,
configs/metaqa_kg_eval.json and scripts/compare_answer_accuracy.py.

Decomposition quality in isolation does not show whether the method is worth
anything. An end-to-end number is what makes this comparable to other people's
work.

Two backends, both accepting a decompositions file from any earlier experiment,
so unguided, router-guided, best-pool and fine-tuned decompositions all run
through the same evaluation.

MuSiQue backend, new: answer each sub-question against the MuSiQue corpus,
chaining #k references so each step substitutes the previous answer. Report
answer exact match and F1, plus a step-level failure taxonomy — broken
reference, unresolvable entity, empty retrieval, wrong intermediate answer.

MetaQA backend: reuse the existing compile-and-execute path in
evaluate_decompositions.py. Do not rewrite the compiler — its own comment says
the step templates and relation rules are deliberately code, a compiler for
MetaQA's question grammar rather than hyperparameters. Wrap it, preserving the
compile-error taxonomy and execution-error categories, and report coverage,
exact match and Jaccard. Add a test proving the wrapper reproduces the existing
fixture outcomes unchanged.

The MetaQA path is meant to route through my supervisor's GRAG system, which is
not in either repository. Check v1 as well as v2, determine precisely what is
missing, and tell me clearly — it is an external dependency I need to chase with
him and I would rather know now than in October. Do not stub GRAG as though it
were available, and do not substitute something else for it.

Log entry before each run.
```
