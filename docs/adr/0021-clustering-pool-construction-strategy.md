# 0021. Clustering-Based Pool Construction: The Implementer's Design

- **Status**: Accepted (design agent-authored, pending Jahid)
- **Date**: 2026-08-20

## Context

ADR [0006](./0006-drop-the-jury-fix-dataset-roles-and-the-few-shot-method.md) fixes the
few-shot pool size at **2000** and frames the contribution as the strategy for
*constructing* the pool and retrieving from it — so construction is the thing under study,
and the sweep is free to vary it instead of size. Until now the sweep had two construction
strategies, both random: `imbalanced` (uniform draw over the whole pool) and `balanced`
(equal quota per coarse hop bucket), selected by `--balance` in
`MusiQue/scripts/sample_pool.py`.

Issue #14 asks for a **third, clustering-based strategy** in the same sweep, evaluated
against the others at pool size 2000. That is the part Jahid decided: *that* a clustering
strategy exists and gets measured. He did **not** decide how it clusters — what is embedded,
how many clusters, or which member of a cluster ends up in the pool. This record states the
design the implementation adopted so the November write-up does not later mistake it for a
research decision, and so changing it is a cheap, visible edit rather than an archaeology
exercise.

**Nothing in this record is a methodological finding, and nothing in it has been measured
yet.** The clustered arm has produced no decomposition metrics; the evaluation at pool size
2000 runs later on the GPU and lands in `experiments/log.md` as its own entry (which is why
the PR for this work references #14 rather than closing it).

## Decision

**The clustering strategy is a third value of the sweep's existing construction axis, not a
new stage.** `--balance clustered` in `MusiQue/scripts/sample_pool.py`, `clustered_sizes` in
`configs/pool_sweep.json`, and everything downstream — pool directory names, run keys,
`summary/all_runs.csv` columns, the plots — is unchanged. A clustered cell is therefore
directly comparable with a balanced or imbalanced cell of the same size, trial seed, dev
sample, retriever variant and retrieval mode, which is the only way issue #14's "sits next
to the other strategies" is true.

**It is swept at size 2000 only** (`clustered_sizes: [2000]`), because ADR 0006 fixes the
pool size and the question this arm exists to answer is construction-versus-construction at
that size.

**Everything about how it clusters is config, not code**: `sample_pool.clustering` in
`configs/musique_prep.json` holds the text field, the bi-encoder reference, the cluster-count
rule, the representative rule and the k-means parameters. No clustering constant lives in the
source.

## Implementation conventions (agent-authored)

The design below is the implementer's. It was not reviewed as methodology by anyone, it is
not derived from the README or from any decision of Jahid's or his supervisor's, and it is
changeable without touching issue #14's outcome. Where a choice is arbitrary this says so
rather than inventing a rationale.

1. **Seeded k-means over bi-encoder embeddings, keeping the row nearest each centroid.**
   Concretely: embed every candidate row, fit `sklearn.cluster.KMeans` with
   `random_state = <the trial's pool seed>`, and take the row closest to each centroid.
   k-means was chosen because it is the obvious, standard, cheap choice with one
   interpretable knob, and because "one representative per cluster" reaches an exact target
   size without a second selection rule. It is **not** claimed to be the best clustering for
   this data — no alternative was measured.

2. **The cluster count follows the target size**: `k = ceil(size / examples_per_cluster)`,
   with `examples_per_cluster = 1` by default, so `k = 2000` at the fixed pool size and each
   cluster contributes exactly one row. The knob exists so a coarser clustering with several
   representatives per cluster (`examples_per_cluster = 5` → `k = 400`) is a config change
   rather than a rewrite. **The default of 1 is arbitrary** in the sense that no evidence says
   one representative per cluster beats five; it is the setting that makes "the pool is a
   diverse cover of the candidate set" most literal.

3. **The representative rule is `nearest_to_centroid`, and it is the only one implemented.**
   Any other value is refused at run time rather than silently ignored, in the shape ADR
   [0019](./0019-musique-answering-backend-conventions.md) used for `context.policy`: a second
   selection regime should require changing this record. Within a cluster, rows are ranked by
   Euclidean distance to their centroid, ties broken by row id then row index, so the result
   never depends on library iteration order.

4. **Representatives are taken rank-major.** Every cluster contributes its closest row before
   any cluster contributes a second, so truncating at the target size cannot starve a cluster.
   When clusters cannot supply the target at all (empty clusters, or duplicate points
   collapsing distinct centroids), the shortfall is **topped up** from the unselected rows in
   the same global `(distance, id, index)` order and **counted** in the run's metrics
   (`selected_from_topup`) rather than passing silently. Ascending distance for the top-up is
   an arbitrary choice: "closest to some centroid first" is as defensible as its opposite, and
   neither was measured.

5. **The embedded text is the row's stored `question` field** — the raw question — read as-is.
   Two things about this. First, **no masker runs**: the strategy reads a field the pool file
   already carries, so ADR [0003](./0003-mask-queries-only-never-re-mask-the-few-shot-pool.md)
   (never re-mask the few-shot pool) holds by construction, and the rows written out are the
   input rows value-for-value. Second, the choice of *which* stored field is genuinely
   arguable: clustering on `question_masked_typed` would cluster on question *structure*,
   which is closer to what the retrieval stage matches on. The raw field is the default
   because one pool is consumed by all three retrieval modes (`raw` / `typed` / `uniform`) in
   the same sweep, and ADR [0018](./0018-resolve-the-carried-v1-research-decisions.md) records
   the masking default as **reopened and unsettled** — so tying pool construction to typed
   masking would bake an unsettled variable into the pool itself. `text_field` is the knob;
   pointing it at a masked field is a one-line config change and needs no code.

6. **The bi-encoder is the one the sweep already uses**, `intfloat/e5-small-v2` via the
   `e5-small` alias (ADR 0018). "One sweep, one embedding model" is **threaded, not assumed**:
   for a clustered cell the orchestrator passes its own `embed_model`, `device` *and*
   `configs.similarity` (as `--similarity-config`) to the pool stage, and the run's metrics
   record the registry it resolved through as an absolute path. Before that, the pool stage
   resolved the alias through the registry named in `configs/musique_prep.json` while every
   retrieval stage used the one named in `configs/pool_sweep.json`; both pointed at
   `similarity.json`, so the claim was true by coincidence (PR #34 review, I-2). Texts get the
   `passage:` prefix — from `needs_e5_prefix` in `src/pool_embeddings.py`, now the single
   definition the retrieval stage shares, so "the same prefix as retrieval" is enforced by one
   function rather than by two identical copies — and the vectors are L2-normalised. The
   parameter count is printed and asserted at load like every other model in this repo
   (`src/model_size.py`, component `retrieval`; measured 33,360,000 parameters), and that the
   assertion brackets the encoder construction is itself guarded at source level
   (`tests/test_pool_clustering_guards.py`, ADR 0016), because no run without weights reaches it.

7. **Determinism is bought explicitly, and it covers the clustering — not the embedding.**
   `kmeans.num_threads` defaults to **1** and the fit runs inside `threadpool_limits`, because
   sklearn's Lloyd loop reduces over OpenMP chunks and the thread count is therefore part of
   the floating-point result. At one thread the fit is bit-reproducible under the seed; raising
   the knob is faster and no longer exactly reproducible, and that trade is stated in the config
   rather than discovered later. `n_init = 1` with seeded `k-means++` init is the default
   because `k = 2000` makes repeated restarts expensive; it is a cost choice, not a quality
   claim.

   **The reproducibility that has been *verified* is CPU-only.** The byte-identical-pool check
   runs the encoder on CPU over six fixture rows; the sweep drives the same code with
   `--device cuda`, and this repo sets no torch determinism flags anywhere, so GPU embedding is
   **not** verified to be bit-reproducible. A pool rebuilt later on different hardware — or on
   the same GPU after a driver, torch or sentence-transformers change — may therefore differ at
   the embedding step and select different representatives, even at the same seed. The
   defensible statement is: the *selection rule* is deterministic given the embeddings, and the
   embeddings are only guaranteed reproducible on the machine and stack that produced them.
   That is why the pool file itself is the artifact to keep, not the recipe alone (kept on the
   compute box / under the ignored `runs/`, never committed — data never enters git).

8. **The pool file is shuffled with the trial's seeded RNG before it is written**, exactly as
   the balanced strategy already does, so file order does not encode distance-to-centroid rank
   for anything that reads the pool in order.

9. **No embedding cache was added.** The retrieval stage has one, but it is private to
   `MusiQue/scripts/check_question_similarity.py`, and extracting it into a shared module is a
   refactor of code this work has no other reason to touch. With one clustered size and one
   trial seed in the current grid, the input pool is embedded once. If the grid grows to
   several clustered trials, this is the first thing to revisit.

10. **Clustering is flat, not hop-stratified.** The clustered arm ignores hop buckets entirely
    and reports whatever hop distribution falls out (`sampled_bucket_counts` in the run's
    `stats.json`), which makes it the diversity-driven counterpart of `imbalanced`. A
    hop-stratified clustering (cluster within each hop bucket, quota per bucket) is a plausible
    fourth strategy and is **not** implemented — it would confound two construction ideas in
    one arm, and nobody asked for it.

## Consequences

- The sweep has a third construction arm that is selectable from committed config and
  comparable with the existing two at the fixed pool size. `--only clustered` restricts a run
  to it.
- `configs/pool_sweep.json` now requires `clustered_sizes`; a sweep config without that key is
  refused loudly by `require()`, in this repo's no-silent-default style. `configs/plot_pool_sweep.json`
  gains `clustered` in `balances`, so the per-strategy plots render it. The
  `balanced_vs_imbalanced_delta` plot still pairs exactly those two strategies — a
  clustered-versus-X delta plot was not added, because which comparison the thesis wants is
  Jahid's call.
- The `balance` axis name now means "construction strategy" and one of its values is not about
  hop balance at all. Renaming the axis was declined: it is a column of `summary/all_runs.csv`,
  a component of every pool directory name and run key, and of the v1 run keys the
  significance tooling reads — a rename would break comparability with rows already produced,
  to buy a better word.
- A clustered cell costs one embedding pass over the input pool plus a k-means fit at
  `k = 2000`, single-threaded by default. **This cost is unmeasured at real scale**; only the
  six-row fixture has been run.
- The clustered arm's *quality* is unmeasured. Nothing in this record supports a claim that
  clustering helps, and no such claim may be made until the run is in `experiments/log.md`.
- **Confound 1 — one trial per arm makes the arms asymmetric in variance.** The sweep config
  carries `num_trials: 1` (`pool_trial_seeds: [42]`). At one trial the clustered arm is
  effectively deterministic — reseeding moves only the k-means init, over the same candidate
  set — while `balanced` and `imbalanced` are each a *single random draw* from a distribution
  of possible pools. A one-trial clustered-minus-random delta therefore mixes the effect of
  the construction strategy with the draw noise of the random arms, and the two cannot be
  separated from the numbers the sweep produces. Whether to raise `num_trials` (which
  multiplies the sweep's decomposer cost by the trial count) is **Jahid's decision**; this
  record states the confound and changes no config. Until it is raised, the honest reading of
  a clustered-versus-random difference is "consistent with", not "caused by", the strategy.
- **Confound 2 — the clustered pool is built in the retriever's own embedding space.** Pool
  construction and retrieval share one encoder (item 6), which is what makes the arms
  comparable on retrieval settings, but it also means the clustered pool is a coverage-optimal
  cover of exactly the space the bi-encoder ranks in. The random arms have no such alignment.
  Any retrieval-mediated metric (top-k neighbour quality, and therefore the decomposition
  metrics downstream of it) is measured under that coupling, so a clustered advantage cannot be
  attributed to "better examples" as distinct from "examples spread out in the retriever's
  metric". Decoupling would need a different encoder for construction than for retrieval, which
  is a design change nobody has asked for; no mitigation is claimed here.
- If the supervisor settles the masking default (ADR 0018), item 5 should be revisited: a
  settled typed-masking default is a reason to consider clustering on the masked field, and
  that would be a config change plus a new sweep, not a code change.

## Alternatives considered

- **A separate `cluster_pool.py` stage.** Rejected: it would need its own output conventions,
  its own skip-existing logic and its own orchestrator stage, and the sweep would then have
  two ways to produce a `pool.jsonl`. Extending the existing strategy flag keeps one producer
  of pool files.
- **A new `strategy` axis alongside `balance`.** Rejected for the comparability reason above —
  it would change every run key and the summary table's column set, which the appender in
  `scripts/pool_sweep_orchestrator.py` deliberately refuses to mix.
- **Agglomerative or HDBSCAN clustering.** Both are defensible and neither reaches an exact
  target size without an extra selection rule; HDBSCAN also leaves noise points that would
  need a policy. Not chosen, not measured, and not ruled out on evidence.
- **Maximal-marginal-relevance / farthest-point selection** (greedy diversity without
  clustering). Simpler than k-means and arguably the same intent; it was not chosen because
  issue #14 names clustering. Worth noting as the nearest neighbour of this design.
- **Clustering the *masked* text by default.** See item 5: the argument for it is real, and the
  reason against it is that ADR 0018 leaves the masking default unsettled. The knob is there.
- **Hop-stratified clustering.** See item 10.
- **Choosing k by a criterion (elbow, silhouette) instead of the target size.** Rejected as
  scope: it needs a sweep of its own to be meaningful, and a criterion picked without measuring
  it would be a fabricated rationale.
