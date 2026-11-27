# 0045 — MetaQA end-to-end by direct-KG execution (not GRAG): the executor's gold convention, the eval set, and the metric mapping

- **Status:** accepted (lead, 2026-11-27, under Jahid's free hand on technical calls; Jahid's to
  reverse). "Decided by the lead" items are the lead's; "Implementer defaults" are the implementer's.
  Items 7-9 and the item 6 confound text were added after the code review of 2f9e2ce, the same
  day, **before any GPU arm ran**.
- **Context:** ADR 0006 gives MetaQA end-to-end evaluation only, through the supervisor's GRAG system.
  GRAG was never handed over (issue #16), so no MetaQA answer number exists in v2. The repo has a
  direct path (PR #42): `scripts/kg.py` builds the KG from `kb.txt`, `scripts/evaluate_decompositions.py`
  compiles each predicted sub-question into a KG op and executes the chain, and
  `scripts/run_metaqa_compile_execute.py` scores the answer set against gold. MetaQA has no gold
  decompositions, so it carries end-to-end evaluation only (Jahid, 2026-11-27: no silver
  decompositions, no MetaQA fine-tune). exp-039 Phase A checked this path before using it.

## What Phase A measured (exp-039, `scripts/metaqa_gold_path_check.py`)

- `kb.txt`: 134,741 lines, 0 malformed, 43,234 entities (same set as `kb_entity_dict.txt`),
  9 relations, 1,159 duplicate lines (harmless: answers are sets). The original paper (1709.04071)
  gives no KB counts; later papers report ~135k triples / 43k entities / 9 relations.
- The official `qa_*_qtype.txt` files are not available, so the gold-path check enumerates every
  relation path of exactly h edges, both directions, from the question's topic entity and asks
  whether one reproduces the gold answer set. It proves every gold set is reachable in the KG; it
  does not prove the path found is the one each question intends.
- On the v1 100/hop set: 100/100, 100/100, 100/100, **provided the topic entity is dropped**: from
  the final set at 2-hop (only 58/100 reproduce without it) and from the intermediate set at 3-hop
  (only 52/100 without it). A 1-hop answer keeps the topic (the documentary "Lemmy" stars Lemmy).
- On the exp-039 set (below): 234/250, 250/250, 250/250. The 16 1-hop misses are all same-title
  films merged into one `kb.txt` entity (e.g. two "Straw Dogs"): gold names one film, the KB
  returns both. No system that reads this KB as given can match those 16 exactly.
- The executor as shipped scored gold-like decompositions of the v1 set at EM 100/43/52 (step
  wording the compiler expects) and 100/43/0 (natural wording), because of four defects (below).

## Decided by the lead

1. **Direct-KG execution is exp-039's backend, and it is not GRAG.** Every artifact says so
   (`backend_label metaqa_compile_execute_direct_kg`, `grag_wired false`). A GRAG number and a
   compile-execute number are different measurements and are never compared as one. ADR 0006's
   GRAG route stays open if GRAG ever arrives.
2. **The executor follows MetaQA's gold convention.** For a chain of two or more steps, the entity
   step 1 names is removed from every later step's result (`configs/metaqa_kg_eval.json`
   `exclude_topic_after_first_step: true`). Gate: in strict compile mode (item 7), gold-like
   decompositions reach EM 300/300 on the v1 set and 734/734 on the reproducible rows of the
   exp-039 set, in each of three wordings: the compiler's templates, natural wording, and natural
   wording that keeps the question's "the film X" phrasing (it differs from plain natural wording on 29/300 and 143/750 rows).
3. **The other three defects are fixed, and empty results are failures.** `write` joins the writer
   rule; a placeholder step's direction is checked against what the referenced step produces
   (item 7); KB names ending in "?" resolve; an empty final result is
   the execution error `empty_result`, never a success. MetaQA gold sets are never empty (0 of
   39,093 test lines).
4. **Eval set: 250 per hop, one seeded draw from the official vanilla test split.** n comes from the
   Phase A power analysis: exact McNemar, 80% power, v1's observed discordance (~0.20), an effect
   of 0.10, Holm m = 6 gives 250. The v1 100/hop set had 54% power for that effect. The files come
   from a public mirror (`huggingface.co/datasets/camazlucas/MetaQA`, revision `f8385409…`), accepted
   because its line counts equal the official README's (9,947 / 14,872 / 14,274), its `kb.txt` is
   byte-identical to ours, and its first 100 lines per hop reproduce the v1 questions and answers
   100/100. Per-file sha256s are pinned in `configs/metaqa_eval_set.json`; the drawn line indices
   and question hashes are in `experiments/exp-039/eval_set_manifest.json`. Question texts that
   repeat within a hop (10 lines at 1-hop, with conflicting answers) are excluded before drawing.
   The v1 set stays as a separate legacy set. **Residual risk:** the mirror is a third-party
   upload, not the authors' release. The checks above pin its test files' size and v1-overlapping
   content, but lines 101 onward were not compared against the official files (which could not be
   downloaded), so an edit there by the uploader would go undetected. If the official Drive files
   are ever obtained, comparing their sha256 with the pinned ones closes this.
5. **Metric mapping.** The primary answer metric is **exact set match** (EM = Jaccard 1.0) over all
   items, a failed row scoring 0. Reported beside it, on the same items: compile rate, execute rate,
   Jaccard, **any-hit** (P ∩ G non-empty) and **RHits@1** = |P ∩ G| / |P|, the expected Hits@1 of a
   random pick from the unranked set. Ranked Hits@1, the literature's usual metric, is undefined for
   a symbolic executor; an EM is never placed in a Hits@1 column (`experiments/exp-039/literature.md`).
6. **Arms and the pre-declared family.** Arms: oracle (gold-like decompositions, CPU), Qwen3.5-9B
   and Mistral-7B prompted (unguided, MetaQA few-shot pool), and each base with its MuSiQue pool
   adapter (exp-026 for Mistral; exp-036 for Qwen, conditional on that adapter existing). All at
   4-bit, so a fine-tune vs prompting pair moves only the adapter and the prompt (the ADR 0044
   item 2 rule). Primary family: exact McNemar on EM, `mistral_ft` vs `mistral_prompted` and
   `qwen_ft` vs `qwen_prompted`, per hop, Holm m = 6 (ADR 0036), fixed in
   `configs/metaqa_arm_report.json` before any arm ran. Secondary: paired bootstrap CI and paired-t
   p on Jaccard, the same six cells, Holm on its own. An arm that cannot run is reported as skipped;
   its two cells leave the family, and the family size used is stated.
   **Confound, stated with the result:** the compiler's templates were written for MetaQA's own
   wording, which the prompted arms also see in their MetaQA few-shot exemplars and the fine-tuned
   arms (`--no-few-shot`, trained on MuSiQue) never see. A fine-tune can therefore lose EM by
   writing correct steps the compiler cannot parse. Two pre-declared secondaries separate this:
   **EM on the items both arms of a pair compiled** (exact McNemar, own Holm family of six), and
   **each arm's compile- and execution-failure reasons per hop**.
7. **Strict compile mode is primary.** A placeholder step's wording decides its direction: a step
   asking for movies ("What (other) films did [#1] direct?") is a reverse lookup and needs `[#k]` to
   hold values; any other ("Who directed [#1]?", "What genres are [#2]?") is a projection and needs
   `[#k]` to hold movies. A step whose need contradicts what step k produced fails to compile as
   `type_mismatch` (`configs/metaqa_kg_eval.json` `compile_mode: strict`). The review showed that
   repairing such steps lets a type-mismatched decomposition ("2. Who directed [#1]?" after a set
   of people) score as gold. The lenient (repairing) mode is the **secondary** answer metric
   (`configs/metaqa_kg_eval_lenient.json`): every arm also runs it, and the report gives lenient EM
   with each arm's repaired-item count per hop, plus a lenient McNemar family of six.
8. **"the film / the movie" is stripped in every movie template**, with the unstripped name tried
   first so a title that starts with the word ("Movie 43", "Film Geek") still resolves.
9. **Junction safety.** Never recursive-delete `.worktrees/`, a worktree or a data junction; remove
   a worktree with `git worktree remove` and a junction only with `cmd /c rmdir <link>`; never link
   from a worktree to the main checkout's `runs/` or another real data folder (docs/compute.md;
   a recursive delete through such a link emptied the main `runs/` on 2026-11-27).

## Implementer defaults

- The compiler gained templates for genre / release year / language / tags steps, because
  MetaQA's schema (nine relations) and the repo's MetaQA few-shot pool ask for them and no template
  compiled them. It also infers a relation without the entity's own words, so a title like "A Star
  Is Born" cannot pick `starred_actors`.
- The oracle arm uses the natural wording (the other two wordings are gated, not reported as
  arms). For the 16 unreproducible 1-hop rows it executes the
  nearest-miss path, so its EM ceiling (0.936 at 1-hop) is measured, not assumed.
- The MetaQA prompting path (KB masking + e5-small few-shot retrieval) has not run for real in v2;
  `run.sh smoke` decodes 6 rows end to end before the arms spend GPU time.

## Consequences

- v1's MetaQA answer numbers (`v1-prior-work/runs/decomposer/*/analysis/answer_metrics.json`) were
  produced by the defective executor and are not a baseline; v1's Gemini-judge numbers were never
  evidence. Every exp-039 number is new.
- A compile failure is now partly a template-coverage limit of the compiler, not of the
  decomposition. The oracle shows the compiler accepts both wordings for every path type in the
  set, but a model's own wording can still miss; compile rate is reported so this stays visible.

## Alternatives considered

- **Wait for GRAG.** Blocked since August with no interface; it would leave MetaQA unmeasured.
- **Keep the v1 100/hop set.** Underpowered for a 0.10 effect (54%), and 98% of its 1-hop rows
  ask one relation type.
- **An LLM answerer instead of symbolic execution.** It would mix answer-model error into the
  decomposition comparison; execution isolates the decomposition.
- **Silver decompositions from question-type paths for a MetaQA fine-tune.** Ruled out by Jahid
  (2026-11-27): MetaQA stays end-to-end only.
