# Jev, the State of the Art, and a Plan to Improve Hop-Aware Question Decomposition on MuSiQue

**Bottom line:** Jev is a real but closed, early-access, cloud-only scoring model. It should stay out of this thesis, and certainly out of the evaluation loop. The thesis numbers are credible and fall between few-shot frontier models and fully supervised decomposers. The largest remaining gain is to fix under-decomposition: predict the hop count and condition on it, both in the few-shot pool track and in fine-tuning, and use string-level reward or selection methods only.

## TL;DR

- **Jev:** Jev (TypeSafe AI, released 15 Sept 2026, stable version jev-1.13.0) is a proprietary discriminative "System One" model. It returns probability distributions over the yes/no, choice and score questions you define. No weights, architecture or paper have been published. TypeSafe's own evaluations treat frontier-LLM answers as the reference, and its documentation says it is weak at multi-hop indirection and counting. Using it to judge decompositions breaks the standing exclusion. Using it inside the system (hop prediction, exemplar reranking, pool filtering, reward) attracts the same "not scientifically defensible" objection. Open-weight substitutes do every one of those jobs reproducibly. Recommendation: do not use it; at most mention it in related work.
- **State of the art:** On MuSiQue, decomposition exact match near 0.10 is normal. A fully supervised Flan-T5-large reaches EM 0.162 / SARI 0.743 / GED 0.228. Few-shot GPT-4o reaches only 0.007 / 0.605 / 0.323, and a fine-tuned Llama-3.1-8B reaches 0.022 / 0.633 / 0.330. The best thesis arm (0.102 / 0.682 / 0.309) is therefore competitive with LLM baselines, though below in-distribution supervised seq2seq. "Fine-tuning beats few-shot at equal model size" matches the literature. Exemplar-selection methods such as vote-k report average relative gains of 12.9% / 11.4% over random selection at annotation budgets of 18 / 100 (Su et al.); training adds more.
- **Plan:** First, cheap levers inside the few-shot-pool framing: a learned hop/composition-type classifier, hop-count-first output, asymmetric "one-hop-up" exemplar hedging, and string-level best-of-N/MBR selection. Second, fine-tuning on a hop-balanced pool, and combining the fine-tuned adapter with retrieved exemplars. Third, EPR-style learned exemplar retrieval. Finally, a small RL arm: DPO or rejection sampling, then optionally GRPO, with a verifiable string-level reward that penalises under-decomposition twice as heavily as over-decomposition. Statistical hygiene comes first: Holm correction, seeds, per-hop tests, and a random-2k control.

---

## Part A — What Jev is, and whether it belongs in this thesis

### A1. What Jev actually is (verified facts)

| Attribute | What the sources say |
|---|---|
| Developer / release | TypeSafe AI, a San Francisco company founded in 2024 by Diogo Almeida, Erik Gafni and Sasha Sheng. Released in limited early access on 15 September 2026, alongside a US$40M seed round led by DCVC. Stable release jev-1.13.0.\[1\] |
| Model type | Discriminative: "used for classification or regression, not generation". It "does not generate natural-language text". Like an LLM, it can do zero-shot classification over categories supplied at inference time.\[1\] |
| "System One models" | TypeSafe's name for the class of models, after Kahneman's fast, intuitive System 1. Described as "a new class of frontier models built to make fast, structured decisions that software can use directly".\[1\]\[2\] |
| Interface | Input: a **state** (string, JSON object, or array of strings) plus one or more typed **questions**. There are three types, all multiple choice: `noul` (true/false), `choice` (unordered options), and `score` (ordered scale).\[3\] Output: one probability distribution per question. The API derives `choice`, `score` and `confidence` from those distributions with fixed formulas.\[1\] |
| Limits | 64k tokens per input; 32k for state plus the longest question. Choice questions support up to 255 options. Questions are answered in parallel and cannot depend on one another; dependent questions need sequential calls.\[1\]\[2\] |
| Architecture / size / weights | Not published: no architecture, weights, parameter count or technical paper. TypeSafe describes it as transformer-based. Outside observers suggest an open-weight LLM or a BERT-like base.\[1\] |
| Training | Synthetic data only, trained with "Reinforcement Learning for Calibrated Decisions (RLCD)". The method is unpublished; observers suspect supervised learning with calibration terms such as the Brier score.\[1\] |
| Access / price | API only. Early-access signups opened to everyone on 20 Sept with $5 credit and were paused on 22 Sept. Also available through Vercel AI Gateway and OpenRouter (`jev-latest`, `jev-1.13`). Input costs $0.042 per million tokens; output is free. TypeSafe claims 70–500 ms latency.\[3\] Hosted in the US only.\[2\]\[3\]\[4\]\[5\] |
| Benchmarks | TypeSafe's own "workflow evals" use the **average of two frontier LLMs (GPT-6 Astra and Fable 5.1) as the reference probabilities**. TypeSafe concedes these workflows were built by its own staff and may be biased.\[2\] Reranking cookbook: top-1 rose from 5% to 18% and top-10 from 38% to 62% on a legal set after a BM25 shortlist.\[3\] Independent spot check (Every): Jev caught 6 of 7 planted defects, while Fable caught all 7.\[3\] |
| Documented weaknesses (TypeSafe "jaggedness" page) | Literal reading; **math and counting** ("recognizes the shape of an answer rather than tallying"); date comparison; **indirection — "multi-hop questions cost accuracy"**; large or noisy state; adversarial content; no generation.\[3\]\[6\]\[7\] |
| Independent findings (inovex) | Answers to related questions can contradict each other. Scores drift across identical runs (one ticket ranged 0.67–0.76 over 20 runs). Reversing option order shifted scores by about 0.1. The probabilities "most accurately indicate how likely a Frontier LLM would be to give this answer, not how often it is correct."\[4\] |
| Open re-implementations | openjev/SemIf reads option logits from Qwen3.5-4B and reached 0.845 modal agreement against Jev's published 0.883.\[3\] Kev (Qwen3.5, Apache 2.0) and djev also exist.\[4\] Laya, a 421M open decision model, rose from 44% to 69% on a relevance task after fine-tuning on 2,380 labels, versus 74% for Jev.\[3\] Ollama added a Jev-style `/api/systemone` endpoint, and OpenAI announced a "Decisions API".\[4\] |

**Source quality.** Two notes:
- The Wikipedia article is about a month old and editable. Its claims are corroborated by TypeSafe's blog and docs (primary sources) and by independent write-ups from inovex, Firecrawl and Every. I treat vendor performance claims as unverified marketing; TypeSafe itself says its evals are self-built and reference-biased.
- I found no peer-reviewed evaluation of Jev and no disclosure of its size.

### A2. Where a discriminative model could sit in this pipeline, and whether that is allowed

The standing exclusion forbids closed commercial models from scoring, rating or judging decomposition quality **in the scientific evaluation loop**. The deeper question is the supervisor's reason for rejecting Gemini-as-judge: a proprietary, unversioned, non-reproducible model cannot carry the scientific claim. That reason applies to any place where Jev's judgement would change the reported numbers.

| Possible use | Inside the evaluation loop? | Letter of the rule | Spirit of the rule | Technical fit | Verdict |
|---|---|---|---|---|---|
| Scoring or judging decompositions for the report card | Yes | **Violates** | Violates | — | Forbidden |
| Best-of-N selector at inference (Jev picks the "best" of N decompositions) | No, it is part of the system under test | Technically allowed | **Violates**: Jev's judgement of decomposition quality then decides which output gets scored, which is LLM-as-judge by the back door | Poor: multi-hop indirection is a documented weakness | Reject |
| Reward model for RL | No | Technically allowed | **Violates**: the policy is optimised to please Jev, whose probabilities track frontier-LLM opinions (GPT-6 Astra / Fable 5.1 averages), so a closed judge's taste is distilled into the model | Poor; not reproducible, and the version may change mid-training | Reject |
| Hop-count predictor (replacing the 38% router) | No | Allowed | Grey: it changes every downstream number, so results cannot be replicated once `jev-latest` moves | **Poor: hop counting is counting plus indirection, both documented weak spots** | Reject; use a fine-tuned open classifier |
| Reranker for few-shot exemplars | No | Allowed | Grey: the contribution *is* "pool construction + retrieval", so a black box in the retrieval step makes the contribution non-reproducible | Moderate (reranking is a showcased use) | Reject for main results; exploratory appendix at most |
| Pool filtering (keep "good" or diverse gold decompositions) | No; offline data curation | Allowed | Grey to acceptable *if* the filtered pool is frozen and released, since the artefact is then reproducible even if the filter is not | Moderate | Possible but unnecessary: string-level and open-model filters do the same job |
| Data-quality audit (flag malformed gold decompositions for human review) | No | Allowed | Acceptable, because a human makes the final call | Moderate | Acceptable but low value |

**Why "inside the system" does not escape the objection.** The thesis claim is "this pool-construction and retrieval strategy improves hop-aware decomposition". If a closed model in that strategy produces the improvement, the improvement belongs to the closed model and cannot be re-run:
- the version may change;
- there is early-access gating, and new signups were already paused;
- scores vary across identical calls, and with option order;
- hosting is US-only, which can be a data-governance issue for a European university.

Jev's calibration target is frontier-LLM agreement rather than ground truth. Putting it in the selection or reward path therefore imports exactly the kind of closed-model judgement the supervisor rejected.

### A3. Open-weight substitutes that fill every role reproducibly

| Role | Open substitute | Why it is enough |
|---|---|---|
| Hop-count / composition-type predictor | DeBERTa-v3-base or ModernBERT, fine-tuned on the 19,938 MuSiQue train questions with labels taken from question-ID prefixes (2hop, 3hop1, 3hop2, 4hop1–3)\[8\] | About an hour on one GPU, with calibrated softmax probabilities. A close precedent: Adaptive-RAG (Jeong et al., NAACL 2024) trained a T5-Large classifier with cross-entropy to predict a three-way query-complexity label (A/B/C), though not hop counts. |
| Exemplar reranker | The existing cross-encoder, or Qwen3-Reranker-0.6B/4B (Apache 2.0), fine-tuned with decomposer-likelihood labels (EPR-style, see C1) | The reranker already wins 9/9 pairs for ROUGE-L/GED; fine-tuning it on task signal is the literature-backed upgrade. |
| Jev-style typed decisions (if the interface is wanted) | Kev, openjev/SemIf (logits from Qwen3.5-4B), or Ollama `/api/systemone` running a local ≤8B model | Same interface, frozen weights, reproducible. |
| Verifier of chain validity | A deterministic parser (placeholder references in range, acyclic, every step except the last consumed) | Better than any model: exact, free, already in the report card. |
| Reward model | String-level verifiable reward (C2), plus answer EM from the open 8B reader | Consistent with the no-model-judge principle. |

### A4. Verdict on Jev

Jev is an interesting product, and its "typed decisions with calibrated probabilities" idea is a useful framing. It is not useful for this thesis:
1. It cannot appear in the evaluation loop.
2. Anywhere it changes results, it fails the same reproducibility and defensibility test that ruled out Gemini-as-judge.
3. Its documented weak spots (multi-hop indirection, counting) are exactly the judgements this thesis needs.
4. Every candidate role has an open, cheaper, reproducible substitute.

The honest recommendation is to cite Jev in related work as an example of commercial "System One" discriminative models, explain why it was excluded, and move on.

---

## Part B — The thesis results against the state of the art

### B1. Intrinsic decomposition quality: how the numbers compare

The most directly comparable recent work is Han & Gardent (NAACL 2025). They evaluate MuSiQue decompositions with the same three Break metrics (EM, SARI, GED) using the official `break-evaluator`, after lowercasing and stripping question marks.\[9\]

| System | Training / prompting | Size | MuSiQue EM ↑ | SARI ↑ | GED ↓ |
|---|---|---|---|---|---|
| Flan-T5-large, fine-tuned on MuSiQue (Han & Gardent) | Full supervised, in-distribution | 0.78B | **0.162** | **0.743** | **0.228** |
| GPT-4o few-shot (Han & Gardent) | 5-shot CoT | closed | 0.007 | 0.605 | 0.323 |
| Llama-3.1-70B few-shot (Han & Gardent) | 5-shot CoT | 70B | 0.008 | 0.635 | 0.314 |
| Llama-3.1-8B few-shot (Han & Gardent) | 5-shot CoT | 8B | 0.015 | 0.608 | 0.379 |
| Qwen2.5-7B few-shot (Han & Gardent) | 5-shot CoT | 7B | 0.007 | 0.578 | 0.399 |
| Llama-3.1-8B FT on synthetic panel data (Han & Gardent) | LoRA on LLM-panel-selected synthetic decompositions | 8B | 0.022 | 0.633 | 0.330 |
| **Thesis exp-008: Mistral-7B + LoRA, full train, zero-shot** | LoRA on gold | 7B | **0.102** | **0.682** | **0.309** |
| **Thesis exp-026: Mistral-7B + LoRA, 2k pool** | LoRA on gold | 7B | 0.097 | 0.680 | 0.319 |
| Thesis exp-022: Qwen2.5-14B, 5-shot retrieved | Few-shot | 14B | 0.033 | 0.580 | 0.461 |
| Thesis exp-027: Qwen3.5-9B, 5-shot retrieved | Few-shot | 9B | 0.055 | 0.598 | 0.484 |

For context from Break (QDMR), the original CopyNet reached test EM 0.157 / SARI 0.746 / GED 0.322, and the leaderboard's best was a curriculum-trained CopyNet at 0.163 / 0.757 / 0.271. On Break's high-level split, CopyNet reached 0.083 EM.\[10\]

**Interpretation:**
- **break_EM ≈ 0.10 is plausible, and good.** Exact string match on multi-sentence natural-language sub-questions is very strict. Han & Gardent argue EM is not very meaningful outside supervised settings and focus on SARI and GED.\[9\] No LLM in their table, including GPT-4o and Llama-70B, exceeds 0.023 EM on MuSiQue. Fine-tuning a 7B model on gold MuSiQue decompositions recovers much of MuSiQue's phrasing (some sub-questions are template-realised), which explains why the thesis's 0.10 is well above few-shot LLMs.
- **SARI 0.68 and GED 0.31 are competitive with LLM baselines** (better than GPT-4o's SARI 0.605, and similar to its GED of 0.323). They remain clearly below an in-distribution supervised Flan-T5-large (0.743 / 0.228). The thesis should say this plainly. A small seq2seq model trained on the full split remains the intrinsic-metric ceiling on MuSiQue, and that ceiling is what "SOTA" means for these metrics.
- **The few-shot arms (SARI 0.57–0.60, GED 0.46–0.49) are in line with few-shot 7–9B models** in Han & Gardent (SARI 0.55–0.61, GED 0.38–0.48), with higher EM (0.03–0.06 vs 0.007–0.016). The higher EM is plausibly due to retrieved, entity-masked, in-domain exemplars rather than fixed random ones.
- **Comparability caveat:** check that the thesis computes EM, SARI and GED with the official `break-evaluator` and the same normalisation (lowercasing, stripping "?"). If not, rerun the scorer on the saved outputs before quoting any cross-paper comparison. This costs no GPU time.
- **step_f1 (0.34), hop_count_EM, chain validity and the under/over pair** are thesis-specific metrics with no published MuSiQue baselines. This is a strength: they are the thesis's own measurement contribution, and the report card should be presented as such.

### B2. Is "fine-tune 7B on 2k beats few-shot 7–14B" consistent with the literature?

Yes, strongly:
- Mosbach et al. (ACL Findings 2023) compared few-shot fine-tuning and in-context learning on equal footing, controlling for the models used, the number of examples, and the number of parameters (ranging from 125M to 30B). They found fine-tuning generalises comparably or better, and that both are unstable across runs.
- Han & Gardent found fine-tuning consistently beat few-shot prompting of the same compact LLM: +2.5 to +4.3 SARI and 2.7 to 5.6 points better GED on their synthetic data.\[9\]
- The thesis's 2k-pool fine-tune nearly matching the full-split fine-tune (SARI 0.680 vs 0.682) fits the "selective annotation" finding that a small, well-chosen set reaches most of the attainable performance. Vote-k reports that 18 vote-k-selected examples beat 100 random ones on 6 of 10 tasks.\[11\]

The thesis's tension is therefore expected rather than embarrassing. The thing to add is a learning curve showing *where* each delivery mechanism saturates (C1.9).

### B3. The ceiling for exemplar selection vs. training

- **Exemplar retrieval and selection:**
  - EPR (Rubin et al., NAACL 2022) improved Break from 26% to 31.9% with GPT-Neo, by training a retriever on LM likelihood labels.\[12\]
  - CEIL (Ye et al., ICML 2023) adds DPP-based set selection and reports gains over EPR, largest on compositional generation tasks.\[13\]
  - Vote-k reports 12.9% / 11.4% relative gains over random selection at annotation budgets of 18 / 100.\[14\]
  - Complexity-based prompting (Fu et al., ICLR 2023) chose exemplars with more reasoning steps. With GPT-3 and Codex it reached new state of the art on GSM8K, MultiArith and MathQA, with an average +5.3 and up to +18 accuracy improvement.
  - Order sensitivity (Lu et al., ACL 2022) is large in classification, but the thesis found order had no effect on decomposition.\[15\] That is a legitimate, reportable null.
  - Expected ceiling: a few points on structure metrics, and possibly more on hop-count behaviour.
- **Training:** decomposition-specific RL and preference results:
  - Collab-RAG (COLM 2025) trains a standalone decomposer with SFT then iterative DPO, using a **rule-based reward** (placeholder-format check × answer EM/accuracy). MuSiQue EM rose from 22.6 (SFT only) to 26.4 for Llama-3.1-8B, and from 23.6 to 25.4 for Qwen-2.5-3B. Removing the format reward cost 2.2 MuSiQue EM for both backbones (26.4 → 24.2 for Llama-3.1-8B; 25.4 → 23.2 for Qwen-2.5-3B).
  - OPERA (AAAI 2026) reports MuSiQue EM rising from 24.3 (SFT) to 34.8 (GRPO), but its ablation is system-wide, and its GRPO variant injects DeepSeek-R1 samples.\[16\]
  - DecomposeR reports +5.3 average points for SFT+RL over SFT on long-form research benchmarks.\[17\]
  - Expected ceiling: modest, in the low single digits over a good SFT model, consistent with RLVR mainly sharpening behaviour already in the model's sample distribution (Yue et al., "Does RL really incentivize reasoning…").\[18\]

### B4. End-to-end answer EM in context

The thesis's EM values (no_decomp 0.180, unguided 0.213, full_train 0.242, oracle 0.280) need a precise statement of the reader setting. Published MuSiQue numbers differ enormously by setting:

| Setting | System | MuSiQue EM |
|---|---|---|
| Gold supporting paragraphs, chain-of-questions | Han & Gardent: Llama-3.1-8B reader over FT-PANEL decompositions | 0.467 (F1 0.634) |
| Gold paragraphs, GPT-4o decomposer | Han & Gardent | 0.464\[9\] |
| Open retrieval, RL-trained 7B search agent | Search-R1 (Qwen2.5-7B) | ≈0.135–0.146 (reproductions)\[19\] |
| Open retrieval, step-wise RL | StepSearch (Qwen2.5-7B) | ≈0.216–0.226\[19\] |
| Open retrieval, adaptive routing (2024) | Adaptive-RAG (FLAN-T5-XL & XXL, as reported in Stop-RAG) | 0.236 (F1 31.8) |
| Supervised readers (2022) | MuSiQue EX(SA) | answer F1 57.9 / 47.9 / 28.1 on 2/3/4-hop\[20\]\[21\] |

If the thesis reader sees MuSiQue's provided paragraphs, 0.24–0.28 is well below the roughly 0.45 gold-paragraph chain-of-questions numbers. That suggests the reader, not the decomposer, is the bottleneck. If the reader is truly closed-book, 0.24 would be unusually high for a 7B model and worth double-checking. If it uses open retrieval, it is comparable to RL-trained 7B search agents. Either way, the clean result is the *ordering* no_decomp < unguided < fine-tuned < oracle. This supports "better decompositions → better answers", which is all the thesis needs from the end-to-end evaluation.

### B5. Strengths and weaknesses of the current framing

**Strong:**
- The unblended six-term report card, especially separating under- from over-decomposition. No published MuSiQue decomposition paper I found reports this.
- Paired bootstrap / McNemar / t-test on a pinned, hop-balanced set.
- Honest nulls (pool strategy, ordering).
- A clear oracle ceiling (Qwen3.5-9B oracle-guided hop_EM 0.873, under-decomp 0.033).

**Weak:**
1. The primary contribution levers (pool strategy, ordering, hop matching) produced mostly nulls, while the strongest arm is a fine-tune.
2. There is no multiple-comparison correction across roughly 13 statistics × many comparisons.
3. Seed variance is absent, although Mosbach et al. stress that both fine-tuning and in-context learning are unstable.
4. There is a hidden distribution shift. MuSiQue train is 14,376 / 4,387 / 1,175 questions (2/3/4-hop), i.e. 72% 2-hop and only 6% 4-hop.\[20\] The pinned evaluation set is 200/200/200.

The fine-tuned arms cut over-decomposition from about 0.32 to 0.03 but *raised* under-decomposition from 0.17 to about 0.26. This is the signature of a model that learned the training prior's 2-hop skew. It is the single most actionable finding in the thesis, and it points directly at hop-aware pool construction.

---

## Part C — Prioritised plan to improve decomposition quality

**Guiding principle:** The oracle-guided result shows the model can decompose well once it commits to the right hop count. Most of the remaining error is a hop-count decision problem, and the worse error (under-decomposition) is biased in a known direction. Every primary-track idea below is a way to make a better, cheaper, non-oracle hop decision through the pool and retrieval.

### C1 — Few-shot pool and retrieval track (primary)

**C1.1 Learned hop and composition-type predictor → hop-aware retrieval (no oracle).**
- **Hypothesis:** A small fine-tuned classifier predicts hop count far above the prompted router's 38%. Retrieval conditioned on its prediction recovers a large share of the oracle hop_count_EM gain (oracle-matched retrieval gave 0.600 vs 0.515 for mixed).
- **Change:** Train DeBERTa-v3-base or ModernBERT on MuSiQue train with two heads: hop (3 classes) and composition type (6 classes, from ID prefixes). Report accuracy and calibration on the pinned 600 as a stand-alone result.
- **Caution from the literature:** question-complexity routers can be weak. In Hi-Q, an Adaptive-RAG router sent 959 of 1,000 MuSiQue questions to multi-step and was beaten by a constant policy.\[22\] Measure, do not assume.
- **Cost:** under 2 GPU-hours.
- **Expected effect:** hop_count_EM up, under-decomposition down, and step metrics roughly unchanged (consistent with the earlier hop-matching null).

**C1.2 Asymmetric "one-hop-up" hedging (novel).**
- **Hypothesis:** Because the supervisor weights under-decomposition as worse, choosing the hop count to minimise *expected asymmetric loss* beats choosing the most probable class.
- **Change:** With classifier probabilities p(h) and cost ratio c (for example c = 2 for under vs over), pick ĥ = argmin over h of Σ_g p(g)·[c·max(0, g−h) + max(0, h−g)]. Then retrieve k−1 exemplars at ĥ hops plus one exemplar at ĥ+1 hops of the same composition shape.
- **Test:** Does showing a single longer exemplar reduce under-decomposition without inflating over-decomposition?
- **Cost:** negligible once C1.1 exists.
- **Expected effect:** under-decomposition down, over-decomposition slightly up. The trade-off is explicit and tunable, which makes for a good thesis figure (an under/over Pareto curve as c varies).

**C1.3 Hop-count-first structured output ("self-guided oracle").**
- **Hypothesis:** Making the model state the hop count before writing steps captures part of the oracle-guided benefit without an oracle.
- **Change:** Annotate exemplars as "[3 hops] 1. … 2. … #1 … 3. … #2". The model must emit "[k hops]" first. Alternatively, inject the C1.1 prediction as a soft hint ("likely 3 hops").
- **Support:** Plan-and-Solve and least-to-most both separate planning from execution. The thesis's own oracle results (Qwen3.5-9B hop_EM 0.873) show conditioning on hop count works.
- **Cost:** inference only, about half a GPU-day for 3 models.
- **Expected effect:** hop_count_EM up; under-decomposition down; chain validity neutral.

**C1.4 Composition-shape retrieval instead of surface similarity.**
- **Hypothesis:** MuSiQue's six graph shapes (linear chains vs. branching 3hop2 / 4hop2 / 4hop3) matter more than surface similarity of the question.\[20\]
- **Change:** Retrieve by predicted composition type (C1.1 head), then by entity-masked similarity within that type. Add a "decomposition template" key: the gold decomposition with entities masked and placeholders kept (e.g. "Who is the [REL] of #1"). Retrieve by predicted template cluster.
- **Support:** CEIL's finding that set-level, compositional exemplar selection helps most on compositional outputs.\[13\]
- **Expected effect:** GED and chain validity improve on 3hop2 and 4hop2/3; break_EM rises slightly.

**C1.5 Diversity-aware selection (MMR / DPP).**
- **Hypothesis:** Five near-duplicate exemplars waste context.
- **Change:** MMR re-selection of the top-20 with λ ∈ {0.5, 0.7}.
- **Support:** CEIL (DPP), vote-k.
- **Cost:** negligible.
- **Expected effect:** small; mainly robustness (lower variance).

**C1.6 Learned exemplar retrieval with decomposer feedback (EPR/CEIL-style). This is the strongest literature-backed lever.**
- **Change:**
  1. For each pool question q (leave-one-out), take its top-50 bi-encoder candidates e.
  2. Score log p(gold_decomp(q) | e, q) under the frozen open decomposer.
  3. Train the cross-encoder or a Qwen3-Reranker-0.6B with positives = top-scoring and negatives = bottom-scoring candidates.
- **Defensibility:** No model judges any output. The label is the likelihood of the *gold* decomposition, exactly as in EPR.
- **Cost:** about 2k × 50 = 100k scoring passes on a 7B 4-bit model, roughly 1–2 GPU-days.
- **Expected effect:** EPR's +5.9 on Break suggests gains of a few points on SARI/GED; possibly larger on hop_count_EM if hop-length alignment is what the scores reward.
- **Bonus — pool pruning by influence (novel):** Average each pool item's usefulness score across all queries where it was a candidate. Drop consistently harmful exemplars and re-run. This turns "pool construction" into a learned, data-driven procedure using the same score matrix. It answers why the three hand-designed pool strategies were indistinguishable: none of them used task feedback.

**C1.7 String-level best-of-N / MBR selection (no model judge).**
- **Change:** Sample N = 8 decompositions (T ≈ 0.7). Select by a deterministic function of the samples only:
  - (a) modal hop count, then the medoid (minimum mean GED to the others) within that mode; or
  - (b) "complexity-biased MBR": the medoid among samples with hop count ≥ the mode, following Fu et al.'s complexity-based consistency.
- **Support:** Self-consistency is a special case of minimum-Bayes-risk decoding (Bertsch et al., "It's MBR All the Way Down").\[23\]
- **Rule compliance:** Allowed. No model scores another model's output, no gold is used, and the selector is pure string agreement. Keep gold-based "oracle best-of-N" strictly as an analysis ceiling, never as a system.
- **Cost:** N× inference, about 1 GPU-day per model at N = 8.
- **Expected effect:** GED and hop_count_EM improve; variant (b) specifically reduces under-decomposition.

**C1.8 Combine the delivery mechanisms.**
- Run the exp-008 and exp-026 adapters *with* 5 retrieved exemplars, which has never been tried.
- Fine-tune Qwen2.5-7B (and 14B if memory allows) on the 2k pool.
- **Most important pool-construction test:** fine-tune on the *hop-balanced* 2k pool vs the *imbalanced* one. The pool strategy was null for in-context learning, but the under-decomposition pattern of the fine-tunes suggests it may matter a lot for fine-tuning. Add a 4-hop-upsampled variant.
- **Cost:** about 3–6 GPU-hours per 7B LoRA.
- **Expected effect:** under-decomposition down for the balanced and upsampled fine-tunes, with little change elsewhere. This would rescue the pool-strategy contribution in a new delivery mechanism.

**C1.9 Pool-size and k learning curves ("same data, three delivery mechanisms").**
- Pool ∈ {250, 500, 1k, 2k, 5k, 19.9k} × {in-context exemplars, LoRA SFT}, and k ∈ {3, 5, 8, 10} for in-context learning.
- **Hypothesis:** In-context learning saturates early (as vote-k suggests) while SFT keeps improving slowly. This produces the thesis's central figure.
- **Cost:** in-context runs cheap; SFT points about 0.5 GPU-day total.

**C1.10 Format hygiene levers.**
- Grammar-constrained decoding (Outlines/xgrammar) that enforces numbered steps and `#k` references to earlier steps only. Biggest expected effect on Qwen3.5-9B, whose chain validity is 0.847.
- Prompt-template isolation (plain vs chat_template for both model families).
- Complexity-ordered exemplars (simple → complex, highest-hop exemplar adjacent to the query). Low priority given the order null.

**Further untested ideas** (flagged as such):
- *Contrastive exemplars:* add one deliberately under-decomposed plan marked "incorrect" next to its correct version.
- *Hop ladder:* show 2-, 3- and 4-hop exemplars of the same composition shape so the model sees the progression.
- *Structural features for the hop classifier:* count of typed masked entities, and relational markers ("the … of the … who …").

### C2 — RL-style training track (secondary)

**Framing:** The 2,000-question pool is one dataset delivered three ways: as in-context exemplars, as SFT data, and as RL prompts with gold anchors. The RL arm is a comparison arm within the pool story, not a new contribution.

**C2.0 A leakage warning first.** The 28 existing experiments produced outputs on the **pinned 600 dev questions**. Building DPO pairs from those outputs would train on the evaluation set. Pairs must be generated from pool or train questions only.

**C2.1 Rejection-sampling fine-tuning / expert iteration (cheapest).**
- Sample N = 8 from the exp-026 adapter on the pool and remaining train questions (excluding anything in the pinned 600). Keep samples with reward ≥ threshold (reward defined below), add them to the gold, and re-SFT.
- **Cost:** about 1 GPU-day.
- **Before running:** compute the SFT model's *oracle pass@8* on hop_count_EM on a held-out train slice. Yue et al. show RL mostly reweights behaviour already in the model's samples.\[24\] If pass@8 is not well above greedy, RL headroom is small.

**C2.2 Offline DPO on pool-generated pairs.**
- **Chosen:** valid chain, correct hop count, lowest GED to gold.
- **Rejected:** under-decomposed (hop count < gold), or invalid chain.
- Use β ≈ 0.1–0.5 and start from exp-026.
- **Support:** Collab-RAG's SFT → iterative DPO decomposer with a rule-based reward (MuSiQue EM 22.6 → 26.4 for Llama-3.1-8B; 23.6 → 25.4 for Qwen-2.5-3B).\[25\]
- **Cost:** under 1 GPU-day with QLoRA (TRL `DPOTrainer`).

**C2.3 GRPO with a verifiable, string-level reward (optional, if C2.1–C2.2 show headroom).**
- **Reward:** R = V · [0.4·1(hop = gold) + 0.3·(1 − GED) + 0.3·SARI] − 0.2·max(0, gold − hop) − 0.1·max(0, hop − gold).
  - V is a hard validity gate: steps are parseable, every `#k` refers to an earlier step, the chain is acyclic, and every step except the last is consumed.
  - Under-decomposition is penalised twice as heavily as over-decomposition, per the supervisor.
- **Anti-hacking checks:**
  - Penalise any step that copies more than about 80% of the original question's tokens. Otherwise SARI and GED can be gamed by restating the question.
  - Cap the number of steps at 6.
  - Reject degenerate chains such as "#1 of #1".
- **Goodhart control:** Because the reward contains report-card metrics, keep step_f1, ROUGE-L, the 45-item anchor set read by eye, and end-to-end EM as **non-optimised** checks, and state this explicitly.
- **Settings:** Base Qwen2.5-7B-Instruct or Mistral-7B, 4-bit QLoRA via TRL/Unsloth; group size 8; completions of 128 tokens or fewer (decompositions are short); KL β ≈ 0.02–0.05; 1–2k prompts from the pool. My estimate is that this fits in roughly 1–2 GPU-days on a single 24–48 GB GPU; it is not benchmarked.
- **Expected gain over SFT:** low single digits on hop_count_EM and GED, consistent with Collab-RAG and DecomposeR. The main hoped-for effect is lower under-decomposition.

**C2.4 Optional answer-correctness reward via the open 8B reader.**
- Add R_ans = EM/F1 of the reader executing the plan over MuSiQue paragraphs. This mirrors Collab-RAG's accuracy reward (which used GPT-4o-mini) and the Search-R1 / StepSearch line, but here the reader is open-weight and the score is string-level, so it complies with the rule.\[25\]
- Use it offline (to build DPO pairs) rather than online, because reader calls per rollout are expensive.

### C3 — Hygiene and rigour fixes (do these regardless of new results)

1. **Pre-declare primary endpoints.** Under-decomposition and hop_count_EM, matching the supervisor's priorities. Treat the other report-card terms as secondary.
2. **Holm–Bonferroni** within each comparison family; report adjusted and unadjusted p-values. A rough paired-McNemar power estimate on n = 600 with about 10% discordant pairs gives a minimum detectable difference of roughly 2.5 points, which tells readers which nulls are informative.
3. **Seeds:** 3 seeds for every LoRA run; 3 pool-resampling seeds for in-context learning. Report the mean ± spread.
4. **Per-hop significance tests** (n = 200 per stratum). The fine-tune's under-decomposition is likely concentrated in 3- and 4-hop questions.
5. **Controls:** a random-2k fine-tune and a hop-stratified-2k fine-tune versus exp-026; a Qwen fine-tune control; prompt-template isolation.
6. **Metric comparability:** rescore everything with the official `break-evaluator` and Han & Gardent's normalisation, so the SOTA comparison is clean.
7. **Masking ablation on the pinned 600.**

### C4 — Experiment plan summary

| # | Experiment | Hypothesis | Cost (1 GPU) | Expected effect | Priority |
|---|---|---|---|---|---|
| 0 | Stats re-analysis (Holm, per-hop, official scorer) | Existing claims survive correction; under-decomposition is concentrated in 3/4-hop | 0 GPU, ~1 day CPU | Cleaner claims | **P0** |
| 1 | Hop/composition-type classifier (C1.1) | ≫38% accuracy | <2 h | Enables 2–4 | **P0** |
| 2 | Hop-first output + predicted-hop hint (C1.3) | Recovers part of the oracle gain | ~0.5 d | hop_EM ↑, under ↓ | **P0** |
| 3 | Asymmetric one-hop-up hedging (C1.2) | Under/over trade-off is tunable | ~0.5 d | under ↓, over ↑ slightly | **P0** |
| 4 | Fine-tune on balanced / 4-hop-upsampled pool vs imbalanced (C1.8) | Pool strategy matters for fine-tuning even though it was null for ICL | ~1 d incl. 3 seeds | under ↓ | **P0** |
| 5 | Random-2k and stratified-2k fine-tune controls | Pool selection beats random for fine-tuning | ~0.5 d | Validates the pool claim | **P0** |
| 6 | Fine-tuned adapter + retrieved exemplars (C1.8) | The two mechanisms are complementary | ~0.3 d | GED ↓, hop_EM ↑ | **P1** |
| 7 | String-level MBR best-of-N (C1.7) | Agreement picks better plans | ~1 d | GED ↓, under ↓ (variant b) | **P1** |
| 8 | Qwen2.5-7B LoRA on pool | Fine-tuning advantage is not Mistral-specific | ~0.5 d | Generality | **P1** |
| 9 | Composition-shape retrieval + MMR (C1.4–1.5) | Shape similarity beats surface similarity | ~0.5 d | GED ↓ on branching types | **P1** |
| 10 | EPR-style learned reranker + influence pruning (C1.6) | Task feedback beats heuristics | 1–2 d | SARI/GED ↑ a few points | **P1/P2** |
| 11 | Pool-size × mechanism learning curve (C1.9) | ICL saturates early, SFT later | ~1 d | Central figure | **P2** |
| 12 | Rejection-sampling FT + DPO on pool pairs (C2.1–2.2) | Preference learning trims under-decomposition | ~1–2 d | under ↓, hop_EM ↑ | **P2** |
| 13 | GRPO with verifiable reward (C2.3) | RL adds modestly over SFT | 1–2 d+ | Low single digits | **P3** |
| 14 | Grammar-constrained decoding; template isolation | Fixes chain-validity outliers | ~0.5 d | chain_validity ↑ (Qwen3.5) | **P2** |

### C5 — Recommended sequencing, given the deadline

- **Days 1–2 (cheap, highest expected value):** #0, #1, #2, #3, #4, #5. Together these turn the thesis's central weakness (nulls in the pool track) into a coherent story: the right pool and retrieval decision is a *hop decision*, under-decomposition comes from the training prior, and hop-aware pools fix it in both delivery mechanisms.
- **Days 3–4:** #6, #7, #8.
- **Only if time remains:** #9–#14, and treat #13 as a future-work paragraph if it cannot be run with seeds.

### C6 — What to tell the supervisor about Jev

"Jev is a closed, early-access, US-hosted discriminative scoring service with no published weights, size or method. Its own evaluations treat frontier-LLM answers as the reference, and its documentation lists multi-hop indirection and counting as weaknesses. Any use that judges decompositions, selects outputs or provides a training reward would be LLM-as-judge by another route. Even neutral uses (hop prediction, exemplar reranking) would make the pool-and-retrieval contribution non-reproducible. Every candidate role has an open-weight substitute: a fine-tuned DeBERTa hop classifier, an EPR-trained open reranker, a deterministic chain-validity parser, and string-level rewards. We therefore do not use Jev, and we mention it only in related work."

## Caveats

- Jev information comes from TypeSafe's blog and docs plus a handful of independent write-ups published within six weeks of launch. Performance claims are unverified, and parameter count and architecture are unknown.
- Cross-paper metric comparisons assume the same `break-evaluator` normalisation. Han & Gardent lowercase and strip question marks, and the thesis should confirm it does the same.\[9\]
- End-to-end EM comparisons depend heavily on the reader and retrieval setting, which the thesis must state precisely.
- RL gain estimates come from adjacent setups (Collab-RAG and OPERA measure downstream QA, not intrinsic decomposition metrics). GPU-time figures are my estimates, not benchmarks.
- The MuSiQue paper gives train and dev counts by hop (dev: 1,252 / 760 / 405) but not by composition type.\[20\]\[26\] Count the ID prefixes in the official dev file to get the six-way breakdown.

## Sources

1. [Jev (AI model)](<https://en.wikipedia.org/wiki/Jev_(AI_model)>)
2. [Introducing System One Models & Jev - TypeSafe AI Blog](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
3. [What Is Jev? Inside TypeSafe's Decision-Only AI Model and Its Developer Use Cases](https://www.firecrawl.dev/blog/what-is-jev)
4. [TypeSafe AI Jev Review: How Good Is the New Model for AI Classification? - inovex GmbH](https://www.inovex.de/en/blog/typesafe-ai-jev-review-how-good-is-the-new-model-for-ai-classification/)
5. [How to build typed decisions with Jev - Flavio Copes](https://flaviocopes.com/jev/)
6. [Jev TypeSafe AI: Enterprise Security Guide](https://neuraltrust.ai/blog/jev-typesafe-ai-security)
7. [Jev AI by TypeSafe: Deep Dive + 10 Use Cases](https://intercode.com/blog/jev-ai)
8. [HPE: Answering Complex Questions over Text by Hybrid Question Parsing and Execution](https://arxiv.org/html/2305.07789v2)
9. <https://aclanthology.org/2025.naacl-long.55.pdf>
10. [A Question Understanding Benchmark](https://allenai.github.io/Break/)
11. [Selective Annotation Makes Language Models Better Few-Shot Learners](https://arxiv.org/pdf/2209.01975)
12. [Learning To Retrieve Prompts for In-Context Learning Ohad Rubin Jonathan Herzig](https://openreview.net/pdf?id=HElebGnWrWq)
13. [\[2302.05698\] Compositional Exemplars for In-context Learning](https://ar5iv.labs.arxiv.org/html/2302.05698)
14. [\[2209.01975\] Selective Annotation Makes Language Models Better Few-Shot Learners](https://arxiv.org/abs/2209.01975)
15. [Fantastically Ordered Prompts and Where to Find Them: Overcoming Few-Shot Prompt Order Sensitivity](https://www.researchgate.net/publication/362944317_Fantastically_Ordered_Prompts_and_Where_to_Find_Them_Overcoming_Few-Shot_Prompt_Order_Sensitivity)
16. [OPERA: A Reinforcement Learning--Enhanced Orchestrated Planner-Executor Architecture for Reasoning-Oriented Multi-Hop Retrieval](https://arxiv.org/pdf/2508.16438)
17. <https://arxiv.org/html/2605.30824>
18. [Paper page - Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?](https://huggingface.co/papers/2504.13837)
19. [Question Answering on MuSiQue (Exact Match (%)) benchmark leaderboard · SOTA2 Research](https://www.sota2.com/research/sota/question-answering-on-musique-exact-match-percent)
20. <https://arxiv.org/pdf/2108.00573>
21. [(PDF) MuSiQue: Multihop Questions via Single-hop Question Composition](https://www.researchgate.net/publication/360407317_MuSiQue_Multihop_Questions_via_Single-hop_Question_Composition)
22. [Hi-Q: Hierarchical Evidence-guided Query Refinement for Multi-Hop Question Answering](https://arxiv.org/pdf/2608.30468)
23. [Improving Minimum Bayes Risk Decoding with Multi-Prompt - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12226151/)
24. [Limit of RLVR](https://limit-of-rlvr.github.io/)
25. <https://arxiv.org/pdf/2504.04915>
26. [\[2108.00573\] ♫ MuSiQue: Multihop Questions via Single-hop Question Composition](https://ar5iv.labs.arxiv.org/html/2108.00573)
