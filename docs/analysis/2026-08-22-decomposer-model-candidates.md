# Decomposer model candidates — open-weight survey through 2026-08-22

**Date**: 2026-08-22 · **Requested by**: Jahid · **Question**: are there newer open-weight models at the same size class that would likely decompose multi-hop questions better than what we use now (`mistralai/Mistral-7B-Instruct-v0.3`, `Qwen/Qwen3.5-9B`)?

**Scope constraints applied**: open weights, runnable locally on one 24 GB GPU at 4-bit; ~8B ceiling with 10B provisionally admitted (ADR 0015, pending supervisor confirmation); no closed commercial models; must be QLoRA-fine-tunable (fine-tuning is a comparison arm, ADR 0012). Anything requiring spend is out of scope to decide here.

This is a survey and a ranked hypothesis list, **not** a measurement. No candidate below has been run on our eval set; every quality claim about them is a proxy from public benchmarks until an exp-004-protocol run produces numbers.

## What the survey found (through August 2026)

Families checked: Qwen (3.5 through 3.8), Llama, Mistral/Ministral, Gemma, Phi, DeepSeek distills, IBM Granite, OLMo/Tülu, NVIDIA Nemotron, Falcon.

**Negative results first** (these prune the zoo honestly):

- **Llama**: Meta shipped no new open-weight small model through mid-2026; Llama 4 Scout/Maverick (Apr 2025) have no ≤10B members. ([H1-2026 retrospective](https://www.digitalapplied.com/blog/open-weight-models-h1-2026-retrospective-deepseek-qwen-llama))
- **Phi**: nothing new ≤10B in 2026. Phi-4-reasoning is 14B; the Mar 2026 release (Phi-4-Reasoning-Vision) is 15B — both over even the provisional 10B ceiling. ([Microsoft Foundry announcement](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/introducing-phi-4-reasoning-vision-to-microsoft-foundry/4499154))
- **DeepSeek**: no R2 and no new small distills; the R1-Distill 7B/8B models (Jan 2025) predate our current pair and bring no new evidence.
- **Qwen after 3.5**: Qwen3.6 (Apr 2026) starts at 27B, Qwen3.7 shipped closed-weight only, Qwen3.8 (Aug 2026) starts at 27B. **Qwen3.5-9B (Mar 2, 2026) remains the newest open-weight Qwen at ≤10B.** ([Qwen 3.5–3.8 guide](https://codersera.com/blog/qwen-3-5-complete-guide-2026/))
- **Nemotron Nano 9B v2** (NVIDIA, Aug 2025): excluded — hybrid Mamba-transformer (QLoRA tooling immature relative to dense transformers), NVIDIA Open Model License rather than Apache 2.0, and Artificial Analysis scores it well below Qwen3.5-9B (15 vs 32). Same reasoning excludes Falcon-H1R-7B (hybrid, AA 16).
- **OLMo 3 7B Instruct** (Ai2, Nov 2025, Apache 2.0): fully open (weights + data + checkpoints), which has genuine thesis value, but its measured intelligence composite is far below the field (AA index 2 vs Qwen3.5-9B's 32); not a credible quality upgrade. ([Ai2 blog](https://allenai.org/blog/olmo3), [AA](https://artificialanalysis.ai/models/olmo-3-7b-instruct))

**Positive result**: the incumbent Qwen3.5-9B is still the strongest open-weight model under 10B by the composite evidence found ("the most intelligent model under 10B parameters", AA index 32, GPQA Diamond 81.7 — [Artificial Analysis](https://artificialanalysis.ai/articles/qwen3-5-small-models)). The survey's main yield is therefore **one strong new candidate inside the uncontested 8B ceiling (Granite 4.1 8B)** and a same-lineage successor to our Mistral base (Ministral 3 8B), not a replacement for Qwen3.5-9B.

## Ranked shortlist

Ranking is by expected decomposition-quality gain **relative to Mistral-7B-Instruct-v0.3** (the model each would displace), weighted by constraint fit. Our own logged evidence already shows Qwen3.5-9B beats Mistral-7B on this exact task on the pinned 600 (exp-005 vs exp-004: unguided composite 0.4212 vs 0.2098, oracle-guided hop_count_EM 0.8733 vs 0.5900 — `experiments/log.md` rows exp-004/exp-005), so the bar for a new candidate is Mistral's slot, not Qwen's.

1. **IBM Granite 4.1 8B Instruct** (`ibm-granite/granite-4.1-8b`) — the strongest new candidate that sits **inside the uncontested ~8B ceiling**. Dense decoder-only transformer (no exotic architecture), Apache 2.0, released 2026-04-29. Model-card IFEval 87.06 and BBH 80.51 are directly relevant proxies for instruction-following and step-wise reasoning; has a CoT "reasoning mode" and structured tool-calling. Standard architecture means QLoRA is drop-in. Caveat: Artificial Analysis's composite scores it only 6 (vs Qwen3.5-9B's 32) — the model-card numbers and the independent composite disagree in magnitude, which is itself a reason to measure rather than trust either.
2. **Ministral 3 8B** (`mistralai/Ministral-3-8B-Instruct-2512`, plus a dedicated `-Reasoning-2512` variant) — the direct in-family successor to our current Mistral base (Dec 2, 2025, Apache 2.0). Scientifically clean swap: same lab lineage isolates "newer model" as the variable, and the paired Instruct/Reasoning variants would let us test whether reasoning post-training helps decomposition at fixed size. Caveat: 8.4B LM + 0.4B vision encoder ≈ 8.8B total — **over the 8B ceiling, inside the provisional 10B**, so it needs the same ADR 0015-style admission as Qwen3.5-9B.
3. **Qwen3.5-4B** (`Qwen/Qwen3.5-4B`) — in-family control inside the uncontested ceiling. AA index 27 ("most intelligent under 5B") is close to the 9B's 32; if it approaches exp-005's numbers it resolves the ADR 0015 tension for free (a ≤8B model with near-9B quality), and it halves fine-tuning cost. Same tokenizer/prompt conventions as the already-integrated 9B lowers integration risk.
4. **Gemma 4 E4B** (`google/gemma-4-E4B`) — 5.1B total / 2.3B effective (Per-Layer-Embeddings architecture), Apache 2.0, Apr 2, 2026. Cheapest to run; native thinking mode. Ranked last of the shortlist: its reasoning proxies (GPQA Diamond 58.6) trail every model above, and QLoRA maturity on the PLE architecture is less proven than on dense transformers. Worth holding as the "small/edge" arm only if a cheap-decomposer arm becomes a thesis question.

**Not a shortlist member but the standing reference**: Qwen3.5-9B stays the quality bar; nothing found supersedes it under 10B.

## Evidence table

All parameter counts and licenses were checked against the model card itself, not headlines. "UNVERIFIED" marks claims found only in secondary sources. **None of these models has any published multi-hop QA or question-decomposition benchmark (MuSiQue, HotpotQA, or otherwise) that I could find — every quality signal below is a proxy** (instruction following, step-wise reasoning composites, math). That absence is the strongest argument for running our own protocol rather than adopting on reputation.

| Model | Params (card) | License (card) | Released | Reasoning/IF evidence (source) | 4-bit on 24 GB | QLoRA | Unknown / caveats |
|---|---|---|---|---|---|---|---|
| Granite 4.1 8B Instruct | 8B dense transformer | Apache 2.0 | 2026-04-29 | IFEval 87.06, BBH(CoT) 80.51, GSM8K 92.49, MMLU 73.84 — self-reported on [model card](https://huggingface.co/ibm-granite/granite-4.1-8b) | Yes (~5–6 GB weights) | Standard dense arch — drop-in | AA composite of 6 conflicts with card numbers (UNVERIFIED which reflects decomposition ability); no multi-hop QA numbers anywhere |
| Ministral 3 8B Instruct / Reasoning | 8.4B LM + 0.4B vision ≈ 8.8B | Apache 2.0 | 2025-12-02 | MMLU 76.1, GPQA-D 66.8, MATH 87.6, Arena-Hard 0.509 — self-reported on [model card](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512); [Mistral 3 announcement](https://mistral.ai/news/mistral-3/) | Yes (unsloth bnb-4bit builds exist) | Proven (community 4-bit + unsloth support) | **>8B: needs ADR 0015-class admission.** IFEval not reported; Reasoning-variant token overhead for a decomposer unmeasured |
| Qwen3.5-4B | 4B dense | Apache 2.0 | 2026-03-02 | AA Intelligence Index 27, "most intelligent under 5B" — [Artificial Analysis](https://artificialanalysis.ai/articles/qwen3-5-small-models) | Yes (~3 GB weights) | Same family as integrated 9B | Per-benchmark card numbers not individually verified here (UNVERIFIED beyond AA composite); gap to 9B on *this* task unmeasured |
| Gemma 4 E4B | 5.1B total / 2.3B effective | Apache 2.0 | 2026-04-02 | GPQA-D 58.6, MMLU-Pro 69.4, AIME'26 42.5 — self-reported on [model card](https://huggingface.co/google/gemma-4-E4B) | Yes | Transformers-compatible; PLE arch less battle-tested for QLoRA (UNVERIFIED) | Weakest reasoning proxies of the shortlist; multimodal stack unused by us |
| *(reference)* Qwen3.5-9B | 9B dense | Apache 2.0 | 2026-03-02 | AA Index 32, GPQA-D 81.7 ([AA](https://artificialanalysis.ai/articles/qwen3-5-small-models), [guide](https://codersera.com/blog/qwen-3-5-complete-guide-2026/)); **our own exp-005 numbers on the pinned 600** | Yes (running today) | Standard dense arch; no adapter trained yet (prompting arms only) | >8B — ADR 0015 confirmation still pending |
| *(current)* Mistral-7B-Instruct-v0.3 | 7.25B | Apache 2.0 | 2024-05 | Our exp-004/exp-008 numbers | In use | In use (exp-001 QLoRA r=16) | Two model generations old; every shortlisted model postdates it by 18+ months |

## How a candidate gets tested fairly

A candidate earns numbers via the **exp-004 protocol** (see `experiments/log.md` rows exp-004 and exp-005, which are this protocol run on Mistral-7B and Qwen3.5-9B respectively):

- All **3 conditions** — `unguided`, `oracle_guided`, `unguided_capped` — via `components/decomposer/run_decomposer.py --model <new_model> --config decomposer_musique.json --condition <cond>`.
- On the **ADR 0007 pinned 600** (200 per hop depth, identical question ids), with the **ADR 0014 retrieval artifact** (v1 9,156-example pool, sha256-pinned), **seed 42**, committed code + committed config, dry-run preflight before launch.
- Evaluated with `scripts/musique_decompositions_evaluator.py`; significance vs the exp-004 (Mistral) and exp-005 (Qwen) arms via the ADR 0009 protocol (paired bootstrap + McNemar + paired t-test), headline metrics per issue #40's finding: step_f1 / ordered_step_accuracy / hop_count_exact_match, composite reported but not headlined.
- Per exp-004/005 precedent this is one GPU pass of ~1,800 generations per model — feasible on the shared 24 GB box under `docs/compute.md` and the run-lock discipline.

Integration cost before any run: a model folder with conforming prompts, an entry in `configs/model_limits.json`, and prompt-parity guards — that touches shared pipeline config, so it goes through a branch + Gate 1 review, not direct to main. If a candidate wins the prompting comparison and should enter the fine-tuning arm, that is a separate QLoRA training run under ADR 0012 conventions (exp-001/exp-008 precedent).

## Governance (explicit)

- **Any model above 8B** (Ministral 3 8B at 8.8B; the incumbent Qwen3.5-9B itself) **requires the supervisor's confirmation of ADR 0015** — that admission is still pending, and if the supervisor reasserts the 8B ceiling it reverts, taking these candidates out of scope.
- **Any model swap is Jahid's call** (with his supervisor where direction is touched). This note ranks hypotheses; it decides nothing.
- Nothing here costs money; if a candidate were only reachable via paid API it was excluded (all shortlisted models are free open weights).

## Recommendation (as hypothesis, not decision)

If Jahid wants one cheap, high-information experiment: run **Granite 4.1 8B** through the exp-004 protocol. It is the only shortlisted model that is simultaneously newer than Mistral-7B-v0.3, inside the uncontested 8B ceiling, Apache 2.0, dense (drop-in QLoRA), and carrying instruction-following/step-reasoning proxies far above the current Mistral base — and its conflicting external composite score makes it exactly the kind of claim this repo settles by measurement. Ministral 3 8B is the natural second run if the supervisor confirms the 10B ceiling.

## Sources

- https://artificialanalysis.ai/articles/qwen3-5-small-models — Qwen3.5 small-model composite scores
- https://codersera.com/blog/qwen-3-5-complete-guide-2026/ — Qwen 3.5→3.8 release timeline, licenses
- https://www.deeplearning.ai/the-batch/alibabas-latest-flagship-models-are-open-weights-moe-performers-in-sizes-from-less-than-1b-parameters — Qwen3.5 release coverage
- https://huggingface.co/ibm-granite/granite-4.1-8b — Granite 4.1 8B model card
- https://datanorth.ai/news/ibm-releases-granite-4-1 — Granite 4.1 release date/family
- https://mistral.ai/news/mistral-3/ — Ministral 3 family announcement
- https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512 — Ministral 3 8B model card (params, benchmarks)
- https://huggingface.co/mistralai/Ministral-3-8B-Reasoning-2512 — reasoning variant
- https://huggingface.co/google/gemma-4-E4B — Gemma 4 E4B model card
- https://ai.google.dev/gemma/docs/core/model_card_4 — Gemma 4 official model card index
- https://www.digitalapplied.com/blog/open-weight-models-h1-2026-retrospective-deepseek-qwen-llama — H1 2026 open-weight recap (no new small Llama)
- https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/introducing-phi-4-reasoning-vision-to-microsoft-foundry/4499154 — Phi 2026 release (15B, over ceiling)
- https://allenai.org/blog/olmo3 and https://huggingface.co/allenai/Olmo-3-7B-Instruct — OLMo 3
- https://artificialanalysis.ai/models/olmo-3-7b-instruct, https://artificialanalysis.ai/models/granite-4-1-8b, https://artificialanalysis.ai/models/ministral-3-8b — independent composite scores
