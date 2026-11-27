# exp-039 — published MetaQA results, and what is comparable to ours

Collected 2026-11-27. Every number below was read in the paper itself (arXiv PDF or HTML, or the
ACL Anthology). A venue marked *(unv.)* was not confirmed on a page. "As reported in X" means the
number is X's baseline table, not the original paper's. Nothing here is a v2 measurement.

## Results on the vanilla MetaQA test sets (1-hop / 2-hop / 3-hop)

| Method (paper, id, venue) | Metric | 1 / 2 / 3-hop | Setting | KB | Backbone |
|---|---|---|---|---|---|
| VRN (1709.04071, AAAI'18 unv.) | Hits@1 | 97.5 / 89.9 / 62.5 | supervised | full | – |
| KV-MemNN, as reported in VRN Table 1 | Hits@1 | 95.8 / 25.1 / 10.1 | supervised | full | – |
| KV-Mem, as reported in PullNet Table 2 | Hits@1 | 96.2 / 82.7 / 48.9 | supervised | full | – |
| GraftNet, as reported in PullNet | Hits@1 | 97.0 / 94.8 / 77.7 | supervised | full | – |
| PullNet (1904.09537, EMNLP'19 unv.) | Hits@1 | 97.0 / 99.9 / 91.4 | supervised | full | – |
| EmbedKGQA (ACL'20, 2020.acl-main.412) | Hits@1 | 97.5 / 98.8 / 94.8 | supervised | full | RoBERTa |
| NSM (2101.03737, WSDM'21) | Hits@1 | 97.1 / 99.9 / 98.9 | supervised | full | – |
| TransferNet (2104.07302, EMNLP'21 unv.) | Hits@1 | 97.5 / 100 / 100 | supervised | full | – |
| KB-BINDER (2305.01750, ACL'23 unv.) | Hits@1 | 93.5 / 99.6 / 96.4 | 5-shot, no training | full | Codex |
| StructGPT (2305.09645, EMNLP'23 unv.) | Hits@1 | 94.2 / 93.9 / 80.2 (zero-shot); 97.1 / 97.3 / 87.0 (32-shot) | training-free | full | gpt-3.5-turbo |
| KG-GPT (2310.11220, unv.) | Hits@1 | 96.3 / 94.4 / 94.0 | 12-shot | full | ChatGPT |
| RoG (2310.01061, ICLR'24) | Hits@1 | – / – / 84.81 (3-hop only) | fine-tuned on 1,000 MetaQA samples | full | LLaMA2-Chat-7B |
| KG-Agent (2402.11163, unv.) | not stated (Hits@1 likely) | 97.1 / 98.0 / 92.1 | instruction-tuned, "one-shot" transfer | full | LLaMA2-7B |
| Interactive-KBQA (2402.15131) | probably F1 | 96.75 / 98.47 / 93.55 | few-shot; **300-question sample** | full | GPT-4 |
| LLM-Planned BFS (2511.19648) | hit rate (any gold answer returned) | 0.874 / 0.752 / 0.230 | zero-shot planning | full | Qwen3-4B |
| same, LoRA on 10k train questions per hop | hit rate | 0.999 / 0.994 / 0.863 | fine-tuned | full | Qwen3-4B |
| LatWeave (2609.27225) | **exact set match** | 0.9447 / 0.6239 / 0.5304 | 9-shot planner, no training | full | not named |

Think-on-Graph (2307.07697, ICLR'24) reports **no** MetaQA results.

## What is comparable to exp-039

exp-039 decomposes the question with a **7–9B open model** (Mistral-7B, Qwen3.5-9B; few-shot or a
MuSiQue-trained adapter), compiles each sub-question into a KG operation by template, and executes
the chain **symbolically on the full `kb.txt`**. It is scored on a **250-per-hop seeded sample** of
the vanilla test split, with **exact set match** (answer EM = Jaccard 1.0) as the primary metric.

1. **Metric.** Almost every row above is **Hits@1**: correct if the single top-ranked answer is
   *any* gold answer. Our executor returns an **unranked set**, so ranked Hits@1 is undefined for
   it. Exact set match is far stricter on multi-answer questions: the one paper reporting both
   (LatWeave) has 0.6239 exact-match vs 0.9983 any-hit at 2-hop. So exp-039 reports, on the same
   items:
   - **EM** (exact set; the thesis metric);
   - **any-hit** (P ∩ G non-empty; comparable to "hit rate" / any-hit rows);
   - **RHits@1** = |P ∩ G| / |P| (the expected Hits@1 of one uniformly random pick from the set;
     Interactive-KBQA's RHits@1 in expectation). This is the closest honest stand-in for Hits@1.
   An exp-039 EM must never be placed in a Hits@1 column.
2. **Setting.** Supervised rows (VRN to TransferNet) train on MetaQA's 300k+ training questions;
   none of our arms trains on MetaQA. The closest settings are the **training-free few-shot LLM
   rows** (KB-BINDER, StructGPT, KG-GPT, LLM-Planned BFS zero-shot). Of these, only LLM-Planned
   BFS uses an open model of our size class (Qwen3-4B); the rest use closed GPT-class models.
3. **Sample.** Ours is 750 questions with bootstrap CIs; most papers report the full test sets
   (39,093 questions) or do not say. A difference inside our CI is not a difference.
4. **Labels.** Our gold-path check (ADR 0045) finds that **16/250 1-hop gold sets cannot be produced
   by any KB relation** (same-title films merged into one entity in `kb.txt`), so 1-hop EM is capped
   at 0.936 for every system that reads this KB as given; the oracle arm measures this ceiling. At
   2- and 3-hop, every gold set is reproduced once the topic entity is dropped from intermediate and
   final sets (MetaQA's convention). Without that convention no relation path reproduces the gold
   set for 91/250 2-hop and 119/250 3-hop questions of our set, which may explain part of the
   low strict-match numbers some papers report (a hypothesis; not checked against their outputs).

Wording rule for the thesis: "on a 250-per-hop sample of the vanilla MetaQA test split, arm X
reaches EM a / any-hit b / RHits@1 c (95% CI ...); for orientation, training-free GPT-class
systems report Hits@1 of ... on the full test sets." The comparison is directional, not a ranking.
