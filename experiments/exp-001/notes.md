# Decomposer LoRA run - 20260818_222259

- Arm: `full_train` (source `/cta/users/fyilmaz/thesis-qav2-data/musique/musique_ans_v1.0_train.jsonl`)
- Base model: `mistral_7b_instruct` / `mistralai/Mistral-7B-Instruct-v0.3`; quantization: 4bit
- Prompt: `/cta/users/fyilmaz/Thesis---QAv2/components/decomposer/models/mistral_7b_instruct/prompt_unguided.md` (style plain, guided False, few-shot examples in prompt: False)
- Examples: 19938 (hops {'2': 14376, '3': 4387, '4': 1175}); seed 42
- Evaluated on hops [2, 3, 4] of the ADR 0007 set (600 ids across hops [2, 3, 4]; 200 per hop / 600 total asserted from eval_set.expected in /cta/users/fyilmaz/Thesis---QAv2/configs/finetune_decomposer.json)
- Train/eval id overlap: 0 (asserted zero)
- Base parameters: 7,248,023,552 (ceiling 8,000,000,000)
- Trainable: 41,943,040 of 7,289,966,592 (0.5754%)
- Peak GPU memory: 5.046 GiB; wall clock 15497.5s
- Adapter: `runs/exp-001/20260818_222259/adapter`
- Formatted examples: `runs/exp-001/20260818_222259/formatted_examples.txt`

- Code: commit `21243337f6516461636a8aed97d7ac88ca23f00a` on `main` — **working tree dirty** (1 file(s)); this run did not come from committed code
- Config snapshot: `runs/exp-001/20260818_222259/config.json`
- Metrics: `runs/exp-001/20260818_222259/metrics.json`
