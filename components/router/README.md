# Router Component

Classifies a question into a hop count (1/2/3). Ported from v1, with the ten
near-identical per-model `router.py` copies consolidated into one runner.

- `run_router.py` — the runner, for every model.
- `models/<model_name>/prompt.md` — the hop-count prompt (few-shot), **byte-identical to v1**.
- `models/<model_name>/config.json` — model id, generation length, loader flags, response parsing.
- `models/prompt_zero_shot.md` — shared zero-shot prompt (no few-shot examples).
- `configs/router.json` — everything that is not model-specific (seed, hops, run counts, output roots).

```bash
python components/router/run_router.py --model qwen2_5_0_5b
python components/router/run_router.py --model qwen2_5_0_5b --prompt-file prompt_zero_shot.md
python components/router/run_router.py --model qwen2_5_0_5b --dry-run   # prompts only, no model
```

Output goes under the runs root from `configs/paths.json`: `router/average_zero_shot/<run_id>/`
for the zero-shot prompt, `router/average_few_shot/<run_id>/` otherwise. Every run writes
`config.json`, `metrics.json`, `notes.md` and `detailed_results*.json`.

`--num-runs N` repeats inference with seeds seed, seed+1, … and reports mean ± std.

**Parameter ceiling.** The runner prints the loaded model's parameter count and asserts it
against `router_max_params` in `configs/model_limits.json` (~600M, per the standing constraint
in CLAUDE.md). Model folders above that ceiling are kept here because their prompts are v1
experimental assets, but the runner will refuse to run them until the ceiling changes — which
is Jahid's decision with his supervisor, not an agent's.

Plots and an HTML report over completed runs:

```bash
python scripts/analyze_runs.py --config configs/analyze_runs.json --component average_zero_shot
```
