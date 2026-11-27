# exp-036 — Qwen3.5-9B pool-only fine-tune (exp-026 replicated, one change: the base model)

**Status: PENDING — not run.** Nothing in this folder is a result yet. Refs #49 (every fine-tune
claim so far is Mistral-only), #50.

**Question:** does the fine-tuning gain exp-026 measured for Mistral-7B hold for a second model
family? exp-036 trains Qwen3.5-9B exactly as exp-026 trained Mistral-7B-Instruct-v0.3 and
evaluates it the same way. **The primary comparison is pair (b): the Qwen fine-tune at 4-bit vs
Qwen3.5-9B prompting at 4-bit (exp-027)** — the fine-tuning gain on Qwen. The cross-family claim
rests on a pre-registered difference-of-differences test (family F3, below); without it, any
cross-family statement would only be qualitative sign agreement of the two gains. Secondary:
pair (a), the Qwen fine-tune at fp16 vs exp-026. No direction is asserted; the paired tests decide.
Decisions behind this design: ADR 0044.

## What is held, what moves

Held (from the **unedited** shared configs, asserted by `tests/test_exp036_plan.py` against
exp-026's own run snapshot): arm `pool_2000_imbalanced` (the pinned 2,000-question pool, sha256
`cc364662…6646`, loaded — never rebuilt, ADR 0035 amendment), 4-bit NF4 QLoRA, LoRA r=16 /
alpha 32 / dropout 0.05 on the same `target_modules` list, 2 epochs, batch 1 x grad-accum 16 (250
optimizer steps), lr 2e-4 cosine, 10 warmup steps, completion-only loss, max_length 1024, bf16,
seed 42; evaluation at fp16 + adapter, `--no-few-shot`, unguided, on the ADR 0007 pinned 600
(retrieval artifact sha256 `3f95c7dc…fdb0`), then `scripts/musique_decompositions_evaluator.py`.

Moves: `--model qwen3_5_9b`. That brings along what the model folder carries — its own
`prompt_unguided.md` rendered through the Qwen chat template with `enable_thinking=false` (the
rule exp-026 followed: the prompting arm's own prompt file, few-shot block empty) — and one
architectural consequence: the identical `target_modules` list adapts attention in only the 8
full-attention layers of Qwen3.5-9B's 32 (the other 24 are Gated DeltaNet layers with different
projection names) and the MLP in all 32. Expected trainable count 29,097,984 (derived from the HF
config, not measured) vs exp-026's 41,943,040. Details: `run_config.json` → `one_change`.

Added beside the replication: a **4-bit** evaluation of the same adapter, **mandatory and
primary**. exp-027 (the Qwen prompting comparator on this pool) ran at 4-bit and no fp16 Qwen3.5-9B
prompting arm exists on this pool, so the 4-bit eval is the precision-matched side of pair (b).
**Precision asymmetry:** the Mistral reference gain (exp-026 vs exp-015, the role exp-015 played)
is fp16 vs fp16; the Qwen gain is 4-bit vs 4-bit. The fp16 eval (exp-026's exact protocol) runs
second and serves pair (a); it is gated for fp16 overflow (at most 1 empty decomposition — exp-027's
recorded count — and 0 rows with a run of `!`, the all-NaN-logits signature), and if it fails, the
primary result stands.

## GPU requirement

>= 22 GiB visible VRAM **and** bf16 (compute capability >= 8.0). Binding step: the (secondary) fp16 evaluation,
**~18-19.5 GiB (estimate)**; training **~12-14 GiB (estimate)** — Qwen's 248k-row vocabulary makes
the fp32-upcast embedding + LM head ~8 GB on their own (exp-026's Mistral peaked at 5.035 GiB).
4-bit (primary) eval ~8 GiB (exp-027 measured 8,026 MiB for the same base without an adapter).
**Colab L4 (24 GB) is the default; A100 (40/80 GB) also passes; T4 and V100 are refused.**

## Expected wall-clock — ESTIMATES, not measurements

Scaled from exp-026 on an RTX 3090 (training 1,568.8 s for 250 steps; fp16 generation 842.3 s
over 600 queries) and exp-027 (4-bit Qwen3.5-9B generation 1,655.1 s). Factors: x1.23 parameters;
x1-2 for Qwen3.5's Gated DeltaNet on the torch fallback (as exp-027 ran it); relative to the 3090,
training A100 x0.5-0.7 / L4 x1.3-2.5, fp16 decoding (bandwidth-bound) A100 x0.5-0.6 / L4 x2.5-3.1,
4-bit eval 20-28 min 3090-equivalent then A100 x0.6 / L4 x1.3-2.

| GPU | train | eval fp16 | eval 4-bit | total incl. ~15-30 min setup | compute units (rate unverified) |
|---|---|---|---|---|---|
| A100 | 16-45 min | 9-21 min | 12-17 min | ~0.9-1.9 h | ~11-22 at ~11.8 CU/h |
| L4 | 42-160 min | 43-107 min | 26-55 min | ~2.1-5.9 h | ~10-28 at ~4.8 CU/h |
| T4 | not supported | | | | |

The CU rates are approximate public figures; the notebook prints the estimate, Colab's Resources
panel shows the real rate.

## How to run (Jahid)

1. The notebook clones `main` (exp-036 merged in PR #66); nothing to push.
2. Make sure `MyDrive/thesis-backup.tar.gz` is on Google Drive.
3. Open `notebooks/exp-036_qwen_pool_ft_colab.ipynb` in Colab (File → Open notebook → GitHub,
   branch `main`), set the runtime to **L4** (or A100), **Run all**.
4. If the runtime disconnects or the L4 runs out of memory in the fp16 eval: reconnect (switch to
   A100 if it was memory), **Run all** again. Finished steps are on Drive
   (`MyDrive/thesis-exp-036/`) and are skipped; an interrupted *training* restarts from step 0.
5. After each evaluation the notebook writes `MyDrive/thesis-exp-036/exp-036_copyback.zip` and
   prints where each file goes.

**Commit pinning:** the first run pins the commit it cloned in `MyDrive/thesis-exp-036/pinned_commit.txt`
(or the full SHA you put in `COMMIT`); every later run checks out exactly that commit, and the
training snapshot and both evaluation snapshots must carry it (gated).

The notebook: clones the branch; refuses an insufficient GPU; builds a **Python 3.13.9** venv with
uv and installs the pins of `requirements.txt` + torch 2.10.0+cu128 (the exp-026 environment) and
refuses on version drift; builds `../thesis-qav2-data` per `docs/compute.md` from the backup and
fetches `reference/musique_official` from StonyBrookNLP/musique @ `922ac98f` (sha256-checked);
verifies the pool and retrieval artifact hashes; downloads Qwen3.5-9B and refuses unless `main`
is still revision `c2022362…37b9a`; trains with gates (hash, 2,000 examples, 1461/413/126, overlap
0, 250 steps, ceiling asserted); evaluates with gates (600 rows 200/200/200, retrieval hash,
adapter prompt parity 0 mismatches, 45 anchors); every output lands on Drive as it is written.
Data extraction is decided from the VM's own disk (a marker under the data root plus the hashes),
so a VM reset re-extracts; archive members with `..`, absolute or drive-anchored names, or a
destination outside the data root / `runs/`, are refused.
No run lock: Colab is a single-user box (`docs/compute.md` "Colab" section, ADR 0044).

## After the run (laptop, CPU)

1. Unzip `exp-036_copyback.zip`. Copy `experiments_exp-036/{config.json,metrics.json,environment.json}`
   into `experiments/exp-036/`; copy `runs/exp-036/` into the main checkout's `runs/exp-036/`
   (gitignored) and `logs/` into `runs/exp-036/colab_logs/`.
2. From the repo root, with any Python that has numpy/scipy/networkx (e.g.
   `C:/Users/suuser/thesis/.venv-cpu/Scripts/python`):

```
# per-pair artifacts, the repo's standard --compare (one directory each)
python scripts/musique_decompositions_evaluator.py --compare runs/exp-036/eval_4bit/eval_per_item.json runs/exp-032/exp-027/eval/eval_per_item.json --run-dir runs/exp-036/compare_vs_exp027_4bit_PRIMARY --out-prefix compare --seed 42
python scripts/musique_decompositions_evaluator.py --compare runs/exp-036/eval_fp16/eval_per_item.json runs/exp-032/exp-026/eval/eval_per_item.json --run-dir runs/exp-036/compare_vs_exp026 --out-prefix compare --seed 42
python scripts/musique_decompositions_evaluator.py --compare runs/exp-036/eval_fp16/eval_per_item.json runs/exp-032/exp-027/eval/eval_per_item.json --run-dir runs/exp-036/compare_vs_exp027_context_fp16 --out-prefix compare --seed 42

# all pairs + the difference of differences, overall + per hop, Holm per pre-registered family
python experiments/exp-036/compare.py            # add --fp16-failed-gate if the fp16 eval failed its gate
```

`compare.py` reuses exp-029's machinery unchanged (the evaluator's paired bootstrap, exact
McNemar and paired t; Holm beside the uncorrected verdict, ADR 0009 / 0036) and writes
`experiments/exp-036/compare_metrics.json`. Comparator files are exp-032's re-parsed
re-evaluations of exp-026, exp-027 (0/600 rows changed by the re-parse) and exp-015 (7/600; the
re-parsed reading is the one a new-parser run is compared against, ADR 0039); the evaluator code is
unchanged between 5c4e442 and this branch, so every side is scored by the same evaluator code.
Families (fixed in `run_config.json` before any exp-036 p-value; ADR 0036), each over the 7
report-card terms:

| family | what | size |
|---|---|---|
| **F1_primary_overall** | pair (b), overall | 7 |
| F2_primary_per_hop | pair (b), hops 2/3/4 | 21 |
| **F3_cross_family_dod** | per-item (Qwen gain − Mistral gain) = (exp-036 4-bit − exp-027) − (exp-026 − exp-015) on the shared 600; paired bootstrap CI + paired-t p (McNemar is undefined for a difference of differences) | 7 |
| F4_secondary_vs_exp026 | pair (a), overall + hops | 28 |

The fp16-vs-exp-027 and fp16-vs-4-bit pairs are context only, in no family. `--fp16-failed-gate`
drops pair (a), the context pairs and F4, and records that; F1-F3 never drop.

3. Then the log row is completed from `metrics.json` and `compare_metrics.json` (no hand-typed
   numbers), and a `notes.md` is written.

## Framing that binds every claim from this experiment

- (b) vs exp-027 — PRIMARY: same base model, same precision (4-bit), same 600; adapter + zero-shot
  prompt vs no adapter + 5 retrieved exemplars — the fine-tuning gain on Qwen, as exp-026 vs
  exp-015 was on Mistral (that one at fp16: the two gains differ in precision).
- Cross-family: only F3 supports a statement that the gain is larger/smaller/indistinguishable
  across families; anything else is qualitative sign agreement and is worded so.
- (a) vs exp-026 — secondary: same recipe, data, protocol and precision; the base model moves, which is
  family **and** generation **and** size together (8.95B vs 7.25B runtime parameters) and LoRA
  coverage differs by architecture. It places a second model's fine-tune beside the first; it is
  not a one-factor family effect.
- Non-significance is "not distinguishable", never "equal" (ADR 0036).
