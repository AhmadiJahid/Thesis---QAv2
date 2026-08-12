#!/usr/bin/env python3
"""Router component: classify a question's hop count.

One runner for every model. v1 kept ten near-identical copies of ``router.py``
(one per model folder); they differed only in the model id, the generation length,
the tokenizer/loader flags and the response-parsing rules. Those differences are
now fields in ``components/router/models/<model>/config.json`` and the code lives
here once. Prompts stay per-model and byte-identical to v1.

Usage::

    python components/router/run_router.py --model qwen2_5_0_5b
    python components/router/run_router.py --model qwen2_5_0_5b --prompt-file prompt_zero_shot.md
    python components/router/run_router.py --model qwen2_5_0_5b --dry-run   # no model load

Every run writes a config snapshot, a metrics JSON and a run note under the
configured runs root, and asserts the model's parameter count against the router
ceiling in ``configs/model_limits.json``.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from model_size import assert_within_ceiling, load_limits, unasserted_note  # noqa: E402
from run_artifacts import now_iso, run_id, write_run_artifacts  # noqa: E402
from run_config import (  # noqa: E402
    load_config,
    load_paths,
    optional,
    require,
    resolve_path,
    runs_path,
)
from seeding import new_rng, set_global_seed  # noqa: E402

_THINK_RX = re.compile(r"<think>.*?</think>", re.DOTALL)

_NUMBER_WORDS = {"ONE": 1, "TWO": 2, "THREE": 3}


def load_questions(file_path: Path) -> list[str]:
    """Load questions from a text file (one per line)."""
    if not file_path.exists():
        print(f"Warning: {file_path} not found.")
        return []
    with file_path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def build_prompt(template: str, question: str) -> str:
    """Fill the question placeholder. Handles both {question} and {{question}}."""
    if "{{question}}" in template:
        return template.replace("{{question}}", question)
    return template.format(question=question)


def parse_hop_response(response: str, question: str, parsing: dict) -> int:
    """Extract a hop count from the model response, per the model's parsing config.

    Mirrors the three parsing variants that existed across v1's router copies:
    an answer-prefix regex (``A:`` or ``Output|A:``), a first-digit fallback, then
    either hop-word labels ("2-hop", "two hop") or number-word labels (ONE/TWO/THREE).
    """
    hops: list[int] = parsing["hops"]
    digit_class = "[" + "".join(str(h) for h in hops) + "]"
    default_hop = int(require(parsing, "default_hop"))

    clean = _THINK_RX.sub("", response, count=0).strip()

    answer_regex = optional(parsing, "answer_regex")
    if answer_regex:
        flags = re.IGNORECASE if parsing.get("answer_regex_ignorecase") else 0
        if match := re.search(answer_regex, clean, flags):
            return int(match.group(1))

    if match := re.search(digit_class, clean):
        return int(match.group())

    fallback = require(parsing, "fallback_labels")
    if fallback == "hop_words":
        lower = clean.lower()
        for hop, words in (
            (1, ("1-hop", "one hop")),
            (2, ("2-hop", "two hop")),
            (3, ("3-hop", "three hop")),
        ):
            if hop in hops and any(w in lower for w in words):
                return hop
    elif fallback == "number_words":
        upper = clean.upper()
        for label, value in _NUMBER_WORDS.items():
            if value in hops and label in upper:
                return value
    else:
        raise SystemExit(f"unknown parsing.fallback_labels: {fallback!r}")

    if numbers := re.findall(digit_class, clean):
        return int(numbers[0])

    print(
        f"Warning: no valid hop count in response for: '{question[:50]}...'. "
        f"Defaulting to {default_hop}."
    )
    if parsing.get("debug_print_response"):
        print(f"DEBUG: Full model response was: '{response}'")
    return default_hop


def classify_hop_count(
    question: str,
    model,
    tokenizer,
    device: str,
    prompt_template: str,
    generation: dict,
    parsing: dict,
) -> int:
    import torch

    prompt = build_prompt(prompt_template, question)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=int(require(generation, "max_new_tokens")),
            temperature=float(require(generation, "temperature")),
            top_p=float(require(generation, "top_p")),
            do_sample=bool(require(generation, "do_sample")),
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()

    try:
        return parse_hop_response(response, question, parsing)
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover - defensive, matches v1 behaviour
        default_hop = int(require(parsing, "default_hop"))
        print(f"Error parsing response for '{question[:50]}...': {exc}. Defaulting to {default_hop}.")
        return default_hop


def compute_metrics(
    predictions: list[int],
    all_expected: list[int],
    run_seed: int,
    model_id: str,
    current_run_id: str,
    hops: list[int],
    run_idx: int | None = None,
) -> dict:
    """Metrics for one run."""
    correct = sum(1 for p, e in zip(predictions, all_expected) if p == e)
    accuracy = correct / len(predictions) if predictions else 0.0
    per_hop: dict[str, float | int] = {}
    for h in hops:
        total = all_expected.count(h)
        match = sum(1 for p, e in zip(predictions, all_expected) if e == h and p == e)
        per_hop[f"hop_{h}_accuracy"] = match / total if total > 0 else 0.0
        per_hop[f"hop_{h}_total"] = total
    metrics = {
        "overall_accuracy": accuracy,
        "total_questions": len(predictions),
        "correct_predictions": correct,
        **per_hop,
        "seed": run_seed,
        "model": model_id,
        "run_id": current_run_id,
    }
    if run_idx is not None:
        metrics["run_index"] = run_idx
    return metrics


def load_model(model_id: str, loader: dict, device: str):
    """Load tokenizer + model with this model's loader flags."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if loader.get("print_gpu_info") and device == "cuda":
        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU: {name} (total VRAM: {total:.1f} GB)")

    trust_remote_code = bool(require(loader, "trust_remote_code"))
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        use_fast=bool(require(loader, "use_fast_tokenizer")),
    )

    dtype_name = require(loader, "cuda_dtype") if device == "cuda" else require(loader, "cpu_dtype")
    dtype = getattr(torch, dtype_name)
    # `dtype=` is the forward-compatible spelling; transformers 5.x accepts both and
    # maps the older `torch_dtype=` onto it.
    model_kwargs = {"trust_remote_code": trust_remote_code, "dtype": dtype}
    if device == "cuda":
        model_kwargs["device_map"] = require(loader, "device_map_cuda")

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    if device != "cuda":
        model = model.to(device)
    model.eval()
    return tokenizer, model


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Model folder under components/router/models/")
    p.add_argument("--config", default="router.json", help="Shared router config (default: configs/router.json)")
    p.add_argument("--prompt-file", default=None, help="Override the model's prompt file (e.g. prompt_zero_shot.md)")
    p.add_argument("--seed", type=int, default=None, help="Override the config seed")
    p.add_argument("--num-runs", type=int, default=None, help="Override config num_runs (seeds seed, seed+1, ...)")
    p.add_argument("--sample-size-per-hop", type=int, default=None, help="Override config sample_size_per_hop")
    p.add_argument("--output-root", default=None, help="Override the run output root")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Assemble prompts and write artifacts without loading a model or generating.",
    )
    p.add_argument(
        "--dry-run-limit",
        type=int,
        default=5,
        help="Questions to assemble prompts for in --dry-run (default: 5).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    cfg = load_config(args.config)
    paths_cfg = load_paths(require(cfg, "paths_config"))
    limits = load_limits(require(cfg, "model_limits_config"))

    models_root = resolve_path(require(paths_cfg, "repo.router_models_dir"), _REPO_ROOT)
    model_dir = models_root / args.model
    if not model_dir.is_dir():
        raise SystemExit(f"model folder not found: {model_dir}")
    model_cfg = load_config(model_dir / "config.json")

    seed = args.seed if args.seed is not None else int(require(cfg, "seed"))
    num_runs = args.num_runs if args.num_runs is not None else int(require(cfg, "num_runs"))
    sample_size = (
        args.sample_size_per_hop
        if args.sample_size_per_hop is not None
        else require(cfg, "sample_size_per_hop")
    )
    hops = [int(h) for h in require(cfg, "hops")]

    prompt_file = args.prompt_file or require(model_cfg, "prompt_file")
    zero_shot = require(cfg, "zero_shot_prompt_marker") in prompt_file
    if args.output_root is not None:
        output_root = Path(args.output_root)
    else:
        subdir = (
            require(cfg, "output_subdir_zero_shot") if zero_shot
            else require(cfg, "output_subdir_few_shot")
        )
        output_root = runs_path(paths_cfg, subdir)

    device = "cpu"
    if not args.dry_run:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    generation = dict(require(model_cfg, "generation"))
    loader = dict(require(model_cfg, "loader"))
    parsing = dict(require(model_cfg, "parsing"))
    parsing["hops"] = hops

    current_run_id = run_id()
    seeded = set_global_seed(seed)

    snapshot = {
        "script": Path(__file__).name,
        "created_utc": now_iso(),
        "run_id": current_run_id,
        "component": "router",
        "model": args.model,
        "model_id": require(model_cfg, "model_id"),
        "model_name": require(model_cfg, "model_name"),
        "prompt_file": prompt_file,
        "zero_shot": zero_shot,
        "seed": seed,
        "seeded": seeded,
        "num_runs": num_runs,
        "sample_size_per_hop": sample_size,
        "hops": hops,
        "device": device,
        "generation": generation,
        "loader": loader,
        "parsing": {k: v for k, v in parsing.items() if k != "hops"},
        "shared_config": cfg.get("_config_path"),
        "model_config": model_cfg.get("_config_path"),
        "output_root": str(output_root),
        "dry_run": args.dry_run,
    }
    print(f"Starting router run {current_run_id} (num_runs={num_runs}, dry_run={args.dry_run})")
    print(json.dumps(snapshot, indent=2, default=str))

    # Prompt template: model folder first, then the shared models dir (v1 behaviour).
    prompt_path = model_dir / prompt_file
    if not prompt_path.exists():
        prompt_path = model_dir.parent / prompt_file
    if not prompt_path.exists():
        raise SystemExit(
            f"prompt file not found: {prompt_file} in {model_dir} or {model_dir.parent}"
        )
    prompt_template = prompt_path.read_text(encoding="utf-8")
    snapshot["prompt_path"] = str(prompt_path)

    # Questions per hop, from the configured data root.
    data_root = Path(paths_cfg["data_root_resolved"])
    template = require(paths_cfg, "datasets." + require(cfg, "questions_template_key"))
    questions_by_hop: dict[int, list[str]] = {}
    for hop in hops:
        path = resolve_path(template.format(hop=hop), data_root)
        questions_by_hop[hop] = load_questions(path)

    if sample_size:
        # Seeded before sampling: v1 sampled before calling set_seed, so its
        # --sample_size draws were not reproducible.
        rng = new_rng(seed)
        for hop in hops:
            pool = questions_by_hop[hop]
            questions_by_hop[hop] = rng.sample(pool, min(len(pool), int(sample_size)))

    all_questions: list[str] = []
    all_expected: list[int] = []
    for hop in hops:
        all_questions.extend(questions_by_hop[hop])
        all_expected.extend([hop] * len(questions_by_hop[hop]))

    if not all_questions:
        raise SystemExit(
            f"no questions loaded from {data_root} "
            f"(expected {template} for hops {hops}); set data_root in configs/paths.json"
        )

    counts = {f"{hop}hop": len(questions_by_hop[hop]) for hop in hops}
    print(f"Loaded {len(all_questions)} questions {counts}")

    output_dir = output_root / current_run_id

    if args.dry_run:
        sample = all_questions[: max(0, args.dry_run_limit)]
        prompts = [build_prompt(prompt_template, q) for q in sample]
        prompts_dir = output_dir / "prompts_log"
        prompts_dir.mkdir(parents=True, exist_ok=True)
        for i, (q, prompt) in enumerate(zip(sample, prompts), start=1):
            (prompts_dir / f"prompt_idx{i:04d}.txt").write_text(
                f"--- Question ---\n{q}\n\n--- Prompt ---\n{prompt}\n", encoding="utf-8"
            )
        metrics = {
            "dry_run": True,
            "total_questions_loaded": len(all_questions),
            "questions_per_hop": counts,
            "prompts_assembled": len(prompts),
            "prompt_chars_mean": (
                sum(len(p) for p in prompts) / len(prompts) if prompts else 0
            ),
            "model_size": unasserted_note("router", require(model_cfg, "model_id")),
            "accuracy_metrics": None,
            "accuracy_metrics_note": "unmeasured: --dry-run does not load a model or generate",
        }
        write_run_artifacts(
            output_dir,
            config_snapshot=snapshot,
            metrics=metrics,
            note_title=f"Router dry run - {current_run_id}",
            note_lines=[
                f"- Model folder: `{args.model}` (model not loaded)",
                f"- Prompt: `{prompt_path}`",
                f"- Questions loaded: {len(all_questions)} {counts}",
                f"- Prompts assembled: {len(prompts)} (logged under `{prompts_dir}`)",
                "- Accuracy: unmeasured (dry run).",
                "- Parameter ceiling: not asserted (no model was loaded).",
            ],
        )
        print(f"\nDry-run artifacts under: {output_dir}")
        return

    model_id = require(model_cfg, "model_id")
    print(f"Loading model: {model_id} on {device} ...")
    tokenizer, model = load_model(model_id, loader, device)
    size_record = assert_within_ceiling(
        model, component="router", model_id=model_id, limits=limits
    )

    all_runs_metrics: list[dict] = []
    all_runs_predictions: list[list[int]] = []
    progress_every = int(require(cfg, "progress_every"))

    for run_idx in range(num_runs):
        run_seed = seed + run_idx
        set_global_seed(run_seed)
        print(f"\n--- Run {run_idx + 1}/{num_runs} (seed={run_seed}) ---")
        predictions: list[int] = []
        for i, question in enumerate(all_questions):
            if (i + 1) % progress_every == 0:
                print(f"Processed {i + 1}/{len(all_questions)}...")
            predictions.append(
                classify_hop_count(
                    question, model, tokenizer, device, prompt_template, generation, parsing
                )
            )
        all_runs_predictions.append(predictions)
        run_metrics = compute_metrics(
            predictions, all_expected, run_seed, model_id, current_run_id, hops, run_idx=run_idx
        )
        all_runs_metrics.append(run_metrics)
        print(f"Run {run_idx + 1} accuracy: {run_metrics['overall_accuracy']:.4f}")

    if num_runs == 1:
        metrics = dict(all_runs_metrics[0])
    else:
        metrics = {
            "num_runs": num_runs,
            "overall_accuracy_mean": statistics.mean(m["overall_accuracy"] for m in all_runs_metrics),
            "overall_accuracy_std": statistics.stdev(m["overall_accuracy"] for m in all_runs_metrics),
            "model": model_id,
            "run_id": current_run_id,
        }
        for hop in hops:
            accs = [m[f"hop_{hop}_accuracy"] for m in all_runs_metrics]
            metrics[f"hop_{hop}_accuracy_mean"] = statistics.mean(accs)
            metrics[f"hop_{hop}_accuracy_std"] = statistics.stdev(accs)
        metrics["per_run"] = all_runs_metrics

    metrics["model_size"] = size_record
    metrics["questions_per_hop"] = counts
    metrics["dry_run"] = False

    print("\n" + "=" * 30)
    if num_runs == 1:
        print(f"Accuracy: {metrics['overall_accuracy']:.4f}")
        for hop in hops:
            print(f"{hop}-hop: {metrics[f'hop_{hop}_accuracy']:.4f}")
    else:
        print(
            f"Accuracy (mean +/- std): {metrics['overall_accuracy_mean']:.4f} "
            f"+/- {metrics['overall_accuracy_std']:.4f}"
        )
        for hop in hops:
            print(
                f"{hop}-hop: {metrics[f'hop_{hop}_accuracy_mean']:.4f} "
                f"+/- {metrics[f'hop_{hop}_accuracy_std']:.4f}"
            )
    print("=" * 30)

    detailed = [
        {"question": q, "expected": e, "predicted": p, "correct": p == e}
        for q, e, p in zip(all_questions, all_expected, all_runs_predictions[0])
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    detailed_name = "detailed_results.json" if num_runs == 1 else "detailed_results_run_0.json"
    (output_dir / detailed_name).write_text(
        json.dumps(detailed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    headline = (
        f"- Overall accuracy: {metrics['overall_accuracy']:.4f}"
        if num_runs == 1
        else (
            f"- Overall accuracy (mean +/- std over {num_runs} runs): "
            f"{metrics['overall_accuracy_mean']:.4f} +/- {metrics['overall_accuracy_std']:.4f}"
        )
    )
    write_run_artifacts(
        output_dir,
        config_snapshot=snapshot,
        metrics=metrics,
        note_title=f"Router run - {current_run_id}",
        note_lines=[
            f"- Model: `{model_id}` ({size_record['parameter_count']:,} parameters, "
            f"ceiling {size_record['parameter_ceiling']:,})",
            f"- Prompt: `{prompt_path}`",
            f"- Seed: {seed} (runs use seed, seed+1, ...)",
            f"- Questions: {len(all_questions)} {counts}",
            headline,
            f"- Detailed predictions: `{output_dir / detailed_name}`",
        ],
    )
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
