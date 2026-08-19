#!/usr/bin/env python3
"""LoRA / QLoRA fine-tuning of the decomposer on MuSiQue (issue #13).

The fine-tuned arm of the prompting-versus-fine-tuning comparison. Everything that
defines a run - base model, quantization, LoRA rank, optimizer settings, which training
data - comes from ``configs/finetune_decomposer.json``; the only CLI arguments are the arm
to train and the usual overrides.

Three training-data arms, selected with ``--arm``:

- ``pool_2000``            - 2000 examples drawn (seeded) from the enriched MuSiQue pool
- ``full_train``           - the full MuSiQue training split
- ``generalisation_2_3hop`` - 2-hop and 3-hop only, for evaluation on 4-hop

Usage::

    # the whole path on a tiny sample, no weights loaded (this is what the smoke test runs)
    python components/decomposer/train_lora.py --arm pool_2000 --dry-run

    # a real run
    python components/decomposer/train_lora.py --arm pool_2000
    python components/decomposer/train_lora.py --arm generalisation_2_3hop --quantization 4bit

A real run holds the **run lock** of ``docs/compute.md`` (the single-GPU box is shared):
acquire ``runs/run.lock`` with the experiment id + timestamp before launching, release it
when the run finishes or fails, and never clear a lock on age alone. ``--dry-run`` loads no
weights and needs no lock.

Guarantees, all of them hard:

- the base model's parameter count is printed and asserted against
  ``configs/model_limits.json`` (``src/model_size.py``) before any training step, and the
  LoRA-wrapped model is asserted again after the adapter is attached;
- training ids are asserted disjoint from the ADR 0007 evaluation ids, naming offenders;
- adapters and checkpoints are written under the gitignored ``runs/`` root, and the run
  leaves a config snapshot, a metrics JSON and a run note;
- the prompt this run trained on is written *inside* the adapter directory
  (``training_provenance.json``), so the evaluation run can refuse an adapter evaluated on a
  prompt it never saw (``run_decomposer.check_adapter_prompt_parity``).

Evaluation is **not** done here: the adapter is scored by running
``components/decomposer/run_decomposer.py --adapter <adapter dir> --no-few-shot`` over the
evaluation set and then ``scripts/musique_decompositions_evaluator.py`` (see
``scripts/compare_decomposer_arms.py``). No commercial model rates decompositions anywhere
in that path (CLAUDE.md standing constraint).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from finetune_data import (  # noqa: E402
    arm_source_path,
    assert_no_eval_overlap,
    build_training_rows,
    load_eval_ids,
    load_jsonl,
    resolve_arm,
    select_arm_examples,
    select_prompt_file,
)
from model_size import (  # noqa: E402
    assert_within_ceiling,
    count_parameters,
    load_limits,
    unasserted_note,
)
from run_artifacts import now_iso, run_id, write_run_artifacts  # noqa: E402
from run_config import load_config, load_paths, require, resolve_path, runs_path  # noqa: E402
from seeding import set_global_seed  # noqa: E402

QUANTIZATIONS = ("none", "4bit", "8bit")


# ----------------------------------------------------------------------- model + adapter


def assert_base_model_within_ceiling(
    model: Any, *, model_id: str, limits: dict[str, Any]
) -> dict[str, Any]:
    """Route the base model through the committed parameter ceiling.

    A thin wrapper so the gate is one named, testable call rather than a line buried in
    ``main``: fine-tuning loads the biggest model in the pipeline, and the ~8B ceiling is a
    standing constraint from Jahid's supervisor.
    """
    return assert_within_ceiling(
        model, component="decomposer", model_id=model_id, limits=limits
    )


def load_base_model(model_id: str, loader: dict[str, Any], quantization: str):
    """Load the base model for training (QLoRA when ``quantization`` is 4bit)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if quantization not in QUANTIZATIONS:
        raise SystemExit(f"unknown quantization {quantization!r} (expected one of {list(QUANTIZATIONS)})")
    if not torch.cuda.is_available():
        raise SystemExit(
            "[finetune] no CUDA device visible. Training needs the GPU of docs/compute.md; "
            "use --dry-run to exercise the data and prompt path on CPU."
        )

    trust_remote_code = bool(require(loader, "trust_remote_code"))
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        # No new token is added: a fresh pad embedding would change the parameter count and
        # is unnecessary for causal SFT, where padding is masked out anyway.
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "device_map": require(loader, "device_map"),
    }
    attn = loader.get("attn_implementation")
    if attn:
        model_kwargs["attn_implementation"] = attn

    if quantization in ("4bit", "8bit"):
        from transformers import BitsAndBytesConfig

        if quantization == "4bit":
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=require(loader, "bnb_4bit_quant_type"),
                bnb_4bit_compute_dtype=getattr(torch, require(loader, "bnb_4bit_compute_dtype")),
                bnb_4bit_use_double_quant=bool(require(loader, "bnb_4bit_use_double_quant")),
            )
        else:
            bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
        model_kwargs["quantization_config"] = bnb_cfg
    else:
        model_kwargs["dtype"] = getattr(torch, require(loader, "dtype"))

    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    except Exception as exc:  # noqa: BLE001 - re-raised as a SystemExit with a hint
        raise SystemExit(
            f"[finetune] failed to load base model {model_id}: {exc}\n"
            "Hint: on 24 GiB, a 7B-9B base needs --quantization 4bit (QLoRA)."
        ) from exc
    return tokenizer, model


def build_lora_config(lora_cfg: dict[str, Any]):
    """A peft ``LoraConfig`` from the committed config. Separate so it is testable."""
    from peft import LoraConfig

    return LoraConfig(
        r=int(require(lora_cfg, "r")),
        lora_alpha=int(require(lora_cfg, "lora_alpha")),
        lora_dropout=float(require(lora_cfg, "lora_dropout")),
        bias=require(lora_cfg, "bias"),
        task_type=require(lora_cfg, "task_type"),
        target_modules=list(require(lora_cfg, "target_modules")),
    )


def build_sft_config(training_cfg: dict[str, Any], *, output_dir: Path, seed: int):
    """A trl ``SFTConfig`` from the committed config. Every field comes from config.

    Separate from ``main`` so a test can build it without a GPU: that is what catches a
    renamed or removed trl field before it costs a training run.
    """
    from trl import SFTConfig

    return SFTConfig(
        output_dir=str(output_dir),
        seed=seed,
        data_seed=seed,
        num_train_epochs=float(require(training_cfg, "num_train_epochs")),
        max_steps=int(require(training_cfg, "max_steps")),
        per_device_train_batch_size=int(require(training_cfg, "per_device_train_batch_size")),
        gradient_accumulation_steps=int(require(training_cfg, "gradient_accumulation_steps")),
        learning_rate=float(require(training_cfg, "learning_rate")),
        lr_scheduler_type=require(training_cfg, "lr_scheduler_type"),
        warmup_steps=int(require(training_cfg, "warmup_steps")),
        weight_decay=float(require(training_cfg, "weight_decay")),
        max_grad_norm=float(require(training_cfg, "max_grad_norm")),
        optim=require(training_cfg, "optim"),
        max_length=int(require(training_cfg, "max_length")),
        packing=bool(require(training_cfg, "packing")),
        completion_only_loss=bool(require(training_cfg, "completion_only_loss")),
        gradient_checkpointing=bool(require(training_cfg, "gradient_checkpointing")),
        bf16=bool(require(training_cfg, "bf16")),
        fp16=bool(require(training_cfg, "fp16")),
        logging_steps=int(require(training_cfg, "logging_steps")),
        save_strategy=require(training_cfg, "save_strategy"),
        save_total_limit=int(require(training_cfg, "save_total_limit")),
        report_to=require(training_cfg, "report_to"),
        dataloader_num_workers=int(require(training_cfg, "dataloader_num_workers")),
    )


def attach_lora(model: Any, lora_cfg: dict[str, Any], *, quantization: str, gradient_checkpointing: bool):
    """Wrap the base model in a LoRA adapter (k-bit prepared when quantized)."""
    from peft import get_peft_model, prepare_model_for_kbit_training

    if quantization in ("4bit", "8bit"):
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=gradient_checkpointing
        )
    peft_config = build_lora_config(lora_cfg)
    return get_peft_model(model, peft_config), peft_config


def trainable_parameter_record(model: Any) -> dict[str, Any]:
    """Trainable vs total parameters. The LoRA claim ('only the adapter trains') measured.

    The denominator goes through :func:`model_size.count_parameters` rather than a raw
    ``numel`` sum. Under 4-bit loading the base weights are ``bitsandbytes`` ``Params4bit``,
    whose storage holds two parameters per element, so a raw sum reports roughly half the
    model - and would contradict ``base_model_size.parameter_count`` in the same metrics JSON,
    which is the count the ~8B ceiling is asserted against. The numerator stays a direct
    ``numel`` sum: what requires grad is the un-quantized LoRA tensors.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = count_parameters(model)
    return {
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_percent": (100.0 * trainable / total) if total else None,
        "counting_note": (
            "total_parameters is src/model_size.py count_parameters (transformers' "
            "num_parameters, which counts a packed 4-bit Params4bit as the parameters it "
            "stores), so it matches base_model_size.parameter_count. trainable_parameters is "
            "a numel sum over requires_grad parameters (the LoRA tensors, not quantized)."
        ),
    }


#: Prompt provenance, written *inside* the adapter directory. The training run's config
#: snapshot next to it says the same thing, but an adapter is a directory that gets copied
#: and moved; this file travels with the weights, and it is what
#: ``run_decomposer.check_adapter_prompt_parity`` reads to refuse evaluating an adapter on a
#: prompt it was never trained on. Same filename as
#: ``run_decomposer.ADAPTER_PROVENANCE_FILE``.
ADAPTER_PROVENANCE_FILE = "training_provenance.json"


def adapter_provenance(snapshot: dict[str, Any]) -> dict[str, Any]:
    """The prompt this run trained on, in the shape the evaluation-side guard reads.

    ``few_shot_examples_in_prompt`` is derived from the rendered ``few_shot_examples`` string
    (this script trains zero-shot, i.e. an empty block) so the guard compares a boolean
    rather than re-deriving the rule.
    """
    prompt = dict(require(snapshot, "prompt"))
    prompt["few_shot_examples_in_prompt"] = bool(str(prompt.get("few_shot_examples", "")).strip())
    return {
        "script": snapshot.get("script"),
        "created_utc": snapshot.get("created_utc"),
        "run_id": snapshot.get("run_id"),
        "arm": snapshot.get("arm"),
        "model": snapshot.get("model"),
        "model_id": snapshot.get("model_id"),
        "prompt": prompt,
        "_note": (
            "What this adapter was trained on, for run_decomposer.py's adapter prompt-parity "
            "guard. The base model is checked separately, from adapter_config.json's "
            "base_model_name_or_path."
        ),
    }


def write_adapter_provenance(adapter_dir: Path, snapshot: dict[str, Any]) -> Path:
    """Write :func:`adapter_provenance` into the adapter directory. Returns the path."""
    path = Path(adapter_dir) / ADAPTER_PROVENANCE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(adapter_provenance(snapshot), indent=2, default=str) + "\n", encoding="utf-8"
    )
    return path


# -------------------------------------------------------------------------------- args


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--arm", default=None, help="Training-data arm (key of the config's 'arms')")
    p.add_argument("--config", default="finetune_decomposer.json", help="Committed config")
    p.add_argument("--model", default=None, help="Model folder under components/decomposer/models/")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--quantization", default=None, choices=list(QUANTIZATIONS))
    p.add_argument("--limit", type=int, default=None, help="Cap the training examples (smoke runs)")
    p.add_argument("--max-steps", type=int, default=None, help="Override training.max_steps")
    p.add_argument("--output-root", default=None)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the dataset, assert the overlap, write artifacts - no weights, no steps.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    cfg = load_config(args.config)
    paths_cfg = load_paths(require(cfg, "paths_config"))
    limits = load_limits(require(cfg, "model_limits_config"))
    decomposer_cfg = load_config(require(cfg, "decomposer_config"))

    arm_name = args.arm if args.arm is not None else require(cfg, "default_arm")
    arm = resolve_arm(cfg, arm_name)
    seed = args.seed if args.seed is not None else int(require(cfg, "seed"))
    model_folder = args.model or require(cfg, "model_folder")
    quantization = args.quantization or require(cfg, "quantization")
    if quantization not in QUANTIZATIONS:
        raise SystemExit(f"unknown quantization {quantization!r} (expected one of {list(QUANTIZATIONS)})")

    models_root = resolve_path(require(paths_cfg, "repo.decomposer_models_dir"), _REPO_ROOT)
    model_dir = models_root / model_folder
    if not model_dir.is_dir():
        raise SystemExit(f"model folder not found: {model_dir}")
    model_cfg = load_config(model_dir / "config.json")
    model_id = require(model_cfg, "model_id")

    prompt_cfg = dict(require(cfg, "prompt"))
    training_cfg = dict(require(cfg, "training"))
    if args.max_steps is not None:
        training_cfg["max_steps"] = int(args.max_steps)
    loader_cfg = dict(require(cfg, "loader"))
    lora_cfg = dict(require(cfg, "lora"))

    current_run_id = run_id()
    seeded = set_global_seed(seed)

    output_root = (
        Path(args.output_root)
        if args.output_root is not None
        else runs_path(paths_cfg, require(cfg, "output_subdir"), arm_name, model_folder)
    )
    output_dir = output_root / current_run_id
    adapter_dir = output_dir / require(cfg, "adapter_dirname")

    # ---- prompt template: the prompting arm's own file, few-shot block empty ----
    guided = bool(require(prompt_cfg, "guided"))
    prompt_file = select_prompt_file(model_cfg, guided=guided)
    prompt_path = model_dir / prompt_file
    if not prompt_path.exists():
        raise SystemExit(f"prompt file not found: {prompt_path}")
    prompt_template = prompt_path.read_text(encoding="utf-8")
    prompt_style = require(model_cfg, "prompt_style")
    chat_marker = (
        require(model_cfg, "chat_template.split_marker") if prompt_style == "chat_template" else None
    )
    enable_thinking = (
        bool(require(model_cfg, "chat_template.enable_thinking"))
        if prompt_style == "chat_template"
        else False
    )
    unguided_hop_placeholder = require(decomposer_cfg, "unguided_hop_placeholder")

    # ---- training data ----
    source_path = arm_source_path(arm, paths_cfg)
    if not source_path.exists():
        raise SystemExit(
            f"training source not found: {source_path} (arm {arm_name!r}); set "
            f"datasets.{arm.get('train_source_key')} or data_root in the paths config"
        )
    rows = load_jsonl(source_path)
    if not rows:
        raise SystemExit(f"training source is empty: {source_path}")

    examples, selection = select_arm_examples(
        arm,
        rows,
        require(cfg, "data"),
        seed=seed,
        max_reported_ids=int(require(cfg, "overlap_check.max_reported_load_ids")),
        limit=args.limit,
    )
    if not examples:
        raise SystemExit(
            f"arm {arm_name!r} selected 0 training examples from {source_path} "
            f"(train_hops={require(arm, 'train_hops')}, max_examples={require(arm, 'max_examples')})"
        )

    # The nested block carries the config path along, so a count-mismatch error names the
    # file whose expected counts it is quoting.
    eval_set_cfg = {**require(cfg, "eval_set"), "_config_path": cfg.get("_config_path")}
    eval_ids, eval_set_record = load_eval_ids(paths_cfg, eval_set_cfg)
    overlap_record = assert_no_eval_overlap(
        examples, eval_ids, max_reported=int(require(cfg, "overlap_check.max_reported_ids"))
    )
    print(
        f"[finetune] arm={arm_name} examples={len(examples)} "
        f"hops={selection['selected_hop_counts']} eval_ids={eval_set_record['num_ids']} "
        f"overlap={overlap_record['overlap_count']}"
    )

    snapshot = {
        "script": Path(__file__).name,
        "created_utc": now_iso(),
        "run_id": current_run_id,
        "component": "decomposer",
        "task": "lora_finetune",
        "arm": arm_name,
        "arm_spec": arm,
        "model": model_folder,
        "model_id": model_id,
        "quantization": quantization,
        "seed": seed,
        "seeded": seeded,
        "prompt": {
            **prompt_cfg,
            "prompt_file": prompt_file,
            "prompt_path": str(prompt_path),
            "prompt_style": prompt_style,
            "unguided_hop_placeholder": unguided_hop_placeholder,
        },
        "loader": loader_cfg,
        "lora": lora_cfg,
        "training": training_cfg,
        "eval_set": eval_set_record,
        "training_source": str(source_path),
        "limit": args.limit,
        "config": cfg.get("_config_path"),
        "model_config": model_cfg.get("_config_path"),
        "decomposer_config": decomposer_cfg.get("_config_path"),
        "output_root": str(output_root),
        "adapter_dir": str(adapter_dir),
        "dry_run": args.dry_run,
    }
    print(f"Starting LoRA run {current_run_id} (arm={arm_name}, dry_run={args.dry_run})")
    print(json.dumps(snapshot, indent=2, default=str))

    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- base model (skipped on a dry run) ----
    tokenizer = None
    model = None
    base_size_record = unasserted_note("decomposer", model_id)
    if not args.dry_run:
        print(f"Loading base model {model_id} (quantization={quantization}) ...")
        tokenizer, model = load_base_model(model_id, loader_cfg, quantization)
        base_size_record = assert_base_model_within_ceiling(
            model, model_id=model_id, limits=limits
        )

    # ---- formatted dataset ----
    training_rows, format_record = build_training_rows(
        examples,
        template=prompt_template,
        prompt_style=prompt_style,
        prompt_cfg=prompt_cfg,
        unguided_hop_placeholder=unguided_hop_placeholder,
        chat_marker=chat_marker,
        tokenizer=tokenizer,
        enable_thinking=enable_thinking,
    )

    examples_logged = int(require(cfg, "formatted_examples_logged"))
    sample_path = output_dir / "formatted_examples.txt"
    sample_blocks = []
    for row in training_rows[:examples_logged]:
        sample_blocks.append(
            f"--- row_id={row['row_id']} hop={row['hop']} ---\n"
            f"--- prompt ---\n{row['prompt']}\n"
            f"--- completion ---\n{row['completion']}\n"
        )
    sample_path.write_text("\n".join(sample_blocks), encoding="utf-8")

    # ---- training ----
    lora_size_record: dict[str, Any] | None = None
    trainable_record: dict[str, Any] | None = None
    train_result: dict[str, Any] | None = None
    peak_gpu_bytes: int | None = None
    wall_clock_seconds: float | None = None
    adapter_provenance_path: Path | None = None

    if not args.dry_run:
        import torch
        from datasets import Dataset
        from trl import SFTTrainer

        model, peft_config = attach_lora(
            model,
            lora_cfg,
            quantization=quantization,
            gradient_checkpointing=bool(require(training_cfg, "gradient_checkpointing")),
        )
        # The adapter adds parameters, so the ceiling is asserted again on what actually
        # trains - not only on the base model.
        lora_size_record = assert_within_ceiling(
            model, component="decomposer", model_id=f"{model_id}+lora", limits=limits
        )
        trainable_record = trainable_parameter_record(model)
        print(
            f"[finetune] trainable {trainable_record['trainable_parameters']:,} / "
            f"{trainable_record['total_parameters']:,} parameters "
            f"({trainable_record['trainable_percent']:.4f}%)"
        )

        dataset = Dataset.from_list(
            [{"prompt": r["prompt"], "completion": r["completion"]} for r in training_rows]
        )
        sft_args = build_sft_config(
            training_cfg, output_dir=output_dir / "checkpoints", seed=seed
        )
        trainer = SFTTrainer(
            model=model,
            args=sft_args,
            train_dataset=dataset,
            processing_class=tokenizer,
        )
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        result = trainer.train()
        wall_clock_seconds = time.time() - t0
        if torch.cuda.is_available():
            peak_gpu_bytes = int(torch.cuda.max_memory_allocated())

        trainer.save_model(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        adapter_provenance_path = write_adapter_provenance(adapter_dir, snapshot)
        print(f"[finetune] adapter prompt provenance -> {adapter_provenance_path}")
        train_result = {
            "global_step": int(result.global_step),
            "training_loss": float(result.training_loss) if result.training_loss is not None else None,
            "metrics": {k: v for k, v in (result.metrics or {}).items()},
            "peft_config": {
                "r": peft_config.r,
                "lora_alpha": peft_config.lora_alpha,
                "lora_dropout": peft_config.lora_dropout,
                "target_modules": sorted(peft_config.target_modules),
            },
        }

    metrics: dict[str, Any] = {
        "dry_run": args.dry_run,
        "arm": arm_name,
        "model": model_folder,
        "model_id": model_id,
        "quantization": quantization,
        "seed": seed,
        "seeded": seeded,
        "training_source": str(source_path),
        "dataset": {
            "num_examples": len(training_rows),
            "hop_counts": selection["selected_hop_counts"],
            **format_record,
        },
        "selection": selection,
        "eval_set": eval_set_record,
        "eval_overlap": overlap_record,
        "base_model_size": base_size_record,
        "lora_model_size": lora_size_record,
        "trainable_parameters": trainable_record,
        "training": training_cfg,
        "train_result": train_result,
        "wall_clock_seconds": wall_clock_seconds,
        "peak_gpu_memory_bytes": peak_gpu_bytes,
        "peak_gpu_memory_gib": (
            round(peak_gpu_bytes / (1024**3), 3) if peak_gpu_bytes is not None else None
        ),
        "adapter_path": str(adapter_dir) if not args.dry_run else None,
        "adapter_provenance_path": (
            str(adapter_provenance_path) if adapter_provenance_path else None
        ),
        "formatted_examples_path": str(sample_path),
        "decomposition_quality": None,
        "decomposition_quality_note": (
            "unmeasured here by design: this script only trains. Score the adapter with "
            "components/decomposer/run_decomposer.py --adapter <adapter dir> --no-few-shot "
            "followed by scripts/musique_decompositions_evaluator.py "
            "(scripts/compare_decomposer_arms.py wires both, plus --compare against the "
            "prompting arm)."
        ),
    }
    if args.dry_run:
        metrics["dry_run_note"] = (
            "no weights were loaded and no optimizer step ran, so the parameter ceiling is "
            "not asserted, trainable_parameters is unmeasured and peak_gpu_memory_bytes is "
            "unmeasured. The data selection, the overlap assertion and the prompt/completion "
            "formatting above ARE exercised."
        )

    write_run_artifacts(
        output_dir,
        config_snapshot=snapshot,
        metrics=metrics,
        note_title=f"Decomposer LoRA {'dry run' if args.dry_run else 'run'} - {current_run_id}",
        note_lines=[
            f"- Arm: `{arm_name}` (source `{source_path}`)",
            f"- Base model: `{model_folder}` / `{model_id}`; quantization: {quantization}"
            + ("" if not args.dry_run else " (not loaded)"),
            f"- Prompt: `{prompt_path}` (style {prompt_style}, guided {guided}, "
            f"few-shot examples in prompt: {format_record['few_shot_examples_in_prompt']})",
            f"- Examples: {len(training_rows)} (hops {selection['selected_hop_counts']}); seed {seed}",
            f"- Evaluated on hops {require(arm, 'eval_hops')} of the ADR 0007 set "
            f"({eval_set_record['num_ids']} ids across hops {eval_set_record['hops']}; "
            f"{eval_set_record['expected_ids_per_hop']} per hop / "
            f"{eval_set_record['expected_total_ids']} total asserted from "
            f"{eval_set_record['expected_counts_source']})",
            f"- Train/eval id overlap: {overlap_record['overlap_count']} (asserted zero)",
            (
                f"- Base parameters: {base_size_record['parameter_count']:,} "
                f"(ceiling {base_size_record['parameter_ceiling']:,})"
                if base_size_record["ceiling_asserted"]
                else "- Parameter ceiling: not asserted (no model was loaded)."
            ),
            (
                f"- Trainable: {trainable_record['trainable_parameters']:,} of "
                f"{trainable_record['total_parameters']:,} "
                f"({trainable_record['trainable_percent']:.4f}%)"
                if trainable_record
                else "- Trainable parameters: unmeasured (dry run)."
            ),
            (
                f"- Peak GPU memory: {metrics['peak_gpu_memory_gib']} GiB; wall clock "
                f"{wall_clock_seconds:.1f}s"
                if peak_gpu_bytes is not None
                else "- Peak GPU memory: unmeasured (dry run)."
            ),
            f"- Adapter: `{adapter_dir}`" if not args.dry_run else "- Adapter: not written (dry run).",
            f"- Formatted examples: `{sample_path}`",
        ],
    )
    print(f"\nRun artifacts in: {output_dir}")


if __name__ == "__main__":
    main()
