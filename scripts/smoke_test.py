#!/usr/bin/env python3
"""End-to-end smoke test on the synthetic fixtures under ``tests/fixtures/``.

Real data and compute are unresolved (issue #2), so this is the only executable check
the ported pipeline has. It runs every stage that does **not** need a model download,
against fabricated fixtures, and asserts that each stage exits 0 and writes the artifacts
it promises (config snapshot + metrics JSON + run note, plus the stage's own outputs).

The runner sets ``QAV2_PATHS_CONFIG=configs/smoke_paths.json``, so every script reads its
normal committed config but resolves data paths into the fixture tree. Output goes to
``runs/smoke/`` (gitignored) and is cleared at the start of each run.

What is NOT covered, and why:

- NER masking, bi-encoder similarity, cross-encoder rerank and the two similarity probes
  need model weights from the network; they are skipped here.
- The router and decomposer runners are exercised with ``--dry-run``: prompts are
  assembled and artifacts written, but no weights are loaded, so the parameter-count
  assertion in ``src/model_size.py`` is *not* exercised by this test.

Usage::

    python scripts/smoke_test.py
    python scripts/smoke_test.py --list
    python scripts/smoke_test.py --only musique_eval,kg_eval
    python scripts/smoke_test.py --keep-output
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from run_config import PATHS_CONFIG_ENV  # noqa: E402

SMOKE_PATHS_CONFIG = REPO_ROOT / "configs" / "smoke_paths.json"
FIXTURES = REPO_ROOT / "tests" / "fixtures"
DATA_ROOT = FIXTURES / "data_root"
RUNS_ROOT = REPO_ROOT / "runs" / "smoke"
WORK = RUNS_ROOT / "work"

MUSIQUE_SCRIPTS = REPO_ROOT / "MusiQue" / "scripts"
SCRIPTS = REPO_ROOT / "scripts"
COMPONENTS = REPO_ROOT / "components"


@dataclass
class Stage:
    name: str
    cmd: list[str]
    expect_files: list[Path] = field(default_factory=list)
    expect_dir_globs: list[tuple[Path, str]] = field(default_factory=list)
    note: str = ""


def _stages() -> list[Stage]:
    prep = "musique_prep.json"

    split_dir = WORK / "split"
    clean_dir = WORK / "clean"
    questions_dir = WORK / "questions"
    combine_root = WORK / "combine"
    enriched = WORK / "enriched" / "pool_enriched.jsonl"
    pool_dir = WORK / "pool_sample"
    dev_out = WORK / "dev_sample" / "dev_sample.jsonl"
    truncated = WORK / "retrieval" / "top5_biencoder.jsonl"
    scored = WORK / "retrieval" / "top5_scored.jsonl"
    musique_eval_dir = WORK / "musique_eval"
    kg_eval_dir = WORK / "kg_eval"
    plots_dir = WORK / "sweep_plots"
    analysis_dir = WORK / "router_analysis"
    router_dry = WORK / "router_dry"
    decomposer_dry = WORK / "decomposer_dry"
    smoke_runner_out = WORK / "qwen_smoke"
    refine_run = WORK / "pool_refine"
    extract_run = WORK / "sample_extract"
    answer_run = WORK / "answer_accuracy"

    # Three stages write next to their input by design (v1 behaviour). They are pointed at
    # copies under runs/smoke so the committed fixtures are never mutated by a smoke run:
    #   compare_answer_accuracy  -> writes into the run directory it is given
    #   combine                  -> writes the combined file into the input root
    #   refine_and_sample_pool   -> writes refined/pool files into the pool directory
    answer_run_copy = answer_run / "decomposer_run"
    combine_input_root = WORK / "combine_input"
    refine_pool_dir = WORK / "metaqa_pool"

    return [
        Stage(
            name="musique_eval",
            cmd=[
                sys.executable, SCRIPTS / "musique_decompositions_evaluator.py",
                "--predictions", FIXTURES / "predictions" / "decomposer_results_musique.json",
                "--run-dir", musique_eval_dir,
            ],
            expect_files=[
                musique_eval_dir / "eval_metrics.json",
                musique_eval_dir / "eval_config.json",
                musique_eval_dir / "eval_notes.md",
                musique_eval_dir / "eval_per_item.json",
            ],
            note="string-level MuSiQue decomposition scoring against gold",
        ),
        Stage(
            name="kg_eval",
            cmd=[
                sys.executable, SCRIPTS / "evaluate_decompositions.py",
                "--predictions", FIXTURES / "predictions" / "decomposer_results_metaqa.json",
                "--run-dir", kg_eval_dir,
            ],
            expect_files=[
                kg_eval_dir / "kg_eval_metrics.json",
                kg_eval_dir / "kg_eval_config.json",
                kg_eval_dir / "kg_eval_notes.md",
                kg_eval_dir / "success.json",
                kg_eval_dir / "compile_fail.json",
                kg_eval_dir / "exec_fail.json",
            ],
            note="compile + execute decompositions against the fabricated MetaQA KG",
        ),
        Stage(
            name="kg_summary",
            cmd=[sys.executable, SCRIPTS / "kg.py"],
            note="KG loader entry point (prints entity/triple/relation counts)",
        ),
        Stage(
            name="split",
            cmd=[
                sys.executable, MUSIQUE_SCRIPTS / "split_musique_train_stratified.py",
                "--config", prep, "--out-dir", split_dir, "--run-dir", split_dir / "_run",
            ],
            expect_files=[
                split_dir / "musique_ans_v1.0_train_0.jsonl",
                split_dir / "musique_ans_v1.0_train_3.jsonl",
                split_dir / "_run" / "metrics.json",
            ],
            note="stratified 4-way split of the 8-row synthetic train file",
        ),
        Stage(
            name="clean",
            cmd=[
                sys.executable, MUSIQUE_SCRIPTS / "clean_musique_train_chunks.py",
                "--config", prep,
                "--inputs", split_dir / "musique_ans_v1.0_train_0.jsonl",
                split_dir / "musique_ans_v1.0_train_1.jsonl",
                "--out-dir", clean_dir, "--run-dir", clean_dir / "_run", "--overwrite",
            ],
            expect_files=[
                clean_dir / "musique_ans_v1.0_train_0_clean.jsonl",
                clean_dir / "_run" / "metrics.json",
            ],
            note="keep id/hop_count/question/decomposition, add a per-file index",
        ),
        Stage(
            name="extract_questions",
            cmd=[
                sys.executable, MUSIQUE_SCRIPTS / "extract_musique_clean_questions.py",
                "--config", prep,
                "--inputs", clean_dir / "musique_ans_v1.0_train_0_clean.jsonl",
                "--out-dir", questions_dir, "--run-dir", questions_dir / "_run", "--overwrite",
            ],
            expect_dir_globs=[(questions_dir, "musique_ans_v1.0_train_0_questions_*.jsonl")],
            expect_files=[questions_dir / "_run" / "metrics.json"],
            note="split a clean chunk into per-stratum question files",
        ),
        Stage(
            name="combine",
            cmd=[
                sys.executable, MUSIQUE_SCRIPTS / "combine_train_split_masked_questions.py",
                "--config", prep,
                "--input-root", combine_input_root,
                "--out-name", "combined_questions_all.jsonl",
                "--run-dir", combine_root, "--overwrite",
            ],
            expect_files=[
                combine_input_root / "roberta_large_ner_english" / "combined_questions_all.jsonl",
                combine_root / "metrics.json",
            ],
            note="merge per-stratum masked chunks, reindexing across files",
        ),
        Stage(
            name="enrich",
            cmd=[
                sys.executable, MUSIQUE_SCRIPTS / "enrich_pool_decompositions.py",
                "--config", prep,
                "--pool", DATA_ROOT / "musique" / "chunks_only_question_masked_fixed"
                / "roberta_large_ner_english" / "musique_ans_v1.0_train_0_questions_2_hop.jsonl",
                "--out", enriched, "--run-dir", enriched.parent / "_run", "--overwrite",
            ],
            expect_files=[enriched, enriched.parent / "_run" / "metrics.json"],
            note="fill few_shot_decomposition_musique from the train source",
        ),
        Stage(
            name="sample_pool_balanced",
            cmd=[
                sys.executable, MUSIQUE_SCRIPTS / "sample_pool.py",
                "--config", prep, "--size", "3", "--balance", "balanced",
                "--out-dir", pool_dir, "--overwrite",
            ],
            expect_files=[
                pool_dir / "pool.jsonl",
                pool_dir / "stats.json",
                pool_dir / "metrics.json",
                pool_dir / "notes.md",
            ],
            note="stratified pool draw, 1 row per coarse hop bucket",
        ),
        Stage(
            name="sample_dev",
            cmd=[
                sys.executable, MUSIQUE_SCRIPTS / "sample_dev.py",
                "--config", prep, "--per-hop", "1", "--out", dev_out, "--overwrite",
            ],
            expect_files=[
                dev_out,
                dev_out.parent / f"{dev_out.stem}_stats.json",
                dev_out.parent / "sample_dev_metrics.json",
            ],
            note="per-hop dev draw across the 2/3/4-hop fixture files",
        ),
        Stage(
            name="truncate",
            cmd=[
                sys.executable, MUSIQUE_SCRIPTS / "truncate_top20.py",
                "--input", FIXTURES / "retrieval" / "top20.jsonl",
                "--out", truncated, "--k", "5", "--run-dir", truncated.parent,
            ],
            expect_files=[truncated, truncated.parent / "truncate_metrics.json"],
            note="bi-encoder-only selector: keep the first k neighbours per mode",
        ),
        Stage(
            name="score_similarity",
            cmd=[
                sys.executable, MUSIQUE_SCRIPTS / "score_similarity_results.py",
                "--input", truncated, "--out", scored, "--run-dir", scored.parent,
            ],
            expect_files=[
                scored,
                scored.parent / f"{scored.stem}_summary.json",
                scored.parent / "score_metrics.json",
            ],
            note="hop-match ladder + cross-mode similarity bonus",
        ),
        Stage(
            name="plot_chunk_stats",
            cmd=[
                sys.executable, MUSIQUE_SCRIPTS / "plot_musique_chunk_stats.py",
                "--config", prep,
                "--inputs", DATA_ROOT / "musique" / "musique_ans_v1.0_train.jsonl",
                "--out-dir", WORK / "chunk_stats",
            ],
            expect_files=[
                WORK / "chunk_stats" / "hop_counts_per_chunk.png",
                WORK / "chunk_stats" / "plot_metrics.json",
            ],
            note="hop/stratum figures over the synthetic train file",
        ),
        Stage(
            name="extract_sample_rows",
            cmd=[
                sys.executable, MUSIQUE_SCRIPTS / "extract_sample_rows.py",
                "--config", prep, "--count", "2",
                "--out-dir", extract_run, "--run-dir", extract_run / "_run",
            ],
            expect_files=[
                extract_run / "sample_bert_large_NER.jsonl",
                extract_run / "sample_roberta_large_ner_english.jsonl",
                extract_run / "_run" / "extract_sample_metrics.json",
            ],
            note="seeded line-number draw across two NER variants of the same rows",
        ),
        Stage(
            name="refine_pool",
            cmd=[
                sys.executable, SCRIPTS / "refine_and_sample_pool.py",
                "--sample-size-per-hop", "2", "--pool-dir", refine_pool_dir,
                "--run-dir", refine_run,
            ],
            expect_files=[
                refine_pool_dir / "qa_train_1hop_refined.txt",
                refine_pool_dir / "1hop_pool.txt",
                refine_run / "refine_metrics.json",
            ],
            note="bracket cleaning + seeded per-hop draw over the MetaQA pool files",
        ),
        Stage(
            name="plot_pool_sweep",
            cmd=[
                sys.executable, SCRIPTS / "plot_pool_sweep.py",
                "--summary-csv", FIXTURES / "pool_sweep_summary" / "all_runs.csv",
                "--out-dir", plots_dir, "--metrics", "step_f1_macro,composite_score",
            ],
            expect_files=[
                plots_dir / "metric_vs_pool_size__step_f1_macro.png",
                plots_dir / "ce_vs_biencoder_delta__step_f1_macro.png",
                plots_dir / "balanced_vs_imbalanced_delta__composite_score.png",
                plots_dir / "metrics_by_cell_mean_std.csv",
                plots_dir / "plots_metrics.json",
            ],
            note="sweep figures from a fabricated summary CSV",
        ),
        Stage(
            name="analyze_runs",
            cmd=[
                sys.executable, SCRIPTS / "analyze_runs.py",
                "--runs-dir", FIXTURES / "router_runs",
                "--component", "average_few_shot",
                "--output-dir", analysis_dir,
            ],
            expect_files=[
                analysis_dir / "overall_accuracy.png",
                analysis_dir / "per_hop_accuracy.png",
                analysis_dir / "report.html",
                analysis_dir / "analysis_metrics.json",
            ],
            note="plots + HTML report over two fabricated router runs",
        ),
        Stage(
            name="answer_accuracy",
            cmd=[
                sys.executable, SCRIPTS / "compare_answer_accuracy.py",
                str(answer_run_copy), "--details",
            ],
            expect_files=[
                answer_run_copy / "analysis" / "answer_accuracy_metrics.json",
                answer_run_copy / "analysis" / "answer_accuracy_notes.md",
                answer_run_copy / "analysis" / "answer_details.json",
            ],
            note="Jaccard comparison of KG results against the fabricated gold answers",
        ),
        Stage(
            name="router_dry_run",
            cmd=[
                sys.executable, COMPONENTS / "router" / "run_router.py",
                "--model", "qwen2_5_0_5b", "--dry-run", "--dry-run-limit", "3",
                "--output-root", router_dry,
            ],
            expect_dir_globs=[
                (router_dry, "*/metrics.json"),
                (router_dry, "*/config.json"),
                (router_dry, "*/notes.md"),
                (router_dry, "*/prompts_log/prompt_idx0001.txt"),
            ],
            note="router prompt assembly + artifacts, no weights loaded",
        ),
        Stage(
            name="decomposer_dry_run_retrieval",
            cmd=[
                sys.executable, COMPONENTS / "decomposer" / "run_decomposer.py",
                "--model", "mistral_7b_instruct", "--dry-run", "--dry-run-limit", "2",
                "--retrieval-input", FIXTURES / "retrieval" / "top20.jsonl",
                "--retrieval-mode", "uniform", "--retrieval-k", "5",
                "--output-root", decomposer_dry / "mistral",
            ],
            expect_dir_globs=[
                (decomposer_dry / "mistral", "*/results.json"),
                (decomposer_dry / "mistral", "*/metrics.json"),
                (decomposer_dry / "mistral", "*/config.json"),
                (decomposer_dry / "mistral", "*/notes.md"),
            ],
            note="retrieval-driven few-shot prompt assembly (5 examples per query)",
        ),
        Stage(
            name="decomposer_dry_run_inline",
            cmd=[
                sys.executable, COMPONENTS / "decomposer" / "run_decomposer.py",
                "--model", "qwen2_5_3b", "--dry-run", "--dry-run-limit", "3",
                "--output-root", decomposer_dry / "qwen",
            ],
            expect_dir_globs=[
                (decomposer_dry / "qwen", "*/results.json"),
                (decomposer_dry / "qwen", "*/metrics.json"),
            ],
            note="inline-examples prompt over the MetaQA question files (few_shot disabled)",
        ),
        Stage(
            name="decomposer_dry_run_chat_template",
            cmd=[
                sys.executable, COMPONENTS / "decomposer" / "run_decomposer.py",
                "--model", "qwen3_5_9b", "--dry-run", "--dry-run-limit", "2",
                "--retrieval-input", FIXTURES / "retrieval" / "top20.jsonl",
                "--retrieval-mode", "typed", "--retrieval-k", "5",
                "--output-root", decomposer_dry / "qwen3_5",
            ],
            expect_dir_globs=[
                (decomposer_dry / "qwen3_5", "*/results.json"),
                (decomposer_dry / "qwen3_5", "*/metrics.json"),
            ],
            note="chat-template prompt split on <<<USER>>> (system/user halves)",
        ),
        Stage(
            name="smoke_runner_subsample",
            cmd=[
                sys.executable, SCRIPTS / "run_qwen_smoke.py",
                "--source", FIXTURES / "retrieval" / "top20.jsonl",
                "--n-per-hop", "1", "--hops", "2", "3",
                "--output-root", smoke_runner_out, "--dry-run",
            ],
            expect_files=[
                smoke_runner_out / "input" / "subset_2_3hop_n1_seed42.jsonl",
                smoke_runner_out / "smoke_metrics.json",
            ],
            note="per-hop subsample builder (decomposer launch skipped)",
        ),
        Stage(
            name="pool_sweep_orchestrator_dry_run",
            cmd=[
                sys.executable, SCRIPTS / "pool_sweep_orchestrator.py",
                "--dry-run", "--only", "size1000_imbalanced", "--modes", "uniform",
                "--variants", "biencoder_only",
            ],
            note="prints the full sweep command graph without executing a stage",
        ),
    ]


def _prepare(keep_output: bool) -> None:
    if RUNS_ROOT.exists() and not keep_output:
        shutil.rmtree(RUNS_ROOT)
    WORK.mkdir(parents=True, exist_ok=True)

    # The three stages that write next to their input get a working copy, so a smoke run
    # never mutates the committed fixtures (and a re-run is never blocked by its own
    # previous output).
    for src, dst in (
        (FIXTURES / "decomposer_run", WORK / "answer_accuracy" / "decomposer_run"),
        (
            DATA_ROOT / "musique" / "chunks_only_question_masked_fixed",
            WORK / "combine_input",
        ),
        (DATA_ROOT / "metaqa" / "pool", WORK / "metaqa_pool"),
    ):
        if dst.exists():
            shutil.rmtree(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)


def _run_stage(stage: Stage, env: dict[str, str]) -> tuple[bool, str, float]:
    cmd = [str(c) for c in stage.cmd]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True)
    elapsed = time.time() - t0
    tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-25:])

    if proc.returncode != 0:
        return False, f"exit code {proc.returncode}\n{tail}", elapsed

    missing: list[str] = []
    for path in stage.expect_files:
        if not Path(path).exists():
            missing.append(str(path))
    for base, pattern in stage.expect_dir_globs:
        if not list(Path(base).glob(pattern)):
            missing.append(f"{base}/{pattern} (no match)")
    if missing:
        return False, "missing expected artifacts:\n  " + "\n  ".join(missing), elapsed

    return True, tail, elapsed


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", action="store_true", help="List the stages and exit.")
    p.add_argument("--only", default=None, help="Comma-separated stage names to run.")
    p.add_argument("--keep-output", action="store_true", help="Do not clear runs/smoke first.")
    p.add_argument("--verbose", action="store_true", help="Print each stage's output tail.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    stages = _stages()

    if args.list:
        for s in stages:
            print(f"{s.name:34s} {s.note}")
        return 0

    if args.only:
        wanted = {t.strip() for t in args.only.split(",") if t.strip()}
        unknown = wanted - {s.name for s in stages}
        if unknown:
            raise SystemExit(f"unknown stage(s): {sorted(unknown)}")
        stages = [s for s in stages if s.name in wanted]

    _prepare(args.keep_output)

    env = os.environ.copy()
    env[PATHS_CONFIG_ENV] = str(SMOKE_PATHS_CONFIG)
    env["PYTHONUNBUFFERED"] = "1"

    print(f"[smoke] repo: {REPO_ROOT}")
    print(f"[smoke] {PATHS_CONFIG_ENV}={SMOKE_PATHS_CONFIG}")
    print(f"[smoke] fixtures: {DATA_ROOT}")
    print(f"[smoke] output: {RUNS_ROOT}")
    print(f"[smoke] stages: {len(stages)}\n")

    results: list[tuple[str, bool, float]] = []
    failures: list[tuple[str, str]] = []
    for stage in stages:
        print(f"[smoke] >>> {stage.name}", flush=True)
        ok, detail, elapsed = _run_stage(stage, env)
        results.append((stage.name, ok, elapsed))
        if ok:
            print(f"[smoke] PASS {stage.name} ({elapsed:.1f}s)")
            if args.verbose and detail:
                print(detail)
        else:
            print(f"[smoke] FAIL {stage.name} ({elapsed:.1f}s)\n{detail}")
            failures.append((stage.name, detail))

    passed = sum(1 for _, ok, _ in results if ok)
    print("\n" + "=" * 72)
    print(f"[smoke] {passed}/{len(results)} stages passed")
    for name, ok, elapsed in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:34s} {elapsed:5.1f}s")
    print("=" * 72)

    summary = {
        "stages_total": len(results),
        "stages_passed": passed,
        "stages_failed": [name for name, ok, _ in results if not ok],
        "per_stage_seconds": {name: round(elapsed, 2) for name, _, elapsed in results},
        "paths_config": str(SMOKE_PATHS_CONFIG),
        "fixtures_data_root": str(DATA_ROOT),
        "not_covered": [
            "NER masking, bi-encoder similarity, cross-encoder rerank and the similarity "
            "probes (need model downloads)",
            "the parameter-count assertion in src/model_size.py (no weights are loaded; "
            "the runners use --dry-run)",
        ],
    }
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    (RUNS_ROOT / "smoke_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[smoke] summary -> {RUNS_ROOT / 'smoke_summary.json'}")

    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
