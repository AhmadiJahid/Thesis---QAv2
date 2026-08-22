#!/usr/bin/env python3
"""MetaQA end-to-end backend: compile a decomposition, execute it, score the answer set.

The MetaQA half of issue #16. ADR 0006 gives MetaQA **end-to-end evaluation only** (it has
no gold decompositions), so a MetaQA decomposition is judged by the answer set executing it
produces. Both halves of that already existed as separate scripts run by hand:

1. ``scripts/evaluate_decompositions.py`` compiles each predicted sub-question into a KG op
   and executes the chain, reporting a compile/execute rate with its error taxonomies;
2. ``scripts/compare_answer_accuracy.py`` compares the executed answer set against MetaQA's
   gold answers with Jaccard (exact match = Jaccard 1.0).

This is the **wrapper** that runs them as one command over one predictions file and writes
one metrics JSON: coverage, exact match, Jaccard, with both error taxonomies preserved
verbatim. It **imports** the compiler and the scorer — it does not reimplement either. The
step-template regexes, the relation rules, the compile-error reasons
(``missing_decomposition``, ``unsupported_template``, ``cannot_infer_relation``,
``compile_error_other``) and the execution-error categories (``entity_not_in_kb``,
``bad_reference_or_plan``, ``exec_error_other``) all stay where they are, unchanged;
``tests/test_metaqa_compile_execute.py`` pins that the wrapper reproduces the standalone
script's fixture outcomes exactly.

**This is not GRAG.** ADR 0006 routes MetaQA end-to-end evaluation through the supervisor's
GRAG system. GRAG is external and nothing about its interface exists in this repo or in the
v1 repo, so it is not wired here and is not stubbed: this path is direct execution against
the MetaQA KG built from ``kb.txt`` (``scripts/kg.py``), it says so in ``backend_label`` and
in ``backend.grag_wired: false`` in every metrics JSON it writes, and a number from it is
not a GRAG number. See ``configs/metaqa_compile_execute.json`` (``_grag_note``).

No model is loaded anywhere in this path — the compiler is regexes and the scorer is set
arithmetic — so there is no parameter count to assert and, as everywhere in this repo, no
model scores, rates or judges anything.

Two denominators are reported for exact match and Jaccard, because they answer different
questions and the difference is large:

- **over executed items with gold** — what ``compare_answer_accuracy.py`` has always
  reported: of the decompositions that compiled *and* executed, how good is the answer set.
  It excludes every compile and execution failure, so it is a conditional number.
- **over all items with gold** — every prediction whose question has a gold answer, with a
  compile or execution failure counted as the empty answer set it produced. This is the
  denominator the MuSiQue answering backend uses (a failed item there is inside the reported
  EM/F1, not excluded), so it is the one that can be read against a MuSiQue number for the
  cross-dataset comparison of issue #41. Which of the two is *the* headline MetaQA metric is
  not this script's call; both are reported with their definitions attached.

A **measured zero is reported as zero**: a run where every decomposition fails reports 0.0%
and 0.0, not null. Null is reserved for the one genuinely unmeasured case — no item in the
run has a gold answer at all. Both metrics apply **one rule to every item**, including the
empty-gold edge case (``jaccard(empty, empty)`` is 1.0 and therefore an exact match), so the
definition printed next to the numbers is true of all of them; when that edge case fires it
is named in ``empty_gold_set_note`` rather than nulled away.

Element matching is **strip-only and case-sensitive** (what
``scripts/compare_answer_accuracy.py`` has always done), which is *not* MuSiQue EM's SQuAD
normalization. The asymmetry is recorded in ``metric_definitions.answer_normalization`` and in
the over-all-items definition rather than removed: changing it would move every MetaQA number
ever reported and is a scoring-rule decision for Jahid.

Which evaluation set was scored is recorded and asserted, mirroring the MuSiQue backend: a
processed question with no gold answer cannot be scored and would silently shrink every
denominator, so the run is **refused** unless ``--allow-unpinned-eval-set`` is given (fixture
and probe runs, which are not experiment arms). MetaQA has no ADR-pinned id subset the way
ADR 0007 pins MuSiQue's 600, so identity is carried by
``evaluation_set.question_set_sha256`` — equal fingerprints mean the same questions were
scored, which is what makes two MetaQA numbers a comparison. The decomposer run that produced
the predictions is traced through its sibling ``config.json`` (run id, commit, config), or
recorded as unrecorded.

Usage::

    python scripts/run_metaqa_compile_execute.py --predictions <decomposer run>/results.json
    python scripts/run_metaqa_compile_execute.py --predictions <...> --run-dir runs/probe

    # a fixture or probe run, whose questions need not all be in the MetaQA gold files
    python scripts/run_metaqa_compile_execute.py --predictions <...> --allow-unpinned-eval-set
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
#: ``sha256_file`` is imported from the decomposer runner rather than copied, for the reason
#: ADR 0019 records about ``assert_pinned_eval_set``: a second copy of "how an input is
#: content-addressed" would be a second thing to keep in step. That module imports nothing
#: heavy at module level (measured: ~60ms, no torch), so a CPU-only script can take it.
sys.path.insert(0, str(_REPO_ROOT / "components" / "decomposer"))

# The two halves, imported rather than reimplemented: one home for the compiler, one home
# for the Jaccard scorer.
from compare_answer_accuracy import jaccard, load_gold_by_hop, run_analysis  # noqa: E402
from evaluate_decompositions import evaluate_decomposition_rate  # noqa: E402
from kg import build_metaqa_kg  # noqa: E402
from run_artifacts import now_iso, write_run_artifacts  # noqa: E402
from run_config import load_config, load_paths, require, resolve_path, runs_path  # noqa: E402
from run_decomposer import sha256_file  # noqa: E402
from seeding import set_global_seed  # noqa: E402

#: Recorded in every artifact. GRAG is external and absent (see the module docstring), so
#: this is a constant of the code, not a config knob a flag could flip.
GRAG_WIRED = False
GRAG_STATUS = (
    "not wired: GRAG is the supervisor's system (ADR 0006) and no interface, endpoint, "
    "client, repository or handover for it exists in this repo or in the v1 repo "
    "(Thesis---QA) as of 2026-08-22. This run executed decompositions directly against the "
    "MetaQA KG built from kb.txt instead, which is a different measurement and is labelled "
    "as one. Obtaining GRAG is an external dependency Jahid chases with his supervisor."
)

#: The dumps ``evaluate_decomposition_rate`` writes, and what each one holds.
DUMP_FILES = ("success.json", "compile_fail.json", "exec_fail.json")


def gold_for_item(item: dict, gold_by_hop: dict[int, dict[str, set[str]]]) -> set[str] | None:
    """The gold answer set for one prediction row, or ``None`` when it has no gold.

    Same lookup key as ``compare_answer_accuracy.run_analysis``: the stripped question text
    within the row's ``hop_count``. Kept to one function so the executed and the failed rows
    are matched to gold by exactly the same rule.
    """
    question = str(item.get("question") or "").strip()
    hop_count = item.get("hop_count")
    if not isinstance(hop_count, int):
        return None
    return gold_by_hop.get(hop_count, {}).get(question)


#: The dumps that hold the rows which produced **no** answer set.
FAILURE_DUMP_FILES = ("compile_fail.json", "exec_fail.json")


def processed_rows(analysis_dir: Path) -> list[dict]:
    """Every row this run processed, read back from its own three dumps.

    ``success.json + compile_fail.json + exec_fail.json`` is exactly the set of rows the
    compile-execute half read (after the row cap), so taking the evaluation set from them
    means the cap and the row-selection rule live in one place — the script being wrapped.
    A missing dump is a refusal for the same reason as in :func:`score_failed_items`.
    """
    rows: list[dict] = []
    for name in DUMP_FILES:
        path = analysis_dir / name
        if not path.exists():
            raise SystemExit(
                f"expected the dump {path} to exist: it is written unconditionally by the "
                "compile-execute half, and the evaluation-set record is derived from all "
                "three dumps. Refusing rather than describing the run's evaluation set from "
                "an unknown subset of it."
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(item for item in payload if isinstance(item, dict))
    return rows


def evaluation_set_record(
    rows: list[dict],
    *,
    predictions_path: Path,
    gold_by_hop: dict[int, dict[str, set[str]]],
    total_per_hop: dict[int, int],
    hops: list[int],
) -> dict[str, Any]:
    """What evaluation set this run scored, as an identity a later run can be checked against.

    The MuSiQue side asserts the pinned ADR 0007 id set by identity, not only by count
    (``run_decomposer.assert_pinned_eval_set``), because "an end-to-end number on a different
    set is not comparable to anything" (ADR 0011). MetaQA has no ADR-pinned subset, so there
    is no id list to check against — but the same discipline applies, and CLAUDE.md's
    evidence rule (a comparison claim requires the *same* evaluation set) has to be
    verifiable from the artifact rather than trusted. So this records:

    - the per-hop row counts and how many of them matched a gold question;
    - ``question_set_sha256``, a fingerprint over the sorted ``hop<TAB>question`` lines of the
      rows processed. Two runs with the same fingerprint scored the same questions; two runs
      with different fingerprints are not a comparison, whatever their row counts say. It is
      a hash, so no dataset text leaves the run in the metrics JSON.
    - ``unmatched_questions``: rows with no gold answer. They are silently absent from every
      accuracy denominator, which is precisely the failure mode worth refusing over.

    ``pinned`` is True when every processed row has a gold answer and there is at least one
    row: the run scored a well-defined subset of the MetaQA gold set. It is deliberately a
    weaker claim than the MuSiQue one, and it says so in ``pinned_definition``.
    """
    per_hop_rows: dict[str, int] = {}
    per_hop_matched: dict[str, int] = {}
    unmatched: list[str] = []
    fingerprint_lines: list[str] = []
    rows_without_hop = 0

    for item in rows:
        question = str(item.get("question") or "").strip()
        hop = item.get("hop_count")
        if not isinstance(hop, int):
            rows_without_hop += 1
            key = "unknown"
        else:
            key = str(hop)
        per_hop_rows[key] = per_hop_rows.get(key, 0) + 1
        fingerprint_lines.append(f"{key}\t{question}")
        gold = gold_for_item(item, gold_by_hop)
        if gold is None:
            unmatched.append(f"{key}: {question}" if question else f"{key}: <no question>")
        else:
            per_hop_matched[key] = per_hop_matched.get(key, 0) + 1

    digest = hashlib.sha256(
        "\n".join(sorted(fingerprint_lines)).encode("utf-8")
    ).hexdigest()

    return {
        "rows_processed": len(rows),
        "rows_per_hop": dict(sorted(per_hop_rows.items())),
        "rows_with_gold_per_hop": dict(sorted(per_hop_matched.items())),
        "rows_without_a_hop_count": rows_without_hop,
        "gold_questions_available_per_hop": {str(h): total_per_hop.get(h, 0) for h in hops},
        "unmatched_question_count": len(unmatched),
        "unmatched_questions_sample": unmatched[:10],
        "question_set_sha256": digest,
        "predictions_path": str(predictions_path),
        "predictions_sha256": sha256_file(predictions_path),
        "pinned": bool(rows) and not unmatched,
        "pinned_definition": (
            "true when every processed row's question was found in the MetaQA gold files for "
            "its hop depth, so the run scored a well-defined subset of the gold set. This is "
            "weaker than the MuSiQue side's assertion (ADR 0007's pinned 600 ids, checked by "
            "identity): MetaQA has no ADR-pinned subset, so there is no id list to check "
            "against. question_set_sha256 is what makes two MetaQA runs comparable - equal "
            "fingerprints mean the same questions were scored."
        ),
    }


def upstream_decomposer_run(predictions_path: Path) -> dict[str, Any]:
    """The provenance of the predictions: which run, which commit, which config.

    A decomposer run writes ``config.json`` / ``metrics.json`` next to its ``results.json``
    (``src/run_artifacts.py``), and ``write_run_artifacts`` stamps git provenance into both.
    Reading it here is what lets a MetaQA number be traced to the decomposition run and the
    commit that produced it, instead of to a bare file path (PR #42 review, finding 3).

    Absence is recorded, never inferred: a hand-made or relocated predictions file has no
    sibling snapshot, and that is reported as ``found: false`` rather than guessed at.
    """
    snapshot_path = predictions_path.parent / "config.json"
    if not snapshot_path.exists():
        return {
            "found": False,
            "note": (
                f"no decomposer run snapshot at {snapshot_path}: the predictions file has no "
                "sibling config.json, so the run and commit that produced these "
                "decompositions are not recorded here. Expected for a hand-made or moved "
                "predictions file; for a real arm it means the run directory was not kept "
                "intact, and the provenance has to come from the experiment log entry."
            ),
        }
    try:
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"found": False, "note": f"could not read {snapshot_path}: {exc}"}
    if not isinstance(snapshot, dict):
        return {"found": False, "note": f"{snapshot_path} is not a JSON object"}

    git = snapshot.get("git") if isinstance(snapshot.get("git"), dict) else {}
    return {
        "found": True,
        "snapshot_path": str(snapshot_path),
        "script": snapshot.get("script"),
        "run_id": snapshot.get("run_id"),
        "component": snapshot.get("component"),
        "model": snapshot.get("model"),
        "model_id": snapshot.get("model_id"),
        "condition": snapshot.get("condition"),
        "shared_config": snapshot.get("shared_config"),
        "model_config": snapshot.get("model_config"),
        "seed": snapshot.get("seed"),
        "commit": git.get("commit"),
        "branch": git.get("branch"),
        "dirty": git.get("dirty"),
    }


def score_failed_items(
    analysis_dir: Path,
    *,
    gold_by_hop: dict[int, dict[str, set[str]]],
    exact_threshold: float,
) -> dict[str, Any]:
    """Score the compile/execute failures as the empty answer set they produced.

    A failed item produced no answer set, so its contribution is ``jaccard(set(), gold)``
    and its exact match is ``that jaccard >= exact_threshold`` — **both** computed with the
    imported scorer and the same threshold the executed items are scored with, so the block's
    stated definition holds for every item in it. The alternative (special-casing the
    empty-gold row) was tried and rejected in review: guarding the mean while leaving the
    exact-match count unguarded made ``exact_match / items_with_gold`` disagree with the
    definition printed next to it (PR #42 review, finding 2). Applying one rule uniformly is
    what keeps the definition text true; ``failed_items_with_an_empty_gold_set`` is reported
    so a reader can see when the empty-vs-empty rule (Jaccard 1.0, therefore an exact match)
    contributed, and the caller attaches a note whenever it did.

    The failed rows are read from the run's own dumps, so "which rows this run processed" is
    taken from the run's output and never re-derived from the predictions file (the row cap
    would then live in two places). A missing dump is a **refusal**, not a skip: the dumps
    are written unconditionally by the compile-execute half, so an absent one means the
    second denominator would silently omit rows it is supposed to count (PR #42 review,
    nit 2).
    """
    failed_with_gold = 0
    failed_jaccard_sum = 0.0
    failed_with_empty_gold = 0
    failed_exact_matches = 0
    for name in FAILURE_DUMP_FILES:
        path = analysis_dir / name
        if not path.exists():
            raise SystemExit(
                f"expected the failure dump {path} to exist: the compile-execute half writes "
                f"{', '.join(FAILURE_DUMP_FILES)} unconditionally, so a missing one means "
                "the over-all-items denominator would omit the rows it is meant to count. "
                "Refusing rather than reporting a number over an unknown subset."
            )
        for item in json.loads(path.read_text(encoding="utf-8")):
            gold = gold_for_item(item, gold_by_hop)
            if gold is None:
                continue
            failed_with_gold += 1
            if not gold:
                failed_with_empty_gold += 1
            score = jaccard(set(), gold)
            failed_jaccard_sum += score
            if score >= exact_threshold:
                failed_exact_matches += 1
    return {
        "failed_items_with_gold": failed_with_gold,
        "failed_items_with_an_empty_gold_set": failed_with_empty_gold,
        "failed_items_jaccard_sum": failed_jaccard_sum,
        "failed_items_exact_match_count": failed_exact_matches,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", default="metaqa_compile_execute.json")
    p.add_argument(
        "--predictions", type=Path, required=True, help="Decomposer run's results.json"
    )
    p.add_argument("--kb", type=Path, default=None, help="Override the kb path from config.")
    p.add_argument("--max-items", type=int, default=None, help="Override the config row cap.")
    p.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Where the artifacts and the per-item dumps go (default: under the runs root).",
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--allow-unpinned-eval-set",
        action="store_true",
        help="Permit a run in which some processed questions have no MetaQA gold answer. "
        "For fixture and probe runs only: such rows are absent from every accuracy "
        "denominator, the metrics then record evaluation_set.pinned false, and the run is "
        "not an experiment arm. Mirrors the same flag on the MuSiQue answering backend.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    cfg = load_config(args.config)
    paths_cfg = load_paths(require(cfg, "paths_config"))
    data_root = Path(paths_cfg["data_root_resolved"])

    # The two composed configs. Every methodology knob is read from them, so the wrapper
    # cannot disagree with the script it wraps about what a number means.
    kg_cfg = load_config(require(cfg, "kg_eval_config"))
    acc_cfg = load_config(require(cfg, "answer_accuracy_config"))

    seed = args.seed if args.seed is not None else int(require(cfg, "seed"))
    seeded = set_global_seed(seed)

    max_items = args.max_items if args.max_items is not None else require(kg_cfg, "max_items")
    hops = [int(h) for h in require(acc_cfg, "hops")]
    analysis_subdir = require(acc_cfg, "analysis_subdir")
    exact_threshold = float(require(acc_cfg, "exact_match_jaccard"))
    out_prefix = require(cfg, "out_prefix")
    backend_label = require(cfg, "backend_label")

    if not args.predictions.exists():
        raise SystemExit(f"predictions not found: {args.predictions}")
    kb_path = args.kb or resolve_path(
        require(paths_cfg, "datasets." + require(kg_cfg, "kb_key")), data_root
    )

    run_dir = (
        args.run_dir
        if args.run_dir is not None
        else runs_path(paths_cfg, require(cfg, "run_subdir"))
    )
    run_dir = Path(run_dir)
    analysis_dir = run_dir / analysis_subdir
    analysis_dir.mkdir(parents=True, exist_ok=True)

    print(f"Backend: {backend_label} (GRAG wired: {GRAG_WIRED})")
    print(f"Loading MetaQA KG from {kb_path} ...")
    kg = build_metaqa_kg(kb_path)

    # ---- half 1: compile + execute (the existing path, called, not reimplemented)
    print(f"Compiling and executing decompositions in {args.predictions} ...")
    rate = evaluate_decomposition_rate(
        kg,
        args.predictions,
        max_items=int(max_items) if max_items is not None else None,
        # Always written: the gold comparison below reads success.json. They hold question
        # and answer text and land under the gitignored runs root, never in git.
        output_dir=analysis_dir,
    )
    coverage = rate.as_dict()

    # ---- half 2: the executed answer sets against MetaQA gold (the existing scorer)
    gold_by_hop, total_per_hop = load_gold_by_hop(
        data_root,
        require(paths_cfg, "datasets." + require(acc_cfg, "questions_template_key")),
        require(paths_cfg, "datasets." + require(acc_cfg, "answers_template_key")),
        hops,
        require(acc_cfg, "answer_separator"),
    )
    if not any(gold_by_hop.values()):
        raise SystemExit(
            f"no MetaQA gold question/answer files found under {data_root}; the executed "
            "answer sets cannot be scored without them (check data_root in the paths config)"
        )

    summary, per_item = run_analysis(
        run_dir,
        gold_by_hop=gold_by_hop,
        total_per_hop=total_per_hop,
        hops=hops,
        analysis_subdir=analysis_subdir,
        exact_threshold=exact_threshold,
        seed=seed,
    )

    # ---- what evaluation set was actually scored, and where the predictions came from
    eval_set_record = evaluation_set_record(
        processed_rows(analysis_dir),
        predictions_path=args.predictions,
        gold_by_hop=gold_by_hop,
        total_per_hop=total_per_hop,
        hops=hops,
    )
    eval_set_record["allow_unpinned_override"] = args.allow_unpinned_eval_set
    upstream = upstream_decomposer_run(args.predictions)
    if (
        require(cfg, "eval_set.require_gold_for_every_item")
        and eval_set_record["unmatched_question_count"]
        and not args.allow_unpinned_eval_set
    ):
        raise SystemExit(
            f"{eval_set_record['unmatched_question_count']} of "
            f"{eval_set_record['rows_processed']} processed question(s) have no MetaQA gold "
            f"answer for their hop depth, e.g. {eval_set_record['unmatched_questions_sample']}"
            "\nThose rows cannot be scored, so they vanish from every exact-match and "
            "Jaccard denominator and the run would report accuracy over a silently smaller "
            "set than it read - and a comparison against another arm would not be a "
            "comparison (CLAUDE.md evidence discipline, ADR 0011).\nEither point "
            "--predictions at a run over MetaQA questions that are in the gold files "
            "(datasets.metaqa_questions_template / metaqa_answers_template), or pass "
            "--allow-unpinned-eval-set for a fixture or probe run that is not an experiment "
            "arm."
        )
    print(
        f"Evaluation set: {eval_set_record['rows_processed']} row(s), pinned="
        f"{eval_set_record['pinned']}, fingerprint="
        f"{eval_set_record['question_set_sha256'][:16]}..., upstream decomposer run="
        + (f"{upstream.get('run_id')} @ {upstream.get('commit')}" if upstream["found"] else "unrecorded")
    )

    # ---- the second denominator: failures counted rather than excluded
    failures = score_failed_items(
        analysis_dir, gold_by_hop=gold_by_hop, exact_threshold=exact_threshold
    )
    executed_with_gold = int(summary["total_with_gold"])
    items_with_gold = executed_with_gold + failures["failed_items_with_gold"]
    exact_matches = int(summary["total_exact_match"]) + failures["failed_items_exact_match_count"]
    # The EXACT sum, not the 4-dp mean multiplied back out (PR #42 review, nit 1). It is 0.0
    # rather than absent when nothing executed, so a run where every item failed reports 0.0
    # and 0.0% - a measured floor - instead of null, which would read as "unmeasured" and is
    # exactly what the review probe caught (finding 1).
    jaccard_sum = float(summary["overall_jaccard_sum"]) + failures["failed_items_jaccard_sum"]
    if items_with_gold:
        overall_exact_rate = round(100.0 * exact_matches / items_with_gold, 2)
        overall_mean_jaccard = round(jaccard_sum / items_with_gold, 4)
    else:
        # No item in the run has a gold answer: nothing was measured, so nothing is reported.
        overall_exact_rate = None
        overall_mean_jaccard = None

    over_all = {
        "definition": (
            "every prediction whose question has a gold answer, with a compile or execution "
            "failure counted as the empty answer set it produced: its Jaccard is "
            "jaccard(empty, gold) and it is an exact match iff that reaches the same "
            f"threshold ({exact_threshold}) the executed items are scored against. For "
            "MetaQA's non-empty gold that contribution is 0 on both metrics. This is the "
            "same convention the MuSiQue answering backend uses - a failed item is inside "
            "the reported metric, not excluded - so it is the denominator that can be read "
            "against a MuSiQue number (issue #41). NOTE the element-level asymmetry with "
            "MuSiQue: matching here is set membership after whitespace stripping only, "
            "CASE-SENSITIVE and with no punctuation or article handling, whereas MuSiQue EM "
            "applies SQuAD's normalize_answer (see answer_normalization in "
            "metric_definitions). The two numbers are therefore comparable in denominator "
            "but not in string strictness; aligning them is a scoring-rule decision for "
            "Jahid, not an implementation detail."
        ),
        "items_with_gold": items_with_gold,
        "items_with_gold_executed": executed_with_gold,
        "items_with_gold_not_executed": failures["failed_items_with_gold"],
        "exact_match_count": exact_matches,
        "exact_match_count_executed": int(summary["total_exact_match"]),
        "exact_match_count_not_executed": failures["failed_items_exact_match_count"],
        "pct_exact": overall_exact_rate,
        "mean_jaccard": overall_mean_jaccard,
        "jaccard_sum": round(jaccard_sum, 6),
        "items_with_an_empty_gold_set": failures["failed_items_with_an_empty_gold_set"],
    }
    if failures["failed_items_with_an_empty_gold_set"]:
        # Both metrics include them, by the same rule, so the definition above still holds -
        # but the empty-vs-empty rule is surprising enough to name when it fires.
        over_all["empty_gold_set_note"] = (
            f"{failures['failed_items_with_an_empty_gold_set']} failed item(s) have an EMPTY "
            "gold answer set. The Jaccard rule scores empty-vs-empty as 1.0, so each of them "
            f"contributed 1.0 to the mean AND counted as an exact match at threshold "
            f"{exact_threshold} - the same rule applied to every other item in this block, "
            "not a special case. MetaQA gold answers are non-empty, so a non-zero count here "
            "points at a malformed gold line rather than at the metric."
        )

    print("\n" + "=" * 56)
    print(f"   METAQA END-TO-END ({backend_label})")
    print("=" * 56)
    print(f"Items:            {coverage['total']}")
    print(f"Compiled OK:      {coverage['compiled_ok']} ({coverage['compiled_ok_rate']:.2%})")
    print(f"Executed OK:      {coverage['executed_ok']} ({coverage['executed_ok_rate']:.2%})")
    print(f"Compile fail:     {coverage['compile_fail']} {coverage['compile_fail_reasons']}")
    print(f"Exec fail:        {coverage['exec_fail']} {coverage['exec_fail_reasons']}")
    print(
        f"Over EXECUTED items with gold ({summary['total_with_gold']}): "
        f"exact {summary['total_exact_match']} ({summary['overall_pct_exact']}%), "
        f"mean Jaccard {summary['overall_mean_jaccard']}"
    )
    print(
        f"Over ALL items with gold ({items_with_gold}): "
        f"exact {over_all['exact_match_count']} ({over_all['pct_exact']}%), "
        f"mean Jaccard {over_all['mean_jaccard']}"
    )
    for hop in hops:
        block = summary["per_hop"][str(hop)]
        print(
            f"  {hop}-hop: answered {block['answered_count']} of "
            f"{block['total_gold_questions']} gold ({block['coverage_pct']}% coverage), "
            f"with_gold {block['with_gold_count']}, exact {block['exact_match_count']} "
            f"({block['pct_exact']}%), mean Jaccard {block['mean_jaccard']}"
        )
    print("=" * 56)

    details_path = analysis_dir / "answer_details.json"
    details_path.write_text(
        json.dumps(per_item, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    metrics: dict[str, Any] = {
        "script": Path(__file__).name,
        "created_utc": now_iso(),
        "seed": seed,
        "seeded": seeded,
        "backend": {
            "label": backend_label,
            "grag_wired": GRAG_WIRED,
            "grag_status": GRAG_STATUS,
            "model_loaded": False,
            "model_note": (
                "no model is loaded in this path: the compiler is regexes over the step "
                "text and the scorer is set arithmetic, so there is no parameter count to "
                "assert and nothing is scored by a model"
            ),
        },
        "predictions_path": str(args.predictions.resolve()),
        "kb_path": str(Path(kb_path).resolve()),
        "kg_entities": len(kg.id_to_entity),
        "kg_triples": len(kg.triples),
        "data_root": str(data_root),
        "evaluation_set": eval_set_record,
        "upstream_decomposer_run": upstream,
        "coverage": coverage,
        "compile_fail_reasons": coverage["compile_fail_reasons"],
        "exec_fail_reasons": coverage["exec_fail_reasons"],
        "answer_accuracy_over_executed_items": summary,
        "answer_accuracy_over_all_items_with_gold": over_all,
        "metric_definitions": {
            "coverage": (
                "compiled_ok_rate and executed_ok_rate over the items read from the "
                "predictions file, from scripts/evaluate_decompositions.py; the "
                "compile_fail_reasons and exec_fail_reasons breakdowns are that script's "
                "taxonomies, preserved verbatim"
            ),
            "per_hop_coverage_pct": (
                "answered items / gold questions available for that hop depth, from "
                "scripts/compare_answer_accuracy.py"
            ),
            "exact_match": f"Jaccard >= {exact_threshold} between the executed answer set "
            "and the gold answer set",
            "jaccard": "|intersection| / |union| of the two answer sets; empty vs empty is 1.0",
            "answer_normalization": (
                "STRIP-ONLY, and CASE-SENSITIVE. A predicted element is the KG entity string "
                "exactly as kb.txt spells it with surrounding whitespace removed; a gold "
                "element is the gold line split on the answer separator with each part "
                "stripped, and empty parts dropped. There is no lowercasing, no punctuation "
                "removal and no article handling, so 'The Tin Compass' and 'the tin compass' "
                "are different elements here. This is what scripts/compare_answer_accuracy.py "
                "has always done and it is left unchanged - it is NOT the rule MuSiQue EM/F1 "
                "uses (SQuAD's normalize_answer: lowercase, strip punctuation, drop a/an/the, "
                "collapse whitespace - src/answer_metrics.py, ADR 0019 decision 3). The "
                "asymmetry is recorded rather than removed: switching MetaQA to SQuAD "
                "normalization would change every MetaQA number ever reported and is a "
                "scoring-rule decision for Jahid with his supervisor (PR #42 review, "
                "finding 4)."
            ),
        },
        "composed_from": {
            "compile_execute": "scripts/evaluate_decompositions.py::evaluate_decomposition_rate",
            "gold_comparison": "scripts/compare_answer_accuracy.py::run_analysis",
            "kg": "scripts/kg.py::build_metaqa_kg",
        },
    }

    snapshot = {
        "script": Path(__file__).name,
        "created_utc": now_iso(),
        "config_path": cfg.get("_config_path"),
        "kg_eval_config_path": kg_cfg.get("_config_path"),
        "answer_accuracy_config_path": acc_cfg.get("_config_path"),
        "backend_label": backend_label,
        "grag_wired": GRAG_WIRED,
        "predictions": str(args.predictions),
        "predictions_sha256": eval_set_record["predictions_sha256"],
        "evaluation_set": eval_set_record,
        "upstream_decomposer_run": upstream,
        "allow_unpinned_eval_set": args.allow_unpinned_eval_set,
        "require_gold_for_every_item": require(cfg, "eval_set.require_gold_for_every_item"),
        "kb": str(kb_path),
        "data_root": str(data_root),
        "run_dir": str(run_dir),
        "analysis_dir": str(analysis_dir),
        "max_items": max_items,
        "hops": hops,
        "exact_match_jaccard": exact_threshold,
        "answer_separator": require(acc_cfg, "answer_separator"),
        "seed": seed,
        "seeded": seeded,
        "seed_sources": {
            "governing": f"{cfg.get('_config_path')} (or --seed)",
            "unused_component_seeds": {
                str(kg_cfg.get("_config_path")): kg_cfg.get("seed"),
                str(acc_cfg.get("_config_path")): acc_cfg.get("seed"),
            },
            "note": "the composed scripts' functions are called directly, so their configs' "
            "seed keys do not apply to this run; they are recorded for provenance only",
        },
        "dumps_written": [str(analysis_dir / name) for name in DUMP_FILES],
    }

    write_run_artifacts(
        run_dir,
        config_snapshot=snapshot,
        metrics=metrics,
        note_title=f"MetaQA end-to-end ({backend_label})",
        note_lines=[
            f"- **Backend: `{backend_label}` — GRAG is NOT wired.** {GRAG_STATUS}",
            f"- Predictions: `{args.predictions}` (sha256 "
            f"{eval_set_record['predictions_sha256'][:16]}...)",
            f"- Evaluation set: {eval_set_record['rows_processed']} row(s) "
            f"{eval_set_record['rows_per_hop']}, pinned={eval_set_record['pinned']} "
            f"({eval_set_record['unmatched_question_count']} without gold"
            + (", --allow-unpinned-eval-set given" if args.allow_unpinned_eval_set else "")
            + f"), question-set fingerprint `{eval_set_record['question_set_sha256']}`",
            "- Upstream decomposer run: "
            + (
                f"`{upstream['run_id']}` ({upstream.get('model')}, condition "
                f"{upstream.get('condition')}) at commit `{upstream.get('commit')}`"
                + (" — **dirty tree**" if upstream.get("dirty") else "")
                if upstream["found"]
                else "unrecorded — " + str(upstream["note"])
            ),
            f"- KB: `{kb_path}` ({len(kg.id_to_entity)} entities, {len(kg.triples)} triples); "
            "no model loaded",
            f"- Coverage: {coverage['total']} item(s), compiled "
            f"{coverage['compiled_ok']} ({coverage['compiled_ok_rate']:.2%}), executed "
            f"{coverage['executed_ok']} ({coverage['executed_ok_rate']:.2%})",
            f"- Compile fail reasons: {coverage['compile_fail_reasons']}",
            f"- Exec fail reasons: {coverage['exec_fail_reasons']}",
            f"- Over EXECUTED items with gold ({summary['total_with_gold']}): exact "
            f"{summary['total_exact_match']} ({summary['overall_pct_exact']}%), mean Jaccard "
            f"{summary['overall_mean_jaccard']}",
            f"- Over ALL items with gold ({items_with_gold}, failures counted as the empty "
            f"answer set): exact {over_all['exact_match_count']} ({over_all['pct_exact']}%), "
            f"mean Jaccard {over_all['mean_jaccard']}"
            + (
                f" — {over_all['empty_gold_set_note']}"
                if over_all.get("empty_gold_set_note")
                else ""
            ),
        ]
        + [
            f"- **{hop}-hop**: answered {summary['per_hop'][str(hop)]['answered_count']} of "
            f"{summary['per_hop'][str(hop)]['total_gold_questions']} gold "
            f"({summary['per_hop'][str(hop)]['coverage_pct']}% coverage), exact "
            f"{summary['per_hop'][str(hop)]['exact_match_count']} "
            f"({summary['per_hop'][str(hop)]['pct_exact']}%), mean Jaccard "
            f"{summary['per_hop'][str(hop)]['mean_jaccard']}"
            for hop in hops
        ]
        + [f"- Per-item: `{details_path}`, dumps: `{analysis_dir}`"],
        prefix=out_prefix,
    )
    print(f"\nResults saved to: {run_dir}")


if __name__ == "__main__":
    main()
