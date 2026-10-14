#!/usr/bin/env python3
"""Build the few-shot retrieval artifact over a whole named evaluation set (issue #56).

exp-014 built ``runs/exp-014/retrieval/s2_pinned600_imbalanced_rerank_top5.jsonl`` - the
artifact every issue #50 arm reads - with two commands over the three pinned files of
ADR 0007. This script runs **the same two commands** (the same scripts, the same pool, the
same flags, the same configs/similarity.json) with only the query side changed: the source
file of a named evaluation set from ``configs/eval_sets.json`` (``musique_dev_full``: the
2,417 answerable dev questions). Everything else comes from ``configs/dev_full_retrieval.json``.

Checks, in order, each a refusal unless stated:

1. the pool's sha256 is the one the config pins (ADR 0028's ``size2000_imbalanced``);
2. the named set loads cleanly, and every row of its source file is in the set (so the
   source file can be passed to the similarity script as the query file unfiltered);
3. pool ids ∩ eval-set ids = 0 (recorded; the question-text overlap is recorded beside it);
4. after the build: the artifact's ids are exactly the named set, one row each, and every
   row carries ``rerank_k`` exemplars in the compared mode;
5. **equivalence**: for every query id in the reference artifact (exp-014's, the pinned
   600), the new row's exemplar pool ids, in order, are compared with the reference row's.
   The match count is recorded; a mismatch exits non-zero after the trail is written.

Usage::

    .venv/bin/python scripts/build_dev_full_retrieval.py
    .venv/bin/python scripts/build_dev_full_retrieval.py --runs-root /abs/path/to/runs
    .venv/bin/python scripts/build_dev_full_retrieval.py --verify-only   # artifact exists
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from eval_sets import hop_from_id, load_named_eval_set  # noqa: E402
from run_artifacts import now_iso, write_run_artifacts  # noqa: E402
from run_config import load_config, load_paths, require  # noqa: E402
from seeding import set_global_seed  # noqa: E402

_WS_RX = re.compile(r"\s+")
SIMILARITY_SCRIPT = _REPO_ROOT / "MusiQue" / "scripts" / "check_question_similarity.py"
RERANK_SCRIPT = _REPO_ROOT / "MusiQue" / "scripts" / "rerank_similarity_results.py"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _norm_q(text: str) -> str:
    return _WS_RX.sub(" ", str(text).strip().lower())


def _exemplar_ids(row: dict[str, Any], mode: str) -> list[Any]:
    return [nb.get("pool_id") for nb in (row.get(f"{mode}_top_k") or [])]


def _run(cmd: list[str]) -> float:
    print("\n$ " + " ".join(cmd), flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True, cwd=str(_REPO_ROOT))
    return round(time.time() - t0, 1)


def compare_to_reference(
    new_rows: dict[str, dict[str, Any]],
    ref_rows: list[dict[str, Any]],
    mode: str,
    prompt_k: int,
) -> dict[str, Any]:
    """Exemplar-id-and-order equivalence of ``new_rows`` against ``ref_rows`` (by query id).

    Beside the strict top-k comparison, two diagnostics are counted (they explain a
    mismatch; they do not excuse one - ``equivalent`` stays strict):

    - ``mismatches_with_identical_score_sequence``: the two lists carry the same rerank
      score at every position, i.e. the difference is only which of several exactly-tied
      exemplars sits where;
    - ``first_prompt_k_match``: rows whose first ``prompt_k`` exemplar ids, in order, match
      (the decomposer shows only its few-shot k of the retrieved list).
    """
    matched = 0
    first_k_matched = 0
    tie_only = 0
    masked_query_matched = 0
    missing: list[str] = []
    mismatched: list[dict[str, Any]] = []
    for ref in ref_rows:
        qid = str(ref.get("query_id"))
        new = new_rows.get(qid)
        if new is None:
            missing.append(qid)
            continue
        if new.get(f"query_question_masked_{mode}") == ref.get(f"query_question_masked_{mode}"):
            masked_query_matched += 1
        a, b = _exemplar_ids(new, mode), _exemplar_ids(ref, mode)
        if a[:prompt_k] == b[:prompt_k]:
            first_k_matched += 1
        if a == b:
            matched += 1
        else:
            scores_a = [nb.get("score") for nb in new.get(f"{mode}_top_k") or []]
            scores_b = [nb.get("score") for nb in ref.get(f"{mode}_top_k") or []]
            if scores_a == scores_b:
                tie_only += 1
            mismatched.append(
                {"query_id": qid, "new": a, "reference": b, "identical_score_sequence": scores_a == scores_b}
            )
    return {
        "mode": mode,
        "reference_rows": len(ref_rows),
        "exemplar_ids_and_order_match": matched,
        "masked_query_text_match": masked_query_matched,
        "missing_from_new_artifact": len(missing),
        "missing_ids_sample": missing[:10],
        "mismatched": len(mismatched),
        "mismatches_with_identical_score_sequence": tie_only,
        "prompt_k": prompt_k,
        "first_prompt_k_match": first_k_matched,
        "mismatched_rows": mismatched,
        "equivalent": matched == len(ref_rows),
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="dev_full_retrieval.json")
    p.add_argument(
        "--runs-root",
        type=Path,
        default=None,
        help="Override configs/paths.json runs_root (e.g. the main checkout's runs/ when "
        "building from a worktree). Recorded in the trail.",
    )
    p.add_argument(
        "--verify-only",
        action="store_true",
        help="Do not build; run checks 1-5 against an artifact that already exists.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    paths_cfg = load_paths(require(cfg, "paths_config"))
    seed = int(require(cfg, "seed"))
    seeded = set_global_seed(seed)
    runs_root = (args.runs_root or Path(paths_cfg["runs_root_resolved"])).resolve()

    pool_path = runs_root / require(cfg, "pool_file")
    out_dir = runs_root / require(cfg, "out_dir")
    top20_path = out_dir / require(cfg, "top20_name")
    out_path = out_dir / require(cfg, "out_name")
    ref_path = runs_root / require(cfg, "reference.path")
    mode = require(cfg, "reference.compare_mode")
    sim_mode = require(cfg, "similarity.mode")
    rerank_k = int(require(cfg, "rerank.rerank_k"))

    # ---- 1. pool identity
    if not pool_path.exists():
        raise SystemExit(f"pool not found: {pool_path}")
    pool_sha = _sha256(pool_path)
    if pool_sha != require(cfg, "pool_sha256"):
        raise SystemExit(
            f"pool sha256 {pool_sha} != pinned {require(cfg, 'pool_sha256')}: {pool_path}"
        )
    pool_rows = _load_jsonl(pool_path)
    print(f"Pool: {len(pool_rows)} rows, sha256 {pool_sha}")

    # ---- 2. the named set, and its source file as the query file
    named = load_named_eval_set(require(cfg, "eval_set"), paths_cfg)
    if named.problems:
        raise SystemExit("eval set definition problems:\n  " + "\n  ".join(named.problems))
    source_path = Path(named.source_path)
    source_rows = _load_jsonl(source_path)
    source_ids = [str(r.get("id")) for r in source_rows]
    if len(source_ids) != len(named.ids) or set(source_ids) != named.ids:
        raise SystemExit(
            f"{source_path} has {len(source_ids)} rows but the named set has {len(named.ids)} "
            "ids: the source file holds rows outside the set, so it cannot be the query file "
            "unfiltered. Build a filtered query file first (not implemented: not needed for "
            "musique_dev_full, whose source rows are all answerable)."
        )
    print(f"Eval set {named.name}: {len(named.ids)} ids, per hop {named.ids_per_hop}")

    # ---- 3. pool ∩ eval set
    pool_ids = {str(r.get("id")) for r in pool_rows}
    id_overlap = sorted(pool_ids & named.ids)
    pool_q = {_norm_q(r.get("question", "")) for r in pool_rows}
    text_overlap = sum(1 for r in source_rows if _norm_q(r.get("question", "")) in pool_q)
    print(f"Pool ∩ eval set: {len(id_overlap)} ids, {text_overlap} normalized question texts")
    if id_overlap:
        raise SystemExit(f"pool and eval set share {len(id_overlap)} ids: {id_overlap[:10]}")

    # ---- build
    timings: dict[str, float] = {}
    commands: list[list[str]] = []
    if not args.verify_only:
        if out_path.exists() or top20_path.exists():
            raise SystemExit(
                f"refusing to overwrite an existing artifact in {out_dir}; remove it or pass "
                "--verify-only"
            )
        out_dir.mkdir(parents=True, exist_ok=True)
        sim_cmd = [
            sys.executable, str(SIMILARITY_SCRIPT),
            "--config", require(cfg, "similarity.config"),
            "--pool-file", str(pool_path),
            "--query-file", str(source_path),
            "--n", str(len(source_rows)),
            "--mode", sim_mode,
            "--top-k", str(int(require(cfg, "similarity.top_k"))),
            "--device", require(cfg, "similarity.device"),
            "--seed", str(seed),
            "--hop-match" if require(cfg, "similarity.hop_match") else "--no-hop-match",
            "--out", str(top20_path),
            "--run-dir", str(out_dir / "similarity"),
        ]
        rerank_cmd = [
            sys.executable, str(RERANK_SCRIPT),
            "--config", require(cfg, "rerank.config"),
            "--input", str(top20_path),
            "--out", str(out_path),
            "--rerank-k", str(rerank_k),
            "--device", require(cfg, "rerank.device"),
            "--seed", str(seed),
            "--run-dir", str(out_dir / "rerank"),
        ]
        commands = [sim_cmd, rerank_cmd]
        timings["similarity_seconds"] = _run(sim_cmd)
        timings["rerank_seconds"] = _run(rerank_cmd)

    # ---- 4. the artifact is exactly the named set
    if not out_path.exists():
        raise SystemExit(f"artifact not found: {out_path}")
    rows = _load_jsonl(out_path)
    by_id: dict[str, dict[str, Any]] = {}
    dups: list[str] = []
    short: list[str] = []
    for r in rows:
        qid = str(r.get("query_id"))
        if qid in by_id:
            dups.append(qid)
        by_id[qid] = r
        if len(_exemplar_ids(r, mode)) != rerank_k:
            short.append(qid)
    per_hop: dict[str, int] = {}
    for qid in by_id:
        h = str(hop_from_id(qid))
        per_hop[h] = per_hop.get(h, 0) + 1
    missing = sorted(named.ids - set(by_id))
    extra = sorted(set(by_id) - named.ids)
    artifact_ok = not (dups or missing or extra or short)

    # ---- 5. equivalence on the reference ids
    ref_sha = _sha256(ref_path)
    if ref_sha != require(cfg, "reference.sha256"):
        raise SystemExit(f"reference sha256 {ref_sha} != pinned {require(cfg, 'reference.sha256')}")
    equivalence = compare_to_reference(
        by_id, _load_jsonl(ref_path), mode, int(require(cfg, "reference.prompt_k"))
    )
    print(
        f"Equivalence vs reference ({mode}): {equivalence['exemplar_ids_and_order_match']}/"
        f"{equivalence['reference_rows']} rows match exemplar ids and order"
    )

    out_sha = _sha256(out_path)
    metrics = {
        "script": Path(__file__).name,
        "created_utc": now_iso(),
        "seed": seed,
        "seeded": seeded,
        "eval_set": named.as_record(),
        "pool_file": str(pool_path),
        "pool_rows": len(pool_rows),
        "pool_sha256": pool_sha,
        "pool_eval_set_id_overlap": len(id_overlap),
        "pool_eval_set_question_text_overlap": text_overlap,
        "artifact": str(out_path),
        "artifact_sha256": out_sha,
        "artifact_rows": len(rows),
        "artifact_rows_per_hop": {k: per_hop[k] for k in sorted(per_hop)},
        "artifact_ids_missing": len(missing),
        "artifact_ids_unexpected": len(extra),
        "artifact_duplicate_ids": len(dups),
        "artifact_rows_without_rerank_k_exemplars": len(short),
        "artifact_matches_eval_set": artifact_ok,
        "top20_artifact": str(top20_path),
        "top20_sha256": _sha256(top20_path) if top20_path.exists() else None,
        "reference": {"path": str(ref_path), "sha256": ref_sha},
        "equivalence": equivalence,
        "timings": timings or "not built in this invocation (--verify-only)",
    }
    write_run_artifacts(
        out_dir,
        config_snapshot={
            "script": Path(__file__).name,
            "config_path": cfg.get("_config_path"),
            "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
            "runs_root": str(runs_root),
            "verify_only": args.verify_only,
            "commands": [" ".join(c) for c in commands],
        },
        metrics=metrics,
        note_title=f"Few-shot retrieval artifact over {named.name} (exp-014's method)",
        note_lines=[
            f"- Pool: `{pool_path}` ({len(pool_rows)} rows, sha256 `{pool_sha}`)",
            f"- Eval set: `{named.name}` - {len(named.ids)} ids, per hop {named.ids_per_hop}, "
            f"source `{named.source_path}` (sha256 `{named.source_sha256}`)",
            f"- Pool ∩ eval set: {len(id_overlap)} ids; {text_overlap} normalized question texts",
            f"- Artifact: `{out_path}` - {len(rows)} rows, per hop {metrics['artifact_rows_per_hop']}, "
            f"sha256 `{out_sha}`; matches the eval set by id: {artifact_ok}",
            f"- Equivalence vs `{ref_path}` ({mode} exemplar pool ids, in order): "
            f"{equivalence['exemplar_ids_and_order_match']}/{equivalence['reference_rows']} rows "
            f"match; masked query text matches {equivalence['masked_query_text_match']}/"
            f"{equivalence['reference_rows']}; mismatches whose rerank-score sequence is "
            f"identical (exact ties reordered): "
            f"{equivalence['mismatches_with_identical_score_sequence']}/"
            f"{equivalence['mismatched']}; first {equivalence['prompt_k']} exemplar ids match "
            f"(what a k={equivalence['prompt_k']} prompt shows): "
            f"{equivalence['first_prompt_k_match']}/{equivalence['reference_rows']}",
            f"- Timings: {metrics['timings']}",
        ],
        prefix="build_",
    )
    if not artifact_ok:
        raise SystemExit(
            f"artifact does not match the eval set: missing {len(missing)}, unexpected "
            f"{len(extra)}, duplicates {len(dups)}, short rows {len(short)}"
        )
    if not equivalence["equivalent"]:
        raise SystemExit(
            f"EQUIVALENCE FAILED: {equivalence['mismatched']} mismatched and "
            f"{equivalence['missing_from_new_artifact']} missing reference rows (see "
            f"{out_dir}/build_metrics.json)"
        )


if __name__ == "__main__":
    main()
