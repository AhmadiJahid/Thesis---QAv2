#!/usr/bin/env python3
"""
Compile predicted MetaQA decompositions into KG query ops and execute them.

A decomposition "executes" when every step maps onto a supported template and the
resulting op chain runs against the MetaQA KG. The output is a compile/execute rate
with a reason breakdown, plus (optionally) the per-item success / compile-fail /
exec-fail dumps used for error analysis.

Ported from v1 ``scripts/evaluate_decompositions.py``. Adapted for v2: the kb path,
row cap and output directory come from ``configs/metaqa_kg_eval.json``, the run
writes the standard config/metrics/notes trail, and the ``scripts.kg`` import is a
plain module import so the script runs from any working directory.

The step-template regexes and relation rules below are deliberately still code:
they are a compiler for MetaQA's question grammar, not hyperparameters.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from kg import MetaQAKG, build_metaqa_kg  # noqa: E402
from run_artifacts import now_iso, write_run_artifacts  # noqa: E402
from run_config import load_config, load_paths, require, resolve_path, runs_path  # noqa: E402
from seeding import set_global_seed  # noqa: E402


# ============================================================
# EXPECTED KG INTERFACE
# ============================================================
# kg must have:
# - kg.triples: List[Tuple[int, str, int]]        (sid, rel, oid)
# - kg.out_adj: Dict[int, List[Tuple[str, int]]]  sid -> [(rel, oid)...]
# - kg.entity_to_id: Dict[str, int]
# - kg.id_to_entity: List[str]
# ============================================================


def build_reverse_index(kg: MetaQAKG) -> Dict[Tuple[str, int], Set[int]]:
    rev: Dict[Tuple[str, int], Set[int]] = defaultdict(set)
    for sid, rel, oid in kg.triples:
        rev[(rel, oid)].add(sid)
    return rev


# ============================================================
# Compiler: decomposition English -> query ops
# ============================================================

OpType = Literal[
    "VALUES_OF_MOVIE",            # movie -> values
    "MOVIES_WITH_VALUE",          # value -> movies (reverse index)
    "PROJECT_VALUES_FROM_MOVIES"  # movieset -> values
]


@dataclass(frozen=True)
class StepOp:
    op: OpType
    relation: str
    movie: Optional[str] = None        # for VALUES_OF_MOVIE
    entity: Optional[str] = None       # for MOVIES_WITH_VALUE (explicit value entity)
    ref_step: Optional[int] = None     # for MOVIES_WITH_VALUE (values from a previous step)


PLACEHOLDER_RX = re.compile(r"\[#(\d+)\]")
STEP_PREFIX_RX = re.compile(r"^\s*\d+\.\s*")


def normalize_step_text(step_line: str) -> str:
    step = STEP_PREFIX_RX.sub("", step_line).strip()
    return step.replace("’", "'").strip()


# ---- Relation inference ----
# We infer the *target KG relation* needed for the step.
# IMPORTANT: many "actor" wordings all map to MetaQA's starred_actors relation.
REL_RULES: List[Tuple[re.Pattern, str]] = [
    # Actors (movie <-> actor)
    (re.compile(r"\b(actor|actors|acted|act|starred|star|appeared|appear|played|cast|co-starred|co-star)\b", re.I), "starred_actors"),

    # Directors
    (re.compile(r"\b(director|directed|direct)\b", re.I), "directed_by"),

    # Writers / screenwriters
    (re.compile(r"\b(writer|writers|wrote|written|screenwriter|screenwriters|scriptwriter|scriptwriters|screenplay)\b", re.I), "written_by"),

    # Genres / types (MetaQA uses has_genre)
    (re.compile(r"\b(genre|genres|type|types)\b", re.I), "has_genre"),

    # Languages
    (re.compile(r"\b(language|languages)\b", re.I), "in_language"),

    # Years / release dates (MetaQA uses release_year)
    (re.compile(r"\b(release year|release years|released|release date|release dates|year|years)\b", re.I), "release_year"),

    # Tags / ratings / votes
    (re.compile(r"\b(tags?)\b", re.I), "has_tags"),
    (re.compile(r"\b(imdb rating)\b", re.I), "has_imdb_rating"),
    (re.compile(r"\b(imdb votes)\b", re.I), "has_imdb_votes"),
]


def infer_relation(step: str) -> str:
    for rx, rel in REL_RULES:
        if rx.search(step):
            return rel
    raise ValueError(f"Cannot infer relation from: {step!r}")


# ---- Step template patterns ----

# A) Person -> movies (reverse lookup): "What movies did PERSON star in?"
RX_PERSON_DID_VERB_IN = re.compile(
    r"^what\s+(movies|films).*\b(did|does|was|were)\b\s+(?P<person>.+?)\s+(?:an\s+)?(?:actor\s+)?(?:act|acted|star|starred|appear|appeared).*\bin\??$",
    re.I,
)
# "What movies was PERSON in?"
RX_PERSON_WAS_IN = re.compile(
    r"^what\s+(movies|films).*\bwas\b\s+(?P<person>.+?)\s+\bin\??$",
    re.I,
)
# "What movies did PERSON direct / write?"
RX_PERSON_DIRECT_WRITE = re.compile(
    r"^what\s+(movies|films).*\b(did|does)\b\s+(?P<person>.+?)\s+\b(direct|directed|write|wrote)\b.*\??$",
    re.I,
)

# B) Movie -> values (forward lookup)
RX_WHO_DIRECTED_MOVIE = re.compile(r"^who\s+directed\s+(?P<movie>.+?)\??$", re.I)
RX_WHO_WROTE_MOVIE = re.compile(
    r"^who\s+(?:wrote|is\s+the\s+writer\s+of|is\s+the\s+screenwriter\s+of|is\s+listed\s+as\s+screenwriter\s+of)\s+(?P<movie>.+?)\??$",
    re.I,
)
RX_WHO_STARRED_IN_MOVIE = re.compile(r"^who\s+(?:starred|acted|appeared)\s+in\s+(?P<movie>.+?)\??$", re.I)
RX_WHO_ARE_ACTORS_IN_MOVIE = re.compile(r"^who\s+(?:are|were)\s+the\s+(?:actors|actor|cast)\s+in\s+(?P<movie>.+?)\??$", re.I)
RX_WHAT_ACTORS_IN_MOVIE = re.compile(r"^what\s+(?:actors|actor|cast)\s+(?:were|was|are)\s+in\s+(?P<movie>.+?)\??$", re.I)

# C) Movie -> values with "of"
RX_WHO_IS_THE_X_OF_MOVIE = re.compile(r"^who\s+is\s+the\s+.+?\s+of\s+(?P<movie>.+?)\??$", re.I)

# D) Movie -> values with "was MOVIE directed by / written by"
RX_WHAT_MOVIES_WAS_MOVIE_DIRECTED_BY = re.compile(r"^what\s+movies?\s+was\s+(?P<movie>.+?)\s+directed\s+by\??$", re.I)
RX_WHAT_MOVIES_WAS_MOVIE_WRITTEN_BY = re.compile(r"^what\s+movies?\s+was\s+(?P<movie>.+?)\s+(?:screenplay\s+)?written\s+by\??$", re.I)

# E) Placeholder-based steps
# "What other movies were directed by [#1]?" -> MOVIES_WITH_VALUE(ref_step=1)
RX_OTHER_MOVIES_BY_PLACEHOLDER = re.compile(r"^what\s+other\s+(movies|films).*\bby\b\s+\[#(?P<k>\d+)\]\??$", re.I)

# If a step contains [#k] and is NOT "other movies by [#k]", it is almost always a
# projection from the movie set produced by step k.
RX_PROJECT_VALUES_FROM_MOVIESET = re.compile(r"^(who|what|when).*\[#(?P<k>\d+)\].*$", re.I)


def compile_decomposition(decomposition: str) -> List[StepOp]:
    lines = [ln.strip() for ln in decomposition.split("\n") if ln.strip()]
    ops: List[StepOp] = []

    for ln in lines:
        step = normalize_step_text(ln)
        rel = infer_relation(step)

        ph = PLACEHOLDER_RX.search(step)
        has_ph = ph is not None

        # ----- 1) Placeholder steps -----
        m = RX_OTHER_MOVIES_BY_PLACEHOLDER.match(step)
        if m:
            # reverse lookup: values from step k -> movies
            ops.append(StepOp(op="MOVIES_WITH_VALUE", relation=rel, ref_step=int(m.group("k"))))
            continue

        if has_ph:
            ops.append(
                StepOp(op="PROJECT_VALUES_FROM_MOVIES", relation=rel, ref_step=int(ph.group(1)))
            )
            continue

        # ----- 2) Non-placeholder steps -----
        m = RX_PERSON_DID_VERB_IN.match(step)
        if m:
            person = m.group("person").strip().rstrip("?")
            ops.append(StepOp(op="MOVIES_WITH_VALUE", relation=rel, entity=person))
            continue

        m = RX_PERSON_WAS_IN.match(step)
        if m:
            person = m.group("person").strip().rstrip("?")
            # Almost always a filmography question => starred_actors
            ops.append(StepOp(op="MOVIES_WITH_VALUE", relation="starred_actors", entity=person))
            continue

        m = RX_PERSON_DIRECT_WRITE.match(step)
        if m:
            person = m.group("person").strip().rstrip("?")
            ops.append(StepOp(op="MOVIES_WITH_VALUE", relation=rel, entity=person))
            continue

        m = RX_WHO_DIRECTED_MOVIE.match(step)
        if m:
            ops.append(
                StepOp(op="VALUES_OF_MOVIE", relation="directed_by", movie=m.group("movie").strip().rstrip("?"))
            )
            continue

        m = RX_WHO_WROTE_MOVIE.match(step)
        if m:
            ops.append(
                StepOp(op="VALUES_OF_MOVIE", relation="written_by", movie=m.group("movie").strip().rstrip("?"))
            )
            continue

        m = RX_WHO_STARRED_IN_MOVIE.match(step)
        if m:
            ops.append(
                StepOp(op="VALUES_OF_MOVIE", relation="starred_actors", movie=m.group("movie").strip().rstrip("?"))
            )
            continue

        m = RX_WHO_ARE_ACTORS_IN_MOVIE.match(step) or RX_WHAT_ACTORS_IN_MOVIE.match(step)
        if m:
            ops.append(
                StepOp(op="VALUES_OF_MOVIE", relation="starred_actors", movie=m.group("movie").strip().rstrip("?"))
            )
            continue

        m = RX_WHO_IS_THE_X_OF_MOVIE.match(step)
        if m:
            ops.append(StepOp(op="VALUES_OF_MOVIE", relation=rel, movie=m.group("movie").strip().rstrip("?")))
            continue

        m = RX_WHAT_MOVIES_WAS_MOVIE_DIRECTED_BY.match(step)
        if m:
            ops.append(
                StepOp(op="VALUES_OF_MOVIE", relation="directed_by", movie=m.group("movie").strip().rstrip("?"))
            )
            continue

        m = RX_WHAT_MOVIES_WAS_MOVIE_WRITTEN_BY.match(step)
        if m:
            ops.append(
                StepOp(op="VALUES_OF_MOVIE", relation="written_by", movie=m.group("movie").strip().rstrip("?"))
            )
            continue

        raise NotImplementedError(f"Unsupported step template: {step!r}")

    return ops


# ============================================================
# Executor
# ============================================================


class KGExecutionError(Exception):
    pass


def must_entity_id(kg: MetaQAKG, name: str) -> int:
    if name not in kg.entity_to_id:
        raise KeyError(name)
    return kg.entity_to_id[name]


def exec_ops(
    kg: MetaQAKG,
    reverse_index: Dict[Tuple[str, int], Set[int]],
    ops: List[StepOp],
) -> Set[int]:
    """Execute ops sequentially; returns the ID set produced by the last op."""
    results: Dict[int, Set[int]] = {}

    for i, op in enumerate(ops, start=1):
        if op.op == "VALUES_OF_MOVIE":
            if not op.movie:
                raise KGExecutionError(f"Step {i} missing movie")
            mid = must_entity_id(kg, op.movie)
            results[i] = {oid for rel, oid in kg.out_adj[mid] if rel == op.relation}
            continue

        if op.op == "MOVIES_WITH_VALUE":
            movie_ids: Set[int] = set()

            if op.entity is not None:
                vid = must_entity_id(kg, op.entity)
                movie_ids |= reverse_index.get((op.relation, vid), set())
                results[i] = movie_ids
                continue

            if op.ref_step is not None:
                prev_vals = results.get(op.ref_step)
                if prev_vals is None:
                    raise KGExecutionError(f"Step {i} refers to missing step #{op.ref_step}")
                for vid in prev_vals:
                    movie_ids |= reverse_index.get((op.relation, vid), set())
                results[i] = movie_ids
                continue

            raise KGExecutionError(f"Step {i} missing entity/ref")

        if op.op == "PROJECT_VALUES_FROM_MOVIES":
            if op.ref_step is None:
                raise KGExecutionError(f"Step {i} missing ref_step")
            prev_movies = results.get(op.ref_step)
            if prev_movies is None:
                raise KGExecutionError(f"Step {i} refers to missing step #{op.ref_step}")

            vals: Set[int] = set()
            for mid in prev_movies:
                for rel, oid in kg.out_adj[mid]:
                    if rel == op.relation:
                        vals.add(oid)
            results[i] = vals
            continue

        raise KGExecutionError(f"Unknown op: {op.op}")

    return results[len(ops)]


# ============================================================
# Decomposition rate (compile + execute) with reason breakdown
# ============================================================


@dataclass
class DecompositionMetrics:
    total: int
    compiled_ok: int
    executed_ok: int
    compile_fail: int
    exec_fail: int
    compile_fail_reasons: Counter
    exec_fail_reasons: Counter

    def as_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "compiled_ok": self.compiled_ok,
            "executed_ok": self.executed_ok,
            "compile_fail": self.compile_fail,
            "exec_fail": self.exec_fail,
            "compiled_ok_rate": self.compiled_ok / self.total if self.total else 0.0,
            "executed_ok_rate": self.executed_ok / self.total if self.total else 0.0,
            "compile_fail_reasons": dict(self.compile_fail_reasons),
            "exec_fail_reasons": dict(self.exec_fail_reasons),
        }


def evaluate_decomposition_rate(
    kg: MetaQAKG,
    results_json_path: str | Path,
    max_items: Optional[int] = None,
    output_dir: Optional[str | Path] = None,
) -> DecompositionMetrics:
    items = json.loads(Path(results_json_path).read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise ValueError("results.json must be a list of dicts")

    if max_items is not None:
        items = items[:max_items]

    reverse_index = build_reverse_index(kg)

    compile_fail_reasons: Counter = Counter()
    exec_fail_reasons: Counter = Counter()
    compiled_ok = executed_ok = compile_fail = exec_fail = 0

    success_items: List[dict] = []
    compile_fail_items: List[dict] = []
    exec_fail_items: List[dict] = []

    for it in items:
        dec = it.get("decomposition")
        if not isinstance(dec, str) or not dec.strip():
            compile_fail += 1
            compile_fail_reasons["missing_decomposition"] += 1
            compile_fail_items.append({**it, "error_reason": "missing_decomposition"})
            continue

        try:
            ops = compile_decomposition(dec)
        except NotImplementedError:
            compile_fail += 1
            compile_fail_reasons["unsupported_template"] += 1
            compile_fail_items.append({**it, "error_reason": "unsupported_template"})
            continue
        except ValueError:
            compile_fail += 1
            compile_fail_reasons["cannot_infer_relation"] += 1
            compile_fail_items.append({**it, "error_reason": "cannot_infer_relation"})
            continue
        except Exception as exc:
            compile_fail += 1
            compile_fail_reasons["compile_error_other"] += 1
            compile_fail_items.append({**it, "error_reason": f"compile_error_other: {exc}"})
            continue

        compiled_ok += 1

        try:
            results = exec_ops(kg, reverse_index, ops)
            executed_ok += 1
            readable = [kg.id_to_entity[rid] for rid in results]
            success_items.append({**it, "ops": [str(o) for o in ops], "kg_results": readable})
        except KeyError as exc:
            exec_fail += 1
            exec_fail_reasons["entity_not_in_kb"] += 1
            exec_fail_items.append(
                {**it, "ops": [str(o) for o in ops], "error_reason": f"entity_not_in_kb: {exc}"}
            )
        except KGExecutionError as exc:
            exec_fail += 1
            exec_fail_reasons["bad_reference_or_plan"] += 1
            exec_fail_items.append(
                {**it, "ops": [str(o) for o in ops], "error_reason": f"bad_reference_or_plan: {exc}"}
            )
        except Exception as exc:
            exec_fail += 1
            exec_fail_reasons["exec_error_other"] += 1
            exec_fail_items.append(
                {**it, "ops": [str(o) for o in ops], "error_reason": f"exec_error_other: {exc}"}
            )

    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        for name, payload in (
            ("success.json", success_items),
            ("compile_fail.json", compile_fail_items),
            ("exec_fail.json", exec_fail_items),
        ):
            (out_path / name).write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
        print(f"\nSaved error analysis files to: {out_path}")

    return DecompositionMetrics(
        total=len(items),
        compiled_ok=compiled_ok,
        executed_ok=executed_ok,
        compile_fail=compile_fail,
        exec_fail=exec_fail,
        compile_fail_reasons=compile_fail_reasons,
        exec_fail_reasons=exec_fail_reasons,
    )


def print_metrics(m: DecompositionMetrics) -> None:
    if m.total == 0:
        print("No items.")
        return
    print("\n" + "=" * 40)
    print("   DECOMPOSITION EXECUTION METRICS")
    print("=" * 40)
    print(f"Total Questions:  {m.total}")
    print(f"Compiled OK:      {m.compiled_ok} ({m.compiled_ok / m.total:.2%})")
    print(f"Executed OK:      {m.executed_ok} ({m.executed_ok / m.total:.2%})")
    print(f"Compile fail:     {m.compile_fail} ({m.compile_fail / m.total:.2%})")
    print(f"Exec fail:        {m.exec_fail} ({m.exec_fail / m.total:.2%})")
    print("\nCompile fail reasons:")
    for k, v in m.compile_fail_reasons.most_common():
        print(f"  - {k}: {v}")
    print("\nExec fail reasons:")
    for k, v in m.exec_fail_reasons.most_common():
        print(f"  - {k}: {v}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="metaqa_kg_eval.json")
    p.add_argument("--predictions", type=Path, required=True, help="Decomposer results.json")
    p.add_argument("--kb", type=Path, default=None, help="Override the kb path from config.")
    p.add_argument("--max-items", type=int, default=None, help="Override the config row cap.")
    p.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Override the run directory (default: <predictions parent>/analysis).",
    )
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    paths_cfg = load_paths(require(cfg, "paths_config"))

    seed = args.seed if args.seed is not None else int(require(cfg, "seed"))
    seeded = set_global_seed(seed)
    max_items = args.max_items if args.max_items is not None else require(cfg, "max_items")

    kb_path = args.kb or resolve_path(
        require(paths_cfg, "datasets." + require(cfg, "kb_key")),
        Path(paths_cfg["data_root_resolved"]),
    )
    if not args.predictions.exists():
        raise SystemExit(f"predictions not found: {args.predictions}")

    if args.run_dir is not None:
        run_dir = args.run_dir
    elif args.predictions.parent.is_dir():
        run_dir = args.predictions.parent / "analysis"
    else:
        run_dir = runs_path(paths_cfg, require(cfg, "run_subdir"))
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading KG from {kb_path} ...")
    kg = build_metaqa_kg(kb_path)

    print(f"Evaluating decompositions in {args.predictions} ...")
    metrics_obj = evaluate_decomposition_rate(
        kg,
        args.predictions,
        max_items=int(max_items) if max_items is not None else None,
        output_dir=run_dir if require(cfg, "write_error_analysis") else None,
    )
    print_metrics(metrics_obj)

    metrics = {
        "script": Path(__file__).name,
        "created_utc": now_iso(),
        "seed": seed,
        "seeded": seeded,
        "predictions_path": str(args.predictions.resolve()),
        "kb_path": str(Path(kb_path).resolve()),
        "kg_entities": len(kg.id_to_entity),
        "kg_triples": len(kg.triples),
        **metrics_obj.as_dict(),
    }
    snapshot = {
        "script": Path(__file__).name,
        "created_utc": now_iso(),
        "config_path": cfg.get("_config_path"),
        "predictions": str(args.predictions),
        "kb": str(kb_path),
        "max_items": max_items,
        "run_dir": str(run_dir),
        "seed": seed,
        "write_error_analysis": require(cfg, "write_error_analysis"),
    }
    write_run_artifacts(
        run_dir,
        config_snapshot=snapshot,
        metrics=metrics,
        note_title="MetaQA decomposition execution",
        note_lines=[
            f"- Predictions: `{args.predictions}`",
            f"- KB: `{kb_path}` ({len(kg.id_to_entity)} entities, {len(kg.triples)} triples)",
            f"- Items: {metrics_obj.total}",
            f"- Compiled OK: {metrics_obj.compiled_ok} ({metrics['compiled_ok_rate']:.2%})",
            f"- Executed OK: {metrics_obj.executed_ok} ({metrics['executed_ok_rate']:.2%})",
            f"- Compile fail reasons: {dict(metrics_obj.compile_fail_reasons)}",
            f"- Exec fail reasons: {dict(metrics_obj.exec_fail_reasons)}",
        ],
        prefix="kg_eval_",
    )


if __name__ == "__main__":
    main()
