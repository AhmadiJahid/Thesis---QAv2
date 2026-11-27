#!/usr/bin/env python3
"""Re-score stored decompositions the way Han & Gardent (NAACL 2025) scored MuSiQue.

ADR 0041 (Tier 0 of the memo's plan, C3 item 6 / C4 row 0) and exp-035. CPU only, no model
is loaded. The input is the evaluator's own ``eval_per_item.json`` (schema
``musique_decomposition_per_item/1``), so the predictions, the gold and the item ids are
exactly the ones the evaluated arm was scored on.

Why this exists. The repo's evaluator ports the official Break trio with three named
deviations (docs/METRICS.md §2.2, ADR 0026): no spaCy lemmas in GED's node cost, no
``return``-stripping / ``;``-splitting, and a GED node cap + time budget that keep every
item instead of dropping it. Han & Gardent ran the official ``allenai/break-evaluator``
through their own fork (github.com/hankelvin/complex_question_decomposition, read at
commit 9c9a6699ec440175192004a12fb2526c1d7a40c8, ``evaluation/evaluate_break_suite.py`` +
``tools/break_evaluator/scripts/evaluate_predictions.py``). Their paper's footnote 11 says
they "lower-case as well as strip question marks and trailing whitespace in every question".
Two variants are therefore scored:

``hg_fork`` (primary) — the code path that produced their numbers, step by step:

1. the steps are joined ``" ;"`` (their CSV's ``decomposition`` column) and passed through
   ``clean_lingering_operators_cot`` (CoT-prefix cut, ``;`` split, ``[...]`` removal,
   whitespace collapse, empty parts dropped, re-joined ``" ;"``);
2. ``format_qdmr``: whitespace collapse, ``;`` split, ``re.sub('return', '', part)``,
   ``#k`` -> ``@@k@@``, empty parts dropped;
3. ``Decomposition.to_string``: parts joined ``" @@SEP@@ "``; then the fork's
   ``strip_qmark_whitespace``: ``re.sub(r'\\?+$', '', s.strip()).strip()`` on the **joined
   string** (so only the final step's question mark goes);
4. EM = ``pred.lower() == gold.lower()`` on those strings; SARI = official ``get_sari`` with
   the question split on ``" "`` as the source; GED = official
   ``normalized_graph_edit_distance`` on the step graphs of step 2 (no ``?`` strip), node cost
   ``1 - ratio`` over **spaCy lemma** sequences, optimizer run to exhaustion, and an item
   whose optimizer exceeds the 180 s budget is **dropped** from the GED mean (official
   ``@exit_after(180)`` + aggregator behaviour). Lowercasing happens only inside EM and in
   spaCy's lemmas: the fork's code lowercases nothing else.

``hg_paper_literal`` (sensitivity) — footnote 11 read literally: every sub-question and the
complex question are lowercased and have trailing ``?`` and whitespace stripped **before**
step 1, then the same pipeline runs.

Deviation that cannot be removed here: the fork pins spaCy 2.1.9 (Python <= 3.8); this
script uses whatever spaCy + ``en_core_web_sm`` the environment has, and records both
versions. Lemmas differ between spaCy 2 and 3 (e.g. pronouns no longer lemmatize to
``-PRON-``), so GED is "official algorithm, spaCy 3 lemmas", not bit-identical to theirs.
Non-comparability that is not a scorer property: Han & Gardent converted MuSiQue's
``entity >> relation`` gold sub-questions to natural language with their own templates;
the gold here is MuSiQue's as stored. Both are stated in every metrics file written.

Run::

    python scripts/break_hg_rescore.py --config configs/break_hg_rescore_exp035.json \\
        --runs-root ../../runs --run-dir ../../runs/exp-035
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable

import networkx as nx
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
sys.path.insert(0, str(_REPO_ROOT / "src"))

# The SARI arithmetic, the edit-distance alignment and the step graph are imported
# unchanged from the evaluator (each is already pinned against the official code by
# tests/test_decomposition_metrics.py), as are the paired tests ADR 0036 feeds to Holm.
from musique_decompositions_evaluator import (  # noqa: E402
    _alignment_matches,
    _decomposition_graph,
    _mcnemar,
    _paired_t_test_row,
    _sari,
)
from run_artifacts import now_iso, write_run_artifacts  # noqa: E402
from run_config import load_config, load_paths  # noqa: E402
from seeding import set_global_seed  # noqa: E402

VARIANTS = ("hg_fork", "hg_paper_literal")
PRIMARY_VARIANT = "hg_fork"
METRICS = ("break_exact_match", "sari", "ged")
STRATA = ("overall", "2", "3", "4")

_BREAK_SEP = " @@SEP@@ "
_COT_PATTERN = re.compile(r".+(the sequence of sub-questions should be: )(.+)", re.IGNORECASE)
_BRACKET_RX = re.compile(r"(\[.*?\])")
_TRAILING_QMARKS_RX = re.compile(r"\?+$")


# --------------------------------------------------------------------------------------
# Normalisation: the fork's string path
# --------------------------------------------------------------------------------------


def literal_normalize(text: str) -> str:
    """Footnote 11 read literally: lowercase, strip trailing ``?`` and whitespace."""
    return _TRAILING_QMARKS_RX.sub("", text.strip()).strip().lower()


def clean_lingering_operators(decomposition: str) -> str:
    """The fork's ``clean_lingering_operators_cot`` with ``cotnshot0=False``."""
    match = _COT_PATTERN.search(decomposition)
    if match and len(match.groups()) > 1 and len(match.group(2).strip()) > 0:
        decomposition = match.group(2)
    parts = []
    for part in decomposition.split(";"):
        if not part.strip():
            continue
        for rep in _BRACKET_RX.findall(part):
            part = part.replace(rep, "")
        parts.append(re.sub(r"\s{2,}", " ", part).strip())
    return " ;".join(parts)


def format_qdmr(decomposition: str) -> list[str]:
    """The fork's ``format_qdmr``: the step texts of its ``Decomposition``."""
    text = " ".join(decomposition.split())
    parts = [re.sub(r"return", "", part.strip().strip("\r")) for part in text.split(";")]
    parts = [re.sub(r"#(\d+)", r"@@\g<1>@@", part) for part in parts]
    return [p for p in parts if p.strip()]


def to_string(parts: list[str]) -> str:
    """``Decomposition.to_string``."""
    return _BREAK_SEP.join(x.replace("  ", " ").strip() for x in parts)


def strip_qmark_whitespace(joined: str) -> str:
    """The fork's ``str_strip_qmark_whitespace`` step, applied to the joined string."""
    return _TRAILING_QMARKS_RX.sub("", joined.strip()).strip()


def prepare(steps: list[str], variant: str) -> tuple[list[str], str]:
    """Steps -> (graph parts, the string EM and SARI compare) for one variant."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    if variant == "hg_paper_literal":
        steps = [literal_normalize(s) for s in steps]
    parts = format_qdmr(clean_lingering_operators(" ;".join(steps)))
    return parts, strip_qmark_whitespace(to_string(parts))


def prepare_question(question: str, variant: str) -> str:
    return literal_normalize(question) if variant == "hg_paper_literal" else question


# --------------------------------------------------------------------------------------
# GED with the official node cost (spaCy lemmas)
# --------------------------------------------------------------------------------------


def make_lemma_tokenizer(model: str) -> Callable[[str], list[str]]:
    """``SequenceMatchScorer.clean_base`` with ``remove_stop_words=False``, cached."""
    import spacy

    parser = spacy.load(model, disable=["ner"])
    cache: dict[str, list[str]] = {}

    def lemmas(text: str) -> list[str]:
        if text not in cache:
            cache[text] = [tok.lemma_ for tok in parser(text)]
        return cache[text]

    return lemmas


def match_ratio(tokens_a: list[str], tokens_b: list[str]) -> float:
    """``edit_distance.SequenceMatcher(a, b).ratio()``, 1.0 when both are empty."""
    if not tokens_a and not tokens_b:
        return 1.0
    return 2.0 * _alignment_matches(tokens_a, tokens_b) / (len(tokens_a) + len(tokens_b))


def normalized_ged(
    pred_parts: list[str],
    gold_parts: list[str],
    lemmas: Callable[[str], list[str]],
    budget_seconds: float,
) -> tuple[float | None, str | None]:
    """Official ``normalized_graph_edit_distance``; ``None`` = dropped (budget exceeded).

    The budget is tested between the optimizer's successive approximations (the official
    ``exit_after`` interrupts mid-approximation; the difference only matters for an item
    that would be dropped either way, and every drop is counted).
    """
    graph_pred = _decomposition_graph(pred_parts)
    graph_gold = _decomposition_graph(gold_parts)
    total_pred = graph_pred.number_of_nodes() + graph_pred.number_of_edges()
    total_gold = graph_gold.number_of_nodes() + graph_gold.number_of_edges()
    normalization = max(total_pred, total_gold)
    if normalization == 0:
        return 0.0, "both_empty"

    def subst(node_a: dict[str, Any], node_b: dict[str, Any]) -> float:
        return 1.0 - match_ratio(lemmas(node_a["label"]), lemmas(node_b["label"]))

    started = time.monotonic()
    distance = None
    for value in nx.optimize_graph_edit_distance(graph_pred, graph_gold, node_subst_cost=subst):
        distance = value
        if time.monotonic() - started > budget_seconds:
            return None, "dropped_time_budget"
    if distance is None:
        return None, "dropped_no_optimizer_result"
    return float(distance) / normalization, None


# --------------------------------------------------------------------------------------
# Scoring one arm
# --------------------------------------------------------------------------------------


def score_item(
    item: dict[str, Any],
    variant: str,
    lemmas: Callable[[str], list[str]],
    budget_seconds: float,
) -> dict[str, Any]:
    pred_parts, pred_str = prepare(list(item["pred_steps"]), variant)
    gold_parts, gold_str = prepare(list(item["gold_steps"]), variant)
    question = prepare_question(item["question"], variant)
    ged, ged_note = normalized_ged(pred_parts, gold_parts, lemmas, budget_seconds)
    return {
        "item_id": item["item_id"],
        "gold_hop_count": int(item["gold_hop_count"]),
        "break_exact_match": 1.0 if pred_str.lower() == gold_str.lower() else 0.0,
        "sari": _sari(question, pred_str, gold_str),
        "ged": ged,
        "ged_note": ged_note,
    }


def load_items(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    items = payload["items"]
    ids = [it["item_id"] for it in items]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"{path}: duplicate item_id values; refusing to score")
    return items


def apply_limit(items: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    """Smoke mode: the first ``limit`` items of EACH gold hop, file order kept."""
    if not limit:
        return items
    seen: dict[int, int] = {}
    out = []
    for it in items:
        hop = int(it["gold_hop_count"])
        if seen.get(hop, 0) < limit:
            seen[hop] = seen.get(hop, 0) + 1
            out.append(it)
    return out


def score_arm(job: dict[str, Any]) -> dict[str, Any]:
    """One (arm, variant) job. Top-level so a process pool can pickle it."""
    lemmas = make_lemma_tokenizer(job["spacy_model"])
    items = load_items(Path(job["per_item_path"]))
    items = apply_limit(items, job.get("limit"))
    started = time.monotonic()
    rows = [score_item(it, job["variant"], lemmas, job["budget_seconds"]) for it in items]
    return {
        "arm": job["arm"],
        "variant": job["variant"],
        "rows": rows,
        "seconds": time.monotonic() - started,
    }


# --------------------------------------------------------------------------------------
# Aggregation, paired family, Holm
# --------------------------------------------------------------------------------------


def stratum_rows(rows: list[dict[str, Any]], stratum: str) -> list[dict[str, Any]]:
    if stratum == "overall":
        return rows
    return [r for r in rows if r["gold_hop_count"] == int(stratum)]


def _mean(values: list[float]) -> float | None:
    """Mean, or None for an empty stratum (never NaN in the metrics JSON)."""
    return float(np.mean(values)) if values else None


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for stratum in STRATA:
        sub = stratum_rows(rows, stratum)
        geds = [r["ged"] for r in sub if r["ged"] is not None]
        out[stratum] = {
            "n": len(sub),
            "break_exact_match": _mean([r["break_exact_match"] for r in sub]),
            "sari": _mean([r["sari"] for r in sub]),
            "ged": _mean(geds),
            "ged_n_scored": len(geds),
            "ged_dropped": len(sub) - len(geds),
        }
    return out


def holm(p_values: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (same arithmetic as exp-029/031/034)."""
    m = len(p_values)
    order = sorted(range(m), key=lambda k: p_values[k])
    adjusted = [0.0] * m
    running = 0.0
    for rank, k in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[k]))
        adjusted[k] = running
    return adjusted


def paired_test(
    rows_a: list[dict[str, Any]],
    rows_b: list[dict[str, Any]],
    metric: str,
    alpha: float,
    min_items: int,
) -> dict[str, Any]:
    if [r["item_id"] for r in rows_a] != [r["item_id"] for r in rows_b]:
        raise SystemExit("paired rows are not aligned by item_id")
    if metric == "break_exact_match":
        row = _mcnemar(rows_a, rows_b, alpha, len(rows_a) < min_items, statistics=(metric,))[metric]
        row["test"] = "mcnemar_exact"
        row["n_paired"] = len(rows_a)
        return row
    keep = [
        i for i, (a, b) in enumerate(zip(rows_a, rows_b))
        if a[metric] is not None and b[metric] is not None
    ]
    values_a = np.array([rows_a[i][metric] for i in keep], dtype=float)
    values_b = np.array([rows_b[i][metric] for i in keep], dtype=float)
    row = _paired_t_test_row(values_a, values_b, alpha, len(keep) < min_items, name=metric)
    row["test"] = "paired_t"
    row["n_paired"] = len(keep)
    row["pairs_dropped_for_missing_ged"] = len(rows_a) - len(keep)
    return row


def family(
    scored: dict[str, list[dict[str, Any]]],
    pairs: list[list[str]],
    alpha: float,
    min_items: int,
) -> list[dict[str, Any]]:
    tests = []
    for arm_a, arm_b in pairs:
        index_b = {r["item_id"]: r for r in scored[arm_b]}
        rows_a = scored[arm_a]
        if set(index_b) != {r["item_id"] for r in rows_a}:
            raise SystemExit(f"{arm_a} vs {arm_b}: item id sets differ")
        rows_b = [index_b[r["item_id"]] for r in rows_a]
        for r_a, r_b in zip(rows_a, rows_b):
            if r_a["gold_hop_count"] != r_b["gold_hop_count"]:
                raise SystemExit(f"{arm_a} vs {arm_b}: gold hop differs on {r_a['item_id']}")
        for stratum in STRATA:
            sub_a = stratum_rows(rows_a, stratum)
            sub_b = stratum_rows(rows_b, stratum)
            for metric in METRICS:
                row = paired_test(sub_a, sub_b, metric, alpha, min_items)
                # Arm names go AFTER the test row and under their own keys: the paired-t
                # row carries numeric "system_a"/"system_b" (the means), which would
                # otherwise overwrite the names (exp-035's metrics.json has that layout).
                tests.append({**row, "arm_a": arm_a, "arm_b": arm_b, "stratum": stratum,
                              "metric": metric})
    usable = [t for t in tests if t.get("p_value") is not None]
    adjusted = holm([t["p_value"] for t in usable])
    for t, p_adj in zip(usable, adjusted):
        t["p_holm"] = p_adj
        t["significant_holm"] = bool(p_adj < alpha)
    for t in tests:
        t.setdefault("p_holm", None)
        t.setdefault("significant_holm", False)
    return tests


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stored_aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
    """The arm's existing evaluator values (repo port), per stratum, for the side-by-side."""
    rows = [
        {"gold_hop_count": int(it["gold_hop_count"]), "break_exact_match": it["break_exact_match"],
         "sari": it["sari"], "ged": it["ged"], "under_decomposition": it["under_decomposition"],
         "over_decomposition": it["over_decomposition"]}
        for it in items
    ]
    out = aggregate(rows)
    for stratum in STRATA:
        sub = stratum_rows(rows, stratum)
        out[stratum]["under_decomposition"] = _mean([r["under_decomposition"] for r in sub])
        out[stratum]["over_decomposition"] = _mean([r["over_decomposition"] for r in sub])
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--runs-root", type=Path, default=None,
                   help="where the arms' per-item paths resolve from (default: paths.json runs_root)")
    p.add_argument("--limit", type=int, default=None,
                   help="smoke: score only the first N items of each gold hop")
    p.add_argument("--workers", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    seed = int(cfg["seed"])
    seeded = set_global_seed(seed)
    runs_root = (args.runs_root or Path(load_paths()["runs_root_resolved"])).resolve()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    import spacy

    arms = cfg["arms"]
    inputs = {}
    for arm, rel in arms.items():
        path = runs_root / rel
        if not path.is_file():
            raise SystemExit(f"arm {arm}: per-item file not found: {path}")
        inputs[arm] = {"path": str(path), "sha256": sha256(path)}

    jobs = [
        {"arm": arm, "variant": variant, "per_item_path": inputs[arm]["path"],
         "spacy_model": cfg["spacy_model"], "budget_seconds": float(cfg["ged_budget_seconds"]),
         "limit": args.limit}
        for arm in arms for variant in VARIANTS
    ]
    started = now_iso()
    t0 = time.monotonic()
    workers = args.workers or int(cfg.get("workers", 1))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(score_arm, jobs))
    elapsed = time.monotonic() - t0

    scored: dict[str, dict[str, list[dict[str, Any]]]] = {v: {} for v in VARIANTS}
    seconds: dict[str, dict[str, float]] = {v: {} for v in VARIANTS}
    for res in results:
        scored[res["variant"]][res["arm"]] = res["rows"]
        seconds[res["variant"]][res["arm"]] = res["seconds"]

    stored = {}
    for arm in arms:
        items = load_items(Path(inputs[arm]["path"]))
        items = apply_limit(items, args.limit)
        stored[arm] = stored_aggregate(items)

    pcfg = cfg["paired_family"]
    tests = family(scored[PRIMARY_VARIANT], pcfg["pairs"], float(pcfg["alpha"]),
                   int(pcfg["min_items_for_significance_claim"]))

    metrics = {
        "schema": "break_hg_rescore/1",
        "exp_id": cfg.get("exp_id"),
        "created_utc": now_iso(),
        "seed": seed,
        "seeded": seeded,
        "limit": args.limit,
        "primary_variant": PRIMARY_VARIANT,
        "variants": {
            "hg_fork": "Han & Gardent's fork code path (strip trailing ?+ on the joined string; EM lowercased; spaCy-lemma GED; >budget items dropped from GED)",
            "hg_paper_literal": "footnote 11 read literally: every sub-question and the question lowercased, trailing ? and whitespace stripped, then the hg_fork path",
        },
        "environment": {
            "python": sys.version.split()[0],
            "spacy": spacy.__version__,
            "spacy_model": cfg["spacy_model"],
            "spacy_model_version": spacy.load(cfg["spacy_model"]).meta.get("version"),
            "networkx": nx.__version__,
            "numpy": np.__version__,
        },
        "deviations_from_han_gardent": cfg["deviations_from_han_gardent"],
        "inputs": inputs,
        "ged_budget_seconds": float(cfg["ged_budget_seconds"]),
        "wall_seconds_total": elapsed,
        "wall_seconds_per_job": seconds,
        "rescored": {v: {arm: aggregate(scored[v][arm]) for arm in arms} for v in VARIANTS},
        "stored_repo_port": stored,
        "paired_family": {
            "variant": PRIMARY_VARIANT,
            "size": len(tests),
            "size_with_p": sum(1 for t in tests if t.get("p_value") is not None),
            "alpha": float(pcfg["alpha"]),
            "correction": "Holm-Bonferroni over the whole family (ADR 0036); uncorrected verdict kept beside it",
            "tests": tests,
        },
    }
    per_item = {v: scored[v] for v in VARIANTS}
    (run_dir / "per_item.json").write_text(json.dumps(per_item), encoding="utf-8")
    note = [
        f"Variants: {', '.join(VARIANTS)} (primary {PRIMARY_VARIANT}). Arms: {', '.join(arms)}.",
        f"Items per arm: {len(scored[PRIMARY_VARIANT][next(iter(arms))])} (limit={args.limit}).",
        f"Wall clock: {elapsed:.1f} s over {workers} worker(s).",
        f"Paired family: {len(tests)} tests, Holm over "
        f"{metrics['paired_family']['size_with_p']} p-values.",
        "Numbers: metrics.json (this directory). Per-item values: per_item.json.",
    ]
    written = write_run_artifacts(
        run_dir,
        config_snapshot={"config_path": str(args.config), "config": cfg,
                         "runs_root": str(runs_root), "limit": args.limit, "workers": workers,
                         "started_utc": started},
        metrics=metrics,
        note_title=f"{cfg.get('exp_id', 'break_hg_rescore')}: Han & Gardent-style Break re-score",
        note_lines=note,
    )
    print(json.dumps({"run_dir": str(run_dir), "written": written,
                      "wall_seconds_total": round(elapsed, 1), "family_size": len(tests)}))


if __name__ == "__main__":
    main()
