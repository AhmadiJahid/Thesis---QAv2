#!/usr/bin/env python3
"""
The junk battery behind the decomposition report card's acceptance criteria.

Break prints a trivial ``Copy`` baseline in its own results table (TACL 2020, Table 9): a
system that echoes the input question, scored beside the real ones. That practice is the
cheapest guard there is against the issue #40 pathology — under the legacy composite, three
fixed junk steps scored **0.2778** and a question echo **0.2333** against exp-004
``unguided``'s **0.2098**, i.e. junk outranked the deployable baseline on its own evaluation
set, and it took two analysis notes to notice.

This script builds six junk systems out of the gold column, scores them with the evaluator's
own :func:`score_item` (no second copy of any metric), and prints the whole report card for
each of them beside every real arm's, plus the pass/fail verdicts:

    A1  the gold itself scores the extremum on every term
    A2  every real arm beats J1-J4 on ged_macro, chain_validity_macro, break_exact_match_rate
    A3  SARI is EXEMPT from A2 by design - its junk floor on this data is high (Break's own
        published Copy row scores SARI 0.431) - so the margin is recorded, not asserted
    A4  the additive blend decomp_mean must rank every junk system below every real arm.
        This is a HARD GATE: failing it disqualifies the blend rather than prompting a
        weight tweak, because the composite's weights were never the disease
    A5  order sensitivity survives: the reversed gold (J5) is caught by break_exact_match
        and ordered_step_accuracy, NOT by GED, which is order-light

It is a **tool, not an experiment** (ADR 0027 point 5): it scores no system of the thesis, it
takes no ``experiments/log.md`` entry, it needs no GPU and it takes no ``runs/run.lock``. The
junk systems are baselines built from the gold, in the same sense as Break's ``Copy`` row.
What it prints is what ADR 0029 records, at the SHA it prints.

The same construction functions are imported by
``tests/test_decomposition_metrics.py::TestJunkBattery``, which asserts A1-A6 over the
committed fixture gold — so the battery is smoke-testable at 4 items before it is run over
600.

Run::

    .venv/bin/python scripts/decomposition_junk_battery.py                     # full
    .venv/bin/python scripts/decomposition_junk_battery.py --limit 20          # tiny
    .venv/bin/python scripts/decomposition_junk_battery.py --json out.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from run_artifacts import now_iso  # noqa: E402
from run_config import load_config, load_paths, require, resolve_path, runs_path  # noqa: E402
from seeding import set_global_seed  # noqa: E402

EVALUATOR = REPO_ROOT / "scripts" / "musique_decompositions_evaluator.py"


def _import_evaluator() -> Any:
    """Import the evaluator by path, as ``scripts/ged_cost_benchmark.py`` does."""
    name = "musique_decompositions_evaluator"
    spec = importlib.util.spec_from_file_location(name, EVALUATOR)
    if spec is None or spec.loader is None:  # pragma: no cover - import plumbing
        raise SystemExit(f"cannot import {EVALUATOR}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


EVAL = _import_evaluator()


# --------------------------------------------------------------------------------------
# The six junk systems. Each is a function from (question, gold steps, junk step texts) to
# the steps the junk system "predicts". They are written out here rather than described,
# for the same reason the GED benchmark's shapes are: a baseline IS the measurement's
# definition, and prose cannot state one precisely enough to re-derive its numbers.
# --------------------------------------------------------------------------------------

JunkSystem = Callable[[str, list[str], list[str]], list[str]]

#: What each junk system is, printed with its row so a table cannot be read without them.
JUNK_DESCRIPTIONS: dict[str, str] = {
    "J1_empty": "empty prediction: zero steps",
    "J2_question_echo": "the question verbatim as a single step (= Break's own 'Copy')",
    "J3_one_fixed_step": "one constant step, the same string for every item",
    "J4_three_fixed_steps": "three constant steps, the same strings for every item",
    "J5_gold_reversed": "the gold steps in reverse order (content-perfect, mis-ordered)",
    "J6_gold": "the gold decomposition verbatim (the anchor)",
}

JUNK_SYSTEMS: dict[str, JunkSystem] = {
    "J1_empty": lambda question, gold, junk: [],
    "J2_question_echo": lambda question, gold, junk: [question],
    "J3_one_fixed_step": lambda question, gold, junk: list(junk[:1]),
    "J4_three_fixed_steps": lambda question, gold, junk: list(junk[:3]),
    "J5_gold_reversed": lambda question, gold, junk: list(reversed(gold)),
    "J6_gold": lambda question, gold, junk: list(gold),
}

#: The junk systems A2 and A4 are asserted against: J5 and J6 are built FROM the gold and
#: are not junk in that sense (J6 is the anchor, J5 is the order probe).
JUNK_BASELINES = ("J1_empty", "J2_question_echo", "J3_one_fixed_step", "J4_three_fixed_steps")

#: The columns a real arm's committed per-item file must carry for this battery to read it.
#: Every one of them predates this script; nothing here re-scores a committed arm.
_REAL_ARM_COLUMNS = (
    "break_exact_match",
    "sari",
    "ged",
    "chain_validity",
    "hop_count_exact_match",
    "ordered_step_accuracy",
    "step_count_signed_error",
)

#: The report card terms this battery prints and compares, in the specification's order.
_SUITE_KEYS = tuple(term.aggregate_key for term in EVAL.DECOMPOSITION_SUITE_TERMS)

#: The three terms A2 asserts on. SARI is deliberately NOT among them (A3).
A2_TERMS = ("ged_macro", "chain_validity_macro", "break_exact_match_rate")


def _lower_is_better(aggregate_key: str) -> bool:
    for term in EVAL.DECOMPOSITION_SUITE_TERMS:
        if term.aggregate_key == aggregate_key:
            return term.per_item_column in EVAL.LOWER_IS_BETTER_STATISTICS
    return False


def _beats(aggregate_key: str, candidate: float, other: float) -> bool:
    """Is ``candidate`` strictly better than ``other`` on this term, direction applied?"""
    return candidate < other if _lower_is_better(aggregate_key) else candidate > other


def _pinned_eval_questions(paths_cfg: dict[str, Any], cfg: dict[str, Any]) -> set[str]:
    """The normalized questions of the pinned evaluation set (ADR 0007: 200 per hop)."""
    template = require(paths_cfg, "datasets." + require(cfg, "eval_set.questions_key"))
    out: set[str] = set()
    for hop in require(cfg, "eval_set.hops"):
        path = resolve_path(
            str(template).format(hop=hop), Path(paths_cfg["data_root_resolved"])
        )
        if not path.exists():
            raise SystemExit(f"pinned evaluation set file not found: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                question = json.loads(line).get("question")
                if isinstance(question, str) and question.strip():
                    out.add(EVAL._normalize_question(question))
    if not out:
        raise SystemExit("the pinned evaluation set files carry no questions.")
    return out


def _assert_same_eval_set(
    scored_keys: set[str], arm_name: str, arm_items: list[dict[str, Any]], path: Path
) -> None:
    """Refuse to put a real arm beside the junk rows unless it is the SAME evaluation set.

    A junk baseline scored over 2411 gold rows and an arm scored over the pinned 600 are not
    comparable, and every A2/A4/A5 verdict is a junk-versus-arm comparison — so this is
    checked rather than assumed (CLAUDE.md, evidence discipline).
    """
    arm_keys = {EVAL._normalize_question(str(item["question"])) for item in arm_items}
    if arm_keys == scored_keys:
        return
    raise SystemExit(
        f"real arm {arm_name!r} was not evaluated on the same set as the junk baselines.\n"
        f"  arm: {path} ({len(arm_keys)} items)\n"
        f"  junk baselines: {len(scored_keys)} items\n"
        f"  only in the arm: {len(arm_keys - scored_keys)}; "
        f"only in the junk set: {len(scored_keys - arm_keys)}\n"
        f"Comparing across two evaluation sets is not a comparison. Item text is not printed "
        f"here: dataset content does not go into an error message (CLAUDE.md)."
    )


def _junk_predictions(
    gold_by_question: dict[str, dict[str, Any]],
    system: JunkSystem,
    junk_steps: list[str],
) -> list[dict[str, Any]]:
    """One prediction row per gold row, built by ``system`` from that row's own gold.

    The question is the gold's RAW text, not the normalized join key: J2 is Break's ``Copy``,
    which copies the input question "without introducing any modifications", and SARI reads
    the question as its source — a lowercased echo would be a different system.
    """
    return [
        {
            "query_id": key,
            "question": gold["question"],
            "decomposition": system(gold["question"], list(gold["steps"]), junk_steps),
        }
        for key, gold in gold_by_question.items()
    ]


def _score_system(
    predictions: list[dict[str, Any]],
    gold_by_question: dict[str, dict[str, Any]],
    ged_policy: dict[str, Any],
    decomp_policy: dict[str, Any],
) -> dict[str, Any]:
    """Score one junk system with the evaluator's own per-item function and aggregate it."""
    rows, missing = EVAL._build_eval_rows(predictions, gold_by_question)
    if missing:
        raise SystemExit(
            f"{missing} junk prediction(s) matched no gold row, which cannot happen when the "
            f"predictions are built from the gold — the question join changed."
        )
    items = [
        EVAL.score_item(
            row,
            ged_max_nodes=ged_policy["max_nodes_for_optimizer"],
            ged_time_budget=ged_policy["per_item_time_budget_seconds"],
            decomp_mean_components=tuple(decomp_policy["components"]),
            ged_clamp=decomp_policy["ged_clamp"],
        )
        for row in rows
    ]
    return _row_of(EVAL._aggregate(items), len(items))


def _row_of(aggregate: dict[str, Any], n_items: int) -> dict[str, Any]:
    row = {key: float(aggregate[key]) for key in _SUITE_KEYS}
    row["decomp_mean_macro"] = float(aggregate["decomp_mean_macro"])
    row["ordered_step_accuracy_macro"] = float(aggregate["ordered_step_accuracy_macro"])
    row["n_items"] = n_items
    return row


def _real_arm_row(
    path: Path, decomp_policy: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read a committed per-item file and aggregate it into the same shape.

    Read, never re-scored: the per-item columns are the ones the arm was scored with, and
    ``decomp_mean`` / the two direction columns are DERIVED from them by the same functions
    the evaluator uses — so an arm scored before those columns existed still gets a row.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise SystemExit(
            f"{path}: not a '{EVAL.PER_ITEM_SCHEMA}' per-item file (no 'items' list)."
        )
    items = payload["items"]
    if not items:
        raise SystemExit(f"{path}: no items.")
    for i, item in enumerate(items):
        missing = [c for c in (*_REAL_ARM_COLUMNS, "question") if c not in item]
        if missing:
            raise SystemExit(
                f"{path}: row {i} is missing {missing}, so this arm cannot be placed beside "
                f"the junk baselines. Re-score it with the evaluator."
            )
    enriched = []
    for item in items:
        signed = int(item["step_count_signed_error"])
        enriched.append(
            {
                **item,
                "over_decomposition": 1.0 if signed > 0 else 0.0,
                "under_decomposition": 1.0 if signed < 0 else 0.0,
                "decomp_mean": EVAL.decomp_mean(
                    item, tuple(decomp_policy["components"]), decomp_policy["ged_clamp"]
                ),
            }
        )
    return _row_of(EVAL._aggregate(enriched), len(enriched)), items


def _verdicts(rows: dict[str, dict[str, Any]], real_names: list[str]) -> dict[str, Any]:
    """A1-A5, computed from the rows rather than asserted in prose."""
    gold = rows["J6_gold"]
    a1_expected = {
        "break_exact_match_rate": 1.0,
        "sari_macro": 1.0,
        "ged_macro": 0.0,
        "chain_validity_macro": 1.0,
        "hop_count_exact_match_rate": 1.0,
        "under_decomposition_rate": 0.0,
        "over_decomposition_rate": 0.0,
    }
    a1_failures = [
        f"{key}: {gold[key]!r} != {value!r}"
        for key, value in a1_expected.items()
        if abs(gold[key] - value) > 1e-9
    ]

    a2_failures: list[str] = []
    for term in A2_TERMS:
        for junk in JUNK_BASELINES:
            for arm in real_names:
                if not _beats(term, rows[arm][term], rows[junk][term]):
                    a2_failures.append(
                        f"{term}: real arm {arm} ({rows[arm][term]:.4f}) does not beat "
                        f"{junk} ({rows[junk][term]:.4f})"
                    )

    junk_sari = {j: rows[j]["sari_macro"] for j in JUNK_BASELINES}
    real_sari = {a: rows[a]["sari_macro"] for a in real_names}
    a3 = {
        "exempt_from_a2": True,
        "why": (
            "SARI's floor on this data is far above 0 (an additive blend containing it hands "
            "junk a nonzero floor), and Break's own published Copy row scores SARI 0.431 "
            "against its best model's 0.748. The ordering is recorded and the MARGIN is "
            "reported; it is not asserted"
        ),
        "junk_max": max(junk_sari.values()),
        "junk_max_system": max(junk_sari, key=junk_sari.get),
        "real_min": min(real_sari.values()) if real_sari else None,
        "real_min_arm": min(real_sari, key=real_sari.get) if real_sari else None,
        "margin": (min(real_sari.values()) - max(junk_sari.values())) if real_sari else None,
        "ordering_holds": (
            bool(min(real_sari.values()) > max(junk_sari.values())) if real_sari else None
        ),
    }

    a4_failures = [
        f"decomp_mean_macro: {junk} ({rows[junk]['decomp_mean_macro']:.4f}) ranks at or "
        f"above real arm {arm} ({rows[arm]['decomp_mean_macro']:.4f})"
        for junk in JUNK_BASELINES
        for arm in real_names
        if not rows[arm]["decomp_mean_macro"] > rows[junk]["decomp_mean_macro"]
    ]

    reversed_row = rows["J5_gold_reversed"]
    a5_failures = []
    if reversed_row["break_exact_match_rate"] != 0.0:
        a5_failures.append(
            f"break_exact_match_rate on the reversed gold is "
            f"{reversed_row['break_exact_match_rate']!r}, not 0.0"
        )
    for arm in real_names:
        if reversed_row["ordered_step_accuracy_macro"] >= rows[arm]["ordered_step_accuracy_macro"]:
            a5_failures.append(
                f"ordered_step_accuracy_macro: the reversed gold "
                f"({reversed_row['ordered_step_accuracy_macro']:.4f}) is not below real arm "
                f"{arm} ({rows[arm]['ordered_step_accuracy_macro']:.4f})"
            )

    return {
        "A1_gold_anchors": {"passed": not a1_failures, "failures": a1_failures},
        "A2_junk_is_beaten": {
            "passed": not a2_failures,
            "terms": list(A2_TERMS),
            "failures": a2_failures,
        },
        "A3_sari_exempt": a3,
        "A4_blend_gate": {
            "passed": not a4_failures,
            "metric": "decomp_mean_macro",
            "hard_gate": (
                "failing this DISQUALIFIES the blend; it is not a prompt to re-tune weights"
            ),
            "failures": a4_failures,
        },
        "A5_order_sensitivity": {
            "passed": not a5_failures,
            "ged_may_not_carry_it": (
                "GED is order-light and on a 2-step plan order-blind, so this assertion rests "
                "on break_exact_match and ordered_step_accuracy"
            ),
            "reversed_gold_ged_macro": reversed_row["ged_macro"],
            "failures": a5_failures,
        },
    }


def _git_commit() -> dict[str, Any]:
    """The SHA this table was printed at, and whether the tree was dirty (ADR 0027 point 2)."""
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=str(REPO_ROOT), capture_output=True, text=True, check=False
        ).stdout.strip()

    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def _print_table(rows: dict[str, dict[str, Any]], real_names: list[str]) -> None:
    header = ["system"] + [k for k in _SUITE_KEYS] + ["decomp_mean_macro", "n"]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for name in list(JUNK_SYSTEMS) + real_names:
        row = rows[name]
        cells = [f"{row[k]:.4f}" for k in _SUITE_KEYS]
        print(
            f"| {name} | " + " | ".join(cells) + f" | {row['decomp_mean_macro']:.4f} | "
            f"{row['n_items']} |"
        )
    print()
    print("Directions (⇓ = lower is better): " + ", ".join(
        f"{k} {'⇓' if _lower_is_better(k) else '⇑'}" for k in _SUITE_KEYS
    ))
    print("Junk systems: " + "; ".join(f"{k} = {v}" for k, v in JUNK_DESCRIPTIONS.items()))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--config",
        default="decomposition_junk_battery.json",
        help="Config (default: configs/decomposition_junk_battery.json)",
    )
    p.add_argument("--gold", type=Path, default=None, help="Override the gold JSONL from config.")
    p.add_argument(
        "--eval-set",
        choices=("pinned", "all-gold"),
        default="pinned",
        help="Which items to score. 'pinned' (default) restricts the gold to the ADR 0007 "
        "evaluation set, which is the set every committed arm was scored on. 'all-gold' "
        "scores every gold row and is NOT comparable to a committed arm — the same-set check "
        "below will refuse the arms.",
    )
    p.add_argument(
        "--limit", type=int, default=None, help="Score only the first N gold rows (tiny run)."
    )
    p.add_argument(
        "--no-real-arms",
        action="store_true",
        help="Skip the committed arms (they live under the gitignored runs/ and may be "
        "absent on another machine); A2/A4/A5 then have nothing to compare against and are "
        "reported as not evaluated.",
    )
    p.add_argument("--json", type=Path, default=None, help="Also write the table as JSON.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    paths_cfg = load_paths(require(cfg, "paths_config"))
    seed = int(require(cfg, "seed"))
    seeded = set_global_seed(seed)

    # The metric's own parameters come from the evaluator's config, not from this one: two
    # copies of a GED cap or a blend definition would drift, and then this battery would be
    # gating a metric nobody runs.
    eval_cfg = load_config(require(cfg, "evaluator_config"))
    ged_policy = EVAL._ged_policy(eval_cfg)
    decomp_policy = EVAL._decomp_mean_policy(eval_cfg)

    gold_path = (
        args.gold
        if args.gold is not None
        else resolve_path(
            require(paths_cfg, "datasets." + require(cfg, "gold_key")),
            Path(paths_cfg["data_root_resolved"]),
        )
    )
    if not gold_path.exists():
        raise SystemExit(f"gold file not found: {gold_path}")
    gold_by_question = EVAL._load_gold(
        gold_path, int(require(eval_cfg, "gold_validation.max_reported_mismatches"))
    )
    gold_rows_total = len(gold_by_question)
    pinned_missing_from_gold = 0
    if args.eval_set == "pinned":
        pinned = _pinned_eval_questions(paths_cfg, cfg)
        pinned_missing_from_gold = len(pinned - set(gold_by_question))
        gold_by_question = {k: v for k, v in gold_by_question.items() if k in pinned}
        if not gold_by_question:
            raise SystemExit(
                "no gold row is in the pinned evaluation set — the gold file and the pinned "
                "question files do not describe the same data."
            )
    if args.limit is not None:
        gold_by_question = dict(list(gold_by_question.items())[: args.limit])
    if not gold_by_question:
        raise SystemExit("no gold rows to score.")

    junk_steps = list(require(cfg, "fixed_junk_steps"))
    if len(junk_steps) < 3 or any(not isinstance(s, str) or not s.strip() for s in junk_steps):
        raise SystemExit(
            f"fixed_junk_steps must be at least 3 non-empty strings (J4 takes three), got "
            f"{junk_steps!r}"
        )

    rows: dict[str, dict[str, Any]] = {}
    for name, system in JUNK_SYSTEMS.items():
        rows[name] = _score_system(
            _junk_predictions(gold_by_question, system, junk_steps),
            gold_by_question,
            ged_policy,
            decomp_policy,
        )

    real_names: list[str] = []
    real_arm_paths: dict[str, str] = {}
    if not args.no_real_arms:
        for arm in require(cfg, "real_arms"):
            name = require(arm, "name")
            path = runs_path(paths_cfg, require(arm, "per_item"))
            if not path.exists():
                raise SystemExit(
                    f"real arm {name!r}: per-item file not found: {path}\n"
                    f"These live under the gitignored runs/ and are the committed arms of "
                    f"exp-011. Pass --no-real-arms to run the junk half only."
                )
            row, arm_items = _real_arm_row(path, decomp_policy)
            _assert_same_eval_set(set(gold_by_question), name, arm_items, path)
            rows[name] = row
            real_arm_paths[name] = str(path)
            real_names.append(name)

    verdicts = _verdicts(rows, real_names) if real_names else None

    git = _git_commit()
    print(f"# Decomposition junk battery — {now_iso()}")
    print(
        f"commit {git['commit'][:12]}{' (DIRTY TREE)' if git['dirty'] else ''}, seed {seed}, "
        f"gold {gold_path.name} ({gold_rows_total} rows), eval set '{args.eval_set}' -> "
        f"{len(gold_by_question)} items, {len(real_names)} real arm(s)"
    )
    if args.eval_set == "pinned":
        print(
            f"Every row below is scored on the SAME {len(gold_by_question)} items: the junk "
            f"baselines were scored on them here, and each real arm's per-item file was "
            f"checked to cover exactly them"
            + (
                f" ({pinned_missing_from_gold} pinned question(s) have no gold row and are "
                f"in neither)"
                if pinned_missing_from_gold
                else ""
            )
            + "."
        )
    else:
        print(
            "WARNING: --eval-set all-gold scores every gold row, which is NOT the set any "
            "committed arm was evaluated on."
        )
    print()
    _print_table(rows, real_names)
    print()
    if verdicts is None:
        print(
            "A2 / A4 / A5 NOT EVALUATED: no real arms were read (--no-real-arms), and every "
            "one of them is a statement about junk versus a real arm."
        )
    else:
        for name, verdict in verdicts.items():
            if "passed" not in verdict:
                print(
                    f"{name}: ordering_holds={verdict['ordering_holds']} "
                    f"(junk max {verdict['junk_max']:.4f} on {verdict['junk_max_system']}, "
                    f"real min {verdict['real_min']:.4f} on {verdict['real_min_arm']}, "
                    f"margin {verdict['margin']:+.4f}) — EXEMPT from A2 by design"
                )
                continue
            print(f"{name}: {'PASS' if verdict['passed'] else 'FAIL'}")
            for failure in verdict["failures"]:
                print(f"    {failure}")

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "script": Path(__file__).name,
                    "created_utc": now_iso(),
                    "git": git,
                    "seed": seed,
                    "seeded": seeded,
                    "gold_path": str(gold_path.resolve()),
                    "eval_set": args.eval_set,
                    "gold_rows_total": gold_rows_total,
                    "pinned_questions_without_a_gold_row": pinned_missing_from_gold,
                    "n_items": len(gold_by_question),
                    "fixed_junk_steps": junk_steps,
                    "junk_descriptions": JUNK_DESCRIPTIONS,
                    "ged_policy": ged_policy,
                    "decomp_mean_policy": decomp_policy,
                    "real_arm_paths": real_arm_paths,
                    "rows": rows,
                    "verdicts": verdicts,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\nWrote {args.json}")

    if verdicts is not None and not all(
        v.get("passed", True) for v in verdicts.values()
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
