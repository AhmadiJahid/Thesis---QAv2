#!/usr/bin/env python3
"""exp-040: output-health gate for one finished decomposer run directory (CPU, stdlib only).

    python experiments/exp-040/check_health.py --run-dir <decomposer run dir> --kind canary|full \
        --out <health json>

Reads ``results.json`` (one row per query) from the run directory and applies the thresholds
pre-declared in experiments/exp-040/config.json ``output_health_gate``: the expected row count,
at most ``max_empty_fraction`` of rows with an empty ``decomposition``, not all rows empty,
and at most ``max_rows_at_max_new_tokens_fraction`` of rows that hit ``max_new_tokens``. Writes
the counts and the verdict to ``--out`` either way; exits 3 on a failed gate, so run.sh stops
the arm and the notebook stops before the next GPU step.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

CONFIG = Path(__file__).resolve().parent / "config.json"


def health(rows: list[dict], gate: dict, kind: str) -> dict:
    n = len(rows)
    expected = int(gate["rows_expected"][kind])
    empty = sum(1 for r in rows if not str(r.get("decomposition") or "").strip())
    at_cap = sum(1 for r in rows if r.get("hit_max_new_tokens"))
    max_empty = math.floor(float(gate["max_empty_fraction"]) * expected)
    max_cap = math.floor(float(gate["max_rows_at_max_new_tokens_fraction"]) * expected)
    checks = {
        "rows": n == expected,
        "empty_within_threshold": empty <= max_empty,
        "not_all_empty": empty < n if n else False,
        "at_max_new_tokens_within_threshold": at_cap <= max_cap,
    }
    return {"kind": kind, "rows": n, "rows_expected": expected, "empty": empty,
            "max_empty": max_empty, "at_max_new_tokens": at_cap, "max_at_max_new_tokens": max_cap,
            "checks": checks, "ok": all(checks.values())}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--kind", choices=("canary", "full"), required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    gate = json.loads(CONFIG.read_text(encoding="utf-8"))["output_health_gate"]
    rows = json.loads((args.run_dir / "results.json").read_text(encoding="utf-8"))
    rep = {"run_dir": str(args.run_dir), **health(rows, gate, args.kind)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    print(f"[exp-040] output health ({args.kind}): {json.dumps(rep['checks'])} "
          f"rows={rep['rows']} empty={rep['empty']} at_cap={rep['at_max_new_tokens']}")
    return 0 if rep["ok"] else 3


if __name__ == "__main__":
    sys.exit(main())
