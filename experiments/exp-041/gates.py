#!/usr/bin/env python3
"""exp-041 gates (CPU, stdlib only), thresholds from experiments/exp-041/run_config.json.

    python experiments/exp-041/gates.py train  --run-dir <train run dir> --seed 43 --commit <sha> --out <json>
    python experiments/exp-041/gates.py decode --run-dir <decomposer run dir> --kind canary|full \
        --role adapter|prompted --commit <sha> --out <json>

``train``: the training run used the pinned pool (sha256), all 2,000 rows with exp-026's hop mix,
0 overlap with the pinned 600, 4-bit, Mistral, 250 optimizer steps, the requested seed, the launch
commit on a clean tree, and the size ceiling asserted - i.e. exp-026's run with only the seed
changed. ``decode``: row counts (600 = 200/hop for a full run), the retrieval artifact's sha256,
the launch commit, and for adapter runs exp-026's protocol (--no-few-shot, adapter prompt parity 0
mismatches, quantization none); for full runs the 45 anchor prompts; then output health (empty
rows, rows at max_new_tokens). Writes the checks to --out either way and exits 3 if any fails.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

CONFIG = Path(__file__).resolve().parent / "run_config.json"


def train_checks(m: dict, gate: dict, *, seed: int, commit: str) -> dict:
    return {
        "training_source_sha256": m["training_source_sha256"] == gate["training_source_sha256"],
        "num_examples": m["dataset"]["num_examples"] == gate["num_examples"],
        "hop_counts": {str(k): v for k, v in m["dataset"]["hop_counts"].items()} == gate["hop_counts"],
        "eval_overlap": m["eval_overlap"]["overlap_count"] == gate["eval_overlap_count"],
        "quantization": m["quantization"] == gate["quantization"],
        "model_id": m["model_id"] == gate["model_id"],
        "global_step": m["train_result"]["global_step"] == gate["global_step"],
        "seed": int(m["seed"]) == int(seed),
        "ceiling_asserted": bool(m["base_model_size"]["ceiling_asserted"]) and bool(m["lora_model_size"]["ceiling_asserted"]),
        "commit": m["git"]["commit"] == commit,
        "clean_tree": m["git"]["dirty"] is False,
        "not_dry_run": m.get("dry_run") is False,
    }


def health(rows: list[dict], oh: dict, expected: int) -> dict:
    empty = sum(1 for r in rows if not str(r.get("decomposition") or "").strip())
    at_cap = sum(1 for r in rows if r.get("hit_max_new_tokens"))
    return {"rows": len(rows), "empty": empty, "at_max_new_tokens": at_cap,
            "max_empty": math.floor(float(oh["max_empty_fraction"]) * expected),
            "max_at_max_new_tokens": math.floor(float(oh["max_at_max_new_tokens_fraction"]) * expected)}


def decode_checks(dm: dict, dc: dict, rows: list[dict], gate: dict, *, kind: str, role: str,
                  commit: str, canary_rows: int) -> tuple[dict, dict]:
    expected = gate["rows"] if kind == "full" else canary_rows
    h = health(rows, gate["output_health"], expected)
    checks = {
        "rows": len(rows) == expected and dm["rows_loaded_total"] == expected,
        "retrieval_sha256": dc["retrieval"]["input_sha256"] == gate["retrieval_input_sha256"],
        "commit": dc["git"]["commit"] == commit,
        "ceiling_asserted": bool(dm["model_size"]["ceiling_asserted"]),
        "empty_within_threshold": h["empty"] <= h["max_empty"],
        "not_all_empty": h["empty"] < h["rows"] if h["rows"] else False,
        "at_max_new_tokens_within_threshold": h["at_max_new_tokens"] <= h["max_at_max_new_tokens"],
    }
    if kind == "full":
        checks["rows_per_hop"] = {str(k): v for k, v in dm["rows_loaded_per_hop"].items()} == gate["rows_per_hop"]
        checks["anchors"] = dm["anchor_prompts_logged"] == gate["anchor_prompts_expected"]
    if role == "adapter":
        checks["no_few_shot"] = dm["no_few_shot"] is True
        checks["quantization_none"] = dc["quantization"] == "none"
        par = dm["adapter_prompt_parity"]
        checks["adapter_prompt_parity"] = bool(par["checked"]) and len(par["mismatches"]) == gate["adapter_prompt_parity_mismatches"]
    else:
        checks["few_shot"] = dm["no_few_shot"] is False
    return checks, h


def main(argv: list[str] | None = None) -> int:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("train", "decode"))
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--commit", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--kind", choices=("canary", "full"))
    ap.add_argument("--role", choices=("adapter", "prompted"))
    a = ap.parse_args(argv)
    load = lambda name: json.loads((a.run_dir / name).read_text(encoding="utf-8"))  # noqa: E731
    if a.what == "train":
        checks, extra = train_checks(load("metrics.json"), cfg["train"]["gate"], seed=a.seed, commit=a.commit), {}
    else:
        checks, extra = decode_checks(load("metrics.json"), load("config.json"), load("results.json"),
                                      cfg["eval_gate"], kind=a.kind, role=a.role, commit=a.commit,
                                      canary_rows=int(cfg["canary"]["rows"]))
    rep = {"what": a.what, "run_dir": str(a.run_dir), "checks": checks, "health": extra, "ok": all(checks.values())}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    print(f"[exp-041] {a.what} gate {'OK' if rep['ok'] else 'FAILED'}: {json.dumps(checks)} {extra}")
    return 0 if rep["ok"] else 3


if __name__ == "__main__":
    sys.exit(main())
