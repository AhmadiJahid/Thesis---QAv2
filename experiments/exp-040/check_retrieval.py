#!/usr/bin/env python3
"""exp-040: check the rebuilt all-modes retrieval artifact and pin its sha256 (CPU, stdlib only).

Run by ``experiments/exp-040/run.sh retrieval`` right after the build, and again (``--verify``)
before every decode that reads the artifact:

    python experiments/exp-040/check_retrieval.py \
        --artifact runs/exp-040/retrieval/pinned600_allmodes_rerank_top5.jsonl \
        --exp014 runs/exp-014/retrieval/s2_pinned600_imbalanced_rerank_top5.jsonl \
        --pin runs/exp-040/retrieval/PINNED_SHA256 --out runs/exp-040/retrieval/check.json

Checks (any failure exits 2 and writes nothing to the pin file):

- the exp-014 artifact is the one exp-020 read (sha256 from experiments/exp-040/config.json);
- 600 rows, 600 distinct query ids, the same id set as the exp-014 artifact (which the
  decomposer runner itself asserts is the ADR 0007 pinned 600), 200/200/200 by hop prefix;
- every row has raw_top_k, typed_top_k and uniform_top_k with >= k candidates, each carrying a
  pool_id, a pool_question and a few-shot decomposition;
- the pin: the first successful check writes the artifact's sha256 to the pin file; every
  later check refuses a file whose sha256 differs (a rebuilt file is a different input).

Reported (never gating): typed exemplar identity vs exp-014 - per query, whether the ordered
top-k pool_id list equals exp-014's - which decides the typed comparator
(config.json typed_comparator_rule), and the top-k overlap between modes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "config.json"
MODES = ("raw", "typed", "uniform")
HOPS = ("2", "3", "4")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def hop_of(query_id: str) -> str:
    return str(query_id).split("hop__", 1)[0]


def ordered_ids(row: dict, mode: str, k: int) -> list[str]:
    return [str(c.get("pool_id")) for c in (row.get(f"{mode}_top_k") or [])[:k]]


def check(artifact: Path, exp014: Path, *, k: int, exp014_sha256: str, expected_rows: int,
          expected_per_hop: int) -> dict:
    errors: list[str] = []
    got014 = sha256(exp014)
    if got014 != exp014_sha256:
        errors.append(f"exp-014 artifact sha256 {got014} != {exp014_sha256} (not exp-020's input)")
    rows, base = load_rows(artifact), load_rows(exp014)
    by_id = {str(r.get("query_id")): r for r in rows}
    base_by_id = {str(r.get("query_id")): r for r in base}
    if len(rows) != expected_rows or len(by_id) != expected_rows:
        errors.append(f"{len(rows)} rows / {len(by_id)} distinct ids, expected {expected_rows}")
    if set(by_id) != set(base_by_id):
        errors.append(f"id set differs from the exp-014 artifact "
                      f"({len(set(by_id) - set(base_by_id))} extra, {len(set(base_by_id) - set(by_id))} missing)")
    per_hop = {h: sum(1 for q in by_id if hop_of(q) == h) for h in HOPS}
    if any(n != expected_per_hop for n in per_hop.values()):
        errors.append(f"per-hop counts {per_hop}, expected {expected_per_hop} each")
    short: dict[str, int] = {m: 0 for m in MODES}
    malformed = 0
    for r in rows:
        for m in MODES:
            cands = r.get(f"{m}_top_k")
            if not isinstance(cands, list) or len(cands) < k:
                short[m] += 1
                continue
            for c in cands[:k]:
                if not (isinstance(c, dict) and c.get("pool_id") is not None and c.get("pool_question")
                        and c.get("pool_few_shot_decomposition_musique")):
                    malformed += 1
    if any(short.values()):
        errors.append(f"rows with fewer than k={k} candidates per mode: {short}")
    if malformed:
        errors.append(f"{malformed} top-{k} candidates lack pool_id / pool_question / decomposition")

    shared = sorted(set(by_id) & set(base_by_id))
    same = [q for q in shared if ordered_ids(by_id[q], "typed", k) == ordered_ids(base_by_id[q], "typed", k)]
    same_set = [q for q in shared
                if set(ordered_ids(by_id[q], "typed", k)) == set(ordered_ids(base_by_id[q], "typed", k))]
    overlap = {}
    for a, b in (("typed", "uniform"), ("typed", "raw"), ("uniform", "raw")):
        vals = [len(set(ordered_ids(r, a, k)) & set(ordered_ids(r, b, k))) for r in rows]
        overlap[f"{a}_vs_{b}"] = {
            "mean_shared_exemplars_of_k": (sum(vals) / len(vals)) if vals else None,
            "rows_identical_set": sum(1 for v in vals if v == k),
            "rows_disjoint": sum(1 for v in vals if v == 0),
        }
    return {
        "artifact": str(artifact), "artifact_sha256": sha256(artifact),
        "exp014_artifact": str(exp014), "exp014_sha256": got014,
        "k": k, "rows": len(rows), "distinct_ids": len(by_id), "per_hop": per_hop,
        "typed_identity": {
            "compared": len(shared),
            "identical": len(same),
            "identical_as_set_any_order": len(same_set),
            "rule": "ordered top-k pool_id list equal to exp-014's for the same query id",
            "differing_ids_first_20": [q for q in shared if q not in set(same)][:20],
        },
        "mode_overlap_top_k": overlap,
        "errors": errors,
        "ok": not errors,
    }


def apply_pin(pin: Path, digest: str) -> str:
    """Write the pin on first use; refuse a different digest afterwards. Returns the action."""
    if pin.exists():
        want = pin.read_text(encoding="utf-8").strip()
        if want != digest:
            raise SystemExit(f"refusing: artifact sha256 {digest} != pinned {want} ({pin}). "
                             "The artifact changed after the first build; a rebuilt file is a different input.")
        return "verified"
    pin.parent.mkdir(parents=True, exist_ok=True)
    pin.write_text(digest + "\n", encoding="utf-8")
    return "pinned"


def main(argv: list[str] | None = None) -> int:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--exp014", type=Path, required=True)
    ap.add_argument("--pin", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--verify", action="store_true", help="only check sha256 against the existing pin")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args(argv)

    if args.verify:
        if not args.pin.exists():
            raise SystemExit(f"refusing: no pin at {args.pin}; run the retrieval step first")
        print(f"[exp-040] retrieval artifact {apply_pin(args.pin, sha256(args.artifact))}")
        return 0
    rep = check(args.artifact, args.exp014, k=args.k,
                exp014_sha256=cfg["inputs"]["exp014_typed_artifact"]["sha256"],
                expected_rows=600, expected_per_hop=200)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    if not rep["ok"]:
        print("[exp-040] retrieval check FAILED:\n  " + "\n  ".join(rep["errors"]), file=sys.stderr)
        return 2
    rep["pin"] = apply_pin(args.pin, rep["artifact_sha256"])
    if args.out:
        args.out.write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    ti = rep["typed_identity"]
    print(f"[exp-040] retrieval OK, sha256 {rep['artifact_sha256']} ({rep['pin']}); typed exemplars "
          f"identical to exp-014 for {ti['identical']}/{ti['compared']} queries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
