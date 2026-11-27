#!/usr/bin/env python3
"""Draw the pinned MetaQA evaluation set (exp-039, ADR 0045) from the vanilla test split.

Reads ``{hop}-hop/vanilla/qa_test.txt`` (official MetaQA format: ``question<TAB>a1|a2``,
topic entity in square brackets), verifies every source file against the sha256 and
line count pinned in the config, drops question texts that occur more than once within a
hop (the scorer keys gold by question text, so a repeated text with a different answer
set cannot be scored), and draws ``n_per_hop`` lines per hop with one seeded RNG.

Writes, under the data root (never git):
  - ``refined_{hop}hop.txt`` / ``answers_{hop}hop.txt`` in the legacy line format the
    decomposer and the compile-execute scorer already read (brackets removed),
  - ``topics_{hop}hop.txt``: the bracketed topic entity, line-aligned.
Writes to ``--manifest`` (committed): source sha256s, the drawn 0-based line indices and
a sha256 per question text, so the set is identified without committing its text.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from run_artifacts import now_iso  # noqa: E402
from run_config import load_config, load_paths, require, resolve_path  # noqa: E402
from seeding import new_rng, set_global_seed  # noqa: E402

BRACKET_RX = re.compile(r"\[([^\]]+)\]")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def parse_line(line: str) -> tuple[str, str, str]:
    """-> (question without brackets, topic entity, answer field)."""
    q, ans = line.rstrip("\r\n").split("\t")
    m = BRACKET_RX.search(q)
    if m is None:
        raise ValueError(f"no bracketed topic entity: {q!r}")
    return BRACKET_RX.sub(lambda mm: mm.group(1), q), m.group(1), ans


def draw(lines: list[str], n: int, rng) -> tuple[list[int], int]:
    parsed = [parse_line(ln) for ln in lines]
    counts = Counter(p[0] for p in parsed)
    eligible = [i for i, p in enumerate(parsed) if counts[p[0]] == 1]
    picked = sorted(rng.sample(eligible, n))
    return picked, len(lines) - len(eligible)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="metaqa_eval_set.json")
    ap.add_argument("--manifest", type=Path, required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths_cfg = load_paths(require(cfg, "paths_config"))
    data_root = Path(paths_cfg["data_root_resolved"])
    seed = int(require(cfg, "seed"))
    set_global_seed(seed)
    rng = new_rng(seed)
    n = int(require(cfg, "n_per_hop"))
    src_dir = resolve_path(require(paths_cfg, "datasets." + require(cfg, "source_dir_key")), data_root)
    out_dir = resolve_path(require(paths_cfg, "datasets." + require(cfg, "output_dir_key")), data_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"created_utc": now_iso(), "seed": seed, "n_per_hop": n,
                "source": require(cfg, "source"), "hops": {}}
    for hop in require(cfg, "hops"):
        pin = require(cfg, f"source_files.{hop}")
        path = src_dir / f"{hop}-hop" / "vanilla" / "qa_test.txt"
        raw = path.read_bytes()
        digest = sha256_bytes(raw)
        lines = [ln for ln in raw.decode("utf-8").splitlines() if ln.strip()]
        if digest != pin["sha256"] or len(lines) != int(pin["lines"]):
            raise SystemExit(f"{path}: sha256 {digest} / {len(lines)} lines does not match the pinned "
                             f"{pin['sha256']} / {pin['lines']}")
        picked, n_dup = draw(lines, n, rng)
        rows = [parse_line(lines[i]) for i in picked]
        (out_dir / f"refined_{hop}hop.txt").write_text("".join(q + "\n" for q, _, _ in rows), encoding="utf-8")
        (out_dir / f"answers_{hop}hop.txt").write_text("".join(a + "\n" for _, _, a in rows), encoding="utf-8")
        (out_dir / f"topics_{hop}hop.txt").write_text("".join(t + "\n" for _, t, _ in rows), encoding="utf-8")
        manifest["hops"][str(hop)] = {
            "source_file": f"{hop}-hop/vanilla/qa_test.txt",
            "source_sha256": digest,
            "source_lines": len(lines),
            "excluded_repeated_question_text_lines": n_dup,
            "line_indices_0based": picked,
            "question_sha256": [sha256_bytes(q.encode("utf-8")) for q, _, _ in rows],
            "overlap_with_legacy_first_100_lines": sum(1 for i in picked if i < 100),
        }
        print(f"hop {hop}: drew {len(picked)} of {len(lines)} (excluded {n_dup} repeated-text lines), "
              f"overlap with legacy first-100: {manifest['hops'][str(hop)]['overlap_with_legacy_first_100_lines']}")
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"manifest -> {args.manifest}; files -> {out_dir}")


if __name__ == "__main__":
    main()
