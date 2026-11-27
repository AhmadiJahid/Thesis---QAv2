#!/usr/bin/env python3
"""Rebuild the data folder (and selected stored runs) from the thesis backup archive.

For Colab (or any fresh box). The server backup ``thesis-backup.tar.gz`` holds the v1 repo's
data (``Thesis---QA/...``) and the v2 repo's ``runs/`` (``Thesis---QAv2/runs/...``). This
script maps them onto the layout ``docs/compute.md`` "Dataset location" describes, which is what
``configs/paths.json`` (``data_root: ../thesis-qav2-data``) resolves:

    <parent>/Thesis---QAv2          the clone (this repo)
    <parent>/thesis-qav2-data/
        musique/        <- Thesis---QA/MusiQue/Data/*   (minus emb_cache)
        cache/emb_cache <- Thesis---QA/MusiQue/Data/emb_cache
        metaqa/         <- Thesis---QA/Data/*           (part 'metaqa' only)

``reference/musique_official`` and ``metaqa/pool`` are NOT in the archive; nothing here invents
them (no current decomposer/evaluator stage reads them). ``--runs`` restores chosen paths of
``Thesis---QAv2/runs/`` into this repo's ``runs/`` (gitignored), e.g. the retrieval artifact and
the stored baseline evaluations an experiment compares against.

One streaming pass over the archive; only matching regular files (and directories) are
extracted, by reading each member's bytes and writing them to a destination this script
computes - never ``tar.extract``. Every destination is checked before it is written
(:func:`_safe_dest`): a member name that is absolute, drive-anchored or contains a ``..``
part is refused, the resolved destination must lie inside its root, and no existing component
of the path below the root may be a symlink. Links in the archive are skipped. Existing files
are left alone unless ``--overwrite``. Data never enters git: both targets are outside the tree or under
the ignored ``runs/``.

Usage::

    python scripts/colab_setup_data.py --archive /content/thesis-backup.tar.gz \\
        --parts musique --runs exp-014/retrieval/s2_pinned600_imbalanced_rerank_top5.jsonl \\
        exp-032/exp-020/eval exp-032/exp-027/eval
    python scripts/colab_setup_data.py --archive ... --list-parts
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath, PureWindowsPath

REPO_ROOT = Path(__file__).resolve().parents[1]

#: archive prefix -> destination relative to data_root (None = handled by --runs)
PARTS: dict[str, list[tuple[str, str]]] = {
    "musique": [("Thesis---QA/MusiQue/Data/", "musique/")],
    "emb_cache": [("Thesis---QA/MusiQue/Data/emb_cache/", "cache/emb_cache/")],
    "metaqa": [("Thesis---QA/Data/", "metaqa/")],
}
RUNS_PREFIX = "Thesis---QAv2/runs/"

#: Files an exp-037/exp-036-style decomposer + evaluator run reads under data_root; checked
#: after extraction so a missing one fails here, not 20 minutes into a GPU session.
REQUIRED_FOR_MUSIQUE = [
    "musique/dev_data/musique_ans_v1.0_dev_sample_2_hop_200.jsonl",
    "musique/dev_data/musique_ans_v1.0_dev_sample_3_hop_200.jsonl",
    "musique/dev_data/musique_ans_v1.0_dev_sample_4_hop_200.jsonl",
    "musique/dev_data/musique_ans_v1.0_dev_clean.jsonl",
]


class UnsafeMember(ValueError):
    """An archive member whose name or destination would escape the target root."""


def _check_member_name(name: str) -> None:
    pure = PurePosixPath(name)
    if (
        name.startswith(("/", "\\"))
        or pure.is_absolute()
        or PureWindowsPath(name).drive
        or PureWindowsPath(name).is_absolute()
        or any(part == ".." for part in pure.parts)
        or any(part == ".." for part in PureWindowsPath(name).parts)
    ):
        raise UnsafeMember(f"refusing archive member {name!r}: absolute or '..' path")


def _safe_dest(root: Path, rel: str) -> Path:
    """``root / rel``, refused unless it stays inside ``root`` with no symlink on the way."""
    _check_member_name(rel)
    root_resolved = root.resolve()
    dest = root / PurePosixPath(rel)
    # No existing component below the root may be a symlink (it could point anywhere).
    cur = root
    for part in PurePosixPath(rel).parts:
        cur = cur / part
        if cur.is_symlink():
            raise UnsafeMember(f"refusing {rel!r}: {cur} is a symlink")
    if not dest.resolve().is_relative_to(root_resolved):
        raise UnsafeMember(f"refusing {rel!r}: resolves outside {root_resolved}")
    return dest


def _plan(member: str, parts: list[str], runs: list[str]) -> str | None:
    """Destination for one archive member: 'data:<rel>', 'runs:<rel>' or None (skip)."""
    for part in parts:
        for prefix, dest in PARTS[part]:
            if member.startswith(prefix) and member != prefix.rstrip("/"):
                rel = member[len(prefix):]
                # musique/ must not swallow the emb_cache subtree (it has its own part).
                if part == "musique" and rel.startswith("emb_cache"):
                    return None
                return "data:" + dest + rel
    if member.startswith(RUNS_PREFIX):
        rel = member[len(RUNS_PREFIX):]
        for want in runs:
            want = want.strip("/")
            if rel == want or rel.startswith(want + "/"):
                return "runs:" + rel
    return None


def _extract(args, counts: dict, seen_runs: set) -> None:
    with tarfile.open(args.archive, "r:gz") as tar:
        for m in tar:
            if not (m.isfile() or m.isdir()):
                continue  # links (the archive has hardlinks under runs/pool_sweep) are not needed
            _check_member_name(m.name)  # every member, wanted or not: a hostile archive is refused
            target = _plan(m.name, args.parts, args.runs)
            if target is None:
                continue
            kind, rel = target.split(":", 1)
            root = args.data_root if kind == "data" else args.runs_root
            root.mkdir(parents=True, exist_ok=True)
            dest = _safe_dest(root, rel)
            if m.isdir():
                dest.mkdir(parents=True, exist_ok=True)
                continue
            if dest.exists() and not args.overwrite:
                counts["skipped_existing"] += 1
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            src = tar.extractfile(m)
            assert src is not None
            with open(dest, "wb") as out:
                while chunk := src.read(1 << 22):
                    out.write(chunk)
            counts[kind] += 1
            if kind == "runs":
                seen_runs.update(w for w in args.runs if rel.startswith(w.strip("/")))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--archive", required=True, type=Path)
    p.add_argument("--data-root", type=Path, default=REPO_ROOT.parent / "thesis-qav2-data")
    p.add_argument("--runs-root", type=Path, default=REPO_ROOT / "runs")
    p.add_argument("--parts", nargs="*", default=["musique"], choices=sorted(PARTS))
    p.add_argument("--runs", nargs="*", default=[], help="paths under Thesis---QAv2/runs/ to restore")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--sha256", default=None, help="expected archive sha256 (checked first if given)")
    p.add_argument("--list-parts", action="store_true")
    args = p.parse_args()

    if args.list_parts:
        print(json.dumps(PARTS, indent=2))
        return 0
    if not args.archive.exists():
        raise SystemExit(f"archive not found: {args.archive}")
    if args.sha256:
        h = hashlib.sha256()
        with open(args.archive, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        if h.hexdigest() != args.sha256.lower():
            raise SystemExit(f"archive sha256 {h.hexdigest()} != expected {args.sha256}")
        print(f"[setup] archive sha256 OK")

    started = time.time()
    counts = {"data": 0, "runs": 0, "skipped_existing": 0}
    seen_runs: set[str] = set()
    try:
        _extract(args, counts, seen_runs)
    except UnsafeMember as exc:
        raise SystemExit(f"[setup] {exc}") from exc

    missing_runs = [w for w in args.runs if w not in seen_runs and not (args.runs_root / w).exists()]
    missing_data = []
    if "musique" in args.parts:
        missing_data = [r for r in REQUIRED_FOR_MUSIQUE if not (args.data_root / r).exists()]
    print(f"[setup] extracted {counts} in {time.time() - started:.0f}s; "
          f"data_root={args.data_root.resolve()} runs_root={args.runs_root.resolve()}")
    if missing_runs or missing_data:
        raise SystemExit(f"[setup] MISSING runs={missing_runs} data={missing_data}")
    print("[setup] all requested paths present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
