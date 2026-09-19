#!/usr/bin/env python3
"""Draw — once — the fixed anchor set of MuSiQue items that is inspected by hand every run.

The supervisor's instruction (2026-08-27, recorded as item A4 of
``docs/plan/2026-09-14-post-supervisor-meeting-phase.md``) is 15 items per hop count, 45 in
total, drawn at random once and then **fixed over all the experiments**. Per ADR 0027 the 45
ids are not a pasted constant: they are the output of this script, re-derivable from the
committed config and a fixed seed, and ``--verify`` is how a reader checks that the committed
list is still the list this code produces.

The draw source is the ADR 0007 pinned 600 (200 per hop), i.e. exactly the three files
``datasets.musique_eval_questions_template`` names. That choice is what makes the inspection
free: every experiment in this phase is evaluated on the pinned 600, so an anchor item is
always scored as part of the normal run and the hand inspection costs no extra inference.
``scripts/render_anchor_set.py`` is the reader for it — it post-filters a finished run's
``eval_per_item.json`` down to these 45 ids.

Usage::

    # verify the committed list still re-derives (the default, and what CI/tests run)
    python scripts/build_anchor_set.py --verify

    # the one-time draw (refuses to clobber an existing list without --force)
    python scripts/build_anchor_set.py --write

    # tiny smoke path against the synthetic fixtures
    QAV2_PATHS_CONFIG=configs/smoke_paths.json python scripts/build_anchor_set.py \\
        --write --n-per-hop 3 --out runs/smoke/work/anchor/anchor_ids.json

Exit codes: 0 when the committed list matches the re-derivation, 1 when it does not (or the
source files are missing, so the check could not be made).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from run_artifacts import now_iso  # noqa: E402
from run_config import (  # noqa: E402
    load_config,
    load_paths,
    require,
    resolve_path,
)
from seeding import new_rng, set_global_seed  # noqa: E402

#: Shape of the committed id list. Bumped if the file's fields change, so a consumer that
#: reads it can say "this is not the file I was written against" instead of KeyError-ing.
ANCHOR_SET_SCHEMA = "musique_anchor_set/1"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", default="anchor_set_draw.json")
    p.add_argument("--out", type=Path, default=None, help="Override the committed id-list path.")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--n-per-hop", type=int, default=None)
    p.add_argument("--hops", type=int, nargs="+", default=None)
    p.add_argument(
        "--verify",
        action="store_true",
        help="Re-derive the draw and exit non-zero if it differs from the id list on disk. "
        "This is the default action when --write is not given.",
    )
    p.add_argument("--write", action="store_true", help="Write the id list.")
    p.add_argument(
        "--force",
        action="store_true",
        help="Permit --write to overwrite an existing id list. The set is frozen by the "
        "supervisor's instruction, so re-drawing it is a deliberate act.",
    )
    return p.parse_args()


def load_hop_ids(path: Path, id_field: str) -> list[str]:
    """Every id in one pinned evaluation file, in file order.

    A duplicate id is fatal: the draw population would be weighted, and ADR 0007 records
    the pinned files as holding 600 *distinct* ids.
    """
    ids: list[str] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if id_field not in obj:
                raise SystemExit(f"{path}:{lineno}: row has no {id_field!r} field")
            qid = str(obj[id_field]).strip()
            if not qid:
                raise SystemExit(f"{path}:{lineno}: empty {id_field!r}")
            if qid in seen:
                raise SystemExit(f"{path}:{lineno}: duplicate id {qid!r} in the draw population")
            seen.add(qid)
            ids.append(qid)
    if not ids:
        raise SystemExit(f"{path}: no ids read; this cannot be a draw population")
    return ids


def ids_sha256(ids: list[str]) -> str:
    """Fingerprint of a draw population, order-independent (the ids are sorted first)."""
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    """Fingerprint of a source file's bytes, read in chunks.

    Recorded alongside ``ids_sha256`` because they answer different questions: the id
    fingerprint says "the draw population is unchanged", the file fingerprint says "the
    file the draw was made from is byte-for-byte the file on disk today". A source file
    whose question text was edited but whose ids were not would pass the first and fail
    the second, and the hand inspection reads that text.
    """
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_anchor_ids(
    population_by_hop: dict[int, list[str]], hops: list[int], n_per_hop: int, seed: int
) -> dict[str, list[str]]:
    """The draw: ``n_per_hop`` ids per hop, one RNG, hops visited in the given order.

    Two deliberate details, both so the result depends on nothing but (population, hops,
    n_per_hop, seed):

    - the population is **sorted** before sampling, so a re-ordered source file cannot move
      the draw;
    - the drawn ids are sorted in the returned list. Sorting after the draw loses nothing —
      the sample is a set — and it makes the committed file diff-readable.

    One RNG across the hops, in the order given, follows the existing convention in
    ``scripts/run_qwen_smoke.py::_sample_per_hop``. It means the hop list is part of the
    draw: changing it changes the ids, which is exactly why the hop list is committed.
    """
    rng = new_rng(seed)
    drawn: dict[str, list[str]] = {}
    for hop in hops:
        population = sorted(population_by_hop[hop])
        if len(population) < n_per_hop:
            raise SystemExit(
                f"hop {hop}: population has {len(population)} ids, cannot draw {n_per_hop}"
            )
        drawn[str(hop)] = sorted(rng.sample(population, n_per_hop))
    return drawn


def _population(paths_cfg: dict[str, Any], cfg: dict[str, Any], hops: list[int]) -> tuple[
    dict[int, list[str]], dict[str, dict[str, Any]]
]:
    """Read the pinned per-hop files: ({hop: ids}, per-hop provenance)."""
    data_root = Path(paths_cfg["data_root_resolved"])
    template = require(paths_cfg, "datasets." + require(cfg, "questions_template_key"))
    id_field = require(cfg, "questions_jsonl.id_field")
    population: dict[int, list[str]] = {}
    provenance: dict[str, dict[str, Any]] = {}
    for hop in hops:
        rel = template.format(hop=hop)
        path = resolve_path(rel, data_root)
        if not path.exists():
            raise SystemExit(
                f"pinned evaluation file for hop {hop} not found: {path}\n"
                "The anchor set is drawn from the ADR 0007 pinned 600; without those files "
                "the draw cannot be made or checked. Set data_root in configs/paths.json."
            )
        ids = load_hop_ids(path, id_field)
        population[hop] = ids
        provenance[str(hop)] = {
            "file": rel,
            "id_count": len(ids),
            "ids_sha256": ids_sha256(ids),
            "file_sha256": file_sha256(path),
        }
    return population, provenance


def build_payload(
    drawn: dict[str, list[str]],
    provenance: dict[str, dict[str, Any]],
    *,
    cfg: dict[str, Any],
    hops: list[int],
    n_per_hop: int,
    seed: int,
) -> dict[str, Any]:
    """The committed artifact: ids and provenance only — never question text (CLAUDE.md)."""
    counts = {hop: len(ids) for hop, ids in drawn.items()}
    total = sum(counts.values())
    return {
        # The note is a template, not prose with numbers typed into it: a config that said
        # "seed 42" while the draw ran at 43 would be a record that lies quietly.
        "_note": require(cfg, "ids_file_note").format(
            seed=seed, total=total, n_per_hop=n_per_hop
        ),
        "schema": ANCHOR_SET_SCHEMA,
        "generated_by": f"scripts/{Path(__file__).name}",
        "generated_utc": now_iso(),
        # Machine context for the draw. Deliberately OUTSIDE _VERIFIED_FIELDS: random.sample
        # is stable across the CPython versions in use, so a different interpreter is
        # information for a reader, not grounds for --verify to fail. If a draw ever does
        # differ across versions, this is the field that says which one produced the list.
        "python_version": platform.python_version(),
        "seed": seed,
        "hops": hops,
        "n_per_hop": n_per_hop,
        "source": {
            "description": "the ADR 0007 pinned MuSiQue evaluation set (600 ids, 200 per hop)",
            "questions_template_key": require(cfg, "questions_template_key"),
            "id_field": require(cfg, "questions_jsonl.id_field"),
            "per_hop": provenance,
        },
        "counts": {**counts, "total": total},
        "ids_by_hop": drawn,
    }


#: Fields of the committed file that must re-derive identically. ``generated_utc``,
#: ``python_version`` and the note are excluded: a timestamp is not part of the draw, the
#: interpreter that ran it is context rather than a parameter, and prose is not a number.
_VERIFIED_FIELDS = ("schema", "seed", "hops", "n_per_hop", "source", "counts", "ids_by_hop")


def _sample(ids: list[str], limit: int = 5) -> str:
    if not ids:
        return "-"
    shown = ", ".join(ids[:limit])
    return shown + (f" ... (+{len(ids) - limit} more)" if len(ids) > limit else "")


def _ids_diff(expected: dict[str, list[str]], found: Any) -> list[str]:
    """Which ids moved, per hop — rather than two full 15-id lists nobody can read."""
    if not isinstance(found, dict):
        return ["ids_by_hop: the committed file has no id object"]
    problems: list[str] = []
    for hop in sorted(set(expected) | set(found)):
        want = list(expected.get(hop, []))
        got = found.get(hop, [])
        got = list(got) if isinstance(got, list) else []
        if want == got:
            continue
        only_committed = sorted(set(got) - set(want))
        only_derived = sorted(set(want) - set(got))
        if only_committed or only_derived:
            problems.append(
                f"ids_by_hop[{hop}]: {len(only_committed)} id(s) only in the committed file "
                f"({_sample(only_committed)}), {len(only_derived)} only in the re-derivation "
                f"({_sample(only_derived)})"
            )
        else:
            problems.append(f"ids_by_hop[{hop}]: the same ids in a different order")
    return problems


def _diff(expected: dict[str, Any], found: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    for field in _VERIFIED_FIELDS:
        if field not in found:
            problems.append(f"{field}: missing from the committed file")
        elif found[field] != expected[field]:
            if field == "ids_by_hop":
                problems.extend(_ids_diff(expected[field], found[field]))
            else:
                problems.append(
                    f"{field}: committed {json.dumps(found[field], sort_keys=True)} != "
                    f"re-derived {json.dumps(expected[field], sort_keys=True)}"
                )
    return problems


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    paths_cfg = load_paths(require(cfg, "paths_config"))

    seed = args.seed if args.seed is not None else int(require(cfg, "seed"))
    seeded = set_global_seed(seed)
    n_per_hop = args.n_per_hop if args.n_per_hop is not None else int(require(cfg, "n_per_hop"))
    hops = args.hops or [int(h) for h in require(cfg, "hops")]
    out_path = args.out or resolve_path(require(cfg, "ids_file"), REPO_ROOT)

    population, provenance = _population(paths_cfg, cfg, hops)
    drawn = select_anchor_ids(population, hops, n_per_hop, seed)
    payload = build_payload(
        drawn, provenance, cfg=cfg, hops=hops, n_per_hop=n_per_hop, seed=seed
    )

    print(
        f"[anchor] seed={seed} seeded={json.dumps(seeded)} hops={hops} "
        f"n_per_hop={n_per_hop} -> {payload['counts']}"
    )
    for hop in hops:
        pop = provenance[str(hop)]
        print(f"[anchor] hop {hop}: drew {n_per_hop} of {pop['id_count']} from {pop['file']}")

    if args.write:
        if out_path.exists() and not args.force:
            raise SystemExit(
                f"{out_path} already exists. The anchor set is frozen over all experiments "
                "(supervisor, 2026-08-27), so re-drawing it is deliberate: pass --force if "
                "that is what you mean, or --verify to check the committed list instead."
            )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"[anchor] wrote {out_path}")
        return

    # Default action: verify. Stated as an explicit branch so that running the script with
    # no flags can never overwrite the frozen list.
    if not out_path.exists():
        raise SystemExit(
            f"no id list at {out_path} to verify. Draw it once with --write."
        )
    found = json.loads(out_path.read_text(encoding="utf-8"))
    problems = _diff(payload, found)
    for key, value in (("seed", seed), ("n_per_hop", n_per_hop), ("hops", hops)):
        if found.get(key) != value:
            problems.append(
                f"{key}: the committed file says {found.get(key)!r} but this run was "
                f"parameterised with {value!r} (config {cfg.get('_config_path')})"
            )
    if problems:
        detail = "\n".join(f"  - {p}" for p in problems)
        raise SystemExit(
            f"{out_path} does not match what {Path(__file__).name} re-derives:\n{detail}\n"
            "Either the committed list was edited by hand, the draw source changed, or a "
            "parameter moved. None of those is a thing to fix by re-writing the list "
            "without a decision — the set is frozen over all experiments."
        )
    print(f"[anchor] VERIFIED: {out_path} re-derives exactly ({payload['counts']}).")


if __name__ == "__main__":
    main()
