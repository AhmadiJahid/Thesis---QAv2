#!/usr/bin/env python3
"""MetaQA knowledge graph, built directly from kb.txt.

Ported from v1 ``scripts/kg.py``. Adapted for v2: the demo entry point reads the
kb path from config instead of the hard-coded ``Data/kb.txt``.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, List, Tuple


@dataclass
class MetaQAKG:
    """
    Faithful MetaQA knowledge graph built directly from kb.txt (no inverse edges,
    no normalization).

    - entity_to_id: maps entity string -> int id
    - id_to_entity: list where index -> entity string
    - triples: list of (subj_id, relation, obj_id)
    - out_adj: subj_id -> list of (relation, obj_id)  (fast forward lookups)
    - rel_counts: relation -> frequency
    """

    entity_to_id: Dict[str, int]
    id_to_entity: List[str]
    triples: List[Tuple[int, str, int]]
    out_adj: DefaultDict[int, List[Tuple[str, int]]]
    rel_counts: Counter


def build_metaqa_kg(kb_path: str | Path) -> MetaQAKG:
    kb_path = Path(kb_path)
    if not kb_path.exists():
        raise FileNotFoundError(f"kb.txt not found at: {kb_path.resolve()}")

    entity_to_id: Dict[str, int] = {}
    id_to_entity: List[str] = []
    triples: List[Tuple[int, str, int]] = []
    out_adj: DefaultDict[int, List[Tuple[str, int]]] = defaultdict(list)
    rel_counts: Counter = Counter()

    def get_id(entity: str) -> int:
        """Assign a stable integer id to each unique entity string (exactly as it appears)."""
        if entity not in entity_to_id:
            entity_to_id[entity] = len(id_to_entity)
            id_to_entity.append(entity)
        return entity_to_id[entity]

    with kb_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            parts = line.split("|")
            if len(parts) != 3:
                raise ValueError(f"Bad triple at line {line_no}: {line!r}")

            subj, rel, obj = (p.strip() for p in parts)
            if not subj or not rel or not obj:
                raise ValueError(f"Empty field at line {line_no}: {line!r}")

            sid = get_id(subj)
            oid = get_id(obj)

            triples.append((sid, rel, oid))
            out_adj[sid].append((rel, oid))
            rel_counts[rel] += 1

    return MetaQAKG(
        entity_to_id=entity_to_id,
        id_to_entity=id_to_entity,
        triples=triples,
        out_adj=out_adj,
        rel_counts=rel_counts,
    )


def _main() -> None:
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from run_config import load_config, load_paths, require, resolve_path

    p = argparse.ArgumentParser(description="Print KG summary statistics.")
    p.add_argument("--config", default="metaqa_kg_eval.json")
    p.add_argument("--kb", type=Path, default=None, help="Override the kb path from config.")
    p.add_argument("--top-relations", type=int, default=10)
    args = p.parse_args()

    cfg = load_config(args.config)
    paths_cfg = load_paths(require(cfg, "paths_config"))
    kb_path = args.kb or resolve_path(
        require(paths_cfg, "datasets." + require(cfg, "kb_key")),
        Path(paths_cfg["data_root_resolved"]),
    )

    kg = build_metaqa_kg(kb_path)
    print(f"KB: {kb_path}")
    print("Entities:", len(kg.id_to_entity))
    print("Triples:", len(kg.triples))
    print("Relations:", len(kg.rel_counts))
    print("Top relations:", kg.rel_counts.most_common(args.top_relations))


if __name__ == "__main__":
    _main()
