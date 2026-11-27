#!/usr/bin/env python3
"""Check that the MetaQA KG built by scripts/kg.py can reproduce the gold answers.

MetaQA's official question-type files (``qa_*_qtype.txt``) are not in the data root, so
the gold relation path of each question is unknown. This script runs the closest
equivalent check: for every question in ``refined_{hop}hop.txt`` it finds the KB
entities named in the question text (word-bounded string match), enumerates every
relation path of exactly ``hop`` edges from each one - both edge directions - and asks
whether any path reproduces the gold answer set in ``answers_{hop}hop.txt`` exactly.

It does this under several topic-entity exclusion conventions, because MetaQA's gold
answer sets drop the topic entity from multi-hop answers (e.g. "co-directors of X"
excludes X). The per-convention counts say which convention the gold follows.

What a pass proves: the KG holds every edge needed for the gold answer, and the gold
set is exactly a relation-path image of a question entity. What it does not prove:
that the path found is the question's intended path (without qtype files that is
inferred, not read), nor that the topic entity a pipeline would link is the one found.

Optionally it writes gold-like decompositions (one templated sub-question per edge of
the found path, ``[#k]`` links) in decomposer ``results.json`` shape, so the existing
compile-execute backend (scripts/run_metaqa_compile_execute.py) can be measured on
input that is correct by construction. Two phrasings are written: ``compiler`` uses
the step templates scripts/evaluate_decompositions.py recognises; ``natural`` uses the
wording a decomposer most plausibly emits for a reverse step on a placeholder.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from kg import MetaQAKG, build_metaqa_kg  # noqa: E402
from run_artifacts import now_iso, write_run_artifacts  # noqa: E402
from run_config import load_config, load_paths, require, resolve_path, runs_path  # noqa: E402
from seeding import set_global_seed  # noqa: E402

Edge = Tuple[str, str]  # (relation, "fwd" | "rev")
CONVENTIONS = ("keep_topic", "drop_topic_final", "drop_topic_intermediate", "drop_topic_everywhere")


def entity_mentions(question: str, entities: Sequence[str], min_len: int) -> List[str]:
    """KB entity strings occurring word-bounded in the question, longest first."""
    found = []
    for e in entities:
        if len(e) < min_len:
            continue
        i = question.find(e)
        while i != -1:
            left = question[i - 1] if i > 0 else " "
            j = i + len(e)
            right = question[j] if j < len(question) else " "
            if not left.isalnum() and not right.isalnum():
                found.append(e)
                break
            i = question.find(e, i + 1)
    return sorted(found, key=lambda s: (-len(s), s))


class PathOracle:
    def __init__(self, kg: MetaQAKG) -> None:
        self.kg = kg
        self.fwd: Dict[int, Dict[str, Set[int]]] = defaultdict(lambda: defaultdict(set))
        self.rev: Dict[int, Dict[str, Set[int]]] = defaultdict(lambda: defaultdict(set))
        for s, r, o in kg.triples:
            self.fwd[s][r].add(o)
            self.rev[o][r].add(s)
        self.relations = sorted(kg.rel_counts)

    def paths(self, start: int, hops: int, drop_intermediate: bool) -> List[Tuple[Tuple[Edge, ...], Set[int]]]:
        out: List[Tuple[Tuple[Edge, ...], Set[int]]] = []

        def rec(front: Set[int], path: List[Edge]) -> None:
            if len(path) == hops:
                out.append((tuple(path), front))
                return
            for rel in self.relations:
                for direction, table in (("fwd", self.fwd), ("rev", self.rev)):
                    nxt: Set[int] = set()
                    for x in front:
                        nxt |= table[x][rel]
                    if drop_intermediate and len(path) + 1 < hops:
                        nxt.discard(start)
                    if nxt:
                        rec(nxt, path + [(rel, direction)])

        rec({start}, [])
        return out

    def matches(self, start: int, hops: int, gold: Set[int]) -> Dict[str, List[Tuple[Edge, ...]]]:
        """For each convention, the paths whose (convention-adjusted) image equals gold."""
        hits: Dict[str, List[Tuple[Edge, ...]]] = {c: [] for c in CONVENTIONS}
        for drop_mid in (False, True):
            for path, image in self.paths(start, hops, drop_mid):
                for drop_final in (False, True):
                    got = image - {start} if drop_final else image
                    if got != gold:
                        continue
                    conv = {
                        (False, False): "keep_topic",
                        (False, True): "drop_topic_final",
                        (True, False): "drop_topic_intermediate",
                        (True, True): "drop_topic_everywhere",
                    }[(drop_mid, drop_final)]
                    hits[conv].append(path)
        return hits


# ---- gold-like decomposition rendering -------------------------------------------

_FWD_NAMED = {
    "directed_by": "Who directed {x}?",
    "written_by": "Who wrote {x}?",
    "starred_actors": "Who starred in {x}?",
}
_REV_NAMED = {
    "directed_by": "What movies did {x} direct?",
    "written_by": "What movies did {x} write?",
    "starred_actors": "What movies did {x} star in?",
}
_FWD_REF = {
    "directed_by": "Who directed {x}?",
    "written_by": "Who wrote {x}?",
    "starred_actors": "Who starred in {x}?",
    "has_genre": "What genres are {x}?",
    "in_language": "What languages are {x} in?",
    "release_year": "When were {x} released?",
    "has_tags": "What tags describe {x}?",
}
_REV_REF = {
    "compiler": {
        "directed_by": "What other movies were directed by {x}?",
        "written_by": "What other movies were written by {x}?",
        "starred_actors": "What other movies were starred by {x}?",
    },
    "natural": {
        "directed_by": "What movies did {x} direct?",
        "written_by": "What movies did {x} write?",
        "starred_actors": "What movies did {x} star in?",
    },
}


def render(topic: str, path: Sequence[Edge], phrasing: str) -> Optional[str]:
    lines = []
    for k, (rel, direction) in enumerate(path, start=1):
        x = topic if k == 1 else f"[#{k - 1}]"
        if k == 1:
            table = _FWD_NAMED if direction == "fwd" else _REV_NAMED
        else:
            table = _FWD_REF if direction == "fwd" else _REV_REF[phrasing]
        if rel not in table:
            return None
        lines.append(f"{k}. " + table[rel].format(x=x))
    return "\n".join(lines)


def load_hop(paths_cfg: dict, data_root: Path, hop: int, sep: str) -> Tuple[List[str], List[Set[str]]]:
    q = resolve_path(require(paths_cfg, "datasets.metaqa_questions_template").format(hop=hop), data_root)
    a = resolve_path(require(paths_cfg, "datasets.metaqa_answers_template").format(hop=hop), data_root)
    qs = [ln.strip() for ln in q.read_text(encoding="utf-8").splitlines() if ln.strip()]
    ans = [{x.strip() for x in ln.split(sep) if x.strip()} for ln in a.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(qs) != len(ans):
        raise SystemExit(f"hop {hop}: {len(qs)} questions vs {len(ans)} answer lines")
    return qs, ans


def check(kg: MetaQAKG, questions: List[str], answers: List[Set[str]], hop: int, min_len: int) -> dict:
    oracle = PathOracle(kg)
    entities = list(kg.entity_to_id)
    conv_counts: Counter = Counter()
    path_types: Counter = Counter()
    rows = []
    n_reproduced = n_longest = n_gold_missing = 0
    for q, gold_names in zip(questions, answers):
        missing = [g for g in gold_names if g not in kg.entity_to_id]
        n_gold_missing += len(missing)
        gold = {kg.entity_to_id[g] for g in gold_names if g in kg.entity_to_id}
        cands = entity_mentions(q, entities, min_len)
        row = {"question": q, "hop": hop, "candidates": cands[:5], "topic": None,
               "topic_rank": None, "conventions": [], "path": None}
        if not missing:
            for rank, t in enumerate(cands):
                hits = oracle.matches(kg.entity_to_id[t], hop, gold)
                ok = [c for c in CONVENTIONS if hits[c]]
                if ok:
                    # prefer the convention that holds everywhere, then the path found first
                    pref = "drop_topic_everywhere" if hits["drop_topic_everywhere"] else ok[0]
                    row.update(topic=t, topic_rank=rank, conventions=ok, path=[list(e) for e in sorted(hits[pref])[0]])
                    break
        if row["path"] is not None:
            n_reproduced += 1
            n_longest += row["topic_rank"] == 0
            conv_counts.update(row["conventions"])
            path_types[" -> ".join(f"{r}:{d}" for r, d in row["path"])] += 1
        rows.append(row)
    return {
        "n": len(questions),
        "reproduced_by_some_path": n_reproduced,
        "topic_is_longest_mention": n_longest,
        "gold_answers_not_kg_entities": n_gold_missing,
        "per_convention": {c: conv_counts.get(c, 0) for c in CONVENTIONS},
        "distinct_path_types": len(path_types),
        "path_types": dict(path_types.most_common()),
        "unreproduced": [r["question"] for r in rows if r["path"] is None],
        "_rows": rows,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="metaqa_gold_path_check.json")
    p.add_argument("--run-dir", type=Path, default=None)
    p.add_argument("--data-root", type=Path, default=None, help="Override the data root (fixture runs).")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    paths_cfg = load_paths(require(cfg, "paths_config"))
    seed = args.seed if args.seed is not None else int(require(cfg, "seed"))
    seeded = set_global_seed(seed)
    data_root = args.data_root or Path(paths_cfg["data_root_resolved"])
    kb_path = resolve_path(require(paths_cfg, "datasets.metaqa_kb"), data_root)
    run_dir = args.run_dir or runs_path(paths_cfg, require(cfg, "run_subdir"))
    run_dir.mkdir(parents=True, exist_ok=True)

    kg = build_metaqa_kg(kb_path)
    per_hop = {}
    fixtures: Dict[str, list] = {ph: [] for ph in require(cfg, "phrasings")}
    for hop in require(cfg, "hops"):
        qs, ans = load_hop(paths_cfg, data_root, int(hop), require(cfg, "answer_separator"))
        res = check(kg, qs, ans, int(hop), int(require(cfg, "min_entity_len")))
        for r in res.pop("_rows"):
            if r["path"] is None:
                continue
            for ph in fixtures:
                dec = render(r["topic"], [tuple(e) for e in r["path"]], ph)
                if dec is not None:
                    fixtures[ph].append({"question": r["question"], "hop_count": hop, "decomposition": dec,
                                         "few_shot_source": "gold_path_oracle"})
        per_hop[str(hop)] = res
        print(f"hop {hop}: n={res['n']} reproduced={res['reproduced_by_some_path']} "
              f"topic_longest={res['topic_is_longest_mention']} per_convention={res['per_convention']} "
              f"path_types={res['distinct_path_types']}")
    for ph, rows in fixtures.items():
        (run_dir / f"gold_like_decompositions_{ph}.json").write_text(
            json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {len(rows)} gold-like decompositions ({ph}) to {run_dir}")

    metrics = {"script": Path(__file__).name, "created_utc": now_iso(), "seed": seed, "seeded": seeded,
               "kb_path": str(kb_path), "kg_entities": len(kg.id_to_entity), "kg_triples": len(kg.triples),
               "kg_relations": len(kg.rel_counts), "per_hop": per_hop,
               "gold_like_rows": {ph: len(v) for ph, v in fixtures.items()}}
    write_run_artifacts(
        run_dir,
        config_snapshot={"script": Path(__file__).name, "config_path": cfg.get("_config_path"),
                         "data_root": str(data_root), "seed": seed, **{k: v for k, v in cfg.items() if not k.startswith("_")}},
        metrics=metrics,
        note_title="MetaQA gold-path reproduction check",
        note_lines=[f"- hop {h}: {v['reproduced_by_some_path']}/{v['n']} gold sets reproduced; "
                    f"per convention {v['per_convention']}" for h, v in per_hop.items()],
        prefix="gold_path_",
    )


if __name__ == "__main__":
    main()
