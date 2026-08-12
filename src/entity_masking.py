"""Entity masking for questions: replace movie titles and person names with placeholders.

Loads entities from the MetaQA ``kb.txt``. Uses Aho-Corasick for O(n) masking.
Optional corpus filtering reduces the entity set for a faster build. Used for
structure-based similarity (e.g. router few-shot matching).

Ported from v1 ``src/entity_masking.py``. Adapted for v2: the placeholders and the
root used to resolve relative corpus paths are caller-supplied (they come from
``configs/masking.json``) instead of defaulting inside the module.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

try:
    import ahocorasick
except ImportError:  # pragma: no cover - depends on the environment
    ahocorasick = None  # type: ignore


def load_entities_from_kb(kb_path: Path) -> tuple[set[str], set[str]]:
    """
    Extract movies and people from kb.txt.
    Returns (movies, people). Movies are subjects; people are objects of
    directed_by, written_by, starred_actors.
    """
    movies: set[str] = set()
    people: set[str] = set()
    person_relations = {"directed_by", "written_by", "starred_actors"}

    for line in Path(kb_path).read_text(encoding="utf-8").strip().splitlines():
        parts = line.strip().split("|")
        if len(parts) != 3:
            continue
        subj, pred, obj = parts[0].strip(), parts[1].strip(), parts[2].strip()
        movies.add(subj)
        if pred in person_relations:
            people.add(obj)

    return movies, people


def _load_corpus_text(corpus_paths: list[Path], root: Path) -> str:
    """Load and concatenate text from corpus files (JSON questions or plain text)."""
    chunks: list[str] = []
    for p in corpus_paths:
        path = p if Path(p).is_absolute() else root / p
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            data = json.loads(text)
            for v in (data.values() if isinstance(data, dict) else [data]):
                if isinstance(v, list):
                    for it in v:
                        if isinstance(it, dict) and "question" in it:
                            chunks.append(it["question"])
                elif isinstance(v, str):
                    chunks.append(v)
        else:
            chunks.extend(line.strip() for line in text.splitlines() if line.strip())
    return " ".join(chunks).lower()


def _filter_to_corpus(
    movies: set[str], people: set[str], corpus_text: str
) -> tuple[set[str], set[str]]:
    """Keep only entities that appear (case-insensitive) in the corpus."""
    movies_filtered = {m for m in movies if m.lower() in corpus_text}
    people_filtered = {p for p in people if p.lower() in corpus_text}
    return movies_filtered, people_filtered


def _build_combined_automaton(
    entities_with_placeholders: list[tuple[str, str]]
) -> "ahocorasick.Automaton | None":
    """Build one automaton from (entity, placeholder) pairs. Longest match wins."""
    if ahocorasick is None:
        return None
    aut = ahocorasick.Automaton()
    for entity, placeholder in entities_with_placeholders:
        if not entity:
            continue
        aut.add_word(entity.lower(), (entity, placeholder))
    aut.make_automaton()
    return aut


def _is_word_boundary(text: str, start: int, end: int) -> bool:
    """Check if span [start:end] is at word boundaries (not mid-word)."""
    before_ok = start == 0 or not text[start - 1].isalnum()
    after_ok = end >= len(text) or not text[end].isalnum()
    return before_ok and after_ok


def _apply_automaton(text: str, automaton: "ahocorasick.Automaton") -> str:
    """Replace matches using iter_long (longest non-overlapping), word-boundary aware."""
    text_lower = text.lower()
    matches = list(automaton.iter_long(text_lower))
    if not matches:
        return text
    spans = []
    for end_idx, (original, placeholder) in matches:
        start_idx = end_idx - len(original) + 1
        end_slice = end_idx + 1
        if _is_word_boundary(text_lower, start_idx, end_slice):
            spans.append((start_idx, end_slice, placeholder))
    spans.sort(key=lambda x: x[0])
    # iter_long yields non-overlapping matches
    result_parts = []
    last_end = 0
    for start, end_slice, placeholder in spans:
        result_parts.append(text[last_end:start])
        result_parts.append(placeholder)
        last_end = end_slice
    result_parts.append(text[last_end:])
    return "".join(result_parts)


def build_masker(
    kb_path: Path,
    *,
    movie_placeholder: str,
    person_placeholder: str,
    corpus_paths: list[Path] | None = None,
    corpus_root: Path | None = None,
) -> Callable[[str], str]:
    """
    Build a mask function that replaces movies and people with placeholders.
    Uses Aho-Corasick for O(n) per-question masking, longest match first.
    If corpus_paths is provided, keeps only entities that appear in the corpus.
    """
    if ahocorasick is None:
        raise ImportError(
            "pyahocorasick is required for entity masking. Install: pip install pyahocorasick"
        )

    movies, people = load_entities_from_kb(kb_path)

    if corpus_paths:
        if corpus_root is None:
            raise ValueError("corpus_root is required when corpus_paths is given")
        corpus_text = _load_corpus_text(corpus_paths, corpus_root)
        movies, people = _filter_to_corpus(movies, people, corpus_text)

    # Single automaton: longest match wins. "Michael Tully" (person) beats
    # "Michael"/"Tully" (movies). People are added first so an entity present in
    # both sets gets the person placeholder.
    combined: list[tuple[str, str]] = [
        (p, person_placeholder) for p in people
    ] + [(m, movie_placeholder) for m in movies if m not in people]
    aut = _build_combined_automaton(combined)

    def mask(text: str) -> str:
        if aut:
            return _apply_automaton(text, aut)
        return text

    return mask


def build_masker_from_config(
    masking_cfg: dict,
    *,
    kb_path: Path,
    corpus_paths: list[Path] | None,
    corpus_root: Path,
) -> Callable[[str], str]:
    """Convenience wrapper: placeholders come from configs/masking.json."""
    from run_config import require

    return build_masker(
        kb_path,
        movie_placeholder=require(masking_cfg, "movie_placeholder"),
        person_placeholder=require(masking_cfg, "person_placeholder"),
        corpus_paths=corpus_paths,
        corpus_root=corpus_root,
    )


def mask_question(text: str, mask_fn: Callable[[str], str]) -> str:
    """Apply masker to a question. Thin wrapper for clarity."""
    return mask_fn(text)
