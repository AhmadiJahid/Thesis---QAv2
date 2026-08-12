#!/usr/bin/env python3
"""
NER masking for MuSiQue question JSONLs.

Produces two masked variants of every question from the same spans:
- typed:   PERSON / ORG / PLACE / DATE / NUM / ENTITY placeholders
- uniform: a single [MASK] token

Span handling (unchanged from v1): filters generic/common false positives, merges
NER fragmentation across small gaps, compresses broken adjacent placeholder runs
(``[PERSON][PLACE] [PERSON]`` -> ``[PERSON]``) and removes artefacts like
``[ENTITY]3``.

Ported from v1 ``MusiQue/scripts/ner_mask_musique_question_chunks.py``. Adapted for
v2: input glob, output directory, NER models, device and batch size come from
``configs/musique_prep.json``; the span rule tables come from
``configs/ner_masking_rules.json`` (verbatim v1 values); each NER model's parameter
count is printed and asserted against the ceiling in ``configs/model_limits.json``;
the run writes the standard trail.

    python MusiQue/scripts/ner_mask_musique_question_chunks.py --device 0 --overwrite
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from _prep_common import (
    dataset_path,
    expand_glob,
    load_config,
    load_prep,
    require,
    run_dir_for,
)

from model_size import assert_within_ceiling, load_limits
from run_artifacts import now_iso, write_run_artifacts
from seeding import set_global_seed

_PLACEHOLDER_RE = re.compile(r"\[(?:PERSON|ORG|PLACE|DATE|NUM|ENTITY|MASK)\]")

# Conservative English date patterns.
_DATE_RES = (
    re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),
    re.compile(
        r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
        r"\s+\d{1,2},?\s+\d{4}\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b\d{4}s\b"),
    re.compile(r"\b\d{4}\b"),
)
_NUM_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?(?:\s?(?:%|percent|million|billion|thousand))?\b", re.IGNORECASE
)


class MaskingRules:
    """The span rule tables from configs/ner_masking_rules.json."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.uniform_token: str = require(cfg, "uniform_token")
        self.fallback_typed: str = require(cfg, "fallback_typed")
        self.typed_priority: dict[str, int] = {
            k: int(v) for k, v in require(cfg, "typed_priority").items()
        }
        self.label_to_placeholder: dict[str, str] = require(cfg, "label_to_placeholder")
        self.generic_rejections: set[str] = set(require(cfg, "generic_rejections"))
        self.literal_keep_lower: set[str] = set(require(cfg, "literal_keep_lower"))
        self.literal_type_hints: dict[str, str] = require(cfg, "literal_type_hints")
        self.max_merge_gap_chars: int = int(require(cfg, "max_merge_gap_chars"))
        self.source: str | None = cfg.get("_config_path")

    def priority(self, placeholder: str) -> int:
        return self.typed_priority.get(placeholder, 0)


def load_masking_rules(config: str | Path = "ner_masking_rules.json") -> MaskingRules:
    return MaskingRules(load_config(config))


def _sanitize_slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", s.split("/")[-1]).strip("_")


def _load_tokenizer_for_ner(model_name: str) -> Any:
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        print("  Note: using slow tokenizer (use_fast=False).")
        return tok


def _normalize_entity_label(label: object) -> str:
    s = str(label).strip()
    if len(s) >= 2 and s[1] == "-":
        s = s[2:]
    return s.upper()


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            q = obj.get("question")
            if not isinstance(q, str):
                q = "" if q is None else str(q)
            obj = dict(obj)
            obj["question"] = q
            rows.append(obj)
    return rows


def _ner_placeholder(label: object, typed_mode: bool, rules: MaskingRules) -> str:
    if not typed_mode:
        return rules.uniform_token
    key = _normalize_entity_label(label)
    return rules.label_to_placeholder.get(key, rules.fallback_typed)


def _is_probably_generic_or_bad_span(span_text: str, label: str, rules: MaskingRules) -> bool:
    raw = span_text.strip()
    low = raw.lower()
    if not raw:
        return True
    if low in rules.generic_rejections:
        return True
    # Reject single lowercase common tokens unless they are strongly typed.
    if raw.islower() and " " not in raw and label not in {"[DATE]", "[NUM]"}:
        return True
    # Reject pure punctuation.
    if not any(ch.isalnum() for ch in raw):
        return True
    return False


def _score(ent: dict[str, Any]) -> float:
    s = ent.get("score")
    try:
        return float(s)
    except Exception:
        return 0.0


def _is_mid_word_gap(text: str, gap_start: int, gap_end: int) -> bool:
    """True when the gap sits inside a single word (no whitespace, alnum neighbours)."""
    if gap_start >= gap_end:
        return False
    gap = text[gap_start:gap_end]
    if any(ch.isspace() for ch in gap):
        return False
    before_alnum = gap_start > 0 and text[gap_start - 1].isalnum()
    after_alnum = gap_end < len(text) and text[gap_end].isalnum()
    return before_alnum and after_alnum


def _entities_to_spans(
    entities: list[dict[str, Any]] | None,
    text: str,
    *,
    typed_mode: bool,
    rules: MaskingRules,
) -> list[tuple[int, int, str]]:
    if not entities:
        return []

    spans: list[tuple[int, int, str, float]] = []
    for ent in entities:
        start = ent.get("start")
        end = ent.get("end")
        if start is None or end is None:
            continue
        s = int(start)
        e = int(end)
        if s >= e or s < 0 or e > len(text):
            continue
        ph = _ner_placeholder(
            ent.get("entity_group") or ent.get("entity") or ent.get("label"), typed_mode, rules
        )
        piece = text[s:e]
        if piece.strip().lower() in rules.literal_keep_lower:
            continue
        if _is_probably_generic_or_bad_span(piece, ph, rules):
            continue
        spans.append((s, e, ph, _score(ent)))

    return _merge_and_clean_spans(text, spans, rules)


def _merge_and_clean_spans(
    text: str,
    spans: list[tuple[int, int, str, float]],
    rules: MaskingRules,
) -> list[tuple[int, int, str]]:
    if not spans:
        return []

    max_gap = rules.max_merge_gap_chars
    spans = sorted(spans, key=lambda x: (x[0], x[1], -x[3]))
    merged: list[tuple[int, int, str, float]] = []
    for s, e, ph, sc in spans:
        if not merged:
            merged.append((s, e, ph, sc))
            continue

        ps, pe, pph, psc = merged[-1]
        gap = text[pe:s] if s >= pe else ""
        # Overlap: keep the longer span, or the higher score if similar.
        if s < pe:
            prev_len = pe - ps
            cur_len = e - s
            choose_cur = (cur_len > prev_len) or (cur_len == prev_len and sc > psc)
            if choose_cur:
                merged[-1] = (s, e, ph, sc)
            continue

        # Same type: merge across small gaps. Covers NER fragmentation such as
        # "Tor"[ORG] + "qu" + "ay Lifeboat Station"[ORG]. Skip only when the gap
        # mixes alnum *and* whitespace (likely two separate entities).
        if pph == ph and len(gap) <= max_gap:
            has_alnum = any(ch.isalnum() for ch in gap)
            has_space = any(ch.isspace() for ch in gap)
            if not (has_alnum and has_space):
                merged[-1] = (ps, e, ph, max(psc, sc))
                continue

        # Different types: merge when the gap is mid-word (no whitespace, alnum
        # neighbours on both sides) - one tokenised word split across labels.
        if gap and len(gap) <= max_gap and _is_mid_word_gap(text, pe, s):
            winner = pph if rules.priority(pph) >= rules.priority(ph) else ph
            merged[-1] = (ps, e, winner, max(psc, sc))
            continue

        merged.append((s, e, ph, sc))

    return [(s, e, ph) for s, e, ph, _ in merged]


def _overlaps_blocked(s: int, e: int, blocked: list[tuple[int, int]]) -> bool:
    for bs, be in blocked:
        if s < be and bs < e:
            return True
    return False


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    out: list[tuple[int, int]] = [intervals[0]]
    for s, e in intervals[1:]:
        ps, pe = out[-1]
        if s <= pe:
            out[-1] = (ps, max(pe, e))
        else:
            out.append((s, e))
    return out


def _regex_spans(
    text: str, blocked: list[tuple[int, int]], typed: bool, rules: MaskingRules
) -> list[tuple[int, int, str]]:
    out: list[tuple[int, int, str]] = []

    date_ivs: list[tuple[int, int]] = []
    for rx in _DATE_RES:
        for m in rx.finditer(text):
            if not _overlaps_blocked(m.start(), m.end(), blocked):
                date_ivs.append((m.start(), m.end()))
    date_merged = _merge_intervals(date_ivs)
    blocked2 = blocked + date_merged
    for s, e in date_merged:
        out.append((s, e, "[DATE]" if typed else rules.uniform_token))

    num_ivs: list[tuple[int, int]] = []
    for m in _NUM_RE.finditer(text):
        if not _overlaps_blocked(m.start(), m.end(), blocked2):
            num_ivs.append((m.start(), m.end()))
    for s, e in _merge_intervals(num_ivs):
        out.append((s, e, "[NUM]" if typed else rules.uniform_token))

    return out


def _literal_hint_spans(
    text: str, blocked: list[tuple[int, int]], typed: bool, rules: MaskingRules
) -> list[tuple[int, int, str]]:
    out: list[tuple[int, int, str]] = []
    text_lower = text.lower()
    for lit, placeholder in rules.literal_type_hints.items():
        start = 0
        while True:
            idx = text_lower.find(lit, start)
            if idx == -1:
                break
            end = idx + len(lit)
            if not _overlaps_blocked(idx, end, blocked):
                out.append((idx, end, placeholder if typed else rules.uniform_token))
            start = end
    return out


def _apply_spans(text: str, spans: list[tuple[int, int, str]]) -> str:
    if not spans:
        return text
    spans = sorted(spans, key=lambda x: x[0])
    parts: list[str] = []
    last = 0
    for s, e, ph in spans:
        parts.append(text[last:s])
        parts.append(ph)
        last = e
    parts.append(text[last:])
    return "".join(parts)


def _best_placeholder(placeholders: Iterable[str], uniform: bool, rules: MaskingRules) -> str:
    vals = list(placeholders)
    if uniform:
        return rules.uniform_token
    counts = Counter(vals)
    # Most common wins; ties broken by typed priority.
    return sorted(counts.items(), key=lambda kv: (kv[1], rules.priority(kv[0])), reverse=True)[0][0]


def _compress_placeholder_runs(text: str, typed_mode: bool, rules: MaskingRules) -> str:
    # Pre-pass: collapse placeholders bridged by tiny non-whitespace fragments -
    # NER tokeniser artefacts that survive span merging, e.g. [ORG]qu[ORG] -> [ORG].
    frag_re = re.compile(
        r"(\[(?:PERSON|ORG|PLACE|DATE|NUM|ENTITY|MASK)\])"
        r"([^\[\]\s]{1," + str(rules.max_merge_gap_chars) + r"})"
        r"(\[(?:PERSON|ORG|PLACE|DATE|NUM|ENTITY|MASK)\])"
    )

    def _frag_repl(m: re.Match[str]) -> str:
        left, right = m.group(1), m.group(3)
        if left == right:
            return left
        if not typed_mode:
            return rules.uniform_token
        return left if rules.priority(left) >= rules.priority(right) else right

    prev = None
    while text != prev:
        prev = text
        text = frag_re.sub(_frag_repl, text)

    # Collapse runs such as [PERSON][PLACE] [PERSON] -> [PERSON]
    pattern = re.compile(
        r"((?:\[(?:PERSON|ORG|PLACE|DATE|NUM|ENTITY|MASK)\](?:[\s'’.,;/:-]*)?){2,})"
    )

    def repl(match: re.Match[str]) -> str:
        tokens = _PLACEHOLDER_RE.findall(match.group(1))
        return _best_placeholder(tokens, uniform=not typed_mode, rules=rules)

    out = pattern.sub(repl, text)
    # Remove attached digits: [ENTITY]3 -> [ENTITY]
    out = re.sub(r"(\[(?:PERSON|ORG|PLACE|DATE|NUM|ENTITY|MASK)\])(?=\d)", r"\1 ", out)
    return re.sub(r"\s+", " ", out).strip()


def _cleanup_masked_text(text: str, typed_mode: bool, rules: MaskingRules) -> str:
    out = _compress_placeholder_runs(text, typed_mode, rules)
    # Normalize spaces before punctuation and possessives.
    out = re.sub(r"\s+([,.;:?!])", r"\1", out)
    out = re.sub(r"\[(PERSON|ORG|PLACE|DATE|NUM|ENTITY|MASK)\]\s+'s", r"[\1]'s", out)
    return re.sub(r"\s{2,}", " ", out).strip()


def _mask_from_entities(
    entities: list[dict[str, Any]] | None,
    text: str,
    *,
    typed_mode: bool,
    use_regex: bool,
    use_literal_hints: bool,
    rules: MaskingRules,
) -> str:
    if not text:
        return ""

    spans = _entities_to_spans(entities, text, typed_mode=typed_mode, rules=rules)
    blocked = [(s, e) for s, e, _ in spans]

    extra: list[tuple[int, int, str]] = []
    if use_regex:
        extra.extend(_regex_spans(text, blocked, typed_mode, rules))
        blocked = blocked + [(s, e) for s, e, _ in extra]
    if use_literal_hints:
        extra.extend(_literal_hint_spans(text, blocked, typed_mode, rules))

    all_spans = sorted(spans + extra, key=lambda x: (x[0], x[1]))
    # Final non-overlap pass.
    final_spans: list[tuple[int, int, str]] = []
    last_end = -1
    for s, e, ph in all_spans:
        if s < last_end:
            continue
        final_spans.append((s, e, ph))
        last_end = e

    return _cleanup_masked_text(_apply_spans(text, final_spans), typed_mode, rules)


def _predict_entities_dataset(ner_pipe: Any, ds: Any, batch_size: int) -> list[list[dict[str, Any]]]:
    from transformers.pipelines.pt_utils import KeyDataset

    out: list[list[dict[str, Any]]] = []
    for item in ner_pipe(KeyDataset(ds, "question"), batch_size=batch_size):
        if item is None:
            out.append([])
        elif isinstance(item, list):
            out.append(item)
        elif isinstance(item, dict):
            out.append([item])
        else:
            out.append([])
    if len(out) != len(ds):
        raise RuntimeError(f"NER output length mismatch: got {len(out)}, expected {len(ds)}")
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="musique_prep.json")
    p.add_argument("--rules-config", default="ner_masking_rules.json")
    p.add_argument("--inputs", nargs="*", type=Path, help="Question JSONL files (ordered).")
    p.add_argument("--input-glob", default=None, help="Override the config glob (data-root relative).")
    p.add_argument("--out-dir", type=Path, default=None, help="Root; each model writes under <slug>/.")
    p.add_argument("--run-dir", type=Path, default=None)
    p.add_argument("--ner-model", action="append", dest="ner_models", default=None, help="HF model id (repeatable).")
    p.add_argument("--device", type=int, default=None, help="Transformers device id (-1 = CPU).")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--split-typed-uniform-dirs", action="store_true", default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg, paths_cfg = load_prep(args.config)
    section = require(cfg, "ner_mask")
    rules = load_masking_rules(args.rules_config)
    limits = load_limits("model_limits.json")

    from datasets import Dataset
    from transformers import pipeline

    seed = args.seed if args.seed is not None else int(require(cfg, "seed"))
    seeded = set_global_seed(seed)
    device = args.device if args.device is not None else int(require(section, "device"))
    batch_size = args.batch_size if args.batch_size is not None else int(require(section, "batch_size"))
    models = args.ner_models or list(require(section, "ner_models"))
    use_regex = bool(require(section, "regex_num_date"))
    use_literal_hints = bool(require(section, "literal_place_hints"))
    split_dirs = (
        args.split_typed_uniform_dirs
        if args.split_typed_uniform_dirs is not None
        else bool(require(section, "split_typed_uniform_dirs"))
    )
    exclude_suffix = require(section, "exclude_stem_suffix")

    if args.inputs:
        inputs = [Path(p).resolve() for p in args.inputs]
    else:
        pattern = args.input_glob or require(section, "input_glob")
        inputs = [p for p in expand_glob(paths_cfg, pattern) if not p.stem.endswith(exclude_suffix)]
        if not inputs:
            raise SystemExit(f"no inputs matched glob {pattern!r} under the data root")

    out_dir = args.out_dir or dataset_path(paths_cfg, require(section, "out_dir_key"))
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = args.run_dir or run_dir_for(paths_cfg, require(section, "run_subdir"))

    per_model_stats: dict[str, dict[str, Any]] = {}
    model_size_records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for model_name in models:
        slug = _sanitize_slug(model_name)
        model_root = out_dir / slug
        typed_root = model_root / "typed"
        uniform_root = model_root / "uniform"
        if split_dirs:
            typed_root.mkdir(parents=True, exist_ok=True)
            uniform_root.mkdir(parents=True, exist_ok=True)
        else:
            model_root.mkdir(parents=True, exist_ok=True)

        print(f"Loading NER pipeline: {model_name} (batch_size={batch_size}) ...")
        tokenizer = _load_tokenizer_for_ner(model_name)
        ner_pipe = pipeline(
            "token-classification",
            model=model_name,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            device=device,
        )
        model_size_records.append(
            assert_within_ceiling(
                ner_pipe.model, component="ner", model_id=model_name, limits=limits
            )
        )

        total_rows = 0
        files_out = 0
        for inp in inputs:
            base = inp.name
            if split_dirs:
                t_path = typed_root / base
                u_path = uniform_root / base
                if (t_path.exists() or u_path.exists()) and not args.overwrite:
                    raise SystemExit(f"Refusing to overwrite (use --overwrite): {t_path} or {u_path}")
            else:
                out_path = model_root / base
                if out_path.exists() and not args.overwrite:
                    raise SystemExit(f"Refusing to overwrite (use --overwrite): {out_path}")

            records = _load_jsonl_records(inp)
            ds = Dataset.from_list(records)
            questions = ds["question"]
            try:
                entities_list = _predict_entities_dataset(ner_pipe, ds, batch_size)
            except Exception as ex:
                errors.append({"model": model_name, "file": str(inp), "error": f"batch_ner: {ex}"})
                continue

            if split_dirs:
                tf = t_path.open("w", encoding="utf-8")
                uf = u_path.open("w", encoding="utf-8")
            else:
                tf = out_path.open("w", encoding="utf-8")
                uf = None

            try:
                for i, obj in enumerate(records):
                    try:
                        q = questions[i]
                        ent_i = entities_list[i]
                        mt = _mask_from_entities(
                            ent_i,
                            q,
                            typed_mode=True,
                            use_regex=use_regex,
                            use_literal_hints=use_literal_hints,
                            rules=rules,
                        )
                        mu = _mask_from_entities(
                            ent_i,
                            q,
                            typed_mode=False,
                            use_regex=use_regex,
                            use_literal_hints=use_literal_hints,
                            rules=rules,
                        )
                        row_t = {
                            "id": obj.get("id"),
                            "index": obj.get("index"),
                            "question": q,
                            "question_masked_typed": mt,
                        }
                        if split_dirs:
                            assert uf is not None
                            tf.write(json.dumps(row_t, ensure_ascii=False) + "\n")
                            uf.write(
                                json.dumps(
                                    {
                                        "id": obj.get("id"),
                                        "index": obj.get("index"),
                                        "question": q,
                                        "question_masked_uniform": mu,
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                        else:
                            tf.write(
                                json.dumps({**row_t, "question_masked_uniform": mu}, ensure_ascii=False)
                                + "\n"
                            )
                        total_rows += 1
                    except Exception as ex:
                        errors.append({"model": model_name, "file": str(inp), "error": str(ex)})
            finally:
                tf.close()
                if uf is not None:
                    uf.close()

            files_out += 1
            target = f"{slug}/typed|uniform" if split_dirs else f"{slug}/{base}"
            print(f"  {model_name}: {inp.name} ({len(records)} rows) -> {target}")

        per_model_stats[model_name] = {
            "slug": slug,
            "files": files_out,
            "total_rows": total_rows,
            "batch_size": batch_size,
        }

    metrics = {
        "script": Path(__file__).name,
        "created_utc": now_iso(),
        "seed": seed,
        "seeded": seeded,
        "ner_models": models,
        "model_sizes": model_size_records,
        "batch_size": batch_size,
        "device": device,
        "regex_num_date_enabled": use_regex,
        "literal_place_hints_enabled": use_literal_hints,
        "split_typed_uniform_dirs": split_dirs,
        "masking_rules_config": rules.source,
        "typed_mapping_note": (
            "PER/PERSON->[PERSON], ORG->[ORG], LOC/GPE/FAC->[PLACE], DATE/TIME->[DATE], "
            "numeric labels->[NUM], ambiguous labels->[ENTITY]"
        ),
        "cleanup_note": "collapses broken adjacent placeholder runs and removes artefacts like [ENTITY]3",
        "inputs": [str(p) for p in inputs],
        "out_dir": str(out_dir.resolve()),
        "per_model": per_model_stats,
        "errors": errors,
    }
    snapshot = {
        "script": Path(__file__).name,
        "config_path": cfg.get("_config_path"),
        "rules_config_path": rules.source,
        "inputs": [str(p) for p in inputs],
        "out_dir": str(out_dir),
        "ner_models": models,
        "device": device,
        "batch_size": batch_size,
        "seed": seed,
        "overwrite": args.overwrite,
        "split_typed_uniform_dirs": split_dirs,
    }
    write_run_artifacts(
        run_dir,
        config_snapshot=snapshot,
        metrics=metrics,
        note_title="MuSiQue question NER masking",
        note_lines=[
            f"- Seed: {seed}",
            f"- Models: {', '.join(models)}",
            f"- Output root: `{out_dir}`",
            f"- Regex NUM/DATE pass: {use_regex}; literal hints: {use_literal_hints}",
            f"- Batch size: {batch_size}; device: {device}",
            f"- Rows masked per model: { {m: s['total_rows'] for m, s in per_model_stats.items()} }",
            f"- Errors recorded: {len(errors)}",
        ],
    )


if __name__ == "__main__":
    main()
