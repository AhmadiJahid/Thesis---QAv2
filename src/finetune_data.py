"""Training data and prompt construction for the decomposer's LoRA arms (issue #13).

Everything here runs on CPU with no model weights, so the whole selection path -
source loading, hop filtering, the seeded cap, the train/eval overlap assertion and the
prompt/completion formatting - is testable and smoke-testable before a run earns GPU time.

Three things this module is deliberate about:

- **The train/eval overlap assertion is hard, and cannot pass vacuously.** A fine-tuned arm
  that saw evaluation questions is not a comparison arm, so :func:`assert_no_eval_overlap`
  raises and names the offending ids rather than warning. It runs against the whole ADR 0007
  evaluation set (all hop depths), never only the hops an arm is evaluated on. Because an
  assertion against an *empty* id set would pass while proving nothing,
  :func:`load_eval_ids` asserts the config-declared counts (200 per hop, 600 total) and
  :func:`assert_no_eval_overlap` refuses outright when handed no evaluation ids.
- **The training prompt is the prompting arm's prompt.** It is rendered by
  ``components/decomposer/run_decomposer.py``'s own template helpers, imported rather than
  reimplemented, with the few-shot block empty. v1's four copies of ``decomposer.py`` are
  the reason: a second implementation of prompt assembly drifts, and then the fine-tuned
  arm is trained on a string shape inference never produces.
- **Nothing here samples without the seed.** The cap draws through
  :func:`seeding.new_rng`, so an arm's example set is a function of (source, seed, cap).
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from run_config import require, resolve_path
from seeding import new_rng

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DECOMPOSER_DIR = _REPO_ROOT / "components" / "decomposer"

_ID_HOP_RX = re.compile(r"^(?P<h>\d+)hop")
#: A bare "#3" reference (MuSiQue's own convention) that is not already bracketed.
_BARE_REF_RX = re.compile(r"(?<!\[)#(\d+)")

REFERENCE_STYLES = ("as_is", "bracketed")

#: Key in the *paths* config that overrides ``eval_set.expected`` (see
#: :func:`expected_eval_counts`). Only configs/smoke_paths.json carries it.
PATHS_EVAL_EXPECTED_KEY = "eval_set_expected"


@dataclass(frozen=True)
class TrainingExample:
    """One supervised example: a question and its gold decomposition steps."""

    row_id: str
    question: str
    steps: tuple[str, ...]
    hop: int


# ------------------------------------------------------------------------------- loading


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def steps_from_row(row: dict[str, Any], step_fields: list[str]) -> list[str]:
    """Step-question strings from the first populated field in ``step_fields``.

    Two shapes exist in this pipeline: raw MuSiQue rows carry ``question_decomposition``
    (a list of step objects with a ``question``), and the enriched pool carries
    ``few_shot_decomposition_musique`` (a list of step strings written from the same
    source). Which fields to try, and in which order, is a config value.
    """
    for field in step_fields:
        value = row.get(field)
        steps: list[str] = []
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item.strip():
                    steps.append(item.strip())
                elif isinstance(item, dict):
                    question = item.get("question")
                    if isinstance(question, str) and question.strip():
                        steps.append(question.strip())
        if steps:
            return steps
    return []


def hop_from_id(row_id: str | None) -> int | None:
    """Hop depth from a MuSiQue id prefix (``2hop__``, ``3hop1__``, ``4hop2__``)."""
    if not row_id:
        return None
    m = _ID_HOP_RX.match(str(row_id))
    return int(m.group("h")) if m else None


def build_examples(
    rows: list[dict[str, Any]], data_cfg: dict[str, Any], *, max_reported_ids: int
) -> tuple[list[TrainingExample], dict[str, Any]]:
    """Turn source rows into examples, reporting what was dropped and why.

    The hop count comes from the row's own ``hop_count`` field when it has a usable one,
    else from the id prefix, else from the number of steps. Where two of those are
    available and disagree the row is still kept here (this is training data, not gold under
    measurement) but the disagreement is counted and the offending ids are listed, capped at
    ``max_reported_ids``.

    Counting is not always enough: for an arm that *filters* on hop depth the labels are
    load-bearing, so :func:`select_arm_examples` turns any disagreement into a hard failure
    when the arm declares ``train_hops``.
    """
    id_field = require(data_cfg, "id_field")
    question_field = require(data_cfg, "question_field")
    hop_count_field = require(data_cfg, "hop_count_field")
    step_fields = list(require(data_cfg, "step_fields"))

    examples: list[TrainingExample] = []
    stats = {
        "rows_in": len(rows),
        "dropped_missing_id": 0,
        "dropped_missing_question": 0,
        "dropped_missing_steps": 0,
        "hop_disagreement_count": 0,
        "hop_disagreement_ids": [],
        "hop_disagreement_ids_capped_at": int(max_reported_ids),
    }
    for row in rows:
        row_id = row.get(id_field)
        if not isinstance(row_id, str) or not row_id.strip():
            stats["dropped_missing_id"] += 1
            continue
        question = row.get(question_field)
        if not isinstance(question, str) or not question.strip():
            stats["dropped_missing_question"] += 1
            continue
        steps = steps_from_row(row, step_fields)
        if not steps:
            stats["dropped_missing_steps"] += 1
            continue

        field_hop = row.get(hop_count_field)
        field_hop = (
            int(field_hop)
            if isinstance(field_hop, int) and not isinstance(field_hop, bool) and field_hop > 0
            else None
        )
        id_hop = hop_from_id(row_id)
        candidates = [h for h in (field_hop, id_hop) if h is not None]
        if len({*candidates, len(steps)}) > 1:
            stats["hop_disagreement_count"] += 1
            if len(stats["hop_disagreement_ids"]) < int(max_reported_ids):
                stats["hop_disagreement_ids"].append(row_id)
        hop = candidates[0] if candidates else len(steps)

        examples.append(
            TrainingExample(
                row_id=row_id.strip(),
                question=question.strip(),
                steps=tuple(steps),
                hop=int(hop),
            )
        )
    return examples, stats


def expected_eval_counts(
    paths_cfg: dict[str, Any], eval_cfg: dict[str, Any]
) -> tuple[int, int, str]:
    """The declared evaluation-set counts: ``(ids_per_hop, total_ids, where they came from)``.

    ADR 0007's 200-per-hop / 600-total live in ``eval_set.expected`` of the training config.
    A *paths* config may override them under :data:`PATHS_EVAL_EXPECTED_KEY` - only
    ``configs/smoke_paths.json`` does, because its fabricated fixture tree carries one row
    per hop file. The override is a committed config value, not a relaxation in code: the
    counts are asserted either way, and which block was used is recorded in the run's
    metrics.
    """
    override = paths_cfg.get(PATHS_EVAL_EXPECTED_KEY)
    if isinstance(override, dict):
        source = f"{PATHS_EVAL_EXPECTED_KEY} in {paths_cfg.get('_config_path')}"
        block = override
    else:
        source = f"eval_set.expected in {eval_cfg.get('_config_path', '<training config>')}"
        block = require(eval_cfg, "expected")
    per_hop = int(require(block, "ids_per_hop"))
    total = int(require(block, "total_ids"))
    hops = [int(h) for h in require(eval_cfg, "hops")]
    if per_hop < 1 or total < 1:
        raise SystemExit(
            f"[finetune] {source} declares non-positive expected counts "
            f"(ids_per_hop={per_hop}, total_ids={total}). The evaluation set cannot be empty: "
            "the train/eval overlap assertion would have nothing to check."
        )
    if per_hop * len(hops) != total:
        raise SystemExit(
            f"[finetune] {source} is inconsistent: ids_per_hop={per_hop} over "
            f"{len(hops)} hop file(s) is {per_hop * len(hops)}, but total_ids={total}. "
            "A uniform per-hop count is assumed (ADR 0007: 200 per hop, 600 total); a "
            "non-uniform evaluation set is a change to src/finetune_data.py, made "
            "deliberately, not a count edited here."
        )
    return per_hop, total, source


def load_eval_ids(
    paths_cfg: dict[str, Any], eval_cfg: dict[str, Any]
) -> tuple[set[str], dict[str, Any]]:
    """Ids of the ADR 0007 evaluation set, per hop file, with counts.

    Two hard failures, both about the same danger - an overlap assertion that checks against
    fewer evaluation questions than it should still *passes*:

    - a missing hop file is fatal;
    - a hop file that does not yield exactly the declared number of distinct ids is fatal
      (see :func:`expected_eval_counts`). That is what catches a mis-resolved ``id_field``, a
      truncated or re-drawn file, and ids shared between two hop files, instead of letting
      :func:`assert_no_eval_overlap` record ``asserted: true`` over an empty set.
    """
    template = require(paths_cfg, "datasets." + require(eval_cfg, "questions_template_key"))
    id_field = require(eval_cfg, "id_field")
    data_root = Path(paths_cfg["data_root_resolved"])
    expected_per_hop, expected_total, expected_source = expected_eval_counts(paths_cfg, eval_cfg)

    ids: set[str] = set()
    per_hop: dict[str, int] = {}
    files: list[str] = []
    for hop in [int(h) for h in require(eval_cfg, "hops")]:
        path = resolve_path(str(template).format(hop=hop), data_root)
        if not path.exists():
            raise SystemExit(
                f"[finetune] evaluation-set file not found: {path}\n"
                f"It is the {hop}-hop file of the ADR 0007 evaluation set "
                f"(datasets.{require(eval_cfg, 'questions_template_key')} in the paths "
                f"config). Training cannot start without it: the train/eval overlap "
                f"assertion has nothing to check against."
            )
        hop_rows = load_jsonl(path)
        hop_ids = set()
        for row in hop_rows:
            value = row.get(id_field)
            if isinstance(value, str) and value.strip():
                hop_ids.add(value.strip())
        if len(hop_ids) != expected_per_hop:
            raise SystemExit(
                f"[finetune] REFUSING TO TRAIN: the {hop}-hop evaluation file yielded "
                f"{len(hop_ids)} distinct id(s), expected {expected_per_hop} "
                f"({expected_source}).\n"
                f"  file: {path}\n"
                f"  rows read: {len(hop_rows)}; id_field: {id_field!r}\n"
                "An evaluation id set smaller than declared makes the train/eval overlap "
                "assertion weaker than it claims to be - at zero ids it passes while "
                "checking nothing. Usual causes: the wrong id_field for this file, a "
                "truncated or re-drawn file, or duplicate ids inside it. The expected counts "
                "are ADR 0007's; changing them is a decision about the evaluation set, not a "
                "fix for this error."
            )
        per_hop[str(hop)] = len(hop_ids)
        ids |= hop_ids
        files.append(str(path))

    if len(ids) != expected_total:
        raise SystemExit(
            f"[finetune] REFUSING TO TRAIN: the evaluation set yielded {len(ids)} distinct "
            f"id(s) across hops {[int(h) for h in require(eval_cfg, 'hops')]}, expected "
            f"{expected_total} ({expected_source}).\n"
            f"  per-hop counts: {per_hop}\n"
            f"  files: {files}\n"
            "Each hop file matched its own expected count, so the shortfall means ids are "
            "shared between hop files - the evaluation set is not what ADR 0007 describes."
        )

    record = {
        "questions_template_key": require(eval_cfg, "questions_template_key"),
        "hops": [int(h) for h in require(eval_cfg, "hops")],
        "files": files,
        "ids_per_hop": per_hop,
        "num_ids": len(ids),
        "expected_ids_per_hop": expected_per_hop,
        "expected_total_ids": expected_total,
        "expected_counts_source": expected_source,
        "expected_counts_asserted": True,
    }
    return ids, record


def assert_no_eval_overlap(
    examples: list[TrainingExample], eval_ids: set[str], *, max_reported: int
) -> dict[str, Any]:
    """Refuse to train when any training id is an evaluation id.

    Fails loudly and names the offenders (capped at ``max_reported``). Overlap makes the
    arm's evaluation numbers meaningless, and a warning in a log tail is not a guard.

    An empty ``eval_ids`` is refused outright rather than reported as a clean check: over an
    empty set there is nothing to overlap with, so the record would say ``asserted: true``
    having proved nothing. :func:`load_eval_ids` should have failed first; this is the
    backstop for any other caller.
    """
    if not eval_ids:
        raise SystemExit(
            "[finetune] REFUSING TO TRAIN: the evaluation id set is empty, so the train/eval "
            "overlap check would pass without checking anything. Load the ADR 0007 "
            "evaluation ids (finetune_data.load_eval_ids, which asserts the declared "
            "200-per-hop / 600-total counts) before asserting disjointness."
        )
    train_ids = [ex.row_id for ex in examples]
    overlap = sorted({row_id for row_id in train_ids if row_id in eval_ids})
    if overlap:
        shown = overlap[:max_reported]
        more = "" if len(overlap) <= max_reported else f"\n  ... (+{len(overlap) - max_reported} more)"
        raise SystemExit(
            f"[finetune] REFUSING TO TRAIN: {len(overlap)} training example(s) carry an id "
            f"from the evaluation set (ADR 0007). Offending ids:\n  "
            + "\n  ".join(shown)
            + more
            + "\nA fine-tuned arm trained on its own evaluation questions is not a "
            "comparison arm. Fix the training source or the evaluation set selection."
        )
    return {
        "checked_training_ids": len(train_ids),
        "distinct_training_ids": len(set(train_ids)),
        "eval_ids": len(eval_ids),
        "overlap_count": 0,
        "overlap_ids": [],
        "asserted": True,
    }


# ----------------------------------------------------------------------------- selection


def filter_hops(examples: list[TrainingExample], hops: list[int] | None) -> list[TrainingExample]:
    if hops is None:
        return list(examples)
    wanted = {int(h) for h in hops}
    return [ex for ex in examples if ex.hop in wanted]


def cap_examples(
    examples: list[TrainingExample],
    max_examples: int | None,
    *,
    stratify_by_hop: bool,
    seed: int,
) -> list[TrainingExample]:
    """Seeded cap. ``stratify_by_hop`` spreads the cap evenly over the hop buckets.

    The draw is shuffle-then-take from a ``random.Random(seed)``, so the selected set is a
    function of (source order, seed, cap) and nothing else. Ordering of the returned list
    follows the source order, so two runs with the same seed write byte-identical datasets.
    """
    if max_examples is None or len(examples) <= int(max_examples):
        return list(examples)
    cap = int(max_examples)
    rng = new_rng(seed)

    if not stratify_by_hop:
        picked = set(rng.sample(range(len(examples)), cap))
        return [ex for i, ex in enumerate(examples) if i in picked]

    buckets: dict[int, list[int]] = {}
    for i, ex in enumerate(examples):
        buckets.setdefault(ex.hop, []).append(i)

    chosen: set[int] = set()
    remaining = cap
    # Smallest bucket first, so a bucket that cannot fill its share hands the remainder to
    # the larger buckets instead of silently shrinking the total below the cap.
    for pos, hop in enumerate(sorted(buckets, key=lambda h: (len(buckets[h]), h))):
        buckets_left = len(buckets) - pos
        share = min(len(buckets[hop]), -(-remaining // buckets_left))
        chosen |= set(rng.sample(buckets[hop], share))
        remaining -= share
    return [ex for i, ex in enumerate(examples) if i in chosen]


def resolve_arm(cfg: dict[str, Any], arm_name: str) -> dict[str, Any]:
    arms = require(cfg, "arms")
    if arm_name not in arms or arm_name == "_note":
        known = sorted(k for k in arms if k != "_note")
        raise SystemExit(
            f"unknown arm {arm_name!r}; {cfg.get('_config_path')} defines {known}"
        )
    return dict(require(arms, arm_name))


def arm_source_path(arm: dict[str, Any], paths_cfg: dict[str, Any]) -> Path:
    """The arm's training source: an explicit path if set, else its datasets key."""
    explicit = arm.get("train_source_path")
    if isinstance(explicit, str) and explicit.strip():
        return resolve_path(explicit.strip(), Path(paths_cfg["data_root_resolved"]))
    return resolve_path(
        require(paths_cfg, "datasets." + require(arm, "train_source_key")),
        Path(paths_cfg["data_root_resolved"]),
    )


def select_arm_examples(
    arm: dict[str, Any],
    rows: list[dict[str, Any]],
    data_cfg: dict[str, Any],
    *,
    seed: int,
    max_reported_ids: int,
    limit: int | None = None,
) -> tuple[list[TrainingExample], dict[str, Any]]:
    """Apply the arm's hop filter and cap. Returns (examples, a record of the selection).

    For an arm that declares ``train_hops`` (the generalisation arm trains on 2-hop and
    3-hop only), a row whose hop signals disagree is **fatal**: the filter decides which
    rows the model sees, so a wrong hop label there silently changes what the arm is. Arms
    with ``train_hops: null`` train on every row regardless, so for them the disagreement is
    recorded and not fatal.
    """
    examples, load_stats = build_examples(rows, data_cfg, max_reported_ids=max_reported_ids)
    train_hops = require(arm, "train_hops")
    if train_hops is not None and load_stats["hop_disagreement_count"]:
        shown = load_stats["hop_disagreement_ids"]
        more = (
            ""
            if load_stats["hop_disagreement_count"] <= len(shown)
            else f"\n  ... (+{load_stats['hop_disagreement_count'] - len(shown)} more)"
        )
        raise SystemExit(
            f"[finetune] REFUSING TO TRAIN: this arm filters on hop depth "
            f"(train_hops={train_hops}), and {load_stats['hop_disagreement_count']} source "
            f"row(s) carry disagreeing hop signals - the "
            f"{data_cfg.get('hop_count_field')!r} field, the id prefix and the number of "
            f"steps do not agree. Offending ids:\n  "
            + "\n  ".join(shown)
            + more
            + "\nThe hop filter is only as trustworthy as the labels it filters on: with a "
            "wrong label, rows of the excluded depth reach training and the generalisation "
            "claim is void. Fix the source, or use an arm with train_hops: null (where the "
            "disagreement is recorded rather than fatal)."
        )
    after_hops = filter_hops(examples, train_hops)
    max_examples = require(arm, "max_examples")
    stratify = bool(require(arm, "stratify_cap_by_hop"))
    selected = cap_examples(after_hops, max_examples, stratify_by_hop=stratify, seed=seed)
    if limit is not None:
        selected = selected[: max(0, int(limit))]

    record = {
        "source_rows": load_stats,
        "train_hops": train_hops,
        "hop_disagreement_fatal": train_hops is not None,
        "num_after_hop_filter": len(after_hops),
        "max_examples": max_examples,
        "stratify_cap_by_hop": stratify,
        "limit": limit,
        "num_selected": len(selected),
        "selected_hop_counts": hop_counts(selected),
        "seed": seed,
    }
    return selected, record


def hop_counts(examples: list[TrainingExample]) -> dict[str, int]:
    out: dict[str, int] = {}
    for ex in examples:
        key = str(ex.hop)
        out[key] = out.get(key, 0) + 1
    return {k: out[k] for k in sorted(out, key=int)}


# ------------------------------------------------------------------- prompt / completion


def _run_decomposer():
    """The decomposer runner, imported for its prompt-template helpers.

    Importing it (rather than copying ``fill_template``) is what keeps the training prompt
    and the inference prompt the same string. It is import-safe: its module level pulls in
    no torch and no weights.
    """
    if str(_DECOMPOSER_DIR) not in sys.path:
        sys.path.insert(0, str(_DECOMPOSER_DIR))
    import run_decomposer  # noqa: PLC0415 - deliberately local, see docstring

    return run_decomposer


def select_prompt_file(model_cfg: dict[str, Any], *, guided: bool) -> str:
    """The prompt file ``run_decomposer.py`` would use for this guidance setting."""
    unguided_prompt_file = require(model_cfg, "unguided_prompt_file")
    prompt_file = require(model_cfg, "prompt_file")
    if not guided and unguided_prompt_file:
        return str(unguided_prompt_file)
    return str(prompt_file)


def build_prompt(
    template: str,
    *,
    prompt_style: str,
    question: str,
    hop: int | None,
    few_shot_examples: str,
    unguided_hop_placeholder: str,
    chat_marker: str | None = None,
    tokenizer: Any = None,
    enable_thinking: bool = False,
) -> tuple[str, bool]:
    """Render the training prompt exactly as the runner renders it at inference.

    Returns ``(prompt_text, dry_run_placeholder)``. For a ``chat_template`` model the real
    prompt needs the tokenizer's chat template; without a tokenizer (a ``--dry-run``, which
    loads no weights) the messages are dumped as JSON and the flag comes back true, so the
    artifact says the logged text is a placeholder rather than the trained string.
    """
    rd = _run_decomposer()
    if prompt_style == "chat_template":
        if not chat_marker:
            raise SystemExit(
                "prompt_style 'chat_template' needs chat_template.split_marker from the "
                "model config"
            )
        messages = rd.build_chat_messages(
            template,
            marker=chat_marker,
            question=question,
            hop_count=hop,
            few_shot_examples=few_shot_examples,
            unguided_hop_placeholder=unguided_hop_placeholder,
        )
        if tokenizer is None:
            return json.dumps(messages, ensure_ascii=False, indent=2), True
        return (
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            ),
            False,
        )
    if prompt_style != "plain":
        raise SystemExit(f"unknown prompt_style {prompt_style!r} (expected plain or chat_template)")
    return (
        rd.fill_template(
            template,
            question=question,
            hop_count=hop,
            few_shot_examples=few_shot_examples,
            unguided_hop_placeholder=unguided_hop_placeholder,
        ),
        False,
    )


def format_completion(
    steps: tuple[str, ...] | list[str], *, reference_style: str, number_lines: bool
) -> str:
    """The supervised target: one step per line, matching the prompt's stated format.

    ``reference_style`` 'as_is' keeps the gold step text verbatim; 'bracketed' rewrites a
    bare ``#k`` to ``[#k]``, the convention the prompt files state. See the trade-off note
    on ``prompt.target_reference_style`` in configs/finetune_decomposer.json.
    """
    if reference_style not in REFERENCE_STYLES:
        raise SystemExit(
            f"unknown target_reference_style {reference_style!r} (expected one of "
            f"{list(REFERENCE_STYLES)})"
        )
    lines: list[str] = []
    for i, step in enumerate(steps, start=1):
        text = step.strip()
        if reference_style == "bracketed":
            text = _BARE_REF_RX.sub(r"[#\1]", text)
        lines.append(f"{i}. {text}" if number_lines else text)
    return "\n".join(lines)


def build_training_rows(
    examples: list[TrainingExample],
    *,
    template: str,
    prompt_style: str,
    prompt_cfg: dict[str, Any],
    unguided_hop_placeholder: str,
    chat_marker: str | None = None,
    tokenizer: Any = None,
    enable_thinking: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Prompt/completion rows for trl's SFT trainer, plus a record of how they were built.

    The rows are prompt-completion typed on purpose: with
    ``training.completion_only_loss`` the prompt tokens are masked out of the loss, so the
    model is trained on the decomposition text rather than on reproducing the instruction
    block.
    """
    guided = bool(require(prompt_cfg, "guided"))
    few_shot_examples = str(require(prompt_cfg, "few_shot_examples"))
    reference_style = str(require(prompt_cfg, "target_reference_style"))
    number_lines = bool(require(prompt_cfg, "number_target_lines"))

    rows: list[dict[str, Any]] = []
    placeholder_prompts = 0
    for ex in examples:
        prompt_text, placeholder = build_prompt(
            template,
            prompt_style=prompt_style,
            question=ex.question,
            hop=ex.hop if guided else None,
            few_shot_examples=few_shot_examples,
            unguided_hop_placeholder=unguided_hop_placeholder,
            chat_marker=chat_marker,
            tokenizer=tokenizer,
            enable_thinking=enable_thinking,
        )
        placeholder_prompts += int(placeholder)
        rows.append(
            {
                "prompt": prompt_text,
                "completion": format_completion(
                    ex.steps, reference_style=reference_style, number_lines=number_lines
                ),
                "row_id": ex.row_id,
                "hop": ex.hop,
            }
        )

    record = {
        "num_rows": len(rows),
        "guided": guided,
        "few_shot_examples_in_prompt": bool(few_shot_examples),
        "target_reference_style": reference_style,
        "number_target_lines": number_lines,
        "prompt_style": prompt_style,
        "dataset_type": "prompt_completion",
        "prompts_are_dry_run_placeholders": placeholder_prompts > 0,
    }
    if placeholder_prompts:
        record["prompt_placeholder_note"] = (
            f"{placeholder_prompts} prompt(s) were rendered as a messages JSON dump because "
            "no tokenizer was loaded (--dry-run). A real run renders them through the "
            "model's chat template; the text logged here is not the trained string."
        )
    return rows, record
