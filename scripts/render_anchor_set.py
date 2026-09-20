#!/usr/bin/env python3
"""Render the 45 anchor items of a finished run (or several, side by side) for hand inspection.

Supervisor item A4 (2026-08-27, recorded in
``docs/plan/2026-09-14-post-supervisor-meeting-phase.md``): *"I want you to pick some anchor
sample that you will check manually by yourself in every experiment… 15 from each hop… those
will be fixed over all the experiments."* Item A7 adds the cross-model half: *"the same
example, check them with Quen and Mistral"*. This script is the reader for both — one
command, one Markdown sheet, the same 45 ids every time.

**It post-filters existing outputs and nothing else.** The input is one or more
``*_per_item.json`` files the evaluator already wrote, which carry the question, the gold
steps, the predicted steps and every per-item metric. No model is loaded, nothing is
re-scored, and no run path is touched: ``run_decomposer.py`` and
``musique_decompositions_evaluator.py`` are untouched by this file, so the pinned-600
assertion and every existing arm read exactly as they did before. That also means an arm can
only be rendered *after* it has been evaluated on the pinned 600 — the anchor 45 are a subset
of it, so the inspection costs no extra inference.

Usage::

    python scripts/render_anchor_set.py \\
        --arm "exp-015 unguided=runs/exp-015/eval/eval_per_item.json" \\
        --arm "exp-008 full_train=runs/exp-011/exp008_full_train/eval_per_item.json" \\
        --out runs/anchor/2026-09-20-baselines.md

    # label defaults to the file's parent directory
    python scripts/render_anchor_set.py --arm runs/exp-015/eval/eval_per_item.json

**Prompts and raw generations.** Append ``@<decomposer run dir>`` to an arm to add, per
anchor item, a collapsible section holding the *full prompt* that arm sent for that item
and the ``decomposition_raw`` it got back. The prompt is read from
``<run dir>/prompts_log/anchor_<query_id>_hop<h>.txt`` and the raw generation from
``<run dir>/results.json``, joined on ``query_id``::

    python scripts/render_anchor_set.py \\
        --arm "exp-020=runs/exp-020/eval/eval_per_item.json@runs/exp-020/decomposer/20260919_223903" \\
        --out runs/anchor/2026-09-20-exp020.md

An anchor item with no prompt file renders an explicit *prompt not logged for this run*
marker. It never renders as silence: an empty section reads as an empty prompt.

**Recovering the prompts of a run that finished before anchor logging existed.** Every run
up to and including exp-020 logged only the every-5th prompt, by position, so 33 of the 45
anchor items have no prompt file at all and the 12 that happen to sit on a multiple of 5
are identifiable only by counting rows. Those prompts are recoverable on CPU, with no GPU
and no model: ``--dry-run`` rebuilds the prompt byte for byte from the committed config
snapshot and the same retrieval artifact. Read the arm's own ``config.json`` snapshot for
``model``, ``condition`` and ``retrieval.input_resolved``, then::

    .venv/bin/python components/decomposer/run_decomposer.py \\
        --model mistral_7b_instruct --config decomposer_musique.json \\
        --condition unguided \\
        --retrieval-input runs/exp-014/retrieval/s2_pinned600_imbalanced_rerank_top5.jsonl \\
        --dry-run --dry-run-limit 600 \\
        --output-root runs/anchor/exp-020-prompts

``--quantization`` is irrelevant to a dry run (no model is loaded) and the seed comes from
the config. That writes all 600 positional prompts *and* the 45 ``anchor_*.txt`` files.
Point the renderer at it with a per-arm ``--prompts-dir``, keeping ``@<run dir>`` on the
arm so ``decomposition_raw`` still comes from the real run::

    python scripts/render_anchor_set.py \\
        --arm "exp-020=runs/exp-020/eval/eval_per_item.json@runs/exp-020/decomposer/20260919_223903" \\
        --prompts-dir "exp-020=runs/anchor/exp-020-prompts/<run id>/prompts_log" \\
        --out runs/anchor/2026-09-20-exp020.md

The sheet says, per arm and in words, that such a prompt is *regenerated (CPU dry run, not
written by this run)*, so it is never mistaken for one the run itself wrote. When the arm
also carries ``@<run dir>``, the regeneration's own config snapshot is **checked** against
the run's on ``model``, ``condition``, ``prompt_sha256`` and ``retrieval.input_sha256``, and
a mismatch is refused with the field named — pointing ``--prompts-dir`` at the wrong
regeneration would otherwise put one arm's prompt beside another arm's generation and the
page would look perfectly correct. A ``--prompts-dir`` on an arm with no run directory has
nothing to check against and is marked *unverified against a run*.

The sheet goes under ``runs/`` (gitignored) by default and must stay there: it contains
question text and gold decompositions, and data never enters git (CLAUDE.md).

Exit codes: 0 when every arm covers all 45 anchor ids, 1 when one does not (pass
``--allow-missing`` to render a partial sheet anyway — the gaps are then stated in it).
A missing prompt is *not* an exit-1 condition: prompts are an addition to the sheet, and
the arms that predate anchor logging are the normal case.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from anchor_set import AnchorSetError, find_anchor_prompt  # noqa: E402
from anchor_set import load_anchor_ids as _load_anchor_ids  # noqa: E402
from run_artifacts import now_iso  # noqa: E402
from run_config import load_config, load_paths, require, resolve_path, runs_path  # noqa: E402

#: The per-item schema this script reads. Stated so a file written against a different shape
#: is refused by name instead of half-rendering.
PER_ITEM_SCHEMA = "musique_decomposition_per_item/1"

#: Id fields a per-item row may carry, in the evaluator's own precedence (``_item_id``),
#: plus ``item_id``, which is what the file it writes actually uses.
_ID_KEYS = ("item_id", "query_id", "id")


class _Missing:
    """The field is **not in the file**, as distinct from present and ``null``.

    The two must not render the same. ``ged_fallback: null`` means the evaluator ran the
    exact GED and recorded no fallback; a missing ``decomp_mean`` means the file predates
    that metric. Collapsing both to one marker invites reading the second as the first.
    """

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid only
        return "<missing>"


#: Singleton for "this field is absent from the row".
MISSING = _Missing()


def display_path(path: Path | str) -> str:
    """A path as the record should carry it: relative to the repo root when it is inside it.

    A sheet rendered from a throwaway worktree would otherwise record that worktree's
    absolute path, which tells a later reader nothing about which file was read.
    """
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def assert_out_path_never_enters_git(out: Path) -> None:
    """Refuse an ``--out`` that git would let someone commit.

    The sheet carries question text and gold decompositions. CLAUDE.md is unconditional:
    data never enters git. A path outside the working tree is fine; a path inside it is
    only fine when ``git check-ignore`` says the path is ignored — asking git is the only
    check that cannot drift from what ``git add`` would actually do.

    A git that cannot answer (no git on PATH, not a repository, an internal error) is a
    refusal too: "could not prove it is ignored" is not "it is ignored".
    """
    resolved = Path(out).resolve()
    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError:
        return  # Outside the working tree — git has no say over it.

    try:
        proc = subprocess.run(
            ["git", "check-ignore", "-q", "--", str(resolved)],
            cwd=str(REPO_ROOT), capture_output=True, text=True,
        )
    except OSError as exc:  # pragma: no cover - depends on the box, not on the input
        raise SystemExit(
            f"refusing --out {out}: it is inside the working tree and git could not be "
            f"asked whether it is ignored ({exc}). The sheet carries question text and "
            "gold decompositions, and data never enters git (CLAUDE.md). Write it under "
            "runs/ instead."
        ) from exc

    if proc.returncode == 0:
        return  # git says the path is ignored.
    if proc.returncode == 1:
        raise SystemExit(
            f"refusing --out {out}: that path is inside the repository and is NOT "
            "gitignored, so the sheet could be committed. It carries question text and "
            "gold decompositions, and data never enters git (CLAUDE.md). Write it under "
            "runs/ (gitignored) or outside the working tree."
        )
    raise SystemExit(
        f"refusing --out {out}: git check-ignore exited {proc.returncode} "
        f"({proc.stderr.strip() or 'no message'}), so it could not be established that the "
        "path is ignored. The sheet carries question text and gold decompositions, and "
        "data never enters git (CLAUDE.md)."
    )


#: The two provenances a prompt on this sheet can have, written out. "--prompts-dir" is a
#: flag name and tells a reader nothing; "regenerated" tells them the prompt was rebuilt
#: afterwards by a CPU dry run and is not a byte the run emitted.
PROMPTS_FROM_RUN = "written by this run"
PROMPTS_REGENERATED = "regenerated (CPU dry run, not written by this run)"

#: Whether a regeneration was checked against the run it claims to reproduce.
PROMPTS_VERIFIED = "verified against the run's config snapshot"
PROMPTS_UNVERIFIED = "unverified against a run"

#: The config-snapshot fields a regeneration must agree with the run on. Not a full diff:
#: these four are what decides whether the same prompt text comes out — which model folder
#: (prompt style, few-shot block), which condition (what the prompt says about the hop
#: count), the exact prompt file by content, and the exact retrieval artifact by content.
#: Everything else a snapshot carries (quantization, device, timing) cannot move a prompt.
PROMPT_PROVENANCE_FIELDS = (
    "model",
    "condition",
    "prompt_sha256",
    "retrieval.input_sha256",
)


def _snapshot_field(snapshot: dict[str, Any], dotted: str) -> Any:
    node: Any = snapshot
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return MISSING
        node = node[part]
    return node


def assert_regeneration_matches_run(
    label: str, prompts_dir: Path, run_dir: Path
) -> None:
    """Refuse a ``--prompts-dir`` that does not reproduce the run it is shown beside.

    The recovery path rebuilds a finished run's prompts with ``--dry-run``. Nothing stops
    someone pointing it at the wrong regeneration — a different model, a different
    condition, a different retrieval artifact — and the sheet would then show one arm's
    prompt next to another arm's generation with no sign of it. That is the single worst
    failure this script can have, because it is invisible: the page looks right.

    So the regeneration's own config snapshot is compared against the run's on the four
    fields that decide the prompt text. A mismatch is refused with the field named. A
    snapshot that cannot be found is refused too: "could not prove it reproduces the run"
    is not "it reproduces the run", the same stance the out-path guard takes.
    """
    run_snapshot_path = run_dir / "config.json"
    # The regeneration directory is a run's prompts_log, so its snapshot is its parent's.
    regen_snapshot_path = prompts_dir.parent / "config.json"
    for who, path in (("run", run_snapshot_path), ("regeneration", regen_snapshot_path)):
        if not path.exists():
            raise SystemExit(
                f"--prompts-dir {label}: no {who} config snapshot at {path}, so it could "
                "not be established that the regenerated prompts reproduce this arm's run. "
                "Point --prompts-dir at the prompts_log of a run directory written by "
                "run_decomposer.py (it holds config.json next to prompts_log/)."
            )
    run_snapshot = json.loads(run_snapshot_path.read_text(encoding="utf-8"))
    regen_snapshot = json.loads(regen_snapshot_path.read_text(encoding="utf-8"))

    mismatches: list[str] = []
    for field in PROMPT_PROVENANCE_FIELDS:
        run_value = _snapshot_field(run_snapshot, field)
        regen_value = _snapshot_field(regen_snapshot, field)
        if run_value != regen_value:
            mismatches.append(
                f"  {field}: run={run_value!r} regeneration={regen_value!r}"
            )
    if mismatches:
        raise SystemExit(
            f"--prompts-dir {label}: the regenerated prompts do NOT reproduce this arm's "
            "run. These fields of the two config snapshots disagree:\n"
            + "\n".join(mismatches)
            + f"\n  run:          {display_path(run_snapshot_path)}"
            + f"\n  regeneration: {display_path(regen_snapshot_path)}\n"
            "Rendering them together would put one arm's prompt beside another arm's "
            "generation, and the page would look correct. Re-run the dry-run recovery with "
            "this arm's own model, condition and retrieval artifact (they are in the run's "
            "config.json)."
        )


class Arm:
    """One evaluated arm: a label, the file it came from, and its rows keyed by item id.

    Optionally also the arm's decomposer run directory — ``results.json`` (for
    ``decomposition_raw``) and ``prompts_log/`` (for the prompt text). Both are ``None``
    when no ``@<run dir>`` was given, and the sheet says so per item rather than leaving a
    blank where a prompt would be.
    """

    def __init__(
        self,
        label: str,
        path: Path,
        rows: dict[str, dict[str, Any]],
        header: dict[str, Any],
        *,
        run_dir: Path | None = None,
        results: dict[str, dict[str, Any]] | None = None,
        prompts_dir: Path | None = None,
        prompts_dir_source: str | None = None,
        prompts_verified: str | None = None,
    ):
        self.label = label
        self.path = path
        self.rows = rows
        self.header = header
        self.run_dir = run_dir
        self.results = results
        self.prompts_dir = prompts_dir
        #: :data:`PROMPTS_FROM_RUN` or :data:`PROMPTS_REGENERATED`, spelled out in full
        #: wherever the sheet mentions a prompt. A prompt the run itself wrote and a prompt
        #: rebuilt afterwards on CPU are different evidence, and the page has to say which
        #: one the reader is looking at in the words, not in a flag name.
        self.prompts_dir_source = prompts_dir_source
        #: How much the regeneration was checked against the run it claims to reproduce:
        #: :data:`PROMPTS_VERIFIED` or :data:`PROMPTS_UNVERIFIED`. ``None`` when the
        #: question does not arise (the prompts came from the run itself, or there are none).
        self.prompts_verified = prompts_verified

    def prompts_provenance(self) -> str:
        """The one phrase the sheet uses for "where did this prompt come from"."""
        if self.prompts_dir_source is None:
            return "no prompts directory"
        if self.prompts_verified is None:
            return self.prompts_dir_source
        return f"{self.prompts_dir_source}, {self.prompts_verified}"

    def prompt_file(self, qid: str) -> Path | None:
        """The logged prompt file for one anchor id, or ``None`` if there is not one."""
        if self.prompts_dir is None:
            return None
        return find_anchor_prompt(self.prompts_dir, qid)

    def raw_generation(self, qid: str) -> Any:
        """``decomposition_raw`` for one anchor id — ``MISSING`` when there is no row."""
        if self.results is None:
            return MISSING
        row = self.results.get(qid)
        if row is None:
            return MISSING
        return row.get("decomposition_raw", MISSING)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", default="anchor_render.json")
    p.add_argument(
        "--arm",
        action="append",
        default=[],
        metavar="[LABEL=]PER_ITEM_JSON[@RUN_DIR]",
        help="An evaluated arm to render. Repeatable; arms appear as columns in the order "
        "given. Without a LABEL the file's parent directory names the column. With "
        "'@RUN_DIR' (the decomposer output directory holding results.json and "
        "prompts_log/) each anchor item also gets a collapsible section with that arm's "
        "full prompt and its decomposition_raw.",
    )
    p.add_argument(
        "--prompts-dir",
        action="append",
        default=[],
        metavar="LABEL=DIR",
        help="Read one arm's anchor prompts from DIR instead of from its run's "
        "prompts_log/. Repeatable, once per arm. This is how a run that finished before "
        "anchor logging existed gets its prompts: regenerate them on CPU with "
        "'run_decomposer.py --dry-run --dry-run-limit 600' (see the description above) and "
        "point at that run's prompts_log. decomposition_raw still comes from the real run's "
        "results.json via '@RUN_DIR'.",
    )
    p.add_argument(
        "--anchor",
        type=Path,
        default=None,
        help="The committed anchor id list. Default: ids_file from the config.",
    )
    p.add_argument("--out", type=Path, default=None, help="Where the Markdown sheet is written.")
    p.add_argument(
        "--allow-missing",
        action="store_true",
        help="Render even when an arm does not cover all 45 anchor ids. The sheet then says "
        "which ids are missing from which arm, so a gap can never be read as a result.",
    )
    args = p.parse_args()
    if not args.arm:
        raise SystemExit("pass at least one --arm [LABEL=]<eval_per_item.json>")
    return args


def parse_arm_spec(spec: str) -> tuple[str, Path, Path | None]:
    """``"label=path@rundir"`` and its shorter forms -> ``(label, path, run_dir)``.

    Split on the *first* ``=`` only, so a label may not contain one but a path may.
    A bare path is labelled by its parent directory, which is how the run tree already
    names arms (``runs/exp-011/exp008_full_train/eval_per_item.json`` -> ``exp008_full_train``).

    The optional run directory is split off the *last* ``@``, which is unambiguous for
    every path in this repo's run tree. ``run_dir`` is ``None`` when the spec has no ``@``;
    the arm then renders its tables exactly as before and says, per item, that no run
    directory was given.
    """
    if "=" in spec:
        label, _, raw = spec.partition("=")
        label = label.strip()
        raw = raw.strip()
        if not label or not raw:
            raise SystemExit(f"--arm {spec!r}: expected LABEL=PATH with both sides non-empty")
    else:
        raw = spec.strip()
        label = ""
    run_dir: Path | None = None
    if "@" in raw:
        raw, _, run_raw = raw.rpartition("@")
        raw = raw.strip()
        run_raw = run_raw.strip()
        if not raw or not run_raw:
            raise SystemExit(
                f"--arm {spec!r}: expected PATH@RUN_DIR with both sides non-empty"
            )
        run_dir = Path(run_raw)
    path = Path(raw)
    return (label or path.parent.name or path.stem), path, run_dir


def parse_prompts_dir_spec(spec: str) -> tuple[str, Path]:
    """``"label=dir"`` -> ``(label, dir)``. Both sides are required: a bare directory could
    not be attributed to an arm, and guessing which arm it belongs to would be worse than
    refusing."""
    label, sep, raw = spec.partition("=")
    label, raw = label.strip(), raw.strip()
    if not sep or not label or not raw:
        raise SystemExit(f"--prompts-dir {spec!r}: expected LABEL=DIR with both sides non-empty")
    return label, Path(raw)


def load_results(path: Path) -> dict[str, dict[str, Any]]:
    """A decomposer ``results.json`` keyed by ``query_id``.

    The runner writes a JSON *list* of per-row objects (``query_id``, ``question``,
    ``decomposition``, ``decomposition_raw``, cost fields). The prompt text is deliberately
    not in it — ``prompts_log/`` is the prompt's home — which is why the sheet joins the two
    files rather than reading one.
    """
    if not path.exists():
        raise SystemExit(
            f"results.json not found: {path}\n"
            "The '@<run dir>' of an --arm must be the decomposer output directory (the one "
            "holding results.json, metrics.json and prompts_log/)."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise SystemExit(
            f"{path}: expected a JSON list of result rows, got {type(payload).__name__}"
        )
    by_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(payload):
        if not isinstance(row, dict):
            raise SystemExit(f"{path}: row {index} is not an object")
        qid = row.get("query_id")
        if qid is None:
            continue  # A question source with no ids (MetaQA 'lines'); nothing to join on.
        qid = str(qid)
        if qid in by_id:
            raise SystemExit(
                f"{path}: duplicate query_id {qid!r}; the raw generation for an anchor item "
                "would be ambiguous."
            )
        by_id[qid] = row
    return by_id


def load_anchor_ids(path: Path) -> tuple[list[str], dict[str, str], dict[str, Any]]:
    """``(ids in anchor order, {id: hop}, the list's provenance header)``.

    A thin adapter over :func:`anchor_set.load_anchor_ids`: the parsing lives in ``src/``
    because ``components/decomposer/run_decomposer.py`` reads the same file to decide which
    prompts to log, and two copies of that parser would let the writer and the reader of the
    anchor prompt files drift apart. This adds only the CLI's refusal style and the
    ``ids_file`` provenance line the sheet prints.
    """
    try:
        ordered, hop_of, header = _load_anchor_ids(path)
    except AnchorSetError as exc:
        raise SystemExit(str(exc)) from exc
    header["ids_file"] = display_path(path)
    return ordered, hop_of, header


def _row_id(row: dict[str, Any]) -> str | None:
    for key in _ID_KEYS:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
    return None


def load_per_item(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Load an evaluator ``*_per_item.json`` into ``({item_id: row}, header)``.

    A duplicate id is fatal: with two rows for one anchor item the sheet would show one of
    them and there would be no way to tell which.
    """
    if not path.exists():
        raise SystemExit(f"per-item file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise SystemExit(
            f"{path}: not an object with an 'items' list. This script reads the evaluator's "
            f"'{PER_ITEM_SCHEMA}' per-item artifact."
        )
    schema = payload.get("schema")
    if schema != PER_ITEM_SCHEMA:
        raise SystemExit(
            f"{path}: schema is {schema!r}, expected {PER_ITEM_SCHEMA!r}. Re-score the arm "
            "with scripts/musique_decompositions_evaluator.py rather than rendering a file "
            "of unknown shape."
        )
    by_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(payload["items"]):
        if not isinstance(row, dict):
            raise SystemExit(f"{path}: items[{index}] is not an object")
        qid = _row_id(row)
        if qid is None:
            raise SystemExit(f"{path}: items[{index}] carries none of {_ID_KEYS}")
        if qid in by_id:
            raise SystemExit(f"{path}: duplicate item id {qid!r}")
        by_id[qid] = row
    header = {k: v for k, v in payload.items() if k != "items"}
    header["per_item_path"] = display_path(path)
    header["item_count"] = len(by_id)
    return by_id, header


def header_value(arm: "Arm", field: str) -> Any:
    """One arm's header field, or ``MISSING`` when the file does not carry it."""
    return arm.header.get(field, MISSING)


def metric_fields(arms: list[Arm], anchor_ids: list[str], structural: list[str]) -> list[str]:
    """Every non-structural field the anchor rows carry, in first-seen order.

    First-seen across arms in the order the arms were given, which is the evaluator's own
    emission order for the first arm and then anything a later arm adds. The script does not
    curate the list: the sheet shows what the files say.
    """
    seen: list[str] = []
    known = set(structural)
    for arm in arms:
        for qid in anchor_ids:
            row = arm.rows.get(qid)
            if row is None:
                continue
            for key in row:
                if key in known:
                    continue
                known.add(key)
                seen.append(key)
    return seen


def cell(value: Any, *, float_format: str, empty: str, null: str = "null") -> str:
    """One Markdown table cell: pipes escaped, newlines folded, floats formatted.

    ``MISSING`` (the field is absent from the file) and ``None`` (the field is present and
    ``null``) render differently and deliberately — see :class:`_Missing`.
    """
    if isinstance(value, _Missing):
        return empty
    if value is None:
        return null
    if isinstance(value, bool):
        text = "true" if value else "false"
    elif isinstance(value, float):
        text = float_format.format(value)
    elif isinstance(value, (list, tuple)):
        text = " | ".join(str(v) for v in value)
    elif isinstance(value, dict):
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    else:
        text = str(value)
    text = text.replace("\\", "\\\\").replace("|", "\\|")
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br>")
    return text.strip() or empty


#: Markers for the two ways a prompt can be absent. Stated, never silent: a collapsible
#: section that opens onto nothing reads as "the prompt was empty", which is a different
#: and much more alarming fact than "this run did not log it".
NO_RUN_DIR_MARKER = (
    "_prompt not logged for this run — no run directory was given for this arm "
    "(`--arm LABEL=<eval_per_item.json>@<decomposer run dir>`)._"
)
NO_PROMPT_FILE_MARKER = (
    "_prompt not logged for this run — no `anchor_<id>_hop*.txt` in `{dir}`. Runs that "
    "finished before anchor logging existed only logged every Nth prompt by position; "
    "regenerate this run's prompts on CPU with `run_decomposer.py --dry-run "
    "--dry-run-limit 600` and pass `--prompts-dir {label}=<that run>/prompts_log`._"
)
NO_RESULTS_ROW_MARKER = "_no `results.json` row for this id in this arm._"


def fenced(text: str, *, language: str = "text") -> list[str]:
    """``text`` in a fence long enough to survive whatever backticks are inside it.

    Prompt files contain code fences (few-shot exemplars, model output). A fixed ```` ``` ````
    would be closed early by the content and the rest of the prompt would leak into the
    page as Markdown — silently, and looking like a rendering quirk rather than a truncated
    prompt.
    """
    longest = 0
    run = 0
    for ch in text:
        run = run + 1 if ch == "`" else 0
        longest = max(longest, run)
    fence = "`" * max(3, longest + 1)
    return [fence + language, *text.split("\n"), fence]


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out.extend("| " + " | ".join(r) + " |" for r in rows)
    return out


def _steps(row: dict[str, Any] | None, key: str) -> list[str]:
    if row is None:
        return []
    value = row.get(key)
    if not isinstance(value, list):
        return []
    return [str(v) for v in value]


def prompt_coverage(arm: Arm, anchor_ids: list[str]) -> tuple[int, int]:
    """``(anchor items with a prompt file, anchor items with a results.json row)``."""
    prompts = sum(1 for qid in anchor_ids if arm.prompt_file(qid) is not None)
    raws = sum(1 for qid in anchor_ids if not isinstance(arm.raw_generation(qid), _Missing))
    return prompts, raws


def render_prompt_section(arm: Arm, qid: str) -> list[str]:
    """One arm's collapsible prompt + raw generation for one anchor item."""
    out = ["<details>", f"<summary><strong>{arm.label}</strong> — prompt and raw generation "
           f"for <code>{qid}</code></summary>", ""]

    prompt_path = arm.prompt_file(qid)
    if prompt_path is None:
        if arm.prompts_dir is None:
            out.append(NO_RUN_DIR_MARKER)
        else:
            out.append(
                NO_PROMPT_FILE_MARKER.format(
                    dir=display_path(arm.prompts_dir), label=arm.label
                )
            )
    else:
        out.append(
            f"Prompt log `{display_path(prompt_path)}` "
            f"({arm.prompts_provenance()}). The file is shown whole — its header, the "
            "prompt body, and whatever the run recorded under `--- Raw generation ---`, "
            "which is empty in a prompt regenerated by `--dry-run` because a dry run "
            "generates nothing. The generation for such an arm is the block below, read "
            "from the real run's `results.json`."
        )
        out.append("")
        out.extend(fenced(prompt_path.read_text(encoding="utf-8")))
    out.append("")

    raw = arm.raw_generation(qid)
    out.append("**`decomposition_raw`** (from the arm's `results.json`):")
    out.append("")
    if isinstance(raw, _Missing):
        if arm.results is None:
            out.append(NO_RUN_DIR_MARKER)
        else:
            out.append(NO_RESULTS_ROW_MARKER)
    elif raw is None:
        out.append("_`decomposition_raw` is null in this arm's `results.json` row._")
    elif str(raw) == "":
        out.append("_`decomposition_raw` is the empty string in this arm's row._")
    else:
        out.extend(fenced(str(raw)))
    out.append("")
    out.append("</details>")
    out.append("")
    return out


def render(
    arms: list[Arm],
    anchor_ids: list[str],
    hop_of: dict[str, str],
    anchor_header: dict[str, Any],
    *,
    structural: list[str],
    arm_header_fields: list[str],
    comparability_fields: list[str],
    float_format: str,
    empty: str,
    null: str,
) -> str:
    """The whole Markdown sheet, as one string."""
    fmt = {"float_format": float_format, "empty": empty, "null": null}
    fields = metric_fields(arms, anchor_ids, structural)
    labels = [a.label for a in arms]
    coverage = {a.label: sum(1 for qid in anchor_ids if qid in a.rows) for a in arms}
    total = len(anchor_ids)
    # Prompts are an optional addition: with no arm carrying a run directory the sheet is
    # byte-for-byte the shape it had before, so an existing sheet does not change meaning.
    with_prompts = any(a.run_dir is not None or a.prompts_dir is not None for a in arms)
    prompt_counts = {a.label: prompt_coverage(a, anchor_ids) for a in arms} if with_prompts else {}

    lines: list[str] = []
    lines.append(f"# Anchor set — {total} items, hand inspection")
    lines.append("")
    lines.append(
        f"Rendered {now_iso()} by `scripts/render_anchor_set.py`. This sheet is a "
        "**post-filter of finished evaluations**: every number below is copied from the "
        "per-item file the evaluator wrote for that arm. Nothing here re-scores or re-runs."
    )
    lines.append("")
    lines.append(
        f"Anchor list `{anchor_header.get('ids_file')}` — seed {anchor_header.get('seed')}, "
        f"{anchor_header.get('n_per_hop')} per hop, hops {anchor_header.get('hops')}, "
        f"counts {json.dumps(anchor_header.get('counts'), sort_keys=True)}, drawn "
        f"{anchor_header.get('generated_utc')} by `{anchor_header.get('generated_by')}`."
    )
    lines.append("")
    lines.append("## Arms and coverage")
    lines.append("")
    prompt_columns = ["prompts found", "raw generations found", "prompts from"] if with_prompts else []
    lines.extend(
        _table(
            [
                "arm", "per-item file", "rows in file", "anchor items found",
                *prompt_columns, *arm_header_fields,
            ],
            [
                [
                    cell(a.label, **fmt),
                    "`" + cell(display_path(a.path), **fmt) + "`",
                    cell(a.header.get("item_count", MISSING), **fmt),
                    f"{coverage[a.label]}/{total}",
                    *(
                        [
                            f"{prompt_counts[a.label][0]}/{total}",
                            f"{prompt_counts[a.label][1]}/{total}",
                            (
                                "`" + display_path(a.prompts_dir) + "` "
                                f"({a.prompts_provenance()})"
                                if a.prompts_dir is not None
                                else cell(MISSING, **fmt)
                            ),
                        ]
                        if with_prompts
                        else []
                    ),
                    *(cell(header_value(a, field), **fmt) for field in arm_header_fields),
                ]
                for a in arms
            ],
        )
    )
    lines.append("")
    lines.append(
        "The columns after the counts are the arms' own per-item headers, copied verbatim. "
        "They are here so the scoring settings a sheet compares under are visible on the "
        "same page as the comparison."
    )
    lines.append("")

    # The same shape of banner the question / gold checks emit further down: a difference
    # that would make two columns non-comparable is stated, never quietly averaged over.
    for field in comparability_fields:
        values = {a.label: header_value(a, field) for a in arms}
        distinct = {json.dumps(v, sort_keys=True, default=str) for v in values.values()}
        if len(distinct) > 1:
            lines.append(
                f"> **The arms were scored under different `{field}`.** "
                + "; ".join(f"{label}: {cell(value, **fmt)}" for label, value in values.items())
                + ". Any column-to-column reading of a metric that depends on it is not a "
                "comparison until that is explained."
            )
            lines.append("")
    for arm in arms:
        missing = [qid for qid in anchor_ids if qid not in arm.rows]
        if missing:
            lines.append(
                f"- **{arm.label} is missing {len(missing)} anchor item(s)**: "
                + ", ".join(f"`{m}`" for m in missing)
                + ". Those cells are empty below and must not be read as a result."
            )
    if any(coverage[a.label] < total for a in arms):
        lines.append("")

    for index, qid in enumerate(anchor_ids, start=1):
        rows = [arm.rows.get(qid) for arm in arms]
        present = [r for r in rows if r is not None]
        lines.append(f"## {index}. `{qid}` — hop {hop_of.get(qid, '?')}")
        lines.append("")

        questions = {str(r.get("question", "")).strip() for r in present}
        if not present:
            lines.append("_No arm carries this item._")
            lines.append("")
            continue
        if len(questions) > 1:
            lines.append(
                "> **The arms disagree on the question text for this id.** Each arm's own "
                "text is shown under its column heading; do not compare these rows until "
                "that is explained."
            )
            lines.append("")
            for arm, row in zip(arms, rows):
                if row is not None:
                    lines.append(f"- **{arm.label}:** {str(row.get('question', '')).strip()}")
            lines.append("")
        else:
            lines.append(f"**Question.** {questions.pop()}")
            lines.append("")

        gold_variants = {tuple(_steps(r, "gold_steps")) for r in present}
        if len(gold_variants) > 1:
            lines.append(
                "> **The arms disagree on the gold steps for this id** — they were scored "
                "against different gold. The per-arm gold is shown in the steps table."
            )
            lines.append("")
            step_header = ["#", "gold (per arm)"] + labels
        else:
            gold = list(gold_variants.pop())
            lines.append("**Gold steps.**")
            lines.append("")
            if gold:
                lines.extend(
                    f"{i}. {s}" for i, s in enumerate(gold, start=1)
                )
            else:
                lines.append("_(none recorded)_")
            lines.append("")
            step_header = ["#"] + labels

        per_arm_steps = [_steps(r, "pred_steps") for r in rows]
        per_arm_gold = [_steps(r, "gold_steps") for r in rows]
        depth = max([len(s) for s in per_arm_steps] + [len(g) for g in per_arm_gold] + [0])
        lines.append("**Predicted steps.**")
        lines.append("")
        table_rows: list[list[str]] = []
        for i in range(depth):
            row_cells = [str(i + 1)]
            if len(step_header) == len(labels) + 2:
                row_cells.append(
                    " / ".join(
                        cell(g[i] if i < len(g) else MISSING, **fmt)
                        for g in per_arm_gold
                    )
                )
            row_cells.extend(
                cell(s[i] if i < len(s) else MISSING, **fmt)
                for s in per_arm_steps
            )
            table_rows.append(row_cells)
        if table_rows:
            lines.extend(_table(step_header, table_rows))
        else:
            lines.append("_(no predicted steps in any arm)_")
        lines.append("")

        lines.append("**Per-item metrics, as recorded by the evaluator.**")
        lines.append("")
        metric_rows = [
            [f"`{field}`"]
            + [
                cell(row.get(field, MISSING) if row is not None else MISSING, **fmt)
                for row in rows
            ]
            for field in fields
        ]
        if metric_rows:
            lines.extend(_table(["metric"] + labels, metric_rows))
        else:
            lines.append("_(the per-item rows carry no metric fields)_")
        lines.append("")

        if with_prompts:
            lines.append("**Prompts and raw generations.**")
            lines.append("")
            for arm in arms:
                lines.extend(render_prompt_section(arm, qid))

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    paths_cfg = load_paths(require(cfg, "paths_config"))

    # First, before a single file is read: a sheet that could be committed is not written
    # at all. Cheap, and it fails at the point the mistake was made.
    out = args.out or runs_path(paths_cfg, require(cfg, "out_subdir"), "anchor_set.md")
    assert_out_path_never_enters_git(out)

    ids_file = args.anchor or resolve_path(require(cfg, "ids_file"), REPO_ROOT)
    anchor_ids, hop_of, anchor_header = load_anchor_ids(ids_file)

    prompts_overrides: dict[str, Path] = {}
    for spec in args.prompts_dir:
        label, directory = parse_prompts_dir_spec(spec)
        if label in prompts_overrides:
            raise SystemExit(f"--prompts-dir given twice for label {label!r}")
        if not directory.is_dir():
            raise SystemExit(f"--prompts-dir {label}: not a directory: {directory}")
        prompts_overrides[label] = directory

    arms: list[Arm] = []
    for spec in args.arm:
        label, path, run_dir = parse_arm_spec(spec)
        if any(a.label == label for a in arms):
            raise SystemExit(f"--arm label {label!r} is used twice; column labels must differ")
        rows, header = load_per_item(path)
        results = None
        if run_dir is not None:
            if not run_dir.is_dir():
                raise SystemExit(f"--arm {label}: run directory not found: {run_dir}")
            results = load_results(run_dir / "results.json")
        # An explicit --prompts-dir wins over the run's own prompts_log: that is the whole
        # point of the recovery path, where the run's own log has no anchor files.
        prompts_verified = None
        if label in prompts_overrides:
            prompts_dir: Path | None = prompts_overrides.pop(label)
            prompts_source = PROMPTS_REGENERATED
            if run_dir is not None:
                assert_regeneration_matches_run(label, prompts_dir, run_dir)
                prompts_verified = PROMPTS_VERIFIED
            else:
                # Nothing to check it against. Rendered, but never as though it had been.
                prompts_verified = PROMPTS_UNVERIFIED
        elif run_dir is not None:
            prompts_dir = run_dir / "prompts_log"
            prompts_source = PROMPTS_FROM_RUN
        else:
            prompts_dir, prompts_source = None, None
        arms.append(
            Arm(
                label, path, rows, header,
                run_dir=run_dir, results=results,
                prompts_dir=prompts_dir, prompts_dir_source=prompts_source,
                prompts_verified=prompts_verified,
            )
        )

    if prompts_overrides:
        raise SystemExit(
            "--prompts-dir names arms that were not given with --arm: "
            + ", ".join(sorted(prompts_overrides))
            + ". A prompts directory that matches no arm would silently render nothing."
        )

    total = len(anchor_ids)
    coverage = {a.label: sum(1 for qid in anchor_ids if qid in a.rows) for a in arms}
    for arm in arms:
        line = f"[anchor-render] {arm.label}: {coverage[arm.label]}/{total} anchor items in {arm.path}"
        if arm.prompts_dir is not None or arm.run_dir is not None:
            prompts, raws = prompt_coverage(arm, anchor_ids)
            line += f"; prompts {prompts}/{total} ({arm.prompts_provenance()}), raw {raws}/{total}"
        print(line)

    short = [label for label, n in coverage.items() if n < total]
    if short and not args.allow_missing:
        raise SystemExit(
            "these arms do not cover the full anchor set: "
            + ", ".join(f"{label} ({coverage[label]}/{total})" for label in short)
            + "\nAn arm evaluated on the ADR 0007 pinned 600 contains all of them, so a gap "
            "means the arm was scored on something else. Pass --allow-missing to render it "
            "anyway; the sheet will state the gap."
        )

    sheet = render(
        arms,
        anchor_ids,
        hop_of,
        anchor_header,
        structural=list(require(cfg, "structural_fields")),
        arm_header_fields=list(require(cfg, "arm_header_fields")),
        comparability_fields=list(require(cfg, "comparability_fields")),
        float_format=require(cfg, "float_format"),
        empty=require(cfg, "empty_cell"),
        null=require(cfg, "null_cell"),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(sheet, encoding="utf-8")
    print(f"[anchor-render] wrote {out} ({total} items, {len(arms)} arm(s))")


if __name__ == "__main__":
    main()
