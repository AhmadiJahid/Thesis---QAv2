"""Pure helpers for exp-038, the weight-space repair track (ADR 0042).

Everything here is numpy-only (or, for the two projection functions, works unchanged on any
array type with ``@``, ``reshape`` and broadcasting - numpy arrays and torch tensors alike),
so it is unit-tested on CPU (``tests/test_weight_space.py``). The torch side - loading the
model, capturing activations, hooks, writing edited weights - lives in
``components/decomposer/weight_space_repair.py`` and calls into this module for every piece
of arithmetic that decides a number.

Vocabulary (ADR 0042):

- A **step boundary** is the generated token that ends a step line: the position whose
  next-token prediction is either a newline (the model goes on to another step) or EOS (it
  ends the plan). Boundary ``j`` ends step ``j`` (1-based).
- Each boundary gets one of four labels from the gold hop count ``k`` and whether the model
  stopped there (:func:`label_boundary`). ``premature_stop`` is the defect under repair.
- A **direction** is a difference of class means of the residual stream at boundaries,
  matched on the boundary index ``j`` so that "where in the plan" cannot masquerade as
  "stop vs continue" (:func:`matched_diff_of_means`).
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

PREMATURE_STOP = "premature_stop"
CONTINUE_NEEDED = "continue_needed"
CORRECT_STOP = "correct_stop"
LATE_CONTINUE = "late_continue"
LATE_STOP = "late_stop"
LABELS = (PREMATURE_STOP, CONTINUE_NEEDED, CORRECT_STOP, LATE_CONTINUE, LATE_STOP)

_ID_HOP_RX = re.compile(r"^(?P<h>\d)hop")


# ------------------------------------------------------------------ boundary labelling


class MixedNewlineToken(ValueError):
    """A generated token holds both step text and a newline; its item is skipped, counted."""


def label_boundary(j: int, gold_k: int, stopped: bool) -> str:
    """Label step boundary ``j`` (1-based) of an item whose gold plan has ``gold_k`` steps.

    ``stopped`` is whether the model emitted EOS right after this boundary.
    """
    if j < 1 or gold_k < 1:
        raise ValueError(f"boundary index and gold hop count must be >= 1 (got j={j}, k={gold_k})")
    if j < gold_k:
        return PREMATURE_STOP if stopped else CONTINUE_NEEDED
    if not stopped:
        return LATE_CONTINUE
    return CORRECT_STOP if j == gold_k else LATE_STOP


def step_boundaries(token_texts: Sequence[str], ended_with_eos: bool) -> list[int]:
    """Positions (indices into ``token_texts``) of the last token of each non-empty step.

    ``token_texts`` are the decoded generated tokens, one string per token, WITHOUT the EOS
    token. A step ends at a token followed by a token whose text starts with a newline, or,
    for the final step, at the last token when the generation ended with EOS. Runs of blank
    lines do not create empty steps; a token that is itself only whitespace/newlines is never
    a boundary. A trailing step that was not closed by EOS (a truncated generation) is not
    given a boundary, because the model never made a stop/continue decision there.
    """
    out: list[int] = []
    n = len(token_texts)
    for p, text in enumerate(token_texts):
        if "\n" in text and text.strip():
            # A merged token like ".\n" carries both the end of a step and the newline, so
            # no single position holds the stop/continue decision. Refuse; the caller skips
            # the item and counts it (Mistral-7B-v0.3 byte-falls-back newlines, so this is
            # expected to be zero - the count is recorded to show it).
            raise MixedNewlineToken(f"token {p} {text!r} mixes step text and a newline")
        if not text.strip():
            continue
        # Look past whitespace-only tokens that hold no newline (a stray space).
        q = p + 1
        while q < n and not token_texts[q].strip() and "\n" not in token_texts[q]:
            q += 1
        if q < n and "\n" in token_texts[q]:
            out.append(p)
        elif q >= n and ended_with_eos:
            out.append(p)
    return out


def boundary_records(
    n_boundaries: int, gold_k: int, ended_with_eos: bool
) -> list[dict[str, Any]]:
    """One ``{j, label}`` per boundary of an item that produced ``n_boundaries`` steps."""
    recs = []
    for j in range(1, n_boundaries + 1):
        stopped = ended_with_eos and j == n_boundaries
        recs.append({"j": j, "label": label_boundary(j, gold_k, stopped)})
    return recs


# ------------------------------------------------------------------ step-count summary


def hop_from_id(qid: str) -> int:
    m = _ID_HOP_RX.match(str(qid))
    if not m:
        raise ValueError(f"cannot parse a hop depth from id {qid!r}")
    return int(m.group("h"))


def step_histogram(counts: Iterable[int]) -> dict[str, int]:
    """``{"<steps>": n_items}``, keys sorted numerically (JSON needs string keys)."""
    hist: dict[int, int] = {}
    for c in counts:
        hist[int(c)] = hist.get(int(c), 0) + 1
    return {str(k): hist[k] for k in sorted(hist)}


def step_count_summary(
    pred_counts: Sequence[int], gold_counts: Sequence[int]
) -> dict[str, Any]:
    """hop-count EM / under / over rates and mean signed error, overall and per gold hop.

    Same definitions as the evaluator's report-card terms (ADR 0029): exact when the number
    of predicted step lines equals the gold hop count, under when fewer, over when more.
    Used only for the train-side sweep; the pinned-600 numbers come from the evaluator.
    """
    if len(pred_counts) != len(gold_counts):
        raise ValueError("pred_counts and gold_counts differ in length")

    def block(pairs: list[tuple[int, int]]) -> dict[str, Any]:
        n = len(pairs)
        if n == 0:
            return {"n": 0}
        em = sum(p == g for p, g in pairs)
        under = sum(p < g for p, g in pairs)
        over = sum(p > g for p, g in pairs)
        return {
            "n": n,
            "hop_count_exact_match_rate": em / n,
            "under_decomposition_rate": under / n,
            "over_decomposition_rate": over / n,
            "mean_pred_steps": sum(p for p, _ in pairs) / n,
            "mean_signed_error": sum(p - g for p, g in pairs) / n,
            "counts": {"exact": em, "under": under, "over": over},
            "pred_step_hist": step_histogram(p for p, _ in pairs),
        }

    pairs = list(zip((int(x) for x in pred_counts), (int(x) for x in gold_counts)))
    out = {"overall": block(pairs)}
    for h in sorted({g for _, g in pairs}):
        out[f"hop{h}"] = block([(p, g) for p, g in pairs if g == h])
    return out


# ------------------------------------------------------------------ directions


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n == 0.0:
        raise ValueError("cannot normalise a zero vector")
    return v / n


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(unit(a), unit(b)))


def matched_diff_of_means(
    acts: np.ndarray,
    labels: Sequence[str],
    strata: Sequence[int],
    pos: str,
    neg: str,
    *,
    min_per_cell: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Stratum-matched difference of means ``mean(pos) - mean(neg)``.

    For every stratum (boundary index ``j``) holding at least ``min_per_cell`` rows of BOTH
    classes, take the within-stratum difference of class means; combine the strata weighted
    by ``min(n_pos, n_neg)``. Strata with only one class contribute nothing, so a class that
    lives only at late boundaries cannot turn "late in the plan" into the direction.

    ``acts`` is ``(n, d)``. Returns the (unnormalised) direction and a record of the cells.
    """
    acts = np.asarray(acts, dtype=np.float64)
    labels = list(labels)
    strata = [int(s) for s in strata]
    if acts.ndim != 2 or acts.shape[0] != len(labels) or len(labels) != len(strata):
        raise ValueError("acts must be (n, d) with one label and one stratum per row")
    cells: list[dict[str, Any]] = []
    total = np.zeros(acts.shape[1], dtype=np.float64)
    wsum = 0.0
    for s in sorted(set(strata)):
        ip = [i for i, (l, t) in enumerate(zip(labels, strata)) if t == s and l == pos]
        ineg = [i for i, (l, t) in enumerate(zip(labels, strata)) if t == s and l == neg]
        used = len(ip) >= min_per_cell and len(ineg) >= min_per_cell
        cells.append({"stratum": s, "n_pos": len(ip), "n_neg": len(ineg), "used": used})
        if not used:
            continue
        w = float(min(len(ip), len(ineg)))
        total += w * (acts[ip].mean(axis=0) - acts[ineg].mean(axis=0))
        wsum += w
    if wsum == 0.0:
        raise ValueError(f"no stratum holds both {pos!r} and {neg!r} (min_per_cell={min_per_cell})")
    return total / wsum, {"pos": pos, "neg": neg, "cells": cells, "weight_total": wsum}


def auroc(scores_pos: Sequence[float], scores_neg: Sequence[float]) -> float:
    """P(score_pos > score_neg) + 0.5 P(tie): the Mann-Whitney AUROC, exact, no sklearn."""
    a = np.asarray(scores_pos, dtype=np.float64)
    b = np.asarray(scores_neg, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        raise ValueError("AUROC needs at least one score in each class")
    allv = np.concatenate([a, b])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty_like(allv)
    sorted_v = allv[order]
    i = 0
    while i < len(sorted_v):
        j = i
        while j + 1 < len(sorted_v) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    r_pos = ranks[: a.size].sum()
    return float((r_pos - a.size * (a.size + 1) / 2.0) / (a.size * b.size))


def random_unit_orthogonal(r: np.ndarray, seed: int) -> np.ndarray:
    """A seeded random unit vector orthogonal to ``r`` (the random-direction control).

    Unit norm, like the ablated direction, so the control removes the same amount of
    "direction" from every writer; orthogonal so it cannot partly be the real one.
    """
    r = unit(r)
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(r.shape[0])
    v -= np.dot(v, r) * r
    return unit(v)


# ------------------------------------------------------------------ weight orthogonalisation


def project_out_of_outputs(W: Any, r: Any, alpha: float = 1.0) -> Any:
    """``W - alpha * r r^T W`` for a writer ``W`` of shape ``(d_model, d_in)``.

    ``nn.Linear`` computes ``y = W x``, so this removes ``alpha`` of the ``r`` component from
    everything the matrix writes into the residual stream (``o_proj``, ``down_proj``).
    ``r`` must be unit norm. Works on numpy arrays and torch tensors alike.
    """
    r_col = r.reshape(-1, 1)
    return W - alpha * (r_col * (r.reshape(1, -1) @ W))


def project_out_of_rows(E: Any, r: Any, alpha: float = 1.0) -> Any:
    """``E - alpha * (E r) r^T`` for a matrix whose ROWS are residual vectors (embeddings)."""
    return E - alpha * ((E @ r.reshape(-1, 1)) * r.reshape(1, -1))


# ------------------------------------------------------------------ splits and leakage


def assert_disjoint(a: Iterable[str], b: Iterable[str], label: str) -> dict[str, Any]:
    sa, sb = set(a), set(b)
    both = sorted(sa & sb)
    if both:
        raise SystemExit(
            f"[weight_space] LEAKAGE REFUSED ({label}): {len(both)} id(s) in common, e.g. {both[:5]}"
        )
    return {"check": label, "n_a": len(sa), "n_b": len(sb), "overlap": 0}


def normalise_question(text: str) -> str:
    return " ".join(re.sub(r"[^0-9a-z ]+", " ", str(text).lower()).split())


def split_train_ids(
    candidates_by_hop: Mapping[int, Sequence[str]],
    sizes: Mapping[str, Mapping[int, int]],
    seed: int,
) -> dict[str, list[str]]:
    """Seeded, hop-stratified, mutually disjoint draws of ids, one list per named split.

    ``candidates_by_hop`` are the eligible ids (already excluding anything that must be
    excluded); they are sorted before shuffling, so the result depends only on the set of
    ids and the seed, never on file order. ``sizes`` maps split name -> {hop: n}. Splits are
    drawn in sorted-name order from one shuffled list per hop.
    """
    out: dict[str, list[str]] = {name: [] for name in sizes}
    for hop in sorted(candidates_by_hop):
        pool = sorted(set(candidates_by_hop[hop]))
        rng = random.Random(f"{seed}:{hop}")
        rng.shuffle(pool)
        need = sum(int(sizes[name].get(hop, 0)) for name in sizes)
        if need > len(pool):
            raise SystemExit(f"[weight_space] hop {hop}: need {need} ids, only {len(pool)} eligible")
        cursor = 0
        for name in sorted(sizes):
            n = int(sizes[name].get(hop, 0))
            out[name].extend(pool[cursor : cursor + n])
            cursor += n
    return {k: sorted(v) for k, v in out.items()}


def ids_sha256(ids: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


# ------------------------------------------------------------------ edit specs

EDIT_KINDS = ("none", "orthogonalize", "final_norm_scale", "token_logit_bias")
WRITER_MATRICES = ("o_proj", "down_proj", "embed_tokens")


def validate_edit_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Check an edit spec (the JSON ``run_decomposer.py --weight-edit`` reads) and normalise it.

    Kinds:
      none              - merge the adapter, change nothing (the paired baseline)
      orthogonalize     - ``direction_file`` (.npy, d_model), ``alpha`` in (0, 1],
                          ``layers`` (list of decoder-layer indices or "all"),
                          ``matrices`` subset of o_proj/down_proj/embed_tokens
      final_norm_scale  - ``scale`` > 0 on the final RMSNorm gain (the norm-only sanity
                          check: under greedy decoding it cannot change an argmax)
      token_logit_bias  - an ADDITIVE ``bias`` (finite, non-zero; negative lowers the token)
                          on the logits of ``token_ids`` ('eos' or ints), written as an
                          lm_head bias vector that is zero everywhere else - the length-bias
                          control. Additive rather than a row scale: scaling a row multiplies
                          a logit whose sign varies with the context, so it can raise it or
                          lower it.
    """
    if not isinstance(spec, Mapping):
        raise SystemExit("[weight_space] edit spec must be a JSON object")
    kind = spec.get("kind")
    if kind not in EDIT_KINDS:
        raise SystemExit(f"[weight_space] edit kind {kind!r} not in {EDIT_KINDS}")
    out: dict[str, Any] = {"kind": kind, "label": str(spec.get("label") or kind)}
    if kind == "orthogonalize":
        for key in ("direction_file", "alpha", "layers", "matrices"):
            if key not in spec:
                raise SystemExit(f"[weight_space] orthogonalize spec needs {key!r}")
        alpha = float(spec["alpha"])
        if not 0.0 < alpha <= 1.0:
            raise SystemExit(f"[weight_space] alpha must be in (0, 1], got {alpha}")
        layers = spec["layers"]
        if layers != "all":
            if not isinstance(layers, list) or not layers or not all(
                isinstance(x, int) and x >= 0 for x in layers
            ):
                raise SystemExit("[weight_space] layers must be 'all' or a non-empty list of ints >= 0")
            layers = sorted(set(layers))
        mats = list(spec["matrices"])
        bad = [m for m in mats if m not in WRITER_MATRICES]
        if bad or not mats:
            raise SystemExit(f"[weight_space] matrices must be a non-empty subset of {WRITER_MATRICES}")
        out.update(
            direction_file=str(spec["direction_file"]),
            alpha=alpha,
            layers=layers,
            matrices=sorted(set(mats)),
        )
        if "direction_sha256" in spec:
            out["direction_sha256"] = str(spec["direction_sha256"])
    elif kind == "final_norm_scale":
        scale = float(spec.get("scale", 0))
        if scale <= 0:
            raise SystemExit(f"[weight_space] {kind} needs scale > 0")
        out["scale"] = scale
    elif kind == "token_logit_bias":
        bias = float(spec.get("bias", 0.0))
        if bias == 0.0 or not np.isfinite(bias):
            raise SystemExit("[weight_space] token_logit_bias needs a finite, non-zero bias")
        out["bias"] = bias
        ids = spec.get("token_ids")
        if ids == "eos":
            out["token_ids"] = "eos"
        elif isinstance(ids, list) and ids and all(isinstance(x, int) for x in ids):
            out["token_ids"] = sorted(set(ids))
        else:
            raise SystemExit("[weight_space] token_logit_bias needs token_ids: 'eos' or a list of ints")
    for key in ("chosen_by", "choice_note", "source_candidate"):
        if key in spec:
            out[key] = spec[key]
    return out


# ------------------------------------------------------------------ multiple comparisons


def holm(ps: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in input order (ADR 0036)."""
    m = len(ps)
    order = sorted(range(m), key=lambda k: ps[k])
    adj = [0.0] * m
    running = 0.0
    for rank, k in enumerate(order):
        running = max(running, min(1.0, (m - rank) * float(ps[k])))
        adj[k] = running
    return adj


# ------------------------------------------------------------------ candidate selection


def select_candidate(
    summaries: Mapping[str, Mapping[str, Any]],
    *,
    baseline: str,
    edit_names: Sequence[str],
    control_names: Sequence[str],
    max_over_increase: float,
    max_hop2_em_drop: float,
) -> dict[str, Any]:
    """The pre-registered default choice rule of ADR 0042 (Claude may overrule it in writing).

    Among ``edit_names`` whose val over-decomposition rate rises by at most
    ``max_over_increase`` over the baseline and whose 2-hop hop-count EM falls by at most
    ``max_hop2_em_drop``, pick the one with the highest overall hop-count EM (ties: lower
    over-decomposition, then name). Report whether it beats every control in
    ``control_names`` on overall hop-count EM.
    """
    base = summaries[baseline]
    b_over = base["overall"]["over_decomposition_rate"]
    b_h2 = base["hop2"]["hop_count_exact_match_rate"]
    admissible, rejected = [], {}
    for name in edit_names:
        s = summaries[name]
        reasons = []
        if s["overall"]["over_decomposition_rate"] - b_over > max_over_increase + 1e-12:
            reasons.append("over_decomposition rise above limit")
        if b_h2 - s["hop2"]["hop_count_exact_match_rate"] > max_hop2_em_drop + 1e-12:
            reasons.append("2-hop hop-count EM drop above limit")
        if reasons:
            rejected[name] = reasons
        else:
            admissible.append(name)
    if not admissible:
        return {"chosen": None, "admissible": [], "rejected": rejected}
    chosen = sorted(
        admissible,
        key=lambda n: (
            -summaries[n]["overall"]["hop_count_exact_match_rate"],
            summaries[n]["overall"]["over_decomposition_rate"],
            n,
        ),
    )[0]
    em = summaries[chosen]["overall"]["hop_count_exact_match_rate"]
    beats = {
        c: em > summaries[c]["overall"]["hop_count_exact_match_rate"] for c in control_names
    }
    return {
        "chosen": chosen,
        "admissible": admissible,
        "rejected": rejected,
        "beats_controls_on_val_hop_em": beats,
        "beats_all_controls": all(beats.values()) if beats else None,
    }


def closest_by_mean_steps(
    summaries: Mapping[str, Mapping[str, Any]], target: str, candidates: Sequence[str]
) -> tuple[str, float]:
    """The candidate whose overall mean predicted step count is closest to ``target``'s, and
    the signed gap (candidate minus target) in mean steps.

    Picks the length-matched EOS-bias control: a control that lengthens plans as much as the
    edit does, so "the edit just makes plans longer" is tested at equal length.
    """
    if not candidates:
        raise ValueError("no length-control candidates")
    t = summaries[target]["overall"]["mean_pred_steps"]
    best = sorted(candidates, key=lambda c: (abs(summaries[c]["overall"]["mean_pred_steps"] - t), c))[0]
    return best, summaries[best]["overall"]["mean_pred_steps"] - t


def refine_biases(points: Sequence[tuple[float, float]], target: float, n_new: int) -> list[float]:
    """New EOS-bias values to try when no swept bias matched the edit's length within tolerance.

    ``points`` are (bias, mean predicted steps) already swept. A lower EOS bias is expected to
    lengthen plans, but that is not assumed: the bracket comes from the measured points
    sorted by bias. If ``target`` lies between two neighbouring points, return ``n_new``
    evenly spaced biases strictly inside that bracket. If it lies outside every point, step
    past the end whose mean is nearer the target, by the swept range (or 1.0 with a single
    point). Never returns an already-swept value or 0.
    """
    if not points:
        raise ValueError("refine_biases needs at least one swept point")
    pts = sorted((float(b), float(m)) for b, m in points)
    swept = {round(b, 4) for b, _ in pts}
    for (b0, m0), (b1, m1) in zip(pts, pts[1:]):
        if min(m0, m1) <= target <= max(m0, m1):
            new = [round(b0 + (b1 - b0) * k / (n_new + 1), 4) for k in range(1, n_new + 1)]
            return [b for b in new if b not in swept and b != 0.0]
    lo, hi = pts[0], pts[-1]
    step = (hi[0] - lo[0]) if len(pts) > 1 else 1.0
    if abs(lo[1] - target) <= abs(hi[1] - target):
        new = [round(lo[0] - step * k, 4) for k in range(1, n_new + 1)]
    else:
        new = [round(hi[0] + step * k, 4) for k in range(1, n_new + 1)]
    return [b for b in new if b not in swept and b != 0.0]


# ------------------------------------------------------------------ freezing and done-files


def check_freeze(
    spec_hashes: Mapping[str, str],
    choice_commits: Sequence[str],
    existing: Mapping[str, Any] | None,
) -> str:
    """May the specs be frozen? Returns 'write' or 'unchanged'; refuses (SystemExit) otherwise.

    - ``choice.json`` must have exactly ONE commit: a second commit means the choice was
      revised, and the run must not carry on under a choice that changed.
    - An existing freeze record is never rewritten. The same hashes and the same choice
      commit mean a resumed session ('unchanged'); anything else is refused.
    """
    if len(choice_commits) != 1:
        raise SystemExit(
            f"[weight_space] FREEZE REFUSED: experiments/exp-038/choice.json has "
            f"{len(choice_commits)} commit(s); exactly one is allowed (ADR 0042 item 5)"
        )
    if existing is None:
        return "write"
    if dict(existing.get("files", {})) == dict(spec_hashes) and existing.get("choice_commit") == choice_commits[0]:
        return "unchanged"
    raise SystemExit("[weight_space] FREEZE REFUSED: specs_frozen.json exists with different specs or choice commit")


def keyed_name(name: str, sha: str) -> str:
    """A done-file stem tied to the exact spec or candidate content, so a revised spec cannot
    reuse an earlier output that has the same name."""
    return f"{name}.{sha[:12]}"


def candidate_sha256(cand: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(cand, sort_keys=True).encode("utf-8")).hexdigest()


# ------------------------------------------------------------------ archive extraction


def safe_member_path(root: str, rel: str) -> str:
    """``root/rel`` resolved; refuses absolute names, drive letters, '..' parts and anything
    that resolves outside ``root`` (archive member names are untrusted input)."""
    from pathlib import Path as _Path, PurePosixPath, PureWindowsPath

    if not rel or rel.startswith(("/", "\\")) or PureWindowsPath(rel).drive:
        raise ValueError(f"unsafe archive path {rel!r}")
    parts = PurePosixPath(rel.replace("\\", "/")).parts
    if ".." in parts:
        raise ValueError(f"unsafe archive path {rel!r}")
    base = _Path(root).resolve()
    dest = base.joinpath(*parts).resolve()
    if not dest.is_relative_to(base):
        raise ValueError(f"archive path {rel!r} resolves outside {root}")
    return str(dest)
