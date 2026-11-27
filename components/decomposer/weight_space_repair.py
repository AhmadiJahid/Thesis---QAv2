#!/usr/bin/env python3
"""exp-038 weight-space repair: locate a stop-too-early direction, edit weights, run controls.

ADR 0042. A separate track OUTSIDE the few-shot pool study. Target: the exp-026 pool-only
LoRA fine-tune of Mistral-7B-Instruct-v0.3, merged in fp16. Every number this script uses
comes from ``configs/exp038_weight_space.json``; every piece of arithmetic that decides a
number lives in ``src/weight_space.py`` (CPU unit-tested). This file is the torch side:
loading, generating, capturing activations, writing edited weights. It needs a GPU and is
driven from ``notebooks/exp-038_weight_space_repair_colab.ipynb``.

Leakage guard: the locating and validation items are MuSiQue TRAIN rows, minus the
adapter's own 2,000 training rows, asserted disjoint (by id and by normalised question
text) from the whole MuSiQue dev file, which contains the ADR 0007 pinned 600. Nothing in
this script reads the pinned 600; the pinned-600 evaluation runs through
``run_decomposer.py --weight-edit <spec>`` and the repo evaluator.

Subcommands (each resumable; each writes into ``--out``)::

    split       CPU only. Draw locate / val_select / val_confirm ids; write splits.json.
    locate      GPU. Greedy-generate every split with the unedited merged model; capture the
                residual stream at every step boundary of locate and val_select.
    directions  CPU. Per-layer stratum-matched difference-of-means directions, val AUROC,
                cosines, and a proposed sweep (candidates.json). Writes directions_table.md
                for inspection.
    sweep       GPU. Generate a split under each candidate edit (weight edits applied and
                undone in place; hook diagnostics), write per-candidate summaries.
    choose      CPU. Apply ADR 0042's pre-registered default rule to the val_select sweep;
                write proposed_choice.json (Claude confirms or overrides in choices.md).
    make-specs  CPU. Turn choice.json into the edit specs run_decomposer.py --weight-edit
                reads: edit, random-direction control, EOS-row (length) control, norm-only
                control, and the unedited merged baseline.
    equivalence GPU. Hook-based ablation vs the weight edit on a few locate prompts: the
                largest logit difference, to show the weight edit does what the hook does.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_HERE))

import weight_space as ws  # noqa: E402
from model_size import assert_within_ceiling, load_limits  # noqa: E402
from run_config import load_config, load_paths, require, resolve_path  # noqa: E402
from seeding import set_global_seed  # noqa: E402
from step_lines import step_line_count, with_scaffold_truncate_markers  # noqa: E402

DEFAULT_CONFIG = "exp038_weight_space.json"


# ============================================================================ config


def _load(config: str) -> tuple[dict, dict]:
    cfg = load_config(config)
    paths = load_paths(require(cfg, "paths_config"))
    return cfg, paths


def _data_path(paths: dict, key: str) -> Path:
    return resolve_path(require(paths, "datasets." + key), _data_root(paths))


def _data_root(paths: dict) -> Path:
    return resolve_path(require(paths, "data_root"), _REPO_ROOT)


def _runs_root(paths: dict) -> Path:
    return resolve_path(require(paths, "runs_root"), _REPO_ROOT)


def _jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _rd():
    import run_decomposer  # components/decomposer, on sys.path above

    return run_decomposer


# ============================================================================ split (CPU)


def cmd_split(cfg: dict, paths: dict, out: Path) -> dict:
    """Draw the train-side splits and prove they cannot touch the evaluation set."""
    data = require(cfg, "data")
    seed = int(require(cfg, "seed"))
    train = _jsonl(_data_path(paths, require(data, "train_key")))
    dev = _jsonl(_data_path(paths, require(data, "dev_key")))
    pinned_tmpl = require(paths, "datasets." + require(data, "pinned_eval_template_key"))
    pinned: list[dict] = []
    for h in require(data, "hops"):
        pinned += _jsonl(resolve_path(pinned_tmpl.format(hop=h), _data_root(paths)))
    pool_path = resolve_path(require(cfg, "adapter_training_pool_path"), _runs_root(paths))
    pool = _jsonl(pool_path)
    pool_sha = _rd().sha256_file(pool_path)
    if pool_sha != require(cfg, "adapter_training_pool_sha256"):
        raise SystemExit(f"[exp-038] adapter training pool sha256 {pool_sha} != config's")

    dev_ids = {r["id"] for r in dev}
    dev_q = {ws.normalise_question(r["question"]) for r in dev}
    pinned_ids = {r["id"] for r in pinned}
    if len(pinned_ids) != 600 or not pinned_ids <= dev_ids:
        raise SystemExit(f"[exp-038] pinned set: {len(pinned_ids)} ids, subset of dev: {pinned_ids <= dev_ids}")
    pool_ids = {r["id"] for r in pool}

    eligible: dict[int, list[str]] = {}
    excluded = {"adapter_training_rows": 0, "dev_question_text": 0}
    for r in train:
        if r["id"] in pool_ids:
            excluded["adapter_training_rows"] += 1
            continue
        if ws.normalise_question(r["question"]) in dev_q:
            excluded["dev_question_text"] += 1
            continue
        h = ws.hop_from_id(r["id"])
        if len(r["question_decomposition"]) != h:
            raise SystemExit(f"[exp-038] {r['id']}: id hop {h} != {len(r['question_decomposition'])} steps")
        eligible.setdefault(h, []).append(r["id"])
    sizes = {name: {int(h): int(n) for h, n in hops.items()} for name, hops in require(data, "splits").items()}
    splits = ws.split_train_ids(eligible, sizes, seed)

    checks = [ws.assert_disjoint(dev_ids, pool_ids, "dev vs adapter training pool")]
    all_split: set[str] = set()
    for name, ids in splits.items():
        checks.append(ws.assert_disjoint(ids, dev_ids, f"{name} vs MuSiQue dev (ids)"))
        checks.append(ws.assert_disjoint(ids, pinned_ids, f"{name} vs pinned 600 (ids)"))
        checks.append(ws.assert_disjoint(ids, pool_ids, f"{name} vs adapter training pool"))
        checks.append(ws.assert_disjoint(ids, all_split, f"{name} vs earlier splits"))
        all_split |= set(ids)
    by_id = {r["id"]: r for r in train}
    for name, ids in splits.items():
        qs = {ws.normalise_question(by_id[i]["question"]) for i in ids}
        checks.append(ws.assert_disjoint(qs, dev_q, f"{name} vs MuSiQue dev (normalised question text)"))

    record = {
        "seed": seed,
        "train_rows": len(train),
        "excluded": excluded,
        "eligible_per_hop": {str(h): len(v) for h, v in sorted(eligible.items())},
        "splits": {name: {"n": len(ids), "ids_sha256": ws.ids_sha256(ids)} for name, ids in splits.items()},
        "leakage_checks": checks,
        "adapter_training_pool_sha256": pool_sha,
    }
    _write_json(out / "splits.json", {"record": record, "ids": splits})
    print(json.dumps(record, indent=1))
    return record


def _split_rows(cfg: dict, paths: dict, out: Path) -> dict[str, list[dict]]:
    payload = json.loads((out / "splits.json").read_text(encoding="utf-8"))
    train = {r["id"]: r for r in _jsonl(_data_path(paths, require(cfg, "data.train_key")))}
    return {
        name: [
            {"id": i, "question": train[i]["question"], "gold_k": len(train[i]["question_decomposition"])}
            for i in ids
        ]
        for name, ids in payload["ids"].items()
    }


# ============================================================================ model


class Target:
    """The merged exp-026 model plus everything needed to prompt and parse it like the runner."""

    def __init__(self, cfg: dict, paths: dict) -> None:
        import torch

        rd = _rd()
        self.torch = torch
        dcfg = load_config(require(cfg, "decomposer_config"))
        folder = _REPO_ROOT / "components" / "decomposer" / "models" / require(cfg, "model_folder")
        mcfg = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        self.model_id = require(mcfg, "model_id")
        prompt_path = folder / require(mcfg, "unguided_prompt_file")
        self.template = prompt_path.read_text(encoding="utf-8")
        self.placeholder = require(dcfg, "unguided_hop_placeholder")
        self.post_cfg = with_scaffold_truncate_markers(
            dict(require(mcfg, "post_process")), list(require(dcfg, "step_parser.scaffold_truncate_at"))
        )
        adapter = resolve_path(require(cfg, "adapter_path"), _runs_root(paths))
        prov = json.loads((adapter / "training_provenance.json").read_text(encoding="utf-8"))
        prompt_sha = rd.sha256_file(prompt_path)
        if prompt_sha != prov["prompt"]["prompt_sha256"]:
            raise SystemExit(f"[exp-038] prompt sha256 {prompt_sha} != adapter's training prompt")
        self.tokenizer, model = rd.load_model(self.model_id, require(mcfg, "loader"), "cuda", require(cfg, "quantization"))
        model, base_rec = rd.attach_adapter(model, adapter, model_id=self.model_id)
        self.model = model.merge_and_unload()
        self.model.eval()
        self.size_record = assert_within_ceiling(
            self.model,
            component="decomposer",
            model_id=f"{self.model_id}+adapter(merged)",
            limits=load_limits(require(cfg, "model_limits_config")),
        )
        self.record = {
            "model_id": self.model_id,
            "adapter": str(adapter),
            "adapter_base_model": base_rec,
            "prompt_sha256": prompt_sha,
            "merged": True,
            "model_size": self.size_record,
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0),
        }
        self.eos = int(self.tokenizer.eos_token_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def prompt(self, question: str) -> str:
        return _rd().fill_template(
            self.template,
            question=question,
            hop_count=None,
            few_shot_examples="",
            unguided_hop_placeholder=self.placeholder,
        )

    @property
    def layers(self):
        return self.model.model.layers

    def generate(self, questions: list[str], gen_cfg: dict) -> list[dict]:
        """Greedy batched generation with left padding (train-side only, ADR 0042)."""
        torch = self.torch
        tok = self.tokenizer
        tok.padding_side = "left"
        out: list[dict] = []
        bs = int(require(gen_cfg, "batch_size"))
        max_new = int(require(gen_cfg, "max_new_tokens"))
        for i in range(0, len(questions), bs):
            prompts = [self.prompt(q) for q in questions[i : i + bs]]
            enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                gen = self.model.generate(
                    **enc,
                    max_new_tokens=max_new,
                    do_sample=bool(require(gen_cfg, "do_sample")),
                    temperature=None,
                    top_p=None,
                    pad_token_id=tok.pad_token_id,
                    eos_token_id=self.eos,
                )
            new = gen[:, enc["input_ids"].shape[1] :].tolist()
            for p, ids in zip(prompts, new):
                ended = self.eos in ids
                ids = ids[: ids.index(self.eos)] if ended else ids
                raw = tok.decode(ids, skip_special_tokens=True)
                dec = _rd().post_process(raw, self.post_cfg).strip()
                out.append(
                    {
                        "prompt_ids": tok(p)["input_ids"],
                        "gen_ids": ids,
                        "ended_with_eos": ended,
                        "hit_max_new_tokens": (not ended) and len(ids) >= max_new,
                        "raw": raw.strip(),
                        "decomposition": dec,
                        "pred_steps": step_line_count(dec),
                    }
                )
        return out

    def boundary_acts(self, prompt_ids: list[int], gen_ids: list[int], positions: list[int]) -> np.ndarray:
        """Residual stream after every decoder layer at the given generated positions.

        Returns ``(len(positions), n_layers, d_model)`` float32. Read by forward hooks on the
        decoder layers rather than ``output_hidden_states`` because the last element of that
        tuple has the final norm applied in HF Mistral, which would make the last layer
        incomparable with the others.
        """
        torch = self.torch
        abs_pos = torch.tensor([len(prompt_ids) + p for p in positions], device="cuda")
        grabbed: list[Any] = [None] * len(self.layers)

        def make_hook(idx: int):
            def hook(_mod, _inp, output):
                hs = output[0] if isinstance(output, tuple) else output
                grabbed[idx] = hs[0, abs_pos, :].float().cpu().numpy()

            return hook

        handles = [layer.register_forward_hook(make_hook(i)) for i, layer in enumerate(self.layers)]
        try:
            ids = torch.tensor([prompt_ids + gen_ids], device="cuda")
            with torch.no_grad():
                self.model(input_ids=ids)
        finally:
            for h in handles:
                h.remove()
        return np.stack(grabbed, axis=1)


# ============================================================================ edits


def _writers(model, layers, matrices):
    """(name, parameter) pairs an orthogonalize edit touches."""
    n_layers = len(model.model.layers)
    idx = range(n_layers) if layers == "all" else layers
    out = []
    for i in idx:
        layer = model.model.layers[i]
        if "o_proj" in matrices:
            out.append((f"layers.{i}.self_attn.o_proj", layer.self_attn.o_proj.weight, "outputs"))
        if "down_proj" in matrices:
            out.append((f"layers.{i}.mlp.down_proj", layer.mlp.down_proj.weight, "outputs"))
    if "embed_tokens" in matrices:
        out.append(("embed_tokens", model.model.embed_tokens.weight, "rows"))
    return out


def edit_targets(model, spec: dict, eos_token_id: int) -> list[tuple[str, Any, Any]]:
    """(name, parameter, row-index-or-None) for every tensor the spec changes (for backup)."""
    kind = spec["kind"]
    if kind == "orthogonalize":
        return [(n, w, None) for n, w, _ in _writers(model, spec["layers"], spec["matrices"])]
    if kind == "final_norm_scale":
        return [("norm", model.model.norm.weight, None)]
    if kind == "token_row_scale":
        ids = [eos_token_id] if spec["token_ids"] == "eos" else spec["token_ids"]
        return [("lm_head", model.lm_head.weight, ids)]
    return []


def apply_edit(model, spec: dict, *, direction: np.ndarray | None, eos_token_id: int) -> dict:
    """Apply a validated spec in place. Computes in float32, writes back in the weight dtype."""
    import torch

    kind = spec["kind"]
    rec: dict[str, Any] = {"kind": kind, "label": spec["label"], "tensors": []}
    with torch.no_grad():
        if kind == "orthogonalize":
            r_np = ws.unit(direction)
            for name, w, mode in _writers(model, spec["layers"], spec["matrices"]):
                r = torch.tensor(r_np, dtype=torch.float32, device=w.device)
                w32 = w.data.float()
                new = (
                    ws.project_out_of_outputs(w32, r, spec["alpha"])
                    if mode == "outputs"
                    else ws.project_out_of_rows(w32, r, spec["alpha"])
                )
                delta = float((new - w32).norm())
                w.data.copy_(new.to(w.dtype))
                rec["tensors"].append({"name": name, "delta_fro": delta, "numel": w.numel()})
        elif kind == "final_norm_scale":
            w = model.model.norm.weight
            w.data.copy_((w.data.float() * spec["scale"]).to(w.dtype))
            rec["tensors"].append({"name": "norm", "numel": w.numel()})
        elif kind == "token_row_scale":
            ids = [eos_token_id] if spec["token_ids"] == "eos" else spec["token_ids"]
            w = model.lm_head.weight
            w.data[ids] = (w.data[ids].float() * spec["scale"]).to(w.dtype)
            rec["tensors"].append({"name": "lm_head", "rows": ids, "numel": len(ids) * w.shape[1]})
            rec["token_ids_resolved"] = ids
    rec["parameters_changed"] = int(sum(t["numel"] for t in rec["tensors"]))
    return rec


class Backup:
    """CPU copies of exactly the tensors (or rows) an edit touches, to undo it exactly."""

    def __init__(self, targets: list[tuple[str, Any, Any]]) -> None:
        self.items = []
        for name, w, rows in targets:
            data = w.data[rows].detach().cpu().clone() if rows is not None else w.data.detach().cpu().clone()
            self.items.append((w, rows, data))

    def restore(self) -> None:
        import torch

        with torch.no_grad():
            for w, rows, data in self.items:
                if rows is not None:
                    w.data[rows] = data.to(w.device)
                else:
                    w.data.copy_(data.to(w.device))


def load_spec(spec_path: str | Path) -> tuple[dict, np.ndarray | None, dict]:
    """Read and validate an edit spec; load and hash its direction file if it has one."""
    spec_path = Path(spec_path)
    raw = json.loads(spec_path.read_text(encoding="utf-8"))
    spec = ws.validate_edit_spec(raw)
    direction = None
    prov: dict[str, Any] = {"spec_path": str(spec_path), "spec_sha256": ws.spec_sha256(raw)}
    if spec["kind"] == "orthogonalize":
        dpath = Path(spec["direction_file"])
        if not dpath.is_absolute():
            dpath = spec_path.parent / dpath
        direction = np.load(dpath).astype(np.float64)
        sha = _rd().sha256_file(dpath)
        if "direction_sha256" in spec and spec["direction_sha256"] != sha:
            raise SystemExit(f"[exp-038] direction file {dpath} sha256 {sha} != spec's {spec['direction_sha256']}")
        prov.update(direction_path=str(dpath), direction_sha256=sha, direction_norm=float(np.linalg.norm(direction)))
    return spec, direction, prov


def apply_edit_spec(model, spec_path: str | Path, *, eos_token_id: int):
    """Entry point for ``run_decomposer.py --weight-edit``: merge any adapter, apply the spec.

    Returns ``(model, record)``. The record goes into the run's metrics and config snapshot.
    """
    spec, direction, prov = load_spec(spec_path)
    merged = False
    if hasattr(model, "merge_and_unload"):
        model = model.merge_and_unload()
        merged = True
    model.eval()
    rec = apply_edit(model, spec, direction=direction, eos_token_id=eos_token_id)
    rec.update(prov)
    rec["adapter_merged_before_edit"] = merged
    rec["spec"] = spec
    print(f"[weight_edit] applied {spec['kind']} ({spec['label']}): {rec['parameters_changed']:,} parameters changed")
    return model, rec


# ============================================================================ locate (GPU)


def _summary_rows(rows: list[dict], gens: list[dict]) -> dict:
    return ws.step_count_summary([g["pred_steps"] for g in gens], [r["gold_k"] for r in rows])


def _write_generations(path: Path, rows: list[dict], gens: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r, g in zip(rows, gens):
            f.write(
                json.dumps(
                    {
                        "id": r["id"],
                        "gold_k": r["gold_k"],
                        "pred_steps": g["pred_steps"],
                        "ended_with_eos": g["ended_with_eos"],
                        "hit_max_new_tokens": g["hit_max_new_tokens"],
                        "decomposition": g["decomposition"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def cmd_locate(cfg: dict, paths: dict, out: Path, target: Target | None = None) -> None:
    set_global_seed(int(require(cfg, "seed")))
    target = target or Target(cfg, paths)
    _write_json(out / "target.json", target.record)
    splits = _split_rows(cfg, paths, out)
    gen_cfg = require(cfg, "generation")
    for name in ("locate", "val_select", "val_confirm"):
        done = out / f"locate_{name}.done.json"
        if done.exists():
            print(f"[locate] {name}: already done, skipping")
            continue
        rows = splits[name]
        t0 = time.time()
        gens = target.generate([r["question"] for r in rows], gen_cfg)
        _write_generations(out / f"generations_baseline_{name}.jsonl", rows, gens)
        skipped = {"hit_max_new_tokens": 0, "boundary_count_mismatch": 0, "mixed_newline_token": 0}
        acts, meta = [], []
        if name != "val_confirm":
            tok = target.tokenizer
            for r, g in zip(rows, gens):
                if not g["ended_with_eos"]:
                    skipped["hit_max_new_tokens"] += 1
                    continue
                texts = [tok.decode([t]) for t in g["gen_ids"]]
                try:
                    pos = ws.step_boundaries(texts, g["ended_with_eos"])
                except ws.MixedNewlineToken:
                    skipped["mixed_newline_token"] += 1
                    continue
                if len(pos) != g["pred_steps"] or not pos:
                    skipped["boundary_count_mismatch"] += 1
                    continue
                recs = ws.boundary_records(len(pos), r["gold_k"], True)
                acts.append(target.boundary_acts(g["prompt_ids"], g["gen_ids"], pos).astype(np.float16))
                for rec in recs:
                    meta.append({"id": r["id"], "gold_k": r["gold_k"], **rec})
            np.save(out / f"acts_{name}.npy", np.concatenate(acts, axis=0))
            with (out / f"boundaries_{name}.jsonl").open("w", encoding="utf-8") as f:
                for m in meta:
                    f.write(json.dumps(m) + "\n")
        labels: dict[str, int] = {}
        for m in meta:
            labels[m["label"]] = labels.get(m["label"], 0) + 1
        _write_json(
            done,
            {
                "split": name,
                "n_items": len(rows),
                "seconds": round(time.time() - t0, 1),
                "baseline_summary": _summary_rows(rows, gens),
                "boundaries": len(meta),
                "boundary_labels": labels,
                "skipped_items": skipped,
            },
        )
        print(f"[locate] {name}: {len(rows)} items, {len(meta)} boundaries {labels}, skipped {skipped}")


# ============================================================================ directions (CPU)


def _load_bounds(out: Path, name: str) -> tuple[np.ndarray, list[dict]]:
    acts = np.load(out / f"acts_{name}.npy")
    meta = _jsonl(out / f"boundaries_{name}.jsonl")
    if acts.shape[0] != len(meta):
        raise SystemExit(f"[exp-038] acts_{name}.npy rows {acts.shape[0]} != boundaries {len(meta)}")
    return acts, meta


def cmd_directions(cfg: dict, out: Path) -> dict:
    lc = require(cfg, "locate")
    acts, meta = _load_bounds(out, "locate")
    vacts, vmeta = _load_bounds(out, "val_select")
    n_layers = acts.shape[1]
    lo, hi = require(lc, "candidate_layer_range")
    kinds = require(lc, "directions")
    dirs: dict[str, np.ndarray] = {}
    table: dict[str, list[dict]] = {}
    for kind, spec in kinds.items():
        pos, neg = spec["pos"], spec["neg"]
        rows = []
        vecs = np.zeros((n_layers, acts.shape[2]), dtype=np.float64)
        vsel_pos = [i for i, m in enumerate(vmeta) if m["label"] == pos]
        vsel_neg = [i for i, m in enumerate(vmeta) if m["label"] == neg]
        for layer in range(n_layers):
            vec, rec = ws.matched_diff_of_means(
                acts[:, layer, :],
                [m["label"] for m in meta],
                [m["j"] for m in meta],
                pos,
                neg,
                min_per_cell=int(require(lc, "min_per_cell")),
            )
            vecs[layer] = vec
            u = ws.unit(vec)
            proj = vacts[:, layer, :].astype(np.float64) @ u
            auc = ws.auroc(proj[vsel_pos], proj[vsel_neg]) if vsel_pos and vsel_neg else None
            resid_norm = float(np.linalg.norm(acts[:, layer, :].astype(np.float64), axis=1).mean())
            rows.append(
                {
                    "layer": layer,
                    "diff_norm": float(np.linalg.norm(vec)),
                    "mean_resid_norm": resid_norm,
                    "relative_norm": float(np.linalg.norm(vec)) / resid_norm,
                    "val_auroc": auc,
                    "val_n_pos": len(vsel_pos),
                    "val_n_neg": len(vsel_neg),
                    "cells": rec["cells"],
                    "in_candidate_range": lo <= layer <= hi,
                }
            )
        dirs[kind] = vecs
        table[kind] = rows
    names = list(dirs)
    cos = (
        [ws.cosine(dirs[names[0]][layer], dirs[names[1]][layer]) for layer in range(n_layers)]
        if len(names) == 2
        else None
    )
    np.savez(out / "directions.npz", **{k: v.astype(np.float32) for k, v in dirs.items()})
    top_k = int(require(lc, "top_layers_per_direction"))
    top = {
        kind: [
            r["layer"]
            for r in sorted(
                (r for r in rows if r["in_candidate_range"] and r["val_auroc"] is not None),
                key=lambda r: (-r["val_auroc"], r["layer"]),
            )[:top_k]
        ]
        for kind, rows in table.items()
    }
    summary = {"layers": n_layers, "directions": table, "cosine_between_kinds": cos, "top_layers": top}
    _write_json(out / "directions_summary.json", summary)
    _write_candidates(cfg, out, top)
    _write_table_md(out, table, cos, top)
    print(json.dumps({"top_layers": top}, indent=1))
    return summary


def _write_candidates(cfg: dict, out: Path, top: dict[str, list[int]]) -> None:
    sw = require(cfg, "sweep")
    cands: list[dict] = [{"name": "baseline", "kind": "none"}]
    for kind, layers in top.items():
        for layer in layers:
            for a in require(sw, "alphas"):
                cands.append(
                    {
                        "name": f"orth_{kind}_L{layer}_a{a}",
                        "kind": "orthogonalize",
                        "direction": kind,
                        "direction_layer": layer,
                        "alpha": a,
                        "layers": require(sw, "ablation_layers"),
                        "matrices": require(sw, "matrices"),
                        "role": "edit",
                    }
                )
        if layers:
            cands.append(
                {
                    "name": f"actadd_{kind}_L{layers[0]}",
                    "kind": "actadd",
                    "direction": kind,
                    "direction_layer": layers[0],
                    "role": "diagnostic",
                    "_note": "inference-time steering: subtract the class-mean gap along the direction at this one layer's output",
                }
            )
    for s in require(sw, "eos_row_scales"):
        cands.append({"name": f"eos_row_x{s}", "kind": "token_row_scale", "token_ids": "eos", "scale": s, "role": "control"})
    cands.append(
        {"name": f"final_norm_x{require(sw, 'final_norm_scale')}", "kind": "final_norm_scale", "scale": require(sw, "final_norm_scale"), "role": "control"}
    )
    _write_json(out / "candidates.json", {"_note": "Proposed by `directions`; Claude may edit this file before `sweep` and records why in choices.md.", "candidates": cands})


def _write_table_md(out: Path, table: dict, cos, top) -> None:
    lines = ["# exp-038 direction statistics (generated by weight_space_repair.py directions)", ""]
    for kind, rows in table.items():
        lines += [f"## {kind}", "", "| layer | val AUROC | ||diff|| | ||diff|| / mean ||resid|| | in range |", "|---|---|---|---|---|"]
        for r in rows:
            auc = "n/a" if r["val_auroc"] is None else f"{r['val_auroc']:.3f}"
            lines.append(f"| {r['layer']} | {auc} | {r['diff_norm']:.2f} | {r['relative_norm']:.4f} | {r['in_candidate_range']} |")
        cells = rows[0]["cells"]
        lines += ["", "Locate cells (j: n_pos / n_neg / used): " + ", ".join(f"{c['stratum']}: {c['n_pos']}/{c['n_neg']}/{c['used']}" for c in cells), ""]
    if cos:
        lines += ["## cosine between the two directions, per layer", "", " ".join(f"L{i}:{c:+.2f}" for i, c in enumerate(cos)), ""]
    lines += ["## proposed top layers (val AUROC, within candidate range)", "", json.dumps(top), ""]
    (out / "directions_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ============================================================================ sweep (GPU)


def _candidate_direction(out: Path, cand: dict) -> np.ndarray:
    with np.load(out / "directions.npz") as dirs:
        return dirs[cand["direction"]][int(cand["direction_layer"])].astype(np.float64)


def _actadd_hook(cfg: dict, target: Target, out: Path, cand: dict):
    """Diagnostic steering: at one layer's output, subtract the locate-split class-mean gap
    along the unit direction (moves every position from the 'pos' class mean toward 'neg')."""
    torch = target.torch
    u_np = ws.unit(_candidate_direction(out, cand))
    acts, meta = _load_bounds(out, "locate")
    kind = require(cfg, "locate.directions")[cand["direction"]]
    pos, neg = kind["pos"], kind["neg"]
    layer = int(cand["direction_layer"])
    proj = acts[:, layer, :].astype(np.float64) @ u_np
    gap = float(proj[[m["label"] == pos for m in meta]].mean() - proj[[m["label"] == neg for m in meta]].mean())
    u = torch.tensor(u_np, dtype=torch.float16, device="cuda")

    def hook(_mod, _inp, output):
        if isinstance(output, tuple):
            return (output[0] - gap * u,) + tuple(output[1:])
        return output - gap * u

    return target.layers[layer].register_forward_hook(hook), {"layer": layer, "gap": gap}


def cmd_sweep(cfg: dict, paths: dict, out: Path, split: str, candidates_file: Path, target: Target | None = None) -> None:
    set_global_seed(int(require(cfg, "seed")))
    target = target or Target(cfg, paths)
    rows = _split_rows(cfg, paths, out)[split]
    cands = json.loads(Path(candidates_file).read_text(encoding="utf-8"))["candidates"]
    sdir = out / f"sweep_{split}"
    sdir.mkdir(parents=True, exist_ok=True)
    for cand in cands:
        done = sdir / f"{cand['name']}.summary.json"
        if done.exists():
            print(f"[sweep] {split}/{cand['name']}: done, skipping")
            continue
        t0 = time.time()
        backup, handle, extra, edit_rec = None, None, {}, None
        if cand["kind"] == "actadd":
            handle, extra = _actadd_hook(cfg, target, out, cand)
        elif cand["kind"] != "none":
            raw = {**cand, "label": cand["name"]}
            spec_direction = None
            if cand["kind"] == "orthogonalize":
                # The sweep reads the direction straight from directions.npz; the spec files
                # make-specs writes carry it as a hashed .npy instead.
                raw["direction_file"] = "directions.npz"
                spec_direction = _candidate_direction(out, cand)
            spec = ws.validate_edit_spec(raw)
            backup = Backup(edit_targets(target.model, spec, target.eos))
            edit_rec = apply_edit(target.model, spec, direction=spec_direction, eos_token_id=target.eos)
        try:
            gens = target.generate([r["question"] for r in rows], require(cfg, "generation"))
        finally:
            if handle is not None:
                handle.remove()
            if backup is not None:
                backup.restore()
        _write_generations(sdir / f"{cand['name']}.generations.jsonl", rows, gens)
        summary = {
            "candidate": cand,
            "split": split,
            "seconds": round(time.time() - t0, 1),
            "summary": _summary_rows(rows, gens),
            "hit_max_new_tokens": sum(g["hit_max_new_tokens"] for g in gens),
            "edit": edit_rec,
            **extra,
        }
        _write_json(done, summary)
        s = summary["summary"]["overall"]
        print(
            f"[sweep] {split}/{cand['name']}: hopEM {s['hop_count_exact_match_rate']:.3f} "
            f"under {s['under_decomposition_rate']:.3f} over {s['over_decomposition_rate']:.3f} "
            f"mean steps {s['mean_pred_steps']:.2f} ({summary['seconds']}s)"
        )


def _summaries(out: Path, split: str) -> dict[str, dict]:
    res = {}
    for p in sorted((out / f"sweep_{split}").glob("*.summary.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        res[d["candidate"]["name"]] = d
    return res


def cmd_choose(cfg: dict, out: Path) -> dict:
    rule = require(cfg, "sweep.selection_rule")
    sums = _summaries(out, "val_select")
    flat = {k: v["summary"] for k, v in sums.items()}
    edits = [k for k, v in sums.items() if v["candidate"].get("role") == "edit"]
    eos = [k for k, v in sums.items() if v["candidate"]["kind"] == "token_row_scale"]
    norm = [k for k, v in sums.items() if v["candidate"]["kind"] == "final_norm_scale"]
    res = ws.select_candidate(
        flat,
        baseline="baseline",
        edit_names=edits,
        control_names=eos + norm,
        max_over_increase=float(rule["max_over_increase"]),
        max_hop2_em_drop=float(rule["max_hop2_em_drop"]),
    )
    if res["chosen"]:
        res["length_matched_eos_control"] = ws.closest_by_mean_steps(flat, res["chosen"], eos)
        res["chosen_candidate"] = sums[res["chosen"]]["candidate"]
    res["rule"] = rule
    res["table"] = {
        k: {m: v["overall"][m] for m in ("hop_count_exact_match_rate", "under_decomposition_rate", "over_decomposition_rate", "mean_pred_steps")}
        for k, v in flat.items()
    }
    _write_json(out / "proposed_choice.json", res)
    print(json.dumps({k: res.get(k) for k in ("chosen", "length_matched_eos_control", "beats_all_controls", "rejected")}, indent=1))
    return res


def cmd_make_specs(cfg: dict, out: Path, choice_file: Path) -> None:
    """choice.json -> spec_*.json + direction files that run_decomposer.py --weight-edit reads."""
    choice = json.loads(Path(choice_file).read_text(encoding="utf-8"))
    for key in ("edit_candidate", "eos_control_candidate", "chosen_by", "choice_note"):
        if key not in choice:
            raise SystemExit(f"[exp-038] {choice_file} needs {key!r}")
    # The candidates actually swept (candidates.json or Claude's override), as each sweep
    # summary recorded them - not whatever candidates.json says now.
    cands = {name: d["candidate"] for name, d in _summaries(out, "val_select").items()}
    for key in ("edit_candidate", "eos_control_candidate"):
        if choice[key] not in cands:
            raise SystemExit(f"[exp-038] {key} {choice[key]!r} was not swept on val_select")
    edit = cands[choice["edit_candidate"]]
    if edit["kind"] != "orthogonalize":
        raise SystemExit("[exp-038] the edit must be an orthogonalize candidate (the weight edit proper)")
    sdir = out / "specs"
    sdir.mkdir(parents=True, exist_ok=True)
    r = ws.unit(_candidate_direction(out, edit)).astype(np.float32)
    rnd = ws.random_unit_orthogonal(r.astype(np.float64), int(require(cfg, "sweep.random_direction_seed"))).astype(np.float32)
    np.save(sdir / "direction_edit.npy", r)
    np.save(sdir / "direction_random.npy", rnd)
    sha = _rd().sha256_file
    common = {"chosen_by": choice["chosen_by"], "choice_note": choice["choice_note"]}
    base = {k: edit[k] for k in ("alpha", "layers", "matrices")}
    specs = {
        "baseline": {"kind": "none", "label": "baseline_merged_unedited", **common},
        "edit": {"kind": "orthogonalize", "label": edit["name"], "direction_file": "direction_edit.npy",
                 "direction_sha256": sha(sdir / "direction_edit.npy"), "source_candidate": edit["name"], **base, **common},
        "random": {"kind": "orthogonalize", "label": f"random_orth_to_{edit['name']}", "direction_file": "direction_random.npy",
                   "direction_sha256": sha(sdir / "direction_random.npy"), **base, **common},
        "eos_control": {**{k: cands[choice["eos_control_candidate"]][k] for k in ("kind", "token_ids", "scale")},
                        "label": choice["eos_control_candidate"], **common},
        "norm_control": {"kind": "final_norm_scale", "scale": float(require(cfg, "sweep.final_norm_scale")),
                         "label": f"final_norm_x{require(cfg, 'sweep.final_norm_scale')}", **common},
    }
    for name, spec in specs.items():
        ws.validate_edit_spec(spec)
        _write_json(sdir / f"spec_{name}.json", spec)
    confirm = [{"name": n, "spec_file": str(sdir / f"spec_{n}.json")} for n in specs]
    _write_json(out / "confirm_candidates.json", {"candidates": confirm})
    print(f"[make-specs] wrote {len(specs)} specs to {sdir}")


def cmd_confirm(cfg: dict, paths: dict, out: Path, target: Target | None = None) -> None:
    """Run the five final specs once on val_confirm (never used for choosing)."""
    target = target or Target(cfg, paths)
    rows = _split_rows(cfg, paths, out)["val_confirm"]
    sdir = out / "confirm_val_confirm"
    sdir.mkdir(parents=True, exist_ok=True)
    for item in json.loads((out / "confirm_candidates.json").read_text(encoding="utf-8"))["candidates"]:
        done = sdir / f"{item['name']}.summary.json"
        if done.exists():
            continue
        spec, direction, prov = load_spec(item["spec_file"])
        backup = Backup(edit_targets(target.model, spec, target.eos))
        rec = apply_edit(target.model, spec, direction=direction, eos_token_id=target.eos)
        try:
            gens = target.generate([r["question"] for r in rows], require(cfg, "generation"))
        finally:
            backup.restore()
        _write_generations(sdir / f"{item['name']}.generations.jsonl", rows, gens)
        _write_json(done, {"name": item["name"], "edit": {**rec, **prov}, "summary": _summary_rows(rows, gens)})
        s = _summary_rows(rows, gens)["overall"]
        print(f"[confirm] {item['name']}: hopEM {s['hop_count_exact_match_rate']:.3f} under {s['under_decomposition_rate']:.3f} over {s['over_decomposition_rate']:.3f}")


def cmd_equivalence(cfg: dict, paths: dict, out: Path, n_prompts: int = 8, target: Target | None = None) -> dict:
    """Hook ablation (project r out of every writer's output) vs the weight edit, same prompts."""
    target = target or Target(cfg, paths)
    torch = target.torch
    spec, direction, _ = load_spec(out / "specs" / "spec_edit.json")
    rows = _split_rows(cfg, paths, out)["locate"][:n_prompts]
    u = torch.tensor(ws.unit(direction), dtype=torch.float32, device="cuda")
    a = spec["alpha"]

    def ablate(_m, _i, output):
        hs = output[0] if isinstance(output, tuple) else output
        h32 = hs.float()
        new = (h32 - a * (h32 @ u).unsqueeze(-1) * u).to(hs.dtype)
        return (new,) + tuple(output[1:]) if isinstance(output, tuple) else new

    mods = []
    idx = range(len(target.layers)) if spec["layers"] == "all" else spec["layers"]
    for i in idx:
        if "o_proj" in spec["matrices"]:
            mods.append(target.layers[i].self_attn.o_proj)
        if "down_proj" in spec["matrices"]:
            mods.append(target.layers[i].mlp.down_proj)
    if "embed_tokens" in spec["matrices"]:
        mods.append(target.model.model.embed_tokens)

    def logits(q):
        ids = target.tokenizer(target.prompt(q), return_tensors="pt").to("cuda")
        with torch.no_grad():
            return target.model(**ids).logits[0, -1].float()

    diffs = []
    for r in rows:
        hs = [m.register_forward_hook(ablate) for m in mods]
        try:
            lh = logits(r["question"])
        finally:
            for h in hs:
                h.remove()
        backup = Backup(edit_targets(target.model, spec, target.eos))
        apply_edit(target.model, spec, direction=direction, eos_token_id=target.eos)
        try:
            lw = logits(r["question"])
        finally:
            backup.restore()
        diffs.append({"id": r["id"], "max_abs_logit_diff": float((lh - lw).abs().max()), "same_argmax": bool(lh.argmax() == lw.argmax())})
    res = {"n": len(diffs), "max_abs_logit_diff": max(d["max_abs_logit_diff"] for d in diffs), "all_same_argmax": all(d["same_argmax"] for d in diffs), "rows": diffs}
    _write_json(out / "equivalence.json", res)
    print(json.dumps({k: res[k] for k in ("n", "max_abs_logit_diff", "all_same_argmax")}))
    return res


# ============================================================================ CLI


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["split", "locate", "directions", "sweep", "choose", "make-specs", "confirm", "equivalence"])
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--out", required=True, type=Path, help="Run directory (e.g. runs/exp-038, or a Drive folder)")
    p.add_argument("--split", default="val_select")
    p.add_argument("--candidates", type=Path, default=None)
    p.add_argument("--choice", type=Path, default=None)
    args = p.parse_args()
    cfg, paths = _load(args.config)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == "split":
        cmd_split(cfg, paths, args.out)
    elif args.command == "locate":
        cmd_locate(cfg, paths, args.out)
    elif args.command == "directions":
        cmd_directions(cfg, args.out)
    elif args.command == "sweep":
        cmd_sweep(cfg, paths, args.out, args.split, args.candidates or args.out / "candidates.json")
    elif args.command == "choose":
        cmd_choose(cfg, args.out)
    elif args.command == "make-specs":
        cmd_make_specs(cfg, args.out, args.choice or args.out / "choice.json")
    elif args.command == "confirm":
        cmd_confirm(cfg, paths, args.out)
    elif args.command == "equivalence":
        cmd_equivalence(cfg, paths, args.out)


if __name__ == "__main__":
    main()
