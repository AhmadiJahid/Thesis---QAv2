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
                reads: edit, random-direction control, length-matched EOS-logit-bias control,
                norm-only sanity check, and the unedited merged baseline.
    freeze      CPU. Record every spec's sha256 and the single commit of choice.json in
                specs_frozen.json; never rewrites it.
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
        untied = assert_untied(self.model)
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
            **untied,
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


def assert_untied(model) -> dict:
    """Refuse a model whose lm_head shares storage with the input embedding.

    With tied weights an edit to the embedding (orthogonalize with embed_tokens) would also
    edit the unembedding, and an lm_head edit would also edit the inputs. Mistral-7B-v0.3 is
    untied; this asserts it rather than assuming it.
    """
    lm = model.lm_head.weight
    emb = model.model.embed_tokens.weight
    if lm.data_ptr() == emb.data_ptr() or lm is emb:
        raise SystemExit("[weight_edit] REFUSING: lm_head.weight and embed_tokens.weight share storage (tied)")
    return {"lm_head_embed_untied": True}


def _writers(model, layers, matrices):
    """(name, parameter, mode) triples an orthogonalize edit touches."""
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


def _token_ids(spec: dict, eos_token_id: int) -> list[int]:
    return [eos_token_id] if spec["token_ids"] == "eos" else list(spec["token_ids"])


def edit_targets(model, spec: dict, eos_token_id: int) -> list[tuple[str, Any, Any]]:
    """(name, parameter, row-index-or-None) for every weight tensor the spec changes."""
    kind = spec["kind"]
    if kind == "orthogonalize":
        return [(n, w, None) for n, w, _ in _writers(model, spec["layers"], spec["matrices"])]
    if kind == "final_norm_scale":
        return [("norm", model.model.norm.weight, None)]
    return []


def apply_edit(model, spec: dict, *, direction: np.ndarray | None, eos_token_id: int) -> dict:
    """Apply a validated spec in place. Computes in float32, writes back in the weight dtype."""
    import torch

    kind = spec["kind"]
    rec: dict[str, Any] = {"kind": kind, "label": spec["label"], "tensors": []}
    if kind != "none":
        rec.update(assert_untied(model))
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
                norm = float(w32.norm())
                w.data.copy_(new.to(w.dtype))
                del w32, new
                rec["tensors"].append({"name": name, "delta_fro": delta, "weight_fro": norm, "numel": w.numel()})
            rec["delta_fro_total"] = float(sum(t["delta_fro"] ** 2 for t in rec["tensors"]) ** 0.5)
            rec["weight_fro_total"] = float(sum(t["weight_fro"] ** 2 for t in rec["tensors"]) ** 0.5)
            rec["delta_fro_ratio"] = rec["delta_fro_total"] / rec["weight_fro_total"]
        elif kind == "final_norm_scale":
            w = model.model.norm.weight
            w.data.copy_((w.data.float() * spec["scale"]).to(w.dtype))
            rec["tensors"].append({"name": "norm", "numel": w.numel()})
        elif kind == "token_logit_bias":
            ids = _token_ids(spec, eos_token_id)
            head = model.lm_head
            w = head.weight
            if head.bias is None:
                head.bias = torch.nn.Parameter(torch.zeros(w.shape[0], dtype=w.dtype, device=w.device), requires_grad=False)
                rec["lm_head_bias_created"] = True
            head.bias.data[ids] += spec["bias"]
            rec["tensors"].append({"name": "lm_head.bias", "rows": ids, "numel": len(ids)})
            rec["token_ids_resolved"] = ids
    rec["parameters_changed"] = int(sum(t["numel"] for t in rec["tensors"]))
    return rec


class Backup:
    """CPU copies of exactly what an edit touches, to undo it exactly.

    Covers weight tensors (``edit_targets``) and the lm_head bias, which a
    ``token_logit_bias`` edit creates when the model has none: restoring removes it again.
    """

    def __init__(self, model, targets: list[tuple[str, Any, Any]]) -> None:
        self.model = model
        self.items = []
        for _name, w, rows in targets:
            data = w.data[rows].detach().cpu().clone() if rows is not None else w.data.detach().cpu().clone()
            self.items.append((w, rows, data))
        bias = model.lm_head.bias
        self.lm_bias = None if bias is None else bias.data.detach().cpu().clone()

    def restore(self) -> None:
        import torch

        with torch.no_grad():
            for w, rows, data in self.items:
                if rows is not None:
                    w.data[rows] = data.to(w.device)
                else:
                    w.data.copy_(data.to(w.device))
            head = self.model.lm_head
            if self.lm_bias is None:
                if head.bias is not None:
                    head.register_parameter("bias", None)
            else:
                head.bias.data.copy_(self.lm_bias.to(head.bias.device))


def load_spec(spec_path: str | Path) -> tuple[dict, np.ndarray | None, dict]:
    """Read and validate an edit spec; hash the spec FILE and, if it has one, its direction file."""
    spec_path = Path(spec_path)
    raw = json.loads(spec_path.read_text(encoding="utf-8"))
    spec = ws.validate_edit_spec(raw)
    direction = None
    prov: dict[str, Any] = {"spec_path": str(spec_path), "spec_file_sha256": _rd().sha256_file(spec_path)}
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
    The caller refuses quantized models before getting here (an edit to 4-/8-bit weights is
    not exact).
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


def _edited_run(target: "Target", spec: dict, direction, fn):
    """Back up, apply, run ``fn()``, and always restore - even if the apply itself fails."""
    backup = Backup(target.model, edit_targets(target.model, spec, target.eos))
    try:
        rec = apply_edit(target.model, spec, direction=direction, eos_token_id=target.eos)
        return rec, fn()
    finally:
        backup.restore()


def _mirror(out: Path, mirror: Path | None, files: list[Path]) -> None:
    """Copy finished files to the Drive mirror right away (a disconnect loses at most one item)."""
    if mirror is None:
        return
    import shutil

    for f in files:
        dest = mirror / f.relative_to(out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest)


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


def cmd_locate(cfg: dict, paths: dict, out: Path, target: Target | None = None, mirror: Path | None = None) -> None:
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
        _mirror(out, mirror, [f for f in out.glob(f"*_{name}*") if f.is_file()] + [out / "target.json"])
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
    for b in require(sw, "length_match.eos_logit_biases"):
        cands.append(_eos_bias_candidate(b))
    cands.append(
        {"name": f"final_norm_x{require(sw, 'final_norm_scale')}", "kind": "final_norm_scale", "scale": require(sw, "final_norm_scale"),
         "role": "sanity_check", "_note": "under greedy decoding a uniform logit scale cannot change an argmax"}
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


def _eos_bias_candidate(bias: float) -> dict:
    return {"name": f"eos_bias_{bias:+g}", "kind": "token_logit_bias", "token_ids": "eos", "bias": float(bias), "role": "control"}


def _sweep_manifest(out: Path, split: str) -> dict[str, str]:
    path = out / f"sweep_{split}" / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def cmd_sweep(
    cfg: dict,
    paths: dict,
    out: Path,
    split: str,
    candidates_file: Path,
    target: Target | None = None,
    mirror: Path | None = None,
) -> None:
    """Generate ``split`` under each candidate. Done-files are keyed to the candidate's content
    hash, and manifest.json maps each name to the hash it was last swept with, so a changed
    candidate is re-run instead of an old output being reused."""
    set_global_seed(int(require(cfg, "seed")))
    cands = json.loads(Path(candidates_file).read_text(encoding="utf-8"))["candidates"]
    sdir = out / f"sweep_{split}"
    sdir.mkdir(parents=True, exist_ok=True)
    manifest = _sweep_manifest(out, split)
    todo = [c for c in cands if not (sdir / f"{ws.keyed_name(c['name'], ws.candidate_sha256(c))}.summary.json").exists()]
    for cand in cands:
        manifest[cand["name"]] = ws.candidate_sha256(cand)
    _write_json(sdir / "manifest.json", manifest)
    if not todo:
        print(f"[sweep] {split}: all {len(cands)} candidates done")
        return
    target = target or Target(cfg, paths)
    rows = _split_rows(cfg, paths, out)[split]
    questions = [r["question"] for r in rows]
    gen_cfg = require(cfg, "generation")
    for cand in todo:
        stem = ws.keyed_name(cand["name"], ws.candidate_sha256(cand))
        t0 = time.time()
        extra: dict[str, Any] = {}
        edit_rec = None
        if cand["kind"] == "actadd":
            handle, extra = _actadd_hook(cfg, target, out, cand)
            try:
                gens = target.generate(questions, gen_cfg)
            finally:
                handle.remove()
        elif cand["kind"] == "none":
            gens = target.generate(questions, gen_cfg)
        else:
            raw = {**cand, "label": cand["name"]}
            direction = None
            if cand["kind"] == "orthogonalize":
                # The sweep reads the direction from directions.npz; the spec files written by
                # make-specs carry it as a hashed .npy instead.
                raw["direction_file"] = "directions.npz"
                direction = _candidate_direction(out, cand)
            spec = ws.validate_edit_spec(raw)
            edit_rec, gens = _edited_run(target, spec, direction, lambda: target.generate(questions, gen_cfg))
        gen_path = sdir / f"{stem}.generations.jsonl"
        _write_generations(gen_path, rows, gens)
        summary = {
            "candidate": cand,
            "candidate_sha256": ws.candidate_sha256(cand),
            "split": split,
            "seconds": round(time.time() - t0, 1),
            "summary": _summary_rows(rows, gens),
            "hit_max_new_tokens": sum(g["hit_max_new_tokens"] for g in gens),
            "edit": edit_rec,
            **extra,
        }
        done = sdir / f"{stem}.summary.json"
        _write_json(done, summary)
        _mirror(out, mirror, [gen_path, done, sdir / "manifest.json"])
        s = summary["summary"]["overall"]
        print(
            f"[sweep] {split}/{cand['name']}: hopEM {s['hop_count_exact_match_rate']:.3f} "
            f"under {s['under_decomposition_rate']:.3f} over {s['over_decomposition_rate']:.3f} "
            f"mean steps {s['mean_pred_steps']:.2f} ({summary['seconds']}s)"
        )


def _summaries(out: Path, split: str) -> dict[str, dict]:
    """The current summary of every candidate in the manifest (keyed by its swept hash)."""
    res = {}
    sdir = out / f"sweep_{split}"
    for name, sha in _sweep_manifest(out, split).items():
        path = sdir / f"{ws.keyed_name(name, sha)}.summary.json"
        if not path.exists():
            raise SystemExit(f"[exp-038] {split}: candidate {name} is in the manifest but not swept ({path.name})")
        res[name] = json.loads(path.read_text(encoding="utf-8"))
    return res


def cmd_choose(cfg: dict, out: Path) -> dict:
    """ADR 0042's default rule, the length-matched EOS-bias control and, when no swept bias
    matches the edit's mean step count within tolerance, a refinement round to sweep next."""
    rule = require(cfg, "sweep.selection_rule")
    lm = require(cfg, "sweep.length_match")
    sums = _summaries(out, "val_select")
    flat = {k: v["summary"] for k, v in sums.items()}
    edits = [k for k, v in sums.items() if v["candidate"].get("role") == "edit"]
    eos = [k for k, v in sums.items() if v["candidate"]["kind"] == "token_logit_bias"]
    norm = [k for k, v in sums.items() if v["candidate"]["kind"] == "final_norm_scale"]
    res = ws.select_candidate(
        flat,
        baseline="baseline",
        edit_names=edits,
        control_names=eos + norm,
        max_over_increase=float(rule["max_over_increase"]),
        max_hop2_em_drop=float(rule["max_hop2_em_drop"]),
    )
    tol = float(lm["tolerance_mean_steps"])
    res["length_match"] = {"tolerance_mean_steps": tol}
    refine_path = out / "refine_candidates.json"
    if res["chosen"]:
        best, gap = ws.closest_by_mean_steps(flat, res["chosen"], eos)
        rounds = sum(1 for v in sums.values() if v["candidate"].get("refine_round"))
        res["length_matched_eos_control"] = best
        res["chosen_candidate"] = sums[res["chosen"]]["candidate"]
        res["length_match"].update(
            {
                "edit_mean_steps_val_select": flat[res["chosen"]]["overall"]["mean_pred_steps"],
                "control_mean_steps_val_select": flat[best]["overall"]["mean_pred_steps"],
                "gap_val_select": gap,
                "within_tolerance": abs(gap) <= tol,
                "refined_candidates_swept": rounds,
            }
        )
        if abs(gap) > tol:
            points = [(sums[k]["candidate"]["bias"], flat[k]["overall"]["mean_pred_steps"]) for k in eos]
            new = ws.refine_biases(points, flat[res["chosen"]]["overall"]["mean_pred_steps"], int(lm["refine_points"]))
            cands = [{**_eos_bias_candidate(b), "refine_round": True} for b in new]
            _write_json(refine_path, {"candidates": cands})
            res["length_match"]["refine_candidates_file"] = str(refine_path)
        elif refine_path.exists():
            refine_path.unlink()
    res["rule"] = rule
    res["table"] = {
        k: {m: v["overall"][m] for m in ("hop_count_exact_match_rate", "under_decomposition_rate", "over_decomposition_rate", "mean_pred_steps")}
        for k, v in flat.items()
    }
    _write_json(out / "proposed_choice.json", res)
    print(json.dumps({k: res.get(k) for k in ("chosen", "length_matched_eos_control", "length_match", "beats_all_controls", "rejected")}, indent=1))
    return res


def cmd_make_specs(cfg: dict, out: Path, choice_file: Path) -> None:
    """choice.json -> spec_*.json + direction files that run_decomposer.py --weight-edit reads.

    Refused once the specs are frozen (specs_frozen.json exists): frozen specs are never
    rewritten, by this or anything else.
    """
    if (out / "specs_frozen.json").exists():
        raise SystemExit("[exp-038] specs are frozen (specs_frozen.json exists); make-specs refuses to rewrite them")
    choice = json.loads(Path(choice_file).read_text(encoding="utf-8"))
    for key in ("edit_candidate", "eos_control_candidate", "chosen_by", "choice_note"):
        if key not in choice:
            raise SystemExit(f"[exp-038] {choice_file} needs {key!r}")
    # The candidates actually swept (candidates.json, Claude's override, refinements), as each
    # sweep summary recorded them - not whatever candidates.json says now.
    sums = _summaries(out, "val_select")
    cands = {name: d["candidate"] for name, d in sums.items()}
    for key in ("edit_candidate", "eos_control_candidate"):
        if choice[key] not in cands:
            raise SystemExit(f"[exp-038] {key} {choice[key]!r} was not swept on val_select")
    edit = cands[choice["edit_candidate"]]
    ctrl = cands[choice["eos_control_candidate"]]
    if edit["kind"] != "orthogonalize":
        raise SystemExit("[exp-038] the edit must be an orthogonalize candidate (the weight edit proper)")
    if ctrl["kind"] != "token_logit_bias":
        raise SystemExit("[exp-038] the length control must be a token_logit_bias candidate")
    gap = sums[ctrl["name"]]["summary"]["overall"]["mean_pred_steps"] - sums[edit["name"]]["summary"]["overall"]["mean_pred_steps"]
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
        "eos_control": {"kind": "token_logit_bias", "token_ids": ctrl["token_ids"], "bias": ctrl["bias"], "label": ctrl["name"],
                        "length_match_val_select": {"gap_mean_steps": gap,
                                                    "tolerance_mean_steps": float(require(cfg, "sweep.length_match.tolerance_mean_steps"))},
                        **common},
        "norm_control": {"kind": "final_norm_scale", "scale": float(require(cfg, "sweep.final_norm_scale")),
                         "label": f"final_norm_x{require(cfg, 'sweep.final_norm_scale')}_sanity_check", **common},
    }
    for name, spec in specs.items():
        ws.validate_edit_spec(spec)
        _write_json(sdir / f"spec_{name}.json", spec)
    confirm = [{"name": n, "spec_file": str(sdir / f"spec_{n}.json")} for n in specs]
    _write_json(out / "confirm_candidates.json", {"candidates": confirm})
    print(f"[make-specs] wrote {len(specs)} specs to {sdir}; length gap on val_select {gap:+.3f} mean steps")


def cmd_freeze(out: Path, choice_commits: list[str]) -> dict:
    """Write specs_frozen.json once: every spec and direction file's sha256 plus the one commit
    of choice.json. Never rewrites it (src/weight_space.py check_freeze)."""
    sdir = out / "specs"
    hashes = {p.name: _rd().sha256_file(p) for p in sorted(sdir.iterdir()) if p.is_file()}
    path = out / "specs_frozen.json"
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    verdict = ws.check_freeze(hashes, choice_commits, existing)
    if verdict == "write":
        _write_json(path, {"choice_commit": choice_commits[0], "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "files": hashes})
    print(f"[freeze] {verdict}: {len(hashes)} files, choice commit {choice_commits[0]}")
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_frozen(out: Path, spec_file: Path) -> str:
    """The spec's sha256, after checking it against the freeze record."""
    frozen = json.loads((out / "specs_frozen.json").read_text(encoding="utf-8"))
    sha = _rd().sha256_file(spec_file)
    if frozen["files"].get(Path(spec_file).name) != sha:
        raise SystemExit(f"[exp-038] {spec_file} does not match specs_frozen.json - refusing")
    return sha


def cmd_confirm(cfg: dict, paths: dict, out: Path, target: Target | None = None, mirror: Path | None = None) -> None:
    """Run the five frozen specs once on val_confirm (never used for choosing)."""
    sdir = out / "confirm_val_confirm"
    sdir.mkdir(parents=True, exist_ok=True)
    items = json.loads((out / "confirm_candidates.json").read_text(encoding="utf-8"))["candidates"]
    rows = None
    for item in items:
        sha = _assert_frozen(out, Path(item["spec_file"]))
        stem = ws.keyed_name(item["name"], sha)
        done = sdir / f"{stem}.summary.json"
        if done.exists():
            continue
        target = target or Target(cfg, paths)
        rows = rows or _split_rows(cfg, paths, out)["val_confirm"]
        spec, direction, prov = load_spec(item["spec_file"])
        rec, gens = _edited_run(target, spec, direction, lambda: target.generate([r["question"] for r in rows], require(cfg, "generation")))
        gen_path = sdir / f"{stem}.generations.jsonl"
        _write_generations(gen_path, rows, gens)
        summ = _summary_rows(rows, gens)
        _write_json(done, {"name": item["name"], "spec_file_sha256": sha, "edit": {**rec, **prov}, "summary": summ})
        _mirror(out, mirror, [gen_path, done])
        s = summ["overall"]
        print(f"[confirm] {item['name']}: hopEM {s['hop_count_exact_match_rate']:.3f} under {s['under_decomposition_rate']:.3f} "
              f"over {s['over_decomposition_rate']:.3f} mean steps {s['mean_pred_steps']:.2f}")


def cmd_equivalence(cfg: dict, paths: dict, out: Path, target: Target | None = None) -> dict:
    """Hook ablation (project r out of every writer's output) vs the weight edit, same prompts.

    Passes only if every prompt has the same argmax and the largest absolute logit difference
    is within ``equivalence.max_abs_logit_diff``; otherwise the run stops here.
    """
    eq = require(cfg, "equivalence")
    target = target or Target(cfg, paths)
    torch = target.torch
    spec, direction, _ = load_spec(out / "specs" / "spec_edit.json")
    rows = _split_rows(cfg, paths, out)["locate"][: int(eq["n_prompts"])]
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
        _rec, lw = _edited_run(target, spec, direction, lambda: logits(r["question"]))
        diffs.append({"id": r["id"], "max_abs_logit_diff": float((lh - lw).abs().max()), "same_argmax": bool(lh.argmax() == lw.argmax())})
    worst = max(d["max_abs_logit_diff"] for d in diffs)
    same = all(d["same_argmax"] for d in diffs)
    thr = float(eq["max_abs_logit_diff"])
    res = {"n": len(diffs), "max_abs_logit_diff": worst, "threshold": thr, "all_same_argmax": same,
           "passed": bool(same and worst <= thr), "rows": diffs}
    _write_json(out / "equivalence.json", res)
    print(json.dumps({k: res[k] for k in ("n", "max_abs_logit_diff", "threshold", "all_same_argmax", "passed")}))
    if not res["passed"]:
        raise SystemExit("[exp-038] EQUIVALENCE FAILED: the weight edit does not reproduce the hook ablation; see equivalence.json")
    return res


# ============================================================================ CLI


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["split", "locate", "directions", "sweep", "choose", "make-specs", "freeze", "confirm", "equivalence"])
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--out", required=True, type=Path, help="Run directory (fast local disk)")
    p.add_argument("--mirror", type=Path, default=None, help="Drive folder; each finished item is copied there at once")
    p.add_argument("--split", default="val_select")
    p.add_argument("--candidates", type=Path, default=None)
    p.add_argument("--choice", type=Path, default=None)
    p.add_argument("--choice-commits", default=None, help="freeze: comma-separated commits that touched choice.json")
    args = p.parse_args()
    cfg, paths = _load(args.config)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == "split":
        cmd_split(cfg, paths, args.out)
    elif args.command == "locate":
        cmd_locate(cfg, paths, args.out, mirror=args.mirror)
    elif args.command == "directions":
        cmd_directions(cfg, args.out)
    elif args.command == "sweep":
        cmd_sweep(cfg, paths, args.out, args.split, args.candidates or args.out / "candidates.json", mirror=args.mirror)
    elif args.command == "choose":
        cmd_choose(cfg, args.out)
    elif args.command == "make-specs":
        cmd_make_specs(cfg, args.out, args.choice or args.out / "choice.json")
    elif args.command == "freeze":
        commits = [c for c in (args.choice_commits or "").split(",") if c.strip()]
        cmd_freeze(args.out, commits)
    elif args.command == "confirm":
        cmd_confirm(cfg, paths, args.out, mirror=args.mirror)
    elif args.command == "equivalence":
        cmd_equivalence(cfg, paths, args.out)


if __name__ == "__main__":
    main()
