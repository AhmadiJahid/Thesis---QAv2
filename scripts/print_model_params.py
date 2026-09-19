#!/usr/bin/env python3
"""Print the parameter count of every registered model, against its committed ceiling.

ADR 0027: a number that justifies a config constant ships as a script that prints it,
not as prose. ``configs/model_limits.json`` sets ``decomposer_max_params`` to 1.5e10
because Qwen2.5-14B-Instruct has 14,770,033,664 parameters (ADR 0031 item 4) — so that
number has to be re-derivable by anyone, at any time, without trusting a quoted figure.

What it does **not** do: load weights, or download them. A parameter count is read from
safetensors *metadata* only:

- **cache** — every shard's safetensors header (an 8-byte little-endian length followed
  by that many bytes of JSON) is parsed from the local HF cache and the tensor shapes are
  summed. Exact, and it reads a few KB per shard rather than the file.
- **hub** — for a model that is not cached, the Hub API's ``safetensors.total`` field.
  Exact per HF's own indexing, a single metadata request, no weights.

A model that is neither cached nor reachable prints ``unmeasured`` with the reason. It is
never estimated from a config's layer dimensions: an invented count that looks right is
worse than a missing one, and the ceiling is a standing constraint.

Run::

    .venv/bin/python scripts/print_model_params.py                      # every folder
    .venv/bin/python scripts/print_model_params.py --offline            # cache only, no network
    .venv/bin/python scripts/print_model_params.py --model qwen2_5_3b   # the tiny mode

``--model`` against a small cached folder is the smoke test for the full path (ADR 0027
point 5): same code, one folder, a second of wall clock.

Exit code is 1 if any measured model breaches its ceiling, 0 otherwise. An unmeasured
model does not fail the run — it is reported as unmeasured, which is the honest state.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from model_size import ceiling_for, load_limits  # noqa: E402
from run_config import load_config  # noqa: E402

#: Where the registries live. The folder list is read from these committed configs — one
#: ``config.json`` per model folder, each naming its ``model_id`` — so the script can never
#: drift from the registry the runner actually loads.
COMPONENT_REGISTRIES = {
    "decomposer": Path("components") / "decomposer" / "models",
    "router": Path("components") / "router" / "models",
}


def registry_models(component: str) -> list[tuple[str, str]]:
    """``(folder name, model_id)`` for every folder of one component's registry."""
    root = REPO_ROOT / COMPONENT_REGISTRIES[component]
    out = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        config = folder / "config.json"
        if not config.is_file():
            continue
        out.append((folder.name, load_config(config)["model_id"]))
    return out


def safetensors_header(path: Path) -> dict[str, Any]:
    """The JSON header of one safetensors file, read without touching the tensor data."""
    with path.open("rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        return json.loads(handle.read(length))


def count_from_header(header: dict[str, Any]) -> int:
    """Sum the element counts of every tensor a safetensors header describes.

    ``__metadata__`` is the format's reserved key and holds no tensor, so it is skipped.
    An entry's ``shape`` is the tensor's dimensions; an empty shape is a scalar (1).
    """
    total = 0
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        count = 1
        for dim in entry["shape"]:
            count *= dim
        total += count
    return total


def cached_snapshot(model_id: str) -> Path | None:
    """The local HF cache snapshot for ``model_id``, or None when it is not cached."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return None
    try:
        return Path(snapshot_download(model_id, local_files_only=True))
    except Exception:
        return None


#: The manifest a sharded repo ships: ``weight_map`` names, per tensor, which file holds it.
SAFETENSORS_INDEX = "model.safetensors.index.json"


def indexed_shards(snapshot: Path) -> list[Path] | None:
    """The shards :data:`SAFETENSORS_INDEX` names, or None when the snapshot has no index.

    Globbing ``*.safetensors`` is wrong whenever a snapshot holds BOTH the shards and a
    whole-model file beside them - ``consolidated.safetensors``, a merged export, a stale
    single-file revision - because every tensor is then counted twice and the total is
    silently double. The index is the repo's own statement of which files make up the model,
    so when it exists it decides; only a genuinely unsharded snapshot falls back to the glob.
    A file the index names but the cache does not hold is skipped by the caller, which then
    reports a partial count rather than pretending to a whole one.
    """
    index = snapshot / SAFETENSORS_INDEX
    if not index.is_file():
        return None
    weight_map = json.loads(index.read_text(encoding="utf-8")).get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        return None
    return sorted({snapshot / name for name in weight_map.values()})


def count_from_cache(model_id: str) -> tuple[int | None, str]:
    """Exact parameter count from the cached shards, as ``(count, detail)``."""
    snapshot = cached_snapshot(model_id)
    if snapshot is None:
        return None, "not in the local HF cache"
    named = indexed_shards(snapshot)
    if named is None:
        shards = sorted(snapshot.glob("*.safetensors"))
        source = "cached shard(s)"
    else:
        shards = [p for p in named if p.is_file()]
        missing = len(named) - len(shards)
        source = f"cached shard(s) of {len(named)} in {SAFETENSORS_INDEX}"
        if missing:
            source += f", {missing} MISSING - count is partial"
    if not shards:
        return None, f"cached at {snapshot.name} but no .safetensors shard"
    total = sum(count_from_header(safetensors_header(shard)) for shard in shards)
    return total, f"{len(shards)} {source}"


def count_from_hub(model_id: str) -> tuple[int | None, str]:
    """Parameter count from the Hub API's safetensors index. One metadata request."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return None, "huggingface_hub is not installed"
    try:
        info = HfApi().model_info(model_id)
    except Exception as exc:  # offline, gated, renamed, typo'd
        return None, f"hub lookup failed: {type(exc).__name__}"
    safetensors = getattr(info, "safetensors", None)
    if safetensors is None or getattr(safetensors, "total", None) is None:
        return None, "the hub reports no safetensors index for this model"
    return int(safetensors.total), f"hub index, revision {info.sha[:12]}"


def measure(model_id: str, *, offline: bool) -> dict[str, Any]:
    """Parameter count for one model id, preferring the cache and never downloading."""
    count, detail = count_from_cache(model_id)
    if count is not None:
        return {"count": count, "source": "cache", "detail": detail}
    if offline:
        return {"count": None, "source": "unmeasured", "detail": f"{detail}; --offline set"}
    hub_count, hub_detail = count_from_hub(model_id)
    if hub_count is not None:
        return {"count": hub_count, "source": "hub", "detail": hub_detail}
    return {"count": None, "source": "unmeasured", "detail": f"{detail}; {hub_detail}"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--component",
        default="decomposer",
        choices=sorted(COMPONENT_REGISTRIES),
        help="which registry to read, and whose ceiling to compare against",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="one folder name from that registry (the tiny mode: one model, no network "
        "if it is cached)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="local HF cache only; an uncached model is reported unmeasured rather than "
        "looked up",
    )
    parser.add_argument("--limits-config", default="model_limits.json")
    args = parser.parse_args()

    limits = load_limits(args.limits_config)
    ceiling = ceiling_for(args.component, limits)
    models = registry_models(args.component)
    if args.model is not None:
        models = [(name, mid) for name, mid in models if name == args.model]
        if not models:
            parser.error(
                f"no folder {args.model!r} in the {args.component} registry; known: "
                f"{[name for name, _ in registry_models(args.component)]}"
            )

    print(
        f"component={args.component}  ceiling={ceiling:,} "
        f"(configs/{args.limits_config})"
    )
    print(f"{'folder':<24} {'parameters':>16} {'% ceiling':>10}  {'verdict':<9} source")
    breaches = 0
    for name, model_id in models:
        record = measure(model_id, offline=args.offline)
        count = record["count"]
        if count is None:
            print(
                f"{name:<24} {'unmeasured':>16} {'-':>10}  {'-':<9} "
                f"{record['detail']}"
            )
            continue
        over = count > ceiling
        breaches += int(over)
        print(
            f"{name:<24} {count:>16,} {100.0 * count / ceiling:>9.1f}%  "
            f"{'BREACH' if over else 'within':<9} {record['source']} ({record['detail']})"
        )
        print(f"{'':<24} {model_id}")
    if breaches:
        print(
            f"\n{breaches} model(s) above the {args.component} ceiling. Raising it is "
            f"Jahid's decision with his supervisor, not this script's."
        )
    return 1 if breaches else 0


if __name__ == "__main__":
    raise SystemExit(main())
