"""Parameter-count ceilings, printed and asserted at model load.

CLAUDE.md records a standing constraint from Jahid's supervisor: roughly 8B
parameters overall and ~600M for the router. The numbers live in
``configs/model_limits.json`` so they are a committed value rather than a
constant buried in code; the assertion is hard, because a ceiling that is only
assumed is not a ceiling.

Changing a limit is a human decision (Jahid with his supervisor), not an agent's.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from run_config import load_config, require

_LIMIT_KEY = {
    "router": "router_max_params",
    "decomposer": "default_max_params",
    "retrieval": "default_max_params",
    "ner": "default_max_params",
    "reranker": "default_max_params",
}


def load_limits(limits_config: str | Path = "model_limits.json") -> dict[str, Any]:
    cfg = load_config(limits_config)
    require(cfg, "router_max_params")
    require(cfg, "default_max_params")
    return cfg


def count_parameters(model: Any) -> int:
    """Total parameter count for a torch / transformers model."""
    num_parameters = getattr(model, "num_parameters", None)
    if callable(num_parameters):
        return int(num_parameters())
    params = getattr(model, "parameters", None)
    if callable(params):
        return int(sum(p.numel() for p in params()))
    raise TypeError(f"cannot count parameters of {type(model).__name__}")


def ceiling_for(component: str, limits: dict[str, Any]) -> int:
    if component not in _LIMIT_KEY:
        raise KeyError(
            f"unknown component {component!r} for the parameter ceiling; "
            f"known: {sorted(_LIMIT_KEY)}"
        )
    return int(require(limits, _LIMIT_KEY[component]))


def assert_within_ceiling(
    model: Any,
    *,
    component: str,
    model_id: str,
    limits: dict[str, Any],
) -> dict[str, Any]:
    """Print the parameter count and fail hard when it breaches the ceiling."""
    count = count_parameters(model)
    ceiling = ceiling_for(component, limits)
    print(
        f"[model_size] {component} model={model_id} "
        f"parameters={count:,} ceiling={ceiling:,} "
        f"({100.0 * count / ceiling:.1f}% of ceiling)"
    )
    if count > ceiling:
        raise SystemExit(
            f"[model_size] REFUSING TO RUN: {component} model {model_id} has "
            f"{count:,} parameters, above the committed ceiling of {ceiling:,} "
            f"(configs/model_limits.json). The ceiling is a standing constraint from "
            f"Jahid's supervisor (see CLAUDE.md); raising it is his decision, not this "
            f"script's."
        )
    return {
        "component": component,
        "model_id": model_id,
        "parameter_count": count,
        "parameter_ceiling": ceiling,
        "ceiling_asserted": True,
    }


def unasserted_note(component: str, model_id: str | None) -> dict[str, Any]:
    """Record for runs that never loaded a model (e.g. --dry-run)."""
    return {
        "component": component,
        "model_id": model_id,
        "parameter_count": None,
        "parameter_ceiling": None,
        "ceiling_asserted": False,
        "reason": "no model was loaded in this run; parameter count is unmeasured",
    }
