"""One seed, threaded through every source of randomness.

Call :func:`set_global_seed` once, as early as possible, and *before* any
sampling. v1 sampled its evaluation questions before seeding, which made those
draws irreproducible; this module exists so that mistake cannot repeat quietly.

The return value records which libraries were actually seeded so a run note can
state it instead of assuming it.
"""
from __future__ import annotations

import os
import random
from typing import Any


def set_global_seed(seed: int) -> dict[str, Any]:
    """Seed python, numpy and torch (when importable). Returns what was seeded."""
    if not isinstance(seed, int):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")

    seeded: dict[str, Any] = {"seed": seed, "python_random": True}
    random.seed(seed)

    # PYTHONHASHSEED is read by the interpreter at startup, so setting it here does
    # nothing for *this* process's hash randomization. It is set only so subprocesses
    # (e.g. the stages the pool-sweep orchestrator launches) inherit it, and the field
    # name says exactly that rather than implying this process is covered.
    os.environ["PYTHONHASHSEED"] = str(seed)
    seeded["pythonhashseed_exported_for_subprocesses"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
        seeded["numpy"] = True
    except ImportError:
        seeded["numpy"] = False

    try:
        import torch

        torch.manual_seed(seed)
        seeded["torch"] = True
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            seeded["torch_cuda"] = True
        else:
            seeded["torch_cuda"] = False
    except ImportError:
        seeded["torch"] = False
        seeded["torch_cuda"] = False

    return seeded


def new_rng(seed: int) -> random.Random:
    """A local, independently seeded RNG for a single sampling call."""
    return random.Random(seed)
