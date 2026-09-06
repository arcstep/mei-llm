"""Single deployment/QAT dtype policy for the canonical 51M tensor inventory."""

from __future__ import annotations

import math
from typing import Sequence


GROUP_SIZE = 128
QUANT_MATH_ID = "mei-cq-v2-g128-wht-codebook"


def lm_storage_dtype(name: str, shape: Sequence[int]) -> str:
    """Return the portable v2 storage dtype for one canonical LM tensor."""

    shape = tuple(int(value) for value in shape)
    safe = (
        name.endswith(".scale")
        or name.endswith(".bias")
        or name.endswith("attn_gate")
        or (name.startswith("engrams.") and name.endswith(".taps"))
        or name.endswith((".mlp.d1", ".mlp.d2", ".mlp.d3"))
        or name.startswith("conf_")
        or not shape
    )
    if safe:
        return "f16"
    if name == "embed.weight" or name.startswith("mhc_"):
        return "cq4"
    return "cq2"


def uniform_group_bits(name: str, shape: Sequence[int]) -> tuple[int, ...] | None:
    """Materialize the explicit per-group map used by QAT and package export."""

    dtype = lm_storage_dtype(name, shape)
    if dtype == "f16":
        return None
    n_params = math.prod(tuple(int(value) for value in shape)) or 1
    groups = (n_params + GROUP_SIZE - 1) // GROUP_SIZE
    width = 4 if dtype == "cq4" else 2
    return (width,) * groups
