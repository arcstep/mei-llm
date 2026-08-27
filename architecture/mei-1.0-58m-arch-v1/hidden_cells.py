"""Collect embedding + per-block residual cells from NeedleZh."""

from __future__ import annotations

from typing import Any

import mlx.core as mx


def collect_cells(model, tokens: mx.array, **kwargs) -> list[mx.array]:
    out = model(tokens, return_cells=True, **kwargs)
    cells = out.get("cells") or []
    if not cells:
        raise RuntimeError("model did not return cells")
    return list(cells)
