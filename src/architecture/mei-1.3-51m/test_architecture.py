from __future__ import annotations

import mlx.core as mx
import mlx.utils as xu

from architecture import (
    NeedleZh,
    count_params,
    sinkhorn,
    walsh_hadamard,
    walsh_matrix,
)
from config import NeedleZhConfig


def _parameter_names(model) -> set[str]:
    flat = xu.tree_flatten(model.parameters())
    leaves = flat[0] if isinstance(flat, tuple) else flat
    return {str(item[0]) for item in leaves if isinstance(item, tuple) and len(item) == 2}


def test_fixed_walsh_matches_dense_reference() -> None:
    mx.random.seed(7)
    for width in (2, 4, 8, 16, 64, 512):
        x = mx.random.normal((3, width))
        actual = walsh_hadamard(x)
        expected = x @ walsh_matrix(width)
        mx.eval(actual, expected)
        assert float(mx.max(mx.abs(actual - expected))) <= 1e-5


def test_fixed_walsh_custom_vjp_is_self_adjoint() -> None:
    mx.random.seed(9)
    weight = mx.random.normal((2, 512))
    x = mx.random.normal((2, 512))
    grad = mx.grad(lambda value: mx.sum(walsh_hadamard(value) * weight))(x)
    expected = walsh_hadamard(weight)
    mx.eval(grad, expected)
    assert float(mx.max(mx.abs(grad - expected))) <= 1e-5


def test_full_parameter_topology_excludes_fixed_arrays() -> None:
    model = NeedleZh(NeedleZhConfig.from_spec())
    names = _parameter_names(model)
    assert count_params(model) == 51_463_797
    assert not any(name.endswith(".H") or name == "H" for name in names)
    assert "mhc_pre_off" not in names
    assert "mhc_post_off" not in names


def test_sinkhorn_contract_is_doubly_stochastic() -> None:
    mx.random.seed(11)
    routed = sinkhorn(mx.random.normal((2, 3, 4, 4)), iters=20)
    mx.eval(routed)
    assert float(mx.max(mx.abs(mx.sum(routed, axis=-1) - 1.0))) <= 1e-4
    assert float(mx.max(mx.abs(mx.sum(routed, axis=-2) - 1.0))) <= 1e-4


def test_engram_twelve_token_prefix_matches_full_sequence() -> None:
    cfg = NeedleZhConfig().tiny()
    model = NeedleZh(cfg)
    tokens = mx.array([[2, *range(4, 23)]], dtype=mx.int32)
    full = model._engram_stack(tokens)
    prefix = tokens[:, -13:-1]
    current = tokens[:, -1:]
    incremental = model._engram_stack(current, prefix)
    mx.eval(*full, *incremental)
    assert cfg.engram_history == 12
    assert float(mx.max(mx.abs(full[0][:, :, -1:, :] - incremental[0]))) <= 1e-5
    assert float(mx.max(mx.abs(full[1][:, :, -1:, :] - incremental[1]))) <= 1e-5


def test_tiny_forward_is_finite() -> None:
    model = NeedleZh(NeedleZhConfig().tiny())
    out = model(mx.array([[2, 8, 9, 1]], dtype=mx.int32))
    mx.eval(out["logits"])
    assert tuple(out["logits"].shape) == (1, 4, 24_000)
    assert bool(mx.all(mx.isfinite(out["logits"])))
