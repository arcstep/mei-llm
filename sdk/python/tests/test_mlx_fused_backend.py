from __future__ import annotations

import sys
import unittest
from pathlib import Path

try:
    import mlx.core as mx
    import mlx.nn as nn
except ImportError:
    mx = nn = None

SDK_ROOT = Path(__file__).resolve().parents[2]
PACKAGE = SDK_ROOT / "packages" / "mei-1.0-58m-base-scratch300m-v1"


def _rms(x):
    return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-6)


def _sinkhorn(logits):
    for _ in range(8):
        logits = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        logits = logits - mx.logsumexp(logits, axis=-2, keepdims=True)
    return mx.exp(logits)


@unittest.skipIf(mx is None, "MLX is not installed")
class FusedBackendTests(unittest.TestCase):
    def test_fused_hadamard_and_mhc_operators_match_reference(self):
        from mei_sdk import Engine

        engine = Engine.load(PACKAGE, backend="mlx-fused")
        from fused_ops import fused_hadamard_mlp, fused_mhc_post, fused_mhc_pre

        model = engine.runtime.model
        block = model.blocks[0]
        x = mx.random.normal((1, 1, 512))
        expected_mlp = block.mlp.d1 * x @ block.mlp.H
        expected_mlp = nn.silu(block.mlp.d2 * expected_mlp) @ block.mlp.H
        expected_mlp = block.mlp.d3 * expected_mlp
        actual_mlp = fused_hadamard_mlp(
            x,
            block.mlp.H,
            block.mlp.d1,
            block.mlp.d2,
            block.mlp.d3,
        )

        lanes = mx.random.normal((1, 1, 4, 512))
        i = 0
        nx = _rms(lanes.reshape(1, 1, 2048))
        hpre = mx.sigmoid(
            model.mhc_a_pre[i] * (nx @ model.mhc_phi_pre[i])
            + model.mhc_b_pre[i]
            + model.mhc_pre_off[i]
        )
        expected_u = mx.einsum("btn,btnc->btc", hpre, lanes)
        hpost = 2 * mx.sigmoid(
            model.mhc_a_post[i] * (nx @ model.mhc_phi_post[i])
            + model.mhc_b_post[i]
            + model.mhc_post_off[i]
        )
        hres = _sinkhorn(
            model.mhc_a_res[i] * (nx @ model.mhc_phi_res[i]).reshape(1, 1, 4, 4)
            + model.mhc_b_res[i]
        )
        y = mx.random.normal((1, 1, 512))
        expected_lanes = (
            mx.einsum("btij,btjc->btic", hres, lanes)
            + hpost[..., None] * y[:, :, None, :]
        )
        actual_u, actual_hpost, actual_hres = fused_mhc_pre(
            lanes,
            model._mei_mhc_phi_packed[i],
            model.mhc_a_pre[i],
            model.mhc_a_post[i],
            model.mhc_a_res[i],
            model.mhc_b_pre[i],
            model.mhc_b_post[i],
            model.mhc_b_res[i],
            model.mhc_pre_off[i],
            model.mhc_post_off[i],
        )
        actual_lanes = fused_mhc_post(lanes, y, actual_hpost, actual_hres)
        mx.eval(
            expected_mlp, actual_mlp, expected_u, actual_u, expected_lanes, actual_lanes
        )
        self.assertLessEqual(
            float(mx.max(mx.abs(expected_mlp - actual_mlp)).item()), 2e-4
        )
        self.assertLessEqual(float(mx.max(mx.abs(expected_u - actual_u)).item()), 2e-4)
        self.assertLessEqual(
            float(mx.max(mx.abs(expected_lanes - actual_lanes)).item()), 2e-4
        )

    def test_fused_chunk_matches_reference_tokens_and_logits(self):
        from mei_sdk import Engine

        reference = Engine.load(PACKAGE, backend="mlx-reference").runtime
        fused = Engine.load(PACKAGE, backend="mlx-fused").runtime
        from kv_manager import KVManager

        ids = reference.tokenizer.encode("成都天气怎么样", add_bos=True)
        ref_kv = KVManager(ordinary_cap=256)
        fused_kv = KVManager(ordinary_cap=256)
        ref_out = ref_kv.prefill_forward(
            reference.model,
            ids[:2],
            ids[2:],
            reserve_tokens=8,
        )
        fused_out = fused_kv.prefill_forward(
            fused.model,
            ids[:2],
            ids[2:],
            reserve_tokens=8,
        )
        ref_logits = ref_out["logits"][:, -1, :]
        expected_ids = []
        for _ in range(8):
            token = int(mx.argmax(ref_logits).item())
            expected_ids.append(token)
            ref_logits = ref_kv.decode_step(reference.model, token)["logits"][:, -1, :]

        chunk = fused_kv.decode_chunk(
            fused.model,
            fused_out["logits"][:, -1, :],
            chunk_size=8,
        )
        self.assertIs(chunk["available"], True)
        self.assertEqual(chunk["ids"], expected_ids)
        mx.eval(ref_logits, chunk["logits"])
        self.assertLessEqual(
            float(mx.max(mx.abs(ref_logits - chunk["logits"])).item()),
            2e-4,
        )


if __name__ == "__main__":
    unittest.main()
