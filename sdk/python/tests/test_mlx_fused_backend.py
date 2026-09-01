from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

try:
    import mlx.core as mx
    import mlx.nn as nn
except ImportError:
    mx = nn = None

MEI_LLM = Path(__file__).resolve().parents[3]
ARCH = MEI_LLM / "architecture" / "mei-1.0-51m-arch-v1"
RUNTIME_SHARED = MEI_LLM / "runtime" / "_shared"
for path in (ARCH, RUNTIME_SHARED):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


@unittest.skipIf(mx is None, "MLX is not installed")
class FusedBackendTests(unittest.TestCase):
    def test_activation_int8_ste_uses_qdq_forward_and_identity_backward(self):
        from architecture import _ste_activation_int8

        x = mx.array(
            [[[-1.0, -0.377, 0.119, 0.731], [-100.0, -37.7, 11.9, 73.1]]],
            dtype=mx.float32,
        )
        maximum = mx.max(mx.abs(x), axis=-1, keepdims=True)
        scale = mx.where(maximum > 0, maximum / 127.0, 1.0)
        expected = mx.clip(mx.round(x / scale), -128.0, 127.0) * scale
        actual = _ste_activation_int8(x)
        gradient = mx.grad(lambda value: mx.sum(_ste_activation_int8(value)))(x)
        extended = mx.concatenate(
            [x, mx.array([[[10000.0, -3770.0, 1190.0, 7310.0]]], dtype=mx.float32)],
            axis=-2,
        )
        extended_prefix = _ste_activation_int8(extended)[..., : int(x.shape[-2]), :]
        mx.eval(expected, actual, gradient, extended_prefix)
        self.assertLessEqual(float(mx.max(mx.abs(expected - actual))), 1e-7)
        self.assertLessEqual(float(mx.max(mx.abs(gradient - 1.0))), 1e-7)
        self.assertLessEqual(float(mx.max(mx.abs(actual - extended_prefix))), 1e-7)

    def test_attention_qat_quantizes_input_output_and_kv_but_not_q(self):
        import architecture as arch
        from config import NeedleZhConfig

        cfg = NeedleZhConfig.from_spec()
        attention = arch.GroupedAttention(cfg)
        seen: list[tuple[int, ...]] = []
        original = arch._ste_activation_int8

        def tracked(value):
            seen.append(tuple(int(part) for part in value.shape))
            return original(value)

        arch._ste_activation_int8 = tracked
        arch.QAT_ACTIVATION_STE = True
        try:
            x = mx.random.normal((1, 3, cfg.d_model))
            rope = arch.precompute_rope(cfg.head_dim, 3, cfg.rope_theta)
            output, cache = attention(x, rope, mask=arch.causal_mask(1, 3))
            mx.eval(output, cache)
        finally:
            arch.QAT_ACTIVATION_STE = False
            arch._ste_activation_int8 = original

        self.assertEqual(seen.count((1, 3, cfg.d_model)), 2)
        self.assertEqual(seen.count((1, cfg.n_kv_heads, 3, cfg.head_dim)), 2)
        self.assertNotIn((1, cfg.n_heads, 3, cfg.head_dim), seen)

    def test_q2_bitplane_linear_matches_portable_code_layout_without_extra_bytes(self):
        from cq2_metal import PackedCqMatrix

        rng = np.random.default_rng(51)
        rows, cols = 32, 512
        groups = rows * (cols // 128)
        data = rng.integers(0, 256, size=groups * 32, dtype=np.uint8).tobytes()
        scales = rng.uniform(0.01, 0.2, size=groups).astype("<f2").tobytes()
        bit_map = bytes((groups + 7) // 8)
        x = mx.array(rng.normal(size=(1, 1, cols)).astype(np.float32))
        matrix = PackedCqMatrix(
            name="test.q2",
            shape=(rows, cols),
            bits=2,
            data=data,
            scales_f16=scales,
            bit_map=bit_map,
        )
        expected = matrix.linear(x)
        resident_bytes = matrix.resident_bytes
        matrix.enable_q2_bitplane_linear(drop_codes=True)
        actual = matrix.linear(x)
        mx.eval(expected, actual)
        self.assertEqual(matrix.resident_bytes, resident_bytes)
        self.assertLessEqual(float(mx.max(mx.abs(expected - actual))), 3e-6)

    def test_current_fixed_wht_mhc20_and_full_tap_engram_match_reference(self):
        from architecture import (
            NeedleZh,
            mhc_offsets,
            rms_unit,
            sinkhorn,
            walsh_hadamard,
        )
        from config import NeedleZhConfig
        from fused_ops import fused_fixed_wht_mlp, fused_mhc_pre

        mx.random.seed(123)
        cfg = NeedleZhConfig.from_spec()
        model = NeedleZh(cfg)

        x = mx.random.normal((1, 1, 512))
        mlp = model.blocks[0].mlp
        expected_mlp = mlp.d3 * walsh_hadamard(
            nn.silu(mlp.d2 * walsh_hadamard(mlp.d1 * x))
        )
        actual_mlp = fused_fixed_wht_mlp(x, mlp.d1, mlp.d2, mlp.d3)

        lanes = mx.random.normal((1, 1, 4, 512))
        pre_off, post_off = mhc_offsets(cfg.n_layers, cfg.mhc_lanes)
        nx = rms_unit(lanes.reshape(1, 1, 2048))
        expected_u = mx.einsum(
            "btn,btnc->btc",
            mx.sigmoid(
                model.mhc_a_pre[0] * (nx @ model.mhc_phi_pre[0])
                + model.mhc_b_pre[0]
                + pre_off[0]
            ),
            lanes,
        )
        expected_hpost = 2 * mx.sigmoid(
            model.mhc_a_post[0] * (nx @ model.mhc_phi_post[0])
            + model.mhc_b_post[0]
            + post_off[0]
        )
        expected_hres = sinkhorn(
            (
                model.mhc_a_res[0] * (nx @ model.mhc_phi_res[0])
            ).reshape(1, 1, 4, 4)
            + model.mhc_b_res[0],
            20,
        )
        packed_phi = mx.concatenate(
            [
                model.mhc_phi_pre[0].T,
                model.mhc_phi_post[0].T,
                model.mhc_phi_res[0].T,
            ],
            axis=0,
        )
        actual_u, actual_hpost, actual_hres = fused_mhc_pre(
            lanes,
            packed_phi,
            model.mhc_a_pre[0],
            model.mhc_a_post[0],
            model.mhc_a_res[0],
            model.mhc_b_pre[0],
            model.mhc_b_post[0],
            model.mhc_b_res[0],
            pre_off[0],
            post_off[0],
        )

        prefix = mx.array([[2, *range(4, 15)]], dtype=mx.int32)
        current = mx.array([[99]], dtype=mx.int32)
        expected_engram = model._engram_stack(current, prefix)
        model.set_inference_backend("mlx-fused")
        actual_engram = model._engram_stack(current, prefix)
        mx.eval(
            expected_mlp,
            actual_mlp,
            expected_u,
            actual_u,
            expected_hpost,
            actual_hpost,
            expected_hres,
            actual_hres,
            *expected_engram,
            *actual_engram,
        )
        self.assertLessEqual(float(mx.max(mx.abs(expected_mlp - actual_mlp))), 2e-5)
        self.assertLessEqual(float(mx.max(mx.abs(expected_u - actual_u))), 2e-5)
        self.assertLessEqual(
            float(mx.max(mx.abs(expected_hpost.reshape(-1) - actual_hpost))),
            2e-5,
        )
        self.assertLessEqual(
            float(mx.max(mx.abs(expected_hres.reshape(-1) - actual_hres))),
            2e-5,
        )
        self.assertLessEqual(
            float(mx.max(mx.abs(expected_engram[0] - actual_engram[0]))),
            2e-5,
        )
        self.assertLessEqual(
            float(mx.max(mx.abs(expected_engram[1] - actual_engram[1]))),
            2e-5,
        )

    def test_compiled_chunk_matches_sequential_fused_decode(self):
        from architecture import NeedleZh
        from config import NeedleZhConfig
        from kv_manager import KVManager

        mx.random.seed(23)
        model = NeedleZh(NeedleZhConfig.from_spec())
        model.set_inference_backend("mlx-fused")
        sink = [2, 10, 11, 12]
        ordinary = list(range(20, 36))
        sequential = KVManager()
        chunked = KVManager()
        sequential_out = sequential.prefill_forward(
            model, sink, ordinary, reserve_tokens=8
        )
        chunked_out = chunked.prefill_forward(model, sink, ordinary, reserve_tokens=8)
        logits = sequential_out["logits"][:, -1, :]
        expected_ids = []
        for _ in range(4):
            token = int(mx.argmax(logits).item())
            expected_ids.append(token)
            logits = sequential.decode_step(model, token)["logits"][:, -1, :]
        chunk = chunked.decode_chunk(
            model,
            chunked_out["logits"][:, -1, :],
            chunk_size=4,
        )
        mx.eval(logits, chunk["logits"])
        self.assertIs(chunk["available"], True)
        self.assertEqual(chunk["ids"], expected_ids)
        self.assertLessEqual(float(mx.max(mx.abs(logits - chunk["logits"]))), 2e-5)
        self.assertEqual(sequential.snapshot()["recomputed_decode_steps"], 0)
        self.assertEqual(chunked.snapshot()["recomputed_decode_steps"], 0)

    def test_native_v2_uses_bounded_int8_code_scale_cache(self):
        from architecture import NeedleZh
        from config import NeedleZhConfig
        from kv_manager import KVManager

        mx.random.seed(51)
        cfg = NeedleZhConfig.from_spec().tiny(vocab_size=256)
        model = NeedleZh(cfg)
        # Exercise deployment quantization with float weights. Packed-weight
        # loading has separate coverage and must not be required to prove the
        # code/scale cache representation itself.
        model._set_native_v2_quantization(True)
        manager = KVManager(
            ordinary_cap=8,
            sink_cap=4,
            max_context=32,
            output_reserve=4,
        )
        out = manager.prefill_forward(
            model,
            [2, 10, 11, 12],
            [20, 21, 22, 23, 24, 25],
            reserve_tokens=4,
        )
        mx.eval(out["logits"], out["cache"])

        self.assertEqual(len(out["cache"]), cfg.n_layers)
        for layer in out["cache"]:
            self.assertEqual(len(layer), 4)
            key_codes, key_scales, value_codes, value_scales = layer
            self.assertEqual(key_codes.dtype, mx.int8)
            self.assertEqual(value_codes.dtype, mx.int8)
            self.assertEqual(key_scales.dtype, mx.float32)
            self.assertEqual(value_scales.dtype, mx.float32)
            self.assertEqual(int(key_codes.shape[-1]), cfg.head_dim)
            self.assertEqual(int(value_codes.shape[-1]), cfg.head_dim)
            self.assertEqual(int(key_scales.shape[-1]), 1)
            self.assertEqual(int(value_scales.shape[-1]), 1)

        snapshot = manager.snapshot()
        self.assertEqual(snapshot["measured_cache_dtype"], "mlx.core.int8")
        self.assertEqual(snapshot["activation_dtype"], "int8-qdq")
        self.assertTrue(snapshot["cache_growth_bounded"])
        self.assertEqual(snapshot["visible_tokens"], 10)
        self.assertTrue(manager.ram_bound_ok(cfg.n_layers, cfg.n_kv_heads, cfg.head_dim))

        token = int(mx.argmax(manager.last_logits).item())
        manager.decode_step(model, token)
        self.assertEqual(manager.snapshot()["incremental_decode_steps"], 1)
        self.assertEqual(manager.snapshot()["recomputed_decode_steps"], 0)
        self.assertEqual(manager.snapshot()["measured_cache_dtype"], "mlx.core.int8")


if __name__ == "__main__":
    unittest.main()
