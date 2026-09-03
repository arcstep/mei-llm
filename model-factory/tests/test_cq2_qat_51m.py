from __future__ import annotations

import sys
import unittest
from pathlib import Path

import mlx.core as mx
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = next(parent for parent in HERE.parents if (parent / "CURRENT.json").is_file())
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "platform/python-sdk"))

from training.qat.cq2_policy_51m import lm_storage_dtype, uniform_group_bits  # noqa: E402
from training.qat.cq2_qat_51m import fake_quant_weight  # noqa: E402
from mei_sdk.cq2 import dequantize, quantize  # noqa: E402


class Cq2QatTests(unittest.TestCase):
    def test_mlx_fake_quant_matches_portable_dequant(self) -> None:
        values = np.asarray(
            [np.sin(index * 0.173) * 0.8 + ((index % 11) - 5) * 0.03 for index in range(200)],
            dtype=np.float32,
        )
        expected = np.asarray(dequantize(quantize(values, [2, 4])), dtype=np.float32)
        actual = np.asarray(fake_quant_weight(mx.array(values), (2, 4), ste=False))
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-6)

    def test_ste_has_identity_gradient(self) -> None:
        values = mx.array(np.linspace(-0.7, 0.9, 128, dtype=np.float32))
        gradient = mx.grad(lambda item: mx.sum(fake_quant_weight(item, (2,), ste=True)))(values)
        np.testing.assert_allclose(np.asarray(gradient), np.ones((128,), dtype=np.float32))

    def test_policy_is_shared_per_group_not_legacy_block64(self) -> None:
        self.assertEqual(lm_storage_dtype("embed.weight", (24000, 512)), "cq4")
        self.assertEqual(lm_storage_dtype("blocks.0.attn.q_proj.weight", (512, 512)), "cq2")
        self.assertEqual(lm_storage_dtype("blocks.0.attn_norm.scale", (512,)), "f16")
        self.assertEqual(lm_storage_dtype("engrams.0.taps", (4, 512)), "f16")
        self.assertEqual(lm_storage_dtype("blocks.0.mlp.d1", (512,)), "f16")
        self.assertEqual(lm_storage_dtype("mhc_b_res", (27, 4, 4)), "cq4")
        self.assertEqual(uniform_group_bits("blocks.0.attn.q_proj.weight", (129,)), (2, 2))
        self.assertIsNone(uniform_group_bits("blocks.0.attn_norm.scale", (512,)))


if __name__ == "__main__":
    unittest.main()
