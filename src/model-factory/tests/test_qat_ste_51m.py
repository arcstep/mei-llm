from __future__ import annotations

import unittest

try:
    import mlx.core as mx
except ImportError:
    mx = None


@unittest.skipIf(mx is None, "MLX is not installed")
class QatSteContractTests(unittest.TestCase):
    def test_legacy_weight_ste_is_fake_quant_forward_identity_backward(self):
        from release.quant_pack_51m import mlx_fake_quant_weight, mlx_ste_quantize

        x = mx.array([[-1.0, -0.377, 0.119, 0.731]], dtype=mx.float32)
        expected = mlx_fake_quant_weight(x, bits=4, block_size=4)
        actual = mlx_ste_quantize(x, bits=4, block_size=4)
        gradient = mx.grad(
            lambda value: mx.sum(mlx_ste_quantize(value, bits=4, block_size=4))
        )(x)
        mx.eval(expected, actual, gradient)
        self.assertLessEqual(float(mx.max(mx.abs(expected - actual))), 1e-7)
        self.assertLessEqual(float(mx.max(mx.abs(gradient - 1.0))), 1e-7)

    def test_activation_ste_is_qdq_forward_identity_backward(self):
        from release.quant_pack_51m import mlx_fake_quant_activation, mlx_ste_activation

        x = mx.array(
            [[[-1.0, -0.377, 0.119, 0.731], [-100.0, -37.7, 11.9, 73.1]]],
            dtype=mx.float32,
        )
        expected = mlx_fake_quant_activation(x, bits=8)
        actual = mlx_ste_activation(x, bits=8)
        gradient = mx.grad(lambda value: mx.sum(mlx_ste_activation(value, bits=8)))(x)
        extended = mx.concatenate(
            [x, mx.array([[[10000.0, -3770.0, 1190.0, 7310.0]]], dtype=mx.float32)],
            axis=-2,
        )
        extended_prefix = mlx_ste_activation(extended, bits=8)[..., : int(x.shape[-2]), :]
        mx.eval(expected, actual, gradient, extended_prefix)
        self.assertLessEqual(float(mx.max(mx.abs(expected - actual))), 1e-7)
        self.assertLessEqual(float(mx.max(mx.abs(gradient - 1.0))), 1e-7)
        self.assertLessEqual(float(mx.max(mx.abs(actual - extended_prefix))), 1e-7)


if __name__ == "__main__":
    unittest.main()
