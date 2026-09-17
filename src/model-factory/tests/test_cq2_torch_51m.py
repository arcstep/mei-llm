import unittest
import numpy as np

try:
    import torch
    from mei_sdk.cq2 import quantize, dequantize
    from training.qat.cq2_torch_51m import fake_quant_weight, fake_quant_safe_f16, activation_int8, forward_qat
    from training.torch_backend.model import NeedleZh, NeedleZhConfig
except ModuleNotFoundError:  # CUDA 侧：torch/mei_sdk 只在 A10 环境
    torch=None


@unittest.skipIf(torch is None, "CUDA 侧测试：需要 torch（A10 环境），本机 MLX 环境跳过")
@unittest.skipUnless(torch is not None and torch.cuda.is_available(), "requires authorized A10 CUDA tests")
class Cq2TorchTests(unittest.TestCase):
    def test_mixed_width_and_tail_matches_portable_decoder(self):
        data = np.asarray([np.sin(i * .173) * .8 + (i % 11 - 5) * .03 for i in range(200)], dtype=np.float32)
        expected = np.asarray(dequantize(quantize(data, [2, 4])), dtype=np.float32)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            actual = fake_quant_weight(torch.tensor(data, device="cuda"), [2, 4], ste=False)
        np.testing.assert_allclose(actual.cpu().numpy(), expected, rtol=0, atol=2e-6)

    def test_zero_and_tiny_scales_match_portable_decoder(self):
        for amplitude in (0., 1e-10, 1e-6):
            data = np.linspace(-amplitude, amplitude, 129, dtype=np.float32)
            expected = dequantize(quantize(data, [2, 2]))
            actual = fake_quant_weight(torch.tensor(data, device="cuda"), [2, 2], ste=False)
            np.testing.assert_allclose(actual.cpu().numpy(), expected, rtol=0, atol=2e-6)

    def test_ste_identity_for_weights_and_safe_tensors(self):
        for fn in (lambda x: fake_quant_weight(x, [2], ste=True), lambda x: fake_quant_safe_f16(x, ste=True)):
            data = torch.linspace(-.7, .9, 128, device="cuda", requires_grad=True)
            fn(data).sum().backward()
            self.assertTrue(torch.equal(data.grad, torch.ones_like(data)))

    def test_activation_qdq_changes_forward_and_has_identity_gradient(self):
        data = torch.tensor([[.0013, .5521, -.0414, 1.]], device="cuda", requires_grad=True)
        result = activation_int8(data)
        self.assertFalse(torch.equal(result, data))
        result.sum().backward()
        self.assertTrue(torch.equal(data.grad, torch.ones_like(data)))

    def test_activation_quantization_is_per_vector_and_batch_independent(self):
        a = torch.tensor([[.0013, .5521, -.0414, 1.]], device="cuda")
        batch = torch.cat([a, a * 100000], dim=0)
        self.assertTrue(torch.equal(activation_int8(batch)[0], activation_int8(a)[0]))

    def test_functional_qat_does_not_replace_master_weights(self):
        torch.manual_seed(5)
        model = NeedleZh(NeedleZhConfig().tiny(vocab_size=512, n_layers=3, engram_layers=(0, 2))).cuda()
        tokens = torch.randint(4, 512, (1, 16), device="cuda")
        master = {k: v.detach().clone() for k, v in model.named_parameters()}
        with self.assertRaisesRegex(ValueError, "activation/KV"):
            forward_qat(model, tokens)
        model.enable_qat()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = forward_qat(model, tokens)["logits"]
        result.float().square().mean().backward()
        self.assertTrue(all(torch.equal(v, master[k]) for k, v in model.named_parameters()))
        self.assertTrue(all(v.dtype == torch.float32 for v in model.parameters()))
        self.assertTrue(all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None))
        self.assertGreater(float(model.embed.weight.grad.norm()), 0)
        model.enable_qat(False)
        self.assertTrue(all(not block.attn.qat_activation_ste for block in model.blocks))

    def test_wrong_group_map_is_rejected(self):
        with self.assertRaises(ValueError):
            fake_quant_weight(torch.ones(129, device="cuda"), [2], ste=True)


if __name__ == "__main__":
    unittest.main()
