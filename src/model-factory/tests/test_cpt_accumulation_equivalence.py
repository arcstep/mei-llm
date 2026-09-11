import unittest

import mlx.core as mx
import mlx.nn as nn

from common.train_common import train_lm_steps


class SmallLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = mx.arange(49, dtype=mx.float32).reshape(7, 7) / 100

    def __call__(self, inputs, return_confidence=False):
        return {"logits": self.weight[inputs]}


class AccumulationEquivalenceTests(unittest.TestCase):
    def test_token_target_with_partial_final_accumulation(self):
        from common.train_common import cosine_lr_tokens
        previous_device = mx.default_device()
        mx.set_default_device(mx.cpu)
        try:
            windows = [{"x": [1, 2, 3, 4], "y": [2, 3, 4, 5], "mask": [1, 1, 1, 1]} for index in range(10)]
            reports = []
            result = train_lm_steps(SmallLanguageModel(), windows, lr=0.0003, lr_final=0.00003,
                                    start_step=19, start_tokens_seen=100, target_tokens=140,
                                    horizon_tokens=40, lr_token_offset=100, batch_size=1, grad_accum=8,
                                    allow_repeat=False, compile_train=False,
                                    on_step=lambda step, row: reports.append((step, row)))
            self.assertEqual(result["steps"], 2)
            self.assertEqual(result["tokens_seen"], 140)
            self.assertEqual([row[1]["tokens_seen_step"] for row in reports], [32, 8])
            self.assertEqual([row[0] for row in reports], [19, 20])
            self.assertAlmostEqual(reports[0][1]["lr"], 0.0003, places=8)
            self.assertAlmostEqual(reports[1][1]["lr"], cosine_lr_tokens(32, 40, 0.0003, 0.00003), places=8)
        finally:
            mx.set_default_device(previous_device)

    def test_mask_weighted_accumulation_matches_full_batch(self):
        previous_device = mx.default_device()
        mx.set_default_device(mx.cpu)
        try:
            windows = [{"x": [index % 7, 2, 3, 4], "y": [1, 3, 4, 5],
                        "mask": [1, 1, 1, int(index != 7)]} for index in range(8)]
            full = SmallLanguageModel()
            accumulated = SmallLanguageModel()
            reports = []
            for model, batch_size, grad_accum in ((full, 8, 1), (accumulated, 1, 8)):
                reports.append(train_lm_steps(model, windows, steps=1, lr=0.0003,
                                             batch_size=batch_size, grad_accum=grad_accum,
                                             allow_repeat=False, compile_train=False))
            self.assertLess(float(mx.max(mx.abs(full.weight - accumulated.weight))), 1e-6)
            self.assertEqual(reports[0]["tokens_seen"], 31)
            self.assertEqual(reports[1]["tokens_seen"], 31)
            self.assertEqual(reports[0]["steps"], 1)
            self.assertEqual(reports[1]["steps"], 1)
        finally:
            mx.set_default_device(previous_device)
