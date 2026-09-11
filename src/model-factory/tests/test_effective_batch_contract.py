import unittest

from training.cpt.effective_batch_contract import continuation_batch_error


class EffectiveBatchTests(unittest.TestCase):
    def test_rejects_uncompensated_microbatch_reduction(self):
        parent = {"batch_size": 8, "grad_accum": 1, "seq_len": 2048}
        stage = {"batch_size": 1, "grad_accum": 1, "seq_len": 2048}
        self.assertIn("effective batch changed", continuation_batch_error(parent, stage))

    def test_accepts_memory_saving_accumulation(self):
        parent = {"batch_size": 8, "grad_accum": 1, "seq_len": 2048}
        stage = {"batch_size": 1, "grad_accum": 8, "seq_len": 2048}
        self.assertIsNone(continuation_batch_error(parent, stage))

    def test_missing_explicit_binding_fails_closed(self):
        self.assertIsNotNone(continuation_batch_error({}, {}))
