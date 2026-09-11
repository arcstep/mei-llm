import unittest

from diagnostics.cpt_accumulation_resource import validate_config


class ResourceConfigTests(unittest.TestCase):
    def test_probe_cannot_be_expanded_into_training(self):
        config = {"schema": "mei-cpt-accumulation-resource-config-v1",
                  "scope": "synthetic_resource_only_no_saved_weights", "batch_size": 1,
                  "grad_accum": 8, "seq_len": 2048, "steps": 2, "lr": 3e-5,
                  "max_peak_allocation_bytes": 17179869184}
        validate_config(config)
        for key, value in (("steps", 1000), ("lr", 3e-4), ("scope", "cpt"), ("grad_accum", 1)):
            with self.assertRaises(ValueError):
                validate_config({**config, key: value})
