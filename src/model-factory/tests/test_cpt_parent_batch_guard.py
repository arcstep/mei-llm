import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from common.paths import ensure_formal_on_path

ensure_formal_on_path()
from training.cpt.cpt_gates import refuse_cpt_parent


class ParentBatchGuardTests(unittest.TestCase):
    def test_parent_validation_rejects_uncompensated_batch_change(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "state.npz"
            state.write_bytes(b"fixture")
            metadata = {"architecture_id": "mei-1.0-51m-arch-v1", "params": 51463797,
                        "weight_contract_sha256": "fixture", "tokens_seen": 1200006656,
                        "batch_size": 8, "grad_accum": 1, "seq_len": 2048}
            state.with_suffix(".meta.json").write_text(json.dumps(metadata))
            schedule = {"kind": "cpt", "parent_tokens_seen": 1200006656,
                        "parent_state_sha256": "fixture-hash",
                        "curriculum": [{"batch_size": 1, "grad_accum": 1, "seq_len": 2048}]}
            with patch("training.cpt.cpt_gates.architecture_contracts", return_value={"weight_contract_sha256": "fixture"}), \
                    patch("training.cpt.cpt_gates.file_sha256", return_value="fixture-hash"):
                self.assertIn("effective batch changed", refuse_cpt_parent(state, schedule))
                schedule["curriculum"][0]["grad_accum"] = 8
                self.assertIsNone(refuse_cpt_parent(state, schedule))
