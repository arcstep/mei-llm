from __future__ import annotations

import unittest

from _repo import ARCHITECTURE_ID, architecture_contracts
from checkpoint import validate_expected_meta
from run_scratch_curriculum import SCRATCH_PROFILES


class Architecture51MReadinessTest(unittest.TestCase):
    def test_weight_contract_refuses_wrong_geometry(self) -> None:
        contracts = architecture_contracts()
        expected = {
            "architecture_id": ARCHITECTURE_ID,
            "weight_contract_sha256": contracts["weight_contract_sha256"],
            "params": 51_463_797,
        }
        wrong = dict(
            expected,
            weight_contract_sha256="0" * 64,
            params=58_541_901,
        )
        with self.assertRaisesRegex(ValueError, "weight_contract_sha256 mismatch"):
            validate_expected_meta(wrong, expected, "strict")

    def test_temporary_arch_v2_checkpoint_is_refused(self) -> None:
        contracts = architecture_contracts()
        temporary = {
            "architecture_id": "mei-1.0-51m-arch-v2",
            "weight_contract_sha256": contracts["weight_contract_sha256"],
            "params": 51_463_797,
        }
        expected = dict(temporary, architecture_id=ARCHITECTURE_ID)
        with self.assertRaisesRegex(ValueError, "architecture_id mismatch"):
            validate_expected_meta(temporary, expected, "strict")

    def test_51m_curriculum_profile_is_unique_and_explicit(self) -> None:
        self.assertEqual(set(SCRATCH_PROFILES), {ARCHITECTURE_ID})
        profile = SCRATCH_PROFILES[ARCHITECTURE_ID]
        self.assertEqual(profile["300m_run"], "pretrain-mei-1.0-51m-base-scratch300m-v1")
        self.assertEqual(profile["recipe"], "pretrain-51m-rungs.json")
        self.assertEqual(profile["pilot_run"], "pretrain-mei-1.0-51m-pilot-5m-v1")

    def test_architecture_contracts_are_separate_and_stable(self) -> None:
        contracts = architecture_contracts()
        self.assertEqual(contracts["weight_contract"]["trainable_params"], 51_463_797)
        hashes = {
            contracts["weight_contract_sha256"],
            contracts["runtime_profile_sha256"],
            contracts["training_aux_sha256"],
        }
        self.assertEqual(len(hashes), 3)
        for digest in hashes:
            self.assertRegex(digest, r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
